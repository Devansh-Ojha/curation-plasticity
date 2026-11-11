"""Resumable real A0/A3 experiment with Phoenix tracing and code evaluators.

Each invocation runs exactly one arm/seed. Hosted model calls are real and
require NVIDIA_API_KEY in the caller's environment; analysis labels stay local.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from phoenix.client import Client

from run_real_a3_smoke import CURATOR_PROMPT_TEMPLATE, REFLECTOR_PROMPT_TEMPLATE, LoggedNVIDIAClient
from src.agents.nvidia_client import NVIDIAAPIError
from src.agents.real_llm_actor import RealLLMActor
from src.analysis.metrics import compute_curator_confusion
from src.analysis.reflection_grounding import make_reflection_trace_records, check_rule_grounding
from src.loop.provenance import Playbook
from src.obs.dataset_upload import _CATEGORY_TERMS
from src.obs.experiments import run_saved_experiment
from src.obs.phoenix_setup import flush_phoenix, init_phoenix_tracer
from src.obs.spans import span
from src.testbed.grader import Grader
from src.testbed.mock_api import MockAPI
from src.testbed.task_gen import generate_task_suites

ACTOR_MODEL = "nvidia/nemotron-3.5-bolt-e4b"
OPTIMIZER_MODEL = "meta/llama-3.2-90b-vision-instruct"
EPOCHS = 8
BATCH_SIZE = 10
TRAIN_POOL_SIZE = 20
EPISODE_WORKERS = 5
HELDOUT_EPOCHS = (0, 4, 8)
TASK_SECRET = re.compile(r"secret_test_key|nvapi-[A-Za-z0-9_-]+", re.IGNORECASE)
_JSONL_LOCK = threading.Lock()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_jsonl(path: Path, value: Dict[str, Any]) -> None:
    with _JSONL_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(value, ensure_ascii=False) + "\n")


def parse_json(text: str) -> Any:
    clean = (text or "").strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1]
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
    return json.loads(clean.strip())


def is_hard_cap(exc: BaseException) -> bool:
    return "HARD CAP" in str(exc)


def restore_playbook(state: Dict[str, Any], max_tokens: int) -> Playbook:
    pb = Playbook(max_tokens=max_tokens)
    saved = state.get("playbook", {})
    pb.rules = saved.get("rules", {})
    pb.next_rule_num = saved.get("next_rule_num", 1)
    return pb


def as_tuple(value: Any) -> Any:
    return tuple(as_tuple(v) for v in value) if isinstance(value, list) else value


def safe_episode(actor: RealLLMActor, task: Dict[str, Any], playbook: Playbook,
                 api: MockAPI, trace_attrs: Dict[str, Any], failures: List[Dict[str, Any]],
                 *, phase: str) -> Dict[str, Any]:
    traced_task = dict(task, _trace_context={**trace_attrs, "phase": phase})
    error = ""
    for episode_attempt in range(3):
        try:
            result = actor.run_episode(traced_task, playbook, api)
            if not result.get("api_failure"):
                return result
            error = "episode returned api_failure"
        except Exception as exc:
            if is_hard_cap(exc):
                raise
            if not isinstance(exc, (NVIDIAAPIError, TimeoutError, OSError)):
                raise
            error = str(exc)
        if episode_attempt < 2:
            append_jsonl(actor.client.log_path.parent / "optimizer_failures.jsonl", {
                "phase": phase, "task_id": task["task_id"],
                "error": error, "episode_retry": episode_attempt + 1,
            })
            continue
        failures.append({"task_id": task["task_id"], "phase": phase, "error": error})
        raise NVIDIAAPIError(
            f"Episode {task['task_id']} in {phase} still failed on an API error after 3 whole-episode attempts: {error}")


def run_episodes_concurrently(actor: RealLLMActor, tasks: List[Dict[str, Any]],
                              playbook: Playbook, trace_attrs: Dict[str, Any],
                              failures: List[Dict[str, Any]], *, phase: str) -> Dict[str, Dict[str, Any]]:
    """Run independent episodes concurrently; the shared NVIDIA client rate-limits starts."""
    results = {}
    if not tasks:
        return results
    with ThreadPoolExecutor(max_workers=min(EPISODE_WORKERS, len(tasks)),
                            thread_name_prefix=f"episode-{phase}") as pool:
        futures = {
            pool.submit(safe_episode, actor, task, playbook, MockAPI(), trace_attrs,
                        failures, phase=phase): task["task_id"]
            for task in tasks
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def call_curator(client: LoggedNVIDIAClient, *, task: Dict[str, Any], history: List[Dict[str, Any]],
                 trusted_success_ids: List[str], arm: str, seed: int, epoch: int,
                 optimizer_model: str) -> Dict[str, Any]:
    history_lines = []
    for row in history[-4:]:
        history_lines.append({
            "epoch": row.get("epoch"), "passed": row.get("passed"),
            "feedback": row.get("feedback"), "trajectory": row.get("trajectory", []),
        })
    user = (f"Task ID: {task['task_id']}\nInstruction: {task['instruction']}\n"
            f"Available corroborated success IDs: {json.dumps(trusted_success_ids)}\n"
            f"Execution history: {json.dumps(history_lines, ensure_ascii=False)}\n"
            "Return the required diagnostic JSON.")
    messages = [{"role": "system", "content": CURATOR_PROMPT_TEMPLATE},
                {"role": "user", "content": user}]
    last_error = None
    for attempt in range(2):
        try:
            client.current_phase = "curate"
            with span("curator_decision", phase="curate", attributes={
                "arm": arm, "seed": seed, "epoch": epoch, "task_id": task["task_id"],
                "model_id": optimizer_model, "execution_mode": "real",
            }):
                response = client.chat_completion(optimizer_model, messages,
                                                   temperature=0.0, max_tokens=1200)
            parsed = parse_json(response.get("text") or "")
            if not isinstance(parsed, dict) or parsed.get("decision") not in {"run", "skip"}:
                raise ValueError("Curator response must be a JSON object with decision run/skip")
            if parsed["decision"] == "skip":
                parsed["anchor_success_id"] = None
            return {"task_id": task["task_id"], "decision": parsed["decision"],
                    "label": parsed.get("label", "Uncertain"),
                    "skip_epochs": max(1, int(parsed.get("skip_epochs", 1))),
                    "anchor_success_id": parsed.get("anchor_success_id"),
                    "reason": parsed.get("reason", ""), "called_llm": True}
        except Exception as exc:
            if is_hard_cap(exc):
                raise
            last_error = str(exc)
            append_jsonl(client.log_path.parent / "optimizer_failures.jsonl", {
                "phase": "curate", "task_id": task["task_id"], "attempt": attempt + 1,
                "error": last_error,
            })
            if attempt == 0:
                messages.extend([{"role": "assistant", "content": ""},
                                 {"role": "user", "content":
                                  "Previous request failed or was invalid. Return strict valid JSON."}])
    return {"task_id": task["task_id"], "decision": "run", "label": "Uncertain",
            "skip_epochs": 1, "anchor_success_id": None, "reason": last_error,
            "called_llm": True, "api_failure": True}


def call_reflector(client: LoggedNVIDIAClient, failed: List[Dict[str, Any]],
                   current: Playbook, anchors: List[Dict[str, Any]],
                   histories: Dict[str, List[Dict[str, Any]]], arm: str, seed: int,
                   epoch: int, optimizer_model: str) -> tuple[list, list, list]:
    records = make_reflection_trace_records(failed, histories)
    bundle = [{"task_id": r["task_id"], "failed_requests": r["failed_requests"],
               "successful_request_quotes": r["successful_request_quotes"]}
              for r in records]
    anchor_bundle = [{"task_id": a["task_id"], "instruction": a["instruction"],
                      "successful_trajectories": a["successful_trajectories"]}
                     for a in anchors]
    user = (f"Current Playbook:\n{current.to_markdown()}\n\n"
            "Trace bundle (only failed requests, error bodies, and observed successful requests):\n"
            + json.dumps({"traces": bundle, "trusted_success_anchors": anchor_bundle},
                                             ensure_ascii=False, indent=2))
    messages = [{"role": "system", "content": REFLECTOR_PROMPT_TEMPLATE},
                {"role": "user", "content": user}]
    rejections: List[Dict[str, Any]] = []
    accepted_ops: List[Dict[str, Any]] = []
    for attempt in range(1, 3):
        try:
            client.current_phase = "reflect"
            with span("reflector_call", phase="reflect", attributes={
                "arm": arm, "seed": seed, "epoch": epoch,
                "model_id": optimizer_model, "execution_mode": "real",
            }):
                response = client.chat_completion(optimizer_model, messages,
                                                   temperature=0.2 * attempt, max_tokens=1800)
            ops = parse_json(response.get("text") or "")
            if not isinstance(ops, list) or not all(isinstance(op, dict) for op in ops):
                raise ValueError("Reflector output must be a JSON array of operations")
            if any(op.get("op") not in {"add", "modify", "delete"} for op in ops):
                raise ValueError("Reflector operation must be add, modify, or delete")
            if any(op.get("op") == "add" and "rule_id" in op for op in ops):
                raise ValueError("Rule IDs are assigned by the harness")
            issues = check_rule_grounding(ops, records) if arm != "A0-raw" else []
            if issues:
                reason = json.dumps(issues, ensure_ascii=False)
                rejections.append({"epoch": epoch, "attempt": attempt, "kind": "grounding",
                                  "reason": reason, "ops": ops})
                raise GroundingRejected(reason)
            if any(op.get("op") in {"modify", "delete"} and op.get("rule_id") not in current.rules
                   for op in ops):
                raise GateRejected("operation references a rule ID absent from current playbook")
            accepted_ops = ops
            return accepted_ops, rejections, records
        except GroundingRejected:
            if attempt == 2:
                break
        except GateRejected as exc:
            rejections.append({"epoch": epoch, "attempt": attempt, "kind": "gate",
                              "reason": str(exc)})
        except Exception as exc:
            if is_hard_cap(exc):
                raise
            kind = "api" if isinstance(exc, (NVIDIAAPIError, TimeoutError, OSError)) else "format"
            rejections.append({"epoch": epoch, "attempt": attempt, "kind": kind,
                              "reason": str(exc)})
            if attempt == 2:
                break
        if attempt < 2:
            messages.extend([{"role": "assistant", "content": ""},
                             {"role": "user", "content":
                              "The previous attempt was rejected. Return grounded strict JSON only."}])
    return [], rejections, records


class GroundingRejected(ValueError):
    pass


class GateRejected(ValueError):
    pass


def upload_seed_dataset(client: Client, tasks: List[Dict[str, Any]], seed: int):
    name = f"curation-plasticity-real-heldout-id-seed-{seed}"
    try:
        return client.datasets.get_dataset(dataset=name)
    except Exception:
        rows = []
        for task in tasks:
            text = str(task["instruction"])
            if _CATEGORY_TERMS.search(text):
                raise ValueError(f"Task {task['task_id']} contains a category label; refusing Phoenix upload")
            rows.append({"task_id": str(task["task_id"]),
                         "task_text": TASK_SECRET.sub("[REDACTED]", text)})
        return client.datasets.create_dataset(
            name=name,
            dataset_description="Seed-specific held-out task text and opaque IDs only.",
            inputs=rows,
            example_id_key="task_id",
        )


def log_heldout_experiment(client: Client, dataset: Any, arm: str, seed: int,
                           epoch: int, rows: Dict[str, Dict[str, Any]],
                           run_tag: Optional[str] = None) -> None:
    tag_suffix = f"-{run_tag}" if run_tag else ""
    name = f"{arm}-seed-{seed}{tag_suffix}-epoch-{epoch}"
    try:
        existing = client.experiments.list(dataset_id=dataset.id)
        if any(exp.get("name") == name for exp in existing):
            return
    except Exception:
        pass
    run_saved_experiment(client, dataset, arm=arm, seed=seed, epoch=epoch, rows=rows)


def run_one(arm: str, seed: int, config_path: Path, epochs: int = EPOCHS,
            eval_epochs: Optional[List[int]] = None,
            run_tag: Optional[str] = None) -> None:
    if epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if eval_epochs is None:
        eval_epoch_set = {e for e in HELDOUT_EPOCHS if e <= epochs} | {0, epochs}
    else:
        eval_epoch_set = set(eval_epochs)
        if any(e < 0 or e > epochs for e in eval_epoch_set):
            raise ValueError(f"eval epochs must be between 0 and --epochs ({epochs})")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    key_env = config.get("API_KEY_ENV_VAR", "NVIDIA_API_KEY")
    if not os.getenv(key_env):
        raise SystemExit(f"STOP: {key_env} is not set; real mode does not fall back to mock.")
    for model_id in (ACTOR_MODEL, OPTIMIZER_MODEL):
        if model_id not in config.get("REFERENCE_PRICE_TABLE", {}):
            raise SystemExit(f"Missing config.yaml hypothetical price for {model_id}")
    if config.get("ACTOR", {}).get("model_id") != ACTOR_MODEL:
        raise SystemExit(f"config ACTOR.model_id must be {ACTOR_MODEL}")
    if config.get("OPTIMIZER", {}).get("model_id") != OPTIMIZER_MODEL:
        raise SystemExit(f"config OPTIMIZER.model_id must be {OPTIMIZER_MODEL}")

    init_phoenix_tracer(force=True)
    if run_tag is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", run_tag):
        raise ValueError("--run-tag may contain only letters, numbers, hyphens, and underscores")
    run_suffix = f"-{run_tag}" if run_tag else ""
    root = Path("outputs/real_experiment") / f"seed-{seed}-{arm}{run_suffix}"
    logs = Path("logs/real_experiment") / f"seed-{seed}-{arm}{run_suffix}"
    state_path = root / "checkpoint.json"
    results_path = root / "results.json"
    calls_path = logs / "calls.jsonl"
    root.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    train, heldout_id, _, labels = generate_task_suites(
        noise_frac=0.20, seed=seed, output_dir=str(root / "task_data"),
        num_train=TRAIN_POOL_SIZE)
    if len(train) != TRAIN_POOL_SIZE or len(heldout_id) != 40:
        raise RuntimeError("Generated task suite did not match 20 train / 40 held-out ID.")
    actor = RealLLMActor(LoggedNVIDIAClient(config, calls_path), ACTOR_MODEL,
                         max_steps=config.get("MAX_STEPS_PER_EPISODE", 12))
    actor.client.current_phase = "resume"
    old_events = read_jsonl(calls_path)
    actor.client.call_logs = [row for row in old_events if row.get("event") == "call"]
    actor.client.retry_events = [row for row in old_events if row.get("event") == "retry"]
    actor.client.total_hypothetical_usd = sum(
        float(row.get("hypothetical_cost_usd") or 0.0) for row in actor.client.call_logs)

    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    playbook = restore_playbook(state, config.get("MAX_PLAYBOOK_TOKENS", 2000))
    histories = state.get("task_histories", {t["task_id"]: [] for t in train})
    skip_until = state.get("skip_until", {})
    clean_success_ids = state.get("clean_success_ids", [])
    curator_decisions = state.get("curator_decisions", [])
    curator_events = state.get("curator_events", [])
    reflector_events = state.get("reflector_events", [])
    reflector_acceptances = state.get("reflector_acceptances", [])
    reflector_noops = state.get("reflector_noops", [])
    api_failures = state.get("api_failures", [])
    epochs_data = state.get("epochs", [])
    heldout_by_epoch = state.get("heldout_by_epoch", {})
    completed = int(state.get("completed_epoch", 0))
    grader = Grader(seed=seed)
    if state.get("grader_rng_state"):
        grader.rng.setstate(as_tuple(state["grader_rng_state"]))
    phoenix_client = Client(base_url="http://localhost:6006")
    dataset = upload_seed_dataset(phoenix_client, heldout_id, seed)

    training_episode_count = epochs * min(BATCH_SIZE, len(train))
    evaluation_episode_count = len(heldout_id) * len(eval_epoch_set)
    actor_episode_count = training_episode_count + evaluation_episode_count
    expected_actor_calls = round(actor_episode_count * 4.4)
    reflector_call_ceiling = epochs * 2
    gate_episode_ceiling = training_episode_count
    expected_gate_calls = round(gate_episode_ceiling * 4.4)
    estimated_calls_with_gate_ceiling = expected_actor_calls + reflector_call_ceiling + expected_gate_calls
    print(
        "CALL_ESTIMATE "
        f"arm={arm} seed={seed} epochs={epochs} train_pool={len(train)} batch={BATCH_SIZE} "
        f"eval_epochs={','.join(map(str, sorted(eval_epoch_set)))} "
        f"actor_episodes={actor_episode_count} expected_actor_calls~{expected_actor_calls} "
        f"reflector_calls<={reflector_call_ceiling} "
        f"expected_calls_without_gate~{expected_actor_calls + reflector_call_ceiling} "
        f"optional_gate_episodes<={gate_episode_ceiling} optional_gate_calls~<={expected_gate_calls} "
        f"estimated_calls_with_gate_ceiling~<={estimated_calls_with_gate_ceiling} "
        f"rate_limit_rpm={config.get('RATE_LIMIT_RPM', 35)} workers={EPISODE_WORKERS}"
    )

    def save_state() -> None:
        nonlocal epochs_data, heldout_by_epoch
        current_costs = phase_cost_totals(actor.client.call_logs)
        snapshot = {
            "schema_version": 1, "arm": arm, "seed": seed,
            "actor_model": ACTOR_MODEL, "optimizer_model": OPTIMIZER_MODEL,
            "execution_mode": "real", "cost_label": "hypothetical pricing",
            "total_epochs": epochs, "eval_epochs": sorted(eval_epoch_set),
            "completed_epoch": completed,
            "playbook": {"rules": playbook.rules, "next_rule_num": playbook.next_rule_num,
                         "max_tokens": playbook.max_tokens},
            "task_histories": histories, "skip_until": skip_until,
            "clean_success_ids": clean_success_ids,
            "curator_decisions": curator_decisions, "curator_events": curator_events,
            "reflector_events": reflector_events, "api_failures": api_failures,
            "reflector_acceptances": reflector_acceptances,
            "reflector_noops": reflector_noops,
            "epochs": epochs_data, "heldout_by_epoch": heldout_by_epoch,
            "grader_rng_state": grader.rng.getstate(),
            "phase_costs_hypothetical": current_costs,
            "total_hypothetical_usd": actor.client.total_hypothetical_usd,
            "raw_call_log": str(calls_path),
        }
        atomic_json(state_path, snapshot)
        atomic_json(results_path, snapshot)

    # Baseline checkpoint and held-out eval at epoch zero.
    if 0 in eval_epoch_set and "0" not in heldout_by_epoch:
        heldout_by_epoch["0"] = evaluate_heldout(
            actor, heldout_id, playbook, labels, grader, arm, seed, 0, api_failures)
        save_state()
    for checkpoint_epoch in sorted(eval_epoch_set):
        rows = heldout_by_epoch.get(str(checkpoint_epoch))
        if rows is not None:
            log_heldout_experiment(phoenix_client, dataset, arm, seed, checkpoint_epoch,
                                   rows, run_tag=run_tag)

    for epoch in range(completed + 1, epochs + 1):
        epoch_started = time.monotonic()
        phase_cost_before = phase_cost_totals(actor.client.call_logs)
        rng = random.Random(seed * 10_000 + epoch)
        eligible = [t for t in train if int(skip_until.get(t["task_id"], 0)) <= epoch]
        if not eligible:
            eligible = train
        selected = rng.sample(eligible, min(BATCH_SIZE, len(eligible)))
        batch_results: Dict[str, bool] = {}
        failed: List[Dict[str, Any]] = []
        training_regressions = 0
        episode_rows = []
        with span("training_epoch", phase="epoch", attributes={
            "arm": arm, "seed": seed, "epoch": epoch, "execution_mode": "real",
        }):
            actor.client.current_phase = "act"
            training_results = run_episodes_concurrently(
                actor, selected, playbook, {"arm": arm, "seed": seed, "epoch": epoch},
                api_failures, phase="act")
            for task in selected:
                task_id = task["task_id"]
                previous_passes = [h for h in histories.get(task_id, []) if h.get("passed")]
                result = training_results[task_id]
                category = labels[task_id]["category"]
                passed, feedback = grader.grade(task, category, result["trajectory"], result["final_answer"])
                passed = bool(passed and not result.get("api_failure")
                              and not result.get("format_errors") and not result.get("ended_by_step_cap"))
                if not passed and previous_passes:
                    training_regressions += 1
                row = {"epoch": epoch, "passed": passed, "feedback": feedback,
                       "steps_used": result["steps_used"],
                       "read_playbook": result["read_playbook_called"],
                       "trajectory": result["trajectory"],
                       "prior_successful_trajectories": [h["trajectory"] for h in previous_passes]}
                histories.setdefault(task_id, []).append(row)
                batch_results[task_id] = passed
                episode_rows.append({"task_id": task_id, "passed": passed,
                                     "steps_used": result["steps_used"],
                                     "playbook_read": result["read_playbook_called"],
                                     "api_failure": bool(result.get("api_failure"))})
                if passed and task_id not in clean_success_ids:
                    clean_success_ids.append(task_id)
                if not passed:
                    failed.append({"task_id": task_id, "instruction": task["instruction"],
                                   "feedback": feedback, "trajectory": result["trajectory"],
                                   "prior_successful_trajectories": row["prior_successful_trajectories"]})

            selected_ids = [t["task_id"] for t in selected]
            curator_skips = []
            anchors = []
            tasks_to_reflect = failed
            if arm == "A3":
                actor.client.current_phase = "curate"
                tasks_to_reflect = []
                failed_by_id = {f["task_id"]: f for f in failed}
                for task_obj in selected:
                    task_id = task_obj["task_id"]
                    history = histories[task_id]
                    if len(history) < 2:
                        decision = {"task_id": task_id, "decision": "run", "label": "Uncertain",
                                    "skip_epochs": 1, "anchor_success_id": None,
                                    "reason": "Underexplored; collecting another trajectory.",
                                    "called_llm": False}
                    else:
                        trusted = [tid for tid in clean_success_ids
                                   if sum(1 for h in histories.get(tid, []) if h.get("passed")) >= 2]
                        decision = call_curator(actor.client, task=task_obj, history=history,
                                                trusted_success_ids=trusted, arm=arm, seed=seed,
                                                epoch=epoch, optimizer_model=OPTIMIZER_MODEL)
                    curator_decisions.append(decision)
                    curator_events.append(decision)
                    if decision["decision"] == "skip":
                        skip_until[task_id] = epoch + decision.get("skip_epochs", 1) + 1
                        curator_skips.append(task_id)
                        continue
                    failure = failed_by_id.get(task_id)
                    if failure is None:
                        continue
                    tasks_to_reflect.append(failure)
                    anchor_id = decision.get("anchor_success_id")
                    if anchor_id and anchor_id in clean_success_ids:
                        success_history = [h for h in histories.get(anchor_id, []) if h.get("passed")]
                        if len(success_history) >= 2:
                            anchor_task = next((t for t in train if t["task_id"] == anchor_id), None)
                            if anchor_task:
                                anchors.append({"task_id": anchor_id,
                                                "instruction": anchor_task["instruction"],
                                                "successful_trajectories": [h["trajectory"] for h in success_history[-2:]]})

            reflect_rejections = []
            accepted_ops: List[Dict[str, Any]] = []
            candidate = playbook.clone()
            gate_info: Dict[str, Any] = {"attempted": False, "accepted": False,
                                         "regressed_task_ids": []}
            if tasks_to_reflect:
                accepted_ops, reflect_rejections, trace_records = call_reflector(
                    actor.client, tasks_to_reflect, playbook, anchors, histories,
                    arm, seed, epoch, OPTIMIZER_MODEL)
                for event in reflect_rejections:
                    reflector_events.append(event)
                source_ids = list(dict.fromkeys(
                    op.get("evidence", {}).get("task_id") for op in accepted_ops
                    if op.get("op") in {"add", "modify"} and isinstance(op.get("evidence"), dict)))
                valid = candidate.apply_ops(accepted_ops, epoch=epoch, source_task_ids=source_ids)
                if accepted_ops and not valid:
                    gate_rejection = {"epoch": epoch, "attempt": 0, "kind": "gate",
                                      "reason": "candidate exceeds playbook token cap"}
                    reflector_events.append(gate_rejection)
                    reflect_rejections.append(gate_rejection)
                if valid and accepted_ops:
                    gate_info["attempted"] = True
                    gate_outcomes = {}
                    actor.client.current_phase = "gate"
                    with span("gate_rerun", phase="gate", attributes={
                        "arm": arm, "seed": seed, "epoch": epoch,
                        "execution_mode": "real",
                    }):
                        actor.client.current_phase = "gate"
                        gate_results = run_episodes_concurrently(
                            actor, selected, candidate,
                            {"arm": arm, "seed": seed, "epoch": epoch},
                            api_failures, phase="gate")
                        for task in selected:
                            gate_result = gate_results[task["task_id"]]
                            cat = labels[task["task_id"]]["category"]
                            passed, _ = grader.grade(task, cat, gate_result["trajectory"],
                                                     gate_result["final_answer"])
                            gate_outcomes[task["task_id"]] = bool(
                                passed and not gate_result.get("api_failure")
                                and not gate_result.get("format_errors")
                                and not gate_result.get("ended_by_step_cap"))
                    before_score = sum(batch_results.values())
                    after_score = sum(gate_outcomes.values())
                    gate_info.update({"current_score": before_score, "candidate_score": after_score,
                                      "accepted": after_score >= before_score,
                                      "regressed_task_ids": [tid for tid, ok in batch_results.items()
                                                              if ok and not gate_outcomes.get(tid, False)]})
                    gate_info["accepted_but_regressed"] = bool(
                        gate_info["accepted"] and gate_info["regressed_task_ids"])
                    if gate_info["accepted"]:
                        playbook = candidate
                        reflector_acceptances.append({
                            "epoch": epoch, "ops_count": len(accepted_ops),
                            "reason": f"grounding passed; gate score {after_score} >= {before_score}",
                        })
                    else:
                        gate_rejection = {"epoch": epoch, "kind": "gate",
                                          "reason": "candidate batch score below baseline",
                                          "current_score": before_score,
                                          "candidate_score": after_score}
                        reflector_events.append(gate_rejection)
                        reflect_rejections.append(gate_rejection)
                elif not accepted_ops and not any(r.get("kind") in {"api", "format", "grounding"}
                                                   for r in reflect_rejections):
                    reflector_noops.append({
                        "epoch": epoch, "ops_count": 0,
                        "reason": "valid empty operation list; no-op, no rules changed",
                    })

        epoch_costs = phase_cost_totals(actor.client.call_logs)
        epoch_cost_delta = {phase: round(epoch_costs[phase] - phase_cost_before.get(phase, 0.0), 6)
                            for phase in epoch_costs}
        epoch_record = {
            "arm": arm, "seed": seed, "epoch": epoch, "selected_task_ids": selected_ids,
            "episodes": episode_rows, "curator_skipped_task_ids": curator_skips,
            "anchors_supplied": [a["task_id"] for a in anchors],
            "reflector_ops_accepted": accepted_ops, "reflector_rejections": reflect_rejections,
            "reflector_rejection_counts": {
                "grounding": sum(r.get("kind") == "grounding" for r in reflect_rejections),
                "gate": sum(r.get("kind") == "gate" for r in reflect_rejections),
                "api": sum(r.get("kind") == "api" for r in reflect_rejections),
                "format": sum(r.get("kind") == "format" for r in reflect_rejections),
            },
            "reflector_acceptances": [r for r in reflector_acceptances if r.get("epoch") == epoch],
            "reflector_noops": [r for r in reflector_noops if r.get("epoch") == epoch],
            "gate": gate_info, "training_regressions": training_regressions,
            "pollution": playbook.compute_pollution(labels),
            "playbook": {"rules": playbook.rules, "next_rule_num": playbook.next_rule_num},
            "phase_costs_hypothetical": epoch_cost_delta,
            "cumulative_phase_costs_hypothetical": epoch_costs,
        }
        epochs_data = [e for e in epochs_data if e.get("epoch") != epoch] + [epoch_record]
        epochs_data.sort(key=lambda row: row["epoch"])
        completed = epoch
        if epoch in eval_epoch_set:
            heldout_by_epoch[str(epoch)] = evaluate_heldout(
                actor, heldout_id, playbook, labels, grader, arm, seed, epoch, api_failures)
        final_epoch_costs = phase_cost_totals(actor.client.call_logs)
        epoch_record["cumulative_phase_costs_hypothetical"] = final_epoch_costs
        epoch_record["phase_costs_hypothetical"] = {
            phase: round(final_epoch_costs[phase] - phase_cost_before.get(phase, 0.0), 6)
            for phase in final_epoch_costs
        }
        save_state()
        if epoch in eval_epoch_set:
            log_heldout_experiment(phoenix_client, dataset, arm, seed, epoch,
                                   heldout_by_epoch[str(epoch)], run_tag=run_tag)
        epoch_wall_sec = time.monotonic() - epoch_started
        print(f"CHECKPOINT arm={arm} seed={seed} epoch={epoch} "
              f"wall_sec={epoch_wall_sec:.1f} cost_usd={actor.client.total_hypothetical_usd:.6f}")

    flush_phoenix()
    print(f"RESULTS {results_path}")


def evaluate_heldout(actor: RealLLMActor, tasks: List[Dict[str, Any]], playbook: Playbook,
                     labels: Dict[str, Any], grader: Grader, arm: str, seed: int,
                     epoch: int, api_failures: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    result_rows: Dict[str, Dict[str, Any]] = {}
    actor.client.current_phase = "eval"
    eval_results = run_episodes_concurrently(
        actor, tasks, playbook, {"arm": arm, "seed": seed, "epoch": epoch},
        api_failures, phase="eval")
    for task in tasks:
        with span("heldout_eval", phase="eval", attributes={
            "arm": arm, "seed": seed, "epoch": epoch, "task_id": task["task_id"],
            "model_id": ACTOR_MODEL, "execution_mode": "real",
        }):
            result = eval_results[task["task_id"]]
            passed, _ = grader.grade(task, labels[task["task_id"]]["category"],
                                     result["trajectory"], result["final_answer"])
            passed = bool(passed and not result.get("api_failure")
                          and not result.get("format_errors")
                          and not result.get("ended_by_step_cap"))
            result_rows[task["task_id"]] = {
                "success": passed, "steps_used": result["steps_used"],
                "playbook_read": bool(result["read_playbook_called"]),
            }
    return result_rows


def phase_cost_totals(calls: List[Dict[str, Any]]) -> Dict[str, float]:
    totals = {phase: 0.0 for phase in ("act", "curate", "reflect", "gate", "eval")}
    for call in calls:
        phase = call.get("phase")
        if phase in totals:
            totals[phase] += float(call.get("hypothetical_cost_usd") or 0.0)
    return {k: round(v, 6) for k, v in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("A0", "A0-raw", "A3"))
    parser.add_argument("--seed", required=True, type=int, choices=(42, 43))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--eval-epochs", type=lambda value: [int(x) for x in value.split(",") if x.strip()],
                        default=None, help="comma-separated held-out evaluation epochs (for example 0,3)")
    parser.add_argument("--run-tag", help="optional namespace for fresh output/log folders and Phoenix experiment names")
    args = parser.parse_args()
    run_one(args.arm, args.seed, args.config, epochs=args.epochs,
            eval_epochs=args.eval_epochs, run_tag=args.run_tag)


if __name__ == "__main__":
    main()
