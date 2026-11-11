"""
Real A3 Smoke Test Runner:
- Seed 42, 4 epochs, 10 train tasks, 5 held-out tasks.
- Actor: nvidia/nemotron-3.5-bolt-e4b
- Optimizer: meta/llama-3.2-90b-vision-instruct
- Strict $1.00 budget cap.
- Collects raw unedited call logs, raw Curator/Reflector prompts and outputs.
- Captures rejected Reflector attempts and exact reasons.
- Security scanner: grep nvapi- and scan forbidden terms.
"""
import os
import argparse
import yaml
import json
import time
import threading
from pathlib import Path
from src.testbed.task_gen import generate_task_suites
from src.testbed.mock_api import MockAPI
from src.testbed.grader import Grader
from src.loop.provenance import Playbook
from src.agents.nvidia_client import NVIDIAAPIError, NVIDIAClient
from src.agents.real_llm_actor import RealLLMActor
from src.analysis.leak_scanner import scan_prompts_and_tasks_for_leaks, FIXED_CURATOR_TAXONOMY
from src.analysis.metrics import compute_curator_confusion
from src.analysis.reflection_grounding import make_reflection_trace_records, check_rule_grounding

ACTOR_MODEL = "nvidia/nemotron-3.5-bolt-e4b"
OPTIMIZER_MODEL = "meta/llama-3.2-90b-vision-instruct"
SEED = 42
EPOCHS = 4
TRAIN_TASK_COUNT = 10
HELDOUT_TASK_COUNT = 5
BATCH_SIZE = 5


class LoggedNVIDIAClient(NVIDIAClient):
    """Append each raw real call immediately; credentials are never logged."""
    def __init__(self, config, log_path):
        super().__init__(config)
        self.log_path = Path(log_path)
        self.current_phase = "unassigned"
        self._log_lock = threading.Lock()

    def chat_completion(self, *args, **kwargs):
        try:
            result = super().chat_completion(*args, **kwargs)
            self._append_new_retries()
            result["call_record"]["phase"] = self.current_phase
            record = dict(result["call_record"])
            if record.get("execution_mode") != "real":
                raise RuntimeError("Safety stop: call was not marked execution_mode=real")
            self._append({"event": "call", **record})
            print("RAW_CALL " + json.dumps({k: record[k] for k in (
                "model_id", "request_id", "input_tokens", "output_tokens",
                "reasoning_tokens", "finish_reason", "latency_sec", "cost_label", "hypothetical_cost_usd"
            )}, ensure_ascii=False))
            return result
        except Exception as exc:
            self._append_new_retries()
            key = os.environ.get(self.key_env_var, "")
            message = str(exc).replace(key, "[REDACTED]") if key else str(exc)
            self._append({"event": "call_error", "phase": self.current_phase,
                          "model_id": kwargs.get("model_id", args[0] if args else None),
                          "request_id": None, "input_tokens": None, "output_tokens": None,
                          "reasoning_tokens": None, "finish_reason": None, "error": message})
            if isinstance(exc, NVIDIAAPIError):
                raise NVIDIAAPIError(message) from None
            raise RuntimeError(message) from None

    def _append_new_retries(self):
        for retry in self._take_thread_retry_events():
            self._append({"event": "retry", "phase": self.current_phase, **retry})

    def _append(self, item):
        with self._log_lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(item, ensure_ascii=False)
            key = os.environ.get(self.key_env_var, "")
            if key and key in serialized:
                raise RuntimeError("Safety stop: refusing to persist a record containing key material")
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(serialized + "\n")


def chat_json_with_one_retry(client, model_id, messages, temperature, max_tokens, validator, kind):
    """Make at most two billed calls for a truncated or invalid JSON response."""
    working_messages = [dict(message) for message in messages]
    attempts = []
    final_response = None
    last_error = None
    for attempt in range(1, 3):
        response = client.chat_completion(
            model_id=model_id, messages=working_messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        final_response = response
        raw = (response.get("text") or "").strip()
        finish_reason = response["call_record"].get("finish_reason")
        parsed = None
        if finish_reason == "length":
            last_error = "Truncated response (finish_reason=length)."
        else:
            clean = raw
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1]
                if clean.endswith("```"):
                    clean = clean.rsplit("```", 1)[0]
            try:
                parsed = json.loads(clean.strip())
                if not validator(parsed):
                    raise ValueError(f"Invalid {kind} JSON structure.")
                attempts.append({
                    "attempt": attempt, "prompt_messages": working_messages,
                    "raw_output": raw, "finish_reason": finish_reason,
                })
                return parsed, response, attempts, None
            except Exception as exc:
                last_error = f"Invalid {kind} JSON: {exc}"
        attempts.append({
            "attempt": attempt, "prompt_messages": working_messages,
            "raw_output": raw, "finish_reason": finish_reason,
            "rejection_reason": last_error,
        })
        if attempt == 1:
            working_messages.append({
                "role": "user",
                "content": f"{last_error} Return one complete valid {kind} JSON response only. Do not repeat any category names.",
            })
    return None, final_response, attempts, last_error


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def checkpoint_state(epoch, playbook, task_histories, clean_success_ids,
                     curator_skip_until, phase_costs, format_errors_by_phase,
                     truncations_by_phase, curator_samples, curator_decisions,
                     reflector_samples, rejected_reflector_attempts,
                     eval_success=0, completed_eval_ids=None,
                     grounding_rejections=0, gate_rejections=0):
    return {
        "completed_epochs": epoch,
        "playbook": {"max_tokens": playbook.max_tokens, "rules": playbook.rules,
                     "next_rule_num": playbook.next_rule_num},
        "task_histories": task_histories,
        "clean_success_ids": clean_success_ids,
        "curator_skip_until": curator_skip_until,
        "phase_costs": phase_costs,
        "format_errors_by_phase": format_errors_by_phase,
        "truncations_by_phase": truncations_by_phase,
        "curator_samples": curator_samples,
        "curator_decisions": curator_decisions,
        "reflector_samples": reflector_samples,
        "rejected_reflector_attempts": rejected_reflector_attempts,
        "reflector_rejection_counts": {
            "grounding_rejections": grounding_rejections,
            "gate_rejections": gate_rejections,
            "format_rejections": format_errors_by_phase["reflect"],
        },
        "eval_success": eval_success,
        "completed_eval_ids": completed_eval_ids or [],
        "reflector_rejections": {
            "grounding": grounding_rejections,
            "gate": gate_rejections,
            "format": format_errors_by_phase.get("reflect", 0),
        },
    }


def phase_cost_totals(call_logs):
    totals = {"act": 0.0, "curate": 0.0, "reflect": 0.0, "gate": 0.0, "eval": 0.0}
    for call in call_logs:
        phase = call.get("phase")
        if phase in totals:
            totals[phase] += call.get("hypothetical_cost_usd", 0.0) or 0.0
    return totals

CURATOR_PROMPT_TEMPLATE = """You are an expert diagnostic Curator for a self-improving AI agent.
Your job is to inspect an individual task's recent execution history and decide whether to RUN or SKIP this task in the upcoming optimization cycle, and optionally select at most ONE trusted past success as a contrastive anchor for the Reflector.

""" + FIXED_CURATOR_TAXONOMY + """

Decision Rules:
1. Skip ONLY on strong evidence from the repeated execution history; uncertainty means run.
2. A SKIP decision must NEVER carry an anchor (anchor_success_id must be null).
3. An anchor requires a genuinely correct, reproducible success corroborated by at least one other independent clean success. If none qualifies, abstain with anchor_success_id: null.
4. Output MUST be valid JSON with reason before decision.

Output JSON format:
{
  "reason": "Detailed analytical justification based purely on cited execution logs",
  "label": "One of the 8 labels",
  "decision": "run" or "skip",
  "skip_epochs": 1,
  "anchor_reason": "Reason for selecting or abstaining from anchor",
  "anchor_success_id": "success_task_id" or null,
  "supporting_success_ids": []
}"""

REFLECTOR_PROMPT_TEMPLATE = """You are the Reflector for this specific API. Use only behavior explicitly shown in the trace bundle supplied by the user message.

Grounding requirements:
- Propose a rule only when a supplied trace shows a failed HTTP request and its error body.
- For each failed task, use only its failed request, error body, and successful request(s), if shown. Do not use full trajectories or unrelated task details.
- Every add or modify operation must include evidence with task_id, exact failed_request_quote, exact error_body_quote, and successful_request_quote only when supplied (otherwise null). Evidence quotes belong in the evidence object, not the rule text.
- Rules must follow: "When <request/endpoint> returns <error text>, then <change field/header X to value Y>". Mention an exact header, parameter, or field from the failed request or error body.
- If the error explicitly names the needed change (such as a missing required header), that change is grounded by the error. Any proposed value or format not stated by the error must appear in an observed successful request. For other fixes, use a successful request as evidence; do not infer that an "invalid field" should be removed or reformatted.
- Reject any rule that contradicts a supplied successful request, including forbidding a header or body it used.
- Forbid generic REST conventions and any parameter, header, field, operator, pagination scheme, format, or behavior not demonstrated in the supplied traces.
- Do not invent request parameters or claim an API behavior from general knowledge.
- Rule IDs are assigned by the harness for add operations. Never provide a rule_id for add. For modify or delete, use only an existing ID shown in the current playbook.
- Keep the resulting playbook under 2,000 tokens.

Output a JSON array only, with no prose and no markdown fences. Allowed forms:
[{"op":"add","content":"When ... returns ..., then ...","evidence":{"task_id":"...","failed_request_quote":"...","error_body_quote":"...","successful_request_quote":null}}]
[{"op":"modify","rule_id":"existing rule ID","content":"When ... returns ..., then ...","evidence":{"task_id":"...","failed_request_quote":"...","error_body_quote":"...","successful_request_quote":"..."}}]
[{"op":"delete","rule_id":"existing rule ID"}]
An empty array is valid when the traces do not support a rule."""

def run_a3_smoke_test(resume=False):
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    from src.obs.phoenix_setup import init_phoenix_tracer
    init_phoenix_tracer()

    key_env = config.get("API_KEY_ENV_VAR", "NVIDIA_API_KEY")
    if not os.environ.get(key_env):
        raise SystemExit(f"STOP: {key_env} is not set; no mock fallback is permitted.")
    actor_model = ACTOR_MODEL
    optimizer_model = OPTIMIZER_MODEL
    if config["OPTIMIZER"]["model_id"] != optimizer_model:
        raise SystemExit(f"STOP: config optimizer must be {optimizer_model}.")
    for model_id in (actor_model, optimizer_model):
        if model_id not in config.get("REFERENCE_PRICE_TABLE", {}):
            raise SystemExit(f"STOP: config.yaml has no pricing entry for {model_id}.")

    Path("logs/a3").mkdir(parents=True, exist_ok=True)
    Path("outputs/a3").mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path("outputs/a3/active-checkpoint-seed-42.json")
    if resume:
        if not checkpoint_path.exists():
            raise SystemExit(f"STOP: no checkpoint found at {checkpoint_path}")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("actor_model") != actor_model or checkpoint.get("optimizer_model") != optimizer_model:
            raise SystemExit("STOP: checkpoint model configuration does not match this smoke test")
        if checkpoint.get("complete"):
            raise SystemExit("STOP: checkpoint is already complete; run without --resume for a fresh test")
        run_id = checkpoint["run_id"]
    else:
        if checkpoint_path.exists():
            existing = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if not existing.get("complete"):
                raise SystemExit(f"STOP: incomplete run exists; resume it with --resume ({checkpoint_path})")
        run_id = time.strftime("%Y%m%dT%H%M%S")
        checkpoint = {
            "schema_version": 1, "run_id": run_id, "actor_model": actor_model,
            "optimizer_model": optimizer_model, "seed": SEED, "complete": False,
            "completed_epochs": 0, "state": {},
            "call_log_path": f"logs/a3/calls-{run_id}.jsonl",
            "episode_log_path": f"outputs/a3/episodes-{run_id}.jsonl",
        }
        atomic_write_json(checkpoint_path, checkpoint)

    client = LoggedNVIDIAClient(config, checkpoint["call_log_path"])
    previous_log_rows = read_jsonl(client.log_path)
    client.call_logs = [row for row in previous_log_rows if row.get("event") == "call"]
    client.retry_events = [row for row in previous_log_rows if row.get("event") == "retry"]
    client.total_hypothetical_usd = sum(
        row.get("hypothetical_cost_usd", 0.0) or 0.0 for row in client.call_logs
    )
    episode_log_path = Path(checkpoint["episode_log_path"])
    episode_cache = {row["cache_key"]: row for row in read_jsonl(episode_log_path)}
    print(f"RAW_CALL_LOG_PATH {client.log_path}")
    print(f"EPISODE_CHECKPOINT_LOG_PATH {episode_log_path}")
    print(f"RESUME_MODE {resume}; COMPLETED_EPOCHS {checkpoint.get('completed_epochs', 0)}")

    actor = RealLLMActor(client=client, model_id=actor_model,
                         max_steps=config.get("MAX_STEPS_PER_EPISODE", 12))
    grader = Grader(seed=SEED)

    train_tasks, heldout_id, _, _ = generate_task_suites(noise_frac=0.20, seed=SEED, output_dir="data_smoke_real")
    # Read the saved analysis labels explicitly; these stay out of model prompts.
    with open("data_smoke_real/labels.json", encoding="utf-8") as f:
        analysis_labels = json.load(f)
    smoke_train = train_tasks[:TRAIN_TASK_COUNT]
    smoke_heldout = heldout_id[:HELDOUT_TASK_COUNT]

    playbook = Playbook(max_tokens=config.get("MAX_PLAYBOOK_TOKENS", 2000))

    task_histories = {t["task_id"]: [] for t in smoke_train}
    clean_success_ids = []
    curator_skip_until = {}

    phase_costs = {"act": 0.0, "curate": 0.0, "reflect": 0.0, "gate": 0.0, "eval": 0.0}
    format_errors_by_phase = {"act": 0, "curate": 0, "reflect": 0, "eval": 0}
    truncations_by_phase = {"act": 0, "curate": 0, "reflect": 0, "eval": 0}
    grounding_rejections = 0
    gate_rejections = 0

    curator_samples = []
    curator_decisions = []
    reflector_samples = []
    rejected_reflector_attempts = []
    eval_success = 0
    completed_eval_ids = set()

    state = checkpoint.get("state", {})
    if state:
        saved_playbook = state.get("playbook", {})
        playbook.rules = saved_playbook.get("rules", {})
        playbook.next_rule_num = saved_playbook.get("next_rule_num", 1)
        task_histories = state.get("task_histories", task_histories)
        clean_success_ids = state.get("clean_success_ids", [])
        curator_skip_until = state.get("curator_skip_until", {})
        phase_costs = state.get("phase_costs", phase_costs)
        format_errors_by_phase = state.get("format_errors_by_phase", format_errors_by_phase)
        truncations_by_phase = state.get("truncations_by_phase", truncations_by_phase)
        rejection_counts = state.get("reflector_rejections", {})
        grounding_rejections = rejection_counts.get("grounding", 0)
        gate_rejections = rejection_counts.get("gate", 0)
        curator_samples = state.get("curator_samples", [])
        curator_decisions = state.get("curator_decisions", [])
        reflector_samples = state.get("reflector_samples", [])
        rejected_reflector_attempts = state.get("rejected_reflector_attempts", [])
        eval_success = state.get("eval_success", 0)
        completed_eval_ids = set(state.get("completed_eval_ids", []))

    print("==================================================================")
    print("STARTING A3 SMOKE TEST (REAL API CALLS, NVIDIA HOSTED)")
    print(f"Actor Model:     {actor_model}")
    print(f"Optimizer Model: {optimizer_model}")
    print(f"Seed/Epochs/Train/Held-out: {SEED}/{EPOCHS}/{len(smoke_train)}/{len(smoke_heldout)}")
    print(f"Budget Cap:      ${config.get('MAX_USD', 1.00):.2f} (Hypothetical Pricing)")
    print("==================================================================")

    for epoch in range(checkpoint.get("completed_epochs", 0) + 1, EPOCHS + 1):
        print(f"\n--- EPOCH {epoch} ---")

        # Rotate through all 10 training tasks in 5-task batches.
        eligible = [t for t in smoke_train if curator_skip_until.get(t["task_id"], 0) <= epoch]
        if len(eligible) <= BATCH_SIZE:
            selected = eligible
        else:
            start = ((epoch - 1) * BATCH_SIZE) % len(eligible)
            selected = (eligible[start:] + eligible[:start])[:BATCH_SIZE]

        # 2. Act
        client.current_phase = "act"
        batch_outcomes = {}
        failed_tasks_info = []
        cost_before_act = client.total_hypothetical_usd

        for t in selected:
            t_id = t["task_id"]
            cache_key = f"train:{epoch}:{t_id}"
            cached = episode_cache.get(cache_key)
            if cached:
                run_res, passed, fb = cached["run_result"], cached["passed"], cached["feedback"]
            else:
                cat = analysis_labels[t_id]["category"]
                traced_task = dict(t, _trace_context={"arm": "A3", "seed": SEED,
                                                     "epoch": epoch, "phase": "act"})
                run_res = actor.run_episode(traced_task, playbook, MockAPI())
                passed, fb = grader.grade(t, cat, run_res["trajectory"], run_res["final_answer"])
                passed = bool(passed and not run_res["format_errors"] and not run_res["ended_by_step_cap"])
                episode_event = {
                    "cache_key": cache_key, "task_id": t_id, "epoch": epoch,
                    "run_result": run_res, "passed": passed, "feedback": fb,
                }
                serialized_episode = json.dumps(episode_event, ensure_ascii=False)
                key = os.environ.get(key_env, "")
                if key and key in serialized_episode:
                    raise RuntimeError("Safety stop: refusing to persist an episode containing key material")
                append_jsonl(episode_log_path, episode_event)
                episode_cache[cache_key] = episode_event
            format_errors_by_phase["act"] += run_res["format_errors"]
            truncations_by_phase["act"] += run_res["truncations"]
            batch_outcomes[t_id] = passed

            task_histories[t_id].append({
                "epoch": epoch,
                "passed": passed,
                "feedback": fb,
                "actions": [s.get("action") for s in run_res["trajectory"]],
                "trajectory": run_res["trajectory"],
            })

            # Success IDs are derived from outcomes only; task labels do not
            # affect prompt contents or candidate selection.
            if passed:
                if t_id not in clean_success_ids:
                    clean_success_ids.append(t_id)
            else:
                failed_tasks_info.append({
                    "task_id": t_id,
                    "instruction": t["instruction"],
                    "feedback": fb,
                    "trajectory": run_res["trajectory"]
                })

        phase_costs = phase_cost_totals(client.call_logs)

        # 3. Curate (A3)
        client.current_phase = "curate"
        cost_before_curate = client.total_hypothetical_usd
        tasks_to_reflect = []
        anchors = []

        for f_info in failed_tasks_info:
            f_id = f_info["task_id"]
            t_obj = next(t for t in selected if t["task_id"] == f_id)
            hist = task_histories[f_id]

            if len(hist) < 2:
                # Underexplored-first: run without curator call
                tasks_to_reflect.append(f_info)
                continue

            # Build Curator prompt
            history_str = ""
            for h in hist[-2:]:
                history_str += f"- Run (Epoch {h['epoch']}): Outcome: {'Passed' if h['passed'] else 'Failed'}\n"
                history_str += f"  Feedback: {h['feedback']}\n"
                history_str += f"  Actions: {json.dumps(h['actions'])}\n"

            curator_user_content = f"Task ID: {f_id}\nInstruction: {t_obj['instruction']}\nAvailable Success IDs: {json.dumps(clean_success_ids)}\n\nExecution History:\n{history_str}\nOutput diagnostic JSON:"
            curator_messages = [
                {"role": "system", "content": CURATOR_PROMPT_TEMPLATE},
                {"role": "user", "content": curator_user_content}
            ]

            from src.obs.spans import span
            with span("curator_decision", phase="curate", attributes={
                "arm": "A3", "seed": SEED, "epoch": epoch, "task_id": f_id,
                "model_id": optimizer_model, "execution_mode": "real",
            }):
                cur_json, cur_resp, cur_attempts, cur_error = chat_json_with_one_retry(
                    client, optimizer_model, curator_messages, temperature=0.0,
                    max_tokens=1200,
                    validator=lambda value: isinstance(value, dict) and value.get("decision") in {"run", "skip"},
                    kind="Curator",
                )
            if cur_json is None:
                format_errors_by_phase["curate"] += 1
                cur_json = {"decision": "run", "reason": cur_error or "Malformed Curator response.",
                            "label": "Uncertain", "anchor_success_id": None}
            raw_cur = (cur_resp.get("text") or "").strip()
            truncations_by_phase["curate"] += sum(
                attempt.get("finish_reason") == "length" for attempt in cur_attempts
            )
            curator_decisions.append({"task_id": f_id, "decision": cur_json["decision"]})

            curator_samples.append({
                "task_id": f_id,
                "epoch": epoch,
                "prompt_messages": curator_messages,
                "attempts": cur_attempts,
                "raw_output": raw_cur,
                "finish_reason": cur_resp["call_record"].get("finish_reason"),
                "truncated": cur_resp["call_record"].get("finish_reason") == "length",
                "parsed_json": cur_json
            })

            if cur_json.get("decision") == "skip":
                curator_skip_until[f_id] = epoch + cur_json.get("skip_epochs", 1)
            else:
                tasks_to_reflect.append(f_info)
                if cur_json.get("anchor_success_id"):
                    anchors.append({"task_id": cur_json["anchor_success_id"]})

        phase_costs = phase_cost_totals(client.call_logs)

        # 4. Reflect
        client.current_phase = "reflect"
        cost_before_reflect = client.total_hypothetical_usd
        if tasks_to_reflect:
            trace_records = make_reflection_trace_records(tasks_to_reflect, task_histories)
            failure_trace_bundle = []
            for failure in tasks_to_reflect:
                history = task_histories.get(failure["task_id"], [])
                previous_successes = [
                    {"epoch": item.get("epoch"), "trajectory": item.get("trajectory", [])}
                    for item in history if item.get("passed")
                ]
                failure_trace_bundle.append({
                    "task_id": failure["task_id"],
                    "instruction": failure["instruction"],
                    "grader_feedback": failure["feedback"],
                    "failed_trajectory": failure["trajectory"],
                    "prior_successful_trajectories": previous_successes,
                })
            reflector_user_content = (
                f"Current Playbook:\n{playbook.to_markdown()}\n\n"
                "Trace bundle (use these exact request and response strings as evidence):\n"
                + json.dumps({"traces": failure_trace_bundle,
                              "evidence_records": trace_records}, ensure_ascii=False, indent=2)
            )
            reflector_messages = [
                {"role": "system", "content": REFLECTOR_PROMPT_TEMPLATE},
                {"role": "user", "content": reflector_user_content}
            ]

            # One retry after a malformed, truncated, or rejected reflection.
            candidate_playbook = playbook.clone()
            reflect_accepted = False

            for attempt in range(1, 3):
                rejection_kind = "format"
                attempt_messages = [dict(message) for message in reflector_messages]
                from src.obs.spans import span
                with span("reflector_call", phase="reflect", attributes={
                    "arm": "A3", "seed": SEED, "epoch": epoch,
                    "model_id": optimizer_model, "execution_mode": "real",
                }):
                    ref_resp = client.chat_completion(
                        model_id=optimizer_model,
                        messages=reflector_messages,
                        temperature=0.2 * attempt,
                        max_tokens=1600
                    )
                raw_ref = (ref_resp.get("text") or "").strip()
                clean_ref = raw_ref
                if clean_ref.startswith("```"):
                    clean_ref = clean_ref.split("\n", 1)[-1]
                    if clean_ref.endswith("```"):
                        clean_ref = clean_ref.rsplit("```", 1)[0]

                try:
                    if ref_resp["call_record"].get("finish_reason") == "length":
                        raise ValueError("Truncated response (finish_reason=length).")
                    ops = json.loads(clean_ref.strip())
                    if not isinstance(ops, list) or not all(
                        isinstance(op, dict) and op.get("op") in {"add", "modify", "delete"}
                        for op in ops
                    ):
                        raise ValueError("Output must be a JSON list of operations with 'op' key.")
                    if any(op.get("op") == "add" and "rule_id" in op for op in ops):
                        raise ValueError("The harness assigns rule IDs; add operations must omit rule_id.")

                    current_rule_ids = set(candidate_playbook.rules)
                    if any(op.get("op") in {"modify", "delete"}
                           and op.get("rule_id") not in current_rule_ids for op in ops):
                        rejection_kind = "gate"
                        gate_rejections += 1
                        raise ValueError("Operation references a rule ID not present in the current playbook.")

                    grounding_issues = check_rule_grounding(ops, trace_records)
                    if grounding_issues:
                        grounding_rejections += 1
                        rejection_kind = "grounding"
                        raise ValueError("Mechanical grounding check rejected rule(s): "
                                         + json.dumps(grounding_issues, ensure_ascii=False))

                    rejection_kind = "gate"
                    source_ids = list(dict.fromkeys(
                        op.get("evidence", {}).get("task_id")
                        for op in ops
                        if op.get("op") in {"add", "modify"}
                        and isinstance(op.get("evidence"), dict)
                    ))
                    with span("gate_rerun", phase="gate", attributes={
                        "arm": "A3", "seed": SEED, "epoch": epoch,
                        "execution_mode": "real",
                    }):
                        valid = candidate_playbook.apply_ops(ops, epoch=epoch, source_task_ids=source_ids)
                    if not valid:
                        gate_rejections += 1
                        raise ValueError("Ops violated 2,000 token budget cap.")

                    rejection_kind = "format"

                    reflector_samples.append({
                        "epoch": epoch,
                        "attempt": attempt,
                        "prompt_messages": attempt_messages,
                        "raw_output": raw_ref,
                        "finish_reason": ref_resp["call_record"].get("finish_reason"),
                        "parsed_ops": ops,
                        "accepted_reason": "Valid JSON operations passed the playbook token cap."
                    })
                    reflect_accepted = True
                    break

                except Exception as e:
                    if ref_resp["call_record"].get("finish_reason") == "length":
                        truncations_by_phase["reflect"] += 1
                    if attempt == 2 and rejection_kind == "format":
                        format_errors_by_phase["reflect"] += 1
                    rejected_reflector_attempts.append({
                        "epoch": epoch,
                        "attempt": attempt,
                        "prompt_messages": attempt_messages,
                        "reason": str(e),
                        "rejection_kind": rejection_kind,
                        "finish_reason": ref_resp["call_record"].get("finish_reason"),
                        "raw_output": raw_ref
                    })
                    reflector_messages.append({"role": "assistant", "content": raw_ref})
                    reflector_messages.append({"role": "user", "content": f"Attempt {attempt} rejected: {str(e)}. Please output valid JSON list of ops only."})

            if reflect_accepted:
                playbook = candidate_playbook

        phase_costs = phase_cost_totals(client.call_logs)

        state = checkpoint_state(
            epoch, playbook, task_histories, clean_success_ids, curator_skip_until,
            phase_costs, format_errors_by_phase, truncations_by_phase,
            curator_samples, curator_decisions, reflector_samples,
            rejected_reflector_attempts, eval_success, sorted(completed_eval_ids),
            grounding_rejections, gate_rejections,
        )
        batch_path = Path(f"outputs/a3/batches/{run_id}/epoch-{epoch:02d}.json")
        atomic_write_json(batch_path, {
            "run_id": run_id, "epoch": epoch, "actor_model": actor_model,
            "optimizer_model": optimizer_model, "phase_costs_hypothetical": phase_costs,
            "playbook": {"rules": playbook.rules, "next_rule_num": playbook.next_rule_num},
            "outcomes": batch_outcomes,
        })
        checkpoint.update({"completed_epochs": epoch, "state": state, "complete": False})
        atomic_write_json(checkpoint_path, checkpoint)
        print(f"BATCH_CHECKPOINT {batch_path}")

    # 5. Held-out Evaluation on 5 tasks
    cost_before_eval = client.total_hypothetical_usd
    client.current_phase = "eval"
    eval_actor = RealLLMActor(client=client, model_id=actor_model,
                              max_steps=config.get("MAX_STEPS_PER_EPISODE", 12))
    for t in smoke_heldout:
        if t["task_id"] in completed_eval_ids:
            continue
        cache_key = f"eval:{t['task_id']}"
        cached = episode_cache.get(cache_key)
        if cached:
            res, p = cached["run_result"], cached["passed"]
        else:
            cat = analysis_labels[t["task_id"]]["category"]
            traced_task = dict(t, _trace_context={"arm": "A3", "seed": SEED,
                                                 "epoch": EPOCHS, "phase": "eval"})
            from src.obs.spans import span
            with span("heldout_eval", phase="eval", attributes={
                "arm": "A3", "seed": SEED, "epoch": EPOCHS,
                "task_id": t["task_id"], "model_id": actor_model,
                "execution_mode": "real",
            }):
                res = eval_actor.run_episode(traced_task, playbook, MockAPI())
            p, _ = grader.grade(t, cat, res["trajectory"], res["final_answer"])
            p = bool(p and not res["format_errors"] and not res["ended_by_step_cap"])
            eval_event = {"cache_key": cache_key, "task_id": t["task_id"],
                          "run_result": res, "passed": p}
            serialized_eval = json.dumps(eval_event, ensure_ascii=False)
            key = os.environ.get(key_env, "")
            if key and key in serialized_eval:
                raise RuntimeError("Safety stop: refusing to persist an evaluation episode containing key material")
            append_jsonl(episode_log_path, eval_event)
            episode_cache[cache_key] = eval_event
        format_errors_by_phase["eval"] += res["format_errors"]
        truncations_by_phase["eval"] += res["truncations"]
        if p:
            eval_success += 1
        completed_eval_ids.add(t["task_id"])
        phase_costs = phase_cost_totals(client.call_logs)
        cost_before_eval = client.total_hypothetical_usd
        state = checkpoint_state(
            EPOCHS, playbook, task_histories, clean_success_ids, curator_skip_until,
            phase_costs, format_errors_by_phase, truncations_by_phase,
            curator_samples, curator_decisions, reflector_samples,
            rejected_reflector_attempts, eval_success, sorted(completed_eval_ids),
            grounding_rejections, gate_rejections,
        )
        checkpoint.update({"completed_epochs": EPOCHS, "state": state, "complete": False})
        atomic_write_json(checkpoint_path, checkpoint)

    curator_confusion = compute_curator_confusion(curator_decisions, analysis_labels)

    # Save summary
    results_summary = {
        "actor_model": actor_model,
        "optimizer_model": optimizer_model,
        "phase_costs_hypothetical": phase_costs,
        "format_errors_by_phase": format_errors_by_phase,
        "truncations_by_phase": truncations_by_phase,
        "total_hypothetical_usd": client.total_hypothetical_usd,
        "seed": SEED,
        "epochs": EPOCHS,
        "train_tasks": len(smoke_train),
        "heldout_tasks": len(smoke_heldout),
        "heldout_success_rate": eval_success / len(smoke_heldout) if smoke_heldout else 0.0,
        "curator_decisions": curator_decisions,
        "curator_confusion_matrix_vs_labels_json": curator_confusion,
        "total_calls": len(client.call_logs),
        "rejected_reflector_attempts": rejected_reflector_attempts,
        "reflector_rejection_counts": {
            "grounding_rejections": grounding_rejections,
            "gate_rejections": gate_rejections,
            "format_rejections": format_errors_by_phase["reflect"],
        },
        "curator_samples": curator_samples,
        "reflector_samples": reflector_samples,
        "call_logs": client.call_logs
    }
    # Leak scan runs against the actual real-call prompt log and task text.
    all_prompt_tasks = smoke_train + smoke_heldout
    leak_scan = scan_prompts_and_tasks_for_leaks(client.call_logs, all_prompt_tasks)
    results_summary["prompt_leak_scan"] = leak_scan
    results_summary["retry_events"] = client.retry_events
    key = os.environ.get(key_env, "")
    if key and key in json.dumps(results_summary, ensure_ascii=False):
        raise RuntimeError("Safety stop: refusing to print or persist a result containing key material")
    for sample in curator_samples:
        print("FULL_CURATOR_SAMPLE " + json.dumps(sample, ensure_ascii=False))
    for sample in reflector_samples:
        print("FULL_REFLECTOR_SAMPLE " + json.dumps(sample, ensure_ascii=False))
        print("ACCEPTED_REFLECTION " + json.dumps({
            "epoch": sample["epoch"], "attempt": sample["attempt"],
            "reason": sample["accepted_reason"], "ops": sample["parsed_ops"]
        }, ensure_ascii=False))
    for sample in rejected_reflector_attempts:
        print("REJECTED_REFLECTOR_ATTEMPT " + json.dumps(sample, ensure_ascii=False))
    print("PHASE_COSTS_HYPOTHETICAL_PRICING " + json.dumps(phase_costs, ensure_ascii=False))
    print("FORMAT_ERRORS_BY_PHASE " + json.dumps(format_errors_by_phase, ensure_ascii=False))
    print("TRUNCATIONS_BY_PHASE " + json.dumps(truncations_by_phase, ensure_ascii=False))
    print("REFLECTOR_REJECTION_COUNTS " + json.dumps(results_summary["reflector_rejection_counts"], ensure_ascii=False))
    print("CURATOR_CONFUSION_MATRIX_VS_LABELS_JSON " + json.dumps(curator_confusion, ensure_ascii=False))
    print("PROMPT_LEAK_SCAN " + json.dumps(leak_scan, ensure_ascii=False))
    result_path = Path(f"outputs/a3/smoke_results-{run_id}.json")
    serialized_summary = json.dumps(results_summary, indent=2)
    if key and key in serialized_summary:
        raise RuntimeError("Safety stop: refusing to persist a summary containing key material")
    with result_path.open("w") as f:
        f.write(serialized_summary)
    print(f"RESULT_FILE {result_path}")

    # Security scan required by the run protocol. Report every matching line
    # from the raw call log and generated result file, without modifying them.
    security_terms = ["impossible", "learnable", "flaky", "noise", "planted",
                      "near-miss", "labels.json", "nonexistent", "missing",
                      "invalid", "unavailable"]
    for scan_path in [client.log_path, result_path]:
        with scan_path.open(encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                lowered = line.lower()
                for term in security_terms:
                    if term in lowered:
                        print(f"SECURITY_TERM_MATCH {scan_path}:{line_no}: {term}")

    checkpoint["complete"] = True
    checkpoint["result_path"] = str(result_path)
    checkpoint["state"] = checkpoint_state(
        EPOCHS, playbook, task_histories, clean_success_ids, curator_skip_until,
        phase_costs, format_errors_by_phase, truncations_by_phase,
        curator_samples, curator_decisions, reflector_samples,
        rejected_reflector_attempts, eval_success, sorted(completed_eval_ids),
        grounding_rejections, gate_rejections,
    )
    atomic_write_json(checkpoint_path, checkpoint)
    return results_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run or resume the real A3 smoke test")
    parser.add_argument("--resume", action="store_true", help="resume the latest incomplete seed-42 run")
    arguments = parser.parse_args()
    run_a3_smoke_test(resume=arguments.resume)
