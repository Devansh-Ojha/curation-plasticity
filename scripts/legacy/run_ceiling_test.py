"""Compare empty and hand-written oracle playbooks on 20 learnable tasks.

This runner makes real hosted actor calls. Task labels are used only by this
analysis harness to select learnable tasks, grade episodes, and report quirks.
"""
import json
import os
import time
import argparse
from collections import defaultdict
from pathlib import Path

import yaml

from src.agents.nvidia_client import NVIDIAClient
from src.agents.real_llm_actor import RealLLMActor
from src.loop.provenance import Playbook
from src.testbed.grader import Grader
from src.testbed.mock_api import MockAPI
from src.testbed.task_gen import generate_task_suites

ACTOR_MODEL = "nvidia/nemotron-3.5-bolt-e4b"
OPTIMIZER_MODEL = "meta/llama-3.2-90b-vision-instruct"
REPEATS = 3

ORACLE_RULES = [
    "Authentication: include the task's supplied API key in the X-Api-Key header on every request; never use Authorization. Example: GET /invoices/inv_502 with headers {\"X-Api-Key\": \"<key from task>\"}.",
    "Filtering: put filters in params using the literal filter[field] key. Example: GET /users with params {\"filter[role]\": \"admin\"}; for orders use {\"filter[status]\": \"completed\"}.",
    "Pagination: after a list response, send its next_page_token as params.page_token until next_page_token is null, then combine all returned items. Example: GET /orders params {\"page_token\": \"page_2\"}, then {\"page_token\": \"page_3\"} when returned.",
    "Idempotency: every POST requires a unique Idempotency-Key header. Example: POST /orders headers {\"Idempotency-Key\": \"order-u1-20240510\"}.",
    "Money: use body.amount_cents as an integer number of cents, never amount. Example: $50.00 is {\"amount_cents\": 5000}.",
    "Date: the order date field is named date and uses YYYY-MM-DD. Example: {\"date\": \"2024-05-10\"}; do not send order_date.",
    "IDs: preserve IDs exactly as given; POST /users uses customer_id, not cust_id. Example: {\"customer_id\": \"u6\"}; use /orders/ord_101 and /invoices/inv_502 as supplied.",
]


class LoggedClient(NVIDIAClient):
    def __init__(self, config, log_path):
        super().__init__(config)
        self.log_path = Path(log_path)

    def chat_completion(self, *args, **kwargs):
        result = super().chat_completion(*args, **kwargs)
        record = dict(result["call_record"])
        if record.get("execution_mode") != "real":
            raise RuntimeError("Safety stop: call was not marked execution_mode=real")
        serialized = json.dumps({"event": "call", **record}, ensure_ascii=False)
        key = os.environ.get(self.key_env_var, "")
        if key and key in serialized:
            raise RuntimeError("Safety stop: refusing to persist a call containing key material")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(serialized + "\n")
        print("RAW_CALL " + json.dumps({k: record.get(k) for k in (
            "model_id", "request_id", "input_tokens", "output_tokens",
            "reasoning_tokens", "finish_reason", "latency_sec", "cost_label",
            "hypothetical_cost_usd"
        )}, ensure_ascii=False))
        return result


def select_twenty_balanced_tasks(train, labels):
    by_instruction = defaultdict(list)
    for task in train:
        if labels[task["task_id"]]["category"] == "learnable":
            by_instruction[task["instruction"]].append(task)
    selected = []
    keys = sorted(by_instruction)
    while len(selected) < 20:
        progressed = False
        for instruction in keys:
            if by_instruction[instruction] and len(selected) < 20:
                selected.append(by_instruction[instruction].pop(0))
                progressed = True
        if not progressed:
            raise RuntimeError("Could not select 20 learnable tasks from the generated suite.")
    return selected


def write_progress(path, payload, key):
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    if key and key in serialized:
        raise RuntimeError("Safety stop: refusing to persist results containing key material")
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(serialized, encoding="utf-8")
    tmp_path.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Compare empty and oracle playbooks with real actor calls")
    parser.add_argument("--resume", help="Resume an existing incremental result JSON")
    args = parser.parse_args()

    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    key_env = config.get("API_KEY_ENV_VAR", "NVIDIA_API_KEY")
    if not os.environ.get(key_env):
        raise SystemExit(f"STOP: {key_env} is not set; no mock fallback is permitted.")
    if config["OPTIMIZER"]["model_id"] != OPTIMIZER_MODEL:
        raise SystemExit(f"STOP: config optimizer must be {OPTIMIZER_MODEL}.")
    if ACTOR_MODEL not in config.get("REFERENCE_PRICE_TABLE", {}):
        raise SystemExit(f"STOP: no config.yaml pricing entry for {ACTOR_MODEL}.")

    output_dir = Path("outputs/ceiling")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        result_path = Path(args.resume)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        run_id = result_path.stem.removeprefix("ceiling-")
        if payload.get("actor_model") != ACTOR_MODEL or payload.get("optimizer_model") != OPTIMIZER_MODEL:
            raise SystemExit("STOP: resume file model IDs do not match this ceiling test.")
    else:
        run_id = time.strftime("%Y%m%dT%H%M%S")
        result_path = output_dir / f"ceiling-{run_id}.json"
        payload = None
    call_log_path = Path("logs/ceiling") / f"calls-{run_id}.jsonl"
    client = LoggedClient(config, call_log_path)
    if args.resume and call_log_path.exists():
        prior_calls = [json.loads(line) for line in call_log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        client.call_logs = [{k: v for k, v in item.items() if k != "event"}
                            for item in prior_calls if item.get("event") == "call"]
        # Include calls from an episode that crashed before its result was saved.
        client.total_hypothetical_usd = sum(item.get("hypothetical_cost_usd", 0.0) for item in client.call_logs)

    train, _, _, labels = generate_task_suites(
        noise_frac=0.0, seed=42, output_dir=str(output_dir / "task_data")
    )
    tasks = select_twenty_balanced_tasks(train, labels)
    empty = Playbook(max_tokens=config.get("MAX_PLAYBOOK_TOKENS", 2000))
    oracle = Playbook(max_tokens=config.get("MAX_PLAYBOOK_TOKENS", 2000))
    valid = oracle.apply_ops(
        [{"op": "add", "content": rule} for rule in ORACLE_RULES],
        epoch=0, source_task_ids=[task["task_id"] for task in tasks]
    )
    if not valid:
        raise SystemExit("STOP: hand-written oracle playbook exceeded the configured token cap.")

    expected_tasks = [{"task_id": t["task_id"], "instruction": t["instruction"],
                       "required_quirks": labels[t["task_id"]]["requires_quirks"]} for t in tasks]
    if payload is None:
        payload = {
            "execution_mode": "real", "actor_model": ACTOR_MODEL,
            "optimizer_model": OPTIMIZER_MODEL, "cost_label": "hypothetical pricing",
            "tasks": expected_tasks, "oracle_rules": ORACLE_RULES, "repeats": REPEATS,
            "episodes": [], "conditions": {}, "total_hypothetical_cost_usd": 0.0,
        }
    elif payload.get("tasks") != expected_tasks or payload.get("oracle_rules") != ORACLE_RULES:
        raise SystemExit("STOP: resume file task set or oracle rules differ from this script.")
    completed = {(r["condition"], r["repeat"], r["task_id"]) for r in payload.get("episodes", [])}
    payload["total_hypothetical_cost_usd"] = client.total_hypothetical_usd
    write_progress(result_path, payload, os.environ.get(key_env, ""))

    for condition, playbook in (("empty", empty), ("oracle", oracle)):
        condition_records = [r for r in payload["episodes"] if r["condition"] == condition]
        for repeat in range(1, REPEATS + 1):
            grader = Grader(seed=41 + repeat)
            actor = RealLLMActor(
                client=client, model_id=ACTOR_MODEL,
                max_steps=config.get("MAX_STEPS_PER_EPISODE", 12)
            )
            for task in tasks:
                if (condition, repeat, task["task_id"]) in completed:
                    continue
                episode = actor.run_episode(task, playbook.clone(), MockAPI())
                category = labels[task["task_id"]]["category"]  # analysis/grading only
                passed, feedback = grader.grade(task, category, episode["trajectory"], episode["final_answer"])
                passed = bool(passed and not episode["format_errors"] and not episode["ended_by_step_cap"])
                record = {
                    "condition": condition, "repeat": repeat,
                    "task_id": task["task_id"], "passed": passed,
                    "feedback": feedback if passed else "Failed: Validation check failed. Output did not match expected target.",
                    "format_errors": episode["format_errors"],
                    "truncations": episode["truncations"],
                    "cap_hit": episode["ended_by_step_cap"],
                    "read_playbook_called": episode["read_playbook_called"],
                    "tokens": episode["tokens"],
                    "required_quirks": labels[task["task_id"]]["requires_quirks"],
                }
                condition_records.append(record)
                payload["episodes"].append(record)
                completed.add((condition, repeat, task["task_id"]))
                payload["total_hypothetical_cost_usd"] = client.total_hypothetical_usd
                payload["conditions"][condition] = summarize(condition_records)
                write_progress(result_path, payload, os.environ.get(key_env, ""))
                print("EPISODE " + json.dumps(record, ensure_ascii=False))
        payload["conditions"][condition] = summarize(condition_records)
        print("CONDITION_SUMMARY " + json.dumps({"condition": condition, **payload["conditions"][condition]}, ensure_ascii=False))
        write_progress(result_path, payload, os.environ.get(key_env, ""))

    failures = []
    task_by_id = {t["task_id"]: t for t in tasks}
    for record in payload["episodes"]:
        if not record["passed"]:
            failures.append({
                "condition": record["condition"], "repeat": record["repeat"],
                "task_id": record["task_id"], "instruction": task_by_id[record["task_id"]]["instruction"],
                "required_quirks": record["required_quirks"],
            })
    payload["failed_task_quirks"] = failures
    payload["total_hypothetical_cost_usd"] = client.total_hypothetical_usd
    payload["raw_call_logs"] = client.call_logs
    write_progress(result_path, payload, os.environ.get(key_env, ""))
    print("FAILED_TASK_QUIRKS " + json.dumps(failures, ensure_ascii=False))
    print("RESULT_FILE " + str(result_path))


def summarize(records):
    count = len(records)
    return {
        "episodes": count,
        "pass_rate": sum(r["passed"] for r in records) / count if count else 0.0,
        "cap_hit_rate": sum(r["cap_hit"] for r in records) / count if count else 0.0,
        "read_playbook_rate": sum(r["read_playbook_called"] for r in records) / count if count else 0.0,
        "avg_tokens_per_episode": sum(r["tokens"]["total_tokens"] for r in records) / count if count else 0.0,
    }


if __name__ == "__main__":
    main()
