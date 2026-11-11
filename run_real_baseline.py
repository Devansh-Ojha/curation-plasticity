"""Compare two real hosted actor models on the same stratified 30-task baseline.

Labels are loaded only by this analysis runner for grading and summaries. They
are never included in task prompts or sent to a model.
"""
import json
import os
import time
from pathlib import Path

import yaml

from src.agents.nvidia_client import NVIDIAClient
from src.agents.nvidia_client import NVIDIAAPIError
from src.agents.real_llm_actor import RealLLMActor
from src.analysis.leak_scanner import scan_prompts_and_tasks_for_leaks
from src.loop.provenance import Playbook
from src.testbed.grader import Grader
from src.testbed.mock_api import MockAPI
from src.testbed.task_gen import generate_task_suites

MODELS = ["minimaxai/minimax-m3.1-flash", "nvidia/nemotron-3.5-bolt-e4b"]
FAILURE_TEXT = "Failed: Validation check failed. Output did not match expected target."


def score_baseline_result(grader, task, category, result):
    passed, feedback = grader.grade(task, category, result["trajectory"], result["final_answer"])
    if result.get("format_errors", 0) or result.get("api_failure") or result.get("ended_by_step_cap"):
        passed = False
    return passed, feedback if passed else FAILURE_TEXT


class LoggedClient(NVIDIAClient):
    def __init__(self, config, log_path):
        super().__init__(config)
        self.log_path = Path(log_path)
        self.call_events = []

    def chat_completion(self, *args, **kwargs):
        started = time.time()
        try:
            result = super().chat_completion(*args, **kwargs)
            record = dict(result["call_record"])
            if record.get("execution_mode") != "real":
                raise RuntimeError("Safety stop: call was not marked execution_mode=real")
            self._append({"event": "call", **record})
            self.call_events.append({"event": "call", **record})
            print("RAW_CALL " + json.dumps({k: record[k] for k in (
                "model_id", "request_id", "input_tokens", "output_tokens",
                "reasoning_tokens", "finish_reason", "latency_sec", "cost_label", "hypothetical_cost_usd"
            )}, ensure_ascii=False))
            return result
        except Exception as exc:
            # Preserve the provider error text while ensuring a credential can
            # never be persisted if a provider unexpectedly echoes it.
            key = os.environ.get(self.key_env_var, "")
            message = str(exc).replace(key, "[REDACTED]") if key else str(exc)
            error_record = {"event": "call_error", "model_id": kwargs.get("model_id", args[0] if args else None),
                            "request_id": None, "input_tokens": None, "output_tokens": None,
                            "reasoning_tokens": None, "finish_reason": None,
                            "elapsed_sec": round(time.time() - started, 3), "error": message}
            self._append(error_record)
            self.call_events.append(error_record)
            raise type(exc)(message) from None

    def _append(self, item):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(item, ensure_ascii=False)
        key = os.environ.get(self.key_env_var, "")
        if key and key in serialized:
            raise RuntimeError("Safety stop: refusing to persist a record containing key material")
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(serialized + "\n")


def main():
    from src.obs.phoenix_setup import init_phoenix_tracer
    init_phoenix_tracer()
    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    key_env = config.get("API_KEY_ENV_VAR", "NVIDIA_API_KEY")
    if not os.environ.get(key_env):
        raise SystemExit(f"STOP: {key_env} is not set; no mock fallback is permitted.")

    output = Path("outputs/baseline")
    logs = Path("logs/baseline")
    output.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    # Keep prior raw logs intact; choose a fresh run suffix.
    run_id = time.strftime("%Y%m%dT%H%M%S")
    client = LoggedClient(config, logs / f"calls-{run_id}.jsonl")
    all_train, _, _, labels = generate_task_suites(
        noise_frac=0.60, seed=42, output_dir=str(output / "task_data")
    )
    # The generated 30-task split has six examples in each of five categories
    # at noise_frac=.60. Interleave categories to avoid category-order effects.
    by_category = {}
    for task in all_train:
        by_category.setdefault(labels[task["task_id"]]["category"], []).append(task)
    expected_categories = ["easy", "learnable", "impossible", "flaky_judge", "flaky_ambiguous"]
    if set(by_category) != set(expected_categories) or any(len(by_category[c]) != 6 for c in expected_categories):
        raise SystemExit("STOP: expected exactly 6 stratified tasks per category.")
    tasks = [by_category[category][i] for i in range(6) for category in expected_categories]
    grader = Grader(seed=42)
    all_results = {}
    result_path = output / f"baseline-{run_id}.json"

    def write_progress():
        payload = {
            "execution_mode": "real", "cost_label": "hypothetical pricing",
            "models": MODELS, "total_hypothetical_cost_usd": client.total_hypothetical_usd,
            "results": all_results, "raw_call_logs": client.call_logs,
            "call_events": client.call_events, "retry_events": client.retry_events,
        }
        serialized = json.dumps(payload, indent=2, ensure_ascii=False)
        secret = os.environ.get(key_env, "")
        if secret and secret in serialized:
            raise RuntimeError("Safety stop: refusing to persist a result containing key material")
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(result_path)

    write_progress()
    for model_id in MODELS:
        retry_start = len(client.retry_events)
        print(f"\n=== REAL EMPTY-PLAYBOOK BASELINE: {model_id} ===")
        actor = RealLLMActor(client=client, model_id=model_id,
                             max_steps=config.get("MAX_STEPS_PER_EPISODE", 12))
        records = []
        all_results[model_id] = {"episodes": records, "summary": None}
        for task in tasks:
            # This is a deliberately empty playbook for the baseline.
            category = labels[task["task_id"]]["category"]  # grader-only analysis
            call_start = len(client.call_logs)
            try:
                traced_task = dict(task, _trace_context={"arm": "baseline", "seed": 42,
                                                        "epoch": 0, "phase": "act"})
                res = actor.run_episode(traced_task, Playbook(), MockAPI())
            except NVIDIAAPIError as exc:
                episode_calls = client.call_logs[call_start:]
                res = {
                    "task_id": task["task_id"], "final_answer": None,
                    "trajectory": [{"step": 0, "action": "api_error", "error": str(exc),
                                    "method": None, "path": None, "params": None,
                                    "headers": None, "body": None}],
                    "format_errors": 0, "truncations": 0, "ended_by_step_cap": False,
                    "api_failure": True,
                    "tokens": {
                        "input_tokens": sum(c["input_tokens"] for c in episode_calls),
                        "output_tokens": sum(c["output_tokens"] for c in episode_calls),
                        "total_tokens": sum(c["input_tokens"] + c["output_tokens"] for c in episode_calls),
                    },
                    "hypothetical_cost_usd": sum(c["hypothetical_cost_usd"] for c in episode_calls),
                }
            passed, feedback = score_baseline_result(grader, task, category, res)
            errors = [step.get("result") for step in res["trajectory"]
                      if step.get("action") == "http_request"
                      and isinstance(step.get("result"), dict)
                      and step["result"].get("status", 200) >= 400]
            record = {
                "model_id": model_id, "task_id": task["task_id"],
                "category": category,
                "passed": passed, "feedback": feedback,
                "format_errors": res["format_errors"],
                "truncations": res.get("truncations", 0),
                "api_failure": res.get("api_failure", False),
                "ended_by_step_cap": res["ended_by_step_cap"],
                "tokens": res["tokens"], "api_errors_verbatim": errors,
                "trajectory": res["trajectory"]
            }
            records.append(record)
            all_results[model_id]["episodes"] = records
            write_progress()
            rendered_record = json.dumps(record, ensure_ascii=False)
            key_value = os.environ.get(key_env, "")
            if key_value and key_value in rendered_record:
                raise SystemExit("STOP: refusing to print or persist an episode containing key material.")
            print("EPISODE " + rendered_record)
            if client.consecutive_api_failures >= 5:
                write_progress()
                raise SystemExit("STOP: 5 consecutive API failures; baseline halted after saving progress.")
        category_summaries = {}
        for category in expected_categories:
            category_records = [r for r in records if r["category"] == category]
            http_calls = [step for r in category_records for step in r["trajectory"]
                          if step.get("action") == "http_request"]
            tool_errors = [step for step in http_calls
                           if isinstance(step.get("result"), dict) and step["result"].get("status", 200) >= 400]
            category_summaries[category] = {
                "episodes": len(category_records),
                "pass_rate": sum(r["passed"] for r in category_records) / len(category_records),
                "cap_hit_rate": sum(r["ended_by_step_cap"] for r in category_records) / len(category_records),
                "tool_error_rate": len(tool_errors) / len(http_calls) if http_calls else 0.0,
                "tool_errors": len(tool_errors), "http_tool_calls": len(http_calls),
                "avg_tokens_per_episode": sum(r["tokens"]["total_tokens"] for r in category_records) / len(category_records),
            }
        http_calls = [step for r in records for step in r["trajectory"] if step.get("action") == "http_request"]
        tool_errors = [step for step in http_calls
                       if isinstance(step.get("result"), dict) and step["result"].get("status", 200) >= 400]
        model_calls = [e for e in client.call_events if e.get("model_id") == model_id]
        format_error_count = sum(r["format_errors"] for r in records)
        truncation_count = sum(1 for e in model_calls if e.get("finish_reason") == "length")
        summary = {
            "episodes": len(records),
            "avg_tokens_per_episode": sum(r["tokens"]["total_tokens"] for r in records) / len(records),
            "format_errors": sum(r["format_errors"] for r in records),
            "format_error_rate": format_error_count / len(model_calls) if model_calls else 0.0,
            "truncations": truncation_count,
            "truncation_rate": truncation_count / len(model_calls) if model_calls else 0.0,
            "cap_hits": sum(r["ended_by_step_cap"] for r in records),
            "cap_hit_rate": sum(r["ended_by_step_cap"] for r in records) / len(records),
            "tool_error_rate": len(tool_errors) / len(http_calls) if http_calls else 0.0,
            "tool_errors": len(tool_errors), "http_tool_calls": len(http_calls),
            "api_errors_verbatim": [e for r in records for e in r["api_errors_verbatim"]],
            "successes": sum(r["passed"] for r in records),
            "timeouts": sum(any(term in str(e.get("error", "")).lower() for term in ("timeout", "timed out")) for e in client.retry_events[retry_start:]),
            "429s": sum(e.get("type") == "429" for e in client.retry_events[retry_start:]),
        }
        print("MODEL_SUMMARY " + json.dumps({"model_id": model_id, **summary, "by_category": category_summaries}, ensure_ascii=False))
        all_results[model_id] = {"episodes": records, "summary": summary, "by_category": category_summaries}
        write_progress()

    scan = scan_prompts_and_tasks_for_leaks(client.call_logs, tasks)
    print("PROMPT_LEAK_SCAN " + json.dumps(scan, ensure_ascii=False))
    all_results["prompt_leak_scan"] = scan
    all_results["retry_events"] = client.retry_events
    write_progress()
    print(f"RESULT_FILE {result_path}")


if __name__ == "__main__":
    main()
