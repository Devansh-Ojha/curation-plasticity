"""Compare every chat model advertised by /v1/models on the same five traces.

This tool is user-run because it makes real API calls. It deliberately applies
no grounding or playbook gate: raw proposed rules are shown side by side.
"""
import copy
import json
import os
import time
import urllib.request
from pathlib import Path

import yaml

from src.agents.nvidia_client import NVIDIAClient
from src.analysis.reflection_grounding import make_reflection_trace_records
from run_real_a3_smoke import REFLECTOR_PROMPT_TEMPLATE


def append_jsonl(path, item, key):
    serialized = json.dumps(item, ensure_ascii=False)
    if key and key in serialized:
        raise RuntimeError("Safety stop: refusing to persist comparison data containing key material")
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(serialized + "\n")


def atomic_json(path, item, key):
    serialized = json.dumps(item, indent=2, ensure_ascii=False)
    if key and key in serialized:
        raise RuntimeError("Safety stop: refusing to persist comparison data containing key material")
    target = Path(path)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(serialized, encoding="utf-8")
    temp.replace(target)


def request_model_list(config, key):
    url = config.get("API_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/") + "/models"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    ids = sorted({item["id"] for item in payload.get("data", []) if isinstance(item, dict) and item.get("id")})
    return ids


def first_five_failed_traces(path, task_path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    selected = []
    seen_task_ids = set()
    for row in rows:
        if (row.get("cache_key", "").startswith("train:") and not row.get("passed")
                and row.get("task_id") not in seen_task_ids):
            selected.append(row)
            seen_task_ids.add(row["task_id"])
            if len(selected) == 5:
                break
    if len(selected) != 5:
        raise RuntimeError(f"Expected at least 5 unique failed training traces; found {len(selected)}")

    public_tasks = json.loads(Path(task_path).read_text(encoding="utf-8"))
    instruction_by_id = {task["task_id"]: task["instruction"] for task in public_tasks}

    histories = {}
    for row in rows:
        if row.get("cache_key", "").startswith("train:"):
            histories.setdefault(row["task_id"], []).append({
                "epoch": row.get("epoch"), "passed": row.get("passed"),
                "trajectory": row.get("run_result", {}).get("trajectory", []),
            })

    failures = []
    for row in selected:
        failures.append({
            "task_id": row["task_id"],
            "instruction": instruction_by_id.get(row["task_id"], ""),
            "feedback": row.get("feedback", ""),
            "trajectory": row.get("run_result", {}).get("trajectory", []),
        })
    return failures, histories, [row["task_id"] for row in selected]


def main():
    with open("config.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    key_env = config.get("API_KEY_ENV_VAR", "NVIDIA_API_KEY")
    key = os.environ.get(key_env)
    if not key:
        raise SystemExit(f"STOP: {key_env} is not set; no mock fallback is permitted")

    results_file = os.environ.get("A3_RESULTS", "outputs/a3/smoke_results-20261108T191622.json")
    episodes_file = os.environ.get("A3_EPISODES", "outputs/a3/episodes-20261108T191622.jsonl")
    if not Path(results_file).exists() or not Path(episodes_file).exists():
        raise SystemExit("STOP: set A3_RESULTS and A3_EPISODES to the completed smoke run files")

    run_results = json.loads(Path(results_file).read_text(encoding="utf-8"))
    failures, histories, task_ids = first_five_failed_traces(episodes_file, "data_smoke_real/train.json")
    trace_records = make_reflection_trace_records(failures, histories)
    trace_bundle = []
    for failure, record in zip(failures, trace_records):
        trace_bundle.append({
            "task_id": failure["task_id"],
            "instruction": failure["instruction"],
            "failed_trajectory": failure["trajectory"],
            "evidence_record": record,
            "prior_successful_trajectories": [
                {"epoch": h["epoch"], "trajectory": h["trajectory"]}
                for h in histories.get(failure["task_id"], []) if h.get("passed")
            ],
        })
    messages = [
        {"role": "system", "content": REFLECTOR_PROMPT_TEMPLATE},
        {"role": "user", "content": "Apply this prompt to the same five failure traces for every model. Do not infer beyond the trace evidence.\n"
         + json.dumps(trace_bundle, ensure_ascii=False, indent=2)},
    ]

    # Enumerate model IDs first. All model testing then uses one configured
    # hypothetical comparison rate and the regular client budget guard.
    model_ids = request_model_list(config, key)
    print("V1_MODELS_COUNT", len(model_ids))
    print("V1_MODEL_IDS " + json.dumps(model_ids, ensure_ascii=False))
    print("TEST_TRACE_TASK_IDS " + json.dumps(task_ids))

    comparison_pricing = config.get("CANDIDATE_COMPARISON_PRICING")
    if not isinstance(comparison_pricing, dict):
        raise SystemExit("STOP: CANDIDATE_COMPARISON_PRICING is missing from config.yaml")
    client_config = copy.deepcopy(config)
    client_config["REFERENCE_PRICE_TABLE"] = {model_id: comparison_pricing for model_id in model_ids}
    client = NVIDIAClient(client_config)

    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_dir = Path("outputs/reflector_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    call_log_path = output_dir / f"calls-{stamp}.jsonl"
    result_path = output_dir / f"comparison-{stamp}.json"
    comparison_rows = []
    stop_reason = None

    for model_id in model_ids:
        try:
            response = client.chat_completion(
                model_id=model_id, messages=messages,
                temperature=0.0, max_tokens=1000,
            )
            record = dict(response["call_record"])
            raw_output = response.get("text") or ""
            try:
                cleaned = raw_output.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[-1]
                    if cleaned.endswith("```"):
                        cleaned = cleaned.rsplit("```", 1)[0]
                parsed_ops = json.loads(cleaned.strip())
                if not isinstance(parsed_ops, list):
                    parsed_ops = None
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed_ops = None
            row = {
                "model_id": model_id, "responded": True,
                "request_id": record.get("request_id"),
                "input_tokens": record.get("input_tokens"),
                "output_tokens": record.get("output_tokens"),
                "reasoning_tokens": record.get("reasoning_tokens"),
                "finish_reason": record.get("finish_reason"),
                "latency_sec": record.get("latency_sec"),
                "hypothetical_cost_usd": record.get("hypothetical_cost_usd"),
                "raw_output": raw_output, "proposed_ops": parsed_ops,
                "valid_json_array": parsed_ops is not None,
            }
            append_jsonl(call_log_path, {"event": "call", **record}, key)
        except Exception as exc:
            error = str(exc).replace(key, "[REDACTED]")
            row = {"model_id": model_id, "responded": False, "error": error}
            append_jsonl(call_log_path, {"event": "call_error", **row}, key)
            if "HARD CAP" in error:
                stop_reason = "hypothetical MAX_USD guard reached"
            elif ("429" in error or "HTTP Error 5" in error
                  or "timed out" in error.lower() or "timeout" in error.lower()):
                stop_reason = f"provider throttling/timeout at {model_id}"

        comparison_rows.append(row)
        summary = {
            "source_results": results_file,
            "source_episodes": episodes_file,
            "source_actor_model": run_results.get("actor_model"),
            "source_optimizer_model": run_results.get("optimizer_model"),
            "trace_task_ids": task_ids,
            "enumerated_model_ids": model_ids,
            "tested_models": comparison_rows,
            "responded_model_ids": [r["model_id"] for r in comparison_rows if r.get("responded")],
            "stop_reason": stop_reason,
            "total_hypothetical_usd": client.total_hypothetical_usd,
            "retry_events": client.retry_events,
            "cost_label": "hypothetical pricing",
            "call_log_path": str(call_log_path),
        }
        atomic_json(result_path, summary, key)
        print("OPTIMIZER_COMPARISON_ROW " + json.dumps(row, ensure_ascii=False))
        if stop_reason:
            break

    responded = [row for row in comparison_rows if row.get("responded")]
    print("RESPONDED_MODEL_IDS " + json.dumps([row["model_id"] for row in responded], ensure_ascii=False))
    print("SIDE_BY_SIDE_RULES_BEGIN")
    for row in responded:
        print(json.dumps({"model_id": row["model_id"], "proposed_ops": row.get("proposed_ops"),
                          "raw_output": row.get("raw_output")}, ensure_ascii=False))
    print("SIDE_BY_SIDE_RULES_END")
    print("STOP_REASON", stop_reason)
    print("TOTAL_HYPOTHETICAL_COST_USD", round(client.total_hypothetical_usd, 6))
    print("CALL_LOG_PATH", call_log_path)
    print("RESULT_FILE", result_path)


if __name__ == "__main__":
    main()
