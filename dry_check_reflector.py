"""Run the current Reflector checks against archived epoch-1 output; no model calls."""
import argparse
import json
from pathlib import Path

from src.analysis.reflection_grounding import check_rule_grounding, make_reflection_trace_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/real_experiment/archive/noise-floor-empty-playbook/seed-42-A0/checkpoint.json"))
    args = parser.parse_args()
    state = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    epoch = next(row for row in state["epochs"] if row.get("epoch") == 1)
    saved_attempts = [r for r in epoch.get("reflector_rejections", []) if r.get("kind") == "grounding"]
    if not saved_attempts:
        raise SystemExit("No saved epoch-1 grounded Reflector outputs found")
    failures = []
    for attempt in saved_attempts:
        for op in attempt.get("ops", []):
            task_id = (op.get("evidence") or {}).get("task_id")
            failures.append({"task_id": task_id, "trajectory": next((h.get("trajectory", [])
                               for h in state.get("task_histories", {}).get(task_id, [])
                               if h.get("epoch") == 1), [])})
    histories = state.get("task_histories", {})
    records = make_reflection_trace_records(failures, histories)
    print("Dry check: saved epoch-1 proposals; no model calls")
    for attempt_no, attempt in enumerate(saved_attempts, 1):
        for index, op in enumerate(attempt.get("ops", [])):
            issues = check_rule_grounding([op], records)
            title = op.get("content", "").split(".", 1)[0]
            if issues:
                print(f"attempt {attempt_no} proposal {index + 1}: FAIL — {issues[0]['reason']}\n  {title}")
            else:
                print(f"attempt {attempt_no} proposal {index + 1}: PASS — failed request/error pair is trace-grounded; field/header is named and observed fix is not contradicted\n  {title}")


if __name__ == "__main__":
    main()
