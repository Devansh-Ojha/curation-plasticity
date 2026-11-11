"""Offline replay of saved Reflector proposals through the current validator."""
import argparse
import json
from pathlib import Path

from src.analysis.reflection_grounding import check_rule_grounding, make_reflection_trace_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path(
        "outputs/real_experiment/archive/pre-trace-capture-fix/seed-42-A0/results.json"))
    args = parser.parse_args()
    state = json.loads(args.results.read_text(encoding="utf-8"))
    total = 0
    confirmed = {"remove body": False, "remove status": False, "set body": False}
    for epoch in state.get("epochs", []):
        for rejection in epoch.get("reflector_rejections", []):
            ops = rejection.get("ops") or []
            if not ops:
                continue
            failures = []
            for op in ops:
                task_id = (op.get("evidence") or {}).get("task_id")
                history = next((row for row in state.get("task_histories", {}).get(task_id, [])
                                if row.get("epoch") == epoch.get("epoch")), {})
                failures.append({"task_id": task_id, "trajectory": history.get("trajectory", [])})
            # Reconstruct only history available at that epoch; never let a
            # future success retroactively ground an earlier proposal.
            available_histories = {
                task_id: [row for row in rows if int(row.get("epoch", 0)) <= int(epoch.get("epoch", 0))]
                for task_id, rows in state.get("task_histories", {}).items()
            }
            records = make_reflection_trace_records(failures, available_histories)
            for index, op in enumerate(ops, 1):
                total += 1
                issues = check_rule_grounding([op], records)
                content = op.get("content", "")
                label = "PASS" if not issues else "FAIL"
                reason = "trace-grounded" if not issues else "; ".join(i["reason"] for i in issues)
                print(f"epoch={epoch.get('epoch')} attempt={rejection.get('attempt')} proposal={index} {label}: {content}\n  reason: {reason}")
                lowered = content.lower()
                for name in confirmed:
                    matched = name in lowered
                    if name == "remove body":
                        matched = "remove" in lowered and "body" in lowered
                    elif name == "remove status":
                        matched = "remove" in lowered and "status" in lowered
                    elif name == "set body":
                        matched = "set" in lowered and "body" in lowered
                    if matched and issues:
                        confirmed[name] = True
    print(f"\nSummary: {total} proposals replayed")
    for name, failed in confirmed.items():
        print(f"{name}: {'confirmed FAIL' if failed else 'not found'}")


if __name__ == "__main__":
    main()
