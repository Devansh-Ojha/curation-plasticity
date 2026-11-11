"""
Baseline Evaluator:
Runs empty-playbook baseline on 10 training tasks.
Reports:
- Pass/Fail frequency per ground-truth category
- Exact API error responses and HTTP status codes returned
Evaluates whether learnable tasks provide sufficient error signals for the Reflector.
"""
import json
from src.testbed.task_gen import generate_task_suites
from src.agents.actor import ActorAgent
from src.testbed.mock_api import MockAPI
from src.testbed.grader import Grader
from src.loop.provenance import Playbook

def run_empty_playbook_baseline():
    train_tasks, _, _, labels = generate_task_suites(noise_frac=0.20, seed=42, output_dir="data_baseline")
    tasks_10 = train_tasks[:10]

    api = MockAPI()
    grader = Grader(seed=42)
    empty_playbook = Playbook()
    actor = ActorAgent(model_name="gemini-2.5-flash")

    print("==================================================================")
    print("EMPTY-PLAYBOOK BASELINE EVALUATION (10 TRAIN TASKS)")
    print("==================================================================")

    results = []
    category_counts = {}

    for i, t in enumerate(tasks_10):
        t_id = t["task_id"]
        cat = labels[t_id]["category"]
        if cat not in category_counts:
            category_counts[cat] = {"total": 0, "passed": 0, "failed": 0}
        category_counts[cat]["total"] += 1

        run_res = actor.run_episode(t, empty_playbook, api)
        passed, fb = grader.grade(t, cat, run_res["trajectory"], run_res["final_answer"])

        if passed:
            category_counts[cat]["passed"] += 1
        else:
            category_counts[cat]["failed"] += 1

        # Extract last API response from trajectory
        last_api_call = "None"
        last_api_response = "None"
        for step in run_res.get("trajectory", []):
            if step.get("action") in ["http_request", "post_user", "post_order", "get_users_filter"]:
                last_api_call = step.get("call") or step.get("action")
                last_api_response = step.get("result", {})

        record = {
            "index": i + 1,
            "task_id": t_id,
            "category": cat,
            "instruction": t["instruction"],
            "passed": passed,
            "final_answer": run_res["final_answer"],
            "last_api_call": last_api_call,
            "api_response": last_api_response
        }
        results.append(record)

        status_str = "PASSED" if passed else "FAILED"
        print(f"\nTask {i+1} [{cat}] -> {status_str}")
        print(f"  Instruction:  {t['instruction']}")
        print(f"  Final Answer: {run_res['final_answer']}")
        if not passed:
            print(f"  API Response: {last_api_response}")

    print("\n================== SUMMARY PER CATEGORY ==================")
    for cat, counts in category_counts.items():
        pass_rate = (counts["passed"] / counts["total"]) * 100
        print(f"Category: {cat:15s} | Passed: {counts['passed']}/{counts['total']} ({pass_rate:.1f}%) | Failed: {counts['failed']}")
    print("==========================================================")

    return results, category_counts

if __name__ == "__main__":
    run_empty_playbook_baseline()
