"""
Baseline Comparison Runner for Candidate Actors:
Runs empty-playbook baseline on 30 stratified train tasks with:
1. minimaxai/minimax-m3.1-flash
2. nvidia/nemotron-3.5-bolt-e4b

Reports per model:
- Pass rate per category (easy, learnable, impossible, flaky_judge, flaky_ambiguous)
- Tool-call / format error rate
- Average tokens per episode (input and output)
- Verbatim API error bodies encountered
"""
import os
import yaml
import json
from src.testbed.task_gen import generate_task_suites
from src.testbed.mock_api import MockAPI
from src.testbed.grader import Grader
from src.loop.provenance import Playbook
from src.agents.nvidia_client import OpenAICompatibleClient
from src.agents.real_llm_actor import RealLLMActor

def evaluate_actor_model(model_id: str, tasks: list, labels: dict, config: dict):
    print(f"\n=======================================================")
    print(f"EVALUATING CANDIDATE ACTOR: {model_id}")
    print(f"Tasks: {len(tasks)} (Stratified) | Playbook: Empty")
    print(f"=======================================================")

    client = OpenAICompatibleClient(config)
    actor = RealLLMActor(client=client, model_id=model_id, max_steps=6)
    grader = Grader(seed=42)
    empty_playbook = Playbook()

    category_stats = {}
    verbatim_errors = []
    total_tokens_all = []
    total_format_errors = 0

    for idx, t in enumerate(tasks):
        t_id = t["task_id"]
        cat = labels[t_id]["category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "passed": 0, "failed": 0}
        category_stats[cat]["total"] += 1

        mock_api = MockAPI() # fresh API state
        run_res = actor.run_episode(t, empty_playbook, mock_api)
        passed, fb = grader.grade(t, cat, run_res["trajectory"], run_res["final_answer"])

        if passed:
            category_stats[cat]["passed"] += 1
        else:
            category_stats[cat]["failed"] += 1

        total_tokens_all.append(run_res["tokens"]["total_tokens"])
        total_format_errors += run_res["format_errors"]

        # Collect verbatim API error responses
        for step in run_res.get("trajectory", []):
            if step.get("action") == "http_request":
                res_obj = step.get("result", {})
                status = res_obj.get("status", 200)
                if status >= 400:
                    err_entry = {
                        "task_id": t_id,
                        "category": cat,
                        "status": status,
                        "body": res_obj.get("body")
                    }
                    if err_entry not in verbatim_errors:
                        verbatim_errors.append(err_entry)

        res_str = "PASS" if passed else "FAIL"
        print(f"[{idx+1:02d}/{len(tasks)}] {t_id} ({cat:15s}) -> {res_str} | format_errs={run_res['format_errors']} tokens={run_res['tokens']['total_tokens']}")

    avg_tokens = sum(total_tokens_all) / len(total_tokens_all) if total_tokens_all else 0.0
    format_error_rate = (total_format_errors / len(tasks))

    print(f"\n--- Model Summary: {model_id} ---")
    print(f"Average Tokens / Episode: {avg_tokens:.1f}")
    print(f"Tool-call / Format Error Rate: {format_error_rate:.2f} per episode")
    print("\nPass Rates by Category:")
    for cat, s in category_stats.items():
        pct = (s["passed"] / s["total"]) * 100 if s["total"] > 0 else 0.0
        print(f"  - {cat:17s}: {s['passed']}/{s['total']} ({pct:.1f}%)")

    print("\nVerbatim API Error Bodies Encountered:")
    for err in verbatim_errors[:8]:
        print(f"  [{err['status']} - {err['category']}] {json.dumps(err['body'])}")

    return {
        "model_id": model_id,
        "avg_tokens": avg_tokens,
        "format_error_rate": format_error_rate,
        "category_stats": category_stats,
        "verbatim_errors": verbatim_errors
    }

def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    train_tasks, _, _, labels = generate_task_suites(noise_frac=0.20, seed=42, output_dir="data_eval_actors")
    tasks_30 = train_tasks[:30]

    m1 = config.get("ACTOR_CANDIDATE_1", "minimaxai/minimax-m3.1-flash")
    m2 = config.get("ACTOR_CANDIDATE_2", "nvidia/nemotron-3.5-bolt-e4b")

    res_m1 = evaluate_actor_model(m1, tasks_30, labels, config)
    res_m2 = evaluate_actor_model(m2, tasks_30, labels, config)

    with open("results/actor_candidates_eval.json", "w") as f:
        json.dump([res_m1, res_m2], f, indent=2)

if __name__ == "__main__":
    main()
