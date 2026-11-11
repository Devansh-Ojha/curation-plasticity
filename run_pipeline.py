"""
End-to-End Orchestrator:
Runs MVP: Arms A0, A1, A2, A3 across 3 seeds on 8 epochs.
Evaluates checkpoints on heldout_id and heldout_ood.
Computes Plasticity, Pollution, Regressions, and Curator quality metrics.
Exports summary results to results/results_table.json and markdown summary.
"""
import os
import yaml
import json
import argparse
from pathlib import Path
from src.testbed.task_gen import generate_task_suites
from src.loop.train_loop import run_train_arm
from src.obs.experiments import evaluate_checkpoint_on_split
from src.agents.actor import ActorAgent
from src.testbed.grader import Grader
from src.analysis.metrics import compute_plasticity, compute_curator_confusion

def main():
    parser = argparse.ArgumentParser(description="Run Curation Plasticity Experiments")
    parser.add_argument("--dry-run", action="store_true", help="Run with mock LLM (deterministic)")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--arms", nargs="+", default=["A0", "A1", "A2", "A3"], help="Arms to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Seeds to run")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    epochs = args.epochs
    seeds = args.seeds
    arms = args.arms
    noise_frac = config.get("NOISE_FRAC", 0.15)
    batch_size = config.get("BATCH_SIZE", 8)

    print("==================================================================")
    print("STARTING EXPERIMENT: Curation Plasticity Benchmark (Arize Phoenix)")
    print(f"Arms: {arms} | Seeds: {seeds} | Epochs: {epochs} | Noise: {noise_frac*100:.0f}%")
    print("==================================================================")

    # 1. Generate or load datasets
    train_tasks, heldout_id, heldout_ood, labels = generate_task_suites(
        noise_frac=noise_frac,
        seed=42,
        output_dir="data"
    )

    eval_actor = ActorAgent(model_name="mock-actor", price_table=config.get("PRICE_TABLE", {}))
    grader = Grader(seed=42)

    all_results = []
    Path("results").mkdir(exist_ok=True)

    for arm in arms:
        for seed in seeds:
            print(f"\n>>> Running Arm {arm} (Seed {seed}) ...")
            train_output = run_train_arm(
                arm=arm,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                noise_frac=noise_frac,
                train_tasks=train_tasks,
                labels=labels,
                config=config,
                checkpoints_dir="checkpoints"
            )

            # Evaluate each checkpoint on heldout_id and heldout_ood
            arm_seed_dir = Path("checkpoints") / f"{arm}_s{seed}"
            id_scores = []
            ood_scores = []
            cumulative_costs = []
            curator_decisions_all = []

            for epoch in range(epochs + 1):
                cp_dir = arm_seed_dir / f"checkpoint_{epoch}"
                playbook_file = cp_dir / "playbook.md"

                # Eval heldout ID
                res_id = evaluate_checkpoint_on_split(str(playbook_file), heldout_id, labels, eval_actor, grader)
                # Eval heldout OOD
                res_ood = evaluate_checkpoint_on_split(str(playbook_file), heldout_ood, labels, eval_actor, grader)

                id_scores.append(res_id["success_rate"])
                ood_scores.append(res_ood["success_rate"])

                if epoch > 0:
                    with open(cp_dir / "metadata.json", "r") as f:
                        meta = json.load(f)
                    cumulative_costs.append(meta["cumulative_cost_usd"]["total"])
                    curator_decisions_all.extend(meta.get("curator_decisions", []))
                else:
                    cumulative_costs.append(0.0)

            # Metrics
            initial_id = id_scores[0]
            final_id = id_scores[-1]
            total_learn_cost = cumulative_costs[-1]

            plasticity_info = compute_plasticity(initial_id, final_id, total_learn_cost)
            
            # Final pollution
            with open(arm_seed_dir / f"checkpoint_{epochs}" / "metadata.json", "r") as f:
                final_meta = json.load(f)
            final_pollution = final_meta.get("pollution", {})

            # Curator confusion
            curator_metrics = compute_curator_confusion(curator_decisions_all, labels) if curator_decisions_all else {}

            run_summary = {
                "arm": arm,
                "seed": seed,
                "initial_heldout_id": initial_id,
                "final_heldout_id": final_id,
                "gain_pp": plasticity_info["gain_pp"],
                "total_learning_cost_usd": total_learn_cost,
                "plasticity_per_1k": plasticity_info["plasticity_per_k"],
                "raw_plasticity": plasticity_info["raw_plasticity_per_k"],
                "pollution_rate": final_pollution.get("pollution_rate", 0.0),
                "strict_pollution_rate": final_pollution.get("strict_pollution_rate", 0.0),
                "total_rules": final_pollution.get("total_rules", 0),
                "id_score_curve": id_scores,
                "ood_score_curve": ood_scores,
                "cost_curve": cumulative_costs,
                "curator_metrics": curator_metrics
            }
            all_results.append(run_summary)
            print(f"[{arm} s{seed}] Gain: +{plasticity_info['gain_pp']:.1f}pp | Cost: ${total_learn_cost:.4f} | Plasticity: {plasticity_info['plasticity_per_k']:.1f}/$1k | Pollution: {final_pollution.get('pollution_rate', 0.0)*100:.0f}%")

    # Save results
    with open("results/all_runs.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Group by arm
    import pandas as pd
    df = pd.DataFrame(all_results)
    grouped = df.groupby("arm").agg({
        "gain_pp": ["mean", "std"],
        "total_learning_cost_usd": ["mean"],
        "plasticity_per_1k": ["mean", "std"],
        "pollution_rate": ["mean"],
        "strict_pollution_rate": ["mean"]
    })
    print("\n================== SUMMARY RESULTS TABLE ==================")
    print(grouped)
    print("===========================================================")

if __name__ == "__main__":
    main()
