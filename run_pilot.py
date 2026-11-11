"""
Small-scale single seed pilot runner:
- Arms: A0 (Uncurated) and A3 (Curator)
- Seed: 42
- Epochs: 6
- Noise: 0.20
- Held-out ID: 10 tasks evaluated at epochs [0, 3, 6]
- Pre-flight cost printed per phase with MAX_USD guard
- Tracks:
  - Tasks dropped by A2 / skipped by A3 (categorized by Curator label)
  - Anchors supplied and anchor abstentions
  - Rule ablation harm rate on final playbook
  - Traces exported to Phoenix project 'curation-plasticity-pilot'
"""
import os
import yaml
import json
from pathlib import Path
from src.testbed.task_gen import generate_task_suites
from src.loop.train_loop import run_train_arm
from src.obs.experiments import evaluate_checkpoint_on_split
from src.agents.actor import ActorAgent
from src.testbed.grader import Grader
from src.analysis.metrics import compute_plasticity, compute_curator_confusion
from src.analysis.rule_ablation import evaluate_rule_ablation_harm
from src.obs.cost_guard import estimate_run_cost

def run_pilot():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Pre-flight cost estimation
    cost_est = estimate_run_cost(
        config=config,
        num_arms=2,
        num_seeds=1,
        num_epochs=6,
        batch_size=10,
        heldout_size=10,
        eval_epochs_count=3
    )

    print("================================================================")
    print("PHASE 1: PRE-FLIGHT COST ESTIMATION & BUDGET GUARD")
    print(f"Actor Model:    {config.get('ACTOR_MODEL')} (Price per 1M: ${config['PRICE_TABLE'][config.get('ACTOR_MODEL')]['prompt_per_1m']} / ${config['PRICE_TABLE'][config.get('ACTOR_MODEL')]['completion_per_1m']})")
    print(f"Optimizer Model:{config.get('OPTIMIZER_MODEL')} (Price per 1M: ${config['PRICE_TABLE'][config.get('OPTIMIZER_MODEL')]['prompt_per_1m']} / ${config['PRICE_TABLE'][config.get('OPTIMIZER_MODEL')]['completion_per_1m']})")
    print("Estimated Spend by Phase:")
    print(f"  - Actor Rollouts:         ${cost_est['actor_rollout_usd']:.4f}")
    print(f"  - Curator Evaluations:    ${cost_est['curation_usd']:.4f}")
    print(f"  - Reflector Multi-Tries:  ${cost_est['reflection_usd']:.4f}")
    print(f"  - Gate Re-Runs:           ${cost_est['gate_reruns_usd']:.4f}")
    print(f"  - Held-out Evaluations:   ${cost_est['heldout_eval_usd']:.4f}")
    print(f"Estimated Total:            ${cost_est['grand_total_usd']:.4f} (MAX_USD Cap: ${cost_est['max_usd_cap']:.2f})")
    if cost_est["exceeds_cap"]:
        raise RuntimeError("Estimated cost exceeds MAX_USD! Aborting run.")
    print("================================================================")

    # 2. Resized Task Matrix Generation
    train_tasks, heldout_id, heldout_ood, labels = generate_task_suites(
        noise_frac=0.20,
        seed=42,
        output_dir="data_pilot"
    )
    pilot_heldout_id = heldout_id[:10]

    eval_actor = ActorAgent(model_name=config.get("ACTOR_MODEL", "claude-haiku-5-5"), price_table=config.get("PRICE_TABLE", {}))
    grader = Grader(seed=42)

    pilot_results = []
    Path("results/pilot").mkdir(parents=True, exist_ok=True)

    for arm in ["A0", "A3"]:
        print(f"\n>>> Running Pilot: Arm {arm} (Seed 42, 6 Epochs, 20% Noise) ...")
        train_output = run_train_arm(
            arm=arm,
            seed=42,
            epochs=6,
            batch_size=10,
            noise_frac=0.20,
            train_tasks=train_tasks,
            labels=labels,
            config=config,
            checkpoints_dir="checkpoints/pilot"
        )

        arm_dir = Path("checkpoints/pilot") / f"{arm}_s42"
        id_evals = {}
        all_curator_decisions = []

        # Evaluate only at epochs 0, 3, 6
        for ep in [0, 3, 6]:
            pb_file = arm_dir / f"checkpoint_{ep}" / "playbook.md"
            eval_res = evaluate_checkpoint_on_split(str(pb_file), pilot_heldout_id, labels, eval_actor, grader)
            id_evals[ep] = eval_res["success_rate"]

        # Collect curator stats
        for ep in range(1, 7):
            with open(arm_dir / f"checkpoint_{ep}" / "metadata.json") as f:
                meta = json.load(f)
            all_curator_decisions.extend(meta.get("curator_decisions", []))

        # Checkpoint 6 final playbook
        final_pb = train_output["final_playbook"]

        # 3. Rule Ablation Harm Rate Evaluation
        harm_info = evaluate_rule_ablation_harm(
            final_playbook=final_pb,
            heldout_tasks=pilot_heldout_id,
            labels=labels,
            actor=eval_actor,
            grader=grader,
            repeats=2,
            noise_threshold_pp=2.5
        )

        # 4. Curator audit & metrics
        curator_metrics = compute_curator_confusion(all_curator_decisions, labels) if all_curator_decisions else {}
        
        # Count Curator actions
        skipped_by_label = {}
        anchors_supplied = 0
        anchor_abstentions = 0
        for d in all_curator_decisions:
            if d.get("decision") == "skip":
                lbl = d.get("label", "Unknown")
                skipped_by_label[lbl] = skipped_by_label.get(lbl, 0) + 1
            if d.get("anchor_success_id"):
                anchors_supplied += 1
            else:
                anchor_abstentions += 1

        total_cost = train_output["epoch_logs"][-1]["cumulative_cost_usd"]["total"]
        gain_pp = id_evals[6] - id_evals[0]
        plasticity = compute_plasticity(id_evals[0], id_evals[6], total_cost)

        summary = {
            "arm": arm,
            "seed": 42,
            "initial_score": id_evals[0],
            "final_score": id_evals[6],
            "gain_pp": gain_pp,
            "cost_usd": total_cost,
            "plasticity_per_1k": plasticity["plasticity_per_k"],
            "raw_plasticity": plasticity["raw_plasticity_per_k"],
            "pollution": final_pb.compute_pollution(labels),
            "harm_evaluation": harm_info,
            "curator_actions": {
                "total_decisions": len(all_curator_decisions),
                "skipped_by_label": skipped_by_label,
                "anchors_supplied": anchors_supplied,
                "anchor_abstentions": anchor_abstentions,
                "confusion": curator_metrics
            }
        }
        pilot_results.append(summary)

        print(f"[{arm}] Initial: {id_evals[0]:.1f}% -> Final: {id_evals[6]:.1f}% (+{gain_pp:.1f}pp)")
        print(f"[{arm}] Cumulative Learning Cost: ${total_cost:.4f}")
        print(f"[{arm}] Rule Harm Rate: {harm_info['harm_rate']*100:.1f}% ({harm_info['harmful_rules_count']}/{harm_info['total_rules']} rules)")
        if arm == "A3":
            print(f"[{arm}] Curator Skips: {sum(skipped_by_label.values())} (Breakdown: {skipped_by_label})")
            print(f"[{arm}] Anchors Supplied: {anchors_supplied} | Abstentions: {anchor_abstentions}")

    with open("results/pilot/pilot_summary.json", "w") as f:
        json.dump(pilot_results, f, indent=2)

    return pilot_results

if __name__ == "__main__":
    run_pilot()
