"""
Plots curves for held-out ID/OOD success, cumulative learning costs,
and rule pollution across checkpoints for each experimental arm.
Outputs figures to results/figures/
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_curves(results_file: str = "results/all_runs.json", out_dir: str = "results/figures"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(results_file, "r") as f:
        runs = json.load(f)

    # Group runs by arm
    arms_data = {}
    for r in runs:
        arm = r["arm"]
        if arm not in arms_data:
            arms_data[arm] = {
                "id_curves": [],
                "ood_curves": [],
                "cost_curves": [],
                "pollution": []
            }
        arms_data[arm]["id_curves"].append(r["id_score_curve"])
        arms_data[arm]["ood_curves"].append(r["ood_score_curve"])
        arms_data[arm]["cost_curves"].append(r["cost_curve"])
        arms_data[arm]["pollution"].append(r["pollution_rate"])

    epochs = range(len(list(arms_data.values())[0]["id_curves"][0]))
    colors = {"A0": "#e74c3c", "A1": "#2ecc71", "A2": "#f39c12", "A3": "#3498db"}
    labels = {
        "A0": "A0: Uncurated (Baseline)",
        "A1": "A1: Oracle (Clean upper-bound)",
        "A2": "A2: Numeric Curriculum",
        "A3": "A3: Curator (Trace-aware)"
    }

    # 1. Held-out ID Success Curve
    plt.figure(figsize=(9, 5))
    for arm, d in arms_data.items():
        arr = np.array(d["id_curves"])
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        plt.plot(epochs, mean, label=labels.get(arm, arm), color=colors.get(arm, "black"), lw=2.2)
        plt.fill_between(epochs, mean - std, mean + std, color=colors.get(arm, "black"), alpha=0.15)
    
    plt.title("Held-out ID Success Curve Across Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Checkpoint (Epoch)", fontsize=11)
    plt.ylabel("Held-out ID Success (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/heldout_id_curves.png", dpi=300)
    plt.close()

    # 2. Cumulative Learning Cost vs Held-out Gain (Plasticity Frontier)
    plt.figure(figsize=(9, 5))
    for arm, d in arms_data.items():
        cost_arr = np.mean(np.array(d["cost_curves"]), axis=0)
        id_arr = np.mean(np.array(d["id_curves"]), axis=0)
        plt.plot(cost_arr, id_arr, marker="o", label=labels.get(arm, arm), color=colors.get(arm, "black"), lw=2)

    plt.title("Plasticity Frontier: Held-out Gain vs Cumulative Learning Cost", fontsize=13, fontweight="bold")
    plt.xlabel("Cumulative Learning Cost (USD)", fontsize=11)
    plt.ylabel("Held-out ID Success (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/plasticity_frontier.png", dpi=300)
    plt.close()

    # 3. Rule Pollution Rate Comparison
    plt.figure(figsize=(7, 4.5))
    arm_names = list(arms_data.keys())
    poll_means = [np.mean(arms_data[a]["pollution"]) * 100 for a in arm_names]
    bar_colors = [colors.get(a, "gray") for a in arm_names]
    
    bars = plt.bar(arm_names, poll_means, color=bar_colors, width=0.55)
    plt.title("Playbook Rule Pollution Rate (%) Under 15% Planted Noise", fontsize=12, fontweight="bold")
    plt.ylabel("Pollution Rate (%)", fontsize=11)
    plt.ylim(0, 100)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/pollution_rate.png", dpi=300)
    plt.close()

    print(f"Generated evaluation plots in {out_dir}")

if __name__ == "__main__":
    plot_curves()
