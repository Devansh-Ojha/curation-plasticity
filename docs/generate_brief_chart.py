"""Regenerate the Matplotlib chart used in learning-from-experience.html."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path("outputs/real_experiment/seed-42-A0-full/results.json"),
                        help="saved real-run results.json")
    parser.add_argument("--output", type=Path,
                        default=Path("docs/assets/heldout-success-a0-seed42.png"),
                        help="chart PNG path")
    args = parser.parse_args()
    if not args.results.exists():
        parser.error(f"results file not found: {args.results}")

    results = json.loads(args.results.read_text(encoding="utf-8"))
    heldout = results.get("heldout_by_epoch", {})
    epochs = [epoch for epoch in (0, 4) if str(epoch) in heldout]
    if not epochs:
        parser.error("results file has no held-out evaluation at epoch 0 or 4")
    rows = [heldout[str(epoch)] for epoch in epochs]
    totals = [len(run) for run in rows]
    if any(total == 0 for total in totals):
        parser.error("held-out evaluation is empty")
    counts = [sum(bool(task.get("success")) for task in run.values()) for run in rows]
    rates = [count / total * 100 for count, total in zip(counts, totals)]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelcolor": "#4d5964", "text.color": "#26323c",
        "xtick.color": "#596671", "ytick.color": "#596671",
        "axes.edgecolor": "#cbd2d8", "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(7.2, 3.15), constrained_layout=True)
    colors = ["#6c7f8d" if epoch == 0 else "#a56a58" for epoch in epochs]
    bars = ax.bar([f"Epoch {epoch}" for epoch in epochs], rates,
                  width=.48, color=colors, zorder=3)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Held-out task success (%)", labelpad=10)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color="#e5e9ec", linewidth=.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#cbd2d8")
    ax.tick_params(axis="y", length=0, pad=7)
    ax.tick_params(axis="x", length=0, pad=9)
    for bar, count, total, rate in zip(bars, counts, totals, rates):
        ax.annotate(f"{count}/{total} · {rate:.0f}%",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    va="bottom", color="#26323c", fontsize=10)
    if len(rates) == 2:
        ax.text(.99, .94, f"Change: {rates[1] - rates[0]:+.0f} percentage points",
                transform=ax.transAxes, ha="right", va="top",
                color="#596671", fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", pad_inches=.12)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
