"""Generate the article's figures from saved A0 results and the study design."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


INK = "#26323c"
MUTED = "#596671"
GRID = "#e5e9ec"
EDGE = "#cbd2d8"
SLATE = "#6c7f8d"
SALMON = "#a56a58"
PALE = "#f6f7f7"


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.labelcolor": MUTED, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": EDGE, "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def read_result(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Results file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def heldout_count(result: dict, epoch: int) -> tuple[int, int]:
    rows = result.get("heldout_by_epoch", {}).get(str(epoch))
    if not isinstance(rows, dict) or not rows:
        raise SystemExit(f"No held-out results at epoch {epoch} in saved results")
    total = len(rows)
    successes = sum(bool(row.get("success")) for row in rows.values())
    return successes, total


def save_checkpoint_comparison(short: dict, long: dict, out: Path) -> None:
    short_epochs = sorted(int(e) for e in short.get("heldout_by_epoch", {}))
    long_epochs = sorted(int(e) for e in long.get("heldout_by_epoch", {}))
    if len(short_epochs) < 2 or len(long_epochs) < 2:
        raise SystemExit("Both result files need baseline and follow-up held-out evaluations")
    series = [
        ("Three-epoch run · complete", short, short_epochs[-1],
         f"Completed through epoch {short.get('completed_epoch')}"),
        ("Longer A0 run · incomplete", long, long_epochs[-1],
         f"Stopped at checkpoint {long.get('completed_epoch')} of {long.get('total_epochs')}"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True,
                             constrained_layout=True)
    for ax, (title, result, last_epoch, note) in zip(axes, series):
        epochs = [0, last_epoch]
        counts = [heldout_count(result, epoch) for epoch in epochs]
        rates = [100 * n / total for n, total in counts]
        ax.plot(epochs, rates, color=SLATE, linewidth=1.6, zorder=2)
        ax.scatter([epochs[0]], [rates[0]], s=56, color=SLATE, zorder=3)
        ax.scatter([epochs[1]], [rates[1]], s=60, color=SALMON, zorder=3)
        ax.set_title(title, loc="left", fontsize=10, fontweight="medium", pad=12)
        ax.set_xticks(epochs, [f"Epoch {e}" for e in epochs])
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color=GRID, linewidth=.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(EDGE)
        ax.tick_params(axis="y", length=0, pad=6)
        ax.tick_params(axis="x", length=0, pad=7)
        for x, (n, total), rate in zip(epochs, counts, rates):
            ax.annotate(f"{n}/{total} · {rate:.1f}%",
                        (x, rate), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=9, color=INK)
        ax.text(.02, -.27, note, transform=ax.transAxes, fontsize=8.3,
                color=MUTED, ha="left", va="top")
    axes[0].set_ylabel("Held-out task success (%)", labelpad=9)
    fig.savefig(out, dpi=190, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def save_transitions(result: dict, out: Path) -> None:
    heldout = result.get("heldout_by_epoch", {})
    if "0" not in heldout or "3" not in heldout:
        raise SystemExit("Three-epoch results need held-out evaluations at epochs 0 and 3")
    baseline, followup = heldout["0"], heldout["3"]
    common = sorted(set(baseline) & set(followup))
    if len(common) != len(baseline) or len(common) != len(followup):
        raise SystemExit("Epoch 0 and 3 held-out task IDs do not match")
    counts = Counter((bool(baseline[tid]["success"]), bool(followup[tid]["success"]))
                     for tid in common)
    matrix = [[counts[(True, True)], counts[(True, False)]],
              [counts[(False, True)], counts[(False, False)]]]
    max_count = max(max(row) for row in matrix) or 1
    fig, ax = plt.subplots(figsize=(7.3, 3.5), constrained_layout=True)
    colors = {"stable": "#e8ecef", "improved": "#e5eeea", "regressed": "#eadbd5", "zero": "#fafafa"}
    names = [["Stayed successful", "Regressed"], ["Improved", "Stayed unsuccessful"]]
    for row in range(2):
        for col in range(2):
            n = matrix[row][col]
            category = ("stable" if row == col else ("regressed" if row == 0 else "improved"))
            face = colors["zero"] if n == 0 else colors[category]
            rect = Rectangle((col, 1-row), 1, 1, facecolor=face,
                             edgecolor="white", linewidth=3)
            ax.add_patch(rect)
            ax.text(col + .5, 1.58-row, f"{n}", ha="center", va="center",
                    fontsize=21, color=SALMON if category == "regressed" else INK,
                    fontweight="medium")
            ax.text(col + .5, 1.27-row, names[row][col], ha="center", va="center",
                    fontsize=9, color=MUTED)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([.5, 1.5], ["Success", "Unsuccessful"])
    ax.set_yticks([1.5, .5], ["Success", "Unsuccessful"])
    ax.set_xlabel("Epoch 3 outcome", labelpad=8)
    ax.set_ylabel("Epoch 0 outcome", labelpad=8)
    ax.tick_params(length=0, pad=7)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_title(f"Same {len(common)} held-out tasks · epoch 0 to epoch 3",
                 loc="left", fontsize=10, pad=12)
    fig.savefig(out, dpi=190, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def save_cost(result: dict, out: Path) -> None:
    costs = result.get("phase_costs_hypothetical", {})
    phases = [(p, float(costs.get(p, 0))) for p in ("eval", "act", "gate", "reflect", "curate")]
    total = sum(value for _, value in phases)
    if total <= 0:
        raise SystemExit("No phase cost estimates in the three-epoch results")
    labels = {"eval": "Held-out evaluation", "act": "Training episodes",
              "gate": "Gate reruns", "reflect": "Reflection", "curate": "Curation"}
    colors = {"eval": SALMON, "act": SLATE, "gate": "#8997a1",
              "reflect": "#b6bec4", "curate": "#d8dde1"}
    fig, ax = plt.subplots(figsize=(7.3, 3.2), constrained_layout=True)
    y = list(range(len(phases)))
    bars = ax.barh(y, [v for _, v in phases], color=[colors[p] for p, _ in phases], height=.54)
    ax.set_yticks(y, [labels[p] for p, _ in phases])
    ax.invert_yaxis()
    ax.set_xlabel("Configured hypothetical reference cost (USD)", labelpad=8)
    ax.set_xlim(0, max(v for _, v in phases) * 1.38)
    ax.grid(axis="x", color=GRID, linewidth=.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_color(EDGE)
    ax.tick_params(axis="y", length=0, pad=7); ax.tick_params(axis="x", length=0, pad=6)
    for bar, (phase, value) in zip(bars, phases):
        pct = value / total * 100
        ax.text(bar.get_width() + max(v for _, v in phases) * .035,
                bar.get_y() + bar.get_height()/2,
                f"${value:.4f} · {pct:.0f}%", va="center", fontsize=8.8, color=INK)
    ax.set_title(f"Three-epoch run · total estimate ${total:.4f}",
                 loc="left", fontsize=10, pad=12)
    fig.savefig(out, dpi=190, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def save_learning_loop(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.7, 4.2), constrained_layout=True)
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.25); ax.axis("off")
    cards = [
        ("01  EXPERIENCE", "Run training tasks\nthrough the actor and API.\n\nKeep the failed request,\nerror body, and any later\nsuccessful request.", SLATE),
        ("02  REFLECT", "The Reflector proposes\na rule from the trace.\n\nA3 can filter tasks and\nsupply a prior success.", SALMON),
        ("03  CHECK", "Grounding checks trace\nsupport before the gate.\n\nThe gate compares candidate\nand current scores on the\ntraining batch.", SALMON),
        ("04  RETAIN", "Keep the candidate if its\nbatch score does not fall.\n\nSave the resulting rule\nin the shared playbook.", SLATE),
    ]
    card_x = [.18, 2.68, 5.18, 7.68]
    card_w, card_y, card_h = 2.12, 1.43, 2.25
    for x, (heading, body, accent) in zip(card_x, cards):
        ax.add_patch(FancyBboxPatch((x, card_y), card_w, card_h,
                                   boxstyle="round,pad=0.04,rounding_size=0.06",
                                   facecolor="white", edgecolor=EDGE, linewidth=.9))
        ax.text(x+.17, card_y+card_h-.28, heading, ha="left", va="center",
                fontsize=8.8, color=accent, fontweight="bold")
        ax.plot([x+.17, x+card_w-.17], [card_y+card_h-.52]*2,
                color=GRID, linewidth=.8)
        ax.text(x+.17, card_y+card_h-.72, body, ha="left", va="top",
                fontsize=8.25, color=INK, linespacing=1.35)
    for i in range(3):
        left = card_x[i]+card_w+.035
        right = card_x[i+1]-.035
        ax.add_patch(FancyArrowPatch((left, card_y+card_h/2), (right, card_y+card_h/2),
                                     arrowstyle="-|>", mutation_scale=9,
                                     linewidth=1, color=SLATE))
    ax.text(5, 4.02, "FROM EXPERIENCE TO A TESTABLE PLAYBOOK",
            ha="center", va="center", fontsize=9, color=MUTED, fontweight="medium")
    # Separate evaluation lane: a frozen checkpoint is measured, never trained on.
    ax.plot([.18, 9.8], [1.12, 1.12], color=GRID, linewidth=1)
    ax.text(.22, .82, "HELD-OUT CHECK", ha="left", va="center",
            fontsize=8.4, color=SLATE, fontweight="bold")
    ax.text(2.12, .82,
            "At scheduled checkpoints, score the saved playbook on held-out tasks. These outcomes do not feed back into rule updates.",
            ha="left", va="center", fontsize=8.4, color=INK)
    # Show checkpoint evaluation as an observation from the retained playbook, with no return path.
    ax.add_patch(FancyArrowPatch((8.74, card_y-.03), (8.74, 1.16),
                                 arrowstyle="-|>", mutation_scale=8,
                                 linewidth=.9, linestyle="--", color=SLATE))
    ax.text(.22, .27,
            "A0: grounding + gate     ·     A0-raw: gate only     ·     A3: curate + ground + gate",
            ha="left", va="center", fontsize=8.2, color=MUTED)
    fig.savefig(out, dpi=190, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-results", type=Path,
                        default=Path("outputs/real_experiment/seed-42-A0/results.json"),
                        help="completed three-epoch A0 results.json")
    parser.add_argument("--long-results", type=Path,
                        default=Path("outputs/real_experiment/seed-42-A0-full/results.json"),
                        help="longer, incomplete A0 results.json")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    style()
    short, long = read_result(args.short_results), read_result(args.long_results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "learning-loop.png": lambda p: save_learning_loop(p),
        "heldout-success-a0-seed42.png": lambda p: save_checkpoint_comparison(short, long, p),
        "heldout-transitions-a0-seed42.png": lambda p: save_transitions(short, p),
        "estimated-cost-by-phase-a0-seed42.png": lambda p: save_cost(short, p),
    }
    for filename, render in outputs.items():
        target = args.output_dir / filename
        render(target)
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
