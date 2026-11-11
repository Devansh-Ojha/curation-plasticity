"""Offline analysis of saved real A0/A3 results. Makes no network/API calls."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.analysis.metrics import compute_curator_confusion
from src.loop.provenance import Playbook

ROOT = Path("outputs/real_experiment")
SEEDS = (42, 43)
ARMS = ("A0", "A3")
CHECKPOINTS = (0, 4, 8, 12)


def load_run(arm: str, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]] | None:
    base = ROOT / f"seed-{seed}-{arm}"
    result_path = base / "results.json"
    labels_path = base / "task_data" / "labels.json"
    if not result_path.exists() or not labels_path.exists():
        return None
    return json.loads(result_path.read_text(encoding="utf-8")), json.loads(labels_path.read_text(encoding="utf-8"))


def success_rate(rows: Dict[str, Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return 100.0 * sum(bool(r.get("success")) for r in rows.values()) / len(rows)


def plot_results(aggregate: Dict[str, Any]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    import os
    mpl_cache = ROOT / ".matplotlib-cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm in ARMS:
        x = [e for e in CHECKPOINTS if aggregate[arm]["heldout_mean"].get(str(e)) is not None]
        y = [aggregate[arm]["heldout_mean"][str(e)] for e in x]
        ax.plot(x, y, marker="o", label=arm)
    ax.set(title="Held-out ID success by checkpoint", xlabel="Epoch", ylabel="Success (%)",
           xticks=CHECKPOINTS, ylim=(0, 100))
    ax.grid(True, alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "heldout_success.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    phases = ("act", "curate", "reflect", "gate", "eval")
    width = .36
    for i, arm in enumerate(ARMS):
        vals = [aggregate[arm]["phase_cost_mean"].get(p, 0.0) for p in phases]
        ax.bar([j + (i - .5) * width for j in range(len(phases))], vals,
               width=width, label=arm)
    ax.set(title="Mean hypothetical cost by phase", ylabel="USD", xticks=range(len(phases)),
           xticklabels=phases)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "phase_cost.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ARMS, [aggregate[a]["plasticity_per_1k_mean"] for a in ARMS])
    ax.set(title="Plasticity per $1,000 hypothetical", ylabel="pp / $1,000")
    fig.tight_layout()
    fig.savefig(ROOT / "plasticity.png", dpi=160)
    plt.close(fig)


def analyze() -> str:
    aggregate: Dict[str, Any] = {}
    lines = ["# Real A0 vs A3 Experiment Analysis", "",
             "Success metrics use the 40 held-out ID tasks. Costs are labeled hypothetical pricing.",
             "Category labels are read only from local `labels.json` for the Curator confusion and harness pollution analyses.", ""]
    confusion_totals = {arm: Counter() for arm in ARMS}

    for arm in ARMS:
        runs = []
        for seed in SEEDS:
            loaded = load_run(arm, seed)
            if loaded:
                runs.append((seed, *loaded))
        checkpoint_rates: Dict[str, List[float]] = defaultdict(list)
        phase_costs: Dict[str, List[float]] = defaultdict(list)
        plasticities = []
        raw_gains = []
        raw_plasticities = []
        pollution = []
        regression_counts: Dict[int, List[int]] = defaultdict(list)
        skipped: List[str] = []
        anchors: List[str] = []
        accepted_reflections = []
        rejected_reflections = []
        decisions = []
        heldout_regressions: Dict[int, List[int]] = defaultdict(list)

        for seed, result, labels in runs:
            heldout = result.get("heldout_by_epoch", {})
            for epoch in CHECKPOINTS:
                rows = heldout.get(str(epoch))
                if rows is not None:
                    checkpoint_rates[str(epoch)].append(success_rate(rows))
            prior_rows = None
            for epoch in CHECKPOINTS:
                rows = heldout.get(str(epoch))
                if rows is not None and prior_rows is not None:
                    heldout_regressions[epoch].append(sum(
                        bool(prior_rows.get(tid, {}).get("success")) and not bool(row.get("success"))
                        for tid, row in rows.items()))
                if rows is not None:
                    prior_rows = rows

            epochs = result.get("epochs", [])
            for record in epochs:
                epoch = int(record["epoch"])
                regression_counts[epoch].append(int(record.get("training_regressions", 0)))
                skipped.extend(f"seed{seed}:{tid}" for tid in record.get("curator_skipped_task_ids", []))
                anchors.extend(f"seed{seed}:{tid}" for tid in record.get("anchors_supplied", []))
                accepted_reflections.extend({"seed": seed, **r}
                                            for r in record.get("reflector_acceptances", []))
                rejected_reflections.extend({"seed": seed, **r}
                                            for r in record.get("reflector_rejections", []))
                for r in record.get("reflector_rejections", []):
                    # Gate rejection events added after the reflector attempt are stored at run level.
                    pass
            rejected_reflections.extend({"seed": seed, **r} for r in result.get("reflector_events", []))
            decisions.extend(result.get("curator_decisions", []))
            matrix = compute_curator_confusion(result.get("curator_decisions", []), labels)
            confusion_totals[arm].update({k: matrix[k] for k in ("tp", "fp", "tn", "fn")})

            final = heldout.get("12", {})
            initial = heldout.get("0", {})
            if initial and final:
                gain = success_rate(final) - success_rate(initial)
                total_learning_cost = sum(float(result.get("phase_costs_hypothetical", {}).get(p, 0.0))
                                          for p in ("act", "curate", "reflect", "gate"))
                raw = gain / total_learning_cost * 1000 if total_learning_cost > 0 else 0.0
                raw_gains.append(gain)
                raw_plasticities.append(raw)
                plasticities.append(raw if gain >= 10.0 else 0.0)

            final_epoch = next((e for e in reversed(epochs) if e.get("epoch") == 12), None)
            if final_epoch:
                rules = final_epoch.get("playbook", {}).get("rules", {})
                pb = Playbook()
                pb.rules = rules
                pollution.append(pb.compute_pollution(labels))
            for phase, value in result.get("phase_costs_hypothetical", {}).items():
                phase_costs[phase].append(float(value or 0.0))

        unique_rejections = {}
        for item in rejected_reflections:
            key = (item.get("seed"), item.get("epoch"), item.get("attempt"),
                   item.get("kind"), item.get("reason"))
            unique_rejections[key] = item
        rejected_reflections = list(unique_rejections.values())
        heldout_mean = {str(e): (sum(checkpoint_rates[str(e)]) / len(checkpoint_rates[str(e)])
                                 if checkpoint_rates[str(e)] else None) for e in CHECKPOINTS}
        phase_mean = {phase: (sum(values) / len(values) if values else 0.0)
                      for phase, values in phase_costs.items()}
        counters = Counter(r.get("kind", "unknown") for r in rejected_reflections)
        aggregate[arm] = {
            "heldout_mean": heldout_mean, "phase_cost_mean": phase_mean,
            "plasticity_per_1k_mean": sum(plasticities) / len(plasticities) if plasticities else 0.0,
        }
        lines += [f"## {arm}", "", "### Held-out ID success", "",
                  "| Epoch | Mean pass rate | Per-seed rates |", "|---:|---:|---|"]
        for epoch in CHECKPOINTS:
            per_seed = []
            for seed, result, _labels in runs:
                rows = result.get("heldout_by_epoch", {}).get(str(epoch))
                if rows is not None:
                    per_seed.append(f"{seed}: {success_rate(rows):.1f}%")
            mean = heldout_mean[str(epoch)]
            lines.append(f"| {epoch} | {mean:.1f}% | {', '.join(per_seed) or 'missing'} |" if mean is not None
                         else f"| {epoch} | missing | missing |")

        lines += ["", "### Costs, plasticity, and pollution", "",
                  "| Metric | Value |", "|---|---:|"]
        for phase in ("act", "curate", "reflect", "gate", "eval"):
            lines.append(f"| Mean {phase} cost (hypothetical USD) | ${phase_mean.get(phase, 0.0):.6f} |")
        lines.append(f"| Raw held-out gain (pp, mean) | {sum(raw_gains)/len(raw_gains) if raw_gains else 0:.2f} |")
        lines.append(f"| Plasticity per $1,000 (zero when gain <10 pp, mean) | {aggregate[arm]['plasticity_per_1k_mean']:.2f} |")
        lines.append(f"| Raw plasticity per $1,000 (mean) | {sum(raw_plasticities)/len(raw_plasticities) if raw_plasticities else 0:.2f} |")
        lines.append(f"| Polluted rules at epoch 12 (mean rate) | {sum(p['pollution_rate'] for p in pollution)/len(pollution) if pollution else 0:.3f} |")
        lines.append(f"| Strictly polluted rules at epoch 12 (mean rate) | {sum(p['strict_pollution_rate'] for p in pollution)/len(pollution) if pollution else 0:.3f} |")
        lines += ["", "### Regressions by epoch", "", "| Epoch | Training regressions | Held-out regressions at checkpoint |",
                  "|---:|---:|---:|"]
        for epoch in range(1, 13):
            train_vals = regression_counts.get(epoch, [])
            held_vals = heldout_regressions.get(epoch, [])
            lines.append(f"| {epoch} | {sum(train_vals)} | {sum(held_vals)} |")

        matrix = confusion_totals[arm]
        lines += ["", "### Curator confusion matrix (noise vs solvable)", "",
                  "| | Curator skip | Curator run |", "|---|---:|---:|",
                  f"| Ground-truth noise | {matrix['tp']} | {matrix['fn']} |",
                  f"| Ground-truth solvable | {matrix['fp']} | {matrix['tn']} |",
                  "", f"Tasks skipped: {len(skipped)}", "", f"Anchors supplied: {len(anchors)}", "",
                  f"Reflector acceptances: {len(accepted_reflections)}", "",
                  "Accepted reflection reasons:"]
        if accepted_reflections:
            for accepted in accepted_reflections:
                lines.append(f"- seed {accepted.get('seed')}, epoch {accepted.get('epoch')}, "
                             f"{accepted.get('ops_count', 0)} ops: {accepted.get('reason', '')}")
        else:
            lines.append("- None")
        lines += ["", "### Reflector rejections", "", f"By kind: {dict(counters)}", ""]
        for rejection in rejected_reflections:
            reason = str(rejection.get("reason", "" )).replace("|", "\\|").replace("\n", " ")
            lines.append(f"- seed {rejection.get('seed')}, epoch {rejection.get('epoch')}, "
                         f"{rejection.get('kind', 'unknown')}: {reason}")
        if not rejected_reflections:
            lines.append("- None")
        lines.append("")
        lines.append("Skipped task IDs: " + (", ".join(skipped) if skipped else "none"))
        lines.append("Anchor task IDs: " + (", ".join(anchors) if anchors else "none"))
        lines.append("")

    plot_results(aggregate)
    report = "\n".join(lines)
    (ROOT / "results.md").write_text(report, encoding="utf-8")
    return report


if __name__ == "__main__":
    print(analyze())
