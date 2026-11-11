"""
Analysis and metrics computation:
- Plasticity: held-out score gain per $1,000 learning cost
- Rule pollution rate and strict pollution rate
- Regressions (tasks pass-before, fail-after)
- Accepted-but-regressed count
- Curator confusion matrix vs ground-truth labels
"""
from typing import Dict, Any, List
import json
import numpy as np

def compute_plasticity(
    initial_score: float,
    final_score: float,
    cumulative_cost_usd: float
) -> Dict[str, float]:
    """
    Plasticity at final checkpoint = (S_final - S_0) / C_final * 1000
    If gain < 10 percentage points, reports plasticity as 0 (matching Agent Plasticity paper),
    while also reporting raw gain.
    """
    gain_pp = final_score - initial_score
    if cumulative_cost_usd <= 0:
        return {"plasticity_per_k": 0.0, "gain_pp": gain_pp, "cost_usd": cumulative_cost_usd}

    raw_plasticity = (gain_pp / cumulative_cost_usd) * 1000.0
    official_plasticity = raw_plasticity if gain_pp >= 10.0 else 0.0

    return {
        "plasticity_per_k": official_plasticity,
        "raw_plasticity_per_k": raw_plasticity,
        "gain_pp": gain_pp,
        "cost_usd": cumulative_cost_usd
    }

def compute_curator_confusion(
    curator_decisions: List[Dict[str, Any]],
    labels: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes precision, recall, false-skip rate, and confusion matrix of Curator vs planted noise.
    """
    tp = 0 # True noise skipped
    fp = 0 # Solvable task skipped (false skip)
    tn = 0 # Solvable task run
    fn = 0 # Noise task run

    for dec in curator_decisions:
        t_id = dec["task_id"]
        true_cat = labels.get(t_id, {}).get("category")
        is_true_noise = true_cat in ["impossible", "flaky_judge", "flaky_ambiguous"]
        decision = dec.get("decision")

        if decision == "skip":
            if is_true_noise:
                tp += 1
            else:
                fp += 1
        elif decision == "run":
            if is_true_noise:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_skip_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "false_skip_rate": false_skip_rate
    }
