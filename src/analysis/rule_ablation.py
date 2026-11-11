"""
Harm rate evaluator via systematic Rule Ablation on the final checkpoint playbook.
For each rule in the final playbook:
1. Removes that single rule to create an ablated playbook.
2. Re-evaluates heldout_id with 2 repeats to filter evaluation noise.
3. If removing the rule raises held-out mean score by >= 1 task-equivalent (> 2.5 pp),
   the rule is declared CAUSALLY HARMFUL.
Reports share of total rules harmful and share of polluted rules harmful.
"""
from typing import Dict, Any, List
import copy
from src.agents.actor import ActorAgent
from src.testbed.grader import Grader
from src.testbed.mock_api import MockAPI
from src.loop.provenance import Playbook
from src.obs.experiments import evaluate_checkpoint_on_split

def evaluate_rule_ablation_harm(
    final_playbook: Playbook,
    heldout_tasks: List[Dict[str, Any]],
    labels: Dict[str, Dict[str, Any]],
    actor: ActorAgent,
    grader: Grader,
    repeats: int = 2,
    noise_threshold_pp: float = 2.5 # >= 1 task equivalent on 40 tasks (1/40 = 2.5%)
) -> Dict[str, Any]:
    """
    Performs rule ablation harm analysis.
    """
    if not final_playbook.rules:
        return {
            "total_rules": 0,
            "harmful_rules": [],
            "harm_rate": 0.0,
            "polluted_rules_harm_rate": 0.0
        }

    # 1. Baseline evaluation of full final playbook (averaged over repeats)
    base_scores = []
    for _ in range(repeats):
        res = evaluate_checkpoint_on_split(
            playbook_path="", # passed directly via playbook object
            tasks_split=heldout_tasks,
            labels=labels,
            actor=actor,
            grader=grader,
            playbook_obj=final_playbook
        )
        base_scores.append(res["success_rate"])
    baseline_score = sum(base_scores) / len(base_scores)

    harmful_rules = []
    polluted_harmful_rules = []

    # 2. Iterate each rule and ablate
    for r_id, r_info in final_playbook.rules.items():
        ablated = final_playbook.clone()
        del ablated.rules[r_id]

        ablated_scores = []
        for _ in range(repeats):
            res = evaluate_checkpoint_on_split(
                playbook_path="",
                tasks_split=heldout_tasks,
                labels=labels,
                actor=actor,
                grader=grader,
                playbook_obj=ablated
            )
            ablated_scores.append(res["success_rate"])
        ablated_mean = sum(ablated_scores) / len(ablated_scores)

        # Delta: ablated_score - baseline_score
        # If removing the rule IMPROVES heldout score, the rule was harmful!
        delta = ablated_mean - baseline_score
        is_harmful = delta >= noise_threshold_pp

        # Check if rule was polluted via provenance
        sources = r_info.get("source_task_ids", [])
        is_polluted = any(
            labels.get(s, {}).get("category") in ["impossible", "flaky_judge", "flaky_ambiguous"]
            for s in sources
        )

        if is_harmful:
            harmful_rules.append({"rule_id": r_id, "delta_pp": delta, "is_polluted": is_polluted})
            if is_polluted:
                polluted_harmful_rules.append(r_id)

    total_rules = len(final_playbook.rules)
    harm_rate = len(harmful_rules) / total_rules if total_rules > 0 else 0.0
    
    polluted_count = sum(
        1 for r in final_playbook.rules.values()
        if any(labels.get(s, {}).get("category") in ["impossible", "flaky_judge", "flaky_ambiguous"] for s in r.get("source_task_ids", []))
    )
    polluted_harm_rate = len(polluted_harmful_rules) / polluted_count if polluted_count > 0 else 0.0

    return {
        "baseline_score": baseline_score,
        "total_rules": total_rules,
        "polluted_rules_count": polluted_count,
        "harmful_rules_count": len(harmful_rules),
        "harm_rate": harm_rate,
        "polluted_rules_harm_rate": polluted_harm_rate,
        "harmful_details": harmful_rules
    }
