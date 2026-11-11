"""
GEPA-style Minibatch Gating:
Re-evaluates the minibatch after reflection with the candidate playbook.
Accepts if candidate minibatch score >= current score, else rejects (reverts playbook).
Gate re-run costs count toward learning cost.
Tracks accepted-but-regressed events (minibatch score rose or held while a previously passing task regressed).
"""
from typing import Dict, Any, List, Tuple
from src.agents.actor import ActorAgent
from src.testbed.mock_api import MockAPI
from src.testbed.grader import Grader
from src.loop.provenance import Playbook

def evaluate_minibatch(
    tasks: List[Dict[str, Any]],
    playbook: Playbook,
    actor: ActorAgent,
    grader: Grader,
    labels: Dict[str, Dict[str, Any]],
    mock_api: MockAPI
) -> Tuple[int, Dict[str, bool], float, Dict[str, int]]:
    """
    Runs actor on each task in minibatch.
    Returns:
        (total_score, per_task_outcome, total_cost_usd, tokens_dict)
    """
    score = 0
    outcomes = {}
    total_cost = 0.0
    prompt_tokens = 0
    completion_tokens = 0

    for t in tasks:
        t_id = t["task_id"]
        cat = labels.get(t_id, {}).get("category", "learnable")
        res = actor.run_episode(t, playbook, mock_api)
        passed, _ = grader.grade(t, cat, res["trajectory"], res["final_answer"])
        outcomes[t_id] = passed
        if passed:
            score += 1
        total_cost += res["cost_usd"]
        prompt_tokens += res["tokens"]["prompt_tokens"]
        completion_tokens += res["tokens"]["completion_tokens"]

    return score, outcomes, total_cost, {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}

def gate_minibatch(
    tasks: List[Dict[str, Any]],
    current_playbook: Playbook,
    candidate_playbook: Playbook,
    current_outcomes: Dict[str, bool],
    current_score: int,
    actor: ActorAgent,
    grader: Grader,
    labels: Dict[str, Dict[str, Any]],
    mock_api: MockAPI
) -> Tuple[bool, Playbook, Dict[str, Any]]:
    """
    Evaluates candidate playbook on the minibatch.
    Accepts if candidate_score >= current_score.
    Logs regression: any task that was True in current_outcomes and is False in candidate_outcomes.
    """
    candidate_score, candidate_outcomes, gate_cost, gate_tokens = evaluate_minibatch(
        tasks, candidate_playbook, actor, grader, labels, mock_api
    )

    accepted = candidate_score >= current_score

    # Check for accepted-but-regressed
    regressed_tasks = [
        t_id for t_id, passed in current_outcomes.items()
        if passed and not candidate_outcomes.get(t_id, False)
    ]
    accepted_but_regressed = accepted and len(regressed_tasks) > 0

    final_playbook = candidate_playbook if accepted else current_playbook

    gate_info = {
        "accepted": accepted,
        "current_score": current_score,
        "candidate_score": candidate_score,
        "gate_cost_usd": gate_cost,
        "gate_tokens": gate_tokens,
        "regressed_task_ids": regressed_tasks,
        "accepted_but_regressed": accepted_but_regressed
    }

    return accepted, final_playbook, gate_info
