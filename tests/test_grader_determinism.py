import pytest
import json
from src.testbed.grader import Grader
from src.agents.actor import ActorAgent
from src.testbed.mock_api import MockAPI
from src.loop.provenance import Playbook
from src.obs.experiments import evaluate_checkpoint_on_split

def test_easy_and_learnable_graders_are_100_percent_deterministic():
    grader1 = Grader(seed=42)
    grader2 = Grader(seed=999) # different seed must not alter clean outcomes

    task = {"task_id": "t_clean", "expected": {"type": "exact", "target_val": "alice@example.com"}}

    for g in [grader1, grader2]:
        p, _ = g.grade(task, "learnable", {}, "alice@example.com")
        assert p is True
        p, _ = g.grade(task, "learnable", {}, "wrong@example.com")
        assert p is False

def test_re_evaluating_same_checkpoint_twice_yields_identical_scores():
    # Load held-out ID tasks
    with open("data/heldout_id.json") as f:
        heldout = json.load(f)
    with open("data/labels.json") as f:
        labels = json.load(f)

    actor = ActorAgent(model_name="mock-actor")
    grader = Grader(seed=42)
    playbook_file = "tests/fixtures/playbooks/empty.md"

    # Evaluate run 1
    eval_1 = evaluate_checkpoint_on_split(playbook_file, heldout, labels, actor, grader)
    
    # Evaluate run 2
    eval_2 = evaluate_checkpoint_on_split(playbook_file, heldout, labels, actor, grader)

    assert eval_1["success_count"] == eval_2["success_count"], \
        f"Non-deterministic eval: {eval_1['success_count']} != {eval_2['success_count']}"
    assert eval_1["success_rate"] == eval_2["success_rate"]


def test_exact_value_and_count_accept_natural_language_answers():
    grader = Grader(seed=42)
    exact_task = {"expected": {"type": "exact", "target_val": "ord_105"}}
    count_task = {"expected": {"type": "count", "target_val": 5}}

    assert grader.grade(exact_task, "learnable", {}, "The last order ID is ord_105.")[0] is True
    assert grader.grade(exact_task, "learnable", {}, "The last order ID is ord_1050.")[0] is False
    assert grader.grade(count_task, "learnable", {}, "The total count is 5.")[0] is True
