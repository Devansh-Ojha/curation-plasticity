import pytest
from src.loop.provenance import Playbook
from src.loop.gating import gate_minibatch
from src.agents.actor import ActorAgent
from src.testbed.mock_api import MockAPI
from src.testbed.grader import Grader

def test_provenance_recorded_by_harness():
    playbook = Playbook(max_tokens=2000)
    source_tasks = ["t_noise_01", "t_noise_02"]
    ops = [{"op": "add", "content": "Always pass header X-Api-Key."}]
    
    success = playbook.apply_ops(ops, epoch=1, source_task_ids=source_tasks)
    assert success is True
    assert "rule_001" in playbook.rules
    assert playbook.rules["rule_001"]["source_task_ids"] == source_tasks

def test_cost_accounting_includes_components():
    actor = ActorAgent(
        model_name="mock-actor",
        price_table={"mock-actor": {"prompt_per_1m": 1.0, "completion_per_1m": 2.0}}
    )
    api = MockAPI()
    task = {"task_id": "t_01", "instruction": "Fetch user details for ID 'u1'"}
    res = actor.run_episode(task, Playbook(), api)
    
    assert res["cost_usd"] > 0
    assert "prompt_tokens" in res["tokens"]
    assert "completion_tokens" in res["tokens"]

def test_playbook_token_cap_rejection():
    # Playbook with tiny cap
    playbook = Playbook(max_tokens=5)
    ops = [{"op": "add", "content": "This is a very long rule content that exceeds the tiny token budget cap."}]
    success = playbook.apply_ops(ops, epoch=1, source_task_ids=["t_01"])
    assert success is False
    assert len(playbook.rules) == 0 # Rolled back
