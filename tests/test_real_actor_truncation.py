from src.agents.real_llm_actor import RealLLMActor
from src.loop.provenance import Playbook
from src.testbed.grader import Grader
from src.testbed.mock_api import MockAPI
from run_real_baseline import score_baseline_result


class SequenceClient:
    def __init__(self):
        self.responses = [
            ("{\"action\": \"finish\",", "length"),
            ('{"action":"finish","answer":"done"}', "stop"),
        ]
        self.calls = 0

    def chat_completion(self, **kwargs):
        text, finish_reason = self.responses[self.calls]
        self.calls += 1
        return {
            "text": text,
            "call_record": {
                "input_tokens": 12, "output_tokens": 4,
                "hypothetical_cost_usd": 0.0,
                "finish_reason": finish_reason,
            },
        }


def test_truncated_json_is_retried_once_and_episode_continues():
    client = SequenceClient()
    actor = RealLLMActor(client=client, model_id="test-model", max_steps=12)
    task = {"task_id": "t_truncated", "instruction": "Return done", "expected": {"type": "exact", "target_val": "done"}}

    result = actor.run_episode(task, Playbook(), MockAPI())
    passed, _ = score_baseline_result(Grader(seed=42), task, "easy", result)

    assert client.calls == 2
    assert result["final_answer"] == "done"
    assert result["truncations"] == 1
    assert result["format_errors"] == 0
    assert passed is True


def test_format_error_after_retry_counts_once_and_next_step_runs():
    client = SequenceClient()
    client.responses = [
        ("{\"action\":", "length"),
        ("{not json", "stop"),
        ('{"action":"finish","answer":"done"}', "stop"),
    ]
    actor = RealLLMActor(client=client, model_id="test-model", max_steps=12)
    task = {"task_id": "t_truncated_retry", "instruction": "Return done", "expected": {"type": "exact", "target_val": "done"}}

    result = actor.run_episode(task, Playbook(), MockAPI())
    passed, _ = score_baseline_result(Grader(seed=42), task, "easy", result)

    assert client.calls == 3
    assert result["final_answer"] == "done"
    assert result["truncations"] == 1
    assert result["format_errors"] == 1
    assert passed is False


def test_null_content_on_truncation_is_retried_without_crashing():
    client = SequenceClient()
    client.responses = [
        (None, "length"),
        ('{"action":"finish","answer":"done"}', "stop"),
    ]
    actor = RealLLMActor(client=client, model_id="test-model", max_steps=12)
    task = {"task_id": "t_null_truncated", "instruction": "Return done",
            "expected": {"type": "exact", "target_val": "done"}}

    result = actor.run_episode(task, Playbook(), MockAPI())
    passed, _ = score_baseline_result(Grader(seed=42), task, "easy", result)

    assert client.calls == 2
    assert result["final_answer"] == "done"
    assert result["truncations"] == 1
    assert result["format_errors"] == 0
    assert passed is True
