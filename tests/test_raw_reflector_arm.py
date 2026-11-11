import json

import run_real_experiment
from src.loop.provenance import Playbook


class FixedClient:
    current_phase = None

    def chat_completion(self, *args, **kwargs):
        return {"text": json.dumps([{"op": "add", "content": "intentionally ungrounded proposal"}])}


def test_a0_raw_skips_grounding_but_keeps_operation_flow(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("raw arm must bypass the grounding check")

    monkeypatch.setattr(run_real_experiment, "check_rule_grounding", fail_if_called)
    ops, rejections, _ = run_real_experiment.call_reflector(
        FixedClient(), [], Playbook(max_tokens=2000), [], {}, "A0-raw", 42, 1, "optimizer-model")
    assert ops == [{"op": "add", "content": "intentionally ungrounded proposal"}]
    assert rejections == []
