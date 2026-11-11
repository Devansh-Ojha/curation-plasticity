from pathlib import Path
from types import SimpleNamespace
import io
import json
import time
import urllib.error

import pytest

from src.agents.nvidia_client import NVIDIAAPIError, NVIDIAClient
from scripts.experiments.run_real_experiment import safe_episode


class FlakyActor:
    def __init__(self, log_path, failures=2):
        self.client = SimpleNamespace(log_path=Path(log_path))
        self.failures = failures
        self.calls = 0

    def run_episode(self, task, playbook, api):
        self.calls += 1
        if self.calls <= self.failures:
            raise NVIDIAAPIError("temporary timeout")
        return {"api_failure": False, "final_answer": "ok"}


def test_api_error_retries_whole_episode(tmp_path):
    actor = FlakyActor(tmp_path / "calls.jsonl")
    result = safe_episode(actor, {"task_id": "t1"}, None, None, {}, [], phase="eval")
    assert actor.calls == 3
    assert result["final_answer"] == "ok"


def test_persistent_api_error_is_not_returned_for_scoring(tmp_path):
    actor = FlakyActor(tmp_path / "calls.jsonl", failures=5)
    failures = []
    with pytest.raises(NVIDIAAPIError, match="whole-episode attempts"):
        safe_episode(actor, {"task_id": "t1"}, None, None, {}, failures, phase="act")
    assert actor.calls == 3
    assert failures and failures[0]["task_id"] == "t1"


def test_reflector_phase_uses_increased_read_timeout(monkeypatch):
    monkeypatch.setenv("TEST_NVIDIA_KEY", "test-key")
    seen = []

    def timeout(req, timeout):
        seen.append(timeout)
        raise TimeoutError("simulated timeout")

    monkeypatch.setattr("urllib.request.urlopen", timeout)
    client = NVIDIAClient({
        "API_KEY_ENV_VAR": "TEST_NVIDIA_KEY",
        "REFERENCE_PRICE_TABLE": {"test-model": {
            "input_per_1m": 0.0, "output_per_1m": 0.0,
        }},
        "MAX_USD": 1.0,
        "MAX_429_RETRIES": 0,
        "REFLECTOR_READ_TIMEOUT_SECONDS": 301,
    })
    client.current_phase = "reflect"
    with pytest.raises(NVIDIAAPIError):
        client._chat_completion("test-model", [{"role": "user", "content": "x"}], max_tokens=1)
    assert seen == [301]


def test_429_sets_shared_endpoint_cooldown_and_retries(monkeypatch):
    monkeypatch.setenv("TEST_NVIDIA_KEY", "test-key")
    monkeypatch.setattr("time.sleep", lambda _: None)
    attempts = []
    success_body = json.dumps({"usage": {"prompt_tokens": 1, "completion_tokens": 1},
                               "choices": [{"message": {"content": "ok"},
                                            "finish_reason": "stop"}]}).encode()

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return success_body

    def urlopen(req, timeout):
        attempts.append(timeout)
        if len(attempts) == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {},
                                         io.BytesIO(b'{"status":429}'))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    endpoint = "https://test-cooldown.invalid/v1"
    client = NVIDIAClient({
        "API_BASE_URL": endpoint,
        "API_KEY_ENV_VAR": "TEST_NVIDIA_KEY",
        "REFERENCE_PRICE_TABLE": {"test-model": {
            "input_per_1m": 0.0, "output_per_1m": 0.0,
        }},
        "MAX_USD": 1.0,
        "MAX_429_RETRIES": 1,
        "ACTOR_THROTTLE_SECONDS": 0,
        "RATE_LIMIT_RPM": 100000,
    })
    result = client._chat_completion("test-model", [{"role": "user", "content": "x"}], max_tokens=1)
    assert result["text"] == "ok"
    assert len(attempts) == 2
    assert NVIDIAClient._global_cooldown_until[endpoint] > time.monotonic()
    assert NVIDIAClient._adaptive_global_interval[endpoint] >= 2 * client.global_min_interval
