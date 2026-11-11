import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.obs.replay import infer_a3_phase, run_start_ns


def _call(model_id, system):
    return {"model_id": model_id,
            "prompt_used": json.dumps([
                {"role": "system", "content": system},
                {"role": "user", "content": "Task"},
            ])}


def test_legacy_a3_call_phases_are_inferred_without_exporting_prompts():
    assert infer_a3_phase(_call("meta/model", "You are an expert diagnostic Curator.")) == "curate"
    assert infer_a3_phase(_call("meta/model", "You are an expert prompt and playbook optimizer.")) == "reflect"
    assert infer_a3_phase(_call("nvidia/model", "You are an API actor.")) == "act"


def test_replay_filename_timestamp_uses_los_angeles_timezone():
    path = Path("calls-20261108T190538.jsonl")
    actual = datetime.fromtimestamp(run_start_ns(path) / 1_000_000_000, tz=ZoneInfo("UTC"))
    expected = datetime(2026, 11, 8, 19, 5, 38, tzinfo=ZoneInfo("America/Los_Angeles"))
    assert actual == expected
