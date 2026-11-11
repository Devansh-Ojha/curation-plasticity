import json

from src.analysis.leak_scanner import FIXED_CURATOR_TAXONOMY, scan_prompts_and_tasks_for_leaks
from scripts.experiments.run_real_a3_smoke import CURATOR_PROMPT_TEMPLATE


def call_with_messages(messages):
    return {"prompt_used": json.dumps(messages)}


def test_scanner_exempts_only_fixed_curator_taxonomy_and_reports_counts():
    clean = scan_prompts_and_tasks_for_leaks(
        [call_with_messages([{"role": "system", "content": "Curator labels:\n" + FIXED_CURATOR_TAXONOMY}])],
        [{"task_id": "t1", "instruction": "Fetch user u1."}],
    )
    assert clean["passed"] is True
    assert clean["items_scanned"] == {
        "call_log_entries": 1,
        "prompt_messages": 1,
        "task_instructions": 1,
        "unparsed_prompts": 0,
    }


def test_scanner_still_detects_taxonomy_terms_outside_the_exempt_block():
    result = scan_prompts_and_tasks_for_leaks(
        [call_with_messages([
            {"role": "system", "content": FIXED_CURATOR_TAXONOMY},
            {"role": "user", "content": "This task is impossible."},
        ])],
        [],
    )
    assert result["passed"] is False
    assert any(item["term"] == "impossible" for item in result["prompt_violations"])
    assert result["items_scanned"]["prompt_messages"] == 2


def test_scanner_does_not_exempt_taxonomy_in_user_message_or_task_text():
    result = scan_prompts_and_tasks_for_leaks(
        [call_with_messages([{"role": "user", "content": FIXED_CURATOR_TAXONOMY}])],
        [{"task_id": "t1", "instruction": "This resource is unavailable."}],
    )
    assert result["passed"] is False
    assert result["prompt_violations"]
    assert result["task_violations"]


def test_current_smoke_curator_prompt_scans_clean_with_only_taxonomy_exempt():
    result = scan_prompts_and_tasks_for_leaks(
        [call_with_messages([{"role": "system", "content": CURATOR_PROMPT_TEMPLATE}])],
        [],
    )
    assert result["passed"] is True
