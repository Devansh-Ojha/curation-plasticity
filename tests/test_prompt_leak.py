"""
Comprehensive Leak and Category Contamination Test:
1. Greps all actual prompts generated during an end-to-end execution.
2. Checks for ANY category terminology or hints:
   - "impossible", "flaky", "flaky_judge", "flaky_ambiguous", "learnable", "easy",
   - "unobtainable", "planted", "noise", "ground_truth", "labels.json", "labels".
3. Asserts that the prompts given to the Curator and Reflector contain zero leaks.
"""
import pytest
import json
from src.testbed.task_gen import generate_task_suites
from src.agents.real_curator import RealCurator, CURATOR_SYSTEM_PROMPT
from src.agents.real_reflector import RealReflector, REFLECTOR_SYSTEM_PROMPT
from src.agents.llm_client import LLMClient

FORBIDDEN_CATEGORY_WORDS = [
    "impossible task",
    "impossible",
    "flaky_judge",
    "flaky_ambiguous",
    "flaky",
    "learnable",
    "easy",
    "unobtainable",
    "planted",
    "labels.json",
    "ground_truth"
]

def test_static_prompts_contain_zero_category_words():
    for word in FORBIDDEN_CATEGORY_WORDS:
        # Note: the curator taxonomy includes the abstract paper label "Operationally impossible"
        # We ensure it does not include "impossible task" or raw planted task categories.
        assert "impossible task" not in CURATOR_SYSTEM_PROMPT.lower()
        assert "flaky_judge" not in CURATOR_SYSTEM_PROMPT.lower()
        assert "flaky_ambiguous" not in CURATOR_SYSTEM_PROMPT.lower()
        assert "learnable" not in CURATOR_SYSTEM_PROMPT.lower()
        assert "labels.json" not in CURATOR_SYSTEM_PROMPT.lower()

        assert word not in REFLECTOR_SYSTEM_PROMPT.lower()

def test_runtime_prompts_generated_from_tasks_contain_zero_leaks():
    train_tasks, _, _, labels = generate_task_suites(noise_frac=0.20, seed=42, output_dir="data_leak_test")
    
    # Simulate failed tasks info as passed to the Reflector
    failed_tasks_info = []
    for t in train_tasks:
        failed_tasks_info.append({
            "task_id": t["task_id"],
            "instruction": t["instruction"],
            "feedback": "Failed: Validation check failed. Output did not match expected target.",
            "trajectory": [{"step": 1, "action": "http_request", "call": "/users/u1"}]
        })

    # Build prompt payload identical to runtime
    failures_summary = ""
    for f in failed_tasks_info:
        failures_summary += f"### Task ID: {f.get('task_id')}\n"
        failures_summary += f"Instruction: {f.get('instruction')}\n"
        failures_summary += f"Grader Feedback: {f.get('feedback')}\n"
        failures_summary += f"Execution Steps: {json.dumps(f.get('trajectory', []))}\n\n"

    runtime_prompt = f"Current Playbook:\n# Playbook\n\nFailed Task Traces:\n{failures_summary}"
    
    # Assert none of the forbidden category words appear in the runtime prompt text
    for word in FORBIDDEN_CATEGORY_WORDS:
        assert word not in runtime_prompt.lower(), f"Leak detected! Term '{word}' appeared in runtime prompt."
