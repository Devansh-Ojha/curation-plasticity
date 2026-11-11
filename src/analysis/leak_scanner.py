"""Scan prompts and task instructions while allowing one fixed taxonomy block."""
import json
from typing import List, Dict, Any

FORBIDDEN_PROMPT_TERMS = [
    "impossible", "easy", "noise", "planted", "near-miss",
    "learnable", "flaky", "labels.json",
]
FORBIDDEN_TASK_WORDS = ["nonexistent", "missing", "invalid", "unavailable"]

# This exact block is part of the fixed Curator instructions. Only its
# occurrence in a system message is exempt; the same terms elsewhere scan normally.
FIXED_CURATOR_TAXONOMY = """Labels (assign exactly one):
- Mastered: Multiple stable clean successes across separate rollouts, no recent regression. One success is not enough.
- Prompt-sensitive: Before/after pairs show prompt/playbook changes improve behavior (even partially). Run these.
- Operationally impossible: Available tools genuinely lack the needed capability, supported by repeated independent attempts. One unsupported claim is insufficient.
- No current optimization signal: The same failure persists across multiple relevant mutations with no progress.
- Flaky or uninformative: Materially similar behavior and end states repeatedly get inconsistent outcomes. Do not infer from one disagreement.
- Evaluator-broken: Explicit evidence of a wrong label or contradictory expectation.
- Infrastructure failure: Timeouts/errors; no evidence about the playbook.
- Uncertain: Run rather than skip."""


def _matches(text: str, terms: List[str], source: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    found = []
    lowered = text.lower()
    for term in terms:
        start = 0
        needle = term.lower()
        while True:
            index = lowered.find(needle, start)
            if index < 0:
                break
            found.append({**source, key: term, "offset": index,
                          "snippet": text[max(0, index - 30):index + len(term) + 30]})
            start = index + len(needle)
    return found


def scan_prompts_and_tasks_for_leaks(
    call_logs: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prompt_violations = []
    task_violations = []
    prompt_messages_scanned = 0
    unparsed_prompts = 0

    for call_index, call in enumerate(call_logs):
        prompt_used = call.get("prompt_used", "")
        try:
            messages = json.loads(prompt_used) if isinstance(prompt_used, str) else prompt_used
            if not isinstance(messages, list):
                raise ValueError("prompt is not a message list")
        except (TypeError, ValueError, json.JSONDecodeError):
            unparsed_prompts += 1
            prompt_violations.extend(_matches(
                str(prompt_used), FORBIDDEN_PROMPT_TERMS,
                {"call_index": call_index, "role": None}, "term",
            ))
            continue

        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            prompt_messages_scanned += 1
            content = str(message.get("content", ""))
            if message.get("role") == "system" and FIXED_CURATOR_TAXONOMY in content:
                content = content.replace(FIXED_CURATOR_TAXONOMY, "", 1)
            prompt_violations.extend(_matches(
                content, FORBIDDEN_PROMPT_TERMS,
                {"call_index": call_index, "message_index": message_index,
                 "role": message.get("role")}, "term",
            ))

    task_instructions_scanned = 0
    for task in tasks:
        task_instructions_scanned += 1
        instruction = str(task.get("instruction", ""))
        task_violations.extend(_matches(
            instruction, FORBIDDEN_TASK_WORDS,
            {"task_id": task.get("task_id")}, "word",
        ))

    passed = not prompt_violations and not task_violations and not unparsed_prompts
    return {
        "passed": passed,
        "items_scanned": {
            "call_log_entries": len(call_logs),
            "prompt_messages": prompt_messages_scanned,
            "task_instructions": task_instructions_scanned,
            "unparsed_prompts": unparsed_prompts,
        },
        "prompt_violations": prompt_violations,
        "task_violations": task_violations,
    }
