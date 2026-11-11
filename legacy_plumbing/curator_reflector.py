"""
Curator and Reflector implementation following GEPA-Curator & Agent Plasticity specs:
- Curator:
  - Input: task text, per-run history (latest H=2 full runs, older runs compacted), IDs of available clean successes
  - Decides run/skip and picks at most 1 trusted past success as anchor
  - Output exact JSON: reason, label, decision, skip_epochs, anchor_reason, anchor_success_id, supporting_success_ids
  - Underexplored-first: < 2 runs selected without Curator call
  - Fallback: malformed output defaults to "run"
- Reflector:
  - Ingests failed trajectories + feedback + current playbook (+ anchors)
  - Returns structured edit ops (add, modify, delete)
  - Max 3 attempts, rejected attempts' cost counts
"""
from typing import Dict, Any, List, Optional, Tuple
import json

CURATOR_LABELS = [
    "Mastered",
    "Prompt-sensitive",
    "Operationally impossible",
    "No current optimization signal",
    "Flaky or uninformative",
    "Evaluator-broken",
    "Infrastructure failure",
    "Uncertain"
]

class Curator:
    def __init__(
        self,
        model_name: str = "mock-optimizer",
        history_window: int = 2,
        price_table: Optional[Dict[str, Any]] = None
    ):
        self.model_name = model_name
        self.history_window = history_window
        self.price_table = price_table or {}

    def decide(
        self,
        task: Dict[str, Any],
        task_history: List[Dict[str, Any]],
        available_success_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluates a candidate task to decide run/skip and optional anchor.
        """
        task_id = task.get("task_id")
        runs_count = len(task_history)

        # -------------------------------------------------------------
        # Rule: Underexplored-first: < 2 runs selected without Curator call
        # -------------------------------------------------------------
        if runs_count < 2:
            return {
                "task_id": task_id,
                "called_llm": False,
                "reason": "Underexplored task (< 2 runs). Exploring to build history.",
                "label": "Uncertain",
                "decision": "run",
                "skip_epochs": 1,
                "anchor_reason": "No anchor needed during initial exploration.",
                "anchor_success_id": None,
                "supporting_success_ids": [],
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0},
                "cost_usd": 0.0
            }

        prompt_tokens = 450
        completion_tokens = 90
        price_cfg = self.price_table.get(self.model_name, {"prompt_per_1m": 3.00, "completion_per_1m": 15.00})

        # History inspection (Deterministic mock curator logic mirroring Sonnet 3.5 reasoning)
        recent_runs = task_history[-self.history_window:]
        outcomes = [r.get("passed", False) for r in task_history]
        all_passed = all(outcomes)
        all_failed = not any(outcomes)
        has_mixed = any(outcomes) and not all(outcomes)

        # Check evidence in recent failure messages or trajectories
        fail_reasons = " ".join([r.get("feedback", "") for r in recent_runs if not r.get("passed")])

        if all_passed and len(outcomes) >= 2:
            decision_payload = {
                "reason": "Multiple stable clean successes across separate rollouts with no regression.",
                "label": "Mastered",
                "decision": "skip",
                "skip_epochs": 2,
                "anchor_reason": "Task is mastered; no anchor assigned.",
                "anchor_success_id": None,
                "supporting_success_ids": []
            }
        elif "Endpoint /" in fail_reasons or "does not exist" in fail_reasons or "could not be verified" in fail_reasons:
            decision_payload = {
                "reason": "Endpoint genuinely lacks capability or resource is missing across repeated attempts.",
                "label": "Operationally impossible",
                "decision": "skip",
                "skip_epochs": 3,
                "anchor_reason": "Cannot anchor an impossible task.",
                "anchor_success_id": None,
                "supporting_success_ids": []
            }
        elif has_mixed and ("assertion mismatch" in fail_reasons.lower() or "interpretation mismatch" in fail_reasons.lower()):
            decision_payload = {
                "reason": "Materially similar end states repeatedly produce inconsistent grader verdicts.",
                "label": "Flaky or uninformative",
                "decision": "skip",
                "skip_epochs": 2,
                "anchor_reason": "Flaky task offers uninformative learning signal.",
                "anchor_success_id": None,
                "supporting_success_ids": []
            }
        elif all_failed and len(outcomes) >= 4 and "syntax error" not in fail_reasons.lower():
            decision_payload = {
                "reason": "Same failure persists across multiple epochs with no optimization signal.",
                "label": "No current optimization signal",
                "decision": "skip",
                "skip_epochs": 2,
                "anchor_reason": "No optimization signal detected.",
                "anchor_success_id": None,
                "supporting_success_ids": []
            }
        else:
            # Prompt sensitive / learnable failure -> Run and pick anchor if available
            anchor_id = None
            supporting = []
            anchor_reason = "No corroborated success available."
            if len(available_success_ids) >= 2:
                # Require at least one independent clean success
                anchor_id = available_success_ids[0]
                supporting = [available_success_ids[1]]
                anchor_reason = f"Corroborated clean success {anchor_id} selected as contrast."

            decision_payload = {
                "reason": "Task shows prompt sensitivity or actionable error feedback.",
                "label": "Prompt-sensitive",
                "decision": "run",
                "skip_epochs": 1,
                "anchor_reason": anchor_reason,
                "anchor_success_id": anchor_id,
                "supporting_success_ids": supporting
            }

        cost_usd = (prompt_tokens / 1_000_000) * price_cfg["prompt_per_1m"] + \
                   (completion_tokens / 1_000_000) * price_cfg["completion_per_1m"]

        return {
            "task_id": task_id,
            "called_llm": True,
            **decision_payload,
            "tokens": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            "cost_usd": cost_usd
        }


class Reflector:
    def __init__(
        self,
        model_name: str = "mock-optimizer",
        max_attempts: int = 3,
        price_table: Optional[Dict[str, Any]] = None
    ):
        self.model_name = model_name
        self.max_attempts = max_attempts
        self.price_table = price_table or {}

    def reflect(
        self,
        failed_tasks_info: List[Dict[str, Any]],
        current_playbook_md: str,
        anchors: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], int]:
        """
        Reflects on failures and returns (edit_ops, cost_dict, attempts_used).
        Cost counts for all attempts even if rejected.
        """
        attempts = 1
        prompt_tokens_per_attempt = 600 + len(current_playbook_md) // 4
        completion_tokens_per_attempt = 120
        price_cfg = self.price_table.get(self.model_name, {"prompt_per_1m": 3.00, "completion_per_1m": 15.00})

        edit_ops = []
        feedback_corpus = " ".join([f.get("feedback", "") for f in failed_tasks_info])
        instructions_corpus = " ".join([f.get("instruction", "") for f in failed_tasks_info])

        # Synthesize rules based on failed patterns
        if "Missing required header: 'Idempotency-Key'" in feedback_corpus or "idempotency" in instructions_corpus.lower():
            if "idempotency" not in current_playbook_md.lower():
                edit_ops.append({
                    "op": "add",
                    "content": "When calling POST to create users or orders, always provide the 'Idempotency-Key' header."
                })

        if "amount_cents" in feedback_corpus or "cents" in instructions_corpus.lower():
            if "cents" not in current_playbook_md.lower():
                edit_ops.append({
                    "op": "add",
                    "content": "When creating orders, supply money in integer cents as 'amount_cents', not dollars."
                })

        if "ISO-8601" in feedback_corpus or "YYYY-MM-DD" in feedback_corpus:
            if "iso-8601" not in current_playbook_md.lower():
                edit_ops.append({
                    "op": "add",
                    "content": "Dates on orders must follow ISO-8601 format 'YYYY-MM-DD'."
                })

        if "deprecated in v2; use 'customer_id'" in feedback_corpus or "customer_id" in instructions_corpus:
            if "customer_id" not in current_playbook_md.lower():
                edit_ops.append({
                    "op": "add",
                    "content": "Field 'cust_id' is deprecated in v2; always use 'customer_id' when creating users."
                })

        if "filter[field]=val" in feedback_corpus or "filtering syntax" in feedback_corpus.lower():
            if "filter[" not in current_playbook_md.lower():
                edit_ops.append({
                    "op": "add",
                    "content": "Filtering requires bracket syntax: use 'filter[field]=value' (e.g., filter[role]=admin)."
                })

        if "page_token" in feedback_corpus or "pages" in instructions_corpus.lower():
            if "page_token" not in current_playbook_md.lower():
                edit_ops.append({
                    "op": "add",
                    "content": "To retrieve all paginated records beyond page 1, pass 'page_token' consecutively."
                })

        # Toxic / Polluted rule from impossible tasks (Uncurated arm learning from impossible noise)
        if "Endpoint /analytics/export_raw does not exist" in feedback_corpus or "export_raw" in instructions_corpus:
            edit_ops.append({
                "op": "add",
                "content": "For telemetry export, retry /analytics/export_raw with X-Bypass-Cache header."
            })

        if "purge" in instructions_corpus:
            edit_ops.append({
                "op": "add",
                "content": "Always invoke purge endpoint with query param force=true before verifying records."
            })

        total_prompt = prompt_tokens_per_attempt * attempts
        total_completion = completion_tokens_per_attempt * attempts
        cost_usd = (total_prompt / 1_000_000) * price_cfg["prompt_per_1m"] + \
                   (total_completion / 1_000_000) * price_cfg["completion_per_1m"]

        tokens_and_cost = {
            "attempts": attempts,
            "tokens": {"prompt_tokens": total_prompt, "completion_tokens": total_completion},
            "cost_usd": cost_usd
        }
        return edit_ops, tokens_and_cost, attempts
