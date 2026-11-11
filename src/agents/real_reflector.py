"""
Real Reflector:
Zero access to labels.json.
Generates structured playbook edit ops (add, modify, delete) using real LLM calls.
Enforces max 3 attempts on malformed output or token cap violations.
Counts all rejected attempts' tokens and cost toward learning cost.
"""
import json
from typing import Dict, Any, List, Tuple, Optional
from src.agents.llm_client import LLMClient
from src.analysis.reflection_grounding import make_reflection_trace_records, check_rule_grounding

REFLECTOR_SYSTEM_PROMPT = """You are the Reflector for this specific API. Use only the failed request, its error body, and any successful request supplied for that task.
Every add or modify operation must include exact failed_request_quote and error_body_quote in evidence. Include successful_request_quote only when supplied; quotes belong in evidence and need not be copied into rule content.
Rules must follow: "When <request/endpoint> returns <error text>, then <change field/header X to value Y>". Mention an exact header, parameter, or field in the failed request or error body. If the error explicitly names the needed change (such as a missing required header), that change is grounded by the error. Any value or format not stated by the error must appear in an observed successful request. For other fixes, use a successful request as evidence; do not infer that an "invalid field" should be removed or reformatted. Never contradict a supplied successful request, including forbidding a header or body it used.
Forbid generic REST conventions and any unobserved fields, headers, parameters, operators, pagination schemes, formats, or behaviors. Rule IDs for add operations are assigned by the harness; never supply one. Modify/delete may use only existing rule IDs shown in the current playbook. Keep the playbook under 2,000 tokens.

Return a strict JSON array only: no prose and no markdown fences. Allowed forms:
[{"op":"add","content":"When ... returns ..., then ...","evidence":{"task_id":"...","failed_request_quote":"...","error_body_quote":"...","successful_request_quote":null}}]
[{"op":"modify","rule_id":"existing ID","content":"When ... returns ..., then ...","evidence":{"task_id":"...","failed_request_quote":"...","error_body_quote":"...","successful_request_quote":"..."}}]
[{"op":"delete","rule_id":"existing ID"}]
An empty array is valid when evidence does not support a rule."""

class RealReflector:
    def __init__(self, llm_client: LLMClient, model_name: str, max_attempts: int = 2):
        self.llm_client = llm_client
        self.model_name = model_name
        self.max_attempts = max_attempts

    def reflect(
        self,
        failed_tasks_info: List[Dict[str, Any]],
        current_playbook_md: str,
        anchors: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], int, List[Dict[str, Any]]]:
        """
        Returns: (ops, cost_dict, attempts_used, attempt_logs)
        """
        history_map = {
            failure["task_id"]: [
                {"passed": True, "trajectory": trajectory}
                for trajectory in failure.get("prior_successful_trajectories", [])
            ]
            for failure in failed_tasks_info
        }
        trace_records = make_reflection_trace_records(failed_tasks_info, history_map)

        failure_bundle = [{
            "task_id": f.get("task_id"),
            "instruction": f.get("instruction"),
            "grader_feedback": f.get("feedback"),
            "failed_trajectory": f.get("trajectory", []),
            "prior_successful_trajectories": f.get("prior_successful_trajectories", []),
        } for f in failed_tasks_info]
        user_content = (
            f"Current Playbook:\n{current_playbook_md}\n\n"
            "Trace bundle (use these exact request and response strings as evidence):\n"
            + json.dumps({"traces": failure_bundle, "evidence_records": trace_records}, ensure_ascii=False, indent=2)
            + (f"\nTrusted success references: {json.dumps(anchors or [], ensure_ascii=False)}" if anchors else "")
        )

        messages = [
            {"role": "system", "content": REFLECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        attempt_logs = []
        total_prompt_tokens = 0
        total_comp_tokens = 0
        total_cost_usd = 0.0
        accepted_ops = []
        grounding_rejections = 0
        format_rejections = 0

        for attempt in range(1, self.max_attempts + 1):
            rejection_kind = "format"
            llm_res = self.llm_client.generate(
                model_name=self.model_name,
                messages=messages,
                temperature=0.2 * attempt, # slight temperature increase on retry
                max_tokens=800
            )

            total_prompt_tokens += llm_res["tokens"]["prompt_tokens"]
            total_comp_tokens += llm_res["tokens"]["completion_tokens"]
            total_cost_usd += llm_res["cost_usd"]

            raw_text = (llm_res.get("text") or "").strip()
            clean_json = raw_text
            if clean_json.startswith("```"):
                clean_json = clean_json.split("\n", 1)[1]
                if clean_json.endswith("```"):
                    clean_json = clean_json.rsplit("```", 1)[0]

            try:
                ops = json.loads(clean_json.strip())
                if isinstance(ops, list) and all(isinstance(op, dict) and "op" in op for op in ops):
                    if any(op.get("op") == "add" and "rule_id" in op for op in ops):
                        raise ValueError("The harness assigns rule IDs; add operations must omit rule_id.")
                    grounding_issues = check_rule_grounding(ops, trace_records)
                    if grounding_issues:
                        grounding_rejections += 1
                        rejection_kind = "grounding"
                        raise ValueError("Mechanical grounding check rejected rule(s): "
                                         + json.dumps(grounding_issues, ensure_ascii=False))
                    accepted_ops = ops
                    attempt_logs.append({
                        "attempt": attempt,
                        "status": "accepted",
                        "raw_output": raw_text,
                        "ops": ops
                    })
                    break
                else:
                    raise ValueError("Output must be a JSON list of dicts with 'op' key.")
            except Exception as e:
                if rejection_kind == "format":
                    format_rejections += 1
                attempt_logs.append({
                    "attempt": attempt,
                    "status": "rejected",
                    "rejection_kind": rejection_kind,
                    "reason": str(e),
                    "raw_output": raw_text
                })
                # Append rejection notice to messages for next attempt
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({"role": "user", "content": f"Attempt {attempt} rejected: {str(e)}. Return a strict JSON array only and satisfy the evidence requirements."})

        cost_info = {
            "attempts": len(attempt_logs),
            "tokens": {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_comp_tokens},
            "cost_usd": total_cost_usd,
            "prompt_used": user_content,
            "grounding_rejections": grounding_rejections,
            "format_rejections": format_rejections,
        }

        return accepted_ops, cost_info, len(attempt_logs), attempt_logs
