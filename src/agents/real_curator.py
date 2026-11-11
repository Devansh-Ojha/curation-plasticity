"""
Curator System Prompt and Real LLM Invocation.
Zero access to labels.json.
Uses exact label taxonomy from GEPA-Curator paper.
"""
import json
from typing import Dict, Any, List, Optional
from src.agents.llm_client import LLMClient

CURATOR_SYSTEM_PROMPT = """You are an expert diagnostic Curator for a self-improving AI agent.
Your job is to inspect an individual task's recent execution history and decide whether to RUN or SKIP this task in the upcoming optimization cycle, and optionally select at most ONE trusted past success as a contrastive anchor for the Reflector.

Labels (assign exactly one):
- Mastered: Multiple stable clean successes across separate rollouts, no recent regression. One success is not enough.
- Prompt-sensitive: Before/after pairs show prompt/playbook changes improve behavior (even partially). Run these.
- Operationally impossible: Available tools genuinely lack the needed capability, supported by repeated independent attempts. One unsupported claim is insufficient.
- No current optimization signal: The same failure persists across multiple relevant mutations with no progress.
- Flaky or uninformative: Materially similar behavior and end states repeatedly get inconsistent outcomes. Do not infer from one disagreement.
- Evaluator-broken: Explicit evidence of a wrong label or contradictory expectation.
- Infrastructure failure: Timeouts/errors; no evidence about the playbook.
- Uncertain: Run rather than skip.

Decision Rules:
1. Skip ONLY on strong evidence (e.g. Operationally impossible, Flaky, Mastered, No current optimization signal).
2. A SKIP decision must NEVER carry an anchor (anchor_success_id must be null).
3. An anchor requires a genuinely correct, reproducible success corroborated by at least one other independent clean success. If none qualifies, abstain with anchor_success_id: null.
4. Output MUST be valid JSON with reason before decision.

Output JSON format:
{
  "reason": "Detailed analytical justification based purely on cited execution logs",
  "label": "One of the 8 labels",
  "decision": "run" or "skip",
  "skip_epochs": 1,
  "anchor_reason": "Reason for selecting or abstaining from anchor",
  "anchor_success_id": "success_task_id" or null,
  "supporting_success_ids": []
}
"""

class RealCurator:
    def __init__(self, llm_client: LLMClient, model_name: str, history_window: int = 2):
        self.llm_client = llm_client
        self.model_name = model_name
        self.history_window = history_window

    def decide(
        self,
        task: Dict[str, Any],
        task_history: List[Dict[str, Any]],
        available_success_ids: List[str]
    ) -> Dict[str, Any]:
        task_id = task.get("task_id")
        runs_count = len(task_history)

        # Underexplored-first: < 2 runs selected without Curator LLM call
        if runs_count < 2:
            return {
                "task_id": task_id,
                "called_llm": False,
                "reason": "Underexplored task (< 2 runs). Exploring to build history.",
                "label": "Uncertain",
                "decision": "run",
                "skip_epochs": 1,
                "anchor_reason": "Underexplored; no anchor assigned.",
                "anchor_success_id": None,
                "supporting_success_ids": [],
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0},
                "cost_usd": 0.0,
                "prompt_used": "",
                "raw_response": ""
            }

        # Format history: latest H=2 full runs, older compacted
        history_str = ""
        for i, h in enumerate(task_history):
            if i >= len(task_history) - self.history_window:
                history_str += f"- Run (Epoch {h.get('epoch')}): Outcome: {'Passed' if h.get('passed') else 'Failed'}\n"
                history_str += f"  Feedback: {h.get('feedback')}\n"
                history_str += f"  Playbook Used: {h.get('playbook_version')}\n"
                history_str += f"  Tool Actions: {json.dumps(h.get('actions', []))}\n"
            else:
                history_str += f"- Run (Epoch {h.get('epoch')}): Outcome: {'Passed' if h.get('passed') else 'Failed'} (compacted)\n"

        user_content = f"""Task ID: {task_id}
Instruction: {task.get('instruction')}
Available Corroborated Success IDs: {json.dumps(available_success_ids)}

Execution History:
{history_str}

Analyze this task's history and output your diagnostic JSON:"""

        messages = [
            {"role": "system", "content": CURATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        llm_res = self.llm_client.generate(
            model_name=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=600
        )

        raw_text = llm_res["text"].strip()
        # Parse JSON output
        try:
            # Strip markdown fences if present
            clean_json = raw_text
            if clean_json.startswith("```"):
                clean_json = clean_json.split("\n", 1)[1]
                if clean_json.endswith("```"):
                    clean_json = clean_json.rsplit("```", 1)[0]
            parsed = json.loads(clean_json.strip())
        except Exception as e:
            # Fallback on malformed: default to run, never silently skip
            parsed = {
                "reason": f"Malformed LLM output ({str(e)}). Defaulting to run.",
                "label": "Uncertain",
                "decision": "run",
                "skip_epochs": 1,
                "anchor_reason": "Malformed output; abstained.",
                "anchor_success_id": None,
                "supporting_success_ids": []
            }

        # Guard: a skip must never carry an anchor
        if parsed.get("decision") == "skip":
            parsed["anchor_success_id"] = None

        return {
            "task_id": task_id,
            "called_llm": True,
            **parsed,
            "tokens": llm_res["tokens"],
            "cost_usd": llm_res["cost_usd"],
            "prompt_used": user_content,
            "raw_response": raw_text
        }
