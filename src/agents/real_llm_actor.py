"""
LLM Actor Agent using Real API Calls (Tool-use loop):
- Tools:
  1. read_playbook()
  2. http_request(method, path, params, headers, body)
  3. finish(answer)
- Interacts with in-process MockAPI using JSON tool calls.
- Enforces a configurable 12-step cap per episode; every model action, including read_playbook, consumes one step.
- Tracks tool-call format error rate and token usage per episode.
"""
from typing import Dict, Any, List, Optional
import json
import threading
from src.testbed.mock_api import MockAPI
from src.loop.provenance import Playbook
from src.agents.nvidia_client import NVIDIAClient

ACTOR_SYSTEM_PROMPT = """You are an automated API client agent solving a task against an enterprise REST API.
Available endpoints and methods:
- GET /users and GET /users/{user_id}
- POST /users
- GET /orders and GET /orders/{order_id}
- POST /orders
- GET /invoices and GET /invoices/{invoice_id}

You have access to the following tools:
1. `read_playbook()`: Reads the team's shared operational playbook (conventions, headers, quirks).
2. `http_request(method, path, params, headers, body)`: Sends an HTTP request to the API.
3. `finish(answer)`: Submits the final answer to conclude the task.

At each step, output ONLY a JSON object indicating your action:
{"action": "read_playbook"}
OR
{"action": "http_request", "method": "GET|POST", "path": "/...", "params": {...}, "headers": {...}, "body": {...}}
OR
{"action": "finish", "answer": "<your final result>"}

Do not output explanations outside the JSON object. Output strictly valid JSON."""

class RealLLMActor:
    def __init__(self, client: NVIDIAClient, model_id: str, max_steps: int = 12):
        self.client = client
        self.model_id = model_id
        self.max_steps = max_steps
        self.format_errors = 0
        self.total_episodes = 0
        self._metrics_lock = threading.Lock()

    def run_episode(
        self,
        task: Dict[str, Any],
        playbook: Playbook,
        mock_api: MockAPI
    ) -> Dict[str, Any]:
        from src.obs.spans import span
        trace_context = task.get("_trace_context", {})
        with span("actor_episode", phase="act", attributes={
            "task_id": task.get("task_id"), "model_id": self.model_id,
            "execution_mode": "real", **trace_context
        }) as episode_span:
            result = self._run_episode(task, playbook, mock_api)
            episode_span.set_attribute("input_tokens", result["tokens"]["input_tokens"])
            episode_span.set_attribute("output_tokens", result["tokens"]["output_tokens"])
            episode_span.set_attribute("reasoning_tokens", result["tokens"].get("reasoning_tokens", 0))
            episode_span.set_attribute("hypothetical_cost_usd", result["hypothetical_cost_usd"])
            if result.get("finish_reason") is not None:
                episode_span.set_attribute("finish_reason", result["finish_reason"])
            episode_span.set_attribute("steps_used", result["steps_used"])
            episode_span.set_attribute("playbook_read", result["read_playbook_called"])
            return result

    def _run_episode(
        self,
        task: Dict[str, Any],
        playbook: Playbook,
        mock_api: MockAPI
    ) -> Dict[str, Any]:
        with self._metrics_lock:
            self.total_episodes += 1
        task_id = task["task_id"]
        instruction = task["instruction"]

        messages = [
            {"role": "system", "content": ACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {instruction}\nSubmit your first tool action as JSON."}
        ]

        trajectory = []
        read_playbook_called = False
        final_answer = None
        finished = False
        episode_format_errors = 0
        episode_truncations = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_reasoning_tokens = 0
        total_cost = 0.0
        last_finish_reason = None

        for step in range(1, self.max_steps + 1):
            attempts = []
            action_data = None
            for retry_index in range(2):
                res = self.client.chat_completion(
                    model_id=self.model_id,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=1000
                )
                total_input_tokens += res["call_record"]["input_tokens"]
                total_output_tokens += res["call_record"]["output_tokens"]
                total_reasoning_tokens += res["call_record"].get("reasoning_tokens", 0)
                total_cost += res["call_record"]["hypothetical_cost_usd"]

                # Some hosted endpoints return finish_reason=length with a
                # missing/null message content. Treat that like an empty
                # truncated response so the normal one-retry path can run.
                raw_text = (res.get("text") or "").strip()
                finish_reason = res["call_record"].get("finish_reason")
                last_finish_reason = finish_reason
                attempts.append({"raw_output": raw_text, "finish_reason": finish_reason})
                if finish_reason == "length":
                    episode_truncations += 1
                    retry_error = "Truncated response (finish_reason=length)."
                else:
                    clean_json = raw_text
                    if clean_json.startswith("```"):
                        clean_json = clean_json.split("\n", 1)[-1]
                        if clean_json.endswith("```"):
                            clean_json = clean_json.rsplit("```", 1)[0]
                    try:
                        action_data = json.loads(clean_json.strip())
                        if action_data.get("action") not in {"read_playbook", "http_request", "finish"}:
                            raise ValueError("Unknown action; expected read_playbook, http_request, or finish.")
                        break
                    except Exception as exc:
                        action_data = None
                        retry_error = f"Invalid JSON/tool action: {exc}"

                if retry_index == 0:
                    messages.append({"role": "assistant", "content": raw_text})
                    messages.append({"role": "user", "content": f"{retry_error} Return one complete valid JSON tool action only."})

            if action_data is None:
                episode_format_errors += 1
                with self._metrics_lock:
                    self.format_errors += 1
                trajectory.append({"step": step, "method": None, "path": None,
                                   "params": None, "headers": None, "body": None,
                                   "attempts": attempts, "error": "format_error_after_retry"})
                continue

            action = action_data.get("action")
            if action == "read_playbook":
                read_playbook_called = True
                pb_content = playbook.to_markdown()
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({"role": "user", "content": f"Playbook Content:\n{pb_content}\nNext tool action:"})
                trajectory.append({"step": step, "action": "read_playbook",
                                   "method": None, "path": None, "params": None,
                                   "headers": None, "body": None, "result": pb_content})

            elif action == "http_request":
                method = action_data.get("method", "GET")
                path = action_data.get("path", "")
                params = action_data.get("params") or {}
                headers = action_data.get("headers") or {}
                body = action_data.get("body") or {}

                full_request = {
                    "method": method, "path": path, "params": params,
                    "headers": headers, "body": body,
                }
                from src.obs.spans import span
                with span("tool_call", phase="tool_call", attributes={
                    "task_id": task_id, "http_method": str(method), "http_path": str(path),
                    "tool_name": "http_request", "execution_mode": "real"
                }) as tool_span:
                    api_res = mock_api.request(method=method, path=path, params=params, headers=headers, body=body)
                    if isinstance(api_res, dict):
                        tool_span.set_attribute("http_status", int(api_res.get("status", 200)))
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({"role": "user", "content": f"HTTP Response: {json.dumps(api_res)}\nNext tool action:"})
                trajectory.append({"step": step, "action": "http_request", "call": path,
                                   **full_request, "result": api_res})

            elif action == "finish":
                final_answer = action_data.get("answer")
                trajectory.append({"step": step, "action": "finish", "method": None,
                                   "path": None, "params": None, "headers": None,
                                   "body": None, "final_answer": final_answer})
                finished = True
                break
            else:
                episode_format_errors += 1
                with self._metrics_lock:
                    self.format_errors += 1
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({"role": "user", "content": "Unknown action. Use 'read_playbook', 'http_request', or 'finish'."})
                trajectory.append({"step": step, "method": None, "path": None,
                                   "params": None, "headers": None, "body": None,
                                   "raw_output": raw_text, "error": "unknown_action"})

        ended_by_step_cap = not finished and len(trajectory) >= self.max_steps
        if ended_by_step_cap:
            print(f"[EPISODE STEP CAP] task_id={task_id} max_steps={self.max_steps} steps_used={len(trajectory)}")

        return {
            "task_id": task_id,
            "read_playbook_called": read_playbook_called,
            "steps_used": len(trajectory),
            "ended_by_step_cap": ended_by_step_cap,
            "final_answer": final_answer,
            "trajectory": trajectory,
            "format_errors": episode_format_errors,
            "truncations": episode_truncations,
            "finish_reason": last_finish_reason,
            "tokens": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "reasoning_tokens": total_reasoning_tokens,
                "total_tokens": total_input_tokens + total_output_tokens
            },
            "hypothetical_cost_usd": total_cost
        }
