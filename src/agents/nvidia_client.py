"""
Robust NVIDIA API Client:
- Configured inter-call throttle.
- Exponential backoff on 429: 2s, 4s, 8s, 16s, 32s. Logs every retry.
- Enforces strict $1.00 hard cap: execution stops immediately if hit.
- Logs input, output, reasoning tokens, latency, request_id, model_id.
- Label all costs "hypothetical pricing".
- Never exposes or logs the API key.
"""
import os
import time
import json
import urllib.request
import urllib.error
import threading
from email.utils import parsedate_to_datetime
from typing import Dict, Any, List, Optional


class NVIDIAAPIError(RuntimeError):
    """A provider/network request failed after configured retries."""

class NVIDIAClient:
    # Shared by every client instance in this process so parallel workers do
    # not independently burst the hosted endpoint.
    _limiter_lock = threading.Lock()
    _last_global_start = {}
    _last_model_start = {}
    _global_cooldown_until = {}
    _adaptive_global_interval = {}
    _successes_since_429 = {}

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get("API_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
        self.key_env_var = config.get("API_KEY_ENV_VAR", "NVIDIA_API_KEY")
        self.price_table = config.get("REFERENCE_PRICE_TABLE", {})
        self.throttle_sec = config.get("ACTOR_THROTTLE_SECONDS", 1.7)
        self.model_throttle_sec = config.get("MODEL_THROTTLE_SECONDS", {})
        self.global_min_interval = 60.0 / config.get("RATE_LIMIT_RPM", 35)
        self.max_usd = config.get("MAX_USD", 1.00)
        self.max_retries = config.get("MAX_429_RETRIES", 2)
        self.reflector_read_timeout_sec = config.get("REFLECTOR_READ_TIMEOUT_SECONDS", 60)

        self.last_call_time = 0.0
        self.total_hypothetical_usd = 0.0
        self._budget_lock = threading.Lock()
        self._reserved_hypothetical_usd = 0.0
        self.call_logs: List[Dict[str, Any]] = []
        self.retry_events: List[Dict[str, Any]] = []
        self._thread_local = threading.local()
        self.consecutive_api_failures = 0

    def _record_retry(self, event: Dict[str, Any]) -> None:
        self.retry_events.append(event)
        pending = getattr(self._thread_local, "retry_events", None)
        if pending is not None:
            pending.append(event)

    def _take_thread_retry_events(self) -> List[Dict[str, Any]]:
        pending = getattr(self._thread_local, "retry_events", [])
        self._thread_local.retry_events = []
        return list(pending)

    @staticmethod
    def _retry_after_seconds(headers: Any, fallback: float) -> float:
        """Read Retry-After as seconds or an HTTP date, with a safe fallback."""
        value = headers.get("Retry-After") if headers is not None else None
        if value is None:
            return fallback
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            try:
                return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                return fallback

    def _get_api_key(self) -> str:
        key = os.environ.get(self.key_env_var)
        if not key:
            raise EnvironmentError(f"CRITICAL: Environment variable '{self.key_env_var}' is not set. Real mode cannot run.")
        return key

    def chat_completion(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        self._thread_local.retry_events = []
        from src.obs.spans import span
        max_call_cost = self._estimate_max_call_cost(model_id, messages, max_tokens)
        with self._budget_lock:
            projected = (self.total_hypothetical_usd + self._reserved_hypothetical_usd
                         + max_call_cost)
            if projected > self.max_usd:
                raise RuntimeError(
                    f"HARD CAP PREFLIGHT: concurrent worst-case hypothetical pricing (${projected:.6f}) "
                    f"could exceed MAX_USD (${self.max_usd:.2f}); request not sent.")
            self._reserved_hypothetical_usd += max_call_cost
        try:
            with span("llm_call", phase="llm_call", attributes={
                "model_id": model_id, "execution_mode": "real"
            }) as llm_span:
                result = self._chat_completion(model_id, messages, temperature, max_tokens)
                record = result.get("call_record", {})
                for key in ("input_tokens", "output_tokens", "reasoning_tokens", "finish_reason",
                            "hypothetical_cost_usd", "request_id", "execution_mode"):
                    if record.get(key) is not None:
                        llm_span.set_attribute(key, record[key])
                return result
        finally:
            with self._budget_lock:
                self._reserved_hypothetical_usd -= max_call_cost

    def _estimate_max_call_cost(self, model_id: str, messages: List[Dict[str, str]],
                                max_tokens: int) -> float:
        if model_id not in self.price_table:
            raise ValueError(f"No hypothetical pricing configured for model '{model_id}'.")
        price = self.price_table[model_id]
        prompt_bytes = len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
        completion_rate = max(price["output_per_1m"],
                              price.get("thinking_per_1m", price["output_per_1m"]))
        return ((prompt_bytes / 1e6) * price["input_per_1m"]
                + (max_tokens / 1e6) * completion_rate)

    def _chat_completion(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        # Enforce budget cap strictly
        if self.total_hypothetical_usd >= self.max_usd:
            raise RuntimeError(f"HARD CAP REACHED: Cumulative hypothetical spend (${self.total_hypothetical_usd:.4f}) hit MAX_USD (${self.max_usd:.2f})! Execution halted.")

        api_key = self._get_api_key()
        url = f"{self.base_url}/chat/completions"

        # Conservative upper bound: UTF-8 bytes upper-bound prompt tokens, and
        # max_tokens upper-bounds completion tokens. Refuse the request before
        # transmission if its possible charge could cross the hard cap.
        if model_id not in self.price_table:
            raise ValueError(f"No hypothetical pricing configured for model '{model_id}'.")
        p_cfg = self.price_table[model_id]
        prompt_bytes = len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
        completion_rate = max(p_cfg["output_per_1m"], p_cfg.get("thinking_per_1m", p_cfg["output_per_1m"]))
        max_call_cost = (prompt_bytes / 1e6) * p_cfg["input_per_1m"] + (max_tokens / 1e6) * completion_rate
        if self.total_hypothetical_usd + max_call_cost > self.max_usd:
            raise RuntimeError(
                f"HARD CAP PREFLIGHT: worst-case hypothetical pricing (${self.total_hypothetical_usd + max_call_cost:.6f}) "
                f"could exceed MAX_USD (${self.max_usd:.2f}); request not sent."
            )

        payload_bytes = json.dumps({
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }).encode("utf-8")

        backoff_delays = [2.0, 4.0, 8.0, 16.0, 32.0]
        data = None
        duration = 0.0
        limiter_scope = self.base_url
        model_scope = (self.base_url, model_id)
        model_interval = max(self.throttle_sec, self.model_throttle_sec.get(model_id, 0.0))

        for attempt in range(self.max_retries + 1):
            # Global start-rate and per-model spacing are enforced under one
            # process-wide lock, including retries and separate client objects.
            with self._limiter_lock:
                now = time.monotonic()
                effective_interval = max(
                    self.global_min_interval,
                    self._adaptive_global_interval.get(limiter_scope, self.global_min_interval))
                global_wait = effective_interval - (now - self._last_global_start.get(limiter_scope, 0.0))
                model_wait = model_interval - (now - self._last_model_start.get(model_scope, 0.0))
                cooldown_wait = self._global_cooldown_until.get(limiter_scope, 0.0) - now
                wait_for = max(0.0, global_wait, model_wait, cooldown_wait)
                if wait_for:
                    time.sleep(wait_for)
                start_time = time.monotonic()
                self._last_global_start[limiter_scope] = start_time
                self._last_model_start[model_scope] = start_time
                self.last_call_time = time.time()
            req = urllib.request.Request(
                url,
                data=payload_bytes,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
            )
            t0 = time.time()
            try:
                request_timeout = (self.reflector_read_timeout_sec
                                   if getattr(self, "current_phase", None) == "reflect" else 120)
                with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                    duration = time.time() - t0
                    data = json.loads(resp.read().decode("utf-8"))
                    with self._limiter_lock:
                        self._successes_since_429[limiter_scope] = (
                            self._successes_since_429.get(limiter_scope, 0) + 1)
                        if (self._successes_since_429[limiter_scope] >= 20
                                and limiter_scope in self._adaptive_global_interval):
                            current = self._adaptive_global_interval[limiter_scope]
                            reduced = max(self.global_min_interval, current / 2.0)
                            if reduced <= self.global_min_interval:
                                self._adaptive_global_interval.pop(limiter_scope, None)
                            else:
                                self._adaptive_global_interval[limiter_scope] = reduced
                            self._successes_since_429[limiter_scope] = 0
                    break
            except urllib.error.HTTPError as e:
                err_text = e.read().decode("utf-8", errors="replace")
                if e.code == 429:
                    # Provider 429s persisted through short 2/4/8s retries in
                    # concurrent evals. Give the shared key a longer recovery
                    # window, unless the server supplies Retry-After.
                    fallback_delays = (15.0, 30.0, 60.0)
                    fallback = fallback_delays[min(attempt, len(fallback_delays) - 1)]
                    sleep_time = self._retry_after_seconds(e.headers, fallback)
                    # A 429 applies to the shared key/endpoint, not only this
                    # worker. Pause every client in this process before any
                    # worker (including an eval worker) starts another call.
                    with self._limiter_lock:
                        retry_after = time.monotonic() + sleep_time
                        self._global_cooldown_until[limiter_scope] = max(
                            self._global_cooldown_until.get(limiter_scope, 0.0), retry_after)
                        current = max(self.global_min_interval,
                                      self._adaptive_global_interval.get(limiter_scope, self.global_min_interval))
                        self._adaptive_global_interval[limiter_scope] = min(
                            30.0, max(current + 1.0, current * 2.0))
                        self._successes_since_429[limiter_scope] = 0
                        effective_interval = self._adaptive_global_interval[limiter_scope]
                    self._record_retry({"type": "429", "model_id": model_id,
                                        "retry": attempt + 1, "backoff_sec": sleep_time,
                                        "server_retry_after": (e.headers.get("Retry-After")
                                                               if e.headers is not None else None),
                                        "adaptive_global_interval_sec": effective_interval,
                                        "exhausted": attempt >= self.max_retries})
                    if attempt >= self.max_retries:
                        self.consecutive_api_failures += 1
                        raise NVIDIAAPIError(f"HTTP Error 429 for model '{model_id}': {err_text}") from e
                    print(f"[RETRY LOG] HTTP 429 received for {model_id}. Shared endpoint cooldown {sleep_time}s (Retry {attempt+1}/{self.max_retries})...")
                elif 500 <= e.code <= 599:
                    sleep_time = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                    self._record_retry({"type": f"http_{e.code}", "model_id": model_id,
                                        "retry": attempt + 1, "backoff_sec": sleep_time,
                                        "exhausted": attempt >= self.max_retries})
                    if attempt >= self.max_retries:
                        self.consecutive_api_failures += 1
                        raise NVIDIAAPIError(f"HTTP Error {e.code} for model '{model_id}': {err_text}") from e
                    print(f"[RETRY LOG] HTTP {e.code} received for {model_id}. Backing off {sleep_time}s (Retry {attempt+1}/{self.max_retries})...")
                    time.sleep(sleep_time)
                else:
                    self.consecutive_api_failures += 1
                    raise NVIDIAAPIError(f"HTTP Error {e.code} for model '{model_id}': {err_text}") from e
            except Exception as e:
                sleep_time = backoff_delays[min(attempt, len(backoff_delays)-1)]
                self._record_retry({"type": "transient", "model_id": model_id,
                                    "retry": attempt + 1, "backoff_sec": sleep_time,
                                    "exhausted": attempt >= self.max_retries,
                                    "error": str(e)})
                if attempt >= self.max_retries:
                    self.consecutive_api_failures += 1
                    raise NVIDIAAPIError(str(e)) from e
                print(f"[RETRY LOG] Transient error {e}. Backing off {sleep_time}s (Retry {attempt+1}/{self.max_retries})...")
                time.sleep(sleep_time)

        if not data:
            self.consecutive_api_failures += 1
            raise NVIDIAAPIError(f"Exhausted {self.max_retries} retries for model {model_id}.")

        self.consecutive_api_failures = 0

        if "usage" not in data or data["usage"] is None:
            self.consecutive_api_failures += 1
            raise NVIDIAAPIError(f"CRITICAL: Model '{model_id}' returned no usage field! Response: {data}")

        usage = data["usage"]
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        # Reasoning tokens (if reported)
        reasoning_tokens = 0
        details = usage.get("completion_tokens_details")
        if details and isinstance(details, dict):
            reasoning_tokens = details.get("reasoning_tokens", 0)

        standard_output_tokens = max(0, output_tokens - reasoning_tokens)

        # Reference pricing
        p_cfg = self.price_table[model_id]
        cost = (input_tokens / 1e6) * p_cfg["input_per_1m"] + \
               (standard_output_tokens / 1e6) * p_cfg["output_per_1m"] + \
               (reasoning_tokens / 1e6) * p_cfg.get("thinking_per_1m", p_cfg["output_per_1m"])

        with self._budget_lock:
            self.total_hypothetical_usd += cost

        choice = data.get("choices", [{}])[0]
        text_content = choice.get("message", {}).get("content", "")
        finish_reason = choice.get("finish_reason")
        req_id = data.get("id", "req_unknown")

        call_record = {
            "provider": "openai_compatible",
            "model_id": model_id,
            "request_id": req_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "finish_reason": finish_reason,
            "usage_fields": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "completion_tokens_details": usage.get("completion_tokens_details"),
            },
            "reasoning_token_field": "usage.completion_tokens_details.reasoning_tokens",
            "latency_sec": round(duration, 3),
            "execution_mode": "real",
            "cost_label": "hypothetical pricing",
            "hypothetical_cost_usd": round(cost, 6),
            "prompt_used": json.dumps(messages),
            "raw_output": text_content
        }
        self.call_logs.append(call_record)

        return {
            "text": text_content,
            "call_record": call_record
        }
