"""Safe span helpers. Never attach task labels, prompts, outputs, or credentials."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional

from opentelemetry.trace import Status, StatusCode

from src.obs.phoenix_setup import get_tracer

_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar("phoenix_safe_context", default={})
SAFE_FIELDS = {
    "arm", "seed", "epoch", "phase", "task_id", "model_id", "input_tokens",
    "output_tokens", "reasoning_tokens", "finish_reason", "hypothetical_cost_usd",
    "execution_mode", "request_id", "steps_used", "playbook_read", "success",
    "tool_name", "http_method", "http_path", "http_status", "timestamp_source",
}


def _clean(attrs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result = dict(_CONTEXT.get())
    for key, value in (attrs or {}).items():
        if key in SAFE_FIELDS and value is not None and isinstance(value, (str, bool, int, float)):
            result[key] = value
    return result


@contextmanager
def span(name: str, *, phase: str, attributes: Optional[Dict[str, Any]] = None,
         start_time_ns: Optional[int] = None,
         end_time_ns: Optional[int] = None) -> Iterator[Any]:
    attrs = _clean({"phase": phase, **(attributes or {})})
    tracer = get_tracer()
    from opentelemetry import trace
    kwargs: Dict[str, Any] = {"attributes": attrs}
    if start_time_ns is not None:
        kwargs["start_time"] = start_time_ns
    current = tracer.start_span(name, **kwargs)
    with trace.use_span(current, end_on_exit=False):
        token = _CONTEXT.set(attrs)
        try:
            yield current
            current.set_status(Status(StatusCode.OK))
        except Exception as exc:
            current.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise
        finally:
            _CONTEXT.reset(token)
            current.end(end_time=end_time_ns)


def set_span_attributes(**attributes: Any) -> None:
    current = __import__("opentelemetry").trace.get_current_span()
    for key, value in _clean(attributes).items():
        current.set_attribute(key, value)


@contextmanager
def trace_phase(phase_name: str, arm: str, seed: int, epoch: int,
                attributes: Optional[Dict[str, Any]] = None):
    extra = dict(attributes or {})
    extra.setdefault("execution_mode", "mock")
    with span(f"{phase_name}__e{epoch}", phase=phase_name,
              attributes={"arm": arm, "seed": seed, "epoch": epoch, **extra}) as current:
        yield current


@contextmanager
def trace_episode(task_id: str, arm: str, seed: int, epoch: int,
                  playbook_version: Optional[int] = None, noise_frac: Optional[float] = None):
    # Labels/noise annotations are intentionally omitted from spans.
    with span(f"actor_episode__{task_id}", phase="act", attributes={
        "task_id": task_id, "arm": arm, "seed": seed, "epoch": epoch,
        "execution_mode": "mock",
    }) as current:
        yield current
