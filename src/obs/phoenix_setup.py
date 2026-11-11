"""Phoenix configuration. Tracing is opt-in and exports only to local Phoenix."""
from __future__ import annotations

import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.trace import Tracer

PROJECT_NAME = "curation-plasticity-real"
LOCAL_ENDPOINT = "http://localhost:6006"
_TRACER: Optional[Tracer] = None


def init_phoenix_tracer(project_name: str = PROJECT_NAME, *, force: bool = False) -> Tracer:
    """Register OTLP/HTTP trace export to a local Phoenix collector.

    Set PHOENIX_TRACE=1 (or pass force=True) to enable export. No Phoenix API
    key is read or supplied; this setup is deliberately local-only.
    """
    global _TRACER
    if _TRACER is not None:
        return _TRACER
    _TRACER = trace.get_tracer("curation-plasticity", "0.1.0")
    if not force and os.getenv("PHOENIX_TRACE", "0").lower() not in {"1", "true", "yes"}:
        return _TRACER

    from phoenix.otel import register

    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", f"{LOCAL_ENDPOINT}/v1/traces")
    if not endpoint.startswith(("http://localhost", "http://127.0.0.1")):
        raise ValueError("Phoenix export is restricted to the local Phoenix endpoint.")
    provider = register(
        project_name=project_name,
        endpoint=endpoint,
        protocol="http/protobuf",
        batch=True,
        auto_instrument=False,
        verbose=False,
    )
    _TRACER = provider.get_tracer("curation-plasticity", "0.1.0")
    return _TRACER


def get_tracer() -> Tracer:
    return init_phoenix_tracer()


def flush_phoenix() -> None:
    provider = trace.get_tracer_provider()
    force_flush = getattr(provider, "force_flush", None)
    if callable(force_flush):
        force_flush(timeout_millis=10000)
