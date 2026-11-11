"""Trace-grounding checks for Reflector rules."""
import json
import re
from typing import Any, Dict, List


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def request_from_step(step: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "method": step.get("method"),
        "path": step.get("path"),
        "params": step.get("params") or {},
        "headers": step.get("headers") or {},
        "body": step.get("body") or {},
    }


def make_reflection_trace_records(
    failed_tasks: List[Dict[str, Any]],
    task_histories: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Build exact failed request/error pairs and successes observed afterward."""
    records = []
    for failure in failed_tasks:
        task_id = failure["task_id"]
        failed_requests = []
        successful_request_quotes = []
        failure_seen = False
        for step in failure.get("trajectory", []):
            if step.get("action") != "http_request":
                continue
            response = step.get("result") or {}
            if response.get("status", 200) < 400:
                if failure_seen:
                    quote = compact_json(request_from_step(step))
                    if quote not in successful_request_quotes:
                        successful_request_quotes.append(quote)
                continue
            failure_seen = True
            failed_requests.append({
                "request_quote": compact_json(request_from_step(step)),
                "error_body_quote": compact_json(response.get("body", {})),
            })

        for history in task_histories.get(task_id, []):
            if not history.get("passed"):
                continue
            for step in history.get("trajectory", []):
                if step.get("action") == "http_request" and (step.get("result") or {}).get("status", 200) < 400:
                    quote = compact_json(request_from_step(step))
                    if quote not in successful_request_quotes:
                        successful_request_quotes.append(quote)

        records.append({
            "task_id": task_id,
            "failed_requests": failed_requests,
            "successful_request_quotes": successful_request_quotes,
        })
    return records


def check_rule_grounding(ops: List[Dict[str, Any]], trace_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate that rules are supported by the supplied request/error traces.

    Evidence belongs in the evidence object; actionable wording must refer to
    fields named in those traces. An error that explicitly says a header is
    required can support adding that header. Other fixes, and values/formats
    the error does not state, need a successful request trace.
    """
    records_by_task = {record["task_id"]: record for record in trace_records}
    rejections = []
    for index, op in enumerate(ops):
        if op.get("op") not in {"add", "modify"}:
            continue
        evidence = op.get("evidence")
        content = op.get("content")
        reasons = []
        if not isinstance(evidence, dict):
            rejections.append({"op_index": index, "reason": "missing evidence object"})
            continue
        task_id = evidence.get("task_id")
        record = records_by_task.get(task_id)
        if record is None:
            reasons.append("evidence task_id is not among supplied traces")
        else:
            request_quote = evidence.get("failed_request_quote")
            error_quote = evidence.get("error_body_quote")
            matching_pair = any(
                item.get("request_quote") == request_quote
                and item.get("error_body_quote") == error_quote
                for item in record["failed_requests"]
            )
            if not matching_pair:
                reasons.append("failed request and error body do not exactly match a supplied trace pair")
            successes = record["successful_request_quotes"]
            success_quote = evidence.get("successful_request_quote")
            if not successes and success_quote is not None:
                reasons.append("successful request quote supplied when none is available")
            if not isinstance(content, str):
                content = ""
            supplied_traces = [item.get("request_quote") for item in record.get("failed_requests", [])]
            supplied_traces += [item.get("error_body_quote") for item in record.get("failed_requests", [])]
            supplied_traces += list(record.get("successful_request_quotes", []))
            if any(q is not None and q not in supplied_traces
                   for q in (request_quote, error_quote, success_quote)):
                reasons.append("evidence quote does not appear verbatim in supplied traces")
            if record and matching_pair:
                request_obj = json.loads(request_quote)
                error_obj = json.loads(error_quote)
                success_obj = json.loads(success_quote) if success_quote in successes else None
                names = set()
                observed_values = set()
                def visit(value, collect_values=False):
                    if isinstance(value, dict):
                        for key, val in value.items():
                            names.add(str(key))
                            visit(val, collect_values)
                    elif isinstance(value, list):
                        for val in value: visit(val, collect_values)
                    elif value is not None:
                        if collect_values:
                            observed_values.add(str(value))
                        if isinstance(value, str):
                            try:
                                visit(json.loads(value), collect_values)
                            except (ValueError, TypeError):
                                pass

                for request in [request_obj]:
                    for section in ("headers", "params", "body"):
                        value = request.get(section) or {}
                        if value: names.add(section)
                        visit(value)
                error_text_parts = []
                def collect_error_text(value):
                    if isinstance(value, dict):
                        for val in value.values(): collect_error_text(val)
                    elif isinstance(value, list):
                        for val in value: collect_error_text(val)
                    elif isinstance(value, str):
                        error_text_parts.append(value)
                collect_error_text(error_obj)
                error_text = " ".join(error_text_parts)
                names.update(re.findall(r"['\"]([A-Za-z_][\w-]*)['\"]", error_text))
                for name in ("X-Api-Key", "Authorization", "Content-Type"):
                    if name.lower() in error_text.lower(): names.add(name)
                if not any(name and name.casefold() in content.casefold() for name in names):
                    reasons.append("rule does not mention an exact header, parameter, or field name from the quoted traces")
                if successes:
                    success_objects = [json.loads(q) for q in successes]
                    for success in success_objects:
                        for section in ("headers", "params", "body"):
                            visit(success.get(section) or {}, collect_values=True)

                # A missing-required-header error directly supports adding the
                # named header. Other suggested fixes need a successful trace.
                direct_missing_header = False
                missing_header = re.search(
                    r"missing required (?:authentication )?header ['\"]?([A-Za-z_][\w-]+)",
                    error_text, re.I)
                if missing_header:
                    header_name = missing_header.group(1)
                    direct_missing_header = (
                        header_name.casefold() in content.casefold()
                        and re.search(r"\b(?:add|include|send|set)\b", content, re.I) is not None)

                assignment_patterns = (
                    r"\b(?:change|set)\s+(?:the\s+)?(?:(?:field|header|parameter)\s+)?[A-Za-z_][\w-]*\s+(?:to(?:\s+value)?|=)\s*['\"]?([A-Za-z0-9_/@+-]+(?:\.[A-Za-z0-9_/@+-]+)*)",
                    r"\b(?:add|include|set)\s+(?:the\s+)?(?:field|header|parameter)\s+[A-Za-z_][\w-]*\s+(?:with\s+value|to)\s+['\"]?([A-Za-z0-9_/@+-]+(?:\.[A-Za-z0-9_/@+-]+)*)",
                    r"\b(?:add|include)\s+[A-Za-z_][\w-]*\s+(?:with\s+value|to)\s+['\"]?([A-Za-z0-9_/@+-]+(?:\.[A-Za-z0-9_/@+-]+)*)",
                )
                proposed_values = [m.group(1) for p in assignment_patterns
                                   for m in re.finditer(p, content, re.I)]
                if re.search(r"\bwith (?:a |an )?(?:valid|well-formed|proper) [\w -]+", content, re.I):
                    proposed_values.append("<unstated format/value>")
                for value in proposed_values:
                    # The missing-header error authorizes the header itself;
                    # its placeholder value is intentionally not grounded by
                    # this rule and does not need a successful trace.
                    if direct_missing_header:
                        continue
                    if value == "<unstated format/value>":
                        value_is_named = False
                        value_is_successful = any(
                            any(header.casefold() in content.casefold() and bool(v)
                                for header, v in (json.loads(q).get("headers") or {}).items())
                            for q in successes)
                    else:
                        value_is_named = value.casefold() in error_text.casefold()
                        value_is_successful = value in observed_values
                    if not value_is_named and not value_is_successful:
                        reasons.append(f"proposed value or format {value!r} is absent from a successful request and not named by the error")

                proposes_fix = re.search(
                    r"\b(?:then|include|omit|do not|don't|set|change|use|add|remove|send)\b",
                    content, re.I) is not None
                if proposes_fix and not successes and not direct_missing_header:
                    reasons.append("error names the field but does not specify this fix; no successful request supports the proposal")
                for success_quote_item in successes:
                    observed_success = json.loads(success_quote_item)
                    # Detect explicit prohibitions of anything present in the
                    # successful request (header/field/parameter names or values).
                    lowered = content.lower()
                    prohibitions = ("do not include", "don't include", "omit", "forbid", "must not", "never include")
                    for key in observed_success.get("headers", {}):
                        if key.lower() in lowered and any(p in lowered for p in prohibitions):
                            reasons.append(f"rule contradicts successful trace by forbidding header {key}")
                    for section in ("body", "params"):
                        for key in (observed_success.get(section) or {}):
                            if key.lower() in lowered and any(p in lowered for p in prohibitions):
                                reasons.append(f"rule contradicts successful trace by forbidding {section} field {key}")
        if not isinstance(content, str) or not content.strip():
            reasons.append("rule content is empty")
        if reasons:
            rejections.append({"op_index": index, "task_id": task_id, "reason": "; ".join(reasons)})
    return rejections
