from src.analysis.reflection_grounding import check_rule_grounding, make_reflection_trace_records


def trace_record(successes=None):
    return {
        "task_id": "t_123",
        "failed_requests": [{
            "request_quote": '{"body":{"amount":50.0},"headers":{},"method":"POST","params":{},"path":"/orders"}',
            "error_body_quote": '{"error":"Bad request: invalid field \'amount\'."}',
        }],
        "successful_request_quotes": successes or [],
    }


def test_grounding_accepts_rule_quoting_exact_request_and_error():
    request = trace_record()["failed_requests"][0]["request_quote"]
    error = trace_record()["failed_requests"][0]["error_body_quote"]
    ops = [{
        "op": "add",
        "content": f'Observed request "{request}" returned "{error}".',
        "evidence": {
            "task_id": "t_123", "failed_request_quote": request,
            "error_body_quote": error, "successful_request_quote": None,
        },
    }]
    assert check_rule_grounding(ops, [trace_record()]) == []


def test_grounding_rejects_generic_rule_even_with_detached_citation():
    request = trace_record()["failed_requests"][0]["request_quote"]
    error = trace_record()["failed_requests"][0]["error_body_quote"]
    ops = [{
        "op": "add", "content": "Use standard pagination with limit and offset.",
        "evidence": {
            "task_id": "t_123", "failed_request_quote": request,
            "error_body_quote": error, "successful_request_quote": None,
        },
    }]
    issues = check_rule_grounding(ops, [trace_record()])
    assert issues and "exact header, parameter, or field name" in issues[0]["reason"]


def test_grounding_rejects_guessed_fix_without_successful_request():
    request = trace_record()["failed_requests"][0]["request_quote"]
    error = trace_record()["failed_requests"][0]["error_body_quote"]
    success = '{"body":{"amount_cents":5000},"headers":{},"method":"POST","params":{},"path":"/orders"}'
    record = trace_record()
    ops = [{
        "op": "add", "content": "When POST /orders returns invalid amount, then change amount to value 5000.",
        "evidence": {
            "task_id": "t_123", "failed_request_quote": request,
            "error_body_quote": error, "successful_request_quote": success,
        },
    }]
    issues = check_rule_grounding(ops, [record])
    assert any("when none is available" in item["reason"] for item in issues)


def test_grounding_accepts_trace_quotes_outside_rule_text():
    request = trace_record()["failed_requests"][0]["request_quote"]
    error = trace_record()["failed_requests"][0]["error_body_quote"]
    ops = [{"op": "add", "content": "When POST /orders returns invalid amount, record amount.",
            "evidence": {"task_id": "t_123", "failed_request_quote": request,
                         "error_body_quote": error, "successful_request_quote": None}}]
    assert check_rule_grounding(ops, [trace_record()]) == []


def test_error_named_missing_header_can_ground_header_without_success():
    request = '{"body":{},"headers":{},"method":"GET","params":{},"path":"/users"}'
    error = '{"error":"Missing required authentication header \'X-Api-Key\'."}'
    record = {"task_id": "t_auth", "failed_requests": [{"request_quote": request,
              "error_body_quote": error}], "successful_request_quotes": []}
    op = {"op": "add", "content": "When GET /users returns missing required X-Api-Key, then add header X-Api-Key.",
          "evidence": {"task_id": "t_auth", "failed_request_quote": request,
                       "error_body_quote": error, "successful_request_quote": None}}
    assert check_rule_grounding([op], [record]) == []


def test_invalid_field_error_does_not_ground_guessed_body_or_status_fixes():
    examples = [("body", "remove field 'body'"),
                ("status", "remove parameter 'status'"),
                ("body", 'set body to {"name":"Ada"}')]
    for field, fix in examples:
        request = '{"body":{"%s":"x"},"headers":{},"method":"POST","params":{},"path":"/users"}' % field
        error = '{"error":"Bad request: invalid field \'%s\'."}' % field
        record = {"task_id": "t_bad", "failed_requests": [{"request_quote": request,
                  "error_body_quote": error}], "successful_request_quotes": []}
        op = {"op": "add", "content": f"When POST /users returns invalid field {field}, then {fix}.",
              "evidence": {"task_id": "t_bad", "failed_request_quote": request,
                           "error_body_quote": error, "successful_request_quote": None}}
        assert check_rule_grounding([op], [record])


def test_grounding_rejects_contradiction_of_successful_header():
    request = trace_record()["failed_requests"][0]["request_quote"]
    error = trace_record()["failed_requests"][0]["error_body_quote"]
    success = '{"body":{},"headers":{"X-Api-Key":"key"},"method":"POST","params":{},"path":"/orders"}'
    record = trace_record([success])
    ops = [{"op": "add", "content": "When POST /orders returns invalid amount, then do not include X-Api-Key.",
            "evidence": {"task_id": "t_123", "failed_request_quote": request,
                         "error_body_quote": error, "successful_request_quote": None}}]
    issues = check_rule_grounding(ops, [record])
    assert any("contradicts successful trace" in item["reason"] for item in issues)


def test_grounding_rejects_unobserved_header_value():
    request = trace_record()["failed_requests"][0]["request_quote"]
    error = trace_record()["failed_requests"][0]["error_body_quote"]
    success = '{"body":{},"headers":{"X-Api-Key":"YOUR_API_KEY"},"method":"GET","params":{},"path":"/orders"}'
    record = trace_record([success])
    op = {"op": "add", "content": "When GET /orders returns missing X-Api-Key, then add header X-Api-Key with value BAD_KEY",
          "evidence": {"task_id": "t_123", "failed_request_quote": request,
                       "error_body_quote": error, "successful_request_quote": success}}
    issues = check_rule_grounding([op], [record])
    assert any("absent from a successful request" in item["reason"] for item in issues)


def test_trace_builder_captures_full_request_error_body_and_prior_success():
    failure = {
        "task_id": "t_123",
        "trajectory": [{
            "action": "http_request", "method": "POST", "path": "/orders",
            "params": {"status": "completed"}, "headers": {"X-Api-Key": "YOUR_API_KEY"},
            "body": {"amount": 50.0},
            "result": {"status": 400, "body": {"error": "Bad request: invalid field 'amount'."}},
        }],
    }
    histories = {"t_123": [{"passed": True, "trajectory": [{
        "action": "http_request", "method": "POST", "path": "/orders",
        "params": {}, "headers": {"X-Api-Key": "YOUR_API_KEY"},
        "body": {"amount_cents": 5000}, "result": {"status": 201, "body": {"id": "ord_1"}},
    }]}]}
    record = make_reflection_trace_records([failure], histories)[0]

    assert '"method":"POST"' in record["failed_requests"][0]["request_quote"]
    assert record["failed_requests"][0]["error_body_quote"] == '{"error":"Bad request: invalid field \'amount\'."}'
    assert len(record["successful_request_quotes"]) == 1
    assert '"amount_cents":5000' in record["successful_request_quotes"][0]


def test_trace_builder_captures_success_after_failure_in_same_episode():
    failure = {"task_id": "t_auth", "trajectory": [
        {"action": "http_request", "method": "GET", "path": "/invoices/inv_501",
         "params": {}, "headers": {}, "body": {},
         "result": {"status": 401, "body": {"error": "Missing X-Api-Key"}}},
        {"action": "http_request", "method": "GET", "path": "/invoices/inv_501",
         "params": {}, "headers": {"X-Api-Key": "YOUR_API_KEY"}, "body": {},
         "result": {"status": 200, "body": {"id": "inv_501"}}},
    ]}
    record = make_reflection_trace_records([failure], {})[0]
    assert len(record["successful_request_quotes"]) == 1
    assert '"X-Api-Key":"YOUR_API_KEY"' in record["successful_request_quotes"][0]
    failed_quote = record["failed_requests"][0]["request_quote"]
    error_quote = record["failed_requests"][0]["error_body_quote"]
    success_quote = record["successful_request_quotes"][0]
    op = {"op": "add",
          "content": "When GET /invoices returns a missing X-Api-Key error, then add header X-Api-Key with value YOUR_API_KEY.",
          "evidence": {"task_id": "t_auth", "failed_request_quote": failed_quote,
                       "error_body_quote": error_quote, "successful_request_quote": success_quote}}
    assert check_rule_grounding([op], [record]) == []
