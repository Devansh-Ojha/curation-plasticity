from src.testbed.mock_api import MockAPI


def test_create_order_400_names_field_without_disclosing_format():
    api = MockAPI()
    result = api.request(
        "POST", "/orders", headers={"X-Api-Key": "test", "Idempotency-Key": "k"},
        body={"amount": 50.0},
    )
    assert result["status"] == 400
    assert result["body"]["error"] == "Bad request: invalid field 'amount'."
    assert "amount_cents" not in result["body"]["error"]


def test_missing_idempotency_header_has_identical_wording_for_create_categories():
    api = MockAPI()
    users = api.request("POST", "/users", headers={"X-Api-Key": "test"}, body={})
    orders = api.request("POST", "/orders", headers={"X-Api-Key": "test"}, body={})
    assert users == orders
    assert users["body"]["error"] == "Bad request: missing header 'Idempotency-Key'."


def test_bad_request_shape_returns_400_instead_of_crashing():
    result = MockAPI().request("POST", "/users", headers={}, body='{"name":"Frank"}')
    assert result["status"] == 400
    assert result["body"]["error"] == "Bad request: invalid field 'body'."
