"""
In-process simulated REST server with 8 learnable quirks and deterministic responses.
No real network calls; pure Python dictionary state.
"""
from typing import Dict, Any, Optional
import copy
import re

def bad_request(problem: str, field: str) -> Dict[str, Any]:
    return {"status": 400, "headers": {}, "body": {"error": f"Bad request: {problem} '{field}'."}}

class MockAPI:
    def __init__(self, initial_rate_limit: int = 0):
        # Base database
        self.users = {
            "u1": {"customer_id": "u1", "name": "Alice Smith", "email": "alice@example.com", "role": "admin", "created_at": "2024-01-15"},
            "u2": {"customer_id": "u2", "name": "Bob Jones", "email": "bob@example.com", "role": "user", "created_at": "2024-02-20"},
            "u3": {"customer_id": "u3", "name": "Charlie Brown", "email": "charlie@example.com", "role": "user", "created_at": "2024-03-10"},
            "u4": {"customer_id": "u4", "name": "Dana White", "email": "dana@example.com", "role": "manager", "created_at": "2024-04-05"},
            "u5": {"customer_id": "u5", "name": "Evan Wright", "email": "evan@example.com", "role": "user", "created_at": "2024-05-12"}
        }
        self.orders = {
            "ord_101": {"id": "ord_101", "customer_id": "u1", "amount_cents": 4500, "status": "completed", "date": "2024-05-01"},
            "ord_102": {"id": "ord_102", "customer_id": "u2", "amount_cents": 12500, "status": "pending", "date": "2024-05-02"},
            "ord_103": {"id": "ord_103", "customer_id": "u1", "amount_cents": 8900, "status": "completed", "date": "2024-05-03"},
            "ord_104": {"id": "ord_104", "customer_id": "u3", "amount_cents": 1500, "status": "cancelled", "date": "2024-05-04"},
            "ord_105": {"id": "ord_105", "customer_id": "u4", "amount_cents": 23000, "status": "completed", "date": "2024-05-05"}
        }
        self.invoices = {
            "inv_501": {"id": "inv_501", "order_id": "ord_101", "total_cents": 4500, "due_date": "2024-06-01", "paid": True},
            "inv_502": {"id": "inv_502", "order_id": "ord_102", "total_cents": 12500, "due_date": "2024-06-02", "paid": False}
        }
        self.seen_idempotency_keys = set()
        self.rate_limit_hits = 0
        self.rate_limit_threshold = initial_rate_limit # If > 0, returns 429 on first N requests

    def request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes an API request against the mock server.
        Returns a dict: {"status": int, "headers": dict, "body": dict/str}
        """
        if not isinstance(method, str) or not isinstance(path, str):
            return bad_request("invalid field", "method or path")
        params = {} if params is None else params
        headers = {} if headers is None else headers
        body = {} if body is None else body
        for field_name, field_value in (("params", params), ("headers", headers), ("body", body)):
            if not isinstance(field_value, dict):
                return bad_request("invalid field", field_name)
        method = method.upper()

        # -------------------------------------------------------------
        # Quirk 2: Auth header must be 'X-Api-Key: <key>', not 'Authorization: Bearer'
        # -------------------------------------------------------------
        if "Authorization" in headers:
            return {
                "status": 401,
                "headers": {},
                "body": {"error": "Invalid auth scheme. Bearer tokens unsupported; use 'X-Api-Key' header."}
            }
        if "X-Api-Key" not in headers and "x-api-key" not in headers:
            return {
                "status": 401,
                "headers": {},
                "body": {"error": "Missing required authentication header 'X-Api-Key'."}
            }

        # -------------------------------------------------------------
        # Quirk 4: Rate limit (returns 429 with retry_after header)
        # -------------------------------------------------------------
        if self.rate_limit_threshold > 0 and self.rate_limit_hits < self.rate_limit_threshold:
            self.rate_limit_hits += 1
            return {
                "status": 429,
                "headers": {"retry_after": "1"},
                "body": {"error": "Rate limit exceeded. Please retry after indicated seconds."}
            }

        # -------------------------------------------------------------
        # Routing
        # -------------------------------------------------------------
        parts = [p for p in path.strip("/").split("/") if p]
        if not parts:
            return {"status": 404, "headers": {}, "body": {"error": "Endpoint not found"}}

        resource = parts[0]
        entity_id = parts[1] if len(parts) > 1 else None
        action = parts[2] if len(parts) > 2 else None

        # Check for impossible endpoints planted deliberately
        if resource in ["analytics", "export_raw", "backdoor"] or action in ["export", "purge"]:
            return {
                "status": 404,
                "headers": {},
                "body": {"error": f"Endpoint /{path} does not exist on this API version."}
            }

        if resource == "users":
            return self._handle_users(method, entity_id, params, headers, body)
        elif resource == "orders":
            return self._handle_orders(method, entity_id, params, headers, body)
        elif resource == "invoices":
            return self._handle_invoices(method, entity_id, params, headers, body)
        else:
            return {"status": 404, "headers": {}, "body": {"error": f"Resource /{resource} not found"}}

    def _handle_users(self, method: str, entity_id: Optional[str], params: dict, headers: dict, body: dict) -> dict:
        if method == "GET":
            if entity_id:
                if entity_id not in self.users:
                    return {"status": 404, "headers": {}, "body": {"error": f"User {entity_id} not found"}}
                return {"status": 200, "headers": {}, "body": copy.deepcopy(self.users[entity_id])}
            
            # List users
            # Quirk 1: Pagination beyond page 1 requires page_token
            # Quirk 8: Filtering uses filter[field]=value, not field=value
            items = list(self.users.values())
            
            # Check filter quirk: if passed plain 'role=admin', ignore or warn
            role_filter = None
            if "filter[role]" in params:
                role_filter = params["filter[role]"]
            elif "role" in params:
                return {
                    "status": 400,
                    "headers": {},
                    "body": {"error": "Bad request: invalid field 'role'."}
                }
            
            if role_filter:
                items = [u for u in items if u["role"] == role_filter]

            # Pagination
            page_size = 2
            page_token = params.get("page_token")
            if not page_token:
                # Returns first page
                return {
                    "status": 200,
                    "headers": {},
                    "body": {
                        "users": items[:page_size],
                        "next_page_token": "page_2" if len(items) > page_size else None,
                        "total_count": len(items)
                    }
                }
            elif page_token == "page_2":
                return {
                    "status": 200,
                    "headers": {},
                    "body": {
                        "users": items[page_size:page_size*2],
                        "next_page_token": "page_3" if len(items) > page_size*2 else None,
                        "total_count": len(items)
                    }
                }
            elif page_token == "page_3":
                return {
                    "status": 200,
                    "headers": {},
                    "body": {
                        "users": items[page_size*2:],
                        "next_page_token": None,
                        "total_count": len(items)
                    }
                }
            else:
                return bad_request("invalid field", "page_token")

        elif method == "POST":
            # Quirk 7: Idempotency-Key required for create
            if "Idempotency-Key" not in headers:
                return bad_request("missing header", "Idempotency-Key")
            
            # Quirk 5: Field rename in v2 (cust_id -> customer_id)
            if "cust_id" in body and "customer_id" not in body:
                return bad_request("invalid field", "cust_id")

            user_id = body.get("customer_id", f"u{len(self.users)+1}")
            new_user = {
                "customer_id": user_id,
                "name": body.get("name", "Unknown"),
                "email": body.get("email", ""),
                "role": body.get("role", "user"),
                "created_at": "2024-05-15"
            }
            self.users[user_id] = new_user
            return {"status": 201, "headers": {}, "body": new_user}

        return {"status": 405, "headers": {}, "body": {"error": f"Method {method} not allowed on users"}}

    def _handle_orders(self, method: str, entity_id: Optional[str], params: dict, headers: dict, body: dict) -> dict:
        if method == "GET":
            if entity_id:
                if entity_id not in self.orders:
                    return {"status": 404, "headers": {}, "body": {"error": f"Order {entity_id} not found"}}
                return {"status": 200, "headers": {}, "body": copy.deepcopy(self.orders[entity_id])}
            
            items = list(self.orders.values())
            
            # Quirk 8: Filtering uses filter[field]=value
            if "filter[status]" in params:
                status_filter = params["filter[status]"]
                items = [o for o in items if o["status"] == status_filter]
            elif "status" in params:
                return {
                    "status": 400,
                    "headers": {},
                    "body": {"error": "Bad request: invalid field 'status'."}
                }

            # Quirk 1: Pagination
            page_size = 2
            page_token = params.get("page_token")
            if not page_token:
                return {
                    "status": 200,
                    "headers": {},
                    "body": {
                        "orders": items[:page_size],
                        "next_page_token": "page_2" if len(items) > page_size else None,
                        "total_count": len(items)
                    }
                }
            elif page_token == "page_2":
                return {
                    "status": 200,
                    "headers": {},
                    "body": {
                        "orders": items[page_size:page_size*2],
                        "next_page_token": "page_3" if len(items) > page_size*2 else None,
                        "total_count": len(items)
                    }
                }
            else:
                return {
                    "status": 200,
                    "headers": {},
                    "body": {"orders": items[page_size*2:], "next_page_token": None, "total_count": len(items)}
                }

        elif method == "POST":
            # Quirk 7: Idempotency-Key
            if "Idempotency-Key" not in headers:
                return bad_request("missing header", "Idempotency-Key")

            # Quirk 6: Money in integer cents
            if "amount" in body:
                return bad_request("invalid field", "amount")
            if "amount_cents" not in body:
                return bad_request("missing field", "amount_cents")
            if not isinstance(body["amount_cents"], int):
                return bad_request("invalid field", "amount_cents")

            # Quirk 3: Date format ISO-8601 YYYY-MM-DD
            if "order_date" in body and "date" not in body:
                return bad_request("invalid field", "order_date")
            date_val = body.get("date")
            if date_val and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(date_val)):
                return bad_request("invalid field", "date")

            order_id = f"ord_{100 + len(self.orders) + 1}"
            new_order = {
                "id": order_id,
                "customer_id": body.get("customer_id", "u1"),
                "amount_cents": body["amount_cents"],
                "status": body.get("status", "pending"),
                "date": date_val or "2024-05-10"
            }
            self.orders[order_id] = new_order
            return {"status": 201, "headers": {}, "body": new_order}

        return {"status": 405, "headers": {}, "body": {"error": f"Method {method} not allowed on orders"}}

    def _handle_invoices(self, method: str, entity_id: Optional[str], params: dict, headers: dict, body: dict) -> dict:
        if method == "GET":
            if entity_id:
                if entity_id not in self.invoices:
                    return {"status": 404, "headers": {}, "body": {"error": f"Invoice {entity_id} not found"}}
                return {"status": 200, "headers": {}, "body": copy.deepcopy(self.invoices[entity_id])}
            return {"status": 200, "headers": {}, "body": {"invoices": list(self.invoices.values())}}
        return {"status": 405, "headers": {}, "body": {"error": f"Method {method} not allowed on invoices"}}
