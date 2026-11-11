"""
Generates resized task matrices:
- train: 30 tasks with configurable noise_frac (0.10, 0.20, 0.30)
- heldout_id: 40 clean tasks
- heldout_ood: 20 clean tasks

Near-miss impossible tasks:
Planted on the EXACT SAME endpoints, filters, and ID formats as solvable tasks
so that invalid/hallucinatory rules learned from them directly conflict with and harm solvable tasks:
1. Missing customer ID 'u99' in /users/u99 (same endpoint as 'u1', 'u2')
2. Missing order ID 'ord_999' in /orders/ord_999 (same endpoint as 'ord_101')
3. Unsupported role filter filter[role]=superadmin on /users (same bracket syntax as role=admin)
4. Nonexistent invoice 'inv_999' on /invoices/inv_999
"""
from typing import Dict, Any, List, Tuple
import hashlib
import json
import random
from pathlib import Path

def _hash_id(prefix: str, content: str) -> str:
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{h}"

# Clean Easy Templates (Solvable without playbook)
EASY_TEMPLATES = [
    {
        "instruction": "Fetch user details for ID 'u1' and return their email address.",
        "expected": {"type": "exact", "target_val": "alice@example.com"},
        "requires_quirks": []
    },
    {
        "instruction": "Check the status of order 'ord_101' and return the status string.",
        "expected": {"type": "exact", "target_val": "completed"},
        "requires_quirks": []
    },
    {
        "instruction": "Fetch user details for ID 'u2' and return their full name.",
        "expected": {"type": "exact", "target_val": "Bob Jones"},
        "requires_quirks": []
    },
    {
        "instruction": "Check invoice 'inv_501' and return whether it is paid (True or False).",
        "expected": {"type": "exact", "target_val": "True"},
        "requires_quirks": []
    }
]

# Learnable Templates (Requires 1-2 quirks)
LEARNABLE_TEMPLATES = [
    {
        "instruction": "List all users and report the total count of users across all pages.",
        "expected": {"type": "count", "target_val": 5},
        "requires_quirks": ["pagination"]
    },
    {
        "instruction": "List all orders across all pages and return only the ID of the last order in the system.",
        "expected": {"type": "exact", "target_val": "ord_105"},
        "requires_quirks": ["pagination"]
    },
    {
        "instruction": "Retrieve invoice 'inv_502' using the provided API key 'secret_test_key' and return its total_cents.",
        "expected": {"type": "exact", "target_val": "12500"},
        "requires_quirks": ["auth_header"]
    },
    {
        "instruction": "Create a new order for customer 'u1' with total amount of 50.00 dollars on date '2024-05-10'. Return the new order ID.",
        "expected": {"type": "contains", "target_val": "ord_"},
        "requires_quirks": ["cents", "iso_date", "idempotency_key"]
    },
    {
        "instruction": "Create a new user named 'Frank Miller' with email 'frank@example.com' and ID 'u6'. Return the created customer_id.",
        "expected": {"type": "exact", "target_val": "u6"},
        "requires_quirks": ["customer_id_rename", "idempotency_key"]
    },
    {
        "instruction": "Find all users with role 'admin' and return their customer_id.",
        "expected": {"type": "exact", "target_val": "u1"},
        "requires_quirks": ["bracket_filtering"]
    },
    {
        "instruction": "Find all orders with status 'completed' and return only the number of orders found as an integer.",
        "expected": {"type": "count", "target_val": 3},
        "requires_quirks": ["bracket_filtering"]
    }
]

# Near-Miss Impossible Tasks: Shares identical endpoints & filter formats
IMPOSSIBLE_NEAR_MISS_TEMPLATES = [
    {
        # Tempts rule: "If user not found, strip prefix or query /users"
        "instruction": "Fetch user details for ID 'u99' and return their email address.",
        "expected": {"type": "exact", "target_val": "unobtainable_u99@example.com"},
        "requires_quirks": []
    },
    {
        # Tempts rule: "If order missing, retry /orders without filter"
        "instruction": "Check the status of order 'ord_999' and return the status string.",
        "expected": {"type": "exact", "target_val": "completed"},
        "requires_quirks": []
    },
    {
        # Tempts rule: "Filter syntax is invalid, omit filter[field]"
        "instruction": "Find all users with role 'superadmin' and return their customer_id.",
        "expected": {"type": "exact", "target_val": "u99"},
        "requires_quirks": ["bracket_filtering"]
    },
    {
        "instruction": "Retrieve invoice 'inv_999' and return its total_cents.",
        "expected": {"type": "exact", "target_val": "99900"},
        "requires_quirks": []
    }
]

FLAKY_JUDGE_TEMPLATES = [
    {
        "instruction": "Fetch user 'u3' and return their user role.",
        "expected": {"type": "exact", "target_val": "user"},
        "requires_quirks": []
    },
    {
        "instruction": "Retrieve the due date of invoice 'inv_501'.",
        "expected": {"type": "exact", "target_val": "2024-06-01"},
        "requires_quirks": []
    }
]

FLAKY_AMBIGUOUS_TEMPLATES = [
    {
        "instruction": "Get user 'u4' created timestamp in standard format.",
        "expected": {
            "type": "ambiguous",
            "reading_a": "2024-04-05",
            "reading_b": "04/05/2024"
        },
        "requires_quirks": []
    },
    {
        "instruction": "Return invoice 'inv_501' total monetary value.",
        "expected": {
            "type": "ambiguous",
            "reading_a": "4500",
            "reading_b": "$45.00"
        },
        "requires_quirks": []
    }
]

def generate_task_suites(
    noise_frac: float = 0.20,
    seed: int = 42,
    output_dir: str = "data",
    num_train: int = 30,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Train set
    if num_train < 7:
        raise ValueError("num_train must leave room for 6 easy tasks and at least one learnable task")
    num_noise = int(round(num_train * noise_frac)) # e.g. 6 tasks at 20%
    num_easy = 6
    num_learnable = num_train - num_noise - num_easy

    train_tasks = []
    labels = {}

    def add_task(template: dict, cat: str, idx: int) -> dict:
        unique_str = f"{cat}_{idx}_{template['instruction']}"
        t_id = _hash_id("t", unique_str)
        public_task = {
            "task_id": t_id,
            "instruction": template["instruction"],
            "expected": template["expected"]
        }
        labels[t_id] = {
            "task_id": t_id,
            "category": cat,
            "requires_quirks": template.get("requires_quirks", [])
        }
        return public_task

    # Add easy
    for i in range(num_easy):
        tpl = EASY_TEMPLATES[i % len(EASY_TEMPLATES)]
        train_tasks.append(add_task(tpl, "easy", i))

    # Add learnable
    for i in range(num_learnable):
        tpl = LEARNABLE_TEMPLATES[i % len(LEARNABLE_TEMPLATES)]
        train_tasks.append(add_task(tpl, "learnable", i))

    # Add near-miss noise
    noise_pool = [
        (IMPOSSIBLE_NEAR_MISS_TEMPLATES[0], "impossible"),
        (IMPOSSIBLE_NEAR_MISS_TEMPLATES[1], "impossible"),
        (FLAKY_JUDGE_TEMPLATES[0], "flaky_judge"),
        (FLAKY_JUDGE_TEMPLATES[1], "flaky_judge"),
        (FLAKY_AMBIGUOUS_TEMPLATES[0], "flaky_ambiguous"),
        (FLAKY_AMBIGUOUS_TEMPLATES[1], "flaky_ambiguous")
    ]
    for i in range(num_noise):
        tpl, cat = noise_pool[i % len(noise_pool)]
        train_tasks.append(add_task(tpl, cat, i))

    rng.shuffle(train_tasks)

    # 2. Heldout ID set (40 clean tasks: easy + learnable)
    heldout_id_tasks = []
    for i in range(40):
        if i % 4 == 0:
            tpl = EASY_TEMPLATES[i % len(EASY_TEMPLATES)]
            heldout_id_tasks.append(add_task(tpl, "easy", 1000 + i))
        else:
            tpl = LEARNABLE_TEMPLATES[i % len(LEARNABLE_TEMPLATES)]
            heldout_id_tasks.append(add_task(tpl, "learnable", 1000 + i))

    # 3. Heldout OOD set (20 clean tasks)
    heldout_ood_tasks = []
    for i in range(20):
        tpl = LEARNABLE_TEMPLATES[i % len(LEARNABLE_TEMPLATES)]
        heldout_ood_tasks.append(add_task(tpl, "learnable", 2000 + i))

    # Save
    with open(f"{output_dir}/train.json", "w") as f:
        json.dump(train_tasks, f, indent=2)
    with open(f"{output_dir}/heldout_id.json", "w") as f:
        json.dump(heldout_id_tasks, f, indent=2)
    with open(f"{output_dir}/heldout_ood.json", "w") as f:
        json.dump(heldout_ood_tasks, f, indent=2)
    with open(f"{output_dir}/labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    return train_tasks, heldout_id_tasks, heldout_ood_tasks, labels

if __name__ == "__main__":
    generate_task_suites()
    print("Generated resized task suites: 30 train, 40 held-out ID, 20 held-out OOD.")
