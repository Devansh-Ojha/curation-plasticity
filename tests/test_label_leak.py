import json
import pytest
from pathlib import Path

def test_no_category_leaks_in_tasks():
    data_dir = Path("data")
    forbidden_strings = ["easy", "learnable", "impossible", "flaky_judge", "flaky_ambiguous", "flaky"]

    for split in ["train.json", "heldout_id.json", "heldout_ood.json"]:
        file_path = data_dir / split
        assert file_path.exists(), f"Missing split {split}"
        with open(file_path, "r") as f:
            tasks = json.load(f)
            raw_text = json.dumps(tasks).lower()
            for forbidden in forbidden_strings:
                # Opaque IDs or instructions must not leak the category name
                # Ensure the forbidden category string does not appear in keys or task metadata
                for t in tasks:
                    assert forbidden not in t.get("task_id", "").lower()
                    assert forbidden not in t.get("instruction", "").lower()
                    assert forbidden not in json.dumps(t.get("expected", {})).lower()

def test_opaque_ids_unique():
    data_dir = Path("data")
    seen_ids = set()
    for split in ["train.json", "heldout_id.json", "heldout_ood.json"]:
        with open(data_dir / split, "r") as f:
            tasks = json.load(f)
            for t in tasks:
                t_id = t["task_id"]
                assert t_id.startswith("t_"), f"ID {t_id} does not follow opaque format"
                assert t_id not in seen_ids, f"Duplicate task ID: {t_id}"
                seen_ids.add(t_id)

def test_labels_file_contains_all_tasks():
    data_dir = Path("data")
    with open(data_dir / "labels.json", "r") as f:
        labels = json.load(f)

    for split in ["train.json", "heldout_id.json", "heldout_ood.json"]:
        with open(data_dir / split, "r") as f:
            tasks = json.load(f)
            for t in tasks:
                assert t["task_id"] in labels, f"Task {t['task_id']} missing from labels.json"
