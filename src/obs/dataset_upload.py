"""Upload task-only Phoenix datasets (no labels, expected answers, or API keys)."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from src.obs.experiments import phoenix_client

_CATEGORY_TERMS = re.compile(r"\b(easy|learnable|impossible|flaky(?:_\w+)?|noise|planted)\b", re.I)
_KEY_PLACEHOLDER = re.compile(r"secret_test_key|nvapi-[A-Za-z0-9_-]+", re.I)


def _task_rows(path: Path) -> List[Dict[str, Any]]:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for task in tasks:
        task_id = str(task["task_id"])
        text = str(task["instruction"])
        if _CATEGORY_TERMS.search(text):
            raise ValueError(f"Task text for {task_id} contains a category/taxonomy term; refusing upload")
        if _KEY_PLACEHOLDER.search(text):
            text = _KEY_PLACEHOLDER.sub("[REDACTED]", text)
        rows.append({"task_id": task_id, "task_text": text})
    return rows


def upload(data_dir: Path = Path("data")) -> Dict[str, Any]:
    client = phoenix_client()
    uploaded = {}
    for split in ("train", "heldout_id", "heldout_ood"):
        rows = _task_rows(data_dir / f"{split}.json")
        name = f"curation-plasticity-real-{split}"
        try:
            dataset = client.datasets.get_dataset(dataset=name)
        except Exception:
            dataset = client.datasets.create_dataset(
                name=name,
                dataset_description="Task text and opaque task IDs only; no labels or expected outputs.",
                inputs=[{"task_id": row["task_id"], "task_text": row["task_text"]} for row in rows],
                example_id_key="task_id",
            )
        uploaded[split] = dataset
    return uploaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    for split, dataset in upload(args.data_dir).items():
        print(f"{split}: {dataset.name} ({len(dataset)} examples)")


if __name__ == "__main__":
    main()
