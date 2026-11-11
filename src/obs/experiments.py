"""Phoenix experiment helpers for saved outcomes or explicitly supplied tasks."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, List, Optional

from phoenix.client import Client

from src.obs.phoenix_setup import LOCAL_ENDPOINT
from src.agents.actor import ActorAgent
from src.testbed.mock_api import MockAPI
from src.testbed.grader import Grader
from src.loop.provenance import Playbook


def phoenix_client() -> Client:
    return Client(base_url=LOCAL_ENDPOINT)


def evaluators():
    """Code evaluators over task outputs; no model-based evaluator/API calls."""
    def success(output: Mapping[str, Any]) -> bool:
        return bool(output.get("success", False))

    def steps_used(output: Mapping[str, Any]) -> int:
        return int(output.get("steps_used", 0))

    def playbook_read(output: Mapping[str, Any]) -> bool:
        return bool(output.get("playbook_read", False))

    return {"success": success, "steps_used": steps_used, "playbook_read": playbook_read}


def run_saved_experiment(client: Client, dataset: Any, *, arm: str, seed: int,
                         epoch: int, rows: Mapping[str, Mapping[str, Any]]):
    """Record one (arm, seed, epoch) result set. The task uses saved outcomes only."""
    def task(input: Mapping[str, Any]) -> Dict[str, Any]:
        task_id = input["task_id"]
        if task_id not in rows:
            raise KeyError(f"No saved result for opaque task ID {task_id}")
        return dict(rows[task_id])

    return client.experiments.run_experiment(
        dataset=dataset,
        task=task,
        evaluators=evaluators(),
        experiment_name=f"{arm}-seed-{seed}-epoch-{epoch}",
        experiment_metadata={"arm": arm, "seed": seed, "epoch": epoch,
                             "execution_mode": "saved-log-replay"},
        print_summary=True,
    )


def load_dataset_dfs(data_dir: str = "data"):
    """Legacy local-analysis loader. Phoenix upload uses inputs only."""
    import pandas as pd
    dfs = {}
    for split in ("train", "heldout_id", "heldout_ood"):
        tasks = json.loads((Path(data_dir) / f"{split}.json").read_text(encoding="utf-8"))
        dfs[split] = pd.DataFrame([{"task_id": t["task_id"], "instruction": t["instruction"]}
                                  for t in tasks])
    return dfs


def sync_phoenix_datasets(client: Client, data_dir: str = "data"):
    """Compatibility wrapper; dataset rows contain only task ID and text."""
    from src.obs.dataset_upload import _task_rows
    datasets = {}
    for split in ("train", "heldout_id", "heldout_ood"):
        name = f"curation-plasticity-real-{split}"
        try:
            datasets[split] = client.datasets.get_dataset(dataset=name)
        except Exception:
            rows = _task_rows(Path(data_dir) / f"{split}.json")
            datasets[split] = client.datasets.create_dataset(
                name=name,
                dataset_description="Task text and opaque task IDs only; no labels or expected outputs.",
                inputs=[{"task_id": r["task_id"], "task_text": r["task_text"]} for r in rows],
                example_id_key="task_id",
            )
    return datasets


def evaluate_checkpoint_on_split(
    playbook_path: str,
    tasks_split: List[Dict[str, Any]],
    labels: Dict[str, Dict[str, Any]],
    actor: ActorAgent,
    grader: Grader,
    playbook_obj: Optional[Playbook] = None,
) -> Dict[str, Any]:
    """Run the existing local/mock checkpoint evaluator used by run_pipeline."""
    if playbook_obj is not None:
        playbook = playbook_obj
    else:
        playbook = Playbook()
        path = Path(playbook_path)
        if path.exists():
            text = path.read_text(encoding="utf-8")
            for rule_id, content in re.findall(r"## \[(rule_\d+)\]\n(.*?)(?=\n## |\Z)", text, re.DOTALL):
                playbook.rules[rule_id] = {"id": rule_id, "content": content.strip(),
                                           "created_epoch": 1, "source_task_ids": []}
    results = []
    successes = 0
    api = MockAPI()
    for task in tasks_split:
        task_id = task["task_id"]
        category = labels[task_id]["category"]
        result = actor.run_episode(task, playbook, api)
        passed, _ = grader.grade(task, category, result["trajectory"], result["final_answer"])
        successes += int(passed)
        results.append({"task_id": task_id, "passed": passed,
                        "steps": result["steps_used"],
                        "read_playbook": result["read_playbook_called"]})
    total = len(tasks_split)
    return {"success_rate": 100.0 * successes / total if total else 0.0,
            "success_count": successes, "total_tasks": total, "results": results}
