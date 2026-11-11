"""Run Phoenix code-evaluator experiments from saved per-task outcomes.

This module never calls an actor or hosted model. Supply an outcomes JSON mapping
opaque task IDs to success/steps_used/playbook_read for each selected arm/seed/epoch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.obs.dataset_upload import upload
from src.obs.experiments import phoenix_client, run_saved_experiment
from src.obs.phoenix_setup import flush_phoenix, init_phoenix_tracer
from src.agents.actor import ActorAgent
from src.loop.provenance import Playbook
from src.testbed.grader import Grader
from src.testbed.mock_api import MockAPI


def main() -> None:
    parser = argparse.ArgumentParser(description="Run saved-outcome Phoenix experiments; no model calls.")
    parser.add_argument("--outcomes-json", type=Path,
                        help="JSON keyed by arm, seed, epoch, then opaque task_id")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--arms", nargs="+", default=["A0", "A3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", nargs="+", type=int, required=True)
    parser.add_argument("--from-checkpoints", action="store_true",
                        help="Evaluate saved playbooks with the local mock actor; no model API calls")
    args = parser.parse_args()

    init_phoenix_tracer()
    datasets = upload(args.data_dir)
    client = phoenix_client()
    if not args.from_checkpoints and args.outcomes_json is None:
        parser.error("--outcomes-json is required unless --from-checkpoints is set")
    outcome_data: Dict[str, Any] = (json.loads(args.outcomes_json.read_text(encoding="utf-8"))
                                    if args.outcomes_json else {})
    if args.from_checkpoints:
        # Existing task data and checkpoints only. Labels stay in this local
        # analysis process and are never sent to Phoenix.
        data = {split: json.loads((args.data_dir / f"{split}.json").read_text(encoding="utf-8"))
                for split in ("heldout_id", "heldout_ood")}
        labels_path = args.data_dir / "labels.json"
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        actor = ActorAgent(model_name="mock-actor")
        grader = Grader(seed=42)
        outcome_data = {}
        from src.obs.spans import span
        for arm in args.arms:
            for seed in args.seeds:
                for epoch in args.epochs:
                    key = f"{arm}/{seed}/{epoch}"
                    playbook = Playbook()
                    checkpoint = Path("checkpoints") / f"{arm}_s{seed}" / f"checkpoint_{epoch}" / "playbook.md"
                    if not checkpoint.exists():
                        raise SystemExit(f"No saved playbook at {checkpoint}")
                    import re
                    for rule_id, content in re.findall(r"## \[(rule_\d+)\]\n(.*?)(?=\n## |\Z)",
                                                       checkpoint.read_text(encoding="utf-8"), re.DOTALL):
                        playbook.rules[rule_id] = {"id": rule_id, "content": content.strip(),
                                                   "created_epoch": epoch, "source_task_ids": []}
                    saved_rows = {}
                    for split in ("heldout_id", "heldout_ood"):
                        for task in data[split]:
                            with span("heldout_eval", phase="eval", attributes={
                                "arm": arm, "seed": seed, "epoch": epoch,
                                "task_id": task["task_id"], "execution_mode": "mock",
                            }) as eval_span:
                                result = actor.run_episode(task, playbook, MockAPI())
                                category = labels[task["task_id"]]["category"]
                                passed, _ = grader.grade(task, category, result["trajectory"], result["final_answer"])
                                row = {"success": bool(passed), "steps_used": result["steps_used"],
                                       "playbook_read": bool(result["read_playbook_called"])}
                                eval_span.set_attribute("success", row["success"])
                                eval_span.set_attribute("steps_used", row["steps_used"])
                                eval_span.set_attribute("playbook_read", row["playbook_read"])
                                saved_rows[task["task_id"]] = row
                    outcome_data[key] = saved_rows
    for arm in args.arms:
        for seed in args.seeds:
            for epoch in args.epochs:
                key = f"{arm}/{seed}/{epoch}"
                rows = outcome_data.get(key)
                if not isinstance(rows, dict):
                    raise SystemExit(f"Missing saved outcomes for {key}; no experiment created")
                for split in ("heldout_id", "heldout_ood"):
                    experiment = run_saved_experiment(
                        client, datasets[split], arm=arm, seed=seed, epoch=epoch, rows=rows
                    )
                    print(f"{key} {split}: {experiment.experiment_name}")
    flush_phoenix()


if __name__ == "__main__":
    main()
