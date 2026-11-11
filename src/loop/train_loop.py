"""
Main training loop orchestrator executing arms A0-A3:
- Per epoch:
  1. Select tasks according to arm
  2. Act: run actor with current playbook
  3. Curate (for A3): LLM curator skip/anchor decisions
  4. Reflect: generate rules from filtered failures (+ optional anchor)
  5. Provenance: record rule_id -> source_task_ids by harness
  6. Gate: re-run minibatch with candidate playbook, accept/reject
  7. Commit checkpoint playbook
  8. Track per-phase costs (act, curate, reflect, gate)
"""
from typing import Dict, Any, List, Optional
import json
import random
from pathlib import Path
from src.agents.actor import ActorAgent
from src.agents.curator_reflector import Curator, Reflector
from src.testbed.mock_api import MockAPI
from src.testbed.grader import Grader
from src.loop.provenance import Playbook
from src.loop.arms import select_minibatch, NumericCurriculum
from src.loop.gating import evaluate_minibatch, gate_minibatch
from src.obs.spans import trace_phase, trace_episode

def run_train_arm(
    arm: str,
    seed: int,
    epochs: int,
    batch_size: int,
    noise_frac: float,
    train_tasks: List[Dict[str, Any]],
    labels: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    checkpoints_dir: str = "checkpoints"
) -> Dict[str, Any]:
    """
    Runs full training process for one arm and seed.
    Saves checkpoints and returns detailed epoch logs.
    """
    rng = random.Random(seed)
    arm_dir = Path(checkpoints_dir) / f"{arm}_s{seed}"
    arm_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    actor = ActorAgent(
        model_name=config.get("ACTOR_MODEL", "mock-actor"),
        price_table=config.get("PRICE_TABLE", {})
    )
    curator = Curator(
        model_name=config.get("OPTIMIZER_MODEL", "mock-optimizer"),
        price_table=config.get("PRICE_TABLE", {})
    )
    reflector = Reflector(
        model_name=config.get("OPTIMIZER_MODEL", "mock-optimizer"),
        price_table=config.get("PRICE_TABLE", {})
    )
    grader = Grader(seed=seed)
    
    current_playbook = Playbook(max_tokens=config.get("MAX_PLAYBOOK_TOKENS", 2000))
    
    # Checkpoint 0 (Baseline empty playbook)
    cp0_dir = arm_dir / "checkpoint_0"
    cp0_dir.mkdir(exist_ok=True)
    with open(cp0_dir / "playbook.md", "w") as f:
        f.write(current_playbook.to_markdown())

    # State tracking
    task_histories: Dict[str, List[Dict[str, Any]]] = {t["task_id"]: [] for t in train_tasks}
    clean_success_ids: List[str] = []
    curator_skip_until: Dict[str, int] = {}
    curriculum = NumericCurriculum(
        b=config.get("CURRICULUM_B", 3),
        k=config.get("CURRICULUM_K", 1)
    )

    epoch_logs = []
    cumulative_cost = {"act": 0.0, "curate": 0.0, "reflect": 0.0, "gate": 0.0, "total": 0.0}

    for epoch in range(1, epochs + 1):
        epoch_cost = {"act": 0.0, "curate": 0.0, "reflect": 0.0, "gate": 0.0}
        reflection_rejections = {"grounding_rejections": 0, "format_rejections": 0}
        gate_rejection_count = 0
        
        # 1. Select minibatch
        with trace_phase("select", arm, seed, epoch):
            selected_tasks = select_minibatch(
                arm=arm,
                all_train_tasks=train_tasks,
                batch_size=batch_size,
                rng=rng,
                curriculum=curriculum,
                curator_skip_until=curator_skip_until,
                labels=labels,
                current_epoch=epoch
            )

        # 2. Act
        batch_outcomes = {}
        failed_tasks_info = []
        mock_api = MockAPI() # fresh API state per batch

        with trace_phase("act", arm, seed, epoch) as act_span:
            for t in selected_tasks:
                t_id = t["task_id"]
                cat = labels[t_id]["category"]
                
                with trace_episode(t_id, arm, seed, epoch, current_playbook.token_count(), noise_frac):
                    run_res = actor.run_episode(t, current_playbook, mock_api)
                
                passed, feedback = grader.grade(t, cat, run_res["trajectory"], run_res["final_answer"])
                batch_outcomes[t_id] = passed
                epoch_cost["act"] += run_res["cost_usd"]

                # Record history row
                task_histories[t_id].append({
                    "epoch": epoch,
                    "passed": passed,
                    "feedback": feedback,
                    "steps": run_res["steps_used"],
                    "read_playbook": run_res["read_playbook_called"],
                    "trajectory": run_res["trajectory"]
                })

                if passed:
                    if t_id not in clean_success_ids and cat in ["easy", "learnable"]:
                        clean_success_ids.append(t_id)
                else:
                    failed_tasks_info.append({
                        "task_id": t_id,
                        "instruction": t["instruction"],
                        "feedback": feedback,
                        "trajectory": run_res["trajectory"],
                        "prior_successful_trajectories": [
                            h["trajectory"] for h in task_histories[t_id][:-1]
                            if h.get("passed") and h.get("trajectory")
                        ]
                    })

                # Update numeric curriculum (A2)
                if arm == "A2":
                    curriculum.update(t_id, passed)

            act_span.set_attribute("cost_usd", epoch_cost["act"])

        # 3. Curate (A3)
        curator_decisions = []
        tasks_to_reflect = failed_tasks_info
        anchors = []

        if arm == "A3":
            with trace_phase("curate", arm, seed, epoch) as curate_span:
                filtered_failures = []
                for f_info in failed_tasks_info:
                    f_id = f_info["task_id"]
                    t_obj = next(t for t in selected_tasks if t["task_id"] == f_id)
                    dec = curator.decide(t_obj, task_histories[f_id], clean_success_ids)
                    curator_decisions.append(dec)
                    epoch_cost["curate"] += dec["cost_usd"]

                    if dec["decision"] == "run":
                        filtered_failures.append(f_info)
                        if dec.get("anchor_success_id"):
                            anchors.append({"task_id": dec["anchor_success_id"]})
                    else:
                        # Set skip interval
                        curator_skip_until[f_id] = epoch + dec.get("skip_epochs", 1)

                tasks_to_reflect = filtered_failures
                curate_span.set_attribute("cost_usd", epoch_cost["curate"])

        # 4. Reflect
        candidate_playbook = current_playbook.clone()
        reflect_accepted = False
        gate_info = {}

        if tasks_to_reflect:
            with trace_phase("reflect", arm, seed, epoch) as reflect_span:
                ops, ref_cost_info, attempts = reflector.reflect(
                    tasks_to_reflect,
                    current_playbook.to_markdown(),
                    anchors=anchors
                )
                epoch_cost["reflect"] += ref_cost_info["cost_usd"]
                reflection_rejections = {
                    "grounding_rejections": ref_cost_info.get("grounding_rejections", 0),
                    "format_rejections": ref_cost_info.get("format_rejections", 0),
                }
                reflect_span.set_attribute("cost_usd", ref_cost_info["cost_usd"])
                reflect_span.set_attribute("attempts", attempts)

                # 5. Provenance: Harness records source task IDs
                source_ids = list(dict.fromkeys(
                    op.get("evidence", {}).get("task_id")
                    for op in ops
                    if op.get("op") in {"add", "modify"}
                    and isinstance(op.get("evidence"), dict)
                ))
                valid_ops = candidate_playbook.apply_ops(ops, epoch=epoch, source_task_ids=source_ids)

            # 6. Gate (GEPA-style re-run)
            if valid_ops and ops:
                with trace_phase("gate", arm, seed, epoch) as gate_span:
                    current_batch_score = sum(1 for p in batch_outcomes.values() if p)
                    accepted, final_playbook, gate_info = gate_minibatch(
                        selected_tasks,
                        current_playbook,
                        candidate_playbook,
                        batch_outcomes,
                        current_batch_score,
                        actor,
                        grader,
                        labels,
                        mock_api
                    )
                    epoch_cost["gate"] += gate_info["gate_cost_usd"]
                    if not accepted:
                        gate_rejection_count += 1
                    gate_span.set_attribute("cost_usd", gate_info["gate_cost_usd"])
                    gate_span.set_attribute("accepted", accepted)
                    
                    if accepted:
                        current_playbook = final_playbook
                        reflect_accepted = True

        # Update cumulative costs
        for k in ["act", "curate", "reflect", "gate"]:
            cumulative_cost[k] += epoch_cost[k]
        cumulative_cost["total"] = sum(cumulative_cost[k] for k in ["act", "curate", "reflect", "gate"])

        # 7. Commit Checkpoint
        cp_dir = arm_dir / f"checkpoint_{epoch}"
        cp_dir.mkdir(exist_ok=True)
        with open(cp_dir / "playbook.md", "w") as f:
            f.write(current_playbook.to_markdown())

        # Measure pollution deterministically via provenance
        pollution_metrics = current_playbook.compute_pollution(labels)

        epoch_record = {
            "epoch": epoch,
            "arm": arm,
            "seed": seed,
            "tasks_run": [t["task_id"] for t in selected_tasks],
            "batch_score": sum(1 for p in batch_outcomes.values() if p),
            "reflect_accepted": reflect_accepted,
            "reflector_rejection_counts": reflection_rejections,
            "gate_rejections": gate_rejection_count,
            "gate_info": gate_info,
            "curator_decisions": curator_decisions,
            "pollution": pollution_metrics,
            "cost_usd": dict(epoch_cost),
            "cumulative_cost_usd": dict(cumulative_cost),
            "playbook_tokens": current_playbook.token_count()
        }
        epoch_logs.append(epoch_record)

        with open(cp_dir / "metadata.json", "w") as f:
            json.dump(epoch_record, f, indent=2)

    return {"arm": arm, "seed": seed, "epoch_logs": epoch_logs, "final_playbook": current_playbook}
