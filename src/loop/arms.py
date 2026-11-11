"""
Task selection policies for the experimental arms:
- A0: Uncurated + gating (uniform random minibatch from train pool)
- A1: Oracle (only learnable tasks using ground-truth labels)
- A2: Numeric curriculum (per-task score s_t, failure adds 1, success subtracts k, drop if > b)
- A3: Curator full (trace-based decision + anchors)
"""
from typing import Dict, Any, List, Set, Optional
import random

class NumericCurriculum:
    def __init__(self, b: int = 3, k: int = 1):
        self.b = b
        self.k = k
        self.scores: Dict[str, int] = {}
        self.dropped_tasks: Set[str] = set()

    def update(self, task_id: str, passed: bool):
        if task_id not in self.scores:
            self.scores[task_id] = 0

        if passed:
            self.scores[task_id] = max(0, self.scores[task_id] - self.k)
        else:
            self.scores[task_id] += 1

        if self.scores[task_id] > self.b:
            self.dropped_tasks.add(task_id)

    def is_available(self, task_id: str) -> bool:
        return task_id not in self.dropped_tasks

def select_minibatch(
    arm: str,
    all_train_tasks: List[Dict[str, Any]],
    batch_size: int,
    rng: random.Random,
    curriculum: Optional[NumericCurriculum] = None,
    curator_skip_until: Optional[Dict[str, int]] = None,
    labels: Optional[Dict[str, Dict[str, Any]]] = None,
    current_epoch: int = 1
) -> List[Dict[str, Any]]:
    """
    Selects up to batch_size tasks according to arm policy.
    """
    if arm == "A1":
        # Oracle: Only ground-truth learnable tasks
        eligible = [
            t for t in all_train_tasks 
            if labels.get(t["task_id"], {}).get("category") == "learnable"
        ]
        sample_size = min(batch_size, len(eligible))
        return rng.sample(eligible, sample_size)

    elif arm == "A2":
        # Numeric curriculum: pool filtered by s_t <= b
        assert curriculum is not None, "Curriculum instance required for A2"
        eligible = [t for t in all_train_tasks if curriculum.is_available(t["task_id"])]
        if not eligible:
            eligible = all_train_tasks # fallback if all dropped
        sample_size = min(batch_size, len(eligible))
        return rng.sample(eligible, sample_size)

    elif arm in ["A3", "A4"]:
        # Curator: filter out tasks whose skip interval has not elapsed
        skip_map = curator_skip_until or {}
        eligible = [t for t in all_train_tasks if skip_map.get(t["task_id"], 0) <= current_epoch]
        if not eligible:
            eligible = all_train_tasks
        sample_size = min(batch_size, len(eligible))
        return rng.sample(eligible, sample_size)

    else:
        # A0 (Uncurated) / A5 / A6: uniform random from full train set
        sample_size = min(batch_size, len(all_train_tasks))
        return rng.sample(all_train_tasks, sample_size)
