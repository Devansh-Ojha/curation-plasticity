"""
Playbook artifact manager:
- Stores structured rules: rule_id, content, created_epoch, source_task_ids.
- Enforces token/word budget cap (default 2,000 tokens).
- Harness (not LLM) records provenance: source_task_ids = failed tasks passed to that reflection.
- Computes deterministic pollution metrics based on ground truth labels without LLM judge.
"""
from typing import Dict, Any, List, Optional
import copy

class Playbook:
    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        # Mapping: rule_id -> {"id": str, "content": str, "created_epoch": int, "source_task_ids": List[str]}
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.next_rule_num = 1

    def clone(self) -> 'Playbook':
        p = Playbook(max_tokens=self.max_tokens)
        p.rules = copy.deepcopy(self.rules)
        p.next_rule_num = self.next_rule_num
        return p

    def apply_ops(
        self,
        ops: List[Dict[str, Any]],
        epoch: int,
        source_task_ids: List[str]
    ) -> bool:
        """
        Applies structured edit ops:
        ops = [
            {"op": "add", "content": "..."},
            {"op": "modify", "rule_id": "rule_001", "content": "..."},
            {"op": "delete", "rule_id": "rule_002"}
        ]
        Harness enforces that source_task_ids are attached to any added/modified rule.
        Returns True if successful, False if invalid or exceeds budget.
        """
        candidate = self.clone()

        for op in ops:
            action = op.get("op")
            if action == "add":
                rule_id = f"rule_{candidate.next_rule_num:03d}"
                candidate.next_rule_num += 1
                candidate.rules[rule_id] = {
                    "id": rule_id,
                    "content": op.get("content", "").strip(),
                    "created_epoch": epoch,
                    "source_task_ids": list(source_task_ids) # Recorded by harness
                }
            elif action == "modify":
                r_id = op.get("rule_id")
                if r_id in candidate.rules:
                    candidate.rules[r_id]["content"] = op.get("content", "").strip()
                    # Append new source task IDs
                    for s_id in source_task_ids:
                        if s_id not in candidate.rules[r_id]["source_task_ids"]:
                            candidate.rules[r_id]["source_task_ids"].append(s_id)
            elif action == "delete":
                r_id = op.get("rule_id")
                if r_id in candidate.rules:
                    del candidate.rules[r_id]

        # Check token cap (approx 4 chars per token)
        text_repr = candidate.to_markdown()
        approx_tokens = len(text_repr) // 4
        if approx_tokens > self.max_tokens:
            return False # Exceeded budget

        # Commit update
        self.rules = candidate.rules
        self.next_rule_num = candidate.next_rule_num
        return True

    def to_markdown(self) -> str:
        """Formats the playbook as markdown for the actor agent."""
        if not self.rules:
            return "# Agent Execution Playbook\nNo rules yet."
        
        lines = ["# Agent Execution Playbook\n"]
        for r_id, r in sorted(self.rules.items()):
            lines.append(f"## [{r_id}]\n{r['content']}\n")
        return "\n".join(lines)

    def token_count(self) -> int:
        return len(self.to_markdown()) // 4

    def compute_pollution(self, labels: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Computes deterministic pollution:
        A rule is polluted if >=50% of its source failed tasks are impossible or flaky.
        Strict pollution: 100% of source tasks are noise.
        """
        if not self.rules:
            return {"pollution_rate": 0.0, "strict_pollution_rate": 0.0, "total_rules": 0}

        noise_rules = 0
        strict_noise_rules = 0

        for r in self.rules.values():
            sources = r.get("source_task_ids", [])
            if not sources:
                continue
            noise_sources = sum(
                1 for s in sources 
                if labels.get(s, {}).get("category") in ["impossible", "flaky_judge", "flaky_ambiguous"]
            )
            frac = noise_sources / len(sources)
            if frac >= 0.5:
                noise_rules += 1
            if frac == 1.0:
                strict_noise_rules += 1

        total = len(self.rules)
        return {
            "pollution_rate": noise_rules / total if total > 0 else 0.0,
            "strict_pollution_rate": strict_noise_rules / total if total > 0 else 0.0,
            "total_rules": total
        }
