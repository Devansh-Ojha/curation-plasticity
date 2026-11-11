"""
Deterministic Grader:
- ZERO randomness for 'easy' and 'learnable' categories.
  Identical final_answer + expected target yields identical verdict deterministically.
- 'impossible' category deterministically fails (target state unattainable).
- ONLY 'flaky_judge' and 'flaky_ambiguous' categories may randomize, strictly via a seeded RNG.
"""
from typing import Dict, Any, Tuple
import random
import re

class Grader:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def grade(
        self,
        task: Dict[str, Any],
        ground_truth_category: str,
        execution_trace: Dict[str, Any],
        final_answer: Any
    ) -> Tuple[bool, str]:
        expected = task.get("expected", {})

        # -------------------------------------------------------------
        # 1. Easy & Learnable: Deterministic
        # -------------------------------------------------------------
        if ground_truth_category in ["easy", "learnable"]:
            passed = self._check_deterministic(expected, final_answer)
            if passed:
                return True, "Passed: Validation check succeeded."
            return False, "Failed: Validation check failed. Output did not match expected target."

        # -------------------------------------------------------------
        # 2. Impossible: Deterministically Fails with identical wording
        # -------------------------------------------------------------
        if ground_truth_category == "impossible":
            return False, "Failed: Validation check failed. Output did not match expected target."

        # -------------------------------------------------------------
        # 3. Flaky Judge: Flips verdict with p=0.4 with identical wording
        # -------------------------------------------------------------
        if ground_truth_category == "flaky_judge":
            passed = self._check_deterministic(expected, final_answer)
            if passed:
                flip = self.rng.random() < 0.4
                if flip:
                    return False, "Failed: Validation check failed. Output did not match expected target."
                return True, "Passed: Validation check succeeded."
            return False, "Failed: Validation check failed. Output did not match expected target."

        # -------------------------------------------------------------
        # 4. Flaky Ambiguous: Picks canonical interpretation with identical wording
        # -------------------------------------------------------------
        if ground_truth_category == "flaky_ambiguous":
            reading_a = expected.get("reading_a")
            reading_b = expected.get("reading_b")
            canonical = self.rng.choice([reading_a, reading_b])
            if str(final_answer).strip().lower() == str(canonical).strip().lower():
                return True, "Passed: Validation check succeeded."
            return False, "Failed: Validation check failed. Output did not match expected target."

        # Default fallback
        passed = self._check_deterministic(expected, final_answer)
        return passed, "Passed: Validation check succeeded." if passed else "Failed: Validation check failed. Output did not match expected target."

    def _check_deterministic(self, expected: Dict[str, Any], final_answer: Any) -> bool:
        if not expected:
            return bool(final_answer)

        check_type = expected.get("type")
        target_val = expected.get("target_val")

        if check_type == "exact":
            answer = str(final_answer).strip().lower()
            target = str(target_val).strip().lower()
            if answer == target:
                return True
            # Tasks that ask to "return" a value do not always require bare
            # output. Accept the exact target as a standalone value in a
            # natural-language answer, while rejecting longer IDs/words.
            return bool(target and re.search(
                rf"(?<![a-z0-9_]){re.escape(target)}(?![a-z0-9_])", answer
            ))
        elif check_type == "contains":
            return str(target_val).lower() in str(final_answer).lower()
        elif check_type == "count":
            try:
                return int(final_answer) == int(target_val)
            except (ValueError, TypeError):
                matches = re.findall(r"(?<![A-Za-z0-9])-?\d+(?![A-Za-z0-9])", str(final_answer))
                return len(matches) == 1 and int(matches[0]) == int(target_val)
        elif check_type == "key_val":
            key = expected.get("key")
            if isinstance(final_answer, dict):
                return str(final_answer.get(key)).lower() == str(target_val).lower()
            return False

        return str(target_val).lower() in str(final_answer).lower()
