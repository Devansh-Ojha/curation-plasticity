"""
Numeric Curriculum sensitivity check and early drop verification (b=2, k=1).
Verifies:
1. Tasks drop when score exceeds threshold b.
2. Sensitivity check comparing b in [1, 2, 3].
3. Guard checking drops happen early enough to reduce later noise-task runs.
"""
import pytest
from src.loop.arms import NumericCurriculum

def test_numeric_curriculum_b2_k1_behavior():
    # Documented choice: b=2, k=1
    curr = NumericCurriculum(b=2, k=1)
    
    # Run 1: Failure -> score = 1
    curr.update("t_noise", passed=False)
    assert curr.scores["t_noise"] == 1
    assert curr.is_available("t_noise") is True

    # Run 2: Failure -> score = 2
    curr.update("t_noise", passed=False)
    assert curr.scores["t_noise"] == 2
    assert curr.is_available("t_noise") is True

    # Run 3: Failure -> score = 3 > b(2) -> Dropped
    curr.update("t_noise", passed=False)
    assert curr.scores["t_noise"] == 3
    assert curr.is_available("t_noise") is False

def test_numeric_curriculum_sensitivity_sweep():
    # Sensitivity check across b=1, 2, 3
    drops_by_b = {}
    for b_val in [1, 2, 3]:
        curr = NumericCurriculum(b=b_val, k=1)
        # Simulate 6 consecutive failures
        runs_until_drop = 0
        for i in range(1, 10):
            curr.update("t_noise", passed=False)
            if not curr.is_available("t_noise"):
                runs_until_drop = i
                break
        drops_by_b[b_val] = runs_until_drop

    # b=1 drops after 2 runs (score=2 > 1)
    # b=2 drops after 3 runs (score=3 > 2)
    # b=3 drops after 4 runs (score=4 > 3)
    assert drops_by_b[1] == 2
    assert drops_by_b[2] == 3
    assert drops_by_b[3] == 4

def test_recovery_on_success():
    curr = NumericCurriculum(b=2, k=1)
    curr.update("t_learnable", passed=False) # score = 1
    curr.update("t_learnable", passed=True)  # score = max(0, 1 - 1) = 0
    assert curr.scores["t_learnable"] == 0
    assert curr.is_available("t_learnable") is True
