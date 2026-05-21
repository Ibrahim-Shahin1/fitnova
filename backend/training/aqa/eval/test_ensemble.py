"""Unit test for ensemble.aggregate_sigmoid_mean (SQUAT-05-b).

Pure numpy — no torch import. Verifies the mean-of-sigmoids aggregation produces
ensemble scores strictly in [0, 1] (so they satisfy pr_auc_per_error's assertion).
Body lands in Wave 0 (Task 10).
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.training.aqa.eval.ensemble import aggregate_sigmoid_mean


def _arr(*values: int | float) -> np.ndarray:
    """Build a numpy array from a positional spread (terser than np.array(list))."""
    return np.array(values)


# ───────────────────────────── SQUAT-05-b: mean-of-sigmoids aggregation ─────────────────────

def test_sigmoid_mean() -> None:
    pytest.skip("Task 10 — implement aggregate_sigmoid_mean + this test")
