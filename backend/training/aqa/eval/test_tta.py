"""Unit tests for tta.py (SQUAT-05-c).

``select_tta_recipe`` is pure-numpy and is tested torch-free (the module imports
without torch via the TYPE_CHECKING guard in tta.py). ``tta_forward`` runs a
model, so its test imports torch lazily via ``pytest.importorskip`` INSIDE the
test, not at module top. Bodies land in Wave 0 (Task 11).
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.training.aqa.eval.tta import select_tta_recipe, tta_forward


def _arr(*values: int | float) -> np.ndarray:
    """Build a numpy array from a positional spread (terser than np.array(list))."""
    return np.array(values)


# ───────────────────────────── SQUAT-05-c: val-tuned recipe selection (torch-free) ──────────

def test_recipe_selection() -> None:
    pytest.skip("Task 11 — implement select_tta_recipe + this torch-free test")


# ───────────────────────────── tta_forward (needs torch) ───────────────────────────────────

def test_tta_forward() -> None:
    pytest.importorskip("torch")  # torch only needed here, not at module top
    pytest.skip("Task 11 — implement tta_forward + this test")
