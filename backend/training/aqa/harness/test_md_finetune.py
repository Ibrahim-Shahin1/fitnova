"""Unit test for md_finetune.build_finetune_model (SQUAT-05-a).

Verifies the fine-tune model loads the MD backbone (NOT Kinetics) and attaches a
fresh ``Dropout(0.2) + Linear(512, 2)`` head. Body lands in Wave 0 (Task 9).
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the whole file on torch-less environments (Windows orchestrator shell).
torch = pytest.importorskip("torch")

from backend.training.aqa.harness.md_finetune import FinetuneConfig, build_finetune_model


def _arr(*values: int | float) -> np.ndarray:
    """Build a numpy array from a positional spread (terser than np.array(list))."""
    return np.array(values)


# ───────────────────────────── SQUAT-05-a: fine-tune model build (slow) ─────────────────────

@pytest.mark.slow
def test_finetune_model_build() -> None:
    pytest.skip("Task 9 — implement build_finetune_model + this slow test")
