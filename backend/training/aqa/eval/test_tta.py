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
    # Two candidate recipes on the same N clips: A separates the classes, B is noise.
    # select_tta_recipe must return A (higher val macro-F1). Pure numpy — no torch.
    n = 20
    labels = np.zeros((n, 2), dtype=int)
    labels[: n // 2, 0] = 1            # first half KIE-positive
    labels[:, 1] = np.arange(n) % 2    # alternating KFE
    logits_a = np.where(labels == 1, 5.0, -5.0).astype(float)  # well-separated -> F1 ~1.0
    rng = np.random.default_rng(0)
    logits_b = rng.normal(0.0, 0.1, (n, 2))                    # noise -> low F1
    recipes = {("flip",): logits_a, ("flip", "spatial_5crop"): logits_b}
    assert select_tta_recipe(recipes, labels) == ("flip",)

    # Tiebreak: identical logits -> the shorter/first recipe wins deterministically.
    tie = {("flip", "temporal_jitter"): logits_a, ("flip",): logits_a}
    assert select_tta_recipe(tie, labels) == ("flip",)


# ───────────────────────────── tta_forward (needs torch) ───────────────────────────────────

def test_tta_forward() -> None:
    torch = pytest.importorskip("torch")  # torch only needed here, not at module top
    import torch.nn as nn

    class _Dummy(nn.Module):  # [B,3,T,H,W] -> [B,2]
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Linear(3, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(x.mean(dim=(2, 3, 4)))  # global pool over T,H,W -> [B,3]

    model = _Dummy().eval()
    clip = torch.zeros(3, 32, 112, 112)
    out = tta_forward(
        model, clip, ["temporal_jitter", "spatial_5crop", "flip"], torch.device("cpu")
    )
    assert out.shape == (2,), out.shape
