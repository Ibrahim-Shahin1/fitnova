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
    # All-zero logits -> sigmoid(0)=0.5 -> ensemble 0.5 (hand-computed anchor).
    z = np.zeros((5, 2))
    out = aggregate_sigmoid_mean([z, z, z])
    assert out.shape == (5, 2)
    assert np.allclose(out, 0.5, atol=1e-9), out

    # Saturating logits: KIE +8 -> ~1, KFE -8 -> ~0.
    clip = _arr(8.0, -8.0).reshape(1, 2)
    ens = aggregate_sigmoid_mean([clip, clip, clip])
    assert ens.shape == (1, 2)
    assert ens[0, 0] > 0.99 and ens[0, 1] < 0.01, ens

    # Random logits: output strictly in [0, 1] (the pr_auc_per_error contract).
    rng = np.random.default_rng(0)
    rand = [rng.normal(0.0, 5.0, (7, 2)) for _ in range(3)]
    e = aggregate_sigmoid_mean(rand)
    assert e.min() >= 0.0 and e.max() <= 1.0


def test_sigmoid_mean_raises() -> None:
    with pytest.raises(ValueError):
        aggregate_sigmoid_mean([])
    with pytest.raises(ValueError):
        aggregate_sigmoid_mean([np.zeros((5, 2)), np.zeros((4, 2))])  # shape mismatch
