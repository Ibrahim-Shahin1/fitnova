"""Unit tests for md_pretrain.py + squat_ssl.split_half_cycles (SQUAT-04-a..f).

Pure-function verification of the reconstructed-from-paper logic BEFORE the
12-24h GPU burn (RESEARCH §15 / "Key insight"): the half-cycle splitter, the
triplet distance-ratio loss arithmetic, the projector L2-norm, the model build,
and the SSL checkpoint schema. Test bodies land alongside their implementations
in Wave 0 (Tasks 2/5/6/7/8).
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the whole file on torch-less environments (Windows orchestrator shell).
torch = pytest.importorskip("torch")

from backend.training.aqa.datasets.squat_ssl import split_half_cycles
from backend.training.aqa.harness.md_pretrain import (
    MDConfig,
    ProjectionHead,
    build_md_model,
    md_triplet_loss,
)


def _arr(*values: int | float) -> np.ndarray:
    """Build a numpy array from a positional spread (terser than np.array(list))."""
    return np.array(values)


# ───────────────────────────── SQUAT-04-a: half-cycle splitter ─────────────────────────────

def test_half_cycle_split() -> None:
    # Two synthetic single-rep parabolas, each paired with the sign that detects its middle.
    # NOTE: plan text mislabels these — -(x**2) is concave-DOWN (global MAX at middle ->
    # argmax); x**2 is concave-UP (global MIN at middle -> argmin). Paired correctly here.
    n = 80
    x = np.linspace(-1.0, 1.0, n)
    cases = [
        (-(x**2), True),   # concave-down: bottom (middle) found by argmax
        (x**2, False),     # concave-up:   bottom (middle) found by argmin
    ]
    for traj, is_argmax in cases:
        descent, ascent = split_half_cycles(traj, frames_per_half=16, bottom_is_argmax=is_argmax)
        assert len(descent) == 16 and len(ascent) == 16, (len(descent), len(ascent))
        bottom = int(descent[-1])
        # Contract: every descent index <= bottom <= every ascent index.
        assert descent.max() <= bottom <= ascent.min(), (descent, bottom, ascent)
        # Detected bottom is near the middle of an 80-sample symmetric parabola.
        assert 35 <= bottom <= 44, bottom
        # Half-cycles span the full clip and are sorted, in-range.
        assert descent[0] == 0 and ascent[-1] == n - 1, (descent[0], ascent[-1])
        assert (np.diff(descent) >= 0).all() and (np.diff(ascent) >= 0).all()
        assert descent.min() >= 0 and ascent.max() < n

    # Flipping the sign moves the detected bottom — proves bottom_is_argmax is wired,
    # not hard-coded (T-04-02 mitigation; the real-data sign is Plan 02's gated probe).
    desc_argmax, _ = split_half_cycles(-(x**2), frames_per_half=16, bottom_is_argmax=True)
    desc_argmin, _ = split_half_cycles(-(x**2), frames_per_half=16, bottom_is_argmax=False)
    assert int(desc_argmax[-1]) != int(desc_argmin[-1])

    # Defensive: a degenerate trajectory raises ValueError, not an opaque crash.
    with pytest.raises(ValueError):
        split_half_cycles(np.array([0.5]), frames_per_half=16)


# ───────────────────────────── SQUAT-04-b/c: triplet distance-ratio loss ────────────────────

def test_triplet_loss_known() -> None:
    pytest.skip("Task 5 — implement md_triplet_loss + hand-computed-value test")


def test_triplet_loss_direction() -> None:
    pytest.skip("Task 5 — implement md_triplet_loss + directionality test")


# ───────────────────────────── SQUAT-04-d: projection head L2-norm ──────────────────────────

def test_projector_l2norm() -> None:
    head = ProjectionHead(512, 512, 128).eval()  # eval() -> BatchNorm1d uses running stats
    x = torch.randn(4, 512)
    with torch.no_grad():
        out = head(x)
    assert out.shape == (4, 128), out.shape
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), norms


# ───────────────────────────── SQUAT-04-f: SSL checkpoint schema ────────────────────────────

def test_ssl_checkpoint_schema() -> None:
    pytest.skip("Task 7 — SSL checkpoint payload schema test")


# ───────────────────────────── SQUAT-04-e: model build (slow) ───────────────────────────────

@pytest.mark.slow
def test_md_model_build() -> None:
    pytest.skip("Task 8 — implement build_md_model + this slow test")
