"""Unit tests for md_pretrain.py + squat_ssl.split_half_cycles (SQUAT-04-a..f).

Pure-function verification of the reconstructed-from-paper logic BEFORE the
12-24h GPU burn (RESEARCH §15 / "Key insight"): the half-cycle splitter, the
triplet distance-ratio loss arithmetic, the projector L2-norm, the model build,
and the SSL checkpoint schema. Test bodies land alongside their implementations
in Wave 0 (Tasks 2/5/6/7/8).
"""

from __future__ import annotations

import math

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
    # Known-value: phi_a == phi_p (one-hot unit vector), phi_n antipodal (-phi_a).
    #   d_ap = ||a-p||^2 = 0                  -> num = exp(0) = 1
    #   d_an = ||[1,..]-[-1,..]||^2 = ||[2,..]||^2 = 4 (squared) or sqrt(4)=2 (non-squared)
    #   d_pn = d_an  (since p == a)
    #   loss = -log( num / (num + exp(-d_an) [+ exp(-d_pn) if 3-term]) )
    # The expected value is computed from this closed form via `math` — an INDEPENDENT
    # code path from the torch implementation under test (not a tautology).
    phi_a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    phi_p = phi_a.clone()
    phi_n = -phi_a
    for squared in (True, False):
        d_an = 4.0 if squared else 2.0
        for three_term in (True, False):
            num = 1.0
            den = num + math.exp(-d_an) + (math.exp(-d_an) if three_term else 0.0)
            expected = -math.log(num / den)
            got = float(md_triplet_loss(phi_a, phi_p, phi_n, squared=squared, three_term=three_term))
            assert abs(got - expected) < 1e-5, (squared, three_term, got, expected)

    # Degenerate exact anchor: all three identical -> all distances 0 -> num=1,
    # den=2 (2-term) / 3 (3-term) -> loss = log(2) ~ 0.6931 / log(3) ~ 1.0986.
    phi = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert abs(float(md_triplet_loss(phi, phi, phi, three_term=False)) - math.log(2.0)) < 1e-5
    assert abs(float(md_triplet_loss(phi, phi, phi, three_term=True)) - math.log(3.0)) < 1e-5


def test_triplet_loss_direction() -> None:
    # The loss must be strictly lower when anchor≈positive and far-from-negative than
    # when positive/negative are swapped — proving it pulls positives together and
    # pushes negatives apart (T-04-01 directionality guarantee).
    torch.manual_seed(0)
    d = 128
    nrm = torch.nn.functional.normalize
    anchor = nrm(torch.randn(4, d), dim=-1)
    pos_close = nrm(anchor + 0.01 * torch.randn(4, d), dim=-1)
    neg_far = nrm(-anchor + 0.01 * torch.randn(4, d), dim=-1)
    loss_good = md_triplet_loss(anchor, pos_close, neg_far)
    loss_bad = md_triplet_loss(anchor, neg_far, pos_close)  # positive/negative swapped
    assert loss_good < loss_bad, (float(loss_good), float(loss_bad))


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
    pytest.importorskip("torchvision")  # downloads ~120 MB Kinetics weights on first call
    backbone, projector = build_md_model()
    assert isinstance(backbone.fc, torch.nn.Identity)
    x = torch.zeros(2, 3, 16, 112, 112)  # 16 frames = one half-cycle (NOT the 32-frame clip)
    with torch.no_grad():
        feat = backbone(x)
    assert feat.shape == (2, 512), feat.shape
    proj_out = projector(feat)
    assert proj_out.shape == (2, 128), proj_out.shape
    assert torch.allclose(proj_out.norm(dim=-1), torch.ones(2), atol=1e-5), proj_out.norm(dim=-1)
