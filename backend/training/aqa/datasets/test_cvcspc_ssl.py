"""Unit tests for the CVCSPC SSL cores (IMG-01) — the 3-term loss (closed-form + directionality)
and the projection head. The phase-matched triplet / _traj2phase / masking / traj_nan tests are
added in Task 3."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402


def test_cvcspc_triplet_loss() -> None:
    """The 3-term loss matches the hand-computed closed form within 1e-6 on L2-normalized triplets."""
    from backend.training.aqa.harness.cvcspc_pretrain import cvcspc_triplet_loss

    torch.manual_seed(0)
    a = F.normalize(torch.randn(4, 8), dim=-1, p=2)
    p = F.normalize(torch.randn(4, 8), dim=-1, p=2)
    n = F.normalize(torch.randn(4, 8), dim=-1, p=2)

    an, pn, nn_ = a.numpy(), p.numpy(), n.numpy()
    d_ap = ((an - pn) ** 2).sum(-1)
    d_an = ((an - nn_) ** 2).sum(-1)
    d_pn = ((pn - nn_) ** 2).sum(-1)
    num = np.exp(-d_ap)
    den = num + np.exp(-d_an) + np.exp(-d_pn)
    expected = float((-np.log(num / (den + 1e-8))).mean())

    got = float(cvcspc_triplet_loss(a, p, n))
    assert got == pytest.approx(expected, abs=1e-6), f"got {got}, expected {expected}"


def test_cvcspc_loss_directional() -> None:
    """The loss rewards anchor-close-to-positive / far-from-negative (catches a sign/term bug)."""
    from backend.training.aqa.harness.cvcspc_pretrain import cvcspc_triplet_loss

    e0 = torch.zeros(4, 8)
    e0[:, 0] = 1.0
    e1 = torch.zeros(4, 8)
    e1[:, 1] = 1.0

    # GOOD: anchor == positive (close), negative orthogonal (far).
    good = float(cvcspc_triplet_loss(e0.clone(), e0.clone(), e1.clone()))
    # BAD: anchor == positive == negative (no separation).
    bad = float(cvcspc_triplet_loss(e0.clone(), e0.clone(), e0.clone()))

    assert good < bad, f"loss not directional: good={good} should be < bad={bad}"


def test_projection_head() -> None:
    """The head outputs unit-norm [B, out_dim], has no BatchNorm, and the dims are configurable."""
    import torch.nn as nn

    from backend.training.aqa.harness.cvcspc_pretrain import CVCSPCProjectionHead

    head = CVCSPCProjectionHead(512, 128, 128)
    out = head(torch.randn(4, 512))
    assert out.shape == (4, 128), f"out shape {tuple(out.shape)}"
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), f"rows not unit-norm: {norms.tolist()}"
    assert not any(isinstance(m, nn.BatchNorm1d) for m in head.modules()), "head must not contain BatchNorm1d"

    head_512 = CVCSPCProjectionHead(512, 512, 512)
    out_512 = head_512(torch.randn(2, 512))
    assert out_512.shape == (2, 512), f"configurable dims failed: {tuple(out_512.shape)}"
