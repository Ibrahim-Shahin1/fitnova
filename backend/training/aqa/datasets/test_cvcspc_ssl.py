"""Unit tests for the CVCSPC SSL cores (IMG-01) — the 3-term loss (closed-form + directionality)
and the projection head. The phase-matched triplet / _traj2phase / masking / traj_nan tests are
added in Task 3."""

from __future__ import annotations

import json

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


def _make_ssl_fixture(tmp_path, clips: dict[str, list]):
    """Write frames_root/{cid}/frame_*.jpg (one per trajectory point) + traj_root/{cid}.json."""
    from PIL import Image

    frames_root = tmp_path / "frames"
    traj_root = tmp_path / "traj"
    frames_root.mkdir()
    traj_root.mkdir()
    for cid, traj in clips.items():
        d = frames_root / cid
        d.mkdir()
        for i in range(len(traj)):
            arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"frame_{i:06d}.jpg")
        (traj_root / f"{cid}.json").write_text(json.dumps([float(x) for x in traj]))
    return str(frames_root), str(traj_root)


def test_traj2phase() -> None:
    """A monotone ramp maps min->0deg, max->360deg; NaNs are interpolated out."""
    from backend.training.aqa.datasets.cvcspc_ssl import _traj2phase

    ph = _traj2phase(np.array([0, 1, 2, 3, 4], dtype=float))
    assert np.allclose(ph, [0, 90, 180, 270, 360], atol=1e-6)

    ph2 = _traj2phase(np.array([0, 1, np.nan, 3, 4], dtype=float))
    assert not np.isnan(ph2).any()
    assert np.allclose(ph2, [0, 90, 180, 270, 360], atol=1e-6)


def test_traj2phase_degenerate() -> None:
    """An all-NaN or constant trajectory raises (the divide-by-zero / flat-phase guard)."""
    from backend.training.aqa.datasets.cvcspc_ssl import _traj2phase

    with pytest.raises(ValueError):
        _traj2phase(np.array([np.nan, np.nan, np.nan], dtype=float))
    with pytest.raises(ValueError):
        _traj2phase(np.array([5, 5, 5, 5], dtype=float))


def test_mask(tmp_path) -> None:
    """_mask zeroes the top mask_amt fraction with prob mask_prob; prob=0 leaves the image intact."""
    from backend.training.aqa.datasets.cvcspc_ssl import ShallowSquatSSLDataset

    frames_root, traj_root = _make_ssl_fixture(
        tmp_path, {"c0": [0, 1, 2, 3, 4, 5, 6, 7], "c1": [7, 6, 5, 4, 3, 2, 1, 0]}
    )
    ds = ShallowSquatSSLDataset(
        frames_root=frames_root, trajectories_root=traj_root,
        mask_prob=1.0, mask_amt_lo=0.5, mask_amt_hi=0.5,
    )
    out = ds._mask(torch.ones(3, 224, 224))
    assert torch.all(out[:, :112, :] == 0), "top 112 rows should be masked"
    assert torch.all(out[:, 112:, :] == 1), "bottom rows should be untouched"

    ds.mask_prob = 0.0
    out2 = ds._mask(torch.ones(3, 224, 224))
    assert torch.all(out2 == 1), "mask_prob=0 must leave the image unchanged"


def test_traj_nan_exclusion(tmp_path) -> None:
    """Clips listed in traj_nan.json are excluded from _clip_ids at construction."""
    from backend.training.aqa.datasets.cvcspc_ssl import ShallowSquatSSLDataset

    frames_root, traj_root = _make_ssl_fixture(
        tmp_path, {"c0": [0, 1, 2, 3], "c1": [3, 2, 1, 0], "c2": [0, 2, 4, 6]}
    )
    nan_path = tmp_path / "traj_nan.json"
    nan_path.write_text(json.dumps(["c2.json"]))

    ds = ShallowSquatSSLDataset(
        frames_root=frames_root, trajectories_root=traj_root, traj_nan_path=str(nan_path),
    )
    assert "c2" not in ds._clip_ids
    assert len(ds._clip_ids) == 2


def test_phase_matched_triplet(tmp_path) -> None:
    """anchor@v0 + positive@v1 at a shared phase; negative@v1 at least phase_gap away."""
    import random as _random

    from backend.training.aqa.datasets.cvcspc_ssl import ShallowSquatSSLDataset

    n = 24
    ramp = list(range(n))
    frames_root, traj_root = _make_ssl_fixture(tmp_path, {"c0": ramp, "c1": ramp})
    ds = ShallowSquatSSLDataset(
        frames_root=frames_root, trajectories_root=traj_root,
        ssl_contrastive_phase_gap=30.0, mask_prob=0.0,
    )
    spacing = 360.0 / (n - 1)
    _random.seed(0)
    for _ in range(10):
        sel = ds._select_triplet(0)
        phase0, phase1, p = sel["phase0"], sel["phase1"], sel["p_anchor"]
        a, pi, ni = sel["anchor_idx"], sel["positive_idx"], sel["negative_idx"]
        assert 0 <= a < len(phase0)
        assert 0 <= pi < len(phase1) and 0 <= ni < len(phase1)
        assert abs(phase0[a] - p) <= spacing + 1e-6
        assert abs(phase1[pi] - p) <= spacing + 1e-6
        assert abs(phase1[ni] - p) >= ds.phase_gap - 1e-6

    out = ds[0]
    assert set(out.keys()) == {"anchor", "positive", "negative"}
    for k in out:
        assert out[k].shape == (3, 224, 224)
        assert out[k].dtype == torch.float32
