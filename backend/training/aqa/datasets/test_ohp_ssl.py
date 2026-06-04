"""Unit tests for the OHP SSL dataset: trajectory loader and half-cycle splitter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# ──────────────────────────────────────────────────────────────────────────────
# BBox _load_trajectory
# ──────────────────────────────────────────────────────────────────────────────


def test_trajectory_load(tmp_path: Path) -> None:
    """BBox JSON with 2 empty region-0 frames is interpolated correctly.

    Synthetic 5-frame BBox JSON:
      frame 0: region_0 = [[10, 100, 200, 200, 90]]  → y_center = (100+200)/2 = 150.0
      frame 1: region_0 = []                           → NaN (interpolated)
      frame 2: region_0 = [[10,  60, 200, 160, 88]]  → y_center = (60+160)/2  = 110.0
      frame 3: region_0 = []                           → NaN (interpolated)
      frame 4: region_0 = [[10, 140, 200, 260, 91]]  → y_center = (140+260)/2 = 200.0

    y_center values: 150, ?, 110, ?, 200
    Linear interpolation on indices [0,2,4] with values [150,110,200]:
      frame 1: interp(1, [0,2,4], [150,110,200]) = 150 + (1-0)/(2-0) * (110-150) = 150 - 20 = 130
      frame 3: interp(3, [0,2,4], [150,110,200]) = 110 + (3-2)/(4-2) * (200-110) = 110 + 45 = 155

    Both interpolated values must lie between their neighbors (monotone interpolation check).
    """
    from backend.training.aqa.datasets.ohp_ssl import OHPSSLDataset

    frames = [
        [[[10, 100, 200, 200, 90]], [], []],  # frame 0: region_0 present
        [[], [], []],                          # frame 1: region_0 empty → NaN
        [[[10, 60, 200, 160, 88]], [], []],   # frame 2: region_0 present
        [[], [], []],                          # frame 3: region_0 empty → NaN
        [[[10, 140, 200, 260, 91]], [], []],  # frame 4: region_0 present
    ]
    traj_path = tmp_path / "00001_0.json"
    traj_path.write_text(json.dumps(frames))

    # Instantiate OHPSSLDataset without __init__ to avoid needing real video/traj dirs.
    ds = OHPSSLDataset.__new__(OHPSSLDataset)
    ds._traj_paths = {"00001_0": traj_path}

    y = ds._load_trajectory("00001_0")

    # Shape and dtype
    assert y.shape == (5,), f"Expected shape (5,), got {y.shape}"
    assert y.ndim == 1, "y must be 1-D"

    # No residual NaN after interpolation
    assert not np.isnan(y).any(), f"Residual NaN in y: {y}"

    # y_center = (y1+y2)/2 per ReadMe formula
    assert y[0] == pytest.approx(150.0), f"y[0] expected 150.0, got {y[0]}"
    assert y[2] == pytest.approx(110.0), f"y[2] expected 110.0, got {y[2]}"
    assert y[4] == pytest.approx(200.0), f"y[4] expected 200.0, got {y[4]}"

    # Interpolated frames must lie between their neighbors (linear interpolation property).
    # Frame 1 is between frame 0 (150) and frame 2 (110): 110 <= y[1] <= 150
    assert y[2] <= y[1] <= y[0] or y[0] <= y[1] <= y[2], (
        f"y[1]={y[1]} not between neighbors y[0]={y[0]} and y[2]={y[2]}"
    )
    # Frame 3 is between frame 2 (110) and frame 4 (200): 110 <= y[3] <= 200
    assert y[2] <= y[3] <= y[4] or y[4] <= y[3] <= y[2], (
        f"y[3]={y[3]} not between neighbors y[2]={y[2]} and y[4]={y[4]}"
    )


def test_trajectory_load_all_valid(tmp_path: Path) -> None:
    """_load_trajectory returns correct y_centers when all frames have region_0."""
    from backend.training.aqa.datasets.ohp_ssl import OHPSSLDataset

    frames = [
        [[[0, 10, 100, 50, 95]], [], []],   # y_center = (10+50)/2 = 30.0
        [[[0, 20, 100, 60, 90]], [], []],   # y_center = (20+60)/2 = 40.0
        [[[0, 30, 100, 70, 88]], [], []],   # y_center = (30+70)/2 = 50.0
    ]
    path = tmp_path / "c0.json"
    path.write_text(json.dumps(frames))

    ds = OHPSSLDataset.__new__(OHPSSLDataset)
    ds._traj_paths = {"c0": path}
    y = ds._load_trajectory("c0")

    assert y.shape == (3,)
    assert not np.isnan(y).any()
    assert y[0] == pytest.approx(30.0)
    assert y[1] == pytest.approx(40.0)
    assert y[2] == pytest.approx(50.0)


def test_trajectory_load_raises_on_too_few_valid(tmp_path: Path) -> None:
    """_load_trajectory raises ValueError when < 2 non-NaN samples exist."""
    from backend.training.aqa.datasets.ohp_ssl import OHPSSLDataset

    # 3 frames, only 1 with region_0
    frames = [
        [[[0, 10, 100, 50, 95]], [], []],
        [[], [], []],
        [[], [], []],
    ]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(frames))

    ds = OHPSSLDataset.__new__(OHPSSLDataset)
    ds._traj_paths = {"bad": path}

    with pytest.raises(ValueError, match="<2 non-NaN"):
        ds._load_trajectory("bad")


# ──────────────────────────────────────────────────────────────────────────────
# split_half_cycles argMIN guard on a synthetic V-trajectory
# ──────────────────────────────────────────────────────────────────────────────


def test_split_half_cycles_ohp() -> None:
    """split_half_cycles with bottom_is_argmax=False correctly splits a V-trajectory.

    OHP physical model:
      - y_center DECREASES from shoulder (start) to overhead (lowest y pixel = argMIN mid-rep).
      - y_center INCREASES back to shoulder (end).
      - Turning point = overhead = argMIN of y_center.
      - Both Squat and OHP use bottom_is_argmax=False (argMIN) for different physical reasons:
        * Squat: deepest position of the squat is the lowest y pixel.
        * OHP: barbell at overhead is highest in the image = lowest y value.

    Synthetic V-shape: (np.linspace(-1,1,80)**2)*100+50
      - Parabola: minimum at index ~39-40 (the overhead position mid-rep).
      - argmin of this trajectory is in [35, 44] (middle third).
      - argmax is at index 0 or 79 (the endpoints = shoulder position).

    Acceptance criteria:
      - descent_idx[-1] <= argmin <= ascent_idx[0]   (half-cycle contract)
      - argmin is NOT 0 or 79 (truly in the middle, not an endpoint)
      - Both arrays length 16, sorted, within [0, 80)
      - With bottom_is_argmax=True, the split degenerates to an endpoint (argMAX sign wrong for OHP)
    """
    import scipy.ndimage
    from backend.training.aqa.datasets.ohp_ssl import split_half_cycles

    # V-shaped trajectory: minimum at index ~39-40 (mid-rep overhead position)
    traj = (np.linspace(-1, 1, 80) ** 2) * 100 + 50

    descent_idx, ascent_idx = split_half_cycles(
        traj, frames_per_half=16, bottom_is_argmax=False,
    )

    # argmin on the smoothed trajectory (mirrors what split_half_cycles uses internally)
    sm = scipy.ndimage.gaussian_filter1d(traj, sigma=2.0)
    argmin = int(np.argmin(sm))

    # The minimum must be in the middle (not an endpoint)
    assert 10 < argmin < 69, (
        f"argmin={argmin} should be in the middle of [0,79], not at an endpoint"
    )

    # Half-cycle contract: descent ends at or before argmin; ascent starts at or after argmin
    assert descent_idx[-1] <= argmin, (
        f"descent_idx[-1]={descent_idx[-1]} must be <= argmin={argmin}"
    )
    assert ascent_idx[0] >= argmin, (
        f"ascent_idx[0]={ascent_idx[0]} must be >= argmin={argmin}"
    )

    # Both arrays must have length 16
    assert len(descent_idx) == 16, f"descent_idx length: {len(descent_idx)}"
    assert len(ascent_idx) == 16,  f"ascent_idx length: {len(ascent_idx)}"

    # All indices within [0, 80)
    assert (descent_idx >= 0).all() and (descent_idx < 80).all(), (
        f"descent_idx out of range [0,80): {descent_idx}"
    )
    assert (ascent_idx >= 0).all() and (ascent_idx < 80).all(), (
        f"ascent_idx out of range [0,80): {ascent_idx}"
    )

    # Arrays are sorted (uniform_sample_indices returns sorted indices)
    assert (np.diff(descent_idx) >= 0).all(), "descent_idx not sorted"
    assert (np.diff(ascent_idx) >= 0).all(),  "ascent_idx not sorted"

    # argMAX sign check: with bottom_is_argmax=True, the split lands at an endpoint
    # (argmax of a V-parabola is at index 0 or 79), degenerating one half-cycle.
    # This proves the sign param is wired and argMIN is the correct OHP choice.
    argmax = int(np.argmax(sm))
    assert argmax == 0 or argmax == 79, (
        f"argmax of V-trajectory should be at 0 or 79, got {argmax} — test assumption broken"
    )
    # With argMAX: either descent or ascent covers the full range minus 1 frame
    descent_wrong, ascent_wrong = split_half_cycles(
        traj, frames_per_half=16, bottom_is_argmax=True,
    )
    # The "wrong" argmax is at 0 or 79; one half spans 1 frame (degenerate)
    # descent_wrong samples from [0..argmax+1], ascent_wrong from [argmax..len]
    # If argmax=0: descent_wrong has num_frames=1 → uniform_sample_indices fills with index 0
    # If argmax=79: ascent_wrong has num_frames=1 → fills with index 79
    if argmax == 0:
        assert np.all(descent_wrong == 0), (
            f"With argMAX=0, descent_wrong should all be index 0, got: {descent_wrong}"
        )
    else:  # argmax == 79
        assert np.all(ascent_wrong == 79), (
            f"With argMAX=79, ascent_wrong should all be index 79, got: {ascent_wrong}"
        )
