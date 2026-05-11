"""Smoke test for qevd_extractor.

Asserts the documented per-clip output contract holds against a real MP4
(`C:/Users/tsh_x/Downloads/Good_Squats.mp4`) BEFORE we point the
batch extractor at the QEVD dataset and burn 20 hours of MediaPipe.

This is gated behind the env var ``RUN_SLOW_TESTS=1`` because it spins up
a real PoseLandmarker (~1 s startup) and processes ~12 s of video
(~30 s wall-time).  Run from repo root:

    RUN_SLOW_TESTS=1 pytest backend/training/preprocessing/test_qevd_extractor.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from backend.training.preprocessing.qevd_extractor import (
    STATUS_INTERP,
    STATUS_NO_POSE,
    STATUS_OK,
    extract_clip,
)

# ── Smoke-test fixture path ──────────────────────────────────────────────────
SMOKE_MP4 = Path("C:/Users/tsh_x/Downloads/Good_Squats.mp4")

# Index of `l_knee_flex` in the 22-dim angular feature vector.
# See backend.training.preprocessing.angular_features.ARTICULATION_ANGLES.
L_KNEE_FLEX_IDX = 6
R_KNEE_FLEX_IDX = 7


def _slow_enabled() -> bool:
    return os.getenv("RUN_SLOW_TESTS", "").strip() in {"1", "true", "yes"}


@pytest.mark.skipif(
    not _slow_enabled(),
    reason="set RUN_SLOW_TESTS=1 to run; smoke-tests real MediaPipe pipeline",
)
@pytest.mark.skipif(
    not SMOKE_MP4.exists(),
    reason=f"{SMOKE_MP4} not present (Good_Squats.mp4 fixture missing)",
)
def test_extract_clip_smoke_on_good_squats():
    """End-to-end: real MP4 -> PoseLandmarker -> ClipFeatures contract."""
    feats = extract_clip(SMOKE_MP4)

    # Shape contract
    T, J, C = feats.pose_canon.shape
    assert J == 15, f"expected 15 canonical joints, got {J}"
    assert C == 4,  f"expected 4 pose channels (xyz+vis), got {C}"
    assert T > 50,  f"expected >50 frames in 12+ s video, got {T}"

    # Angle shape contract
    assert feats.angles_raw.shape == (T, 22), feats.angles_raw.shape

    # Status flags only ever take documented values
    unique_statuses = set(np.unique(feats.status_per_frame).tolist())
    allowed = {STATUS_OK, STATUS_INTERP, STATUS_NO_POSE}
    assert unique_statuses.issubset(allowed), (
        f"unexpected status flags: {unique_statuses - allowed}"
    )

    # NaN budget: pose array must be substantially clean
    nan_frac = float(np.isnan(feats.pose_canon).mean())
    assert nan_frac < 0.05, f"NaN fraction {nan_frac:.3f} >= 0.05"

    # No-pose budget: should be small on a 12 s phone-recorded squat
    assert 0.0 <= feats.no_pose_fraction <= 0.10, (
        f"no_pose_fraction {feats.no_pose_fraction:.3f} outside [0, 0.10]"
    )

    # Sanity: knees must bend somewhere in this video
    l_knee_max = float(np.max(feats.angles_raw[:, L_KNEE_FLEX_IDX]))
    r_knee_max = float(np.max(feats.angles_raw[:, R_KNEE_FLEX_IDX]))
    assert max(l_knee_max, r_knee_max) > 0.5, (
        f"max knee flexion {max(l_knee_max, r_knee_max):.3f} rad < 0.5 -- "
        "knees never bent? extractor may be broken"
    )

    # Sanity: native fps in a sane range
    assert 10.0 < feats.fps_native < 240.0, feats.fps_native


@pytest.mark.skipif(
    not _slow_enabled(),
    reason="set RUN_SLOW_TESTS=1 to run",
)
@pytest.mark.skipif(
    not SMOKE_MP4.exists(),
    reason=f"{SMOKE_MP4} not present",
)
def test_extract_clip_save_npz_roundtrip(tmp_path: Path):
    """Save .npz then reload; tensors must match the documented schema."""
    feats = extract_clip(SMOKE_MP4)
    out_path = tmp_path / "smoke.npz"
    feats.save_npz(out_path)

    assert out_path.exists()

    with np.load(out_path) as z:
        keys = set(z.files)
        assert keys == {
            "pose_canon", "angles_raw", "fps_native",
            "status_per_frame", "no_pose_fraction",
        }, keys
        assert z["pose_canon"].dtype == np.float32
        assert z["angles_raw"].dtype == np.float32
        assert z["status_per_frame"].dtype == np.uint8
        # fps_native + no_pose_fraction are 0-d float32
        assert z["fps_native"].shape == ()
        assert z["no_pose_fraction"].shape == ()


# ── Always-on (fast) smoke checks: import surface + dataclass shape ─────────
def test_extract_clip_raises_on_missing_file(tmp_path: Path):
    bogus = tmp_path / "does_not_exist.mp4"
    with pytest.raises(FileNotFoundError):
        extract_clip(bogus)


def test_clip_features_save_npz_roundtrip_synthetic(tmp_path: Path):
    """Feed synthetic arrays through ClipFeatures.save_npz to verify the
    dataclass + npz schema work without MediaPipe."""
    from backend.training.preprocessing.qevd_extractor import ClipFeatures

    T = 30
    feats = ClipFeatures(
        pose_canon=np.zeros((T, 15, 4), dtype=np.float32),
        angles_raw=np.zeros((T, 22), dtype=np.float32),
        fps_native=30.0,
        status_per_frame=np.zeros(T, dtype=np.uint8),
        no_pose_fraction=0.0,
    )
    out_path = tmp_path / "synth.npz"
    feats.save_npz(out_path)
    with np.load(out_path) as z:
        assert z["pose_canon"].shape == (T, 15, 4)
        assert z["angles_raw"].shape == (T, 22)
        assert z["status_per_frame"].shape == (T,)
        assert float(z["fps_native"]) == 30.0
        assert float(z["no_pose_fraction"]) == 0.0
