"""
Tests that all 22 angular features are Y-rotation invariant.

This is the GA gate prerequisite from the Phase 4 plan. Any Y-axis rotation
of the canonical skeleton must produce identical feature vectors, so that
`_augment_joints3d` (which applies random Y-rotations) can rotate the
*corrupted* canonical joints without changing the quality label.
"""

import numpy as np
import pytest

from backend.training.preprocessing.angular_features import (
    N_ANGULAR,
    compute_frame_angles,
)
from backend.training.preprocessing.joint_mapping import N_CANONICAL


def _y_rotation_matrix(theta_rad: float) -> np.ndarray:
    """Standard 3x3 rotation matrix around Y (up) axis."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32
    )


def _random_canonical_pose(rng: np.random.Generator) -> np.ndarray:
    """
    Build a plausible standing canonical skeleton (15 joints, 3D metres).
    Coordinates roughly match post-normalize_skeleton convention:
    pelvis at origin, subject facing forward but NOT azimuthally aligned.
    """
    # Base canonical positions (rough standing anatomy, Y-up, facing +Z)
    base = np.zeros((N_CANONICAL, 3), dtype=np.float32)
    # Shoulders
    base[0] = [-0.20, 0.55, 0.0]   # L_SH
    base[1] = [ 0.20, 0.55, 0.0]   # R_SH
    # Elbows
    base[2] = [-0.25, 0.25, 0.02]  # L_EL
    base[3] = [ 0.25, 0.25, 0.02]  # R_EL
    # Wrists
    base[4] = [-0.27, -0.05, 0.04] # L_WR
    base[5] = [ 0.27, -0.05, 0.04] # R_WR
    # Hips
    base[6] = [-0.10, 0.0, 0.0]    # L_HI
    base[7] = [ 0.10, 0.0, 0.0]    # R_HI
    # Knees
    base[8] = [-0.11, -0.45, 0.05] # L_KN
    base[9] = [ 0.11, -0.45, 0.05] # R_KN
    # Ankles
    base[10] = [-0.11, -0.90, 0.0] # L_AN
    base[11] = [ 0.11, -0.90, 0.0] # R_AN
    # Pelvis / neck / spine_mid
    base[12] = [ 0.0, 0.0, 0.0]    # PELV
    base[13] = [ 0.0, 0.60, 0.0]   # NECK
    base[14] = [ 0.0, 0.30, 0.0]   # SP

    # Add small per-subject jitter to simulate 20 different people
    jitter = rng.normal(0.0, 0.03, size=base.shape).astype(np.float32)
    return base + jitter


# Match plan spec: 20 subjects × 3 frames × 4 rotations
SUBJECTS = 20
FRAMES_PER_SUBJECT = 3
ROTATIONS_DEG = [30.0, 60.0, 90.0, 180.0]
TOL = 1e-4  # arccos precision limit on float32


@pytest.mark.parametrize("theta_deg", ROTATIONS_DEG)
def test_all_22_features_invariant_under_y_rotation(theta_deg: float) -> None:
    rng = np.random.default_rng(seed=int(theta_deg))
    theta_rad = np.deg2rad(theta_deg)
    R = _y_rotation_matrix(theta_rad)

    max_diff_per_feature = np.zeros(N_ANGULAR, dtype=np.float64)

    for subj in range(SUBJECTS):
        for _f in range(FRAMES_PER_SUBJECT):
            pose = _random_canonical_pose(rng)
            rotated = (R @ pose.T).T.astype(np.float32)

            angles_before = compute_frame_angles(pose)
            angles_after = compute_frame_angles(rotated)

            diff = np.abs(angles_before - angles_after)
            max_diff_per_feature = np.maximum(max_diff_per_feature, diff)

    # Report the worst offender so failures localise the broken feature
    offenders = [
        (i, float(d)) for i, d in enumerate(max_diff_per_feature) if d > TOL
    ]
    assert not offenders, (
        f"Features not Y-rotation invariant at {theta_deg}°. "
        f"Max |Δ| per feature (idx, diff): {offenders}"
    )


def test_spine_features_are_distinct_after_fix() -> None:
    """
    Features 16 (sag) and 17 (front) must NOT both collapse to the same value
    for a subject leaning asymmetrically — that would indicate the fix
    degenerated into a single tilt scalar.
    """
    rng = np.random.default_rng(seed=42)
    pose = _random_canonical_pose(rng)
    # Force a 0.2 m forward lean AND a 0.1 m left lean on the neck
    pose[13][2] += 0.20  # forward (+Z)
    pose[13][0] -= 0.10  # left (-X)

    angles = compute_frame_angles(pose)
    sag = angles[16]
    front = angles[17]
    # Must differ by at least 2° — forward lean (0.2 m) should dominate frontal (0.1 m)
    assert abs(sag - front) > np.deg2rad(2.0), (
        f"Features 16 (sag={sag:.4f}) and 17 (front={front:.4f}) collapsed to "
        f"near-identical values — fix degenerated the pair into one scalar."
    )
