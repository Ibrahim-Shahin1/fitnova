"""
Data augmentation for exercise rep sequences.

Applied to (T, 15, 3) canonical joint arrays BEFORE angular feature computation.
All augmentations preserve the semantic meaning of the exercise.
"""

import numpy as np
from typing import Tuple
from .normalize import resample_sequence, TARGET_FRAMES

_RNG = np.random.default_rng(42)

# ── Symmetric exercise list: L/R joints can be safely mirrored ────────────────
SYMMETRIC_EXERCISES = {
    "squat", "deadlift", "pushup", "diamond_pushup",
    "barbell_shrug", "burpees", "clean_and_press",
    "overhead_extension_thruster", "standing_ab_twists",
    "dumbbell_overhead_shoulder_press", "neutral_overhead_shoulder_press",
    "dumbbell_biceps_curls", "dumbbell_hammer_curls", "dumbbell_curl_trifecta",
    "dumbbell_high_pulls", "dumbbell_scaptions", "side_lateral_raise",
    "overhead_trap_raises", "w_raise", "band_pull_apart",
}

# Canonical index pairs that get swapped during mirror augmentation
MIRROR_SWAP_PAIRS = [
    (0, 1),   # l_shoulder ↔ r_shoulder
    (2, 3),   # l_elbow    ↔ r_elbow
    (4, 5),   # l_wrist    ↔ r_wrist
    (6, 7),   # l_hip      ↔ r_hip
    (8, 9),   # l_knee     ↔ r_knee
    (10, 11), # l_ankle    ↔ r_ankle
]


def mirror_rep(joints: np.ndarray) -> np.ndarray:
    """
    Mirror skeleton left↔right by:
    1. Negating the X coordinate.
    2. Swapping left/right joint pairs.

    Args:
        joints: (T, 15, 3)
    Returns:
        mirrored: (T, 15, 3)
    """
    out = joints.copy()
    out[:, :, 0] *= -1.0  # flip X axis
    for l, r in MIRROR_SWAP_PAIRS:
        out[:, [l, r], :] = out[:, [r, l], :]
    return out


def temporal_jitter(
    joints: np.ndarray,
    max_jitter: int = 5,
    rng: np.random.Generator = _RNG,
) -> np.ndarray:
    """
    Randomly trim up to max_jitter frames from the start and end,
    simulating slightly imperfect rep boundary detection.

    Args:
        joints: (T, 15, 3)
    Returns:
        jittered: (T', 15, 3) where T' ≤ T - 2
    """
    T = joints.shape[0]
    trim_start = rng.integers(0, max_jitter + 1)
    trim_end   = rng.integers(0, max_jitter + 1)
    end = T - trim_end
    if end <= trim_start + 10:
        return joints
    return joints[trim_start:end]


def add_angular_noise(
    joints: np.ndarray,
    sigma: float = 0.003,
    rng: np.random.Generator = _RNG,
) -> np.ndarray:
    """
    Add small Gaussian noise to joint positions (in metres).
    sigma=0.003 ≈ 3mm — realistic MediaPipe noise level.

    Args:
        joints: (T, 15, 3)
    Returns:
        noisy: (T, 15, 3)
    """
    noise = rng.normal(0.0, sigma, size=joints.shape).astype(np.float32)
    return joints + noise


def temporal_scale(
    joints: np.ndarray,
    min_len: int = 48,
    max_len: int = 80,
    rng: np.random.Generator = _RNG,
) -> np.ndarray:
    """
    Randomly resample to a length in [min_len, max_len], then return.
    Combined with later resampling to TARGET_FRAMES this simulates speed variation.

    Args:
        joints: (T, 15, 3)
    Returns:
        scaled: (new_len, 15, 3)
    """
    new_len = int(rng.integers(min_len, max_len + 1))
    return resample_sequence(joints, new_len)


def joint_dropout(
    joints: np.ndarray,
    drop_prob: float = 0.1,
    rng: np.random.Generator = _RNG,
) -> np.ndarray:
    """
    Randomly zero out individual joint positions with probability drop_prob
    per joint per frame, simulating MediaPipe landmark detection failures.

    Args:
        joints: (T, 15, 3)
    Returns:
        dropped: (T, 15, 3)
    """
    out = joints.copy()
    mask = rng.random(size=(joints.shape[0], joints.shape[1])) < drop_prob
    out[mask] = 0.0
    return out


def augment_rep(
    joints: np.ndarray,
    exercise: str,
    apply_mirror: bool = True,
    apply_jitter: bool = True,
    apply_noise: bool = True,
    apply_scale: bool = True,
    apply_dropout: bool = True,
    rng: np.random.Generator = _RNG,
) -> Tuple[np.ndarray, bool]:
    """
    Apply a random subset of augmentations to one rep.

    Returns:
        (augmented_joints (T, 15, 3), was_mirrored: bool)

    Mirror flag is returned so the label can be updated for asymmetric exercises.
    """
    mirrored = False

    if apply_jitter:
        joints = temporal_jitter(joints, rng=rng)

    if apply_scale and rng.random() < 0.5:
        joints = temporal_scale(joints, rng=rng)

    if apply_noise:
        joints = add_angular_noise(joints, rng=rng)

    if apply_dropout and rng.random() < 0.3:
        joints = joint_dropout(joints, rng=rng)

    if apply_mirror and exercise in SYMMETRIC_EXERCISES and rng.random() < 0.5:
        joints = mirror_rep(joints)
        mirrored = True

    return joints, mirrored


def generate_augmented_reps(
    rep_joints: np.ndarray,
    exercise: str,
    n_augments: int = 3,
    rng: np.random.Generator = _RNG,
) -> list:
    """
    Generate n_augments augmented versions of one rep.

    Returns list of (augmented_joints, mirrored) tuples.
    """
    results = []
    for _ in range(n_augments):
        aug, mirrored = augment_rep(rep_joints.copy(), exercise, rng=rng)
        results.append((aug, mirrored))
    return results
