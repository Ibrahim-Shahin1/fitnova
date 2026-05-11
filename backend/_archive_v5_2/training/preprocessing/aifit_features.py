"""
AIFit-style active/passive feature mining (v5).

This module replaces v4's synthetic-corruption labelling with the analytical
pipeline from AIFit (Fieraru et al., the canonical Fit3D paper).

The core idea:
  • Each exercise has a fixed set of 22 angular features (computed by
    ``angular_features.py``).
  • Some of those features are MEANT to vary across the rep — e.g. knee
    flexion in a squat, elbow flexion in a curl. AIFit calls these *active*.
  • The remaining features should stay near-constant — e.g. trunk lean
    during a curl, ankle angle during a press. AIFit calls these *passive*.
  • Activeness is discovered automatically per exercise from the instructor's
    motion energy; we do not hard-code which features matter for which
    exercise.

The instructor reference is subject s03 (per AIFit's protocol — "one licensed
fitness instructor… reference for correctness").

Signature format
----------------
For each exercise we store an "AIFit signature" — a fixed-length statistical
descriptor of how the instructor performs that exercise:

    Active features  → (mean, std, max, min, range)
    Passive features → (mean, std)

A rep's deviation from the instructor signature is the L2 distance in
z-score space, restricted to the active-feature subset (passive features
also contribute, but down-weighted).

Quality scalar
--------------
    quality = exp(-‖deviation‖₂ / τ)

τ is calibrated on a held-out instructor split so the instructor's own
reps cluster near 0.95 and the worst novice reps land near 0.30.

Per-joint-group attribution
---------------------------
Each angular feature belongs to one or more of 5 joint groups
(``knee``, ``hip``, ``back``, ``shoulder``, ``elbow``). The per-frame deviation
of each feature contributes to its joint group(s); thresholding at p75 of
the instructor frames gives a binary "this group is deviating now" signal.

This module is stateless / pure-function. It does NOT load any files —
that's the dataset builder's job.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .angular_features import (
    ARTICULATION_ANGLES,
    AXIS_ANGLES,
    N_ANGULAR,
    compute_sequence_angles,
)

# ── 5 joint groups (down from v4's 10) ────────────────────────────────────────
JOINT_GROUPS_V5 = ["knee", "hip", "back", "shoulder", "elbow"]
N_JOINT_GROUPS_V5 = len(JOINT_GROUPS_V5)
GROUP_TO_IDX_V5 = {g: i for i, g in enumerate(JOINT_GROUPS_V5)}

# Angular feature index → list of joint groups it belongs to.
# Indexed exactly per angular_features.py:
#   0-1   l/r_elbow_flex            → elbow
#   2-5   l/r_shoulder_flex/abd     → shoulder
#   6-7   l/r_knee_flex             → knee
#   8-11  l/r_hip_flex/abd          → hip
#   12    trunk_flex                → back
#   13    trunk_lateral             → back
#   14    neck_flex (alias spine)   → back
#   15    pelvis_tilt               → hip
#   16-17 spine_vert_sag/front      → back
#   18-19 l/r_thigh_vert            → hip + knee
#   20-21 l/r_arm_vert              → shoulder + elbow
FEATURE_TO_GROUPS: Dict[int, List[str]] = {
    0:  ["elbow"],
    1:  ["elbow"],
    2:  ["shoulder"],
    3:  ["shoulder"],
    4:  ["shoulder"],
    5:  ["shoulder"],
    6:  ["knee"],
    7:  ["knee"],
    8:  ["hip"],
    9:  ["hip"],
    10: ["hip"],
    11: ["hip"],
    12: ["back"],
    13: ["back"],
    14: ["back"],
    15: ["hip"],
    16: ["back"],
    17: ["back"],
    18: ["hip", "knee"],
    19: ["hip", "knee"],
    20: ["shoulder", "elbow"],
    21: ["shoulder", "elbow"],
}


def feature_to_group_matrix() -> np.ndarray:
    """Return (N_ANGULAR, N_JOINT_GROUPS_V5) float32 attribution matrix.

    Row f, column g = 1.0 / k if feature f belongs to group g (one of k groups).
    Otherwise 0. Rows sum to 1, so attributing a feature's deviation to
    multiple groups conserves total magnitude.
    """
    M = np.zeros((N_ANGULAR, N_JOINT_GROUPS_V5), dtype=np.float32)
    for f, groups in FEATURE_TO_GROUPS.items():
        share = 1.0 / len(groups)
        for g in groups:
            M[f, GROUP_TO_IDX_V5[g]] = share
    return M


# ── Active / passive classification ───────────────────────────────────────────


def compute_motion_energy(angles: np.ndarray) -> np.ndarray:
    """Per-feature motion energy across one rep.

    Args:
        angles: (T, N_ANGULAR) angular features for ONE rep, in radians.

    Returns:
        energy: (N_ANGULAR,) the variance of each feature across time.

    AIFit defines motion energy as the temporal variance of an angular
    feature within a rep. Active features have high variance (the joint
    moves a lot during the rep); passive features have low variance.
    """
    return np.var(angles, axis=0).astype(np.float32)


def classify_active_features(
    instructor_reps: List[np.ndarray],
    threshold: float = 0.5,
) -> np.ndarray:
    """Determine which of the 22 features are 'active' for one exercise.

    Args:
        instructor_reps: list of (T, 22) angle arrays, all reps of ONE
            exercise performed by the instructor.
        threshold: a feature is active if its mean motion-energy is at least
            ``threshold * max(mean_energy)``. Default 0.5 (the elbow-of-the-
            -bend in motion-energy distributions).

    Returns:
        active_mask: (N_ANGULAR,) bool — True if active.
    """
    if not instructor_reps:
        # Fall back: nothing is active. Caller should treat this as
        # 'this exercise has no signature' and skip it.
        return np.zeros(N_ANGULAR, dtype=bool)

    per_rep_energy = np.stack(
        [compute_motion_energy(r) for r in instructor_reps], axis=0
    )  # (n_reps, 22)
    mean_energy = per_rep_energy.mean(axis=0)  # (22,)

    if mean_energy.max() < 1e-6:
        return np.zeros(N_ANGULAR, dtype=bool)

    cutoff = threshold * mean_energy.max()
    return mean_energy >= cutoff


# ── Instructor signature ──────────────────────────────────────────────────────


def build_instructor_signature(
    instructor_reps: List[np.ndarray],
    active_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute the AIFit signature for one exercise from instructor reps.

    For each feature we store:
      • mean : per-frame mean of the feature across all instructor reps
      • std  : per-frame std (clipped to a small floor so we never divide
              by zero during z-scoring)

    For active features we ALSO store:
      • max, min, range (max - min) over the rep

    Args:
        instructor_reps: list of (T, 22) instructor reps for ONE exercise.
        active_mask: (22,) bool from ``classify_active_features``.

    Returns:
        Dict with keys ``mean``, ``std``, ``max``, ``min``, ``range``,
        ``active_mask`` — all numpy arrays of shape (22,) (except active_mask
        which is bool).
    """
    if not instructor_reps:
        return {
            "mean": np.zeros(N_ANGULAR, dtype=np.float32),
            "std": np.ones(N_ANGULAR, dtype=np.float32),
            "max": np.zeros(N_ANGULAR, dtype=np.float32),
            "min": np.zeros(N_ANGULAR, dtype=np.float32),
            "range": np.zeros(N_ANGULAR, dtype=np.float32),
            "active_mask": active_mask.astype(np.bool_),
        }

    # Concatenate all instructor frames across reps for the per-frame mean/std
    all_frames = np.concatenate(instructor_reps, axis=0)  # (sum_T, 22)
    mean = all_frames.mean(axis=0).astype(np.float32)
    std = all_frames.std(axis=0).astype(np.float32)
    std = np.clip(std, 0.05, None)  # 0.05 rad ≈ 2.9° floor (matches normalize.py)

    # Per-rep max/min, then average across reps → "typical" range
    per_rep_max = np.stack([r.max(axis=0) for r in instructor_reps], axis=0)
    per_rep_min = np.stack([r.min(axis=0) for r in instructor_reps], axis=0)
    mx = per_rep_max.mean(axis=0).astype(np.float32)
    mn = per_rep_min.mean(axis=0).astype(np.float32)
    rng = (mx - mn).astype(np.float32)

    return {
        "mean": mean,
        "std": std,
        "max": mx,
        "min": mn,
        "range": rng,
        "active_mask": active_mask.astype(np.bool_),
    }


# ── Per-rep deviation + quality scalar ────────────────────────────────────────


def per_frame_zscore(
    angles_rep: np.ndarray,
    signature: Dict[str, np.ndarray],
) -> np.ndarray:
    """Per-frame z-score of a rep against the instructor signature.

    Returns:
        (T, 22) z-scores. A score of 0 means "exactly on the instructor
        mean for that feature"; |z| > 2 means "outside 2 std".
    """
    mean = signature["mean"]
    std = signature["std"]
    return ((angles_rep - mean) / std).astype(np.float32)


def per_frame_group_deviation(
    angles_rep: np.ndarray,
    signature: Dict[str, np.ndarray],
    feature_group_matrix: np.ndarray,
    active_weight: float = 1.0,
    passive_weight: float = 0.4,
) -> np.ndarray:
    """Per-frame deviation magnitude per joint group.

    For each frame, project the absolute z-scores onto the 5 joint groups
    via ``feature_group_matrix``, weighting active features more than
    passive ones (passive features are mostly position constraints, not
    movement signal — they shouldn't dominate quality).

    Args:
        angles_rep: (T, 22) angular features for one rep.
        signature: instructor signature dict.
        feature_group_matrix: (22, 5) attribution from
            ``feature_to_group_matrix()``.
        active_weight: weight applied to active features' contribution.
        passive_weight: weight applied to passive features' contribution.

    Returns:
        (T, 5) float32 — per-frame, per-group deviation magnitude in
        z-score units.
    """
    z = np.abs(per_frame_zscore(angles_rep, signature))  # (T, 22)
    active_mask = signature["active_mask"].astype(np.float32)  # (22,)
    weights = active_mask * active_weight + (1.0 - active_mask) * passive_weight
    z_weighted = z * weights[None, :]
    return (z_weighted @ feature_group_matrix).astype(np.float32)  # (T, 5)


def quality_scalar(
    angles_rep: np.ndarray,
    signature: Dict[str, np.ndarray],
    tau: float,
    active_weight: float = 1.0,
    passive_weight: float = 0.4,
) -> float:
    """Single quality scalar in (0, 1] for one rep.

    Aggregates the per-frame group deviation across time and groups, then
    maps to (0, 1] via ``exp(-d/τ)``.

    Args:
        angles_rep: (T, 22)
        signature: instructor signature dict
        tau: calibration scale. Larger τ ⇒ more lenient (instructor reps
            cluster nearer 1.0). Calibrate on held-out instructor data so
            instructor median ≈ 0.95.
        active_weight: see ``per_frame_group_deviation``
        passive_weight: see ``per_frame_group_deviation``

    Returns:
        quality in (0, 1] float.
    """
    M = feature_to_group_matrix()
    d_per_frame = per_frame_group_deviation(
        angles_rep, signature, M,
        active_weight=active_weight, passive_weight=passive_weight,
    )  # (T, 5)
    # Aggregate: sum across groups, mean across frames → scalar
    d_scalar = float(d_per_frame.sum(axis=1).mean())
    return float(np.exp(-d_scalar / max(tau, 1e-6)))


def per_frame_joint_errors(
    angles_rep: np.ndarray,
    signature: Dict[str, np.ndarray],
    instructor_p75_per_group: np.ndarray,
    active_weight: float = 1.0,
    passive_weight: float = 0.4,
) -> np.ndarray:
    """Binary per-frame per-group error labels for one rep.

    A group is flagged as 'erroring' on a frame if its deviation magnitude
    exceeds the p75 deviation of the instructor on that same group (sparse,
    interpretable, real).

    Args:
        angles_rep: (T, 22)
        signature: instructor signature dict
        instructor_p75_per_group: (5,) the 75th-percentile per-group
            deviation across all instructor frames of that exercise.
            Built once per exercise — see ``build_instructor_p75``.
        active_weight: see above
        passive_weight: see above

    Returns:
        (T, 5) bool — per-frame, per-group error flags. ``True`` ⇒ erroring.
    """
    M = feature_to_group_matrix()
    d_per_frame = per_frame_group_deviation(
        angles_rep, signature, M,
        active_weight=active_weight, passive_weight=passive_weight,
    )  # (T, 5)
    return d_per_frame > instructor_p75_per_group[None, :]


def build_instructor_p75(
    instructor_reps: List[np.ndarray],
    signature: Dict[str, np.ndarray],
    active_weight: float = 1.0,
    passive_weight: float = 0.4,
) -> np.ndarray:
    """Compute per-group p75 deviation across all instructor frames.

    The instructor's own reps will of course have small deviations from the
    instructor signature — but not zero (within-subject variation). We
    threshold at p75 of these self-deviations: a "real" error is one that
    exceeds the upper quartile of the instructor's own jitter.

    Returns:
        (5,) float32 — p75 per joint group.
    """
    if not instructor_reps:
        return np.zeros(N_JOINT_GROUPS_V5, dtype=np.float32)
    M = feature_to_group_matrix()
    all_dev = []
    for rep in instructor_reps:
        d = per_frame_group_deviation(
            rep, signature, M,
            active_weight=active_weight, passive_weight=passive_weight,
        )  # (T, 5)
        all_dev.append(d)
    cat = np.concatenate(all_dev, axis=0)  # (sum_T, 5)
    return np.percentile(cat, 75, axis=0).astype(np.float32)


# ── Tau calibration helper ────────────────────────────────────────────────────


def calibrate_tau(
    instructor_reps: List[np.ndarray],
    signature: Dict[str, np.ndarray],
    target_instructor_quality: float = 0.95,
    active_weight: float = 1.0,
    passive_weight: float = 0.4,
) -> float:
    """Choose τ so the median instructor self-quality lands near a target.

    Procedure:
      1. Compute per-rep deviation magnitude d_i for every instructor rep.
      2. We want median(exp(-d_i/τ)) ≈ target.
      3. Solve τ = -median(d_i) / ln(target).

    Args:
        instructor_reps: list of (T, 22) instructor reps for ONE exercise.
        signature: built from those reps (or a held-out subset for honesty).
        target_instructor_quality: where the median instructor rep should land.
        active_weight, passive_weight: see other functions.

    Returns:
        tau: positive float.
    """
    if not instructor_reps:
        return 1.0
    M = feature_to_group_matrix()
    devs = []
    for rep in instructor_reps:
        d_per_frame = per_frame_group_deviation(
            rep, signature, M,
            active_weight=active_weight, passive_weight=passive_weight,
        )
        devs.append(float(d_per_frame.sum(axis=1).mean()))
    median_d = float(np.median(devs))
    target = float(np.clip(target_instructor_quality, 1e-3, 1 - 1e-3))
    tau = -median_d / np.log(target)
    return float(max(tau, 1e-3))


# ── Convenience: all per-exercise artifacts in one struct ─────────────────────


def build_exercise_signature_bundle(
    angles_per_subject: Dict[str, List[np.ndarray]],
    instructor_subject: str = "s03",
    activeness_threshold: float = 0.5,
    target_instructor_quality: float = 0.95,
) -> Dict[str, np.ndarray]:
    """End-to-end signature build for ONE exercise.

    Args:
        angles_per_subject: dict ``{subject_id: [angle_arrays per rep]}`` for
            ONE exercise. Each angle array is (T, 22).
        instructor_subject: which subject is the AIFit instructor (default s03).
        activeness_threshold: threshold for active-feature classification.
        target_instructor_quality: τ-calibration target.

    Returns:
        Dict with everything the dataset builder needs for this exercise:
        ``mean``, ``std``, ``max``, ``min``, ``range``, ``active_mask``,
        ``p75_per_group``, ``tau``, ``n_instructor_reps``.
    """
    instructor_reps = angles_per_subject.get(instructor_subject, [])
    active_mask = classify_active_features(
        instructor_reps, threshold=activeness_threshold
    )
    signature = build_instructor_signature(instructor_reps, active_mask)
    p75 = build_instructor_p75(instructor_reps, signature)
    tau = calibrate_tau(
        instructor_reps, signature,
        target_instructor_quality=target_instructor_quality,
    )

    bundle = dict(signature)
    bundle["p75_per_group"] = p75
    bundle["tau"] = np.float32(tau)
    bundle["n_instructor_reps"] = np.int32(len(instructor_reps))
    return bundle
