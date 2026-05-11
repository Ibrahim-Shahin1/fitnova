"""
Synthetic perturbation augmentation in canonical 15-joint space (v5.1).

Why this exists
---------------
v5.0 failed Phase 6. Diagnosis: Fit3D contains zero bad-form examples by
design — gym-experienced subjects, 80% effort, supervised. Trained quality-
label std was 0.04. Model learned to predict the training mean.

This module gives the model artificial bad-form examples while keeping the
labelling math canonical and AIFit-aligned.

How it works
------------
We operate in the project's CANONICAL 15-joint space (see
``joint_mapping.CANONICAL_JOINTS``). Both the MediaPipe-derived input pose
and the MoCap-derived label pose are mapped to this same 15-joint
representation, so the SAME perturbation function applied to both produces
matched (input, label) pairs:

    canon_input  = extract_canonical_from_mediapipe(...)   # for model input
    canon_label  = extract_canonical_from_fit3d(...)       # for label math
    plan         = sample_perturbation_plan(rng, n=6)
    for (group, severity) in plan:
        c_in_p   = perturb_canonical15(canon_input,  group, severity, rng)
        c_lab_p  = perturb_canonical15(canon_label,  group, severity, rng_same)
        # ... compute angles, build labels via aifit_features.quality_scalar etc.

Why this is different from v4's failure
---------------------------------------
v4 corrupted INPUT poses with constant geometric offsets and ASSIGNED
arbitrary scalar quality labels (1.0, 0.7, 0.4, 0.1) from a tier table. The
model learned to detect specific corruption fingerprints, not real form
deviation.

v5.1's perturbation:
  * Operates at the canonical pose level, not on angles directly.
  * Triggers the SAME canonical AIFit signature math used to label clean
    reps. Quality drops naturally because the perturbed angles deviate
    further from the instructor signature.
  * Per-joint-group error labels emerge from the AIFit attribution matrix —
    perturbing the knee group makes l_knee_flex / r_knee_flex / thigh_vert
    deviate, the attribution matrix maps those features back to the knee
    group, and the model's joint-error head sees a real biomechanical
    activation pattern, not invented frame-level masks.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

# ── Canonical 15-joint indices (mirrors joint_mapping.CANONICAL_JOINTS) ──────
# 0=l_shoulder, 1=r_shoulder, 2=l_elbow, 3=r_elbow, 4=l_wrist, 5=r_wrist,
# 6=l_hip,      7=r_hip,      8=l_knee,  9=r_knee, 10=l_ankle, 11=r_ankle,
# 12=pelvis,    13=neck,      14=spine_mid

# ── Perturbation ontology ────────────────────────────────────────────────────
# Each entry: (proximal_pivot_idx, [distal_subtree_idxs], rotation_axis).
# rotation_axis is 'x', 'y', or 'z' in the canonical world frame. Because
# our canonical poses are normalised hip-centred / torso-scaled with the
# subject roughly upright, world-frame axes are a reasonable proxy for the
# biomechanically relevant axes for small (<25 deg) perturbations.
#
# Axis choices:
#   'y' = vertical axis      → rotates the chain in the horizontal plane
#                              (knee valgus/varus, hip rotation)
#   'x' = mediolateral axis  → rotates in the sagittal plane
#                              (hip flex deviation, spine flex/extension)
#   'z' = anteroposterior    → rotates in the frontal plane
#                              (shoulder abd/adduction, elbow ulnar/radial)

PERTURBATIONS_CANONICAL = {
    "knee": [
        # Rotate shin around thigh axis (knee valgus / varus simulation).
        # Distal: ankle.
        (8,  [10], "y"),  # l_knee  pivot, l_ankle  distal
        (9,  [11], "y"),  # r_knee  pivot, r_ankle  distal
    ],
    "hip": [
        # Rotate thigh around hip in the sagittal plane (under-flexion / over-
        # flexion of the hip joint). Distal subtree includes knee + ankle.
        (6,  [8, 10], "x"),  # l_hip pivot
        (7,  [9, 11], "x"),  # r_hip pivot
    ],
    "back": [
        # Rotate the entire upper body (neck + both arms) around the spine_mid
        # pivot in the sagittal plane. Simulates excessive forward flexion or
        # hyperextension of the spine.
        (14, [13, 0, 1, 2, 3, 4, 5], "x"),
    ],
    "shoulder": [
        # Rotate the upper arm around the shoulder in the frontal plane.
        # Distal subtree: elbow + wrist.
        (0,  [2, 4], "z"),
        (1,  [3, 5], "z"),
    ],
    "elbow": [
        # Rotate forearm around elbow in the frontal plane.
        # Distal subtree: wrist.
        (2,  [4], "z"),
        (3,  [5], "z"),
    ],
}

JOINT_GROUPS = list(PERTURBATIONS_CANONICAL.keys())  # 5 groups

# Severity → peak rotation magnitude in degrees (baseline, before per-group
# multiplier). Calibrated so:
#   mild     ≈ small but visible form-error deviation
#   moderate ≈ clearly observable error a coach would call out
#   severe   ≈ large enough to bottom-out quality scalar
SEVERITY_DEG = {
    "mild":     8.0,
    "moderate": 16.0,
    "severe":   25.0,
}

# v5.2 fix: per-group severity multipliers, calibrated so that all five joint
# groups produce comparable canonical-pose displacement magnitudes regardless
# of the kinematic chain's lever-arm length. v5.1 evaluation showed knee /
# shoulder / elbow groups had joint_err F1 ≈ 0 because their tiny lever arms
# (rotating one joint around its proximal parent) produced sub-mm canonical-
# pose deltas that the model couldn't learn from. Boosting the angular
# rotations on these groups equalises the visible signal.
#
# Lever-arm reasoning (canonical units, hip-centred + torso-scaled):
#   back     ~ 0.7 (whole upper-body chain)         multiplier 1.0 (baseline)
#   hip      ~ 0.4 (thigh + shin from hip)          multiplier 1.4
#   knee     ~ 0.2 (just the ankle)                 multiplier 2.0
#   shoulder ~ 0.3 (forearm + wrist)                multiplier 1.5
#   elbow    ~ 0.2 (just the wrist)                 multiplier 2.0
#
# Effective severe rotation per group = 25° × multiplier:
#   back 25°, hip 35°, knee 50°, shoulder 37.5°, elbow 50°.
# All within the range of real bad-form rotations (knee valgus can hit 30-45°
# in collapsed form; elbow hyperextension can hit 30°+).
GROUP_SEVERITY_MULTIPLIER = {
    "back":     1.0,
    "hip":      1.4,
    "knee":     2.0,
    "shoulder": 1.5,
    "elbow":    2.0,
}


# ── Temporal envelope ────────────────────────────────────────────────────────


def _envelope(t: np.ndarray, kind: str = "midpeak") -> np.ndarray:
    """Per-frame perturbation magnitude in [0, 1].

    Real form errors usually peak under load, near the bottom of a rep
    (eccentric→concentric transition). The default midpeak envelope reflects
    that: zero at rep start/end, max at t=0.5.
    """
    t = np.clip(t, 0.0, 1.0)
    if kind == "midpeak":
        return np.sin(np.pi * t).astype(np.float32)
    if kind == "constant":
        return np.ones_like(t, dtype=np.float32)
    if kind == "ramp":
        return t.astype(np.float32)
    raise ValueError(f"Unknown envelope: {kind}")


def _rotation_matrix(axis: str, angle_rad: float) -> np.ndarray:
    """3×3 rotation around 'x', 'y', or 'z' (world frame)."""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    raise ValueError(f"Unknown axis: {axis}")


def _perturb_one_chain(
    canon: np.ndarray,
    proximal_idx: int,
    distal_idxs: List[int],
    axis: str,
    per_frame_angles_rad: np.ndarray,
) -> np.ndarray:
    """Apply per-frame rotations around proximal joint to the distal subtree.

    Args:
        canon                : (T, 15, 3) canonical pose. Not modified in place.
        proximal_idx         : pivot joint index in canonical space.
        distal_idxs          : list of joint indices to rotate together.
        axis                 : 'x' / 'y' / 'z' in the world frame.
        per_frame_angles_rad : (T,) rotation angle for each frame.

    Returns:
        new_canon            : (T, 15, 3) with the distal chain rotated.
    """
    out = canon.copy()
    T = out.shape[0]
    for t in range(T):
        ang = float(per_frame_angles_rad[t])
        if abs(ang) < 1e-6:
            continue
        R = _rotation_matrix(axis, ang)
        pivot = out[t, proximal_idx]
        for j in distal_idxs:
            v = out[t, j] - pivot
            out[t, j] = (R @ v) + pivot
    return out


# ── Public API ───────────────────────────────────────────────────────────────


def perturb_canonical15(
    canon: np.ndarray,
    joint_group: str,
    severity: str,
    rng: np.random.Generator,
    envelope: str = "midpeak",
    randomize_side: bool = True,
    angle_jitter_deg: float = 3.0,
    sign_choice: float | None = None,
) -> np.ndarray:
    """Apply a canonical-space perturbation to one rep.

    Args:
        canon            : (T, 15, 3) canonical pose for one rep, output of
                           ``extract_canonical_from_mediapipe`` or
                           ``extract_canonical_from_fit3d`` after
                           ``normalize_skeleton``.
        joint_group      : one of "knee", "hip", "back", "shoulder", "elbow".
        severity         : one of "mild", "moderate", "severe".
        rng              : numpy random Generator (for reproducibility).
        envelope         : temporal envelope shape ("midpeak" / "constant" /
                           "ramp"). Default "midpeak".
        randomize_side   : if True and the joint_group has bilateral chains,
                           randomly pick left, right, or both. If False,
                           perturb both sides symmetrically.
        angle_jitter_deg : per-frame Gaussian jitter (deg) added to the
                           sinusoidal envelope so the perturbation isn't
                           perfectly symmetric.
        sign_choice      : explicit -1.0 or +1.0 to force perturbation direction.
                           If None, randomly chosen. Set this when applying the
                           SAME perturbation to two paired streams (MoCap &
                           MediaPipe), so the rotation goes the same way on both.

    Returns:
        canon_perturbed  : (T, 15, 3) perturbed pose.
    """
    if joint_group not in PERTURBATIONS_CANONICAL:
        raise ValueError(f"Unknown joint group: {joint_group}")
    if severity not in SEVERITY_DEG:
        raise ValueError(f"Unknown severity: {severity}")

    chains = PERTURBATIONS_CANONICAL[joint_group]
    if randomize_side and len(chains) > 1:
        choice = int(rng.integers(0, 3))
        if choice == 0:
            chains = [chains[0]]
        elif choice == 1:
            chains = [chains[1]]
        # choice == 2: keep both

    T = canon.shape[0]
    t_norm = np.linspace(0.0, 1.0, T, dtype=np.float32)
    env = _envelope(t_norm, kind=envelope)

    # v5.2: peak rotation = base severity × per-group multiplier (lever-arm
    # equalisation; see GROUP_SEVERITY_MULTIPLIER docstring for rationale).
    peak_deg = SEVERITY_DEG[severity] * GROUP_SEVERITY_MULTIPLIER[joint_group]
    sign = float(sign_choice) if sign_choice is not None else float(rng.choice([-1.0, 1.0]))

    out = canon
    for proximal_idx, distal_idxs, axis in chains:
        deg_per_frame = sign * peak_deg * env
        if angle_jitter_deg > 0.0:
            deg_per_frame = deg_per_frame + rng.normal(
                0.0, angle_jitter_deg, size=T
            ).astype(np.float32)
        rad_per_frame = np.deg2rad(deg_per_frame).astype(np.float32)
        out = _perturb_one_chain(
            out, proximal_idx, distal_idxs, axis, rad_per_frame
        )
    return out


# ── Sampling helpers ─────────────────────────────────────────────────────────


def enumerate_perturbations(
    severities: Tuple[str, ...] = ("mild", "moderate", "severe"),
) -> List[Tuple[str, str]]:
    """All (joint_group, severity) pairs available."""
    return [(g, s) for g in JOINT_GROUPS for s in severities]


# Default number of synthetic variants per clean rep for v5.1.
# Empirical: 6 variants × ~1700 train samples = ~10K → enough for the
# quality head to learn a meaningful gradient without synthetic dominating.
DEFAULT_VARIANTS_PER_REP = 6


def sample_perturbation_plan(
    rng: np.random.Generator,
    n_variants: int = DEFAULT_VARIANTS_PER_REP,
    severities: Tuple[str, ...] = ("mild", "moderate", "severe"),
) -> List[Tuple[str, str]]:
    """Sample n distinct (joint_group, severity) pairs.

    Stratifies so that every joint group appears at least once if n_variants
    >= 5, then random-fills.
    """
    all_variants = enumerate_perturbations(severities)
    if n_variants >= len(all_variants):
        return list(all_variants)

    plan: List[Tuple[str, str]] = []
    groups_shuffled = list(JOINT_GROUPS)
    rng.shuffle(groups_shuffled)

    # First pass: cover each joint group up to n_variants
    for g in groups_shuffled[:n_variants]:
        s = str(rng.choice(severities))
        plan.append((g, s))

    # Second pass: random fill
    while len(plan) < n_variants:
        plan.append(
            (str(rng.choice(JOINT_GROUPS)), str(rng.choice(severities)))
        )
    return plan


def sample_paired_perturbation(
    rng: np.random.Generator,
    joint_group: str,
    severity: str,
    randomize_side: bool = True,
) -> dict:
    """Pre-sample the random parameters for one perturbation.

    Returned dict can be passed as ``**kwargs`` (``sign_choice`` and
    ``randomize_side=False``) to ``perturb_canonical15`` so the SAME
    perturbation is applied to two paired pose streams (e.g. MoCap label
    pose and MediaPipe input pose).

    The trick is: we compute the random side/sign ONCE here, then disable
    randomize_side and force sign_choice in subsequent calls.
    """
    sign = float(rng.choice([-1.0, 1.0]))
    chains = PERTURBATIONS_CANONICAL[joint_group]
    side_choice = int(rng.integers(0, 3)) if (randomize_side and len(chains) > 1) else -1
    return {
        "joint_group":    joint_group,
        "severity":       severity,
        "sign":           sign,
        "side_choice":    side_choice,
    }


def apply_paired_perturbation(
    canon: np.ndarray,
    plan: dict,
    rng: np.random.Generator,
    angle_jitter_deg: float = 3.0,
    envelope: str = "midpeak",
) -> np.ndarray:
    """Apply a pre-sampled perturbation plan to a canonical pose.

    Use ``sample_paired_perturbation`` to build the plan once, then call this
    twice (once on MoCap canonical, once on MediaPipe canonical, with the
    same plan and SEPARATE rngs for jitter so the two streams don't share
    angle noise).
    """
    chains = PERTURBATIONS_CANONICAL[plan["joint_group"]]
    side_choice = plan["side_choice"]
    if side_choice == 0:
        chains = [chains[0]]
    elif side_choice == 1:
        chains = [chains[1]]
    # else: keep all (either both sides or single-chain group like "back")

    T = canon.shape[0]
    t_norm = np.linspace(0.0, 1.0, T, dtype=np.float32)
    env = _envelope(t_norm, kind=envelope)

    # v5.2: per-group multiplier for lever-arm equalisation
    peak_deg = SEVERITY_DEG[plan["severity"]] * GROUP_SEVERITY_MULTIPLIER[plan["joint_group"]]
    sign = plan["sign"]

    out = canon
    for proximal_idx, distal_idxs, axis in chains:
        deg_per_frame = sign * peak_deg * env
        if angle_jitter_deg > 0.0:
            deg_per_frame = deg_per_frame + rng.normal(
                0.0, angle_jitter_deg, size=T
            ).astype(np.float32)
        rad_per_frame = np.deg2rad(deg_per_frame).astype(np.float32)
        out = _perturb_one_chain(
            out, proximal_idx, distal_idxs, axis, rad_per_frame
        )
    return out


# ── Smoke test / sanity ─────────────────────────────────────────────────────


if __name__ == "__main__":
    """Quick sanity: feed one synthetic squat-shaped pose through every
    (group, severity) and verify perturbation magnitude scales with severity.
    """
    rng = np.random.default_rng(42)

    # Build a fake squat-shaped canonical pose: T=80 frames, person standing
    # upright at t=0, descending into a squat by t=40, ascending by t=80.
    T = 80
    canon = np.zeros((T, 15, 3), dtype=np.float32)
    # Standing: shoulders at y=1, hips at y=0.5, knees at y=0.25, ankles at y=0
    rest = np.array([
        [-0.2, 1.0, 0.0],   # 0 l_shoulder
        [ 0.2, 1.0, 0.0],   # 1 r_shoulder
        [-0.3, 0.7, 0.0],   # 2 l_elbow
        [ 0.3, 0.7, 0.0],   # 3 r_elbow
        [-0.3, 0.4, 0.0],   # 4 l_wrist
        [ 0.3, 0.4, 0.0],   # 5 r_wrist
        [-0.1, 0.5, 0.0],   # 6 l_hip
        [ 0.1, 0.5, 0.0],   # 7 r_hip
        [-0.1, 0.25, 0.05], # 8 l_knee
        [ 0.1, 0.25, 0.05], # 9 r_knee
        [-0.1, 0.0, 0.0],   # 10 l_ankle
        [ 0.1, 0.0, 0.0],   # 11 r_ankle
        [ 0.0, 0.5, 0.0],   # 12 pelvis
        [ 0.0, 1.0, 0.0],   # 13 neck
        [ 0.0, 0.75, 0.0],  # 14 spine_mid
    ], dtype=np.float32)
    for t in range(T):
        # Squat down: scale y by sin envelope (0 at start/end, 0.5 at mid)
        depth = 0.3 * np.sin(np.pi * t / T)
        pose = rest.copy()
        pose[[6, 7, 12], 1] -= depth        # hips drop
        pose[[8, 9], 1] -= 0.5 * depth      # knees drop a little
        canon[t] = pose

    print("Canonical squat-shaped pose, T=80")
    print(f"  knee y at t=0:  {canon[0, 8, 1]:.3f}")
    print(f"  knee y at t=40: {canon[40, 8, 1]:.3f}")
    print(f"  knee y at t=80: {canon[-1, 8, 1]:.3f}")
    print()

    print("Perturbation sanity (mean absolute joint delta from clean):")
    for group in JOINT_GROUPS:
        for sev in ("mild", "moderate", "severe"):
            local_rng = np.random.default_rng(42)
            p = perturb_canonical15(canon, group, sev, local_rng)
            delta = float(np.mean(np.abs(p - canon)))
            print(f"  {group:<10s} {sev:<9s} mean abs delta: {delta:.5f}")
    print()

    print("Paired perturbation sanity (MoCap + MediaPipe both perturbed identically):")
    plan_rng = np.random.default_rng(7)
    plan = sample_paired_perturbation(plan_rng, "knee", "moderate")
    print(f"  plan: {plan}")
    a_rng = np.random.default_rng(11)
    b_rng = np.random.default_rng(11)  # same seed -> identical jitter
    a = apply_paired_perturbation(canon, plan, a_rng)
    b = apply_paired_perturbation(canon, plan, b_rng)
    print(f"  identical-rng A vs B max abs delta: {float(np.max(np.abs(a - b))):.6f} (should be ~0)")
    print()

    print("Sample perturbation plan (n=6):")
    for g, s in sample_perturbation_plan(np.random.default_rng(99), n_variants=6):
        print(f"  {g:<10s} {s}")
