"""
Synthetic form-error corruption for self-supervised label generation.

Strategy
--------
Fit3D provides high-quality exercise motion but no "good/bad form" labels.
Instead of comparing to a single instructor's idiosyncratic trajectory
(the AIFit baseline, which turns out to produce degenerate labels on a
small subject pool), we *synthesise* known-bad-form variants by applying
controlled 3D perturbations to specific joint groups.

For each base rep we generate:
  • 1  clean variant   (quality=1.0,  joint_errors=all zero)
  • K  corrupted variants, each:
       – a random joint group is selected (out of 10)
       – a severity is sampled  (MILD | MODERATE | SEVERE)
       – a geometric perturbation appropriate for that joint is applied
         across all frames of the rep
       – quality label is set according to severity
       – joint_errors[:, joint_group] is set to 1 for all frames

This gives us:
  • guaranteed balanced label distribution
  • known ground-truth for every sample
  • interpretable corruption types that map to real-world form faults

At inference time the model has learned to recognise angular/spatial
deviations from nominal form — which is exactly what real bad form looks
like to a coach.

Corruption types per joint group
---------------------------------
Each joint group has one or two geometric corruptions that create a
realistic-looking error pattern:

  • elbow       → rotate forearm (wrist)  around the upper-arm axis
                 (simulates collapsed elbow during press/row)
  • shoulder    → rotate upper arm (elbow) around the torso axis
                 (simulates flared elbows / poor scapular positioning)
  • knee        → rotate shin (ankle) around the thigh axis
                 (simulates valgus or cut-short ROM)
  • hip         → rotate thigh (knee)  around the pelvis axis
                 (simulates hip shift or butt-wink)
  • trunk       → rotate upper body around the pelvis
                 (simulates forward lean / twist)
  • neck        → rotate head/neck around the spine mid
                 (simulates cervical flexion / lookdown)

All rotations are applied as constant offsets for the whole rep — this
mirrors how a real form fault tends to be a consistent postural habit
rather than a momentary slip.
"""

import numpy as np
from typing import List, Tuple

# ── Canonical joint indices (must match joint_mapping.py CANONICAL_JOINTS) ─────
L_SHO, R_SHO  = 0, 1
L_ELB, R_ELB  = 2, 3
L_WRI, R_WRI  = 4, 5
L_HIP, R_HIP  = 6, 7
L_KNE, R_KNE  = 8, 9
L_ANK, R_ANK  = 10, 11
PELVIS        = 12
NECK          = 13
SPINE_MID     = 14

# For each joint group (10 total) declare:
#   root:        joint the rotation happens AROUND
#   hinge:       the joint whose position is rotated
#   descendants: joints that follow along (must also be rotated to preserve chain)
JOINT_GROUP_SPECS = {
    0: {  # l_elbow
        "root": L_ELB,  "hinge": L_WRI,  "descendants": [],
    },
    1: {  # r_elbow
        "root": R_ELB,  "hinge": R_WRI,  "descendants": [],
    },
    2: {  # l_shoulder
        "root": L_SHO,  "hinge": L_ELB,  "descendants": [L_WRI],
    },
    3: {  # r_shoulder
        "root": R_SHO,  "hinge": R_ELB,  "descendants": [R_WRI],
    },
    4: {  # l_knee
        "root": L_KNE,  "hinge": L_ANK,  "descendants": [],
    },
    5: {  # r_knee
        "root": R_KNE,  "hinge": R_ANK,  "descendants": [],
    },
    6: {  # l_hip
        "root": L_HIP,  "hinge": L_KNE,  "descendants": [L_ANK],
    },
    7: {  # r_hip
        "root": R_HIP,  "hinge": R_KNE,  "descendants": [R_ANK],
    },
    8: {  # trunk
        "root": PELVIS, "hinge": SPINE_MID,
        "descendants": [NECK, L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI],
    },
    9: {  # neck
        "root": SPINE_MID, "hinge": NECK, "descendants": [],
    },
}

# Severity → (angle range in radians, quality penalty)
# Angle: magnitude of the rotation applied
# Quality: label assigned to the corrupted rep (higher severity → lower quality)
SEVERITY_LEVELS = {
    "mild":     {"angle_range": (0.15, 0.25), "quality": 0.80},  # ~9-14°
    "moderate": {"angle_range": (0.25, 0.40), "quality": 0.55},  # ~14-23°
    "severe":   {"angle_range": (0.40, 0.60), "quality": 0.25},  # ~23-34°
}
SEVERITY_PROBS = [0.5, 0.35, 0.15]   # favour mild so labels aren't skewed severe
SEVERITY_NAMES = list(SEVERITY_LEVELS.keys())


def _rotation_matrix_about_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' formula. axis should be unit length."""
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = axis
    C = 1 - c
    return np.array([
        [c + x*x*C,    x*y*C - z*s,  x*z*C + y*s],
        [y*x*C + z*s,  c + y*y*C,    y*z*C - x*s],
        [z*x*C - y*s,  z*y*C + x*s,  c + z*z*C],
    ], dtype=np.float32)


def _pick_rotation_axis(rng: np.random.Generator) -> np.ndarray:
    """
    Random unit vector on the unit sphere. Using a random axis rather than a
    fixed anatomical plane keeps the model from learning the exact corruption
    recipe, and covers the variety of directions real form faults can occur in.
    """
    v = rng.standard_normal(3)
    return v / (np.linalg.norm(v) + 1e-9)


def apply_joint_group_corruption(
    canonical_seq: np.ndarray,   # (T, 15, 3)
    joint_group: int,            # 0..9
    severity: str = "moderate",
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, float]:
    """
    Apply a geometric corruption to one joint group across every frame.

    The corruption is deterministic per (joint_group, sampled angle, sampled axis):
    a fixed rotation of the hinge joint (and its descendants) around the root joint.

    Returns:
        corrupted_seq: (T, 15, 3)  — new sequence, original untouched
        quality_label: float       — ground-truth quality for this rep
    """
    if rng is None:
        rng = np.random.default_rng()

    spec = JOINT_GROUP_SPECS[joint_group]
    root_idx  = spec["root"]
    hinge_idx = spec["hinge"]
    desc_idx  = spec["descendants"]

    lvl   = SEVERITY_LEVELS[severity]
    angle = rng.uniform(*lvl["angle_range"])
    # Random sign (rotate either direction)
    angle *= 1.0 if rng.random() < 0.5 else -1.0
    axis  = _pick_rotation_axis(rng)

    # Build rotation matrix (same for every frame — constant postural offset)
    R = _rotation_matrix_about_axis(axis, angle)   # (3,3)

    out = canonical_seq.copy()
    T = out.shape[0]

    joints_to_rotate = [hinge_idx] + list(desc_idx)
    for j in joints_to_rotate:
        # Rotate j around root: v' = R @ (v - root) + root
        rel = out[:, j, :] - out[:, root_idx, :]         # (T, 3)
        rotated = rel @ R.T                               # (T, 3)
        out[:, j, :] = rotated + out[:, root_idx, :]

    return out, lvl["quality"]


# ═════════════════════════════════════════════════════════════════════════════
#  Biomechanically motivated corruption types (R3)
# ═════════════════════════════════════════════════════════════════════════════
#
#  The original `apply_joint_group_corruption` rotates ONE joint group by a
#  CONSTANT offset across the entire rep. Real form faults rarely look like
#  that. The four functions below simulate failure modes that coaches
#  actually cue against:
#
#      apply_rom_compression     — half-reps / insufficient depth
#      apply_phase_specific_offset — "error at the bottom/top" faults
#      apply_asymmetric_load     — one-sided compensation
#      apply_trajectory_jitter   — unstable / grinding reps
#
#  All four return (corrupted_seq, quality_label, errors_per_group) so they
#  slot into `generate_label_variants` interchangeably.
#
#  Each corruption maps to one or more of the 10 joint_error groups so the
#  per-group head still gets a localised signal:
#
#      ROM compression         -> groups: hip+knee     (for lower-body lifts)
#                                         or shoulder+elbow (upper-body lifts)
#      phase-specific offset   -> groups: the single joint_group it rotates
#      asymmetric load         -> groups: both sides of the affected joint pair
#      trajectory jitter       -> groups: the joint_group receiving jitter
#

def _descend(joint: int) -> List[int]:
    """Return the set of joints that follow `joint` in the kinematic chain."""
    return {
        L_SHO: [L_ELB, L_WRI],
        R_SHO: [R_ELB, R_WRI],
        L_ELB: [L_WRI],
        R_ELB: [R_WRI],
        L_HIP: [L_KNE, L_ANK],
        R_HIP: [R_KNE, R_ANK],
        L_KNE: [L_ANK],
        R_KNE: [R_ANK],
        PELVIS: [SPINE_MID, NECK,
                 L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI,
                 L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK],
        SPINE_MID: [NECK,
                    L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI],
        NECK: [],
    }.get(joint, [])


def apply_rom_compression(
    canonical_seq: np.ndarray,      # (T, 15, 3)
    scale: float = 0.6,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Simulate a half-rep / insufficient ROM by contracting every frame toward
    the mean frame of the rep. scale in (0, 1]; 1.0 = no change, 0.5 = half ROM.

    The quality label scales smoothly with `scale`: 0.5 -> 0.25, 0.8 -> 0.7.
    All four limb groups (both knees, both hips, both elbows, both shoulders)
    are flagged because ROM compression is a whole-body symptom.
    """
    if rng is None:
        rng = np.random.default_rng()
    T = canonical_seq.shape[0]
    mean_frame = canonical_seq.mean(axis=0, keepdims=True)           # (1, 15, 3)
    out = mean_frame + scale * (canonical_seq - mean_frame)
    # quality: 0.5 scale => ~0.25, 0.8 scale => ~0.70
    quality = float(np.clip(0.05 + 0.80 * (scale - 0.4) / 0.4, 0.15, 0.85))
    errors = np.zeros((T, 10), dtype=np.float32)
    for g in (2, 3, 4, 5, 6, 7):   # both shoulders, knees, hips
        errors[:, g] = 1.0
    return out.astype(np.float32), quality, errors


def apply_phase_specific_offset(
    canonical_seq: np.ndarray,      # (T, 15, 3)
    joint_group: int = None,
    severity: str = "moderate",
    phase: Tuple[float, float] = (0.25, 0.75),
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Apply a rotation to one joint group ONLY during the given phase of the
    rep (e.g. frames 25 %-75 % => "error at the bottom"). Uses a raised-cosine
    envelope so the rotation fades in and out smoothly — avoids teaching the
    model to look for a sudden step.

    errors[:, joint_group] is set to 1 only during the active phase.
    """
    if rng is None:
        rng = np.random.default_rng()
    if joint_group is None:
        joint_group = int(rng.integers(0, 10))

    spec = JOINT_GROUP_SPECS[joint_group]
    root_idx, hinge_idx = spec["root"], spec["hinge"]
    desc_idx = spec["descendants"]

    lvl   = SEVERITY_LEVELS[severity]
    angle = rng.uniform(*lvl["angle_range"]) * (1.0 if rng.random() < 0.5 else -1.0)
    axis  = _pick_rotation_axis(rng)
    R     = _rotation_matrix_about_axis(axis, angle)   # (3,3)

    T = canonical_seq.shape[0]
    p0, p1 = int(phase[0] * T), int(phase[1] * T)
    p0, p1 = max(0, p0), min(T, p1)

    # Raised cosine envelope: 0 outside [p0,p1], 1 in the middle, smooth edges
    env = np.zeros(T, dtype=np.float32)
    if p1 > p0:
        local = np.linspace(0, np.pi, p1 - p0)
        env[p0:p1] = 0.5 * (1.0 - np.cos(local))  # 0..1..0

    out = canonical_seq.copy()
    joints_to_rotate = [hinge_idx] + list(desc_idx)
    for t in range(T):
        if env[t] == 0.0:
            continue
        theta = angle * env[t]
        R_t = _rotation_matrix_about_axis(axis, theta)
        for j in joints_to_rotate:
            rel = out[t, j] - out[t, root_idx]
            out[t, j] = rel @ R_t.T + out[t, root_idx]

    errors = np.zeros((T, 10), dtype=np.float32)
    errors[env > 0.1, joint_group] = 1.0

    # Quality penalty scaled by the fraction of the rep affected
    frac = float((env > 0.1).mean())
    quality = float(np.clip(1.0 - (1.0 - lvl["quality"]) * frac, 0.2, 0.95))
    return out.astype(np.float32), quality, errors


def apply_asymmetric_load(
    canonical_seq: np.ndarray,      # (T, 15, 3)
    side: str = None,               # "left" or "right"; random if None
    region: str = None,             # "upper" or "lower"; random if None
    severity: str = "moderate",
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Apply a larger rotation on one side than the other of the selected region.
    Simulates one-sided dominance (e.g. right leg working harder than left).
    """
    if rng is None:
        rng = np.random.default_rng()

    side   = side   or rng.choice(["left", "right"])
    region = region or rng.choice(["upper", "lower"])
    lvl    = SEVERITY_LEVELS[severity]

    # Map (region, side) -> (strong_side_group, weak_side_group)
    if region == "upper":
        groups = (2, 3) if side == "left" else (3, 2)   # (strong, weak) shoulders
    else:
        groups = (6, 7) if side == "left" else (7, 6)   # (strong, weak) hips
    strong_g, weak_g = groups

    # Strong side gets full magnitude, weak side gets 30 % (opposite sign)
    big_angle   = rng.uniform(*lvl["angle_range"]) * (1.0 if rng.random() < 0.5 else -1.0)
    small_angle = -0.3 * big_angle
    axis        = _pick_rotation_axis(rng)

    out = canonical_seq.copy()
    for g, theta in ((strong_g, big_angle), (weak_g, small_angle)):
        spec = JOINT_GROUP_SPECS[g]
        R = _rotation_matrix_about_axis(axis, theta)
        for j in [spec["hinge"]] + list(spec["descendants"]):
            rel = out[:, j, :] - out[:, spec["root"], :]
            out[:, j, :] = rel @ R.T + out[:, spec["root"], :]

    T = canonical_seq.shape[0]
    errors = np.zeros((T, 10), dtype=np.float32)
    errors[:, strong_g] = 1.0
    errors[:, weak_g]   = 1.0
    return out.astype(np.float32), float(lvl["quality"]), errors


def apply_trajectory_jitter(
    canonical_seq: np.ndarray,      # (T, 15, 3)
    joint_group: int = None,
    sigma: float = 0.03,            # per-frame angular noise std (radians)
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Apply high-frequency angular jitter to one joint group — simulates
    unstable / grinding reps.  For each frame we sample a small random
    rotation (axis and angle ~ N(0, sigma)) and apply it to the joint
    group's hinge + descendants about the root.
    """
    if rng is None:
        rng = np.random.default_rng()
    if joint_group is None:
        joint_group = int(rng.integers(0, 10))

    spec = JOINT_GROUP_SPECS[joint_group]
    root_idx, hinge_idx = spec["root"], spec["hinge"]
    desc_idx = spec["descendants"]

    out = canonical_seq.copy()
    T   = out.shape[0]
    joints_to_rotate = [hinge_idx] + list(desc_idx)

    # 3 Hz bandpass-ish noise: resample low-frequency noise up to per-frame
    # (prevents easy smoothing from cancelling it).
    n_knots = max(4, T // 4)
    knot_angles = rng.normal(0.0, sigma, size=(n_knots, 3))
    # Linear interpolation of the 3-vector noise
    xs = np.linspace(0, n_knots - 1, T)
    lo = np.floor(xs).astype(int)
    hi = np.minimum(lo + 1, n_knots - 1)
    alpha = (xs - lo)[:, None]
    noise = (1 - alpha) * knot_angles[lo] + alpha * knot_angles[hi]   # (T, 3)

    for t in range(T):
        v = noise[t]
        theta = float(np.linalg.norm(v))
        if theta < 1e-6:
            continue
        axis = v / theta
        R_t = _rotation_matrix_about_axis(axis, theta)
        for j in joints_to_rotate:
            rel = out[t, j] - out[t, root_idx]
            out[t, j] = rel @ R_t.T + out[t, root_idx]

    errors = np.zeros((T, 10), dtype=np.float32)
    errors[:, joint_group] = 1.0
    # Jitter is inherently moderate — quality 0.5
    return out.astype(np.float32), 0.50, errors


# ── Mixture sampler used by generate_label_variants ───────────────────────────

CORRUPTION_TYPES = [
    "rotation",          # original constant-rotation corruption (baseline)
    "phase_offset",      # error at the bottom / top
    "rom_compression",   # half-reps
    "asymmetric",        # one-sided dominance
    "jitter",            # grinding / unstable reps
]
# Keep the original rotation at 25 % (per plan); spread remaining 75 % across
# the four biomechanical types so each still gets a healthy share.
CORRUPTION_PROBS = [0.25, 0.25, 0.20, 0.15, 0.15]


def _sample_corruption(
    canonical_seq: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Sample ONE corruption variant according to CORRUPTION_PROBS."""
    kind = rng.choice(CORRUPTION_TYPES, p=CORRUPTION_PROBS)
    severity = rng.choice(SEVERITY_NAMES, p=SEVERITY_PROBS)

    if kind == "rotation":
        joint_group = int(rng.integers(0, 10))
        corr, q = apply_joint_group_corruption(canonical_seq, joint_group, severity, rng=rng)
        T = canonical_seq.shape[0]
        err = np.zeros((T, 10), dtype=np.float32)
        err[:, joint_group] = 1.0
        return corr, q, err

    if kind == "phase_offset":
        joint_group = int(rng.integers(0, 10))
        # Random phase window: symmetric around middle, width 30-60 % of rep
        width = rng.uniform(0.30, 0.60)
        centre = rng.uniform(0.30, 0.70)
        phase = (max(0.0, centre - width / 2), min(1.0, centre + width / 2))
        return apply_phase_specific_offset(
            canonical_seq, joint_group, severity=severity, phase=phase, rng=rng,
        )

    if kind == "rom_compression":
        scale = rng.uniform(0.50, 0.80)
        return apply_rom_compression(canonical_seq, scale=scale, rng=rng)

    if kind == "asymmetric":
        return apply_asymmetric_load(canonical_seq, severity=severity, rng=rng)

    # jitter
    sigma = rng.uniform(0.02, 0.04)
    return apply_trajectory_jitter(canonical_seq, sigma=sigma, rng=rng)


def generate_label_variants(
    canonical_seq: np.ndarray,       # (T, 15, 3)  clean, already normalised
    n_corruptions: int = 4,
    rng: np.random.Generator = None,
) -> List[Tuple[np.ndarray, float, np.ndarray]]:
    """
    Generate 1 clean + n_corruptions corrupted variants for one rep.

    Each corrupted variant is drawn from CORRUPTION_TYPES according to
    CORRUPTION_PROBS. This teaches the model to recognise a *family* of
    real form faults rather than only the original constant-rotation
    corruption (see R3 in glistening-inventing-steele.md).

    Returns a list of tuples:
        (canonical_seq, quality_label, joint_errors (T, 10))

    Where:
        - First element is always the clean variant: quality=1.0, errors=0.
        - Subsequent elements are corrupted variants.
    """
    if rng is None:
        rng = np.random.default_rng()

    T = canonical_seq.shape[0]
    variants: List[Tuple[np.ndarray, float, np.ndarray]] = []

    # Clean variant
    clean_errors = np.zeros((T, 10), dtype=np.float32)
    variants.append((canonical_seq.copy(), 1.0, clean_errors))

    # Corrupted variants — sampled from the mixture
    for _ in range(n_corruptions):
        corr, q, err = _sample_corruption(canonical_seq, rng)
        variants.append((corr, q, err))

    return variants
