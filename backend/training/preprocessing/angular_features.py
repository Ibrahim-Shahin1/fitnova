"""
Compute 22 angular features per frame from canonical 15-joint skeleton.

16 articulation angles (3D angle at vertex joint B in chain A→B→C)
 6 axis-relative angles (segment vs. vertical Y-axis)

All angles in radians, range [0, π].
"""

import numpy as np
from .joint_mapping import CANONICAL_JOINTS, N_CANONICAL

# Canonical joint indices (short aliases for readability)
L_SH  = 0   # l_shoulder
R_SH  = 1   # r_shoulder
L_EL  = 2   # l_elbow
R_EL  = 3   # r_elbow
L_WR  = 4   # l_wrist
R_WR  = 5   # r_wrist
L_HI  = 6   # l_hip
R_HI  = 7   # r_hip
L_KN  = 8   # l_knee
R_KN  = 9   # r_knee
L_AN  = 10  # l_ankle
R_AN  = 11  # r_ankle
PELV  = 12  # pelvis
NECK  = 13  # neck
SP    = 14  # spine_mid

N_ANGULAR = 22  # total angular features per frame

# ── Articulation angle definitions (A, B_vertex, C) ───────────────────────────
# angle = arccos(dot(B→A, B→C) / (|B→A| * |B→C|))
ARTICULATION_ANGLES = [
    # (name,          A,    B,    C)
    ("l_elbow_flex",  L_SH, L_EL, L_WR),   # 0
    ("r_elbow_flex",  R_SH, R_EL, R_WR),   # 1
    ("l_shoulder_flex", L_EL, L_SH, L_HI), # 2
    ("r_shoulder_flex", R_EL, R_SH, R_HI), # 3
    ("l_shoulder_abd",  L_EL, L_SH, NECK), # 4
    ("r_shoulder_abd",  R_EL, R_SH, NECK), # 5
    ("l_knee_flex",   L_HI, L_KN, L_AN),   # 6
    ("r_knee_flex",   R_HI, R_KN, R_AN),   # 7
    ("l_hip_flex",    L_KN, L_HI, PELV),   # 8
    ("r_hip_flex",    R_KN, R_HI, PELV),   # 9
    ("l_hip_abd",     L_KN, L_HI, R_HI),   # 10
    ("r_hip_abd",     R_KN, R_HI, L_HI),   # 11
    ("trunk_flex",    NECK, SP,   PELV),    # 12
    ("trunk_lateral", L_SH, SP,   R_SH),   # 13
    ("neck_flex",     NECK, SP,   PELV),    # 14  (alias: spine angle)
    ("pelvis_tilt",   SP,   PELV, L_KN),   # 15
]  # 16 angles total

# ── Axis-relative angle definitions ───────────────────────────────────────────
# Each entry: (name, joint_from, joint_to)
# angle between the segment (from→to) and the vertical Y-axis [0,1,0]
#
# NOTE: Features 16 (spine_vert_sag) and 17 (spine_vert_front) are computed
# separately from this list because they require a subject-relative basis
# (hip-line) to remain Y-rotation invariant. See _spine_subject_relative_angles.
# All four of the thigh/arm-vs-vertical angles below are already rotation-
# invariant because they compare 3D magnitudes against world-up Y.
AXIS_ANGLES = [
    ("spine_vert_sag",   PELV, NECK),   # 16 — subject-relative sagittal lean
    ("spine_vert_front", PELV, NECK),   # 17 — subject-relative frontal lean
    ("l_thigh_vert",     L_HI, L_KN),  # 18
    ("r_thigh_vert",     R_HI, R_KN),  # 19
    ("l_arm_vert",       L_SH, L_EL),  # 20
    ("r_arm_vert",       R_SH, R_EL),  # 21
]  # 6 angles total

ANGULAR_NAMES = (
    [a[0] for a in ARTICULATION_ANGLES] +
    [a[0] for a in AXIS_ANGLES]
)
assert len(ANGULAR_NAMES) == N_ANGULAR

# Vertical axis
_VERTICAL = np.array([0.0, 1.0, 0.0], dtype=np.float32)
_EPS = 1e-8


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """3D angle between two vectors (radians, [0, π])."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < _EPS or n2 < _EPS:
        return 0.0
    cos_val = np.dot(v1, v2) / (n1 * n2)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return float(np.arccos(cos_val))


def _articulation_angle(joints: np.ndarray, a: int, b: int, c: int) -> float:
    """Angle at vertex B in chain A-B-C."""
    ba = joints[a] - joints[b]
    bc = joints[c] - joints[b]
    return _angle_between(ba, bc)


def _axis_relative_angle(joints: np.ndarray, from_j: int, to_j: int,
                          plane: str = "3d") -> float:
    """Angle between segment (from→to) and vertical Y-axis.

    Full-3D `plane="3d"` is rotation-invariant around Y because it only uses
    the segment's magnitude and its Y-component (inner product with [0,1,0]).
    The projected variants (`"sagittal"`, `"frontal"`) are NOT Y-rotation
    invariant; kept only for backward compatibility. Prefer `"3d"`.
    """
    seg = joints[to_j] - joints[from_j]
    if plane == "sagittal":   # project to Y-Z plane (x=0) — NOT rotation-invariant
        seg = np.array([0.0, seg[1], seg[2]], dtype=np.float32)
    elif plane == "frontal":  # project to Y-X plane (z=0) — NOT rotation-invariant
        seg = np.array([seg[0], seg[1], 0.0], dtype=np.float32)
    return _angle_between(seg, _VERTICAL)


def _spine_subject_relative_angles(joints: np.ndarray) -> tuple[float, float]:
    """Rotation-invariant sagittal + frontal spine lean.

    Uses the hip-line (L_HI → R_HI) projected to the horizontal plane to
    define a subject-local basis:
      • right_hat   = unit vector from L_HI to R_HI, horizontal component only
      • forward_hat = cross(up, right_hat) — points out of subject's chest
    The spine segment (PELV → NECK) is decomposed onto (forward_hat, up,
    right_hat). Sagittal lean uses the forward+up plane; frontal lean uses
    the right+up plane. Both bases rotate rigidly with the subject around
    the Y-axis, so both returned angles are invariant to any world-frame
    Y-rotation of the input joints.

    Returns:
        (spine_vert_sag, spine_vert_front) in radians, both in [0, π/2].
    """
    # Hip line in the horizontal plane (zero out Y component)
    hip_vec = joints[R_HI] - joints[L_HI]
    right_horiz = np.array([hip_vec[0], 0.0, hip_vec[2]], dtype=np.float32)
    right_norm = np.linalg.norm(right_horiz)
    if right_norm < _EPS:
        # Degenerate (hips stacked vertically — should not happen) — fall back to
        # a total-tilt measurement duplicated across both features.
        spine = joints[NECK] - joints[PELV]
        spine_norm = np.linalg.norm(spine)
        if spine_norm < _EPS:
            return 0.0, 0.0
        horiz = float(np.sqrt(spine[0] ** 2 + spine[2] ** 2))
        tilt = float(np.arctan2(horiz, abs(spine[1])))
        return tilt, tilt

    right_hat = right_horiz / right_norm
    # Forward = up × right  (right-handed; for Y-up: forward = [-right_z, 0, right_x])
    forward_hat = np.array(
        [-right_hat[2], 0.0, right_hat[0]], dtype=np.float32
    )

    spine = joints[NECK] - joints[PELV]
    sag_component = float(np.dot(spine, forward_hat))   # forward lean magnitude (signed)
    front_component = float(np.dot(spine, right_hat))   # lateral lean magnitude (signed)
    up_component = float(spine[1])                      # vertical magnitude (signed)

    # Both outputs = angle from vertical in the respective plane,
    # mapped into [0, π/2] via abs (lean direction symmetry — form-quality
    # metrics care about magnitude of deviation, not direction).
    sag = float(np.arctan2(abs(sag_component), abs(up_component)))
    front = float(np.arctan2(abs(front_component), abs(up_component)))
    return sag, front


def compute_frame_angles(joints: np.ndarray) -> np.ndarray:
    """
    Compute 22 angular features for a single frame.

    Args:
        joints: (15, 3) canonical joint positions

    Returns:
        angles: (22,) float32 array in radians
    """
    assert joints.shape == (N_CANONICAL, 3), f"Expected (15,3), got {joints.shape}"

    angles = np.zeros(N_ANGULAR, dtype=np.float32)

    # 16 articulation angles
    for i, (_, a, b, c) in enumerate(ARTICULATION_ANGLES):
        angles[i] = _articulation_angle(joints, a, b, c)

    # 6 axis-relative angles
    # Features 16 + 17 use subject-relative basis (hip-line) — Y-rotation invariant.
    sag, front = _spine_subject_relative_angles(joints)
    angles[16] = sag
    angles[17] = front
    angles[18] = _axis_relative_angle(joints, L_HI, L_KN)
    angles[19] = _axis_relative_angle(joints, R_HI, R_KN)
    angles[20] = _axis_relative_angle(joints, L_SH, L_EL)
    angles[21] = _axis_relative_angle(joints, R_SH, R_EL)

    return angles


def compute_sequence_angles(joints_seq: np.ndarray) -> np.ndarray:
    """
    Compute angular features for an entire sequence.

    Args:
        joints_seq: (T, 15, 3) canonical joint sequence

    Returns:
        angles_seq: (T, 22) float32
    """
    T = joints_seq.shape[0]
    angles = np.zeros((T, N_ANGULAR), dtype=np.float32)
    for t in range(T):
        angles[t] = compute_frame_angles(joints_seq[t])
    return angles
