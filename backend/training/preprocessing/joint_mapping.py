"""
Joint index mappings between Fit3D (25 joints) and MediaPipe Pose (33 landmarks).
Both are mapped to a canonical 15-joint representation used for angular feature computation.
"""

# ── Fit3D 25-joint indices ─────────────────────────────────────────────────────
FIT3D_JOINTS = {
    "pelvis":     0,
    "l_hip":      1,
    "r_hip":      2,
    "spine1":     3,
    "l_knee":     4,
    "r_knee":     5,
    "spine2":     6,
    "l_ankle":    7,
    "r_ankle":    8,
    "spine3":     9,
    "l_foot":    10,
    "r_foot":    11,
    "neck":      12,
    "l_collar":  13,
    "r_collar":  14,
    "head":      15,
    "l_shoulder":16,
    "r_shoulder":17,
    "l_elbow":   18,
    "r_elbow":   19,
    "l_wrist":   20,
    "r_wrist":   21,
    "l_hand":    22,
    "r_hand":    23,
    "nose":      24,
}

# ── MediaPipe Pose 33-landmark indices ─────────────────────────────────────────
MEDIAPIPE_JOINTS = {
    "nose":        0,
    "l_ear":       7,   # B5 fix: needed for anatomical neck estimation
    "r_ear":       8,   # B5 fix: needed for anatomical neck estimation
    "l_shoulder": 11,
    "r_shoulder": 12,
    "l_elbow":    13,
    "r_elbow":    14,
    "l_wrist":    15,
    "r_wrist":    16,
    "l_hip":      23,
    "r_hip":      24,
    "l_knee":     25,
    "r_knee":     26,
    "l_ankle":    27,
    "r_ankle":    28,
    "l_heel":     29,
    "r_heel":     30,
    "l_foot_index": 31,
    "r_foot_index": 32,
}

# ── Canonical 15-joint names (shared representation) ──────────────────────────
CANONICAL_JOINTS = [
    "l_shoulder",   # 0
    "r_shoulder",   # 1
    "l_elbow",      # 2
    "r_elbow",      # 3
    "l_wrist",      # 4
    "r_wrist",      # 5
    "l_hip",        # 6
    "r_hip",        # 7
    "l_knee",       # 8
    "r_knee",       # 9
    "l_ankle",      # 10
    "r_ankle",      # 11
    "pelvis",       # 12  (midpoint l_hip+r_hip in MediaPipe)
    "neck",         # 13  (midpoint l_shoulder+r_shoulder in MediaPipe)
    "spine_mid",    # 14  (midpoint pelvis+neck)
]
N_CANONICAL = len(CANONICAL_JOINTS)  # 15

# ── Fit3D → Canonical extraction ──────────────────────────────────────────────
# Each entry: canonical_index → Fit3D index (or tuple for midpoint average)
FIT3D_TO_CANONICAL = {
    0:  FIT3D_JOINTS["l_shoulder"],
    1:  FIT3D_JOINTS["r_shoulder"],
    2:  FIT3D_JOINTS["l_elbow"],
    3:  FIT3D_JOINTS["r_elbow"],
    4:  FIT3D_JOINTS["l_wrist"],
    5:  FIT3D_JOINTS["r_wrist"],
    6:  FIT3D_JOINTS["l_hip"],
    7:  FIT3D_JOINTS["r_hip"],
    8:  FIT3D_JOINTS["l_knee"],
    9:  FIT3D_JOINTS["r_knee"],
    10: FIT3D_JOINTS["l_ankle"],
    11: FIT3D_JOINTS["r_ankle"],
    12: (FIT3D_JOINTS["l_hip"], FIT3D_JOINTS["r_hip"]),        # pelvis = midpoint
    13: FIT3D_JOINTS["neck"],
    14: (FIT3D_JOINTS["spine2"],),                              # spine_mid ≈ spine2
}

# ── MediaPipe → Canonical extraction ──────────────────────────────────────────
# Each entry: canonical_index → MediaPipe index (or tuple for midpoint average)
#
# B5 fix (2026-04-18): neck (canonical 13) was previously set to
# midpoint(l_shoulder, r_shoulder), which is anatomically wrong — the neck
# joint (C7) sits above the shoulders, not between them.  Fit3D uses a real
# C7 landmark.  Using the shoulder midpoint biased 6 of 22 angular features
# (trunk_flex, neck_flex, l/r_shoulder_abd, spine_vert_sag/front).
#
# Fix: neck = 0.75 * shoulder_mid + 0.25 * ear_mid
# MediaPipe ear landmarks (7=l_ear, 8=r_ear) are at approximately the
# external acoustic meatus, which is ~midway up the head. Taking 25% of the
# ear midpoint moves the estimated "neck" ~1/4 of the way from shoulders
# toward the head, landing close to the Fit3D C7 position.
#
# This is implemented as a special-case computation in
# extract_canonical_from_mediapipe (see normalize.py) rather than a simple
# index tuple.  The sentinel value "neck_weighted" triggers that path.
MEDIAPIPE_TO_CANONICAL = {
    0:  MEDIAPIPE_JOINTS["l_shoulder"],
    1:  MEDIAPIPE_JOINTS["r_shoulder"],
    2:  MEDIAPIPE_JOINTS["l_elbow"],
    3:  MEDIAPIPE_JOINTS["r_elbow"],
    4:  MEDIAPIPE_JOINTS["l_wrist"],
    5:  MEDIAPIPE_JOINTS["r_wrist"],
    6:  MEDIAPIPE_JOINTS["l_hip"],
    7:  MEDIAPIPE_JOINTS["r_hip"],
    8:  MEDIAPIPE_JOINTS["l_knee"],
    9:  MEDIAPIPE_JOINTS["r_knee"],
    10: MEDIAPIPE_JOINTS["l_ankle"],
    11: MEDIAPIPE_JOINTS["r_ankle"],
    12: (MEDIAPIPE_JOINTS["l_hip"], MEDIAPIPE_JOINTS["r_hip"]),           # pelvis
    13: "neck_weighted",   # B5 fix: 0.75*shoulder_mid + 0.25*ear_mid (see normalize.py)
    14: None,              # spine_mid computed as midpoint(pelvis, neck) after extraction
}

# ── 10 joint groups for per-joint error reporting ─────────────────────────────
JOINT_GROUPS = [
    "l_elbow",      # 0
    "r_elbow",      # 1
    "l_shoulder",   # 2
    "r_shoulder",   # 3
    "l_knee",       # 4
    "r_knee",       # 5
    "l_hip",        # 6
    "r_hip",        # 7
    "trunk",        # 8  (spine_mid + neck + pelvis)
    "neck",         # 9
]
N_JOINT_GROUPS = len(JOINT_GROUPS)  # 10

# ── Skeleton connections for visualisation (pairs of canonical indices) ────────
SKELETON_CONNECTIONS = [
    (0, 2),   # l_shoulder → l_elbow
    (2, 4),   # l_elbow    → l_wrist
    (1, 3),   # r_shoulder → r_elbow
    (3, 5),   # r_elbow    → r_wrist
    (0, 1),   # l_shoulder → r_shoulder  (collar)
    (0, 6),   # l_shoulder → l_hip
    (1, 7),   # r_shoulder → r_hip
    (6, 8),   # l_hip      → l_knee
    (8, 10),  # l_knee     → l_ankle
    (7, 9),   # r_hip      → r_knee
    (9, 11),  # r_knee     → r_ankle
    (6, 7),   # l_hip      → r_hip       (pelvis)
    (12, 14), # pelvis     → spine_mid
    (14, 13), # spine_mid  → neck
]
