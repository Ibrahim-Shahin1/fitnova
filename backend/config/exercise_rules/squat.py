"""Geometric form rules for squat.

Coordinate system reminder
--------------------------
Landmarks are MediaPipe image-normalised (x, y, z) in the [0, 1] range:
  x: 0 = left edge of frame,  1 = right edge
  y: 0 = TOP of frame,        1 = BOTTOM   (gravity points to higher y)
  z: relative depth (rarely used)

So during a squat:
  STANDING  → hip_y is SMALL (high in frame)
  BOTTOM    → hip_y is LARGE (low in frame)

Canonical 15-joint indices (from joint_mapping.CANONICAL_JOINTS):
   0 l_shoulder    1 r_shoulder    2 l_elbow      3 r_elbow
   4 l_wrist       5 r_wrist       6 l_hip        7 r_hip
   8 l_knee        9 r_knee       10 l_ankle     11 r_ankle
  12 pelvis (hip mid)            13 neck       14 spine_mid

10-group joint_errors output indices (from joint_mapping.JOINT_GROUPS):
   0 l_elbow     1 r_elbow     2 l_shoulder    3 r_shoulder
   4 l_knee      5 r_knee      6 l_hip         7 r_hip
   8 trunk       9 neck
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Canonical joint indices — keep here so rule files don't import joint_mapping
# (avoids circular imports during early-init)
# ─────────────────────────────────────────────────────────────────────────────

L_SHOULDER, R_SHOULDER = 0, 1
L_HIP, R_HIP           = 6, 7
L_KNEE, R_KNEE         = 8, 9
L_ANKLE, R_ANKLE       = 10, 11
PELVIS, NECK, SPINE_MID = 12, 13, 14

# Joint-group output indices
G_L_KNEE, G_R_KNEE = 4, 5
G_L_HIP, G_R_HIP   = 6, 7
G_TRUNK            = 8


# ─────────────────────────────────────────────────────────────────────────────
# Rule config (consumed by GeometricFormValidator)
# ─────────────────────────────────────────────────────────────────────────────


SQUAT_RULES = {
    # ── Which joint groups matter for this exercise ─────────────────────────
    # The session uses this to FILTER the merged ML+geometric joint_errors
    # before sending to Flutter. Reason: v6's ML head over-fires on
    # shoulders/elbows/neck even during a squat (an unfixable degeneracy
    # without retraining). Restricting to the biomechanically-relevant set
    # makes the UI feel correct.
    #
    # Indices correspond to JOINT_GROUPS (form_session.JOINT_GROUP_NAMES):
    #   0 L_Elbow  1 R_Elbow  2 L_Shoulder  3 R_Shoulder
    #   4 L_Knee   5 R_Knee   6 L_Hip       7 R_Hip
    #   8 Trunk    9 Neck
    "relevant_joint_groups": [G_L_KNEE, G_R_KNEE, G_L_HIP, G_R_HIP, G_TRUNK],

    # ── Rep-counter config ──────────────────────────────────────────────────
    "rep_tracker": {
        # Track the pelvis (midpoint of L/R hip) — stable proxy for body
        # vertical position during a squat. Ankles also work but are
        # affected by foot lift / heel rise; pelvis is more robust.
        "key_joint_idx": PELVIS,
        "axis":          "y",          # image y (lower y = higher in space)

        # Calibration: how many frames to observe before locking the
        # "standing baseline" Y. Longer = more stable but slower first rep.
        "calibration_frames": 20,

        # Trigger phase changes when the change exceeds these thresholds
        # (units: image-y, fraction of frame height)
        # Squat descent of ~12% of frame height is conservative — even
        # someone taking up only the middle 50% of the frame produces a
        # ~24% body-height descent during a real squat, which corresponds
        # to ~12% of frame.
        "descent_threshold":          0.07,  # must descend by this to enter DESCENDING
        "velocity_reversal_threshold": 0.015, # peak hip_y - current hip_y to detect bottom
        "return_threshold":           0.04,   # within this of baseline = STANDING again

        # Minimum total descent to count as a "real rep" (filters out
        # bobbing, fidgeting). Bottom_y - standing_y must exceed this.
        "min_rep_displacement": 0.08,

        # Trigger policy: "completion" fires +1 only after ASCENDING returns
        # to STANDING (most reliable, never double-counts), "commitment"
        # fires +1 the moment STANDING → DESCENDING transitions.
        "trigger": "completion",
    },

    # ── Form-quality rules ──────────────────────────────────────────────────
    #
    # Each rule emits a categorical severity label instead of a 0-1 number.
    # The Flutter UI shows the LABEL (e.g. "Small Knee Caving"), not the raw
    # severity float. The 0-1 severity is still used internally to:
    #   - rank rules (more severe wins display priority)
    #   - drive joint-error heat in the skeleton overlay
    #   - compute the quality penalty
    #
    # Severity → label mapping (per rule):
    #   < tier_thresholds[0]   → not active (don't show)
    #   < tier_thresholds[1]   → labels[0]   (mild)
    #   < tier_thresholds[2]   → labels[1]   (moderate)
    #   >= tier_thresholds[2]  → labels[2]   (dangerous)
    "form_checks": [
        # Excessive forward lean / back rounded
        # Trunk angle (hip→neck vector) from vertical UP. At top of squat
        # should be < 25°; at bottom can reach ~50° for a deep squat. > 70°
        # = severe forward lean / lumbar flexion.
        {
            "name":             "back_excessive_lean",
            "display_name":     "Forward Lean",
            "compute":          "trunk_angle_from_vertical",
            "thresholds":       {
                "safe_max_deg":  60.0,    # below this is fine
                "red_flag_deg":  85.0,    # above this is severe
            },
            "joint_groups":     [G_TRUNK],
            "penalty":          0.30,     # quality penalty when active
            "severity_tiers":   [0.30, 0.60, 0.85],
            "severity_labels":  [
                "Slight Forward Lean",
                "Notable Forward Lean",
                "Dangerous Forward Lean",
            ],
        },

        # Knee valgus (knees caving inward)
        # ratio = dist(L_knee, R_knee) / dist(L_ankle, R_ankle)
        #
        # IMPORTANT (per user feedback 2026-05-11): the threshold used to be
        # too sensitive (safe_min=0.75 was firing on most reasonable squats).
        # Real biomechanics: it's safe for knees to track in slightly from
        # the feet, as long as they stay roughly over the line of the toes.
        # We raise tolerance dramatically:
        #   ratio >= 0.55  → silently fine (knees within "in line with toes")
        #   0.40 - 0.55    → "Small Knee Caving"
        #   0.25 - 0.40    → "Notable Knee Caving"
        #   < 0.25         → "Dangerous Knee Caving"
        {
            "name":             "knees_caving",
            "display_name":     "Knee Caving",
            "compute":          "knee_to_ankle_width_ratio",
            "thresholds":       {
                "safe_min":      0.55,    # was 0.75 — too aggressive
                "red_flag_min":  0.25,
            },
            "joint_groups":     [G_L_KNEE, G_R_KNEE],
            "penalty":          0.25,
            "severity_tiers":   [0.30, 0.60, 0.85],
            "severity_labels":  [
                "Small Knee Caving",
                "Notable Knee Caving",
                "Dangerous Knee Caving",
            ],

            # Side-view filter: require minimum ankle-ankle separation
            # (ankles overlap horizontally in side view; the ratio is
            # meaningless there). Note we now filter on ANKLE separation
            # (not knee) — knees may have collapsed but ankles are still
            # planted at stance width in a front view.
            "min_horizontal_separation": 0.05,
        },

        # Insufficient depth (only checked at BOTTOM phase)
        # At the bottom of a real squat, hip_y should be at or below
        # knee_y (pelvis y >= knee y in image coords). If pelvis stays
        # above knees throughout the rep, it's a quarter squat.
        {
            "name":             "insufficient_depth",
            "display_name":     "Squat Depth",
            "compute":          "hip_above_knee_at_bottom",
            "thresholds":       {
                "shallow_margin": -0.06,   # hip 6% of frame above knee
            },
            "joint_groups":     [G_L_HIP, G_R_HIP],
            "penalty":          0.15,
            "fire_only_at":     "BOTTOM",
            "severity_tiers":   [0.30, 0.65, 0.85],
            "severity_labels":  [
                "Shallow Squat",
                "Quarter Squat",
                "Too Shallow — Bend Knees More",
            ],
        },
    ],

    # ── Quality calculation ────────────────────────────────────────────────
    # Final quality = max(0, 1.0 - sum of active rule penalties)
    # Multiple simultaneous flags compound (capped at 0).
    "quality_floor": 0.0,
    "quality_ceil":  1.0,

    # ── Overall quality → categorical UI label ─────────────────────────────
    # Replaces the percentage bar in the Flutter UI.
    # Pick the LAST tier whose lower_bound is <= quality.
    "quality_labels": [
        (0.85, "Excellent Form"),
        (0.70, "Good Form"),
        (0.50, "Fair Form"),
        (0.30, "Needs Work"),
        (0.00, "Bad Form"),
    ],
}
