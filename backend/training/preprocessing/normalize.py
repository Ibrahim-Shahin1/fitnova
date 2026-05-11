"""
Normalization utilities for Fit3D skeleton data.

1. extract_canonical_joints  — Fit3D 25-joint array → canonical 15-joint array
2. normalize_skeleton         — translate pelvis to origin, scale by torso length
3. resample_sequence          — linear interpolation to fixed T frames
4. build_angle_stats          — compute mean/std over training set
5. normalize_angles           — standardize using pre-computed stats

AXIS CONVENTION
---------------
v5.0 trained without any axis remap; v5.1 keeps that convention to maintain
SSL-encoder compatibility (the SSL pretrain weights are reused). A
``Y-up axis remap`` was tested and reverted: it didn't fix the underlying
angular-feature variance issue, and it would have caused a distribution
mismatch between the SSL pretrain and the supervised fine-tune. The
underlying issue (``thigh_vert`` having ~zero motion energy on real squats)
was traced not to the axis but to the joint indices in ``joint_mapping.py``
not matching the actual Fit3D 25-joint layout. v5.1 sidesteps the joint-
indexing problem entirely by replacing the AIFit-signature-derived labels
with synthetic-perturbation severity labels (see
``dataset_builder_v5_1.py``).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .joint_mapping import FIT3D_TO_CANONICAL, N_CANONICAL, MEDIAPIPE_TO_CANONICAL

TARGET_FRAMES = 64    # fixed temporal length after resampling
_EPS = 1e-8
# Minimum std clip for angle normalisation.
# Prevents divide-by-near-zero on angles that are near-constant across the training set
# (root cause of the std-explosion that produced -3601 outliers in v1).
# 0.05 rad (~2.9°) is well below real biomechanical variation and above noise floor.
_MIN_ANGLE_STD = 0.05


# ── 1. Extract canonical joints from Fit3D ────────────────────────────────────

def extract_canonical_from_fit3d(joints25: np.ndarray) -> np.ndarray:
    """
    Convert (T, 25, 3) Fit3D skeleton → (T, 15, 3) canonical skeleton.

    Entries that are tuples in FIT3D_TO_CANONICAL are averaged.
    Spine_mid (index 14) is computed as midpoint(pelvis, neck).

    NOTE on axis convention: this function passes raw Fit3D coordinates
    through unchanged. An earlier "Y-up remap" attempt was reverted because
    (a) it didn't actually fix the angular-feature variance issue (root cause
    was the joint indexing in joint_mapping.py, not the axis), and (b) it
    would have created a distribution mismatch with the v5.0 SSL encoder
    weights which we reuse.
    """
    T = joints25.shape[0]
    canonical = np.zeros((T, N_CANONICAL, 3), dtype=np.float32)

    for can_idx, fit_idx in FIT3D_TO_CANONICAL.items():
        if can_idx == 14:
            # spine_mid = midpoint of pelvis (12) and neck (13) in canonical
            # Will be filled after pelvis and neck are computed
            continue
        if isinstance(fit_idx, tuple):
            canonical[:, can_idx, :] = np.mean(
                [joints25[:, i, :] for i in fit_idx], axis=0
            )
        else:
            canonical[:, can_idx, :] = joints25[:, fit_idx, :]

    # spine_mid = midpoint(pelvis=12, neck=13)
    canonical[:, 14, :] = 0.5 * (canonical[:, 12, :] + canonical[:, 13, :])

    return canonical


def extract_canonical_from_mediapipe(landmarks_mp: np.ndarray) -> np.ndarray:
    """
    Convert (T, 33, 3) or (33, 3) MediaPipe world landmarks → (T, 15, 3) canonical.

    For a single frame pass shape (33, 3); returns (15, 3).

    B5 fix: canonical neck (index 13) is estimated as a weighted blend of the
    shoulder midpoint (75%) and ear midpoint (25%) to better match Fit3D's real
    C7 neck joint.  See joint_mapping.py for rationale.
    """
    from .joint_mapping import MEDIAPIPE_JOINTS

    single = landmarks_mp.ndim == 2
    if single:
        landmarks_mp = landmarks_mp[np.newaxis]  # (1, 33, 3)

    T = landmarks_mp.shape[0]
    canonical = np.zeros((T, N_CANONICAL, 3), dtype=np.float32)

    for can_idx, mp_idx in MEDIAPIPE_TO_CANONICAL.items():
        if can_idx == 14:
            continue  # spine_mid filled below
        if mp_idx is None:
            continue
        if mp_idx == "neck_weighted":
            # B5 fix: neck = 0.75 * shoulder_mid + 0.25 * ear_mid
            l_sho = landmarks_mp[:, MEDIAPIPE_JOINTS["l_shoulder"], :]
            r_sho = landmarks_mp[:, MEDIAPIPE_JOINTS["r_shoulder"], :]
            l_ear = landmarks_mp[:, MEDIAPIPE_JOINTS["l_ear"],      :]
            r_ear = landmarks_mp[:, MEDIAPIPE_JOINTS["r_ear"],      :]
            shoulder_mid = 0.5 * (l_sho + r_sho)
            ear_mid      = 0.5 * (l_ear + r_ear)
            canonical[:, can_idx, :] = 0.75 * shoulder_mid + 0.25 * ear_mid
        elif isinstance(mp_idx, tuple):
            canonical[:, can_idx, :] = np.mean(
                [landmarks_mp[:, i, :] for i in mp_idx], axis=0
            )
        else:
            canonical[:, can_idx, :] = landmarks_mp[:, mp_idx, :]

    # spine_mid = midpoint(pelvis=12, neck=13)
    canonical[:, 14, :] = 0.5 * (canonical[:, 12, :] + canonical[:, 13, :])

    return canonical[0] if single else canonical


# ── 2. Skeleton normalisation ─────────────────────────────────────────────────

def normalize_skeleton(canonical: np.ndarray) -> np.ndarray:
    """
    Per-frame skeleton normalisation:
      1. Translate so pelvis (index 12) is at origin.
      2. Scale by torso length (pelvis → neck distance).

    Args:
        canonical: (T, 15, 3)

    Returns:
        normalised: (T, 15, 3)  — same shape, position/scale invariant
    """
    out = canonical.copy()
    pelvis = out[:, 12:13, :]          # (T, 1, 3)
    out -= pelvis                       # translate to origin

    torso_len = np.linalg.norm(
        out[:, 13, :] - out[:, 12, :], axis=-1, keepdims=True
    )  # (T, 1)
    torso_len = np.clip(torso_len, _EPS, None)
    out /= torso_len[:, np.newaxis, :]  # (T, 1, 1) broadcast

    return out


# ── 3. Temporal resampling ────────────────────────────────────────────────────

def resample_sequence(seq: np.ndarray, target_len: int = TARGET_FRAMES) -> np.ndarray:
    """
    Linearly resample a sequence to target_len frames.

    Args:
        seq:        (T, ...) any shape where T is the temporal dimension
        target_len: desired output length

    Returns:
        resampled:  (target_len, ...)
    """
    T = seq.shape[0]
    if T == target_len:
        return seq.copy()

    src_idx = np.linspace(0, T - 1, target_len)
    lo = np.floor(src_idx).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    alpha = (src_idx - lo).reshape(-1, *([1] * (seq.ndim - 1)))  # broadcast

    return (1 - alpha) * seq[lo] + alpha * seq[hi]


# ── 4 & 5. Angular feature normalisation ─────────────────────────────────────

class AngleNormalizer:
    """Standardise angular features using per-dimension mean and std."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None   # (22,)
        self.std: Optional[np.ndarray] = None    # (22,)

    def fit(self, angles_list: List[np.ndarray]) -> None:
        """
        Compute mean/std from a list of (T, 22) angle arrays.
        Each array may have different T.

        std is clipped to MIN_ANGLE_STD to prevent divide-by-near-zero
        on near-constant angles (e.g. when an angle barely changes
        across a small training set).
        """
        all_angles = np.concatenate(angles_list, axis=0)  # (N_total_frames, 22)
        self.mean = all_angles.mean(axis=0).astype(np.float32)
        raw_std = all_angles.std(axis=0)
        self.std = np.clip(raw_std, _MIN_ANGLE_STD, None).astype(np.float32)

    def transform(self, angles: np.ndarray) -> np.ndarray:
        """
        Standardise (T, 22) or (22,) angles, then clip the output to [-8, 8]
        as a final safety net against any residual outliers (e.g. from
        corrupted joint positions during label synthesis).
        """
        assert self.mean is not None, "Call fit() first"
        out = (angles - self.mean) / self.std
        out = np.clip(out, -8.0, 8.0)
        return out.astype(np.float32)

    def inverse_transform(self, norm_angles: np.ndarray) -> np.ndarray:
        return (norm_angles * self.std + self.mean).astype(np.float32)

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "AngleNormalizer":
        data = np.load(path)
        obj = cls()
        obj.mean = data["mean"]
        obj.std = data["std"]
        return obj


# ── Convenience: full preprocessing for a single rep ─────────────────────────

def preprocess_rep_fit3d(
    rep_joints25: np.ndarray,
    normalizer: Optional[AngleNormalizer] = None,
    target_frames: int = TARGET_FRAMES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full pipeline for one Fit3D rep.

    Returns:
        angles_norm: (target_frames, 22) float32  — normalised angular features
        joints_norm: (target_frames, 15*3) float32 — flattened normalised joints
    """
    from .angular_features import compute_sequence_angles

    canonical = extract_canonical_from_fit3d(rep_joints25)   # (T, 15, 3)
    canonical = normalize_skeleton(canonical)                  # (T, 15, 3)
    canonical = resample_sequence(canonical, target_frames)    # (64, 15, 3)

    angles = compute_sequence_angles(canonical)                # (64, 22)
    if normalizer is not None:
        angles = normalizer.transform(angles)

    joints_flat = canonical.reshape(target_frames, -1)         # (64, 45)
    return angles, joints_flat
