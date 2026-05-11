"""QEVD v6 dataset assembly: split logic + sample weighting.

This is a SKELETON file: ``tf.data.Dataset`` wiring is deferred to D4
(needs a real qevd_extracted/ directory), but the subject-disjoint split
and stratified-weight logic is implemented + unit-tested here so D4
can pick it up unchanged.

See:
- plans/i-am-now-on-zazzy-brooks.md section II.4.5
- plans/phase-1-complete-critical-snappy-flurry.md section D0.7
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Source labels for the four splits
# ─────────────────────────────────────────────────────────────────────────────

SplitName = Literal["train", "val", "in_domain_test", "ood_test", "unused"]
ALL_SPLITS: tuple[SplitName, ...] = (
    "train", "val", "in_domain_test", "ood_test", "unused",
)

# Per master plan section II.4.5: train + val drawn from FIT-300K train (subject-
# disjoint); in_domain_test from FIT-300K test split; ood_test from FIT-COACH test.
# Anything else (e.g. FIT-COACH train) is unused for the supervised loop.
SourceLabel = Literal[
    "fit300k_train",
    "fit300k_test",
    "fitcoach_train",
    "fitcoach_test",
]

# ─────────────────────────────────────────────────────────────────────────────
# Subject-disjoint split for FIT-300K training subjects
# ─────────────────────────────────────────────────────────────────────────────


def split_subjects(
    human_ids: Sequence[str],
    val_frac: float = 0.09,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    """Subject-disjoint shuffle into train/val.

    Master plan default: 9% val, 91% train, deterministic on `seed`.
    Returns (train_humans, val_humans), guaranteed disjoint and
    union-covering of the input.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    ids = list(dict.fromkeys(human_ids))   # de-dupe, preserve order
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac)))
    val_set = set(ids[:n_val])
    train_set = set(ids[n_val:])
    assert val_set.isdisjoint(train_set)
    return train_set, val_set


# ─────────────────────────────────────────────────────────────────────────────
# Per-clip routing
# ─────────────────────────────────────────────────────────────────────────────


def assign_clip_to_split(
    clip_meta: dict,
    train_humans: set[str],
    val_humans: set[str],
) -> SplitName:
    """Route one clip to a split based on its source + human_id.

    `clip_meta` must contain at least `{source, human_id}` where `source`
    is one of the SourceLabel values. Anything outside the documented
    plan is routed to "unused" (e.g. FIT-COACH train, which is not used
    in the v6 supervised pipeline).
    """
    source = clip_meta.get("source", "")
    human = clip_meta.get("human_id", "")

    if source == "fit300k_train":
        if human in train_humans:
            return "train"
        if human in val_humans:
            return "val"
        return "unused"   # human not in either set (caller bug)
    if source == "fit300k_test":
        return "in_domain_test"
    if source == "fitcoach_test":
        return "ood_test"
    # fitcoach_train and any unknown source land here
    return "unused"


# ─────────────────────────────────────────────────────────────────────────────
# Stratified sample weights
# ─────────────────────────────────────────────────────────────────────────────


def stratified_sample_weights(
    labels: np.ndarray,
    qualities: np.ndarray,
    n_buckets: int = 4,
) -> np.ndarray:
    """Inverse-frequency weights over (exercise_idx x quality_bucket).

    Parameters
    ----------
    labels : (N,) int  -- exercise_idx
    qualities : (N,) float -- per-clip quality scalar in [0, 1]
    n_buckets : int -- number of quality strata

    Returns
    -------
    weights : (N,) float32, normalised so mean(weights) == 1.0.
    """
    labels = np.asarray(labels)
    qualities = np.asarray(qualities)
    N = labels.shape[0]
    if qualities.shape != (N,):
        raise ValueError(
            f"qualities shape {qualities.shape} != labels shape ({N},)"
        )
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be >= 1, got {n_buckets}")

    if N == 0:
        return np.zeros(0, dtype=np.float32)

    # Bucketise quality on edges [0, 1/n, 2/n, ..., 1]
    inner_edges = np.linspace(0.0, 1.0, n_buckets + 1)[1:-1]
    quality_bucket = np.digitize(qualities, inner_edges)
    quality_bucket = np.clip(quality_bucket, 0, n_buckets - 1).astype(np.int64)

    # Joint stratum key: ensure unique per (label, bucket)
    key = labels.astype(np.int64) * n_buckets + quality_bucket

    unique, counts = np.unique(key, return_counts=True)
    count_map = {int(u): int(c) for u, c in zip(unique, counts)}
    freq = np.array([count_map[int(k)] for k in key], dtype=np.float64)

    weights = 1.0 / freq
    # Normalise so mean == 1.0
    weights = weights * (N / weights.sum())
    return weights.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# QEVD-aware split builder (uses worker_ids = participants, verified 2026-05-06)
# ─────────────────────────────────────────────────────────────────────────────


def build_qevd_splits(
    clip_to_split: dict[str, str],
    clip_to_human: dict[str, str],
    *,
    val_frac: float = 0.09,
    seed: int = 42,
) -> dict[str, set[str]]:
    """Carve up FIT-300K clips into train / val / in_domain_test using
    *participant-disjoint* val carve-out within Qualcomm's `split=train`.

    Inputs are typically taken straight off ``QEVDLabels``::

        labels = QEVDLabels.from_files(..., worker_ids_path=...)
        splits = build_qevd_splits(labels.clip_to_split, labels.clip_to_human)
        train_clips = splits['train']
        val_clips   = splits['val']

    The function is fully deterministic on ``seed`` and is conservative on
    missing data:
      * clips with no `split` value are routed to "unused"
      * clips with no `human_id` (e.g. worker_ids manifest absent) are still
        routed by Qualcomm's split — but val carve-out can't apply, so they
        end up in train. A warning is logged with the count.

    Parameters
    ----------
    clip_to_split : dict[str, str]
        Maps clip_id -> "train" | "test" (Qualcomm's pre-defined split).
    clip_to_human : dict[str, str]
        Maps clip_id -> worker_id. Empty dict is allowed (degraded mode).
    val_frac : float
        Fraction of TRAIN PARTICIPANTS to hold out for validation.
    seed : int

    Returns
    -------
    dict with keys "train", "val", "in_domain_test", "unused" — each
    mapping to a set of clip_ids. The four sets are disjoint and their
    union covers ``clip_to_split.keys()``.
    """
    import logging
    logger = logging.getLogger(__name__)

    # 1. Discover the train participant set
    train_humans_full = {
        clip_to_human[cid]
        for cid, s in clip_to_split.items()
        if s == "train" and cid in clip_to_human
    }
    n_clips_no_human = sum(
        1 for cid, s in clip_to_split.items()
        if s == "train" and cid not in clip_to_human
    )
    if n_clips_no_human > 0:
        logger.warning(
            "build_qevd_splits: %d train clips have no worker_id; "
            "routing to 'train' without val carve-out", n_clips_no_human,
        )

    # 2. Subject-disjoint val carve-out
    if train_humans_full:
        train_humans, val_humans = split_subjects(
            sorted(train_humans_full), val_frac=val_frac, seed=seed,
        )
    else:
        train_humans, val_humans = set(), set()

    # 3. Walk every clip and route
    out: dict[str, set[str]] = {
        "train": set(), "val": set(),
        "in_domain_test": set(), "unused": set(),
    }
    for cid, s in clip_to_split.items():
        if s == "test":
            out["in_domain_test"].add(cid)
            continue
        if s != "train":
            out["unused"].add(cid)
            continue
        # split == "train": route by participant if known, else default train
        wid = clip_to_human.get(cid)
        if wid is None:
            out["train"].add(cid)
        elif wid in val_humans:
            out["val"].add(cid)
        else:
            out["train"].add(cid)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# v6 model contract (matches backend.training.models.st_gcn_v6)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TARGET_FRAMES   = 64
DEFAULT_N_JOINTS        = 15
DEFAULT_N_POSE_CHANNELS = 4    # x, y, z, visibility
DEFAULT_N_ANGULAR       = 22
DEFAULT_N_JOINT_GROUPS  = 10   # matches form_session.JOINT_GROUP_NAMES
DEFAULT_N_EXERCISES     = 25   # 24 named QEVD exercises + 1 "other"


# ─────────────────────────────────────────────────────────────────────────────
# .npz loader
# ─────────────────────────────────────────────────────────────────────────────


def load_clip_npz(path: Path) -> dict:
    """Load a single .npz produced by qevd_extractor.

    Returns a dict with keys: pose_canon (T,15,4), angles_raw (T,22),
    fps_native (float), status_per_frame (T,), no_pose_fraction (float).
    """
    with np.load(path) as z:
        return {
            "pose_canon":       z["pose_canon"].copy(),
            "angles_raw":       z["angles_raw"].copy(),
            "fps_native":       float(z["fps_native"]),
            "status_per_frame": z["status_per_frame"].copy(),
            "no_pose_fraction": float(z["no_pose_fraction"]),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Exercise mapping (24 named QEVD exercises + 1 "other")
# ─────────────────────────────────────────────────────────────────────────────


OTHER_KEY = "__other__"


def build_exercise_map(
    labels,                              # backend.training.preprocessing.qevd_label_builder.QEVDLabels
    n_named: int = DEFAULT_N_EXERCISES - 1,
    fixed_exercises: Optional[Sequence[str]] = None,
) -> dict[str, int]:
    """Build {exercise_name -> idx} for the action head.

    Top-N most frequent exercises in the labels manifest get a dedicated
    index 0..n_named-1. Everything else maps to 'other' = n_named.

    Pass `fixed_exercises` when reproducing an existing run (so the
    mapping doesn't drift if clip counts shift between manifest versions).
    """
    if fixed_exercises is not None:
        top = list(fixed_exercises)[:n_named]
    else:
        counts: Counter = Counter()
        for rec in labels.clip_to_fine.values():
            ex = rec.get("exercise")
            if ex:
                counts[ex] += 1
        top = [e for e, _ in counts.most_common(n_named)]
    out = {ex: i for i, ex in enumerate(top)}
    out[OTHER_KEY] = n_named
    return out


def save_exercise_map(mapping: dict[str, int], path: Path) -> None:
    """Persist the exercise map to JSON for later reload."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)


def load_exercise_map(path: Path) -> dict[str, int]:
    """Load an exercise map written by save_exercise_map."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def map_exercise_to_idx(
    exercise_name: Optional[str],
    mapping: dict[str, int],
) -> int:
    """Look up an exercise name; unknown -> 'other' index."""
    if exercise_name is None or exercise_name not in mapping:
        return int(mapping[OTHER_KEY])
    return int(mapping[exercise_name])


# ─────────────────────────────────────────────────────────────────────────────
# Per-clip target tensors
# ─────────────────────────────────────────────────────────────────────────────


def derive_v6_targets(
    feedbacks: Sequence,
    labels_list: Sequence[str],
    exercise_name: Optional[str],
    exercise_map: dict[str, int],
    target_frames: int = DEFAULT_TARGET_FRAMES,
    n_joint_groups: int = DEFAULT_N_JOINT_GROUPS,
    pose_canon: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Build the supervised-target tensors for one clip.

    Outputs (matching backend.training.models.st_gcn_v6 head shapes):
      quality   : (1,)        clipped to [0, 1]
      action    : ()          int32 idx into exercise_map
      rep_count : (1,)        weak supervision via hip-Y peaks (or 1.0 fallback)
      boundary  : (T, 1)      per-frame rep-boundary signal
      joint_err : (T, J)      per-frame multi-label, currently clip-tiled
    """
    # Lazy import to avoid forcing ST-GCN deps when only split helpers used.
    from backend.training.preprocessing.qevd_label_builder import (
        compute_clip_quality, derive_joint_groups, derive_joint_groups_from_class,
        N_JOINT_GROUPS,
    )

    quality = compute_clip_quality(feedbacks, list(labels_list))
    action_idx = map_exercise_to_idx(exercise_name, exercise_map)

    # Joint groups: union of feedback-derived + class-derived (10-vec bool)
    groups = np.zeros(n_joint_groups, dtype=np.float32)
    for fb in feedbacks or []:
        text = fb if isinstance(fb, str) else fb.get("text", "")
        groups = np.maximum(groups, derive_joint_groups(text)[:n_joint_groups]
                                       .astype(np.float32))
    for lbl in labels_list or []:
        groups = np.maximum(
            groups,
            derive_joint_groups_from_class(lbl)[:n_joint_groups].astype(np.float32),
        )
    # Tile across T frames (FIT-300K has clip-level annotations only).
    joint_err = np.tile(groups, (target_frames, 1)).astype(np.float32)

    # Rep-count + per-frame boundary derived from hip-Y trajectory if pose is
    # available.  Master plan section II.4.4 (FIT-300K weak supervision).
    if pose_canon is not None and pose_canon.shape[0] >= 8:
        rep_count_val, boundary_seq = _hip_y_peaks(
            pose_canon, target_frames=target_frames,
        )
    else:
        rep_count_val = 1.0
        boundary_seq = np.zeros((target_frames, 1), dtype=np.float32)

    return {
        "quality":   np.array([quality],    dtype=np.float32),
        "action":    np.int32(action_idx),
        "rep_count": np.array([rep_count_val], dtype=np.float32),
        "boundary":  boundary_seq,
        "joint_err": joint_err,
    }


def _hip_y_peaks(
    pose_canon: np.ndarray,
    target_frames: int,
) -> tuple[float, np.ndarray]:
    """Estimate rep-count + per-frame boundary signal from hip-Y peaks.

    Inputs
    ------
    pose_canon : (T_orig, 15, 4)  canonical pose with visibility
    target_frames : int           length of the resampled boundary sequence

    Returns
    -------
    rep_count : float
    boundary  : (target_frames, 1) float32, smoothed Gaussian around each peak
    """
    try:
        from scipy.signal import find_peaks
    except ImportError:
        return 1.0, np.zeros((target_frames, 1), dtype=np.float32)

    # Negative hip-Y so squat *descents* (low Y in image-coords) become peaks
    pelv_y = -pose_canon[:, 12, 1]   # joint index 12 = pelvis
    if pelv_y.shape[0] < 8 or float(np.std(pelv_y)) < 1e-6:
        return 1.0, np.zeros((target_frames, 1), dtype=np.float32)

    peaks, _ = find_peaks(pelv_y, distance=8, prominence=0.05)
    n_peaks = max(1, int(len(peaks)))

    # Map peak indices from T_orig to target_frames, smooth with Gaussian
    T = pose_canon.shape[0]
    mapped = (peaks * (target_frames - 1) / max(1, T - 1)).astype(int)
    boundary = np.zeros(target_frames, dtype=np.float32)
    sigma = 1.5
    xs = np.arange(target_frames)
    for p in mapped:
        boundary = np.maximum(
            boundary,
            np.exp(-0.5 * ((xs - p) / sigma) ** 2).astype(np.float32),
        )
    return float(n_peaks), boundary[:, None]


# ─────────────────────────────────────────────────────────────────────────────
# Resample + assemble inputs
# ─────────────────────────────────────────────────────────────────────────────


def build_v6_sample(
    clip_npz: dict,
    targets: dict[str, np.ndarray],
    target_frames: int = DEFAULT_TARGET_FRAMES,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Resample inputs to target_frames; return (inputs, targets) tuple."""
    from backend.training.preprocessing.normalize import resample_sequence

    pose   = clip_npz["pose_canon"].astype(np.float32)
    angles = clip_npz["angles_raw"].astype(np.float32)
    pose_r   = resample_sequence(pose,   target_frames).astype(np.float32)
    angles_r = resample_sequence(angles, target_frames).astype(np.float32)
    inputs = {
        "pose":        pose_r,
        "angles":      angles_r,
        "exercise_id": np.int32(targets["action"]),
    }
    return inputs, targets


# ─────────────────────────────────────────────────────────────────────────────
# tf.data.Dataset builder
# ─────────────────────────────────────────────────────────────────────────────


def make_clip_dataset(
    npz_dirs: Iterable[Path],
    clip_ids: Iterable[str],
    labels,                                  # QEVDLabels
    exercise_map: dict[str, int],
    target_frames: int = DEFAULT_TARGET_FRAMES,
    n_joint_groups: int = DEFAULT_N_JOINT_GROUPS,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
):
    """Build a ``tf.data.Dataset`` that yields ``(inputs, targets)`` for
    ``keras.Model.fit``.

    Parameters
    ----------
    npz_dirs : iterable of Path
        Directories that hold ``<clip_id>.npz`` files. Searched in order
        per clip until a hit. (Allows you to point at multiple Part-N
        directories without copying them.)
    clip_ids : iterable of str
        Subset to include in this dataset (e.g. the train clip IDs from
        ``build_qevd_splits``).  Each ID is normalised via the
        QEVDLabels rules (zero-padded 8-digit form for FIT-300K).
    labels : QEVDLabels
        Loaded label manifest used to look up exercise/feedbacks/labels.
    exercise_map : dict[str, int]
        Output of ``build_exercise_map``.
    """
    # tf import is local so the rest of this module works without TF.
    import tensorflow as tf

    npz_dirs = [Path(d) for d in npz_dirs]
    clip_ids_list = [str(c) for c in clip_ids]

    def _resolve_npz(cid: str) -> Optional[Path]:
        cid_padded = cid.zfill(8) if cid.isdigit() else cid
        for d in npz_dirs:
            p = d / f"{cid_padded}.npz"
            if p.exists():
                return p
        return None

    def _gen():
        rng = np.random.default_rng(seed if shuffle else None)
        ids = list(clip_ids_list)
        if shuffle:
            rng.shuffle(ids)
        for cid in ids:
            npz_path = _resolve_npz(cid)
            if npz_path is None:
                continue
            try:
                clip_data = load_clip_npz(npz_path)
            except Exception as e:
                logger.warning("skipping %s: %s", npz_path, e)
                continue
            rec = labels.lookup(cid)
            targets = derive_v6_targets(
                rec.get("feedbacks", []),
                rec.get("labels", []),
                rec.get("exercise"),
                exercise_map,
                target_frames=target_frames,
                n_joint_groups=n_joint_groups,
                pose_canon=clip_data["pose_canon"],
            )
            inputs, tgts = build_v6_sample(clip_data, targets, target_frames)
            yield inputs, tgts

    output_signature = (
        {
            "pose":        tf.TensorSpec(
                shape=(target_frames, DEFAULT_N_JOINTS, DEFAULT_N_POSE_CHANNELS),
                dtype=tf.float32, name="pose"),
            "angles":      tf.TensorSpec(
                shape=(target_frames, DEFAULT_N_ANGULAR),
                dtype=tf.float32, name="angles"),
            "exercise_id": tf.TensorSpec(
                shape=(), dtype=tf.int32, name="exercise_id"),
        },
        {
            "quality":   tf.TensorSpec(shape=(1,),                  dtype=tf.float32),
            "action":    tf.TensorSpec(shape=(),                    dtype=tf.int32),
            "rep_count": tf.TensorSpec(shape=(1,),                  dtype=tf.float32),
            "boundary":  tf.TensorSpec(shape=(target_frames, 1),    dtype=tf.float32),
            "joint_err": tf.TensorSpec(
                shape=(target_frames, n_joint_groups), dtype=tf.float32),
        },
    )

    ds = tf.data.Dataset.from_generator(_gen, output_signature=output_signature)
    if batch_size and batch_size > 1:
        ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


__all__ = [
    "ALL_SPLITS",
    "DEFAULT_N_ANGULAR",
    "DEFAULT_N_EXERCISES",
    "DEFAULT_N_JOINTS",
    "DEFAULT_N_JOINT_GROUPS",
    "DEFAULT_N_POSE_CHANNELS",
    "DEFAULT_TARGET_FRAMES",
    "OTHER_KEY",
    "SourceLabel",
    "SplitName",
    "assign_clip_to_split",
    "build_exercise_map",
    "build_qevd_splits",
    "build_v6_sample",
    "derive_v6_targets",
    "load_clip_npz",
    "load_exercise_map",
    "make_clip_dataset",
    "map_exercise_to_idx",
    "save_exercise_map",
    "split_subjects",
    "stratified_sample_weights",
]
