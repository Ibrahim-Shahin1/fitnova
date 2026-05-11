"""v6.1 tf.data pipeline — paper-faithful multi-label classification.

Forks the relevant bits of ``qevd_dataset.py`` (v6) to emit the v6.1
target shape:

    inputs  = {pose: (T, 15, 4), angles: (T, 22), exercise_id: ()}
    targets = (num_classes,)  float32  multi-label sigmoid target

vs v6's five-head target dict. Reuses the same split logic,
``load_clip_npz``, and ``resample_sequence`` so the .npz files extracted
by ``qevd_extractor.py`` work unchanged.

Filters applied per clip (see plan §6 + §9):
  1. .npz must exist on disk in one of ``npz_dirs``.
  2. Clip's exercise prefix must be in the v6.1 cleaned class space
     (otherwise no usable supervision — clip dropped, not zero-target).
  3. Multi-label target must have at least one positive class
     (the cleanup pass drops "not visible" + long-tail variants, and
     ~11% of clips end up with all their labels filtered out — those
     clips are dropped, not trained as all-negatives).

See plan §3 for why this differs from v6's silence-default approach.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from backend.training.preprocessing.qevd_class_space import (
    QEVDClassSpace,
    get_default_class_space,
)
from backend.training.preprocessing.qevd_dataset import (
    DEFAULT_N_ANGULAR,
    DEFAULT_N_JOINTS,
    DEFAULT_N_POSE_CHANNELS,
    DEFAULT_TARGET_FRAMES,
    load_clip_npz,
)

logger = logging.getLogger(__name__)


def derive_v6_1_target(
    labels_list: Sequence[str],
    exercise_prefix: Optional[str],
    class_space: QEVDClassSpace,
) -> tuple[Optional[np.ndarray], Optional[int]]:
    """Build the v6.1 supervised target for one clip.

    Args:
        labels_list: Raw label strings from the QEVD manifest, e.g.
            ["squats - shallow", "squats - back not straight"].
        exercise_prefix: The clip's exercise prefix (used for the
            embedding input — independent of the multi-label target).
            Typically derived from the first label's prefix or from a
            separate `exercise` field on the manifest record.
        class_space: v6.1 class-index loader.

    Returns:
        (target, prefix_idx). Either may be None:
          - target is None when no labels survived cleanup (clip should
            be filtered out at the dataset level — it has no usable
            supervision).
          - prefix_idx is None when the exercise prefix was dropped by
            cleanup (also filter out — embedding input would be a
            sentinel).
        When both are non-None, the clip is fit to train. The target
        is shape (num_classes,) float32 with 1.0 at positive class
        indices.
    """
    target = class_space.encode_labels(labels_list or [])
    if target.sum() == 0:
        return None, None

    prefix = exercise_prefix
    if prefix is None and labels_list:
        # Fallback: parse the first label's prefix.
        first = labels_list[0]
        prefix = first.split(" - ", 1)[0].strip() if " - " in first else first.strip()
    prefix_idx = class_space.prefix_idx_for(prefix) if prefix else None
    if prefix_idx is None:
        return None, None

    return target, prefix_idx


def _resample_inputs(
    clip_npz: dict,
    target_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample pose + angles to target_frames; return (pose, angles)."""
    from backend.training.preprocessing.normalize import resample_sequence

    pose = clip_npz["pose_canon"].astype(np.float32)
    angles = clip_npz["angles_raw"].astype(np.float32)
    pose_r = resample_sequence(pose, target_frames).astype(np.float32)
    angles_r = resample_sequence(angles, target_frames).astype(np.float32)
    return pose_r, angles_r


def build_v6_1_sample(
    clip_npz: dict,
    target: np.ndarray,
    prefix_idx: int,
    target_frames: int = DEFAULT_TARGET_FRAMES,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Assemble one (inputs, target) pair for the v6.1 dataset."""
    pose_r, angles_r = _resample_inputs(clip_npz, target_frames)
    inputs = {
        "pose":        pose_r,
        "angles":      angles_r,
        "exercise_id": np.int32(prefix_idx),
    }
    return inputs, target.astype(np.float32)


def make_v6_1_clip_dataset(
    npz_dirs: Iterable[Path],
    clip_ids: Iterable[str],
    labels,                                # QEVDLabels-like (has .lookup(cid))
    class_space: Optional[QEVDClassSpace] = None,
    target_frames: int = DEFAULT_TARGET_FRAMES,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
):
    """Build a ``tf.data.Dataset`` for v6.1 training/eval.

    Yields ``(inputs, target)`` where ``target`` is a single
    (num_classes,) float32 vector — NOT the v6 multi-task dict.

    Clips are filtered out (silently skipped, with a counter logged at
    end of generation) when:
      - .npz is missing in all ``npz_dirs``
      - target has zero positive labels (all filtered by cleanup)
      - exercise prefix is not in the cleaned class space
      - .npz load fails (corrupt file, etc.)

    The dataset is NOT cached; the generator is re-run per epoch. Use
    ``tf.data.Dataset.cache()`` upstream if disk I/O is the bottleneck
    (it is not on Colab L4 with the current Drive throughput).
    """
    import tensorflow as tf

    cs = class_space or get_default_class_space()
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
        n_yielded = 0
        n_skip_npz = 0
        n_skip_target = 0
        n_skip_prefix = 0
        n_skip_load = 0
        for cid in ids:
            npz_path = _resolve_npz(cid)
            if npz_path is None:
                n_skip_npz += 1
                continue
            try:
                clip_data = load_clip_npz(npz_path)
            except Exception as e:
                logger.warning("skipping %s: %s", npz_path, e)
                n_skip_load += 1
                continue
            rec = labels.lookup(cid)
            target, prefix_idx = derive_v6_1_target(
                rec.get("labels", []),
                rec.get("exercise"),
                cs,
            )
            if target is None:
                # Could be either reason; finer-grained accounting not
                # worth the complexity here — both are "drop the clip".
                if not rec.get("labels"):
                    n_skip_target += 1
                else:
                    n_skip_target += 1
                continue
            if prefix_idx is None:
                n_skip_prefix += 1
                continue
            inputs, tgt = build_v6_1_sample(clip_data, target, prefix_idx, target_frames)
            n_yielded += 1
            yield inputs, tgt
        # End-of-epoch accounting helps catch dataset-shrink regressions.
        logger.info(
            "v6_1 gen: yielded=%d skip_npz=%d skip_target=%d "
            "skip_prefix=%d skip_load=%d",
            n_yielded, n_skip_npz, n_skip_target, n_skip_prefix, n_skip_load,
        )

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
        tf.TensorSpec(shape=(cs.num_classes,), dtype=tf.float32, name="target"),
    )

    ds = tf.data.Dataset.from_generator(_gen, output_signature=output_signature)
    if batch_size and batch_size > 1:
        ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


__all__ = [
    "build_v6_1_sample",
    "derive_v6_1_target",
    "make_v6_1_clip_dataset",
]
