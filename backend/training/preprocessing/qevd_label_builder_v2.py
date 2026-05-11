"""v6.1 label builder — paper-faithful multi-label classification targets.

Replaces the v6 lexicon classifier (`qevd_label_builder.py`). The v6
pipeline derived a 1-D quality scalar + 10-bit joint-group bool from
free-text via regex, throwing away the QEVD paper's native 1,851-class
variation taxonomy. v6.1 fixes that: each clip's `labels` list is mapped
to discrete class indices in the cleaned class space, and the target is
a `(num_classes,)` float32 multi-label vector for sigmoid + BCE.

There is NO quality scalar, NO joint-group derivation, NO boundary
target, NO action softmax. The model's single multi-label head is the
sole supervisory signal. Coaching language is generated downstream by
GPT-4o-mini at runtime, not derived here.

See plan §6 for the full architecture rationale.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from backend.training.preprocessing.qevd_class_space import (
    QEVDClassSpace,
    get_default_class_space,
)


def build_v6_1_target(
    labels: Sequence[str],
    class_space: QEVDClassSpace | None = None,
) -> np.ndarray:
    """Encode a clip's raw labels as a multi-label sigmoid target.

    Args:
        labels: Raw label strings from `fine_grained_labels.json`, e.g.
            ["squats - shallow", "squats - back not straight"].
        class_space: Class-index loader. Defaults to the cached singleton
            for the v6.1 spec.

    Returns:
        np.ndarray of shape (class_space.num_classes,), dtype float32.
        1.0 for every label that survived the cleanup pass; 0.0 for the
        rest. Labels filtered out by the cleanup (e.g. 'not visible',
        long-tail variants, single-variation general-activity prefixes)
        contribute nothing — they are silently skipped, not encoded as
        zero targets. The downstream BCE loss treats zero targets as
        "negative class" because the cleanup retains 92% of all label
        occurrences.
    """
    cs = class_space or get_default_class_space()
    return cs.encode_labels(labels)


def build_v6_1_clip_record(
    clip_record: dict,
    class_space: QEVDClassSpace | None = None,
    *,
    keep_filtered_labels: bool = True,
) -> dict:
    """Build a per-clip dict with target + traceability fields.

    Args:
        clip_record: A record from `fine_grained_labels.json`. Must have
            `video_path` and `labels` keys; may also have `split` and
            `labels_descriptive`.
        class_space: Class-index loader (default singleton).
        keep_filtered_labels: Include the labels that were filtered out
            by cleanup in the output (under `filtered_labels`). Useful
            for D7 evaluation reports that need to know "this clip's
            ground-truth was filtered, expect the model to predict
            nothing on it."

    Returns:
        Dict with:
          - clip_id: int (parsed from "00012345" in video_path)
          - video_path: str (raw)
          - split: str ("train" | "test")
          - labels: list[str] (raw labels kept after cleanup)
          - filtered_labels: list[str] (raw labels dropped by cleanup),
              omitted if `keep_filtered_labels` is False
          - label_indices: list[int] (class indices for the kept labels)
          - target: np.ndarray (num_classes,), float32 multi-label
    """
    cs = class_space or get_default_class_space()

    video_path = clip_record.get("video_path", "")
    clip_id = _parse_clip_id(video_path)
    split = clip_record.get("split", "train")
    raw_labels: Sequence[str] = clip_record.get("labels", [])

    label_indices: list[int] = []
    kept: list[str] = []
    filtered: list[str] = []
    for label in raw_labels:
        idx = cs.class_idx_for_label(label)
        if idx is None:
            filtered.append(label)
        else:
            label_indices.append(idx)
            kept.append(label)

    target = np.zeros(cs.num_classes, dtype=np.float32)
    if label_indices:
        target[label_indices] = 1.0

    out = {
        "clip_id": clip_id,
        "video_path": video_path,
        "split": split,
        "labels": kept,
        "label_indices": label_indices,
        "target": target,
    }
    if keep_filtered_labels:
        out["filtered_labels"] = filtered
    return out


def stack_targets(
    records: Iterable[dict],
    class_space: QEVDClassSpace | None = None,
) -> np.ndarray:
    """Build a stacked (B, num_classes) target tensor from a record batch.

    Convenience for offline ablation / smoke-eval scripts. The training
    dataset pipeline (`qevd_dataset_v2.py`, Day 2) builds targets
    incrementally inside `tf.data.Dataset.from_generator` and does not
    use this helper.
    """
    cs = class_space or get_default_class_space()
    rows = [cs.encode_labels(r.get("labels", [])) for r in records]
    if not rows:
        return np.zeros((0, cs.num_classes), dtype=np.float32)
    return np.stack(rows, axis=0)


def _parse_clip_id(video_path: str) -> int:
    """Extract the integer clip id from a path like './00012345.mp4'.

    Returns -1 if parsing fails (caller should treat as "unknown" rather
    than crash; the dataset pipeline tolerates this).
    """
    if not video_path:
        return -1
    stem = video_path.rstrip("/").split("/")[-1]
    if stem.endswith(".mp4"):
        stem = stem[:-4]
    try:
        return int(stem)
    except ValueError:
        return -1
