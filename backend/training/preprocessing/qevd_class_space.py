"""Loader for the v6.1 cleaned QEVD class space.

This module is the single source of truth for the v6.1 multi-label
classification class index. It loads the JSON written by
`_build_qevd_class_space.py` and exposes immutable lookup helpers used by
the dataset pipeline (`qevd_label_builder_v2.py`), the model head
(`st_gcn_v6_1.py`), the training loop, and the runtime form analyzer.

The class index is data, not code: the JSON is reproducible from the
dataset alone with the cleanup rules documented in the build script. No
manual curation is involved. See plan §6.
"""
from __future__ import annotations

import functools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC_PATH = REPO_ROOT / "backend" / "data" / "qevd_class_space_v6_1.json"


class QEVDClassSpace:
    """Immutable class-index for v6.1 multi-label classification."""

    def __init__(self, spec: dict) -> None:
        self._spec = spec
        self._classes: list[dict] = spec["classes"]
        self._num_classes: int = len(self._classes)

        self._pair_to_idx: dict[tuple[str, str], int] = {
            (c["prefix"], c["variation"]): c["idx"] for c in self._classes
        }
        self._prefix_to_indices: dict[str, list[int]] = {
            prefix: list(indices)
            for prefix, indices in spec["prefix_to_class_indices"].items()
        }
        self._counts: np.ndarray = np.array(
            [c["n_clips_total"] for c in self._classes], dtype=np.float32
        )

        # Stable alphabetical prefix order for the v6.1 embedding input.
        # Sorted so the index is deterministic across runs without needing
        # a separate JSON file.
        self._prefix_ordered: list[str] = sorted(self._prefix_to_indices)
        self._prefix_to_prefix_idx: dict[str, int] = {
            p: i for i, p in enumerate(self._prefix_ordered)
        }

    # ---- factory ---------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str | None = None) -> "QEVDClassSpace":
        path = Path(path) if path is not None else DEFAULT_SPEC_PATH
        with open(path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        return cls(spec)

    # ---- properties ------------------------------------------------------

    @property
    def version(self) -> str:
        return self._spec["version"]

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_prefixes(self) -> int:
        return len(self._prefix_to_indices)

    @property
    def class_counts(self) -> np.ndarray:
        """Per-class total clip count (training distribution)."""
        return self._counts.copy()

    # ---- lookups ---------------------------------------------------------

    def class_idx_for_label(self, label: str) -> int | None:
        """Map a raw label string like 'squats - shallow' to its class idx.

        Returns None if the label was filtered out by the cleanup pass
        (e.g. 'not visible', long-tail variant). Callers should treat None
        as 'no supervisory signal from this label' and skip it during
        target construction.
        """
        if " - " in label:
            prefix, variation = label.split(" - ", 1)
        else:
            prefix, variation = label, ""
        return self._pair_to_idx.get((prefix.strip(), variation.strip()))

    def decode(self, idx: int) -> tuple[str, str]:
        """Inverse of class_idx_for_label — return (prefix, variation)."""
        if not 0 <= idx < self._num_classes:
            raise IndexError(f"idx {idx} outside [0, {self._num_classes})")
        c = self._classes[idx]
        return c["prefix"], c["variation"]

    def class_indices_for_prefix(self, prefix: str) -> list[int]:
        """Class indices belonging to a given exercise prefix.

        Used for runtime sigmoid-logit masking when the user has
        pre-selected an exercise (the model's output is restricted to
        the variations valid for that prefix).
        """
        return list(self._prefix_to_indices.get(prefix, []))

    def has_prefix(self, prefix: str) -> bool:
        return prefix in self._prefix_to_indices

    def prefix_idx_for(self, prefix: str) -> int | None:
        """Map an exercise prefix to its v6.1 embedding-input index.

        Order is deterministic (alphabetical). Returns None for prefixes
        that did not survive cleanup; the dataset pipeline should drop
        the clip (no usable supervision) when this is None.
        """
        return self._prefix_to_prefix_idx.get(prefix)

    def decode_prefix(self, prefix_idx: int) -> str:
        if not 0 <= prefix_idx < len(self._prefix_ordered):
            raise IndexError(
                f"prefix_idx {prefix_idx} outside [0, {len(self._prefix_ordered)})"
            )
        return self._prefix_ordered[prefix_idx]

    def prefix_idx_for_label(self, label: str) -> int | None:
        """Map a raw label string to its v6.1 prefix-embedding index.

        Convenience for dataset code that has the raw label string and
        wants the embedding index without splitting manually.
        """
        if " - " in label:
            prefix, _ = label.split(" - ", 1)
        else:
            prefix = label
        return self.prefix_idx_for(prefix.strip())

    # ---- training helpers -----------------------------------------------

    def inverse_frequency_weights(
        self,
        beta: float = 0.999,
        min_weight: float = 0.1,
        max_weight: float = 10.0,
    ) -> np.ndarray:
        """Cui et al. (2019) effective-number class weights, clipped.

        Used as the BCE pos_weight tensor. beta=0.999 is the standard
        long-tail-recognition default. Clipping prevents extreme weights
        on the smallest classes from dominating the gradient.
        """
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1); got {beta}")
        effective_num = 1.0 - np.power(beta, self._counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        weights = weights / weights.mean()
        return np.clip(weights, min_weight, max_weight).astype(np.float32)

    def encode_labels(self, labels: Iterable[str]) -> np.ndarray:
        """Encode a clip's label list to a multi-label sigmoid target.

        Returns shape (num_classes,) float32 with 1.0 for each label that
        survived cleanup, 0.0 otherwise. Labels that map to None (filtered)
        contribute nothing — they are silently skipped, NOT encoded as a
        zero target (the BCE loss should mask the unsupervised classes if
        there are any; in practice the cleanup pass keeps 92% of labels
        so the unmasked-zero approach is acceptable).
        """
        target = np.zeros(self._num_classes, dtype=np.float32)
        for label in labels:
            idx = self.class_idx_for_label(label)
            if idx is not None:
                target[idx] = 1.0
        return target


@functools.lru_cache(maxsize=1)
def get_default_class_space() -> QEVDClassSpace:
    """Cached singleton — typical callsite uses the default spec path."""
    return QEVDClassSpace.load()
