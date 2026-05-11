"""Phase-E confusion-aware mismatch detector.

Watches the MT-TCN argmax stream inside a FormSession. If the user pre-
selected exercise X but the model is persistently predicting a *different*
class that is NOT one of X's typical confusions, emit a one-shot warning.

Modes
-----
Smart mode (confusion_matrix.npy present next to the weights):
    For each class we precompute the 3 most-confusable neighbour classes from
    the row-normalised confusion matrix. A mismatch is only flagged if the
    persistent prediction is OUTSIDE that top-3 set — known confusions stay
    silent.

Degraded mode (confusion_matrix.npy absent):
    No neighbour suppression. To avoid toast fatigue we instead raise the
    per-frame confidence floor from 0.70 to 0.80. This keeps the warning
    useful when the model is saved without the matrix (pre-A6 weights).
"""
from __future__ import annotations

import json
import logging
import os
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Rolling-window parameters
_BUFFER_SIZE           = 8     # last N inferences
_MIN_MISMATCH_COUNT    = 5     # need >=5 of last 8 disagreeing
_CONFIDENCE_SMART_MODE = 0.70
_CONFIDENCE_DEGRADED   = 0.80  # when confusion matrix is missing
_TOP_K_NEIGHBOURS      = 3


class ExerciseMismatchDetector:
    """One instance per FormSession.

    Call `update(predicted_name, confidence)` after each MT-TCN window
    inference. Returns a warning dict exactly once per session, or None.
    """

    _class_index: Optional[dict[str, int]] = None
    _neighbours:  Optional[dict[str, set[str]]] = None
    _mode:        str = "uninitialised"  # "smart" | "degraded"

    def __init__(self, selected_exercise: Optional[str], model_dir: str):
        self.selected_exercise = selected_exercise
        self._buffer: deque[tuple[str, float]] = deque(maxlen=_BUFFER_SIZE)
        self._fired = False

        if not self._class_index:
            ExerciseMismatchDetector._bootstrap(model_dir)

    @classmethod
    def _bootstrap(cls, model_dir: str) -> None:
        """Load labels + confusion matrix once per process.

        Label-file detection priority:
          1. ``qevd_exercise_map.json``   (v6 — QEVD class names)
          2. ``exercise_labels.json``     (v5.2 / v4 — FitNova names)

        Confusion matrix priority (any match wins):
          ``confusion_matrix_test.npy`` (s11 held-out — closest to real-world
          generalisation), then ``confusion_matrix.npy`` (manual rename), then
          ``confusion_matrix_val.npy`` as a fallback.
        """
        v6_labels_path  = os.path.join(model_dir, "qevd_exercise_map.json")
        v52_labels_path = os.path.join(model_dir, "exercise_labels.json")
        if os.path.isfile(v6_labels_path):
            labels_path = v6_labels_path
            label_source = "v6"
        else:
            labels_path = v52_labels_path
            label_source = "v5.2/v4"

        cm_path = None
        for candidate in ("confusion_matrix_test.npy",
                          "confusion_matrix.npy",
                          "confusion_matrix_val.npy"):
            p = os.path.join(model_dir, candidate)
            if os.path.exists(p):
                cm_path = p
                break
        if cm_path is None:
            cm_path = os.path.join(model_dir, "confusion_matrix.npy")  # for log message below
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)  # {name: idx}
            logger.info(
                "ExerciseMismatchDetector: loaded %d labels from %s (source=%s)",
                len(labels), os.path.basename(labels_path), label_source,
            )
        except Exception as e:
            logger.warning(
                "ExerciseMismatchDetector: cannot load labels from %s or %s "
                "(%s); detector disabled",
                v6_labels_path, v52_labels_path, e,
            )
            cls._class_index = {}
            cls._neighbours  = {}
            cls._mode        = "disabled"
            return

        cls._class_index = dict(labels)
        idx_to_name = {v: k for k, v in labels.items()}

        if os.path.exists(cm_path):
            try:
                cm = np.load(cm_path)  # (K, K) row-normalised or counts
                if cm.shape[0] == cm.shape[1] == len(labels):
                    # Row-normalise in case raw counts were saved
                    row_sums = cm.sum(axis=1, keepdims=True)
                    row_sums[row_sums == 0] = 1.0
                    cm_norm = cm / row_sums
                    neighbours: dict[str, set[str]] = {}
                    for name, idx in labels.items():
                        row = cm_norm[idx].copy()
                        row[idx] = -1.0  # exclude self
                        top = np.argsort(-row)[:_TOP_K_NEIGHBOURS]
                        neighbours[name] = {idx_to_name[int(i)] for i in top}
                    cls._neighbours = neighbours
                    cls._mode       = "smart"
                    logger.info("ExerciseMismatchDetector: smart mode (confusion matrix loaded)")
                    return
                else:
                    logger.warning(
                        f"confusion_matrix.npy shape {cm.shape} does not match "
                        f"{len(labels)} classes; falling back to degraded mode."
                    )
            except Exception as e:
                logger.warning(f"Could not parse confusion_matrix.npy: {e}; degraded mode.")
        else:
            logger.info("confusion_matrix.npy not found; degraded mode (tighter confidence floor).")

        cls._neighbours = {}
        cls._mode       = "degraded"

    def update(self, predicted_name: str, confidence: float) -> Optional[dict]:
        """Call once per inference; returns the warning dict or None."""
        if self._fired:
            return None
        if not self.selected_exercise:
            return None
        if self._mode == "disabled":
            return None
        if not self._class_index or self.selected_exercise not in self._class_index:
            return None

        floor = _CONFIDENCE_SMART_MODE if self._mode == "smart" else _CONFIDENCE_DEGRADED
        self._buffer.append((predicted_name, float(confidence)))

        if len(self._buffer) < _BUFFER_SIZE:
            return None

        disagreeing = [
            (p, c) for (p, c) in self._buffer
            if p != self.selected_exercise and c >= floor
        ]
        if len(disagreeing) < _MIN_MISMATCH_COUNT:
            return None

        # Pick the most frequent disagreeing class
        from collections import Counter
        dominant, _ = Counter(p for p, _ in disagreeing).most_common(1)[0]
        avg_conf    = float(np.mean([c for p, c in disagreeing if p == dominant]))

        # Smart mode: suppress if `dominant` is a known confusion of `selected`
        if self._mode == "smart":
            neighbours = self._neighbours.get(self.selected_exercise, set()) if self._neighbours else set()
            if dominant in neighbours:
                return None

        self._fired = True
        return {
            "type":       "exercise_mismatch_warning",
            "selected":   self.selected_exercise,
            "predicted":  dominant,
            "confidence": round(avg_conf, 3),
            "mode":       self._mode,
        }

    @property
    def mode(self) -> str:
        return self._mode
