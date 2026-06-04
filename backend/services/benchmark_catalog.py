"""BenchmarkCatalog — loads the Fitness-AQA test splits + results.pkl files at startup.

Provides get_catalog(exercise), media_path(exercise, clip_id), and whitelisted_ids
(the startup-built set used by the media endpoint's path-traversal gate).

Supports three exercises: squat (244 clips), ohp (339 clips), shallow (540 clips).
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# AQA sub-paths (relative to the fitness_aqa_root argument)
# ─────────────────────────────────────────────────────────────────────────────

_SQUAT_SPLIT_JSON = "Squat/Labeled_Dataset/Splits/test_keys.json"
_SQUAT_LABEL_KIE  = "Squat/Labeled_Dataset/Labels/error_knees_inward.json"
_SQUAT_LABEL_KFE  = "Squat/Labeled_Dataset/Labels/error_knees_forward.json"
_SQUAT_VIDEOS_DIR = "Squat/Labeled_Dataset/videos_extracted/videos"

_OHP_SPLIT_JSON = "OHP/Labeled_Dataset/Splits/test_keys.json"
_OHP_LABEL_ELBOWS = "OHP/Labeled_Dataset/Labels/error_elbows.json"
_OHP_LABEL_KNEES  = "OHP/Labeled_Dataset/Labels/error_knees.json"
_OHP_VIDEOS_DIR = "OHP/Labeled_Dataset/videos_extracted/videos"

_SHALLOW_SPLIT_JSON  = "Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/splits/test_ids.json"
_SHALLOW_LABEL_JSON  = "Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/labels_shallow_depth.json"
_SHALLOW_CROPS_DIR   = "Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/crops_unaligned"

# ─────────────────────────────────────────────────────────────────────────────
# results.pkl sub-paths (relative to results_dir)
# ─────────────────────────────────────────────────────────────────────────────

_SQUAT_PKL   = "squat/results.pkl"
_OHP_PKL     = "ohp/results.pkl"
_SHALLOW_PKL = "shallow/results.pkl"


class BenchmarkCatalog:
    """Load all three Fitness-AQA test splits at construction time.

    Args:
        fitness_aqa_root: Path to the Fitness-AQA dataset root directory.
        results_dir:      Path to the directory holding the per-exercise results.pkl files.
    """

    def __init__(self, fitness_aqa_root: str, results_dir: str) -> None:
        self._root = Path(fitness_aqa_root)
        self._results_dir = Path(results_dir)
        self._catalog: dict[str, list[dict[str, Any]]] = {}
        self._whitelisted_ids: dict[str, set[str]] = {}
        self._load_all()

    # ─────────────────────────────────────────────────────────────────────────
    # Internal loaders
    # ─────────────────────────────────────────────────────────────────────────

    def _load_all(self) -> None:
        self._catalog["squat"]   = self._load_squat()
        self._catalog["ohp"]     = self._load_ohp()
        self._catalog["shallow"] = self._load_shallow()
        for ex, entries in self._catalog.items():
            self._whitelisted_ids[ex] = {e["clip_id"] for e in entries}
        logger.info(
            "BenchmarkCatalog loaded: squat=%d, ohp=%d, shallow=%d",
            len(self._catalog["squat"]),
            len(self._catalog["ohp"]),
            len(self._catalog["shallow"]),
        )

    def _load_squat(self) -> list[dict[str, Any]]:
        ids: list[str] = json.loads((self._root / _SQUAT_SPLIT_JSON).read_text())
        kie_label_map: dict[str, list] = json.loads((self._root / _SQUAT_LABEL_KIE).read_text())
        kfe_label_map: dict[str, list] = json.loads((self._root / _SQUAT_LABEL_KFE).read_text())

        with open(self._results_dir / _SQUAT_PKL, "rb") as f:
            data = pickle.load(f)

        scores = data["raw"]["ens_test_scores"]   # (244, 2) — col0=KIE, col1=KFE
        labels = data["raw"]["test_labels"]       # (244, 2)
        assert len(ids) == len(scores) == len(labels), (
            f"squat catalog misalignment: ids={len(ids)} scores={len(scores)} labels={len(labels)}"
        )

        entries: list[dict[str, Any]] = []
        for i, cid in enumerate(ids):
            # Ground-truth from results.pkl (authoritative, verified vs label JSONs).
            entries.append({
                "clip_id": cid,
                "ground_truth": {
                    "KIE": int(labels[i, 0]),
                    "KFE": int(labels[i, 1]),
                },
                "score": {
                    "KIE": float(scores[i, 0]),
                    "KFE": float(scores[i, 1]),
                },
            })
        return entries

    def _load_ohp(self) -> list[dict[str, Any]]:
        with open(self._results_dir / _OHP_PKL, "rb") as f:
            data = pickle.load(f)

        # OHP results.pkl carries test_clip_ids as the authoritative ordering; use it
        # instead of test_keys.json because eval used this list to build the score matrix.
        ids: list[str] = data["test_clip_ids"]
        scores = data["ensemble_test_scores"]   # (339, 2) — col0=ELBOWS, col1=KNEES
        labels = data["test_labels"]            # (339, 2)
        assert len(ids) == len(scores) == len(labels), (
            f"ohp catalog misalignment: ids={len(ids)} scores={len(scores)} labels={len(labels)}"
        )

        entries: list[dict[str, Any]] = []
        for i, cid in enumerate(ids):
            entries.append({
                "clip_id": cid,
                "ground_truth": {
                    "ELBOWS": int(labels[i, 0]),
                    "KNEES":  int(labels[i, 1]),
                },
                "score": {
                    "ELBOWS": float(scores[i, 0]),
                    "KNEES":  float(scores[i, 1]),
                },
            })
        return entries

    def _load_shallow(self) -> list[dict[str, Any]]:
        # Shallow results.pkl has no test_clip_ids; use test_ids.json list order.
        ids: list[str] = json.loads((self._root / _SHALLOW_SPLIT_JSON).read_text())

        with open(self._results_dir / _SHALLOW_PKL, "rb") as f:
            data = pickle.load(f)

        scores = data["ensemble_test_scores"]   # (540,) — single DEPTH score
        labels = data["test_labels"]            # (540,)
        assert len(ids) == len(scores) == len(labels), (
            f"shallow catalog misalignment: ids={len(ids)} scores={len(scores)} labels={len(labels)}"
        )

        entries: list[dict[str, Any]] = []
        for i, cid in enumerate(ids):
            entries.append({
                "clip_id": cid,
                "ground_truth": {
                    "DEPTH": int(labels[i]),
                },
                "score": {
                    "DEPTH": float(scores[i]),
                },
            })
        return entries

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def get_catalog(self, exercise: str) -> list[dict[str, Any]]:
        """Return the full test-split entry list for `exercise`.

        Returns an empty list for unknown exercises (caller enforces allowlist).
        """
        return self._catalog.get(exercise, [])

    def media_path(self, exercise: str, clip_id: str) -> Path:
        """Return the filesystem path for `clip_id`'s media file.

        Callers MUST check that clip_id is whitelisted before calling this method —
        no path-safety check is performed here.
        """
        if exercise == "squat":
            return self._root / _SQUAT_VIDEOS_DIR / f"{clip_id}.mp4"
        if exercise == "ohp":
            return self._root / _OHP_VIDEOS_DIR / f"{clip_id}.mp4"
        if exercise == "shallow":
            return self._root / _SHALLOW_CROPS_DIR / f"{clip_id}.jpg"
        raise ValueError(f"Unknown exercise: {exercise!r}")

    @property
    def whitelisted_ids(self) -> dict[str, set[str]]:
        """Startup-built whitelist: {exercise: {clip_id, ...}} from the official split JSONs."""
        return self._whitelisted_ids
