"""Single source of truth for exercise metadata (Phase C — plan GC).

Loads backend/config/exercises.json once per process and exposes lookup
helpers used by both the training scripts and the runtime services.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_JSON_PATH = os.path.join(_HERE, "exercises.json")


@lru_cache(maxsize=1)
def _load() -> dict[str, dict[str, Any]]:
    with open(_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def exercise_to_idx() -> dict[str, int]:
    return {name: meta["idx"] for name, meta in _load().items()}


@lru_cache(maxsize=1)
def idx_to_exercise() -> dict[int, str]:
    return {meta["idx"]: name for name, meta in _load().items()}


def get_metadata(name: str) -> dict[str, Any]:
    data = _load()
    if name not in data:
        raise KeyError(f"Unknown exercise: {name!r}")
    return data[name]


def all_exercises() -> dict[str, dict[str, Any]]:
    """Return the full mapping (copy-safe for API responses)."""
    return {k: dict(v) for k, v in _load().items()}


# Convenience module-level constants (evaluated lazily on first access via
# the cached functions above). Consumers typically do:
#     from backend.config.exercises import EXERCISE_TO_IDX, IDX_TO_EXERCISE
EXERCISE_TO_IDX = exercise_to_idx()
IDX_TO_EXERCISE = idx_to_exercise()


# ── Semantic joint mapping ───────────────────────────────────────────────────
# MT-TCN joint-error head outputs a 10-vector over these groups (order fixed
# by training; see backend/services/form_session.py::JOINT_GROUP_NAMES).
_JOINT_GROUP_ORDER = [
    "Left Elbow", "Right Elbow",
    "Left Shoulder", "Right Shoulder",
    "Left Knee", "Right Knee",
    "Left Hip", "Right Hip",
    "Trunk / Spine", "Neck",
]

# Mapping from the short semantic labels used in exercises.json
# (``key_errors_detected``) to the indices of the 10-vector above.
_SEMANTIC_TO_INDICES: dict[str, list[int]] = {
    "knees":     [4, 5],   # L/R knee
    "hips":      [6, 7],   # L/R hip
    "back":      [8],      # trunk / spine
    "spine":     [8],
    "trunk":     [8],
    "neck":      [9],
    "elbows":    [0, 1],   # L/R elbow — curls, presses
    "shoulders": [2, 3],   # L/R shoulder
    "arms":      [0, 1, 2, 3],
}


def joint_group_names() -> list[str]:
    """Return the fixed-order 10-joint-group labels used by the MT-TCN head."""
    return list(_JOINT_GROUP_ORDER)


def relevant_joint_indices(name: str) -> list[int]:
    """Return the subset of the 10-joint-group vector that matters for `name`.

    Driven by ``key_errors_detected`` in ``exercises.json``. Returns the
    full 10-index list as a safe fallback if the exercise is unknown or the
    semantic tag isn't recognised, so consumers can always iterate.
    """
    meta = _load().get(name)
    if meta is None:
        return list(range(10))
    indices: list[int] = []
    for tag in meta.get("key_errors_detected", []):
        for idx in _SEMANTIC_TO_INDICES.get(tag.lower(), []):
            if idx not in indices:
                indices.append(idx)
    return indices or list(range(10))


def relevant_joint_names(name: str) -> list[str]:
    """Like :func:`relevant_joint_indices` but returns the human-readable names."""
    return [_JOINT_GROUP_ORDER[i] for i in relevant_joint_indices(name)]
