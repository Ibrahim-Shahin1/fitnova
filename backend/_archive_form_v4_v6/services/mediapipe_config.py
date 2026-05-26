"""Shared MediaPipe configuration for FitNova.

Single source of truth for PoseLandmarker options. Both the live inference
path (backend.services.form_analyzer) and the QEVD training-data extractor
(backend.training.preprocessing.qevd_extractor) import from here so that
pose features are produced under identical settings.

If you change ANY constant here, retrain or re-extract: drift between
training-time and inference-time MediaPipe was the root cause of v4's
real-phone failure (HANDOFF.md section 5).
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Pinned versions (informational; enforce via requirements.txt) ────────────
# Match exactly what's installed on the user's Windows PC where reference
# outputs were generated. Drift between this pin and the actual local
# install would re-introduce v4's distribution-mismatch bug.
PINNED_MEDIAPIPE_VERSION = "0.10.33"
PINNED_PROTOBUF_VERSION = "4.25.3"

# ── Detection confidences (must match training extraction) ───────────────────
MIN_POSE_DETECTION_CONFIDENCE = 0.5
MIN_POSE_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
NUM_POSES = 1

# Running mode VIDEO (not IMAGE) — applies temporal smoothing across frames,
# matching how training .npy files were extracted. See B6 fix in form_analyzer.
RUNNING_MODE = "VIDEO"

# ── Model asset ──────────────────────────────────────────────────────────────
# This module lives at backend/services/mediapipe_config.py; parents[2] = repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
POSE_LANDMARKER_MODEL_PATH = (
    REPO_ROOT / "backend" / "models" / "pose_landmarker_full.task"
)
POSE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)


def ensure_model_downloaded(model_path: Path | None = None) -> Path:
    """Return the cached PoseLandmarker model path, downloading on first use.

    ~29 MB float16 asset from Google's MediaPipe model zoo.
    """
    p = Path(model_path) if model_path is not None else POSE_LANDMARKER_MODEL_PATH
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MediaPipe model (~29 MB) to %s ...", p)
    urllib.request.urlretrieve(POSE_LANDMARKER_MODEL_URL, str(p))
    logger.info("MediaPipe model downloaded.")
    return p


def build_pose_landmarker_options(
    *,
    model_path: Path | None = None,
    running_mode: str | None = None,
) -> Any:
    """Construct ``mediapipe.tasks.vision.PoseLandmarkerOptions`` with FitNova's
    pinned settings.

    Lazy-imports mediapipe so callers that just need the constants (e.g.
    unit tests) can import this module without mediapipe installed.
    """
    import mediapipe as mp  # lazy: keeps the module import side-effect-free

    asset_path = ensure_model_downloaded(model_path)

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    mode_name = running_mode or RUNNING_MODE
    try:
        running_mode_enum = getattr(VisionRunningMode, mode_name)
    except AttributeError as exc:
        raise ValueError(
            f"Unknown running_mode {mode_name!r}; expected one of "
            f"{[m.name for m in VisionRunningMode]}"
        ) from exc

    return PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(asset_path)),
        running_mode=running_mode_enum,
        num_poses=NUM_POSES,
        min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_POSE_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )


__all__ = [
    "MIN_POSE_DETECTION_CONFIDENCE",
    "MIN_POSE_PRESENCE_CONFIDENCE",
    "MIN_TRACKING_CONFIDENCE",
    "NUM_POSES",
    "PINNED_MEDIAPIPE_VERSION",
    "PINNED_PROTOBUF_VERSION",
    "POSE_LANDMARKER_MODEL_PATH",
    "POSE_LANDMARKER_MODEL_URL",
    "REPO_ROOT",
    "RUNNING_MODE",
    "build_pose_landmarker_options",
    "ensure_model_downloaded",
]
