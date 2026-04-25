"""
FormAnalyzer — real-time inference service.

Responsibilities:
  1. Decode JPEG frames
  2. Run MediaPipe Pose (VIDEO mode — matches training extraction) to extract 33 landmarks
  3. Map landmarks to 15 canonical joints
  4. Compute 22 normalised angular features
  5. Return raw landmarks (for Flutter skeleton overlay) + angular features (for TCN)

B6 fix (2026-04-18): switched from VisionRunningMode.IMAGE to VisionRunningMode.VIDEO.
  Training .npy files were extracted with VIDEO mode (detect_for_video + timestamps),
  which applies temporal smoothing.  Using IMAGE mode at inference shifts the angle
  distribution and collapses exercise accuracy from ~89% to ~11%.
  VIDEO mode requires a monotonically increasing timestamp per frame.
  Call reset_video_state(fps) at the start of each new video / session.

B7 fix (2026-04-18): forward-fill on no-pose frames.
  When MediaPipe cannot detect a pose, the previous valid frame's result is
  returned (up to MAX_FILL_FRAMES consecutive).  This matches training's linear
  interpolation strategy for short occlusions.
"""

import base64
import json
import logging
import os
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-import heavy deps at class init to keep startup fast
_MEDIAPIPE_AVAILABLE = True
try:
    import mediapipe as mp
except ImportError:
    _MEDIAPIPE_AVAILABLE = False
    logger.warning("mediapipe not installed — FormAnalyzer will be unavailable.")

_TF_AVAILABLE = True
try:
    import tensorflow as tf
except ImportError:
    _TF_AVAILABLE = False

# Maximum consecutive no-pose frames that will be forward-filled.
# After this many misses we emit "no_pose" instead of recycling the last result.
MAX_FILL_FRAMES = 5


class FormAnalyzer:
    """
    Stateless per-frame analyzer.
    Loaded once at backend startup and shared across all WebSocket sessions.

    Thread safety: process_frame() mutates _ts_ms and _prev_result, so do NOT
    share a single FormAnalyzer across concurrent sessions without a lock.
    """

    def __init__(self, model_dir: str):
        """
        Args:
            model_dir: path to backend/models/form_model/
                       Must contain: mt_tcn_weights.weights.h5, model_config.json,
                                     angle_stats.npz, exercise_labels.json
        """
        self.model_dir = model_dir
        self._model = None
        self._normalizer = None
        self._exercise_labels: dict = {}
        self._idx_to_exercise: dict = {}

        # VIDEO mode state (reset per video/session)
        self._ts_ms: int = 0
        self._frame_interval_ms: int = 33   # default 30 fps → ~33 ms/frame
        self._prev_result: Optional[dict] = None
        self._fill_streak: int = 0           # consecutive no-pose frames filled

        self._init_mediapipe()
        self._load_model()

    # ── Video-state management ────────────────────────────────────────────────

    def reset_video_state(self, fps: float = 30.0) -> None:
        """
        Call at the start of every new video or live session.

        MediaPipe's PoseLandmarker (VIDEO mode) maintains an internal timestamp
        counter and throws "Input timestamp must be monotonically increasing" if
        you pass a smaller timestamp than the last one.  The only safe way to
        reset is to close the current instance and create a fresh one.
        """
        # ── Recreate the MediaPipe landmarker so the timestamp counter resets ──
        if hasattr(self, "_mp_landmarker") and self._mp_landmarker is not None:
            try:
                self._mp_landmarker.close()
            except Exception:
                pass  # tolerate close errors (e.g. already closed)

        if hasattr(self, "_mp_options") and self._mp_options is not None:
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            self._mp_landmarker = PoseLandmarker.create_from_options(self._mp_options)
            self.mp_pose = self._mp_landmarker   # keep alias in sync

        # ── Reset per-video state ───────────────────────────────────────────────
        self._ts_ms             = 0
        self._frame_interval_ms = max(1, int(1000.0 / fps))
        self._prev_result       = None
        self._fill_streak       = 0

    # ── MediaPipe ─────────────────────────────────────────────────────────────

    def _init_mediapipe(self):
        if not _MEDIAPIPE_AVAILABLE:
            logger.error("mediapipe not available — pose estimation disabled.")
            self.mp_pose = None
            self.mp_drawing = None
            return

        # ── MediaPipe 0.10.x Tasks API ────────────────────────────────────────
        import urllib.request

        # Cache the model file next to the other model artefacts
        model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "models",
                         "pose_landmarker_full.task")
        )
        if not os.path.exists(model_path):
            url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_full/float16/latest/"
                "pose_landmarker_full.task"
            )
            logger.info("Downloading MediaPipe model (~29 MB) to %s …", model_path)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            urllib.request.urlretrieve(url, model_path)
            logger.info("MediaPipe model downloaded.")

        BaseOptions           = mp.tasks.BaseOptions
        PoseLandmarker        = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode     = mp.tasks.vision.RunningMode

        # VIDEO mode: applies temporal smoothing across frames, matching how
        # the training .npy files were extracted (see FitNova_Colab.ipynb cell-12).
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,   # ← B6 fix (was IMAGE)
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mp_options        = options            # stored for reset_video_state re-creation
        self._mp_landmarker     = PoseLandmarker.create_from_options(options)
        self._mp_image_cls      = mp.Image
        self._mp_image_fmt      = mp.ImageFormat
        self._VisionRunningMode = VisionRunningMode   # stored for mode checks

        # Keep mp_pose as an alias so app.py can call mp_pose.close() unchanged
        self.mp_pose    = self._mp_landmarker
        self.mp_drawing = None   # drawing_utils not needed; skeleton drawn by Flutter
        logger.info("MediaPipe PoseLandmarker initialised (Tasks API, VIDEO mode).")

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        # Keras 3 requires weights filenames to end in ".weights.h5" (that is what
        # train_form_model.py writes). We still accept the legacy "mt_tcn_weights.h5"
        # for backward compatibility with older checkpoints.
        candidates = [
            os.path.join(self.model_dir, "mt_tcn_weights.weights.h5"),
            os.path.join(self.model_dir, "mt_tcn_best.weights.h5"),
            os.path.join(self.model_dir, "mt_tcn_weights.h5"),
        ]
        weights_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
        config_path  = os.path.join(self.model_dir, "model_config.json")
        stats_path   = os.path.join(self.model_dir, "angle_stats.npz")
        labels_path  = os.path.join(self.model_dir, "exercise_labels.json")

        if not os.path.exists(weights_path):
            logger.warning(
                f"MT-TCN weights not found at {weights_path}. "
                "Train the model first: python -m backend.training.train_form_model"
            )
            return

        if not _TF_AVAILABLE:
            logger.error("TensorFlow not available — MT-TCN loading skipped.")
            return

        # Import here to avoid circular import at module level
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from backend.training.models.mt_tcn import build_mt_tcn
        from backend.training.preprocessing.normalize import AngleNormalizer

        with open(config_path) as f:
            cfg = json.load(f)

        self._model = build_mt_tcn(
            target_frames  = cfg["target_frames"],
            n_angular      = cfg["n_angular"],
            n_joints_flat  = cfg["n_joints_flat"],
            n_exercises    = cfg["n_exercises"],
            n_joint_groups = cfg["n_joint_groups"],
        )
        # NOTE: Keras 3's .weights.h5 format uses positional layer matching.
        # by_name=True is only supported for legacy .h5/.hdf5 files and raises
        # an error on .weights.h5.  Positional loading is safe here because
        # build_mt_tcn() always constructs layers in the same order.
        self._model.load_weights(weights_path)
        logger.info("MT-TCN weights loaded (positional, %.1f MB).",
                    os.path.getsize(weights_path) / 1e6)

        self._normalizer = AngleNormalizer.load(stats_path)
        logger.info("Angle normaliser loaded.")

        with open(labels_path) as f:
            self._exercise_labels = json.load(f)
        self._idx_to_exercise = {v: k for k, v in self._exercise_labels.items()}
        logger.info(f"Exercise labels loaded: {len(self._exercise_labels)} classes.")

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def model_ready(self) -> bool:
        return self._model is not None and self._normalizer is not None

    def process_frame(self, jpeg_bytes: bytes,
                      timestamp_ms: Optional[int] = None) -> dict:
        """
        Process one JPEG frame.

        Args:
            jpeg_bytes:   JPEG-encoded image bytes.
            timestamp_ms: Optional explicit timestamp in milliseconds.
                          If None, an internal counter is auto-incremented by
                          frame_interval_ms (default 33 ms = 30 fps).
                          Pass explicit timestamps when the caller has real
                          frame timing (e.g. from Flutter's camera stream).

        Returns:
            {
              "landmarks": [[x,y,z], ...] (15 canonical joints, image-normalised 0-1)
                           or None if no pose detected
              "angles_norm": [float × 22]  normalised angular features
                             or None if no pose detected
              "status": "ok" | "no_pose" | "pose_interpolated" |
                        "mediapipe_unavailable" | "decode_error"
            }
        """
        if not _MEDIAPIPE_AVAILABLE or not hasattr(self, "_mp_landmarker"):
            return {"landmarks": None, "joints_norm": None,
                    "angles_norm": None, "status": "mediapipe_unavailable"}

        # ── Timestamp ─────────────────────────────────────────────────────────
        if timestamp_ms is not None:
            self._ts_ms = int(timestamp_ms)
        current_ts = self._ts_ms
        self._ts_ms += self._frame_interval_ms

        # Decode JPEG
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return {"landmarks": None, "joints_norm": None,
                    "angles_norm": None, "status": "decode_error"}

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_img    = self._mp_image_cls(image_format=self._mp_image_fmt.SRGB,
                                       data=image_rgb)

        # VIDEO mode: pass the monotonically increasing timestamp
        result = self._mp_landmarker.detect_for_video(mp_img, current_ts)

        if not result.pose_landmarks or not result.pose_world_landmarks:
            # ── B7 fix: forward-fill from last valid frame ─────────────────
            if (self._prev_result is not None
                    and self._fill_streak < MAX_FILL_FRAMES):
                self._fill_streak += 1
                filled = dict(self._prev_result)
                filled["status"] = "pose_interpolated"
                return filled
            return {"landmarks": None, "joints_norm": None,
                    "angles_norm": None, "status": "no_pose"}

        # Pose detected — reset fill streak
        self._fill_streak = 0

        # ── Extract world landmarks (3D, metric scale) ────────────────────────
        mp_world = np.array(
            [[lm.x, lm.y, lm.z] for lm in result.pose_world_landmarks[0]],
            dtype=np.float32,
        )  # (33, 3)

        # ── Image landmarks (normalised 0-1 for display) ──────────────────────
        mp_image = np.array(
            [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks[0]],
            dtype=np.float32,
        )  # (33, 3) — x,y in [0,1], z relative depth

        # ── Map to canonical 15 joints ────────────────────────────────────────
        from backend.training.preprocessing.normalize import (
            extract_canonical_from_mediapipe, normalize_skeleton,
        )
        canonical_world = extract_canonical_from_mediapipe(mp_world)  # (15, 3)
        canonical_img   = extract_canonical_from_mediapipe(mp_image)  # (15, 3)

        # Skeleton-normalise world coords before angle computation
        canon_norm = normalize_skeleton(canonical_world[np.newaxis])[0]  # (15, 3)

        # ── Angular features ──────────────────────────────────────────────────
        from backend.training.preprocessing.angular_features import compute_frame_angles
        angles_raw = compute_frame_angles(canon_norm)  # (22,)

        if self._normalizer is not None:
            angles_norm = self._normalizer.transform(angles_raw)
        else:
            angles_norm = angles_raw

        result_dict = {
            "landmarks":        canonical_img.tolist(),    # 15 × [x,y,z] image coords (display)
            "joints_norm":      canon_norm.tolist(),       # 15 × [x,y,z] skeleton-normalised world coords
            "angles_norm":      angles_norm.tolist(),       # 22 normalised angles
            "status":           "ok",
        }
        self._prev_result = result_dict          # cache for B7 forward-fill
        return result_dict

    def predict_window(
        self,
        angles_window: np.ndarray,
        joints_window: np.ndarray,
    ) -> dict:
        """
        Run MT-TCN on a 64-frame window.

        Args:
            angles_window: (64, 22) float32
            joints_window: (64, 45) float32

        Returns:
            {
              "exercise": str,
              "exercise_confidence": float,
              "boundary": [float × 64],   per-frame boundary probability
              "rep_count": float,
              "quality": float,
              "joint_errors": [[float × 10] × 64]  per-frame per-joint
            }
        """
        if not self.model_ready:
            return {
                "exercise": "unknown", "exercise_confidence": 0.0,
                "boundary": [0.0] * 64, "rep_count": 0.0,
                "quality": 0.5, "joint_errors": [[0.0] * 10] * 64,
            }

        inp = {
            "angular_input": angles_window[np.newaxis].astype(np.float32),
            "joints_input":  joints_window[np.newaxis].astype(np.float32),
        }
        preds = self._model.predict(inp, verbose=0)

        ex_probs   = preds["exercise"][0]           # (27,)
        ex_idx     = int(np.argmax(ex_probs))
        ex_conf    = float(ex_probs[ex_idx])
        ex_name    = self._idx_to_exercise.get(ex_idx, "unknown")

        boundary   = preds["boundary"][0, :, 0].tolist()     # (64,)
        rep_count  = float(preds["rep_count"][0, 0])
        quality    = float(preds["quality"][0, 0])
        joint_err  = preds["joint_errors"][0].tolist()        # (64, 10)

        return {
            "exercise":            ex_name,
            "exercise_confidence": ex_conf,
            "boundary":            boundary,
            "rep_count":           rep_count,
            "quality":             quality,
            "joint_errors":        joint_err,
        }
