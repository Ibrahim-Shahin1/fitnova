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
        # All FitNova-specific MediaPipe config (confidences, model asset,
        # VIDEO running-mode) lives in backend.services.mediapipe_config so
        # the QEVD training extractor (backend.training.preprocessing.
        # qevd_extractor) and this inference path produce features under
        # identical settings.  Drift between training/inference MediaPipe
        # was the root cause of v4's real-phone failure (HANDOFF.md §5).
        from backend.services.mediapipe_config import (
            build_pose_landmarker_options,
        )

        PoseLandmarker    = mp.tasks.vision.PoseLandmarker
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = build_pose_landmarker_options()  # VIDEO mode + B6 fix
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
        """Detect model version from model_dir and load accordingly.

        Detection priority (highest first):
          v6 — QEVD-trained ST-GCN with native 10-channel joint_err and 25-class
               action head. Detected by ``v6_supervised.weights.h5``.
          v5.2 — Fit3D-trained ST-GCN with 5-group joint_err (remapped to 10
                 for UI compat). Detected by ``v5_2_supervised.weights.h5``.
          v4 — Legacy MT-TCN. Used when neither v6 nor v5.2 weights are present.
        """
        v6_weights   = os.path.join(self.model_dir, "v6_supervised.weights.h5")
        v5_2_weights = os.path.join(self.model_dir, "v5_2_supervised.weights.h5")

        if os.path.exists(v6_weights) and _TF_AVAILABLE:
            self._model_version = "v6"
            self._load_v6_model(v6_weights)
            return

        if os.path.exists(v5_2_weights) and _TF_AVAILABLE:
            self._model_version = "v5.2"
            self._load_v5_2_model(v5_2_weights)
            return

        # ── v4 legacy path ────────────────────────────────────────────────────
        self._model_version = "v4"
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
                f"No v5.2 or v4 weights found in {self.model_dir}. "
                "Train the model first."
            )
            return

        if not _TF_AVAILABLE:
            logger.error("TensorFlow not available — model loading skipped.")
            return

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
        self._model.load_weights(weights_path)
        logger.info("v4 MT-TCN weights loaded (%.1f MB).", os.path.getsize(weights_path) / 1e6)

        self._normalizer = AngleNormalizer.load(stats_path)
        logger.info("Angle normaliser loaded.")

        with open(labels_path) as f:
            self._exercise_labels = json.load(f)
        self._idx_to_exercise = {v: k for k, v in self._exercise_labels.items()}
        logger.info(f"v4 exercise labels loaded: {len(self._exercise_labels)} classes.")

    def _load_v5_2_model(self, weights_path: str) -> None:
        """Load the v5.2 ST-GCN multi-task model.

        v5.2 takes 3 inputs (pose, angles, exercise_id) and produces 5 named
        outputs (quality, action, rep_count, boundary, joint_err) plus a
        trunk-features tensor for multi-view consistency loss (unused at
        inference). The model file ``v5_2_supervised.weights.h5`` contains
        only the trainable weights (Keras 3 ``.weights.h5`` format), so
        layers are matched positionally.
        """
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from backend.training.models.st_gcn import build_v5_model

        config_path = os.path.join(self.model_dir, "model_config.json")
        labels_path = os.path.join(self.model_dir, "exercise_labels.json")

        with open(config_path) as f:
            cfg = json.load(f)

        self._model = build_v5_model(
            target_frames    = cfg.get("target_frames", 64),
            n_joints         = cfg.get("n_canonical_joints", 15),
            n_pose_channels  = cfg.get("n_pose_channels", 4),
            n_angular        = cfg.get("n_angular", 22),
            n_exercises      = cfg.get("n_exercises", 15),
            n_joint_groups   = cfg.get("n_joint_groups", 5),
        )
        self._model.load_weights(weights_path)
        logger.info("v5.2 ST-GCN weights loaded (%.1f MB, %s params).",
                    os.path.getsize(weights_path) / 1e6,
                    f"{self._model.count_params():,}")

        with open(labels_path) as f:
            self._exercise_labels = json.load(f)
        self._idx_to_exercise = {v: k for k, v in self._exercise_labels.items()}
        logger.info(f"v5.2 exercise labels loaded: {len(self._exercise_labels)} classes.")
        # The angle "normaliser" from v4 (per-feature mean/std) is not used in
        # v5.2 — angles go in raw. Mark as None so process_frame skips the
        # transform.
        self._normalizer = None

    def _load_v6_model(self, weights_path: str) -> None:
        """Load the v6 QEVD-trained ST-GCN multi-task model.

        v6 takes 3 inputs (pose, angles, exercise_id) — same shape as v5.2 —
        but produces NATIVE 10-channel joint_err (no remap needed) and a
        25-class action head trained on QEVD's top-24 named exercises plus
        '__other__'. Exercise map lives in qevd_exercise_map.json (NOT
        exercise_labels.json — that's a v4/v5.2 file).

        The known D9 failure mode is documented in
        backend/data/qevd_phase_reports/D9_reality_check.json. This integration
        wires v6 in for end-to-end testing; the recovery (Tier 2 GPT-4o-mini
        relabel + retrain) is tracked separately.
        """
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from backend.training.models.st_gcn_v6 import build_v6_model

        config_path = os.path.join(self.model_dir, "model_config.json")
        ex_map_path = os.path.join(self.model_dir, "qevd_exercise_map.json")

        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"v6 missing model_config.json at {config_path}"
            )
        if not os.path.isfile(ex_map_path):
            raise FileNotFoundError(
                f"v6 missing qevd_exercise_map.json at {ex_map_path}"
            )

        with open(config_path) as f:
            cfg = json.load(f)

        self._model = build_v6_model(
            target_frames    = cfg.get("target_frames", 64),
            n_joints         = cfg.get("n_canonical_joints", 15),
            n_pose_channels  = cfg.get("n_pose_channels", 4),
            n_angular        = cfg.get("n_angular", 22),
            n_exercises      = cfg.get("n_exercises", 25),
            n_joint_groups   = cfg.get("n_joint_groups", 10),
        )
        self._model.load_weights(weights_path)
        logger.info("v6 ST-GCN weights loaded (%.1f MB, %s params).",
                    os.path.getsize(weights_path) / 1e6,
                    f"{self._model.count_params():,}")

        # Load QEVD exercise map (name → idx). Both directions kept for
        # convenience: name→idx for conditioning, idx→name for top-1 display.
        with open(ex_map_path) as f:
            self._exercise_labels = json.load(f)
        self._idx_to_exercise = {v: k for k, v in self._exercise_labels.items()}
        logger.info(
            f"v6 QEVD exercise map loaded: {len(self._exercise_labels)} classes "
            f"(includes '__other__'={self._exercise_labels.get('__other__', '?')})."
        )
        # v6 uses raw angles (same as v5.2) — no AngleNormalizer.
        self._normalizer = None

        # ── DEBUG: dump model summary if FITNOVA_DEBUG=1 ─────────────────────
        if os.environ.get("FITNOVA_DEBUG") == "1":
            try:
                summary_lines = []
                self._model.summary(print_fn=lambda l: summary_lines.append(l))
                logger.info("v6 model summary:\n%s", "\n".join(summary_lines))
            except Exception as e:
                logger.warning("could not produce model summary: %s", e)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def model_ready(self) -> bool:
        if self._model is None:
            return False
        # v6 + v5.2 don't use the AngleNormalizer (raw angles in); v4 requires it.
        if getattr(self, "_model_version", "v4") in ("v6", "v5.2"):
            return True
        return self._normalizer is not None

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

        # ── v5.2: also build a per-canonical-joint visibility channel (15, 1) ──
        # Mirrors what dataset_builder_v5_1._slice_canonicalize() does. The
        # canonical-15 visibility is the average of source-MediaPipe joints
        # that map to each canonical index (with the B5 neck weighting). Used
        # only by the v5.2 model's pose input; the v4 model ignores it.
        from backend.training.preprocessing.joint_mapping import (
            MEDIAPIPE_JOINTS, MEDIAPIPE_TO_CANONICAL, N_CANONICAL,
        )
        mp_vis = np.array(
            [[float(getattr(lm, "visibility", 0.0))]
             for lm in result.pose_world_landmarks[0]],
            dtype=np.float32,
        )  # (33, 1)
        canon_vis = np.zeros((N_CANONICAL, 1), dtype=np.float32)
        for can_idx, mp_idx in MEDIAPIPE_TO_CANONICAL.items():
            if can_idx == 14:
                continue
            if mp_idx is None:
                continue
            if mp_idx == "neck_weighted":
                l = MEDIAPIPE_JOINTS["l_shoulder"]
                r = MEDIAPIPE_JOINTS["r_shoulder"]
                le = MEDIAPIPE_JOINTS["l_ear"]
                re = MEDIAPIPE_JOINTS["r_ear"]
                canon_vis[can_idx] = 0.25 * (mp_vis[l] + mp_vis[r] + mp_vis[le] + mp_vis[re])
            elif isinstance(mp_idx, tuple):
                canon_vis[can_idx] = np.mean([mp_vis[i] for i in mp_idx], axis=0)
            else:
                canon_vis[can_idx] = mp_vis[mp_idx]
        canon_vis[14] = 0.5 * (canon_vis[12] + canon_vis[13])

        # (15, 4) pose tensor — matches v5.2 model input schema
        pose_canon = np.concatenate([canon_norm, canon_vis], axis=-1).astype(np.float32)

        result_dict = {
            "landmarks":        canonical_img.tolist(),    # 15 × [x,y,z] image coords (display)
            "joints_norm":      canon_norm.tolist(),       # 15 × [x,y,z] skeleton-normalised world coords
            "pose_canon":       pose_canon.tolist(),       # 15 × [x,y,z,vis] for v5.2 model input
            "angles_norm":      angles_norm.tolist(),       # 22 angular features
            "status":           "ok",
        }
        self._prev_result = result_dict          # cache for B7 forward-fill
        return result_dict

    def predict_window(
        self,
        angles_window: np.ndarray,
        joints_window: np.ndarray,
        pose_window: Optional[np.ndarray] = None,
        exercise_idx: Optional[int] = None,
    ) -> dict:
        """
        Run multi-task model on a 64-frame window. Dispatches by version:

        v5.2 mode (uses ``pose_window`` (64, 15, 4) + ``angles_window`` (64, 22)
        + ``exercise_idx``): conditions on user-selected exercise; returns
        5-group joint_err which we expand to 10-group for backward
        compatibility with the existing FormSession + Flutter UI.

        v4 mode (uses ``angles_window`` (64, 22) + ``joints_window`` (64, 45)):
        legacy MT-TCN path, returns 10-group joint_err directly.

        Returns:
            {
              "exercise": str, "exercise_confidence": float,
              "boundary": [float × 64], "rep_count": float,
              "quality": float,
              "joint_errors": [[float × 10] × 64],  # always 10-group, see remap
              "model_version": "v5.2" | "v4",
            }
        """
        if not self.model_ready:
            return {
                "exercise": "unknown", "exercise_confidence": 0.0,
                "boundary": [0.0] * 64, "rep_count": 0.0,
                "quality": 0.5, "joint_errors": [[0.0] * 10] * 64,
                "model_version": "none",
            }

        version = getattr(self, "_model_version", "v4")
        if version == "v6":
            return self._predict_window_v6(
                angles_window=angles_window,
                pose_window=pose_window,
                exercise_idx=exercise_idx if exercise_idx is not None else 0,
            )
        if version == "v5.2":
            return self._predict_window_v5_2(
                angles_window=angles_window,
                pose_window=pose_window,
                exercise_idx=exercise_idx if exercise_idx is not None else 0,
            )
        return self._predict_window_v4(angles_window, joints_window)

    def _predict_window_v4(
        self, angles_window: np.ndarray, joints_window: np.ndarray,
    ) -> dict:
        inp = {
            "angular_input": angles_window[np.newaxis].astype(np.float32),
            "joints_input":  joints_window[np.newaxis].astype(np.float32),
        }
        preds = self._model.predict(inp, verbose=0)

        ex_probs = preds["exercise"][0]
        ex_idx   = int(np.argmax(ex_probs))
        ex_conf  = float(ex_probs[ex_idx])
        ex_name  = self._idx_to_exercise.get(ex_idx, "unknown")

        return {
            "exercise":            ex_name,
            "exercise_confidence": ex_conf,
            "boundary":            preds["boundary"][0, :, 0].tolist(),
            "rep_count":           float(preds["rep_count"][0, 0]),
            "quality":             float(preds["quality"][0, 0]),
            "joint_errors":        preds["joint_errors"][0].tolist(),  # (64, 10)
            "model_version":       "v4",
        }

    def _predict_window_v5_2(
        self,
        angles_window: np.ndarray,
        pose_window: Optional[np.ndarray],
        exercise_idx: int,
    ) -> dict:
        """v5.2 ST-GCN inference. ``pose_window`` is (64, 15, 4)."""
        if pose_window is None:
            logger.error("v5.2 predict_window called without pose_window — returning neutral")
            return {
                "exercise": "unknown", "exercise_confidence": 0.0,
                "boundary": [0.0] * 64, "rep_count": 0.0,
                "quality": 0.5, "joint_errors": [[0.0] * 10] * 64,
                "model_version": "v5.2",
            }

        inp = {
            "pose":        pose_window[np.newaxis].astype(np.float32),
            "angles":      angles_window[np.newaxis].astype(np.float32),
            "exercise_id": np.array([int(exercise_idx)], dtype=np.int32),
        }
        preds = self._model.predict(inp, verbose=0)

        ex_probs = preds["action"][0]                             # (15,)
        ex_idx   = int(np.argmax(ex_probs))
        ex_conf  = float(ex_probs[ex_idx])
        ex_name  = self._idx_to_exercise.get(ex_idx, "unknown")
        boundary  = preds["boundary"][0, :, 0].tolist()           # (64,)
        rep_count = float(preds["rep_count"][0, 0])
        quality   = float(preds["quality"][0, 0])
        je_v5     = preds["joint_err"][0]                         # (64, 5)

        # ── v5 → v4 joint-group remapping for Flutter UI compatibility ──────
        # v5 order: [knee, hip, back, shoulder, elbow]
        # v4 order: [L_elbow, R_elbow, L_shoulder, R_shoulder,
        #            L_knee,  R_knee,  L_hip,      R_hip,      trunk, neck]
        # Bilateral groups duplicated; back maps to both trunk + neck.
        je_v4 = np.zeros((je_v5.shape[0], 10), dtype=np.float32)
        je_v4[:, 0] = je_v5[:, 4]   # L_elbow    ← elbow
        je_v4[:, 1] = je_v5[:, 4]   # R_elbow    ← elbow
        je_v4[:, 2] = je_v5[:, 3]   # L_shoulder ← shoulder
        je_v4[:, 3] = je_v5[:, 3]   # R_shoulder ← shoulder
        je_v4[:, 4] = je_v5[:, 0]   # L_knee     ← knee
        je_v4[:, 5] = je_v5[:, 0]   # R_knee     ← knee
        je_v4[:, 6] = je_v5[:, 1]   # L_hip      ← hip
        je_v4[:, 7] = je_v5[:, 1]   # R_hip      ← hip
        je_v4[:, 8] = je_v5[:, 2]   # trunk      ← back
        je_v4[:, 9] = je_v5[:, 2]   # neck       ← back

        return {
            "exercise":            ex_name,
            "exercise_confidence": ex_conf,
            "boundary":            boundary,
            "rep_count":           rep_count,
            "quality":             quality,
            "joint_errors":        je_v4.tolist(),
            "joint_errors_v5":     je_v5.tolist(),  # also expose raw 5-group
            "model_version":       "v5.2",
        }

    def _predict_window_v6(
        self,
        angles_window: np.ndarray,
        pose_window: Optional[np.ndarray],
        exercise_idx: int,
    ) -> dict:
        """v6 ST-GCN inference (QEVD-trained).

        v6's joint_err head is NATIVE 10-channel — same order as
        ``form_session.JOINT_GROUP_NAMES``. No remap step needed.

        Heavy debug logging when ``FITNOVA_DEBUG=1`` env var is set: every
        prediction logs per-head distribution stats so we can correlate
        Flutter UI behaviour with raw model outputs.
        """
        debug = os.environ.get("FITNOVA_DEBUG") == "1"

        if pose_window is None:
            logger.error(
                "v6 predict_window called without pose_window — returning neutral"
            )
            return {
                "exercise": "unknown", "exercise_confidence": 0.0,
                "boundary": [0.0] * 64, "rep_count": 0.0,
                "quality": 0.5, "joint_errors": [[0.0] * 10] * 64,
                "model_version": "v6",
            }

        # ── Validate input shapes (catch silent contract drift early) ────────
        T = pose_window.shape[0]
        if pose_window.shape != (T, 15, 4):
            logger.error(
                "v6 pose_window has wrong shape %s, expected (T, 15, 4); "
                "returning neutral", pose_window.shape,
            )
            return {
                "exercise": "unknown", "exercise_confidence": 0.0,
                "boundary": [0.0] * 64, "rep_count": 0.0,
                "quality": 0.5, "joint_errors": [[0.0] * 10] * 64,
                "model_version": "v6",
            }
        if angles_window.shape != (T, 22):
            logger.error(
                "v6 angles_window has wrong shape %s, expected (T, 22); "
                "returning neutral", angles_window.shape,
            )
            return {
                "exercise": "unknown", "exercise_confidence": 0.0,
                "boundary": [0.0] * 64, "rep_count": 0.0,
                "quality": 0.5, "joint_errors": [[0.0] * 10] * 64,
                "model_version": "v6",
            }

        if debug:
            logger.info(
                "v6 input: pose %s [%.3f..%.3f], angles %s [%.3f..%.3f], "
                "exercise_idx=%d (%s)",
                pose_window.shape, float(pose_window.min()), float(pose_window.max()),
                angles_window.shape, float(angles_window.min()), float(angles_window.max()),
                int(exercise_idx),
                self._idx_to_exercise.get(int(exercise_idx), "?"),
            )

        inp = {
            "pose":        pose_window[np.newaxis].astype(np.float32),
            "angles":      angles_window[np.newaxis].astype(np.float32),
            "exercise_id": np.array([int(exercise_idx)], dtype=np.int32),
        }
        preds = self._model.predict(inp, verbose=0)

        # ── Action head (25 classes incl. __other__) ─────────────────────────
        action_probs = preds["action"][0]                   # (25,)
        action_idx   = int(np.argmax(action_probs))
        action_conf  = float(action_probs[action_idx])
        action_name  = self._idx_to_exercise.get(action_idx, "unknown")

        # ── Other heads ──────────────────────────────────────────────────────
        boundary  = preds["boundary"][0, :, 0].tolist()     # (64,)
        rep_count = float(preds["rep_count"][0, 0])
        quality   = float(preds["quality"][0, 0])
        je        = preds["joint_err"][0]                    # (64, 10) NATIVE

        if debug:
            logger.info(
                "v6 output: action top1=%d conf=%.3f (%s)  q=%.3f  rep=%.3f  "
                "bnd_max=%.3f  je_max=%.3f  je_per_group_max=%s",
                action_idx, action_conf, action_name,
                quality, rep_count, max(boundary, default=0.0),
                float(je.max()),
                ["%.2f" % x for x in je.max(axis=0).tolist()],
            )

        return {
            "exercise":            action_name,
            "exercise_confidence": action_conf,
            "boundary":            boundary,
            "rep_count":           rep_count,
            "quality":             quality,
            "joint_errors":        je.tolist(),     # native (64, 10)
            "model_version":       "v6",
        }
