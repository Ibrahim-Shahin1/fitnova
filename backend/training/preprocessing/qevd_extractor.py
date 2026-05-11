"""MediaPipe pose extractor for the QEVD dataset (v6 pipeline).

Mirrors the live-inference pipeline in backend.services.form_analyzer
EXACTLY -- both use ``backend.services.mediapipe_config`` for the
PoseLandmarker options (VIDEO mode, confidence 0.5, 1 pose).  Drift
between training-time and inference-time MediaPipe was the root cause of
v4's real-phone failure (HANDOFF.md section 5).

Per-clip output schema (one .npz per clip):
    pose_canon       (T, 15, 4)  float32   canonical joints + visibility
    angles_raw       (T, 22)     float32   raw articulation/axis angles (rad)
    fps_native       float32                source video fps
    status_per_frame (T,)        uint8      0=ok 1=interp 2=no_pose
    no_pose_fraction float32

Usage (single clip):
    from pathlib import Path
    from backend.training.preprocessing.qevd_extractor import extract_clip
    feats = extract_clip(Path("video.mp4"))

Usage (batch):
    extract_directory(in_dir=Path("backend/data/qevd_raw/fit300k/train"),
                      out_dir=Path("backend/data/qevd_extracted/fit300k/train"),
                      n_workers=4)

See:
- plans/i-am-now-on-zazzy-brooks.md section II.4.3
- plans/phase-1-complete-critical-snappy-flurry.md section D0.5
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Output flags (as documented in master plan section II.4.3)
STATUS_OK = 0
STATUS_INTERP = 1
STATUS_NO_POSE = 2

# Maximum consecutive no-pose frames that will be forward-filled.
# Mirrors backend.services.form_analyzer.MAX_FILL_FRAMES.
MAX_FILL_FRAMES = 5

N_CANONICAL = 15
N_POSE_CHANNELS = 4   # (x, y, z, visibility)
N_ANGULAR = 22


# ─────────────────────────────────────────────────────────────────────────────
# Output container
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClipFeatures:
    """Per-clip extraction result."""
    pose_canon: np.ndarray              # (T, 15, 4) float32
    angles_raw: np.ndarray              # (T, 22)    float32
    fps_native: float
    status_per_frame: np.ndarray        # (T,)       uint8
    no_pose_fraction: float

    def save_npz(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            pose_canon=self.pose_canon.astype(np.float32),
            angles_raw=self.angles_raw.astype(np.float32),
            fps_native=np.float32(self.fps_native),
            status_per_frame=self.status_per_frame.astype(np.uint8),
            no_pose_fraction=np.float32(self.no_pose_fraction),
        )


@dataclass
class ExtractStats:
    """Aggregate statistics over a batch run."""
    n_clips_total: int = 0
    n_clips_ok: int = 0
    n_clips_dropped: int = 0
    n_clips_failed: int = 0
    aggregate_no_pose_fraction: float = 0.0
    aggregate_interp_fraction: float = 0.0
    failures: list = field(default_factory=list)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_clips_total":               self.n_clips_total,
            "n_clips_ok":                  self.n_clips_ok,
            "n_clips_dropped":             self.n_clips_dropped,
            "n_clips_failed":              self.n_clips_failed,
            "aggregate_no_pose_fraction":  self.aggregate_no_pose_fraction,
            "aggregate_interp_fraction":   self.aggregate_interp_fraction,
            "failures":                    self.failures[:50],  # cap for JSON
            "elapsed_s":                   self.elapsed_s,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Single-clip extraction (pure function)
# ─────────────────────────────────────────────────────────────────────────────


def extract_clip(mp4_path: Path, *, landmarker=None) -> ClipFeatures:
    """Extract pose + angles from one MP4.

    Parameters
    ----------
    mp4_path
        Path to the source MP4.
    landmarker
        Optional pre-built ``mediapipe.tasks.vision.PoseLandmarker`` instance.
        Pass one when calling repeatedly to avoid model-load overhead. When
        ``None``, builds a fresh one (slow: ~1 s startup per call).

    Returns
    -------
    ClipFeatures
    """
    import mediapipe as mp_lib
    from backend.services.mediapipe_config import (
        build_pose_landmarker_options,
    )
    from backend.training.preprocessing.angular_features import (
        compute_frame_angles,
    )
    from backend.training.preprocessing.joint_mapping import (
        MEDIAPIPE_JOINTS, MEDIAPIPE_TO_CANONICAL,
    )
    from backend.training.preprocessing.normalize import (
        extract_canonical_from_mediapipe, normalize_skeleton,
    )

    mp4_path = Path(mp4_path)
    if not mp4_path.exists():
        raise FileNotFoundError(mp4_path)

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture failed to open {mp4_path}")

    fps_native = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_interval_ms = max(1, int(1000.0 / fps_native))

    # ── Build landmarker if not supplied ──────────────────────────────────────
    own_landmarker = landmarker is None
    if own_landmarker:
        PoseLandmarker = mp_lib.tasks.vision.PoseLandmarker
        landmarker = PoseLandmarker.create_from_options(
            build_pose_landmarker_options()
        )

    mp_image_cls = mp_lib.Image
    mp_image_fmt = mp_lib.ImageFormat

    pose_frames: list[np.ndarray] = []   # (15, 4) per frame
    angle_frames: list[np.ndarray] = []  # (22,)   per frame
    statuses: list[int] = []
    prev_pose_canon: Optional[np.ndarray] = None
    prev_angles: Optional[np.ndarray] = None
    fill_streak = 0
    ts_ms = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp_image_cls(image_format=mp_image_fmt.SRGB,
                                   data=frame_rgb)

            # detect_for_video requires monotonically increasing timestamps
            current_ts = ts_ms
            ts_ms += frame_interval_ms

            result = landmarker.detect_for_video(mp_img, current_ts)

            if (not result.pose_landmarks
                    or not result.pose_world_landmarks):
                # Forward-fill if possible (matches form_analyzer B7 fix)
                if prev_pose_canon is not None and fill_streak < MAX_FILL_FRAMES:
                    fill_streak += 1
                    pose_frames.append(prev_pose_canon)
                    angle_frames.append(prev_angles)
                    statuses.append(STATUS_INTERP)
                else:
                    pose_frames.append(
                        np.zeros((N_CANONICAL, N_POSE_CHANNELS), dtype=np.float32)
                    )
                    angle_frames.append(
                        np.zeros(N_ANGULAR, dtype=np.float32)
                    )
                    statuses.append(STATUS_NO_POSE)
                continue

            # Pose detected — reset streak
            fill_streak = 0

            mp_world = np.array(
                [[lm.x, lm.y, lm.z] for lm in result.pose_world_landmarks[0]],
                dtype=np.float32,
            )  # (33, 3)

            # ── Canonical 15 joints, hip-centred + torso-scaled ──────────────
            canonical_world = extract_canonical_from_mediapipe(mp_world)  # (15, 3)
            canon_norm = normalize_skeleton(canonical_world[np.newaxis])[0]

            # ── Visibility per canonical joint (matches form_analyzer.py:362-392)
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
                    canon_vis[can_idx] = 0.25 * (
                        mp_vis[l] + mp_vis[r] + mp_vis[le] + mp_vis[re]
                    )
                elif isinstance(mp_idx, tuple):
                    canon_vis[can_idx] = np.mean(
                        [mp_vis[i] for i in mp_idx], axis=0,
                    )
                else:
                    canon_vis[can_idx] = mp_vis[mp_idx]
            canon_vis[14] = 0.5 * (canon_vis[12] + canon_vis[13])

            pose_canon = np.concatenate(
                [canon_norm, canon_vis], axis=-1,
            ).astype(np.float32)  # (15, 4)

            angles = compute_frame_angles(canon_norm).astype(np.float32)  # (22,)

            pose_frames.append(pose_canon)
            angle_frames.append(angles)
            statuses.append(STATUS_OK)
            prev_pose_canon = pose_canon
            prev_angles = angles
    finally:
        cap.release()
        if own_landmarker:
            try:
                landmarker.close()
            except Exception:
                pass

    if not pose_frames:
        raise RuntimeError(
            f"No frames decoded from {mp4_path} -- is the file empty/corrupt?"
        )

    pose_arr = np.stack(pose_frames, axis=0).astype(np.float32)
    angles_arr = np.stack(angle_frames, axis=0).astype(np.float32)
    status_arr = np.array(statuses, dtype=np.uint8)
    no_pose_frac = float((status_arr == STATUS_NO_POSE).mean())

    return ClipFeatures(
        pose_canon=pose_arr,
        angles_raw=angles_arr,
        fps_native=fps_native,
        status_per_frame=status_arr,
        no_pose_fraction=no_pose_frac,
    )


def extract_clip_to_npz(mp4_path: Path, out_path: Path) -> ClipFeatures:
    """Extract one clip and save it as .npz. Returns the features."""
    feats = extract_clip(mp4_path)
    feats.save_npz(out_path)
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Multi-process directory walk
# ─────────────────────────────────────────────────────────────────────────────

def _worker_extract(args: tuple) -> dict:
    """Pool task: extract one clip; return a status dict.

    Creates a fresh PoseLandmarker for each clip rather than sharing one
    across clips. Sharing trips MediaPipe's monotonic-timestamp invariant
    in VIDEO mode (each new clip resets ts to 0, which is < the last ts
    seen by the shared landmarker → ValueError). The fresh-per-clip cost
    is ~100 ms overhead per clip, which is dominated by the ~5 s of pose
    extraction; over 298K clips at 4 workers this adds about 2 hr of
    aggregate setup time but eliminates the bug class entirely.
    """
    mp4_path, out_path = args
    try:
        feats = extract_clip(Path(mp4_path))   # fresh landmarker each call
        feats.save_npz(Path(out_path))
        n_frames = int(feats.status_per_frame.shape[0])
        n_interp = int((feats.status_per_frame == STATUS_INTERP).sum())
        return {
            "ok":               True,
            "mp4":              str(mp4_path),
            "n_frames":         n_frames,
            "no_pose_fraction": feats.no_pose_fraction,
            "interp_fraction":  (n_interp / n_frames) if n_frames else 0.0,
            "fps_native":       feats.fps_native,
        }
    except Exception as e:
        return {
            "ok":    False,
            "mp4":   str(mp4_path),
            "error": f"{type(e).__name__}: {e}",
        }


def extract_directory(
    in_dir: Path,
    out_dir: Path,
    n_workers: int = 4,
    progress_every: int = 10_000,
    drop_threshold: float = 0.50,
) -> ExtractStats:
    """Walk `in_dir/<human_id>/<clip>.mp4` and extract each.

    Outputs `<out_dir>/<human_id>/<clip>.npz`.

    Parameters
    ----------
    in_dir : Path
    out_dir : Path
    n_workers : int
        ``multiprocessing.Pool`` size.  4 is the master-plan default.
    progress_every : int
        How often to log + flush a partial D2 metrics report.
    drop_threshold : float
        Per-clip ``no_pose_fraction`` above which the clip is logged as
        dropped (not used for training).  See master plan section II.4.3.
    """
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    if not in_dir.exists():
        raise FileNotFoundError(in_dir)

    mp4s = sorted(in_dir.rglob("*.mp4"))
    if not mp4s:
        logger.warning("No .mp4 files under %s", in_dir)
        return ExtractStats()

    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    n_skipped = 0
    for mp4 in mp4s:
        rel = mp4.relative_to(in_dir).with_suffix(".npz")
        out_path = out_dir / rel
        # Resume support: skip clips whose .npz already exists. This makes
        # the extractor restartable across PC reboots / interruptions.
        if out_path.exists() and out_path.stat().st_size > 0:
            n_skipped += 1
            continue
        tasks.append((str(mp4), str(out_path)))

    if n_skipped:
        logger.info("qevd_extractor: skipping %d clips with existing .npz "
                    "(resume mode); %d clips remaining",
                    n_skipped, len(tasks))

    stats = ExtractStats(n_clips_total=len(tasks))
    no_pose_running = 0.0
    interp_running = 0.0
    n_done = 0
    t_start = time.time()

    logger.info(
        "qevd_extractor: %d clips, %d workers (%s -> %s)",
        len(tasks), n_workers, in_dir, out_dir,
    )

    ctx = mp.get_context("spawn")  # spawn is required on Windows
    with ctx.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_worker_extract, tasks, chunksize=8):
            n_done += 1
            if result["ok"]:
                stats.n_clips_ok += 1
                no_pose_running += result["no_pose_fraction"]
                interp_running += result["interp_fraction"]
                if result["no_pose_fraction"] > drop_threshold:
                    stats.n_clips_dropped += 1
                    stats.failures.append({
                        "mp4":              result["mp4"],
                        "reason":           "no_pose_fraction > drop_threshold",
                        "no_pose_fraction": result["no_pose_fraction"],
                    })
            else:
                stats.n_clips_failed += 1
                stats.failures.append({
                    "mp4":   result["mp4"],
                    "error": result.get("error", "unknown"),
                })

            if n_done % progress_every == 0:
                logger.info(
                    "  %d/%d clips done; no_pose=%.4f, interp=%.4f, "
                    "drop=%d, fail=%d",
                    n_done, len(tasks),
                    no_pose_running / max(1, stats.n_clips_ok),
                    interp_running / max(1, stats.n_clips_ok),
                    stats.n_clips_dropped, stats.n_clips_failed,
                )

    if stats.n_clips_ok > 0:
        stats.aggregate_no_pose_fraction = no_pose_running / stats.n_clips_ok
        stats.aggregate_interp_fraction = interp_running / stats.n_clips_ok
    stats.elapsed_s = time.time() - t_start
    logger.info(
        "qevd_extractor done: %d ok, %d dropped, %d failed in %.1f s",
        stats.n_clips_ok, stats.n_clips_dropped, stats.n_clips_failed,
        stats.elapsed_s,
    )
    return stats


__all__ = [
    "ClipFeatures",
    "ExtractStats",
    "MAX_FILL_FRAMES",
    "STATUS_INTERP",
    "STATUS_NO_POSE",
    "STATUS_OK",
    "extract_clip",
    "extract_clip_to_npz",
    "extract_directory",
]
