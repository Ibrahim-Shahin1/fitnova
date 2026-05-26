"""Serving-side video clip decode (cv2 backend).

WHY THIS EXISTS (Phase 5 deviation — documented):
  The training pipeline decodes video via `backend.training.aqa.datasets.transforms.decode_clip`,
  which wraps `torchvision.io.read_video` / `read_video_timestamps`. On the serving machine
  **torchvision 0.27 has removed both** (the TorchCodec migration the Phase-2 decisions
  anticipated), and neither `torchcodec` nor PyAV is installed — so `decode_clip` raises
  `AttributeError: module 'torchvision.io' has no attribute 'read_video'`.

  cv2 (opencv-python-headless, already a dependency and already used by `rep_segmenter.py`)
  is the only available video decoder. This module wraps it to produce the EXACT output
  contract `decode_clip` produced — a uint8 `[T, 3, H, W]` **RGB** tensor at the requested
  frame indices — so the downstream `spatial_val` (short-side resize → center crop → /255 →
  Kinetics-400 normalization) runs identically.

PARITY NOTE (the #1 Phase-5 risk):
  The model was trained on Colab with `read_video` (RGB, TCHW). cv2 reads BGR; we convert
  BGR→RGB to match. Both decode the same H.264 frames; residual decoder differences are
  negligible relative to the spatial normalization that follows and to the phone-camera
  domain shift itself. This is a parity-APPROX, not a bit-identical reproduction — and it is
  the only option on torchvision 0.27.

Used by: the latency/domain-shift probe (`backend/scripts/probe_squat_inference.py`) and the
upload endpoint `POST /analyze-form-video` (Plan 05-03).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_frame_count_and_fps(video_path: str) -> tuple[int, float]:
    """Return (frame_count, fps) for an mp4 via cv2.

    CAP_PROP_FRAME_COUNT is an estimate for some codecs; callers that need the exact
    count should rely on `decode_clip_cv2`, which reads sequentially and tolerates a
    count that is slightly high (missing trailing frames are mapped to the nearest
    decoded frame). fps defaults to 30.0 (Phase-1: Fitness-AQA Squat is uniformly 30fps).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"clip_decode: cv2 could not open {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    cap.release()
    return n, fps


def decode_clip_cv2(video_path: str, indices: torch.Tensor) -> torch.Tensor:
    """Decode the frames at `indices` from `video_path` as uint8 `[T, 3, H, W]` RGB.

    Drop-in replacement for `transforms.decode_clip` on the serving machine (torchvision
    0.27 removed `read_video`). Reads sequentially up to `max(indices)` and keeps only the
    requested frames (memory-bounded to len(indices) full-resolution frames, NOT the whole
    clip), converts BGR→RGB, and returns frames in the order given by `indices` (which may
    repeat indices when `num_frames < target`). The downstream `spatial_val` does the resize
    + center-crop + Kinetics normalization, so frames are returned at native resolution.

    Args:
        video_path: absolute path to the .mp4.
        indices:    LongTensor of frame indices (from `uniform_sample_indices`).

    Returns:
        uint8 tensor `[len(indices), 3, H, W]` (RGB, TCHW) — same contract as `decode_clip`.

    Raises:
        ValueError: cv2 could not open the file or decoded zero frames.
    """
    idx_list = [int(i) for i in (indices.tolist() if hasattr(indices, "tolist") else list(indices))]
    if not idx_list:
        raise ValueError("decode_clip_cv2: empty indices")
    want = set(idx_list)
    max_want = max(idx_list)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"decode_clip_cv2: cv2 could not open {video_path}")

    frames_by_idx: dict[int, np.ndarray] = {}
    fi = 0
    while fi <= max_want:
        ret, frame = cap.read()  # BGR uint8 [H, W, 3]
        if not ret:
            break
        if fi in want:
            frames_by_idx[fi] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB uint8 [H, W, 3]
        fi += 1
    cap.release()

    if not frames_by_idx:
        raise ValueError(f"decode_clip_cv2: decoded 0 frames from {video_path}")

    if len(frames_by_idx) < len(want):
        logger.warning(
            "decode_clip_cv2: requested %d distinct indices but decoded %d (clip shorter than "
            "CAP_PROP_FRAME_COUNT estimate); missing indices mapped to nearest decoded frame.",
            len(want),
            len(frames_by_idx),
        )

    available = sorted(frames_by_idx)
    picked: list[np.ndarray] = []
    for i in idx_list:
        if i in frames_by_idx:
            picked.append(frames_by_idx[i])
        else:
            nearest = min(available, key=lambda k: abs(k - i))
            picked.append(frames_by_idx[nearest])

    arr = np.stack(picked, axis=0)                       # [T, H, W, 3] uint8 RGB
    arr = np.ascontiguousarray(arr.transpose(0, 3, 1, 2))  # [T, 3, H, W] uint8 RGB
    return torch.from_numpy(arr)
