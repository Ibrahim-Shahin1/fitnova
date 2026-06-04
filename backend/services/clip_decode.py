"""Serving-side video clip decode (cv2 backend) + knee-aware spatial preprocessing.

Why this exists:
  The training pipeline decodes video via `backend.training.aqa.datasets.transforms.decode_clip`,
  which wraps `torchvision.io.read_video` / `read_video_timestamps`. On the serving machine
  **torchvision 0.27 has removed both** (the TorchCodec migration), and neither `torchcodec`
  nor PyAV is installed — so `decode_clip` raises
  `AttributeError: module 'torchvision.io' has no attribute 'read_video'`.

  cv2 (opencv-python-headless, already a dependency and already used by `rep_segmenter.py`)
  is the only available video decoder. This module wraps it to produce the EXACT output
  contract `decode_clip` produced — a uint8 `[T, 3, H, W]` **RGB** tensor at the requested
  frame indices — so the downstream `spatial_val` (short-side resize → center crop → /255 →
  Kinetics-400 normalization) runs identically.

PARITY NOTE:
  The model was trained on Colab with `read_video` (RGB, TCHW). cv2 reads BGR; we convert
  BGR→RGB to match. Both decode the same H.264 frames; residual decoder differences are
  negligible relative to the spatial normalization that follows and to the phone-camera
  domain shift itself. This is a parity-APPROX, not a bit-identical reproduction — and it is
  the only option on torchvision 0.27.

DOMAIN-SHIFT FIX — `kneeaware_spatial_val`:
  The real-camera domain-shift test revealed that on PORTRAIT phone video (e.g.
  576×1024), the standard `spatial_val` path (short-side resize → 128 → center-crop 112²)
  keeps the MIDDLE horizontal band = the torso, cropping the knees out entirely. With the
  knees absent from the crop, the model gave zero KIE/KFE separation.

  A LOWER-BODY (knee-centered) pre-crop on portrait frames FIXES it — validated on 4 real
  clips: KFE separated good≈0.49 vs bad≈0.60; KIE went from noise to directionally correct
  (good≈0.06 vs bad≈0.13–0.24). `kneeaware_spatial_val` applies this portrait pre-crop
  then delegates to `spatial_val`, leaving landscape/square input UNCHANGED (offline-eval
  parity: dataset clips are landscape gym video — changing landscape behavior would shift the
  reported 0.6304 F1).

Used by: the latency/domain-shift probe (`backend/scripts/probe_squat_inference.py`), the
upload endpoint `POST /analyze-form-video`, and `SquatFormService.classify_clip`.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch

from backend.training.aqa.datasets.transforms import spatial_val

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Knee-aware spatial preprocessing constant
# ─────────────────────────────────────────────────────────────────────────────

# On portrait phone video the standard center-crop window falls on the TORSO,
# excluding the knees.  We pre-crop to the lower body (rows below this fraction
# of H) before delegating to spatial_val so the knees land in the center of the
# 112² crop.  Value tuned on 4 real phone clips.
# Only applied when H > W (portrait); landscape input is passed through unchanged.
PORTRAIT_LOWERBODY_TOP_FRAC: float = 0.42


def get_frame_count_and_fps(video_path: str) -> tuple[int, float]:
    """Return (frame_count, fps) for an mp4 via cv2.

    CAP_PROP_FRAME_COUNT is an estimate for some codecs; callers that need the exact
    count should rely on `decode_clip_cv2`, which reads sequentially and tolerates a
    count that is slightly high (missing trailing frames are mapped to the nearest
    decoded frame). fps defaults to 30.0 (Fitness-AQA Squat is uniformly 30fps).
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


def kneeaware_spatial_val(
    frames_tchw_uint8: torch.Tensor,
    *,
    crop_size: int = 112,
    resize_short: int = 128,
) -> torch.Tensor:
    """Portrait-aware serving preprocessing: lower-body pre-crop for portrait video, then spatial_val.

    WHY (domain-shift fix):
      On portrait phone video (H > W, e.g. 1024×576) the standard `spatial_val` path
      (short-side resize to 128, then 112² center-crop) keeps the middle horizontal band —
      the TORSO — and crops the knees entirely out of view.  With no knees in the input the
      model returns near-random KIE/KFE scores (validated: all clips scored within noise,
      KIE/KFE confidences ~0.1 with no good/bad separation).

      A lower-body pre-crop (rows starting at `PORTRAIT_LOWERBODY_TOP_FRAC * H`) moves the
      knee region into the center before the 112² crop is applied.  Validated on 4 real
      clips: KFE separation good≈0.49 vs bad≈0.60 restored; KIE separation good≈0.06 vs
      bad≈0.13–0.24 (directionally correct).

    PARITY RATIONALE (landscape unchanged):
      Fitness-AQA dataset clips are landscape gym video.  The offline 0.6304 F1 was measured
      with the standard `spatial_val` path.  Changing landscape preprocessing would break
      offline-eval reproducibility.  This function passes landscape/square input through to
      `spatial_val` UNCHANGED so the serving path and the offline-eval path stay identical
      for the dataset distribution.

    Args:
        frames_tchw_uint8: uint8 tensor `[T, 3, H, W]` RGB (from `decode_clip_cv2`).
        crop_size:         output spatial size (default 112, R(2+1)D-18 native).
        resize_short:      short-side resize target passed to `spatial_val` (default 128).

    Returns:
        float32 tensor `[3, T, crop_size, crop_size]` Kinetics-normalised — same output
        contract as `spatial_val`.  Portrait input goes through the lower-body pre-crop
        first; landscape/square input is passed directly to `spatial_val`.
    """
    _, _, H, W = frames_tchw_uint8.shape

    if H > W:
        # Portrait mode — pre-crop to the lower body so knees land in the center crop.
        top_row = int(PORTRAIT_LOWERBODY_TOP_FRAC * H)
        region_height = H - top_row          # rows from waist down
        # Square the region to the shorter of region_height and W so the center crop
        # is not distorted; take the top `side` rows of the region + all columns.
        side = min(region_height, W)
        lower_body = frames_tchw_uint8[:, :, top_row : top_row + side, :]
        # Center-crop columns to `side` if W > side (unlikely but defensive).
        if W > side:
            col_start = (W - side) // 2
            lower_body = lower_body[:, :, :, col_start : col_start + side]
        return spatial_val(lower_body, crop_size=crop_size, resize_short=resize_short)

    # Landscape or square — pass through unchanged for offline-eval parity.
    return spatial_val(frames_tchw_uint8, crop_size=crop_size, resize_short=resize_short)
