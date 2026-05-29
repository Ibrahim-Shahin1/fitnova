"""Window-bounded video decode + 32-frame uniform sampler + spatial preprocessing for the Squat KIE/KFE pipeline.

Wraps `torchvision.io.read_video` behind `decode_clip(path, indices)` so a future swap to TorchCodec (when/if Colab's torchvision passes 0.24 / `read_video` removal) is a one-line change — see D6 + RESEARCH §5. Window-bounded decode (F11 Option A) caps memory at the picked-indices span instead of reading the full clip, defending against OOM on long-tail Squat clips (Phase 1 measured up to 404 frames at width 480 × height up to 600 — ~350 MB uncompressed if decoded whole).

Spatial pipeline: short-side resize to 128 → 112² random crop (train) or center crop (val/test) → /255 → Kinetics-400 normalization → permute to (C, T, H, W). NO horizontal flip — paper line 576 augmentation list omits it; CVCSPC code has it commented out.

See: `.planning/phases/02-squat-data-pipeline-colab-harness/02-01-PLAN.md` — Task 5 / D1 / D5 / D6 / interfaces block.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Final

import torch
import torchvision.io
import torchvision.transforms.functional as TF

logger = logging.getLogger("aqa.phase02")

# torchvision >= 0.26 REMOVED `read_video` / `read_video_timestamps` (the TorchCodec
# migration the Phase-2 decisions anticipated — confirmed live on Colab 2026-05, tv
# 0.26.0+cu128). On those versions `decode_clip` + `count_frames` transparently fall back
# to a cv2 backend (opencv-python-headless, already a project dependency). Detected once at
# import so the per-call routing is cheap. The cv2 path mirrors the Phase-5 serving twin
# `backend/services/clip_decode.decode_clip_cv2` (kept here so the training transforms layer
# stays self-contained — no training->services import).
_HAS_TV_READ_VIDEO = hasattr(torchvision.io, "read_video") and hasattr(
    torchvision.io, "read_video_timestamps"
)

# Kinetics-400 normalization — matches torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1.
# NOT ImageNet stats (CVCSPC uses ImageNet for its 2D image path — different model family).
KINETICS_MEAN: Final[tuple[float, float, float]] = (0.43216, 0.394666, 0.37645)
KINETICS_STD: Final[tuple[float, float, float]] = (0.22803, 0.22145, 0.216989)

# Module-level normalization tensors (lazy-broadcast over T, H, W in spatial_* below).
# Shape [3, 1, 1, 1] — applied to (T, C, H, W) after `unsqueeze(0)` then squeezed implicitly via broadcasting.
_MEAN_T = torch.tensor(KINETICS_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
_STD_T = torch.tensor(KINETICS_STD, dtype=torch.float32).view(1, 3, 1, 1)


def uniform_sample_indices(
    num_frames: int,
    target: int = 32,
    jitter: int = 0,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick `target` frame indices uniformly across `[0, num_frames-1]`, with optional jitter.

    Args:
        num_frames: total frames in the source clip (from `read_video_timestamps`).
        target:     number of indices to return (default 32 per the paper).
        jitter:     if > 0, each picked index is shifted by `randint(-jitter, +jitter+1)`,
                    then clamped to `[0, num_frames-1]`. Train-time augmentation only;
                    val/test pass `jitter=0` for determinism.
        generator:  torch.Generator for the jitter randomness — required for deterministic
                    resume (D14 / `<determinism_checklist>` precondition 6).

    Returns:
        LongTensor of shape `[target]`. Edge case `num_frames < target` is handled by
        saturating clamp — no exception; Phase 1 confirms no Squat clip is < 49 frames,
        so this is defensive only.
    """
    base = torch.round(torch.linspace(0, max(num_frames - 1, 0), target)).long()
    if jitter > 0:
        shift = torch.randint(-jitter, jitter + 1, (target,), generator=generator)
        base = (base + shift).clamp(0, max(num_frames - 1, 0))
    return base


def count_frames(path: str) -> int:
    """Frame count for `path`, version-robust across the torchvision read_video removal.

    torchvision < 0.26: `read_video_timestamps` (exact decoded frame count).
    torchvision >= 0.26: cv2 `CAP_PROP_FRAME_COUNT` (an estimate for some codecs; the
    cv2 `decode_clip` path tolerates a slightly-high count — out-of-range indices map to
    the nearest decoded frame). Replaces the datasets' direct `read_video_timestamps`
    probe so they decode identically on both torchvision generations.
    """
    if _HAS_TV_READ_VIDEO:
        pts_list, _fps = torchvision.io.read_video_timestamps(path, pts_unit="sec")
        return len(pts_list)
    import cv2  # opencv-python-headless — project dependency

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"count_frames: cv2 could not open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _decode_clip_cv2(path: str, indices: torch.Tensor) -> torch.Tensor:
    """cv2 fallback for `decode_clip` (torchvision >= 0.26 removed read_video).

    Sequential read up to `max(indices)`, BGR->RGB, returns uint8 `[len(indices), 3, H, W]`
    RGB — the SAME contract as the read_video path (so `spatial_train` / `spatial_val` run
    identically). Mirrors `backend/services/clip_decode.decode_clip_cv2` (the serving twin),
    duplicated here to keep this training layer free of a services import.
    """
    import cv2
    import numpy as np

    idx_list = [int(i) for i in indices.tolist()]
    if not idx_list:
        raise ValueError("_decode_clip_cv2: empty indices")
    want = set(idx_list)
    max_want = max(idx_list)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"_decode_clip_cv2: cv2 could not open {path}")

    frames_by_idx: dict[int, "np.ndarray"] = {}
    fi = 0
    while fi <= max_want:
        ret, frame = cap.read()  # BGR uint8 [H, W, 3]
        if not ret:
            break
        if fi in want:
            frames_by_idx[fi] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fi += 1
    cap.release()

    if not frames_by_idx:
        raise ValueError(f"_decode_clip_cv2: decoded 0 frames from {path}")

    available = sorted(frames_by_idx)
    picked = [
        frames_by_idx[i] if i in frames_by_idx
        else frames_by_idx[min(available, key=lambda k: abs(k - i))]
        for i in idx_list
    ]
    arr = np.ascontiguousarray(np.stack(picked, axis=0).transpose(0, 3, 1, 2))  # [T,3,H,W] RGB
    return torch.from_numpy(arr)


def decode_clip(path: str, indices: torch.Tensor) -> torch.Tensor:
    """Decode a window of frames from `path` at the positions in `indices`.

    Window-bounded decode (F11 Option A, RESEARCH §5): rather than decode the entire
    clip and fancy-index — which risks OOM on long Squat clips (up to 404 frames at
    480×600×3 uint8 ≈ 350 MB raw) — we (1) cheaply probe `read_video_timestamps` for
    fps + total length, (2) compute `start_pts` and `end_pts` from `indices.min/max`,
    (3) decode only that window, (4) re-base `indices` to the decoded window and
    fancy-index. Memory usage is bounded by the indices span, not the clip length.

    Phase 3+ swap target: `torchcodec.decoders.VideoDecoder` (when `read_video` is
    removed in torchvision ≥ 0.24, or when profiling demands).

    Args:
        path:    absolute path to the .mp4 (typically `/content/squat_videos/{clip_id}.mp4`).
        indices: LongTensor of frame indices to extract (shape `[target]`, from
                 `uniform_sample_indices`). Values must be in `[0, num_frames-1]`.

    Returns:
        uint8 tensor of shape `[len(indices), 3, H, W]`. Discards audio + info per
        RESEARCH §5 footgun.

    Raises:
        AssertionError: returned shape doesn't match expected `[len(indices), 3, H, W]`.
    """
    # torchvision >= 0.26 removed read_video — route to the cv2 backend (same output contract).
    if not _HAS_TV_READ_VIDEO:
        return _decode_clip_cv2(path, indices)

    # F11 (RESEARCH §5): cheap probe for fps + frame count — no frames decoded.
    pts_list, video_fps = torchvision.io.read_video_timestamps(path, pts_unit="sec")
    fps = float(video_fps) if video_fps else 30.0  # Phase 1: Squat is uniformly 30fps; defensive default.

    idx_min = int(indices.min().item())
    idx_max = int(indices.max().item())

    # F11 (RESEARCH §5): window-bounded decode — `+1/fps` on end gives the decoder
    # one extra frame of slack on the right edge to handle floating-point boundary.
    start_pts = idx_min / fps
    end_pts = (idx_max + 1) / fps

    # B-fix: `output_format="TCHW"` is required — default is "THWC" which would force
    # an extra permute downstream. `pts_unit="sec"` matches the timestamps probe.
    # Suppress the chatty UserWarning about packed B-frames — known torchvision behavior,
    # not actionable from here.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
        video, _audio, _info = torchvision.io.read_video(
            path,
            start_pts=start_pts,
            end_pts=end_pts,
            output_format="TCHW",
            pts_unit="sec",
        )

    # F11 (RESEARCH §5): re-base indices to the decoded window.
    # decoded_start_frame is the absolute frame index where the decoder started.
    decoded_start_frame = round(start_pts * fps)
    # clamp guards against fp edge cases where the actual decoded window is 1 frame off.
    rebased = (indices - decoded_start_frame).clamp(0, max(video.shape[0] - 1, 0))

    out = video[rebased]

    # Hard shape guard — wrong shape here would silently corrupt Task 6's batches.
    assert out.shape[0] == indices.shape[0], (
        f"decode_clip: expected {indices.shape[0]} frames, got {out.shape[0]}; "
        f"path={path}, indices={indices.tolist()[:5]}..., video.shape={tuple(video.shape)}"
    )
    assert out.shape[1] == 3, f"decode_clip: expected 3 channels, got {out.shape[1]}"
    return out


def _resize_short_side(
    clip_tchw: torch.Tensor,
    resize_short: int,
) -> torch.Tensor:
    """Aspect-preserving resize so `min(H, W) == resize_short`. Operates on `[T, C, H, W]`."""
    _, _, h, w = clip_tchw.shape
    scale = resize_short / min(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    # antialias=True matches torchvision's recommendation for downscaling 8-bit images.
    return TF.resize(clip_tchw, [new_h, new_w], antialias=True)


def _normalize_kinetics(clip_tchw_float: torch.Tensor) -> torch.Tensor:
    """In-place-safe Kinetics-400 normalization on `[T, C, H, W]` float32 in `[0, 1]`."""
    return (clip_tchw_float - _MEAN_T.to(clip_tchw_float.device)) / _STD_T.to(clip_tchw_float.device)


def spatial_train(
    clip_tchw: torch.Tensor,
    *,
    crop_size: int = 112,
    resize_short: int = 128,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Train-time spatial pipeline: aspect-preserving resize → random crop → /255 → Kinetics norm → permute.

    Args:
        clip_tchw:    uint8 tensor `[T, 3, H, W]` from `decode_clip`.
        crop_size:    output spatial size (default 112; D1 — R(2+1)D-18 Kinetics-V1 native).
        resize_short: short-side target after aspect-preserving resize (default 128 —
                      12.5% margin above crop_size so random crops have room).
        generator:    torch.Generator for the random crop offset (determinism precondition 7).

    Returns:
        float32 tensor `[3, T, crop_size, crop_size]` — ready for R(2+1)D-18 (C-T-H-W).
    """
    resized = _resize_short_side(clip_tchw, resize_short)
    _, _, h, w = resized.shape

    # Random crop offsets — generator-driven for resume determinism.
    max_top = h - crop_size
    max_left = w - crop_size
    if max_top < 0 or max_left < 0:
        raise ValueError(
            f"spatial_train: resized clip too small for crop "
            f"({h}x{w} < {crop_size}x{crop_size}); raise resize_short or lower crop_size"
        )
    top = int(torch.randint(0, max_top + 1, (1,), generator=generator).item()) if max_top > 0 else 0
    left = int(torch.randint(0, max_left + 1, (1,), generator=generator).item()) if max_left > 0 else 0
    cropped = resized[:, :, top : top + crop_size, left : left + crop_size]

    floated = cropped.to(torch.float32) / 255.0
    normed = _normalize_kinetics(floated)

    # Permute (T, C, H, W) -> (C, T, H, W) for R(2+1)D-18.
    return normed.permute(1, 0, 2, 3).contiguous()


def spatial_val(
    clip_tchw: torch.Tensor,
    *,
    crop_size: int = 112,
    resize_short: int = 128,
) -> torch.Tensor:
    """Val/test spatial pipeline: aspect-preserving resize → center crop → /255 → Kinetics norm → permute.

    Deterministic — no generator parameter. Used by val/test loaders.
    """
    resized = _resize_short_side(clip_tchw, resize_short)
    _, _, h, w = resized.shape

    if h < crop_size or w < crop_size:
        raise ValueError(
            f"spatial_val: resized clip too small for center crop "
            f"({h}x{w} < {crop_size}x{crop_size}); raise resize_short or lower crop_size"
        )
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    cropped = resized[:, :, top : top + crop_size, left : left + crop_size]

    floated = cropped.to(torch.float32) / 255.0
    normed = _normalize_kinetics(floated)

    return normed.permute(1, 0, 2, 3).contiguous()


def decode_clip_cached(
    cache_dir: str | None,
    cache_key: str,
    path: str,
    indices: torch.Tensor,
    *,
    resize_short: int = 128,
) -> torch.Tensor:
    """decode_clip with a disk memo so repeated epochs don't re-decode the same frames.

    cache_dir None -> plain decode_clip (native res). cache_dir set -> the frames are
    decoded once, short-side-resized to `resize_short` (the pre-crop size spatial_* uses
    anyway, so the downstream random/center crop is preserved) and stored uint8 at
    {cache_dir}/{cache_key}.pt; later calls load it. Atomic write (tmp + os.replace) so a
    killed worker can't leave a half-written file; a corrupt/short read falls back to decode.
    """
    if not cache_dir:
        return decode_clip(path, indices)
    cpath = os.path.join(cache_dir, cache_key + ".pt")
    if os.path.exists(cpath):
        try:
            cached = torch.load(cpath)
            if isinstance(cached, torch.Tensor) and cached.shape[0] == int(indices.shape[0]):
                return cached
        except Exception:
            pass
    frames = _resize_short_side(decode_clip(path, indices), resize_short)
    if frames.dtype != torch.uint8:
        frames = frames.round().clamp(0, 255).to(torch.uint8)
    os.makedirs(cache_dir, exist_ok=True)
    tmp = f"{cpath}.tmp{os.getpid()}"
    torch.save(frames, tmp)
    os.replace(tmp, cpath)
    return frames


if __name__ == "__main__":
    # Standalone smoke — synthetic tensor exercise (no real video needed).
    # The F11 acceptance gate (on a ≥400-frame .mp4 with peak memory < 200 MB) runs in
    # the notebook via Step 2, where real /content/squat_videos/*.mp4 are available.
    fake = torch.randint(0, 255, (60, 3, 480, 600), dtype=torch.uint8)

    g = torch.Generator().manual_seed(42)
    idx0 = uniform_sample_indices(60, 32, jitter=0)
    idx2 = uniform_sample_indices(60, 32, jitter=2, generator=g)
    assert idx0.shape == (32,) and idx0.dtype == torch.long
    assert (idx2 - idx0).abs().max().item() <= 2, "jitter exceeded ±2"
    assert idx2.min().item() >= 0 and idx2.max().item() <= 59, "jitter clamp failed"

    train_out = spatial_train(fake, generator=torch.Generator().manual_seed(0))
    val_out = spatial_val(fake)
    assert train_out.shape == (3, 60, 112, 112) and train_out.dtype == torch.float32
    assert val_out.shape == (3, 60, 112, 112) and val_out.dtype == torch.float32

    # Normalization sanity: post-Kinetics-norm values should center near 0.
    assert -3.0 < val_out.mean().item() < 3.0, "normalization mean drift"

    print("transforms.py smoke: ok")
    print(f"  uniform_sample_indices(60, 32, jitter=0)[:8] = {idx0[:8].tolist()}")
    print(f"  uniform_sample_indices(60, 32, jitter=2)[:8] = {idx2[:8].tolist()}")
    print(f"  spatial_train  shape = {tuple(train_out.shape)}  mean={train_out.mean().item():+.3f}")
    print(f"  spatial_val    shape = {tuple(val_out.shape)}    mean={val_out.mean().item():+.3f}")
