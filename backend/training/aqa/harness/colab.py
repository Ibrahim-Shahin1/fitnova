"""Resumable Colab harness primitives — Drive mount, idempotent zip-stage, RNG capture/restore, atomic checkpoint save/load/prune.

Implementation lands across Tasks 7, 9, 10 (Phase 2):

- Task 7 (slice 1, THIS COMMIT): `mount_drive`, `stage_squat_videos` — Drive FUSE mount
  and per-session copy of `Squat/Labeled_Dataset/videos.zip` → `/content/squat_videos/`.
  Idempotent: re-running a cell does not re-extract.
- Task 9 (slice 2): `capture_rng_state`, `restore_rng_state` — full 4-RNG capture (Python
  `random`, NumPy, torch CPU, torch CUDA). Sets `cudnn.deterministic=True`,
  `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True)`. Determinism
  preconditions in PLAN.md `<determinism_checklist>` apply.
- Task 10 (slice 3): `hash_config`, `atomic_save_checkpoint`, `load_latest_checkpoint`,
  `prune_checkpoints`, `CheckpointConfigMismatchError`. Atomic write contract (D11):
  `torch.save` to tmp → reload-verify → `os.replace` → write `latest.txt` LAST. **Drive
  FUSE rename is NOT atomic** (RESEARCH §6) — the contract defends against partial
  checkpoints. The `latest.txt`-written-last ordering means a crash mid-write leaves
  the prior good file pointer untouched.

See: `.planning/phases/02-squat-data-pipeline-colab-harness/02-01-PLAN.md` — Tasks 7/9/10 / D11 / D13 / interfaces block / risk register R3.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
import zipfile
from pathlib import Path

logger = logging.getLogger("aqa.phase02")

# Phase 1 verified — videos.zip ships 1739 mp4s for the Squat labeled set (116 extra
# beyond the 1623 official-split ids, intentionally excluded by the split JSONs).
_SQUAT_VIDEOS_EXPECT_COUNT = 1739
_DRIVE_MOUNT_POINT = "/content/drive"
_DRIVE_MYDRIVE = "/content/drive/MyDrive"


def mount_drive() -> str:
    """Mount Google Drive at `/content/drive`. Idempotent — `drive.mount` itself no-ops on a second call.

    Returns:
        Absolute path to MyDrive: `"/content/drive/MyDrive"`.

    Raises:
        RuntimeError: not running in Colab (google.colab unavailable), or the post-mount
                      MyDrive directory doesn't exist.
    """
    if os.path.isdir(_DRIVE_MYDRIVE):
        logger.info("Drive already mounted at %s (no-op)", _DRIVE_MOUNT_POINT)
        return _DRIVE_MYDRIVE

    try:
        from google.colab import drive  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "mount_drive() requires google.colab — not available outside Colab. "
            "For local testing, mount the dataset at /content/drive/MyDrive manually "
            "or override the drive_root argument to stage_squat_videos()."
        ) from exc

    drive.mount(_DRIVE_MOUNT_POINT)
    if not os.path.isdir(_DRIVE_MYDRIVE):
        raise RuntimeError(f"Drive mount completed but {_DRIVE_MYDRIVE} is missing")
    return _DRIVE_MYDRIVE


def stage_squat_videos(
    drive_root: str,
    *,
    local_root: str = "/content/squat_videos",
    expect_count: int = _SQUAT_VIDEOS_EXPECT_COUNT,
) -> str:
    """Copy Squat `videos.zip` from Drive → `/content/`, extract, verify count. Idempotent (D15).

    Drive FUSE per-frame reads are too slow for training (Phase 1 measurement: ~2.5s per
    archive open from Drive vs. effectively zero from local disk). Per-session staging
    to `/content/` local disk is the build constraint for any video iteration.

    Cache hit semantics: if `local_root` already contains exactly `expect_count` `.mp4`
    files (any nesting), the function is a no-op and returns immediately. Otherwise it
    copies the zip to `/content/squat_videos.zip`, extracts to `local_root`, verifies
    the post-extraction count, then deletes the local zip to free disk (Colab disk is
    tight — the extracted tree is what training reads, not the zip).

    Args:
        drive_root:   path to Drive MyDrive (typically `"/content/drive/MyDrive"`).
        local_root:   destination for the extracted `.mp4` files.
        expect_count: Phase 1 verified count (1739 for Squat labeled).

    Returns:
        Absolute path to `local_root` (the directory containing the extracted .mp4s).

    Raises:
        FileNotFoundError: source zip missing from Drive.
        RuntimeError:      post-extraction mp4 count doesn't match `expect_count`.
    """
    local_root_p = Path(local_root)

    # D15 idempotence — cache hit short-circuits before any I/O.
    if local_root_p.is_dir():
        existing = list(local_root_p.rglob("*.mp4"))
        if len(existing) == expect_count:
            logger.info(
                "stage cache hit: %s already has %d .mp4 files; skipping copy + unzip",
                local_root, expect_count,
            )
            return str(local_root_p)
        logger.info(
            "stage cache miss: %s has %d mp4s (expected %d); re-staging",
            local_root, len(existing), expect_count,
        )

    src_zip = (
        Path(drive_root)
        / "Fitness-AQA_dataset_release/Squat/Labeled_Dataset/videos.zip"
    )
    if not src_zip.is_file():
        raise FileNotFoundError(f"Source zip not found on Drive: {src_zip}")

    local_zip = Path("/content/squat_videos.zip")

    t0 = time.perf_counter()
    shutil.copy(str(src_zip), str(local_zip))
    t_copy = time.perf_counter() - t0
    size_mb = src_zip.stat().st_size / 1e6
    logger.info("copy %s -> %s (%.1f MB) in %.2fs", src_zip.name, local_zip, size_mb, t_copy)

    local_root_p.mkdir(parents=True, exist_ok=True)
    t1 = time.perf_counter()
    with zipfile.ZipFile(str(local_zip), "r") as zf:
        zf.extractall(str(local_root_p))
    t_unzip = time.perf_counter() - t1
    logger.info("unzip %s -> %s in %.2fs", local_zip.name, local_root, t_unzip)

    mp4s = list(local_root_p.rglob("*.mp4"))
    if len(mp4s) != expect_count:
        raise RuntimeError(
            f"stage_squat_videos: expected {expect_count} mp4s after unzip, got {len(mp4s)} "
            f"(check Drive zip integrity: {src_zip})"
        )

    # Reclaim ~1+ GB by removing the local zip — extraction already succeeded.
    try:
        local_zip.unlink()
        logger.info("removed local zip %s", local_zip)
    except OSError as exc:
        logger.warning("could not remove local zip %s: %s", local_zip, exc)

    logger.info(
        "stage_squat_videos: %d mp4s ready at %s (copy %.2fs + unzip %.2fs, %.1f MB source)",
        expect_count, local_root, t_copy, t_unzip, size_mb,
    )
    return str(local_root_p)


# ──────────────────────────────────────────────────────────────────────────────
# Task 9 placeholders — implementation lands with RNG capture/restore + cudnn.
# ──────────────────────────────────────────────────────────────────────────────


def capture_rng_state() -> dict:
    raise NotImplementedError("capture_rng_state lands in Task 9 (colab.py slice 2)")


def restore_rng_state(state: dict) -> None:
    raise NotImplementedError("restore_rng_state lands in Task 9 (colab.py slice 2)")


# ──────────────────────────────────────────────────────────────────────────────
# Task 10 placeholders — implementation lands with checkpoint primitives.
# ──────────────────────────────────────────────────────────────────────────────


class CheckpointConfigMismatchError(RuntimeError):
    """Raised when load_latest_checkpoint detects a config hash drift (Task 10)."""


def hash_config(config: dict) -> str:
    raise NotImplementedError("hash_config lands in Task 10 (colab.py slice 3)")


def atomic_save_checkpoint(payload: dict, target_path: str) -> None:
    raise NotImplementedError("atomic_save_checkpoint lands in Task 10 (colab.py slice 3)")


def load_latest_checkpoint(
    run_dir: str,
    *,
    expected_config_hash: str,
    map_location: str = "cpu",
) -> dict | None:
    raise NotImplementedError("load_latest_checkpoint lands in Task 10 (colab.py slice 3)")


def prune_checkpoints(
    run_dir: str,
    *,
    keep_last: int = 3,
    keep_best: bool = True,
) -> None:
    raise NotImplementedError("prune_checkpoints lands in Task 10 (colab.py slice 3)")
