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


def _copy_with_resume_and_progress(
    src: Path, dst: Path, *, chunk_size: int = 16 * 1024 * 1024,
) -> float:
    """Byte-resumable chunked copy with tqdm byte-progress. Returns wall time in seconds.

    Resume semantics (per the working agreement — disconnect is the default case):
    - If `dst` exists with size **equal** to `src.size`, no-op (copy already complete).
    - If `dst` exists with size **less than** `src.size`, **resume** from `dst.size`
      using `fin.seek` + `ab` (append) mode on dst. The tqdm bar is initialized at
      `initial=dst_size` so the bar reflects already-transferred bytes.
    - If `dst` exists with size **greater than** `src.size` (defensive — shouldn't
      happen), truncate and restart.
    - If `dst` doesn't exist, start fresh from byte 0.

    A kernel-interrupt mid-copy leaves `dst` at the partial size; re-running the cell
    picks up exactly where it stopped. Full runtime restart (Colab wipes `/content/`)
    is the only failure mode that requires copy-from-zero, and that's outside our
    contract.
    """
    try:
        from tqdm.auto import tqdm  # type: ignore[import-not-found]
    except ImportError:
        tqdm = None  # graceful fallback — log only

    src_size = src.stat().st_size

    if dst.exists():
        dst_size = dst.stat().st_size
        if dst_size == src_size:
            logger.info(
                "copy resume: %s already complete (%.1f MB) — no-op",
                dst.name, src_size / 1e6,
            )
            return 0.0
        if dst_size > src_size:
            logger.warning(
                "copy resume: %s is larger than src (%d > %d) — truncating and restarting",
                dst, dst_size, src_size,
            )
            dst.unlink()
            dst_size = 0
        else:
            logger.info(
                "copy resume: %s partial (%.1f / %.1f MB) — continuing from byte %d",
                dst.name, dst_size / 1e6, src_size / 1e6, dst_size,
            )
    else:
        dst_size = 0

    t0 = time.perf_counter()
    open_mode = "ab" if dst_size > 0 else "wb"
    with src.open("rb") as fin:
        fin.seek(dst_size)
        with dst.open(open_mode) as fout:
            if tqdm is None:
                while True:
                    buf = fin.read(chunk_size)
                    if not buf:
                        break
                    fout.write(buf)
            else:
                with tqdm(
                    total=src_size, initial=dst_size,
                    unit="B", unit_scale=True, unit_divisor=1024,
                    desc=f"copy {src.name}", leave=True,
                ) as pbar:
                    while True:
                        buf = fin.read(chunk_size)
                        if not buf:
                            break
                        fout.write(buf)
                        pbar.update(len(buf))
    return time.perf_counter() - t0


def _extract_with_resume_and_progress(zip_path: Path, dest: Path) -> float:
    """Per-member-resumable `zipfile` extract with tqdm progress. Returns wall time in seconds.

    Resume semantics: iterate `zf.infolist()` rather than `extractall`. For each member,
    if the target file already exists at `dest / m.filename` with size matching
    `m.file_size` (the uncompressed size from the zip header), skip the extract. Files
    that are missing or have wrong size are (re-)extracted; existing-matching files are
    a no-op.

    A kernel-interrupt mid-extract leaves a partial set; re-running the cell skips the
    already-extracted members and continues from where it stopped. The post-extract
    count check in `stage_squat_videos` is the integrity gate.

    Tqdm postfix shows `extracted=` and `skipped=` counters so progress reflects both
    new work and resume reuse.
    """
    try:
        from tqdm.auto import tqdm  # type: ignore[import-not-found]
    except ImportError:
        tqdm = None

    t0 = time.perf_counter()
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        members = zf.infolist()

        if tqdm is None:
            extracted = skipped = 0
            for m in members:
                target = dest / m.filename
                if target.exists() and target.stat().st_size == m.file_size:
                    skipped += 1
                else:
                    zf.extract(m, path=str(dest))
                    extracted += 1
            logger.info("unzip resume: extracted=%d skipped=%d", extracted, skipped)
        else:
            with tqdm(
                total=len(members),
                desc=f"unzip {zip_path.name}",
                unit="file",
                leave=True,
            ) as pbar:
                extracted = skipped = 0
                for m in members:
                    target = dest / m.filename
                    if target.exists() and target.stat().st_size == m.file_size:
                        skipped += 1
                    else:
                        zf.extract(m, path=str(dest))
                        extracted += 1
                    pbar.update(1)
                    pbar.set_postfix(extracted=extracted, skipped=skipped)
    return time.perf_counter() - t0


def stage_squat_videos(
    drive_root: str,
    *,
    local_root: str = "/content/squat_videos",
    expect_count: int = _SQUAT_VIDEOS_EXPECT_COUNT,
) -> str:
    """Copy Squat `videos.zip` from Drive → `/content/`, extract, verify count. Idempotent (D15) with **byte-level resume**.

    Drive FUSE per-frame reads are too slow for training (Phase 1 measurement: ~2.5s per
    archive open from Drive vs. effectively zero from local disk). Per-session staging
    to `/content/` local disk is the build constraint for any video iteration.

    **Three-layer resume design (per the working-agreement disconnect-by-default rule):**

    1. **Cache hit** (top-level fast path): if `local_root` already contains exactly
       `expect_count` `.mp4` files, the function returns in milliseconds — no I/O at all.
    2. **Copy byte-level resume**: `_copy_with_resume_and_progress` checks if
       `/content/squat_videos.zip` already exists; if it's at the correct size the copy
       is a no-op; if partial, the copy resumes from `dst_size` via `seek + ab`. A
       kernel-interrupt mid-copy survives because the partial zip is kept.
    3. **Extract per-member resume**: `_extract_with_resume_and_progress` iterates zip
       members; files already at the target with matching uncompressed size are skipped.
       A kernel-interrupt mid-extract survives because already-extracted files stay put.

    Tqdm progress bars cover both copy and extract so the cell is never silent.

    Failure mode requiring full restart: a true runtime restart (Colab wipes `/content/`)
    — that's outside Colab's contract, not something resume can defend against. In that
    case the next cell run pays the full copy + extract cost, as expected.

    Post-extract, the local zip is deleted to reclaim disk (the extracted tree is what
    training reads, not the zip). If the extract verify fails the zip is NOT deleted, so
    the next attempt benefits from copy cache.

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
    size_mb = src_zip.stat().st_size / 1e6

    t_copy = _copy_with_resume_and_progress(src_zip, local_zip)
    logger.info("copy %s -> %s (%.1f MB) in %.2fs", src_zip.name, local_zip, size_mb, t_copy)

    local_root_p.mkdir(parents=True, exist_ok=True)
    t_unzip = _extract_with_resume_and_progress(local_zip, local_root_p)
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
