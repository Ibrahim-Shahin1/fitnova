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
import random
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("aqa.phase02")

# F8 audit: by the time this module loads, _envinit.py has already set
# CUBLAS_WORKSPACE_CONFIG. So `torch.use_deterministic_algorithms(True)` will not raise
# because of missing workspace config. `warn_only=True` keeps any genuinely non-
# deterministic op from crashing the run — they'll log a warning instead.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

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
    """Per-member-resumable extract with **flat mp4 layout**. Returns wall time in seconds.

    Strips the zip's internal directory prefix — every `.mp4` member lands at
    `dest / basename(member)` directly. The Fitness-AQA `videos.zip` is structured as
    `videos/12345.mp4`; `splits.py` (D7) builds `video_path = videos_root + "/" + clip_id +
    ".mp4"` and expects mp4s at the **top** of videos_root. Extracting flat avoids a
    separate flatten step that would temporarily double disk usage.

    Resume semantics: for each `.mp4` member, if `dest / basename(member)` already exists
    at the correct uncompressed size, skip. Otherwise stream the member content via
    `shutil.copyfileobj` to the target. Non-mp4 members and directory entries are logged
    and skipped.

    A kernel-interrupt mid-extract survives because already-finished mp4s stay in place
    and skip-on-next-run. Tqdm postfix shows `extracted=` vs `skipped=` counts.

    Raises:
        RuntimeError: zip contains two .mp4 members with the same basename in different
                      subdirs (would collide on flat layout). Defensive — the Squat
                      videos.zip is single-directory so this shouldn't fire.
    """
    try:
        from tqdm.auto import tqdm  # type: ignore[import-not-found]
    except ImportError:
        tqdm = None

    t0 = time.perf_counter()
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        all_members = zf.infolist()
        mp4_members = [m for m in all_members if not m.is_dir() and m.filename.lower().endswith(".mp4")]

        # Collision check — if two zip members would flatten to the same basename, refuse.
        basenames = [Path(m.filename).name for m in mp4_members]
        if len(set(basenames)) != len(basenames):
            from collections import Counter
            dupes = [n for n, c in Counter(basenames).items() if c > 1]
            raise RuntimeError(
                f"zip {zip_path.name} has duplicate .mp4 basenames across subdirs "
                f"(would collide on flat layout): {dupes[:5]}{'...' if len(dupes) > 5 else ''}"
            )

        non_mp4_files = sum(1 for m in all_members if not m.is_dir() and not m.filename.lower().endswith(".mp4"))
        if non_mp4_files:
            logger.info("zip has %d non-mp4 files (skipping — flat layout is mp4-only)", non_mp4_files)

        dest.mkdir(parents=True, exist_ok=True)

        extracted = skipped = 0

        def _do_one(m: zipfile.ZipInfo) -> None:
            nonlocal extracted, skipped
            target = dest / Path(m.filename).name
            if target.exists() and target.stat().st_size == m.file_size:
                skipped += 1
                return
            with zf.open(m) as src_f, target.open("wb") as dst_f:
                shutil.copyfileobj(src_f, dst_f)
            extracted += 1

        if tqdm is None:
            for m in mp4_members:
                _do_one(m)
            logger.info("unzip resume: extracted=%d skipped=%d (flat layout)", extracted, skipped)
        else:
            with tqdm(
                total=len(mp4_members),
                desc=f"unzip {zip_path.name}",
                unit="file",
                leave=True,
            ) as pbar:
                for m in mp4_members:
                    _do_one(m)
                    pbar.update(1)
                    pbar.set_postfix(extracted=extracted, skipped=skipped)

    return time.perf_counter() - t0


def _migrate_nested_mp4s_to_top(local_root: Path) -> int:
    """Migration helper: move any nested `.mp4` files up to `local_root` itself.

    Returns:
        Number of files actually moved.

    Idempotent: if all mp4s are already at the top, this is a no-op. If a target
    already exists at the top, the nested duplicate is deleted (we know it came from
    the same source zip and is identical content). Empty subdirectories are removed
    after the move loop.

    Used for one-time migration of a session whose `videos.zip` was extracted with the
    old behavior (preserving the `videos/` subdir prefix) so we don't have to re-download
    1+ GB from Drive on the upgrade. Tqdm bar makes the move visible.
    """
    try:
        from tqdm.auto import tqdm  # type: ignore[import-not-found]
    except ImportError:
        tqdm = None

    nested = [p for p in local_root.rglob("*.mp4") if p.parent != local_root]
    if not nested:
        return 0

    moved = 0
    deduped = 0

    def _do_one(src_p: Path) -> None:
        nonlocal moved, deduped
        target = local_root / src_p.name
        if target.exists():
            if target.stat().st_size == src_p.stat().st_size:
                # Same content — drop the nested duplicate.
                src_p.unlink()
                deduped += 1
            else:
                logger.warning(
                    "migrate: target %s already exists with DIFFERENT size; not overwriting",
                    target,
                )
            return
        src_p.rename(target)
        moved += 1

    if tqdm is None:
        for p in nested:
            _do_one(p)
    else:
        with tqdm(total=len(nested), desc="flatten nested mp4s", unit="file", leave=True) as pbar:
            for p in nested:
                _do_one(p)
                pbar.update(1)
                pbar.set_postfix(moved=moved, deduped=deduped)

    # Remove now-empty subdirectories under local_root (sorted deepest-first).
    for d in sorted(
        (p for p in local_root.rglob("*") if p.is_dir()),
        key=lambda p: -len(p.parts),
    ):
        try:
            d.rmdir()
        except OSError:
            pass  # not empty — leave it

    logger.info(
        "migrate: moved=%d deduped=%d to top of %s", moved, deduped, local_root,
    )
    return moved + deduped


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

    # D15 idempotence — three checks before falling through to copy + extract:
    #   (a) cache hit (flat layout): mp4s at the top of local_root, count matches.
    #   (b) migration: mp4s nested in a subdir (legacy/buggy state) — flatten them up
    #       to avoid re-downloading 1+ GB just to fix the layout. Re-check (a) after.
    #   (c) cache miss: partial or no extraction — fall through to full copy + extract.
    if local_root_p.is_dir():
        flat_existing = list(local_root_p.glob("*.mp4"))
        if len(flat_existing) == expect_count:
            logger.info(
                "stage cache hit (flat): %d mp4s already at top of %s",
                expect_count, local_root,
            )
            return str(local_root_p)

        nested = [p for p in local_root_p.rglob("*.mp4") if p.parent != local_root_p]
        if nested:
            logger.info(
                "stage migrate: %d mp4s found nested under %s — flattening to top",
                len(nested), local_root,
            )
            _migrate_nested_mp4s_to_top(local_root_p)
            flat_existing = list(local_root_p.glob("*.mp4"))
            if len(flat_existing) == expect_count:
                logger.info(
                    "stage cache hit (post-migration): %d mp4s now flat at top of %s",
                    expect_count, local_root,
                )
                return str(local_root_p)
            logger.info(
                "stage post-migration: %d at top + %d still nested — proceeding to copy + extract",
                len(flat_existing),
                sum(1 for _ in local_root_p.rglob("*.mp4")) - len(flat_existing),
            )
        else:
            logger.info(
                "stage cache miss: %s has %d top-level mp4s (expected %d); proceeding",
                local_root, len(flat_existing), expect_count,
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

    # Flat-layout check — extract should have produced exactly `expect_count` mp4s at
    # the TOP of local_root (the splits.py D7 contract).
    mp4s = list(local_root_p.glob("*.mp4"))
    if len(mp4s) != expect_count:
        nested_count = sum(1 for _ in local_root_p.rglob("*.mp4")) - len(mp4s)
        raise RuntimeError(
            f"stage_squat_videos: expected {expect_count} mp4s at top of {local_root}, "
            f"got {len(mp4s)} flat + {nested_count} nested. "
            f"Check Drive zip integrity: {src_zip}"
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
# Task 9 — RNG capture/restore. Required for Task 14's bitwise-resume assertion
# (fresh-2-epoch vs. resumed-from-epoch-0-into-epoch-1 must produce byte-identical
# epoch-1 loss trajectories). All 4 RNG sources captured so seed-restoration
# is exhaustive: any one missing would let post-resume sampling drift.
# ──────────────────────────────────────────────────────────────────────────────


def capture_rng_state() -> dict:
    """Snapshot all 4 RNG sources used by the pipeline. Returns dict with keys:

    - `python`         : `random.getstate()` — Python's `random` module
    - `numpy`          : `np.random.get_state()` — NumPy's global RNG
    - `torch_cpu`      : `torch.get_rng_state()` — torch CPU RNG (ByteTensor)
    - `torch_cuda_all` : `torch.cuda.get_rng_state_all()` if CUDA available, else `[]`

    The 4-key contract is asserted by Task 9's verification cell and D14's
    `<determinism_checklist>` precondition 2.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    """Restore all 4 RNG sources from a `capture_rng_state` snapshot.

    Order: **CUDA first, then CPU**. The plan calls this "to avoid drift" — by
    setting CUDA before any CPU op that might internally allocate or seed a CUDA
    tensor, we keep CUDA's seed exactly at the captured value. The CPU/numpy/python
    restores can then happen in any order since they're independent streams.

    CUDA restore is gated on `torch.cuda.is_available()` so a CPU-only host
    (e.g., the local Windows machine) doesn't fail.
    """
    if torch.cuda.is_available() and state.get("torch_cuda_all"):
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])
    torch.set_rng_state(state["torch_cpu"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


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
