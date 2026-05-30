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
import warnings
import zipfile
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("aqa.phase02")

# Suppress the torchvision 0.22+ "video decoding deprecated" UserWarning. We KNOW
# read_video is deprecated; the swap to TorchCodec is documented in D6 and runs when
# Colab's torchvision passes 0.24 (`read_video` removal). Until then the warning is
# noise — it fires once per `read_video_timestamps` + `read_video` call, polluting
# stdout during the tiny training run.
warnings.filterwarnings(
    "ignore",
    message=r".*video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
)

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


_SQUAT_UNLABELED_VIDEOS_EXPECT_COUNT = 4970  # Phase 1 verified count (Squat Unlabeled_Dataset)


def stage_unlabeled_squat_videos(
    drive_root: str,
    *,
    local_videos_root: str = "/content/squat_unlabeled_videos",
    local_traj_root: str = "/content/squat_trajectories",
    expect_count: int = _SQUAT_UNLABELED_VIDEOS_EXPECT_COUNT,
) -> tuple[str, str]:
    """Stage the unlabeled Squat `videos.zip` + `bar_trajectories_raw.zip` from Drive (§10).

    Mirrors `stage_squat_videos`' three-layer resume for the videos (cache hit → byte-resume
    copy → per-member mp4 extract, reusing the existing helpers). The trajectory archive
    holds per-clip JSONs, so it is copied (byte-resume) then extracted with
    `zipfile.extractall` — `_extract_with_resume_and_progress` is mp4-only and would skip
    JSON members. The trajectory zip is small; its exact on-disk layout/format is confirmed
    by Plan 02's gated probe before the SSL dataset's `_load_trajectory` is finalized (§8).

    Args:
        drive_root:        path to Drive MyDrive (typically `"/content/drive/MyDrive"`).
        local_videos_root: destination for the extracted unlabeled `.mp4` files.
        local_traj_root:   destination for the extracted trajectory JSONs.
        expect_count:      Phase 1 verified count (4970 unlabeled Squat clips).

    Returns:
        `(local_videos_root, local_traj_root)` — paths to the staged assets.

    Raises:
        FileNotFoundError: a source zip is missing from Drive.
        RuntimeError:      post-extraction mp4 count doesn't match `expect_count`.
    """
    videos_root_p = Path(local_videos_root)
    traj_root_p = Path(local_traj_root)

    # Cache hit on the expensive asset (the 4,970 mp4s) + a populated trajectory dir.
    if videos_root_p.is_dir():
        flat = list(videos_root_p.glob("*.mp4"))
        if (
            len(flat) == expect_count
            and traj_root_p.is_dir()
            and any(traj_root_p.iterdir())
        ):
            logger.info(
                "stage_unlabeled cache hit: %d mp4s + trajectories already staged", expect_count
            )
            return str(videos_root_p), str(traj_root_p)

    base = Path(drive_root) / "Fitness-AQA_dataset_release/Squat/Unlabeled_Dataset"
    src_videos = base / "videos.zip"
    src_traj = base / "bar_trajectories_raw.zip"
    if not src_videos.is_file():
        raise FileNotFoundError(f"Unlabeled videos zip not found on Drive: {src_videos}")
    if not src_traj.is_file():
        raise FileNotFoundError(f"Unlabeled trajectory zip not found on Drive: {src_traj}")

    # Videos: byte-resume copy → per-member mp4 extract (reuse the existing primitives).
    local_videos_zip = Path("/content/squat_unlabeled_videos.zip")
    vid_mb = src_videos.stat().st_size / 1e6
    t_vcopy = _copy_with_resume_and_progress(src_videos, local_videos_zip)
    logger.info("copy %s (%.1f MB) in %.2fs", src_videos.name, vid_mb, t_vcopy)
    videos_root_p.mkdir(parents=True, exist_ok=True)
    t_vextract = _extract_with_resume_and_progress(local_videos_zip, videos_root_p)
    mp4s = list(videos_root_p.glob("*.mp4"))
    if len(mp4s) != expect_count:
        raise RuntimeError(
            f"stage_unlabeled_squat_videos: expected {expect_count} mp4s at top of "
            f"{local_videos_root}, got {len(mp4s)}. Check Drive zip: {src_videos}"
        )

    # Trajectories: byte-resume copy → generic extractall (JSONs; the mp4-only extractor
    # cannot handle them). Small archive; resume-safe via the copy + idempotent extractall.
    local_traj_zip = Path("/content/squat_trajectories.zip")
    t_tcopy = _copy_with_resume_and_progress(src_traj, local_traj_zip)
    traj_root_p.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(local_traj_zip), "r") as zf:
        zf.extractall(traj_root_p)
    n_traj_files = sum(1 for p in traj_root_p.rglob("*") if p.is_file())
    logger.info(
        "stage_unlabeled_squat_videos: %d mp4s at %s + %d trajectory files at %s "
        "(videos copy %.2fs + extract %.2fs, traj copy %.2fs)",
        expect_count, local_videos_root, n_traj_files, local_traj_root,
        t_vcopy, t_vextract, t_tcopy,
    )

    # Reclaim disk — both extractions succeeded.
    for z in (local_videos_zip, local_traj_zip):
        try:
            z.unlink()
        except OSError as exc:
            logger.warning("could not remove %s: %s", z, exc)

    return str(videos_root_p), str(traj_root_p)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 6 OHP staging (D4 / RESEARCH §5 / Pitfall 4)
# ──────────────────────────────────────────────────────────────────────────────

# Phase 6 D4 verified: labeled zip holds 2,367 mp4s (2,260 official + 107 extra —
# like Squat's 1,739-vs-1,623; the split JSONs are the id source of truth).
_OHP_LABELED_VIDEOS_EXPECT_COUNT   = 2367
# Phase 6 D4 verified: 5,490 unlabeled OHP clips.
_OHP_UNLABELED_VIDEOS_EXPECT_COUNT = 5490


def stage_ohp_videos(
    drive_root: str,
    *,
    local_root: str = "/content/ohp_videos",
    expect_count: int = _OHP_LABELED_VIDEOS_EXPECT_COUNT,
) -> str:
    """Copy OHP labeled videos.zip from Drive → /content/ohp_videos/, extract, verify. Idempotent.

    Source: {drive_root}/Fitness-AQA_dataset_release/OHP/Labeled_Dataset/videos.zip
    (in the -3-001 release folder alongside the OHP labeled data and trajectory zip).

    Mirrors stage_squat_videos exactly with three-layer resume:
      1. Cache hit: if local_root contains exactly expect_count .mp4s, return immediately.
      2. Copy byte-level resume: _copy_with_resume_and_progress.
      3. Extract per-member resume: _extract_with_resume_and_progress.

    Args:
        drive_root:   path to Drive MyDrive (typically "/content/drive/MyDrive").
        local_root:   destination for the extracted .mp4 files (default /content/ohp_videos).
        expect_count: Phase 6 D4 verified count (2367 for OHP labeled zip).

    Returns:
        Absolute path to local_root.

    Raises:
        FileNotFoundError: source zip missing from Drive.
        RuntimeError:      post-extraction mp4 count doesn't match expect_count.
    """
    local_root_p = Path(local_root)

    # Layer 1 — cache hit (flat layout).
    if local_root_p.is_dir():
        flat_existing = list(local_root_p.glob("*.mp4"))
        if len(flat_existing) == expect_count:
            logger.info(
                "stage_ohp cache hit (flat): %d mp4s already at top of %s",
                expect_count, local_root,
            )
            return str(local_root_p)

        nested = [p for p in local_root_p.rglob("*.mp4") if p.parent != local_root_p]
        if nested:
            logger.info(
                "stage_ohp migrate: %d mp4s found nested under %s — flattening to top",
                len(nested), local_root,
            )
            _migrate_nested_mp4s_to_top(local_root_p)
            flat_existing = list(local_root_p.glob("*.mp4"))
            if len(flat_existing) == expect_count:
                logger.info(
                    "stage_ohp cache hit (post-migration): %d mp4s now flat at top of %s",
                    expect_count, local_root,
                )
                return str(local_root_p)
        else:
            logger.info(
                "stage_ohp cache miss: %s has %d top-level mp4s (expected %d); proceeding",
                local_root, len(flat_existing), expect_count,
            )

    src_zip = (
        Path(drive_root)
        / "Fitness-AQA_dataset_release/OHP/Labeled_Dataset/videos.zip"
    )
    if not src_zip.is_file():
        raise FileNotFoundError(f"OHP labeled videos zip not found on Drive: {src_zip}")

    local_zip = Path("/content/ohp_videos.zip")
    size_mb = src_zip.stat().st_size / 1e6

    t_copy = _copy_with_resume_and_progress(src_zip, local_zip)
    logger.info("copy %s -> %s (%.1f MB) in %.2fs", src_zip.name, local_zip, size_mb, t_copy)

    local_root_p.mkdir(parents=True, exist_ok=True)
    t_unzip = _extract_with_resume_and_progress(local_zip, local_root_p)
    logger.info("unzip %s -> %s in %.2fs", local_zip.name, local_root, t_unzip)

    # Flat-layout check.
    mp4s = list(local_root_p.glob("*.mp4"))
    if len(mp4s) != expect_count:
        nested_count = sum(1 for _ in local_root_p.rglob("*.mp4")) - len(mp4s)
        raise RuntimeError(
            f"stage_ohp_videos: expected {expect_count} mp4s at top of {local_root}, "
            f"got {len(mp4s)} flat + {nested_count} nested. "
            f"Check Drive zip integrity: {src_zip}"
        )

    # Reclaim disk.
    try:
        local_zip.unlink()
        logger.info("removed local zip %s", local_zip)
    except OSError as exc:
        logger.warning("could not remove local zip %s: %s", local_zip, exc)

    logger.info(
        "stage_ohp_videos: %d mp4s ready at %s (copy %.2fs + unzip %.2fs, %.1f MB source)",
        expect_count, local_root, t_copy, t_unzip, size_mb,
    )
    return str(local_root_p)


def stage_unlabeled_ohp_videos(
    drive_root_3001: str,
    drive_root_3002: str,
    *,
    local_videos_root: str = "/content/ohp_unlabeled_videos",
    local_traj_root: str = "/content/ohp_trajectories",
    expect_count: int = _OHP_UNLABELED_VIDEOS_EXPECT_COUNT,
) -> tuple[str, str]:
    """Stage OHP unlabeled videos.zip (from -3-002) + bar_trajectories_raw.zip (from -3-001).

    OHP archive is SPLIT across two release folders (D4, Pitfall 4 / RESEARCH §5):
      - videos.zip:               {drive_root_3002}/.../OHP/Unlabeled_Dataset/videos.zip
      - bar_trajectories_raw.zip: {drive_root_3001}/.../OHP/Unlabeled_Dataset/bar_trajectories_raw.zip

    Two separate drive_root parameters handle this split (OHP differs from Squat which has
    all its data in one folder). Trajectory archive uses zipfile.extractall (JSON members,
    flat layout: bar_trajectories_raw/{clip_id}.json) — NOT _extract_with_resume_and_progress
    (mp4-only). D7 landmine #7: always use zipfile.extractall for JSON trajectory zips.

    Mirrors stage_unlabeled_squat_videos body; only the source paths differ.

    Args:
        drive_root_3001: path to Drive root of the -3-001 release folder (has the trajectory zip).
        drive_root_3002: path to Drive root of the -3-002 release folder (has unlabeled videos).
        local_videos_root: destination for extracted unlabeled .mp4 files.
        local_traj_root:   destination for extracted trajectory JSONs.
        expect_count:      Phase 6 D4 verified count (5490 unlabeled OHP clips).

    Returns:
        ``(local_videos_root, local_traj_root)`` — paths to the staged assets.

    Raises:
        FileNotFoundError: a source zip is missing from Drive.
        RuntimeError:      post-extraction mp4 count doesn't match expect_count.
    """
    videos_root_p = Path(local_videos_root)
    traj_root_p   = Path(local_traj_root)

    # Cache hit on the expensive asset (5,490 mp4s) + a populated trajectory dir.
    if videos_root_p.is_dir():
        flat = list(videos_root_p.glob("*.mp4"))
        if (
            len(flat) == expect_count
            and traj_root_p.is_dir()
            and any(traj_root_p.iterdir())
        ):
            logger.info(
                "stage_unlabeled_ohp cache hit: %d mp4s + trajectories already staged",
                expect_count,
            )
            return str(videos_root_p), str(traj_root_p)

    # Two Drive roots — OHP archive is split across -3-001 (traj) and -3-002 (videos).
    base_3001 = Path(drive_root_3001) / "Fitness-AQA_dataset_release/OHP/Unlabeled_Dataset"
    base_3002 = Path(drive_root_3002) / "Fitness-AQA_dataset_release/OHP/Unlabeled_Dataset"
    src_traj   = base_3001 / "bar_trajectories_raw.zip"
    src_videos = base_3002 / "videos.zip"

    if not src_videos.is_file():
        raise FileNotFoundError(f"OHP unlabeled videos zip not found on Drive: {src_videos}")
    if not src_traj.is_file():
        raise FileNotFoundError(f"OHP trajectory zip not found on Drive: {src_traj}")

    # Videos: byte-resume copy → per-member mp4 extract (reuse the existing primitives).
    local_videos_zip = Path("/content/ohp_unlabeled_videos.zip")
    vid_mb = src_videos.stat().st_size / 1e6
    t_vcopy = _copy_with_resume_and_progress(src_videos, local_videos_zip)
    logger.info("copy %s (%.1f MB) in %.2fs", src_videos.name, vid_mb, t_vcopy)
    videos_root_p.mkdir(parents=True, exist_ok=True)
    t_vextract = _extract_with_resume_and_progress(local_videos_zip, videos_root_p)
    mp4s = list(videos_root_p.glob("*.mp4"))
    if len(mp4s) != expect_count:
        raise RuntimeError(
            f"stage_unlabeled_ohp_videos: expected {expect_count} mp4s at top of "
            f"{local_videos_root}, got {len(mp4s)}. Check Drive zip: {src_videos}"
        )

    # Trajectories: byte-resume copy → zipfile.extractall (JSON members, flat layout).
    # D7 landmine #7: NOT _extract_with_resume_and_progress (mp4-only).
    # OHP traj zip extracts to bar_trajectories_raw/{clip_id}.json (flat one-level).
    local_traj_zip = Path("/content/ohp_trajectories.zip")
    t_tcopy = _copy_with_resume_and_progress(src_traj, local_traj_zip)
    traj_root_p.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(local_traj_zip), "r") as zf:
        zf.extractall(traj_root_p)
    n_traj_files = sum(1 for p in traj_root_p.rglob("*") if p.is_file())
    logger.info(
        "stage_unlabeled_ohp_videos: %d mp4s at %s + %d trajectory files at %s "
        "(videos copy %.2fs + extract %.2fs, traj copy %.2fs)",
        expect_count, local_videos_root, n_traj_files, local_traj_root,
        t_vcopy, t_vextract, t_tcopy,
    )

    # Reclaim disk — both extractions succeeded.
    for z in (local_videos_zip, local_traj_zip):
        try:
            z.unlink()
        except OSError as exc:
            logger.warning("could not remove %s: %s", z, exc)

    return str(videos_root_p), str(traj_root_p)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 7 — Shallow-Squat image staging + CVCSPC SSL frame extraction (additive)
# ──────────────────────────────────────────────────────────────────────────────

_SHALLOW_SQUAT_IMAGES_EXPECT_COUNT = 3738  # images.zip ships 3,738 crop JPEGs


def stage_shallow_squat_images(
    drive_root_3001: str,
    *,
    local_root: str = "/content/squat_shallow_images",
    expect_count: int = _SHALLOW_SQUAT_IMAGES_EXPECT_COUNT,
) -> str:
    """Stage the Shallow-Squat images.zip + labels + splits from Drive (-3-001) to /content/.

    Source: {drive_root_3001}/Fitness-AQA_dataset_release/Squat/Labeled_Dataset/
            Shallow_Squat_Error_Dataset/{images.zip, labels_shallow_depth.json, splits/}

    Extracts to local_root/crops_unaligned/{id}.jpg and copies the labels + splits alongside,
    so the training/EDA notebooks read images_root=local_root/crops_unaligned,
    labels_path=local_root/labels_shallow_depth.json, splits_root=local_root/splits.

    Returns the local_root path.
    """
    local_root_p = Path(local_root)
    crops_dir = local_root_p / "crops_unaligned"
    src_dir = (
        Path(drive_root_3001)
        / "Fitness-AQA_dataset_release/Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset"
    )

    def _copy_aux() -> None:
        local_root_p.mkdir(parents=True, exist_ok=True)
        labels_src = src_dir / "labels_shallow_depth.json"
        if labels_src.is_file():
            shutil.copy2(labels_src, local_root_p / "labels_shallow_depth.json")
        splits_src = src_dir / "splits"
        if splits_src.is_dir():
            (local_root_p / "splits").mkdir(parents=True, exist_ok=True)
            for j in splits_src.glob("*.json"):
                shutil.copy2(j, local_root_p / "splits" / j.name)

    # Cache hit — crops already extracted at the expected count.
    if crops_dir.is_dir():
        n_jpg = sum(1 for _ in crops_dir.rglob("*.jpg"))
        if n_jpg == expect_count:
            _copy_aux()
            logger.info("stage_shallow_squat cache hit: %d jpgs at %s", expect_count, crops_dir)
            return str(local_root_p)

    src_zip = src_dir / "images.zip"
    if not src_zip.is_file():
        raise FileNotFoundError(f"Shallow-Squat images.zip not found on Drive: {src_zip}")

    local_root_p.mkdir(parents=True, exist_ok=True)
    local_zip = Path("/content/squat_shallow_images.zip")
    t_copy = _copy_with_resume_and_progress(src_zip, local_zip)
    # JPEG members — zipfile.extractall (the mp4-only extractor would skip every JPEG).
    with zipfile.ZipFile(str(local_zip), "r") as zf:
        zf.extractall(local_root_p)
    _copy_aux()

    n_jpg = sum(1 for _ in crops_dir.rglob("*.jpg"))
    if n_jpg != expect_count:
        raise RuntimeError(
            f"stage_shallow_squat_images: expected {expect_count} jpgs under {crops_dir}, "
            f"got {n_jpg}. Check Drive zip: {src_zip}"
        )

    try:
        local_zip.unlink()
    except OSError as exc:
        logger.warning("could not remove %s: %s", local_zip, exc)

    logger.info("stage_shallow_squat_images: %d jpgs at %s (copy %.2fs)", expect_count, crops_dir, t_copy)
    return str(local_root_p)


def extract_frames_for_ssl(
    videos_root: str,
    frames_root: str,
    *,
    skip_existing: bool = True,
) -> int:
    """Decode each unlabeled Squat mp4 into per-clip JPEG frame dirs for the CVCSPC dataloader.

    Output layout: {frames_root}/{video_id}/frame_{i:06d}.jpg (the per-video subdir the SSL
    dataset indexes). Disconnect-safe: a clip whose dir already exists and is non-empty is
    skipped. Returns the total number of frames written this call.
    """
    import cv2

    try:
        from tqdm.auto import tqdm  # type: ignore[import-not-found]
    except ImportError:
        tqdm = None

    videos_root_p = Path(videos_root)
    frames_root_p = Path(frames_root)
    frames_root_p.mkdir(parents=True, exist_ok=True)

    mp4s = sorted(videos_root_p.glob("*.mp4"))
    total_frames = 0
    skipped = 0
    iterator = mp4s if tqdm is None else tqdm(mp4s, desc="extract frames", unit="clip")
    for mp4 in iterator:
        out_dir = frames_root_p / mp4.stem
        if skip_existing and out_dir.exists() and any(out_dir.iterdir()):
            skipped += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(mp4))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(str(out_dir / f"frame_{i:06d}.jpg"), frame)
            i += 1
        cap.release()
        total_frames += i
        if tqdm is not None:
            iterator.set_postfix(extracted=total_frames, skipped=skipped)

    logger.info("extract_frames_for_ssl: %d frames written, %d clips skipped", total_frames, skipped)
    return total_frames


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
# Task 10 — atomic checkpoint primitives. Drive FUSE rename is NOT atomic
# (RESEARCH §6); these primitives defend against partial writes with:
#   tmp write → torch.load round-trip verify → os.replace → latest.txt LAST.
# A crash at any step leaves the prior good `latest.txt` pointer untouched.
# ──────────────────────────────────────────────────────────────────────────────

import hashlib
import json
import re


class CheckpointConfigMismatchError(RuntimeError):
    """Raised by `load_latest_checkpoint` when a checkpoint's `config_hash` doesn't match the caller's `expected_config_hash`.

    Carries both hashes + the `config_repr` (the JSON-safe mirror of the config) in
    the message so the user can see *what* drifted at-a-glance. Recovery: start a
    new run name, or reconcile the config back to the checkpointed values.
    """


def hash_config(config: dict) -> str:
    """SHA-256 of `json.dumps(config, sort_keys=True, default=str)`, truncated to 16 hex chars (D12).

    `sort_keys=True` means `{"a":1, "b":2}` and `{"b":2, "a":1}` produce identical
    hashes — the caller doesn't have to enforce ordering. `default=str` lets the
    config contain non-JSON-serializable objects (e.g., `torch.dtype`); they're
    stringified before hashing.

    The caller is responsible for passing only the hash-relevant keys (seed,
    crop_size, num_frames, learning_rate, model arch, etc.). Things like
    `run_name` or `start_time` should NOT be in the hashed config or every run
    would mismatch trivially.
    """
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _atomic_write_text(path: str, content: str, *, retries: int = 5, delay: float = 0.5) -> None:
    """Atomic-ish text write via tmp + os.replace, hardened for Drive FUSE.

    Drive FUSE is eventually-consistent for small files: a freshly-written tmp may not be visible
    to os.replace yet, which raises FileNotFoundError on the source under rapid writes (the
    checkpoint path dodges this via its torch.load read-back). Mirror that read-back here to force
    materialization, fsync, and retry the whole write+replace on the transient FUSE error.
    """
    dir_ = os.path.dirname(path) or "."
    base = os.path.basename(path)
    last_exc: Exception | None = None
    for attempt in range(retries):
        tmp = os.path.join(dir_, f".tmp_{base}.{os.getpid()}.{attempt}")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(content)
                fh.flush()
                os.fsync(fh.fileno())
            with open(tmp, "r", encoding="utf-8") as fh:
                if fh.read() != content:
                    raise OSError(f"tmp read-back mismatch: {tmp}")
            os.replace(tmp, path)
            return
        except OSError as exc:
            last_exc = exc
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            if attempt < retries - 1:
                time.sleep(delay)
    raise RuntimeError(f"_atomic_write_text failed after {retries} attempts: {path}") from last_exc


def atomic_save_checkpoint(
    payload: dict, target_path: str, *, update_latest: bool = True
) -> None:
    """Atomically save `payload` to `target_path`; update `latest.txt` iff `update_latest` (D11/D7).

    `update_latest` (default True) preserves the exact Phase 2/3 behavior. Pass
    `update_latest=False` for `backbone.pt` / `best.pt` writes so the `latest.txt`
    epoch-resume pointer is never clobbered (D7 / RESEARCH §10 / Pitfall 4) — otherwise
    a `backbone.pt` save would make SSL resume jump to the wrong (best, not latest) epoch.

    Contract (in order — D11):
      1. `torch.save(payload, tmp)` where `tmp = .tmp_{basename}.{pid}` next to target.
      2. `torch.load(tmp, map_location='cpu')` — round-trip verify the write is readable.
      3. `os.replace(tmp, target_path)` — POSIX rename, the closest thing to atomic we
         have on the platform. On most filesystems this is atomic for same-FS ops;
         Drive FUSE doesn't guarantee atomicity but does guarantee the destination
         isn't visible until the source is unlinked, which is good enough.
      4. Write `latest.txt` LAST, using the same tmp+replace pattern. A crash before
         this step leaves the **prior** good `latest.txt` untouched — so a subsequent
         `load_latest_checkpoint` reads the previous epoch, not a partial one.

    On any verify failure, the tmp file is removed and the exception propagates;
    `target_path` and `latest.txt` are NEVER touched if the save was bad.
    """
    target_dir = os.path.dirname(target_path) or "."
    target_name = os.path.basename(target_path)
    os.makedirs(target_dir, exist_ok=True)

    tmp = os.path.join(target_dir, f".tmp_{target_name}.{os.getpid()}")

    # Step 1: torch.save to tmp.
    try:
        torch.save(payload, tmp)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise

    # Step 2: round-trip verify.
    try:
        _ = torch.load(tmp, map_location="cpu", weights_only=False)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise

    # Step 3: atomic rename into place.
    os.replace(tmp, target_path)

    # Step 4: latest.txt LAST. Crash here leaves prior latest.txt pointing to the
    # last successful checkpoint — `target_path` is now also on disk but unreferenced
    # until a subsequent successful save rewrites latest.txt.
    if update_latest:
        latest_path = os.path.join(target_dir, "latest.txt")
        _atomic_write_text(latest_path, target_name + "\n")
        logger.info("atomic_save_checkpoint: %s (+ latest.txt -> %s)", target_path, target_name)
    else:
        # D7 / Pitfall 4: backbone.pt / best.pt write — leave the epoch-resume pointer alone.
        logger.info("atomic_save_checkpoint: %s (latest.txt NOT updated)", target_path)


def load_latest_checkpoint(
    run_dir: str,
    *,
    expected_config_hash: str,
    map_location: str = "cpu",
) -> dict | None:
    """Read `{run_dir}/latest.txt`, load the pointed-at checkpoint, verify config hash.

    Returns:
        The torch-loaded payload dict on success, or `None` if no `latest.txt`
        exists (fresh run) or the pointed-at file is missing (recoverable — caller
        treats as "no prior epoch").

    Raises:
        CheckpointConfigMismatchError: the checkpoint's `config_hash` doesn't match
            `expected_config_hash`. Message includes both hashes + the
            stored `config_repr` for diagnosis. Recovery: rename the run or
            reconcile the config.
    """
    latest_path = os.path.join(run_dir, "latest.txt")
    if not os.path.isfile(latest_path):
        logger.info("load_latest_checkpoint: no %s (fresh run)", latest_path)
        return None

    with open(latest_path, "r", encoding="utf-8") as fh:
        ckpt_name = fh.read().strip()

    ckpt_path = os.path.join(run_dir, ckpt_name)
    if not os.path.isfile(ckpt_path):
        logger.warning(
            "load_latest_checkpoint: latest.txt points to %s but file is missing — "
            "treating as no prior epoch (recoverable)",
            ckpt_path,
        )
        return None

    payload = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    actual_hash = payload.get("config_hash")
    if actual_hash != expected_config_hash:
        raise CheckpointConfigMismatchError(
            f"checkpoint {ckpt_path} hash {actual_hash!r} != expected {expected_config_hash!r} — "
            f"start a new run name or reconcile config; "
            f"checkpoint config_repr={payload.get('config_repr')!r}"
        )

    logger.info(
        "load_latest_checkpoint: loaded %s (epoch=%s, config_hash matches)",
        ckpt_path, payload.get("epoch"),
    )
    return payload


_EPOCH_PT_RE = re.compile(r"^epoch_\d+\.pt$")


def prune_checkpoints(
    run_dir: str,
    *,
    keep_last: int = 3,
    keep_best: bool = True,
) -> None:
    """Keep the `keep_last` most-recent `epoch_*.pt` (by mtime) + optionally `best.pt`; delete the rest (D17).

    Sort by mtime descending — newer first. Top `keep_last` are kept. Anything else
    named `epoch_*.pt` is deleted. If `keep_best=True` and `best.pt` exists, it's
    always kept regardless of position in the sorted list.

    Other files in `run_dir` (e.g., `latest.txt`, JSON metric sidecars, tmp files
    from in-progress writes) are NOT touched — this only manages `epoch_*.pt`.
    """
    if not os.path.isdir(run_dir):
        return

    epochs = [
        os.path.join(run_dir, name)
        for name in os.listdir(run_dir)
        if _EPOCH_PT_RE.match(name)
    ]
    # Sort by mtime descending — most-recent first.
    epochs.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    keep_set = set(epochs[:keep_last])
    if keep_best:
        best_path = os.path.join(run_dir, "best.pt")
        if os.path.isfile(best_path):
            keep_set.add(best_path)  # idempotent (set), best.pt isn't an epoch_*.pt anyway

    deleted = 0
    for p in epochs:
        if p in keep_set:
            continue
        try:
            os.remove(p)
            deleted += 1
        except OSError as exc:
            logger.warning("prune_checkpoints: could not remove %s: %s", p, exc)

    logger.info(
        "prune_checkpoints: %s — kept %d epoch_*.pt%s, deleted %d",
        run_dir,
        min(len(epochs), keep_last),
        " + best.pt" if keep_best and os.path.isfile(os.path.join(run_dir, "best.pt")) else "",
        deleted,
    )
