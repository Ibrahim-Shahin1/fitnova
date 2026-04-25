"""
Targeted video extraction for Fit3D (R2a).

Why we don't extract the whole tarball
--------------------------------------
`fit3d_train.tar.gz` is ~18 GB because every subject has 4 camera angles and
~30 exercises including warmups. For MediaPipe-native training we only need
ONE camera angle (`60457274`) per subject for the 27 exercises we train on
(no warmups, no `walk_the_box`). That brings the working set down to
~2.4 GB — small enough to ship on the same laptop that will run training.

What this script does
---------------------
1. Opens fit3d_train.tar.gz in streaming mode (never loads all names into RAM).
2. Walks every member; keeps only paths that look like
       train/<subject>/videos/<camera>/<exercise>.mp4
   where <subject> is in --subjects, <camera> == --camera,
   and <exercise> is in SUPPORTED_EXERCISES (no warmups, no walk_the_box).
3. Skips members whose destination file already exists with the correct size
   (re-runs are cheap and resumable).
4. Extracts each surviving member, preserving directory structure.

The extracted tree matches what the rest of the pipeline already expects:
    <dest>/train/<subject>/videos/<camera>/<exercise>.mp4

Usage
-----
    python -m backend.training.preprocessing.extract_fit3d_videos \
        [--tar <path to fit3d_train.tar.gz>] \
        [--dest <dataset_parent_dir>]         \
        [--camera 60457274] [--subjects s03,s04,...]
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import time
from typing import Iterable, List, Optional, Set

# The 27 exercises we actually train on.
from .fit3d_loader import SUPPORTED_EXERCISES

DEFAULT_TAR       = r"C:\Users\tsh_x\Desktop\FitNova Drafts2\Form Correction\fit3d_train.tar.gz"
DEFAULT_DEST      = r"C:\Users\tsh_x\Desktop\FitNova Datasets\fit3d\train"
DEFAULT_CAMERA    = "60457274"
DEFAULT_SUBJECTS  = ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]


def _is_wanted(
    path: str,
    camera: str,
    subjects: Set[str],
    exercises: Set[str],
) -> bool:
    """Match `train/<subject>/videos/<camera>/<exercise>.mp4`."""
    if not path.endswith(".mp4"):
        return False
    parts = path.replace("\\", "/").split("/")
    # Expect: ["train", subject, "videos", camera, "<exercise>.mp4"]
    if len(parts) != 5:
        return False
    if parts[0] != "train" or parts[2] != "videos":
        return False
    if parts[3] != camera:
        return False
    if parts[1] not in subjects:
        return False
    exercise = os.path.splitext(parts[4])[0]
    return exercise in exercises


def extract_videos(
    tar_path: str = DEFAULT_TAR,
    dest: str = DEFAULT_DEST,
    camera: str = DEFAULT_CAMERA,
    subjects: Optional[Iterable[str]] = None,
    exercises: Optional[Iterable[str]] = None,
    skip_if_present: bool = True,
    verbose: bool = True,
) -> dict:
    """Extract the matching .mp4 files.

    Returns a stats dict: {"extracted": N, "skipped": M, "elapsed_s": float,
    "bytes": int}.
    """
    subjects = set(subjects or DEFAULT_SUBJECTS)
    exercises = set(exercises or SUPPORTED_EXERCISES)

    if not os.path.isfile(tar_path):
        raise FileNotFoundError(f"tar not found: {tar_path}")
    os.makedirs(dest, exist_ok=True)

    if verbose:
        print(f"[extract] tar     : {tar_path}")
        print(f"[extract] dest    : {dest}")
        print(f"[extract] camera  : {camera}")
        print(f"[extract] subjects: {sorted(subjects)}")
        print(f"[extract] exercises: {len(exercises)} target exercises")
        print("[extract] scanning archive (streaming; this takes ~10-20 s)...")

    n_extract = 0
    n_skip    = 0
    n_bytes   = 0
    t0 = time.time()

    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            name = member.name
            if not _is_wanted(name, camera, subjects, exercises):
                continue

            target_path = os.path.join(dest, name.replace("/", os.sep))

            if skip_if_present and os.path.isfile(target_path):
                try:
                    existing_size = os.path.getsize(target_path)
                except OSError:
                    existing_size = -1
                if existing_size == member.size:
                    n_skip += 1
                    if verbose:
                        print(f"  skip  {name}  ({member.size/1e6:.1f} MB, already present)")
                    continue

            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            if verbose:
                print(f"  xtract {name}  ({member.size/1e6:.1f} MB) ...", flush=True)

            # Stream file object from tar into destination
            f_in = tf.extractfile(member)
            if f_in is None:
                if verbose:
                    print("     (not a regular file, skipping)")
                continue
            try:
                with open(target_path, "wb") as f_out:
                    while True:
                        chunk = f_in.read(1 << 20)   # 1 MB chunks
                        if not chunk:
                            break
                        f_out.write(chunk)
            finally:
                f_in.close()

            n_extract += 1
            n_bytes   += member.size

    elapsed = time.time() - t0
    if verbose:
        print()
        print(f"[extract] extracted: {n_extract}")
        print(f"[extract] skipped  : {n_skip}")
        print(f"[extract] size     : {n_bytes/1e9:.2f} GB")
        print(f"[extract] elapsed  : {elapsed:.1f} s")
    return {"extracted": n_extract, "skipped": n_skip,
            "bytes": n_bytes, "elapsed_s": elapsed}


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--tar",     default=DEFAULT_TAR)
    p.add_argument("--dest",    default=DEFAULT_DEST)
    p.add_argument("--camera",  default=DEFAULT_CAMERA)
    p.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS),
                   help="Comma-separated subjects (e.g. s03,s04,s10)")
    p.add_argument("--no-skip", action="store_true",
                   help="Re-extract even if destination file already exists with correct size")
    args = p.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    extract_videos(
        tar_path=args.tar, dest=args.dest, camera=args.camera,
        subjects=subjects, skip_if_present=(not args.no_skip),
    )


if __name__ == "__main__":
    _cli()
