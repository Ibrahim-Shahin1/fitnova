"""
Batch MediaPipe Pose extraction from Fit3D videos (R2b).

Purpose
-------
Produce MediaPipe-native training data. For every extracted
`train/<subject>/videos/<camera>/<exercise>.mp4` we run MediaPipe Pose
(model_complexity=1, same as inference) frame-by-frame and save:

    mediapipe_fit3d/<subject>/<exercise>.npy          # (n_frames, 33, 4)
                                                       #   channels = [x, y, z, visibility]
    mediapipe_fit3d/<subject>/<exercise>.json         # fps, n_frames,
                                                       #   n_detected, mean_visibility

World landmarks (`pose_world_landmarks`) are saved — they're metric 3D and
match the dimensions of Fit3D joints3d_25 after canonical mapping.

Parallel strategy
-----------------
MediaPipe holds a TensorFlow Lite interpreter per process, so we run one
video per worker process with a `ProcessPoolExecutor`. 4 workers is a
sensible default on a laptop (MediaPipe saturates ~2 cores per video).

Usage
-----
    python -m backend.training.preprocessing.mediapipe_extractor \\
        [--video-root <...>] [--out-root <...>] [--camera 60457274] \\
        [--subjects s03,s04,...] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

# We import SUPPORTED_EXERCISES lazily inside workers (below) to avoid the
# (sometimes slow) Fit3D package import when the pool starts a subprocess.

DEFAULT_VIDEO_ROOT = r"C:\Users\tsh_x\Desktop\FitNova Datasets\fit3d\train\train"
DEFAULT_OUT_ROOT   = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "mediapipe_fit3d")
)
DEFAULT_CAMERA     = "60457274"
DEFAULT_SUBJECTS   = ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]
DEFAULT_WORKERS    = 4


# ── Worker ────────────────────────────────────────────────────────────────────

def _extract_one(
    video_path: str,
    out_npy: str,
    out_json: str,
    model_complexity: int = 1,
) -> dict:
    """Run MediaPipe Pose on one video and save the landmark tensor.

    Executed in a subprocess — all heavy imports are local.
    Returns a summary dict. Never raises (errors are captured into the dict).

    NOTE: Designed for Colab where mediapipe is pre-installed correctly.
    """
    t0 = time.time()
    try:
        import mediapipe as mp
    except ImportError as e:
        return {"video": video_path, "error": f"mediapipe import failed: {e}"}

    if not os.path.isfile(video_path):
        return {"video": video_path, "error": "missing_input"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"video": video_path, "error": "cv2_open_failed"}

    fps        = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarks: List[np.ndarray] = []
    visibilities: List[np.ndarray] = []
    n_detected = 0

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_world_landmarks is not None:
            lms = res.pose_world_landmarks.landmark   # 33
            xyz = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            vis = np.array([[lm.visibility] for lm in lms], dtype=np.float32)
            n_detected += 1
        else:
            # No pose — fill with NaN so downstream code can detect dropouts
            xyz = np.full((33, 3), np.nan, dtype=np.float32)
            vis = np.zeros((33, 1),          dtype=np.float32)

        landmarks.append(xyz)
        visibilities.append(vis)
        frame_idx += 1

    cap.release()
    pose.close()

    if frame_idx == 0:
        return {"video": video_path, "error": "no_frames_decoded"}

    arr_xyz = np.stack(landmarks, axis=0)                  # (T, 33, 3)
    arr_vis = np.stack(visibilities, axis=0)               # (T, 33, 1)
    combined = np.concatenate([arr_xyz, arr_vis], axis=-1) # (T, 33, 4)

    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, combined)

    with open(out_json, "w") as f:
        json.dump({
            "video":           video_path,
            "fps":             fps,
            "n_frames_video":  n_frames,
            "n_frames_saved":  frame_idx,
            "n_detected":      n_detected,
            "mean_visibility": float(np.nanmean(arr_vis)) if n_detected else 0.0,
            "model_complexity": model_complexity,
            "seconds_to_extract": round(time.time() - t0, 2),
        }, f, indent=2)

    return {
        "video": video_path,
        "out_npy": out_npy,
        "n_frames": frame_idx,
        "n_detected": n_detected,
        "seconds": round(time.time() - t0, 1),
    }


# ── Driver ────────────────────────────────────────────────────────────────────

def _plan_jobs(
    video_root: str,
    out_root: str,
    camera: str,
    subjects: List[str],
    exercises: List[str],
    skip_existing: bool,
) -> List[Tuple[str, str, str]]:
    """Return list of (video_path, out_npy, out_json) triples to process."""
    jobs: List[Tuple[str, str, str]] = []
    for subj in subjects:
        video_dir = os.path.join(video_root, subj, "videos", camera)
        if not os.path.isdir(video_dir):
            continue
        for ex in exercises:
            vp = os.path.join(video_dir, f"{ex}.mp4")
            if not os.path.isfile(vp):
                continue
            out_npy  = os.path.join(out_root, subj, f"{ex}.npy")
            out_json = os.path.join(out_root, subj, f"{ex}.json")
            if skip_existing and os.path.isfile(out_npy) and os.path.isfile(out_json):
                continue
            jobs.append((vp, out_npy, out_json))
    return jobs


def extract_all(
    video_root: str = DEFAULT_VIDEO_ROOT,
    out_root: str = DEFAULT_OUT_ROOT,
    camera: str = DEFAULT_CAMERA,
    subjects: Optional[Iterable[str]] = None,
    exercises: Optional[Iterable[str]] = None,
    workers: int = DEFAULT_WORKERS,
    skip_existing: bool = True,
    verbose: bool = True,
) -> dict:
    from .fit3d_loader import SUPPORTED_EXERCISES
    subjects = list(subjects) if subjects is not None else DEFAULT_SUBJECTS
    exercises = list(exercises) if exercises is not None else SUPPORTED_EXERCISES

    jobs = _plan_jobs(video_root, out_root, camera,
                      subjects, exercises, skip_existing)

    if verbose:
        print(f"[mp-extract] video_root: {video_root}")
        print(f"[mp-extract] out_root  : {out_root}")
        print(f"[mp-extract] camera    : {camera}")
        print(f"[mp-extract] jobs      : {len(jobs)} videos")
        print(f"[mp-extract] workers   : {workers}")
        if not jobs:
            print("[mp-extract] nothing to do (all targets already cached).")
            return {"n_jobs": 0, "n_success": 0, "failures": []}

    t0 = time.time()
    failures: List[dict] = []
    successes = 0

    # Serial path for workers == 1 (useful for debugging or low-RAM machines)
    if workers <= 1:
        for i, (vp, out_npy, out_json) in enumerate(jobs):
            res = _extract_one(vp, out_npy, out_json)
            if "error" in res:
                failures.append(res)
                if verbose:
                    print(f"  [{i+1}/{len(jobs)}] FAIL {vp} :: {res['error']}")
            else:
                successes += 1
                if verbose:
                    print(f"  [{i+1}/{len(jobs)}] ok   {vp}  "
                          f"({res['n_frames']}f, {res['n_detected']} det, {res['seconds']}s)")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_extract_one, vp, out_npy, out_json): (vp, i)
                for i, (vp, out_npy, out_json) in enumerate(jobs)
            }
            for fut in as_completed(futures):
                vp, i = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"video": vp, "error": f"worker_exception: {e}"}
                if "error" in res:
                    failures.append(res)
                    if verbose:
                        print(f"  [{i+1}/{len(jobs)}] FAIL {vp} :: {res['error']}")
                else:
                    successes += 1
                    if verbose:
                        print(f"  [{i+1}/{len(jobs)}] ok   {vp}  "
                              f"({res['n_frames']}f, {res['n_detected']} det, {res['seconds']}s)")

    elapsed = time.time() - t0
    summary = {
        "n_jobs":    len(jobs),
        "n_success": successes,
        "n_failed":  len(failures),
        "failures":  failures,
        "elapsed_s": round(elapsed, 1),
        "out_root":  out_root,
    }
    if verbose:
        print()
        print(f"[mp-extract] {successes}/{len(jobs)} succeeded in {elapsed/60:.1f} min")
        if failures:
            print(f"[mp-extract] {len(failures)} failed (see JSON log)")
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "_extraction_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    p.add_argument("--out-root",   default=DEFAULT_OUT_ROOT)
    p.add_argument("--camera",     default=DEFAULT_CAMERA)
    p.add_argument("--subjects",   default=",".join(DEFAULT_SUBJECTS))
    p.add_argument("--workers",    type=int, default=DEFAULT_WORKERS)
    p.add_argument("--no-skip",    action="store_true",
                   help="Re-extract even if the target npy/json already exist")
    args = p.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    extract_all(
        video_root=args.video_root,
        out_root=args.out_root,
        camera=args.camera,
        subjects=subjects,
        workers=args.workers,
        skip_existing=(not args.no_skip),
    )


if __name__ == "__main__":
    _cli()
