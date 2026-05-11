"""
Batch MediaPipe Pose extraction from Fit3D videos (v5 / Tasks API).

Purpose
-------
Produce MediaPipe-native training data that matches the inference pipeline in
``backend/services/form_analyzer.py`` byte-for-byte.

For every video at::

    <fit3d_root>/<split>/<subject>/videos/<camera>/<exercise>.mp4   (train)
    <fit3d_root>/<split>/<subject>/videos/<exercise>.mp4            (test, no camera subdir)

we run MediaPipe Pose Landmarker (Tasks API, VIDEO running mode, model
``pose_landmarker_full.task``) frame-by-frame and save::

    mediapipe_fit3d_v5/<split>/<subject>/<camera>/<exercise>.npy   (T, 33, 4)
    mediapipe_fit3d_v5/<split>/<subject>/<camera>/<exercise>.json  (metadata)

Channels of the .npy are ``[x, y, z, visibility]`` from
``result.pose_world_landmarks[0]`` — metric 3D coordinates that align with
Fit3D ``joints3d_25`` after canonical mapping.

For test subjects (which only have one camera and no camera subdir), the
camera id is recorded as ``"single"`` in the output path so the same downstream
loaders work uniformly.

Why Tasks API and not legacy ``mp.solutions.pose``?
---------------------------------------------------
The legacy ``mp.solutions`` namespace was removed in MediaPipe 0.10.33+; the
v4 extraction script (which imported ``mp.solutions.pose``) crashed on every
single video with ``module 'mediapipe' has no attribute 'solutions'`` (see
``backend/data/mediapipe_fit3d/_extraction_summary.json``). The Tasks API is
the canonical 0.10.x path AND it is exactly what ``form_analyzer.py`` uses at
inference time, so train and inference now run identical code paths.

Crash-resume
------------
``skip_existing=True`` (default) checks for the .npy + .json pair before
re-running. Each (video → npy) is atomic: if the worker dies mid-video, the
.npy is never written, so re-running picks up exactly where it stopped.

Usage
-----
Smoke test on one already-extracted video::

    python -m backend.training.preprocessing.mediapipe_extractor --smoke

Full extraction (after tarballs are unpacked)::

    python -m backend.training.preprocessing.mediapipe_extractor \\
        --fit3d-root "C:/Users/tsh_x/Desktop/FitNova Drafts2/Form Correction"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_FIT3D_ROOT = r"C:\Users\tsh_x\Desktop\FitNova Drafts2\Form Correction"
DEFAULT_OUT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "mediapipe_fit3d_v5")
)
DEFAULT_TASK_MODEL = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "models", "pose_landmarker_full.task"
    )
)

# Train layout: <fit3d_root>/train/<subject>/videos/<camera>/<exercise>.mp4
# Test  layout: <fit3d_root>/test/<subject>/videos/<exercise>.mp4 (no camera subdir)
DEFAULT_TRAIN_SUBJECTS = ["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"]
DEFAULT_TEST_SUBJECTS = ["s02", "s12", "s13"]
DEFAULT_TRAIN_CAMERAS = ["50591643", "58860488", "60457274", "65906101"]
DEFAULT_WORKERS = 4

# ── Worker ────────────────────────────────────────────────────────────────────


def _extract_one(
    video_path: str,
    out_npy: str,
    out_json: str,
    task_model_path: str,
) -> dict:
    """Run MediaPipe Pose (Tasks API, VIDEO mode) on one video and save .npy.

    Executed in a subprocess. All heavy imports are local. Never raises —
    errors are captured into the returned dict.

    Returned dict has either an "error" key (failure) or full result fields.
    """
    t0 = time.time()
    try:
        import mediapipe as mp
    except Exception as e:  # pragma: no cover
        return {"video": video_path, "error": f"mediapipe import failed: {e}"}

    if not os.path.isfile(video_path):
        return {"video": video_path, "error": "missing_input"}
    if not os.path.isfile(task_model_path):
        return {
            "video": video_path,
            "error": f"missing_task_model: {task_model_path}",
        }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"video": video_path, "error": "cv2_open_failed"}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_interval_ms = max(1, int(round(1000.0 / fps)))

    # ── Build MediaPipe Tasks API PoseLandmarker (matches form_analyzer.py) ──
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=task_model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        landmarker = PoseLandmarker.create_from_options(options)
    except Exception as e:
        cap.release()
        return {
            "video": video_path,
            "error": f"landmarker_create_failed: {type(e).__name__}: {e}",
        }

    landmarks: List[np.ndarray] = []   # (T, 33, 3)
    visibilities: List[np.ndarray] = []  # (T, 33, 1)
    n_detected = 0
    frame_idx = 0
    ts_ms = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            res = landmarker.detect_for_video(mp_img, ts_ms)
            ts_ms += frame_interval_ms

            if res.pose_world_landmarks:
                lms = res.pose_world_landmarks[0]
                xyz = np.array(
                    [[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32
                )
                vis = np.array(
                    [[float(getattr(lm, "visibility", 0.0))] for lm in lms],
                    dtype=np.float32,
                )
                n_detected += 1
            else:
                xyz = np.full((33, 3), np.nan, dtype=np.float32)
                vis = np.zeros((33, 1), dtype=np.float32)

            landmarks.append(xyz)
            visibilities.append(vis)
            frame_idx += 1
    except Exception as e:
        cap.release()
        try:
            landmarker.close()
        except Exception:
            pass
        return {
            "video": video_path,
            "error": f"frame_loop_exc: {type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=4),
        }

    cap.release()
    try:
        landmarker.close()
    except Exception:
        pass

    if frame_idx == 0:
        return {"video": video_path, "error": "no_frames_decoded"}

    arr_xyz = np.stack(landmarks, axis=0)  # (T, 33, 3)
    arr_vis = np.stack(visibilities, axis=0)  # (T, 33, 1)
    combined = np.concatenate([arr_xyz, arr_vis], axis=-1)  # (T, 33, 4)

    # Atomic write: tmp file then rename. Prevents half-written .npy on crash.
    # NOTE: np.save() auto-appends ".npy" if the path doesn't already end in
    # ".npy", so we pass an open file handle to bypass that behavior.
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    tmp_npy = out_npy + ".tmp"
    with open(tmp_npy, "wb") as fh:
        np.save(fh, combined)
    os.replace(tmp_npy, out_npy)

    mean_vis = float(np.nanmean(arr_vis)) if n_detected else 0.0
    summary = {
        "video": video_path,
        "fps": fps,
        "n_frames_video_meta": n_frames_meta,
        "n_frames_saved": frame_idx,
        "n_detected": n_detected,
        "mean_visibility": mean_vis,
        "task_model": os.path.basename(task_model_path),
        "running_mode": "VIDEO",
        "model_complexity": "full",  # pose_landmarker_full.task
        "seconds_to_extract": round(time.time() - t0, 2),
    }
    tmp_json = out_json + ".tmp"
    with open(tmp_json, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp_json, out_json)

    return {
        "video": video_path,
        "out_npy": out_npy,
        "n_frames": frame_idx,
        "n_detected": n_detected,
        "mean_visibility": mean_vis,
        "seconds": round(time.time() - t0, 1),
    }


# ── Driver ────────────────────────────────────────────────────────────────────


def _list_exercises_in_dir(video_dir: str) -> List[str]:
    """Return every exercise name (.mp4 stem) present in a video directory."""
    if not os.path.isdir(video_dir):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(video_dir)
        if f.lower().endswith(".mp4")
    )


def _plan_jobs(
    fit3d_root: str,
    out_root: str,
    train_subjects: List[str],
    train_cameras: List[str],
    test_subjects: List[str],
    skip_existing: bool,
    exercises_filter: Optional[List[str]] = None,
) -> List[Tuple[str, str, str, str]]:
    """Walk the dataset and return (video_path, out_npy, out_json, label) tuples.

    ``label`` is a short human string for log lines.
    """
    jobs: List[Tuple[str, str, str, str]] = []

    # ── Train: <fit3d_root>/train/<subj>/videos/<camera>/<ex>.mp4 ────────────
    train_root = os.path.join(fit3d_root, "train")
    if os.path.isdir(train_root):
        for subj in train_subjects:
            for cam in train_cameras:
                video_dir = os.path.join(train_root, subj, "videos", cam)
                if not os.path.isdir(video_dir):
                    continue
                for ex in _list_exercises_in_dir(video_dir):
                    if exercises_filter and ex not in exercises_filter:
                        continue
                    vp = os.path.join(video_dir, f"{ex}.mp4")
                    out_npy = os.path.join(
                        out_root, "train", subj, cam, f"{ex}.npy"
                    )
                    out_json = os.path.join(
                        out_root, "train", subj, cam, f"{ex}.json"
                    )
                    if (
                        skip_existing
                        and os.path.isfile(out_npy)
                        and os.path.isfile(out_json)
                    ):
                        continue
                    label = f"train/{subj}/{cam}/{ex}"
                    jobs.append((vp, out_npy, out_json, label))

    # ── Test: <fit3d_root>/test/<subj>/videos/<ex>.mp4 (no camera subdir) ────
    test_root = os.path.join(fit3d_root, "test")
    if os.path.isdir(test_root):
        for subj in test_subjects:
            video_dir = os.path.join(test_root, subj, "videos")
            if not os.path.isdir(video_dir):
                continue
            for ex in _list_exercises_in_dir(video_dir):
                if exercises_filter and ex not in exercises_filter:
                    continue
                vp = os.path.join(video_dir, f"{ex}.mp4")
                out_npy = os.path.join(
                    out_root, "test", subj, "single", f"{ex}.npy"
                )
                out_json = os.path.join(
                    out_root, "test", subj, "single", f"{ex}.json"
                )
                if (
                    skip_existing
                    and os.path.isfile(out_npy)
                    and os.path.isfile(out_json)
                ):
                    continue
                label = f"test/{subj}/single/{ex}"
                jobs.append((vp, out_npy, out_json, label))

    return jobs


def extract_all(
    fit3d_root: str = DEFAULT_FIT3D_ROOT,
    out_root: str = DEFAULT_OUT_ROOT,
    task_model_path: str = DEFAULT_TASK_MODEL,
    train_subjects: Optional[Iterable[str]] = None,
    train_cameras: Optional[Iterable[str]] = None,
    test_subjects: Optional[Iterable[str]] = None,
    workers: int = DEFAULT_WORKERS,
    skip_existing: bool = True,
    exercises_filter: Optional[Iterable[str]] = None,
    verbose: bool = True,
) -> dict:
    """Plan and run extraction across train + test partitions."""
    train_subjects = list(train_subjects or DEFAULT_TRAIN_SUBJECTS)
    train_cameras = list(train_cameras or DEFAULT_TRAIN_CAMERAS)
    test_subjects = list(test_subjects or DEFAULT_TEST_SUBJECTS)
    exercises_filter = list(exercises_filter) if exercises_filter else None

    jobs = _plan_jobs(
        fit3d_root,
        out_root,
        train_subjects,
        train_cameras,
        test_subjects,
        skip_existing=skip_existing,
        exercises_filter=exercises_filter,
    )

    if verbose:
        print(f"[mp-extract-v5] fit3d_root  : {fit3d_root}")
        print(f"[mp-extract-v5] out_root    : {out_root}")
        print(f"[mp-extract-v5] task_model  : {task_model_path}")
        print(f"[mp-extract-v5] train subj  : {train_subjects}")
        print(f"[mp-extract-v5] train cams  : {train_cameras}")
        print(f"[mp-extract-v5] test  subj  : {test_subjects}")
        print(f"[mp-extract-v5] jobs        : {len(jobs)}")
        print(f"[mp-extract-v5] workers     : {workers}")
        if exercises_filter:
            print(f"[mp-extract-v5] exercise filter: {exercises_filter}")
        if not jobs:
            print("[mp-extract-v5] nothing to do (all targets already cached).")
            return {"n_jobs": 0, "n_success": 0, "failures": [], "skipped": True}

    t0 = time.time()
    failures: List[dict] = []
    successes = 0

    if workers <= 1:
        for i, (vp, out_npy, out_json, label) in enumerate(jobs):
            res = _extract_one(vp, out_npy, out_json, task_model_path)
            if "error" in res:
                failures.append(res)
                if verbose:
                    print(f"  [{i+1}/{len(jobs)}] FAIL {label} :: {res['error']}")
            else:
                successes += 1
                if verbose:
                    print(
                        f"  [{i+1}/{len(jobs)}] ok   {label}  "
                        f"({res['n_frames']}f, {res['n_detected']} det, "
                        f"vis={res['mean_visibility']:.2f}, {res['seconds']}s)"
                    )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _extract_one, vp, out_npy, out_json, task_model_path
                ): (label, i)
                for i, (vp, out_npy, out_json, label) in enumerate(jobs)
            }
            for fut in as_completed(futures):
                label, i = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {
                        "video": label,
                        "error": f"worker_exception: {type(e).__name__}: {e}",
                    }
                if "error" in res:
                    failures.append(res)
                    if verbose:
                        print(
                            f"  [{i+1}/{len(jobs)}] FAIL {label} :: {res['error']}"
                        )
                else:
                    successes += 1
                    if verbose:
                        print(
                            f"  [{i+1}/{len(jobs)}] ok   {label}  "
                            f"({res['n_frames']}f, {res['n_detected']} det, "
                            f"vis={res['mean_visibility']:.2f}, {res['seconds']}s)"
                        )

    elapsed = time.time() - t0
    summary = {
        "n_jobs": len(jobs),
        "n_success": successes,
        "n_failed": len(failures),
        "failures": failures,
        "elapsed_s": round(elapsed, 1),
        "out_root": out_root,
        "task_model": task_model_path,
        "running_mode": "VIDEO",
    }
    if verbose:
        print()
        print(
            f"[mp-extract-v5] {successes}/{len(jobs)} succeeded in "
            f"{elapsed/60:.1f} min"
        )
        if failures:
            print(f"[mp-extract-v5] {len(failures)} failed (see _extraction_summary.json)")

    os.makedirs(out_root, exist_ok=True)
    summary_path = os.path.join(out_root, "_extraction_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ── CLI / smoke test ──────────────────────────────────────────────────────────


def _smoke_test(out_root: str = DEFAULT_OUT_ROOT,
                task_model_path: str = DEFAULT_TASK_MODEL) -> int:
    """Run on the one already-extracted video as a quick sanity check.

    Returns 0 on success, non-zero on failure. Used by the CLI ``--smoke`` flag
    so we don't trigger the full Phase-1 pipeline by accident.
    """
    smoke_video = os.path.join(
        DEFAULT_FIT3D_ROOT, "train", "s05", "videos", "60457274", "squat.mp4"
    )
    if not os.path.isfile(smoke_video):
        print(f"[smoke] FAIL: smoke video not found: {smoke_video}")
        return 1

    smoke_out_dir = os.path.join(out_root, "_smoke")
    out_npy = os.path.join(smoke_out_dir, "s05_60457274_squat.npy")
    out_json = os.path.join(smoke_out_dir, "s05_60457274_squat.json")

    # Force re-extraction even if cached
    for p in (out_npy, out_json):
        if os.path.exists(p):
            os.remove(p)

    print(f"[smoke] extracting one video: {smoke_video}")
    res = _extract_one(smoke_video, out_npy, out_json, task_model_path)
    if "error" in res:
        print(f"[smoke] FAIL: {res['error']}")
        if "trace" in res:
            print(res["trace"])
        return 2

    arr = np.load(out_npy)
    nan_pct = float(np.isnan(arr).mean()) * 100
    print()
    print(f"[smoke] OK   {res['n_frames']} frames, "
          f"{res['n_detected']} detected, "
          f"mean_vis={res['mean_visibility']:.3f}")
    print(f"[smoke] .npy shape={arr.shape} dtype={arr.dtype} NaN={nan_pct:.1f}%")
    print(f"[smoke] saved   {out_npy}")
    print(f"[smoke]         {out_json}")
    print(f"[smoke] elapsed {res['seconds']}s")
    if arr.shape[1:] != (33, 4):
        print("[smoke] WARN: expected (T, 33, 4) shape")
        return 3
    if nan_pct > 30:
        print("[smoke] WARN: high NaN rate — model may be missing pose")
    return 0


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--fit3d-root", default=DEFAULT_FIT3D_ROOT,
                   help="Root with train/ and test/ subfolders")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--task-model", default=DEFAULT_TASK_MODEL,
                   help="Path to pose_landmarker_full.task")
    p.add_argument("--train-subjects", default=",".join(DEFAULT_TRAIN_SUBJECTS))
    p.add_argument("--train-cameras", default=",".join(DEFAULT_TRAIN_CAMERAS))
    p.add_argument("--test-subjects", default=",".join(DEFAULT_TEST_SUBJECTS))
    p.add_argument("--exercises", default="",
                   help="Optional comma-separated exercise filter")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--no-skip", action="store_true",
                   help="Re-extract even if .npy/.json already exist")
    p.add_argument("--smoke", action="store_true",
                   help="Run only the single-video smoke test and exit")
    args = p.parse_args()

    if args.smoke:
        sys.exit(_smoke_test(args.out_root, args.task_model))

    train_subjects = [s.strip() for s in args.train_subjects.split(",") if s.strip()]
    train_cameras = [s.strip() for s in args.train_cameras.split(",") if s.strip()]
    test_subjects = [s.strip() for s in args.test_subjects.split(",") if s.strip()]
    exercises_filter = (
        [s.strip() for s in args.exercises.split(",") if s.strip()]
        or None
    )

    extract_all(
        fit3d_root=args.fit3d_root,
        out_root=args.out_root,
        task_model_path=args.task_model,
        train_subjects=train_subjects,
        train_cameras=train_cameras,
        test_subjects=test_subjects,
        workers=args.workers,
        skip_existing=(not args.no_skip),
        exercises_filter=exercises_filter,
    )


if __name__ == "__main__":
    _cli()
