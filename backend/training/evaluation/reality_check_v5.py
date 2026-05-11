"""
Phase 6 reality-check gate — v5 model on user-recorded squat videos.

The four sub-criteria from the v5 plan §7 (HARD GATE):
  1. "squat" classification on all 4 clips
  2. Measurable rep-boundary spikes (>0.5 confidence somewhere)
  3. Quality gap >= 0.15 between mean(Good_*) and mean(Bad_*)
  4. Knee or hip channel signal > 0.2 somewhere in Bad_*, < 0.1 throughout Good_*

If any of these fail, v5 is not shippable.

Pipeline matches `form_analyzer.py` exactly so the phone runs the same code path:
  - MediaPipe Pose Tasks API, VIDEO running mode, ``pose_landmarker_full.task``
  - 33 world landmarks → 15 canonical joints (with B5 neck-weighted blend)
  - hip-centred, torso-scaled normalisation
  - 22 AIFit angular features per frame
  - Slide 64-frame windows (stride 8) across the whole clip
  - Aggregate per-window predictions to per-clip statistics

Usage:
    python -m backend.training.evaluation.reality_check_v5
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from backend.training.models.st_gcn import build_v5_model
from backend.training.preprocessing.angular_features import compute_sequence_angles
from backend.training.preprocessing.normalize import (
    extract_canonical_from_mediapipe,
    normalize_skeleton,
    resample_sequence,
    TARGET_FRAMES,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# Default to v5.1; CLI/env can override via --model-dir.
MODEL_DIR = os.environ.get(
    "FITNOVA_MODEL_DIR",
    os.path.join(PROJECT_ROOT, "backend", "models", "form_model_v5_1"),
)
DATASET_DIR = os.environ.get(
    "FITNOVA_DATASET_DIR",
    os.path.join(PROJECT_ROOT, "backend", "data", "v5_1_dataset"),
)
TASK_MODEL = os.path.join(PROJECT_ROOT, "backend", "models", "pose_landmarker_full.task")
WEIGHTS_FILENAME = os.environ.get("FITNOVA_WEIGHTS_FILE", "v5_1_supervised.weights.h5")

USER_VIDEOS = [
    (r"C:\Users\tsh_x\Downloads\Good_Squats.mp4",  "Good_Squats",  "good"),
    (r"C:\Users\tsh_x\Downloads\Good_Squats2.mp4", "Good_Squats2", "good"),
    (r"C:\Users\tsh_x\Downloads\Bad_Squats.mp4",   "Bad_Squats",   "bad"),
    (r"C:\Users\tsh_x\Downloads\Bad_Squats2.mp4",  "Bad_Squats2",  "bad"),
]

EXERCISE = "squat"
WINDOW_STRIDE = 8

# Joint group order from aifit_features.JOINT_GROUPS_V5 = ["knee","hip","back","shoulder","elbow"]
GROUP_NAMES = ["knee", "hip", "back", "shoulder", "elbow"]


# ── MediaPipe extraction (same as training extractor) ─────────────────────────


def extract_mediapipe_joints(video_path: str, task_model: str = TASK_MODEL) -> np.ndarray:
    """Run MediaPipe Pose VIDEO-mode on a clip; return (T, 33, 4) world landmarks."""
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_interval_ms = max(1, int(round(1000.0 / fps)))

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=task_model),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    landmarks: List[np.ndarray] = []
    visibilities: List[np.ndarray] = []
    n_detected = 0
    ts_ms = 0
    frame_idx = 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect_for_video(mp_img, ts_ms)
        ts_ms += frame_interval_ms

        if res.pose_world_landmarks:
            lms = res.pose_world_landmarks[0]
            xyz = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
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

    cap.release()
    landmarker.close()

    if frame_idx == 0:
        raise RuntimeError("no frames decoded")

    arr = np.concatenate([np.stack(landmarks), np.stack(visibilities)], axis=-1)
    print(f"    extracted {frame_idx} frames @ {fps:.1f} fps "
          f"({n_detected} detected, vis_mean={float(np.nanmean(arr[..., 3])):.3f})")
    return arr  # (T, 33, 4)


# ── Pre-process to model inputs ──────────────────────────────────────────────


def joints_to_inputs(mp_joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(T_full, 33, 4) MediaPipe → (T_full, 15, 4) canonical_pose, (T_full, 22) angles."""
    xyz = mp_joints[..., :3]
    vis = mp_joints[..., 3:4]

    # Forward-fill any NaN frames (when MediaPipe loses pose)
    if np.isnan(xyz).any():
        T = xyz.shape[0]
        for t in range(T):
            if np.isnan(xyz[t]).any() and t > 0:
                xyz[t] = xyz[t - 1]
                vis[t] = vis[t - 1]
        # Backfill if first frames are NaN
        for t in range(T):
            if np.isnan(xyz[t]).any():
                for tt in range(T):
                    if not np.isnan(xyz[tt]).any():
                        xyz[t] = xyz[tt]
                        vis[t] = vis[tt]
                        break

    canon_xyz = extract_canonical_from_mediapipe(xyz)        # (T, 15, 3)
    canon_xyz = normalize_skeleton(canon_xyz)                 # (T, 15, 3)

    # Build per-canonical-joint visibility (mirrors dataset_builder_v5)
    from backend.training.preprocessing.joint_mapping import (
        MEDIAPIPE_JOINTS, MEDIAPIPE_TO_CANONICAL,
    )
    from backend.training.preprocessing.joint_mapping import N_CANONICAL
    T = xyz.shape[0]
    canon_vis = np.zeros((T, N_CANONICAL, 1), dtype=np.float32)
    for can_idx, mp_idx in MEDIAPIPE_TO_CANONICAL.items():
        if can_idx == 14:
            continue
        if mp_idx is None:
            continue
        if mp_idx == "neck_weighted":
            l = MEDIAPIPE_JOINTS["l_shoulder"]
            r = MEDIAPIPE_JOINTS["r_shoulder"]
            le = MEDIAPIPE_JOINTS["l_ear"]
            re = MEDIAPIPE_JOINTS["r_ear"]
            canon_vis[:, can_idx] = 0.25 * (
                vis[:, l] + vis[:, r] + vis[:, le] + vis[:, re]
            )
        elif isinstance(mp_idx, tuple):
            canon_vis[:, can_idx] = np.mean(
                [vis[:, i] for i in mp_idx], axis=0
            )
        else:
            canon_vis[:, can_idx] = vis[:, mp_idx]
    canon_vis[:, 14] = 0.5 * (canon_vis[:, 12] + canon_vis[:, 13])

    pose = np.concatenate([canon_xyz, canon_vis], axis=-1)    # (T, 15, 4)
    angles = compute_sequence_angles(canon_xyz)                # (T, 22)
    return pose.astype(np.float32), angles.astype(np.float32)


# ── Sliding-window inference ──────────────────────────────────────────────────


def sliding_predict(model, pose: np.ndarray, angles: np.ndarray,
                    exercise_idx: int, stride: int = WINDOW_STRIDE) -> Dict[str, np.ndarray]:
    """Run model on overlapping 64-frame windows; return aggregated per-clip outputs.

    Returns dict with:
      windows_quality   (n_win,)
      windows_action    (n_win, n_exercises)
      per_frame_boundary  (T_total,)
      per_frame_joint_err (T_total, 5)
    """
    T = pose.shape[0]
    if T < TARGET_FRAMES:
        # Pad by repeating last frame
        pad = TARGET_FRAMES - T
        pose = np.concatenate([pose, np.tile(pose[-1:], (pad, 1, 1))], axis=0)
        angles = np.concatenate([angles, np.tile(angles[-1:], (pad, 1))], axis=0)
        T = TARGET_FRAMES

    starts = list(range(0, T - TARGET_FRAMES + 1, stride))
    if starts[-1] != T - TARGET_FRAMES:
        starts.append(T - TARGET_FRAMES)

    n_win = len(starts)
    pose_b = np.stack([pose[s:s + TARGET_FRAMES] for s in starts], axis=0)
    ang_b = np.stack([angles[s:s + TARGET_FRAMES] for s in starts], axis=0)
    ex_b = np.full((n_win,), exercise_idx, dtype=np.int32)

    preds = model.predict(
        {"pose": pose_b, "angles": ang_b, "exercise_id": ex_b},
        batch_size=16, verbose=0,
    )
    # preds: dict with keys quality, action, rep_count, boundary, joint_err, trunk

    # Aggregate per-frame outputs by averaging overlapping windows
    boundary_acc = np.zeros(T, dtype=np.float32)
    boundary_cnt = np.zeros(T, dtype=np.float32)
    joint_err_acc = np.zeros((T, 5), dtype=np.float32)
    joint_err_cnt = np.zeros((T, 5), dtype=np.float32)
    for w, s in enumerate(starts):
        boundary_acc[s:s + TARGET_FRAMES] += preds["boundary"][w, :, 0]
        boundary_cnt[s:s + TARGET_FRAMES] += 1.0
        joint_err_acc[s:s + TARGET_FRAMES] += preds["joint_err"][w]
        joint_err_cnt[s:s + TARGET_FRAMES] += 1.0
    boundary_per_frame = boundary_acc / np.maximum(boundary_cnt, 1.0)
    # Element-wise: both joint_err_acc and joint_err_cnt are (T, 5)
    joint_err_per_frame = joint_err_acc / np.maximum(joint_err_cnt, 1.0)

    return {
        "windows_quality":   preds["quality"][:, 0],         # (n_win,)
        "windows_action":    preds["action"],                # (n_win, n_exercises)
        "windows_rep_count": preds["rep_count"][:, 0],
        "boundary":          boundary_per_frame,             # (T,)
        "joint_err":         joint_err_per_frame,            # (T, 5)
    }


# ── Main reality check ───────────────────────────────────────────────────────


def main() -> int:
    print("=" * 78)
    print("Phase 6 reality-check gate — v5 model on user-recorded squat videos")
    print("=" * 78)
    print()

    # Load exercise label map
    with open(os.path.join(DATASET_DIR, "exercise_labels.json")) as f:
        ex2idx = json.load(f)
    idx2ex = {v: k for k, v in ex2idx.items()}
    n_exercises = len(ex2idx)
    exercise_idx = ex2idx[EXERCISE]

    print(f"Loading model from {MODEL_DIR}")
    model = build_v5_model(
        target_frames=TARGET_FRAMES,
        n_joints=15,
        n_pose_channels=4,
        n_angular=22,
        n_exercises=n_exercises,
        n_joint_groups=5,
    )
    weights_path = os.path.join(MODEL_DIR, WEIGHTS_FILENAME)
    model.load_weights(weights_path)
    print(f"  loaded {model.count_params():,} params from {os.path.basename(weights_path)}")
    print(f"  exercise lock = '{EXERCISE}' (idx {exercise_idx})")
    print()

    # Process each video
    results: Dict[str, Dict] = {}
    for video_path, label, kind in USER_VIDEOS:
        print(f"[{label}] processing {os.path.basename(video_path)}")
        if not os.path.isfile(video_path):
            print(f"  MISSING: {video_path}")
            continue

        t0 = time.time()
        joints33 = extract_mediapipe_joints(video_path)
        pose, angles = joints_to_inputs(joints33)
        out = sliding_predict(model, pose, angles, exercise_idx)
        elapsed = time.time() - t0

        # Aggregate stats
        q_mean = float(np.mean(out["windows_quality"]))
        q_std = float(np.std(out["windows_quality"]))
        # Action: mean window vote
        action_mean = out["windows_action"].mean(axis=0)
        action_top1 = int(np.argmax(action_mean))
        action_correct = (action_top1 == exercise_idx)
        # Boundary: peak across the clip
        bdy_max = float(out["boundary"].max())
        # Joint-err: per-group max and mean
        je = out["joint_err"]              # (T, 5)
        je_max = je.max(axis=0)             # (5,)
        je_mean = je.mean(axis=0)           # (5,)
        # Rep count proxy: sum of windows / n_windows (rough)
        rep_count_mean = float(out["windows_rep_count"].mean())

        results[label] = {
            "kind":            kind,
            "n_windows":       int(len(out["windows_quality"])),
            "duration_s":      pose.shape[0] / 30.0,  # approximate
            "quality_mean":    q_mean,
            "quality_std":     q_std,
            "action_top1_idx": action_top1,
            "action_top1_name": idx2ex[action_top1],
            "action_top1_conf": float(action_mean[action_top1]),
            "action_squat_conf": float(action_mean[exercise_idx]),
            "action_correct":  action_correct,
            "boundary_peak":   bdy_max,
            "joint_err_max":   je_max.tolist(),
            "joint_err_mean":  je_mean.tolist(),
            "rep_count_mean":  rep_count_mean,
            "elapsed_s":       round(elapsed, 1),
        }

        print(f"  -> quality={q_mean:.3f} (sigma={q_std:.3f})")
        print(f"  -> action top1='{idx2ex[action_top1]}' conf={action_mean[action_top1]:.3f}, "
              f"squat_conf={action_mean[exercise_idx]:.3f}")
        print(f"  -> boundary_peak={bdy_max:.3f}, rep_count_mean={rep_count_mean:.2f}")
        print(f"  -> joint_err_max [knee={je_max[0]:.3f}, hip={je_max[1]:.3f}, "
              f"back={je_max[2]:.3f}, shoulder={je_max[3]:.3f}, elbow={je_max[4]:.3f}]")
        print(f"  -> {elapsed:.1f}s")
        print()

    # ── Summary table ────────────────────────────────────────────────────────
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'Clip':<14s} | {'Q':>6s} | {'Action':<14s} | {'B':>5s} | "
          f"{'knee':>6s} {'hip':>6s} {'back':>6s} {'sho':>6s} {'elb':>6s}")
    print("-" * 78)
    for _, label, _ in USER_VIDEOS:
        r = results.get(label)
        if not r:
            continue
        print(f"{label:<14s} | {r['quality_mean']:>6.3f} | "
              f"{r['action_top1_name']:<14s} | "
              f"{r['boundary_peak']:>5.2f} | "
              f"{r['joint_err_max'][0]:>6.3f} {r['joint_err_max'][1]:>6.3f} "
              f"{r['joint_err_max'][2]:>6.3f} {r['joint_err_max'][3]:>6.3f} "
              f"{r['joint_err_max'][4]:>6.3f}")
    print()

    # ── 4 ship-it sub-criteria ──────────────────────────────────────────────
    good_quality = [results[l]["quality_mean"] for _, l, k in USER_VIDEOS if k == "good" and l in results]
    bad_quality  = [results[l]["quality_mean"] for _, l, k in USER_VIDEOS if k == "bad"  and l in results]

    quality_gap = (np.mean(good_quality) - np.mean(bad_quality)) if good_quality and bad_quality else 0.0

    # Knee/hip in Bad must exceed 0.2; in Good must stay under 0.1
    bad_labels  = [l for _, l, k in USER_VIDEOS if k == "bad"  and l in results]
    good_labels = [l for _, l, k in USER_VIDEOS if k == "good" and l in results]
    knee_hip_bad_max = max(
        (max(results[l]["joint_err_max"][0], results[l]["joint_err_max"][1]) for l in bad_labels),
        default=0.0,
    )
    knee_hip_good_max = max(
        (max(results[l]["joint_err_max"][0], results[l]["joint_err_max"][1]) for l in good_labels),
        default=0.0,
    )

    all_squat = all(r["action_correct"] for r in results.values())
    boundary_ok = all(r["boundary_peak"] > 0.5 for r in results.values())

    print("Ship-it sub-criteria (plan §7):")
    print(f"  1. All 4 classified as 'squat':          "
          f"{'PASS' if all_squat else 'FAIL'}  "
          f"({sum(r['action_correct'] for r in results.values())}/{len(results)})")
    print(f"  2. Boundary peaks > 0.5 on every clip:   "
          f"{'PASS' if boundary_ok else 'FAIL'}  "
          f"(min peak = {min(r['boundary_peak'] for r in results.values()):.3f})")
    print(f"  3. Quality gap >= 0.15  (good - bad):    "
          f"{'PASS' if quality_gap >= 0.15 else 'FAIL'}  "
          f"(gap = {quality_gap:+.3f}, "
          f"mean_good={np.mean(good_quality):.3f}, "
          f"mean_bad={np.mean(bad_quality):.3f})")
    print(f"  4. Knee/hip > 0.2 in Bad, < 0.1 in Good: "
          f"{'PASS' if knee_hip_bad_max > 0.2 and knee_hip_good_max < 0.1 else 'FAIL'}  "
          f"(bad_max={knee_hip_bad_max:.3f}, good_max={knee_hip_good_max:.3f})")
    print()
    overall = (
        all_squat and boundary_ok
        and quality_gap >= 0.15
        and knee_hip_bad_max > 0.2 and knee_hip_good_max < 0.1
    )
    print(f"OVERALL: {'PASS - v5.0 ships' if overall else 'FAIL - iterate'}")
    print()

    # Save the full report
    out_path = os.path.join(MODEL_DIR, "phase6_reality_check.json")
    with open(out_path, "w") as f:
        json.dump({
            "results": results,
            "gates": {
                "all_squat":         all_squat,
                "boundary_ok":       boundary_ok,
                "quality_gap":       float(quality_gap),
                "quality_gap_pass":  bool(quality_gap >= 0.15),
                "knee_hip_bad_max":  float(knee_hip_bad_max),
                "knee_hip_good_max": float(knee_hip_good_max),
                "knee_hip_pass":     bool(knee_hip_bad_max > 0.2 and knee_hip_good_max < 0.1),
                "overall_pass":      bool(overall),
            },
        }, f, indent=2)
    print(f"Report saved to {out_path}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
