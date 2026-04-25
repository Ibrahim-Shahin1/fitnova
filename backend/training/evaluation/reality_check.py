"""
reality_check.py — dual-path evaluation of the MT-TCN model.

Problem we are measuring
------------------------
Our training metrics say exercise_accuracy = 83.1%, but training inputs are
Fit3D MoCap joints3d_25 (near-perfect 3D joints), and real inference inputs
are MediaPipe's 2.5D video-derived landmarks. We have never directly measured
the performance gap between those two distributions.

This script takes the already-extracted camera-60457274 videos for subject s05
and runs each video through two pipelines simultaneously:

    Path A  (training-domain)
        Fit3D joints3d_25 -> canonical 15 -> angles -> MT-TCN
        Uses the rep boundaries from rep_ann.json to cut clean 64-frame windows.

    Path B  (real inference)
        Video -> MediaPipe Pose -> canonical 15 -> angles -> MT-TCN
        Sliding 64-frame window, same cadence as FormSession.

For each of the 27 exercises present we record, per path:
    * predicted exercise label (at the video / first-rep level)
    * mean quality score across windows
    * mean joint-error profile (10-vector)
    * per-feature mean absolute angular difference between the two paths,
      on rep-aligned 64-frame windows (quantifies the raw domain gap)

We emit a JSON report and a few matplotlib figures.

Usage
-----
    python -m backend.training.evaluation.reality_check

        [--dataset-root <path>]  default: C:\\Users\\tsh_x\\Desktop\\FitNova Datasets\\fit3d\\train\\train
        [--camera <id>]           default: 60457274
        [--subject <id>]          default: s05
        [--model-dir <dir>]
        [--out-dir <dir>]         default: backend/models/form_model/reality_check/
        [--every <N>]             MediaPipe window cadence (default 10)
        [--max-exercises <N>]     cap to first N exercises (debugging)

Interpretation of the output
----------------------------
    * If the MediaPipe-path exercise label disagrees with the MoCap-path label
      on many exercises -> the model has overfit to MoCap-specific geometry.
    * If the MediaPipe-path quality distribution is a narrow spike (e.g. all
      0.92) -> the quality head has memorised the synthetic-corruption axis
      and is not sensitive to real motion noise.
    * If per-feature angular MAD is large for the axis-relative features
      (indices 16-21) -> MediaPipe's depth / scale assumptions are the main
      cause of the gap, and we should warm-start on MediaPipe data (R2).

Expected outcome
----------------
Exercise accuracy on path B is noticeably lower than 83 %, and rep boundaries
on path B mis-fire (because the single-rep training labels never taught the
model what a sliding-window boundary looks like — R5 is the fix).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from backend.services.form_analyzer import FormAnalyzer  # noqa: E402
from backend.training.preprocessing.fit3d_loader import (  # noqa: E402
    SUPPORTED_EXERCISES, load_joints, load_rep_boundaries, segment_reps,
)
from backend.training.preprocessing.normalize import (  # noqa: E402
    TARGET_FRAMES, extract_canonical_from_fit3d, normalize_skeleton, resample_sequence,
)
from backend.training.preprocessing.angular_features import (  # noqa: E402
    compute_sequence_angles, ANGULAR_NAMES,
)

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW_SIZE       = TARGET_FRAMES  # 64
EXERCISE_CONF_THRESHOLD = 0.40
JOINT_GROUP_NAMES = [
    "l_elbow", "r_elbow",
    "l_shoulder", "r_shoulder",
    "l_knee", "r_knee",
    "l_hip", "r_hip",
    "trunk", "neck",
]

DEFAULT_DATASET_ROOT = r"C:\Users\tsh_x\Desktop\FitNova Datasets\fit3d\train\train"
DEFAULT_MODEL_DIR    = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "form_model")
)
DEFAULT_OUT_DIR      = os.path.join(DEFAULT_MODEL_DIR, "reality_check")
# Pre-extracted MediaPipe .npy files (shape T×33×4) — see mediapipe_extractor.py
DEFAULT_MP_ROOT      = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "mediapipe_fit3d")
)


# ── Path A: MoCap -> MT-TCN ───────────────────────────────────────────────────

def _predict_mocap_reps(
    analyzer: FormAnalyzer,
    joints25: np.ndarray,
    boundaries: List[int],
) -> Dict:
    """Run MT-TCN on every rep of one Fit3D exercise (MoCap path).

    Returns a dict with per-rep predictions and aggregate stats.
    """
    reps = segment_reps(joints25, boundaries)
    if not reps:
        return {"n_reps": 0}

    ex_labels: List[str] = []
    qualities:   List[float] = []
    joint_errs:  List[List[float]] = []
    rep_counts:  List[float] = []
    ang_cache:   List[np.ndarray] = []

    for rep in reps:
        canon  = extract_canonical_from_fit3d(rep)
        canon  = normalize_skeleton(canon)
        canon  = resample_sequence(canon, WINDOW_SIZE)
        angles = compute_sequence_angles(canon)
        ang_cache.append(angles.copy())
        angles = analyzer._normalizer.transform(angles)   # normalised
        joints_flat = canon.reshape(WINDOW_SIZE, -1)

        inf = analyzer.predict_window(angles, joints_flat)
        ex_labels.append(inf["exercise"])
        qualities.append(inf["quality"])
        joint_errs.append(list(np.mean(inf["joint_errors"], axis=0)))
        rep_counts.append(inf["rep_count"])

    ex_mode = max(set(ex_labels), key=ex_labels.count) if ex_labels else "unknown"
    return {
        "n_reps":            len(reps),
        "per_rep_exercise":  ex_labels,
        "dominant_exercise": ex_mode,
        "mean_quality":      float(np.mean(qualities)),
        "std_quality":       float(np.std(qualities)),
        "quality_per_rep":   [float(q) for q in qualities],
        "mean_joint_errors": [float(x) for x in np.mean(joint_errs, axis=0)],
        "mean_rep_count":    float(np.mean(rep_counts)),
        "_angle_cache":      ang_cache,   # kept for MAD computation (stripped later)
    }


# ── Path B2: MediaPipe .npy -> MT-TCN (no JPEG round-trip) ───────────────────

def _predict_mediapipe_npy_reps(
    analyzer: FormAnalyzer,
    npy_path: str,
    boundaries: List[int],
) -> Dict:
    """
    Run MT-TCN on every rep of one exercise using the pre-extracted MediaPipe .npy file.

    This path is equivalent to the training pipeline (same canonical extraction,
    same normalisation) but skips the JPEG encode/decode step that distorts
    landmarks in the live-video path.

    Returns the same dict structure as `_predict_mocap_reps`.
    """
    from backend.training.preprocessing.normalize import (
        extract_canonical_from_mediapipe, normalize_skeleton,
    )
    from backend.training.preprocessing.mediapipe_loader import _interpolate_nans_3d

    if not os.path.exists(npy_path):
        return {"n_reps": 0, "error": f"npy not found: {npy_path}"}

    arr = np.load(npy_path)          # (T, 33, 4) — x,y,z,visibility
    if arr.ndim != 3 or arr.shape[1] != 33:
        return {"n_reps": 0, "error": "unexpected npy shape"}

    xyz = arr[:, :, :3].astype(np.float32)   # (T, 33, 3) world landmarks
    # Interpolate NaN frames (same as mediapipe_loader does during training)
    xyz = _interpolate_nans_3d(xyz)

    if not boundaries:
        return {"n_reps": 0}

    # Segment into reps (same logic as fit3d_loader.segment_reps)
    reps_xyz = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e > s:
            reps_xyz.append(xyz[s:e])

    if not reps_xyz:
        return {"n_reps": 0}

    ex_labels: List[str]        = []
    qualities: List[float]      = []
    joint_errs: List[List[float]] = []
    ang_cache: List[np.ndarray] = []

    for rep_xyz in reps_xyz:
        from backend.training.preprocessing.normalize import resample_sequence
        canon  = extract_canonical_from_mediapipe(rep_xyz)     # (T, 15, 3)
        canon  = normalize_skeleton(canon)                       # (T, 15, 3)
        canon  = resample_sequence(canon, WINDOW_SIZE)          # (64, 15, 3)
        angles = compute_sequence_angles(canon)                  # (64, 22)
        ang_cache.append(angles.copy())
        angles = analyzer._normalizer.transform(angles)          # normalised
        joints_flat = canon.reshape(WINDOW_SIZE, -1)             # (64, 45)

        inf = analyzer.predict_window(angles, joints_flat)
        ex_labels.append(inf["exercise"])
        qualities.append(inf["quality"])
        joint_errs.append(list(np.mean(inf["joint_errors"], axis=0)))

    ex_mode = max(set(ex_labels), key=ex_labels.count) if ex_labels else "unknown"
    return {
        "n_reps":            len(reps_xyz),
        "per_rep_exercise":  ex_labels,
        "dominant_exercise": ex_mode,
        "mean_quality":      float(np.mean(qualities)) if qualities else 0.0,
        "std_quality":       float(np.std(qualities)) if qualities else 0.0,
        "quality_per_rep":   [float(q) for q in qualities],
        "mean_joint_errors": [float(x) for x in np.mean(joint_errs, axis=0)] if joint_errs else [],
        "_angle_cache":      ang_cache,
    }


# ── Path B: MediaPipe video -> MT-TCN ─────────────────────────────────────────

def _run_video_pipeline(
    analyzer: FormAnalyzer,
    video_path: str,
    every_n: int,
    boundary_threshold: float,
    min_rep_gap_frames: int,
    max_frames: Optional[int] = None,
) -> Dict:
    """Sliding-window MT-TCN inference on a video (MediaPipe path).

    Returns the same keys as the MoCap path (where meaningful) plus
    per-window aggregates so we can plot a quality distribution.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"n_windows": 0, "error": f"cv2 could not open {video_path}"}

    from collections import deque
    ang_buf = deque(maxlen=WINDOW_SIZE)
    jnt_buf = deque(maxlen=WINDOW_SIZE)

    qualities:   List[float] = []
    joint_errs:  List[List[float]] = []
    ex_labels:   List[str]         = []
    confidences: List[float]       = []
    boundary_events: int = 0
    angles_per_window: List[np.ndarray] = []
    recent_exercises: List[str] = []

    last_boundary_at = -min_rep_gap_frames
    prev_boundary_prob = 0.0
    frame_idx = 0
    no_pose_n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            frame_idx += 1
            continue
        r = analyzer.process_frame(jpeg.tobytes())

        if r["status"] in ("ok", "pose_interpolated"):
            ang_buf.append(np.asarray(r["angles_norm"], dtype=np.float32))
            # Use skeleton-normalised world coords (matches training X_joints format).
            # Fall back to image landmarks if joints_norm absent (legacy responses).
            jnt_key = "joints_norm" if "joints_norm" in r else "landmarks"
            jnt_buf.append(np.asarray(r[jnt_key], dtype=np.float32).flatten())
        else:
            no_pose_n += 1

        if len(ang_buf) == WINDOW_SIZE and frame_idx % every_n == 0:
            ang_win = np.stack(list(ang_buf), axis=0)
            jnt_win = np.stack(list(jnt_buf), axis=0)
            inf = analyzer.predict_window(ang_win, jnt_win)

            qualities.append(inf["quality"])
            joint_errs.append(list(np.mean(inf["joint_errors"], axis=0)))
            ex_labels.append(inf["exercise"])
            confidences.append(inf["exercise_confidence"])
            angles_per_window.append(ang_win.copy())

            # Exercise gating: boundary only counts if exercise prediction is stable and confident
            recent_exercises.append(inf["exercise"])
            if len(recent_exercises) > 5:
                recent_exercises.pop(0)
            if len(recent_exercises) == 5:
                mode_label = max(set(recent_exercises), key=recent_exercises.count)
                mode_count = recent_exercises.count(mode_label)
                exercise_stable = (mode_count >= 4 and mode_label == inf["exercise"])
            else:
                exercise_stable = False
            exercise_confident = float(inf.get("exercise_confidence", 1.0)) >= EXERCISE_CONF_THRESHOLD

            max_prob = float(max(inf["boundary"]))  # peak anywhere in window
            max_idx = int(np.argmax(np.asarray(inf["boundary"])))
            if (max_prob > boundary_threshold and
                    prev_boundary_prob <= boundary_threshold and
                    frame_idx - last_boundary_at > min_rep_gap_frames and
                    exercise_stable and exercise_confident):
                boundary_events += 1
                # Anchor the boundary timestamp to where the peak actually occurred in absolute frames
                last_boundary_at = frame_idx - (WINDOW_SIZE - 1 - max_idx)
            prev_boundary_prob = max_prob

        frame_idx += 1

    cap.release()

    if not qualities:
        return {
            "n_windows":  0,
            "frames":     frame_idx,
            "frames_no_pose": no_pose_n,
            "error":      "no_windows_inferred",
        }

    ex_mode = max(set(ex_labels), key=ex_labels.count)
    return {
        "n_windows":         len(qualities),
        "frames":            frame_idx,
        "frames_no_pose":    no_pose_n,
        "per_window_exercise": ex_labels,
        "dominant_exercise": ex_mode,
        "mean_confidence":   float(np.mean(confidences)),
        "mean_quality":      float(np.mean(qualities)),
        "std_quality":       float(np.std(qualities)),
        "quality_per_window":[float(q) for q in qualities],
        "mean_joint_errors": [float(x) for x in np.mean(joint_errs, axis=0)],
        "boundary_events":   boundary_events,
        "_angle_cache":      angles_per_window,
    }


# ── MAD computation ───────────────────────────────────────────────────────────

def _compute_angular_mad(
    mocap_angles: List[np.ndarray],
    mediapipe_angles: List[np.ndarray],
) -> Optional[Dict[str, float]]:
    """Mean absolute difference between MoCap and MediaPipe angles, per feature.

    Sequences are aligned by resampling both to a common length, then averaging
    absolute differences over all reps and windows. Designed to flag which
    of the 22 angular features are most affected by the MediaPipe domain shift.

    Note: both inputs are NORMALISED (post AngleNormalizer) so the MAD is
    dimensionless (in standardised units).
    """
    if not mocap_angles or not mediapipe_angles:
        return None

    # Resample to same length (64 frames; they already are)
    arr_a = np.stack([a for a in mocap_angles],     axis=0)  # (N_reps,    64, 22)
    arr_b = np.stack([a for a in mediapipe_angles], axis=0)  # (N_windows, 64, 22)

    # Collapse temporal axis then compare distributions (robust to slight
    # phase shifts between MoCap reps and video windows).
    mean_a = arr_a.mean(axis=1)   # (N_reps,    22)
    mean_b = arr_b.mean(axis=1)   # (N_windows, 22)

    # Broadcast-difference: every rep vs every window, then average.
    diffs = np.abs(mean_a[:, None, :] - mean_b[None, :, :]).mean(axis=(0, 1))  # (22,)
    return {name: float(d) for name, d in zip(ANGULAR_NAMES, diffs)}


# ── Plotting ──────────────────────────────────────────────────────────────────

def _save_plots(report: Dict, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[reality_check] matplotlib not installed; skipping plots.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # 1. Quality score histograms (one per exercise, stacked figure)
    exercises = sorted(report["per_exercise"].keys())
    n = len(exercises)
    if n == 0:
        return

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.2))
    axes = np.atleast_2d(axes)
    for i, ex in enumerate(exercises):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        rec = report["per_exercise"][ex]
        a  = rec.get("mocap", {}).get("quality_per_rep", [])
        b  = rec.get("mediapipe", {}).get("quality_per_window", [])
        b2 = rec.get("mediapipe_npy", {}).get("quality_per_rep", [])
        if a:
            ax.hist(a,  bins=np.linspace(0, 1, 11), alpha=0.55, label="MoCap",    color="#2b8cbe")
        if b:
            ax.hist(b,  bins=np.linspace(0, 1, 11), alpha=0.40, label="MP-video", color="#e34a33")
        if b2:
            ax.hist(b2, bins=np.linspace(0, 1, 11), alpha=0.55, label="MP-npy",   color="#31a354")
        ax.set_title(ex, fontsize=8)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6)

    # Hide unused subplots
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Per-exercise quality distribution — MoCap vs MediaPipe", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, "quality_histograms.png"), dpi=120)
    plt.close(fig)

    # 2. Bar chart of per-feature angular MAD (averaged across exercises)
    mads = [
        rec["angle_mad"] for rec in report["per_exercise"].values()
        if rec.get("angle_mad")
    ]
    if mads:
        names = list(mads[0].keys())
        mean_mad = np.mean([[d[k] for k in names] for d in mads], axis=0)
        fig, ax = plt.subplots(figsize=(10, 3.2))
        idx = np.arange(len(names))
        ax.bar(idx, mean_mad, color="#6a51a3")
        ax.set_xticks(idx)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Mean |MoCap - MediaPipe| (std units)")
        ax.set_title("Domain gap per angular feature (lower is better)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "angular_mad.png"), dpi=120)
        plt.close(fig)

    # 3. Exercise classification agreement heatmap
    ok = []
    labels = []
    for ex, rec in report["per_exercise"].items():
        mp_ex  = rec.get("mediapipe",     {}).get("dominant_exercise", "—")
        npy_ex = rec.get("mediapipe_npy", {}).get("dominant_exercise", "—")
        mc_ex  = rec.get("mocap",         {}).get("dominant_exercise", "—")
        # Green = npy matches, orange = video matches, red = both wrong
        ok.append(1 if npy_ex == ex else (0.5 if mp_ex == ex else 0))
        labels.append(f"{ex}\n npy:{npy_ex}  vid:{mp_ex}")
    fig, ax = plt.subplots(figsize=(10, max(3, 0.3 * len(labels))))
    y = np.arange(len(labels))
    colours = ["#2ca25f" if v == 1 else "#fd8d3c" if v == 0.5 else "#d7301f" for v in ok]
    ax.barh(y, [1] * len(labels), color=colours, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.set_title("Exercise accuracy: green=npy✓, orange=video✓only, red=both wrong")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exercise_agreement.png"), dpi=120)
    plt.close(fig)


# ── Top-level runner ──────────────────────────────────────────────────────────

def run_reality_check(
    dataset_root: str = DEFAULT_DATASET_ROOT,
    camera: str = "60457274",
    subject: str = "s05",
    model_dir: str = DEFAULT_MODEL_DIR,
    out_dir: str = DEFAULT_OUT_DIR,
    every_n: int = 10,
    boundary_threshold: float = 0.55,
    min_rep_gap_frames: int = 20,
    max_exercises: Optional[int] = None,
    max_frames_per_video: Optional[int] = None,
    mediapipe_root: str = DEFAULT_MP_ROOT,
    verbose: bool = True,
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    analyzer = FormAnalyzer(model_dir=model_dir)
    if not analyzer.model_ready:
        raise RuntimeError("MT-TCN not ready; check model_dir has weights + stats + labels.")

    subject_dir   = os.path.join(dataset_root, subject)
    video_dir     = os.path.join(subject_dir, "videos", camera)
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"Video dir not found: {video_dir}")

    rep_boundaries = load_rep_boundaries(subject_dir)

    # Only evaluate the 27 exercises we actually trained on.
    candidate = sorted(set(SUPPORTED_EXERCISES).intersection(rep_boundaries.keys()))
    if max_exercises:
        candidate = candidate[:max_exercises]

    report: Dict = {
        "model_dir":    model_dir,
        "subject":      subject,
        "camera":       camera,
        "dataset_root": dataset_root,
        "per_exercise": {},
    }

    agreement_rows: List[Tuple[str, str, str, bool]] = []
    t0 = time.time()

    for i, ex in enumerate(candidate):
        if verbose:
            print(f"\n[{i+1}/{len(candidate)}] {ex}")

        video_path = os.path.join(video_dir, f"{ex}.mp4")
        if not os.path.isfile(video_path):
            if verbose:
                print(f"  skip (missing video): {video_path}")
            continue

        # ── Path A: MoCap
        joints = load_joints(subject_dir, ex)
        if joints is None:
            if verbose:
                print("  skip (missing joints3d_25)")
            continue
        mocap = _predict_mocap_reps(analyzer, joints, rep_boundaries[ex])
        if not mocap.get("n_reps"):
            if verbose:
                print("  skip (no reps after segmentation)")
            continue
        if verbose:
            print(f"  MoCap    : {mocap['n_reps']} reps  -> {mocap['dominant_exercise']}  "
                  f"q={mocap['mean_quality']:.2f}")

        # ── Path B: MediaPipe video (JPEG path)
        # Reset VIDEO-mode timestamp + forward-fill cache for each new video
        analyzer.reset_video_state(fps=30.0)
        mpipe = _run_video_pipeline(
            analyzer, video_path, every_n,
            boundary_threshold, min_rep_gap_frames,
            max_frames=max_frames_per_video,
        )
        if mpipe.get("n_windows"):
            if verbose:
                print(f"  MediaPipe: {mpipe['n_windows']} windows  "
                      f"-> {mpipe['dominant_exercise']}  "
                      f"q={mpipe['mean_quality']:.2f}  "
                      f"rep_events={mpipe['boundary_events']}")
        else:
            if verbose:
                print(f"  MediaPipe: FAIL  ({mpipe.get('error')})")

        # ── Path B2: MediaPipe .npy (no JPEG — in-distribution with training)
        npy_path = os.path.join(mediapipe_root, subject, f"{ex}.npy")
        # boundaries may contain indices beyond the .npy length; _predict handles clamping
        boundaries_for_npy = rep_boundaries[ex]
        analyzer.reset_video_state(fps=30.0)   # clean state before .npy path
        mpipe_npy = _predict_mediapipe_npy_reps(analyzer, npy_path, boundaries_for_npy)
        if mpipe_npy.get("n_reps"):
            if verbose:
                print(f"  MP-npy  : {mpipe_npy['n_reps']} reps  "
                      f"-> {mpipe_npy['dominant_exercise']}  "
                      f"q={mpipe_npy['mean_quality']:.2f}")
        else:
            if verbose:
                print(f"  MP-npy  : SKIP  ({mpipe_npy.get('error', 'no reps')})")

        # ── MAD (MoCap vs MediaPipe-video angles)
        mad = None
        if mpipe.get("n_windows"):
            mad = _compute_angular_mad(mocap["_angle_cache"], mpipe["_angle_cache"])

        report["per_exercise"][ex] = {
            "mocap":         {k: v for k, v in mocap.items() if not k.startswith("_")},
            "mediapipe":     {k: v for k, v in mpipe.items() if not k.startswith("_")},
            "mediapipe_npy": {k: v for k, v in mpipe_npy.items() if not k.startswith("_")},
            "angle_mad":     mad,
        }

        # B / B2 agreement with ground-truth label
        b_match  = mpipe.get("dominant_exercise") == ex if mpipe.get("n_windows") else None
        b2_match = mpipe_npy.get("dominant_exercise") == ex if mpipe_npy.get("n_reps") else None
        agreement_rows.append((
            ex,
            mocap["dominant_exercise"],
            mpipe.get("dominant_exercise", "—"),
            mpipe_npy.get("dominant_exercise", "—"),
            b_match,
            b2_match,
        ))

    # ── Aggregate ────────────────────────────────────────────────────────────
    if agreement_rows:
        n = len(agreement_rows)
        # r = (ex, mocap_pred, mpipe_pred, mpipe_npy_pred, b_match, b2_match)
        b_rows  = [r for r in agreement_rows if r[4] is not None]
        b2_rows = [r for r in agreement_rows if r[5] is not None]

        npy_quals = [
            report["per_exercise"][r[0]]["mediapipe_npy"].get("mean_quality", 0.0)
            for r in b2_rows
        ]

        report["aggregate"] = {
            "n_exercises_evaluated": n,
            "exercise_match_mocap_to_gt":
                float(sum(1 for r in agreement_rows if r[1] == r[0])) / n,
            "exercise_match_mediapipe_video_to_gt":
                float(sum(r[4] for r in b_rows)) / len(b_rows) if b_rows else None,
            "exercise_match_mediapipe_npy_to_gt":
                float(sum(r[5] for r in b2_rows)) / len(b2_rows) if b2_rows else None,
            "mean_quality_mocap":
                float(np.mean([report["per_exercise"][r[0]]["mocap"]["mean_quality"]
                               for r in agreement_rows])),
            "mean_quality_mediapipe_video":
                float(np.mean([report["per_exercise"][r[0]]["mediapipe"]["mean_quality"]
                               for r in b_rows])) if b_rows else None,
            "mean_quality_mediapipe_npy":
                float(np.mean(npy_quals)) if npy_quals else None,
            "wall_clock_seconds": round(time.time() - t0, 1),
        }

    # ── Emit ─────────────────────────────────────────────────────────────────
    out_json = os.path.join(out_dir, "report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    if verbose:
        print(f"\n[reality_check] report written -> {out_json}")

    _save_plots(report, out_dir)

    if verbose and "aggregate" in report:
        agg = report["aggregate"]
        def _pct(v):
            return f"{v*100:.1f}%" if v is not None else "n/a"
        def _q(v):
            return f"{v:.3f}" if v is not None else "n/a"
        print("\n== aggregate ==")
        print(f"  MoCap         exercise accuracy : {_pct(agg['exercise_match_mocap_to_gt'])}")
        print(f"  MediaPipe-vid exercise accuracy : {_pct(agg['exercise_match_mediapipe_video_to_gt'])}"
              f"  (JPEG path — expect degraded)")
        print(f"  MediaPipe-npy exercise accuracy : {_pct(agg['exercise_match_mediapipe_npy_to_gt'])}"
              f"  (in-distribution, target >=75%)")
        print(f"  mean quality (MoCap)            : {_q(agg['mean_quality_mocap'])}")
        print(f"  mean quality (MP-video)         : {_q(agg['mean_quality_mediapipe_video'])}")
        print(f"  mean quality (MP-npy)           : {_q(agg['mean_quality_mediapipe_npy'])}")
        print(f"  wall-clock                      : {agg['wall_clock_seconds']}s")

    return report


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    p.add_argument("--camera",       default="60457274")
    p.add_argument("--subject",      default="s05")
    p.add_argument("--model-dir",    default=DEFAULT_MODEL_DIR)
    p.add_argument("--out-dir",      default=DEFAULT_OUT_DIR)
    p.add_argument("--mediapipe-root", default=DEFAULT_MP_ROOT,
                   help="Root dir for pre-extracted MediaPipe .npy files "
                        "(default: backend/data/mediapipe_fit3d)")
    p.add_argument("--every",        type=int, default=10)
    p.add_argument("--max-exercises", type=int, default=None,
                   help="Cap to first N exercises (debugging)")
    p.add_argument("--max-frames-per-video", type=int, default=None,
                   help="Cap MediaPipe processing to N frames per video (debugging)")
    args = p.parse_args()

    run_reality_check(
        dataset_root=args.dataset_root,
        camera=args.camera,
        subject=args.subject,
        model_dir=args.model_dir,
        out_dir=args.out_dir,
        mediapipe_root=args.mediapipe_root,
        every_n=args.every,
        max_exercises=args.max_exercises,
        max_frames_per_video=args.max_frames_per_video,
    )


if __name__ == "__main__":
    _cli()
