"""
visualize_pipeline.py — end-to-end inference visualisation.

Takes a video file, runs the full real-time pipeline
(MediaPipe -> canonical 15 joints -> angular features -> MT-TCN),
and writes an annotated mp4 next to the input video.

The output overlays:
  * 14-bone skeleton (only 14 of the 15 canonical joints are visible:
    pelvis / neck / spine_mid are interpolated — we draw the 11 "real" ones
    plus the 3 midpoints for completeness).
  * Current predicted exercise + confidence
  * Per-frame quality score (colour bar, green=high, red=low)
  * Top-3 joint-error groups for the current frame
  * Running rep count (from boundary head)
  * Frame index / timestamp

Why it matters
--------------
Our training metrics are computed on synthetic 64-frame single-rep windows.
The real inference path is a sliding window over a MediaPipe video. This tool
is how we *see* whether those two worlds agree. It is also the deliverable
for the R6 user-recorded validation: the annotated .mp4 is shown in the
defence slides.

Usage
-----
    python -m backend.training.evaluation.visualize_pipeline <video_path> \
        [--model-dir <dir>] [--out <out.mp4>] [--every <N>] [--max-frames <N>]

    --every N         Run MT-TCN every N frames (default 5). Smaller = smoother,
                      slower. Larger = faster but more stale predictions.
    --max-frames N    Stop after N video frames (useful for quick tests).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

# Local imports (package-relative)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from backend.services.form_analyzer import FormAnalyzer  # noqa: E402
from backend.training.preprocessing.joint_mapping import (  # noqa: E402
    SKELETON_CONNECTIONS,
)
from backend.config.exercises import (  # noqa: E402
    get_metadata,
    relevant_joint_indices,
)

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW_SIZE       = 64
EXERCISE_CONF_THRESHOLD = 0.40
# Short labels match the 10-index order of the MT-TCN joint-error head.
JOINT_GROUP_NAMES = [
    "L Elbow", "R Elbow",
    "L Shoulder", "R Shoulder",
    "L Knee", "R Knee",
    "L Hip", "R Hip",
    "Trunk", "Neck",
]

DEFAULT_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "form_model")
)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _quality_colour(q: float) -> Tuple[int, int, int]:
    """Green (q=1) -> yellow (q=0.5) -> red (q=0)."""
    q = float(np.clip(q, 0.0, 1.0))
    if q >= 0.5:
        t = (q - 0.5) * 2.0         # 0..1  yellow->green
        b, g, r = 0, 255, int(255 * (1.0 - t))
    else:
        t = q * 2.0                 # 0..1  red->yellow
        b, g, r = 0, int(255 * t), 255
    return (b, g, r)


def _draw_skeleton(img: np.ndarray, landmarks_15: List[List[float]]) -> None:
    """Draw 14 bones + 15 joints onto `img` in-place.

    `landmarks_15` is the analyzer's image-normalised (x,y,z) list where x,y in [0,1].
    """
    if landmarks_15 is None:
        return
    h, w = img.shape[:2]
    pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks_15]

    # Bones
    for i, j in SKELETON_CONNECTIONS:
        if 0 <= i < len(pts) and 0 <= j < len(pts):
            cv2.line(img, pts[i], pts[j], (255, 255, 255), 2, cv2.LINE_AA)

    # Joints
    for p in pts:
        cv2.circle(img, p, 4, (0, 200, 255), -1, cv2.LINE_AA)


def _draw_hud(
    img: np.ndarray,
    exercise: str,
    confidence: float,
    quality: float,
    rep_count: int,
    shown_errors: List[Tuple[str, float]],
    frame_idx: int,
    fps: float,
    locked: bool = False,
) -> None:
    """Draw the heads-up display overlay onto `img` in-place.

    When ``locked`` is true (user pre-selected the exercise), we never show
    a classifier guess — the exercise name is displayed with a "(selected)"
    tag and confidence is hidden, because the user is authoritative.
    """
    h, w = img.shape[:2]

    # Translucent panel, top-left. Height scales with number of joint rows.
    panel_h = 168 + 20 * max(1, len(shown_errors[:4]))
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (400, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, img, 0.42, 0, img)

    def _t(txt, y, colour=(255, 255, 255), scale=0.65, thickness=2):
        cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, colour, thickness, cv2.LINE_AA)

    ex_display = exercise.replace('_', ' ').title()
    if locked:
        _t(f"Exercise: {ex_display}", 28, (120, 255, 180))
        _t("(selected by user)", 52, (150, 200, 170), 0.48, 1)
    else:
        _t(f"Exercise: {ex_display}", 28)
        _t(f"Confidence: {confidence*100:.0f}%", 54, (200, 200, 200), 0.55, 1)
    _t(f"Reps: {rep_count}", 84)

    # Quality bar
    q_col = _quality_colour(quality)
    _t(f"Quality: {quality*100:.0f}%", 114, q_col)
    bar_x, bar_y, bar_w, bar_h = 12, 124, 260, 14
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (80, 80, 80), -1)
    fill = int(bar_w * float(np.clip(quality, 0, 1)))
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h),
                  q_col, -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (220, 220, 220), 1)

    # Relevant joint issues (already filtered + sorted by caller)
    err_y = 158
    label = "Relevant joint issues:" if locked else "Top issues:"
    _t(label, err_y, (200, 200, 200), 0.55, 1)
    for i, (name, prob) in enumerate(shown_errors[:4]):
        colour = (
            (0, 0, 255)     if prob > 0.5 else
            (0, 180, 255)   if prob > 0.3 else
            (180, 180, 180)
        )
        _t(f"- {name}: {prob*100:.0f}%", err_y + 22 + 20 * i, colour, 0.5, 1)

    # Bottom-right: frame idx / fps
    ts = f"f={frame_idx}  {fps:.1f} fps"
    (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(img, ts, (w - tw - 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def visualise(
    video_path: str,
    model_dir: str = DEFAULT_MODEL_DIR,
    out_path: Optional[str] = None,
    every_n: int = 5,
    max_frames: Optional[int] = None,
    boundary_threshold: float = 0.55,
    min_rep_gap_frames: int = 20,
    verbose: bool = True,
    selected_exercise: Optional[str] = None,
) -> dict:
    """Run the end-to-end pipeline on a video and write an annotated mp4.

    Returns a stats dict summarising the run.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {video_path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if out_path is None:
        base, _ = os.path.splitext(video_path)
        out_path = base + "_annotated.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_in, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2 could not open writer for {out_path}")

    if verbose:
        print(f"[visualise] input :  {video_path}  ({w}x{h} @ {fps_in:.1f}fps)")
        print(f"[visualise] output:  {out_path}")
        print(f"[visualise] model :  {model_dir}")

    analyzer = FormAnalyzer(model_dir=model_dir)
    if not analyzer.model_ready:
        raise RuntimeError(
            "FormAnalyzer could not load MT-TCN weights. "
            "Train the model first or pass --model-dir."
        )

    # Resolve locked-exercise metadata once up-front.
    locked_display: Optional[str] = None
    relevant_idx: List[int] = list(range(10))  # default: show all
    if selected_exercise:
        try:
            meta = get_metadata(selected_exercise)
            locked_display = meta.get("display_name", selected_exercise)
            relevant_idx = relevant_joint_indices(selected_exercise)
            if verbose:
                names = [JOINT_GROUP_NAMES[i] for i in relevant_idx]
                print(f"[visualise] locked exercise: {locked_display} "
                      f"(classifier head ignored)")
                print(f"[visualise] relevant joints: {names}")
        except KeyError:
            print(f"[warn] unknown exercise '{selected_exercise}' — "
                  f"falling back to classifier label.")

    # Buffers + inference state
    ang_buf: Deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
    jnt_buf: Deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)

    latest = {
        "exercise":   locked_display or "detecting...",
        "confidence": 1.0 if locked_display else 0.0,
        "quality":    0.5,
        "joint_errors_frame": [0.0] * 10,
    }
    rep_count = 0
    last_boundary_at = -min_rep_gap_frames
    prev_boundary_prob = 0.0
    recent_exercises: List[str] = []
    quality_history: List[float] = []
    exercise_history: List[str]  = []

    frame_idx = 0
    t_start = time.time()
    no_pose_n = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        # Encode -> JPEG for analyzer.process_frame (reuses the production path)
        ok, jpeg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            frame_idx += 1
            continue
        result = analyzer.process_frame(jpeg.tobytes())

        if result["status"] in ("ok", "pose_interpolated"):
            ang_buf.append(np.asarray(result["angles_norm"], dtype=np.float32))
            # Use skeleton-normalised world coords (matches training X_joints format).
            jnt_key = "joints_norm" if "joints_norm" in result else "landmarks"
            jnt_buf.append(np.asarray(result[jnt_key], dtype=np.float32).flatten())
        else:
            no_pose_n += 1

        # Run MT-TCN every `every_n` frames once the window is full
        if len(ang_buf) == WINDOW_SIZE and frame_idx % every_n == 0:
            ang_win = np.stack(list(ang_buf), axis=0)
            jnt_win = np.stack(list(jnt_buf), axis=0)
            inf = analyzer.predict_window(ang_win, jnt_win)

            # The classifier head is only consulted when the user hasn't told us
            # what they're doing. Otherwise we keep the locked display name and
            # treat the exercise as "always stable, always confident" — the user
            # is authoritative.
            if locked_display is not None:
                latest["exercise"]   = locked_display
                latest["confidence"] = 1.0
                exercise_stable     = True
                exercise_confident  = True
            else:
                latest["exercise"]   = inf["exercise"]
                latest["confidence"] = inf["exercise_confidence"]
                recent_exercises.append(inf["exercise"])
                if len(recent_exercises) > 5:
                    recent_exercises.pop(0)
                if len(recent_exercises) == 5:
                    mode_label = max(set(recent_exercises), key=recent_exercises.count)
                    mode_count = recent_exercises.count(mode_label)
                    exercise_stable = (mode_count >= 4 and mode_label == inf["exercise"])
                else:
                    exercise_stable = False
                exercise_confident = (
                    float(inf.get("exercise_confidence", 1.0)) >= EXERCISE_CONF_THRESHOLD
                )

            latest["quality"]    = inf["quality"]
            mid_joint_err = inf["joint_errors"][WINDOW_SIZE // 2]
            latest["joint_errors_frame"] = mid_joint_err

            # Rep boundary detection — rising edge above threshold
            max_prob = float(max(inf["boundary"]))  # peak anywhere in window
            max_idx = int(np.argmax(np.asarray(inf["boundary"])))
            if (max_prob > boundary_threshold and
                    prev_boundary_prob <= boundary_threshold and
                    frame_idx - last_boundary_at > min_rep_gap_frames and
                    exercise_stable and exercise_confident):
                rep_count += 1
                # Anchor the boundary timestamp to where the peak actually occurred in absolute frames
                last_boundary_at = frame_idx - (WINDOW_SIZE - 1 - max_idx)
            prev_boundary_prob = max_prob

        # Build the joint-error list. When locked, we only surface the joints
        # that matter for the selected exercise (e.g. for squats: knees, hips,
        # back). When unlocked, we show the top-3 raw predictions.
        all_errs = list(zip(JOINT_GROUP_NAMES, latest["joint_errors_frame"]))
        if locked_display is not None:
            shown_errors = [all_errs[i] for i in relevant_idx]
            shown_errors.sort(key=lambda x: -x[1])
        else:
            shown_errors = sorted(all_errs, key=lambda x: -x[1])[:3]

        _draw_skeleton(frame_bgr, result.get("landmarks"))
        _draw_hud(
            frame_bgr,
            exercise=latest["exercise"],
            confidence=latest["confidence"],
            quality=latest["quality"],
            rep_count=rep_count,
            shown_errors=shown_errors,
            frame_idx=frame_idx,
            fps=fps_in,
            locked=(locked_display is not None),
        )

        writer.write(frame_bgr)
        quality_history.append(latest["quality"])
        exercise_history.append(latest["exercise"])

        frame_idx += 1
        if verbose and frame_idx % 50 == 0:
            elapsed = time.time() - t_start
            print(f"  frame {frame_idx}  "
                  f"ex={latest['exercise']:<32s}  "
                  f"q={latest['quality']:.2f}  reps={rep_count}  "
                  f"({frame_idx / max(elapsed, 1e-6):.1f} fps)")

    cap.release()
    writer.release()

    total_s = time.time() - t_start
    stats = {
        "video":                video_path,
        "annotated_out":        out_path,
        "frames_processed":     frame_idx,
        "frames_no_pose":       no_pose_n,
        "rep_count":            rep_count,
        "mean_quality":         float(np.mean(quality_history)) if quality_history else None,
        "selected_exercise":    selected_exercise,
        "relevant_joints":      [JOINT_GROUP_NAMES[i] for i in relevant_idx] if locked_display else None,
        "dominant_exercise":    (
            max(set(exercise_history), key=exercise_history.count)
            if exercise_history else None
        ),
        "proc_fps":             frame_idx / max(total_s, 1e-6),
    }

    if verbose:
        print("[visualise] done.")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    return stats


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("video", help="Path to input video (.mp4)")
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                   help=f"Directory with MT-TCN weights (default: {DEFAULT_MODEL_DIR})")
    p.add_argument("--out", default=None,
                   help="Output annotated mp4 (default: <video>_annotated.mp4)")
    p.add_argument("--every", type=int, default=5,
                   help="Run MT-TCN every N frames (default 5)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after N video frames")
    p.add_argument("--selected-exercise", default=None,
                   help="User-selected exercise name (e.g. 'squat'). "
                        "When set, the classifier head is IGNORED for the "
                        "overlay label and rep-gating, and joint-error display "
                        "is masked to that exercise's key_errors_detected.")
    args = p.parse_args()

    visualise(
        video_path=args.video,
        model_dir=args.model_dir,
        out_path=args.out,
        every_n=args.every,
        max_frames=args.max_frames,
        selected_exercise=args.selected_exercise,
    )


if __name__ == "__main__":
    _cli()
