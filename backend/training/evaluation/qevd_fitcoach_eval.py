"""D8 OOD evaluation — v6 model on FIT-COACH long-range videos.

Master plan section II.8 thresholds:
  * Quality Spearman ρ ≥ 0.3 (degradation expected on OOD)
  * Rep-count MAE ≤ 2.0

Pipeline:
  1. Load v6 model + exercise map
  2. For each long-range video:
       - Extract pose features (full 3.5 min)
       - Slide 64-frame windows, stride 8
       - Run model on each window
  3. Compare per-30s-segment predictions to ground-truth from
     feedbacks_long_range.json
  4. Compute Spearman correlation + rep-count MAE
  5. Write D8_ood_eval.json

Usage:
    python -m backend.training.evaluation.qevd_fitcoach_eval \\
        --model-dir backend/models/form_model_v6 \\
        --videos-dir backend/data/qevd_raw/QEVD-FIT-COACH-Benchmark/long_range_videos \\
        --feedbacks backend/data/qevd_raw/QEVD-FIT-COACH-Benchmark/feedbacks_long_range.json \\
        --report    backend/data/qevd_phase_reports/D8_ood_eval.json

Notes
-----
- Designed to be re-runnable; if the user only has FIT-COACH train videos
  (not benchmark), passing those still works — the gate just measures
  in-distribution quality, not OOD.
- 74 videos × 3.5 min × ~30 fps ≈ 466K frames. With stride-8 windows of
  size 64, that's ~58K window inferences. On L4 GPU expect ~10-20 min.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Master-plan thresholds
# ─────────────────────────────────────────────────────────────────────────────


QUALITY_SPEARMAN_MIN = 0.30
REP_COUNT_MAE_MAX    = 2.00
SEGMENT_LENGTH_S     = 30.0     # paper section 3.1: long-range cuts every 30 s


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth derivation from feedbacks_long_range.json
# ─────────────────────────────────────────────────────────────────────────────


def _segment_index(t: float, segment_length_s: float) -> int:
    return int(t // segment_length_s)


def derive_segment_targets(
    record: dict,
    segment_length_s: float = SEGMENT_LENGTH_S,
) -> tuple[list[float], list[int]]:
    """Convert a FIT-COACH long-range record into per-segment ground-truth.

    Each long video has dense per-frame feedbacks (most are empty strings;
    non-empty strings are the actual coaching utterances tagged by type).

    For each 30-s segment we derive:
      quality_gt:    1.0 - 0.6*frac_corrective + 0.2*frac_affirmative + 0.1*frac_ack
      rep_count_gt:  count of 'rep-counting' typed feedbacks in segment

    Returns
    -------
    quality_gt : list[float]   length = ceil(video_duration / segment_length)
    rep_gt     : list[int]     same length
    """
    from backend.training.preprocessing.qevd_label_builder import (
        classify_feedback_type, FIT_COACH_PASSTHROUGH_TYPES,
    )

    feedbacks_dense = record.get("feedbacks") or []
    timestamps      = record.get("feedback_timestamps") or []
    if isinstance(feedbacks_dense, list) and not feedbacks_dense:
        return [], []

    # Some manifests use parallel lists; others have list-of-dicts. Handle both.
    if feedbacks_dense and isinstance(feedbacks_dense[0], dict):
        # List of {text, t_start, t_end, type?}
        events = []
        for fb in feedbacks_dense:
            text = fb.get("text") or ""
            t = fb.get("t_start") or fb.get("timestamp") or 0.0
            ftype = fb.get("type")
            if ftype not in FIT_COACH_PASSTHROUGH_TYPES:
                ftype = classify_feedback_type(text) if text else "informative"
            events.append((float(t), text, ftype))
    else:
        # Parallel arrays — feedbacks is dense per-frame, mostly empty
        if len(timestamps) != len(feedbacks_dense):
            # Trust whichever is shorter
            n = min(len(feedbacks_dense), len(timestamps))
            feedbacks_dense = feedbacks_dense[:n]
            timestamps      = timestamps[:n]
        events = []
        last_text = ""
        for text, t in zip(feedbacks_dense, timestamps):
            text = str(text or "")
            if text and text != last_text:
                ftype = classify_feedback_type(text)
                events.append((float(t), text, ftype))
            last_text = text

    if not events:
        return [], []

    max_t = max(t for t, *_ in events)
    n_segments = max(1, int(np.ceil(max_t / segment_length_s)))

    counts = [{
        "corrective": 0, "affirmative": 0, "acknowledgment": 0,
        "informative": 0, "rep-counting": 0,
    } for _ in range(n_segments)]
    for (t, text, ftype) in events:
        idx = _segment_index(t, segment_length_s)
        if 0 <= idx < n_segments:
            if ftype not in counts[idx]:
                counts[idx][ftype] = 0
            counts[idx][ftype] += 1

    quality_gt: list[float] = []
    rep_gt:     list[int]   = []
    for c in counts:
        non_rep_total = (c["corrective"] + c["affirmative"]
                          + c["acknowledgment"] + c["informative"])
        if non_rep_total == 0:
            q = 0.7        # silence default (master plan section II.3)
        else:
            frac_c = c["corrective"]      / non_rep_total
            frac_a = c["affirmative"]     / non_rep_total
            frac_k = c["acknowledgment"]  / non_rep_total
            q = 1.0 - 0.6 * frac_c + 0.2 * frac_a + 0.1 * frac_k
            q = float(max(0.0, min(1.0, q)))
        quality_gt.append(float(q))
        rep_gt.append(int(c["rep-counting"]))

    return quality_gt, rep_gt


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────


def spearman_correlation(x: list[float], y: list[float]) -> float:
    """Spearman rank-correlation. Returns NaN when either input is constant."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return float("nan")
    # If either ORIGINAL series is constant, correlation is undefined
    # (argsort would still produce unique ranks, so we have to gate on
    # the input variance, not the rank variance).
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    return float(np.dot(rx_c, ry_c)
                 / (np.sqrt(np.dot(rx_c, rx_c) * np.dot(ry_c, ry_c)) + 1e-9))


def mean_absolute_error(pred: list[float], gt: list[float]) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    gt   = np.asarray(gt,   dtype=np.float64)
    n = min(pred.size, gt.size)
    if n == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[:n] - gt[:n])))


# ─────────────────────────────────────────────────────────────────────────────
# Per-video inference
# ─────────────────────────────────────────────────────────────────────────────


def _infer_video_segments(
    model,
    video_path: Path,
    exercise_idx: int,
    target_frames: int,
    stride: int,
    segment_length_s: float = SEGMENT_LENGTH_S,
) -> dict:
    """Run sliding-window inference; aggregate to per-segment statistics.

    Returns dict with quality_per_segment + rep_count_per_segment.
    """
    import tensorflow as tf
    from backend.training.preprocessing.qevd_extractor import extract_clip

    feats = extract_clip(video_path)
    pose = feats.pose_canon
    angles = feats.angles_raw
    fps = feats.fps_native
    duration_s = pose.shape[0] / max(1.0, fps)

    qualities, rep_counts, window_starts_t = [], [], []
    for start in range(0, max(1, pose.shape[0] - target_frames + 1), stride):
        w_pose = pose[start:start + target_frames]
        w_ang  = angles[start:start + target_frames]
        if w_pose.shape[0] < target_frames:
            pad = target_frames - w_pose.shape[0]
            w_pose = np.concatenate([w_pose,
                                       np.zeros((pad,) + w_pose.shape[1:],
                                                 dtype=w_pose.dtype)], axis=0)
            w_ang  = np.concatenate([w_ang,
                                       np.zeros((pad,) + w_ang.shape[1:],
                                                 dtype=w_ang.dtype)], axis=0)
        out = model({
            "pose":        tf.constant(w_pose[None].astype(np.float32)),
            "angles":      tf.constant(w_ang[None].astype(np.float32)),
            "exercise_id": tf.constant(np.array([exercise_idx], dtype=np.int32)),
        }, training=False)
        qualities.append(float(out["quality"][0, 0].numpy()))
        rep_counts.append(float(out["rep_count"][0, 0].numpy()))
        window_starts_t.append(start / max(1.0, fps))

    # Aggregate windows into segments
    n_segments = max(1, int(np.ceil(duration_s / segment_length_s)))
    seg_quality = [[] for _ in range(n_segments)]
    seg_repcount = [[] for _ in range(n_segments)]
    for q, r, t_start in zip(qualities, rep_counts, window_starts_t):
        idx = _segment_index(t_start, segment_length_s)
        if 0 <= idx < n_segments:
            seg_quality[idx].append(q)
            seg_repcount[idx].append(r)

    quality_per_segment = [
        float(np.mean(qs)) if qs else 0.7  # silence default
        for qs in seg_quality
    ]
    # Rep count per segment: mean predicted across all windows, scaled by
    # segment length / window length so a 30-s segment that has 5 reps in
    # the window estimate -> ~5 * (30/window_seconds).
    window_duration_s = target_frames / max(1.0, fps)
    rep_count_per_segment = [
        float(np.mean(rs)) * (segment_length_s / window_duration_s)
        if rs else 0.0
        for rs in seg_repcount
    ]
    return {
        "duration_s":            duration_s,
        "quality_per_segment":   quality_per_segment,
        "rep_count_per_segment": rep_count_per_segment,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir",   type=Path, required=True)
    parser.add_argument("--videos-dir",  type=Path, required=True,
                        help="Directory containing FIT-COACH long_range *.mp4 files.")
    parser.add_argument("--feedbacks",   type=Path, required=True,
                        help="Path to feedbacks_long_range.json (benchmark or train).")
    parser.add_argument("--report",      type=Path, required=True)
    parser.add_argument("--target-frames", type=int, default=64)
    parser.add_argument("--stride",      type=int, default=8)
    parser.add_argument("--exercise",    type=str, default="squat")
    parser.add_argument("--max-videos",  type=int, default=0,
                        help="If >0, only evaluate the first N videos (smoke).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(message)s")

    # Avoid circular: import reality_check_v6's loader
    from backend.training.evaluation.reality_check_v6 import _load_model

    print(f"[1] loading v6 model from {args.model_dir} ...")
    model, cfg, ex_map = _load_model(args.model_dir)
    exercise_idx = int(ex_map.get(args.exercise, ex_map.get("__other__", 0)))
    print(f"    exercise_idx for conditioning: {exercise_idx}")

    print(f"[2] loading feedbacks_long_range from {args.feedbacks} ...")
    feedbacks = json.loads(args.feedbacks.read_text())
    if not isinstance(feedbacks, list):
        raise ValueError(f"unexpected feedbacks_long_range schema: "
                          f"{type(feedbacks).__name__}")
    print(f"    {len(feedbacks)} long-range records")

    if args.max_videos > 0:
        feedbacks = feedbacks[:args.max_videos]
        print(f"    [smoke] truncated to first {len(feedbacks)} records")

    all_pred_q,  all_gt_q  = [], []
    all_pred_rc, all_gt_rc = [], []
    per_video: list[dict] = []

    for i, rec in enumerate(feedbacks):
        # Resolve video path (record stores ./long_range_videos/0001.mp4 form)
        rel = (rec.get("long_range_video_file") or
               rec.get("video_path") or "")
        rel = rel.lstrip("./")
        if "/" in rel:
            rel = rel.rsplit("/", 1)[-1]
        if "\\" in rel:
            rel = rel.rsplit("\\", 1)[-1]
        video_path = args.videos_dir / rel
        if not video_path.exists():
            print(f"  [{i+1}/{len(feedbacks)}] {rel}: NOT FOUND, skipping")
            continue

        gt_q, gt_rc = derive_segment_targets(rec)
        if not gt_q:
            print(f"  [{i+1}/{len(feedbacks)}] {rel}: no GT segments, skipping")
            continue

        print(f"  [{i+1}/{len(feedbacks)}] {video_path.name}: "
              f"{len(gt_q)} GT segments ...")
        t0 = time.time()
        pred = _infer_video_segments(
            model, video_path, exercise_idx,
            target_frames=cfg["target_frames"], stride=args.stride,
        )
        elapsed = time.time() - t0
        n = min(len(pred["quality_per_segment"]), len(gt_q))
        all_pred_q.extend(pred["quality_per_segment"][:n])
        all_gt_q.extend(gt_q[:n])
        all_pred_rc.extend(pred["rep_count_per_segment"][:n])
        all_gt_rc.extend(gt_rc[:n])
        per_video.append({
            "name":         video_path.name,
            "n_segments":   n,
            "elapsed_s":    round(elapsed, 1),
        })

    # Compute aggregate metrics
    print()
    print(f"[3] computing aggregate metrics on {len(all_pred_q)} segments ...")
    quality_rho = spearman_correlation(all_pred_q, all_gt_q)
    rep_mae     = mean_absolute_error(all_pred_rc, all_gt_rc)
    print(f"    quality Spearman ρ: {quality_rho:.4f}  (need >= {QUALITY_SPEARMAN_MIN})")
    print(f"    rep-count MAE:      {rep_mae:.4f}      (need <= {REP_COUNT_MAE_MAX})")

    quality_pass = (not np.isnan(quality_rho)
                     and quality_rho >= QUALITY_SPEARMAN_MIN)
    rep_pass     = (not np.isnan(rep_mae) and rep_mae <= REP_COUNT_MAE_MAX)
    overall_pass = bool(quality_pass and rep_pass)

    report = {
        "gate_id":   "D8_ood_eval",
        "status":    "PASS" if overall_pass else "FAIL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "model_dir":         str(args.model_dir),
            "videos_dir":        str(args.videos_dir),
            "feedbacks":         str(args.feedbacks),
            "exercise":          args.exercise,
            "n_videos":          len(per_video),
            "n_segments_total":  len(all_pred_q),
            "quality_spearman":  round(float(quality_rho), 4) if not np.isnan(quality_rho) else None,
            "rep_count_mae":     round(float(rep_mae),     4) if not np.isnan(rep_mae)     else None,
            "quality_pass":      bool(quality_pass),
            "rep_count_pass":    bool(rep_pass),
        },
        "per_video": per_video,
        "failures":  [],
    }
    if not quality_pass:
        report["failures"].append({
            "metric": "quality_spearman",
            "actual": report["metrics"]["quality_spearman"],
            "bound":  f">={QUALITY_SPEARMAN_MIN}",
        })
    if not rep_pass:
        report["failures"].append({
            "metric": "rep_count_mae",
            "actual": report["metrics"]["rep_count_mae"],
            "bound":  f"<={REP_COUNT_MAE_MAX}",
        })

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\n[4] D8 verdict: {report['status']}")
    print(f"    report -> {args.report}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
