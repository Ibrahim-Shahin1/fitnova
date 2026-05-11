"""D9 reality-check — v6 model on the user's 4 squat videos.

This is THE gate that determines if v6 ships. Master plan section II.8.

Hard gates (all four must pass):
  1. Top-1 action == "squat" (or related QEVD class) on all 4 clips
  2. mean(quality(Good_*)) - mean(quality(Bad_*)) >= 0.15
  3. Knee/hip group probability > 0.2 somewhere in Bad_*,
     < 0.1 throughout Good_*
  4. Boundary head fires >= 1 spike per ~3 s in Good_Squats (rep-count sanity)

Pipeline (mirrors `backend.services.form_analyzer` exactly):
  - MediaPipe Pose Tasks API VIDEO mode, pinned config
  - 33 world landmarks -> 15 canonical joints (B5 neck-weighted blend)
  - hip-centred + torso-scaled normalisation
  - 22 AIFit angular features per frame
  - Slide 64-frame windows (stride 8) across the whole clip
  - Aggregate per-window predictions

Usage:
    python -m backend.training.evaluation.reality_check_v6 \\
        --model-dir backend/models/form_model_v6 \\
        --report   backend/data/qevd_phase_reports/D9_reality_check.json

Requires:
  <model-dir>/v6_supervised.weights.h5     (output of train_form_model_v6.py)
  <model-dir>/model_config.json
  <model-dir>/qevd_exercise_map.json
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
# Defaults — locked to the master plan
# ─────────────────────────────────────────────────────────────────────────────


DEFAULT_USER_VIDEOS = [
    ("C:/Users/tsh_x/Downloads/Good_Squats.mp4",  "Good_Squats",  "good"),
    ("C:/Users/tsh_x/Downloads/Good_Squats2.mp4", "Good_Squats2", "good"),
    ("C:/Users/tsh_x/Downloads/Bad_Squats.mp4",   "Bad_Squats",   "bad"),
    ("C:/Users/tsh_x/Downloads/Bad_Squats2.mp4",  "Bad_Squats2",  "bad"),
]

DEFAULT_WINDOW_STRIDE = 8
DEFAULT_TARGET_FRAMES = 64

# Master plan section II.8 thresholds
QUALITY_GAP_MIN          = 0.15
KNEE_HIP_BAD_MIN         = 0.20
KNEE_HIP_GOOD_MAX        = 0.10
BOUNDARY_SPIKE_THRESHOLD = 0.5


# v6 joint-group order matches form_session.JOINT_GROUP_NAMES exactly
JOINT_GROUP_NAMES_V6 = [
    "L_elbow", "R_elbow", "L_shoulder", "R_shoulder",
    "L_knee",  "R_knee",  "L_hip",      "R_hip",
    "Trunk",   "Neck",
]
KNEE_HIP_INDICES = [
    JOINT_GROUP_NAMES_V6.index(n)
    for n in ("L_knee", "R_knee", "L_hip", "R_hip")
]


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_model(model_dir: Path):
    """Load v6 model weights + config + exercise map."""
    import tensorflow as tf
    from backend.training.models.st_gcn_v6 import build_v6_model

    config_path  = model_dir / "model_config.json"
    weights_path = model_dir / "v6_supervised.weights.h5"
    ex_map_path  = model_dir / "qevd_exercise_map.json"
    for p in (config_path, weights_path, ex_map_path):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")

    cfg = json.loads(config_path.read_text())
    ex_map = json.loads(ex_map_path.read_text())

    model = build_v6_model(
        target_frames=cfg["target_frames"],
        n_exercises=cfg["n_exercises"],
        n_joint_groups=cfg["n_joint_groups"],
    )
    model.load_weights(str(weights_path))
    return model, cfg, ex_map


def _extract_pose_features(mp4_path: Path):
    """Pull pose_canon + angles_raw from a clip via the same path used for
    training data extraction (single source of truth)."""
    from backend.training.preprocessing.qevd_extractor import extract_clip
    feats = extract_clip(mp4_path)
    return feats.pose_canon, feats.angles_raw, feats.fps_native


def _sliding_windows(arr: np.ndarray, target_frames: int, stride: int):
    """Yield (start_index, window) tuples covering arr along axis 0."""
    T = arr.shape[0]
    if T < target_frames:
        # Pad with zeros at the end
        pad_amount = target_frames - T
        pad_shape = (pad_amount,) + arr.shape[1:]
        padded = np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)],
                                axis=0)
        yield 0, padded
        return
    for start in range(0, T - target_frames + 1, stride):
        yield start, arr[start:start + target_frames]
    last_start = T - target_frames
    if last_start % stride != 0:
        yield last_start, arr[last_start:]


def _resolve_squat_indices(exercise_map: dict[str, int]) -> list[int]:
    """Return action-head indices that correspond to 'squat' or variants
    in the exercise map. Used by gate #1."""
    indices = []
    for name, idx in exercise_map.items():
        if name == "__other__":
            continue
        n = name.lower()
        if "squat" in n:
            indices.append(int(idx))
    return indices


# ─────────────────────────────────────────────────────────────────────────────
# Per-clip inference
# ─────────────────────────────────────────────────────────────────────────────


def _infer_clip(
    model,
    pose_canon: np.ndarray,
    angles_raw: np.ndarray,
    fps_native: float,
    *,
    exercise_idx: int,
    target_frames: int,
    stride: int,
) -> dict:
    """Run sliding-window inference. Returns aggregated per-clip stats.

    Outputs:
      n_windows, duration_s, quality_mean, quality_std, action_top1_idx,
      action_top1_conf, action_squat_conf, boundary_max,
      joint_err_max (per-group), joint_err_mean (per-group),
      rep_count_mean, knee_hip_max
    """
    import tensorflow as tf

    if pose_canon.shape[0] < target_frames // 2:
        # too short to be useful
        return {"error": f"clip too short ({pose_canon.shape[0]} frames)"}

    qualities, action_logits, boundary_seqs, je_seqs, rep_counts = (
        [], [], [], [], [],
    )
    n_windows = 0
    for (start_p, w_pose), (start_a, w_ang) in zip(
        _sliding_windows(pose_canon, target_frames, stride),
        _sliding_windows(angles_raw, target_frames, stride),
    ):
        n_windows += 1
        out = model({
            "pose":        tf.constant(w_pose[None].astype(np.float32)),
            "angles":      tf.constant(w_ang[None].astype(np.float32)),
            "exercise_id": tf.constant(np.array([exercise_idx], dtype=np.int32)),
        }, training=False)
        qualities.append(float(out["quality"][0, 0].numpy()))
        action_logits.append(out["action"][0].numpy())
        boundary_seqs.append(out["boundary"][0, :, 0].numpy())
        je_seqs.append(out["joint_err"][0].numpy())
        rep_counts.append(float(out["rep_count"][0, 0].numpy()))

    quality_arr   = np.array(qualities)
    action_arr    = np.stack(action_logits, axis=0).mean(axis=0)
    boundary_arr  = np.concatenate(boundary_seqs)
    je_arr        = np.stack(je_seqs, axis=0)
    je_mean       = je_arr.mean(axis=(0, 1))
    je_max        = je_arr.max(axis=(0, 1))
    knee_hip_max  = float(je_arr[..., KNEE_HIP_INDICES].max())
    rep_count_arr = np.array(rep_counts)

    return {
        "n_windows":           int(n_windows),
        "duration_s":          float(pose_canon.shape[0] / max(1.0, fps_native)),
        "quality_mean":        float(quality_arr.mean()),
        "quality_std":         float(quality_arr.std()),
        "action_top1_idx":     int(np.argmax(action_arr)),
        "action_top1_conf":    float(action_arr.max()),
        "action_full":         action_arr.tolist(),
        "boundary_max":        float(boundary_arr.max()),
        "boundary_seq":        boundary_arr.tolist(),
        "joint_err_max":       je_max.tolist(),
        "joint_err_mean":      je_mean.tolist(),
        "knee_hip_max":        knee_hip_max,
        "rep_count_mean":      float(rep_count_arr.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# D9 gate evaluation
# ─────────────────────────────────────────────────────────────────────────────


def _evaluate_gates(
    results: dict[str, dict],
    squat_indices: list[int],
    fps_native: float = 30.0,
) -> dict:
    """Apply the four D9 gates and return a verdict dict."""
    gates: dict = {}

    # Gate 1: top-1 action == squat on all 4
    actions_match = []
    for label, info in results.items():
        top_idx = int(info["action_top1_idx"])
        actions_match.append(top_idx in squat_indices)
    gates["all_squat"] = bool(all(actions_match))

    # Gate 2: quality gap >= 0.15
    good_q = [info["quality_mean"]
              for label, info in results.items()
              if "good" in label.lower()]
    bad_q  = [info["quality_mean"]
              for label, info in results.items()
              if "bad" in label.lower()]
    gates["quality_good_mean"] = float(np.mean(good_q)) if good_q else float("nan")
    gates["quality_bad_mean"]  = float(np.mean(bad_q))  if bad_q  else float("nan")
    gap = (gates["quality_good_mean"] - gates["quality_bad_mean"]
            if good_q and bad_q else float("nan"))
    gates["quality_gap"] = float(gap) if not np.isnan(gap) else None
    gates["quality_gap_pass"] = bool(gap >= QUALITY_GAP_MIN) if not np.isnan(gap) else False

    # Gate 3: knee/hip discrimination
    bad_max  = max((info["knee_hip_max"]
                     for label, info in results.items()
                     if "bad" in label.lower()), default=0.0)
    # For "good", we want knee/hip MAX over the clip to be < 0.10
    good_max = max((info["knee_hip_max"]
                     for label, info in results.items()
                     if "good" in label.lower()), default=0.0)
    gates["knee_hip_bad_max"]  = float(bad_max)
    gates["knee_hip_good_max"] = float(good_max)
    gates["knee_hip_pass"] = bool(
        bad_max > KNEE_HIP_BAD_MIN and good_max < KNEE_HIP_GOOD_MAX
    )

    # Gate 4: boundary spikes in Good_Squats — at least 1 spike per ~3 s
    # (i.e. for a 12 s clip, at least 4 spikes; rate >= 1/3 spikes/sec)
    good_clips = [(label, info) for label, info in results.items()
                  if "good" in label.lower()]
    if good_clips:
        label, info = good_clips[0]
        seq = np.array(info.get("boundary_seq", []))
        # Count non-overlapping crossings above threshold (rough rep proxy).
        spikes = 0
        in_spike = False
        for v in seq:
            if v > BOUNDARY_SPIKE_THRESHOLD:
                if not in_spike:
                    spikes += 1
                    in_spike = True
            else:
                in_spike = False
        clip_duration_s = info["duration_s"]
        spike_rate = spikes / max(1e-6, clip_duration_s)
        gates["boundary_spikes"]    = int(spikes)
        gates["boundary_rate_per_s"] = float(spike_rate)
        gates["boundary_pass"] = bool(spike_rate >= 1.0 / 3.0)
    else:
        gates["boundary_pass"] = False

    gates["overall_pass"] = bool(
        gates["all_squat"] and
        gates["quality_gap_pass"] and
        gates["knee_hip_pass"] and
        gates["boundary_pass"]
    )
    return gates


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Directory containing v6_supervised.weights.h5 + "
                             "model_config.json + qevd_exercise_map.json.")
    parser.add_argument("--report", type=Path, required=True,
                        help="Output path for the D9 reality-check JSON.")
    parser.add_argument("--videos", type=Path, nargs="+", default=None,
                        help="Override the default 4 user videos (paths). "
                             "If not given, uses the standard Good/Bad set.")
    parser.add_argument("--target-frames", type=int, default=DEFAULT_TARGET_FRAMES)
    parser.add_argument("--stride", type=int, default=DEFAULT_WINDOW_STRIDE)
    parser.add_argument("--exercise", type=str, default="squat",
                        help="Exercise name to look up in the map (default 'squat').")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(message)s")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"[1] loading v6 model from {args.model_dir} ...")
    model, cfg, ex_map = _load_model(args.model_dir)
    print(f"    target_frames={cfg['target_frames']}  "
          f"n_exercises={cfg['n_exercises']}  "
          f"n_joint_groups={cfg['n_joint_groups']}")

    # ── Resolve exercise idx (for conditioning input) ──────────────────────
    if args.exercise in ex_map:
        exercise_idx = int(ex_map[args.exercise])
    elif "__other__" in ex_map:
        exercise_idx = int(ex_map["__other__"])
        print(f"    [warn] '{args.exercise}' not in exercise map -> using 'other' idx {exercise_idx}")
    else:
        exercise_idx = 0
    print(f"    exercise_idx for conditioning: {exercise_idx} ({args.exercise!r})")

    squat_indices = _resolve_squat_indices(ex_map)
    print(f"    squat-related action indices: {squat_indices}")

    # ── Resolve videos ─────────────────────────────────────────────────────
    if args.videos is None:
        video_specs = [(Path(p), label, kind) for p, label, kind in DEFAULT_USER_VIDEOS]
    else:
        video_specs = []
        for p in args.videos:
            label = p.stem
            kind = "good" if "good" in label.lower() else \
                   ("bad" if "bad" in label.lower() else "unknown")
            video_specs.append((p, label, kind))

    # ── Run inference per clip ─────────────────────────────────────────────
    results: dict = {}
    for path, label, kind in video_specs:
        print(f"\n[2] {label} ({path}) ...")
        if not path.exists():
            print(f"    [skip] file missing")
            results[label] = {"error": "file missing", "kind": kind}
            continue
        t0 = time.time()
        pose_canon, angles_raw, fps = _extract_pose_features(path)
        print(f"    extracted {pose_canon.shape[0]} frames at {fps:.1f} fps "
              f"in {time.time() - t0:.1f}s")
        clip_t0 = time.time()
        info = _infer_clip(
            model, pose_canon, angles_raw, fps,
            exercise_idx=exercise_idx,
            target_frames=cfg["target_frames"],
            stride=args.stride,
        )
        info["kind"] = kind
        info["elapsed_s"] = round(time.time() - clip_t0, 2)
        results[label] = info
        print(f"    quality_mean = {info.get('quality_mean', float('nan')):.3f}  "
              f"action_top1 = {info.get('action_top1_idx', '?')}  "
              f"knee_hip_max = {info.get('knee_hip_max', float('nan')):.3f}  "
              f"boundary_max = {info.get('boundary_max', float('nan')):.3f}")

    # ── Apply gates ────────────────────────────────────────────────────────
    print(f"\n[3] applying D9 gates ...")
    gates = _evaluate_gates(results, squat_indices)
    for k, v in gates.items():
        print(f"    {k}: {v}")

    # ── Save report ────────────────────────────────────────────────────────
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "gate_id":    "D9_reality_check",
        "status":     "PASS" if gates["overall_pass"] else "FAIL",
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "metrics":    {
            "model_dir":    str(args.model_dir),
            "exercise":     args.exercise,
            "exercise_idx": exercise_idx,
            "squat_action_indices": squat_indices,
            "stride":       args.stride,
            "target_frames": cfg["target_frames"],
        },
        "per_clip":   results,
        "gates":      gates,
        "failures":   [],
    }
    if not gates["overall_pass"]:
        for k in ("all_squat", "quality_gap_pass",
                  "knee_hip_pass", "boundary_pass"):
            if not gates[k]:
                report["failures"].append({
                    "metric": k, "actual": gates[k], "bound": "True",
                })

    args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\n[4] D9 verdict: {report['status']}")
    print(f"    report -> {args.report}")
    return 0 if gates["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
