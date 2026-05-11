"""D9 reality-check — v6.1 multi-label model on the user's 4 squat videos.

Forks reality_check_v6.py for the v6.1 architecture: single multi-label
sigmoid head over the cleaned 1,387-class QEVD variation taxonomy.
No quality scalar, no boundary head, no per-frame joint_err.

Gate (per plan §5):
  - GOOD clips: a "good-form" variation must appear in top-3 predictions
    on >= 60% of GOOD clips.
  - BAD clips: a "defect" variation must appear in top-3 predictions
    on >= 60% of BAD clips.
  - Top-K is computed AFTER masking the sigmoid logits to the variations
    that belong to the user-selected exercise prefix (here: squats).

The good-form / defect partitions for squats are drawn directly from the
QEVD variation taxonomy — NOT invented. They mirror the way QEVD's
crowd-workers labelled form: variations like "no obvious issue" /
"shoulder-width" / "90 degrees" describe a clip the annotator considered
correct, while "shallow" / "back not straight" / "knees over toes" /
"narrow" / "wide" / "insufficient" describe a clip the annotator flagged
as a form mistake.

Usage:
    python -m backend.training.evaluation.reality_check_v6_1 \\
        --model-dir backend/models/form_model_v6_1 \\
        --report    backend/data/qevd_phase_reports/D9_reality_check_v6_1.json

Requires:
  <model-dir>/v6_1_supervised.weights.h5  (output of train_form_model_v6_1.py)
  <model-dir>/model_config_v6_1.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

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
DEFAULT_TOP_K = 3

# Gate thresholds (per plan §8 Day 6)
GOOD_TOPK_PRECISION_MIN = 0.6
BAD_TOPK_PRECISION_MIN  = 0.6

# Squats partition of the QEVD variation taxonomy. Drawn from the actual
# `fine_grained_labels.json` — these are real strings the annotators
# used. Variations not in either set (e.g. "plie", "hold") are tempo
# / stance variants we don't grade as good or bad on a regular squat.
SQUAT_GOOD_VARIATIONS = {
    "no obvious issue",
    "90 degrees",
    "shoulder-width",
    "over 90 degrees",
}
SQUAT_DEFECT_VARIATIONS = {
    "shallow",
    "back not straight",
    "knees over toes",
    "narrow",
    "wide",
    "insufficient",
    "starting late",
}


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_v6_1_model(model_dir: Path):
    """Load v6.1 weights + config + class space."""
    from backend.training.models.st_gcn_v6_1 import build_v6_1_model
    from backend.training.preprocessing.qevd_class_space import (
        get_default_class_space,
    )

    config_path  = model_dir / "model_config_v6_1.json"
    weights_path = model_dir / "v6_1_supervised.weights.h5"
    for p in (config_path, weights_path):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")

    cfg = json.loads(config_path.read_text())
    cs  = get_default_class_space()

    if cfg.get("num_classes") != cs.num_classes:
        raise ValueError(
            f"model_config.num_classes ({cfg.get('num_classes')}) != "
            f"class_space.num_classes ({cs.num_classes}). "
            f"Rebuild the class space JSON or retrain."
        )

    model = build_v6_1_model(
        num_classes=cs.num_classes,
        n_exercises=cs.num_prefixes,
        target_frames=cfg["target_frames"],
    )
    model.load_weights(str(weights_path))
    return model, cfg, cs


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
        pad = target_frames - T
        pad_shape = (pad,) + arr.shape[1:]
        padded = np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0)
        yield 0, padded
        return
    for start in range(0, T - target_frames + 1, stride):
        yield start, arr[start:start + target_frames]
    last_start = T - target_frames
    if last_start % stride != 0:
        yield last_start, arr[last_start:]


# ─────────────────────────────────────────────────────────────────────────────
# Per-clip inference
# ─────────────────────────────────────────────────────────────────────────────


def _infer_clip(
    model,
    pose_canon: np.ndarray,
    angles_raw: np.ndarray,
    fps_native: float,
    *,
    exercise_prefix_idx: int,
    target_frames: int,
    stride: int,
    class_indices_mask: Sequence[int],
    top_k: int,
    class_space,
) -> dict:
    """Run sliding-window inference; aggregate per-window predictions.

    Returns:
      n_windows           : int
      duration_s          : float
      mean_probs_masked   : list[float] over class_indices_mask
      top_k_classes       : list[{idx, variation, prob}]
      good_present_in_topk: bool — any SQUAT_GOOD_VARIATIONS in top-k?
      defect_present_in_topk: bool — any SQUAT_DEFECT_VARIATIONS in top-k?
    """
    import tensorflow as tf

    if pose_canon.shape[0] < target_frames // 2:
        return {"error": f"clip too short ({pose_canon.shape[0]} frames)"}

    probs_per_window: list[np.ndarray] = []
    n_windows = 0
    for (start_p, w_pose), (start_a, w_ang) in zip(
        _sliding_windows(pose_canon, target_frames, stride),
        _sliding_windows(angles_raw, target_frames, stride),
    ):
        n_windows += 1
        out = model({
            "pose":        tf.constant(w_pose[None].astype(np.float32)),
            "angles":      tf.constant(w_ang[None].astype(np.float32)),
            "exercise_id": tf.constant(np.array([exercise_prefix_idx], dtype=np.int32)),
        }, training=False)
        probs_per_window.append(out["probs"][0].numpy())  # (num_classes,)

    probs_arr = np.stack(probs_per_window, axis=0)        # (n_win, num_classes)
    mean_probs = probs_arr.mean(axis=0)                   # (num_classes,)

    # Mask to the user-selected exercise's variations, then pick top-K.
    mask_arr = np.array(class_indices_mask, dtype=np.int64)
    masked_probs = mean_probs[mask_arr]                   # (n_masked,)
    order = np.argsort(-masked_probs)                     # descending
    top_k_local = order[:top_k]

    top_k_records = []
    top_k_variations = []
    for local_idx in top_k_local:
        global_idx = int(mask_arr[local_idx])
        prefix, variation = class_space.decode(global_idx)
        prob = float(masked_probs[local_idx])
        top_k_records.append({
            "global_idx": global_idx,
            "prefix":     prefix,
            "variation":  variation,
            "prob":       prob,
        })
        top_k_variations.append(variation)

    good_in_topk = any(v in SQUAT_GOOD_VARIATIONS for v in top_k_variations)
    defect_in_topk = any(v in SQUAT_DEFECT_VARIATIONS for v in top_k_variations)

    return {
        "n_windows":              int(n_windows),
        "duration_s":             float(pose_canon.shape[0] / max(1.0, fps_native)),
        "top_k_predictions":      top_k_records,
        "top_k_variations":       top_k_variations,
        "good_present_in_topk":   bool(good_in_topk),
        "defect_present_in_topk": bool(defect_in_topk),
        "masked_n_classes":       int(len(class_indices_mask)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gate evaluation
# ─────────────────────────────────────────────────────────────────────────────


def _evaluate_gates(results: dict[str, dict]) -> dict:
    """Compute per-group precision and the overall pass verdict."""
    good_clips = [(label, info) for label, info in results.items()
                  if info.get("kind") == "good" and "error" not in info]
    bad_clips  = [(label, info) for label, info in results.items()
                  if info.get("kind") == "bad" and "error" not in info]

    good_hits = sum(1 for _, info in good_clips if info["good_present_in_topk"])
    bad_hits  = sum(1 for _, info in bad_clips  if info["defect_present_in_topk"])

    good_precision = (good_hits / len(good_clips)) if good_clips else 0.0
    bad_precision  = (bad_hits  / len(bad_clips))  if bad_clips  else 0.0

    gates = {
        "n_good_clips":           len(good_clips),
        "n_bad_clips":            len(bad_clips),
        "good_topk_hits":         good_hits,
        "bad_topk_hits":          bad_hits,
        "good_topk_precision":    float(good_precision),
        "bad_topk_precision":     float(bad_precision),
        "good_topk_precision_pass": bool(good_precision >= GOOD_TOPK_PRECISION_MIN),
        "bad_topk_precision_pass":  bool(bad_precision  >= BAD_TOPK_PRECISION_MIN),
    }
    gates["overall_pass"] = bool(
        gates["good_topk_precision_pass"] and gates["bad_topk_precision_pass"]
    )
    return gates


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Directory containing v6_1_supervised.weights.h5 + "
                             "model_config_v6_1.json.")
    parser.add_argument("--report", type=Path, required=True,
                        help="Output path for the D9 v6.1 reality-check JSON.")
    parser.add_argument("--videos", type=Path, nargs="+", default=None,
                        help="Override the default 4 user videos.")
    parser.add_argument("--exercise-prefix", type=str, default="squats",
                        help="QEVD exercise prefix to mask the sigmoid logits to. "
                             "Default 'squats' (the user's test set). Use 'pushups' / "
                             "'lunges' / etc. for other exercises.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--stride", type=int, default=DEFAULT_WINDOW_STRIDE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(message)s")

    # ── Load model + class space ───────────────────────────────────────────
    print(f"[1] loading v6.1 model from {args.model_dir} ...")
    model, cfg, cs = _load_v6_1_model(args.model_dir)
    print(f"    target_frames={cfg['target_frames']}  "
          f"num_classes={cs.num_classes}  num_prefixes={cs.num_prefixes}")

    # ── Resolve exercise prefix + class-index mask ─────────────────────────
    prefix_idx = cs.prefix_idx_for(args.exercise_prefix)
    if prefix_idx is None:
        raise SystemExit(
            f"exercise prefix {args.exercise_prefix!r} not in v6.1 class space"
        )
    class_indices_mask = cs.class_indices_for_prefix(args.exercise_prefix)
    if not class_indices_mask:
        raise SystemExit(
            f"no classes for prefix {args.exercise_prefix!r} — class space corrupt?"
        )
    print(f"    exercise_prefix='{args.exercise_prefix}'  "
          f"prefix_idx={prefix_idx}  n_masked_classes={len(class_indices_mask)}")
    print(f"    variations in this prefix:")
    for idx in class_indices_mask:
        _, v = cs.decode(idx)
        tag = ""
        if v in SQUAT_GOOD_VARIATIONS:   tag = "  (good)"
        elif v in SQUAT_DEFECT_VARIATIONS: tag = "  (defect)"
        print(f"      [{idx:4d}] {v!r}{tag}")

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

    # ── Per-clip inference ─────────────────────────────────────────────────
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
            exercise_prefix_idx=prefix_idx,
            target_frames=cfg["target_frames"],
            stride=args.stride,
            class_indices_mask=class_indices_mask,
            top_k=args.top_k,
            class_space=cs,
        )
        info["kind"] = kind
        info["elapsed_s"] = round(time.time() - clip_t0, 2)
        results[label] = info
        if "error" in info:
            print(f"    [error] {info['error']}")
        else:
            print(f"    top-{args.top_k}: {info['top_k_variations']}")
            print(f"    good_in_topk={info['good_present_in_topk']}  "
                  f"defect_in_topk={info['defect_present_in_topk']}")

    # ── Apply gates ────────────────────────────────────────────────────────
    print(f"\n[3] applying D9 v6.1 gates ...")
    gates = _evaluate_gates(results)
    for k, v in gates.items():
        print(f"    {k}: {v}")

    # ── Save report ────────────────────────────────────────────────────────
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "gate_id":    "D9_reality_check_v6_1",
        "status":     "PASS" if gates["overall_pass"] else "FAIL",
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "metrics":    {
            "model_dir":         str(args.model_dir),
            "exercise_prefix":   args.exercise_prefix,
            "prefix_idx":        prefix_idx,
            "n_masked_classes":  len(class_indices_mask),
            "top_k":             args.top_k,
            "stride":            args.stride,
            "target_frames":     cfg["target_frames"],
            "class_space_version": cs.version,
        },
        "per_clip":   results,
        "gates":      gates,
        "failures":   [],
    }
    if not gates["overall_pass"]:
        for k in ("good_topk_precision_pass", "bad_topk_precision_pass"):
            if not gates[k]:
                report["failures"].append({
                    "metric": k, "actual": gates[k], "bound": "True",
                })

    args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\n[4] D9 v6.1 verdict: {report['status']}")
    print(f"    report -> {args.report}")
    return 0 if gates["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
