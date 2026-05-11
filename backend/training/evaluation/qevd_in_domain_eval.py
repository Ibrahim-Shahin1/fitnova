"""D7 in-domain evaluation — v6 model on the FIT-300K test split.

Master plan section II.8 thresholds:
  * Action top-1 accuracy           >= 0.55
  * Quality Spearman rho            >= 0.40
  * Joint-err mean AP (10 groups)   >= 0.30

Pipeline
--------
1. Load v6 model + config + exercise map.
2. Load QEVDLabels manifests, then call build_qevd_splits to get the
   in_domain_test clip-id set (FIT-300K split == "test" — Qualcomm's
   pre-defined held-out set, NOT participant-disjoint val).
3. Restrict to clips that have a .npz on disk in any of --extracted-dirs.
4. For each clip:
     - Load pose_canon + angles_raw from .npz
     - Resample to 64 frames (single forward pass — clips are ~5 s,
       so a sliding-window strategy adds nothing here)
     - Predict: action softmax, quality scalar, per-frame joint_err (10).
5. Compute aggregate metrics:
     - action_top1_accuracy:  argmax(action) == map_exercise_to_idx(GT)
     - quality_spearman:      pred_q vs compute_clip_quality(GT) over N clips
     - joint_err_mean_ap:     per-group AP using max-over-time pred score
                              against the union-of-feedback+class GT vector;
                              groups with no positives are dropped from mean
6. Write backend/data/qevd_phase_reports/D7_in_domain.json.

Usage
-----
python -m backend.training.evaluation.qevd_in_domain_eval \
    --model-dir       backend/models/form_model_v6 \
    --extracted-dirs  backend/data/qevd_extracted/fit300k/test \
    --fine-grained    backend/data/qevd_raw/.../fine_grained_labels.json \
    --feedbacks       backend/data/qevd_raw/.../feedbacks_short_clips.json \
    --worker-ids      backend/data/qevd_raw/.../fine_grained_labels_with_worker_ids.json \
    --report          backend/data/qevd_phase_reports/D7_in_domain.json

Notes
-----
- The v6 action head outputs over (n_named + 1) classes, with index
  n_named being '__other__'. We DO include __other__ in the top-1
  accuracy: a model that confuses a niche exercise for "other" is not
  a correct detection.
- Joint-err GT comes from the SAME path qevd_dataset.derive_v6_targets
  uses for training (union of feedback-derived + class-derived 10-vec).
  This is the contract the model was trained to satisfy.
- Pred score for AP is max-over-time of joint_err[t, g] — same
  aggregation reality_check_v6 uses for knee_hip_max. Mean-over-time
  is too smooth for a per-frame signal that's sparse by design.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Master-plan thresholds (section II.8)
# ─────────────────────────────────────────────────────────────────────────────

ACTION_TOP1_MIN      = 0.55
QUALITY_SPEARMAN_MIN = 0.40
JOINT_ERR_MAP_MIN    = 0.30


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────


def action_top1_accuracy(pred_idx: list[int], gt_idx: list[int]) -> float:
    if not pred_idx:
        return float("nan")
    n = min(len(pred_idx), len(gt_idx))
    if n == 0:
        return float("nan")
    correct = sum(int(p == g) for p, g in zip(pred_idx[:n], gt_idx[:n]))
    return float(correct) / float(n)


def quality_spearman(pred_q: list[float], gt_q: list[float]) -> float:
    """Spearman rank correlation; NaN when either input is constant."""
    x = np.asarray(pred_q, dtype=np.float64)
    y = np.asarray(gt_q,   dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return float("nan")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    return float(np.dot(rx_c, ry_c)
                 / (np.sqrt(np.dot(rx_c, rx_c) * np.dot(ry_c, ry_c)) + 1e-9))


def joint_err_mean_ap(
    pred_scores: np.ndarray,         # (N, J) float in [0, 1]
    gt_labels:   np.ndarray,         # (N, J) bool / 0-1
    group_names: Optional[list[str]] = None,
) -> dict:
    """Per-group average precision + mean across groups with positives.

    Groups with all-positive or all-negative ground truth are excluded
    from the mean (AP is undefined there).
    """
    from sklearn.metrics import average_precision_score

    pred = np.asarray(pred_scores, dtype=np.float64)
    gt   = np.asarray(gt_labels,   dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(
            f"shape mismatch: pred {pred.shape} vs gt {gt.shape}"
        )
    if pred.ndim != 2:
        raise ValueError(f"expected 2-D arrays, got pred.ndim={pred.ndim}")

    n_groups = pred.shape[1]
    per_group: list[Optional[float]] = []
    n_pos:     list[int] = []
    for j in range(n_groups):
        y_true = gt[:, j]
        n_pos.append(int(y_true.sum()))
        if n_pos[-1] == 0 or n_pos[-1] == y_true.size:
            per_group.append(None)
            continue
        per_group.append(float(average_precision_score(y_true, pred[:, j])))
    valid = [v for v in per_group if v is not None]
    mean_ap = float(np.mean(valid)) if valid else float("nan")
    return {
        "per_group_ap":     per_group,
        "per_group_n_pos":  n_pos,
        "mean_ap":          mean_ap,
        "n_valid_groups":   len(valid),
        "group_names":      list(group_names) if group_names is not None else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Inference (single forward pass per clip)
# ─────────────────────────────────────────────────────────────────────────────


def _infer_clip_singlepass(
    model,
    clip_data: dict,
    exercise_idx: int,
    target_frames: int,
) -> dict:
    """Resample the clip to target_frames, single forward pass.

    Returns dict with action/quality/joint_err/rep_count predictions.
    """
    import tensorflow as tf
    from backend.training.preprocessing.normalize import resample_sequence

    pose   = clip_data["pose_canon"].astype(np.float32)
    angles = clip_data["angles_raw"].astype(np.float32)
    pose_r   = resample_sequence(pose,   target_frames).astype(np.float32)
    angles_r = resample_sequence(angles, target_frames).astype(np.float32)

    out = model({
        "pose":        tf.constant(pose_r[None]),
        "angles":      tf.constant(angles_r[None]),
        "exercise_id": tf.constant(np.array([exercise_idx], dtype=np.int32)),
    }, training=False)

    action_full = out["action"][0].numpy()                     # (n_exercises,)
    je          = out["joint_err"][0].numpy()                  # (T, J)
    return {
        "action_argmax":  int(np.argmax(action_full)),
        "action_top1":    float(action_full.max()),
        "quality":        float(out["quality"][0, 0].numpy()),
        "rep_count":      float(out["rep_count"][0, 0].numpy()),
        "boundary_max":   float(out["boundary"][0, :, 0].numpy().max()),
        "joint_err_max":  je.max(axis=0).astype(np.float32),   # (J,)
        "joint_err_mean": je.mean(axis=0).astype(np.float32),  # (J,)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Clip resolution
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_npz(cid: str, dirs: list[Path]) -> Optional[Path]:
    """Find <cid>.npz in any of the provided directories. Tries both
    raw and zero-padded 8-digit numeric form."""
    candidates = [cid]
    if cid.isdigit():
        candidates.append(cid.zfill(8))
    seen: set[str] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        for d in dirs:
            p = d / f"{c}.npz"
            if p.exists():
                return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Dir containing v6_supervised.weights.h5 + "
                             "model_config.json + qevd_exercise_map.json.")
    parser.add_argument("--extracted-dirs", type=Path, nargs="+", required=True,
                        help="One or more dirs holding <clip_id>.npz files.")
    parser.add_argument("--fine-grained", type=Path, required=True,
                        help="Path to fine_grained_labels.json.")
    parser.add_argument("--feedbacks",    type=Path, required=True,
                        help="Path to feedbacks_short_clips.json.")
    parser.add_argument("--worker-ids",   type=Path, default=None,
                        help="Optional fine_grained_labels_with_worker_ids.json "
                             "(only used to validate split membership; not "
                             "required for D7 since FIT-300K already ships "
                             "split=test).")
    parser.add_argument("--report",       type=Path, required=True)
    parser.add_argument("--max-clips",    type=int, default=0,
                        help="If >0, only evaluate the first N test clips "
                             "(after a deterministic shuffle on --seed). 0 = all.")
    parser.add_argument("--target-frames", type=int, default=64)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(message)s")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"[1] loading v6 model from {args.model_dir} ...")
    from backend.training.evaluation.reality_check_v6 import _load_model
    model, cfg, ex_map = _load_model(args.model_dir)
    print(f"    target_frames={cfg['target_frames']}  "
          f"n_exercises={cfg['n_exercises']}  "
          f"n_joint_groups={cfg['n_joint_groups']}")

    # Lazy imports — keep TF init out of help/dry-run paths.
    from backend.training.preprocessing.qevd_dataset import (
        build_qevd_splits, load_clip_npz, map_exercise_to_idx,
    )
    from backend.training.preprocessing.qevd_label_builder import (
        QEVDLabels, compute_clip_quality, derive_joint_groups,
        derive_joint_groups_from_class, N_JOINT_GROUPS,
    )
    from backend.training.evaluation.reality_check_v6 import JOINT_GROUP_NAMES_V6

    # ── Load labels ────────────────────────────────────────────────────────
    print(f"[2] loading QEVD labels ...")
    labels = QEVDLabels.from_files(
        feedbacks_short_clips_path=args.feedbacks,
        fine_grained_labels_path=args.fine_grained,
        worker_ids_path=args.worker_ids,
    )
    n_clips_seen = len(set(labels.clip_to_feedbacks) | set(labels.clip_to_fine))
    print(f"    {n_clips_seen} clips with feedback or fine_grained records")

    # ── Build splits → in_domain_test clip set ─────────────────────────────
    splits = build_qevd_splits(
        labels.clip_to_split,
        labels.clip_to_human,
        seed=args.seed,
    )
    test_clips = sorted(splits.get("in_domain_test", set()))
    print(f"    in_domain_test split has {len(test_clips)} clip ids")

    # ── Filter to clips with .npz on disk ──────────────────────────────────
    print(f"[3] resolving .npz files in {args.extracted_dirs} ...")
    extract_dirs = [Path(d) for d in args.extracted_dirs]
    available: list[tuple[str, Path]] = []
    n_missing = 0
    for cid in test_clips:
        p = _resolve_npz(cid, extract_dirs)
        if p is None:
            n_missing += 1
            continue
        available.append((cid, p))
    print(f"    {len(available)} of {len(test_clips)} test clips have .npz "
          f"({n_missing} missing)")

    if not available:
        report = {
            "gate_id":   "D7_in_domain",
            "status":    "FAIL",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics":   {"n_clips_evaluated": 0},
            "failures":  [{"metric": "n_clips_evaluated",
                           "actual": 0, "bound": ">=1"}],
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"\n[!] no .npz files found for any in_domain_test clips.")
        print(f"    report -> {args.report}")
        return 1

    # Deterministic shuffle + truncation
    rng = np.random.default_rng(args.seed)
    rng.shuffle(available)
    if args.max_clips and args.max_clips > 0:
        available = available[:args.max_clips]
        print(f"    [smoke] truncated to first {len(available)} clips")

    # ── Iterate ────────────────────────────────────────────────────────────
    print(f"[4] running inference on {len(available)} clips ...")
    pred_action_idx: list[int] = []
    gt_action_idx:   list[int] = []
    pred_quality:    list[float] = []
    gt_quality:      list[float] = []
    pred_je_max:     list[np.ndarray] = []     # (J,) per clip
    gt_je:           list[np.ndarray] = []     # (J,) per clip
    n_no_pose_skipped = 0
    action_class_counts: Counter = Counter()
    t_start = time.time()
    log_every = max(1, len(available) // 20)
    for i, (cid, npz_path) in enumerate(available):
        try:
            clip_data = load_clip_npz(npz_path)
        except Exception as e:
            logger.warning("skipping %s: %s", npz_path, e)
            continue
        if clip_data["no_pose_fraction"] >= 0.5:
            n_no_pose_skipped += 1
            continue

        rec = labels.lookup(cid)
        feedbacks   = rec.get("feedbacks") or []
        labels_list = rec.get("labels") or []
        gt_action   = map_exercise_to_idx(rec.get("exercise"), ex_map)
        gt_q        = compute_clip_quality(feedbacks, labels_list)
        groups = np.zeros(N_JOINT_GROUPS, dtype=bool)
        for fb in feedbacks:
            text = fb if isinstance(fb, str) else fb.get("text", "")
            if text:
                groups |= derive_joint_groups(text)
        for lbl in labels_list:
            groups |= derive_joint_groups_from_class(lbl)

        infer = _infer_clip_singlepass(
            model, clip_data,
            exercise_idx=gt_action,
            target_frames=cfg["target_frames"],
        )

        pred_action_idx.append(infer["action_argmax"])
        gt_action_idx.append(int(gt_action))
        pred_quality.append(infer["quality"])
        gt_quality.append(float(gt_q))
        pred_je_max.append(infer["joint_err_max"])
        gt_je.append(groups.astype(np.float32))
        action_class_counts[int(gt_action)] += 1

        if (i + 1) % log_every == 0 or (i + 1) == len(available):
            elapsed = time.time() - t_start
            rate = (i + 1) / max(1e-6, elapsed)
            print(f"    [{i+1}/{len(available)}] "
                  f"{rate:.1f} clips/s  "
                  f"({elapsed:.1f}s elapsed)")

    n_eval = len(pred_action_idx)
    print(f"    evaluated {n_eval} clips "
          f"(skipped {n_no_pose_skipped} for no_pose>=0.5)")

    # ── Compute aggregate metrics ──────────────────────────────────────────
    print(f"[5] computing metrics ...")
    top1 = action_top1_accuracy(pred_action_idx, gt_action_idx)
    rho  = quality_spearman(pred_quality, gt_quality)
    je_pred_arr = np.stack(pred_je_max, axis=0)              # (N, J)
    je_gt_arr   = np.stack(gt_je,        axis=0)             # (N, J)
    ap_info = joint_err_mean_ap(
        je_pred_arr, je_gt_arr,
        group_names=JOINT_GROUP_NAMES_V6[:cfg["n_joint_groups"]],
    )
    print(f"    action_top1_accuracy : {top1:.4f}  "
          f"(need >= {ACTION_TOP1_MIN})")
    print(f"    quality_spearman_rho : {rho:.4f}  "
          f"(need >= {QUALITY_SPEARMAN_MIN})")
    print(f"    joint_err_mean_ap    : {ap_info['mean_ap']:.4f}  "
          f"(need >= {JOINT_ERR_MAP_MIN})")
    if ap_info["per_group_ap"] is not None:
        for name, ap, n_pos in zip(
            ap_info.get("group_names") or [f"g{i}" for i in range(len(ap_info["per_group_ap"]))],
            ap_info["per_group_ap"],
            ap_info["per_group_n_pos"],
        ):
            ap_str = f"{ap:.3f}" if ap is not None else "  -  "
            print(f"      {name:<14s}  AP={ap_str}  n_pos={n_pos}")

    # ── Build verdict ──────────────────────────────────────────────────────
    action_pass  = (not np.isnan(top1) and top1 >= ACTION_TOP1_MIN)
    quality_pass = (not np.isnan(rho)  and rho  >= QUALITY_SPEARMAN_MIN)
    je_pass      = (not np.isnan(ap_info["mean_ap"])
                    and ap_info["mean_ap"] >= JOINT_ERR_MAP_MIN)
    overall_pass = bool(action_pass and quality_pass and je_pass)

    failures: list[dict] = []
    if not action_pass:
        failures.append({"metric": "action_top1_accuracy",
                          "actual": None if np.isnan(top1) else round(top1, 4),
                          "bound":  f">={ACTION_TOP1_MIN}"})
    if not quality_pass:
        failures.append({"metric": "quality_spearman",
                          "actual": None if np.isnan(rho) else round(rho, 4),
                          "bound":  f">={QUALITY_SPEARMAN_MIN}"})
    if not je_pass:
        failures.append({"metric": "joint_err_mean_ap",
                          "actual": None if np.isnan(ap_info["mean_ap"])
                                          else round(ap_info["mean_ap"], 4),
                          "bound":  f">={JOINT_ERR_MAP_MIN}"})

    report = {
        "gate_id":   "D7_in_domain",
        "status":    "PASS" if overall_pass else "FAIL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "model_dir":              str(args.model_dir),
            "extracted_dirs":         [str(d) for d in extract_dirs],
            "n_test_clips_in_split":  len(test_clips),
            "n_test_clips_resolved":  len(available),
            "n_clips_evaluated":      n_eval,
            "n_clips_skipped_nopose": n_no_pose_skipped,
            "action_top1_accuracy":   None if np.isnan(top1) else round(float(top1), 4),
            "quality_spearman":       None if np.isnan(rho)  else round(float(rho),  4),
            "joint_err_mean_ap":      None if np.isnan(ap_info["mean_ap"])
                                            else round(float(ap_info["mean_ap"]), 4),
            "joint_err_per_group_ap": [
                None if v is None else round(float(v), 4)
                for v in ap_info["per_group_ap"]
            ],
            "joint_err_per_group_n_pos": ap_info["per_group_n_pos"],
            "joint_err_group_names":  ap_info.get("group_names"),
            "joint_err_n_valid_groups": ap_info["n_valid_groups"],
            "n_unique_action_classes_seen": len(action_class_counts),
        },
        "failures":  failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\n[6] D7 verdict: {report['status']}")
    print(f"    report -> {args.report}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
