"""
Diagnose the currently-built form dataset.

Goal: determine whether the quality + joint_error labels are degenerate
(i.e. almost all zeros or almost all ones), which would explain why the
first training run learned nothing useful on val.

Run:
    python -m backend.training.diagnose_dataset
"""

import json
import os
import sys
import numpy as np

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "form_dataset"
)


def _hist(arr: np.ndarray, bins: int = 10) -> str:
    """Return a compact text histogram with counts."""
    counts, edges = np.histogram(arr, bins=bins)
    lines = []
    for i, c in enumerate(counts):
        pct = 100.0 * c / counts.sum() if counts.sum() else 0.0
        bar = "#" * int(pct / 2)
        lines.append(f"    [{edges[i]:5.2f}, {edges[i+1]:5.2f}] {c:6d}  {pct:5.1f}%  {bar}")
    return "\n".join(lines)


def diagnose_split(split_dir: str, split_name: str) -> dict:
    print(f"\n{'='*78}")
    print(f"  SPLIT: {split_name.upper()}  ({split_dir})")
    print(f"{'='*78}")

    paths = {k: os.path.join(split_dir, f"{k}.npy") for k in [
        "X_angles", "X_joints", "y_exercise", "y_quality", "y_joints",
        "y_boundary", "y_count",
    ]}
    arrays = {k: np.load(p) for k, p in paths.items() if os.path.exists(p)}

    N = len(arrays["y_quality"])
    print(f"\n  n_samples: {N}")
    print(f"  X_angles shape: {arrays['X_angles'].shape}")
    print(f"  X_joints shape: {arrays['X_joints'].shape}")

    # ── Exercise class balance ────────────────────────────────────────────────
    ex = arrays["y_exercise"]
    ex_counts = np.bincount(ex)
    print(f"\n  Exercise class distribution ({len(ex_counts)} classes):")
    print(f"    min={ex_counts.min()}  max={ex_counts.max()}  mean={ex_counts.mean():.1f}")
    print(f"    zero-count classes: {(ex_counts == 0).sum()}")

    # ── Quality score distribution (THE KEY CHECK) ────────────────────────────
    q = arrays["y_quality"]
    print(f"\n  Quality score stats:")
    print(f"    min={q.min():.4f}  max={q.max():.4f}  mean={q.mean():.4f}  std={q.std():.4f}")
    print(f"    exact 1.0 : {(q == 1.0).sum()} / {N}  ({100*(q==1.0).mean():.1f}%)")
    print(f"    exact 0.0 : {(q == 0.0).sum()} / {N}  ({100*(q==0.0).mean():.1f}%)")
    print(f"    >= 0.95   : {(q >= 0.95).sum()} / {N}  ({100*(q>=0.95).mean():.1f}%)")
    print(f"    <= 0.5    : {(q <= 0.5).sum()} / {N}  ({100*(q<=0.5).mean():.1f}%)")
    print(f"    unique values: {len(np.unique(q))}")
    print(f"    histogram:")
    print(_hist(q, bins=10))

    # ── Per-joint-group error positive rate ───────────────────────────────────
    je = arrays["y_joints"]   # (N, 64, 10)
    frames_total = je.shape[0] * je.shape[1]
    pos_rate_overall = je.mean()
    print(f"\n  Joint-error labels (N={je.shape[0]}, T={je.shape[1]}, G={je.shape[2]}):")
    print(f"    overall positive rate: {pos_rate_overall:.4f}  ({100*pos_rate_overall:.2f}%)")
    group_names = [
        "l_elbow", "r_elbow", "l_shoulder", "r_shoulder",
        "l_knee",  "r_knee",  "l_hip",      "r_hip",
        "trunk",   "neck",
    ]
    print(f"    per-group positive rate:")
    for g in range(je.shape[2]):
        r = je[:, :, g].mean()
        print(f"      {group_names[g]:>12}: {r:.4f}  ({100*r:.2f}%)")

    # ── Per-rep "at least one error somewhere" rate ───────────────────────────
    any_err_per_rep = (je.sum(axis=(1, 2)) > 0).mean()
    print(f"    fraction of reps with ANY error flag: {any_err_per_rep:.4f}")

    # ── Boundary label sanity ─────────────────────────────────────────────────
    b = arrays["y_boundary"]
    print(f"\n  Boundary labels (N={b.shape[0]}, T={b.shape[1]}):")
    print(f"    positive rate: {b.mean():.4f}")
    print(f"    mean positives per sample: {b.sum(axis=1).mean():.2f}")

    # ── Rep count sanity ──────────────────────────────────────────────────────
    rc = arrays["y_count"]
    print(f"\n  Rep count labels: unique={np.unique(rc)}  mean={rc.mean():.2f}")

    # ── Angular feature sanity ────────────────────────────────────────────────
    xa = arrays["X_angles"]
    print(f"\n  Angular features: mean={xa.mean():.3f}  std={xa.std():.3f}  "
          f"min={xa.min():.3f}  max={xa.max():.3f}")

    return {
        "n": N,
        "quality_mean": float(q.mean()),
        "quality_std": float(q.std()),
        "quality_frac_1": float((q == 1.0).mean()),
        "quality_unique": int(len(np.unique(q))),
        "joint_err_pos_rate": float(pos_rate_overall),
        "any_err_per_rep": float(any_err_per_rep),
    }


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIR
    print(f"Diagnosing dataset in: {data_dir}")

    summary = {}
    for split in ["train", "val"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"  [skip] {split_dir} not found")
            continue
        summary[split] = diagnose_split(split_dir, split)

    print(f"\n{'='*78}")
    print("  VERDICT")
    print(f"{'='*78}")
    for split, s in summary.items():
        q_ok = 0.05 < s["quality_mean"] < 0.95 and s["quality_std"] > 0.1
        j_ok = 0.05 < s["joint_err_pos_rate"] < 0.5
        print(f"  {split}:  quality_mean={s['quality_mean']:.3f}  "
              f"quality_std={s['quality_std']:.3f}  "
              f"joint_pos={s['joint_err_pos_rate']:.3f}  "
              f"→ {'OK' if q_ok and j_ok else 'DEGENERATE'}")

    print("\n  Interpretation:")
    print("    - quality_mean should be ~0.5-0.8 with std > 0.1 for useful learning signal")
    print("    - joint_err_pos_rate should be ~0.1-0.3 (not all-zero, not all-one)")
    print("    - if both train AND val are 'DEGENERATE' the labels must be rebuilt")


if __name__ == "__main__":
    main()
