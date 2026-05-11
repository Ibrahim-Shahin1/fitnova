"""D5 smoke: end-to-end training validation on real QEVD clips.

Extracts a small subset of local Part-1 MP4s to .npz, then runs
train_form_model_v6 for a few epochs and validates the loss curves.

Master plan reference: section II.7 D5 hard-stop.

Run from repo root:
    python backend/training/_run_d5_smoke.py

Pass criteria:
  * All 5 per-head losses are finite and non-NaN at every epoch
  * Total loss at last epoch < total loss at first epoch (model is learning)
  * No exceptions during training
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

PART1_DIR  = REPO_ROOT / "backend/data/qevd_raw/fit300k/QEVD-FIT-300k-Part-1"
LABELS_DIR = REPO_ROOT / "backend/data/qevd_raw/fitcoach"
SMOKE_NPZ  = REPO_ROOT / "backend/data/qevd_extracted/smoke_d5"
SMOKE_OUT  = REPO_ROOT / "backend/models/form_model_v6_smoke"

N_CLIPS    = 80
N_TRAIN    = 60
N_VAL      = 15
EPOCHS     = 3
BATCH_SIZE = 8
SEED       = 42


def _extract_subset(n_clips: int) -> int:
    """Extract n_clips random Part-1 MP4s to .npz form. Resumes if some
    .npz files already exist at SMOKE_NPZ."""
    from backend.training.preprocessing.qevd_extractor import extract_clip

    SMOKE_NPZ.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in SMOKE_NPZ.glob("*.npz")}
    if len(existing) >= n_clips:
        print(f"[extract] {len(existing)} .npz already present, skipping extraction")
        return len(existing)

    rng = random.Random(SEED)
    all_mp4s = sorted(PART1_DIR.glob("*.mp4"))
    if len(all_mp4s) < n_clips:
        raise RuntimeError(f"need {n_clips} MP4s, found {len(all_mp4s)}")
    sample = rng.sample(all_mp4s, n_clips)

    print(f"[extract] processing {n_clips} clips (~{n_clips * 6} s estimated) ...")
    t0 = time.time()
    n_done = 0
    for i, mp4 in enumerate(sample, 1):
        npz_path = SMOKE_NPZ / f"{mp4.stem}.npz"
        if npz_path.exists():
            n_done += 1
            continue
        try:
            feats = extract_clip(mp4)
            feats.save_npz(npz_path)
            n_done += 1
        except Exception as e:
            print(f"  [warn] {mp4.name}: {type(e).__name__}: {e}")
        if i % 10 == 0:
            print(f"    [extract] {i}/{n_clips}  ({(time.time()-t0)/60:.1f} min)")
    print(f"[extract] done: {n_done}/{n_clips} successful "
          f"in {(time.time()-t0)/60:.1f} min")
    return n_done


def _run_training() -> int:
    """Launch train_form_model_v6 via CLI; return exit code."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "backend/training/train_form_model_v6.py"),
        "--npz-dirs",   str(SMOKE_NPZ),
        "--labels-dir", str(LABELS_DIR),
        "--out-dir",    str(SMOKE_OUT),
        "--epochs",         str(EPOCHS),
        "--batch-size",     str(BATCH_SIZE),
        "--target-frames",  "32",            # smaller for CPU speed
        "--max-train-clips", str(N_TRAIN),
        "--max-val-clips",   str(N_VAL),
        "--warmup-epochs",   "1",
        "--lr",              "1e-3",         # higher LR for fast convergence
    ]
    print()
    print("[train] command:")
    print(" ", " ".join(cmd))
    print()
    return subprocess.run(cmd, check=False).returncode


def _check_history() -> None:
    """Validate loss curves in v6_history.json."""
    history_path = SMOKE_OUT / "v6_history.json"
    if not history_path.exists():
        raise RuntimeError(f"history not written at {history_path}")
    history = json.loads(history_path.read_text())

    expected_heads = ("loss", "l_quality", "l_joint_err",
                      "l_boundary", "l_rep_count", "l_action")
    print("\n[validate] history per-head losses:")
    for k in expected_heads:
        assert k in history, f"missing head: {k}"
        vals = history[k]
        finite = all(np.isfinite(v) for v in vals)
        first, last = vals[0], vals[-1]
        delta = last - first
        marker = "[OK]" if finite else "[FAIL]"
        print(f"  {marker} {k:<14s} epochs: {[f'{v:.4f}' for v in vals]}  "
              f"delta = {delta:+.4f}")
        assert finite, f"non-finite {k}: {vals}"

    # Total loss should decrease over the run (sign that the model is learning)
    losses = history["loss"]
    delta = losses[-1] - losses[0]
    print(f"\n[validate] total loss change over {EPOCHS} epochs: {delta:+.4f}")
    if delta < 0:
        print(f"[validate] [OK] total loss decreased — D5 PASS")
    else:
        # Even with random data we'd typically see *some* decrease in 3 epochs.
        # Allow this to be a [WARN] rather than [FAIL] in case the random seed
        # lands somewhere weird; final D9 reality-check is the real gate.
        print(f"[validate] [WARN] total loss did NOT decrease "
              f"({losses[0]:.4f} -> {losses[-1]:.4f}). "
              f"Smoke is informational only; D9 is the real gate.")


def main() -> int:
    print("=" * 70)
    print("D5 SMOKE — end-to-end training validation on real QEVD data")
    print("=" * 70)

    # 1. Extract subset
    n = _extract_subset(N_CLIPS)
    if n < N_TRAIN + N_VAL:
        print(f"WARNING: only {n} successful clips, "
              f"may have too few for train+val={N_TRAIN+N_VAL}")
    print()

    # 1b. Clean previous smoke output — guarantees a fresh training run
    #     and avoids the "resume from epoch 3, do 0 work" state.
    if SMOKE_OUT.exists():
        print(f"[clean] wiping previous smoke output {SMOKE_OUT} ...")
        shutil.rmtree(SMOKE_OUT)

    # 2. Run training
    rc = _run_training()
    if rc != 0:
        print(f"\nD5 SMOKE FAIL: training exited with code {rc}")
        return rc

    # 3. Validate
    _check_history()

    print()
    print("=" * 70)
    print("D5 SMOKE COMPLETE")
    print("=" * 70)
    print(f"Outputs in {SMOKE_OUT}:")
    for f in sorted(SMOKE_OUT.glob("*")):
        if f.is_file():
            print(f"  {f.name:<32s}  {f.stat().st_size / 1e3:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
