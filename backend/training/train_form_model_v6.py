"""Supervised training for FitNova v6 (QEVD-trained ST-GCN).

Trains backend.training.models.st_gcn_v6.build_v6_model on labels derived
from QEVD by qevd_label_builder + qevd_dataset.

Master plan reference: section II.7.

Usage examples
--------------
D5 smoke (overfit ~200 clips, ~5 epochs, validates the loop):
    python backend/training/train_form_model_v6.py \\
        --npz-dirs backend/data/qevd_extracted/smoke_d5 \\
        --labels-dir backend/data/qevd_raw/fitcoach \\
        --out-dir backend/models/form_model_v6_smoke \\
        --max-train-clips 200 --max-val-clips 30 \\
        --epochs 5 --batch-size 16

D6 full Colab run (~5-8 hours on L4):
    python backend/training/train_form_model_v6.py \\
        --npz-dirs /content/drive/MyDrive/fitnova_v6/qevd_extracted/Part-1 \\
                   /content/drive/MyDrive/fitnova_v6/qevd_extracted/Part-2 \\
                   /content/drive/MyDrive/fitnova_v6/qevd_extracted/Part-3 \\
                   /content/drive/MyDrive/fitnova_v6/qevd_extracted/Part-4 \\
        --labels-dir /content/drive/MyDrive/fitnova_v6/labels \\
        --out-dir /content/drive/MyDrive/fitnova_v6/models/form_model_v6 \\
        --epochs 30 --batch-size 32

Outputs (in --out-dir):
    v6_supervised.weights.h5    final model weights
    v6_history.json             per-epoch loss/metric history
    qevd_exercise_map.json      exercise_name -> idx mapping (saved on first
                                  build, reused on resume)
    model_config.json           target_frames / n_exercises / etc.
    checkpoints/epoch_NN.weights.h5  rotating last-3 checkpoints (resume)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Loss config (master plan §II.7)
# ─────────────────────────────────────────────────────────────────────────────


LOSS_WEIGHTS = {
    "quality":   1.0,
    "joint_err": 2.0,    # primary signal — drives the D9 reality-check
    "boundary":  0.3,
    "rep_count": 0.2,
    "action":    0.5,
}


# ─────────────────────────────────────────────────────────────────────────────
# Multi-task model wrapper
# ─────────────────────────────────────────────────────────────────────────────


def _build_v6_compiled(
    target_frames: int,
    n_exercises: int,
    n_joint_groups: int,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    clipnorm: float = 1.0,
):
    """Build st_gcn_v6 + AdamW with cosine-decay schedule + custom train_step."""
    import tensorflow as tf
    from tensorflow import keras
    from backend.training.models.st_gcn_v6 import build_v6_model

    base = build_v6_model(
        target_frames=target_frames,
        n_exercises=n_exercises,
        n_joint_groups=n_joint_groups,
    )

    total_steps  = max(1, total_epochs * steps_per_epoch)
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    decay_steps  = max(1, total_steps - warmup_steps)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr,
        decay_steps=decay_steps,
        alpha=1e-2,                           # min lr = 1% of initial
        warmup_target=lr,
        warmup_steps=warmup_steps,
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=weight_decay,
        clipnorm=clipnorm,
    )

    return _V6Trainer(base), optimizer


class _V6Trainer:
    """Thin wrapper holding the base model + custom train/test step.

    NOT a keras.Model subclass — kept simple to avoid the 3-layer indirection
    that bit v5.2 (build_v5_model returned a keras.Model that was wrapped
    in a class that was wrapped in another keras.Model). Just hold the base
    and run train_step manually with our own loop in `fit_loop()` below.
    """

    def __init__(self, base):
        self.base = base

    @property
    def trainable_variables(self):
        return self.base.trainable_variables

    def __call__(self, inputs, training=False):
        return self.base(inputs, training=training)


def _multi_task_loss(outputs, targets):
    """Return total_loss + per-head dict for logging."""
    import tensorflow as tf
    from tensorflow import keras
    eps = 1e-7

    quality_pred = outputs["quality"]
    quality_tgt  = targets["quality"]
    l_q = tf.reduce_mean(tf.square(quality_pred - quality_tgt))

    je_pred = tf.clip_by_value(outputs["joint_err"], eps, 1.0 - eps)
    je_tgt  = targets["joint_err"]
    l_j = tf.reduce_mean(
        keras.losses.binary_crossentropy(je_tgt, je_pred)
    )

    b_pred = tf.clip_by_value(outputs["boundary"], eps, 1.0 - eps)
    b_tgt  = targets["boundary"]
    l_b = tf.reduce_mean(
        keras.losses.binary_crossentropy(b_tgt, b_pred)
    )

    c_pred = outputs["rep_count"]
    c_tgt  = targets["rep_count"]
    l_c = tf.keras.losses.huber(c_tgt, c_pred)
    l_c = tf.reduce_mean(l_c)

    a_logits = outputs["action"]
    a_tgt    = targets["action"]
    l_a = tf.reduce_mean(
        keras.losses.sparse_categorical_crossentropy(a_tgt, a_logits)
    )

    total = (
        LOSS_WEIGHTS["quality"]   * l_q
        + LOSS_WEIGHTS["joint_err"] * l_j
        + LOSS_WEIGHTS["boundary"]  * l_b
        + LOSS_WEIGHTS["rep_count"] * l_c
        + LOSS_WEIGHTS["action"]    * l_a
    )
    return total, {
        "loss":      total,
        "l_quality":   l_q,
        "l_joint_err": l_j,
        "l_boundary":  l_b,
        "l_rep_count": l_c,
        "l_action":    l_a,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training loop with checkpoint + resume
# ─────────────────────────────────────────────────────────────────────────────


def fit_loop(
    trainer,
    optimizer,
    train_ds,
    val_ds,
    epochs: int,
    out_dir: Path,
    initial_epoch: int = 0,
    history_acc: Optional[dict] = None,
):
    import tensorflow as tf

    history_acc = dict(history_acc) if history_acc else {}
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    @tf.function(reduce_retracing=True)
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            outputs = trainer(inputs, training=True)
            total, parts = _multi_task_loss(outputs, targets)
        grads = tape.gradient(total, trainer.trainable_variables)
        optimizer.apply_gradients(zip(grads, trainer.trainable_variables))
        return parts

    @tf.function(reduce_retracing=True)
    def val_step(inputs, targets):
        outputs = trainer(inputs, training=False)
        _, parts = _multi_task_loss(outputs, targets)
        return parts

    # Per-batch progress: print a heartbeat line every PRINT_EVERY batches
    # so users get visible progress in long runs (especially via Colab's
    # subprocess.Popen pipe, where tqdm's \r redraws don't render reliably).
    PRINT_EVERY = 100
    for epoch in range(initial_epoch, epochs):
        epoch_t0 = time.time()
        train_metrics: dict = {}
        n_train_batches = 0
        last_t = time.time()
        for inputs, targets in train_ds:
            parts = train_step(inputs, targets)
            for k, v in parts.items():
                train_metrics.setdefault(k, 0.0)
                train_metrics[k] += float(v.numpy())
            n_train_batches += 1
            if n_train_batches % PRINT_EVERY == 0:
                t_now = time.time()
                rate = PRINT_EVERY / max(1e-6, t_now - last_t)
                last_t = t_now
                running = {k: v / n_train_batches
                            for k, v in train_metrics.items()}
                loss_str = "  ".join(
                    f"{k}={running[k]:.3f}"
                    for k in ("total", "quality", "joint_err",
                              "boundary", "rep_count", "action")
                    if k in running
                )
                print(f"  ep{epoch+1:02d} step {n_train_batches:5d}  "
                      f"{rate:5.1f} batch/s  {loss_str}",
                      flush=True)
        if n_train_batches:
            train_metrics = {k: v / n_train_batches
                             for k, v in train_metrics.items()}

        # Val pass — one heartbeat at start, one at end (val is short)
        print(f"  ep{epoch+1:02d} val pass starting "
              f"({n_train_batches} train batches done) ...",
              flush=True)
        val_metrics: dict = {}
        n_val_batches = 0
        for inputs, targets in val_ds:
            parts = val_step(inputs, targets)
            for k, v in parts.items():
                val_metrics.setdefault(k, 0.0)
                val_metrics[k] += float(v.numpy())
            n_val_batches += 1
        if n_val_batches:
            val_metrics = {k: v / n_val_batches for k, v in val_metrics.items()}

        # Log + persist
        elapsed = time.time() - epoch_t0
        train_str = "  ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
        val_str   = "  ".join(f"val_{k}={v:.4f}"
                                for k, v in val_metrics.items())
        print(f"[epoch {epoch+1:3d}/{epochs}] {elapsed:.1f}s  "
              f"train: {train_str}  val: {val_str}")

        for k, v in train_metrics.items():
            history_acc.setdefault(k, []).append(v)
        for k, v in val_metrics.items():
            history_acc.setdefault(f"val_{k}", []).append(v)

        # Save checkpoint + history
        ckpt_path = ckpt_dir / f"epoch_{epoch+1:02d}.weights.h5"
        trainer.base.save_weights(str(ckpt_path))
        with open(out_dir / "v6_history.json", "w") as f:
            json.dump(history_acc, f, indent=2)
        # Keep only last 3 checkpoints
        files = sorted(p for p in ckpt_dir.glob("epoch_*.weights.h5"))
        for old in files[:-3]:
            try: old.unlink()
            except OSError: pass

    return history_acc


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _find_latest_checkpoint(ckpt_dir: Path) -> int:
    """Find epoch_NN from filenames like 'epoch_NN.weights.h5'.

    Path.stem strips only the LAST extension, so for 'epoch_01.weights.h5'
    we get 'epoch_01.weights' — parse the leading 'epoch_NN' chunk
    instead of relying on stem.
    """
    if not ckpt_dir.is_dir():
        return 0
    files = list(ckpt_dir.glob("epoch_*.weights.h5"))
    nums: list[int] = []
    for f in files:
        prefix = f.name.split(".", 1)[0]    # 'epoch_NN'
        parts = prefix.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            nums.append(int(parts[1]))
    return max(nums) if nums else 0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz-dirs", type=Path, nargs="+", required=True,
                        help="One or more directories containing <clip_id>.npz files. "
                             "Searched in order per clip until a hit. "
                             "Pass all 4 Part-N dirs for full training.")
    parser.add_argument("--labels-dir", type=Path, required=True,
                        help="Directory containing the QEVD label JSONs "
                             "(feedbacks_short_clips.json, fine_grained_labels.json, "
                             "fine_grained_labels_with_worker_ids.json).")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for weights, history, exercise map, etc.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--target-frames", type=int, default=64)
    parser.add_argument("--max-train-clips", type=int, default=0,
                        help="If > 0, sample this many train clips (for D5 smoke).")
    parser.add_argument("--max-val-clips", type=int, default=0,
                        help="If > 0, cap val clips for fast eval.")
    parser.add_argument("--val-frac", type=float, default=0.09,
                        help="Fraction of train participants held out for val.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf
    from backend.training.preprocessing.qevd_dataset import (
        DEFAULT_N_EXERCISES, DEFAULT_N_JOINT_GROUPS,
        build_exercise_map, build_qevd_splits, load_exercise_map,
        make_clip_dataset, save_exercise_map,
    )
    from backend.training.preprocessing.qevd_label_builder import QEVDLabels

    # ── 1. Load labels ─────────────────────────────────────────────────────
    fbs_path  = args.labels_dir / "feedbacks_short_clips.json"
    fine_path = args.labels_dir / "fine_grained_labels.json"
    workers_path = args.labels_dir / "fine_grained_labels_with_worker_ids.json"
    print(f"[1] loading labels from {args.labels_dir} ...")
    labels = QEVDLabels.from_files(
        feedbacks_short_clips_path=fbs_path  if fbs_path.exists() else None,
        fine_grained_labels_path  =fine_path if fine_path.exists() else None,
        worker_ids_path           =workers_path if workers_path.exists() else None,
    )
    stats = labels.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # ── 2. Build splits + exercise map ─────────────────────────────────────
    print(f"\n[2] building splits + exercise map ...")
    splits = build_qevd_splits(
        labels.clip_to_split, labels.clip_to_human,
        val_frac=args.val_frac, seed=args.seed,
    )
    print(f"  splits: train={len(splits['train']):,}  val={len(splits['val']):,}  "
          f"in_domain_test={len(splits['in_domain_test']):,}")

    ex_map_path = args.out_dir / "qevd_exercise_map.json"
    if ex_map_path.exists():
        ex_map = load_exercise_map(ex_map_path)
        print(f"  loaded existing exercise map ({len(ex_map)} entries)")
    else:
        ex_map = build_exercise_map(labels)
        save_exercise_map(ex_map, ex_map_path)
        print(f"  built new exercise map ({len(ex_map)} entries) -> {ex_map_path}")

    # ── 3. Filter to clips that actually have .npz on disk, then subsample ─
    rng = np.random.default_rng(args.seed)

    def _filter_to_existing_npz(ids, dirs):
        # Return only clip_ids that have a corresponding .npz in any dir.
        out = []
        for cid in ids:
            cid_padded = cid.zfill(8) if cid.isdigit() else cid
            for d in dirs:
                if (d / f"{cid_padded}.npz").exists():
                    out.append(cid); break
        return out

    train_avail = _filter_to_existing_npz(sorted(splits["train"]), args.npz_dirs)
    val_avail   = _filter_to_existing_npz(sorted(splits["val"]),   args.npz_dirs)
    print(f"  train clips with .npz on disk: {len(train_avail):,} / {len(splits['train']):,}")
    print(f"  val   clips with .npz on disk: {len(val_avail):,} / {len(splits['val']):,}")
    if not train_avail:
        raise RuntimeError(
            "No train .npz files found in any of --npz-dirs. "
            f"Searched: {[str(d) for d in args.npz_dirs]}"
        )
    if not val_avail:
        # Fallback: hold out a fraction of train_avail for val
        n_holdout = max(1, len(train_avail) // 10)
        rng.shuffle(train_avail)
        val_avail   = train_avail[:n_holdout]
        train_avail = train_avail[n_holdout:]
        print(f"  no val clips on disk; held out {n_holdout} train clips for val")

    train_ids = list(train_avail)
    val_ids   = list(val_avail)
    if args.max_train_clips > 0 and args.max_train_clips < len(train_ids):
        idx = rng.choice(len(train_ids), size=args.max_train_clips, replace=False)
        train_ids = [train_ids[i] for i in idx]
        print(f"  subsampled train -> {len(train_ids)} clips (--max-train-clips)")
    if args.max_val_clips > 0 and args.max_val_clips < len(val_ids):
        idx = rng.choice(len(val_ids), size=args.max_val_clips, replace=False)
        val_ids = [val_ids[i] for i in idx]
        print(f"  subsampled val -> {len(val_ids)} clips (--max-val-clips)")

    # ── 4. Build tf.data datasets ──────────────────────────────────────────
    print(f"\n[3] building tf.data datasets (batch={args.batch_size}) ...")
    train_ds = make_clip_dataset(
        npz_dirs=args.npz_dirs, clip_ids=train_ids,
        labels=labels, exercise_map=ex_map,
        target_frames=args.target_frames,
        n_joint_groups=DEFAULT_N_JOINT_GROUPS,
        batch_size=args.batch_size, shuffle=True, seed=args.seed,
    )
    val_ds = make_clip_dataset(
        npz_dirs=args.npz_dirs, clip_ids=val_ids,
        labels=labels, exercise_map=ex_map,
        target_frames=args.target_frames,
        n_joint_groups=DEFAULT_N_JOINT_GROUPS,
        batch_size=args.batch_size, shuffle=False, seed=args.seed,
    )

    # Roughly estimate steps/epoch for the LR schedule
    steps_per_epoch = max(1, len(train_ids) // args.batch_size)
    print(f"  steps_per_epoch ~ {steps_per_epoch}")

    # ── 5. Build + compile model ───────────────────────────────────────────
    print(f"\n[4] building model + optimizer ...")
    trainer, optimizer = _build_v6_compiled(
        target_frames=args.target_frames,
        n_exercises=DEFAULT_N_EXERCISES,
        n_joint_groups=DEFAULT_N_JOINT_GROUPS,
        lr=args.lr, weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs, steps_per_epoch=steps_per_epoch,
    )
    n_params = trainer.base.count_params()
    print(f"  v6 model params: {n_params:,}")

    # Save model_config.json (consumed by form_analyzer.py at integration day)
    model_config = {
        "target_frames":   args.target_frames,
        "n_canonical_joints": 15,
        "n_pose_channels": 4,
        "n_angular":       22,
        "n_exercises":     DEFAULT_N_EXERCISES,
        "n_joint_groups":  DEFAULT_N_JOINT_GROUPS,
        "version":         "v6",
        "source":          "qevd_v6",
    }
    with open(args.out_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    # ── 6. Resume from latest checkpoint if any ─────────────────────────────
    ckpt_dir = args.out_dir / "checkpoints"
    initial_epoch = _find_latest_checkpoint(ckpt_dir)
    history_acc = {}
    if initial_epoch > 0:
        last_ckpt = ckpt_dir / f"epoch_{initial_epoch:02d}.weights.h5"
        print(f"\n[5] resuming from epoch {initial_epoch} ({last_ckpt})")
        trainer.base.load_weights(str(last_ckpt))
        hist_path = args.out_dir / "v6_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                history_acc = json.load(f)
    else:
        print(f"\n[5] fresh training run")

    # ── 7. Fit ─────────────────────────────────────────────────────────────
    print(f"\n[6] starting fit_loop for {args.epochs - initial_epoch} epochs ...")
    fit_loop(
        trainer, optimizer, train_ds, val_ds,
        epochs=args.epochs, out_dir=args.out_dir,
        initial_epoch=initial_epoch, history_acc=history_acc,
    )

    # ── 8. Save final weights ──────────────────────────────────────────────
    final_path = args.out_dir / "v6_supervised.weights.h5"
    trainer.base.save_weights(str(final_path))
    print(f"\n[7] saved final weights -> {final_path}")
    print(f"    history -> {args.out_dir / 'v6_history.json'}")
    print(f"    config  -> {args.out_dir / 'model_config.json'}")
    print(f"    exer-map -> {ex_map_path}")


if __name__ == "__main__":
    main()
