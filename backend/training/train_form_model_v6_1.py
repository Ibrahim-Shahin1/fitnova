"""Supervised training for FitNova v6.1 — paper-faithful multi-label.

Forks ``train_form_model_v6.py`` to match the v6.1 architecture
(``st_gcn_v6_1.build_v6_1_model``): single Dense(num_classes, sigmoid)
head over the cleaned QEVD variation taxonomy. No quality scalar, no
joint_err head, no boundary head, no separate action softmax.

Loss
----
Class-weighted BCE on logits. Per-class weights come from
``QEVDClassSpace.inverse_frequency_weights()`` (Cui et al. 2019
effective-number formulation, clipped to [0.1, 10.0]). The smallest
classes (~50 clips) get roughly an order of magnitude more weight than
the most populous (~3,400 clips).

Metrics (computed on val at end of each epoch)
----------------------------------------------
- val_loss               (BCE)
- val_macro_f1           per-class F1 at prob > 0.5, macro-averaged
- val_top1_acc           Did the top-1 prediction match any true label?
- val_top3_recall        Fraction of true labels in the top-3 predicted
- val_active_classes     Number of distinct classes that fired >= 0.5
                         in val (drops to <10 means class collapse)

Usage examples
--------------
D5 smoke (overfit ~1k clips, ~5 epochs, validates the loop):
    python -m backend.training.train_form_model_v6_1 \\
        --npz-dirs backend/data/qevd_extracted/smoke_d5 \\
        --labels-dir backend/data/qevd_raw/fitcoach \\
        --out-dir backend/models/form_model_v6_1_smoke \\
        --max-train-clips 1000 --max-val-clips 100 \\
        --epochs 5 --batch-size 16

Full Colab L4 run (~10-15 h):
    python -m backend.training.train_form_model_v6_1 \\
        --npz-dirs /content/drive/MyDrive/fitnova_v6/qevd_extracted/Part-1 \\
                   /content/drive/MyDrive/fitnova_v6/qevd_extracted/Part-2 \\
                   /content/drive/MyDrive/fitnova_v6/qevd_extracted/Part-3 \\
                   /content/drive/MyDrive/fitnova_v6/qevd_extracted/Part-4 \\
        --labels-dir /content/drive/MyDrive/fitnova_v6/labels \\
        --out-dir   /content/drive/MyDrive/fitnova_v6_1/models/form_model_v6_1 \\
        --epochs 30 --batch-size 32

Outputs (in --out-dir):
    v6_1_supervised.weights.h5         final weights
    v6_1_history.json                  per-epoch metrics
    model_config_v6_1.json             reproducibility metadata
    checkpoints/epoch_NN.weights.h5    rotating last-3 checkpoints
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Loss + metrics
# ─────────────────────────────────────────────────────────────────────────────


def _weighted_bce_loss(logits, targets, pos_weight):
    """Weighted binary cross-entropy on logits.

    pos_weight is the per-class weight tensor of shape (num_classes,).
    The weight is applied to BOTH positive and negative examples per
    class (a simple multiplicative scaling), which is the standard
    Cui-et-al style class-weighted BCE for long-tail multi-label.
    """
    import tensorflow as tf
    # Use the from-logits BCE for numerical stability.
    per_class = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=targets, logits=logits,
    )  # (B, C)
    return tf.reduce_mean(per_class * pos_weight)


def _accumulate_val_metrics(
    all_probs: np.ndarray,
    all_targets: np.ndarray,
    *,
    threshold: float = 0.5,
    top_k: int = 3,
) -> dict:
    """Compute multi-label diagnostics from stacked val predictions.

    Args:
        all_probs:   (N, C) sigmoid outputs.
        all_targets: (N, C) ground-truth multi-hot.
    """
    preds = (all_probs >= threshold).astype(np.float32)
    tgts = (all_targets >= 0.5).astype(np.float32)

    # Macro F1 across classes with any positive support in val.
    tp = (preds * tgts).sum(axis=0)        # (C,)
    fp = (preds * (1 - tgts)).sum(axis=0)  # (C,)
    fn = ((1 - preds) * tgts).sum(axis=0)  # (C,)
    support = tgts.sum(axis=0)             # (C,)
    has_support = support > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where((tp + fp) > 0, tp / (tp + fp + 1e-8), 0.0)
        recall    = np.where((tp + fn) > 0, tp / (tp + fn + 1e-8), 0.0)
        f1        = np.where(
            (precision + recall) > 0,
            2.0 * precision * recall / (precision + recall + 1e-8),
            0.0,
        )
    macro_f1 = float(f1[has_support].mean()) if has_support.any() else 0.0

    # Top-1: did the top-1 prediction match ANY true label per clip?
    top1_idx = all_probs.argmax(axis=1)               # (N,)
    top1_hit = tgts[np.arange(len(top1_idx)), top1_idx] > 0
    top1_acc = float(top1_hit.mean()) if len(top1_hit) else 0.0

    # Top-K recall: fraction of TRUE labels recovered in top-K predictions.
    topk_idx = np.argsort(-all_probs, axis=1)[:, :top_k]    # (N, K)
    n_clips, n_classes = tgts.shape
    topk_hit_per_clip = []
    for i in range(n_clips):
        true_positives = np.where(tgts[i] > 0)[0]
        if len(true_positives) == 0:
            continue
        in_topk = np.isin(true_positives, topk_idx[i]).sum()
        topk_hit_per_clip.append(in_topk / len(true_positives))
    topk_recall = float(np.mean(topk_hit_per_clip)) if topk_hit_per_clip else 0.0

    # Class collapse diagnostic: how many distinct classes fired >= threshold?
    fired_classes = int((preds.sum(axis=0) > 0).sum())

    return {
        "macro_f1":       macro_f1,
        "top1_acc":       top1_acc,
        f"top{top_k}_recall": topk_recall,
        "active_classes": fired_classes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model + optimizer
# ─────────────────────────────────────────────────────────────────────────────


def _build_v6_1_compiled(
    num_classes: int,
    n_exercises: int,
    target_frames: int,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    clipnorm: float = 1.0,
):
    """Build v6.1 model + AdamW with cosine-decay schedule."""
    from tensorflow import keras
    from backend.training.models.st_gcn_v6_1 import build_v6_1_model

    base = build_v6_1_model(
        num_classes=num_classes,
        n_exercises=n_exercises,
        target_frames=target_frames,
    )

    total_steps  = max(1, total_epochs * steps_per_epoch)
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    decay_steps  = max(1, total_steps - warmup_steps)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr,
        decay_steps=decay_steps,
        alpha=1e-2,
        warmup_target=lr,
        warmup_steps=warmup_steps,
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=weight_decay,
        clipnorm=clipnorm,
    )
    return base, optimizer


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────


def fit_loop(
    model,
    optimizer,
    train_ds,
    val_ds,
    pos_weight_np: np.ndarray,
    epochs: int,
    out_dir: Path,
    initial_epoch: int = 0,
    history_acc: Optional[dict] = None,
):
    import tensorflow as tf

    history_acc = dict(history_acc) if history_acc else {}
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pos_weight_tf = tf.constant(pos_weight_np, dtype=tf.float32)

    @tf.function(reduce_retracing=True)
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            out = model(inputs, training=True)
            loss = _weighted_bce_loss(out["logits"], targets, pos_weight_tf)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function(reduce_retracing=True)
    def val_step(inputs, targets):
        out = model(inputs, training=False)
        loss = _weighted_bce_loss(out["logits"], targets, pos_weight_tf)
        return loss, out["probs"]

    PRINT_EVERY = 50
    for epoch in range(initial_epoch, epochs):
        epoch_t0 = time.time()
        train_loss_sum = 0.0
        n_train_batches = 0
        last_t = time.time()
        for inputs, targets in train_ds:
            loss = train_step(inputs, targets)
            train_loss_sum += float(loss.numpy())
            n_train_batches += 1
            if n_train_batches % PRINT_EVERY == 0:
                t_now = time.time()
                rate = PRINT_EVERY / max(1e-6, t_now - last_t)
                last_t = t_now
                running_loss = train_loss_sum / n_train_batches
                print(f"  ep{epoch+1:02d} step {n_train_batches:5d}  "
                      f"{rate:5.1f} batch/s  loss={running_loss:.4f}",
                      flush=True)
        train_loss = train_loss_sum / max(1, n_train_batches)

        # Val pass + metric accumulation
        print(f"  ep{epoch+1:02d} val pass starting "
              f"({n_train_batches} train batches done) ...",
              flush=True)
        val_loss_sum = 0.0
        n_val_batches = 0
        all_probs = []
        all_targets = []
        for inputs, targets in val_ds:
            loss, probs = val_step(inputs, targets)
            val_loss_sum += float(loss.numpy())
            n_val_batches += 1
            all_probs.append(probs.numpy())
            all_targets.append(targets.numpy() if hasattr(targets, "numpy") else np.asarray(targets))
        val_loss = val_loss_sum / max(1, n_val_batches)
        if all_probs:
            all_probs_np   = np.concatenate(all_probs,   axis=0)
            all_targets_np = np.concatenate(all_targets, axis=0)
            metrics = _accumulate_val_metrics(all_probs_np, all_targets_np)
        else:
            metrics = {"macro_f1": 0.0, "top1_acc": 0.0,
                       "top3_recall": 0.0, "active_classes": 0}

        # Log + persist
        elapsed = time.time() - epoch_t0
        m_str = "  ".join(f"{k}={v:.4f}" if isinstance(v, float)
                            else f"{k}={v}" for k, v in metrics.items())
        print(f"[epoch {epoch+1:3d}/{epochs}] {elapsed:.1f}s  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  {m_str}")

        history_acc.setdefault("train_loss", []).append(train_loss)
        history_acc.setdefault("val_loss",   []).append(val_loss)
        for k, v in metrics.items():
            history_acc.setdefault(f"val_{k}", []).append(v)

        ckpt_path = ckpt_dir / f"epoch_{epoch+1:02d}.weights.h5"
        model.save_weights(str(ckpt_path))
        with open(out_dir / "v6_1_history.json", "w") as f:
            json.dump(history_acc, f, indent=2)
        # Rotate: keep last 3
        files = sorted(p for p in ckpt_dir.glob("epoch_*.weights.h5"))
        for old in files[:-3]:
            try: old.unlink()
            except OSError: pass

    return history_acc


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _find_latest_checkpoint(ckpt_dir: Path) -> int:
    if not ckpt_dir.is_dir():
        return 0
    files = list(ckpt_dir.glob("epoch_*.weights.h5"))
    nums: list[int] = []
    for f in files:
        prefix = f.name.split(".", 1)[0]
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
                        help="Directories holding <clip_id>.npz files.")
    parser.add_argument("--labels-dir", type=Path, required=True,
                        help="Directory with the QEVD label JSONs.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--target-frames", type=int, default=64)
    parser.add_argument("--max-train-clips", type=int, default=0)
    parser.add_argument("--max-val-clips", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.09)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from backend.training.preprocessing.qevd_class_space import get_default_class_space
    from backend.training.preprocessing.qevd_dataset import build_qevd_splits
    from backend.training.preprocessing.qevd_dataset_v6_1 import make_v6_1_clip_dataset
    from backend.training.preprocessing.qevd_label_builder import QEVDLabels

    cs = get_default_class_space()
    print(f"[v6.1] class space: {cs.num_classes} classes, "
          f"{cs.num_prefixes} prefixes")

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
    for k, v in labels.stats().items():
        print(f"  {k}: {v}")

    # ── 2. Build splits ────────────────────────────────────────────────────
    print(f"\n[2] building splits ...")
    splits = build_qevd_splits(
        labels.clip_to_split, labels.clip_to_human,
        val_frac=args.val_frac, seed=args.seed,
    )
    print(f"  train={len(splits['train']):,}  val={len(splits['val']):,}  "
          f"in_domain_test={len(splits['in_domain_test']):,}")

    rng = np.random.default_rng(args.seed)

    def _filter_to_existing_npz(ids, dirs):
        out = []
        for cid in ids:
            cid_padded = cid.zfill(8) if cid.isdigit() else cid
            for d in dirs:
                if (d / f"{cid_padded}.npz").exists():
                    out.append(cid); break
        return out

    train_avail = _filter_to_existing_npz(sorted(splits["train"]), args.npz_dirs)
    val_avail   = _filter_to_existing_npz(sorted(splits["val"]),   args.npz_dirs)
    print(f"  train clips with .npz: {len(train_avail):,} / {len(splits['train']):,}")
    print(f"  val   clips with .npz: {len(val_avail):,} / {len(splits['val']):,}")
    if not train_avail:
        raise RuntimeError("No train .npz found.")
    if not val_avail:
        n_hold = max(1, len(train_avail) // 10)
        rng.shuffle(train_avail)
        val_avail   = train_avail[:n_hold]
        train_avail = train_avail[n_hold:]
        print(f"  no val on disk; held out {n_hold} train for val")

    train_ids = list(train_avail)
    val_ids   = list(val_avail)
    if args.max_train_clips > 0 and args.max_train_clips < len(train_ids):
        idx = rng.choice(len(train_ids), size=args.max_train_clips, replace=False)
        train_ids = [train_ids[i] for i in idx]
        print(f"  subsampled train -> {len(train_ids)}")
    if args.max_val_clips > 0 and args.max_val_clips < len(val_ids):
        idx = rng.choice(len(val_ids), size=args.max_val_clips, replace=False)
        val_ids = [val_ids[i] for i in idx]
        print(f"  subsampled val -> {len(val_ids)}")

    # ── 3. Build datasets ──────────────────────────────────────────────────
    print(f"\n[3] building v6.1 tf.data datasets (batch={args.batch_size}) ...")
    train_ds = make_v6_1_clip_dataset(
        npz_dirs=args.npz_dirs, clip_ids=train_ids,
        labels=labels, class_space=cs,
        target_frames=args.target_frames,
        batch_size=args.batch_size, shuffle=True, seed=args.seed,
    )
    val_ds = make_v6_1_clip_dataset(
        npz_dirs=args.npz_dirs, clip_ids=val_ids,
        labels=labels, class_space=cs,
        target_frames=args.target_frames,
        batch_size=args.batch_size, shuffle=False, seed=args.seed,
    )
    steps_per_epoch = max(1, len(train_ids) // args.batch_size)
    print(f"  steps_per_epoch ~ {steps_per_epoch}")

    # ── 4. Build model + class weights ─────────────────────────────────────
    print(f"\n[4] building model + class weights ...")
    model, optimizer = _build_v6_1_compiled(
        num_classes=cs.num_classes,
        n_exercises=cs.num_prefixes,
        target_frames=args.target_frames,
        lr=args.lr, weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs, steps_per_epoch=steps_per_epoch,
    )
    pos_weight = cs.inverse_frequency_weights()
    print(f"  model params:  {model.count_params():,}")
    print(f"  class weights: min={pos_weight.min():.3f}  max={pos_weight.max():.3f}  "
          f"mean={pos_weight.mean():.3f}")

    model_config = {
        "version":            "v6.1",
        "source":             "qevd_v6_1",
        "target_frames":      args.target_frames,
        "n_canonical_joints": 15,
        "n_pose_channels":    4,
        "n_angular":          22,
        "num_classes":        int(cs.num_classes),
        "num_prefixes":       int(cs.num_prefixes),
        "class_space_version": cs.version,
    }
    with open(args.out_dir / "model_config_v6_1.json", "w") as f:
        json.dump(model_config, f, indent=2)

    # ── 5. Resume ──────────────────────────────────────────────────────────
    ckpt_dir = args.out_dir / "checkpoints"
    initial_epoch = _find_latest_checkpoint(ckpt_dir)
    history_acc: dict = {}
    if initial_epoch > 0:
        last_ckpt = ckpt_dir / f"epoch_{initial_epoch:02d}.weights.h5"
        print(f"\n[5] resuming from epoch {initial_epoch} ({last_ckpt})")
        model.load_weights(str(last_ckpt))
        hist_path = args.out_dir / "v6_1_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                history_acc = json.load(f)
    else:
        print(f"\n[5] fresh training run")

    # ── 6. Fit ─────────────────────────────────────────────────────────────
    print(f"\n[6] starting fit_loop for {args.epochs - initial_epoch} epochs ...")
    fit_loop(
        model, optimizer, train_ds, val_ds, pos_weight,
        epochs=args.epochs, out_dir=args.out_dir,
        initial_epoch=initial_epoch, history_acc=history_acc,
    )

    # ── 7. Save final ──────────────────────────────────────────────────────
    final_path = args.out_dir / "v6_1_supervised.weights.h5"
    model.save_weights(str(final_path))
    print(f"\n[7] saved final weights -> {final_path}")
    print(f"    history  -> {args.out_dir / 'v6_1_history.json'}")
    print(f"    config   -> {args.out_dir / 'model_config_v6_1.json'}")


if __name__ == "__main__":
    main()
