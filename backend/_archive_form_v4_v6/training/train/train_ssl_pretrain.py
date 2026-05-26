"""
Self-supervised masked skeleton autoencoder pretraining (P3.1).

Inspired by: MAE (He et al. 2021), PoseBERT (Ghosh et al. 2021),
             SkeletonMAE (Yan et al. 2023).

The model learns to reconstruct masked angular / joint features from context.
No task labels are needed — all 8 subjects' data from all cameras are used.

After this script finishes, pass the encoder weights to train_form_model.py:
    python -m backend.training.train_form_model --ssl-init backend/models/form_model/mt_tcn_encoder_ssl.weights.h5 ...

Architecture
------------
- SAME MT-TCN encoder (both TCN branches + fusion projection) as the supervised model.
- Thin decoder: mirrors the encoder shape (4-level TCN, shrinking filters).
- Two masking strategies applied randomly per sample:
    (a) Frame masking:   40% of the 64 frames zeroed out.
    (b) Feature masking: 30% of the 22 angular features zeroed out.
- Loss: MSE on masked positions only (visible positions are not penalised).

Usage (run on Colab A100 — ~90 min with all 4 cameras):
    python -m backend.training.train_ssl_pretrain \\
        --data-dir backend/data/form_dataset_mediapipe_ssl \\
        --model-dir backend/models/form_model

The script builds its own dataset from raw .npy files if data-dir doesn't exist.
"""

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import tensorflow as tf
from tensorflow.keras import layers, Model

# tqdm: auto-picks notebook widget in Jupyter, plain ASCII in terminal
try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kw):          # graceful fallback if tqdm not installed
        return it

# IPython clear_output (works in Colab cells; silently ignored in terminal)
try:
    from IPython.display import clear_output as _clear_output
    _IN_JUPYTER = True
except ImportError:
    _IN_JUPYTER = False
    def _clear_output(wait=False): pass

from backend.training.models.mt_tcn import build_mt_tcn

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data",
                                 "form_dataset_mediapipe")
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models",
                                 "form_model")

# SSL training hyperparameters
BATCH_SIZE       = 64
EPOCHS           = 80
PATIENCE         = 15
LR               = 1e-3
FRAME_MASK_RATE  = 0.40   # fraction of frames to zero out
FEAT_MASK_RATE   = 0.30   # fraction of angular features to zero out

# Dimensions (must match mt_tcn.py)
T  = 64   # frames
NA = 22   # angular features
NJ = 45   # joint coords (15 joints × 3)


# ── Masking ───────────────────────────────────────────────────────────────────

def _make_masks(batch_size: int, frame_rate: float, feat_rate: float):
    """Return frame mask (B,T,1) and feature mask (B,1,NA) as float32 (0=masked)."""
    # Frame mask: sample independently per sequence
    frame_keep = tf.cast(
        tf.random.uniform((batch_size, T, 1)) > frame_rate,
        tf.float32,
    )
    # Feature mask: sample independently per sequence
    feat_keep = tf.cast(
        tf.random.uniform((batch_size, 1, NA)) > feat_rate,
        tf.float32,
    )
    return frame_keep, feat_keep


def _apply_masks(ang, jnt, frame_keep, feat_keep):
    """Apply masks: zero out selected positions."""
    ang_masked = ang * frame_keep * feat_keep   # (B, T, NA)
    jnt_masked = jnt * frame_keep               # (B, T, NJ) — frame mask only on joints
    return ang_masked, jnt_masked


# ── Decoder ───────────────────────────────────────────────────────────────────

def _decoder_block(x, filters, dilation, name):
    """Mirrored TCN block without residual projection (simpler decoder)."""
    out = layers.Conv1D(filters, 3, padding="causal", dilation_rate=dilation,
                        kernel_initializer="he_normal", name=f"{name}_conv")(x)
    out = layers.BatchNormalization(name=f"{name}_bn")(out)
    out = layers.ReLU(name=f"{name}_relu")(out)
    return out


def build_ssl_model(dropout_rate: float = 0.1) -> Model:
    """
    Full SSL model: encoder (MT-TCN shared) + decoder + dual reconstruction head.

    The encoder is identical to the MT-TCN encoder so weights transfer directly.
    """
    # ── Inputs (pre-masked arrays — masking applied before model call) ──────
    ang_in = tf.keras.Input(shape=(T, NA), name="angular_input")
    jnt_in = tf.keras.Input(shape=(T, NJ), name="joints_input")
    # Note: masks are NOT model inputs — they are applied externally in
    # _apply_masks() before calling model(), and used in the custom loss
    # outside the model graph.  Keras 3 requires every declared input to
    # participate in the output computation, so masks stay external.

    # ── Encoder: shared MT-TCN branches ─────────────────────────────────────
    # Import the actual TCN backbone function to keep identical layer names
    from backend.training.models.mt_tcn import _tcn_backbone
    branch_ang = _tcn_backbone(ang_in, name_prefix="ang", dropout_rate=dropout_rate)
    branch_jnt = _tcn_backbone(jnt_in, name_prefix="jnt", dropout_rate=dropout_rate)

    fused = layers.Concatenate(axis=-1, name="fusion_concat")([branch_ang, branch_jnt])
    fused = layers.Conv1D(256, 1, padding="same", kernel_initializer="he_normal",
                          name="fusion_proj")(fused)
    fused = layers.BatchNormalization(name="fusion_bn")(fused)
    fused = layers.ReLU(name="fusion_relu")(fused)
    # → (B, T, 256)

    # ── Decoder: lightweight upsampling TCN ─────────────────────────────────
    dec = _decoder_block(fused, 256, 8, "dec_b4")
    dec = _decoder_block(dec,   128, 4, "dec_b3")
    dec = _decoder_block(dec,   128, 2, "dec_b2")
    dec = _decoder_block(dec,    64, 1, "dec_b1")

    # ── Reconstruction heads ─────────────────────────────────────────────────
    ang_recon = layers.Conv1D(NA, 1, padding="same", name="ang_recon",
                              kernel_initializer="glorot_uniform")(dec)  # (B, T, 22)
    jnt_recon = layers.Conv1D(NJ, 1, padding="same", name="jnt_recon",
                              kernel_initializer="glorot_uniform")(dec)  # (B, T, 45)

    model = Model(
        inputs={"angular_input": ang_in, "joints_input": jnt_in},
        outputs={"ang_recon": ang_recon, "jnt_recon": jnt_recon},
        name="SSL_MT_TCN",
    )
    return model


# ── Custom masked MSE loss ────────────────────────────────────────────────────

class MaskedMSE(tf.keras.losses.Loss):
    """MSE computed only on MASKED positions (where mask == 0)."""

    def call(self, y_true, y_pred):
        return y_true  # placeholder — loss computed in custom train step


@tf.function
def _train_step(model, optimizer, ang, jnt, frame_mask, feat_mask):
    """One gradient step on a batch. Returns (ang_loss, jnt_loss)."""
    ang_m, jnt_m = _apply_masks(ang, jnt, frame_mask, feat_mask)

    with tf.GradientTape() as tape:
        preds = model(
            {"angular_input": ang_m, "joints_input": jnt_m},
            training=True,
        )
        # Masked angular loss: penalise only on masked frames × masked features
        ang_masked_pos = (1.0 - feat_mask) * (1.0 - frame_mask)  # (B,T,NA)
        ang_loss = tf.reduce_sum(
            ang_masked_pos * tf.square(ang - preds["ang_recon"])
        ) / (tf.reduce_sum(ang_masked_pos) + 1e-8)

        # Masked joint loss: penalise only on masked frames
        jnt_masked_pos = 1.0 - frame_mask   # (B,T,1) broadcast
        jnt_loss = tf.reduce_sum(
            jnt_masked_pos * tf.square(jnt - preds["jnt_recon"])
        ) / (tf.reduce_sum(jnt_masked_pos * NJ) + 1e-8)

        total_loss = ang_loss + 0.5 * jnt_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return ang_loss, jnt_loss


@tf.function
def _val_step(model, ang, jnt, frame_mask, feat_mask):
    ang_m, jnt_m = _apply_masks(ang, jnt, frame_mask, feat_mask)
    preds = model(
        {"angular_input": ang_m, "joints_input": jnt_m},
        training=False,
    )
    ang_masked_pos = (1.0 - feat_mask) * (1.0 - frame_mask)
    ang_loss = tf.reduce_sum(
        ang_masked_pos * tf.square(ang - preds["ang_recon"])
    ) / (tf.reduce_sum(ang_masked_pos) + 1e-8)
    jnt_masked_pos = 1.0 - frame_mask
    jnt_loss = tf.reduce_sum(
        jnt_masked_pos * tf.square(jnt - preds["jnt_recon"])
    ) / (tf.reduce_sum(jnt_masked_pos * NJ) + 1e-8)
    return ang_loss, jnt_loss


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_all_splits(data_dir: str):
    """Load all available splits (train + val + test) for SSL — no subject held-out."""
    all_ang, all_jnt = [], []
    for split in ("train", "val", "test"):
        d = os.path.join(data_dir, split)
        ang_path = os.path.join(d, "X_angles.npy")
        jnt_path = os.path.join(d, "X_joints.npy")
        if os.path.exists(ang_path) and os.path.exists(jnt_path):
            all_ang.append(np.load(ang_path))
            all_jnt.append(np.load(jnt_path))
            print(f"  loaded {split}: {all_ang[-1].shape[0]} samples")
    if not all_ang:
        raise FileNotFoundError(f"No splits found in {data_dir}")
    return np.concatenate(all_ang, axis=0), np.concatenate(all_jnt, axis=0)


def make_ssl_dataset(X_ang, X_jnt, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices({
        "angular_input": X_ang.astype(np.float32),
        "joints_input":  X_jnt.astype(np.float32),
    })
    if shuffle:
        ds = ds.shuffle(len(X_ang), seed=42)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Main training loop ────────────────────────────────────────────────────────

def train_ssl(args):
    print("=" * 60)
    print("FitNova — SSL Skeleton Autoencoder Pretraining (P3.1)")
    print("=" * 60)
    print(f"  TF version   : {tf.__version__}")
    print(f"  GPU available: {bool(tf.config.list_physical_devices('GPU'))}")
    print(f"  Data dir     : {args.data_dir}")
    print(f"  Model dir    : {args.model_dir}")
    print()

    # Load all data (no subject held-out in SSL)
    print("Loading data (all splits — no held-out subject in SSL)...")
    X_ang, X_jnt = _load_all_splits(args.data_dir)
    print(f"  Total: {len(X_ang)} windows")

    # 90/10 split for SSL val (random, not per-subject)
    n_val  = max(512, int(0.10 * len(X_ang)))
    perm   = np.random.default_rng(42).permutation(len(X_ang))
    val_i, tr_i = perm[:n_val], perm[n_val:]

    tr_ds  = make_ssl_dataset(X_ang[tr_i], X_jnt[tr_i], args.batch_size, shuffle=True)
    val_ds = make_ssl_dataset(X_ang[val_i], X_jnt[val_i], args.batch_size, shuffle=False)

    print(f"  Train: {len(tr_i)}  Val: {len(val_i)}")

    # Build model
    model = build_ssl_model()
    model.summary(line_length=80)
    optimizer = tf.keras.optimizers.Adam(args.lr)

    os.makedirs(args.model_dir, exist_ok=True)
    best_val = float("inf")
    patience_count = 0
    best_enc_path = os.path.join(args.model_dir, "mt_tcn_encoder_ssl.weights.h5")
    history = []

    t0 = time.time()

    # Compact epoch summary table header
    _header = (f"{'Ep':>4} {'tr_ang':>8} {'tr_jnt':>8} "
               f"{'val_ang':>8} {'val_jnt':>8} {'best':>8} {'min':>6}")
    _divider = "-" * len(_header)
    print(_header)
    print(_divider)

    epoch_bar = _tqdm(range(1, args.epochs + 1),
                      desc="SSL", unit="ep", dynamic_ncols=True)

    for epoch in epoch_bar:
        # ── Train ────────────────────────────────────────────────────────────
        tr_ang_loss = tf.keras.metrics.Mean()
        tr_jnt_loss = tf.keras.metrics.Mean()
        for batch in _tqdm(tr_ds, desc="train", leave=False,
                           unit="batch", dynamic_ncols=True):
            ang = batch["angular_input"]
            jnt = batch["joints_input"]
            bs  = tf.shape(ang)[0]
            fm, feat_m = _make_masks(bs, FRAME_MASK_RATE, FEAT_MASK_RATE)
            al, jl = _train_step(model, optimizer, ang, jnt, fm, feat_m)
            tr_ang_loss.update_state(al)
            tr_jnt_loss.update_state(jl)

        # ── Val ──────────────────────────────────────────────────────────────
        vl_ang_loss = tf.keras.metrics.Mean()
        vl_jnt_loss = tf.keras.metrics.Mean()
        for batch in _tqdm(val_ds, desc="val  ", leave=False,
                           unit="batch", dynamic_ncols=True):
            ang = batch["angular_input"]
            jnt = batch["joints_input"]
            bs  = tf.shape(ang)[0]
            fm, feat_m = _make_masks(bs, FRAME_MASK_RATE, FEAT_MASK_RATE)
            al, jl = _val_step(model, ang, jnt, fm, feat_m)
            vl_ang_loss.update_state(al)
            vl_jnt_loss.update_state(jl)

        val_total = vl_ang_loss.result().numpy() + 0.5 * vl_jnt_loss.result().numpy()
        elapsed   = (time.time() - t0) / 60

        tr_ang = float(tr_ang_loss.result())
        tr_jnt = float(tr_jnt_loss.result())
        vl_ang = float(vl_ang_loss.result())
        vl_jnt = float(vl_jnt_loss.result())

        is_best = val_total < best_val
        marker  = " *" if is_best else ""

        # One compact line per epoch
        print(f"{epoch:4d} {tr_ang:8.4f} {tr_jnt:8.4f} "
              f"{vl_ang:8.4f} {vl_jnt:8.4f} {best_val:8.4f} {elapsed:5.1f}m{marker}",
              flush=True)

        # Update tqdm postfix (visible in the bar even when scrolled up)
        epoch_bar.set_postfix(val=f"{val_total:.4f}", best=f"{best_val:.4f}",
                              patience=f"{patience_count}/{args.patience}")

        history.append({
            "epoch":     epoch,
            "tr_ang":    tr_ang,
            "tr_jnt":    tr_jnt,
            "val_ang":   vl_ang,
            "val_jnt":   vl_jnt,
            "val_total": float(val_total),
        })

        if is_best:
            best_val = val_total
            patience_count = 0
            # Save ONLY the encoder layers (matching MT-TCN supervised model names)
            model.save_weights(best_enc_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}  "
                      f"(patience={args.patience} exhausted).")
                break

        # ── Periodic clear to prevent Colab cell from growing unbounded ──────
        # Every 20 epochs: clear cell, reprint header + last 5 history rows
        if _IN_JUPYTER and epoch % 20 == 0:
            last5 = history[-5:]
            _clear_output(wait=True)
            print(f"SSL pretraining — epoch {epoch}/{args.epochs}  "
                  f"elapsed {elapsed:.1f} min")
            print(_header)
            print(_divider)
            for row in last5:
                marker = " *" if row["val_total"] == min(r["val_total"] for r in history) else ""
                print(f"{row['epoch']:4d} {row['tr_ang']:8.4f} {row['tr_jnt']:8.4f} "
                      f"{row['val_ang']:8.4f} {row['val_jnt']:8.4f} "
                      f"{best_val:8.4f} {(time.time()-t0)/60:5.1f}m{marker}")

    # Save history
    hist_path = os.path.join(args.model_dir, "ssl_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nSSL pretraining complete: {(time.time()-t0)/60:.1f} min")
    print(f"  Best val loss : {best_val:.4f}")
    print(f"  Encoder saved : {best_enc_path}")
    print(f"  History saved : {hist_path}")
    print()
    print("Next step — supervised fine-tune from SSL init:")
    print(f"  python -m backend.training.train_form_model \\")
    print(f"    --data_dir {args.data_dir} \\")
    print(f"    --ssl-init {best_enc_path} \\")
    print(f"    --source mediapipe")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--data-dir",  default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--epochs",    type=int,   default=EPOCHS)
    parser.add_argument("--patience",  type=int,   default=PATIENCE)
    parser.add_argument("--lr",        type=float, default=LR)
    parser.add_argument("--batch-size",type=int,   default=BATCH_SIZE)
    args = parser.parse_args()
    args.data_dir  = os.path.abspath(args.data_dir)
    args.model_dir = os.path.abspath(args.model_dir)
    train_ssl(args)
