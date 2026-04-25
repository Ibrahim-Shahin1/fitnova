"""
Phase 1 — Diagnostic P1.1: Weight probe.

Answers three questions:
  1. Does positional load_weights (current code) produce DIFFERENT weights
     than by_name=True load_weights?  → If yes, B1a is confirmed.
  2. Does the model correctly classify a TRAINING sample?
     → If no, the weights themselves are garbage regardless of loading method.
  3. What Keras version saved these weights? Does it match local Keras?

Run:
    python -m backend.training.evaluation.weight_probe
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import tensorflow as tf
from backend.training.models.mt_tcn import build_mt_tcn

# ── Paths ──────────────────────────────────────────────────────────────────────
# Phase-4 A5: data path is no longer hardcoded to form_dataset_mediapipe.
# Read ``source`` from model_config.json (written by train_form_model.py) and
# pick the matching dataset dir (form_dataset_fit3d | form_dataset_mediapipe).
# Falls back to fit3d if the config is missing the key (older checkpoints).
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "models", "form_model")
WEIGHTS     = os.path.join(MODEL_DIR, "mt_tcn_weights.weights.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "exercise_labels.json")
STATS_PATH  = os.path.join(MODEL_DIR, "angle_stats.npz")
_CFG_PATH   = os.path.join(MODEL_DIR, "model_config.json")


def _resolve_data_dir() -> str:
    source = "fit3d"
    if os.path.exists(_CFG_PATH):
        try:
            with open(_CFG_PATH) as _cf:
                cfg = json.load(_cf)
            source = cfg.get("source", source)
        except Exception:
            pass
    return os.path.join(
        os.path.dirname(__file__), "..", "..", "data",
        f"form_dataset_{source}",
    )


DATA_DIR = _resolve_data_dir()

print("=" * 60)
print("FitNova — Weight Probe (P1.1)")
print("=" * 60)
print(f"  TF / Keras    : {tf.__version__}")
print(f"  Weights file  : {WEIGHTS}")
print(f"  Weights size  : {os.path.getsize(WEIGHTS)/1e6:.2f} MB")
print()


# ── 1. Keras version inside HDF5 ──────────────────────────────────────────────
print("── Keras version in weights file ──────────────────────────")
try:
    import h5py
    with h5py.File(WEIGHTS, "r") as f:
        top_keys = list(f.keys())
        print(f"  HDF5 top-level keys: {top_keys}")
        # Try common metadata locations
        for attr_path in ["", "keras_metadata", "model_weights"]:
            node = f if not attr_path else f.get(attr_path)
            if node is not None:
                for attr in node.attrs:
                    if "keras" in attr.lower() or "version" in attr.lower():
                        print(f"  [{attr_path or '/'}].{attr} = {node.attrs[attr]}")
except Exception as e:
    print(f"  Could not inspect HDF5: {e}")
print()


# ── 2. Load two ways and compare first-conv kernels ───────────────────────────
print("── Weight loading comparison ───────────────────────────────")

def _first_kernel_stats(model):
    """Mean/std/norm of the first trainable Conv1D kernel."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv1D) and layer.trainable_weights:
            w = layer.get_weights()[0]  # (kernel_size, in_ch, out_ch)
            return layer.name, float(np.mean(w)), float(np.std(w)), float(np.linalg.norm(w.ravel()))
    return None, 0.0, 0.0, 0.0


# (a) Positional — as current form_analyzer does
model_pos = build_mt_tcn()
model_pos.load_weights(WEIGHTS)
name_a, mean_a, std_a, norm_a = _first_kernel_stats(model_pos)
print(f"  (a) positional load_weights:")
print(f"      layer={name_a}  mean={mean_a:.6f}  std={std_a:.6f}  norm={norm_a:.4f}")

# (b) by_name=True
model_named = build_mt_tcn()
try:
    model_named.load_weights(WEIGHTS, by_name=True, skip_mismatch=False)
    name_b, mean_b, std_b, norm_b = _first_kernel_stats(model_named)
    print(f"  (b) by_name=True load_weights:")
    print(f"      layer={name_b}  mean={mean_b:.6f}  std={std_b:.6f}  norm={norm_b:.4f}")

    max_diff = abs(norm_a - norm_b)
    if max_diff < 1e-4:
        print(f"\n  ✓  MATCH  (norm diff={max_diff:.2e}) — B1a NOT confirmed via kernel comparison")
    else:
        print(f"\n  ✗  MISMATCH  (norm diff={max_diff:.4f}) — B1a CONFIRMED: positional loading is broken")
except Exception as e:
    print(f"  (b) by_name load FAILED: {e}")
    print("      → B1a is likely the cause (layer name mismatch in saved file)")

print()

# ── 3. Classify training samples ─────────────────────────────────────────────
print("── Training-sample sanity check ────────────────────────────")
train_dir = os.path.join(DATA_DIR, "train")
val_dir   = os.path.join(DATA_DIR, "val")

with open(LABELS_PATH) as f:
    exercise_labels = json.load(f)
idx_to_exercise = {v: k for k, v in exercise_labels.items()}

for split_name, split_dir in [("train", train_dir), ("val", val_dir)]:
    if not os.path.exists(os.path.join(split_dir, "X_angles.npy")):
        print(f"  {split_name}: not found — skip")
        continue

    X_ang = np.load(os.path.join(split_dir, "X_angles.npy"))
    X_jnt = np.load(os.path.join(split_dir, "X_joints.npy"))
    y_ex  = np.load(os.path.join(split_dir, "y_exercise.npy"))
    meta_path = os.path.join(split_dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else None

    # Pick first 64 samples (one batch) and evaluate both models
    N = min(64, len(X_ang))
    X_a_batch = X_ang[:N].astype(np.float32)
    X_j_batch = X_jnt[:N].astype(np.float32)
    y_batch   = y_ex[:N]

    inp = {"angular_input": X_a_batch, "joints_input": X_j_batch}

    # Model (a) positional
    p_pos   = model_pos.predict(inp, verbose=0)
    ex_pos  = np.argmax(p_pos["exercise"], axis=-1)
    acc_pos = np.mean(ex_pos == y_batch)

    # Model (b) by_name
    try:
        p_name  = model_named.predict(inp, verbose=0)
        ex_name = np.argmax(p_name["exercise"], axis=-1)
        acc_name= np.mean(ex_name == y_batch)
    except Exception:
        acc_name = -1.0

    print(f"\n  {split_name} split (first {N} samples):")
    print(f"    (a) positional  accuracy = {acc_pos:.3f}  "
          f"({int(acc_pos*N)}/{N}  correct)")
    print(f"    (b) by_name     accuracy = {acc_name:.3f}  "
          f"({int(acc_name*N)}/{N}  correct)")

    # Print first 8 GT vs pred (positional model)
    print(f"    First 8 GT → predicted (positional):")
    for i in range(min(8, N)):
        gt  = idx_to_exercise.get(int(y_batch[i]), "?")
        pr  = idx_to_exercise.get(int(ex_pos[i]),  "?")
        match = "✓" if gt == pr else "✗"
        subj  = meta[i]["subject"] if meta else "?"
        print(f"      [{i:2d}] {subj} GT={gt:35s}  pred={pr:35s} {match}")

print()

# ── 4. Quality score distribution ────────────────────────────────────────────
print("── Quality score distribution (by_name, train first 256) ───")
if os.path.exists(os.path.join(train_dir, "X_angles.npy")):
    X_ang = np.load(os.path.join(train_dir, "X_angles.npy"))
    X_jnt = np.load(os.path.join(train_dir, "X_joints.npy"))
    N = min(256, len(X_ang))
    try:
        p = model_named.predict(
            {"angular_input": X_ang[:N].astype(np.float32),
             "joints_input":  X_jnt[:N].astype(np.float32)},
            verbose=0)
        q = p["quality"].squeeze()
        print(f"  n={N}  min={q.min():.3f}  max={q.max():.3f}  "
              f"mean={q.mean():.3f}  std={q.std():.3f}")
        # Histogram
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        counts, _ = np.histogram(q, bins=bins)
        for lo, hi, c in zip(bins, bins[1:], counts):
            bar = "█" * int(c / max(counts) * 30)
            print(f"  [{lo:.1f}-{hi:.1f}]: {c:4d}  {bar}")
    except Exception as e:
        print(f"  Could not compute: {e}")

print()
print("=" * 60)
print("Interpretation:")
print("  • If (a) positional acc ≈ (b) by_name acc on TRAIN split and both are high (>80%)")
print("    → weights are fine; bug is in reality_check's pre-processing path.")
print("  • If (a) positional acc << (b) by_name acc on TRAIN split")
print("    → B1a confirmed: fix form_analyzer._load_model() with by_name=True.")
print("  • If BOTH (a) and (b) are near-random on TRAIN split (acc ≈ 4%)")
print("    → weights are garbage; full retrain is required.")
print("=" * 60)
