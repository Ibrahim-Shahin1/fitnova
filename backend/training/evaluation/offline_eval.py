"""
Phase 1 — Diagnostic P1.2: Offline tensor evaluation.

Loads val/X_*.npy directly (no MediaPipe, no video) and evaluates both
loading methods on the full validation set.

This reproduces the exact evaluation path from train_form_model.py's
evaluate_model() — the one that reported 88.87% at training time.

If this gives 88% → weights are fine; the gap is in reality_check's
                     MediaPipe/MoCap preprocessing pipeline.
If this gives 11% → weights are broken; full retrain required.

Run:
    python -m backend.training.evaluation.offline_eval
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr

from backend.training.models.mt_tcn import build_mt_tcn, compile_mt_tcn

# ── Paths ──────────────────────────────────────────────────────────────────────
# Phase-4 A5: pick the dataset dir that matches the trained model's `source`,
# read from model_config.json. Previously hardcoded to form_dataset_mediapipe,
# which silently fed the wrong feature distribution to a Fit3D-trained model.
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "form_model")
WEIGHTS   = os.path.join(MODEL_DIR, "mt_tcn_weights.weights.h5")
BEST_W    = os.path.join(MODEL_DIR, "mt_tcn_best.weights.h5")
_CFG_PATH = os.path.join(MODEL_DIR, "model_config.json")


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
print("FitNova — Offline Tensor Eval (P1.2)")
print("=" * 60)
print(f"  TF version : {tf.__version__}")
print(f"  Data dir   : {DATA_DIR}")
print()


def load_split(data_dir, split):
    d = os.path.join(data_dir, split)
    return (
        np.load(os.path.join(d, "X_angles.npy")),
        np.load(os.path.join(d, "X_joints.npy")),
        np.load(os.path.join(d, "y_exercise.npy")),
        np.load(os.path.join(d, "y_quality.npy")),
        np.load(os.path.join(d, "y_joints.npy")),
        np.load(os.path.join(d, "y_boundary.npy")),
        np.load(os.path.join(d, "y_count.npy")),
    )


def make_ds(X_ang, X_jnt, y_ex, y_qlt, y_jnt, y_bnd, y_cnt, batch=64):
    n = len(X_ang)
    y_ex_oh = np.eye(27, dtype=np.float32)[y_ex.astype(np.int32)]
    inputs  = {"angular_input": X_ang.astype(np.float32),
               "joints_input":  X_jnt.astype(np.float32)}
    targets = {"exercise":     y_ex_oh,
               "quality":      y_qlt[:, None].astype(np.float32),
               "joint_errors": y_jnt.astype(np.float32),
               "boundary":     y_bnd.astype(np.float32),
               "rep_count":    y_cnt[:, None].astype(np.float32)}
    return tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(batch).prefetch(2)


def evaluate(model, X_ang, X_jnt, y_ex, y_qlt, y_jnt, label=""):
    inp   = {"angular_input": X_ang.astype(np.float32),
             "joints_input":  X_jnt.astype(np.float32)}
    preds = model.predict(inp, batch_size=64, verbose=0)

    ex_pred  = np.argmax(preds["exercise"], axis=-1)
    qlt_pred = preds["quality"].squeeze()

    ex_acc   = accuracy_score(y_ex, ex_pred)
    ex_f1    = f1_score(y_ex, ex_pred, average="macro", zero_division=0)

    try:
        qlt_r, _ = pearsonr(y_qlt, qlt_pred)
    except Exception:
        qlt_r = float("nan")

    # Per-class breakdown (top 5 worst)
    per_class_acc = {}
    for cls in np.unique(y_ex):
        mask = y_ex == cls
        per_class_acc[int(cls)] = float(np.mean(ex_pred[mask] == cls))

    worst5 = sorted(per_class_acc.items(), key=lambda x: x[1])[:5]
    best5  = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)[:5]

    print(f"\n  [{label}]")
    print(f"    Exercise accuracy : {ex_acc:.4f}  ({int(ex_acc*len(y_ex))}/{len(y_ex)})")
    print(f"    Exercise macro-F1 : {ex_f1:.4f}")
    print(f"    Quality Pearson r : {qlt_r:.4f}")
    print(f"    Quality mean/std  : {qlt_pred.mean():.3f} / {qlt_pred.std():.3f}")
    print(f"    Worst 5 classes (by accuracy):")
    with open(os.path.join(MODEL_DIR, "exercise_labels.json")) as f:
        lbl = json.load(f)
    i2e = {v: k for k, v in lbl.items()}
    for cidx, cacc in worst5:
        print(f"      {i2e.get(cidx,'?'):40s}: {cacc:.3f}")
    print(f"    Best 5 classes (by accuracy):")
    for cidx, cacc in best5:
        print(f"      {i2e.get(cidx,'?'):40s}: {cacc:.3f}")

    return ex_acc


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading splits from disk...")
tr = load_split(DATA_DIR, "train")
vl = load_split(DATA_DIR, "val")
print(f"  train: {tr[0].shape[0]} samples, val: {vl[0].shape[0]} samples")
print()

# ── Evaluate both loading methods ─────────────────────────────────────────────
results = {}
for wpath, wname in [(WEIGHTS, "mt_tcn_weights"), (BEST_W, "mt_tcn_best")]:
    if not os.path.exists(wpath):
        print(f"Skipping {wname} — file not found")
        continue
    print(f"── {wname} ──────────────────────────────────────────────")

    # (a) Positional
    m_pos = build_mt_tcn()
    m_pos.load_weights(wpath)
    acc_pos_val = evaluate(m_pos, *vl[:3], vl[3], vl[4], label=f"{wname} / positional / val")
    acc_pos_tr  = evaluate(m_pos, *tr[:3], tr[3], tr[4], label=f"{wname} / positional / train")

    # (b) by_name
    m_named = build_mt_tcn()
    try:
        m_named.load_weights(wpath, by_name=True, skip_mismatch=False)
        acc_named_val = evaluate(m_named, *vl[:3], vl[3], vl[4], label=f"{wname} / by_name / val")
        acc_named_tr  = evaluate(m_named, *tr[:3], tr[3], tr[4], label=f"{wname} / by_name / train")
    except Exception as e:
        print(f"  by_name loading FAILED: {e}")
        acc_named_val = acc_named_tr = -1.0

    results[wname] = {
        "positional_val":  acc_pos_val,
        "positional_train": acc_pos_tr,
        "by_name_val":     acc_named_val,
        "by_name_train":   acc_named_tr,
    }

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
for wname, r in results.items():
    print(f"\n  {wname}:")
    print(f"    positional — val={r['positional_val']:.3f}  train={r['positional_train']:.3f}")
    print(f"    by_name    — val={r['by_name_val']:.3f}  train={r['by_name_train']:.3f}")

print()
print("Decision:")
best_train = max(
    max(r["positional_train"], r["by_name_train"]) for r in results.values()
)
if best_train >= 0.80:
    print("  ✓  Weights VALID — at least one loading method achieves ≥80% on train.")
    print("     Phase 2 pipeline fixes are still required (B2-B7),")
    print("     but a full retrain may not be needed if we fix the loading.")
elif best_train >= 0.40:
    print("  ⚠  Weights PARTIALLY degraded — best train acc is between 40-80%.")
    print("     Proceed to Phase 2 full rebuild.")
else:
    print("  ✗  Weights BROKEN — even on training data accuracy is near-random.")
    print("     Full retrain is required after Phase 2 pipeline fixes.")
    print("     The 88.87% training accuracy was memorization of synthetic variants.")

# Save results
out = os.path.join(MODEL_DIR, "reality_check", "offline_eval.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
import json
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved → {out}")
