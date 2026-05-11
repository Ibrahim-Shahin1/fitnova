"""
Build v5.1 Colab artifacts.

Run once locally:

    python _build_v5_1_colab_bundle.py

Produces:
    colab_notebooks/v5_1_supervised_train.ipynb     — Phase 5.1 retrain on perturbed dataset
    fitnova_v5_1_src.zip                             — backend source for Drive (now includes synthetic_perturbation.py + dataset_builder_v5_1.py)
    v5_1_dataset.zip                                 — pre-built v5.1 dataset

You then:
  1. Upload BOTH new zips to MyDrive/fitnova_v5/ (overwrite existing if any)
  2. Open colab_notebooks/v5_1_supervised_train.ipynb in Colab → Run all
     The SSL encoder from your earlier run is reused; supervised takes ~60-90 min.
  3. Download v5.1 weights to backend/models/form_model_v5_1/
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent
COLAB_DIR = PROJECT_ROOT / "colab_notebooks"
COLAB_DIR.mkdir(exist_ok=True)


def md(text):
    return nbf.v4.new_markdown_cell(text.lstrip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.lstrip("\n"))


def write_notebook(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "colab":      {"provenance": []},
        "language_info": {"name": "python"},
    }
    out = COLAB_DIR / name
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"  wrote {out}")
    return out


# ── Common Drive/source/data setup cells ──────────────────────────────────────

CELL_GPU_DRIVE = code("""
# ── GPU + Drive mount ────────────────────────────────────────────────────────
import os, sys, json, time, shutil
import tensorflow as tf
print("TF:    ", tf.__version__)
print("Keras: ", tf.keras.__version__)
print("GPUs:  ", tf.config.list_physical_devices("GPU"))

from google.colab import drive
drive.mount("/content/drive")

DRIVE_BASE  = "/content/drive/MyDrive/fitnova_v5"
RESULTS_DIR = "/content/drive/MyDrive/fitnova_v5_results"
os.makedirs(DRIVE_BASE,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("DRIVE_BASE :", DRIVE_BASE)
print("RESULTS_DIR:", RESULTS_DIR)
""")

CELL_EXTRACT_SRC = code("""
# ── Extract v5.1 backend source from Drive ──────────────────────────────────
SRC_ZIP  = f"{DRIVE_BASE}/fitnova_v5_1_src.zip"
WORK_DIR = "/content/fitnova_v5_1"

assert os.path.isfile(SRC_ZIP), (
    f"Source zip not found at {SRC_ZIP}.\\n"
    f"Build it locally: python _build_v5_1_colab_bundle.py\\n"
    f"Then upload fitnova_v5_1_src.zip to MyDrive/fitnova_v5/"
)

# Always re-extract so we pick up code changes.
shutil.rmtree(WORK_DIR, ignore_errors=True)
os.makedirs(WORK_DIR, exist_ok=True)
!unzip -q "$SRC_ZIP" -d "$WORK_DIR"

if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)

print("Source ready at", WORK_DIR)
print("  has st_gcn:                 ", os.path.exists(f"{WORK_DIR}/backend/training/models/st_gcn.py"))
print("  has synthetic_perturbation: ", os.path.exists(f"{WORK_DIR}/backend/training/preprocessing/synthetic_perturbation.py"))
print("  has dataset_builder_v5_1:   ", os.path.exists(f"{WORK_DIR}/backend/training/preprocessing/dataset_builder_v5_1.py"))
""")

CELL_EXTRACT_DATA = code("""
# ── Extract v5.1 pre-built dataset from Drive ───────────────────────────────
DATA_ZIP = f"{DRIVE_BASE}/v5_1_dataset.zip"
DATA_DIR = "/content/v5_1_dataset"

assert os.path.isfile(DATA_ZIP), (
    f"Dataset zip not found at {DATA_ZIP}.\\n"
    f"Build it locally: python _build_v5_1_colab_bundle.py\\n"
    f"Then upload v5_1_dataset.zip to MyDrive/fitnova_v5/"
)

shutil.rmtree(DATA_DIR, ignore_errors=True)
os.makedirs(DATA_DIR, exist_ok=True)
!unzip -q "$DATA_ZIP" -d "$DATA_DIR"

print("Dataset files:", sorted(os.listdir(DATA_DIR)))

with open(f"{DATA_DIR}/dataset_info.json") as f:
    DATASET_INFO = json.load(f)
for k in ("version", "label_strategy", "n_exercises", "n_joint_groups",
          "target_frames", "n_canonical_joints", "n_angular",
          "perturbations_per_rep_train", "severity_to_quality"):
    print(f"  {k:<32s}: {DATASET_INFO[k]}")
""")


# ── v5.1 supervised notebook ──────────────────────────────────────────────────


def build_v5_1_supervised_notebook():
    cells = [
        md("""
# FitNova v5.1 — Supervised retrain on synthetic-perturbation dataset

Replaces v5.0's AIFit-signature labels with **synthetic perturbation severity**
labels (Fitness-AQA, Parmar et al. 2022). The architecture and SSL encoder
from v5.0 are reused unchanged; only the supervised dataset is different.

**Cost:** ~60-90 min on L4 / ~30-45 min on A100. ~3 Colab compute units.

**Inputs (MyDrive/fitnova_v5/ + MyDrive/fitnova_v5_results/):**
- `fitnova_v5_1_src.zip`  (NEW — has `synthetic_perturbation.py` + `dataset_builder_v5_1.py`)
- `v5_1_dataset.zip`      (NEW — perturbed dataset)
- `ssl_encoder.weights.h5` (REUSE from v5.0 — already in Drive)

**Outputs (MyDrive/fitnova_v5_results/):**
- `v5_1_supervised.weights.h5`
- `v5_1_history.json`
- `v5_1_test_metrics.json`
- `v5_1_val_perturbed_metrics.json`  (NEW — measures quality-gradient learning)
- `v5_1_supervised_checkpoints/epoch_NN.weights.h5`
- `v5_1_confusion_matrix_test.npy`

**Crash-resume:** rerun all cells from the top; the training cell auto-resumes.
"""),
        CELL_GPU_DRIVE,
        CELL_EXTRACT_SRC,
        CELL_EXTRACT_DATA,
        code("""
# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
from collections import defaultdict
from tensorflow import keras
from backend.training.models.st_gcn import build_v5_model, build_v5_ssl_model

SUP_DIR     = f"{RESULTS_DIR}/v5_1_supervised_checkpoints"
SUP_HISTORY = f"{RESULTS_DIR}/v5_1_history.json"
SUP_FINAL   = f"{RESULTS_DIR}/v5_1_supervised.weights.h5"
SUP_TEST    = f"{RESULTS_DIR}/v5_1_test_metrics.json"
SUP_VPERT   = f"{RESULTS_DIR}/v5_1_val_perturbed_metrics.json"
SSL_FINAL   = f"{RESULTS_DIR}/ssl_encoder.weights.h5"  # reused from v5.0
os.makedirs(SUP_DIR, exist_ok=True)
"""),
        code("""
# ── Hyperparameters ──────────────────────────────────────────────────────────
EPOCHS        = 60
BATCH_SIZE    = 16          # 16 view-groups × 2 views = 32 per step
LR            = 1e-3
LR_FINAL      = 1e-5
WARMUP_EPOCHS = 2
LOSS_WEIGHTS = {
    "quality":   3.0,        # bumped from 2.5 — quality is the headline gate
    "joint_err": 1.5,
    "boundary":  0.3,        # lowered — boundary is degenerate by design
    "rep_count": 0.1,
    "action":    0.3,
    "mv_cons":   0.5,
}
SEED = 42

T_FRAMES       = DATASET_INFO["target_frames"]
J              = DATASET_INFO["n_canonical_joints"]
N_EXERCISES    = DATASET_INFO["n_exercises"]
N_JOINT_GROUPS = DATASET_INFO["n_joint_groups"]
N_ANGULAR      = DATASET_INFO["n_angular"]
"""),
        code("""
# ── Load splits ──────────────────────────────────────────────────────────────
train_npz         = np.load(f"{DATA_DIR}/train.npz")
val_npz           = np.load(f"{DATA_DIR}/val.npz")
val_perturbed_npz = np.load(f"{DATA_DIR}/val_perturbed.npz")
test_npz          = np.load(f"{DATA_DIR}/test.npz")

print(f"train:         {train_npz['pose'].shape[0]} samples  "
      f"(clean={int((train_npz['perturb_severity_idx']==0).sum())}  "
      f"perturbed={int((train_npz['perturb_severity_idx']>0).sum())})")
print(f"val:           {val_npz['pose'].shape[0]} samples (clean only)")
print(f"val_perturbed: {val_perturbed_npz['pose'].shape[0]} samples "
      f"(includes perturbed variants for tracking quality-gradient)")
print(f"test:          {test_npz['pose'].shape[0]} samples (clean only)")
"""),
        code("""
# ── Build view-groups for multi-view sampling ───────────────────────────────
# Pair only samples sharing (subject, exercise, rep, perturb_group, severity)
# so multi-view consistency loss compares matched perturbation variants only.
def build_view_groups(npz):
    groups = defaultdict(list)
    for i, (s, e, r, pg, ps) in enumerate(zip(
        npz["subject_idx"], npz["exercise_idx"], npz["rep_idx"],
        npz["perturb_group"], npz["perturb_severity_idx"],
    )):
        groups[(int(s), int(e), int(r), int(pg), int(ps))].append(i)
    return list(groups.values())

train_groups = build_view_groups(train_npz)
val_groups   = build_view_groups(val_perturbed_npz)

n_multiview = sum(1 for g in train_groups if len(g) >= 2)
print(f"Train view-groups: {len(train_groups)} total, {n_multiview} with >=2 views")
"""),
        code("""
# ── tf.data pipeline ─────────────────────────────────────────────────────────
def make_supervised_dataset(npz, view_groups, batch_size, shuffle, seed, repeat=True):
    n_groups = len(view_groups)
    pose       = npz["pose"]
    angles     = npz["angles"]
    exercise   = npz["exercise_idx"].astype(np.int32)
    quality    = npz["quality"].astype(np.float32).reshape(-1, 1)
    joint_err  = npz["joint_err"].astype(np.float32)
    boundary   = npz["boundary"].astype(np.float32).reshape(-1, T_FRAMES, 1)
    rep_count  = npz["rep_count"].astype(np.float32).reshape(-1, 1)

    def gen():
        rng = np.random.default_rng(seed)
        order = np.arange(n_groups)
        while True:
            if shuffle:
                rng.shuffle(order)
            for i in range(0, n_groups, batch_size):
                idx_groups = order[i:i + batch_size]
                if len(idx_groups) < batch_size:
                    break
                view1, view2 = [], []
                for g_idx in idx_groups:
                    g = view_groups[g_idx]
                    if len(g) >= 2:
                        a, b = rng.choice(g, size=2, replace=False)
                    else:
                        a = b = g[0]
                    view1.append(int(a))
                    view2.append(int(b))
                idx = np.array(view1 + view2, dtype=np.int64)
                yield (
                    {
                        "pose":        pose[idx],
                        "angles":      angles[idx],
                        "exercise_id": exercise[idx],
                    },
                    {
                        "quality":   quality[idx],
                        "joint_err": joint_err[idx],
                        "boundary":  boundary[idx],
                        "rep_count": rep_count[idx],
                        "action":    exercise[idx],
                    },
                )
            if not repeat:
                break

    sig = (
        {
            "pose":        tf.TensorSpec(shape=(None, T_FRAMES, J, 4), dtype=tf.float32),
            "angles":      tf.TensorSpec(shape=(None, T_FRAMES, N_ANGULAR), dtype=tf.float32),
            "exercise_id": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        },
        {
            "quality":   tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "joint_err": tf.TensorSpec(shape=(None, T_FRAMES, N_JOINT_GROUPS), dtype=tf.float32),
            "boundary":  tf.TensorSpec(shape=(None, T_FRAMES, 1), dtype=tf.float32),
            "rep_count": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            "action":    tf.TensorSpec(shape=(None,), dtype=tf.int32),
        },
    )
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(2)

steps_per_epoch = max(1, len(train_groups) // BATCH_SIZE)
val_steps       = max(1, len(val_groups)   // BATCH_SIZE)
train_ds = make_supervised_dataset(train_npz, train_groups, BATCH_SIZE, shuffle=True,  seed=SEED)
val_ds   = make_supervised_dataset(val_perturbed_npz, val_groups, BATCH_SIZE, shuffle=False, seed=SEED+1)
print(f"steps/epoch: {steps_per_epoch}   val steps: {val_steps}")
"""),
        code("""
# ── Custom training step with multi-view consistency loss ────────────────────
class V5_1Model(keras.Model):
    def __init__(self, base_model, mv_weight, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.base = base_model
        self.mv_weight = float(mv_weight)
        self.bs = int(batch_size)

    def call(self, inputs, training=False):
        return self.base(inputs, training=training)

    def _step(self, data, training):
        x, y = data
        with tf.GradientTape(watch_accessed_variables=training) as tape:
            preds = self.base(x, training=training)
            l_q = tf.reduce_mean(tf.square(preds["quality"]   - y["quality"]))
            l_j = tf.reduce_mean(keras.losses.binary_crossentropy(y["joint_err"], preds["joint_err"]))
            l_b = tf.reduce_mean(keras.losses.binary_crossentropy(y["boundary"],  preds["boundary"]))
            l_c = tf.reduce_mean(tf.square(preds["rep_count"] - y["rep_count"]))
            l_a = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y["action"], preds["action"]))
            B = tf.shape(preds["trunk"])[0]
            half = B // 2
            t1 = preds["trunk"][:half]
            t2 = preds["trunk"][half:half * 2]
            l_mv = tf.reduce_mean(tf.square(t1 - t2))
            total = (
                LOSS_WEIGHTS["quality"]   * l_q
                + LOSS_WEIGHTS["joint_err"] * l_j
                + LOSS_WEIGHTS["boundary"]  * l_b
                + LOSS_WEIGHTS["rep_count"] * l_c
                + LOSS_WEIGHTS["action"]    * l_a
                + self.mv_weight            * l_mv
            )
        if training:
            grads = tape.gradient(total, self.base.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.base.trainable_variables))
        # Pearson correlation on quality, computed within this batch (rough metric)
        q_pred = tf.reshape(preds["quality"], [-1])
        q_true = tf.reshape(y["quality"], [-1])
        q_pred_c = q_pred - tf.reduce_mean(q_pred)
        q_true_c = q_true - tf.reduce_mean(q_true)
        denom = tf.sqrt(tf.reduce_sum(q_pred_c**2) * tf.reduce_sum(q_true_c**2))
        q_pearson = tf.reduce_sum(q_pred_c * q_true_c) / tf.maximum(denom, 1e-6)
        action_acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.cast(tf.argmax(preds["action"], axis=-1), tf.int32), y["action"]),
            tf.float32,
        ))
        return {
            "loss": total, "l_q": l_q, "l_j": l_j, "l_b": l_b,
            "l_c": l_c, "l_a": l_a, "l_mv": l_mv,
            "q_pearson": q_pearson, "action_acc": action_acc,
        }

    def train_step(self, data):
        return self._step(data, training=True)

    def test_step(self, data):
        return self._step(data, training=False)
"""),
        code("""
# ── Build base model + load v5.0 SSL pretrain encoder ───────────────────────
base = build_v5_model(
    target_frames=T_FRAMES, n_joints=J, n_pose_channels=4,
    n_angular=N_ANGULAR, n_exercises=N_EXERCISES, n_joint_groups=N_JOINT_GROUPS,
)

if os.path.isfile(SSL_FINAL):
    ssl_template = build_v5_ssl_model(
        target_frames=T_FRAMES, n_joints=J, n_pose_channels=4
    )
    ssl_template.load_weights(SSL_FINAL)
    n_loaded = 0
    for layer_sup in base.layers:
        for layer_ssl in ssl_template.layers:
            if (layer_sup.name == layer_ssl.name
                and layer_sup.weights and layer_ssl.weights
                and [w.shape for w in layer_sup.weights] == [w.shape for w in layer_ssl.weights]):
                layer_sup.set_weights(layer_ssl.get_weights())
                n_loaded += 1
                break
    print(f"Loaded {n_loaded} SSL-pretrained layers (reused from v5.0)")
    del ssl_template
else:
    print("WARNING: no SSL pretrain — training from scratch.")
"""),
        code("""
# ── Optimizer + wrapper ─────────────────────────────────────────────────────
total_steps  = steps_per_epoch * EPOCHS
warmup_steps = steps_per_epoch * WARMUP_EPOCHS
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LR,
    decay_steps=max(1, total_steps - warmup_steps),
    alpha=LR_FINAL / LR,
    warmup_target=LR,
    warmup_steps=warmup_steps,
)
opt = keras.optimizers.AdamW(
    learning_rate=lr_schedule, weight_decay=1e-4, clipnorm=1.0,
)
model = V5_1Model(base, mv_weight=LOSS_WEIGHTS["mv_cons"], batch_size=BATCH_SIZE)
model.compile(optimizer=opt)

_ = base({
    "pose":        np.zeros((1, T_FRAMES, J, 4), dtype=np.float32),
    "angles":      np.zeros((1, T_FRAMES, N_ANGULAR), dtype=np.float32),
    "exercise_id": np.zeros((1,), dtype=np.int32),
})
print(f"Model has {base.count_params():,} params")
"""),
        code("""
# ── Crash-resume detection ──────────────────────────────────────────────────
def find_latest_epoch(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return 0
    files = [f for f in os.listdir(ckpt_dir) if f.startswith("epoch_") and f.endswith(".weights.h5")]
    if not files:
        return 0
    return max(int(f.split("_")[1].split(".")[0]) for f in files)

initial_epoch = find_latest_epoch(SUP_DIR)
if initial_epoch > 0:
    last_ckpt = f"{SUP_DIR}/epoch_{initial_epoch:02d}.weights.h5"
    print(f"Resuming from epoch {initial_epoch} ({last_ckpt})")
    base.load_weights(last_ckpt)
else:
    print("Fresh start (no prior v5.1 checkpoints found)")

history_acc = {}
if os.path.isfile(SUP_HISTORY):
    with open(SUP_HISTORY) as f:
        history_acc = json.load(f)
    print(f"Loaded prior history: {len(next(iter(history_acc.values()), []))} epochs")
"""),
        code("""
# ── Per-epoch checkpoint callback ──────────────────────────────────────────
class DriveCheckpoint(keras.callbacks.Callback):
    def __init__(self, ckpt_dir, history_path, history_acc, base_model):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.history_path = history_path
        self.history_acc  = history_acc
        self.base_model   = base_model

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        ep1 = epoch + 1
        ckpt = f"{self.ckpt_dir}/epoch_{ep1:02d}.weights.h5"
        self.base_model.save_weights(ckpt)
        for k, v in logs.items():
            self.history_acc.setdefault(k, []).append(float(v))
        with open(self.history_path, "w") as f:
            json.dump(self.history_acc, f, indent=2)
        files = sorted(
            f for f in os.listdir(self.ckpt_dir)
            if f.startswith("epoch_") and f.endswith(".weights.h5")
        )
        for f in files[:-3]:
            try:
                os.remove(os.path.join(self.ckpt_dir, f))
            except OSError:
                pass

cb = DriveCheckpoint(SUP_DIR, SUP_HISTORY, history_acc, base)
"""),
        code("""
# ── Train ────────────────────────────────────────────────────────────────────
if initial_epoch >= EPOCHS:
    print(f"Already trained for {initial_epoch} epochs — nothing to do.")
else:
    model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=[cb],
        verbose=2,
    )
"""),
        code("""
# ── Save final weights ───────────────────────────────────────────────────────
base.save_weights(SUP_FINAL)
print(f"Saved final v5.1 weights to {SUP_FINAL}")
print(f"  size: {os.path.getsize(SUP_FINAL) / 1e6:.2f} MB")
"""),
        code("""
# ── Evaluate on test (s11 clean) and val_perturbed (quality gradient) ────────
def evaluate_on_split(npz, name, batch_size=16):
    pose       = npz["pose"]
    angles     = npz["angles"]
    exercise   = npz["exercise_idx"].astype(np.int32)
    quality    = npz["quality"].astype(np.float32)
    joint_err  = npz["joint_err"].astype(np.float32)
    boundary   = npz["boundary"].astype(np.float32)

    preds = base.predict(
        {"pose": pose, "angles": angles, "exercise_id": exercise},
        batch_size=batch_size, verbose=0,
    )
    pq = preds["quality"][:, 0]
    if quality.std() < 1e-6:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(pq, quality)[0, 1])
    q_mae = float(np.mean(np.abs(pq - quality)))
    pj = (preds["joint_err"] > 0.5).astype(np.float32)
    j_acc = float((pj == joint_err).mean())
    eps = 1e-9
    j_f1 = float(2 * (pj * joint_err).sum() / max(eps, (pj + joint_err).sum()))
    pb = (preds["boundary"][..., 0] > 0.5).astype(np.float32)
    bb = (boundary > 0.5).astype(np.float32)
    inter = float((pb * bb).sum())
    union = float(((pb + bb) > 0.5).sum())
    iou = inter / max(eps, union)
    ax = preds["action"].argmax(axis=-1)
    a_acc = float((ax == exercise).mean())
    out = {
        "n":               int(len(pose)),
        "quality_pearson": pearson,
        "quality_mae":     q_mae,
        "quality_pred_mean": float(pq.mean()),
        "quality_pred_std":  float(pq.std()),
        "joint_err_acc":   j_acc,
        "joint_err_f1":    j_f1,
        "boundary_iou":    iou,
        "action_acc":      a_acc,
    }
    return out

test_metrics = evaluate_on_split(test_npz, "test (s11 clean)")
print("test:", json.dumps(test_metrics, indent=2))
with open(SUP_TEST, "w") as f:
    json.dump(test_metrics, f, indent=2)

vpert_metrics = evaluate_on_split(val_perturbed_npz, "val_perturbed")
print("val_perturbed:", json.dumps(vpert_metrics, indent=2))
with open(SUP_VPERT, "w") as f:
    json.dump(vpert_metrics, f, indent=2)
"""),
        code("""
# ── Per-severity quality breakdown on val_perturbed (the key signal) ────────
sev_names = ["clean", "mild", "moderate", "severe"]
sev_targets = [1.0, 0.7, 0.4, 0.1]
preds_val = base.predict(
    {"pose": val_perturbed_npz["pose"],
     "angles": val_perturbed_npz["angles"],
     "exercise_id": val_perturbed_npz["exercise_idx"].astype(np.int32)},
    batch_size=16, verbose=0,
)["quality"][:, 0]
sev_idx = val_perturbed_npz["perturb_severity_idx"]
print(f"{'severity':<10s} {'target':>8s}   {'pred μ':>8s} {'pred σ':>8s}   {'n':>5s}")
for i, name in enumerate(sev_names):
    mask = sev_idx == i
    if mask.sum() == 0:
        continue
    p = preds_val[mask]
    print(f"{name:<10s} {sev_targets[i]:>8.2f}   {p.mean():>8.3f} {p.std():>8.3f}   {int(mask.sum()):>5d}")
"""),
        code("""
# ── Confusion matrix on test ────────────────────────────────────────────────
preds_action = base.predict(
    {"pose": test_npz["pose"], "angles": test_npz["angles"],
     "exercise_id": test_npz["exercise_idx"].astype(np.int32)},
    batch_size=16, verbose=0,
)["action"].argmax(axis=-1)
true_action = test_npz["exercise_idx"]
n_ex = N_EXERCISES
cm = np.zeros((n_ex, n_ex), dtype=np.int32)
for t, p in zip(true_action, preds_action):
    cm[int(t), int(p)] += 1
np.save(f"{RESULTS_DIR}/v5_1_confusion_matrix_test.npy", cm)
print("Confusion matrix saved.")
"""),
        md("""
## Done

Outputs in `MyDrive/fitnova_v5_results/`:
- `v5_1_supervised.weights.h5`
- `v5_1_history.json`
- `v5_1_test_metrics.json`
- `v5_1_val_perturbed_metrics.json`  ← the key signal: per-severity quality predictions
- `v5_1_confusion_matrix_test.npy`

Download to `backend/models/form_model_v5_1/` on the laptop, then run Phase 6 reality-check.
"""),
    ]
    return write_notebook("v5_1_supervised_train.ipynb", cells)


# ── src + dataset zips ───────────────────────────────────────────────────────


SRC_ROOTS_V5_1 = [
    "backend/__init__.py",
    "backend/training/__init__.py",
    "backend/training/preprocessing/__init__.py",
    "backend/training/preprocessing/aifit_features.py",            # group constants
    "backend/training/preprocessing/angular_features.py",
    "backend/training/preprocessing/dataset_builder_v5.py",
    "backend/training/preprocessing/dataset_builder_v5_1.py",      # NEW
    "backend/training/preprocessing/synthetic_perturbation.py",    # NEW
    "backend/training/preprocessing/fit3d_loader.py",
    "backend/training/preprocessing/joint_mapping.py",
    "backend/training/preprocessing/mediapipe_extractor.py",
    "backend/training/preprocessing/normalize.py",
    "backend/training/models/__init__.py",
    "backend/training/models/st_gcn.py",
]

DATASET_FILES_V5_1 = [
    "backend/data/v5_1_dataset/train.npz",
    "backend/data/v5_1_dataset/val.npz",
    "backend/data/v5_1_dataset/val_perturbed.npz",
    "backend/data/v5_1_dataset/test.npz",
    "backend/data/v5_1_dataset/dataset_info.json",
    "backend/data/v5_1_dataset/exercise_labels.json",
]


def build_zip(out_path: Path, files: list, strip_prefix: str | None = None):
    print(f"Building {out_path.name} ...")
    n = 0
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            full = PROJECT_ROOT / rel
            if not full.exists():
                if rel.endswith("__init__.py"):
                    full.parent.mkdir(parents=True, exist_ok=True)
                    full.write_text("")
                else:
                    print(f"  WARN missing: {rel}")
                    continue
            arc = rel
            if strip_prefix and rel.startswith(strip_prefix):
                arc = rel[len(strip_prefix):]
            zf.write(full, arcname=arc)
            n += 1
    size_mb = out_path.stat().st_size / 1e6
    print(f"  wrote {n} files, {size_mb:.2f} MB")


# ── Main ─────────────────────────────────────────────────────────────────────


def build_v5_1_ssl_pretrain_notebook():
    """SSL pretrain rebuilt for v5.1 dataset (in case the v5.0 SSL output is gone).

    Reads from v5_1_dataset.zip and filters to CLEAN samples only
    (perturb_severity_idx == 0) so SSL trains on the same pose distribution
    as v5.0 originally did.
    """
    cells = [
        md("""
# FitNova v5.1 — SSL Pretrain (rebuild, reads v5_1_dataset.zip)

Use this only if `ssl_encoder.weights.h5` is missing from your Drive
`MyDrive/fitnova_v5_results/` folder. If it's there, skip this and go
straight to `v5_1_supervised_train.ipynb`.

This notebook is functionally identical to v5.0's SSL pretrain — it just
reads the v5.1 dataset and filters to clean samples (the SSL distribution
should match what supervised will see at clean inputs).

**Cost:** ~25–45 min on L4. ~0.7 Colab compute units.

**Inputs (MyDrive/fitnova_v5/):**
- `fitnova_v5_1_src.zip`
- `v5_1_dataset.zip`

**Outputs (MyDrive/fitnova_v5_results/):**
- `ssl_encoder.weights.h5`  ← what `v5_1_supervised_train.ipynb` reads
- `ssl_history.json`
- `ssl_checkpoints/epoch_NN.weights.h5` (rolling 3 latest, for crash-resume)
"""),
        CELL_GPU_DRIVE,
        CELL_EXTRACT_SRC,
        CELL_EXTRACT_DATA,
        code("""
# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
from tensorflow import keras
from backend.training.models.st_gcn import build_v5_ssl_model

SSL_DIR     = f"{RESULTS_DIR}/ssl_checkpoints"
SSL_HISTORY = f"{RESULTS_DIR}/ssl_history.json"
SSL_FINAL   = f"{RESULTS_DIR}/ssl_encoder.weights.h5"
os.makedirs(SSL_DIR, exist_ok=True)
"""),
        code("""
# ── Hyperparameters (same as v5.0 SSL) ───────────────────────────────────────
EPOCHS     = 20
BATCH_SIZE = 32
MASK_RATIO = 0.30
LR         = 1e-3
SEED       = 42

T_FRAMES = DATASET_INFO["target_frames"]
J        = DATASET_INFO["n_canonical_joints"]
"""),
        code("""
# ── Load training pose (clean samples only) ──────────────────────────────────
train_npz = np.load(f"{DATA_DIR}/train.npz")
val_npz   = np.load(f"{DATA_DIR}/val.npz")

train_clean_mask = train_npz["perturb_severity_idx"] == 0
val_clean_mask   = val_npz["perturb_severity_idx"] == 0
train_pose = train_npz["pose"][train_clean_mask].astype(np.float32)
val_pose   = val_npz["pose"][val_clean_mask].astype(np.float32)

print(f"train_pose: {train_pose.shape}  (clean only out of {len(train_npz['pose'])} total)")
print(f"val_pose:   {val_pose.shape}    (clean only out of {len(val_npz['pose'])} total)")
print(f"finite:     {np.isfinite(train_pose).all()}, {np.isfinite(val_pose).all()}")
"""),
        code("""
# ── Build masked-joint dataset (random per-joint per-frame mask) ─────────────
def make_ssl_dataset(pose, mask_ratio, batch_size, shuffle, seed):
    pose = pose.astype(np.float32)
    pose_xyz = pose[..., :3]
    n = len(pose)

    def gen():
        rng = np.random.default_rng(seed)
        order = np.arange(n)
        while True:
            if shuffle:
                rng.shuffle(order)
            for i in range(0, n, batch_size):
                idx  = order[i:i+batch_size]
                p    = pose[idx].copy()
                mask = (rng.random((*p.shape[:3], 1)) < mask_ratio).astype(np.float32)
                p_masked = p.copy()
                p_masked[..., :3] *= (1.0 - mask)
                yield (
                    {"pose_masked": p_masked, "mask": mask},
                    pose_xyz[idx],
                )

    sig = (
        {
            "pose_masked": tf.TensorSpec(shape=(None, T_FRAMES, J, 4), dtype=tf.float32),
            "mask":        tf.TensorSpec(shape=(None, T_FRAMES, J, 1), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(None, T_FRAMES, J, 3), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(2)

steps_per_epoch  = max(1, len(train_pose) // BATCH_SIZE)
val_steps        = max(1, len(val_pose)   // BATCH_SIZE)
print(f"steps/epoch: {steps_per_epoch}   val steps: {val_steps}")

train_ds = make_ssl_dataset(train_pose, MASK_RATIO, BATCH_SIZE, shuffle=True,  seed=SEED)
val_ds   = make_ssl_dataset(val_pose,   MASK_RATIO, BATCH_SIZE, shuffle=False, seed=SEED + 1)
"""),
        code("""
# ── Build model ──────────────────────────────────────────────────────────────
def masked_recon_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

ssl_model = build_v5_ssl_model(
    target_frames=T_FRAMES,
    n_joints=J,
    n_pose_channels=4,
)
ssl_model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=LR, weight_decay=1e-4, clipnorm=1.0),
    loss={"pose_recon": masked_recon_loss},
)
ssl_model.summary(line_length=120)
"""),
        code("""
# ── Crash-resume detection ──────────────────────────────────────────────────
def find_latest_epoch(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return 0
    files = [f for f in os.listdir(ckpt_dir) if f.startswith("epoch_") and f.endswith(".weights.h5")]
    if not files:
        return 0
    return max(int(f.split("_")[1].split(".")[0]) for f in files)

initial_epoch = find_latest_epoch(SSL_DIR)
if initial_epoch > 0:
    last_ckpt = f"{SSL_DIR}/epoch_{initial_epoch:02d}.weights.h5"
    print(f"Resuming from epoch {initial_epoch} ({last_ckpt})")
    ssl_model.load_weights(last_ckpt)
else:
    print("Fresh start (no prior SSL checkpoints found)")

history_acc = {"loss": [], "val_loss": []}
if os.path.isfile(SSL_HISTORY):
    with open(SSL_HISTORY) as f:
        history_acc = json.load(f)
    history_acc.setdefault("loss", [])
    history_acc.setdefault("val_loss", [])
    print(f"Loaded prior history: {len(history_acc['loss'])} epochs")
"""),
        code("""
# ── Per-epoch checkpoint callback ───────────────────────────────────────────
class DriveCheckpoint(keras.callbacks.Callback):
    def __init__(self, ckpt_dir, history_path, history_acc, total_epochs):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.history_path = history_path
        self.history_acc  = history_acc
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        ep1 = epoch + 1
        ckpt = f"{self.ckpt_dir}/epoch_{ep1:02d}.weights.h5"
        self.model.save_weights(ckpt)
        self.history_acc["loss"].append(float(logs.get("loss", 0.0)))
        self.history_acc["val_loss"].append(float(logs.get("val_loss", 0.0)))
        with open(self.history_path, "w") as f:
            json.dump(self.history_acc, f, indent=2)
        files = sorted(
            f for f in os.listdir(self.ckpt_dir)
            if f.startswith("epoch_") and f.endswith(".weights.h5")
        )
        for f in files[:-3]:
            try:
                os.remove(os.path.join(self.ckpt_dir, f))
            except OSError:
                pass
        print(f"  [saved {os.path.basename(ckpt)}]")
"""),
        code("""
# ── Train ────────────────────────────────────────────────────────────────────
if initial_epoch >= EPOCHS:
    print(f"Already trained for {initial_epoch} epochs — nothing to do. Skip to next cell.")
else:
    callbacks = [DriveCheckpoint(SSL_DIR, SSL_HISTORY, history_acc, EPOCHS)]
    ssl_model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=2,
    )
"""),
        code("""
# ── Save final weights and summarise ─────────────────────────────────────────
ssl_model.save_weights(SSL_FINAL)
print(f"Saved final SSL encoder to {SSL_FINAL}")
print(f"  size: {os.path.getsize(SSL_FINAL) / 1e6:.2f} MB")

with open(SSL_HISTORY) as f:
    h = json.load(f)
print(f"\\nFinal SSL stats over {len(h['loss'])} epochs:")
print(f"  train loss: first={h['loss'][0]:.6f}  last={h['loss'][-1]:.6f}  best={min(h['loss']):.6f}")
print(f"  val   loss: first={h['val_loss'][0]:.6f}  last={h['val_loss'][-1]:.6f}  best={min(h['val_loss']):.6f}")
print()
print("Done. Now go run v5_1_supervised_train.ipynb.")
"""),
    ]
    return write_notebook("v5_1_ssl_pretrain.ipynb", cells)


def main():
    print("== v5.1 Notebooks ==")
    build_v5_1_ssl_pretrain_notebook()
    build_v5_1_supervised_notebook()
    print()
    print("== v5.1 Zips ==")
    build_zip(PROJECT_ROOT / "fitnova_v5_1_src.zip", SRC_ROOTS_V5_1)
    build_zip(PROJECT_ROOT / "v5_1_dataset.zip", DATASET_FILES_V5_1,
              strip_prefix="backend/data/v5_1_dataset/")
    print()
    print("=" * 70)
    print("Next:")
    print("  1. Upload fitnova_v5_1_src.zip + v5_1_dataset.zip to MyDrive/fitnova_v5/")
    print("  2. If you're missing ssl_encoder.weights.h5 in MyDrive/fitnova_v5_results/:")
    print("     run v5_1_ssl_pretrain.ipynb FIRST (~25-45 min)")
    print("  3. Then run v5_1_supervised_train.ipynb (~60-90 min on L4)")
    print("  4. Download weights to backend/models/form_model_v5_1/")


if __name__ == "__main__":
    main()
