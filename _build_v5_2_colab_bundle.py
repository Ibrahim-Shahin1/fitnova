"""
Build v5.2 Colab artifacts.

Run once locally:

    python _build_v5_2_colab_bundle.py

v5.2 changes vs v5.1:
  * Per-group severity multipliers (synthetic_perturbation.py now scales
    knee/shoulder/elbow rotations 1.5–2.0× to equalise canonical-pose
    displacement against hip/back). Source code change baked into
    fitnova_v5_2_src.zip; new dataset already built into v5_2_dataset.zip.
  * Class-weighted BCE on joint_err (positive class weighted 30× in the
    new supervised notebook to fix the joint_err = 0 collapse on knee /
    shoulder / elbow).

Outputs:
    colab_notebooks/v5_2_supervised_train.ipynb
    fitnova_v5_2_src.zip
    v5_2_dataset.zip
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
        "kernelspec":   {"name": "python3", "display_name": "Python 3", "language": "python"},
        "colab":        {"provenance": []},
        "language_info":{"name": "python"},
    }
    out = COLAB_DIR / name
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"  wrote {out}")
    return out


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
# ── Extract v5.2 backend source from Drive ──────────────────────────────────
SRC_ZIP  = f"{DRIVE_BASE}/fitnova_v5_2_src.zip"
WORK_DIR = "/content/fitnova_v5_2"

assert os.path.isfile(SRC_ZIP), (
    f"Source zip not found at {SRC_ZIP}.\\n"
    f"Build it locally: python _build_v5_2_colab_bundle.py\\n"
    f"Then upload fitnova_v5_2_src.zip to MyDrive/fitnova_v5/"
)

shutil.rmtree(WORK_DIR, ignore_errors=True)
os.makedirs(WORK_DIR, exist_ok=True)
!unzip -q "$SRC_ZIP" -d "$WORK_DIR"

if WORK_DIR not in sys.path:
    sys.path.insert(0, WORK_DIR)

print("Source ready at", WORK_DIR)
print("  has st_gcn:                 ", os.path.exists(f"{WORK_DIR}/backend/training/models/st_gcn.py"))
print("  has synthetic_perturbation: ", os.path.exists(f"{WORK_DIR}/backend/training/preprocessing/synthetic_perturbation.py"))
""")

CELL_EXTRACT_DATA = code("""
# ── Extract v5.2 pre-built dataset from Drive ───────────────────────────────
DATA_ZIP = f"{DRIVE_BASE}/v5_2_dataset.zip"
DATA_DIR = "/content/v5_2_dataset"

assert os.path.isfile(DATA_ZIP), (
    f"Dataset zip not found at {DATA_ZIP}.\\n"
    f"Build it locally: python _build_v5_2_colab_bundle.py\\n"
    f"Then upload v5_2_dataset.zip to MyDrive/fitnova_v5/"
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


def build_supervised_v5_2_notebook():
    cells = [
        md("""
# FitNova v5.2 — Supervised retrain with rebalanced perturbations + class-weighted joint_err loss

v5.1 evaluation revealed:
- Action classifier now correct on user phone clips (4/4 squat). Big win.
- Quality range compressed [0.41 .. 0.58] vs target [0.10 .. 1.00].
- Joint_err head DEAD on knee/shoulder/elbow (F1=0). Decent on hip (F1=0.57)
  and back (F1=0.84).

v5.2 fixes both joint_err issues:
1. Per-group severity multipliers in synthetic_perturbation.py (knee/elbow ×2,
   shoulder ×1.5, hip ×1.4, back ×1.0) so all groups produce comparable
   canonical-pose displacement magnitude.
2. Class-weighted BCE on joint_err (pos_weight=30) to fix the imbalance
   collapse — was ~1% positive rate per group, BCE trivially predicted 0.

**Cost:** ~60-90 min on L4. ~3 Colab compute units.

**Inputs:**
- `fitnova_v5_2_src.zip`
- `v5_2_dataset.zip`
- `ssl_encoder.weights.h5` (reused from v5.0)

**Outputs (MyDrive/fitnova_v5_results/):**
- `v5_2_supervised.weights.h5`
- `v5_2_history.json`
- `v5_2_test_metrics.json`
- `v5_2_val_perturbed_metrics.json`
- `v5_2_supervised_checkpoints/epoch_NN.weights.h5`
- `v5_2_confusion_matrix_test.npy`
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

SUP_DIR     = f"{RESULTS_DIR}/v5_2_supervised_checkpoints"
SUP_HISTORY = f"{RESULTS_DIR}/v5_2_history.json"
SUP_FINAL   = f"{RESULTS_DIR}/v5_2_supervised.weights.h5"
SUP_TEST    = f"{RESULTS_DIR}/v5_2_test_metrics.json"
SUP_VPERT   = f"{RESULTS_DIR}/v5_2_val_perturbed_metrics.json"
SSL_FINAL   = f"{RESULTS_DIR}/ssl_encoder.weights.h5"
os.makedirs(SUP_DIR, exist_ok=True)
"""),
        code("""
# ── Hyperparameters ──────────────────────────────────────────────────────────
EPOCHS        = 60
BATCH_SIZE    = 16
LR            = 1e-3
LR_FINAL      = 1e-5
WARMUP_EPOCHS = 2
LOSS_WEIGHTS = {
    "quality":   3.0,
    "joint_err": 1.0,        # lowered from 1.5 — pos_weight=30 already inflates magnitude
    "boundary":  0.3,
    "rep_count": 0.1,
    "action":    0.3,
    "mv_cons":   0.5,
}
JOINT_ERR_POS_WEIGHT = 30.0   # NEW v5.2: weight on positive class for joint_err BCE
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
print(f"train:         {train_npz['pose'].shape[0]} samples")
print(f"val:           {val_npz['pose'].shape[0]} samples")
print(f"val_perturbed: {val_perturbed_npz['pose'].shape[0]} samples")
print(f"test:          {test_npz['pose'].shape[0]} samples")
"""),
        code("""
# ── Build view-groups (same logic as v5.1) ──────────────────────────────────
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
print(f"Train view-groups: {len(train_groups)} ({sum(1 for g in train_groups if len(g)>=2)} with >=2 views)")
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
            if shuffle: rng.shuffle(order)
            for i in range(0, n_groups, batch_size):
                idx_groups = order[i:i + batch_size]
                if len(idx_groups) < batch_size: break
                view1, view2 = [], []
                for g_idx in idx_groups:
                    g = view_groups[g_idx]
                    if len(g) >= 2:
                        a, b = rng.choice(g, size=2, replace=False)
                    else:
                        a = b = g[0]
                    view1.append(int(a)); view2.append(int(b))
                idx = np.array(view1 + view2, dtype=np.int64)
                yield (
                    {"pose": pose[idx], "angles": angles[idx], "exercise_id": exercise[idx]},
                    {"quality": quality[idx], "joint_err": joint_err[idx],
                     "boundary": boundary[idx], "rep_count": rep_count[idx],
                     "action": exercise[idx]},
                )
            if not repeat: break

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
# ── Custom training step ─────────────────────────────────────────────────────
# v5.2 NEW: weighted BCE for joint_err (pos_weight=30 to fix class imbalance).
def weighted_bce(y_true, y_pred, pos_weight):
    eps = 1e-7
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    return -tf.reduce_mean(
        pos_weight * y_true * tf.math.log(y_pred)
        + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
    )

class V5_2Model(keras.Model):
    def __init__(self, base_model, mv_weight, pos_weight, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.base = base_model
        self.mv_weight = float(mv_weight)
        self.pos_weight = float(pos_weight)
        self.bs = int(batch_size)

    def call(self, inputs, training=False):
        return self.base(inputs, training=training)

    def _step(self, data, training):
        x, y = data
        with tf.GradientTape(watch_accessed_variables=training) as tape:
            preds = self.base(x, training=training)
            l_q = tf.reduce_mean(tf.square(preds["quality"]   - y["quality"]))
            l_j = weighted_bce(y["joint_err"], preds["joint_err"], self.pos_weight)
            l_b = tf.reduce_mean(keras.losses.binary_crossentropy(y["boundary"],  preds["boundary"]))
            l_c = tf.reduce_mean(tf.square(preds["rep_count"] - y["rep_count"]))
            l_a = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y["action"], preds["action"]))
            B = tf.shape(preds["trunk"])[0]
            half = B // 2
            t1 = preds["trunk"][:half]; t2 = preds["trunk"][half:half * 2]
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
        # Pearson on quality (batch-level rough)
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
        # Per-group joint_err recall (positive class) — useful diagnostic
        je_pred_b = tf.cast(preds["joint_err"] > 0.5, tf.float32)
        je_true_b = y["joint_err"]
        eps = 1e-7
        recall_per_group = tf.reduce_sum(je_pred_b * je_true_b, axis=[0, 1]) / tf.maximum(
            tf.reduce_sum(je_true_b, axis=[0, 1]), eps,
        )
        je_recall_mean = tf.reduce_mean(recall_per_group)
        return {
            "loss": total, "l_q": l_q, "l_j": l_j, "l_b": l_b,
            "l_c": l_c, "l_a": l_a, "l_mv": l_mv,
            "q_pearson": q_pearson, "action_acc": action_acc,
            "je_recall": je_recall_mean,
        }

    def train_step(self, data): return self._step(data, training=True)
    def test_step(self, data):  return self._step(data, training=False)
"""),
        code("""
# ── Build model + load SSL encoder ──────────────────────────────────────────
base = build_v5_model(
    target_frames=T_FRAMES, n_joints=J, n_pose_channels=4,
    n_angular=N_ANGULAR, n_exercises=N_EXERCISES, n_joint_groups=N_JOINT_GROUPS,
)
if os.path.isfile(SSL_FINAL):
    ssl_template = build_v5_ssl_model(
        target_frames=T_FRAMES, n_joints=J, n_pose_channels=4,
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
opt = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4, clipnorm=1.0)
model = V5_2Model(base, mv_weight=LOSS_WEIGHTS["mv_cons"],
                  pos_weight=JOINT_ERR_POS_WEIGHT, batch_size=BATCH_SIZE)
model.compile(optimizer=opt)
_ = base({
    "pose":        np.zeros((1, T_FRAMES, J, 4), dtype=np.float32),
    "angles":      np.zeros((1, T_FRAMES, N_ANGULAR), dtype=np.float32),
    "exercise_id": np.zeros((1,), dtype=np.int32),
})
print(f"Model has {base.count_params():,} params")
"""),
        code("""
# ── Crash-resume + training ─────────────────────────────────────────────────
def find_latest_epoch(ckpt_dir):
    if not os.path.isdir(ckpt_dir): return 0
    files = [f for f in os.listdir(ckpt_dir) if f.startswith("epoch_") and f.endswith(".weights.h5")]
    if not files: return 0
    return max(int(f.split("_")[1].split(".")[0]) for f in files)

initial_epoch = find_latest_epoch(SUP_DIR)
if initial_epoch > 0:
    last_ckpt = f"{SUP_DIR}/epoch_{initial_epoch:02d}.weights.h5"
    print(f"Resuming from epoch {initial_epoch} ({last_ckpt})")
    base.load_weights(last_ckpt)
else:
    print("Fresh start (no prior v5.2 checkpoints)")

history_acc = {}
if os.path.isfile(SUP_HISTORY):
    with open(SUP_HISTORY) as f:
        history_acc = json.load(f)

class DriveCheckpoint(keras.callbacks.Callback):
    def __init__(self, ckpt_dir, history_path, history_acc, base_model):
        super().__init__()
        self.ckpt_dir = ckpt_dir; self.history_path = history_path
        self.history_acc = history_acc; self.base_model = base_model
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}; ep1 = epoch + 1
        ckpt = f"{self.ckpt_dir}/epoch_{ep1:02d}.weights.h5"
        self.base_model.save_weights(ckpt)
        for k, v in logs.items():
            self.history_acc.setdefault(k, []).append(float(v))
        with open(self.history_path, "w") as f:
            json.dump(self.history_acc, f, indent=2)
        files = sorted(f for f in os.listdir(self.ckpt_dir)
                       if f.startswith("epoch_") and f.endswith(".weights.h5"))
        for f in files[:-3]:
            try: os.remove(os.path.join(self.ckpt_dir, f))
            except OSError: pass

cb = DriveCheckpoint(SUP_DIR, SUP_HISTORY, history_acc, base)

if initial_epoch >= EPOCHS:
    print(f"Already trained for {initial_epoch} epochs — skipping.")
else:
    model.fit(
        train_ds, validation_data=val_ds,
        steps_per_epoch=steps_per_epoch, validation_steps=val_steps,
        epochs=EPOCHS, initial_epoch=initial_epoch,
        callbacks=[cb], verbose=2,
    )
"""),
        code("""
# ── Save final weights + per-severity quality breakdown ─────────────────────
base.save_weights(SUP_FINAL)
print(f"Saved {SUP_FINAL}  ({os.path.getsize(SUP_FINAL)/1e6:.2f} MB)")

# Per-severity breakdown (the key v5.2 signal)
print()
print("=== v5.2 per-severity quality on val_perturbed ===")
preds_val = base.predict(
    {"pose": val_perturbed_npz["pose"],
     "angles": val_perturbed_npz["angles"],
     "exercise_id": val_perturbed_npz["exercise_idx"].astype(np.int32)},
    batch_size=16, verbose=0,
)
q_pred = preds_val["quality"][:, 0]
sev_idx = val_perturbed_npz["perturb_severity_idx"]
sev_names = ["clean", "mild", "moderate", "severe"]
sev_targets = [1.0, 0.7, 0.4, 0.1]
print(f"  {'severity':<10s} {'target':>7s}   {'pred μ':>7s}   {'pred σ':>7s}   {'n':>5s}")
for i, name in enumerate(sev_names):
    mask = sev_idx == i
    if mask.sum() == 0: continue
    p = q_pred[mask]
    print(f"  {name:<10s} {sev_targets[i]:>7.2f}   {p.mean():>7.3f}   {p.std():>7.3f}   {int(mask.sum()):>5d}")

# Per-joint-group quality + F1
print()
print("=== v5.2 per-group quality + joint_err F1 (perturbed only) ===")
group_names = ["knee", "hip", "back", "shoulder", "elbow"]
group_idx = val_perturbed_npz["perturb_group"]
je_pred = (preds_val["joint_err"] > 0.5).astype(np.float32)
je_true = val_perturbed_npz["joint_err"].astype(np.float32)
print(f"  {'group':<10s}   {'pred μ Q':>8s}   {'F1':>5s}")
for g_idx, g_name in enumerate(group_names):
    mask = (group_idx == g_idx) & (sev_idx > 0)
    if mask.sum() == 0: continue
    pq = q_pred[mask]
    pp = je_pred[..., g_idx]
    tt = je_true[..., g_idx]
    inter = (pp * tt).sum(); pred_pos = pp.sum(); true_pos = tt.sum()
    eps = 1e-9
    prec = inter / max(pred_pos, eps); rec = inter / max(true_pos, eps)
    f1 = 2 * prec * rec / max(prec + rec, eps)
    print(f"  {g_name:<10s}   {pq.mean():>8.3f}   {f1:>5.3f}")
"""),
        code("""
# ── Test set + confusion matrix ─────────────────────────────────────────────
def evaluate_on_split(npz, batch_size=16):
    preds = base.predict(
        {"pose": npz["pose"], "angles": npz["angles"],
         "exercise_id": npz["exercise_idx"].astype(np.int32)},
        batch_size=batch_size, verbose=0,
    )
    quality = npz["quality"].astype(np.float32)
    pq = preds["quality"][:, 0]
    pearson = float("nan") if quality.std() < 1e-6 else float(np.corrcoef(pq, quality)[0, 1])
    pj = (preds["joint_err"] > 0.5).astype(np.float32)
    je_true = npz["joint_err"].astype(np.float32)
    j_acc = float((pj == je_true).mean())
    eps = 1e-9
    j_f1 = float(2*(pj*je_true).sum() / max(eps, (pj+je_true).sum()))
    pb = (preds["boundary"][..., 0] > 0.5).astype(np.float32)
    bb = (npz["boundary"].astype(np.float32) > 0.5).astype(np.float32)
    inter = float((pb*bb).sum()); union = float(((pb+bb)>0.5).sum())
    iou = inter / max(eps, union)
    a_acc = float((preds["action"].argmax(axis=-1) == npz["exercise_idx"]).mean())
    return {
        "n": int(len(npz["pose"])),
        "quality_pearson": pearson,
        "quality_mae": float(np.mean(np.abs(pq - quality))),
        "quality_pred_mean": float(pq.mean()),
        "quality_pred_std":  float(pq.std()),
        "joint_err_acc": j_acc, "joint_err_f1": j_f1,
        "boundary_iou": iou, "action_acc": a_acc,
    }

test_metrics = evaluate_on_split(test_npz)
print("test:", json.dumps(test_metrics, indent=2))
with open(SUP_TEST, "w") as f: json.dump(test_metrics, f, indent=2)
vpert_metrics = evaluate_on_split(val_perturbed_npz)
print("val_perturbed:", json.dumps(vpert_metrics, indent=2))
with open(SUP_VPERT, "w") as f: json.dump(vpert_metrics, f, indent=2)

# Confusion matrix
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
np.save(f"{RESULTS_DIR}/v5_2_confusion_matrix_test.npy", cm)
print("Saved confusion matrix.")
"""),
        md("""
## Done

Outputs in `MyDrive/fitnova_v5_results/`:
- `v5_2_supervised.weights.h5`
- `v5_2_history.json`
- `v5_2_test_metrics.json`
- `v5_2_val_perturbed_metrics.json`
- `v5_2_confusion_matrix_test.npy`
- `v5_2_supervised_checkpoints/`

Download to `backend/models/form_model_v5_2/` on the laptop, then run
Phase 6 reality-check.
"""),
    ]
    return write_notebook("v5_2_supervised_train.ipynb", cells)


SRC_ROOTS_V5_2 = [
    "backend/__init__.py",
    "backend/training/__init__.py",
    "backend/training/preprocessing/__init__.py",
    "backend/training/preprocessing/aifit_features.py",
    "backend/training/preprocessing/angular_features.py",
    "backend/training/preprocessing/dataset_builder_v5.py",
    "backend/training/preprocessing/dataset_builder_v5_1.py",
    "backend/training/preprocessing/synthetic_perturbation.py",
    "backend/training/preprocessing/fit3d_loader.py",
    "backend/training/preprocessing/joint_mapping.py",
    "backend/training/preprocessing/mediapipe_extractor.py",
    "backend/training/preprocessing/normalize.py",
    "backend/training/models/__init__.py",
    "backend/training/models/st_gcn.py",
]

DATASET_FILES_V5_2 = [
    "backend/data/v5_2_dataset/train.npz",
    "backend/data/v5_2_dataset/val.npz",
    "backend/data/v5_2_dataset/val_perturbed.npz",
    "backend/data/v5_2_dataset/test.npz",
    "backend/data/v5_2_dataset/dataset_info.json",
    "backend/data/v5_2_dataset/exercise_labels.json",
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


def main():
    print("== v5.2 Notebook ==")
    build_supervised_v5_2_notebook()
    print()
    print("== v5.2 Zips ==")
    build_zip(PROJECT_ROOT / "fitnova_v5_2_src.zip", SRC_ROOTS_V5_2)
    build_zip(PROJECT_ROOT / "v5_2_dataset.zip", DATASET_FILES_V5_2,
              strip_prefix="backend/data/v5_2_dataset/")
    print()
    print("=" * 70)
    print("Next:")
    print("  1. Upload fitnova_v5_2_src.zip + v5_2_dataset.zip to MyDrive/fitnova_v5/")
    print("  2. Open colab_notebooks/v5_2_supervised_train.ipynb -> Run all (~60-90 min)")
    print("     (SSL encoder is reused from v5.0; no need to rerun SSL)")
    print("  3. Download weights to backend/models/form_model_v5_2/")


if __name__ == "__main__":
    main()
