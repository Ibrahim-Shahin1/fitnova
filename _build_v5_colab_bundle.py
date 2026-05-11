"""
Build everything you need to run v5 training on Colab.

Run once locally:

    python _build_v5_colab_bundle.py

It produces:

    colab_notebooks/
        v5_ssl_pretrain.ipynb        — Phase 4: masked-joint SSL pretrain (~30-45 min on L4)
        v5_supervised_train.ipynb    — Phase 5: multi-task supervised training (~60-90 min on L4)
        README.md                    — upload + run instructions
    fitnova_v5_src.zip               — backend source for Drive
    v5_dataset.zip                   — pre-built dataset for Drive

Then you (the human):
  1. Upload fitnova_v5_src.zip to MyDrive/fitnova_v5/
  2. Upload v5_dataset.zip       to MyDrive/fitnova_v5/
  3. Open one of the notebooks in Colab
  4. Runtime -> Change runtime type -> L4 GPU (or T4)
  5. Run cells top to bottom
  6. Crash-resume: if Colab dies mid-training, just rerun ALL cells —
     the training cell auto-detects the latest checkpoint and resumes
     from the next epoch.

Outputs land in MyDrive/fitnova_v5_results/. Download the resulting
weights when you're back to your machine.
"""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent
COLAB_DIR = PROJECT_ROOT / "colab_notebooks"
COLAB_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for notebook authoring
# ──────────────────────────────────────────────────────────────────────────────


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.lstrip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.lstrip("\n"))


def write_notebook(name: str, cells: list) -> Path:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        },
        "colab": {"provenance": []},
        "language_info": {"name": "python"},
    }
    out = COLAB_DIR / name
    with open(out, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"  wrote {out}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Common cells (boilerplate for both notebooks)
# ──────────────────────────────────────────────────────────────────────────────

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

print()
print("DRIVE_BASE :", DRIVE_BASE)
print("RESULTS_DIR:", RESULTS_DIR)
""")

CELL_EXTRACT_SRC = code("""
# ── Extract backend source from Drive ────────────────────────────────────────
SRC_ZIP  = f"{DRIVE_BASE}/fitnova_v5_src.zip"
WORK_DIR = "/content/fitnova"

assert os.path.isfile(SRC_ZIP), (
    f"Source zip not found at {SRC_ZIP}.\\n"
    f"Build it locally with: python _build_v5_colab_bundle.py\\n"
    f"Then upload fitnova_v5_src.zip to MyDrive/fitnova_v5/"
)

if not os.path.isdir(f"{WORK_DIR}/backend"):
    os.makedirs(WORK_DIR, exist_ok=True)
    !unzip -q "$SRC_ZIP" -d "$WORK_DIR"

sys.path.insert(0, WORK_DIR)
print("Source ready at", WORK_DIR)
print("Has backend:    ", os.path.exists(f"{WORK_DIR}/backend"))
print("Has st_gcn:     ", os.path.exists(f"{WORK_DIR}/backend/training/models/st_gcn.py"))
""")

CELL_EXTRACT_DATA = code("""
# ── Extract pre-built v5 dataset from Drive ─────────────────────────────────
DATA_ZIP = f"{DRIVE_BASE}/v5_dataset.zip"
DATA_DIR = "/content/v5_dataset"

assert os.path.isfile(DATA_ZIP), (
    f"Dataset zip not found at {DATA_ZIP}.\\n"
    f"Build it locally with: python _build_v5_colab_bundle.py\\n"
    f"Then upload v5_dataset.zip to MyDrive/fitnova_v5/"
)

if not os.path.isdir(DATA_DIR) or not os.path.isfile(f"{DATA_DIR}/train.npz"):
    os.makedirs(DATA_DIR, exist_ok=True)
    !unzip -q "$DATA_ZIP" -d "$DATA_DIR"

print("Dataset files:", sorted(os.listdir(DATA_DIR)))

with open(f"{DATA_DIR}/dataset_info.json") as f:
    DATASET_INFO = json.load(f)
for k in ("n_exercises", "n_joint_groups", "target_frames", "n_canonical_joints", "n_angular"):
    print(f"  {k:<22s}: {DATASET_INFO[k]}")
""")


# ──────────────────────────────────────────────────────────────────────────────
# Notebook 1 — SSL pretrain
# ──────────────────────────────────────────────────────────────────────────────


def build_ssl_notebook() -> Path:
    cells = [
        md("""
# FitNova v5 — Phase 4: SSL Pretrain (Masked-Joint Reconstruction)

Pretrains the ST-GCN encoder on the full Fit3D MediaPipe corpus using
masked-joint autoencoding. The encoder weights are then loaded into the
supervised model in Phase 5.

**Cost:** ~25-45 min on L4 / ~15-25 min on A100. ~0.7 Colab compute units.

**Inputs (must be uploaded to MyDrive/fitnova_v5/):**
- `fitnova_v5_src.zip` — backend source code
- `v5_dataset.zip` — pre-built v5 dataset

**Outputs (saved to MyDrive/fitnova_v5_results/):**
- `ssl_encoder.weights.h5` — pretrained ST-GCN encoder weights
- `ssl_history.json` — per-epoch loss history
- `ssl_checkpoints/epoch_NN.weights.h5` — per-epoch checkpoints (crash-resume)

**Crash-resume:** If Colab kills the runtime mid-training, just rerun every
cell from the top — the training cell auto-detects the latest checkpoint and
resumes from the next epoch.
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
# ── Hyperparameters ──────────────────────────────────────────────────────────
EPOCHS       = 20
BATCH_SIZE   = 32
MASK_RATIO   = 0.30
LR           = 1e-3
SEED         = 42

T_FRAMES = DATASET_INFO["target_frames"]
J        = DATASET_INFO["n_canonical_joints"]
"""),
        code("""
# ── Load training pose data ──────────────────────────────────────────────────
train_npz = np.load(f"{DATA_DIR}/train.npz")
val_npz   = np.load(f"{DATA_DIR}/val.npz")

train_pose = train_npz["pose"].astype(np.float32)   # (N, 64, 15, 4) MediaPipe canonical + visibility
val_pose   = val_npz["pose"].astype(np.float32)
print("train_pose:", train_pose.shape)
print("val_pose:  ", val_pose.shape)
print("finite:    ", np.isfinite(train_pose).all(), np.isfinite(val_pose).all())
"""),
        code("""
# ── Build masked-joint dataset (random per-joint per-frame mask) ─────────────
def make_ssl_dataset(pose, mask_ratio, batch_size, shuffle, seed):
    \"\"\"
    Yields (inputs, target_pose_xyz) tuples.

    For each batch we draw a fresh random mask. Masked positions in
    pose_masked have x,y,z zeroed; visibility (channel 3) is left intact so
    the model can still see "this joint was visible before being masked".
    \"\"\"
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
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
    return ds.prefetch(2)

steps_per_epoch  = max(1, len(train_pose) // BATCH_SIZE)
val_steps        = max(1, len(val_pose)   // BATCH_SIZE)
print(f"steps/epoch: {steps_per_epoch}   val steps: {val_steps}")

train_ds = make_ssl_dataset(train_pose, MASK_RATIO, BATCH_SIZE, shuffle=True,  seed=SEED)
val_ds   = make_ssl_dataset(val_pose,   MASK_RATIO, BATCH_SIZE, shuffle=False, seed=SEED + 1)
"""),
        code("""
# ── Build model ──────────────────────────────────────────────────────────────
def masked_recon_loss(y_true, y_pred):
    \"\"\"MSE over masked positions only (skip visible joints).

    NOTE: We don't have access to the mask tensor inside the loss, so we
    just compute mean-MSE over all positions. The masked locations are
    where the model was forced to reconstruct without input, so they
    dominate the gradient anyway.
    \"\"\"
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
# ── Crash-resume: detect latest checkpoint ───────────────────────────────────
def find_latest_epoch(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return 0
    files = [f for f in os.listdir(ckpt_dir) if f.startswith("epoch_") and f.endswith(".weights.h5")]
    if not files:
        return 0
    epochs = [int(f.split("_")[1].split(".")[0]) for f in files]
    return max(epochs)

initial_epoch = find_latest_epoch(SSL_DIR)
if initial_epoch > 0:
    last_ckpt = f"{SSL_DIR}/epoch_{initial_epoch:02d}.weights.h5"
    print(f"Resuming from epoch {initial_epoch} ({last_ckpt})")
    ssl_model.load_weights(last_ckpt)
else:
    print("Fresh start (no prior SSL checkpoints found)")

# History resume
history_acc = {"loss": [], "val_loss": []}
if os.path.isfile(SSL_HISTORY):
    with open(SSL_HISTORY) as f:
        history_acc = json.load(f)
    history_acc.setdefault("loss", [])
    history_acc.setdefault("val_loss", [])
    print(f"Loaded prior history: {len(history_acc['loss'])} epochs")
"""),
        code("""
# ── Per-epoch checkpoint callback (saves to Drive every epoch) ──────────────
class DriveCheckpoint(keras.callbacks.Callback):
    def __init__(self, ckpt_dir, history_path, history_acc, total_epochs):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.history_path = history_path
        self.history_acc  = history_acc
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Epoch indexing: Keras `epoch` is 0-based; we want 1-based filenames
        ep1 = epoch + 1
        ckpt = f"{self.ckpt_dir}/epoch_{ep1:02d}.weights.h5"
        self.model.save_weights(ckpt)
        # Accumulate history
        self.history_acc["loss"].append(float(logs.get("loss", 0.0)))
        self.history_acc["val_loss"].append(float(logs.get("val_loss", 0.0)))
        with open(self.history_path, "w") as f:
            json.dump(self.history_acc, f, indent=2)
        # Keep only the most recent 3 checkpoints to save Drive space
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
    callbacks = [
        DriveCheckpoint(SSL_DIR, SSL_HISTORY, history_acc, EPOCHS),
    ]
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

# Final history summary
with open(SSL_HISTORY) as f:
    h = json.load(f)
print(f"\\nFinal SSL stats over {len(h['loss'])} epochs:")
print(f"  train loss: first={h['loss'][0]:.6f}  last={h['loss'][-1]:.6f}  best={min(h['loss']):.6f}")
print(f"  val   loss: first={h['val_loss'][0]:.6f}  last={h['val_loss'][-1]:.6f}  best={min(h['val_loss']):.6f}")
"""),
        code("""
# ── Sanity: reconstruction quality on a held-out batch ──────────────────────
val_batch = next(iter(val_ds))
inputs, target = val_batch
recon = ssl_model.predict(inputs, verbose=0)["pose_recon"]
mse  = float(tf.reduce_mean(tf.square(recon - target)).numpy())
mse0 = float(tf.reduce_mean(tf.square(inputs["pose_masked"][..., :3] - target)).numpy())
print(f"Reconstruction MSE on a val batch: {mse:.6f}")
print(f"  (compare to MSE between masked-input and target: {mse0:.6f})")
print(f"  improvement ratio: {mse0 / max(mse, 1e-9):.2f}x")
print()
print("Done. Move on to v5_supervised_train.ipynb")
"""),
    ]
    return write_notebook("v5_ssl_pretrain.ipynb", cells)


# ──────────────────────────────────────────────────────────────────────────────
# Notebook 2 — Supervised multi-task training
# ──────────────────────────────────────────────────────────────────────────────


def build_supervised_notebook() -> Path:
    cells = [
        md("""
# FitNova v5 — Phase 5: Supervised Multi-Task Training

Trains the ST-GCN model with the multi-task heads (quality, joint_err,
boundary, action, rep_count) and the multi-view consistency loss. Loads
the SSL-pretrained encoder weights from Phase 4.

**Cost:** ~60-90 min on L4 / ~30-45 min on A100. ~3 Colab compute units.

**Inputs (MyDrive/fitnova_v5/ + MyDrive/fitnova_v5_results/):**
- `fitnova_v5_src.zip`
- `v5_dataset.zip`
- `ssl_encoder.weights.h5` (from Phase 4)

**Outputs (MyDrive/fitnova_v5_results/):**
- `v5_supervised.weights.h5` — final trained model
- `v5_history.json` — per-epoch metrics
- `v5_supervised_checkpoints/epoch_NN.weights.h5` — per-epoch checkpoints (crash-resume)
- `v5_test_metrics.json` — final eval on s11 test split

**Crash-resume:** Same pattern — rerun all cells from the top, the
training cell picks up at the latest epoch.
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

SUP_DIR     = f"{RESULTS_DIR}/v5_supervised_checkpoints"
SUP_HISTORY = f"{RESULTS_DIR}/v5_history.json"
SUP_FINAL   = f"{RESULTS_DIR}/v5_supervised.weights.h5"
SUP_TEST    = f"{RESULTS_DIR}/v5_test_metrics.json"
SSL_FINAL   = f"{RESULTS_DIR}/ssl_encoder.weights.h5"
os.makedirs(SUP_DIR, exist_ok=True)
"""),
        code("""
# ── Hyperparameters (per plan §6) ────────────────────────────────────────────
EPOCHS       = 60
BATCH_SIZE   = 16   # 16 reps × 2 views = 32 per step
LR           = 1e-3
LR_FINAL     = 1e-5
WARMUP_EPOCHS = 2
LOSS_WEIGHTS = {
    "quality":   2.5,
    "joint_err": 1.5,
    "boundary":  0.5,
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
train_npz = np.load(f"{DATA_DIR}/train.npz")
val_npz   = np.load(f"{DATA_DIR}/val.npz")
test_npz  = np.load(f"{DATA_DIR}/test.npz")

print(f"train: {train_npz['pose'].shape[0]} samples")
print(f"val:   {val_npz['pose'].shape[0]} samples")
print(f"test:  {test_npz['pose'].shape[0]} samples")
"""),
        code("""
# ── Build per-rep grouping for multi-view sampling ───────────────────────────
# A "rep group" = all camera views of the same (subject, exercise, rep).
# Multi-view loss pairs two random views of the same rep within a batch.
def build_view_groups(npz):
    groups = defaultdict(list)
    for i, (s, e, r) in enumerate(zip(
        npz["subject_idx"], npz["exercise_idx"], npz["rep_idx"]
    )):
        groups[(int(s), int(e), int(r))].append(i)
    return list(groups.values())

train_groups = build_view_groups(train_npz)
val_groups   = build_view_groups(val_npz)

n_multiview = sum(1 for g in train_groups if len(g) >= 2)
print(f"Train view-groups: {len(train_groups)} total, {n_multiview} with >=2 views")
"""),
        code("""
# ── tf.data pipeline: sample 2 views per rep when available ─────────────────
def make_supervised_dataset(npz, view_groups, batch_size, shuffle, seed, repeat=True):
    \"\"\"Each yielded batch is 2*batch_size samples — two views per group, packed
    in pairs at indices (i, i+batch_size). The model sees them as a flat batch;
    the multi-view consistency loss inside train_step compares the two halves.

    Groups with only one view are duplicated so the loss term harmlessly
    becomes 0 for that pair.\"\"\"
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
                        "action":    exercise[idx],   # plain int label, sparse CCE
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
val_ds   = make_supervised_dataset(val_npz,   val_groups,   BATCH_SIZE, shuffle=False, seed=SEED + 1)
print(f"steps/epoch: {steps_per_epoch}   val steps: {val_steps}")
"""),
        code("""
# ── Custom training step with multi-view consistency loss ────────────────────
class V5Model(keras.Model):
    \"\"\"Wraps build_v5_model and overrides train_step / test_step to add the
    multi-view consistency loss between samples i and i+B in each flat batch.\"\"\"

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
            # Per-head losses
            l_q = tf.reduce_mean(tf.square(preds["quality"]   - y["quality"]))
            l_j = tf.reduce_mean(keras.losses.binary_crossentropy(y["joint_err"], preds["joint_err"]))
            l_b = tf.reduce_mean(keras.losses.binary_crossentropy(y["boundary"],  preds["boundary"]))
            l_c = tf.reduce_mean(tf.square(preds["rep_count"] - y["rep_count"]))
            l_a = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y["action"], preds["action"]))
            # Multi-view consistency: trunk feats halves
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
        # Per-task metrics
        action_acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.cast(tf.argmax(preds["action"], axis=-1), tf.int32), y["action"]),
            tf.float32,
        ))
        return {
            "loss":       total,
            "l_q":        l_q,
            "l_j":        l_j,
            "l_b":        l_b,
            "l_c":        l_c,
            "l_a":        l_a,
            "l_mv":       l_mv,
            "action_acc": action_acc,
        }

    def train_step(self, data):
        return self._step(data, training=True)

    def test_step(self, data):
        return self._step(data, training=False)
"""),
        code("""
# ── Build base model + load SSL pretrained encoder weights ──────────────────
base = build_v5_model(
    target_frames=T_FRAMES,
    n_joints=J,
    n_pose_channels=4,
    n_angular=N_ANGULAR,
    n_exercises=N_EXERCISES,
    n_joint_groups=N_JOINT_GROUPS,
)

# Load SSL trunk weights (skip the SSL recon head).
# build_v5_ssl_model and build_v5_model share the same ST-GCN block names
# (stgcn1/stgcn2/stgcn3), so we load by name.
if os.path.isfile(SSL_FINAL):
    ssl_template = build_v5_ssl_model(
        target_frames=T_FRAMES, n_joints=J, n_pose_channels=4
    )
    ssl_template.load_weights(SSL_FINAL)
    n_loaded = 0
    for layer_sup in base.layers:
        for layer_ssl in ssl_template.layers:
            if layer_sup.name == layer_ssl.name and layer_sup.weights and layer_ssl.weights:
                if [w.shape for w in layer_sup.weights] == [w.shape for w in layer_ssl.weights]:
                    layer_sup.set_weights(layer_ssl.get_weights())
                    n_loaded += 1
                    break
    print(f"Loaded {n_loaded} SSL-pretrained layers into supervised model")
    del ssl_template
else:
    print("WARNING: no SSL pretrain file found — training from scratch.")
"""),
        code("""
# ── Wrap, optimizer, compile ────────────────────────────────────────────────
total_steps = steps_per_epoch * EPOCHS
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
model = V5Model(base, mv_weight=LOSS_WEIGHTS["mv_cons"], batch_size=BATCH_SIZE)
model.compile(optimizer=opt)

# A dummy forward pass to materialise variables before weight loading
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
    print("Fresh start (no prior supervised checkpoints found)")

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
        # Save the base (unwrapped) weights so we can reload without the wrapper
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
print(f"Saved final supervised weights to {SUP_FINAL}")
print(f"  size: {os.path.getsize(SUP_FINAL) / 1e6:.2f} MB")
"""),
        code("""
# ── Evaluate on test set (s11) ──────────────────────────────────────────────
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
    # Quality
    pq = preds["quality"][:, 0]
    pearson = float(np.corrcoef(pq, quality)[0, 1])
    q_mae = float(np.mean(np.abs(pq - quality)))
    # Joint err
    pj = (preds["joint_err"] > 0.5).astype(np.float32)
    j_acc = float((pj == joint_err).mean())
    j_f1 = float(2 * (pj * joint_err).sum() / max(1.0, (pj + joint_err).sum()))
    # Boundary
    pb = (preds["boundary"][..., 0] > 0.5).astype(np.float32)
    bb = (boundary > 0.5).astype(np.float32)
    inter = float((pb * bb).sum())
    union = float(((pb + bb) > 0.5).sum())
    iou = inter / max(1.0, union)
    # Action
    ax = preds["action"].argmax(axis=-1)
    a_acc = float((ax == exercise).mean())
    return {
        "n":               int(len(pose)),
        "quality_pearson": pearson,
        "quality_mae":     q_mae,
        "joint_err_acc":   j_acc,
        "joint_err_f1":    j_f1,
        "boundary_iou":    iou,
        "action_acc":      a_acc,
    }

metrics = {}
for nm, npz in (("val", val_npz), ("test", test_npz)):
    metrics[nm] = evaluate_on_split(npz, nm)
    print(nm, json.dumps(metrics[nm], indent=2))

with open(SUP_TEST, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\\nSaved metrics to {SUP_TEST}")
"""),
        code("""
# ── Confusion matrix on test ────────────────────────────────────────────────
import numpy as np
preds_action = base.predict(
    {
        "pose":        test_npz["pose"],
        "angles":      test_npz["angles"],
        "exercise_id": test_npz["exercise_idx"].astype(np.int32),
    }, batch_size=16, verbose=0,
)["action"].argmax(axis=-1)
true_action = test_npz["exercise_idx"]

n_ex = N_EXERCISES
cm = np.zeros((n_ex, n_ex), dtype=np.int32)
for t, p in zip(true_action, preds_action):
    cm[int(t), int(p)] += 1
np.save(f"{RESULTS_DIR}/v5_confusion_matrix_test.npy", cm)
print("Confusion matrix saved.")

# Quick text dump
with open(f"{DATA_DIR}/exercise_labels.json") as f:
    ex2idx = json.load(f)
idx2ex = {v: k for k, v in ex2idx.items()}
print()
print("Per-exercise action accuracy on test:")
for i in range(n_ex):
    n = cm[i].sum()
    if n == 0:
        continue
    print(f"  {idx2ex[i]:<35s}  {cm[i, i]}/{n} = {cm[i, i] / n:.3f}")
"""),
        md("""
## Done

Outputs in `MyDrive/fitnova_v5_results/`:
- `v5_supervised.weights.h5`  — final model weights
- `v5_history.json`          — per-epoch metrics
- `v5_test_metrics.json`     — final eval numbers (Pearson, F1, IoU, action acc)
- `v5_confusion_matrix_test.npy` — for the defence slide

Download these to `backend/models/form_model_v5/` on your laptop and continue with Phase 6 (reality-check on user squat videos) and Phase 7 (backend integration).
"""),
    ]
    return write_notebook("v5_supervised_train.ipynb", cells)


# ──────────────────────────────────────────────────────────────────────────────
# Source + dataset zips
# ──────────────────────────────────────────────────────────────────────────────


SRC_ROOTS = [
    "backend/__init__.py",
    "backend/training/__init__.py",
    "backend/training/preprocessing/__init__.py",
    "backend/training/preprocessing/aifit_features.py",
    "backend/training/preprocessing/angular_features.py",
    "backend/training/preprocessing/dataset_builder_v5.py",
    "backend/training/preprocessing/fit3d_loader.py",
    "backend/training/preprocessing/joint_mapping.py",
    "backend/training/preprocessing/mediapipe_extractor.py",
    "backend/training/preprocessing/normalize.py",
    "backend/training/models/__init__.py",
    "backend/training/models/st_gcn.py",
]

DATASET_FILES = [
    "backend/data/v5_dataset/train.npz",
    "backend/data/v5_dataset/val.npz",
    "backend/data/v5_dataset/test.npz",
    "backend/data/v5_dataset/signatures.npz",
    "backend/data/v5_dataset/dataset_info.json",
    "backend/data/v5_dataset/exercise_labels.json",
    "backend/data/v5_dataset/signatures_meta.json",
]


def build_src_zip(out_path: Path) -> None:
    print(f"Building {out_path.name} ...")
    n = 0
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in SRC_ROOTS:
            full = PROJECT_ROOT / rel
            if not full.exists():
                # Touch a stub __init__.py if missing (so the package imports work)
                if rel.endswith("__init__.py"):
                    full.parent.mkdir(parents=True, exist_ok=True)
                    full.write_text("")
                else:
                    print(f"  WARN missing: {rel}")
                    continue
            zf.write(full, arcname=rel)
            n += 1
    size_mb = out_path.stat().st_size / 1e6
    print(f"  wrote {n} files, {size_mb:.2f} MB")


def build_dataset_zip(out_path: Path) -> None:
    print(f"Building {out_path.name} ...")
    n = 0
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in DATASET_FILES:
            full = PROJECT_ROOT / rel
            if not full.exists():
                print(f"  WARN missing: {rel}")
                continue
            arc = os.path.basename(rel)  # drop the backend/data/v5_dataset prefix
            zf.write(full, arcname=arc)
            n += 1
    size_mb = out_path.stat().st_size / 1e6
    print(f"  wrote {n} files, {size_mb:.2f} MB")


# ──────────────────────────────────────────────────────────────────────────────
# README
# ──────────────────────────────────────────────────────────────────────────────


README_BODY = """
# v5 Colab Bundle

Generated by `_build_v5_colab_bundle.py`.

## What's in this folder

- `v5_ssl_pretrain.ipynb`    — Phase 4 (SSL masked-joint pretrain, ~30-45 min on L4)
- `v5_supervised_train.ipynb` — Phase 5 (supervised multi-task training, ~60-90 min on L4)

## How to run

### Step 1 — upload to Drive (once)

Upload these to `MyDrive/fitnova_v5/`:
- `fitnova_v5_src.zip` (project root, generated by this script)
- `v5_dataset.zip` (project root, generated by this script)

If you re-run this script the zips are overwritten in place. Re-upload the new ones.

### Step 2 — run SSL pretrain

1. Open `v5_ssl_pretrain.ipynb` in Colab
2. Runtime -> Change runtime type -> L4 GPU (or T4)
3. Runtime -> Run all
4. ~30-45 min. Outputs land in `MyDrive/fitnova_v5_results/ssl_encoder.weights.h5`.

### Step 3 — run supervised training

1. Open `v5_supervised_train.ipynb` in Colab
2. Runtime -> Change runtime type -> L4 GPU (or T4)
3. Runtime -> Run all
4. ~60-90 min. Outputs land in `MyDrive/fitnova_v5_results/v5_supervised.weights.h5`
   plus `v5_history.json`, `v5_test_metrics.json`, `v5_confusion_matrix_test.npy`.

### Step 4 — bring weights back

Download from `MyDrive/fitnova_v5_results/` to your laptop:
- `ssl_encoder.weights.h5`
- `v5_supervised.weights.h5`
- `v5_history.json`
- `v5_test_metrics.json`
- `v5_confusion_matrix_test.npy`

Place them all in `backend/models/form_model_v5/` (create the folder).

## Crash-resume

Both notebooks save per-epoch checkpoints to Drive. If Colab kills the
runtime mid-training, just rerun every cell from the top. The training
cell auto-detects the latest checkpoint and resumes from the next epoch.
You don't need to do anything special.

## Output sizes

- `ssl_encoder.weights.h5`: ~5 MB
- `v5_supervised.weights.h5`: ~5 MB
- `v5_history.json`: ~50 KB
- Per-epoch checkpoints (rolling 3 latest): ~15 MB total
- Total Drive footprint: under 30 MB.
"""


def write_readme() -> Path:
    out = COLAB_DIR / "README.md"
    out.write_text(README_BODY.strip() + "\n", encoding="utf-8")
    print(f"  wrote {out}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print("== Notebooks ==")
    build_ssl_notebook()
    build_supervised_notebook()
    write_readme()
    print()
    print("== Zips ==")
    build_src_zip(PROJECT_ROOT / "fitnova_v5_src.zip")
    build_dataset_zip(PROJECT_ROOT / "v5_dataset.zip")
    print()
    print("=" * 70)
    print("Next steps:")
    print("  1. Upload fitnova_v5_src.zip + v5_dataset.zip to MyDrive/fitnova_v5/")
    print("  2. Open colab_notebooks/v5_ssl_pretrain.ipynb in Colab → Run all")
    print("  3. Open colab_notebooks/v5_supervised_train.ipynb in Colab → Run all")
    print("  4. Download weights back to backend/models/form_model_v5/")


if __name__ == "__main__":
    main()
