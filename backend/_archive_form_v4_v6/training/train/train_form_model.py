"""
Training script for the MT-TCN form detection model.

Usage:
    python -m backend.training.train_form_model

Or with custom paths:
    python -m backend.training.train_form_model \
        --dataset_root "C:/Users/tsh_x/Desktop/FitNova Datasets/fit3d/train/train" \
        --data_dir     "backend/data/form_dataset" \
        --model_dir    "backend/models/form_model"
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf

# ── Allow running from project root ───────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.training.preprocessing.dataset_builder import build_dataset
from backend.training.models.mt_tcn import build_mt_tcn, compile_mt_tcn


# IPython clear_output for Colab (silently ignored in terminal)
try:
    from IPython.display import clear_output as _clear_output
    _IN_JUPYTER = True
except ImportError:
    _IN_JUPYTER = False
    def _clear_output(wait=False): pass


class PeriodicClearOutput(tf.keras.callbacks.Callback):
    """Clears Colab cell output every N epochs to prevent browser lag.

    Reprints a compact header so the user can still see recent progress.
    No-op outside Jupyter/Colab.
    """

    def __init__(self, every: int = 10, keep_last: int = 10):
        super().__init__()
        self.every = every
        self.keep_last = keep_last
        self._lines: list[str] = []

    def on_epoch_end(self, epoch, logs=None):
        if not _IN_JUPYTER:
            return
        logs = logs or {}
        # Build one-line summary for this epoch
        loss      = logs.get("loss", float("nan"))
        val_loss  = logs.get("val_loss", float("nan"))
        ex_acc    = logs.get("exercise_accuracy",
                            logs.get("val_exercise_accuracy", float("nan")))
        vx_acc    = logs.get("val_exercise_accuracy", float("nan"))
        self._lines.append(
            f"ep {epoch+1:3d}  loss {loss:7.4f}  val_loss {val_loss:7.4f}"
            f"  val_ex_acc {vx_acc:5.3f}"
        )
        # Every `every` epochs, clear and reprint last `keep_last` lines
        if (epoch + 1) % self.every == 0:
            _clear_output(wait=True)
            print(f"Training — epoch {epoch+1} (showing last {self.keep_last})")
            for line in self._lines[-self.keep_last:]:
                print(line)


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DATASET_ROOT = r"C:\Users\tsh_x\Desktop\FitNova Datasets\fit3d\train\train"
DEFAULT_DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data", "form_dataset")
DEFAULT_MODEL_DIR    = os.path.join(os.path.dirname(__file__), "..", "models", "form_model")

BATCH_SIZE         = 32
EPOCHS             = 150
PATIENCE           = 25
LEARNING_RATE      = 1e-3
# Fine-tuning defaults (used when --warm-start is given)
FT_EPOCHS          = 30
FT_PATIENCE        = 10
FT_LEARNING_RATE   = 1e-4


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(data_dir: str, split: str):
    """Load one split (train/val) from numpy arrays."""
    d = os.path.join(data_dir, split)
    X_ang = np.load(os.path.join(d, "X_angles.npy"))    # (N, 64, 22)
    X_jnt = np.load(os.path.join(d, "X_joints.npy"))    # (N, 64, 45)
    y_ex  = np.load(os.path.join(d, "y_exercise.npy"))  # (N,)
    y_qlt = np.load(os.path.join(d, "y_quality.npy"))   # (N,)
    y_jnt = np.load(os.path.join(d, "y_joints.npy"))    # (N, 64, 10)
    y_bnd = np.load(os.path.join(d, "y_boundary.npy"))  # (N, 64, 1)
    y_cnt = np.load(os.path.join(d, "y_count.npy"))     # (N,)
    return X_ang, X_jnt, y_ex, y_qlt, y_jnt, y_bnd, y_cnt


def make_tf_dataset(X_ang, X_jnt, y_ex, y_qlt, y_jnt, y_bnd, y_cnt,
                    batch_size: int, shuffle: bool = True,
                    n_exercises: int = 27):
    """Wrap numpy arrays in a tf.data.Dataset."""
    # One-hot encode exercise labels (required by CategoricalCrossentropy w/ label smoothing)
    y_ex_onehot = np.eye(n_exercises, dtype=np.float32)[y_ex.astype(np.int32)]

    inputs = {
        "angular_input": X_ang.astype(np.float32),
        "joints_input":  X_jnt.astype(np.float32),
    }
    targets = {
        "exercise":     y_ex_onehot,
        "quality":      y_qlt[:, np.newaxis].astype(np.float32),
        "joint_errors": y_jnt.astype(np.float32),
        "boundary":     y_bnd.astype(np.float32),
        "rep_count":    y_cnt[:, np.newaxis].astype(np.float32),
    }
    ds = tf.data.Dataset.from_tensor_slices((inputs, targets))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X_ang), seed=42)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_model(model, val_ds, val_y_ex, val_y_qlt, val_y_jnt,
                   output_dir: str,
                   metrics_filename: str = "training_metrics.json",
                   confusion_filename: str | None = None,
                   class_names: list[str] | None = None) -> dict:
    """
    Compute detailed evaluation metrics beyond Keras defaults.
    Saves metrics to ``output_dir/<metrics_filename>``.

    Phase-4 A4 fix: ``metrics_filename`` is parameterised so that the training-
    set and test-set invocations write to distinct files. Previously both calls
    wrote to ``training_metrics.json`` and the test-set call relied on a
    post-hoc ``shutil.copy`` that (because the same function was called twice)
    overwrote the first file — making both outputs identical.

    Phase-4 A6: if ``confusion_filename`` and ``class_names`` are provided,
    save a 27×27 confusion matrix as both ``.npy`` and ``.png`` next to the
    metrics JSON. Used downstream by ``exercise_sanity.py`` for confusion-
    aware mismatch detection.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, mean_absolute_error,
        mean_squared_error
    )
    from scipy.stats import pearsonr

    print("\n── Detailed Evaluation ──────────────────────────────────")

    # Collect predictions
    preds = model.predict(val_ds, verbose=0)
    ex_pred  = np.argmax(preds["exercise"], axis=-1)
    qlt_pred = preds["quality"].squeeze()
    jnt_pred = (preds["joint_errors"] > 0.5).astype(int)  # (N, 64, 10)

    # Exercise classification
    ex_acc = accuracy_score(val_y_ex, ex_pred)
    ex_f1  = f1_score(val_y_ex, ex_pred, average="macro", zero_division=0)
    print(f"  Exercise  — Accuracy: {ex_acc:.4f}  Macro-F1: {ex_f1:.4f}")

    # Form quality
    qlt_mse = mean_squared_error(val_y_qlt, qlt_pred)
    qlt_mae = mean_absolute_error(val_y_qlt, qlt_pred)
    qlt_corr, _ = pearsonr(val_y_qlt, qlt_pred)
    print(f"  Quality   — MSE: {qlt_mse:.4f}  MAE: {qlt_mae:.4f}  Pearson r: {qlt_corr:.4f}")

    # Accuracy at delta thresholds (comparable to AIFit's delta metric)
    for delta in [0.3, 0.5, 0.7]:
        correct = np.abs(val_y_qlt - qlt_pred) < delta
        print(f"  Quality   — Accuracy @ δ={delta}: {correct.mean():.4f}")

    # Per-joint error (collapse time dimension → any error per sample)
    jnt_true_collapsed = (val_y_jnt.max(axis=1) > 0.5).astype(int)  # (N, 10)
    jnt_pred_collapsed = (jnt_pred.max(axis=1) > 0).astype(int)      # (N, 10)
    jnt_f1 = f1_score(jnt_true_collapsed, jnt_pred_collapsed,
                      average="macro", zero_division=0)
    print(f"  Joint err — Macro-F1: {jnt_f1:.4f}")

    # Rep counting (OBO and MAE — directly comparable to AIFit's 0.140 / 0.253)
    # Since we train on single-rep windows, rep_count target is always 1
    cnt_pred = preds["rep_count"].squeeze()
    cnt_mae  = mean_absolute_error(np.ones(len(cnt_pred)), cnt_pred)
    cnt_obo  = np.mean(np.abs(np.ones(len(cnt_pred)) - np.round(cnt_pred)) <= 1)
    print(f"  Rep count — MAE: {cnt_mae:.4f}  OBO: {1-cnt_obo:.4f}")

    metrics = {
        "exercise_accuracy": float(ex_acc),
        "exercise_macro_f1": float(ex_f1),
        "quality_mse":       float(qlt_mse),
        "quality_mae":       float(qlt_mae),
        "quality_pearson_r": float(qlt_corr),
        "quality_acc_d03":   float(np.mean(np.abs(val_y_qlt - qlt_pred) < 0.3)),
        "quality_acc_d05":   float(np.mean(np.abs(val_y_qlt - qlt_pred) < 0.5)),
        "quality_acc_d07":   float(np.mean(np.abs(val_y_qlt - qlt_pred) < 0.7)),
        "joint_error_f1":    float(jnt_f1),
        "rep_count_mae":     float(cnt_mae),
        "rep_count_obo":     float(1 - cnt_obo),
    }

    metrics_path = os.path.join(output_dir, metrics_filename)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved → {metrics_path}")

    # Phase-4 A6: optional confusion matrix (feeds exercise_sanity.py downstream).
    if confusion_filename is not None and class_names is not None:
        try:
            from sklearn.metrics import confusion_matrix
            import matplotlib
            matplotlib.use("Agg")  # headless (Colab-safe)
            import matplotlib.pyplot as plt

            cm = confusion_matrix(
                val_y_ex, ex_pred, labels=list(range(len(class_names)))
            )
            npy_path = os.path.join(output_dir, confusion_filename + ".npy")
            png_path = os.path.join(output_dir, confusion_filename + ".png")
            np.save(npy_path, cm)

            # Normalised (row-wise) confusion matrix for readability
            with np.errstate(invalid="ignore", divide="ignore"):
                cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm, nan=0.0)
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0, cmap="Blues")
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=90, fontsize=7)
            ax.set_yticklabels(class_names, fontsize=7)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Confusion matrix ({confusion_filename}, row-normalised)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(png_path, dpi=120)
            plt.close(fig)
            print(f"  Confusion matrix saved → {npy_path} + {png_path}")
        except Exception as cm_err:  # don't crash training on viz failure
            print(f"  [warn] confusion matrix save failed: {cm_err}")

    return metrics


# ── Main training loop ────────────────────────────────────────────────────────

def train(args):
    # Fine-tuning mode picks up shorter defaults if the user didn't override
    is_finetune = bool(args.warm_start)
    epochs        = args.epochs   or (FT_EPOCHS   if is_finetune else EPOCHS)
    patience      = args.patience or (FT_PATIENCE if is_finetune else PATIENCE)
    learning_rate = args.learning_rate or (FT_LEARNING_RATE if is_finetune else LEARNING_RATE)

    # ── Step 1: Build dataset (if not already built) ──────────────────────────
    train_arrays_path = os.path.join(args.data_dir, "train", "X_angles.npy")
    if not os.path.exists(train_arrays_path):
        print(f"Dataset not found at {args.data_dir}. Building "
              f"from source={args.source!r}...")
        build_dataset(
            dataset_root=args.dataset_root,
            output_dir=args.data_dir,
            source=args.source,
            mediapipe_root=args.mediapipe_root,
            verbose=True,
        )
    else:
        print(f"Dataset found at {args.data_dir} — skipping rebuild.")

    # ── Step 2: Load data ─────────────────────────────────────────────────────
    print("\nLoading training data...")
    tr = load_split(args.data_dir, "train")
    print(f"  Train: {tr[0].shape[0]} samples")

    print("Loading validation data...")
    vl = load_split(args.data_dir, "val")
    print(f"  Val:   {vl[0].shape[0]} samples")

    train_ds = make_tf_dataset(*tr, batch_size=args.batch_size, shuffle=True)
    val_ds   = make_tf_dataset(*vl, batch_size=args.batch_size, shuffle=False)

    # ── Step 3: Build model ───────────────────────────────────────────────────
    print("\nBuilding MT-TCN model...")
    model = build_mt_tcn()
    compile_mt_tcn(model, learning_rate=learning_rate)
    print(f"  Total parameters: {model.count_params():,}")

    # Warm-start from existing supervised weights
    if args.warm_start:
        if not os.path.isfile(args.warm_start):
            raise FileNotFoundError(
                f"--warm-start weights not found: {args.warm_start}"
            )
        print(f"\nWarm-starting from {args.warm_start}")
        model.load_weights(args.warm_start)
        print(f"  learning_rate : {learning_rate:g}  "
              f"(dropped from {LEARNING_RATE:g})")
        print(f"  epochs        : {epochs}  "
              f"(dropped from {EPOCHS})")
        print(f"  patience      : {patience}  "
              f"(dropped from {PATIENCE})")

    # P3.2: SSL encoder init — load encoder layers from SSL pretraining
    # The SSL model has identical layer names to the supervised MT-TCN encoder
    # (ang_b1_*, ang_b2_*, ..., fusion_proj, fusion_bn, fusion_relu), so
    # loading the SSL weights into the supervised model populates the backbone
    # while leaving all task heads at random init.
    if args.ssl_init:
        if not os.path.isfile(args.ssl_init):
            raise FileNotFoundError(
                f"--ssl-init weights not found: {args.ssl_init}"
            )
        if args.warm_start:
            print("\n[warning] --ssl-init ignored because --warm-start was also specified.")
        else:
            print(f"\nLoading SSL encoder init from {args.ssl_init}")
            # ── Phase A: copy encoder weights layer-by-layer (matching names) ─
            from backend.training.train_ssl_pretrain import build_ssl_model
            ssl_model = build_ssl_model()
            ssl_model.load_weights(args.ssl_init)
            # Encoder layer names are shared between SSL and supervised model
            # (ang_b1_*, ang_b2_*, ..., fusion_proj, fusion_bn, fusion_relu)
            n_loaded = 0
            ssl_layer_map = {l.name: l for l in ssl_model.layers}
            encoder_names = set()
            for layer in model.layers:
                if layer.name in ssl_layer_map and layer.trainable_weights:
                    try:
                        layer.set_weights(ssl_layer_map[layer.name].get_weights())
                        n_loaded += 1
                        encoder_names.add(layer.name)
                    except Exception as e:
                        print(f"  [skip] {layer.name}: {e}")
            print(f"  SSL encoder layers loaded : {n_loaded}")
            print(f"  Task head layers (random) : {len(model.layers) - n_loaded}")
            del ssl_model   # free memory

            # ── Phase B: 2-phase LR strategy ────────────────────────────────
            # Phase 1 (first 20 epochs): freeze encoder, train heads fast
            # Phase 2 (remaining epochs): unfreeze all, fine-tune with lower LR
            # This is a clean Keras-compatible approach (no per-param-group tricks).
            # We store encoder layer names for the phase-2 unfreeze callback.
            args._ssl_encoder_names = encoder_names

            # Freeze encoder backbone for phase 1
            for layer in model.layers:
                if layer.name in encoder_names:
                    layer.trainable = False
            # Recompile so Keras picks up the trainability changes
            compile_mt_tcn(model, learning_rate=learning_rate)
            n_frozen = sum(1 for l in model.layers if not l.trainable and l.name in encoder_names)
            print(f"  Phase 1: encoder frozen ({n_frozen} layers), heads lr={learning_rate:g}")

    # ── Step 4: Callbacks factory ─────────────────────────────────────────────
    os.makedirs(args.model_dir, exist_ok=True)
    weights_path = os.path.join(args.model_dir, "mt_tcn_best.weights.h5")

    def _make_callbacks():
        """Fresh callback set for each model.fit() call."""
        return [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=weights_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(5, patience // 2),
                min_lr=1e-6,
                verbose=1,
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(args.model_dir, "logs"),
                histogram_freq=0,
            ),
            PeriodicClearOutput(every=10, keep_last=10),
        ]

    # ── Step 5: Train ─────────────────────────────────────────────────────────
    # NOTE: Keras 3 does NOT allow model.compile() inside model.fit() callbacks
    # (it sets train_function=None and crashes the next epoch).  The 2-phase SSL
    # strategy is therefore implemented as two separate model.fit() calls with an
    # explicit unfreeze+recompile between them.
    t0 = time.time()
    combined_history: dict = {}

    def _merge(h):
        for k, v in h.history.items():
            combined_history.setdefault(k, []).extend(v)

    if args.ssl_init and not args.warm_start:
        # ── Phase 1: encoder frozen, task heads learn fast (20 epochs) ────────
        SSL_P1 = 20
        actual_p1 = min(SSL_P1, epochs)
        print(f"\nSSL Phase 1 — encoder frozen, task heads only "
              f"({actual_p1} epochs, lr={learning_rate:g})...")
        _merge(model.fit(train_ds, validation_data=val_ds,
                         epochs=actual_p1, callbacks=_make_callbacks(), verbose=2))
        elapsed1 = (time.time() - t0) / 60
        print(f"  Phase 1 done ({elapsed1:.1f} min).  "
              f"Best val_loss so far: {min(combined_history['val_loss']):.4f}")

        remaining = epochs - actual_p1
        if remaining > 0:
            # ── Phase 2: unfreeze all layers, fine-tune with 10× lower LR ────
            for layer in model.layers:
                layer.trainable = True
            fine_lr = max(learning_rate * 0.1, 1e-5)
            compile_mt_tcn(model, learning_rate=fine_lr)
            print(f"\nSSL Phase 2 — all layers unfrozen "
                  f"({remaining} epochs, lr={fine_lr:g})...")
            _merge(model.fit(train_ds, validation_data=val_ds,
                             epochs=remaining, callbacks=_make_callbacks(), verbose=2))
    else:
        print(f"\nTraining for up to {epochs} epochs "
              f"(early stopping patience={patience})...")
        _merge(model.fit(train_ds, validation_data=val_ds,
                         epochs=epochs, callbacks=_make_callbacks(), verbose=2))

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} minutes.")

    # ── Step 6: Save final model ──────────────────────────────────────────────
    final_weights = os.path.join(args.model_dir, "mt_tcn_weights.weights.h5")
    model.save_weights(final_weights)
    print(f"Final weights saved → {final_weights}")

    # Save model config for reconstruction at inference
    # Phase-4 A5: persist `source` so downstream eval scripts
    # (weight_probe.py, offline_eval.py) auto-resolve the matching dataset dir.
    config = {
        "target_frames":  64,
        "n_angular":      22,
        "n_joints_flat":  45,
        "n_exercises":    27,
        "n_joint_groups": 10,
        "keras_version":  tf.keras.__version__,   # pin version to catch load mismatches
        "tf_version":     tf.__version__,
        "source":         args.source,
    }
    with open(os.path.join(args.model_dir, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Copy exercise labels and angle stats to model dir
    import shutil
    shutil.copy(
        os.path.join(args.data_dir, "exercise_labels.json"),
        os.path.join(args.model_dir, "exercise_labels.json"),
    )
    shutil.copy(
        os.path.join(args.data_dir, "angle_stats.npz"),
        os.path.join(args.model_dir, "angle_stats.npz"),
    )

    # ── Step 7: Evaluate on val (and test if available) ──────────────────────
    # Phase-4 A4: pass metrics_filename explicitly to avoid the previous bug
    # where the test-set evaluation silently overwrote training_metrics.json.
    # Phase-4 A6: write a confusion matrix alongside each metrics file.
    with open(os.path.join(args.model_dir, "exercise_labels.json")) as _ef:
        _ex_to_idx = json.load(_ef)
    class_names_by_idx = [name for name, _ in sorted(_ex_to_idx.items(), key=lambda kv: kv[1])]

    print("\nRunning detailed evaluation on validation set...")
    _, _, val_y_ex, val_y_qlt, val_y_jnt, _, _ = vl
    evaluate_model(
        model, val_ds, val_y_ex, val_y_qlt, val_y_jnt, args.model_dir,
        metrics_filename="training_metrics.json",
        confusion_filename="confusion_matrix_val",
        class_names=class_names_by_idx,
    )

    test_arrays_path = os.path.join(args.data_dir, "test", "X_angles.npy")
    if os.path.exists(test_arrays_path):
        print("\nTest split detected — running evaluation on HELD-OUT TEST SET...")
        te = load_split(args.data_dir, "test")
        test_ds = make_tf_dataset(*te, batch_size=args.batch_size, shuffle=False)
        _, _, te_y_ex, te_y_qlt, te_y_jnt, _, _ = te
        test_metrics = evaluate_model(
            model, test_ds, te_y_ex, te_y_qlt, te_y_jnt, args.model_dir,
            metrics_filename="test_metrics.json",
            confusion_filename="confusion_matrix_test",
            class_names=class_names_by_idx,
        )
        print(f"Test metrics saved → {os.path.join(args.model_dir, 'test_metrics.json')}")
        print(f"\n*** DEFENCE-READY TEST ACCURACY: {test_metrics['exercise_accuracy']:.4f} ***")
    else:
        print("\n[info] No test/ split found — skipping test evaluation.")
        print("       Rebuild dataset with default split to get s11 test accuracy.")

    # ── Step 8: Training history ──────────────────────────────────────────────
    history_path = os.path.join(args.model_dir, "training_history.json")
    hist_serialisable = {k: [float(v) for v in vals]
                         for k, vals in combined_history.items()}
    with open(history_path, "w") as f:
        json.dump(hist_serialisable, f, indent=2)
    print(f"Training history saved → {history_path}")

    print(f"\nAll model artifacts saved to: {args.model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FitNova MT-TCN form model")
    parser.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--data_dir",     default=DEFAULT_DATA_DIR)
    parser.add_argument("--model_dir",    default=DEFAULT_MODEL_DIR)
    # R2 flags
    parser.add_argument("--source", choices=["fit3d", "mediapipe", "both"],
                        default="fit3d",
                        help="Input modality used to build the dataset "
                             "(default fit3d = MoCap joints3d_25)")
    parser.add_argument("--mediapipe-root", default=None,
                        help="Where MediaPipe .npy extractions live "
                             "(default backend/data/mediapipe_fit3d/)")
    parser.add_argument("--warm-start", default=None,
                        help="Path to a .weights.h5 checkpoint to fine-tune from. "
                             "Enables shorter fine-tuning defaults (30 epochs, "
                             "patience 10, lr 1e-4) unless overridden below.")
    parser.add_argument("--ssl-init", default=None, dest="ssl_init",
                        help="Path to SSL encoder weights (.weights.h5) from "
                             "train_ssl_pretrain.py. Loads the TCN encoder "
                             "backbone; task heads stay at random init.")
    parser.add_argument("--epochs",        type=int,   default=None)
    parser.add_argument("--patience",      type=int,   default=None)
    parser.add_argument("--learning-rate", type=float, default=None,
                        dest="learning_rate")
    parser.add_argument("--batch-size",    type=int,   default=BATCH_SIZE,
                        dest="batch_size",
                        help="Mini-batch size (default 32; use 64 on L4, 128 on A100)")
    args = parser.parse_args()

    # Normalise paths
    args.data_dir  = os.path.abspath(args.data_dir)
    args.model_dir = os.path.abspath(args.model_dir)
    if args.warm_start:
        args.warm_start = os.path.abspath(args.warm_start)
    if args.ssl_init:
        args.ssl_init = os.path.abspath(args.ssl_init)

    print("=" * 60)
    print("FitNova — MT-TCN Form Detection Training")
    print("=" * 60)
    print(f"  Dataset root : {args.dataset_root}")
    print(f"  Data dir     : {args.data_dir}")
    print(f"  Model dir    : {args.model_dir}")
    print(f"  Source       : {args.source}")
    if args.warm_start:
        print(f"  Warm start   : {args.warm_start}")
    print(f"  TF version   : {tf.__version__}")
    print(f"  GPU available: {bool(tf.config.list_physical_devices('GPU'))}")
    print()

    train(args)
