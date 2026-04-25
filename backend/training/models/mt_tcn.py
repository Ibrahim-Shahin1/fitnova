"""
Multi-Task Temporal Convolutional Network (MT-TCN)

Architecture:
  Dual-branch dilated causal TCN backbone (angular features + raw joints)
  → Fused representation
  → 4 task heads:
      A: Exercise classification (27 classes)
      B: Rep boundary detection (per-frame binary)
      B2: Rep count regression (scalar)
      C: Form quality score (scalar 0-1)
      D: Per-joint error detection (per-frame, 10 joint groups)

Beats AIFit by:
  - Frame-level per-joint feedback (vs. per-rep stats only)
  - Learned temporal features (vs. hand-crafted operators)
  - Multi-task joint training
  - Dual input representation
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple


N_ANGULAR   = 22
N_JOINTS_FLAT = 45   # 15 joints × 3 coords
N_EXERCISES = 27
N_JOINT_GROUPS = 10
TARGET_FRAMES  = 64


def _tcn_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    dropout_rate: float = 0.2,
    name: str = "",
) -> tf.Tensor:
    """
    One dilated causal TCN block:
        Conv1D (causal, dilated) → BatchNorm → ReLU → Dropout
    + residual connection (1×1 conv if channel dim changes).
    """
    residual = x

    out = layers.Conv1D(
        filters, kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        kernel_initializer="he_normal",
        name=f"{name}_conv",
    )(x)
    out = layers.BatchNormalization(name=f"{name}_bn")(out)
    out = layers.ReLU(name=f"{name}_relu")(out)
    out = layers.Dropout(dropout_rate, name=f"{name}_drop")(out)

    # Residual projection when channel dims differ
    if residual.shape[-1] != filters:
        residual = layers.Conv1D(
            filters, 1,
            padding="same",
            kernel_initializer="he_normal",
            name=f"{name}_res_proj",
        )(residual)

    out = layers.Add(name=f"{name}_add")([out, residual])
    return out


def _tcn_backbone(
    inp: tf.Tensor,
    name_prefix: str,
    dropout_rate: float = 0.2,
) -> tf.Tensor:
    """
    4-level dilated TCN backbone.
    Receptive field: 1 + 2*(k-1)*(d1+d2+d3+d4) = 1+2*2*(1+2+4+8) = 61 frames
    → covers nearly the full 64-frame rep.
    """
    x = _tcn_block(inp,  64,  3, dilation_rate=1, dropout_rate=dropout_rate, name=f"{name_prefix}_b1")
    x = _tcn_block(x,   128,  3, dilation_rate=2, dropout_rate=dropout_rate, name=f"{name_prefix}_b2")
    x = _tcn_block(x,   128,  3, dilation_rate=4, dropout_rate=dropout_rate, name=f"{name_prefix}_b3")
    x = _tcn_block(x,   256,  3, dilation_rate=8, dropout_rate=dropout_rate, name=f"{name_prefix}_b4")
    return x  # (batch, 64, 256)


def build_mt_tcn(
    target_frames: int = TARGET_FRAMES,
    n_angular: int = N_ANGULAR,
    n_joints_flat: int = N_JOINTS_FLAT,
    n_exercises: int = N_EXERCISES,
    n_joint_groups: int = N_JOINT_GROUPS,
    dropout_rate: float = 0.2,
    clf_dropout: float = 0.3,
) -> Model:
    """
    Build and return the MT-TCN Keras model.

    Inputs:
        angular_input:  (batch, 64, 22)
        joints_input:   (batch, 64, 45)

    Outputs (dict):
        exercise:   (batch, 27)       — softmax probabilities
        boundary:   (batch, 64, 1)    — rep boundary sigmoid
        rep_count:  (batch, 1)        — rep count regression
        quality:    (batch, 1)        — form quality sigmoid
        joint_errors:(batch, 64, 10)  — per-joint error sigmoid
    """

    # ── Inputs ────────────────────────────────────────────────────────────────
    angular_input = tf.keras.Input(
        shape=(target_frames, n_angular), name="angular_input"
    )
    joints_input = tf.keras.Input(
        shape=(target_frames, n_joints_flat), name="joints_input"
    )

    # ── Branch 1: Angular features ────────────────────────────────────────────
    branch_ang = _tcn_backbone(angular_input, name_prefix="ang", dropout_rate=dropout_rate)
    # → (batch, 64, 256)

    # ── Branch 2: Raw joint coordinates ──────────────────────────────────────
    branch_jnt = _tcn_backbone(joints_input, name_prefix="jnt", dropout_rate=dropout_rate)
    # → (batch, 64, 256)

    # ── Fusion ────────────────────────────────────────────────────────────────
    fused = layers.Concatenate(axis=-1, name="fusion_concat")([branch_ang, branch_jnt])
    # → (batch, 64, 512)

    backbone = layers.Conv1D(
        256, 1, padding="same",
        kernel_initializer="he_normal",
        name="fusion_proj",
    )(fused)
    backbone = layers.BatchNormalization(name="fusion_bn")(backbone)
    backbone = layers.ReLU(name="fusion_relu")(backbone)
    # → (batch, 64, 256)

    # ── Head A: Exercise classification ──────────────────────────────────────
    gap = layers.GlobalAveragePooling1D(name="head_cls_gap")(backbone)
    cls = layers.Dense(128, activation="relu",
                       kernel_initializer="glorot_uniform",
                       name="head_cls_d1")(gap)
    cls = layers.Dropout(clf_dropout, name="head_cls_drop")(cls)
    exercise_out = layers.Dense(
        n_exercises, activation="softmax",
        kernel_initializer="glorot_uniform",
        name="exercise",
    )(cls)

    # ── Head B: Rep boundary detection (per-frame) ────────────────────────────
    bnd = layers.Conv1D(
        64, 3, padding="causal",
        activation="relu",
        kernel_initializer="he_normal",
        name="head_bnd_conv",
    )(backbone)
    boundary_out = layers.Conv1D(
        1, 1, padding="same",
        activation="sigmoid",
        kernel_initializer="glorot_uniform",
        name="boundary",
    )(bnd)
    # → (batch, 64, 1)

    # ── Head B2: Rep count regression ─────────────────────────────────────────
    cnt = layers.GlobalAveragePooling1D(name="head_cnt_gap")(backbone)
    cnt = layers.Dense(64, activation="relu",
                       kernel_initializer="glorot_uniform",
                       name="head_cnt_d1")(cnt)
    rep_count_out = layers.Dense(
        1, activation="relu",
        kernel_initializer="glorot_uniform",
        name="rep_count",
    )(cnt)
    # → (batch, 1)

    # ── Head C: Form quality score ────────────────────────────────────────────
    qap = layers.GlobalAveragePooling1D(name="head_qual_gap")(backbone)
    qlt = layers.Dense(128, activation="relu",
                       kernel_initializer="glorot_uniform",
                       name="head_qual_d1")(qap)
    qlt = layers.Dropout(clf_dropout, name="head_qual_drop")(qlt)
    quality_out = layers.Dense(
        1, activation="sigmoid",
        kernel_initializer="glorot_uniform",
        name="quality",
    )(qlt)
    # → (batch, 1)

    # ── Head D: Per-joint error detection (per-frame) ─────────────────────────
    jnt_h = layers.Conv1D(
        128, 3, padding="causal",
        activation="relu",
        kernel_initializer="he_normal",
        name="head_jnt_conv",
    )(backbone)
    joint_errors_out = layers.Conv1D(
        n_joint_groups, 1, padding="same",
        activation="sigmoid",
        kernel_initializer="glorot_uniform",
        name="joint_errors",
    )(jnt_h)
    # → (batch, 64, 10)

    model = Model(
        inputs={"angular_input": angular_input, "joints_input": joints_input},
        outputs={
            "exercise":     exercise_out,
            "boundary":     boundary_out,
            "rep_count":    rep_count_out,
            "quality":      quality_out,
            "joint_errors": joint_errors_out,
        },
        name="MT_TCN",
    )
    return model


def compile_mt_tcn(
    model: Model,
    learning_rate: float = 1e-3,
    w_cls:      float = 1.0,
    w_boundary: float = 0.5,
    w_count:    float = 0.3,
    w_quality:  float = 2.0,
    w_joint:    float = 2.0,
) -> None:
    """
    Compile the MT-TCN with multi-task losses and metrics.

    Loss weights:
      quality and joint errors are weighted 2× — primary objectives.
      boundary and count are auxiliary regularisers.
    """
    # Label smoothing (0.05) softens the 27-class one-hot targets — prevents
    # the classifier from becoming over-confident on training subjects when
    # generalising to held-out subjects with slightly different motion styles.
    # Implemented via CategoricalCrossentropy since SparseCCE does not support
    # label_smoothing; labels will be one-hot encoded at training time.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "exercise":     tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            "boundary":     tf.keras.losses.BinaryCrossentropy(),
            "rep_count":    tf.keras.losses.MeanSquaredError(),
            "quality":      tf.keras.losses.MeanSquaredError(),
            "joint_errors": tf.keras.losses.BinaryCrossentropy(),
        },
        loss_weights={
            "exercise":     w_cls,
            "boundary":     w_boundary,
            "rep_count":    w_count,
            "quality":      w_quality,
            "joint_errors": w_joint,
        },
        metrics={
            "exercise":     [tf.keras.metrics.CategoricalAccuracy(name="acc")],
            "quality":      [tf.keras.metrics.MeanAbsoluteError(name="mae")],
            "joint_errors": [tf.keras.metrics.BinaryAccuracy(name="bin_acc")],
        },
    )


def model_summary(model: Model) -> None:
    model.summary(line_length=100)
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")


if __name__ == "__main__":
    m = build_mt_tcn()
    compile_mt_tcn(m)
    model_summary(m)
