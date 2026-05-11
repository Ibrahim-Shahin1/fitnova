"""ST-GCN single-head multi-label classifier for FitNova v6.1.

Forks ``st_gcn_v6.py``. The trunk (3 ST-GCN blocks + angular MLP +
exercise embedding + global pool) is unchanged. The output side
collapses v6's five heads (quality / joint_err / boundary / rep_count /
action) into a single ``Dense(num_classes, sigmoid)`` over the QEVD
fine-grained variation taxonomy.

Why the change is correct (plan §5b + §6):
- The QEVD paper's vision backbone is a multi-class classifier on the
  ~1,800-class fine-grained variation taxonomy. v6 compressed those
  labels into a scalar + 10-bit bool — that compression was the cause of
  the +0.004 quality_gap on D9. v6.1 uses the labels as discrete
  supervisors, the way the paper does.
- No quality scalar, no joint_err head, no boundary head, no separate
  action head. The predicted variation NAMES are the form feedback;
  rep counting comes from the geometric state machine; coaching language
  comes from GPT-4o-mini at runtime.

Inputs (same shape contract as v6, expanded `n_exercises` vocab):
    pose         (B, T=64, J=15, C=4)
    angles       (B, T=64, F=22)
    exercise_id  (B,)  int32 in [0, n_exercises)
                       Cleaned-prefix index; see
                       qevd_class_space.QEVDClassSpace.prefix_idx_for.

Outputs:
    logits       (B, num_classes)  pre-sigmoid logits
    probs        (B, num_classes)  sigmoid(logits)
    trunk        (B, trunk_dim)    optional feature output for ablations

Notes:
- Logits and probs are both exposed: training uses logits (with
  `from_logits=True` BCE for numerical stability), runtime uses probs
  for the top-K decoding + exercise-prefix masking.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, ops

from backend.training.models.st_gcn import (
    SKELETON_EDGES_15,
    _stgcn_block,
    build_normalized_adjacency,
)

# Constants — keep the same input shape contract as v6 so the .npz
# pipeline and runtime pose extractor do not need to change.
N_JOINTS_V6_1        = 15
N_POSE_CHANNELS_V6_1 = 4
N_ANGULAR_V6_1       = 22
TARGET_FRAMES_V6_1   = 64


def build_v6_1_model(
    num_classes:      int,
    n_exercises:      int,
    target_frames:    int   = TARGET_FRAMES_V6_1,
    n_joints:         int   = N_JOINTS_V6_1,
    n_pose_channels:  int   = N_POSE_CHANNELS_V6_1,
    n_angular:        int   = N_ANGULAR_V6_1,
    exercise_emb_dim: int   = 16,
    angular_mlp_dim:  int   = 64,
    stgcn_channels:   Tuple[int, int, int] = (64, 128, 256),
    stgcn_dilations:  Tuple[int, int, int] = (1, 2, 4),
    temporal_kernel:  int   = 9,
    head_hidden:      int   = 256,
    head_dropout:     float = 0.3,
) -> keras.Model:
    """Build the v6.1 multi-label ST-GCN model.

    Args:
        num_classes: Size of the variation-taxonomy multi-label head.
            Use ``QEVDClassSpace.num_classes`` from the v6.1 spec.
        n_exercises: Vocabulary size of the exercise-prefix embedding
            input. Use ``QEVDClassSpace.num_prefixes`` (140 in the v6.1
            cleaned space).
        head_hidden: Bottleneck Dense layer width before the multi-label
            output. Wider than v6's quality head (128) because the head
            now has to project 336-D pooled features into 1.4K logits.
        head_dropout: Dropout before the bottleneck. Higher than v6
            (0.2 -> 0.3) to slow down overfit on the long-tail classes.

    Returns:
        keras.Model with inputs {pose, angles, exercise_id} and outputs
        {logits, probs, trunk}.
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be > 0; got {num_classes}")
    if n_exercises <= 0:
        raise ValueError(f"n_exercises must be > 0; got {n_exercises}")

    adjacency = build_normalized_adjacency(n_joints)

    # ── Inputs ────────────────────────────────────────────────────────────
    pose_in = keras.Input(
        shape=(target_frames, n_joints, n_pose_channels), name="pose"
    )
    angles_in = keras.Input(
        shape=(target_frames, n_angular), name="angles"
    )
    exercise_in = keras.Input(shape=(), dtype="int32", name="exercise_id")

    # ── ST-GCN encoder (identical to v6 trunk) ────────────────────────────
    h = pose_in
    for i, (c, d) in enumerate(zip(stgcn_channels, stgcn_dilations)):
        h = _stgcn_block(
            h,
            out_channels=c,
            adjacency=adjacency,
            temporal_kernel=temporal_kernel,
            dilation=d,
            name=f"stgcn{i+1}",
        )
    pose_per_frame = layers.Lambda(
        lambda t: ops.mean(t, axis=2),
        output_shape=(target_frames, stgcn_channels[-1]),
        name="pose_spatial_pool",
    )(h)

    # ── Angular branch ────────────────────────────────────────────────────
    a = layers.Dense(32, activation="relu", name="ang_dense1")(angles_in)
    a = layers.Dense(angular_mlp_dim, activation="relu", name="ang_dense2")(a)

    # ── Exercise embedding (v6.1: 140-prefix vocab, not 25) ───────────────
    ex_emb = layers.Embedding(
        input_dim=n_exercises,
        output_dim=exercise_emb_dim,
        name="exercise_emb",
    )(exercise_in)
    ex_emb_tiled = layers.Lambda(
        lambda t: ops.tile(
            ops.expand_dims(t, axis=1),
            [1, target_frames, 1],
        ),
        output_shape=(target_frames, exercise_emb_dim),
        name="exercise_emb_tile",
    )(ex_emb)

    # ── Per-frame trunk → global pool ────────────────────────────────────
    trunk_per_frame = layers.Concatenate(axis=-1, name="trunk_per_frame")(
        [pose_per_frame, a, ex_emb_tiled]
    )
    trunk_pooled = layers.GlobalAveragePooling1D(name="trunk_pool")(
        trunk_per_frame
    )

    # ── Multi-label head ─────────────────────────────────────────────────
    h = layers.Dropout(head_dropout, name="head_drop")(trunk_pooled)
    h = layers.Dense(head_hidden, activation="relu", name="head_dense")(h)
    logits = layers.Dense(num_classes, activation=None, name="logits")(h)
    probs = layers.Activation("sigmoid", name="probs")(logits)

    trunk = layers.Identity(name="trunk")(trunk_pooled)

    model = keras.Model(
        inputs={
            "pose":        pose_in,
            "angles":      angles_in,
            "exercise_id": exercise_in,
        },
        outputs={
            "logits": logits,
            "probs":  probs,
            "trunk":  trunk,
        },
        name="fitnova_v6_1_stgcn",
    )
    return model


__all__ = [
    "N_JOINTS_V6_1",
    "N_POSE_CHANNELS_V6_1",
    "N_ANGULAR_V6_1",
    "TARGET_FRAMES_V6_1",
    "build_v6_1_model",
]


# ── Quick sanity check (run as module to print summary) ──────────────────────


if __name__ == "__main__":
    from backend.training.preprocessing.qevd_class_space import get_default_class_space
    cs = get_default_class_space()
    m = build_v6_1_model(num_classes=cs.num_classes, n_exercises=cs.num_prefixes)
    m.summary(line_length=120)
    print()

    B, T = 2, TARGET_FRAMES_V6_1
    pose   = np.random.randn(B, T, N_JOINTS_V6_1, N_POSE_CHANNELS_V6_1).astype(np.float32)
    angles = np.random.randn(B, T, N_ANGULAR_V6_1).astype(np.float32)
    ex_id  = np.array([0, 13], dtype=np.int32)
    out = m({"pose": pose, "angles": angles, "exercise_id": ex_id})
    print("Output shapes:")
    for k, v in out.items():
        print(f"  {k:<10s} {tuple(v.shape)}  dtype={v.dtype}")
    print(f"\nTotal params: {m.count_params():,}")
    print(f"num_classes: {cs.num_classes}")
    print(f"num_prefixes: {cs.num_prefixes}")
