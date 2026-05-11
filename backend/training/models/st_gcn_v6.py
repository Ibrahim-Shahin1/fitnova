"""ST-GCN encoder + multi-task heads for FitNova v6 (QEVD).

Forks v5.2's ``st_gcn.py``. Architecture is identical (3 ST-GCN blocks
over a 15-joint skeleton + angular MLP + exercise embedding); only the
two output-head dimensions change to match v6's QEVD-derived label
contract.

v6 changes vs v5.2 (master plan section II.3 / II.6):

    head             v5.2 (Fit3D)   v6 (QEVD)
    -----------------------------------------
    joint_err         5 channels     10 channels   (drops the v5->v4 remap
                                                     in form_analyzer.py:525-540)
    action            15 classes     25 classes    (24 named QEVD exercises
                                                     overlapping FitNova's
                                                     exercise menu + 1 "other")

The shared ST-GCN primitives (``GraphConv``, ``_stgcn_block``, the
skeleton adjacency, etc.) are imported from ``st_gcn`` so the trunk
architecture stays in one place.

Inputs (unchanged from v5.2):
    pose         (B, T=64, J=15, C=4)   MediaPipe canonical + visibility
    angles       (B, T=64, F=22)        AIFit angular features
    exercise_id  (B,)                   user-selected exercise (int32 in [0, 25))

Outputs:
    quality      (B, 1)                  primary regression head
    action       (B, 25)                 softmax over 25-way QEVD subset
    rep_count    (B, 1)                  auxiliary
    boundary     (B, T, 1)               per-frame rep-boundary
    joint_err    (B, T, 10)              per-frame multi-label
    trunk        (B, trunk_dim)          exposed for multi-view loss
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, ops

from backend.training.models.st_gcn import (
    GraphConv,
    SKELETON_EDGES_15,
    _stgcn_block,
    build_normalized_adjacency,
)

# Constants — explicit so downstream code can import them without instantiating the model.
N_JOINTS_V6        = 15        # canonical 15-joint skeleton (unchanged from v5)
N_POSE_CHANNELS_V6 = 4         # x, y, z, visibility
N_ANGULAR_V6       = 22        # AIFit angular features
N_EXERCISES_V6     = 25        # 24 QEVD-mapped + 1 "other"
N_JOINT_GROUPS_V6  = 10        # matches form_session.JOINT_GROUP_NAMES exactly
TARGET_FRAMES_V6   = 64        # window length


def build_v6_model(
    target_frames:    int   = TARGET_FRAMES_V6,
    n_joints:         int   = N_JOINTS_V6,
    n_pose_channels:  int   = N_POSE_CHANNELS_V6,
    n_angular:        int   = N_ANGULAR_V6,
    n_exercises:      int   = N_EXERCISES_V6,
    n_joint_groups:   int   = N_JOINT_GROUPS_V6,
    exercise_emb_dim: int   = 16,
    angular_mlp_dim:  int   = 64,
    stgcn_channels:   Tuple[int, int, int] = (64, 128, 256),
    stgcn_dilations:  Tuple[int, int, int] = (1, 2, 4),
    temporal_kernel:  int   = 9,
    head_dropout:     float = 0.2,
) -> keras.Model:
    """Build the v6 multi-task ST-GCN model.

    Returns a ``keras.Model`` with three named inputs (``pose``, ``angles``,
    ``exercise_id``) and six named outputs (``quality``, ``action``,
    ``rep_count``, ``boundary``, ``joint_err``, ``trunk``).

    Default ``n_exercises=25`` and ``n_joint_groups=10`` are the v6
    contract. Pass other values for ablations / unit tests.
    """
    adjacency = build_normalized_adjacency(n_joints)

    # ── Inputs ────────────────────────────────────────────────────────────
    pose_in = keras.Input(
        shape=(target_frames, n_joints, n_pose_channels), name="pose"
    )
    angles_in = keras.Input(
        shape=(target_frames, n_angular), name="angles"
    )
    exercise_in = keras.Input(shape=(), dtype="int32", name="exercise_id")

    # ── ST-GCN encoder (identical to v5.2 trunk) ──────────────────────────
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

    # ── Angular branch ───────────────────────────────────────────────────
    a = layers.Dense(32, activation="relu", name="ang_dense1")(angles_in)
    a = layers.Dense(angular_mlp_dim, activation="relu", name="ang_dense2")(a)

    # ── Exercise embedding ───────────────────────────────────────────────
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

    # ── Per-frame trunk ──────────────────────────────────────────────────
    trunk_per_frame = layers.Concatenate(axis=-1, name="trunk_per_frame")(
        [pose_per_frame, a, ex_emb_tiled]
    )

    trunk_pooled = layers.GlobalAveragePooling1D(name="trunk_pool")(
        trunk_per_frame
    )

    # ── Heads ────────────────────────────────────────────────────────────
    # Q — quality (primary regression)
    q = layers.Dropout(head_dropout, name="q_drop")(trunk_pooled)
    q = layers.Dense(128, activation="relu", name="q_dense")(q)
    quality = layers.Dense(1, activation="sigmoid", name="quality")(q)

    # A — action (v6: 25-way softmax instead of 15)
    a_h = layers.Dropout(head_dropout, name="a_drop")(trunk_pooled)
    a_h = layers.Dense(64, activation="relu", name="a_dense")(a_h)
    action = layers.Dense(
        n_exercises, activation="softmax", name="action"
    )(a_h)

    # C — rep count (auxiliary)
    c_h = layers.Dense(64, activation="relu", name="c_dense")(trunk_pooled)
    rep_count = layers.Dense(1, activation="relu", name="rep_count")(c_h)

    # B — rep boundary (per-frame)
    b_h = layers.Conv1D(
        64, 3, padding="same", activation="relu", name="b_conv1"
    )(trunk_per_frame)
    boundary = layers.Conv1D(
        1, 1, padding="same", activation="sigmoid", name="boundary"
    )(b_h)

    # J — joint errors (v6: 10-channel native — no v5->v4 remap downstream)
    j_h = layers.Conv1D(
        64, 3, padding="same", activation="relu", name="j_conv1"
    )(trunk_per_frame)
    joint_err = layers.Conv1D(
        n_joint_groups, 1, padding="same", activation="sigmoid",
        name="joint_err",
    )(j_h)

    trunk = layers.Identity(name="trunk")(trunk_pooled)

    model = keras.Model(
        inputs={
            "pose":        pose_in,
            "angles":      angles_in,
            "exercise_id": exercise_in,
        },
        outputs={
            "quality":   quality,
            "action":    action,
            "rep_count": rep_count,
            "boundary":  boundary,
            "joint_err": joint_err,
            "trunk":     trunk,
        },
        name="fitnova_v6_stgcn",
    )
    return model


__all__ = [
    "N_JOINTS_V6",
    "N_POSE_CHANNELS_V6",
    "N_ANGULAR_V6",
    "N_EXERCISES_V6",
    "N_JOINT_GROUPS_V6",
    "TARGET_FRAMES_V6",
    "build_v6_model",
]


# ── Quick sanity check (run as module to print summary) ──────────────────────


if __name__ == "__main__":
    m = build_v6_model()
    m.summary(line_length=120)
    print()

    B, T = 2, TARGET_FRAMES_V6
    pose   = np.random.randn(B, T, N_JOINTS_V6, N_POSE_CHANNELS_V6).astype(np.float32)
    angles = np.random.randn(B, T, N_ANGULAR_V6).astype(np.float32)
    ex_id  = np.array([0, 13], dtype=np.int32)
    out = m({"pose": pose, "angles": angles, "exercise_id": ex_id})
    print("Output shapes:")
    for k, v in out.items():
        print(f"  {k:<10s} {tuple(v.shape)}  dtype={v.dtype}")
    print(f"\nTotal params: {m.count_params():,}")
