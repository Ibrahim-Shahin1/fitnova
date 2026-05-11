"""
ST-GCN encoder + multi-task heads for FitNova v5.

Replaces v4's MT-TCN. The key wins:
  • Spatial inductive bias from the skeleton graph — sample-efficient at our
    ~4 K-rep training scale.
  • Pose-only ST-GCN beat single-view I3D for AQA on FLEX (per the FLEX
    paper's Table 3) — even though we are not using FLEX, that result
    transfers because the underlying argument is "skeletal joints carry
    most of the AQA signal, and ST-GCN respects skeletal structure".

Architecture (matches plan §5):
    Inputs:
      pose         : (B, T=64, J=15, C=4)   ← MediaPipe canonical + visibility
      angles       : (B, T=64, F=22)        ← AIFit angular features
      exercise_id  : (B,)                   ← user-selected exercise (int32)

    Encoder:
      ST-GCN block 1 :  (B,T,J, 4)  → (B,T,J, 64)  dilation 1
      ST-GCN block 2 :  (B,T,J, 64) → (B,T,J,128)  dilation 2
      ST-GCN block 3 :  (B,T,J,128) → (B,T,J,256)  dilation 4

      spatial mean-pool             → (B, T, 256)         per-frame pose features
      angular MLP   (B,T,22) → (B,T,64)                   per-frame angle features
      exercise emb  (B,)    → (B,16)                      tile to (B,T,16)

      per-frame trunk: concat        → (B, T, 256+64+16=336)
      pooled trunk:   GlobalAvgPool1D over T  → (B, 336)

    Heads:
      Q  quality      : trunk_pooled  → Dense(128) → Dense(1, sigmoid)
      A  action       : trunk_pooled  → Dense(64)  → Dense(15, softmax)
      C  rep count    : trunk_pooled  → Dense(64)  → Dense(1, ReLU)
      B  rep boundary : trunk_per_frame → Conv1D(64,3) → Conv1D(1,1, sigmoid)
      J  joint errors : trunk_per_frame → Conv1D(64,3) → Conv1D(5,1, sigmoid)
      F  trunk feats  : trunk_pooled  (exposed for multi-view consistency loss)

The model is built with the Keras 3 functional API and returns a
``keras.Model`` with 6 named outputs.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, ops

# ── Skeleton adjacency for the 15 canonical joints ────────────────────────────
# (canonical_index_i, canonical_index_j) — bidirectional bones
SKELETON_EDGES_15: List[Tuple[int, int]] = [
    (0, 2),    # l_shoulder → l_elbow
    (2, 4),    # l_elbow    → l_wrist
    (1, 3),    # r_shoulder → r_elbow
    (3, 5),    # r_elbow    → r_wrist
    (0, 1),    # l_shoulder → r_shoulder (collar)
    (0, 6),    # l_shoulder → l_hip
    (1, 7),    # r_shoulder → r_hip
    (6, 8),    # l_hip      → l_knee
    (8, 10),   # l_knee     → l_ankle
    (7, 9),    # r_hip      → r_knee
    (9, 11),   # r_knee     → r_ankle
    (6, 7),    # l_hip      → r_hip (pelvis)
    (12, 14),  # pelvis     → spine_mid
    (14, 13),  # spine_mid  → neck
    (12, 6),   # pelvis     → l_hip
    (12, 7),   # pelvis     → r_hip
    (13, 0),   # neck       → l_shoulder
    (13, 1),   # neck       → r_shoulder
]
N_JOINTS_V5 = 15


def build_normalized_adjacency(
    n_joints: int = N_JOINTS_V5,
    edges: List[Tuple[int, int]] = SKELETON_EDGES_15,
) -> np.ndarray:
    """Symmetric-normalised adjacency: D^(-1/2) (A + I) D^(-1/2).

    Includes self-loops. Returns (n_joints, n_joints) float32.
    """
    A = np.zeros((n_joints, n_joints), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A_hat = A + np.eye(n_joints, dtype=np.float32)  # self-loops
    deg = A_hat.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-6)))
    return (d_inv_sqrt @ A_hat @ d_inv_sqrt).astype(np.float32)


# ── ST-GCN block ──────────────────────────────────────────────────────────────


class GraphConv(layers.Layer):
    """Graph-convolution layer over the joint dimension.

    Input  : (B, T, J, C_in)
    Output : (B, T, J, C_out)

    Computes ``A_norm @ X @ W`` where A_norm is the precomputed normalised
    skeleton adjacency. X is reshaped so the joint dimension is the inner
    product axis with A_norm.
    """

    def __init__(self, out_channels: int, adjacency: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = int(out_channels)
        # Persist adjacency as a non-trainable weight so it travels with the
        # model on save/load.
        self._adj_init = adjacency.astype(np.float32)

    def build(self, input_shape):
        # input_shape: (B, T, J, C_in)
        c_in = input_shape[-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(c_in, self.out_channels),
            initializer="he_normal",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.out_channels,),
            initializer="zeros",
            trainable=True,
        )
        self.adj = self.add_weight(
            name="adj",
            shape=self._adj_init.shape,
            initializer=keras.initializers.Constant(self._adj_init),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x):
        # x: (B, T, J, C_in)
        # Linear projection per joint
        x = ops.matmul(x, self.kernel)        # (B, T, J, C_out)
        # Graph aggregation:  for each (B, T): A_norm @ X
        # Use einsum on the joint axis
        x = ops.einsum("ij,btjc->btic", self.adj, x)  # (B, T, J, C_out)
        x = x + self.bias
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"out_channels": self.out_channels})
        return cfg


def _stgcn_block(
    x,
    out_channels: int,
    adjacency: np.ndarray,
    temporal_kernel: int = 9,
    dilation: int = 1,
    name: str = "stgcn",
):
    """One ST-GCN block: GraphConv → BN → ReLU → TemporalConv → BN → ReLU.

    Adds a residual connection (1×1 projection if in/out channels differ).
    """
    # Spatial graph conv
    h = GraphConv(out_channels, adjacency, name=f"{name}_gconv")(x)
    h = layers.BatchNormalization(name=f"{name}_bn1")(h)
    h = layers.Activation("relu", name=f"{name}_relu1")(h)

    # Temporal conv (kernel along T axis only)
    # x shape: (B, T, J, C). Conv2D with kernel (k_t, 1) acts on T per-joint.
    h = layers.Conv2D(
        filters=out_channels,
        kernel_size=(temporal_kernel, 1),
        padding="same",
        dilation_rate=(dilation, 1),
        name=f"{name}_tconv",
    )(h)
    h = layers.BatchNormalization(name=f"{name}_bn2")(h)

    # Residual
    if x.shape[-1] != out_channels:
        res = layers.Conv2D(
            filters=out_channels,
            kernel_size=(1, 1),
            padding="same",
            name=f"{name}_res_proj",
        )(x)
    else:
        res = x
    h = layers.Add(name=f"{name}_add")([h, res])
    h = layers.Activation("relu", name=f"{name}_relu2")(h)
    return h


# ── Top-level model factory ───────────────────────────────────────────────────


def build_v5_model(
    target_frames: int = 64,
    n_joints: int = N_JOINTS_V5,
    n_pose_channels: int = 4,    # x, y, z, visibility
    n_angular: int = 22,
    n_exercises: int = 15,
    n_joint_groups: int = 5,
    exercise_emb_dim: int = 16,
    angular_mlp_dim: int = 64,
    stgcn_channels: Tuple[int, int, int] = (64, 128, 256),
    stgcn_dilations: Tuple[int, int, int] = (1, 2, 4),
    temporal_kernel: int = 9,
    head_dropout: float = 0.2,
) -> keras.Model:
    """Build the v5 multi-task model.

    Returns a ``keras.Model`` with four named inputs:

        ``pose``         (B, T, J, C_pose)
        ``angles``       (B, T, F_angular)
        ``exercise_id``  (B,)                  int32 in [0, n_exercises)
        ``view_pair``    optional helper (not consumed; left for caller)

    And six named outputs:

        ``quality``      (B, 1)
        ``action``       (B, n_exercises)
        ``rep_count``    (B, 1)
        ``boundary``     (B, T, 1)
        ``joint_err``    (B, T, n_joint_groups)
        ``trunk``        (B, trunk_dim)         — exposed for multi-view loss
    """
    adjacency = build_normalized_adjacency(n_joints)

    # ── Inputs ────────────────────────────────────────────────────────────────
    pose_in = keras.Input(
        shape=(target_frames, n_joints, n_pose_channels), name="pose"
    )
    angles_in = keras.Input(shape=(target_frames, n_angular), name="angles")
    exercise_in = keras.Input(shape=(), dtype="int32", name="exercise_id")

    # ── ST-GCN encoder ────────────────────────────────────────────────────────
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
    # h : (B, T, J, stgcn_channels[-1])
    pose_per_frame = layers.Lambda(
        lambda t: ops.mean(t, axis=2),
        output_shape=(target_frames, stgcn_channels[-1]),
        name="pose_spatial_pool",
    )(h)  # (B, T, stgcn_channels[-1])

    # ── Angular branch ────────────────────────────────────────────────────────
    a = layers.Dense(32, activation="relu", name="ang_dense1")(angles_in)
    a = layers.Dense(angular_mlp_dim, activation="relu", name="ang_dense2")(a)
    # a : (B, T, angular_mlp_dim) — keep per-frame for boundary head

    # ── Exercise embedding ────────────────────────────────────────────────────
    ex_emb = layers.Embedding(
        input_dim=n_exercises,
        output_dim=exercise_emb_dim,
        name="exercise_emb",
    )(exercise_in)  # (B, exercise_emb_dim)

    # Tile to per-frame
    ex_emb_tiled = layers.Lambda(
        lambda t: ops.tile(
            ops.expand_dims(t, axis=1),
            [1, target_frames, 1],
        ),
        output_shape=(target_frames, exercise_emb_dim),
        name="exercise_emb_tile",
    )(ex_emb)  # (B, T, exercise_emb_dim)

    # ── Per-frame trunk ──────────────────────────────────────────────────────
    trunk_per_frame = layers.Concatenate(axis=-1, name="trunk_per_frame")(
        [pose_per_frame, a, ex_emb_tiled]
    )  # (B, T, stgcn_channels[-1] + angular_mlp_dim + exercise_emb_dim)

    # ── Pooled trunk ─────────────────────────────────────────────────────────
    trunk_pooled = layers.GlobalAveragePooling1D(name="trunk_pool")(
        trunk_per_frame
    )  # (B, trunk_dim)

    # ── Heads ────────────────────────────────────────────────────────────────
    # Q — quality
    q = layers.Dropout(head_dropout, name="q_drop")(trunk_pooled)
    q = layers.Dense(128, activation="relu", name="q_dense")(q)
    quality = layers.Dense(1, activation="sigmoid", name="quality")(q)

    # A — action (auxiliary mismatch detector)
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
    )(b_h)  # (B, T, 1)

    # J — joint errors (per-frame, multi-label)
    j_h = layers.Conv1D(
        64, 3, padding="same", activation="relu", name="j_conv1"
    )(trunk_per_frame)
    joint_err = layers.Conv1D(
        n_joint_groups, 1, padding="same", activation="sigmoid",
        name="joint_err",
    )(j_h)  # (B, T, n_joint_groups)

    # Expose trunk_pooled for multi-view consistency loss
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
        name="fitnova_v5_stgcn",
    )
    return model


# ── Optional: SSL-pretrain reconstruction decoder ─────────────────────────────


def build_v5_ssl_model(
    target_frames: int = 64,
    n_joints: int = N_JOINTS_V5,
    n_pose_channels: int = 4,
    n_angular: int = 22,
    stgcn_channels: Tuple[int, int, int] = (64, 128, 256),
    stgcn_dilations: Tuple[int, int, int] = (1, 2, 4),
    temporal_kernel: int = 9,
) -> keras.Model:
    """Masked-joint reconstruction model for SSL pretraining.

    Inputs:
      pose_masked  (B, T, J, C)   ← input with masked joints zeroed
      mask         (B, T, J, 1)   ← 1 where the joint was masked
      angles       (B, T, F)      (optional input, included for parity)

    Output:
      pose_recon   (B, T, J, 3)   ← reconstructed xyz (visibility omitted)

    The encoder shares the ST-GCN blocks with the supervised model. After
    pretraining we copy the ST-GCN block weights into the supervised model
    via name matching.
    """
    adjacency = build_normalized_adjacency(n_joints)

    pose_in = keras.Input(
        shape=(target_frames, n_joints, n_pose_channels), name="pose_masked"
    )
    mask_in = keras.Input(
        shape=(target_frames, n_joints, 1), name="mask"
    )

    h = layers.Concatenate(axis=-1, name="ssl_concat_mask")([pose_in, mask_in])
    for i, (c, d) in enumerate(zip(stgcn_channels, stgcn_dilations)):
        h = _stgcn_block(
            h,
            out_channels=c,
            adjacency=adjacency,
            temporal_kernel=temporal_kernel,
            dilation=d,
            name=f"stgcn{i+1}",
        )
    # h : (B, T, J, stgcn_channels[-1])

    # Decoder: 1×1 conv → 3 channels (xyz)
    pose_recon = layers.Conv2D(
        3, kernel_size=(1, 1), padding="same",
        name="ssl_recon",
    )(h)

    return keras.Model(
        inputs={"pose_masked": pose_in, "mask": mask_in},
        outputs={"pose_recon": pose_recon},
        name="fitnova_v5_ssl",
    )


# ── Quick sanity-check ────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Smoke test
    m = build_v5_model()
    m.summary(line_length=120)
    print()
    print("Adjacency matrix:")
    A = build_normalized_adjacency()
    print(f"  shape={A.shape}  rowsum_min={A.sum(1).min():.3f}  rowsum_max={A.sum(1).max():.3f}")

    # Forward pass with random inputs
    B, T, J, C = 2, 64, N_JOINTS_V5, 4
    pose = np.random.randn(B, T, J, C).astype(np.float32)
    angles = np.random.randn(B, T, 22).astype(np.float32)
    ex_id = np.array([0, 7], dtype=np.int32)
    out = m({"pose": pose, "angles": angles, "exercise_id": ex_id})
    print()
    print("Output shapes:")
    for k, v in out.items():
        print(f"  {k:<10s} {tuple(v.shape)}  dtype={v.dtype}")
    print()
    print(f"Total params: {m.count_params():,}")
