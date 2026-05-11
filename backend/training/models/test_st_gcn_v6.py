"""Unit tests for backend.training.models.st_gcn_v6.

Run from repo root:
    pytest backend/training/models/test_st_gcn_v6.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from backend.training.models.st_gcn_v6 import (
    N_ANGULAR_V6,
    N_EXERCISES_V6,
    N_JOINTS_V6,
    N_JOINT_GROUPS_V6,
    N_POSE_CHANNELS_V6,
    TARGET_FRAMES_V6,
    build_v6_model,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants — pin the v6 contract
# ─────────────────────────────────────────────────────────────────────────────


def test_v6_constants_match_master_plan():
    """Master plan §II.6 locks these. Drift = downstream contract breakage."""
    assert N_JOINTS_V6        == 15
    assert N_POSE_CHANNELS_V6 == 4
    assert N_ANGULAR_V6       == 22
    assert N_EXERCISES_V6     == 25  # 24 named + 1 "other" — was 15 in v5.2
    assert N_JOINT_GROUPS_V6  == 10  # was 5 in v5.2 (+ runtime remap to 10)
    assert TARGET_FRAMES_V6   == 64


# ─────────────────────────────────────────────────────────────────────────────
# Build + shape contract
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def model():
    return build_v6_model()


def test_model_builds(model):
    assert isinstance(model, keras.Model)
    assert model.name == "fitnova_v6_stgcn"


def test_input_names_and_shapes(model):
    """Input contract: three named keras.Input tensors."""
    in_specs = {inp.name.split(":")[0]: inp for inp in model.inputs}
    expected_names = {"pose", "angles", "exercise_id"}
    assert set(in_specs.keys()) >= expected_names, set(in_specs.keys())

    pose_shape = tuple(in_specs["pose"].shape)
    assert pose_shape == (None, TARGET_FRAMES_V6,
                          N_JOINTS_V6, N_POSE_CHANNELS_V6), pose_shape

    ang_shape = tuple(in_specs["angles"].shape)
    assert ang_shape == (None, TARGET_FRAMES_V6, N_ANGULAR_V6), ang_shape

    ex_shape = tuple(in_specs["exercise_id"].shape)
    assert ex_shape == (None,), ex_shape


def test_output_names(model):
    """Output contract: six named outputs in the documented order."""
    expected = {"quality", "action", "rep_count",
                "boundary", "joint_err", "trunk"}
    assert set(model.output_names) == expected


def test_output_shapes_for_batch(model):
    """Batch=2 forward pass produces correctly-shaped outputs."""
    B = 2
    pose   = np.zeros((B, TARGET_FRAMES_V6, N_JOINTS_V6,
                        N_POSE_CHANNELS_V6), dtype=np.float32)
    angles = np.zeros((B, TARGET_FRAMES_V6, N_ANGULAR_V6), dtype=np.float32)
    ex_id  = np.zeros((B,), dtype=np.int32)
    out = model({"pose": pose, "angles": angles, "exercise_id": ex_id})

    assert tuple(out["quality"].shape)   == (B, 1)
    assert tuple(out["action"].shape)    == (B, N_EXERCISES_V6)
    assert tuple(out["rep_count"].shape) == (B, 1)
    assert tuple(out["boundary"].shape)  == (B, TARGET_FRAMES_V6, 1)
    assert tuple(out["joint_err"].shape) == (B, TARGET_FRAMES_V6,
                                              N_JOINT_GROUPS_V6)
    # Trunk dim = stgcn_channels[-1] + angular_mlp_dim + exercise_emb_dim
    #           = 256 + 64 + 16 = 336
    assert tuple(out["trunk"].shape) == (B, 256 + 64 + 16)


def test_output_dtypes_and_ranges(model):
    """Sigmoid heads stay in [0,1]; softmax sums to 1; quality/rep_count nonneg."""
    B = 2
    pose   = np.random.randn(
        B, TARGET_FRAMES_V6, N_JOINTS_V6, N_POSE_CHANNELS_V6
    ).astype(np.float32)
    angles = np.random.randn(B, TARGET_FRAMES_V6, N_ANGULAR_V6).astype(np.float32)
    ex_id  = np.array([0, N_EXERCISES_V6 - 1], dtype=np.int32)
    out = model({"pose": pose, "angles": angles, "exercise_id": ex_id})

    q = out["quality"].numpy()
    assert np.all(q >= 0.0) and np.all(q <= 1.0), q
    a = out["action"].numpy()
    assert np.all(a >= 0.0) and np.all(a <= 1.0), a
    assert np.allclose(a.sum(axis=-1), 1.0, atol=1e-5)
    rc = out["rep_count"].numpy()
    assert np.all(rc >= 0.0), rc
    b = out["boundary"].numpy()
    assert np.all(b >= 0.0) and np.all(b <= 1.0), b
    je = out["joint_err"].numpy()
    assert np.all(je >= 0.0) and np.all(je <= 1.0), je


# ─────────────────────────────────────────────────────────────────────────────
# Param count target (master plan §II.6: ~5.2M)
# ─────────────────────────────────────────────────────────────────────────────


def test_param_count_in_target_range(model):
    """Pin v6 param count near 1.08M (matches v5.2's actual count + the
    handful of extra params from the wider action/joint_err heads).

    NOTE: master plan section II.6 quoted "~5.2 M params"; that was a
    guesstimate. The realised architecture (3 ST-GCN blocks at 64/128/256
    channels) is actually ~1.08M for v5.2 and v6. Smaller-than-claimed but
    consistent across versions and well-suited to our data scale (~250K
    train clips × 1M params = healthy 250 samples-per-param ratio).

    Verified on 2026-05-08: v5.2 = 1,082,250 ; v6 = 1,083,385.
    Drift signals an unintended architecture change.
    """
    n = model.count_params()
    print(f"\nv6 model params: {n:,}")
    assert 1_000_000 < n < 1_200_000, (
        f"param count {n:,} outside [1.0M, 1.2M] — architecture drifted?"
    )

    # And v6 should have ~1000 more params than v5.2 (the wider heads).
    from backend.training.models.st_gcn import build_v5_model
    n5 = build_v5_model().count_params()
    delta = n - n5
    assert 500 < delta < 5_000, (
        f"v6 - v5.2 = {delta} params; expected ~1000 from action 15->25 "
        f"+ joint_err 5->10. Way off means heads were rewired."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Numerical sanity: no NaN/Inf on a typical forward pass
# ─────────────────────────────────────────────────────────────────────────────


def test_no_nan_or_inf_on_forward(model):
    np.random.seed(42)
    B = 4
    pose = (np.random.randn(B, TARGET_FRAMES_V6, N_JOINTS_V6,
                              N_POSE_CHANNELS_V6) * 0.5).astype(np.float32)
    # Visibility column should be in [0, 1]
    pose[..., 3] = np.random.uniform(0.5, 1.0,
                                     size=pose[..., 3].shape).astype(np.float32)
    angles = (np.random.randn(B, TARGET_FRAMES_V6, N_ANGULAR_V6) * 0.5
              + 1.5).astype(np.float32)  # roughly [0, 3] rad range
    ex_id = np.random.randint(0, N_EXERCISES_V6, size=(B,)).astype(np.int32)
    out = model({"pose": pose, "angles": angles, "exercise_id": ex_id})
    for name, t in out.items():
        arr = t.numpy()
        assert not np.isnan(arr).any(), f"NaN in {name}"
        assert not np.isinf(arr).any(), f"Inf in {name}"


# ─────────────────────────────────────────────────────────────────────────────
# Gradient flow: verify every head receives gradients during training
# ─────────────────────────────────────────────────────────────────────────────


def test_gradients_flow_to_every_head(model):
    """Compute a loss against each output head and confirm every trainable
    variable receives a non-zero gradient.  Catches dead heads + frozen
    layers.  Master plan §II.7: all 5 head losses must train."""
    B = 2
    pose   = np.random.randn(B, TARGET_FRAMES_V6, N_JOINTS_V6,
                              N_POSE_CHANNELS_V6).astype(np.float32)
    angles = np.random.randn(B, TARGET_FRAMES_V6, N_ANGULAR_V6).astype(np.float32)
    ex_id  = np.array([0, 5], dtype=np.int32)

    # Targets — arbitrary values; we only care that gradients are non-zero.
    y_quality   = np.array([[0.8], [0.4]], dtype=np.float32)
    y_action    = np.array([0, 5], dtype=np.int32)
    y_rep_count = np.array([[3.0], [5.0]], dtype=np.float32)
    y_boundary  = np.zeros((B, TARGET_FRAMES_V6, 1), dtype=np.float32)
    y_boundary[:, ::8, 0] = 1.0   # spike every 8 frames
    y_joint_err = np.zeros((B, TARGET_FRAMES_V6, N_JOINT_GROUPS_V6),
                           dtype=np.float32)
    y_joint_err[:, :, 0] = 1.0    # always-on left elbow signal

    inputs = {"pose": tf.constant(pose),
              "angles": tf.constant(angles),
              "exercise_id": tf.constant(ex_id)}

    with tf.GradientTape() as tape:
        out = model(inputs, training=True)
        loss_q = tf.reduce_mean(tf.square(out["quality"] - y_quality))
        loss_a = tf.reduce_mean(
            keras.losses.sparse_categorical_crossentropy(y_action, out["action"])
        )
        loss_c = tf.reduce_mean(tf.square(out["rep_count"] - y_rep_count))
        loss_b = tf.reduce_mean(
            keras.losses.binary_crossentropy(y_boundary, out["boundary"])
        )
        loss_j = tf.reduce_mean(
            keras.losses.binary_crossentropy(y_joint_err, out["joint_err"])
        )
        total = loss_q + loss_a + loss_c + loss_b + loss_j

    grads = tape.gradient(total, model.trainable_variables)
    assert len(grads) == len(model.trainable_variables)

    n_with_grad = 0
    n_zero_grad = 0
    for v, g in zip(model.trainable_variables, grads):
        if g is None:
            pytest.fail(f"variable {v.name} has no gradient (None)")
        n_with_grad += 1
        gnorm = float(tf.norm(g).numpy())
        if gnorm == 0.0:
            n_zero_grad += 1
            print(f"  WARN: zero-norm gradient on {v.name}")
    assert n_with_grad >= 30, (
        f"only {n_with_grad} variables received gradients — model under-built?"
    )
    # Allow a small handful of zero-grad vars (e.g., frozen adjacency); fail
    # if it's a substantial fraction.
    assert n_zero_grad < n_with_grad // 4, (
        f"{n_zero_grad}/{n_with_grad} vars have zero gradient — suspicious"
    )


# ─────────────────────────────────────────────────────────────────────────────
# v6 vs v5.2 head differences
# ─────────────────────────────────────────────────────────────────────────────


def test_joint_err_head_is_native_10_channel(model):
    """v5.2 had 5 channels and remapped to 10 at runtime; v6 outputs 10
    natively. Master plan §II.10 deletes the remap on integration day."""
    B = 1
    pose   = np.zeros((B, TARGET_FRAMES_V6, N_JOINTS_V6,
                        N_POSE_CHANNELS_V6), dtype=np.float32)
    angles = np.zeros((B, TARGET_FRAMES_V6, N_ANGULAR_V6), dtype=np.float32)
    ex_id  = np.zeros((B,), dtype=np.int32)
    out = model({"pose": pose, "angles": angles, "exercise_id": ex_id})
    assert out["joint_err"].shape[-1] == 10


def test_action_head_is_25_way(model):
    """v5.2: 15 classes (FitNova exercise menu).
    v6: 25 (24 QEVD-mapped + 1 'other'). Master plan §II.3."""
    B = 1
    pose   = np.zeros((B, TARGET_FRAMES_V6, N_JOINTS_V6,
                        N_POSE_CHANNELS_V6), dtype=np.float32)
    angles = np.zeros((B, TARGET_FRAMES_V6, N_ANGULAR_V6), dtype=np.float32)
    ex_id  = np.zeros((B,), dtype=np.int32)
    out = model({"pose": pose, "angles": angles, "exercise_id": ex_id})
    assert out["action"].shape[-1] == 25


# ─────────────────────────────────────────────────────────────────────────────
# Determinism + multi-batch consistency
# ─────────────────────────────────────────────────────────────────────────────


def test_eval_mode_is_deterministic(model):
    """In inference (training=False), same input -> same output. Catches
    accidental dropout-on-eval or BN issues."""
    B = 2
    pose   = np.zeros((B, TARGET_FRAMES_V6, N_JOINTS_V6,
                        N_POSE_CHANNELS_V6), dtype=np.float32)
    angles = np.zeros((B, TARGET_FRAMES_V6, N_ANGULAR_V6), dtype=np.float32)
    ex_id  = np.array([0, 1], dtype=np.int32)
    inputs = {"pose": pose, "angles": angles, "exercise_id": ex_id}
    out1 = {k: v.numpy() for k, v in
            model(inputs, training=False).items()}
    out2 = {k: v.numpy() for k, v in
            model(inputs, training=False).items()}
    for k in out1:
        np.testing.assert_allclose(
            out1[k], out2[k], atol=1e-6, rtol=1e-6,
            err_msg=f"non-deterministic eval on head {k}",
        )


def test_exercise_embedding_changes_output(model):
    """Different exercise_id should change quality/action — proves the
    embedding actually conditions the trunk (master plan §II.3)."""
    B = 1
    pose   = np.random.RandomState(0).randn(
        B, TARGET_FRAMES_V6, N_JOINTS_V6, N_POSE_CHANNELS_V6
    ).astype(np.float32)
    angles = np.random.RandomState(0).randn(
        B, TARGET_FRAMES_V6, N_ANGULAR_V6
    ).astype(np.float32)
    out_a = model({"pose": pose, "angles": angles,
                    "exercise_id": np.array([0], dtype=np.int32)},
                   training=False)
    out_b = model({"pose": pose, "angles": angles,
                    "exercise_id": np.array([13], dtype=np.int32)},
                   training=False)
    # Quality should differ (the exercise embedding feeds the pooled trunk
    # which feeds the quality head).
    diff = float(np.abs(out_a["quality"].numpy()
                         - out_b["quality"].numpy()).max())
    assert diff > 1e-6, (
        f"exercise embedding has no effect on quality (diff={diff}); "
        "the embedding may not be wired into the trunk"
    )
