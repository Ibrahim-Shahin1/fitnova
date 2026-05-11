"""Tests for the v6.1 single-head multi-label ST-GCN."""
from __future__ import annotations

import numpy as np
import pytest

from backend.training.models.st_gcn_v6_1 import (
    N_ANGULAR_V6_1,
    N_JOINTS_V6_1,
    N_POSE_CHANNELS_V6_1,
    TARGET_FRAMES_V6_1,
    build_v6_1_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model():
    """Small model for fast tests (num_classes=20, n_exercises=5)."""
    return build_v6_1_model(num_classes=20, n_exercises=5)


def _dummy_inputs(batch_size: int = 2, n_exercises: int = 5):
    return {
        "pose": np.random.randn(
            batch_size, TARGET_FRAMES_V6_1, N_JOINTS_V6_1, N_POSE_CHANNELS_V6_1
        ).astype(np.float32),
        "angles": np.random.randn(
            batch_size, TARGET_FRAMES_V6_1, N_ANGULAR_V6_1
        ).astype(np.float32),
        "exercise_id": np.arange(batch_size, dtype=np.int32) % n_exercises,
    }


# ---------------------------------------------------------------------------
# Build + shape contract
# ---------------------------------------------------------------------------


def test_build_returns_keras_model(tiny_model):
    import tensorflow as tf
    assert isinstance(tiny_model, tf.keras.Model)


def test_build_invalid_num_classes_raises():
    with pytest.raises(ValueError, match="num_classes"):
        build_v6_1_model(num_classes=0, n_exercises=5)


def test_build_invalid_n_exercises_raises():
    with pytest.raises(ValueError, match="n_exercises"):
        build_v6_1_model(num_classes=10, n_exercises=0)


def test_output_keys(tiny_model):
    out = tiny_model(_dummy_inputs())
    assert set(out.keys()) == {"logits", "probs", "trunk"}


def test_logits_shape(tiny_model):
    out = tiny_model(_dummy_inputs(batch_size=3))
    assert tuple(out["logits"].shape) == (3, 20)


def test_probs_shape(tiny_model):
    out = tiny_model(_dummy_inputs(batch_size=3))
    assert tuple(out["probs"].shape) == (3, 20)


def test_trunk_shape(tiny_model):
    # Trunk dim = pose_per_frame(256) + angular(64) + exercise_emb(16) = 336
    out = tiny_model(_dummy_inputs())
    assert tuple(out["trunk"].shape) == (2, 336)


def test_probs_in_unit_range(tiny_model):
    out = tiny_model(_dummy_inputs())
    p = out["probs"].numpy()
    assert (p >= 0.0).all() and (p <= 1.0).all()


def test_probs_equal_sigmoid_of_logits(tiny_model):
    out = tiny_model(_dummy_inputs())
    logits = out["logits"].numpy()
    probs = out["probs"].numpy()
    expected = 1.0 / (1.0 + np.exp(-logits))
    np.testing.assert_allclose(probs, expected, atol=1e-5)


def test_no_multi_task_outputs(tiny_model):
    # The v6.1 model MUST NOT expose quality / joint_err / boundary /
    # rep_count / action outputs. If a refactor reintroduces them, this
    # test fails loudly.
    out = tiny_model(_dummy_inputs())
    forbidden = {"quality", "joint_err", "boundary", "rep_count", "action"}
    assert forbidden.isdisjoint(out.keys())


# ---------------------------------------------------------------------------
# Behaviour contract
# ---------------------------------------------------------------------------


def test_inference_deterministic_in_eval_mode(tiny_model):
    inputs = _dummy_inputs()
    out_a = tiny_model(inputs, training=False)
    out_b = tiny_model(inputs, training=False)
    np.testing.assert_allclose(out_a["logits"], out_b["logits"], atol=1e-6)


def test_exercise_id_affects_output(tiny_model):
    inputs = _dummy_inputs(batch_size=1)
    inputs_a = {**inputs, "exercise_id": np.array([0], dtype=np.int32)}
    inputs_b = {**inputs, "exercise_id": np.array([3], dtype=np.int32)}
    out_a = tiny_model(inputs_a, training=False)
    out_b = tiny_model(inputs_b, training=False)
    diff = float(np.abs(out_a["logits"].numpy() - out_b["logits"].numpy()).max())
    assert diff > 1e-3, "Exercise embedding does not influence logits"


def test_batch_invariance(tiny_model):
    # Per-sample logits should not depend on batch context.
    inputs = _dummy_inputs(batch_size=2)
    out_pair = tiny_model(inputs, training=False)

    single = {
        "pose":        inputs["pose"][:1],
        "angles":      inputs["angles"][:1],
        "exercise_id": inputs["exercise_id"][:1],
    }
    out_single = tiny_model(single, training=False)
    np.testing.assert_allclose(
        out_pair["logits"].numpy()[:1],
        out_single["logits"].numpy(),
        atol=1e-5,
    )


def test_param_count_in_expected_range():
    # Ballpark for the production-size model: ~1.3-1.5M params at the
    # real class space size. Catches accidental head-width changes.
    from backend.training.preprocessing.qevd_class_space import (
        get_default_class_space,
    )
    cs = get_default_class_space()
    m = build_v6_1_model(num_classes=cs.num_classes, n_exercises=cs.num_prefixes)
    n = m.count_params()
    assert 1_000_000 <= n <= 2_000_000, f"unexpected v6.1 param count: {n:,}"
