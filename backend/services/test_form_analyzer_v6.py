"""Unit tests for v6 integration in backend.services.form_analyzer.

Specifically verifies:
  1. _load_model picks v6 when v6_supervised.weights.h5 is present
  2. _load_v6_model loads model_config.json + qevd_exercise_map.json correctly
  3. predict_window dispatches to _predict_window_v6 when version == "v6"
  4. Input shape validation: bad shapes return neutral defaults, don't crash
  5. Output dict has the expected keys + native 10-channel joint_errors

Run from repo root:
    pytest backend/services/test_form_analyzer_v6.py -v

NOTE: These tests build a minimal v6 model with random weights, save it to a
tmp_path, and exercise the full FormAnalyzer loading path. This catches:
  - file path / config-loading bugs
  - shape contract drift between trainer and inference
  - dispatcher routing errors
  - debug logging side-effects when FITNOVA_DEBUG=1

The tests do NOT verify training-time numerical correctness (that's covered
by backend/training/models/test_st_gcn_v6.py + the eval gates D7/D8/D9).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Skip the entire file if TF is unavailable (matches form_analyzer's guard)
tf = pytest.importorskip("tensorflow")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: build a minimal v6 model + write the 4 expected files into tmp_path
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def v6_model_dir(tmp_path):
    """Build a tiny v6 model with random weights and save it as if it were the
    Colab-trained artifact. Returns the directory path."""
    from backend.training.models.st_gcn_v6 import build_v6_model

    model = build_v6_model(
        target_frames=64,
        n_joints=15,
        n_pose_channels=4,
        n_angular=22,
        n_exercises=25,
        n_joint_groups=10,
    )
    weights_path = tmp_path / "v6_supervised.weights.h5"
    model.save_weights(str(weights_path))

    # model_config.json — what train_form_model_v6.py writes
    config = {
        "target_frames":      64,
        "n_canonical_joints": 15,
        "n_pose_channels":    4,
        "n_angular":          22,
        "n_exercises":        25,
        "n_joint_groups":     10,
    }
    (tmp_path / "model_config.json").write_text(json.dumps(config))

    # qevd_exercise_map.json — top-24 + __other__
    ex_map = {f"exercise_{i:02d}": i for i in range(24)}
    ex_map["__other__"] = 24
    # Add a real-looking entry the v6 model would actually have
    ex_map["squats"] = 3
    (tmp_path / "qevd_exercise_map.json").write_text(json.dumps(ex_map))

    return tmp_path


# ─────────────────────────────────────────────────────────────────────────────
# 1. Loading + version detection
# ─────────────────────────────────────────────────────────────────────────────


def test_load_model_detects_v6_when_v6_weights_present(v6_model_dir):
    from backend.services.form_analyzer import FormAnalyzer
    analyzer = FormAnalyzer(model_dir=str(v6_model_dir))
    assert analyzer._model_version == "v6"
    assert analyzer.model_ready is True


def test_load_v6_loads_exercise_map(v6_model_dir):
    from backend.services.form_analyzer import FormAnalyzer
    analyzer = FormAnalyzer(model_dir=str(v6_model_dir))
    # Forward + reverse maps must both be populated
    assert "__other__" in analyzer._exercise_labels
    assert analyzer._exercise_labels["__other__"] == 24
    assert "squats" in analyzer._exercise_labels
    assert analyzer._idx_to_exercise[24] == "__other__"
    assert analyzer._idx_to_exercise[3] == "squats"


def test_load_v6_no_normalizer(v6_model_dir):
    """v6 uses raw angles (same as v5.2). _normalizer must be None."""
    from backend.services.form_analyzer import FormAnalyzer
    analyzer = FormAnalyzer(model_dir=str(v6_model_dir))
    assert analyzer._normalizer is None


def test_load_v6_correct_param_count(v6_model_dir):
    """v6 should be ~1.08M params; large drift means architecture mismatch."""
    from backend.services.form_analyzer import FormAnalyzer
    analyzer = FormAnalyzer(model_dir=str(v6_model_dir))
    n = analyzer._model.count_params()
    assert 1_000_000 < n < 1_200_000, f"unexpected v6 param count: {n}"


def test_load_fails_clearly_when_config_missing(tmp_path):
    """Smoke: drop the weights file but no model_config.json."""
    from backend.training.models.st_gcn_v6 import build_v6_model
    model = build_v6_model()
    model.save_weights(str(tmp_path / "v6_supervised.weights.h5"))
    # Don't write model_config.json or exercise_map

    from backend.services.form_analyzer import FormAnalyzer
    with pytest.raises(FileNotFoundError, match="model_config"):
        FormAnalyzer(model_dir=str(tmp_path))


def test_load_fails_clearly_when_exercise_map_missing(tmp_path):
    from backend.training.models.st_gcn_v6 import build_v6_model
    model = build_v6_model()
    model.save_weights(str(tmp_path / "v6_supervised.weights.h5"))
    config = {"target_frames": 64, "n_canonical_joints": 15,
              "n_pose_channels": 4, "n_angular": 22,
              "n_exercises": 25, "n_joint_groups": 10}
    (tmp_path / "model_config.json").write_text(json.dumps(config))
    # Don't write qevd_exercise_map.json

    from backend.services.form_analyzer import FormAnalyzer
    with pytest.raises(FileNotFoundError, match="qevd_exercise_map"):
        FormAnalyzer(model_dir=str(tmp_path))


# ─────────────────────────────────────────────────────────────────────────────
# 2. predict_window dispatcher routes to v6
# ─────────────────────────────────────────────────────────────────────────────


def test_predict_window_dispatches_to_v6(v6_model_dir):
    from backend.services.form_analyzer import FormAnalyzer
    analyzer = FormAnalyzer(model_dir=str(v6_model_dir))

    pose   = np.random.randn(64, 15, 4).astype(np.float32)
    angles = np.random.randn(64, 22).astype(np.float32)
    out = analyzer.predict_window(
        angles_window=angles,
        joints_window=np.zeros((64, 45)),  # ignored by v6
        pose_window=pose,
        exercise_idx=3,  # squats
    )
    assert out["model_version"] == "v6"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Output contract: keys + shapes + native 10-channel joint_errors
# ─────────────────────────────────────────────────────────────────────────────


def test_v6_output_keys(v6_model_dir):
    from backend.services.form_analyzer import FormAnalyzer
    analyzer = FormAnalyzer(model_dir=str(v6_model_dir))
    pose   = np.random.randn(64, 15, 4).astype(np.float32)
    angles = np.random.randn(64, 22).astype(np.float32)
    out = analyzer.predict_window(angles, np.zeros((64, 45)), pose, exercise_idx=3)

    expected_keys = {
        "exercise", "exercise_confidence", "boundary", "rep_count",
        "quality", "joint_errors", "model_version",
    }
    assert set(out.keys()) >= expected_keys


def test_v6_joint_errors_native_10_channel(v6_model_dir):
    """v6 joint_err must be (64, 10) — no remap from 5 → 10 like v5.2."""
    from backend.services.form_analyzer import FormAnalyzer
    analyzer = FormAnalyzer(model_dir=str(v6_model_dir))
    pose   = np.random.randn(64, 15, 4).astype(np.float32)
    angles = np.random.randn(64, 22).astype(np.float32)
    out = analyzer.predict_window(angles, np.zeros((64, 45)), pose, exercise_idx=3)

    je = np.array(out["joint_errors"])
    assert je.shape == (64, 10), f"expected (64,10), got {je.shape}"
    # v6 outputs sigmoid → [0, 1]
    assert je.min() >= 0.0 and je.max() <= 1.0


def test_v6_quality_in_range(v6_model_dir):
    from backend.services.form_analyzer import FormAnalyzer
    analyzer = FormAnalyzer(model_dir=str(v6_model_dir))
    pose   = np.random.randn(64, 15, 4).astype(np.float32)
    angles = np.random.randn(64, 22).astype(np.float32)
    out = analyzer.predict_window(angles, np.zeros((64, 45)), pose, exercise_idx=3)

    assert 0.0 <= out["quality"] <= 1.0


def test_v6_action_index_in_25_class_range(v6_model_dir):
    from backend.services.form_analyzer import FormAnalyzer
    analyzer = FormAnalyzer(model_dir=str(v6_model_dir))
    pose   = np.random.randn(64, 15, 4).astype(np.float32)
    angles = np.random.randn(64, 22).astype(np.float32)
    out = analyzer.predict_window(angles, np.zeros((64, 45)), pose, exercise_idx=3)

    # exercise is the resolved name string, exercise_confidence is the prob
    assert isinstance(out["exercise"], str)
    assert 0.0 <= out["exercise_confidence"] <= 1.0
    # Must resolve to one of the 26 names (24 named + __other__ + squats overlap)
    assert out["exercise"] in {f"exercise_{i:02d}" for i in range(24)} | {"__other__", "squats"}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Input validation: bad shapes return neutral, don't crash
# ─────────────────────────────────────────────────────────────────────────────


def test_v6_predict_with_no_pose_returns_neutral(v6_model_dir):
    from backend.services.form_analyzer import FormAnalyzer
    analyzer = FormAnalyzer(model_dir=str(v6_model_dir))
    angles = np.random.randn(64, 22).astype(np.float32)
    out = analyzer.predict_window(
        angles_window=angles,
        joints_window=np.zeros((64, 45)),
        pose_window=None,            # missing
        exercise_idx=3,
    )
    assert out["model_version"] == "v6"
    assert out["exercise"] == "unknown"
    assert out["quality"] == 0.5
    assert len(out["joint_errors"]) == 64
    assert len(out["joint_errors"][0]) == 10


def test_v6_predict_with_wrong_pose_shape_returns_neutral(v6_model_dir):
    """Bad pose shape (e.g. 3D coords without visibility) should not crash."""
    from backend.services.form_analyzer import FormAnalyzer
    analyzer = FormAnalyzer(model_dir=str(v6_model_dir))
    pose   = np.random.randn(64, 15, 3).astype(np.float32)  # missing visibility channel
    angles = np.random.randn(64, 22).astype(np.float32)
    out = analyzer.predict_window(angles, np.zeros((64, 45)), pose, exercise_idx=3)
    assert out["model_version"] == "v6"
    assert out["exercise"] == "unknown"


def test_v6_predict_with_wrong_angles_shape_returns_neutral(v6_model_dir):
    from backend.services.form_analyzer import FormAnalyzer
    analyzer = FormAnalyzer(model_dir=str(v6_model_dir))
    pose   = np.random.randn(64, 15, 4).astype(np.float32)
    angles = np.random.randn(64, 33).astype(np.float32)  # wrong feature count
    out = analyzer.predict_window(angles, np.zeros((64, 45)), pose, exercise_idx=3)
    assert out["model_version"] == "v6"
    assert out["exercise"] == "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Determinism: same input → same output across two calls
# ─────────────────────────────────────────────────────────────────────────────


def test_v6_inference_is_deterministic(v6_model_dir):
    from backend.services.form_analyzer import FormAnalyzer
    analyzer = FormAnalyzer(model_dir=str(v6_model_dir))
    pose   = np.random.RandomState(0).randn(64, 15, 4).astype(np.float32)
    angles = np.random.RandomState(1).randn(64, 22).astype(np.float32)

    out1 = analyzer.predict_window(angles, np.zeros((64, 45)), pose, exercise_idx=3)
    out2 = analyzer.predict_window(angles, np.zeros((64, 45)), pose, exercise_idx=3)
    assert out1["quality"] == pytest.approx(out2["quality"], abs=1e-6)
    assert out1["rep_count"] == pytest.approx(out2["rep_count"], abs=1e-6)
    np.testing.assert_array_almost_equal(
        np.array(out1["joint_errors"]),
        np.array(out2["joint_errors"]),
        decimal=5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Different exercise_idx changes the model's output (conditioning works)
# ─────────────────────────────────────────────────────────────────────────────


def test_v6_exercise_idx_affects_output(v6_model_dir):
    """Verify that the exercise_id input actually conditions the network.
    If the embedding layer were dead, the same pose+angles would produce
    identical outputs for any exercise_idx."""
    from backend.services.form_analyzer import FormAnalyzer
    analyzer = FormAnalyzer(model_dir=str(v6_model_dir))
    pose   = np.random.RandomState(0).randn(64, 15, 4).astype(np.float32)
    angles = np.random.RandomState(1).randn(64, 22).astype(np.float32)

    out_squats = analyzer.predict_window(angles, np.zeros((64, 45)), pose, exercise_idx=3)
    out_other  = analyzer.predict_window(angles, np.zeros((64, 45)), pose, exercise_idx=24)
    # At least ONE of (quality, rep_count, joint_errors) must differ
    differs = (
        out_squats["quality"] != out_other["quality"]
        or out_squats["rep_count"] != out_other["rep_count"]
        or not np.allclose(np.array(out_squats["joint_errors"]),
                            np.array(out_other["joint_errors"]), atol=1e-4)
    )
    assert differs, "exercise_idx had no effect on output — embedding may be dead"
