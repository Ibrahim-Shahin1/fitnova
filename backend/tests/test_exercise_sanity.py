import json
import os
import tempfile
import numpy as np
import pytest

from backend.services.exercise_sanity import ExerciseMismatchDetector


@pytest.fixture
def tmp_model_dir():
    with tempfile.TemporaryDirectory() as d:
        labels = {"squat": 0, "deadlift": 1, "pushup": 2, "curl": 3, "row": 4}
        with open(os.path.join(d, "exercise_labels.json"), "w") as f:
            json.dump(labels, f)
        # 5x5 confusion matrix: squat confused with deadlift, pushup, row (top-3).
        # curl is NOT in squat's top-3 confusions (0 confusion with curl).
        cm = np.array([
            [80, 15, 3, 0, 2],   # squat (neighbours: deadlift, pushup, row)
            [10, 85, 3, 1, 1],   # deadlift
            [ 2,  3, 90, 2, 3],  # pushup
            [ 1,  1,  2, 95, 1], # curl
            [ 1,  1,  3, 1, 94], # row
        ], dtype=np.float32)
        np.save(os.path.join(d, "confusion_matrix.npy"), cm)
        # Reset class state between tests (module-level cache)
        ExerciseMismatchDetector._class_index = None
        ExerciseMismatchDetector._neighbours  = None
        ExerciseMismatchDetector._mode        = "uninitialised"
        yield d


@pytest.fixture
def tmp_model_dir_no_cm():
    with tempfile.TemporaryDirectory() as d:
        labels = {"squat": 0, "deadlift": 1, "pushup": 2, "curl": 3}
        with open(os.path.join(d, "exercise_labels.json"), "w") as f:
            json.dump(labels, f)
        ExerciseMismatchDetector._class_index = None
        ExerciseMismatchDetector._neighbours  = None
        ExerciseMismatchDetector._mode        = "uninitialised"
        yield d


def test_smart_mode_known_confusion_muted(tmp_model_dir):
    det = ExerciseMismatchDetector("squat", tmp_model_dir)
    assert det.mode == "smart"
    # Model keeps predicting deadlift (squat's top-1 neighbour) — must NOT fire
    for _ in range(10):
        warning = det.update("deadlift", 0.85)
    assert warning is None


def test_smart_mode_unrelated_class_fires(tmp_model_dir):
    det = ExerciseMismatchDetector("squat", tmp_model_dir)
    # curl is not in squat's top-3 neighbours → should fire
    warning = None
    for _ in range(8):
        warning = det.update("curl", 0.85)
    assert warning is not None
    assert warning["predicted"] == "curl"
    assert warning["selected"]  == "squat"


def test_one_shot_only(tmp_model_dir):
    det = ExerciseMismatchDetector("squat", tmp_model_dir)
    first = None
    for _ in range(8):
        first = det.update("curl", 0.85)
    assert first is not None
    # Further updates must return None
    for _ in range(20):
        assert det.update("curl", 0.90) is None


def test_confidence_floor(tmp_model_dir):
    det = ExerciseMismatchDetector("squat", tmp_model_dir)
    # All confidences below 0.70 floor — must not fire
    for _ in range(20):
        assert det.update("pushup", 0.60) is None


def test_degraded_mode_uses_higher_floor(tmp_model_dir_no_cm):
    det = ExerciseMismatchDetector("squat", tmp_model_dir_no_cm)
    assert det.mode == "degraded"
    # Under 0.80 must not fire
    for _ in range(20):
        assert det.update("pushup", 0.75) is None
    # At/above 0.80 must fire
    det2 = ExerciseMismatchDetector("squat", tmp_model_dir_no_cm)
    warning = None
    for _ in range(8):
        warning = det2.update("pushup", 0.85)
    assert warning is not None


def test_no_selected_exercise_never_fires(tmp_model_dir):
    det = ExerciseMismatchDetector(None, tmp_model_dir)
    for _ in range(20):
        assert det.update("pushup", 0.95) is None


# ─────────────────────────────────────────────────────────────────────────────
# v6 (QEVD) compatibility — loads from qevd_exercise_map.json
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_v6_model_dir():
    """Directory with v6's qevd_exercise_map.json (no v5.2 exercise_labels.json)."""
    with tempfile.TemporaryDirectory() as d:
        ex_map = {
            "__other__": 24, "squats": 3, "pushups": 4, "yoga pushup": 2,
            "elbow plank": 0, "high knees": 8,
        }
        with open(os.path.join(d, "qevd_exercise_map.json"), "w") as f:
            json.dump(ex_map, f)
        # Reset singleton state
        ExerciseMismatchDetector._class_index = None
        ExerciseMismatchDetector._neighbours  = None
        ExerciseMismatchDetector._mode        = "uninitialised"
        yield d


def test_v6_loads_labels_from_qevd_map(tmp_v6_model_dir):
    """v6 model dir has qevd_exercise_map.json instead of exercise_labels.json.
    The detector must detect this and not log 'detector disabled'."""
    det = ExerciseMismatchDetector("squats", tmp_v6_model_dir)
    assert det.mode != "disabled"
    # Class index must be populated with QEVD names
    assert ExerciseMismatchDetector._class_index is not None
    assert "squats" in ExerciseMismatchDetector._class_index
    assert "__other__" in ExerciseMismatchDetector._class_index


def test_v6_disabled_when_no_label_file(tmp_path):
    """If neither qevd_exercise_map.json nor exercise_labels.json exist,
    detector falls back to disabled mode (graceful degradation)."""
    ExerciseMismatchDetector._class_index = None
    ExerciseMismatchDetector._neighbours  = None
    ExerciseMismatchDetector._mode        = "uninitialised"
    det = ExerciseMismatchDetector("squat", str(tmp_path))
    assert det.mode == "disabled"
    # update() must early-return None even with mismatching predictions
    for _ in range(20):
        assert det.update("pushup", 0.95) is None


def test_v6_prefers_qevd_map_over_exercise_labels(tmp_path):
    """If both files exist (unlikely but possible mid-migration), prefer v6."""
    qevd_map = {"squats": 3, "__other__": 24}
    v52_map  = {"squat": 0, "pushup": 1}
    with open(os.path.join(str(tmp_path), "qevd_exercise_map.json"), "w") as f:
        json.dump(qevd_map, f)
    with open(os.path.join(str(tmp_path), "exercise_labels.json"), "w") as f:
        json.dump(v52_map, f)
    ExerciseMismatchDetector._class_index = None
    ExerciseMismatchDetector._neighbours  = None
    ExerciseMismatchDetector._mode        = "uninitialised"
    det = ExerciseMismatchDetector("squats", str(tmp_path))
    # Must have loaded the QEVD names, not v5.2 names
    assert "squats" in ExerciseMismatchDetector._class_index
    assert "squat" not in ExerciseMismatchDetector._class_index   # v5.2 ignored
