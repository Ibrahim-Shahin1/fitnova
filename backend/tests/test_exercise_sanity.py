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
