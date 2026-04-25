"""Guard against drift between exercises.json (SSOT) and the label file
written by train_form_model.py."""
import json
import os
import pytest

from backend.config.exercises import all_exercises, EXERCISE_TO_IDX, IDX_TO_EXERCISE

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LABELS_PATH = os.path.join(ROOT, "backend", "models", "form_model", "exercise_labels.json")


def test_exercises_count():
    assert len(all_exercises()) == 27


def test_indices_are_contiguous_0_to_26():
    idxs = sorted(e["idx"] for e in all_exercises().values())
    assert idxs == list(range(27))


def test_required_fields_present():
    required = {"idx", "display_name", "camera_view", "distance_m",
                "phone_height_cm", "orientation", "key_errors_detected",
                "guidelines"}
    for name, meta in all_exercises().items():
        missing = required - set(meta.keys())
        assert not missing, f"{name} missing fields: {missing}"


def test_camera_view_values():
    allowed = {"side", "front", "either"}
    for name, meta in all_exercises().items():
        assert meta["camera_view"] in allowed, name


def test_orientation_values():
    allowed = {"landscape", "portrait"}
    for name, meta in all_exercises().items():
        assert meta["orientation"] in allowed, name


def test_matches_trained_label_file():
    if not os.path.exists(LABELS_PATH):
        pytest.skip(f"Trained label file not present at {LABELS_PATH}")
    with open(LABELS_PATH) as f:
        trained = json.load(f)
    assert set(trained.keys()) == set(EXERCISE_TO_IDX.keys()), (
        "Class names drift between exercises.json and exercise_labels.json"
    )
    for name, idx in trained.items():
        assert EXERCISE_TO_IDX[name] == idx, (
            f"Index mismatch for {name}: trained={idx} ssot={EXERCISE_TO_IDX[name]}"
        )
