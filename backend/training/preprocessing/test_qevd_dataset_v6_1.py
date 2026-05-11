"""Tests for the v6.1 dataset pipeline pure functions."""
from __future__ import annotations

import json

import numpy as np
import pytest

from backend.training.preprocessing.qevd_class_space import QEVDClassSpace
from backend.training.preprocessing.qevd_dataset_v6_1 import (
    build_v6_1_sample,
    derive_v6_1_target,
)


@pytest.fixture
def cs(tmp_path) -> QEVDClassSpace:
    spec = {
        "version": "test-0.0.1",
        "source_file": "tests/fixture",
        "cleanup_rules": {},
        "stats": {},
        "classes": [
            {"idx": 0, "prefix": "pushups", "variation": "narrow",
             "n_clips_train": 400, "n_clips_test": 20, "n_clips_total": 420},
            {"idx": 1, "prefix": "pushups", "variation": "no obvious issue",
             "n_clips_train": 700, "n_clips_test": 40, "n_clips_total": 740},
            {"idx": 2, "prefix": "squats", "variation": "back not straight",
             "n_clips_train": 600, "n_clips_test": 30, "n_clips_total": 630},
            {"idx": 3, "prefix": "squats", "variation": "no obvious issue",
             "n_clips_train": 1500, "n_clips_test": 80, "n_clips_total": 1580},
            {"idx": 4, "prefix": "squats", "variation": "shallow",
             "n_clips_train": 800, "n_clips_test": 50, "n_clips_total": 850},
        ],
        "prefix_to_class_indices": {"pushups": [0, 1], "squats": [2, 3, 4]},
    }
    p = tmp_path / "tiny.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(spec, f)
    return QEVDClassSpace.load(p)


@pytest.fixture
def fake_clip_npz():
    """Mimic the keys load_clip_npz produces from a real .npz file."""
    return {
        "pose_canon": np.random.randn(50, 15, 4).astype(np.float32),
        "angles_raw": np.random.randn(50, 22).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# derive_v6_1_target — happy path
# ---------------------------------------------------------------------------


def test_derive_basic_single_label(cs):
    target, prefix_idx = derive_v6_1_target(
        labels_list=["squats - shallow"],
        exercise_prefix="squats",
        class_space=cs,
    )
    assert target is not None
    assert target.shape == (5,)
    assert target[4] == 1.0
    assert target.sum() == 1.0
    # Alphabetical: pushups=0, squats=1
    assert prefix_idx == 1


def test_derive_multi_label_same_prefix(cs):
    target, prefix_idx = derive_v6_1_target(
        labels_list=["squats - shallow", "squats - back not straight"],
        exercise_prefix="squats",
        class_space=cs,
    )
    assert target[2] == 1.0
    assert target[4] == 1.0
    assert target.sum() == 2.0
    assert prefix_idx == 1


def test_derive_falls_back_to_first_label_prefix(cs):
    # Caller passes no exercise_prefix; should infer from the first label.
    target, prefix_idx = derive_v6_1_target(
        labels_list=["pushups - narrow"],
        exercise_prefix=None,
        class_space=cs,
    )
    assert target is not None
    assert prefix_idx == 0


# ---------------------------------------------------------------------------
# derive_v6_1_target — filter cases
# ---------------------------------------------------------------------------


def test_derive_all_labels_filtered_returns_none(cs):
    # 'squats - hold' is not in the tiny class space; sole label drops out.
    target, prefix_idx = derive_v6_1_target(
        labels_list=["squats - hold"],
        exercise_prefix="squats",
        class_space=cs,
    )
    assert target is None
    assert prefix_idx is None


def test_derive_empty_labels_returns_none(cs):
    target, prefix_idx = derive_v6_1_target(
        labels_list=[],
        exercise_prefix="squats",
        class_space=cs,
    )
    assert target is None
    assert prefix_idx is None


def test_derive_unknown_prefix_returns_none(cs):
    # The label DOES resolve (somehow — fixture lets us simulate this),
    # but the prefix is not in the cleaned space. In practice the class
    # space cleanup is consistent so this case would not happen; the
    # function still must defend against it.
    target, prefix_idx = derive_v6_1_target(
        labels_list=["squats - shallow"],     # resolves to idx 4
        exercise_prefix="this_is_not_a_prefix",
        class_space=cs,
    )
    # target is non-zero, but prefix_idx is None → entire result is None.
    assert target is None
    assert prefix_idx is None


def test_derive_partial_filter_kept(cs):
    # Some labels survive, others don't — keep the clip with the survivors.
    target, prefix_idx = derive_v6_1_target(
        labels_list=["squats - shallow", "squats - this_does_not_exist"],
        exercise_prefix="squats",
        class_space=cs,
    )
    assert target is not None
    assert target.sum() == 1.0
    assert target[4] == 1.0


# ---------------------------------------------------------------------------
# build_v6_1_sample
# ---------------------------------------------------------------------------


def test_sample_input_shapes(cs, fake_clip_npz):
    target, prefix_idx = derive_v6_1_target(
        ["squats - shallow"], "squats", cs,
    )
    inputs, tgt = build_v6_1_sample(fake_clip_npz, target, prefix_idx, target_frames=64)
    assert inputs["pose"].shape == (64, 15, 4)
    assert inputs["angles"].shape == (64, 22)
    assert inputs["exercise_id"].shape == ()
    assert inputs["exercise_id"] == 1
    assert tgt.shape == (5,)


def test_sample_target_is_float32(cs, fake_clip_npz):
    target, prefix_idx = derive_v6_1_target(
        ["pushups - narrow"], "pushups", cs,
    )
    _, tgt = build_v6_1_sample(fake_clip_npz, target, prefix_idx)
    assert tgt.dtype == np.float32


def test_sample_exercise_id_is_int32(cs, fake_clip_npz):
    target, prefix_idx = derive_v6_1_target(
        ["pushups - narrow"], "pushups", cs,
    )
    inputs, _ = build_v6_1_sample(fake_clip_npz, target, prefix_idx)
    assert inputs["exercise_id"].dtype == np.int32


def test_sample_no_quality_or_joint_err(cs, fake_clip_npz):
    # The v6.1 target is a plain ndarray, NOT a dict. If a refactor
    # changes it to a dict (reintroducing multi-task heads), this test
    # fails.
    target, prefix_idx = derive_v6_1_target(
        ["squats - shallow"], "squats", cs,
    )
    _, tgt = build_v6_1_sample(fake_clip_npz, target, prefix_idx)
    assert isinstance(tgt, np.ndarray)
