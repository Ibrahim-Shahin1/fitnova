"""Tests for the v6.1 multi-label classification target builder."""
from __future__ import annotations

import json

import numpy as np
import pytest

from backend.training.preprocessing.qevd_class_space import QEVDClassSpace
from backend.training.preprocessing.qevd_label_builder_v2 import (
    _parse_clip_id,
    build_v6_1_clip_record,
    build_v6_1_target,
    stack_targets,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cs(tmp_path) -> QEVDClassSpace:
    """A small class space mirroring real QEVD squat / pushup variants."""
    spec = {
        "version": "test-0.0.1",
        "source_file": "tests/fixture",
        "cleanup_rules": {},
        "stats": {},
        "classes": [
            {"idx": 0, "prefix": "squats", "variation": "no obvious issue",
             "n_clips_train": 1500, "n_clips_test": 80, "n_clips_total": 1580},
            {"idx": 1, "prefix": "squats", "variation": "shallow",
             "n_clips_train": 800, "n_clips_test": 50, "n_clips_total": 850},
            {"idx": 2, "prefix": "squats", "variation": "back not straight",
             "n_clips_train": 600, "n_clips_test": 30, "n_clips_total": 630},
            {"idx": 3, "prefix": "pushups", "variation": "narrow",
             "n_clips_train": 400, "n_clips_test": 20, "n_clips_total": 420},
            {"idx": 4, "prefix": "pushups", "variation": "no obvious issue",
             "n_clips_train": 700, "n_clips_test": 40, "n_clips_total": 740},
        ],
        "prefix_to_class_indices": {"squats": [0, 1, 2], "pushups": [3, 4]},
    }
    p = tmp_path / "tiny.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(spec, f)
    return QEVDClassSpace.load(p)


# ---------------------------------------------------------------------------
# build_v6_1_target — pure function
# ---------------------------------------------------------------------------


def test_target_single_label(cs):
    target = build_v6_1_target(["squats - shallow"], cs)
    assert target.shape == (5,)
    assert target.dtype == np.float32
    assert target[1] == 1.0
    assert target.sum() == 1.0


def test_target_multi_label(cs):
    target = build_v6_1_target(
        ["squats - shallow", "squats - back not straight"], cs,
    )
    assert target[1] == 1.0
    assert target[2] == 1.0
    assert target.sum() == 2.0


def test_target_skips_filtered_labels(cs):
    # 'squats - hold' is not in the tiny class space; should be skipped
    target = build_v6_1_target(["squats - shallow", "squats - hold"], cs)
    assert target.sum() == 1.0
    assert target[1] == 1.0


def test_target_empty_labels(cs):
    target = build_v6_1_target([], cs)
    assert target.shape == (5,)
    assert target.sum() == 0.0


def test_target_no_quality_scalar_emitted(cs):
    # The function MUST NOT return any quality field — only the target.
    target = build_v6_1_target(["squats - shallow"], cs)
    assert isinstance(target, np.ndarray)
    # If someone refactors to return a dict, this fails loudly.


# ---------------------------------------------------------------------------
# build_v6_1_clip_record — full per-clip record
# ---------------------------------------------------------------------------


def test_clip_record_basic(cs):
    record = {
        "video_path": "./00012345.mp4",
        "labels": ["squats - shallow"],
        "split": "train",
    }
    out = build_v6_1_clip_record(record, cs)
    assert out["clip_id"] == 12345
    assert out["video_path"] == "./00012345.mp4"
    assert out["split"] == "train"
    assert out["labels"] == ["squats - shallow"]
    assert out["label_indices"] == [1]
    assert out["target"][1] == 1.0


def test_clip_record_separates_kept_and_filtered(cs):
    record = {
        "video_path": "./00000001.mp4",
        "labels": ["squats - shallow", "squats - hold", "squats - back not straight"],
        "split": "train",
    }
    out = build_v6_1_clip_record(record, cs)
    assert "squats - shallow" in out["labels"]
    assert "squats - back not straight" in out["labels"]
    assert "squats - hold" in out["filtered_labels"]
    assert out["target"].sum() == 2.0


def test_clip_record_keep_filtered_false_omits_field(cs):
    record = {
        "video_path": "./00000001.mp4",
        "labels": ["squats - shallow", "squats - hold"],
    }
    out = build_v6_1_clip_record(record, cs, keep_filtered_labels=False)
    assert "filtered_labels" not in out


def test_clip_record_missing_video_path_yields_minus_one(cs):
    out = build_v6_1_clip_record({"labels": ["squats - shallow"]}, cs)
    assert out["clip_id"] == -1


def test_clip_record_default_split_is_train(cs):
    out = build_v6_1_clip_record({"video_path": "./00000001.mp4", "labels": []}, cs)
    assert out["split"] == "train"


def test_clip_record_no_labels(cs):
    record = {"video_path": "./00000999.mp4", "labels": [], "split": "test"}
    out = build_v6_1_clip_record(record, cs)
    assert out["labels"] == []
    assert out["label_indices"] == []
    assert out["target"].sum() == 0.0


def test_clip_record_target_is_independent_per_call(cs):
    # Mutating one returned target must not affect another.
    a = build_v6_1_clip_record({"video_path": "./00000001.mp4", "labels": ["squats - shallow"]}, cs)
    b = build_v6_1_clip_record({"video_path": "./00000002.mp4", "labels": ["squats - shallow"]}, cs)
    a["target"][2] = 1.0
    assert b["target"][2] == 0.0


# ---------------------------------------------------------------------------
# stack_targets — batch helper
# ---------------------------------------------------------------------------


def test_stack_targets_basic(cs):
    records = [
        {"labels": ["squats - shallow"]},
        {"labels": ["squats - back not straight"]},
        {"labels": ["pushups - narrow"]},
    ]
    stacked = stack_targets(records, cs)
    assert stacked.shape == (3, 5)
    assert stacked.dtype == np.float32
    assert stacked[0, 1] == 1.0
    assert stacked[1, 2] == 1.0
    assert stacked[2, 3] == 1.0
    assert stacked.sum() == 3.0


def test_stack_targets_empty(cs):
    stacked = stack_targets([], cs)
    assert stacked.shape == (0, 5)
    assert stacked.dtype == np.float32


def test_stack_targets_handles_missing_labels(cs):
    records = [{"labels": ["squats - shallow"]}, {}, {"labels": []}]
    stacked = stack_targets(records, cs)
    assert stacked.shape == (3, 5)
    assert stacked.sum() == 1.0  # only the first record contributed


# ---------------------------------------------------------------------------
# _parse_clip_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("video_path,expected", [
    ("./00012345.mp4",          12345),
    ("00012345.mp4",            12345),
    ("/some/abs/00000007.mp4",  7),
    ("./12345.mp4",             12345),
    ("./00000000.mp4",          0),
])
def test_parse_clip_id_valid(video_path, expected):
    assert _parse_clip_id(video_path) == expected


@pytest.mark.parametrize("video_path", ["", "./not_a_number.mp4", "./abc.mp4"])
def test_parse_clip_id_invalid_returns_minus_one(video_path):
    assert _parse_clip_id(video_path) == -1


# ---------------------------------------------------------------------------
# Round-trip integration with the real class space
# ---------------------------------------------------------------------------


def test_real_squat_labels_all_resolve():
    """Real squat label strings from QEVD should all resolve in v6.1 space."""
    real_squat_labels = [
        "squats - shallow",
        "squats - back not straight",
        "squats - knees over toes",
        "squats - narrow",
        "squats - wide",
        "squats - 90 degrees",
        "squats - shoulder-width",
        "squats - no obvious issue",
        "squats - insufficient",
    ]
    target = build_v6_1_target(real_squat_labels)
    # All 9 should resolve in the real v6.1 spec (verified via test_qevd_class_space).
    assert int(target.sum()) == len(real_squat_labels)
