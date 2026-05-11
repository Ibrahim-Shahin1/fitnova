"""Tests for the v6.1 cleaned QEVD class-space loader."""
from __future__ import annotations

import json

import numpy as np
import pytest

from backend.training.preprocessing.qevd_class_space import (
    DEFAULT_SPEC_PATH,
    QEVDClassSpace,
    get_default_class_space,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_space() -> QEVDClassSpace:
    """The actual v6.1 class space loaded from the repo's JSON."""
    return QEVDClassSpace.load()


@pytest.fixture
def tiny_space(tmp_path) -> QEVDClassSpace:
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
# Real-spec contract tests (catch class-space drift in CI)
# ---------------------------------------------------------------------------


def test_real_spec_file_exists():
    assert DEFAULT_SPEC_PATH.is_file(), (
        f"v6.1 class space JSON missing at {DEFAULT_SPEC_PATH}. "
        f"Run `python -m backend.training.preprocessing._build_qevd_class_space` "
        f"to regenerate."
    )


def test_real_spec_version_is_v6_1(real_space):
    assert real_space.version.startswith("v6.1")


def test_real_spec_class_count_in_expected_range(real_space):
    # Cleanup rules are deterministic; this catches accidental rule changes.
    assert 1300 <= real_space.num_classes <= 1500


def test_real_spec_prefix_count_in_expected_range(real_space):
    assert 130 <= real_space.num_prefixes <= 160


def test_real_spec_squats_keeps_all_form_variants(real_space):
    indices = real_space.class_indices_for_prefix("squats")
    decoded = {real_space.decode(i)[1] for i in indices}
    # The defect labels we MUST preserve for the powerlifting use case.
    must_have = {"no obvious issue", "shallow", "back not straight",
                 "knees over toes", "narrow", "wide", "90 degrees",
                 "shoulder-width", "insufficient"}
    missing = must_have - decoded
    assert not missing, f"Squat form variants dropped by cleanup: {missing}"


def test_real_spec_drops_not_visible(real_space):
    not_visible = [
        c for c in real_space._classes if c["variation"].lower() == "not visible"
    ]
    assert not_visible == []


def test_real_spec_drops_long_tail(real_space):
    counts = real_space.class_counts
    assert counts.min() >= 50, "min_clips_per_class threshold violated"


# ---------------------------------------------------------------------------
# Lookup correctness (tiny fixture)
# ---------------------------------------------------------------------------


def test_class_idx_for_known_label(tiny_space):
    assert tiny_space.class_idx_for_label("squats - shallow") == 1
    assert tiny_space.class_idx_for_label("pushups - narrow") == 3


def test_class_idx_for_unknown_label_returns_none(tiny_space):
    assert tiny_space.class_idx_for_label("nonexistent - foo") is None
    assert tiny_space.class_idx_for_label("squats - hold") is None  # not in tiny spec


def test_class_idx_handles_no_dash_format(tiny_space):
    # Some labels in the wild may have no " - " separator; should not crash.
    assert tiny_space.class_idx_for_label("plain_no_dash") is None


def test_class_idx_tolerates_extra_whitespace(tiny_space):
    # Stripping happens after the split, so extra surrounding whitespace
    # on either side of the " - " separator should still resolve.
    assert tiny_space.class_idx_for_label("squats  -  shallow") == 1
    assert tiny_space.class_idx_for_label(" squats - shallow ") == 1
    assert tiny_space.class_idx_for_label("squats - shallow") == 1


def test_decode_roundtrip(tiny_space):
    for label in ("squats - shallow", "pushups - no obvious issue"):
        idx = tiny_space.class_idx_for_label(label)
        assert idx is not None
        prefix, variation = tiny_space.decode(idx)
        assert label == f"{prefix} - {variation}"


def test_decode_invalid_idx_raises(tiny_space):
    with pytest.raises(IndexError):
        tiny_space.decode(999)
    with pytest.raises(IndexError):
        tiny_space.decode(-1)


def test_class_indices_for_prefix(tiny_space):
    assert tiny_space.class_indices_for_prefix("squats") == [0, 1, 2]
    assert tiny_space.class_indices_for_prefix("pushups") == [3, 4]
    assert tiny_space.class_indices_for_prefix("nonexistent") == []


def test_has_prefix(tiny_space):
    assert tiny_space.has_prefix("squats")
    assert not tiny_space.has_prefix("backflip")


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def test_encode_single_label(tiny_space):
    target = tiny_space.encode_labels(["squats - shallow"])
    assert target.shape == (5,)
    assert target.dtype == np.float32
    assert target[1] == 1.0
    assert target.sum() == 1.0


def test_encode_multi_label(tiny_space):
    target = tiny_space.encode_labels(["squats - shallow", "squats - back not straight"])
    assert target[1] == 1.0
    assert target[2] == 1.0
    assert target.sum() == 2.0


def test_encode_silently_skips_unknown(tiny_space):
    target = tiny_space.encode_labels(["squats - shallow", "nonexistent - foo"])
    assert target[1] == 1.0
    assert target.sum() == 1.0  # the unknown label contributed nothing


def test_encode_empty_labels(tiny_space):
    target = tiny_space.encode_labels([])
    assert target.shape == (5,)
    assert target.sum() == 0.0


def test_encode_repeated_label_idempotent(tiny_space):
    target = tiny_space.encode_labels(["squats - shallow", "squats - shallow"])
    assert target[1] == 1.0
    assert target.sum() == 1.0


# ---------------------------------------------------------------------------
# Class weights (Cui et al. 2019 effective-number)
# ---------------------------------------------------------------------------


def test_inverse_frequency_weights_shape(tiny_space):
    w = tiny_space.inverse_frequency_weights()
    assert w.shape == (5,)
    assert w.dtype == np.float32


def test_inverse_frequency_weights_inversely_correlated_with_count(tiny_space):
    w = tiny_space.inverse_frequency_weights()
    counts = tiny_space.class_counts
    # The smallest class should weigh strictly more than the largest.
    smallest = int(np.argmin(counts))
    largest = int(np.argmax(counts))
    assert w[smallest] > w[largest]


def test_inverse_frequency_weights_clipped(tiny_space):
    w = tiny_space.inverse_frequency_weights(min_weight=0.5, max_weight=2.0)
    assert w.min() >= 0.5
    assert w.max() <= 2.0


def test_inverse_frequency_weights_invalid_beta_raises(tiny_space):
    with pytest.raises(ValueError):
        tiny_space.inverse_frequency_weights(beta=1.0)
    with pytest.raises(ValueError):
        tiny_space.inverse_frequency_weights(beta=-0.1)


def test_real_class_weights_finite_and_positive(real_space):
    w = real_space.inverse_frequency_weights()
    assert np.all(np.isfinite(w))
    assert np.all(w > 0)


# ---------------------------------------------------------------------------
# Prefix index (v6.1 embedding input)
# ---------------------------------------------------------------------------


def test_prefix_idx_for_known_prefix(tiny_space):
    # Alphabetical order: 'pushups' < 'squats'
    assert tiny_space.prefix_idx_for("pushups") == 0
    assert tiny_space.prefix_idx_for("squats") == 1


def test_prefix_idx_for_unknown_prefix_returns_none(tiny_space):
    assert tiny_space.prefix_idx_for("nonexistent") is None
    assert tiny_space.prefix_idx_for("") is None


def test_decode_prefix_roundtrip(tiny_space):
    for prefix in ("squats", "pushups"):
        idx = tiny_space.prefix_idx_for(prefix)
        assert idx is not None
        assert tiny_space.decode_prefix(idx) == prefix


def test_decode_prefix_invalid_raises(tiny_space):
    with pytest.raises(IndexError):
        tiny_space.decode_prefix(999)
    with pytest.raises(IndexError):
        tiny_space.decode_prefix(-1)


def test_prefix_idx_for_label(tiny_space):
    assert tiny_space.prefix_idx_for_label("squats - shallow") == 1
    assert tiny_space.prefix_idx_for_label("pushups - narrow") == 0
    assert tiny_space.prefix_idx_for_label("nonexistent - foo") is None


def test_prefix_idx_for_label_no_dash(tiny_space):
    # When the label has no " - " separator the whole string is the prefix.
    assert tiny_space.prefix_idx_for_label("squats") == 1


def test_prefix_idx_ordering_is_alphabetical(real_space):
    # Deterministic embedding-input order; this catches accidental ordering
    # changes that would shift trained-model embedding rows out of sync.
    prefixes_in_idx_order = [
        real_space.decode_prefix(i) for i in range(real_space.num_prefixes)
    ]
    assert prefixes_in_idx_order == sorted(prefixes_in_idx_order)


def test_prefix_idx_range_matches_num_prefixes(real_space):
    for i in range(real_space.num_prefixes):
        prefix = real_space.decode_prefix(i)
        assert real_space.prefix_idx_for(prefix) == i


# ---------------------------------------------------------------------------
# Cached singleton
# ---------------------------------------------------------------------------


def test_get_default_class_space_returns_singleton():
    a = get_default_class_space()
    b = get_default_class_space()
    assert a is b
