"""Unit tests for backend.training.evaluation.qevd_fitcoach_eval.

Covers the metric helpers + segment-target derivation. Skips the actual
model-inference path (that needs a trained model + extracted videos).

Run from repo root:
    pytest backend/training/evaluation/test_qevd_fitcoach_eval.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.training.evaluation.qevd_fitcoach_eval import (
    QUALITY_SPEARMAN_MIN, REP_COUNT_MAE_MAX, SEGMENT_LENGTH_S,
    derive_segment_targets, mean_absolute_error, spearman_correlation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants pinning master-plan thresholds
# ─────────────────────────────────────────────────────────────────────────────


def test_thresholds_match_master_plan():
    assert QUALITY_SPEARMAN_MIN == 0.30
    assert REP_COUNT_MAE_MAX    == 2.00
    assert SEGMENT_LENGTH_S     == 30.0


# ─────────────────────────────────────────────────────────────────────────────
# spearman_correlation
# ─────────────────────────────────────────────────────────────────────────────


def test_spearman_perfectly_correlated():
    rho = spearman_correlation([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
    assert math.isclose(rho, 1.0, abs_tol=1e-9)


def test_spearman_anti_correlated():
    rho = spearman_correlation([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
    assert math.isclose(rho, -1.0, abs_tol=1e-9)


def test_spearman_uncorrelated_random():
    rng = np.random.RandomState(0)
    x = rng.randn(200)
    y = rng.randn(200)
    rho = spearman_correlation(x.tolist(), y.tolist())
    assert abs(rho) < 0.2


def test_spearman_constant_input_returns_nan():
    rho = spearman_correlation([1, 1, 1, 1], [1, 2, 3, 4])
    assert math.isnan(rho)


def test_spearman_too_few_points_returns_nan():
    assert math.isnan(spearman_correlation([1.0], [2.0]))
    assert math.isnan(spearman_correlation([], []))


# ─────────────────────────────────────────────────────────────────────────────
# mean_absolute_error
# ─────────────────────────────────────────────────────────────────────────────


def test_mae_zero_when_perfect():
    assert mean_absolute_error([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_mae_basic():
    mae = mean_absolute_error([1.0, 2.0, 3.0], [2.0, 2.0, 5.0])
    assert math.isclose(mae, 1.0, abs_tol=1e-9)


def test_mae_handles_length_mismatch():
    """Truncates to the shorter input — matches our pipeline behaviour
    where window-aggregated segments may not align 1:1 with GT segments."""
    mae = mean_absolute_error([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0])
    assert math.isclose(mae, 0.0, abs_tol=1e-9)


def test_mae_empty():
    assert math.isnan(mean_absolute_error([], []))


# ─────────────────────────────────────────────────────────────────────────────
# derive_segment_targets
# ─────────────────────────────────────────────────────────────────────────────


def test_derive_segment_targets_dense_array():
    """Schema A: parallel feedbacks/timestamps arrays (most empty strings)."""
    record = {
        "feedbacks":           ["", "", "great", "great", "", "watch your form",
                                 "", "rep 1", "rep 1", "rep 2", "rep 2",
                                 "", "lower your hips"],
        "feedback_timestamps": [0.0, 1.0, 2.0, 2.5, 3.0, 6.0,
                                 7.0, 8.0, 8.5, 12.0, 12.5,
                                 15.0, 35.0],
    }
    quality_gt, rep_gt = derive_segment_targets(record, segment_length_s=30.0)
    # Both events fall within [0, 30) (segment 0) except the last at t=35 (seg 1)
    assert len(quality_gt) == 2
    assert len(rep_gt) == 2
    assert 0.0 <= quality_gt[0] <= 1.0
    assert 0.0 <= quality_gt[1] <= 1.0
    # Dense-array path can't reliably classify 'rep N' as rep-counting without
    # explicit type tags (it falls back to classify_feedback_type which labels
    # short utterances as acknowledgment). So rep_gt is 0 for both segments.
    # The list-of-dicts schema (Schema B) is the path that exercises type tags.
    assert rep_gt[0] == 0
    assert rep_gt[1] == 0


def test_derive_segment_targets_list_of_dicts():
    """Schema B: list of {text, t_start, type}."""
    record = {
        "feedbacks": [
            {"text": "great",  "t_start":  2.5, "type": "affirmative"},
            {"text": "watch your knees", "t_start":  6.0, "type": "corrective"},
            {"text": "rep 1",  "t_start": 10.0, "type": "rep-counting"},
            {"text": "rep 2",  "t_start": 14.0, "type": "rep-counting"},
            {"text": "lower",  "t_start": 36.0, "type": "corrective"},
        ],
    }
    q, r = derive_segment_targets(record)
    assert len(q) == 2
    assert r[0] == 2     # rep 1 + rep 2 in seg 0
    assert r[1] == 0     # no rep-counting in seg 1
    # First segment had affirmative + corrective, no acks
    # quality = 1 - 0.6 * (1/2) + 0.2 * (1/2) + 0 = 0.8
    assert math.isclose(q[0], 0.8, abs_tol=1e-3)


def test_derive_segment_targets_empty_record():
    record = {"feedbacks": [], "feedback_timestamps": []}
    q, r = derive_segment_targets(record)
    assert q == [] and r == []


def test_derive_segment_targets_silence_default():
    """A segment with NO non-rep feedback events gets 0.7 (silence default)."""
    record = {
        "feedbacks": [
            {"text": "rep 1", "t_start": 5.0, "type": "rep-counting"},
            {"text": "rep 2", "t_start": 15.0, "type": "rep-counting"},
        ],
    }
    q, r = derive_segment_targets(record)
    # Only rep-counting events in segment 0; no quality content -> default
    assert math.isclose(q[0], 0.7, abs_tol=1e-9)
    assert r[0] == 2
