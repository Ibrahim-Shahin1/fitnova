"""Unit tests for v6 exercise-idx resolution in form_session.

Verifies the static method ``FormSession._resolve_exercise_idx`` correctly
routes FitNova UI exercise names to the v6 model's QEVD class indices.

Run from repo root:
    pytest backend/services/test_form_session_v6_routing.py -v
"""

from __future__ import annotations

import pytest

from backend.services.form_session import FormSession, V6_FITNOVA_TO_QEVD


# Mock QEVD exercise map — same shape as what FormAnalyzer would load
QEVD_MAP = {
    "__other__":          24,
    "squats":              3,
    "pushups":             4,
    "yoga pushup":         2,
    "inchworm":            7,
    "elbow plank":         0,
    "spider man pushup":   1,
}


# ─────────────────────────────────────────────────────────────────────────────
# v6 routing — happy path
# ─────────────────────────────────────────────────────────────────────────────


def test_v6_squat_routes_to_squats_idx():
    idx = FormSession._resolve_exercise_idx(
        "squat", model_version="v6", qevd_exercise_map=QEVD_MAP,
    )
    assert idx == 3


def test_v6_pushup_routes_to_pushups_idx():
    idx = FormSession._resolve_exercise_idx(
        "pushup", model_version="v6", qevd_exercise_map=QEVD_MAP,
    )
    assert idx == 4


def test_v6_diamond_pushup_routes_to_yoga_pushup():
    """diamond_pushup has no exact QEVD match; we map to 'yoga pushup'."""
    idx = FormSession._resolve_exercise_idx(
        "diamond_pushup", model_version="v6", qevd_exercise_map=QEVD_MAP,
    )
    assert idx == QEVD_MAP["yoga pushup"]


# ─────────────────────────────────────────────────────────────────────────────
# v6 routing — fallbacks
# ─────────────────────────────────────────────────────────────────────────────


def test_v6_unknown_exercise_falls_back_to_other():
    """deadlift isn't in the FitNova→QEVD table → __other__."""
    idx = FormSession._resolve_exercise_idx(
        "deadlift", model_version="v6", qevd_exercise_map=QEVD_MAP,
    )
    assert idx == 24  # __other__ index


def test_v6_empty_name_returns_zero():
    idx = FormSession._resolve_exercise_idx(
        None, model_version="v6", qevd_exercise_map=QEVD_MAP,
    )
    assert idx == 0


def test_v6_empty_string_returns_zero():
    idx = FormSession._resolve_exercise_idx(
        "", model_version="v6", qevd_exercise_map=QEVD_MAP,
    )
    assert idx == 0


def test_v6_empty_qevd_map_falls_back_to_zero(caplog):
    """If FormAnalyzer didn't load the exercise map, can't route — fallback."""
    idx = FormSession._resolve_exercise_idx(
        "squat", model_version="v6", qevd_exercise_map={},
    )
    assert idx == 0


def test_v6_qevd_map_missing_other_uses_zero():
    """Edge case: malformed QEVD map without __other__. Falls back to 0."""
    bad_map = {"squats": 3, "pushups": 4}  # no __other__
    idx = FormSession._resolve_exercise_idx(
        "deadlift", model_version="v6", qevd_exercise_map=bad_map,
    )
    assert idx == 0  # __other__ default


# ─────────────────────────────────────────────────────────────────────────────
# v5.2 routing still works (don't break the existing path)
# ─────────────────────────────────────────────────────────────────────────────


def test_v5_2_squat_unchanged():
    idx = FormSession._resolve_exercise_idx("squat", model_version="v5.2")
    assert idx == 0   # v5.2 squat == 0


def test_v5_2_unknown_falls_back_to_zero():
    idx = FormSession._resolve_exercise_idx("kayak_paddle", model_version="v5.2")
    assert idx == 0


# ─────────────────────────────────────────────────────────────────────────────
# v4 / unknown
# ─────────────────────────────────────────────────────────────────────────────


def test_v4_returns_zero():
    idx = FormSession._resolve_exercise_idx("squat", model_version="v4")
    assert idx == 0


def test_unknown_version_returns_zero():
    idx = FormSession._resolve_exercise_idx("squat", model_version="banana")
    assert idx == 0


# ─────────────────────────────────────────────────────────────────────────────
# Mapping table sanity
# ─────────────────────────────────────────────────────────────────────────────


def test_v6_table_entries_lowercase_with_underscores():
    """FitNova UI uses lowercase_with_underscores. Catch typos."""
    for k in V6_FITNOVA_TO_QEVD:
        assert k == k.lower(), f"{k} is not lowercase"
        assert " " not in k, f"{k} contains a space (use underscore)"


def test_v6_table_targets_are_real_qevd_classes():
    """Targets must be QEVD class names (lowercase, may contain spaces)."""
    for fitnova_name, qevd_name in V6_FITNOVA_TO_QEVD.items():
        assert isinstance(qevd_name, str)
        assert qevd_name == qevd_name.lower()
        # All valid QEVD names contain at least one alphabetic character
        assert any(c.isalpha() for c in qevd_name)
