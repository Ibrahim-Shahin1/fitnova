"""Benchmark catalog + media-serving endpoint tests (TestClient).

Covers SRV-04 — the full test-split catalog per exercise (Squat 244 / OHP 339 /
Shallow 540) with ground-truth labels, and the media endpoint's clip_id whitelist
(path-traversal guard returns 404 before any filesystem access).
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

torch = pytest.importorskip("torch")

from fastapi.testclient import TestClient

from backend.app import app


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def catalog_client():
    """TestClient with a real BenchmarkCatalog wired into app.state.

    Loads the real catalog from the local Fitness-AQA tree so count/schema
    assertions are meaningful.  Skips if the dataset or results.pkl files are absent.
    """
    try:
        from backend.services.benchmark_catalog import BenchmarkCatalog
        cat = BenchmarkCatalog(
            fitness_aqa_root=(
                "Fitness-AQA/Fitness-AQA_dataset_release-20260518T073812Z-3-001"
                "/Fitness-AQA_dataset_release"
            ),
            results_dir="backend/data/benchmark",
        )
    except Exception as exc:
        pytest.skip(f"BenchmarkCatalog failed to load (dataset absent?): {exc}")

    app.state.benchmark_catalog = cat
    app.state.recommender = MagicMock()
    app.state.llm_adapter = MagicMock()
    app.state.chat_service = MagicMock()
    app.state.coach_service = MagicMock()
    app.state.squat_form_service = MagicMock()
    app.state.ohp_form_service = MagicMock()
    app.state.shallow_form_service = MagicMock()

    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def mock_catalog_client():
    """TestClient with a minimal mock catalog — no dataset on disk required.

    Used for whitelist-gate and 400/404 tests that must not depend on media files.
    """
    mock_cat = MagicMock()
    mock_cat.whitelisted_ids = {
        "squat":   {"real_squat_id"},
        "ohp":     {"real_ohp_id"},
        "shallow": {"real_shallow_id"},
    }

    def _get_catalog(exercise: str):
        if exercise == "squat":
            return [{"clip_id": "real_squat_id", "ground_truth": {"KIE": 1, "KFE": 0}, "score": {"KIE": 0.7, "KFE": 0.2}}]
        if exercise == "ohp":
            return [{"clip_id": "real_ohp_id", "ground_truth": {"ELBOWS": 0, "KNEES": 1}, "score": {"ELBOWS": 0.3, "KNEES": 0.8}}]
        if exercise == "shallow":
            return [{"clip_id": "real_shallow_id", "ground_truth": {"DEPTH": 1}, "score": {"DEPTH": 0.9}}]
        return []

    mock_cat.get_catalog.side_effect = _get_catalog

    app.state.benchmark_catalog = mock_cat
    app.state.recommender = MagicMock()
    app.state.llm_adapter = MagicMock()
    app.state.chat_service = MagicMock()
    app.state.coach_service = MagicMock()
    app.state.squat_form_service = MagicMock()
    app.state.ohp_form_service = MagicMock()
    app.state.shallow_form_service = MagicMock()

    return TestClient(app, raise_server_exceptions=True)


# ─────────────────────────────────────────────────────────────────────────────
# Catalog counts (real dataset)
# ─────────────────────────────────────────────────────────────────────────────


def test_catalog_counts(catalog_client) -> None:
    """GET catalog for each exercise returns the expected test-split count."""
    for exercise, expected in [("squat", 244), ("ohp", 339), ("shallow", 540)]:
        resp = catalog_client.get(f"/api/benchmark/catalog?exercise={exercise}")
        assert resp.status_code == 200, f"{exercise}: {resp.status_code} {resp.text}"
        body = resp.json()
        assert body["exercise"] == exercise
        clips = body["clips"]
        assert len(clips) == expected, f"{exercise}: expected {expected}, got {len(clips)}"


# ─────────────────────────────────────────────────────────────────────────────
# Catalog schema (real dataset)
# ─────────────────────────────────────────────────────────────────────────────


def test_catalog_schema(catalog_client) -> None:
    """Each catalog entry has clip_id, ground_truth (per applicable error), and score."""
    cases = [
        ("squat",   {"KIE", "KFE"}),
        ("ohp",     {"ELBOWS", "KNEES"}),
        ("shallow", {"DEPTH"}),
    ]
    for exercise, expected_errors in cases:
        resp = catalog_client.get(f"/api/benchmark/catalog?exercise={exercise}")
        assert resp.status_code == 200
        clips = resp.json()["clips"]
        entry = clips[0]
        assert set(entry.keys()) == {"clip_id", "ground_truth", "score"}, (
            f"{exercise} entry keys: {set(entry.keys())}"
        )
        assert isinstance(entry["clip_id"], str) and entry["clip_id"]
        assert set(entry["ground_truth"].keys()) == expected_errors, (
            f"{exercise} ground_truth keys: {set(entry['ground_truth'].keys())}"
        )
        assert set(entry["score"].keys()) == expected_errors
        for err_key in expected_errors:
            assert entry["ground_truth"][err_key] in (0, 1), (
                f"{exercise}.{err_key} ground_truth not 0/1: {entry['ground_truth'][err_key]}"
            )
            assert isinstance(entry["score"][err_key], float)


# ─────────────────────────────────────────────────────────────────────────────
# Catalog 400 on unknown exercise
# ─────────────────────────────────────────────────────────────────────────────


def test_catalog_bad_exercise(mock_catalog_client) -> None:
    """GET /api/benchmark/catalog?exercise=bench returns 400."""
    resp = mock_catalog_client.get("/api/benchmark/catalog?exercise=bench")
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"


# ─────────────────────────────────────────────────────────────────────────────
# Media endpoint whitelist gate (path-traversal + 400 exercise)
# ─────────────────────────────────────────────────────────────────────────────


def test_media_clipid_allowlist(mock_catalog_client) -> None:
    """Media endpoint 404s every non-whitelisted clip_id (incl. traversal); 400s unknown exercise."""

    # Non-existent id in a valid exercise -> 404
    resp = mock_catalog_client.get("/api/benchmark/media/squat/__not_a_real_id__")
    assert resp.status_code == 404, f"Expected 404 for unknown id, got {resp.status_code}"

    # Path-traversal attempt via URL encoding -> 404 (not in whitelist set)
    resp = mock_catalog_client.get("/api/benchmark/media/squat/..%2f..%2fetc%2fpasswd")
    assert resp.status_code == 404, f"Expected 404 for traversal (encoded), got {resp.status_code}"

    # Unknown exercise -> 400
    resp = mock_catalog_client.get("/api/benchmark/media/bogus/any_id")
    assert resp.status_code == 400, f"Expected 400 for unknown exercise, got {resp.status_code}"


def test_media_whitelisted_id_200(tmp_path) -> None:
    """Whitelisted clip_id returns 200 with body when the file exists on disk."""
    # Create a minimal media file so FileResponse has something to serve.
    media_file = tmp_path / "test_clip.mp4"
    media_file.write_bytes(b"\x00" * 100)

    mock_cat = MagicMock()
    mock_cat.whitelisted_ids = {"squat": {"test_clip"}}
    mock_cat.media_path.return_value = media_file

    app.state.benchmark_catalog = mock_cat
    app.state.recommender = MagicMock()
    app.state.llm_adapter = MagicMock()
    app.state.chat_service = MagicMock()
    app.state.coach_service = MagicMock()
    app.state.squat_form_service = MagicMock()
    app.state.ohp_form_service = MagicMock()
    app.state.shallow_form_service = MagicMock()

    client = TestClient(app, raise_server_exceptions=True)
    resp = client.get("/api/benchmark/media/squat/test_clip")
    assert resp.status_code == 200
    assert len(resp.content) > 0
