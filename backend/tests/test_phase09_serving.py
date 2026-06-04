"""Serving integration tests (FastAPI TestClient).

Covers the raw per-rep output schema (confidence + threshold + detected, NO
severity_word / coaching text), multi-exercise routing (squat / ohp / shallow),
and per-rep thumbnail + rep_index/total_reps.
"""

from __future__ import annotations

import base64
import os
import tempfile

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from io import BytesIO
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from backend.app import app


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_synthetic_mp4(n_frames: int = 90, size: tuple[int, int] = (112, 112)) -> bytes:
    """Write random uint8 frames to a temp .mp4 and return its bytes."""
    tmp_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_path, fourcc, 30.0, size)
    rng = np.random.default_rng(seed=0)
    for _ in range(n_frames):
        frame = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    with open(tmp_path, "rb") as fh:
        data = fh.read()
    os.unlink(tmp_path)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

_SQUAT_RAW = {
    "exercise": "squat",
    "errors": [
        {"type": "KIE", "detected": True, "confidence": 0.72, "threshold": 0.614, "intervals": []},
        {"type": "KFE", "detected": False, "confidence": 0.25, "threshold": 0.385, "intervals": []},
    ],
}


@pytest.fixture()
def mock_services_phase09():
    """Wire mocked services onto app.state for the serving integration tests."""

    async def _classify_squat(frames, *, n_seeds=None):
        return _SQUAT_RAW

    mock_squat = MagicMock()
    mock_squat.model_ready = True
    mock_squat.kie_threshold = 0.614
    mock_squat.kfe_threshold = 0.385
    mock_squat._models = [1, 2, 3]
    mock_squat.classify_clip_async = _classify_squat

    mock_catalog = MagicMock()
    mock_catalog.whitelisted_ids = {"squat": set(), "ohp": set(), "shallow": set()}
    mock_catalog.get_catalog.return_value = []

    app.state.squat_form_service = mock_squat
    app.state.ohp_form_service = MagicMock()
    app.state.shallow_form_service = MagicMock()
    app.state.benchmark_catalog = mock_catalog
    app.state.recommender = MagicMock()
    app.state.llm_adapter = MagicMock()
    app.state.chat_service = MagicMock()
    app.state.coach_service = MagicMock()

    yield mock_squat

    app.state.squat_form_service = MagicMock()
    app.state.ohp_form_service = MagicMock()
    app.state.shallow_form_service = MagicMock()
    app.state.benchmark_catalog = MagicMock()


# ─────────────────────────────────────────────────────────────────────────────
# No severity_word, every error has threshold
# ─────────────────────────────────────────────────────────────────────────────


def test_no_severity_word(mock_services_phase09) -> None:
    """No error in any rep carries severity_word; every error carries threshold in [0,1]."""
    client = TestClient(app, raise_server_exceptions=True)
    mp4_bytes = make_synthetic_mp4()
    resp = client.post(
        "/analyze-form-video",
        data={"exercise": "squat"},
        files={"file": ("clip.mp4", BytesIO(mp4_bytes), "video/mp4")},
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    for rep in body["reps"]:
        for err in rep["errors"]:
            assert "severity_word" not in err, f"severity_word present in error: {err}"
            assert "threshold" in err, f"threshold missing from error: {err}"
            assert 0.0 <= err["threshold"] <= 1.0, f"threshold out of range: {err['threshold']}"


# ─────────────────────────────────────────────────────────────────────────────
# Raw output schema — exact error field set
# ─────────────────────────────────────────────────────────────────────────────


def test_raw_output_schema(mock_services_phase09) -> None:
    """Each error has exactly the expected field set (type, detected, confidence, threshold, intervals)."""
    client = TestClient(app, raise_server_exceptions=True)
    mp4_bytes = make_synthetic_mp4()
    resp = client.post(
        "/analyze-form-video",
        data={"exercise": "squat"},
        files={"file": ("clip.mp4", BytesIO(mp4_bytes), "video/mp4")},
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    required_fields = {"type", "detected", "confidence", "threshold", "intervals"}
    for rep in body["reps"]:
        for err in rep["errors"]:
            present = set(err.keys())
            # ground_truth is optional (None for non-catalog uploads) — exclude from required
            assert required_fields <= present, f"Missing fields: {required_fields - present}"
            assert "severity_word" not in present, "severity_word must not appear"
            assert err["type"] in {"KIE", "KFE", "ELBOWS", "KNEES", "DEPTH"}
            assert isinstance(err["detected"], bool)
            assert 0.0 <= err["confidence"] <= 1.0
            assert 0.0 <= err["threshold"] <= 1.0
            assert isinstance(err["intervals"], list)


# ─────────────────────────────────────────────────────────────────────────────
# Rep carries rep_index, total_reps, and a decodable thumbnail
# ─────────────────────────────────────────────────────────────────────────────


def test_rep_info_schema(mock_services_phase09) -> None:
    """Each rep carries rep_index (int), total_reps (int), and a non-empty base64-decodable thumbnail."""
    client = TestClient(app, raise_server_exceptions=True)
    mp4_bytes = make_synthetic_mp4()
    resp = client.post(
        "/analyze-form-video",
        data={"exercise": "squat"},
        files={"file": ("clip.mp4", BytesIO(mp4_bytes), "video/mp4")},
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert len(body["reps"]) >= 1, "Expected at least one rep"
    for rep in body["reps"]:
        assert "rep_index" in rep, "rep_index missing from rep"
        assert "total_reps" in rep, "total_reps missing from rep"
        assert isinstance(rep["rep_index"], int)
        assert isinstance(rep["total_reps"], int)
        thumb = rep.get("thumbnail")
        assert thumb is not None and len(thumb) > 0, "thumbnail is empty or missing"
        decoded = base64.b64decode(thumb)
        assert len(decoded) > 50, f"thumbnail decoded to only {len(decoded)} bytes — likely empty"


# ─────────────────────────────────────────────────────────────────────────────
# OHP routing (ELBOWS + KNEES error types)
# ─────────────────────────────────────────────────────────────────────────────

_OHP_RAW = {
    "exercise": "ohp",
    "errors": [
        {"type": "ELBOWS", "detected": True,  "confidence": 0.71, "threshold": 0.357, "intervals": []},
        {"type": "KNEES",  "detected": False, "confidence": 0.31, "threshold": 0.476, "intervals": []},
    ],
}


def test_ohp_routing(mock_services_phase09) -> None:
    """POST exercise=ohp routes to OHPFormService; response has ELBOWS + KNEES errors with threshold."""

    async def _classify_ohp(frames, *, n_seeds=None):
        return _OHP_RAW

    mock_ohp = MagicMock()
    mock_ohp.model_ready = True
    mock_ohp.elbows_threshold = 0.357
    mock_ohp.knees_threshold = 0.476
    mock_ohp._models = [1, 2]
    mock_ohp.classify_clip_async = _classify_ohp
    app.state.ohp_form_service = mock_ohp

    client = TestClient(app, raise_server_exceptions=True)
    mp4_bytes = make_synthetic_mp4()
    resp = client.post(
        "/analyze-form-video",
        data={"exercise": "ohp"},
        files={"file": ("ohp_clip.mp4", BytesIO(mp4_bytes), "video/mp4")},
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["exercise"] == "ohp"
    assert len(body["reps"]) >= 1
    rep = body["reps"][0]
    error_types = {e["type"] for e in rep["errors"]}
    assert "ELBOWS" in error_types, f"ELBOWS missing from error types: {error_types}"
    assert "KNEES"  in error_types, f"KNEES missing from error types: {error_types}"
    for err in rep["errors"]:
        assert "threshold" in err
        assert 0.0 <= err["threshold"] <= 1.0
        assert "severity_word" not in err


# ─────────────────────────────────────────────────────────────────────────────
# Shallow routing (DEPTH single-rep, catalog clip_id, thumbnail)
# ─────────────────────────────────────────────────────────────────────────────

_SHALLOW_RAW = {
    "exercise": "shallow",
    "errors": [
        {"type": "DEPTH", "detected": True, "confidence": 0.82, "threshold": 0.395, "intervals": []},
    ],
}


def test_shallow_routing(mock_services_phase09, tmp_path) -> None:
    """POST exercise=shallow with a whitelisted clip_id returns DEPTH error, total_reps=1, thumbnail."""
    import cv2
    import numpy as np

    # Create a minimal JPEG to serve as the catalog crop.
    crop_file = tmp_path / "test_crop.jpg"
    frame = np.zeros((112, 112, 3), dtype=np.uint8)
    cv2.imwrite(str(crop_file), frame)

    async def _classify_shallow(image_path, *, n_seeds=None):
        return _SHALLOW_RAW

    mock_shallow = MagicMock()
    mock_shallow.model_ready = True
    mock_shallow.threshold = 0.395
    mock_shallow._models = [1, 2, 3]
    mock_shallow.classify_image_async = _classify_shallow
    app.state.shallow_form_service = mock_shallow

    mock_cat = MagicMock()
    mock_cat.whitelisted_ids = {"squat": set(), "ohp": set(), "shallow": {"test_shallow_id"}}
    mock_cat.media_path.return_value = crop_file
    mock_cat.get_catalog.return_value = [
        {"clip_id": "test_shallow_id", "ground_truth": {"DEPTH": 1}, "score": {"DEPTH": 0.82}}
    ]
    app.state.benchmark_catalog = mock_cat

    client = TestClient(app, raise_server_exceptions=True)

    # Shallow endpoint does not take a file upload — use a dummy file (it's ignored).
    resp = client.post(
        "/analyze-form-video",
        data={"exercise": "shallow", "clip_id": "test_shallow_id"},
        files={"file": ("dummy.mp4", BytesIO(b"\x00" * 10), "video/mp4")},
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["exercise"] == "shallow"
    assert body["total_reps"] == 1
    assert len(body["reps"]) == 1
    rep = body["reps"][0]
    assert rep["total_reps"] == 1
    error_types = {e["type"] for e in rep["errors"]}
    assert "DEPTH" in error_types, f"DEPTH missing: {error_types}"
    assert len(rep["errors"]) == 1, f"Expected 1 DEPTH error, got {len(rep['errors'])}"
    assert rep["errors"][0]["ground_truth"] == 1, "ground_truth not wired from catalog"
    thumb = rep.get("thumbnail")
    assert thumb is not None and len(thumb) > 0, "thumbnail missing for shallow"


def test_shallow_missing_clip_id(mock_services_phase09) -> None:
    """POST exercise=shallow with no clip_id returns 400."""
    client = TestClient(app, raise_server_exceptions=True)
    mp4_bytes = make_synthetic_mp4()
    resp = client.post(
        "/analyze-form-video",
        data={"exercise": "shallow"},
        files={"file": ("clip.mp4", BytesIO(mp4_bytes), "video/mp4")},
    )
    assert resp.status_code == 400, f"Expected 400 for missing clip_id, got {resp.status_code}: {resp.text}"
