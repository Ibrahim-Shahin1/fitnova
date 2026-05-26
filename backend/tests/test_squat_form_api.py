"""
Tests for the Squat inference service, rep segmenter, and the redefined form endpoints.

Wave 0 scaffold — covers:
  - SquatFormService construction + graceful degradation (API-01, D-02)
  - severity_word bands (D-10)
  - classify_clip determinism (spatial_val center-crop, not spatial_train random crop — D-02 / Pitfall 1)
  - RepSegmenter single-rep fallback (API-02, D-03)
  - LiveWindowTrigger gate arithmetic (D-03)

Plan 03 additions — REST integration tests:
  - mock_services_squat fixture (autouse=False — scoped to squat API tests)
  - test_health_schema: squat_model_ready/seeds_loaded/kie_threshold/kfe_threshold present; no model_version
  - test_upload_response_schema: synthetic mp4 + exercise=squat -> 200 + D-05 shape
  - test_upload_no_model: neutral dict -> 200 (not 500)
  - test_upload_too_large: Content-Length > MAX_UPLOAD_BYTES -> 413
  - test_upload_bad_exercise: exercise=bench -> 400
"""

from __future__ import annotations

import base64
import os
import tempfile

import cv2
import numpy as np
import pytest

# Skip the entire module cleanly when torch is not installed (CI without torch).
torch = pytest.importorskip("torch")

from backend.services.squat_form_service import SquatFormService, _severity
from backend.services.clip_decode import kneeaware_spatial_val
from backend.training.aqa.datasets.transforms import spatial_val
from backend.services.rep_segmenter import (
    LiveWindowTrigger,
    LIVE_MIN_GAP_FRAMES,
    LIVE_REP_WINDOW_FRAMES,
    segment_reps_by_motion_energy,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (reused by Plan 03 integration tests)
# ─────────────────────────────────────────────────────────────────────────────


def make_synthetic_mp4(n_frames: int = 90, size: tuple[int, int] = (112, 112)) -> bytes:
    """Write random uint8 frames to a temp .mp4 and return its bytes.

    Uses cv2.VideoWriter with the 'mp4v' fourcc at 30 fps.  The temp file is
    created, read back, and unlinked — caller receives raw bytes.
    """
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


def make_fake_jpeg(size: tuple[int, int] = (112, 112)) -> bytes:
    """Return a minimal valid JPEG (black frame) as bytes."""
    _, buf = cv2.imencode(".jpg", np.zeros((*size, 3), dtype=np.uint8))
    return buf.tobytes()


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — SquatFormService unit tests
# ─────────────────────────────────────────────────────────────────────────────


def test_service_loads() -> None:
    """SquatFormService constructs without raising, exposes bool model_ready and correct thresholds.

    Weights may be present (model_ready True) or absent (model_ready False) — both
    are acceptable here.  The test validates the API surface, not weight presence.
    """
    svc = SquatFormService(model_dir="backend/models/form_model_squat_md")
    assert isinstance(svc.model_ready, bool)
    assert svc.kie_threshold == pytest.approx(0.614)
    assert svc.kfe_threshold == pytest.approx(0.385)


def test_neutral_response_schema(tmp_path) -> None:
    """With no weights present, model_ready is False and classify_clip returns a valid D-05 neutral dict.

    Uses tmp_path (an empty/nonexistent dir) so real staged weights are never loaded
    — important because backend/models/form_model_squat_md/ already has best.pt files.
    """
    # Point at an empty directory so _load() finds no .pt files.
    empty_model_dir = str(tmp_path / "no_weights")
    svc = SquatFormService(model_dir=empty_model_dir)
    assert svc.model_ready is False

    dummy_frames = np.zeros((32, 3, 120, 120), dtype=np.uint8)
    result = svc.classify_clip(dummy_frames)

    assert result.get("model_not_loaded") is True
    errors = result.get("errors", [])
    assert len(errors) == 2
    assert errors[0]["type"] == "KIE"
    assert errors[1]["type"] == "KFE"
    for err in errors:
        assert set(err.keys()) >= {"type", "detected", "confidence", "severity_word", "intervals"}
        assert err["detected"] is False
        assert err["confidence"] == pytest.approx(0.0)
        assert err["severity_word"] == "none"
        assert err["intervals"] == []


def test_severity_word() -> None:
    """_severity returns the four expected bands: strong / moderate / possible / none."""
    assert _severity(0.85, True) == "strong"
    assert _severity(0.80, True) == "strong"   # boundary inclusive
    assert _severity(0.70, True) == "moderate"
    assert _severity(0.65, True) == "moderate"  # boundary inclusive
    assert _severity(0.50, True) == "possible"
    assert _severity(0.40, True) == "possible"
    assert _severity(0.50, False) == "none"
    assert _severity(0.90, False) == "none"     # detected=False always "none"


def test_classify_deterministic() -> None:
    """Repeated classify_clip calls on the same clip return byte-equal confidence values.

    Proves spatial_val (deterministic center-crop) is used, NOT spatial_train (random crop).
    Uses an in-process model built from scratch — no staged weights needed.
    """
    import torch.nn as nn
    from torchvision.models.video import r2plus1d_18

    # Build a tiny in-process model matching the exact serve-time architecture.
    m = r2plus1d_18(weights=None)
    assert m.fc.in_features == 512
    m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 2))
    m.eval()

    # Inject the model directly — bypass _load() so no disk access is needed.
    svc = SquatFormService(model_dir="/nonexistent_dir_for_test")
    svc._models = [m]
    svc._model_ready = True
    svc._default_n_seeds = 1

    # Deterministic random seed for the clip.
    rng = np.random.default_rng(seed=42)
    frames_uint8 = rng.integers(0, 256, (32, 3, 112, 112), dtype=np.uint8)

    result1 = svc.classify_clip(frames_uint8)
    result2 = svc.classify_clip(frames_uint8)

    # Confidence values must be identical (not just close) — exact float equality.
    assert result1["errors"][0]["confidence"] == result2["errors"][0]["confidence"], (
        "KIE confidence differs between calls — spatial_train (random crop) may be used"
    )
    assert result1["errors"][1]["confidence"] == result2["errors"][1]["confidence"], (
        "KFE confidence differs between calls — spatial_train (random crop) may be used"
    )


# ─────────────────────────────────────────────────────────────────────────────
# D-11 knee-aware spatial preprocessing tests
# ─────────────────────────────────────────────────────────────────────────────


def test_kneeaware_landscape_is_spatial_val() -> None:
    """kneeaware_spatial_val on LANDSCAPE input is element-equal to spatial_val.

    Parity guard: landscape input must pass through unchanged so the serving path
    reproduces the offline-eval 0.6304 F1 (Fitness-AQA dataset clips are landscape).
    Synthetic clip [8, 3, 120, 200] — W > H → landscape.
    """
    rng = np.random.default_rng(seed=7)
    frames_uint8 = rng.integers(0, 256, (8, 3, 120, 200), dtype=np.uint8)
    clip_tensor = torch.from_numpy(frames_uint8)

    out_kneeaware = kneeaware_spatial_val(clip_tensor)
    out_spatial_val = spatial_val(clip_tensor)

    assert out_kneeaware.shape == out_spatial_val.shape, (
        f"Shape mismatch: kneeaware={out_kneeaware.shape}, spatial_val={out_spatial_val.shape}"
    )
    assert torch.equal(out_kneeaware, out_spatial_val), (
        "kneeaware_spatial_val on landscape input is NOT element-equal to spatial_val — "
        "offline-eval parity broken"
    )


def test_kneeaware_portrait_differs_and_shape() -> None:
    """kneeaware_spatial_val on PORTRAIT input returns [3, T, 112, 112] and differs from spatial_val.

    Two assertions:
      1. Output shape is [3, 8, 112, 112] float32 — correct output contract.
      2. Output is NOT element-equal to spatial_val on the same input — proves the
         lower-body pre-crop is applied and produces a different crop window.

    Synthetic clip [8, 3, 200, 120] — H > W → portrait.
    """
    rng = np.random.default_rng(seed=13)
    frames_uint8 = rng.integers(0, 256, (8, 3, 200, 120), dtype=np.uint8)
    clip_tensor = torch.from_numpy(frames_uint8)

    out_kneeaware = kneeaware_spatial_val(clip_tensor)

    # Shape check
    assert out_kneeaware.shape == (3, 8, 112, 112), (
        f"kneeaware_spatial_val portrait output shape is {out_kneeaware.shape}, "
        f"expected (3, 8, 112, 112)"
    )
    assert out_kneeaware.dtype == torch.float32, (
        f"Expected float32 output, got {out_kneeaware.dtype}"
    )

    # NOT equal to plain spatial_val (proves the lower-body crop is applied)
    out_spatial_val = spatial_val(clip_tensor)
    assert not torch.equal(out_kneeaware, out_spatial_val), (
        "kneeaware_spatial_val on portrait input is unexpectedly element-equal to spatial_val "
        "— the lower-body pre-crop is not being applied"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — RepSegmenter unit tests
# ─────────────────────────────────────────────────────────────────────────────


def test_rep_segmenter_synthetic(tmp_path) -> None:
    """segment_reps_by_motion_energy returns a non-empty list of (int, int) tuples on a synthetic clip."""
    mp4_bytes = make_synthetic_mp4(n_frames=90, size=(112, 112))
    tmp_mp4 = str(tmp_path / "test_clip.mp4")
    with open(tmp_mp4, "wb") as fh:
        fh.write(mp4_bytes)

    intervals = segment_reps_by_motion_energy(tmp_mp4)

    assert isinstance(intervals, list)
    assert len(intervals) >= 1, "Must return at least one interval (single-rep fallback)"
    for start, end in intervals:
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert start <= end, f"Invalid interval: start={start} > end={end}"


def test_live_window_trigger() -> None:
    """LiveWindowTrigger fires exactly once after LIVE_MIN_GAP_FRAMES calls with buffer_len >= 32.

    Also verifies it never fires when buffer_len < LIVE_REP_WINDOW_FRAMES.
    """
    trigger = LiveWindowTrigger()

    # With buffer_len < LIVE_REP_WINDOW_FRAMES, should never fire.
    for _ in range(LIVE_MIN_GAP_FRAMES + 10):
        fired = trigger.should_fire(buffer_len=10)
        assert not fired, "Trigger fired unexpectedly when buffer_len < LIVE_REP_WINDOW_FRAMES"

    # Fresh trigger: exactly at LIVE_MIN_GAP_FRAMES calls with buffer_len >= 32.
    trigger2 = LiveWindowTrigger()
    fire_count = 0
    for i in range(LIVE_MIN_GAP_FRAMES):
        if trigger2.should_fire(buffer_len=LIVE_REP_WINDOW_FRAMES):
            fire_count += 1
    # The trigger fires exactly on the LIVE_MIN_GAP_FRAMES-th call.
    assert fire_count == 1, (
        f"Expected exactly 1 fire across {LIVE_MIN_GAP_FRAMES} calls with full buffer, "
        f"got {fire_count}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plan 03 — REST integration tests
# ─────────────────────────────────────────────────────────────────────────────
#
# These tests require the FastAPI app with mocked services so they are import-
# and network-free while still exercising the real endpoint routing + Pydantic
# validation.

import asyncio
from io import BytesIO
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from backend.app import MAX_UPLOAD_BYTES, app


# ── Valid D-05 dict returned by the mock service ──────────────────────────────

_VALID_D05 = {
    "exercise": "squat",
    "errors": [
        {
            "type": "KIE",
            "detected": True,
            "confidence": 0.72,
            "severity_word": "moderate",
            "intervals": [],
        },
        {
            "type": "KFE",
            "detected": False,
            "confidence": 0.25,
            "severity_word": "none",
            "intervals": [],
        },
    ],
}

_NEUTRAL_D05 = {
    "exercise": "squat",
    "errors": [
        {"type": "KIE", "detected": False, "confidence": 0.0, "severity_word": "none", "intervals": []},
        {"type": "KFE", "detected": False, "confidence": 0.0, "severity_word": "none", "intervals": []},
    ],
    "model_not_loaded": True,
}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_services_squat():
    """Wire mocked services onto app.state for REST integration tests.

    classify_clip_async is a REAL coroutine (not a MagicMock awaitable) returning
    a valid D-05 dict — required so `await svc.classify_clip_async(...)` works with
    the real endpoint code.
    """

    async def _classify_async(frames, *, n_seeds=None):
        return _VALID_D05

    mock_svc = MagicMock()
    mock_svc.model_ready = True
    mock_svc.kie_threshold = 0.614
    mock_svc.kfe_threshold = 0.385
    mock_svc._models = [1, 2, 3]  # len=3 → seeds_loaded=3
    mock_svc.classify_clip_async = _classify_async

    app.state.squat_form_service = mock_svc
    app.state.recommender = MagicMock()
    app.state.llm_adapter = MagicMock()
    app.state.chat_service = MagicMock()
    app.state.coach_service = MagicMock()

    yield mock_svc

    # Teardown — reset to a bare MagicMock so other test modules are unaffected.
    app.state.squat_form_service = MagicMock()


@pytest.fixture()
def squat_client(mock_services_squat):
    return TestClient(app, raise_server_exceptions=False)


# ── /health ───────────────────────────────────────────────────────────────────


def test_health_schema(squat_client):
    """GET /health returns status 200 with all D-05 squat fields and NO model_version."""
    response = squat_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "squat_model_ready" in data, "squat_model_ready missing from /health"
    assert "seeds_loaded" in data, "seeds_loaded missing from /health"
    assert "kie_threshold" in data, "kie_threshold missing from /health"
    assert "kfe_threshold" in data, "kfe_threshold missing from /health"
    assert "model_version" not in data, "model_version must be removed from /health (old field)"
    assert data["squat_model_ready"] is True
    assert data["seeds_loaded"] == 3


# ── /analyze-form-video ───────────────────────────────────────────────────────


def test_upload_response_schema(squat_client):
    """POST /analyze-form-video with a synthetic mp4 + exercise=squat returns 200 + D-05 shape."""
    mp4_bytes = make_synthetic_mp4(n_frames=90, size=(112, 112))
    response = squat_client.post(
        "/analyze-form-video",
        files={"file": ("test.mp4", BytesIO(mp4_bytes), "video/mp4")},
        data={"exercise": "squat"},
    )
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["exercise"] == "squat"
    assert data["total_reps"] >= 1
    assert isinstance(data["reps"], list)
    assert len(data["reps"]) >= 1
    rep = data["reps"][0]
    assert rep["exercise"] == "squat"
    assert isinstance(rep["errors"], list)
    for err in rep["errors"]:
        assert err["type"] in {"KIE", "KFE"}, f"Unexpected error type: {err['type']}"
        assert isinstance(err["detected"], bool)
        assert 0.0 <= err["confidence"] <= 1.0
        assert err["severity_word"] in {"none", "possible", "moderate", "strong"}
        assert isinstance(err["intervals"], list)


def test_upload_no_model(mock_services_squat, squat_client):
    """When model returns neutral dict (model_not_loaded=True), endpoint returns 200 (not 500)."""

    async def _neutral_async(frames, *, n_seeds=None):
        return _NEUTRAL_D05

    mock_services_squat.model_ready = False
    mock_services_squat.classify_clip_async = _neutral_async

    mp4_bytes = make_synthetic_mp4(n_frames=90, size=(112, 112))
    response = squat_client.post(
        "/analyze-form-video",
        files={"file": ("test.mp4", BytesIO(mp4_bytes), "video/mp4")},
        data={"exercise": "squat"},
    )
    assert response.status_code == 200, (
        f"No-model path returned {response.status_code} — expected 200 (graceful degradation)"
    )
    data = response.json()
    assert data["exercise"] == "squat"
    assert data["total_reps"] >= 1


def test_upload_too_large(squat_client):
    """POST /analyze-form-video with Content-Length > MAX_UPLOAD_BYTES returns 413."""
    oversized = MAX_UPLOAD_BYTES + 1
    response = squat_client.post(
        "/analyze-form-video",
        files={"file": ("big.mp4", BytesIO(b"x"), "video/mp4")},
        data={"exercise": "squat"},
        headers={"content-length": str(oversized)},
    )
    assert response.status_code == 413, (
        f"Expected 413 for oversized upload, got {response.status_code}: {response.text}"
    )


def test_upload_bad_exercise(squat_client):
    """POST /analyze-form-video with exercise=bench returns 400 (allowlist)."""
    mp4_bytes = make_synthetic_mp4(n_frames=30, size=(64, 64))
    response = squat_client.post(
        "/analyze-form-video",
        files={"file": ("test.mp4", BytesIO(mp4_bytes), "video/mp4")},
        data={"exercise": "bench"},
    )
    assert response.status_code == 400, (
        f"Expected 400 for unsupported exercise, got {response.status_code}: {response.text}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plan 04 — WS integration tests (API-04)
# ─────────────────────────────────────────────────────────────────────────────
#
# Uses Starlette's synchronous TestClient.websocket_connect — no pytest-asyncio
# needed (RESEARCH §7 VALIDATION note).  The mock_services_squat fixture is
# reused so the real lifespan (torch model load) is bypassed.


@pytest.fixture()
def client(mock_services_squat):
    """TestClient with raise_server_exceptions=True so assertion errors surface cleanly."""
    return TestClient(app, raise_server_exceptions=True)


def test_ws_session_protocol(client):
    """start_session -> session_started; end_session -> session_summary with exercise + rep_results.

    Verifies the basic three-step WS protocol without sending any frames.
    """
    with client.websocket_connect("/ws/form-session") as ws:
        # Step 1: start the session.
        ws.send_json({"type": "start_session", "exercise_hint": "squat"})
        started = ws.receive_json()
        assert started["type"] == "session_started", (
            f"Expected session_started, got {started}"
        )

        # Step 2: end the session immediately (no frames).
        ws.send_json({"type": "end_session"})
        summary = ws.receive_json()

    assert summary["type"] == "session_summary", (
        f"Expected session_summary as final message, got {summary}"
    )
    assert "exercise" in summary, "session_summary missing 'exercise' key"
    assert "rep_results" in summary, "session_summary missing 'rep_results' key"
    assert summary["exercise"] == "squat"
    assert isinstance(summary["rep_results"], list)


def test_ws_rep_result_schema(client):
    """Sending enough frames causes the trigger to fire and emits a rep_result.

    The LiveWindowTrigger fires when:
      buffer_len >= LIVE_REP_WINDOW_FRAMES=32  AND
      frames_since_last_trigger >= LIVE_MIN_GAP_FRAMES=45

    Sending LIVE_MIN_GAP_FRAMES + 2 = 47 frames (each with a valid JPEG) guarantees
    at least one trigger fire.  The mocked classify_clip_async returns _VALID_D05
    (KIE detected=True) so the rep_result message will contain errors.

    Starlette TestClient WS is synchronous — receive_json() BLOCKS until a message
    arrives.  Because conditional-send means most frames produce NO server reply, we
    send ALL frames first (no receive in the loop), then collect the one rep_result
    that the server queued by calling receive_json() exactly once before end_session.
    """
    n_frames_to_send = LIVE_MIN_GAP_FRAMES + 2  # 47 frames — enough to guarantee one fire
    jpeg_b64 = base64.b64encode(make_fake_jpeg()).decode()

    with client.websocket_connect("/ws/form-session") as ws:
        ws.send_json({"type": "start_session", "exercise_hint": "squat"})
        started = ws.receive_json()
        assert started["type"] == "session_started"

        # Send all frames without receiving — server queues the rep_result internally.
        # The trigger fires somewhere around frame 45; we do NOT block per-frame.
        for i in range(n_frames_to_send):
            ws.send_json({"type": "frame", "data": jpeg_b64, "timestamp_ms": i * 33})

        # The server has queued exactly one rep_result by now — receive it.
        rep = ws.receive_json()

        # Then end the session and collect the summary.
        ws.send_json({"type": "end_session"})
        summary = ws.receive_json()

    assert rep["type"] == "rep_result", (
        f"Expected rep_result after {n_frames_to_send} frames, got {rep}"
    )
    assert rep["rep_number"] >= 1, "rep_number must be >= 1"
    assert "errors" in rep, "rep_result missing 'errors' key"
    for err in rep["errors"]:
        assert err["type"] in {"KIE", "KFE"}, (
            f"Unexpected error type in rep_result: {err['type']}"
        )
    # Also verify the summary arrived cleanly.
    assert summary["type"] == "session_summary"
    assert summary["total_reps"] >= 1


def test_ws_malformed_frame(client):
    """A frame with invalid JPEG data (T-05-10 None-guard) does not crash the session.

    After the bad frame, end_session must still return a valid session_summary —
    proving the session stays alive and the None-guard drops the frame gracefully.
    """
    # base64-encode a string that is NOT valid JPEG bytes.
    bad_b64 = base64.b64encode(b"not-a-jpeg").decode()

    with client.websocket_connect("/ws/form-session") as ws:
        ws.send_json({"type": "start_session", "exercise_hint": "squat"})
        started = ws.receive_json()
        assert started["type"] == "session_started"

        # Send the malformed frame — server must drop it silently (no crash, no error reply).
        ws.send_json({"type": "frame", "data": bad_b64, "timestamp_ms": 0})
        # No reply expected (conditional-send; None returned by add_frame).
        # Send end_session immediately after to verify the session is still alive.
        ws.send_json({"type": "end_session"})
        summary = ws.receive_json()

    assert summary["type"] == "session_summary", (
        f"Expected session_summary after malformed frame, got {summary}"
    )
    assert "rep_results" in summary, "session_summary missing rep_results after malformed frame"
