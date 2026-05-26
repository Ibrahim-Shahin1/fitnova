"""
Tests for the Squat inference service, rep segmenter, and the redefined form endpoints.

Wave 0 scaffold — covers:
  - SquatFormService construction + graceful degradation (API-01, D-02)
  - severity_word bands (D-10)
  - classify_clip determinism (spatial_val center-crop, not spatial_train random crop — D-02 / Pitfall 1)
  - RepSegmenter single-rep fallback (API-02, D-03)
  - LiveWindowTrigger gate arithmetic (D-03)

REST/WebSocket integration tests (test_upload_response_schema, test_ws_session_protocol,
etc.) are added by Plan 03 when app.py is rewritten.  No test in this file connects to
/analyze-form-video or /ws/form-session.
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import pytest

# Skip the entire module cleanly when torch is not installed (CI without torch).
torch = pytest.importorskip("torch")

from backend.services.squat_form_service import SquatFormService, _severity
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
