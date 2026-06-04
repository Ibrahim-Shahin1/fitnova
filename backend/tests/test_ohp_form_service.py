"""OHPFormService unit tests.

Covers the 2-seed R(2+1)D-18 MD-SSL ensemble (Elbows/Knees), validation-tuned
thresholds 0.357 / 0.476, raw output (no severity_word), graceful degradation.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from backend.services.ohp_form_service import OHPFormService


# ─────────────────────────────────────────────────────────────────────────────
# Service load
# ─────────────────────────────────────────────────────────────────────────────


def test_ohp_service_loads() -> None:
    """OHPFormService constructs without raising, exposes bool model_ready and correct thresholds.

    Weights may be present (model_ready True) or absent (model_ready False) — both
    are acceptable here.  The test validates the API surface, not weight presence.
    """
    svc = OHPFormService(model_dir="backend/models/form_model_ohp_md")
    assert isinstance(svc.model_ready, bool)
    assert svc.elbows_threshold == pytest.approx(0.357)
    assert svc.knees_threshold == pytest.approx(0.476)


# ─────────────────────────────────────────────────────────────────────────────
# Neutral response schema (weights absent)
# ─────────────────────────────────────────────────────────────────────────────


def test_ohp_neutral_response_schema(tmp_path) -> None:
    """With no weights present, model_ready is False and classify_clip returns a valid neutral dict.

    Uses tmp_path (an empty/nonexistent dir) so real staged weights are never loaded.
    """
    svc = OHPFormService(model_dir=str(tmp_path / "no_weights"))
    assert svc.model_ready is False

    dummy = np.zeros((32, 3, 120, 120), dtype=np.uint8)
    result = svc.classify_clip(dummy)

    assert result.get("model_not_loaded") is True
    errors = result["errors"]
    assert len(errors) == 2
    assert errors[0]["type"] == "ELBOWS"
    assert errors[1]["type"] == "KNEES"
    for err in errors:
        assert set(err.keys()) >= {"type", "detected", "confidence", "threshold", "intervals"}
        assert err["detected"] is False
        assert err["confidence"] == pytest.approx(0.0)
        assert "severity_word" not in err
        assert "threshold" in err
        assert err["intervals"] == []

    assert errors[0]["threshold"] == pytest.approx(0.357)
    assert errors[1]["threshold"] == pytest.approx(0.476)


# ─────────────────────────────────────────────────────────────────────────────
# Determinism (requires staged weights)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_ohp_classify_determinism() -> None:
    """Repeated classify_clip calls on the same clip return byte-equal confidence values.

    Proves spatial_val (deterministic center-crop) is used, NOT spatial_train (random crop).
    Skipped when staged weights are absent.
    """
    svc = OHPFormService(model_dir="backend/models/form_model_ohp_md")
    if not svc.model_ready:
        pytest.skip("OHP weights not staged — determinism not testable here")

    rng = np.random.default_rng(seed=42)
    frames = rng.integers(0, 256, (32, 3, 120, 160), dtype=np.uint8)  # landscape

    result1 = svc.classify_clip(frames)
    result2 = svc.classify_clip(frames)

    assert result1["errors"][0]["confidence"] == result2["errors"][0]["confidence"], (
        "ELBOWS confidence differs between calls — non-deterministic preprocessing may be used"
    )
    assert result1["errors"][1]["confidence"] == result2["errors"][1]["confidence"], (
        "KNEES confidence differs between calls — non-deterministic preprocessing may be used"
    )
