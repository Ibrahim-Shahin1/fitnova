"""ShallowSquatFormService unit tests.

Covers SRV-03 — the 3-seed ResNet-18 CVCSPC image classifier (single crop:
Resize(256)->CenterCrop(224)->ImageNet norm), single-logit head, inline scalar
mean-of-sigmoids, threshold 0.395, raw DEPTH output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from backend.services.shallow_squat_form_service import ShallowSquatFormService

# ─────────────────────────────────────────────────────────────────────────────
# Path constants (mirrors test_shallow_squat.py:15-21)
# ─────────────────────────────────────────────────────────────────────────────

_LOCAL_SQUAT_3001 = Path(
    "Fitness-AQA/Fitness-AQA_dataset_release-20260518T073812Z-3-001"
)
_SHALLOW_DIR = (
    _LOCAL_SQUAT_3001
    / "Fitness-AQA_dataset_release/Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset"
)
_CROPS_DIR = _SHALLOW_DIR / "crops_unaligned"
_SPLITS_DIR = _SHALLOW_DIR / "splits"
_STAGED_MODEL_DIR = "backend/models/form_model_shallow_cvcspc"

_WEIGHTS_PRESENT = (
    Path(_STAGED_MODEL_DIR) / "seed42" / "best.pt"
).exists()

_CROPS_PRESENT = _CROPS_DIR.exists() and any(_CROPS_DIR.glob("*.jpg"))


# ─────────────────────────────────────────────────────────────────────────────
# test_shallow_service_loads
# ─────────────────────────────────────────────────────────────────────────────


def test_shallow_service_loads() -> None:
    """ShallowSquatFormService constructs from the staged dir, exposes bool model_ready and threshold 0.395.

    Weight presence is not asserted — both True and False are valid (CI may lack weights).
    """
    svc = ShallowSquatFormService(model_dir=_STAGED_MODEL_DIR)
    assert isinstance(svc.model_ready, bool)
    assert svc.threshold == pytest.approx(0.395)


# ─────────────────────────────────────────────────────────────────────────────
# test_shallow_neutral_response_schema
# ─────────────────────────────────────────────────────────────────────────────


def test_shallow_neutral_response_schema(tmp_path) -> None:
    """With no weights, model_ready is False; classify_image returns a schema-valid neutral DEPTH dict.

    Uses tmp_path so real staged weights are never loaded.
    """
    svc = ShallowSquatFormService(model_dir=str(tmp_path / "no_weights"))
    assert svc.model_ready is False

    result = svc.classify_image("x.jpg")

    assert result.get("model_not_loaded") is True
    errors = result.get("errors", [])
    assert len(errors) == 1, f"Expected 1 error (DEPTH only), got {len(errors)}"

    err = errors[0]
    assert err["type"] == "DEPTH"
    assert err["detected"] is False
    assert err["confidence"] == pytest.approx(0.0)
    assert "threshold" in err, "threshold key missing from neutral DEPTH error"
    assert err["threshold"] == pytest.approx(0.395)
    assert "severity_word" not in err, "severity_word must not appear in DEPTH error"
    assert err["intervals"] == []


# ─────────────────────────────────────────────────────────────────────────────
# test_shallow_classify_real_crop
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.skipif(
    not _WEIGHTS_PRESENT or not _CROPS_PRESENT,
    reason="Staged Shallow weights or extracted crops not present",
)
def test_shallow_classify_real_crop() -> None:
    """classify_image on an extracted test-split crop returns one DEPTH error and is deterministic."""
    test_ids = json.loads((_SPLITS_DIR / "test_ids.json").read_text(encoding="utf-8"))
    first_id = test_ids[0]
    crop_path = str(_CROPS_DIR / f"{first_id}.jpg")

    svc = ShallowSquatFormService(model_dir=_STAGED_MODEL_DIR)
    assert svc.model_ready is True, "Weights present but model_ready is False"

    result = svc.classify_image(crop_path)

    errors = result.get("errors", [])
    assert len(errors) == 1, f"Expected 1 DEPTH error, got {len(errors)}"
    err = errors[0]
    assert err["type"] == "DEPTH"
    assert 0.0 <= err["confidence"] <= 1.0, f"confidence out of range: {err['confidence']}"
    assert err["threshold"] == pytest.approx(0.395)
    assert "severity_word" not in err

    # Determinism: eval-mode + center crop must give identical results on repeat
    result2 = svc.classify_image(crop_path)
    assert result2["errors"][0]["confidence"] == err["confidence"], (
        "classify_image is not deterministic — train-time random crop may be in use"
    )
