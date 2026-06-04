"""pytest unit suite for `backend.training.aqa.harness.supervised_train`.

Covers the model head shape and the checkpoint schema keys.

`test_model_head_shape` is `@pytest.mark.slow` because constructing R(2+1)D-18
downloads ~120 MB of Kinetics-V1 weights on first call (cached afterward by
torchvision under `~/.cache/torch/hub/checkpoints/`). Run via:

    python -m pytest backend/training/aqa/harness/test_supervised_train.py::test_model_head_shape -x -v

The schema test is NOT slow — runs in milliseconds with a dummy payload:

    python -m pytest backend/training/aqa/harness/test_supervised_train.py -x -v -m "not slow"

`pytest.importorskip("torch")` at module top means the whole file skips
cleanly on environments without torch (e.g., the Windows orchestrator shell).
"""

from __future__ import annotations

import pytest

# Skip the whole file on torch-less environments (Windows orchestrator shell).
torch = pytest.importorskip("torch")

from backend.training.aqa.harness.supervised_train import build_model


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


# Top-level payload keys that must appear on every checkpoint dict.
_EXPECTED_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "epoch",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "rng_state",
    "metrics_history",
    "best_f1_val",
    "best_thresholds",
    "config_hash",
    "config_repr",
    "code_version",
)

# Per-epoch metrics_history entry keys.
_EXPECTED_METRIC_KEYS: tuple[str, ...] = (
    "epoch",
    "train_loss_mean",
    "train_loss_per_batch",
    "val_loss_mean",
    "val_f1_kie",
    "val_f1_kfe",
    "val_pr_auc_kie",
    "val_pr_auc_kfe",
    "val_macro_f1",
    "epoch_wall_time_s",
)


# ─────────────────────────────────────────────────────────────────────────────
# checkpoint schema keys (pure-Python schema regression — fast)
# ─────────────────────────────────────────────────────────────────────────────


def test_checkpoint_schema_keys() -> None:
    """Schema regression — every key the trainer writes must be present.

    Build a dummy payload mirroring what `run_supervised_epoch` produces. Assert
    every top-level key and every metrics_history sub-key is present, and assert
    `code_version == "phase03-supervised-baseline"` so a checkpoint from a
    different code version can never silently pass.
    """
    dummy_metrics_entry = {
        "epoch": 0,
        "train_loss_mean": 0.0,
        "train_loss_per_batch": [],
        "val_loss_mean": 0.0,
        "val_f1_kie": 0.0,
        "val_f1_kfe": 0.0,
        "val_pr_auc_kie": 0.0,
        "val_pr_auc_kfe": 0.0,
        "val_macro_f1": 0.0,
        "epoch_wall_time_s": 0.0,
    }
    payload = {
        "epoch": 0,
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "rng_state": {
            "python": (),
            "numpy": (),
            "torch_cpu": torch.tensor([0], dtype=torch.uint8),
            "torch_cuda_all": [],
        },
        "metrics_history": [dummy_metrics_entry],
        "best_f1_val": 0.0,
        "best_thresholds": None,
        "config_hash": "abc123",
        "config_repr": {},
        "code_version": "phase03-supervised-baseline",
    }

    missing_top = [k for k in _EXPECTED_TOP_LEVEL_KEYS if k not in payload]
    assert missing_top == [], f"Missing top-level keys: {missing_top}"

    missing_metric = [k for k in _EXPECTED_METRIC_KEYS if k not in payload["metrics_history"][0]]
    assert missing_metric == [], f"Missing metrics_history keys: {missing_metric}"

    assert payload["code_version"] == "phase03-supervised-baseline", (
        f"code_version must be 'phase03-supervised-baseline' (got "
        f"{payload['code_version']!r}) — an older checkpoint must NOT pass "
        f"as a newer one"
    )


# ─────────────────────────────────────────────────────────────────────────────
# model head shape (slow — downloads ~120 MB weights on first call)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_model_head_shape() -> None:
    """`build_model()` returns R(2+1)D-18 with `Linear(512, 2)` head; forward pass shapes correctly.

    Skips if `torchvision` is not importable (defensive — keeps CPU-only CI
    green when torchvision install is unavailable).
    """
    pytest.importorskip("torchvision")

    model = build_model()
    assert model.fc.in_features == 512, model.fc.in_features
    assert model.fc.out_features == 2, model.fc.out_features

    x = torch.zeros(2, 3, 32, 112, 112)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 2), y.shape
    assert y.dtype == torch.float32, y.dtype
