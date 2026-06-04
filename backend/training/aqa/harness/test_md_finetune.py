"""Unit test for md_finetune.build_finetune_model.

Verifies the fine-tune model loads the MD backbone (NOT Kinetics) and attaches a
fresh ``Dropout(0.2) + Linear(512, 2)`` head.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the whole file on torch-less environments (Windows orchestrator shell).
torch = pytest.importorskip("torch")

from backend.training.aqa.harness.md_finetune import FinetuneConfig, build_finetune_model


def _arr(*values: int | float) -> np.ndarray:
    """Build a numpy array from a positional spread (terser than np.array(list))."""
    return np.array(values)


# ───────────────────────────── fine-tune model build (slow) ─────────────────────

@pytest.mark.slow
def test_finetune_model_build(tmp_path) -> None:
    pytest.importorskip("torchvision")
    from backend.training.aqa.harness.md_pretrain import build_md_model

    # Round-trip: save an MD backbone state_dict, then load it into a fresh fine-tune model.
    backbone, _ = build_md_model()  # Kinetics backbone, fc=Identity (weights cached)
    ckpt_path = tmp_path / "backbone.pt"
    torch.save({"backbone_state_dict": backbone.state_dict()}, ckpt_path)

    model = build_finetune_model(str(ckpt_path))
    assert isinstance(model.fc, torch.nn.Sequential), type(model.fc)
    assert isinstance(model.fc[0], torch.nn.Dropout)
    assert isinstance(model.fc[1], torch.nn.Linear)
    assert model.fc[1].out_features == 2

    x = torch.zeros(2, 3, 32, 112, 112)  # 32-frame labeled clip (NOT the 16-frame half-cycle)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 2), y.shape


# ───────────────────────────── overfit monitor (pure logic, fast) ──────────────

def test_d6_overfit_monitor() -> None:
    """Monitor fires iff (epoch < 10 AND val/train BCE ratio > 10.0). Pure logic, no GPU.

    Overfit = val loss far ABOVE train (an earlier run hit ~32x), so the abort signal is val/train > 10.
    """
    from backend.training.aqa.harness.md_finetune import _d6_overfit_abort

    # val/train ~20 (>10) before epoch 10 → ABORT (overfit: val loss >> train).
    assert _d6_overfit_abort(epoch=8, train_loss_mean=0.5, val_loss_mean=10.0) is True
    # Same blow-up but AT/after epoch 10 → no abort (monitor only guards the early epochs).
    assert _d6_overfit_abort(epoch=10, train_loss_mean=0.5, val_loss_mean=10.0) is False
    assert _d6_overfit_abort(epoch=20, train_loss_mean=0.5, val_loss_mean=10.0) is False
    # Healthy (val ≈ train) early → no abort.
    assert _d6_overfit_abort(epoch=3, train_loss_mean=0.50, val_loss_mean=0.50) is False
    # INVERTED case (train >> val) must NOT abort — that's under-fitting, not overfit (guards the bug).
    assert _d6_overfit_abort(epoch=5, train_loss_mean=10.0, val_loss_mean=0.5) is False
    # Boundary: val/train == 10.0 exactly (with the +1e-9 eps, just under) → no abort.
    assert _d6_overfit_abort(epoch=5, train_loss_mean=0.5, val_loss_mean=5.0) is False
    # Just over → abort.
    assert _d6_overfit_abort(epoch=5, train_loss_mean=0.5, val_loss_mean=5.01) is True
