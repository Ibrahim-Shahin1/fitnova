"""Unit test for md_finetune.build_finetune_model (SQUAT-05-a).

Verifies the fine-tune model loads the MD backbone (NOT Kinetics) and attaches a
fresh ``Dropout(0.2) + Linear(512, 2)`` head. Body lands in Wave 0 (Task 9).
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


# ───────────────────────────── SQUAT-05-a: fine-tune model build (slow) ─────────────────────

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
