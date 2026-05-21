"""Behavioral test for atomic_save_checkpoint's update_latest kwarg (T-04-03 / Pitfall 4).

Proves the D7 latest.txt-clobber fix: a backbone.pt / best.pt write (update_latest=False)
writes the checkpoint but leaves the epoch-resume pointer untouched, so SSL resume never
jumps to backbone.pt; while the default (update_latest=True) preserves the exact Phase 2/3
behavior. Closes RESEARCH Pitfall 4 before any 12-24h SSL run.
"""

from __future__ import annotations

import pytest

# atomic_save_checkpoint round-trips a torch payload (torch.save + torch.load verify).
torch = pytest.importorskip("torch")

from backend.training.aqa.harness.colab import atomic_save_checkpoint


def _payload() -> dict:
    return {"epoch": 0, "model_state_dict": {}, "config_hash": "abc"}


def test_update_latest_true_writes_pointer(tmp_path) -> None:
    # Default update_latest=True: latest.txt tracks the most recent epoch checkpoint.
    atomic_save_checkpoint(_payload(), str(tmp_path / "epoch_000.pt"))
    latest = tmp_path / "latest.txt"
    assert latest.exists()
    assert latest.read_text(encoding="utf-8").strip() == "epoch_000.pt"

    atomic_save_checkpoint(_payload(), str(tmp_path / "epoch_001.pt"))
    assert latest.read_text(encoding="utf-8").strip() == "epoch_001.pt"


def test_update_latest_false_preserves_pointer(tmp_path) -> None:
    # Establish latest.txt -> epoch_001.pt, then write backbone.pt with update_latest=False.
    atomic_save_checkpoint(_payload(), str(tmp_path / "epoch_000.pt"))
    atomic_save_checkpoint(_payload(), str(tmp_path / "epoch_001.pt"))
    latest = tmp_path / "latest.txt"
    before = latest.read_bytes()

    atomic_save_checkpoint(_payload(), str(tmp_path / "backbone.pt"), update_latest=False)

    assert (tmp_path / "backbone.pt").exists()  # the checkpoint WAS written
    assert latest.read_bytes() == before  # ...but latest.txt is byte-unchanged
    assert latest.read_text(encoding="utf-8").strip() == "epoch_001.pt"  # still the prior epoch
