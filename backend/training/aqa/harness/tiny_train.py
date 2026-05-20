"""Toy 3D-conv model + multi-epoch training loop wired to the Colab harness — drives Phase 2's end-to-end smoke validation.

Implementation lands in Task 11 (Phase 2). The toy model is intentionally minimal — `Conv3d(3, 4, 3) → AdaptiveAvgPool3d(1) → Linear(4, 2)` — its only job is exercising the shape contract (input `[B, 3, 32, 112, 112]` → output `[B, 2]`), forward / loss / backward / optimizer step / atomic checkpoint write / RNG restore / resumed-forward-pass cycle. No learning, no accuracy.

`run_tiny_epoch(resume, max_epochs, ...) -> dict` returns `{epoch, train_loss_per_batch, val_loss_mean, checkpoint_path, config_hash}`. Phase 2 acceptance Task 14: a fresh 2-epoch run and a resumed-from-epoch-0-into-epoch-1 run produce **bitwise-identical** epoch-1 loss trajectories — proves RNG capture/restore is functionally correct across a simulated restart.

See: `.planning/phases/02-squat-data-pipeline-colab-harness/02-01-PLAN.md` — Tasks 11/12/13/14 / D14 / interfaces block.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("aqa.phase02")

# Placeholder — implementation lands in Task 11.
