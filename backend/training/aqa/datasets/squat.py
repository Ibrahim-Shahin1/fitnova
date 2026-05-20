"""SquatKIEKFEDataset + build_loaders factory — joint multi-label dataset over the official Squat splits.

Implementation lands in Task 6 (Phase 2). Yields `(video[B, 3, 32, 112, 112] float32, labels[B, 2] float32)` from official train/val/test splits. Computes and exposes `pos_weight` at construction (Tensor of length 2 for KIE/KFE), to be consumed by Phase 3's `BCEWithLogitsLoss`. Phase 1 measured KIE = 14.29% positive (severe imbalance) and KFE = 68.33% positive (mild reverse imbalance).

`build_loaders(...)` returns `{"train": DataLoader, "val": DataLoader, "test": DataLoader}`. Phase 2 tiny run uses `num_workers=0` for the bitwise-determinism guarantee (D14); Phase 3 may bump it with a `worker_init_fn`.

See: `.planning/phases/02-squat-data-pipeline-colab-harness/02-01-PLAN.md` — Task 6 / D9 / D14 / interfaces block.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("aqa.phase02")

# Placeholder — implementation lands in Task 6.
