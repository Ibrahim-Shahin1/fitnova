"""Window-bounded video decode + 32-frame uniform sampler + spatial preprocessing for the Squat KIE/KFE pipeline.

Implementation lands in Task 5 (Phase 2). Wraps `torchvision.io.read_video` behind `decode_clip(path, indices)` so a future swap to TorchCodec (when Colab's torchvision passes 0.22) is a one-line change. Uses `start_pts` / `end_pts` to bound the decode window against OOM on long-tail clips (Phase 1 measured up to 404 frames at width 480 × height up to 600).

Spatial pipeline: short-side resize to 128 → 112² random crop (train) or center crop (val/test) → Kinetics-400 normalization (mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]). NO horizontal flip — paper line 576 augmentation list omits it; CVCSPC code has it commented out.

See: `.planning/phases/02-squat-data-pipeline-colab-harness/02-01-PLAN.md` — Task 5 / D1 / D5 / D6 / interfaces block.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("aqa.phase02")

# Placeholder — implementation lands in Task 5.
