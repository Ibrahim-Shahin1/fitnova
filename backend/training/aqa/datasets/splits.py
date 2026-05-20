"""Official Squat split loader — reads {train,val,test}_keys.json + error_knees_{inward,forward}.json into ClipRecord lists.

Implementation lands in Task 4 (Phase 2). Phase 1 dataset report verified split sizes 1136/243/244 and full label coverage; this module surfaces those records to the dataset class in `squat.py` and to the notebook for reconciliation.

See: `.planning/phases/02-squat-data-pipeline-colab-harness/02-01-PLAN.md` — Task 4 / D7 / interfaces block.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("aqa.phase02")

# Placeholder — implementation lands in Task 4.
