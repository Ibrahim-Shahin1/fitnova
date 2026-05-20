"""Resumable Colab harness primitives — Drive mount, idempotent zip-stage, RNG capture/restore, atomic checkpoint save/load/prune.

Implementation lands across Tasks 7, 9, 10 (Phase 2):

- Task 7 (slice 1): `mount_drive`, `stage_squat_videos` — Drive FUSE mount and per-session copy of `Squat/Labeled_Dataset/videos.zip` → `/content/squat_videos/`. Idempotent: re-running a cell does not re-extract.
- Task 9 (slice 2): `capture_rng_state`, `restore_rng_state` — full 4-RNG capture (Python `random`, NumPy, torch CPU, torch CUDA). Sets `cudnn.deterministic=True`, `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True)`. Determinism preconditions in PLAN.md `<determinism_checklist>` apply.
- Task 10 (slice 3): `hash_config`, `atomic_save_checkpoint`, `load_latest_checkpoint`, `prune_checkpoints`, `CheckpointConfigMismatchError`. Atomic write contract (D11): torch.save to tmp → reload-verify → os.replace → write `latest.txt` LAST. Drive FUSE rename is not atomic; the contract defends against partial checkpoints.

See: `.planning/phases/02-squat-data-pipeline-colab-harness/02-01-PLAN.md` — Tasks 7/9/10 / D11 / D13 / interfaces block / risk register R3.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("aqa.phase02")

# Placeholder — implementation lands in Tasks 7, 9, 10.
