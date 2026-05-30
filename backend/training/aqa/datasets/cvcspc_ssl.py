"""CVCSPC phase-contrastive SSL dataset over the unlabeled Back-Squat frames + bar trajectories.

Bodies land in Task 3 (the phase-matched cross-rep triplet, _traj2phase, masking aug, and the
traj_nan.json exclusion). This module holds the class + loader signatures so cvcspc_pretrain.py
imports cleanly.
"""

from __future__ import annotations

import logging

from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("aqa.phase07")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ShallowSquatSSLDataset(Dataset):
    def __init__(
        self,
        *,
        frames_root: str,
        trajectories_root: str,
        traj_nan_path: str | None = None,
        ssl_contrastive_phase_gap: float = 30.0,
        mask_prob: float = 0.5,
        mask_amt_lo: float = 0.4,
        mask_amt_hi: float = 0.5,
        seed: int = 42,
    ) -> None:
        raise NotImplementedError("ShallowSquatSSLDataset body lands in Task 3")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError


def build_cvcspc_loader(dataset, config, *, seed: int = 42) -> DataLoader:
    raise NotImplementedError("build_cvcspc_loader body lands in Task 3")
