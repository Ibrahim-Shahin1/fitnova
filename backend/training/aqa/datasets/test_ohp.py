"""Unit tests for OHP dataset, trajectory loader, and splits."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# ──────────────────────────────────────────────────────────────────────────────
# index_ohp counts test (reads local archive; marked slow)
# ──────────────────────────────────────────────────────────────────────────────

# The -3-001 folder is the parent of "Fitness-AQA_dataset_release/OHP/..."
# index_ohp(split, drive_root=...) prepends the release subdir internally.
_LOCAL_OHP_3001 = Path(
    "Fitness-AQA/Fitness-AQA_dataset_release-20260518T073812Z-3-001"
)
_LOCAL_ARCHIVE_EXISTS = _LOCAL_OHP_3001.exists()


@pytest.mark.skipif(
    not _LOCAL_ARCHIVE_EXISTS,
    reason="Local OHP archive not found — set _LOCAL_OHP_3001 to the -3-001 folder",
)
def test_ohp_splits_count() -> None:
    """index_ohp returns 1582/339/339 against the local archive."""
    from backend.training.aqa.datasets.splits import OHPClipRecord, index_ohp

    drive_root = str(_LOCAL_OHP_3001)
    videos_root = "/dev/null"  # video paths are constructed but files not opened

    for split_name, expected in [("train", 1582), ("val", 339), ("test", 339)]:
        recs = index_ohp(split_name, drive_root=drive_root, videos_root=videos_root)
        assert len(recs) == expected, (
            f"index_ohp('{split_name}'): expected {expected} records, got {len(recs)}"
        )
        for rec in recs:
            assert isinstance(rec, OHPClipRecord), f"record is not OHPClipRecord: {type(rec)}"
            assert rec.label_elbows in (0, 1), f"label_elbows not in {{0,1}}: {rec.label_elbows}"
            assert rec.label_knees in (0, 1), f"label_knees not in {{0,1}}: {rec.label_knees}"

    # Train class balance: Elbows+ ~407 (25.7%), Knees+ ~541 (34.2%)
    train_recs = index_ohp("train", drive_root=drive_root, videos_root=videos_root)
    elbows_pos = sum(r.label_elbows for r in train_recs)
    knees_pos  = sum(r.label_knees  for r in train_recs)
    assert elbows_pos == 407, f"train Elbows+ expected 407, got {elbows_pos}"
    assert knees_pos == 541,  f"train Knees+ expected 541, got {knees_pos}"


# ──────────────────────────────────────────────────────────────────────────────
# pos_weight formula (pure, no I/O)
# ──────────────────────────────────────────────────────────────────────────────


def test_ohp_pos_weight() -> None:
    """_compute_pos_weight returns the hand-computed [3.0, 1.0] on synthetic records."""
    from backend.training.aqa.datasets.splits import OHPClipRecord
    from backend.training.aqa.datasets.ohp import _compute_pos_weight

    # 4 records: 1 Elbows+, 2 Knees+ => w_elbows=(4-1)/1=3.0, w_knees=(4-2)/2=1.0
    records = [
        OHPClipRecord(clip_id="a", video_path="a.mp4", label_elbows=1, label_knees=0),
        OHPClipRecord(clip_id="b", video_path="b.mp4", label_elbows=0, label_knees=1),
        OHPClipRecord(clip_id="c", video_path="c.mp4", label_elbows=0, label_knees=1),
        OHPClipRecord(clip_id="d", video_path="d.mp4", label_elbows=0, label_knees=0),
    ]
    pw = _compute_pos_weight(records)
    assert pw.shape == (2,), f"pos_weight shape: {pw.shape}"
    assert float(pw[0]) == pytest.approx(3.0, rel=1e-6), f"w_elbows: {pw[0]}"
    assert float(pw[1]) == pytest.approx(1.0, rel=1e-6), f"w_knees: {pw[1]}"


# ──────────────────────────────────────────────────────────────────────────────
# shape test (uses synthetic mp4 via torchvision write_video)
# ──────────────────────────────────────────────────────────────────────────────


def _write_synthetic_mp4(path: Path, n_frames: int = 40) -> None:
    """Write a tiny synthetic mp4 (40 frames, 64x64) to path using torchvision."""
    try:
        import torchvision.io
    except ImportError:
        pytest.skip("torchvision not available")
    # [T, H, W, C] uint8 for write_video
    frames = torch.randint(0, 256, (n_frames, 64, 64, 3), dtype=torch.uint8)
    try:
        torchvision.io.write_video(str(path), frames, fps=30)
    except Exception as exc:
        pytest.skip(f"write_video failed (codec unavailable?): {exc}")


@pytest.mark.slow
def test_ohp_dataset_shape(tmp_path: Path) -> None:
    """OHPElbowsKneesDataset __getitem__ returns (clip[3,8,112,112], labels[2])."""
    from backend.training.aqa.datasets.ohp import OHPElbowsKneesDataset

    # Build minimal directory tree: drive_root with OHP split + label JSONs, videos_root with mp4s.
    import json

    clip_ids = ["00001_0", "00001_1"]

    # videos_root: write two synthetic mp4s
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    for cid in clip_ids:
        _write_synthetic_mp4(videos_dir / f"{cid}.mp4")

    # drive_root tree
    splits_dir = tmp_path / "Fitness-AQA_dataset_release/OHP/Labeled_Dataset/Splits"
    labels_dir = tmp_path / "Fitness-AQA_dataset_release/OHP/Labeled_Dataset/Labels"
    splits_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for split_name in ("train", "val", "test"):
        (splits_dir / f"{split_name}_keys.json").write_text(json.dumps(clip_ids))

    # label dicts: clip 0 = Elbows+, clip 1 = Knees+
    elbows = {clip_ids[0]: [[0, 1]], clip_ids[1]: []}
    knees  = {clip_ids[0]: [],       clip_ids[1]: [[0, 1]]}
    (labels_dir / "error_elbows.json").write_text(json.dumps(elbows))
    (labels_dir / "error_knees.json").write_text(json.dumps(knees))

    ds = OHPElbowsKneesDataset(
        split="train",
        drive_root=str(tmp_path),
        videos_root=str(videos_dir),
        num_frames=8,
        crop_size=112,
        train_aug=False,  # deterministic for shape check
    )
    assert len(ds) == 2
    clip, labels = ds[0]
    assert clip.shape == (3, 8, 112, 112), f"clip shape: {clip.shape}"
    assert clip.dtype == torch.float32
    assert labels.shape == (2,), f"labels shape: {labels.shape}"
    assert labels.dtype == torch.float32
    assert ds.pos_weight.shape == (2,)


# ──────────────────────────────────────────────────────────────────────────────
# dataset_cls injection seam (signature-level, no GPU/data required)
# ──────────────────────────────────────────────────────────────────────────────


def test_dataset_cls_injection() -> None:
    """_build_dataloaders + run_md_finetune_epoch have a backward-compatible dataset_cls kwarg."""
    from backend.training.aqa.datasets.squat import SquatKIEKFEDataset
    from backend.training.aqa.harness.supervised_train import _build_dataloaders
    from backend.training.aqa.harness.md_finetune import run_md_finetune_epoch

    # _build_dataloaders: dataset_cls must be keyword-only with default SquatKIEKFEDataset
    sig_bd = inspect.signature(_build_dataloaders)
    assert "dataset_cls" in sig_bd.parameters, "_build_dataloaders missing dataset_cls kwarg"
    param_bd = sig_bd.parameters["dataset_cls"]
    assert param_bd.kind == inspect.Parameter.KEYWORD_ONLY, (
        "_build_dataloaders.dataset_cls must be keyword-only"
    )
    assert param_bd.default is SquatKIEKFEDataset, (
        f"_build_dataloaders.dataset_cls default is {param_bd.default}, expected SquatKIEKFEDataset"
    )

    # run_md_finetune_epoch: same contract
    sig_mf = inspect.signature(run_md_finetune_epoch)
    assert "dataset_cls" in sig_mf.parameters, "run_md_finetune_epoch missing dataset_cls kwarg"
    param_mf = sig_mf.parameters["dataset_cls"]
    assert param_mf.kind == inspect.Parameter.KEYWORD_ONLY, (
        "run_md_finetune_epoch.dataset_cls must be keyword-only"
    )
    assert param_mf.default is SquatKIEKFEDataset, (
        f"run_md_finetune_epoch.dataset_cls default is {param_mf.default}, expected SquatKIEKFEDataset"
    )
