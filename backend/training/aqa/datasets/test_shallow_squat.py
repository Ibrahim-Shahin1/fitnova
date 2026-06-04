"""Unit tests for ShallowSquatDataset — shape, single-head pos_weight/label,
ImageNet norm, build_loaders, and the official-split reconciliation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# The -3-001 folder holds Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset.
_LOCAL_SQUAT_3001 = Path(
    "Fitness-AQA/Fitness-AQA_dataset_release-20260518T073812Z-3-001"
)
_SHALLOW_DIR = (
    _LOCAL_SQUAT_3001
    / "Fitness-AQA_dataset_release/Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset"
)
_LOCAL_ARCHIVE_EXISTS = _SHALLOW_DIR.exists()


def _make_synthetic_dataset(tmp_path: Path, ids: list[str], labels: dict[str, int]):
    """Write synthetic JPEG crops + flat label JSON + split JSONs under tmp_path."""
    from PIL import Image

    crops_dir = tmp_path / "crops_unaligned"
    crops_dir.mkdir()
    for id_ in ids:
        arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(arr).save(crops_dir / f"{id_}.jpg")
    (tmp_path / "labels_shallow_depth.json").write_text(json.dumps(labels))
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    for split_name in ("train", "val", "test"):
        (splits_dir / f"{split_name}_ids.json").write_text(json.dumps(ids))
    return (
        str(crops_dir),
        str(tmp_path / "labels_shallow_depth.json"),
        str(splits_dir),
    )


def test_pos_weight() -> None:
    """_compute_pos_weight returns a length-1 tensor (single head) ≈ 1.0 on 2/4-positive records."""
    from backend.training.aqa.datasets.shallow_squat import _compute_pos_weight

    records = [("a", 1), ("b", 0), ("c", 1), ("d", 0)]
    pw = _compute_pos_weight(records)
    assert tuple(pw.shape) == (1,), f"pos_weight shape {tuple(pw.shape)}, expected (1,) not (2,)"
    assert float(pw[0]) == pytest.approx(1.0, rel=1e-6), f"w={float(pw[0])}"


def test_getitem_shape(tmp_path: Path) -> None:
    """__getitem__ returns (tensor[3,224,224] float32, SCALAR float32 label in {0,1})."""
    from backend.training.aqa.datasets.shallow_squat import ShallowSquatDataset

    ids = ["37803_2_44", "37803_2_45"]
    labels = {id_: i % 2 for i, id_ in enumerate(ids)}
    images_root, labels_path, splits_root = _make_synthetic_dataset(tmp_path, ids, labels)

    ds = ShallowSquatDataset(
        split="train",
        images_root=images_root,
        labels_path=labels_path,
        splits_root=splits_root,
        train_aug=False,
    )
    img, label = ds[0]
    assert img.shape == (3, 224, 224), f"img shape {tuple(img.shape)}"
    assert img.dtype == torch.float32
    assert label.ndim == 0, f"label must be scalar (single head), got ndim={label.ndim}"
    assert label.dtype == torch.float32
    assert float(label) in (0.0, 1.0)


def test_imagenet_norm(tmp_path: Path) -> None:
    """The val transform produces ImageNet-range pixels ([-3, 3]), not Kinetics range."""
    from backend.training.aqa.datasets.shallow_squat import ShallowSquatDataset

    ids = ["v_0_0", "v_0_1"]
    labels = {id_: 0 for id_ in ids}
    images_root, labels_path, splits_root = _make_synthetic_dataset(tmp_path, ids, labels)

    ds = ShallowSquatDataset(
        split="val",
        images_root=images_root,
        labels_path=labels_path,
        splits_root=splits_root,
        train_aug=False,
    )
    img, _ = ds[0]
    assert float(img.min()) > -3.0, f"min {float(img.min())} below ImageNet range"
    assert float(img.max()) < 3.0, f"max {float(img.max())} above ImageNet range"


def test_build_loaders(tmp_path: Path) -> None:
    """build_loaders returns {train,val,test} DataLoaders; the label batch is (B,), not (B,2)."""
    from torch.utils.data import DataLoader

    from backend.training.aqa.datasets.shallow_squat import build_loaders

    ids = ["v_0_0", "v_0_1", "v_0_2", "v_0_3"]
    labels = {id_: i % 2 for i, id_ in enumerate(ids)}
    images_root, labels_path, splits_root = _make_synthetic_dataset(tmp_path, ids, labels)

    loaders = build_loaders(
        images_root=images_root,
        labels_path=labels_path,
        splits_root=splits_root,
        batch_size=2,
        num_workers=0,
    )
    assert set(loaders.keys()) == {"train", "val", "test"}
    for name, loader in loaders.items():
        assert isinstance(loader, DataLoader), f"{name} is not a DataLoader"

    _imgs, batch_labels = next(iter(loaders["val"]))
    assert batch_labels.shape == (2,), f"label batch shape {tuple(batch_labels.shape)}, expected (B,)"


@pytest.mark.skipif(not _LOCAL_ARCHIVE_EXISTS, reason="Local Shallow-Squat archive not found")
def test_split_sizes() -> None:
    """ShallowSquatDataset reconciles to the official 2542/529/540 split sizes."""
    from backend.training.aqa.datasets.shallow_squat import ShallowSquatDataset

    images_root = str(_SHALLOW_DIR / "crops_unaligned")
    labels_path = str(_SHALLOW_DIR / "labels_shallow_depth.json")
    splits_root = str(_SHALLOW_DIR / "splits")
    for split_name, expected in [("train", 2542), ("val", 529), ("test", 540)]:
        ds = ShallowSquatDataset(
            split=split_name,
            images_root=images_root,
            labels_path=labels_path,
            splits_root=splits_root,
            train_aug=False,
        )
        assert len(ds) == expected, f"{split_name}: expected {expected}, got {len(ds)}"
    # pos_weight derives from the train split regardless of which split this instance is.
    train_ids = json.loads((_SHALLOW_DIR / "splits" / "train_ids.json").read_text())
    all_labels = json.loads((_SHALLOW_DIR / "labels_shallow_depth.json").read_text())
    train_pos = sum(int(all_labels[i]) for i in train_ids if i in all_labels)
    expected_w = (len(train_ids) - train_pos) / max(train_pos, 1)
    assert float(ds.pos_weight[0]) == pytest.approx(expected_w, rel=1e-6)


def test_image_trainer_seams() -> None:
    """_build_dataloaders + run_image_epoch expose the dataset_cls / model_builder / checkpoint_phase seams."""
    import inspect

    from backend.training.aqa.datasets.shallow_squat import ShallowSquatDataset
    from backend.training.aqa.harness.image_supervised_train import (
        _build_dataloaders,
        build_resnet18,
        run_image_epoch,
    )

    sig_bd = inspect.signature(_build_dataloaders)
    assert sig_bd.parameters["dataset_cls"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig_bd.parameters["dataset_cls"].default is ShallowSquatDataset

    sig_ri = inspect.signature(run_image_epoch)
    assert sig_ri.parameters["dataset_cls"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig_ri.parameters["dataset_cls"].default is ShallowSquatDataset
    assert sig_ri.parameters["model_builder"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig_ri.parameters["model_builder"].default is build_resnet18
    assert sig_ri.parameters["checkpoint_phase"].default == "phase07"


def test_colab_additions_present() -> None:
    """The colab staging + frame-extraction additions exist with the right signatures."""
    import inspect

    from backend.training.aqa.harness import colab

    assert callable(colab.stage_shallow_squat_images)
    assert callable(colab.extract_frames_for_ssl)

    sig_s = inspect.signature(colab.stage_shallow_squat_images)
    assert "drive_root_3001" in sig_s.parameters
    assert sig_s.parameters["local_root"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig_s.parameters["expect_count"].kind == inspect.Parameter.KEYWORD_ONLY

    sig_e = inspect.signature(colab.extract_frames_for_ssl)
    assert "videos_root" in sig_e.parameters
    assert "frames_root" in sig_e.parameters
    assert sig_e.parameters["skip_existing"].kind == inspect.Parameter.KEYWORD_ONLY
