# Phase 06: Overhead Press — Pattern Map

**Mapped:** 2026-05-27
**Files analyzed:** 10 new/modified files
**Analogs found:** 10 / 10

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `datasets/ohp.py` | dataset + factory | CRUD / batch | `datasets/squat.py` | exact |
| `datasets/ohp_ssl.py` | SSL dataset | batch / transform | `datasets/squat_ssl.py` | exact (with known BBox delta) |
| `datasets/splits.py` (add OHP variant) | data loader / config | CRUD | `datasets/splits.py` (Squat section) | exact (additive) |
| `harness/supervised_train.py` (edit) | trainer / harness | batch | self — `_build_dataloaders` | exact (5-line parameterization) |
| `harness/md_finetune.py` (edit) | trainer / harness | batch | self — `run_md_finetune_epoch` | exact (5-line parameterization) |
| `harness/colab.py` (add staging) | infrastructure / utility | file-I/O | `stage_squat_videos` + `stage_unlabeled_squat_videos` | exact (new analogs) |
| `datasets/test_ohp.py` | test | unit | `harness/test_md_finetune.py` | role-match |
| `notebooks/06_ohp_supervised_baseline.{py,ipynb}` | training notebook | batch / interactive | `notebooks/03_squat_supervised_baseline.py` | exact |
| `notebooks/07_ohp_md_ssl.{py,ipynb}` | training notebook | batch / interactive | `notebooks/04_squat_md_ssl.py` | exact |
| `notebooks/08_ohp_md_finetune.{py,ipynb}` | training notebook | batch / interactive | `notebooks/05_squat_md_finetune.py` | exact |
| `docs/notebooks/05_ohp_eda.{py,ipynb}` through `08_ohp_evaluation.{py,ipynb}` | deliverable notebook | transform / viz | `docs/notebooks/01_squat_eda.py` through `04_squat_evaluation.py` | exact |

---

## Pattern Assignments

---

### `datasets/ohp.py` — NEW (dataset + build_loaders factory)

**Analog:** `backend/training/aqa/datasets/squat.py` (full file, 204 lines)

**Structural delta:** rename `kie/kfe` → `elbows/knees` throughout; swap `splits.index()` → `splits.index_ohp()`; update module docstring numbers (pos_weight ≈ 2.89 Elbows, ≈ 1.92 Knees; co-occurrence note: errors largely independent).

**Imports pattern** (lines 19–36):
```python
from __future__ import annotations

import logging
from typing import Callable, Literal

import torch
import torchvision.io
from torch.utils.data import DataLoader, Dataset

from backend.training.aqa.datasets import splits
from backend.training.aqa.datasets.transforms import (
    decode_clip,
    spatial_train,
    spatial_val,
    uniform_sample_indices,
)

logger = logging.getLogger("aqa.phase06")
```

**`_compute_pos_weight` pattern** (lines 39–56 of squat.py — rename fields, keep formula):
```python
def _compute_pos_weight(train_records: list[splits.OHPClipRecord]) -> torch.Tensor:
    """Compute pos_weight = (N - pos) / max(pos, 1) for Elbows and Knees from train records.

    Returns:
        Float32 tensor of shape [2] — [w_elbows, w_knees]. Phase 6 expected values:
        Elbows ≈ 2.887, Knees ≈ 1.924 (tolerance ±0.05).
    """
    n_total = len(train_records)
    pos_elbows = sum(r.label_elbows for r in train_records)
    pos_knees  = sum(r.label_knees  for r in train_records)
    w_elbows = (n_total - pos_elbows) / max(pos_elbows, 1)
    w_knees  = (n_total - pos_knees)  / max(pos_knees,  1)
    logger.info(
        "pos_weight from train (N=%d): Elbows+ %d -> w=%.4f, Knees+ %d -> w=%.4f",
        n_total, pos_elbows, w_elbows, pos_knees, w_knees,
    )
    return torch.tensor([w_elbows, w_knees], dtype=torch.float32)
```

**Dataset class pattern** (lines 59–156 of squat.py — copy verbatim, rename):
```python
class OHPElbowsKneesDataset(Dataset):
    """Joint multi-label OHP Elbows/Knees dataset over the official splits.
    [same docstring structure as SquatKIEKFEDataset]
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        *,
        drive_root: str,
        videos_root: str,
        num_frames: int = 32,
        crop_size: int = 112,
        train_aug: bool = True,
        train_jitter_frames: int = 2,
        seed: int = 42,
    ) -> None:
        # identical body; swap splits.index → splits.index_ohp
        self.records: list[splits.OHPClipRecord] = splits.index_ohp(
            split, drive_root=drive_root, videos_root=videos_root,
        )
        # pos_weight always from TRAIN (same contract as Squat)
        if split == "train":
            train_records = self.records
        else:
            train_records = splits.index_ohp(
                "train", drive_root=drive_root, videos_root=videos_root,
            )
        self.pos_weight: torch.Tensor = _compute_pos_weight(train_records)
        self._spatial_fn: Callable[..., torch.Tensor] = (
            spatial_train if train_aug else spatial_val
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        # [identical decode + spatial pipeline as SquatKIEKFEDataset.__getitem__]
        label = torch.tensor([rec.label_elbows, rec.label_knees], dtype=torch.float32)
        return clip, label
```

**`build_loaders` pattern** (lines 159–203 of squat.py):
The `build_loaders` function body is a direct copy with `SquatKIEKFEDataset` renamed to `OHPElbowsKneesDataset`. The Phase 2 `num_workers=0` clamp comment changes to: "convenience/notebook-smoke factory only; trainers use `_build_dataloaders` directly with `num_workers>0`."

---

### `datasets/ohp_ssl.py` — NEW (SSL dataset + BBox `_load_trajectory`)

**Analog:** `backend/training/aqa/datasets/squat_ssl.py` (full file, 269 lines)

**Critical delta:** `_load_trajectory` is NOT a copy. OHP trajectories are per-frame BBox arrays, not flat y-lists. See the concrete implementation below. Everything else (`split_half_cycles`, `_augment`, `__getitem__`) is a near-verbatim copy with class/logger renames.

**Imports pattern** (lines 1–48 of squat_ssl.py — copy verbatim, rename logger):
```python
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import scipy.ndimage
import torch
from torch.utils.data import DataLoader, Dataset

from backend.training.aqa.datasets import ssl_augs
from backend.training.aqa.datasets.transforms import (
    decode_clip,
    spatial_train,
    uniform_sample_indices,
)

if TYPE_CHECKING:
    from backend.training.aqa.harness.md_pretrain import MDConfig

logger = logging.getLogger("aqa.phase06")
```

**`split_half_cycles` — REUSE UNCHANGED from squat_ssl.py** (lines 51–96). Copy verbatim into ohp_ssl.py (or import from squat_ssl; copying is simpler and keeps ohp_ssl self-contained). The sign `bottom_is_argmax=False` (argMIN) applies to OHP exactly as to Squat, for different physical reasons (overhead = barbell at lowest y pixel value). Confirmed by RESEARCH §3 inspection of `11681_3.json` (y_center argMIN at frame 47, mid-rep).

**`OHPSSLDataset.__init__` pattern** (mirrors `SquatSSLDataset.__init__` lines 107–141):
```python
class OHPSSLDataset(Dataset):
    def __init__(
        self,
        *,
        videos_root: str,
        trajectories_root: str,
        frames_per_half: int = 16,
        crop_size: int = 112,
        seed: int = 42,
        strong_augs: bool = True,
        use_rotation: bool = False,
        aug_prob: float = 0.5,
    ) -> None:
        # [identical field assignments as SquatSSLDataset]
        # OHP trajectories are in a FLAT directory (not nested subdirs like Squat).
        # Use glob("*.json") here, NOT rglob("*.json") — but keep the stem->Path map.
        vid_stems = {p.stem for p in Path(videos_root).glob("*.mp4")}
        self._traj_paths: dict[str, Path] = {
            p.stem: p for p in Path(trajectories_root).glob("*.json")  # flat dir — no rglob needed
        }
        self._clip_ids: list[str] = sorted(vid_stems & set(self._traj_paths))
        logger.info(
            "OHPSSLDataset: %d clips (videos=%d, trajectories=%d) under %s",
            len(self._clip_ids), len(vid_stems), len(self._traj_paths), videos_root,
        )
```

**`_load_trajectory` — REPLACE ENTIRELY (OHP-specific BBox parser):**
```python
def _load_trajectory(self, clip_id: str) -> np.ndarray:
    """Load OHP barbell trajectory from BBox JSON and return y_center array.

    OHP format (ReadMe.md.docx + offline inspection of 20 files): each JSON is a
    list of frames; each frame = [region_0, region_1, region_2]; each region is a
    list of 0 or 1 bboxes [x1, y1, x2, y2, conf]. Region 0 = barbell (wide
    horizontal bbox covering full image width). y_center = (y1 + y2) / 2 per
    ReadMe formula.

    Missing frames (empty region-0 list, ~40% of files have some) are written as
    np.nan and linearly interpolated, analogous to squat_ssl's NaN interpolation.
    The Squat loader's np.isnan() check is REUSED; only the extraction step differs.

    Returns:
        1-D float64 ndarray, length == number of frames in the clip.
        1:1 with video frames [ASSUMED — MUST be confirmed in the Colab probe].
    """
    path = self._traj_paths[clip_id]
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)  # list of frames

    y = np.empty(len(raw), dtype=float)
    for i, frame in enumerate(raw):
        region_0 = frame[0] if len(frame) > 0 else []
        if region_0:
            x1, y1, x2, y2, conf = region_0[0]
            y[i] = (y1 + y2) / 2.0
        else:
            y[i] = np.nan  # sentinel for empty detection; interpolated below

    nan_mask = np.isnan(y)
    if nan_mask.any():
        valid = ~nan_mask
        if int(valid.sum()) < 2:
            raise ValueError(
                f"trajectory {clip_id}: <2 non-NaN samples ({int(valid.sum())})"
            )
        idx = np.arange(len(y))
        y[nan_mask] = np.interp(idx[nan_mask], idx[valid], y[valid])
    return y
```

**`_augment` and `__getitem__` patterns** (lines 166–238 of squat_ssl.py): copy verbatim; the only change in `__getitem__` is the video path uses `self.videos_root` (same). The `split_half_cycles` call already uses `bottom_is_argmax=False`.

**`build_ssl_loader` pattern** (lines 248–268 of squat_ssl.py): copy verbatim, rename `SquatSSLDataset` → `OHPSSLDataset` in type annotation.

---

### `datasets/splits.py` — MODIFY: add `OHPClipRecord` + `index_ohp()` (additive, no Squat edits)

**Analog:** Existing `datasets/splits.py` — Squat section (lines 20–165)

**Rule:** Do NOT touch any existing code. Add the OHP variant below the existing `__main__` block or in a clearly separated section after line 150.

**Reused helpers (copy nothing — import directly):** `_load_json` (line 46–49) and `_label_value` (line 52–68) are both generic; `index_ohp()` calls them directly.

**New constants to add:**
```python
# ──────────────────────────────────────────────────────────────────────────────
# OHP split constants (Phase 6 D4 verified counts + subdir paths)
# ──────────────────────────────────────────────────────────────────────────────
_OHP_EXPECTED_COUNTS: dict[str, int] = {"train": 1582, "val": 339, "test": 339}
_OHP_LABELS_SUBDIR = "Fitness-AQA_dataset_release/OHP/Labeled_Dataset/Labels"
_OHP_SPLITS_SUBDIR = "Fitness-AQA_dataset_release/OHP/Labeled_Dataset/Splits"
_OHP_LABEL_ELBOWS_FILENAME = "error_elbows.json"
_OHP_LABEL_KNEES_FILENAME  = "error_knees.json"
```

**New `OHPClipRecord` dataclass (mirrors `ClipRecord` lines 29–43):**
```python
@dataclass(frozen=True)
class OHPClipRecord:
    """One labeled OHP clip.

    Attributes:
        clip_id:       stem of the .mp4 (e.g. "11681_3").
        video_path:    absolute path under videos_root.
        label_elbows:  0/1 — non-empty interval list in error_elbows.json => 1.
        label_knees:   0/1 — non-empty interval list in error_knees.json => 1.
    """
    clip_id:      str
    video_path:   str
    label_elbows: int
    label_knees:  int
```

**`index_ohp()` function (mirrors `index()` lines 71–144):**
```python
def index_ohp(
    split: Literal["train", "val", "test"],
    *,
    drive_root: str,
    videos_root: str,
) -> list[OHPClipRecord]:
    """Load the official OHP split and return OHPClipRecords in split-file order.

    Mirrors index() exactly. Label files: error_elbows.json / error_knees.json.
    Expected counts (D4 verified): train=1582, val=339, test=339.
    """
    if split not in _OHP_EXPECTED_COUNTS:
        raise ValueError(f"split must be one of {sorted(_OHP_EXPECTED_COUNTS)}; got '{split}'")

    drive_root_p = Path(drive_root)
    splits_dir = drive_root_p / _OHP_SPLITS_SUBDIR
    labels_dir = drive_root_p / _OHP_LABELS_SUBDIR

    split_keys_path = splits_dir / f"{split}_keys.json"
    label_elbows_path = labels_dir / _OHP_LABEL_ELBOWS_FILENAME
    label_knees_path  = labels_dir / _OHP_LABEL_KNEES_FILENAME

    split_keys = _load_json(split_keys_path)
    if not isinstance(split_keys, list):
        raise TypeError(f"{split_keys_path} must contain a JSON list of clip_id strings")

    elbows_dict = _load_json(label_elbows_path)
    knees_dict  = _load_json(label_knees_path)
    if not isinstance(elbows_dict, dict) or not isinstance(knees_dict, dict):
        raise TypeError("error_elbows.json / error_knees.json must each be a JSON object")

    records: list[OHPClipRecord] = []
    for clip_id in split_keys:
        if not isinstance(clip_id, str):
            raise TypeError(
                f"clip_id in {split_keys_path} is {type(clip_id).__name__}, expected str"
            )
        label_elbows = _label_value(elbows_dict, clip_id)
        label_knees  = _label_value(knees_dict,  clip_id)
        video_path = os.path.join(videos_root, f"{clip_id}.mp4")
        records.append(
            OHPClipRecord(
                clip_id=clip_id,
                video_path=video_path,
                label_elbows=label_elbows,
                label_knees=label_knees,
            )
        )

    expected = _OHP_EXPECTED_COUNTS[split]
    if len(records) != expected:
        logger.warning(
            "OHP split '%s' count drift: got %d records, D4 verified %d",
            split, len(records), expected,
        )

    logger.info("loaded OHP split '%s': %d records", split, len(records))
    return records
```

---

### `harness/supervised_train.py` — MODIFY: `dataset_cls` kwarg on `_build_dataloaders`

**Edit site:** lines 187–189 (verified: function def starts at line 187)

**Current signature (lines 187–189):**
```python
def _build_dataloaders(
    seed: int, config: SupervisedConfig, drive_root: str, videos_root: str,
) -> dict[str, DataLoader]:
```

**New signature (backward-compatible, `SquatKIEKFEDataset` remains default):**
```python
def _build_dataloaders(
    seed: int, config: SupervisedConfig, drive_root: str, videos_root: str,
    *, dataset_cls=SquatKIEKFEDataset,  # NEW kwarg — OHP callers pass OHPElbowsKneesDataset
) -> dict[str, DataLoader]:
```

**Edit to function body** (lines 222–224 — swap hardcoded class for `dataset_cls`):

Current (lines 222–224):
```python
    train_ds = SquatKIEKFEDataset(split="train", train_aug=True, **common_kwargs)
    val_ds = SquatKIEKFEDataset(split="val", train_aug=False, **common_kwargs)
    test_ds = SquatKIEKFEDataset(split="test", train_aug=False, **common_kwargs)
```

Replace with:
```python
    train_ds = dataset_cls(split="train", train_aug=True, **common_kwargs)
    val_ds   = dataset_cls(split="val",   train_aug=False, **common_kwargs)
    test_ds  = dataset_cls(split="test",  train_aug=False, **common_kwargs)
```

**No other changes to `supervised_train.py`** — all other functions (`build_model`, `_val_pass`, `run_supervised_epoch`, `seed_worker`) are exercise-agnostic and used unchanged.

**Note on `videos_root` default:** `run_supervised_epoch` defaults `videos_root="/content/squat_videos"` (line 337). OHP callers MUST pass `videos_root="/content/ohp_videos"` explicitly. Add a guard assertion to `_build_dataloaders` or note it prominently in the OHP notebook.

---

### `harness/md_finetune.py` — MODIFY: `dataset_cls` kwarg on `run_md_finetune_epoch`

**Two edit sites (verified line numbers):**

**Edit site 1 — import at line 37 (informational comment only, not functional):**
```python
from backend.training.aqa.datasets.squat import SquatKIEKFEDataset
```
This import remains — it is the default value for the `dataset_cls` kwarg. No change needed to this line; `SquatKIEKFEDataset` stays importable for backward compatibility.

**Edit site 2 — `run_md_finetune_epoch` signature** (line 134 verified):
Add `dataset_cls=SquatKIEKFEDataset` as a keyword-only argument:
```python
def run_md_finetune_epoch(
    *,
    run_name: str,
    md_backbone_path: str,
    drive_root: str = "/content/drive/MyDrive",
    videos_root: str = "/content/squat_videos",
    seed: int = 42,
    config: FinetuneConfig | None = None,
    resume: bool = True,
    max_epochs: int | None = None,
    dataset_cls=SquatKIEKFEDataset,  # NEW kwarg — forward to _build_dataloaders
) -> dict:
```

**Edit site 3 — call to `_build_dataloaders` at line 193:**

Current (line 193):
```python
    loaders = _build_dataloaders(seed, config, drive_root, videos_root)
```

Replace with:
```python
    loaders = _build_dataloaders(seed, config, drive_root, videos_root, dataset_cls=dataset_cls)
```

**No other changes to `md_finetune.py`** — `build_finetune_model`, `FinetuneConfig`, `_d6_overfit_abort`, the training loop, checkpoint schema, and early-stop are all exercise-agnostic.

---

### `harness/colab.py` — MODIFY: add `stage_ohp_videos` + `stage_unlabeled_ohp_videos`

**Analog:** `stage_squat_videos` (line 317) and `stage_unlabeled_squat_videos` (line 449)

**Add after line 538 (end of `stage_unlabeled_squat_videos`).**

**New constant:**
```python
_OHP_LABELED_VIDEOS_EXPECT_COUNT   = 2367   # Phase 6 D4: 2,367 mp4s in labeled zip (2,260 official + 107 extra)
_OHP_UNLABELED_VIDEOS_EXPECT_COUNT = 5490   # Phase 6 D4 verified count
```

**`stage_ohp_videos` signature pattern** (mirrors `stage_squat_videos` lines 317–322):
```python
def stage_ohp_videos(
    drive_root: str,
    *,
    local_root: str = "/content/ohp_videos",
    expect_count: int = _OHP_LABELED_VIDEOS_EXPECT_COUNT,
) -> str:
    """Copy OHP labeled videos.zip from Drive -> /content/ohp_videos/, extract, verify. Idempotent.

    Source: {drive_root}/Fitness-AQA_dataset_release/OHP/Labeled_Dataset/videos.zip
    (in the -3-001 release folder alongside the Squat labeled data).

    [Three-layer resume: cache-hit -> byte-resume copy -> per-member mp4 extract]
    Mirrors stage_squat_videos exactly; only the source path and local names differ.
    """
    # [copy stage_squat_videos body verbatim, replace paths:]
    src_zip = (
        Path(drive_root)
        / "Fitness-AQA_dataset_release/OHP/Labeled_Dataset/videos.zip"
    )
    local_zip = Path("/content/ohp_videos.zip")
    # [rest of body identical to stage_squat_videos — cache hit check, copy, extract, verify, cleanup]
```

**`stage_unlabeled_ohp_videos` signature pattern** (mirrors `stage_unlabeled_squat_videos` lines 449–455):
```python
def stage_unlabeled_ohp_videos(
    drive_root_3001: str,
    drive_root_3002: str,
    *,
    local_videos_root: str = "/content/ohp_unlabeled_videos",
    local_traj_root: str = "/content/ohp_trajectories",
    expect_count: int = _OHP_UNLABELED_VIDEOS_EXPECT_COUNT,
) -> tuple[str, str]:
    """Stage OHP unlabeled videos.zip (from -3-002) + bar_trajectories_raw.zip (from -3-001).

    OHP archive is SPLIT across two release folders (D4, Pitfall 4):
      - videos.zip:               {drive_root_3002}/.../OHP/Unlabeled_Dataset/videos.zip
      - bar_trajectories_raw.zip: {drive_root_3001}/.../OHP/Unlabeled_Dataset/bar_trajectories_raw.zip

    Two separate drive_root parameters handle this. Trajectory archive uses zipfile.extractall
    (JSON members, flat layout) — NOT _extract_with_resume_and_progress (mp4-only).
    Mirrors stage_unlabeled_squat_videos body; only the source paths differ.
    """
    # [body mirrors stage_unlabeled_squat_videos lines 478–538, with two drive_root args]
    base_3001 = Path(drive_root_3001) / "Fitness-AQA_dataset_release/OHP/Unlabeled_Dataset"
    base_3002 = Path(drive_root_3002) / "Fitness-AQA_dataset_release/OHP/Unlabeled_Dataset"
    src_traj   = base_3001 / "bar_trajectories_raw.zip"
    src_videos = base_3002 / "videos.zip"
    local_videos_zip = Path("/content/ohp_unlabeled_videos.zip")
    local_traj_zip   = Path("/content/ohp_trajectories.zip")
    # [trajectory extraction: zipfile.extractall → flat bar_trajectories_raw/{clip_id}.json layout]
    # [video extraction: _extract_with_resume_and_progress → mp4s at top of local_videos_root]
```

---

### `datasets/test_ohp.py` — NEW (unit tests)

**Analog:** `harness/test_md_finetune.py` (full file, 69 lines) + `eval/test_metrics.py` test style

**Test file structure pattern** (from `test_md_finetune.py` lines 1–16):
```python
"""Unit tests for OHP dataset, trajectory loader, and splits (OHP-01-a through OHP-01-d).

See: .planning/phases/06-overhead-press/06-RESEARCH.md §6 validation architecture.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
```

**Four tests to implement (from RESEARCH §6 test map):**

- `test_ohp_dataset_shape` (OHP-01-a): instantiate `OHPElbowsKneesDataset` with a
  `tmp_path` fixture containing 2 synthetic mp4s + mock split JSONs; assert
  `__getitem__` returns `(clip[3,32,112,112], labels[2])` and `pos_weight` shape is
  `(2,)`.

- `test_trajectory_load` (OHP-01-b): write a synthetic BBox JSON (5 frames, region_0
  present on frames 0/2/4, empty on 1/3); call `OHPSSLDataset._load_trajectory`; assert
  output is a 1-D float array of length 5 with no NaNs and the interpolated values are
  between their neighbors.

- `test_split_half_cycles_ohp` (OHP-01-c): call `split_half_cycles` with a synthetic
  decreasing-then-increasing trajectory (V-shape, argMIN in the middle); assert
  `descent_idx[-1] <= argmin <= ascent_idx[0]`. This is the same logic as Squat (argMIN);
  the test guards the `bottom_is_argmax=False` wiring.

- `test_ohp_splits` (OHP-01-d): calls `splits.index_ohp` against actual local archive
  paths (marked `@pytest.mark.slow` or `skipif` on CI); asserts counts match
  1582/339/339.

**Test pattern for the pure-function tests** (mirrors `test_d6_overfit_monitor` in
`test_md_finetune.py` lines 49–68 — no GPU, fast):
```python
def test_trajectory_load(tmp_path) -> None:
    """BBox JSON with 2 empty region-0 frames is interpolated correctly."""
    from backend.training.aqa.datasets.ohp_ssl import OHPSSLDataset

    # Build a synthetic 5-frame BBox JSON.
    frames = [
        [[[10, 100, 200, 200, 90]], [], []],  # frame 0 — region_0 has bbox
        [[], [], []],                          # frame 1 — region_0 empty → NaN
        [[[10, 60, 200, 160, 88]], [], []],   # frame 2
        [[], [], []],                          # frame 3 — region_0 empty → NaN
        [[[10, 140, 200, 260, 91]], [], []],  # frame 4
    ]
    traj_path = tmp_path / "00001_0.json"
    import json
    traj_path.write_text(json.dumps(frames))

    ds = OHPSSLDataset.__new__(OHPSSLDataset)
    ds._traj_paths = {"00001_0": traj_path}
    y = ds._load_trajectory("00001_0")

    assert y.shape == (5,)
    assert not np.isnan(y).any()
    # y_center: frame 0 = (100+200)/2=150, frame 2 = (60+160)/2=110, frame 4 = (140+260)/2=200
    assert y[0] == pytest.approx(150.0)
    assert y[2] == pytest.approx(110.0)
    assert y[4] == pytest.approx(200.0)
    # Interpolated frames must lie between their neighbors.
    assert y[0] >= y[1] >= y[2] or y[0] <= y[1] <= y[2]  # monotone interpolation
```

---

### OHP Colab training notebooks — NEW

**Analogs:**
- `notebooks/03_squat_supervised_baseline.py` → `notebooks/06_ohp_supervised_baseline.py`
- `notebooks/04_squat_md_ssl.py` → `notebooks/07_ohp_md_ssl.py`
- `notebooks/05_squat_md_finetune.py` → `notebooks/08_ohp_md_finetune.py`

**Cell A bootstrap pattern** (lines 30–61 of `03_squat_supervised_baseline.py` — copy verbatim; only the comment changes):
```python
# %% [markdown]
# ## Cell A — bootstrap (clone-or-pull repo + sys.path)

# %%
import os, subprocess, sys

REPO_DIR = "/content/fitnova"
REPO_URL = "https://github.com/Ibrahim-Shahin1/fitnova.git"
BRANCH   = "fresh-start"

if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "checkout", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only", "origin", BRANCH], check=True)

os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
```

**Step 0 pattern** (from `03_squat_supervised_baseline.py` lines 65–80): F8 constraint — first import MUST be `from backend.training.aqa.harness import _envinit`. OHP-specific Step 0 additions vs Squat Step 0:
- Replace `stage_squat_videos` call with `stage_ohp_videos(drive_root, local_root="/content/ohp_videos")`.
- Pass `videos_root="/content/ohp_videos"` everywhere.
- Phase 6 run_name: `"ohp_supervised_v1"` / `"ohp_md_pretrain_v1"` / `"ohp_md_finetune_seed{42,1337,7}"`.
- Drive layout: `My Drive/FitNova/checkpoints/phase06/{run_name}/`.
- `SupervisedConfig` / `FinetuneConfig` defaults remain; caller passes
  `videos_root="/content/ohp_videos"` and `dataset_cls=OHPElbowsKneesDataset`.

**`run_supervised_epoch` call pattern for OHP notebooks:**
```python
from backend.training.aqa.datasets.ohp import OHPElbowsKneesDataset
from backend.training.aqa.harness.supervised_train import run_supervised_epoch, SupervisedConfig

result = run_supervised_epoch(
    run_name="ohp_supervised_v1",
    drive_root=DRIVE_ROOT,
    videos_root="/content/ohp_videos",    # OHP override (Pitfall 5 guard)
    seed=42,
    config=SupervisedConfig(),
    dataset_cls=OHPElbowsKneesDataset,    # inject OHP dataset
)
```

**`run_md_finetune_epoch` call pattern:**
```python
from backend.training.aqa.harness.md_finetune import run_md_finetune_epoch, FinetuneConfig

for seed in [42, 1337, 7]:
    result = run_md_finetune_epoch(
        run_name=f"ohp_md_finetune_seed{seed}",
        md_backbone_path=BACKBONE_PATH,
        drive_root=DRIVE_ROOT,
        videos_root="/content/ohp_videos",
        seed=seed,
        config=FinetuneConfig(),
        dataset_cls=OHPElbowsKneesDataset,
    )
```

**Disconnect-safe caching pattern** (from Phase 4 SSL notebook — cache Drive walks to pkl):
```python
import pickle, os
CACHE_PATH = f"{DRIVE_ROOT}/FitNova/checkpoints/phase06/ohp_clip_catalogue.pkl"
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        catalogue = pickle.load(f)
else:
    catalogue = build_catalogue(...)  # expensive Drive walk
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(catalogue, f)
```

---

### OHP deliverable notebooks — NEW (docs/notebooks/05–08)

**Analogs:** `docs/notebooks/01_squat_eda.py` through `04_squat_evaluation.py`

**EDA notebook structure pattern** (from `docs/notebooks/01_squat_eda.py` lines 1–80):
```python
# --- jupytext header ---
# %% [markdown]
# # FitNova — OHP Dataset: Exploratory Data Analysis
# Dataset: Fitness-AQA (Parmar et al., ECCV 2022), OHP labeled set.
# Two video errors: Elbows (upper-body extension) and Knees (lower-body alignment).

# %% imports
import os, sys, json, time
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Project root + dataset path discovery (mirror squat_eda.py lines 32–39)
BASE = ("Fitness-AQA/.../OHP/Labeled_Dataset")
elbows = json.load(open(os.path.join(BASE, "Labels/error_elbows.json")))
knees  = json.load(open(os.path.join(BASE, "Labels/error_knees.json")))
splits = {s: list(json.load(open(os.path.join(BASE, f"Splits/{s}_keys.json"))))
          for s in ["train", "val", "test"]}
```

**Visualization style pattern** (from `01_squat_eda.py` lines 55–80 — bar/grouped bar with text annotations, `plt.tight_layout()` + `plt.show()`): mirror color palette `["#1B998B", "#2E86AB", "#D7263D"]` for consistency with Squat figures.

**results.pkl schema pattern** (from Phase 4 context — mirror for OHP):
```python
results = {
    "ssl_epochs": [...],          # per-epoch SSL curves
    "finetune_seeds": {42: ..., 1337: ..., 7: ...},  # per-seed fine-tune curves
    "ensemble_val_scores": ...,   # shape (N_val, 2)
    "ensemble_test_scores": ...,  # shape (N_test, 2)
    "val_labels": ...,
    "test_labels": ...,
    "best_thresholds": {"elbows": ..., "knees": ...},
    "test_f1": {"elbows": ..., "knees": ...},
    "paper_targets": {"elbows": 0.4552, "knees": 0.8452},
}
```

---

## Shared Patterns

### Module docstring pattern
**Source:** `datasets/squat.py` lines 1–17, `harness/md_finetune.py` lines 1–21
**Apply to:** all new Python files (`ohp.py`, `ohp_ssl.py`)
```python
"""OHPElbowsKneesDataset + build_loaders factory — joint multi-label dataset over the official OHP splits.

Yields `(video[3, T, H, W] float32, labels[2] float32)` from official train/val/test splits.
Computes and exposes `pos_weight` (Tensor of length 2 for Elbows/Knees) at construction ...

Phase 6 D4 verified: Elbows+ 25.7% (w ≈ 2.887), Knees+ 34.2% (w ≈ 1.924).
Joint multi-label is justified by shared MD-SSL representation + efficiency (D2).
...
"""

from __future__ import annotations
```

### `from __future__ import annotations` + logger pattern
**Source:** all backend Python modules
**Apply to:** `ohp.py`, `ohp_ssl.py`, `test_ohp.py`
Every new Python file starts with `from __future__ import annotations` and declares a module-level logger:
```python
logger = logging.getLogger("aqa.phase06")
```

### `persistent_workers` guard pattern
**Source:** `harness/supervised_train.py` lines 239–250
**Apply to:** `ohp.py:build_loaders` and any OHP DataLoader construction
```python
_persistent = config.num_workers > 0  # D7 / [[reference_pytorch_persistent_workers]]
DataLoader(..., persistent_workers=_persistent)
```

### `map_location='cpu'` checkpoint load pattern
**Source:** `harness/md_finetune.py` line 213 + `harness/supervised_train.py` line 443
**Apply to:** any `load_latest_checkpoint` call in OHP notebooks/trainers
```python
prior = load_latest_checkpoint(run_dir, expected_config_hash=config_hash_str, map_location="cpu")
```

### Atomic checkpoint write pattern
**Source:** `harness/colab.py` atomic_save_checkpoint contract
**Apply to:** OHP training notebooks (use `atomic_save_checkpoint`, never `torch.save` directly)
- `update_latest=True` (default) for per-epoch `epoch_NNN.pt`
- `update_latest=False` when saving `backbone.pt` from MD pretrain (must NOT overwrite `latest.txt` of the SSL run)

### pytest skip-torchless pattern
**Source:** `harness/test_md_finetune.py` line 13
**Apply to:** `datasets/test_ohp.py`
```python
torch = pytest.importorskip("torch")
```

### Section separator style
**Source:** `harness/supervised_train.py` lines 72, 108, 139, 182, 327
**Apply to:** `ohp.py`, `ohp_ssl.py`, additions to `splits.py`
```python
# ──────────────────────────────────────────────────────────────────────────────
# Section heading
# ──────────────────────────────────────────────────────────────────────────────
```

---

## No Analog Needed (Import Directly — REUSE UNCHANGED)

The following files are exercise-agnostic and must NOT be copied or modified:

| File | Role | Reason |
|---|---|---|
| `datasets/transforms.py` | transforms | `KINETICS_MEAN/STD`, `uniform_sample_indices`, `decode_clip`, `spatial_train`, `spatial_val` — no Squat coupling |
| `datasets/ssl_augs.py` | SSL augmentations | Operates on `[T,3,H,W]` tensors; no exercise-specific logic |
| `eval/metrics.py` | evaluation | Operates on score arrays + label arrays; exercise-agnostic |
| `eval/ensemble.py` | evaluation | `aggregate_sigmoid_mean` over score arrays |
| `eval/tta.py` | evaluation | Per-clip augmented forward passes |
| `harness/md_pretrain.py` | training | Epoch loop, `_linear_probe`, `ProjectionHead`, `md_triplet_loss`, `build_md_model` — no exercise coupling |
| `harness/supervised_train.py` (epoch loop, `build_model`, `_val_pass`) | training | Exercise-agnostic beyond `_build_dataloaders` |

---

## Carry-Forward Landmines (Must Appear in Every Plan Action)

These are non-negotiable constraints from Phase 4 D7. Every plan task that touches a DataLoader,
checkpoint, or notebook cell must include them:

1. `persistent_workers=True` on every `DataLoader` with `num_workers > 0`
2. `map_location='cpu'` for every `load_latest_checkpoint` call
3. Load `metrics_history` from the `latest.txt`-pointed checkpoint, NOT `best.pt`
4. `update_latest=False` when writing `backbone.pt`
5. Every Colab notebook delivered as paired `.py` + `.ipynb`
6. Restart runtime / `importlib.reload` after `git pull` in a running notebook
7. OHP trajectory extraction uses `zipfile.extractall` (JSON), NOT `_extract_with_resume_and_progress` (mp4-only)
8. OHP notebooks always pass `videos_root="/content/ohp_videos"` explicitly (Pitfall 5 guard)

---

## Verified Line Numbers (RESEARCH cross-check)

RESEARCH.md §5 cited the following line numbers; verified against actual source:

| Claim | Actual | Status |
|---|---|---|
| `_build_dataloaders` at lines 222–224 constructs `SquatKIEKFEDataset` | lines 222–224 confirmed | EXACT |
| `md_finetune` imports `SquatKIEKFEDataset` at line 37 | line 37 confirmed | EXACT |
| `md_finetune` calls `_build_dataloaders` at line 193 | line 193 confirmed | EXACT |
| `stage_squat_videos` starts at line 317 | line 317 confirmed | EXACT |
| `stage_unlabeled_squat_videos` starts at line 449 | line 449 confirmed | EXACT |

---

## Metadata

**Analog search scope:** `backend/training/aqa/datasets/`, `backend/training/aqa/harness/`,
`backend/training/aqa/notebooks/`, `docs/notebooks/`
**Files read:** 10 source files
**Pattern extraction date:** 2026-05-27
