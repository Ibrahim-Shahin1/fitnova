"""Official Squat split loader — reads {train,val,test}_keys.json + error_knees_{inward,forward}.json into ClipRecord lists.

Phase 1 dataset report verified split sizes 1136/243/244 and full label coverage across all three splits; this module surfaces those records to the dataset class in `squat.py` and to the notebook for reconciliation. Order matches the JSON list in each `{split}_keys.json` for deterministic indexing across runs.

See: `.planning/phases/02-squat-data-pipeline-colab-harness/02-01-PLAN.md` — Task 4 / D7 / interfaces block.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger("aqa.phase02")

# Phase 1 verified counts (`01-DATASET-REPORT.md` § "Official splits").
_EXPECTED_COUNTS: dict[str, int] = {"train": 1136, "val": 243, "test": 244}

# Dataset layout under the Drive root (Phase 1 verified — `My Drive/Fitness-AQA_dataset_release/...`).
_LABELS_SUBDIR = "Fitness-AQA_dataset_release/Squat/Labeled_Dataset/Labels"
_SPLITS_SUBDIR = "Fitness-AQA_dataset_release/Squat/Labeled_Dataset/Splits"
_LABEL_KIE_FILENAME = "error_knees_inward.json"
_LABEL_KFE_FILENAME = "error_knees_forward.json"


@dataclass(frozen=True)
class ClipRecord:
    """One labeled Squat clip — frozen so records are hashable and safe in sets/dict keys.

    Attributes:
        clip_id:    stem of the .mp4 (e.g. "12345_0").
        video_path: absolute path under `videos_root` (`{videos_root}/{clip_id}.mp4`).
        label_kie:  0/1 — non-empty interval list in error_knees_inward.json ⇒ 1.
        label_kfe:  0/1 — non-empty interval list in error_knees_forward.json ⇒ 1.
    """

    clip_id: str
    video_path: str
    label_kie: int
    label_kfe: int


def _load_json(path: Path) -> object:
    """Read a UTF-8 JSON file. Exceptions propagate with the path in the error chain."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _label_value(label_dict: dict, clip_id: str) -> int:
    """Binary label from an interval-list label dict: any non-empty list ⇒ 1, else 0.

    Defensive: per the Fitness-AQA schema and Phase 1 verification, every split clip_id
    must appear in BOTH label dicts (D7 invariant). Missing keys ⇒ KeyError rather than
    silent zero — we'd rather surface a dataset-format surprise than guess.
    """
    if clip_id not in label_dict:
        raise KeyError(
            f"clip_id '{clip_id}' missing from label dict (D7 invariant: every split id must be labeled)"
        )
    value = label_dict[clip_id]
    if not isinstance(value, list):
        raise TypeError(
            f"clip_id '{clip_id}' label is {type(value).__name__}, expected list of [start, end] intervals"
        )
    return int(bool(value))


def index(
    split: Literal["train", "val", "test"],
    *,
    drive_root: str,
    videos_root: str,
) -> list[ClipRecord]:
    """Load the official Squat split and return ClipRecords in split-file order.

    Args:
        split:       one of "train" / "val" / "test".
        drive_root:  path to the Drive root containing `Fitness-AQA_dataset_release/`
                     (e.g. `/content/drive/MyDrive`).
        videos_root: path where the unzipped `.mp4` files live (e.g. `/content/squat_videos`).
                     Used to construct `video_path` on each record; files are NOT opened here.

    Returns:
        Ordered list of `ClipRecord`. Order matches the JSON list in `{split}_keys.json`
        — load-bearing for deterministic indexing across training runs.

    Raises:
        ValueError:        unknown split name.
        FileNotFoundError: missing split or label JSON file.
        KeyError:          a split clip_id is missing from one of the label dicts.
        TypeError:         a JSON value has unexpected schema (label not a list, etc.).
    """
    if split not in _EXPECTED_COUNTS:
        raise ValueError(f"split must be one of {sorted(_EXPECTED_COUNTS)}; got '{split}'")

    drive_root_p = Path(drive_root)
    splits_dir = drive_root_p / _SPLITS_SUBDIR
    labels_dir = drive_root_p / _LABELS_SUBDIR

    split_keys_path = splits_dir / f"{split}_keys.json"
    label_kie_path = labels_dir / _LABEL_KIE_FILENAME
    label_kfe_path = labels_dir / _LABEL_KFE_FILENAME

    split_keys = _load_json(split_keys_path)
    if not isinstance(split_keys, list):
        raise TypeError(f"{split_keys_path} must contain a JSON list of clip_id strings")

    kie_dict = _load_json(label_kie_path)
    kfe_dict = _load_json(label_kfe_path)
    if not isinstance(kie_dict, dict) or not isinstance(kfe_dict, dict):
        raise TypeError("error_knees_{inward,forward}.json must each be a JSON object")

    records: list[ClipRecord] = []
    for clip_id in split_keys:
        if not isinstance(clip_id, str):
            raise TypeError(
                f"clip_id in {split_keys_path} is {type(clip_id).__name__}, expected str"
            )
        label_kie = _label_value(kie_dict, clip_id)
        label_kfe = _label_value(kfe_dict, clip_id)
        video_path = os.path.join(videos_root, f"{clip_id}.mp4")
        records.append(
            ClipRecord(
                clip_id=clip_id,
                video_path=video_path,
                label_kie=label_kie,
                label_kfe=label_kfe,
            )
        )

    expected = _EXPECTED_COUNTS[split]
    if len(records) != expected:
        logger.warning(
            "split '%s' count drift: got %d records, Phase 1 measured %d",
            split,
            len(records),
            expected,
        )

    logger.info("loaded split '%s': %d records", split, len(records))
    return records


def expected_counts() -> dict[str, int]:
    """Return Phase 1 verified split counts. Used by the notebook for reconciliation."""
    return dict(_EXPECTED_COUNTS)


if __name__ == "__main__":
    # Standalone smoke-check — set DRIVE_ROOT below to your Drive mount path (Colab) or
    # local snapshot path, and uncomment the run block. Off by default so
    # `python -c "import splits"` is side-effect-free.

    # DRIVE_ROOT = "/content/drive/MyDrive"
    # VIDEOS_ROOT = "/content/squat_videos"  # path is interpolated; .mp4s not required for splits.py
    # for split_name in ("train", "val", "test"):
    #     recs = index(split_name, drive_root=DRIVE_ROOT, videos_root=VIDEOS_ROOT)
    #     kie_pos = sum(r.label_kie for r in recs)
    #     kfe_pos = sum(r.label_kfe for r in recs)
    #     print(f"{split_name:5s}: {len(recs):4d} records   KIE+ {kie_pos:3d}   KFE+ {kfe_pos:3d}")

    print("splits.py — smoke block disabled (uncomment under __main__ to run)")
