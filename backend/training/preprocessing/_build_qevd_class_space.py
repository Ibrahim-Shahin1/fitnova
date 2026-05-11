"""Build the v6.1 cleaned QEVD class space JSON.

Run once (or whenever the cleanup rules change) to regenerate
`backend/data/qevd_class_space_v6_1.json` from the raw QEVD labels.

Cleanup rules (v6.1.0):
  1. Drop prefixes that have only one variation in the raw data (general
     activities like "drinking from a bottle"). They are not exercises with
     form variants.
  2. Drop "not visible" variation suffixes. They are exclusion cases.
  3. Drop (prefix, variation) classes with fewer than 50 total clips.
     Long tail noise; ~5,800 clip-occurrences (1.2 percent) discarded.

Output schema is intentionally explicit so the class index is reproducible
from the dataset alone — no human curation. See
`backend/training/preprocessing/qevd_class_space.py` for the loader.
"""
from __future__ import annotations

import collections
import json
import os
from pathlib import Path
from typing import Iterable

VERSION = "v6.1.0"
MIN_CLIPS_PER_CLASS = 50
NOT_VISIBLE_PATTERN = "not visible"

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_LABELS = REPO_ROOT / "backend" / "data" / "qevd_raw" / "fitcoach" / "fine_grained_labels.json"
OUT_PATH = REPO_ROOT / "backend" / "data" / "qevd_class_space_v6_1.json"


def _split_label(label: str) -> tuple[str, str]:
    if " - " in label:
        prefix, variation = label.split(" - ", 1)
    else:
        prefix, variation = label, ""
    return prefix.strip(), variation.strip()


def _enumerate(records: Iterable[dict]) -> tuple[
    collections.Counter,  # pair_counts_total
    collections.Counter,  # pair_counts_train
    collections.Counter,  # pair_counts_test
    dict[str, set[str]],  # prefix_to_variations
]:
    pair_counts_total: collections.Counter = collections.Counter()
    pair_counts_train: collections.Counter = collections.Counter()
    pair_counts_test: collections.Counter = collections.Counter()
    prefix_to_variations: dict[str, set[str]] = collections.defaultdict(set)
    for r in records:
        split = r.get("split", "train")
        for label in r.get("labels", []):
            prefix, variation = _split_label(label)
            pair_counts_total[(prefix, variation)] += 1
            prefix_to_variations[prefix].add(variation)
            if split == "train":
                pair_counts_train[(prefix, variation)] += 1
            elif split == "test":
                pair_counts_test[(prefix, variation)] += 1
    return pair_counts_total, pair_counts_train, pair_counts_test, prefix_to_variations


def _apply_cleanup(
    pair_counts: collections.Counter,
    prefix_to_variations: dict[str, set[str]],
) -> dict[tuple[str, str], int]:
    single_var_prefixes = {p for p, vs in prefix_to_variations.items() if len(vs) <= 1}
    cleaned: dict[tuple[str, str], int] = {}
    for (prefix, variation), n in pair_counts.items():
        if prefix in single_var_prefixes:
            continue
        if variation.lower() == NOT_VISIBLE_PATTERN:
            continue
        if n < MIN_CLIPS_PER_CLASS:
            continue
        cleaned[(prefix, variation)] = n
    return cleaned


def build() -> dict:
    with open(SRC_LABELS, "r", encoding="utf-8") as f:
        records = json.load(f)

    total, train, test, prefix_to_vars = _enumerate(records)
    cleaned = _apply_cleanup(total, prefix_to_vars)

    sorted_classes = sorted(cleaned.items(), key=lambda kv: (kv[0][0], -kv[1]))

    classes_out = []
    prefix_to_indices: dict[str, list[int]] = collections.defaultdict(list)
    for idx, ((prefix, variation), n_total) in enumerate(sorted_classes):
        classes_out.append({
            "idx": idx,
            "prefix": prefix,
            "variation": variation,
            "n_clips_train": train[(prefix, variation)],
            "n_clips_test": test[(prefix, variation)],
            "n_clips_total": n_total,
        })
        prefix_to_indices[prefix].append(idx)

    counts = [c["n_clips_total"] for c in classes_out]
    spec = {
        "version": VERSION,
        "source_file": str(SRC_LABELS.relative_to(REPO_ROOT)).replace("\\", "/"),
        "cleanup_rules": {
            "drop_single_variation_prefixes": True,
            "drop_not_visible": True,
            "min_clips_per_class": MIN_CLIPS_PER_CLASS,
        },
        "stats": {
            "total_records_input": len(records),
            "total_label_occurrences_input": sum(total.values()),
            "total_classes_output": len(classes_out),
            "total_prefixes_output": len(prefix_to_indices),
            "total_clip_occurrences_output": sum(counts),
            "coverage_pct": round(sum(counts) / sum(total.values()) * 100, 2),
            "imbalance_ratio": round(max(counts) / min(counts), 2) if counts else None,
            "median_clips_per_class": int(sorted(counts)[len(counts) // 2]) if counts else None,
        },
        "classes": classes_out,
        "prefix_to_class_indices": dict(prefix_to_indices),
    }
    return spec


def main() -> None:
    spec = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    print(f"Wrote {OUT_PATH}")
    print(f"  version: {spec['version']}")
    print(f"  num_classes: {spec['stats']['total_classes_output']}")
    print(f"  num_prefixes: {spec['stats']['total_prefixes_output']}")
    print(f"  coverage: {spec['stats']['coverage_pct']}%")
    print(f"  imbalance: {spec['stats']['imbalance_ratio']}x")


if __name__ == "__main__":
    main()
