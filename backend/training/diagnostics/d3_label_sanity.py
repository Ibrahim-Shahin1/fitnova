"""D3 gate: label-sanity spot-check.

Validates that the lexicon + variation-tier classifiers produce a
non-unimodal quality distribution and a per-joint-group hit rate in the
acceptable band. Master plan locks this as the single most important
sanity check in the whole pipeline (II.4.4).

Hard-stop checks:
  * quality histogram NOT unimodal at 0.7  (silence-default collapse)
  * each of the 10 joint groups fires on [3%, 50%] of sampled clips
  * D2 (extract quality) gate must already be PASS

Artifacts written:
  * D3_label_sanity.json                        (the gate report)
  * label_spotcheck_100.csv                     (for the user to eyeball)
  * D3_quality_histogram.json                   (5-bucket counts + stats)

Usage:
    python backend/training/diagnostics/d3_label_sanity.py \\
        --labels-dir backend/data/qevd_raw/fitcoach \\
        --sample-size 100

Master plan reference: section II.4.4.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from backend.training.diagnostics.phase_gates import (
    expect, gate, write_gate_report, REPORTS_DIR,
)
from backend.training.preprocessing.qevd_label_builder import (
    QEVDLabels, JOINT_GROUP_NAMES, N_JOINT_GROUPS,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True,
                        help="Directory containing feedbacks_short_clips.json "
                             "+ fine_grained_labels.json (the extracted "
                             "FIT-COACH download root)")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-prior-check", action="store_true",
                        help="Skip the assertion that D2 has passed first; "
                             "useful for label-only validation before "
                             "MediaPipe extraction is run")
    args = parser.parse_args()
    random.seed(args.seed)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    fbs_path = args.labels_dir / "feedbacks_short_clips.json"
    fine_path = args.labels_dir / "fine_grained_labels.json"
    if not fbs_path.exists() or not fine_path.exists():
        write_gate_report("D3_label_sanity", "FAIL", {}, [{
            "metric": "manifest_files_present",
            "actual": f"fbs={fbs_path.exists()} fine={fine_path.exists()}",
            "bound":  "both files must exist",
        }])
        return 1

    logger.info("Loading manifests from %s ...", args.labels_dir)
    labels = QEVDLabels.from_files(
        feedbacks_short_clips_path=fbs_path,
        fine_grained_labels_path=fine_path,
    )
    stats = labels.stats()
    logger.info("Manifests loaded: %s", stats)

    all_clip_ids = sorted(labels.clip_to_fine.keys())
    sample_size = min(args.sample_size, len(all_clip_ids))
    if sample_size < 50:
        logger.warning("Sample size %d < 50; D3 not statistically meaningful",
                       sample_size)
    sample_ids = random.sample(all_clip_ids, sample_size)

    # ── Walk the sample ────────────────────────────────────────────────
    qualities: list[float] = []
    group_hits = np.zeros(N_JOINT_GROUPS, dtype=int)
    quality_sources: Counter = Counter()
    csv_rows: list[dict] = []

    for cid in sample_ids:
        enriched = labels.derive_label(cid)
        q = float(enriched["quality"])
        groups = np.array(enriched["joint_groups"], dtype=bool)
        qualities.append(q)
        group_hits += groups.astype(int)
        quality_sources[enriched["quality_source"]] += 1
        csv_rows.append({
            "clip_id":          enriched["clip_id"],
            "exercise":         enriched.get("exercise") or "",
            "fine_class":       enriched.get("fine_class") or "",
            "n_feedbacks":      enriched["n_feedbacks"],
            "first_feedback":   (enriched["feedbacks"][0]
                                  if enriched["feedbacks"] else ""),
            "derived_q":        round(q, 3),
            "derived_groups":   ",".join(JOINT_GROUP_NAMES[i]
                                          for i, v in enumerate(groups) if v),
            "quality_source":   enriched["quality_source"],
            "split":            enriched.get("split") or "",
        })

    q_arr = np.array(qualities)

    # ── Quality histogram (5 buckets) ──────────────────────────────────
    bins = [(0.0, 0.4), (0.4, 0.55), (0.55, 0.71),
            (0.71, 0.85), (0.85, 1.01)]
    bin_counts = []
    for lo, hi in bins:
        bin_counts.append({
            "lo": lo, "hi": hi,
            "n": int(((q_arr >= lo) & (q_arr < hi)).sum()),
        })
    n_at_silence = int((q_arr == 0.7).sum())

    # ── Write artifacts ─────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    spotcheck_path = REPORTS_DIR / "label_spotcheck_100.csv"
    with open(spotcheck_path, "w", newline="", encoding="utf-8") as f:
        if csv_rows:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    logger.info("CSV spot-check -> %s", spotcheck_path)

    histogram_path = REPORTS_DIR / "D3_quality_histogram.json"
    with open(histogram_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_sampled":             sample_size,
            "bins":                  bin_counts,
            "n_exact_silence_default": n_at_silence,
            "mean":                  round(float(q_arr.mean()), 4),
            "std":                   round(float(q_arr.std()), 4),
            "median":                round(float(np.median(q_arr)), 4),
            "min":                   round(float(q_arr.min()), 4),
            "max":                   round(float(q_arr.max()), 4),
        }, f, indent=2, sort_keys=True)

    # ── Gate context ────────────────────────────────────────────────────
    with gate("D3_label_sanity",
              skip_prior_check=args.skip_prior_check) as g:
        g.metrics["n_sampled"] = sample_size
        g.metrics["n_exact_silence_default"] = n_at_silence
        g.metrics["quality_mean"] = round(float(q_arr.mean()), 4)
        g.metrics["quality_std"]  = round(float(q_arr.std()), 4)
        g.metrics["quality_median"] = round(float(np.median(q_arr)), 4)
        for k, v in stats.items():
            g.metrics[f"manifest__{k}"] = v
        for src, n in quality_sources.items():
            g.metrics[f"quality_source__{src}"] = n

        # Hard-stop 1: not unimodal at 0.7
        # Master plan threshold: hard-stop if >60% at exactly 0.7
        unimodal_thresh = int(0.6 * sample_size)
        expect(n_at_silence < unimodal_thresh, "unimodal_at_0p7",
               f"{n_at_silence}/{sample_size}",
               f"<{unimodal_thresh}/{sample_size}", g.failures)

        # Hard-stop 2: every joint group fires on [3%, 50%]
        for i, name in enumerate(JOINT_GROUP_NAMES):
            rate = float(group_hits[i] / max(1, sample_size))
            g.metrics[f"group_hit_rate__{name}"] = round(rate, 4)
            expect(0.03 <= rate <= 0.50,
                   f"group_hit_rate__{name}_in_[0.03,0.50]",
                   round(rate, 4), "[0.03, 0.50]", g.failures)

        # Soft signal: spot-check CSV exists for user review
        g.metrics["spotcheck_csv_path"]    = str(spotcheck_path)
        g.metrics["quality_histogram_path"] = str(histogram_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
