"""D1 gate: verify QEVD extract integrity.

Runs after the four FIT-300K parts and FIT-COACH have been extracted.
Validates the on-disk shape against the master-plan-locked invariants
and writes ``backend/data/qevd_phase_reports/D1_extract_integrity.json``.

Usage:
    python backend/training/diagnostics/d1_verify_extract.py \\
        --mp4-dir   backend/data/qevd_raw/fit300k \\
        --labels-dir backend/data/qevd_raw/fitcoach \\
        --sample-size 100

Master plan reference: section II.4.2.

Hard-stop checks (any failure -> overall FAIL):
  * total MP4 count in [290_000, 310_000]   (paper Table 2: 281_660 train + 16_429 test = 298_089)
  * 100 percent of clip IDs in labels exist as MP4s on disk (no missing files)
  * mean clip duration in [4.5, 6.5] s     (paper: 5.6 +/- 1.1 s)
  * total hours in [440, 480]              (paper: 460 hours)
  * unique human count >= 1_800 + 100      (paper: 1_800+ train + 100 test)
  * sampled-clip ffprobe success rate >= 0.99

Soft-info metrics:
  * clip-id sequential gaps (looking for missing IDs Qualcomm dropped)
  * per-human clip count distribution (min / median / max)
  * fine-grained class label histogram (top 20)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from backend.training.diagnostics.phase_gates import (
    expect, gate, write_gate_report,
)
from backend.training.preprocessing.qevd_label_builder import QEVDLabels


def _ffprobe_duration(mp4: Path) -> float | None:
    """Return clip duration in seconds via ffprobe; None if not invokable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(mp4)],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        return float(out.stdout.strip())
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        return None


def _opencv_duration(mp4: Path) -> float | None:
    """Fallback: use cv2 to read fps + frame count."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return n_frames / max(1.0, fps)
    except Exception:
        return None


def _clip_duration(mp4: Path) -> float | None:
    """ffprobe first; OpenCV fallback if ffprobe missing."""
    d = _ffprobe_duration(mp4)
    return d if d is not None else _opencv_duration(mp4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mp4-dir", type=Path, required=True,
                        help="Directory containing the extracted FIT-300K "
                             "MP4 files (any nesting depth)")
    parser.add_argument("--labels-dir", type=Path, default=None,
                        help="Directory containing feedbacks_short_clips.json "
                             "+ fine_grained_labels.json (typically the "
                             "extracted FIT-COACH download root)")
    parser.add_argument("--sample-size", type=int, default=100,
                        help="Number of random clips to ffprobe for duration")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(message)s")

    if not args.mp4_dir.exists():
        write_gate_report(
            "D1_extract_integrity", "FAIL", {},
            [{"metric": "mp4_dir_exists", "actual": False,
              "bound": f"{args.mp4_dir} exists"}],
        )
        return 1

    with gate("D1_extract_integrity", skip_prior_check=False) as g:
        # ── 1. MP4 file count ────────────────────────────────────────────
        mp4s = sorted(args.mp4_dir.rglob("*.mp4"))
        n_mp4 = len(mp4s)
        g.metrics["n_mp4"] = n_mp4
        expect(290_000 <= n_mp4 <= 310_000, "n_mp4_in_range",
               n_mp4, "[290000, 310000]", g.failures)

        # ── 2. Sequential ID coverage (Qualcomm uses 00000000..NNNNNNNN) ─
        clip_ids = []
        for p in mp4s:
            try:
                clip_ids.append(int(p.stem))
            except ValueError:
                pass
        if clip_ids:
            min_id, max_id = min(clip_ids), max(clip_ids)
            n_unique = len(set(clip_ids))
            n_expected_range = max_id - min_id + 1
            n_missing = n_expected_range - n_unique
            g.metrics["clip_id_min"] = min_id
            g.metrics["clip_id_max"] = max_id
            g.metrics["n_unique_clip_ids"] = n_unique
            g.metrics["n_missing_clip_ids_in_range"] = n_missing
            expect(n_missing < 1_000, "n_missing_clip_ids_lt_1000",
                   n_missing, "<1000", g.failures)
        else:
            expect(False, "clip_ids_parseable",
                   0, "at least one numeric clip id", g.failures)

        # ── 3. Sample clips for duration sanity ──────────────────────────
        sample_size = min(args.sample_size, n_mp4)
        sample = random.sample(mp4s, sample_size) if mp4s else []
        durations = []
        n_probe_fail = 0
        for mp4 in sample:
            d = _clip_duration(mp4)
            if d is None or d <= 0:
                n_probe_fail += 1
            else:
                durations.append(d)

        probe_success_rate = (
            (len(durations) / sample_size) if sample_size else 0.0
        )
        g.metrics["sampled_clip_count"] = sample_size
        g.metrics["probe_success_rate"] = round(probe_success_rate, 4)
        expect(probe_success_rate >= 0.99, "probe_success_rate_ge_0.99",
               probe_success_rate, ">=0.99", g.failures)

        if durations:
            mean_d = statistics.fmean(durations)
            median_d = statistics.median(durations)
            min_d, max_d = min(durations), max(durations)
            g.metrics["sample_duration_mean_s"] = round(mean_d, 3)
            g.metrics["sample_duration_median_s"] = round(median_d, 3)
            g.metrics["sample_duration_min_s"] = round(min_d, 3)
            g.metrics["sample_duration_max_s"] = round(max_d, 3)
            expect(4.5 <= mean_d <= 6.5, "mean_duration_in_range",
                   round(mean_d, 3), "[4.5, 6.5]", g.failures)

            # Total dataset hours: extrapolate from sample mean
            est_total_hours = (mean_d * n_mp4) / 3600.0
            g.metrics["est_total_hours"] = round(est_total_hours, 1)
            expect(440 <= est_total_hours <= 480, "total_hours_in_range",
                   round(est_total_hours, 1), "[440, 480]", g.failures)

        # ── 4. Labels integrity (only if labels-dir provided) ────────────
        if args.labels_dir is not None and args.labels_dir.exists():
            fbs_path  = args.labels_dir / "feedbacks_short_clips.json"
            fine_path = args.labels_dir / "fine_grained_labels.json"
            workers_path = (
                args.labels_dir / "fine_grained_labels_with_worker_ids.json"
            )
            labels = QEVDLabels.from_files(
                feedbacks_short_clips_path=fbs_path if fbs_path.exists() else None,
                fine_grained_labels_path=fine_path if fine_path.exists() else None,
                worker_ids_path=workers_path if workers_path.exists() else None,
            )
            stats = labels.stats()
            for k, v in stats.items():
                g.metrics[f"labels__{k}"] = v

            # Cross-check: every clip_id in labels must have an MP4
            mp4_ids_set = {p.stem for p in mp4s}
            label_ids_with_fb = set(labels.clip_to_feedbacks.keys())
            label_ids_with_fine = set(labels.clip_to_fine.keys())
            label_ids_all = label_ids_with_fb | label_ids_with_fine
            missing_mp4s = label_ids_all - mp4_ids_set
            g.metrics["n_label_ids_missing_mp4"] = len(missing_mp4s)
            expect(len(missing_mp4s) == 0,
                   "all_label_ids_have_mp4_on_disk",
                   len(missing_mp4s), "0", g.failures)

            # Unique humans (paper: 1_800+ train + 100 test = 1_900+)
            n_humans = stats["n_unique_humans"]
            expect(n_humans >= 1_900, "n_unique_humans_ge_1900",
                   n_humans, ">=1900", g.failures)
        else:
            g.metrics["labels_check_skipped"] = (
                "labels-dir not given or does not exist; D1 cannot fully "
                "verify until FIT-COACH download is extracted"
            )
            g.forced_status = "PARTIAL"

    return 0


if __name__ == "__main__":
    sys.exit(main())
