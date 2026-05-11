"""D2 gate driver: run MediaPipe pose extraction on the full QEVD-FIT-300K
corpus (298,089 clips), aggregate pose-quality stats, write the D2 gate
report. Mirrors the master plan section II.4.3 spec.

Usage:
    # benchmark only (no real run, just measure throughput on N clips)
    python backend/training/diagnostics/d2_extract_features.py --benchmark 100

    # full corpus run
    python backend/training/diagnostics/d2_extract_features.py --workers 4

    # subsample run (stratified is post-hoc; this just caps the count)
    python backend/training/diagnostics/d2_extract_features.py --max-clips 50000
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from backend.training.diagnostics.phase_gates import (
    expect, gate, write_gate_report,
)
from backend.training.preprocessing.qevd_extractor import (
    extract_clip, extract_directory, ExtractStats,
)


def _benchmark(in_dir: Path, n_clips: int, workers: int) -> dict:
    """Run on N random clips with the multiprocessing pool, time it."""
    import shutil
    import tempfile
    mp4s = sorted(in_dir.rglob("*.mp4"))
    if not mp4s:
        raise RuntimeError(f"No MP4s under {in_dir}")
    random.seed(42)
    sample = random.sample(mp4s, min(n_clips, len(mp4s)))
    tmpdir = Path(tempfile.mkdtemp(prefix="qevd_bench_"))

    # Stage the sample into a temp dir so extract_directory only sees those
    sample_dir = tmpdir / "sample_in"
    sample_dir.mkdir(parents=True)
    for mp4 in sample:
        (sample_dir / mp4.name).symlink_to(mp4) if False else \
            shutil.copy2(mp4, sample_dir / mp4.name)

    out_dir = tmpdir / "sample_out"
    t0 = time.time()
    stats = extract_directory(
        in_dir=sample_dir, out_dir=out_dir,
        n_workers=workers, progress_every=max(1, n_clips // 4),
    )
    dt = time.time() - t0

    # Surface the first few failure reasons so we don't have to guess
    if stats.failures:
        print(f"\n--- FIRST 5 FAILURES OUT OF {len(stats.failures)} ---")
        for f in stats.failures[:5]:
            print(f"  mp4:   {f.get('mp4', '?')}")
            print(f"  error: {f.get('error', '?')}")
            print()

    shutil.rmtree(tmpdir, ignore_errors=True)
    return {
        "n_clips":          n_clips,
        "workers":          workers,
        "elapsed_s":        round(dt, 2),
        "clips_per_sec":    round(n_clips / max(0.01, dt), 2),
        "no_pose_mean":     stats.aggregate_no_pose_fraction,
        "interp_mean":      stats.aggregate_interp_fraction,
        "n_dropped":        stats.n_clips_dropped,
        "n_failed":         stats.n_clips_failed,
        "n_ok":             stats.n_clips_ok,
        "est_full_298k_hr": round((298_089 / max(0.01, n_clips / dt)) / 3600, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path,
                        default=REPO_ROOT / "backend" / "data" / "qevd_raw" / "fit300k")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "backend" / "data" / "qevd_extracted" / "fit300k")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=5_000)
    parser.add_argument("--drop-threshold", type=float, default=0.50)
    parser.add_argument("--max-clips", type=int, default=0,
                        help="If >0, cap the run at this many clips (random sample). "
                             "Useful for time-boxed runs.")
    parser.add_argument("--benchmark", type=int, default=0,
                        help="Run a timing benchmark on N clips and exit "
                             "(no real D2 gate report).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.benchmark > 0:
        print(f"=== BENCHMARK: {args.benchmark} clips, "
              f"{args.workers} workers ===")
        result = _benchmark(args.in_dir, args.benchmark, args.workers)
        for k, v in result.items():
            print(f"  {k}: {v}")
        return 0

    # (the --max-clips path below uses _benchmark internals indirectly;
    #  failures are surfaced when the gate writes the JSON report.)

    if args.max_clips > 0:
        print(f"WARNING: --max-clips {args.max_clips} given. Will sample.")
        # For the capped run we stage symlinks/copies into a temp dir
        # and point extract_directory at it; otherwise it walks the full corpus.
        # (Not used in the standard full-corpus run.)
        import shutil, tempfile
        all_mp4s = sorted(args.in_dir.rglob("*.mp4"))
        sample = random.sample(all_mp4s, min(args.max_clips, len(all_mp4s)))
        staging = Path(tempfile.mkdtemp(prefix="qevd_run_"))
        for mp4 in sample:
            shutil.copy2(mp4, staging / mp4.name)
        in_dir = staging
    else:
        in_dir = args.in_dir

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with gate("D2_extract_quality") as g:
        t0 = time.time()
        stats = extract_directory(
            in_dir=in_dir,
            out_dir=args.out_dir,
            n_workers=args.workers,
            progress_every=args.progress_every,
            drop_threshold=args.drop_threshold,
        )
        elapsed = time.time() - t0

        # Stuff stats into the gate
        g.metrics["elapsed_s"] = round(elapsed, 1)
        g.metrics["elapsed_hr"] = round(elapsed / 3600, 2)
        for k, v in stats.to_dict().items():
            g.metrics[f"stats__{k}"] = v
        g.metrics["clips_per_sec"] = round(
            stats.n_clips_total / max(0.01, elapsed), 2
        )

        # Hard-stop checks per master plan section II.4.3
        expect(stats.aggregate_no_pose_fraction < 0.10,
               "aggregate_no_pose_fraction",
               round(stats.aggregate_no_pose_fraction, 4),
               "<0.10", g.failures)
        expect(stats.aggregate_interp_fraction < 0.05,
               "aggregate_interp_fraction",
               round(stats.aggregate_interp_fraction, 4),
               "<0.05", g.failures)
        # Drop rate sanity: <2% of clips should hit the drop_threshold
        drop_rate = stats.n_clips_dropped / max(1, stats.n_clips_total)
        g.metrics["drop_rate"] = round(drop_rate, 4)
        expect(drop_rate < 0.02, "drop_rate", round(drop_rate, 4),
               "<0.02", g.failures)
        # Failure rate: <0.1% should outright fail
        fail_rate = stats.n_clips_failed / max(1, stats.n_clips_total)
        g.metrics["fail_rate"] = round(fail_rate, 4)
        expect(fail_rate < 0.001, "fail_rate", round(fail_rate, 4),
               "<0.001", g.failures)

    return 0


if __name__ == "__main__":
    sys.exit(main())
