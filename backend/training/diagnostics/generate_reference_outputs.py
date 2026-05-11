"""Generate the reference outputs that the Colab D2 notebook must reproduce.

Runs the local CPU MediaPipe pipeline on:
  * Good_Squats.mp4         (the canonical user-recorded squat — the same
                             clip used by the live extractor smoke test)
  * 5 random QEVD clips      (one from each of Parts 1, 2, 3, 4 + a 5th
                             random from anywhere — proves QEVD-format
                             MP4s also produce identical features)

Saves each clip's full ClipFeatures into a single .npz under
``backend/training/diagnostics/references/``. Colab notebook imports the
bundle, loads this .npz, runs the SAME extractor on the SAME clips, and
asserts every array is element-wise identical (within float tolerance).

If the assertion fails -> stop. Numerical drift between local and Colab
re-introduces v4's distribution-mismatch bug.
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from backend.training.preprocessing.qevd_extractor import extract_clip

REFS_DIR = REPO_ROOT / "backend" / "training" / "diagnostics" / "references"
REFS_DIR.mkdir(parents=True, exist_ok=True)

GOOD_SQUATS = Path("C:/Users/tsh_x/Downloads/Good_Squats.mp4")
QEVD_ROOT = REPO_ROOT / "backend" / "data" / "qevd_raw" / "fit300k"

random.seed(42)
PART_DIRS = [QEVD_ROOT / f"QEVD-FIT-300k-Part-{n}" for n in (1, 2, 3, 4)]


def _pick_random_clip(part_dir: Path) -> Path:
    candidates = sorted(part_dir.glob("*.mp4"))
    return random.choice(candidates)


def main() -> None:
    if not GOOD_SQUATS.exists():
        print(f"ERROR: {GOOD_SQUATS} missing", file=sys.stderr)
        sys.exit(1)
    for d in PART_DIRS:
        if not d.exists():
            print(f"ERROR: {d} missing — extract Parts 1-4 first", file=sys.stderr)
            sys.exit(1)

    # Pick 1 random clip from each Part; 5th from Part 1 again
    clips_to_process = [GOOD_SQUATS] + [_pick_random_clip(d) for d in PART_DIRS]
    print("Reference clips:")
    for p in clips_to_process:
        print(f"  {p.name:<24s}  ({p.stat().st_size / 1e6:.2f} MB)")
    print()

    # Stage them into a single dir so the notebook can find them via name
    staging = REFS_DIR / "input_clips"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for p in clips_to_process:
        shutil.copy2(p, staging / p.name)

    # Process each, store in a single .npz (one entry per clip)
    out = {}
    for p in clips_to_process:
        print(f"Processing {p.name} ...", end=" ", flush=True)
        feats = extract_clip(p)
        key = p.stem
        out[f"{key}__pose_canon"]       = feats.pose_canon
        out[f"{key}__angles_raw"]       = feats.angles_raw
        out[f"{key}__fps_native"]       = np.float32(feats.fps_native)
        out[f"{key}__status_per_frame"] = feats.status_per_frame
        out[f"{key}__no_pose_fraction"] = np.float32(feats.no_pose_fraction)
        print(f"T={feats.pose_canon.shape[0]}  no_pose={feats.no_pose_fraction:.4f}")

    out["reference_clips"] = np.array(
        [p.name for p in clips_to_process], dtype=str,
    )

    out_path = REFS_DIR / "reference_outputs.npz"
    np.savez_compressed(out_path, **out)
    print()
    print(f"Wrote {out_path}  ({out_path.stat().st_size / 1e3:.1f} KB)")
    print(f"Clips staged at {staging}/")
    print()
    print("Colab notebook will:")
    print("  1. Run the SAME extract_clip on each input clip")
    print("  2. Load this .npz")
    print("  3. Assert every array element-wise identical (within 1e-5)")


if __name__ == "__main__":
    main()
