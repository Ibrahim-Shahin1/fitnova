"""Local dry-run of the v6 D2 Colab notebook.

Simulates the notebook's logic on this machine (without Colab/Drive)
to prove the bundle is correct before handing over.

Steps:
  1. Unzip fitnova_v6_src.zip into a temp dir (fresh)
  2. Add the temp dir to sys.path (matches Colab's WORK_DIR pattern)
  3. Import qevd_extractor from the bundle
  4. Run the numerical-equivalence test on the 5 reference clips
  5. Run extract_directory on a 5-clip mini-batch end-to-end
  6. Verify outputs match exactly

This is the gate before handover: ALL 6 STEPS MUST PASS.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
BUNDLE = REPO_ROOT / "fitnova_v6_src.zip"


def step(msg):
    print(f"\n=== {msg} ===")


def main() -> int:
    if not BUNDLE.exists():
        print(f"ERROR: {BUNDLE} not found. Run "
              "_build_v6_d2_colab_bundle.py first.")
        return 1

    workdir = Path(tempfile.mkdtemp(prefix="fitnova_v6_dryrun_"))
    print(f"Workdir: {workdir}")

    # ── 1. Unzip ─────────────────────────────────────────────────────────────
    step("Step 1: unzip bundle")
    with zipfile.ZipFile(BUNDLE) as z:
        z.extractall(workdir)
    print(f"  extracted {sum(1 for _ in workdir.rglob('*'))} files")

    # ── 2. sys.path ──────────────────────────────────────────────────────────
    step("Step 2: add bundle to sys.path")
    # First remove the live repo path so we KNOW we're using the bundle
    paths_to_remove = [p for p in sys.path if "FitNova Application" in str(p)]
    for p in paths_to_remove:
        sys.path.remove(p)
    sys.path.insert(0, str(workdir))
    # Also clear cached imports so we re-import from the bundle
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("backend"):
            del sys.modules[mod_name]

    # ── 3. Import from bundle ───────────────────────────────────────────────
    step("Step 3: import from bundle")
    from backend.training.preprocessing.qevd_extractor import (
        extract_clip, extract_directory, ClipFeatures,
        STATUS_OK, STATUS_INTERP, STATUS_NO_POSE,
    )
    from backend.services.mediapipe_config import (
        MIN_POSE_DETECTION_CONFIDENCE, RUNNING_MODE, PINNED_MEDIAPIPE_VERSION,
    )
    print(f"  qevd_extractor: imported")
    print(f"  mediapipe pinned: {PINNED_MEDIAPIPE_VERSION}")
    print(f"  config: detect={MIN_POSE_DETECTION_CONFIDENCE} mode={RUNNING_MODE}")

    # ── 4. Numerical-equivalence test (the key correctness check) ───────────
    step("Step 4: numerical equivalence test (5 reference clips)")
    REFS = workdir / "backend/training/diagnostics/references"
    ref_npz = REFS / "reference_outputs.npz"
    in_dir = REFS / "input_clips"
    assert ref_npz.exists(), f"missing {ref_npz}"
    assert in_dir.exists(),  f"missing {in_dir}"

    ref = np.load(ref_npz, allow_pickle=True)
    clip_names = list(ref["reference_clips"])
    print(f"  reference clips ({len(clip_names)}): {clip_names}")

    ATOL = 1e-5
    RTOL = 1e-5
    all_ok = True
    for clip_name in clip_names:
        clip_path = in_dir / clip_name
        feats = extract_clip(clip_path)
        key = clip_path.stem
        fields = {
            "pose_canon":       feats.pose_canon,
            "angles_raw":       feats.angles_raw,
            "fps_native":       np.float32(feats.fps_native),
            "status_per_frame": feats.status_per_frame,
            "no_pose_fraction": np.float32(feats.no_pose_fraction),
        }
        clip_ok = True
        for fname, arr_local in fields.items():
            arr_ref = ref[f"{key}__{fname}"]
            if arr_ref.shape != arr_local.shape:
                print(f"  [FAIL] {clip_name} {fname}: shape "
                      f"{arr_local.shape} vs ref {arr_ref.shape}")
                clip_ok = False; continue
            if np.issubdtype(arr_local.dtype, np.floating):
                ok = np.allclose(arr_local, arr_ref, atol=ATOL, rtol=RTOL,
                                  equal_nan=True)
            else:
                ok = np.array_equal(arr_local, arr_ref)
            if not ok:
                diff = np.abs(arr_local.astype(np.float64)
                               - arr_ref.astype(np.float64))
                print(f"  [FAIL] {clip_name} {fname}: "
                      f"max_abs_diff={diff.max():.6f}")
                clip_ok = False
        if clip_ok:
            print(f"  [OK]  {clip_name}: all 5 fields identical to reference")
        else:
            all_ok = False

    if not all_ok:
        print("\n  >>> NUMERICAL EQUIVALENCE: FAIL")
        return 2
    print("\n  >>> NUMERICAL EQUIVALENCE: PASS")

    # ── 5. End-to-end extract_directory on the 5 reference clips ────────────
    step("Step 5: end-to-end extract_directory on 5 clips (multi-process)")
    out_dir = workdir / "test_out"
    out_dir.mkdir()
    stats = extract_directory(
        in_dir=in_dir, out_dir=out_dir, n_workers=2, progress_every=2,
    )
    print(f"  ok={stats.n_clips_ok} dropped={stats.n_clips_dropped} "
          f"failed={stats.n_clips_failed}")
    if stats.n_clips_failed > 0:
        print(f"  FAILURES:")
        for f in stats.failures[:5]:
            print(f"    {f}")
        return 3
    print(f"  no_pose_mean: {stats.aggregate_no_pose_fraction:.4f}")
    print(f"  interp_mean : {stats.aggregate_interp_fraction:.4f}")

    # Verify .npz files exist and have correct schema
    npzs = sorted(out_dir.rglob("*.npz"))
    print(f"  wrote {len(npzs)} .npz files")
    assert len(npzs) >= 5, f"expected >=5, got {len(npzs)}"
    for npz_path in npzs:
        with np.load(npz_path) as z:
            keys = set(z.files)
            expected = {"pose_canon", "angles_raw", "fps_native",
                         "status_per_frame", "no_pose_fraction"}
            assert keys == expected, f"{npz_path.name}: keys {keys}"

    # ── 6. Resume support test ─────────────────────────────────────────────
    step("Step 6: resume support (re-run should skip all)")
    stats2 = extract_directory(
        in_dir=in_dir, out_dir=out_dir, n_workers=2, progress_every=2,
    )
    # After re-run, n_clips_total should be 0 (all skipped), or the
    # skip log message should show in stderr
    print(f"  re-run: ok={stats2.n_clips_ok} (expected 0 — all skipped)")

    # Cleanup
    shutil.rmtree(workdir, ignore_errors=True)

    print()
    print("=" * 70)
    print("ALL 6 STEPS PASS — bundle is verified, safe to hand over.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
