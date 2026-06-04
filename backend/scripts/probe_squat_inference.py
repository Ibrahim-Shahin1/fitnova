"""Latency probe + domain-shift diagnostic for the Squat MD-SSL ensemble.

Two checks:
  - measure per-clip CPU forward latency (single-seed vs 3-seed ensemble) on the
    target machine with the real staged weights.
  - run the model end-to-end on a real user phone-camera squat clip and print
    KIE/KFE confidences + raw ensemble sigmoid scores for domain-shift diagnosis.

The per-seed path uses `kneeaware_spatial_val` (from `backend.services.clip_decode`)
instead of the raw `spatial_val` call. This matches `SquatFormService.classify_clip`
exactly: portrait lower-body pre-crop applied, landscape unchanged. The full
classify_clip sanity check (Section B bottom) will therefore agree with the
per-seed scores printed above it.

Usage:
    python -m backend.scripts.probe_squat_inference               # latency only (synthetic clip)
    python -m backend.scripts.probe_squat_inference path/to.mp4  # latency + real-clip test

Output goes to stdout.

IMPORTANT: fp32 only throughout - do NOT add .half() calls.
fp16 hangs on CPU with PyTorch 2.12 on Windows.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap - locate model directory relative to repo root
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MODEL_DIR = _REPO_ROOT / "backend" / "models" / "form_model_squat_md"

# ---------------------------------------------------------------------------
# Helper - median of 3 timed runs
# ---------------------------------------------------------------------------

def _median_time(fn, *, n_runs: int = 3) -> float:
    """Run `fn()` `n_runs` times and return the median wall-clock seconds."""
    times: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    clip_path: str | None = sys.argv[1] if len(sys.argv) > 1 else None

    # -- 1. Load service ---------------------------------------------------------
    print("=" * 70)
    print("Squat inference probe - latency + domain-shift")
    print("=" * 70)
    print(f"Model dir : {_MODEL_DIR}")

    from backend.services.squat_form_service import SquatFormService

    svc = SquatFormService(model_dir=str(_MODEL_DIR))

    if not svc.model_ready:
        print()
        print("ERROR: SquatFormService.model_ready is False - weights not staged.")
        print("Stage the 3 best.pt files from Google Drive:")
        print("  MyDrive/FitNova/checkpoints/phase04/md_finetune_seed{42,1337,7}/best.pt")
        print("  -> backend/models/form_model_squat_md/seed{42,1337,7}/best.pt")
        print("Then re-run this probe.")
        sys.exit(1)

    print(f"Seeds loaded: {len(svc._models)}/3  |  model_ready: {svc.model_ready}")

    # -- 2. LATENCY -------------------------------------------------------
    print()
    print("-" * 70)
    print("Section A - CPU latency")
    print("-" * 70)
    print("Building synthetic clip: np.zeros((32, 3, 128, 128), uint8) ...")
    print("(All-zeros clip exercises the full preprocessing + forward path; fp32 only)")
    print()

    synthetic_clip = np.zeros((32, 3, 128, 128), dtype=np.uint8)

    # Single-seed (n_seeds=1) - uses the first loaded model (seed42)
    print("Timing classify_clip(n_seeds=1) over 3 runs ...")
    single_median = _median_time(
        lambda: svc.classify_clip(synthetic_clip, n_seeds=1),
        n_runs=3,
    )
    print(f"  Single-seed median: {single_median:.3f} s")

    # 3-seed ensemble
    print("Timing classify_clip(n_seeds=3) over 3 runs ...")
    ensemble_median = _median_time(
        lambda: svc.classify_clip(synthetic_clip, n_seeds=3),
        n_runs=3,
    )
    print(f"  Ensemble (3-seed) median: {ensemble_median:.3f} s")

    print()
    print("LATENCY SUMMARY")
    print(f"  single-seed (seed42, ~0.93 s expected) : {single_median:.3f} s")
    print(f"  3-seed ensemble   (~2.8 s expected)    : {ensemble_median:.3f} s")
    print(f"  extrapolated speedup: {ensemble_median / single_median:.2f}x (ideally ~3.0x)")

    # Compare against the reference baseline timings
    ref_single = 0.93
    ref_ensemble = 2.8
    delta_single = single_median - ref_single
    delta_ensemble = ensemble_median - ref_ensemble
    sign_s = "+" if delta_single >= 0 else ""
    sign_e = "+" if delta_ensemble >= 0 else ""
    print(f"  vs reference baseline: single {sign_s}{delta_single:.3f} s, ensemble {sign_e}{delta_ensemble:.3f} s")

    # -- 3. REAL CLIP -----------------------------------------------------
    if clip_path is None:
        print()
        print("-" * 70)
        print("Section B - real-clip domain-shift test")
        print("-" * 70)
        print("No clip path provided. Provide a real phone-camera squat .mp4:")
        print("  python -m backend.scripts.probe_squat_inference path/to/squat.mp4")
        print("Expected: short ~3 s single-rep clip, side-ish angle, good lighting.")
        print()
        print("Probe complete (latency section only).")
        return

    print()
    print("-" * 70)
    print("Section B - real-clip domain-shift test")
    print("-" * 70)
    print(f"Clip path: {clip_path}")

    import cv2
    import torch
    from backend.training.aqa.datasets.transforms import uniform_sample_indices
    from backend.services.clip_decode import decode_clip_cv2

    # Frame count via cv2 — torchvision 0.27 removed read_video / read_video_timestamps,
    # so the training decode_clip is unusable on the serving machine. cv2 is the only
    # available decoder (see backend/services/clip_decode.py).
    print("Probing frame count via cv2.VideoCapture ...")
    _cap = cv2.VideoCapture(clip_path)
    total_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = float(_cap.get(cv2.CAP_PROP_FPS))
    _cap.release()
    fps = video_fps if video_fps else 30.0
    duration_s = total_frames / fps if fps > 0 else 0.0
    print(f"  total_frames: {total_frames}  |  fps: {fps:.1f}  |  duration: {duration_s:.2f} s")

    if total_frames < 32:
        print(f"WARNING: clip has only {total_frames} frames (< 32). Inference may degrade.")
        print("For best results use a clip with >= 32 frames (~1 s at 30 fps).")

    # Sample 32 frames deterministically (jitter=0, serve-time mode)
    print("Sampling 32 frames with uniform_sample_indices(jitter=0) ...")
    indices = uniform_sample_indices(total_frames, target=32, jitter=0)
    print(f"  indices[:8]: {indices[:8].tolist()} ... {indices[-4:].tolist()}")

    # Decode via cv2 (BGR->RGB, same sampled indices) — parity-approx of the training
    # read_video decode; spatial_val (resize/crop/Kinetics-norm) runs identically downstream.
    print("Decoding clip via cv2 (decode_clip_cv2) ...")
    frames_tchw = decode_clip_cv2(clip_path, indices)  # uint8 [32, 3, H, W] RGB
    print(f"  decoded shape: {tuple(frames_tchw.shape)}  dtype: {frames_tchw.dtype}")

    frames_np = frames_tchw.numpy()  # [32, 3, H, W] uint8

    # -- Raw ensemble sigmoid scores (before thresholding) --------------------
    # Run each seed individually to get per-seed logits, then aggregate.
    from backend.training.aqa.eval.ensemble import aggregate_sigmoid_mean
    from backend.services.squat_form_service import SquatFormService
    import torch.nn as nn

    print()
    print("Running per-seed forwards (fp32 only - no .half()) ...")

    # Reuse the service's preprocessing via classify_clip internals - but we need
    # raw per-seed scores.  Use kneeaware_spatial_val (the same path classify_clip
    # uses) so the per-seed numbers here match the classify_clip output.
    from backend.services.clip_decode import kneeaware_spatial_val

    clip_tensor = kneeaware_spatial_val(frames_tchw)  # [3, 32, 112, 112] float32
    batch = clip_tensor.unsqueeze(0)                   # [1, 3, 32, 112, 112]

    per_seed_logits: list[np.ndarray] = []
    per_seed_sigmoid: list[tuple[float, float]] = []

    with torch.no_grad():
        for i, m in enumerate(svc._models):
            logits = m(batch)              # [1, 2]
            sig = torch.sigmoid(logits)   # [1, 2]
            kie_raw = float(sig[0, 0].item())
            kfe_raw = float(sig[0, 1].item())
            per_seed_logits.append(logits.numpy())
            per_seed_sigmoid.append((kie_raw, kfe_raw))
            print(f"  seed[{i}] logits: KIE={float(logits[0,0]):.4f}, KFE={float(logits[0,1]):.4f}"
                  f"  |  sigmoid: KIE={kie_raw:.4f}, KFE={kfe_raw:.4f}")

    # Ensemble via aggregate_sigmoid_mean (identical path to classify_clip)
    ensemble_scores = aggregate_sigmoid_mean(per_seed_logits)   # [1, 2]
    kie_s = float(ensemble_scores[0, 0])
    kfe_s = float(ensemble_scores[0, 1])

    # Apply thresholds
    kie_thresh = svc.kie_threshold     # 0.614
    kfe_thresh = svc.kfe_threshold     # 0.385
    kie_det = kie_s >= kie_thresh
    kfe_det = kfe_s >= kfe_thresh

    def _sev(conf: float, det: bool) -> str:
        if not det:
            return "none"
        if conf >= 0.80:
            return "strong"
        if conf >= 0.65:
            return "moderate"
        return "possible"

    kie_word = _sev(kie_s, kie_det)
    kfe_word = _sev(kfe_s, kfe_det)

    # Also run classify_clip directly (sanity-check same result)
    direct_result = svc.classify_clip(frames_np, n_seeds=len(svc._models))

    print()
    print("-" * 70)
    print("Real-clip results (response schema)")
    print("-" * 70)
    print(f"  exercise: squat")
    print(f"  KIE - confidence: {kie_s:.4f}  detected: {kie_det}"
          f"  severity: {kie_word}  threshold: {kie_thresh}")
    print(f"  KFE - confidence: {kfe_s:.4f}  detected: {kfe_det}"
          f"  severity: {kfe_word}  threshold: {kfe_thresh}")

    print()
    print("RAW ENSEMBLE SIGMOID SCORES (mean-of-sigmoids before thresholding)")
    print(f"  KIE raw: {kie_s:.6f}  (threshold {kie_thresh} - {'ABOVE' if kie_det else 'BELOW'})")
    print(f"  KFE raw: {kfe_s:.6f}  (threshold {kfe_thresh} - {'ABOVE' if kfe_det else 'BELOW'})")

    print()
    print("PER-SEED SIGMOID SCORES")
    seed_names = [42, 1337, 7]
    for i, (kie_si, kfe_si) in enumerate(per_seed_sigmoid):
        seed_label = seed_names[i] if i < len(seed_names) else i
        print(f"  seed{seed_label}: KIE={kie_si:.4f}  KFE={kfe_si:.4f}")

    print()
    print("Full response (via classify_clip - sanity check)")
    import json
    print(json.dumps(direct_result, indent=2))

    print()
    print("-" * 70)
    print("DOMAIN-SHIFT INTERPRETATION GUIDE")
    print("-" * 70)
    print("Expected behavior on a real squat clip:")
    print("  KFE: strong on clips with knees-forward travel (~97% recall offline)")
    print("  KIE: modest (~47% recall); threshold 0.614 is intentionally conservative")
    print()
    print("Healthy signs:")
    print("  - KFE confidence varies with actual knee forward travel")
    print("  - Scores are NOT all ~0.0 (model seeing nothing) or all ~1.0 (model saturated)")
    print("  - Results are stable (same clip -> same scores each run)")
    print()
    print("Problem signs (report these at the checkpoint):")
    print("  - Both KIE AND KFE ~0.0: preprocessing mismatch - check spatial_val is used")
    print("  - Both KIE AND KFE >=0.99: model may be in train mode (check .eval() called)")
    print("  - Large variance across the 3 per-seed scores: unexpected divergence between seeds")
    print()
    print("Probe complete.")


if __name__ == "__main__":
    main()
