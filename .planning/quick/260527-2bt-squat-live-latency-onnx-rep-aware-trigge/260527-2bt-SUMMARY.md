---
quick_id: 260527-2bt
slug: squat-live-latency-onnx-rep-aware-trigge
status: partial   # Units 1+2 (lag fix) done; Unit 3 (upload replay) pending
date: 2026-05-27
branch: fresh-start
tracked_under: 05-HUMAN-UAT.md #2
commits:
  - 63c7409  # plan
  - 02630af  # Unit 1: ONNX inference
  - 1524082  # Unit 2: motion-gated rep-end live trigger
verification: "import OK; 25 backend tests pass; flutter analyze clean; ONNX parity exact, ensemble 1.52s / single 0.51s"
human_verification_required: true   # device test of live latency + motion-trigger reliability
units_done: [1, 2]
units_pending: [3]
---

# Quick Task 260527-2bt: Squat live latency + rep-aware trigger + upload replay — SUMMARY (partial)

**Units 1+2 (the lag fix) are complete and verified. Unit 3 (upload video replay) is pending** —
handed off here for the device test of the lag fix, which can only be validated on the user's real
squats (motion-trigger thresholds are device-dependent).

## Unit 1 — ONNX Runtime inference (02630af)
- `SquatFormService` auto-exports each `best.pt` → `best.onnx` at load (cached, gitignored) and runs
  the forward through onnxruntime CPU, with a PyTorch fallback if onnxruntime/export is unavailable.
- **Numerically identical** (parity 0.00 at output precision; benchmark max abs diff 2.4e-7) — same
  fp32 math, no quality tradeoff. **~1.6× faster:** single 0.75→0.51s, full ensemble 1.9→1.52s.
- `/health` + startup log report `onnx_enabled`. `onnxruntime`/`onnxscript` added to requirements.
- 25 tests pass (incl. a guarded ONNX/PyTorch parity test that runs against staged weights).

## Unit 2 — motion-gated rep-end live trigger (1524082)
- **Root problem fixed:** the old `LiveWindowTrigger` fired on a 45-frame clock (~7s) and classified
  the rolling window at that tick — so pausing after a rep made it analyze the **standing pose**
  (lag + wrong-window bug).
- `LiveRepDetector` (motion-energy state machine, no pose): IDLE → MOVING (energy > adaptive
  threshold) → settle → fire on rep-end, returning the rep's frame span. `SquatLiveSession` now has
  `push_frame` (fast, no inference) + `classify_pending` (classifies the rep's frames). WS sends
  `{analyzing}` immediately on rep-end, then `{rep_result}` after the ~1.5s ensemble forward.
- **Live defaults to the full ensemble** (no quality tradeoff per the user's constraint);
  `SQUAT_INFERENCE_SEEDS=1` dials to single-seed (~0.5s) without a rebuild.
- Flutter: `analyzing` provider state + spinner banner; "CHECKS" count; honest rep-based copy
  ("Do a rep, then pause — feedback after each rep").

## Net latency change (live)
Before: ~7s clock + 0.86s single-seed, on the WRONG frames (standing pose).
After: feedback right after each rep, on the REP's frames, full-ensemble 1.52s (or 0.51s single via knob).

## Deviations
- Kept legacy `LiveWindowTrigger` (+ its test) as dead-but-tested code to avoid churn; `SquatLiveSession`
  uses `LiveRepDetector`.
- Did NOT add torch/torchvision to requirements.txt (pre-existing omission — the form service imports
  them but they're absent; a fresh `pip install -r requirements.txt` wouldn't load the form subsystem).
  Out of this task's scope + CPU-vs-CUDA pin complexity; flagged for a separate fix.

## Human verification required (the device test — gates judging)
1. **Backend with weights** running (`uvicorn ...`); `/health` → `onnx_enabled: true`, `seeds_loaded: 3`.
2. **Live on BlueStacks:** start session standing still (establishes the motion baseline), do a squat,
   pause → expect an "Analyzing rep…" spinner then a KIE/KFE verdict (~1.5s). CHECKS increments per rep.
   - **Motion-trigger thresholds (`LIVE_REP_*` in rep_segmenter.py) are device-dependent** — if it
     mis-fires (fires on fidgeting) or misses reps, report it and we tune the constants on your camera.
   - If 1.5s feels slow: set `SQUAT_INFERENCE_SEEDS=1` (→ ~0.5s, −0.018 macro, invisible live).

## Unit 3 — PENDING (next focused build)
Upload video replay: backend returns each rep's `[start,end]` interval in `UploadResponse`; new Flutter
replay screen plays the clip back with per-rep KIE/KFE verdicts synced to playback + the model's
lower-body crop box overlaid ("eyes of the model"). Resume with `/gsd:quick resume squat-live-latency-onnx-rep-aware-trigge`.
