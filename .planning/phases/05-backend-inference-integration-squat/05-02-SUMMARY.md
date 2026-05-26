---
phase: 05-backend-inference-integration-squat
plan: "02"
subsystem: backend-inference
tags: [squat, domain-shift, preprocessing, knee-aware-crop, pytorch, serving]
dependency_graph:
  requires: [05-01]
  provides: [kneeaware_spatial_val, D-11-validated-serving-crop]
  affects: [squat_form_service, probe_squat_inference, 05-03-upload-endpoint]
tech_stack:
  added: []
  patterns:
    - "Portrait-conditional lower-body pre-crop before spatial_val (serving only)"
    - "PORTRAIT_LOWERBODY_TOP_FRAC = 0.42 — module constant, tuned on 4 real phone clips"
key_files:
  created: []
  modified:
    - backend/services/clip_decode.py
    - backend/services/squat_form_service.py
    - backend/scripts/probe_squat_inference.py
    - backend/tests/test_squat_form_api.py
decisions:
  - "D-11 knee-aware crop: portrait-conditional lower-body pre-crop (0.42*H) applied in serving path; landscape unchanged (offline-eval parity)"
  - "PORTRAIT_LOWERBODY_TOP_FRAC = 0.42 serving heuristic — tuned on 4 real clips, not re-tuned from offline data"
  - "Offline thresholds (KIE 0.614 / KFE 0.385) NOT changed — serving-framing threshold calibration deferred"
metrics:
  duration: "< 1 hour"
  completed: "2026-05-26"
  tasks_completed: 3
  files_modified: 4
---

# Phase 5 Plan 02: Squat Serving Weights + Domain-Shift Probe Summary

**One-liner:** D-11-validated portrait knee-aware serving crop (`kneeaware_spatial_val`) wired into `SquatFormService.classify_clip` and probe, restoring KFE/KIE discrimination on phone video.

---

## What Was Done

### Context — What the Earlier Tasks in This Plan Produced

Plan 05-02 was originally three tasks: (1) stage the 3 Phase-4 `best.pt` weights from Google Drive (human action), (2) document them in a README + `.gitignore` entry, (3) run a latency probe + real-clip domain-shift test. Tasks 1–3 were executed in a prior session and produced the following **validated findings** that this execution round closes:

**Weights pre-staged** — 3 Phase-4 MD-SSL `best.pt` files at `backend/models/form_model_squat_md/seed{42,1337,7}/best.pt` (each ~359 MB full checkpoint; service reads `model_state_dict` only). `.gitignore` protects them (`seed*/` + `**/*.pt` pattern, negation keeps README). `README.md` committed.

**CPU latency measured (D-08)** — single-seed: ~0.85 s, ensemble (3-seed): ~2.9 s on the target Windows CPU. Confirms the D-08 recommendation: single-seed for live WebSocket, 3-seed ensemble for the upload endpoint.

**Decode deviation fixed** — `torchvision 0.27` removed `read_video` / `read_video_timestamps`. Added `backend/services/clip_decode.py` (`decode_clip_cv2`, cv2 BGR→RGB parity-approx) in a prior commit. This deviation affects the upload endpoint (05-03) too.

### D-11 Domain-Shift Finding (The Headline)

Running the probe on real phone-camera squat clips exposed a critical preprocessing failure:

**Root cause:** Portrait phone video (e.g. 576×1024) fed through the standard `spatial_val` path (short-side resize → 128, then 112² center-crop) keeps the MIDDLE horizontal band — the torso. The knees are completely outside the crop window. With no knees visible, the R(2+1)D-18 model returned near-random scores (all seeds ~0.1 for both KIE and KFE, no separation between good and bad form clips).

**Fix validated on 4 real clips:** A lower-body pre-crop (rows starting at 42% of H, producing a square region covering the hips-down) before calling `spatial_val` restores the knees to the center of the 112² crop. Post-fix discrimination:

| Clip type | KFE confidence | KIE confidence |
|-----------|---------------|----------------|
| Good form | ~0.49         | ~0.06          |
| Bad form  | ~0.60         | ~0.13–0.24     |

KFE separation is directionally correct. KIE is directionally correct (good < bad) though modest — consistent with its ~47% offline recall. Both heads were at noise floor before the fix.

**Landscape unchanged:** Fitness-AQA dataset clips are landscape gym video. The offline 0.6304 F1 was measured with plain `spatial_val`. The fix is portrait-conditional: `H > W` triggers the pre-crop; `W >= H` passes through to `spatial_val` unchanged, preserving offline-eval reproducibility.

### Implementation — This Execution Round

**`backend/services/clip_decode.py`**

- Added module-level constant `PORTRAIT_LOWERBODY_TOP_FRAC = 0.42` (named, documented, with rationale comment).
- Added function `kneeaware_spatial_val(frames_tchw_uint8, *, crop_size=112, resize_short=128)`:
  - Portrait (`H > W`): crops rows `[int(0.42*H) : int(0.42*H) + min(H-int(0.42*H), W)]` to produce a square lower-body region, then calls `spatial_val` on it.
  - Landscape/square (`W >= H`): passes through directly to `spatial_val` — no change.
  - Returns `[3, T, 112, 112]` float32 Kinetics-normalised (same contract as `spatial_val`).
  - Module docstring and function docstring explain WHY (portrait excludes knees), the parity rationale (landscape unchanged = offline-eval-faithful), and that it is the D-11-validated fix.
- Added import of `spatial_val` from `backend.training.aqa.datasets.transforms` at module top.

**`backend/services/squat_form_service.py`**

- Replaced `from backend.training.aqa.datasets.transforms import spatial_val` with `from backend.services.clip_decode import kneeaware_spatial_val`.
- In `classify_clip`: replaced `clip = spatial_val(...)` with `clip = kneeaware_spatial_val(...)`.
- Updated the anti-pattern guard comment inline and the module docstring preprocessing contract bullet.
- Ensemble logic, thresholds, `_build_response`, `_neutral_response` — all UNCHANGED.

**`backend/scripts/probe_squat_inference.py`**

- Per-seed manual path: replaced `from backend.training.aqa.datasets.transforms import spatial_val` + `spatial_val(frames_tchw)` with `from backend.services.clip_decode import kneeaware_spatial_val` + `kneeaware_spatial_val(frames_tchw)`.
- The "FULL D-05 RESPONSE via classify_clip" section already routes through `classify_clip`, which now uses `kneeaware_spatial_val` — so per-seed scores and the full D-05 response are now consistent.
- Module docstring updated with a D-11 UPDATE block.

**`backend/tests/test_squat_form_api.py`**

Two new tests added:

1. `test_kneeaware_landscape_is_spatial_val` — synthetic landscape clip `[8, 3, 120, 200]` (W > H); asserts `kneeaware_spatial_val` output is element-equal to `spatial_val` output. Parity guard.
2. `test_kneeaware_portrait_differs_and_shape` — synthetic portrait clip `[8, 3, 200, 120]` (H > W); asserts (a) output shape is `(3, 8, 112, 112)` float32, (b) output is NOT element-equal to `spatial_val` on the same input (proves the lower-body pre-crop is applied).

All 8 tests green: `python -m pytest backend/tests/test_squat_form_api.py -x -q` → `8 passed in 19.30s`.

---

## Threshold Note

The offline-tuned thresholds (KIE 0.614, KFE 0.385) were tuned on the Fitness-AQA dataset distribution (landscape gym video, CVCSPC canonical split). On the 4 validated phone clips with the knee-aware crop, KFE scores ranged ~0.49–0.60 — all clips exceed the 0.385 threshold (i.e. KFE detected for every clip). The separation between good and bad is real but the serving decision threshold needs calibration on more than 4 clips before the threshold should be adjusted.

**Action deferred to serving-tuning:** collect ~20+ real phone clips, measure separation, and set a serving-specific threshold if needed. The offline 0.6304 result is unchanged; thresholds in `squat_form_service.py` are unchanged (D-01: val-tuned, locked).

---

## Deviations from Plan

### Auto-fixed Issues

The plan's three original tasks were completed in prior sessions. This execution round implements the D-11 fix that was validated during Task 3.

**1. [Rule 2 - Missing Critical Functionality] Knee-aware serving crop**
- **Found during:** Task 3 real-clip probe (D-11 domain-shift test)
- **Issue:** Standard `spatial_val` center-crop on portrait phone video excludes the knees entirely, zeroing the model's ability to discriminate KIE/KFE. Without this fix the serving path is non-functional on the most common user input (phone portrait video).
- **Fix:** Added `kneeaware_spatial_val` in `clip_decode.py`; wired into `classify_clip` and probe; added 2 tests.
- **Files modified:** `backend/services/clip_decode.py`, `backend/services/squat_form_service.py`, `backend/scripts/probe_squat_inference.py`, `backend/tests/test_squat_form_api.py`
- **Commit:** `36010ea`

---

## Commits

| Hash | Message |
|------|---------|
| `36010ea` | `feat(05-02): knee-aware serving crop + tests (D-11 domain-shift fix)` |

(Prior commits for Tasks 1–3 from earlier sessions: README + .gitignore staging, probe script, clip_decode cv2 decode deviation.)

---

## Self-Check: PASSED

- `backend/services/clip_decode.py` — exists, contains `kneeaware_spatial_val` and `PORTRAIT_LOWERBODY_TOP_FRAC`
- `backend/services/squat_form_service.py` — `classify_clip` uses `kneeaware_spatial_val`
- `backend/scripts/probe_squat_inference.py` — per-seed path uses `kneeaware_spatial_val`
- `backend/tests/test_squat_form_api.py` — `test_kneeaware_landscape_is_spatial_val` + `test_kneeaware_portrait_differs_and_shape` present
- `git log --oneline` confirms `36010ea` exists
- `8 passed` on full test file
