---
status: partial
phase: 05-backend-inference-integration-squat
source: [05-VERIFICATION.md]
started: 2026-05-26
updated: 2026-05-26
---

# Phase 5 — Human UAT / Demo-Prep Follow-ups

Phase 5's goal (serve the Squat detector via live + upload API) is delivered and verified
(10/10 must-haves, 24/24 tests, real-endpoint E2E passed). These are **non-blocking**
follow-ups for the live demo / dissertation, tracked here so they surface in
`/gsd:progress` and `/gsd:audit-uat`.

## Current Test

[awaiting demo-prep]

## Tests

### 1. Serving-threshold calibration (D-11 follow-up)
expected: On the phone-camera serving framing, the offline thresholds (KIE 0.614 / KFE 0.385)
do not transfer cleanly — KFE 0.385 over-detects (a good clip scored 0.408). The model still
*separates* good vs bad (live WS bad ≈0.85 vs good ≈0.41), so a recalibrated serving threshold
(~0.5–0.55 for KFE) would split them. Gather 5–10+ labeled good/bad phone clips and pick a
serving threshold from their score distribution (do NOT calibrate on the 4–8 clips used so far —
overfitting). The offline 0.6304 (official-split) result is unchanged and remains the academic number.
result: [pending]
why_human: needs a labeled set of real phone-camera clips the user records.

### 2. Live BlueStacks/Flutter end-to-end demo
expected: With the Flutter app on BlueStacks (see reference_bluestacks) pointed at the local
backend, a live squat session streams camera frames to `WS /ws/form-session` and shows per-rep
KIE/KFE feedback; an uploaded clip via `POST /analyze-form-video` returns the result screen.
(The real-endpoint E2E already passed via FastAPI TestClient with real weights + real frames;
this item is the on-device demo dress-rehearsal.)
result: [pending]
why_human: needs the device/emulator + the merged frontend running against the backend.

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps

None — both items are demo-prep enhancements, not phase-goal gaps. Phase 5 goal is met.
