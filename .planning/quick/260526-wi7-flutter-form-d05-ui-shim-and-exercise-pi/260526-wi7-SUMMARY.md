---
quick_id: 260526-wi7
slug: flutter-form-d05-ui-shim-and-exercise-pi
status: complete
date: 2026-05-26
branch: fresh-start
tracked_under: 05-HUMAN-UAT.md #2
commits:
  - 0e3b719  # docs: plan
  - 7b5e08a  # C1 D-05 models
  - eb62b89  # C2 rewire upload+live+results
  - ab9fea1  # C3 exercise picker trim
verification: "flutter analyze lib/ -> No issues found (run twice: after C2, after C3)"
human_verification_required: true  # device run on BlueStacks (analyzer proves compile, not runtime)
---

# Quick Task 260526-wi7: Flutter form D-05 UI shim + exercise trim — SUMMARY

Rewired the Flutter form-correction UI from the dead MediaPipe schema to the Phase-5 D-05
PyTorch-Squat schema, and trimmed the exercise picker to the three Fitness-AQA exercises.
`flutter analyze lib/` clean. **Not yet run on a device** — that's the open human check.

## What was built

### C1 — D-05 Flutter models (`lib/models/form_models.dart`) — 7b5e08a
Added `FormError {type, detected, confidence, severityWord, intervals}` (+ `label`/`summary`
helpers), `FormRep {exercise, repNumber?, errors}`, `FormReport {exercise, totalReps, reps,
sessionFeedback}` with `fromUpload` (UploadResponse) + `fromSession` (session_summary) factories.
The upload path synthesizes a client-side feedback line (backend sends none); live uses the
backend's `session_feedback`.

### C2 — Rewire upload + live + results (eb62b89)
- `api_service.uploadFormVideo` → returns `FormReport` (parses `UploadResponse`).
- `form_session_provider` → D-05 state: `report`, live `liveReps`/`lastRep`/`formCheckCount`;
  methods `addLiveRep`/`setReport`. Removed all skeleton/quality/landmarks/mismatch state.
- `form_check_screen` → `_onServerMessage` handles `session_started`/`rep_result`/`session_summary`
  (+ `error`); removed the skeleton overlay, quality label, active-flag chips, phase pill, and
  mismatch banner; shows a **"CHECKS"** counter (honest — periodic, not reps) + a latest-check
  banner with KIE/KFE severity chips.
- `form_results_screen` → renders `FormReport`: summary card (KIE/KFE flagged counts), coach
  feedback line, per-rep KIE/KFE chips. Removed the quality gauge / averageQuality / common-errors.
- `video_upload_screen` → `setReport` + navigates straight to `/form-results` (replay branch dropped).

### C3 — Exercise picker trim (`lib/screens/exercise_selection_screen.dart`) — ab9fea1
Client-side filter of `/api/exercises` to `[squat, dumbbell_overhead_shoulder_press, barbell_row]`,
relabel OHP → "Overhead Press". Single flat list (dropped side/front/either grouping). Only Squat
is tappable → `/guidelines`; OHP + Barbell Row are disabled "Coming soon" tiles (snackbar on tap)
so they never reach the squat-only backend.

## Deviations from plan

1. **[Kept old models — deviation from "remove dead pose models"]** Deleting `FormFrameResult`/
   `FormSessionSummary`/`RepResult`/`FormFrameTimeline` would cascade-break `form_replay_screen.dart`
   + `main.dart`'s `/form-replay` route (they consume those + `SkeletonPainter`). For a demo shim, a
   compile cascade is worse than dead code, so the old models stay and `form_replay` becomes
   unreachable (upload now goes straight to `/form-results`). **Dead-code cleanup deferred**:
   `form_replay_screen.dart`, `widgets/skeleton_painter.dart`, `widgets/mismatch_banner.dart`, and the
   `/form-replay` route are now unused.
2. **OHP → dumbbell entry, relabelled.** `exercises.json` has no barbell-OHP entry; used
   `dumbbell_overhead_shoulder_press` relabelled "Overhead Press" for the coming-soon tile. Exact
   exercise gets wired in Phase 6 — functionally irrelevant now (tile is disabled).

## Verification

- `flutter analyze lib/` → **No issues found** (after C2 and again after C3).
- Grep for removed provider members (`updateFrame`/`setSummary`/`liveLandmarks`/…) across `lib/` →
  no dangling references (only `form_replay`'s own `widget.summary` field, which is the kept old model).
- Backend untouched — `exercises.json` (27-entry SSOT) + `test_exercises_consistency.py` unchanged;
  recommender/plan feature does not read `exercises.json` (verified pre-edit).

## Human verification required (open)

- **Run on BlueStacks** (live + upload Squat). `flutter analyze` proves it compiles, NOT that it
  behaves on a real device. Expected: live camera streams → periodic KIE/KFE banner + CHECKS count →
  end → results screen; upload Squat clip → results screen with per-rep KIE/KFE + feedback.
- **Serving-threshold calibration (05-HUMAN-UAT #1)** still pending — user records 5–10+ labelled
  clips first, then recalibrate the serving KFE/KIE thresholds.

## Known dead code (deferred cleanup, non-blocking)
`form_replay_screen.dart`, `widgets/skeleton_painter.dart`, `widgets/mismatch_banner.dart`, the
`/form-replay` route in `main.dart`, and the old MediaPipe models in `form_models.dart`.
