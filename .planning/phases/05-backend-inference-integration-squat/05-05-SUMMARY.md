---
phase: 05-backend-inference-integration-squat
plan: "05"
subsystem: infra
tags: [pytorch, archival, visualization, matplotlib, opencv, git-mv]

requires:
  - phase: 05-backend-inference-integration-squat
    provides: Plans 03-04 removed FormAnalyzer/FormSession from app.py — archive now safe

provides:
  - "backend/_archive_form_v4_v6/: reversible archive of old TF+MediaPipe form subsystem (D-06, API-01)"
  - "backend/scripts/render_squat_result.py: headless matplotlib visualization script for KIE/KFE overlay"
  - ".planning/.../figures/squat_result_overlay.png: rendered supervisor deliverable figure"

affects: [phase-06-verification, dissertation-appendix]

tech-stack:
  added: []
  patterns:
    - "git mv for reversible archival: preserves history, file moves are undoable with git mv back"
    - "git rm --cached for untracking already-tracked binaries at new path (post-git-mv gitignore)"
    - "matplotlib.use(Agg) + decode_clip_cv2 for headless visualization on torchvision 0.27"

key-files:
  created:
    - "backend/_archive_form_v4_v6/README.md"
    - "backend/scripts/render_squat_result.py"
    - ".planning/phases/05-backend-inference-integration-squat/figures/squat_result_overlay.png"
  modified:
    - ".gitignore (archive weight gitignore rules)"

key-decisions:
  - "git rm --cached after git mv to untrack .weights.h5 in archive — preserves files on disk but removes from git index so gitignore rules apply going forward"
  - "Also archived backend/tests/test_exercise_sanity.py and validate_user_squats.py — they reference archived modules; Rule 2 addition to prevent dangling imports in test discovery"
  - "viz script uses decode_clip_cv2 + get_frame_count_and_fps instead of dead read_video_timestamps (torchvision 0.27 removed it) — parity with probe_squat_inference.py"
  - "Figure saved to .planning figures/ dir for supervisor demo access; also committed to git for dissertation"

requirements-completed: [API-01]

duration: 65min
completed: 2026-05-26
---

# Phase 05 Plan 05: Archive Old Form Subsystem + Visualization Summary

**Reversible git mv archive of TF+MediaPipe form v4-v6 subsystem to backend/_archive_form_v4_v6/; cv2-backed matplotlib visualization renders KIE/KFE overlay on BadSquat_45.mp4 (KFE detected=True confidence=0.4817)**

## Performance

- **Duration:** ~65 min
- **Started:** 2026-05-26T15:50:00Z
- **Completed:** 2026-05-26T17:05:00Z
- **Tasks:** 2
- **Files modified:** 14 (moves) + 3 created + 1 gitignore update

## Accomplishments

- Archived 5 old form services + 3 co-located tests + 4 TF model arch files + 4 train scripts + 5 weight dirs via `git mv` to `backend/_archive_form_v4_v6/` — reversible, history-preserving, dissertation evidence
- Added `.gitignore` rules for archived `.weights.h5` / `.task` / smoke dirs; ran `git rm --cached` to untrack already-moved weight binaries (11 files)
- `backend.app` imports cleanly and 24/24 tests pass after archival; `backend/training/aqa/**` and all recommender + new frontend services UNTOUCHED
- `render_squat_result.py` uses cv2 decode helpers + SquatFormService.classify_clip; rendered figure committed to figures/ directory

## Task Commits

1. **Task 1: Archive old TF+MediaPipe form subsystem** - `cf036c3` (feat)
2. **Task 2: Visualization script + rendered figure** - `231880e` (feat)

## Files Created/Modified

- `backend/_archive_form_v4_v6/README.md` — manifest documenting every archived path, reason, reversibility
- `backend/_archive_form_v4_v6/services/` — form_analyzer, form_session, form_geometry, mediapipe_config, exercise_sanity + 3 co-located tests
- `backend/_archive_form_v4_v6/tests/` — test_exercise_sanity.py, validate_user_squats.py (Rule 2 addition)
- `backend/_archive_form_v4_v6/training/models/` — mt_tcn, st_gcn, st_gcn_v6, st_gcn_v6_1 + 2 tests
- `backend/_archive_form_v4_v6/training/train/` — train_form_model.py, train_form_model_v6.py, train_form_model_v6_1.py, train_ssl_pretrain.py
- `backend/_archive_form_v4_v6/models/` — form_model, form_model_v5, v5_1, v5_2, v6 (metadata tracked; .weights.h5 gitignored)
- `backend/scripts/render_squat_result.py` — headless matplotlib visualization script
- `.planning/.../figures/squat_result_overlay.png` — rendered figure (646 KB)
- `.gitignore` — archive weight ignore rules added

## Decisions Made

- Used `git rm --cached` after `git mv` to untrack `.weights.h5` files at their new archive location. `git mv` keeps files tracked; the gitignore rules only apply to untracked files. `git rm --cached` removes from index while preserving the files on disk, making the gitignore rules effective going forward.
- Archived `backend/tests/test_exercise_sanity.py` and `backend/tests/validate_user_squats.py` in addition to the plan's explicit list. Both reference archived modules (`exercise_sanity`, `form_analyzer`, `form_session`). Leaving them in place would cause import errors on test discovery (Rule 2 auto-add to prevent dangling references).
- viz script uses `decode_clip_cv2` + `get_frame_count_and_fps` (cv2 helpers from `backend/services/clip_decode.py`) as instructed — the old `read_video_timestamps` + `decode_clip` from transforms.py are dead on torchvision 0.27.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Also archived backend/tests/test_exercise_sanity.py and validate_user_squats.py**
- **Found during:** Task 1 (precondition grep)
- **Issue:** These files in `backend/tests/` import `exercise_sanity`, `form_analyzer`, `form_session` — all being archived. Leaving them in place causes import errors in test discovery and dangling broken imports.
- **Fix:** Included them in the `git mv` to `backend/_archive_form_v4_v6/tests/`
- **Files modified:** archive/tests/test_exercise_sanity.py, archive/tests/validate_user_squats.py
- **Verification:** `python -m pytest ... --collect-only` still collects only 24 tests (none from archive); `python -c "import backend.app"` still exits 0
- **Committed in:** cf036c3 (Task 1 commit)

**2. [Rule 1 - Bug] Fixed Unicode print error in render_squat_result.py**
- **Found during:** Task 2 (first run of the viz script)
- **Issue:** Windows PowerShell uses cp1252 encoding; the `──` box-drawing characters in the print output caused `UnicodeEncodeError`. The figure WAS produced (figure save ran before the print), but the script exited with code 1.
- **Fix:** Replaced box-drawing characters with plain ASCII `--` / `----------------------`
- **Files modified:** backend/scripts/render_squat_result.py
- **Verification:** Second run exits 0, figure produced cleanly
- **Committed in:** 231880e (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 Rule 2 — missing critical, 1 Rule 1 — bug)
**Impact on plan:** Both necessary for correctness. No scope creep.

## Visualization Result — BadSquat_45.mp4

Rendered figure: `.planning/phases/05-backend-inference-integration-squat/figures/squat_result_overlay.png`

```
Clip: C:/Users/tsh_x/Downloads/BadSquat_45.mp4 | frames=121  fps=30.3
KFE detected=True  confidence=0.4817  severity=possible
KIE detected=False  confidence=0.0603  severity=none
```

**Interpretation:** KFE (Knee-Forward Error) detected on the bad-form clip as expected (confidence 0.4817 > threshold 0.385, severity "possible" — slightly above threshold, directionally correct). KIE not detected (confidence 0.0603, well below 0.614 threshold) — consistent with the modest KIE recall (~47%) documented in D-11.

## Domain-Shift Follow-up (D-11)

Per Plan 02 and the domain-shift diagnostics, two items are deferred to phase verification:

1. **Serving-threshold calibration** — the KIE threshold (0.614) was tuned on val clips; on real phone-camera clips the KIE confidence is low (~0.06 on a bad-form clip). The kneeaware_spatial_val pre-crop improves KFE separation but KIE recall remains modest. A re-calibration run on user clips is recommended before the supervisor demo.

2. **D-11 domain shift is not fully closed** — the cv2 BGR→RGB conversion introduces a small decoder residual vs. the training Colab path (torchvision read_video). This is documented as parity-APPROX in `clip_decode.py`. KFE generalizes better; KIE is still directionally correct but borderline on real clips.

## Issues Encountered

- None beyond the two auto-fixed deviations above.

## Next Phase Readiness

- Phase 5 is complete. All 5 plans (01-05) executed.
- API-01 closed: old TF+MediaPipe subsystem archived; new PyTorch SquatFormService is the sole form-correction runtime.
- The archive is reversible (`git mv backend/_archive_form_v4_v6/<path> backend/<path>` to restore).
- Phase 6 verification should check: `python -c "import backend.app"` exits 0; 24 test suite green; viz script produces expected KFE detection on a bad-form clip; archived weights are gitignored (`git check-ignore` returns the path).

---
*Phase: 05-backend-inference-integration-squat*
*Completed: 2026-05-26*
