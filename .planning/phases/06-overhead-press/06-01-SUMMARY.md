---
phase: 06-overhead-press
plan: 01
subsystem: training
tags: [pytorch, fitness-aqa, ohp, ssl, dataset, splits, colab]

# Dependency graph
requires:
  - phase: 04-squat-motion-disentangling-ssl
    provides: squat_ssl.py, split_half_cycles, supervised_train._build_dataloaders, md_finetune.run_md_finetune_epoch, colab staging helpers
  - phase: 03-squat-supervised-baseline
    provides: SquatKIEKFEDataset, build_loaders, transforms.py, _compute_pos_weight pattern
  - phase: 02-squat-data-pipeline-colab-harness
    provides: splits.py (_load_json, _label_value, ClipRecord), colab.py harness primitives
provides:
  - OHPClipRecord dataclass + index_ohp() split loader (1582/339/339, D4 verified)
  - OHPElbowsKneesDataset (joint multi-label, label_elbows/label_knees, train pos_weight ~[2.887, 1.924])
  - OHPSSLDataset with BBox _load_trajectory (y_center=(y1+y2)/2 from region_0, NaN interp)
  - dataset_cls=SquatKIEKFEDataset keyword-only seam on _build_dataloaders + run_md_finetune_epoch
  - stage_ohp_videos + stage_unlabeled_ohp_videos (two-drive-root split for Pitfall 4)
  - pytest suite: 29 passed -m "not slow" (OHP-01-a/b/c/d + seam test + existing Squat suite green)
affects:
  - 06-02 (gated trajectory-format Colab probe consumes OHPSSLDataset._load_trajectory)
  - 06-03 (SSL GPU burn consumes OHPSSLDataset + stage_unlabeled_ohp_videos)
  - 06-04 (fine-tune consumes OHPElbowsKneesDataset + dataset_cls seam)

# Tech tracking
tech-stack:
  added: []  # Zero new pip installs (T-06-SC accept — all libraries Colab-provided or in requirements.txt)
  patterns:
    - "OHP BBox trajectory parser: per-frame [region_0,region_1,region_2]; y_center=(y1+y2)/2 from region_0[0]; empty region_0 -> np.nan -> linear interp"
    - "FLAT trajectory glob: glob('*.json') not rglob (OHP zip extracts flat vs Squat's nested)"
    - "dataset_cls kwarg seam: keyword-only with SquatKIEKFEDataset default — backward-compatible injection"
    - "Two-drive-root staging: stage_unlabeled_ohp_videos(drive_root_3001, drive_root_3002) handles OHP archive split"

key-files:
  created:
    - backend/training/aqa/datasets/ohp.py
    - backend/training/aqa/datasets/ohp_ssl.py
    - backend/training/aqa/datasets/test_ohp.py
    - backend/training/aqa/datasets/test_ohp_ssl.py
    - backend/training/aqa/datasets/conftest.py
  modified:
    - backend/training/aqa/datasets/splits.py (additive: OHPClipRecord + index_ohp + ohp_expected_counts)
    - backend/training/aqa/harness/supervised_train.py (dataset_cls kwarg on _build_dataloaders)
    - backend/training/aqa/harness/md_finetune.py (dataset_cls kwarg on run_md_finetune_epoch)
    - backend/training/aqa/harness/colab.py (additive: stage_ohp_videos + stage_unlabeled_ohp_videos)

key-decisions:
  - "D4: index_ohp reconciles to 1582/339/339; Elbows+ 407 (25.7% w≈2.887), Knees+ 541 (34.2% w≈1.924) — verified against local archive"
  - "D5-format: OHP trajectories are per-frame BBox arrays NOT flat y-lists; _load_trajectory extracts y_center=(y1+y2)/2 from region_0[0]; empty frames -> np.nan -> linear interp (Squat NaN path reused)"
  - "D6-reuse: split_half_cycles COPIED VERBATIM from squat_ssl.py (argMIN, bottom_is_argmax=False); ssl_augs.py/transforms.py reused unchanged"
  - "D6-factoring: dataset_cls=SquatKIEKFEDataset keyword-only seam — backward-compatible; existing Squat call sites pass no dataset_cls and are unaffected (no-regression: 29 passed)"
  - "D7 landmine #4: OHP trajectory glob is glob('*.json') NOT rglob — OHP zip extracts flat, Squat extracts nested"
  - "D7 landmine #7: stage_unlabeled_ohp_videos uses zipfile.extractall for trajectory JSON, NOT _extract_with_resume_and_progress (mp4-only)"

patterns-established:
  - "OHP BBox parser is the template for any future exercise using YOLO/BBox trajectory format"
  - "Two-drive-root staging pattern handles archives split across multiple release folders"

requirements-completed: [OHP-01]  # NOTE: OHP-01 is phase-spanning; Wave 0 scaffold only.
                                   # OHP-01 is NOT satisfied yet — it requires trained/evaluated models
                                   # (Plans 02-05). Do NOT mark OHP-01 complete after Wave 0.
                                   # orchestrator is instructed NOT to run requirements.mark-complete.

# Metrics
duration: 45min
completed: 2026-05-27
---

# Phase 6 Plan 01: OHP Wave-0 Dataset Modules Summary

**OHP BBox trajectory parser + full dataset scaffold (OHPElbowsKneesDataset, OHPSSLDataset, splits.index_ohp, dataset_cls seam, colab staging) CPU-verified with 29 tests green before any GPU burn**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-05-27T00:00:00Z
- **Completed:** 2026-05-27
- **Tasks:** 5 / 5
- **Files modified:** 9 (5 created, 4 edited)

## Accomplishments

- Implemented the ONE genuinely-new logic (T-06-01): OHP BBox `_load_trajectory` parser — per-frame `[region_0,region_1,region_2]`; `y_center=(y1+y2)/2` from `region_0[0]`; empty region_0 → `np.nan` → linear interpolation. Verified on synthetic 5-frame JSON: `y[0]=150, y[2]=110, y[4]=200`, interpolated frames between neighbors, no NaN.
- Established the correct argMIN half-cycle sign for OHP (T-06-02): `split_half_cycles(bottom_is_argmax=False)` — barbell at overhead = lowest y pixel = argMIN. V-trajectory test: `descent[-1] <= argmin <= ascent[0]`; argMAX degenerates one half-cycle (sign param wired).
- Reconciled `index_ohp` to D4-verified counts: 1582/339/339 train/val/test, Elbows+ 407 (25.7%), Knees+ 541 (34.2%), against the local OHP archive.
- Injected backward-compatible `dataset_cls=SquatKIEKFEDataset` seam on `_build_dataloaders` and `run_md_finetune_epoch` — existing Phase 3/4 Squat call sites pass no `dataset_cls` and are byte-for-byte unaffected (no-regression: 29 passed).
- Added OHP staging functions handling the Pitfall 4 two-drive-root split (trajectories in -3-001, unlabeled videos in -3-002).

## Task Commits

1. **Task 1: splits.index_ohp + OHPClipRecord + OHP-01-d count test** - `7220a54` (feat)
2. **Task 2: OHPElbowsKneesDataset + pos_weight** - `e7a929a` (feat)
3. **Task 3: ohp_ssl.py BBox _load_trajectory + OHP-01-b test** - `78fb6bf` (feat)
4. **Task 4: split_half_cycles argMIN guard + conftest.py** - `a3e603c` (test)
5. **Task 5: dataset_cls seam + colab OHP staging + no-regression gate** - `3554b46` (feat)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified

- `backend/training/aqa/datasets/splits.py` — ADD OHPClipRecord + index_ohp() + ohp_expected_counts() (additive, Squat code untouched)
- `backend/training/aqa/datasets/ohp.py` — NEW: OHPElbowsKneesDataset + _compute_pos_weight + build_loaders
- `backend/training/aqa/datasets/ohp_ssl.py` — NEW: OHPSSLDataset + BBox _load_trajectory + split_half_cycles (verbatim copy) + build_ssl_loader
- `backend/training/aqa/datasets/test_ohp.py` — NEW: test_ohp_splits_count, test_ohp_pos_weight, test_ohp_dataset_shape (skipped), test_dataset_cls_injection
- `backend/training/aqa/datasets/test_ohp_ssl.py` — NEW: test_trajectory_load, test_trajectory_load_all_valid, test_trajectory_load_raises_on_too_few_valid, test_split_half_cycles_ohp
- `backend/training/aqa/datasets/conftest.py` — NEW: registers 'slow' mark (suppresses PytestUnknownMarkWarning)
- `backend/training/aqa/harness/supervised_train.py` — EDIT: dataset_cls kwarg on _build_dataloaders (~3 lines)
- `backend/training/aqa/harness/md_finetune.py` — EDIT: dataset_cls kwarg on run_md_finetune_epoch + forward to _build_dataloaders (~2 lines)
- `backend/training/aqa/harness/colab.py` — ADD: stage_ohp_videos + stage_unlabeled_ohp_videos (~170 lines, additive)

## Decisions Made

- D4 verified: `index_ohp` counts reconcile to 1582/339/339 against local archive. Elbows+ 407 / Knees+ 541 — both moderate-minority (w≈2.887/1.924), favorable vs Squat's rare+majority profile.
- D5-format confirmed: OHP BBox parser extracts `y_center=(y1+y2)/2` from `region_0[0]` (ReadMe formula). The live 1:1 traj↔frame alignment is gated to Plan 02's Colab probe before the SSL GPU burn.
- D6-reuse: `split_half_cycles` COPIED VERBATIM from squat_ssl.py. argMIN sign is correct for OHP (overhead = lowest y pixel = argMIN, same sign as Squat for different physical reason). Re-confirmed by Plan 02's probe.
- D6-factoring: additive `OHPClipRecord` + `index_ohp()` in splits.py (no edits to Squat); `dataset_cls` keyword-only seam keeps the default as `SquatKIEKFEDataset` — backward-compatible.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added datasets/conftest.py to register 'slow' pytest mark**
- **Found during:** Task 4 (argMIN guard test)
- **Issue:** Tests using `@pytest.mark.slow` produced PytestUnknownMarkWarning on every run, indicating the mark was not registered. While not a functional issue, unregistered marks risk being silently misfiltered in CI.
- **Fix:** Added `backend/training/aqa/datasets/conftest.py` registering the `slow` mark via `pytest_configure`.
- **Files modified:** `backend/training/aqa/datasets/conftest.py`
- **Verification:** Warnings absent from final no-regression run.
- **Committed in:** `a3e603c` (Task 4 commit)

---

**Total deviations:** 1 auto-fixed (Rule 2 — missing critical marker registration)
**Impact on plan:** Minimal. The conftest.py addition is pure housekeeping; no scope change.

## Known Stubs

None — all OHP code is fully implemented with correct logic. The 1:1 traj↔frame alignment assumption in `_load_trajectory` is documented as `[ASSUMED — confirmed in Colab probe]` (not a stub — it is a known-provisional assumption gated to Plan 02 before any GPU burn).

## Threat Flags

None — offline ML pipeline (no network/auth/untrusted-input surface per D1). T-06-01 (BBox parser) is mitigated by Task 3's unit test. T-06-SC accept (zero new installs).

## Test Results

Final `python -m pytest backend/training/aqa/ -m "not slow" -q`:
```
29 passed, 4 deselected in 7.81s
```

Skipped tests and reasons:
- `test_ohp_dataset_shape` — `@pytest.mark.slow` (requires torchvision write_video; also skipped on Windows CPU due to mp4 codec unavailability). The plan says "skip cleanly if codec unavailable, but pos_weight must pass" — pos_weight passed.
- 3 additional `@pytest.mark.slow` tests (archive-dependent) — deselected by `-m "not slow"`.

The test in RESEARCH §6 (OHP-01-b) references `test_ohp.py::test_trajectory_load` but the plan put trajectory tests in `test_ohp_ssl.py`. Both files are consistent with the plan's `<files>` spec.

## Issues Encountered

None — all 5 tasks executed cleanly. The `test_dataset_cls_injection` failure during Task 2's verify run was expected (the seam edit hadn't been done yet); it passed in Task 5's no-regression run.

## User Setup Required

None — no external service configuration required. All code is CPU-only pure-function scaffold; Colab/Drive/GPU are not touched in this plan.

## Next Phase Readiness

- Plan 02 (Wave 1+2): gated `checkpoint:human-verify` probe on 2-5 real OHP clips confirms the 1:1 traj↔frame alignment and re-confirms argMIN visually, THEN trains the supervised baseline. Requires: Colab L4, Drive with OHP labeled data + trajectory zip staged via `stage_ohp_videos`.
- The Wave-0 GPU-burn gate is locked behind a green pure-function suite. Plan 02 must not start until this SUMMARY is committed and the gate is green.
- T-06-01 (BBox parser): CPU unit test passed; live alignment confirmed at Plan 02.
- OHP-01 requirement: NOT complete after Wave 0. Plans 02-05 deliver the trained/evaluated models.

---
*Phase: 06-overhead-press*
*Completed: 2026-05-27*
