---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: "Phase 7 Plan 01 (Wave 0) COMPLETE — Shallow-Squat CVCSPC scaffold + unit tests GREEN (44 passed, 4 deselected; +15 Phase-7 tests, no Squat/OHP regression). 5 atomic commits 26dce44..62dd9be pushed to origin. NEXT: Plan 02 (Colab) — ImageNet ResNet-18 supervised baseline on the official Shallow-Squat splits (multi-seed, test F1 = SSL-lift control)."
last_updated: "2026-05-30"
last_activity: "2026-05-30 -- Phase 7 Plan 01 executed (Wave-0 local CPU): shallow_squat.py + cvcspc_ssl.py + cvcspc_pretrain.py (3-term loss + phase-matched triplet) + image_supervised_train.py + colab staging/frame-extract + 15 unit tests; Wave-0 gate 44 passed/4 deselected. Pushed 62dd9be."
progress:
  total_phases: 8
  completed_phases: 6
  total_plans: 22
  completed_plans: 18
  percent: 75
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-19)

**Core value:** A user can record/upload a lift and get trustworthy, plain-language form-error feedback grounded in a published dataset and method.
**Current focus:** Phase 7 (Shallow-Squat CVCSPC) PLANNED — ready to execute

## Current Position

Phase: 7 (Shallow-Squat CVCSPC) EXECUTING — Plan 01 (Wave 0) COMPLETE; Plans 02-05 remain. Phases 1–6 COMPLETE. IMG-03 (BarbellRow) DESCOPED.
Plan: Phase 7 = 1/5 executed. Plan 01 (Wave-0 LOCAL CPU) DONE: shallow_squat.py + cvcspc_ssl.py + image_supervised_train.py + cvcspc_pretrain.py + colab staging/frame-extract + 15 unit tests (3-term loss closed-form+directional, phase-matched cross-rep triplet, traj_nan exclusion, masking aug, single-head shapes, no-BN configurable projector, seams); Wave-0 gate 44 passed/4 deselected, no Squat/OHP regression; pushed 26dce44..62dd9be. NEXT: Plan 02 (Colab) = ImageNet ResNet-18 baseline (multi-seed, official Shallow-Squat splits 2542/529/540, test F1 = the SSL-lift control). 03 = CVCSPC SSL pretrain (Colab, FRESH notebook, frame-extract 4970 unlabeled clips + traj_nan exclusion) -> backbone.pt. 04 = fine-tune (model_builder seam) + ensemble + eval + results.pkl. 05 = deliverable pack. checkpoint_phase=phase07. Key facts: single binary error (Linear(512,1)+BCE, pos_weight~1.28); 3-term loss (train_test.py:68, code deviates from paper's 2-term — documented); Adam (official) == AdamW at wd=0; CVCSPC feasible on Squat-alone.
PRIOR (Phase 6, complete): OHP test macro 0.6622 (Elbows 0.4474 / Knees 0.8770), +0.050 SSL lift, paper-matched, val-test gap 0.011; deliverable shipped (docs/figures/ohp_*.png, docs/notebooks/05-08_ohp_*, FINDINGS_OHP, results.pkl).
Status: executing — Phase 7 Plan 01 (Wave 0) complete; Plan 02 baseline (Colab) next
Last activity: 2026-05-30 -- Phase 7 Plan 01 (Wave 0) complete: scaffold + 15 unit tests green (44 passed/4 deselected)

Progress: [████████░░] 75%

## Performance Metrics

**Velocity:**

- Total plans completed: 12 (Phases 1–4)
- Average duration: — (per-plan wall-time not tracked)
- Total execution time: — (interactive Colab execution; heavy training spanned multiple sessions)

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 5 | 5 | - | - |

**Recent Trend:**

- Last 5 plans: —
- Trend: —

*Updated after each plan completion*
| Phase 05 P01 | 437 | 3 tasks | 3 files |
| Phase 05 P02 | 60 | 3 tasks | 4 files |
| Phase 05 P03 | 15 | 2 tasks | 3 files |
| Phase 05-backend-inference-integration-squat P05 | 65 | 2 tasks | 14 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Init: Rebuild form-correction on Fitness-AQA; scrap the v4–v7 / QEVD line
- Init: Squat-first vertical slice; supervised baseline before domain-knowledge SSL
- Init: Binary detection + timing feedback; PyTorch for the form model
- Phase 1: Use the Drive shortcut in place (verified, 8/8 archives readable) — no preprocessing of Drive layout; copy zip → local per Colab session for training
- Phase 1: Joint multi-label head over a shared backbone for Squat (KIE & KFE co-occur — 192 of 232 KIE+ also KFE+)
- Phase 1: Notebook artifacts as `.py` jupytext-percent format (git-friendly; converts to .ipynb when needed)
- Phase 1: OHP barbell-trajectory preprocessing deferred to Phase 6 (track-0 default + NaN interpolation)
- Phase 2 (research override): input crop is 112×112 (torchvision R2Plus1D_18 Kinetics-V1 native), not the paper's 320→224 (which was the Waseda 2DCNN path). `crop_size` parameterized for Phase 3 ablation
- Phase 2 (research override): `Code_Release/motion_disentanglement/` is empty (1-byte README) — no supervised baseline code shipped. Phase 2 Task 1–2 clone the repo and document the gap; pipeline reconstructed from paper text + CVCSPC idioms
- Phase 2 (research override): decoder wrapped behind `decode_clip(path, indices)` — `torchvision.io.read_video` for now, one-line swap target for TorchCodec when Colab's torchvision passes 0.22
- Phase 2 (research override): horizontal flip default OFF (paper line 576 lists augmentations without flip; CVCSPC has it commented out)
- Phase 2 (research override): multi-label imbalance handled via `BCEWithLogitsLoss(pos_weight=Tensor([w_KIE, w_KFE]))`, not `WeightedRandomSampler`; dataset exposes `pos_weight`, Phase 3 consumes
- Phase 2: atomic Drive checkpoint write (tmp + torch.load round-trip verify + os.replace + latest.txt last) — Drive FUSE rename is not atomic
- Phase 2 (deviation): dataset uses torch's default RNG (not per-instance generator) so capture_rng_state/restore_rng_state cover sampling state across resume; per-instance generator would have broken Task 14's bitwise assertion
- Phase 2 (deviation): IPython kept at google.colab's pinned 7.34.0; autoreload enabled via a 4-line `imp` shim (`types.ModuleType` with `importlib.reload`) on Python 3.12
- Phase 2 (deviation): F11 windowing test corrected to clustered indices `[100, 102, 104]` (full-span indices defeat windowing — original plan had `[0, 200, 403]`)
- Phase 3: Supervised baseline closed — test KIE 0.286 / KFE 0.800 / macro 0.543 (within 2% of paper Kinetics row); end-to-end fine-tune overfits on 1136 clips → empirical motivation for Phase 4 SSL
- Phase 4 (discuss): scope A+B+D+E locked, F (aux trajectory head) dropped; AdamW wd=1e-4/dropout=0.2 (50ep+patience, P3 recipe); 3-seed ensemble (42/1337/7, mean of sigmoids); val-tuned TTA (flip OOD-corrected); abort-seed+bump-reg overfit guard (lock recipe on seed 42); one-remediation no-lift policy
- Phase 4 (Plan 02 run-1, 2026-05-22): md_pretrain_v1 (safe-core 4-aug, 60ep cosine) — SSL WORKS: ep5 frozen linear-probe macro 0.5708 vs Kinetics 0.4297 (+0.141; KIE 0.169->0.380 doubled, mirrors paper MD's KIE-concentrated lift). BUT probe peaks ep5 then declines (10:0.566, 15:0.551) with eff_rank collapse 11.8->3.3 = weak-aug contrastive collapse. Paper §3.2/Fig.5 require STRONG augs (we ran 4 mild). DECISION (D2-augs iteration): do NOT ride to 60, do NOT fine-tune v1; re-run paper-faithful md_pretrain_v2 — wire strong augs (translation+zoom+blur+channel-swap; rotation toggle default-OFF, KIE-risky) + 20ep cosine T_max=20; 5-epoch verification gate (probe>0.5708 AND KIE>=0.380 AND eff_rank healthy) BEFORE the full run. v1 backbone.pt kept as the weak-aug ablation datapoint. Exceed-path: 3-seed ensemble + per-head threshold_sweep + val-TTA on the v2 backbone vs paper MD-alone 0.6262.
- [Phase ?]: D-11 knee-aware serving crop: portrait lower-body pre-crop (0.42*H) in kneeaware_spatial_val; landscape unchanged for offline-eval parity; validated on 4 real phone clips
- [Phase ?]: Per-connection SquatLiveSession per WebSocket accept — fresh instance isolates mutable buffer state
- [Phase ?]: Single-seed classify_clip_async(n_seeds=1) for low-latency live inference (~0.93s per rep — D-08)
- [Phase ?]: Deterministic session_summary feedback from KIE/KFE counts, no LLM (D-10)
- [Phase ?]: git rm --cached after git mv to untrack archived weights — gitignore only applies to untracked files
- [Phase ?]: render_squat_result.py uses decode_clip_cv2 (cv2 backend) — torchvision 0.27 removed read_video; KFE detected=True on BadSquat_45 confidence=0.4817
- [Phase ?]: API-01 closed: old TF+MediaPipe form subsystem reversibly archived; new PyTorch SquatFormService is the sole form-correction runtime

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260526-wi7 | Flutter form D-05 UI shim + exercise picker trim to three (demo-prep, 05-HUMAN-UAT #2) | 2026-05-26 | ab9fea1 | [260526-wi7-flutter-form-d05-ui-shim-and-exercise-pi](./quick/260526-wi7-flutter-form-d05-ui-shim-and-exercise-pi/) |
| 260527-2bt | Squat live latency: ONNX (1.6x, parity exact) retained; live-trigger + upload-replay SUPERSEDED by frontend cancellation | 2026-05-27 | 1524082 | [260527-2bt-squat-live-latency-onnx-rep-aware-trigge](./quick/260527-2bt-squat-live-latency-onnx-rep-aware-trigge/) |
| eval-bench | Benchmark eval visualizations (244-clip official test split) + best-example findings — EVAL-01 (comparison) + EVAL-03 (viz pack) partial; defense deliverable | 2026-05-27 | (eval commit) | docs/eval/FINDINGS.md + docs/figures/ |
| squat-nbs | 4 executed Squat analysis notebooks: EDA, data-pipeline/balancing, training-diagnostics (over/underfit + SSL collapse), evaluation (PR/ROC/AUC/confusion) — defense visualization pack | 2026-05-27 | (nb commits) | docs/notebooks/ |

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-05-29T23:34:48.447Z
Stopped at: Phase 7 context gathered (Shallow-Squat CVCSPC; BarbellRow descoped)
Resume file: .planning/phases/07-image-based-errors-cvcspc/07-CONTEXT.md
