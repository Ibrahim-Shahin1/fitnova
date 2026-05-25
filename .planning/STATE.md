---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: "Phase 4 COMPLETE (4/4 plans; HEAD b49a7d2, pushed). MD-SSL ensemble test macro 0.6304 matches paper MD 0.6262 (+0.087 over Phase 3). ROADMAP + STATE reconciled. Next: /gsd:discuss-phase 5 (Backend Inference Integration)."
last_updated: "2026-05-25"
last_activity: 2026-05-25 -- Phase 4 COMPLETE: MD-SSL 3-seed ensemble test macro 0.6304 (KIE 0.4198/KFE 0.8410) MATCHES paper MD 0.6262, beats Phase3 0.5429 (+0.087). TTA evaluated/not adopted (val gain reversed on test). 9 figures + results.pkl + all 4 plan SUMMARYs done. Next: Phase 5 (backend inference integration)
progress:
  total_phases: 8
  completed_phases: 4
  total_plans: 7
  completed_plans: 7
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-19)

**Core value:** A user can record/upload a lift and get trustworthy, plain-language form-error feedback grounded in a published dataset and method.
**Current focus:** Phase 5 — Backend Inference Integration (Squat)

## Current Position

Phase: 4 (Squat Motion-Disentangling SSL) — COMPLETE (4/4 plans)
Plan: all 4 complete; Phase 4 result = MD-SSL 3-seed ensemble test macro 0.6304 (matches paper MD 0.6262; +0.087 over Phase 3)
Status: Phase 4 closed; next = Phase 5 (Backend Inference Integration — serve the Squat detector live + video-upload)
Last activity: 2026-05-25 -- Phase 4 complete (ensemble 0.6304, 9 figures, results.pkl, 4 plan SUMMARYs)

Progress: [█████░░░░░] 50%

## Performance Metrics

**Velocity:**

- Total plans completed: 7 (Phases 1–4)
- Average duration: — (per-plan wall-time not tracked)
- Total execution time: — (interactive Colab execution; heavy training spanned multiple sessions)

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**

- Last 5 plans: —
- Trend: —

*Updated after each plan completion*

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

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-05-25 -- Phase 4 COMPLETE and fully wrapped (HEAD b49a7d2, pushed to origin/fresh-start). MD-SSL 3-seed ensemble test macro 0.6304 (KIE 0.4198 / KFE 0.8410) matches paper MD 0.6262, beats Phase 3 baseline 0.5429 (+0.087). All 4 plan SUMMARYs, 10 figures + results.pkl, and the consolidated 05_squat_md_finetune.{py,ipynb} committed. Production protocol: 3-seed mean-of-sigmoids, thresholds KIE 0.614 / KFE 0.385, NO TTA; single-seed latency fallback ~0.612. Checkpoints live on Drive at MyDrive/FitNova/checkpoints/phase04/ (md_pretrain_v2/backbone.pt + md_finetune_seed{42,1337,7}/best.pt) — NOT in the repo; Phase 5 must transfer them to the serving machine.
Stopped at: Phase 4 closed; ROADMAP + STATE reconciled to reflect completion. Next: /gsd:discuss-phase 5 (Backend Inference Integration — serve the Squat detector live + video-upload, replacing the TF/MediaPipe form_analyzer/form_session). Design tensions to surface in discuss: the new model is clip-level (32-frame, 112², Kinetics-norm RGB -> 2 binary errors) vs the old per-frame pose pipeline, so live mode needs rep segmentation WITHOUT pose; PyTorch+TF coexistence in one FastAPI process; Drive->local-server weight transfer; CPU latency (3x R(2+1)D-18 forward); early real-camera domain-shift test (offline 0.63 != live phone camera).
Resume file: .planning/ROADMAP.md (Phase 5 — no phase dir yet; discuss-phase 5 creates it)
