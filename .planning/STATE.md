---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: "Completed 05-02: kneeaware_spatial_val wired; D-11 fix validated; 8/8 tests green"
last_updated: "2026-05-26T14:44:44.631Z"
last_activity: 2026-05-26
progress:
  total_phases: 8
  completed_phases: 3
  total_plans: 13
  completed_plans: 11
  percent: 38
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-19)

**Core value:** A user can record/upload a lift and get trustworthy, plain-language form-error feedback grounded in a published dataset and method.
**Current focus:** Phase 05 — Backend Inference Integration (Squat)

## Current Position

Phase: 05 (Backend Inference Integration (Squat)) — EXECUTING
Plan: 5 of 5
Status: Ready to execute
Last activity: 2026-05-26

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
| Phase 05 P01 | 437 | 3 tasks | 3 files |
| Phase 05 P02 | 60 | 3 tasks | 4 files |
| Phase 05 P03 | 15 | 2 tasks | 3 files |

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

Last session: 2026-05-26T14:44:44.624Z
Stopped at: Completed 05-02: kneeaware_spatial_val wired; D-11 fix validated; 8/8 tests green
Resume file: None
