---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: "Phase 4 context gathered (discuss-phase complete; A+B+D+E locked, F dropped). Next: /gsd:plan-phase 4 on a fresh L4 Colab notebook."
last_updated: "2026-05-21T10:40:15.062Z"
last_activity: 2026-05-25 -- Phase 4 COMPLETE: MD-SSL 3-seed ensemble test macro 0.6304 (KIE 0.4198/KFE 0.8410) MATCHES paper MD 0.6262, beats Phase3 0.5429 (+0.087). TTA evaluated/not adopted (val gain reversed on test). 9 figures + results.pkl + all 4 plan SUMMARYs done. Next: Phase 5 (backend inference integration)
progress:
  total_phases: 8
  completed_phases: 4
  total_plans: 8
  completed_plans: 8
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-19)

**Core value:** A user can record/upload a lift and get trustworthy, plain-language form-error feedback grounded in a published dataset and method.
**Current focus:** Phase 4 — Squat Motion-Disentangling SSL

## Current Position

Phase: 4 (Squat Motion-Disentangling SSL) — COMPLETE (4/4 plans)
Plan: all 4 complete; Phase 4 result = MD-SSL 3-seed ensemble test macro 0.6304 (matches paper MD 0.6262; +0.087 over Phase 3)
Status: Phase 4 closed; next = Phase 5 (Backend Inference Integration — serve the Squat detector live + video-upload)
Last activity: 2026-05-25 -- Phase 4 complete (ensemble 0.6304, 9 figures, results.pkl, 4 plan SUMMARYs)

Progress: [█████░░░░░] 50%

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: —
- Total execution time: 0 hours

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

Last session: 2026-05-22 -- Phase 4 Plan 02 run-1 interpreted. Control: SSL ep5 frozen-probe macro 0.5708 (KIE 0.380/KFE 0.762) vs Kinetics 0.4297 (KIE 0.169/KFE 0.691) = +0.141 lift. Probe peaks ep5 then declines + eff_rank collapse (weak-aug). DECISION = strong-aug paper-faithful md_pretrain_v2 (D2-augs), 5-epoch verification gate before full 20ep run. Aug change is a tracked code edit to ssl_augs.py + squat_ssl.py._augment; v2 run is heavy training. GATE PASSED (ep0-5, 2026-05-22): v2 ep5 probe macro 0.5844, KIE 0.4554 (>> v1's 0.380, and above paper MD's fine-tuned 0.4186 — different protocol), eff_rank 10.4 vs v1's 5.8 = collapse PREVENTED, macro rising not peaking. KFE 0.7133 dipped below v1's 0.762 (zoom may erode the depth cue — watch; dial back if it stays down). Plan 03 COMPLETE (2026-05-22): 3-seed fine-tune from v2-ep5 backbone.pt. best.pt on Drive at phase04/md_finetune_seed{42,1337,7}/. Val macro-F1: seed42=0.6062 (KIE 0.411/KFE 0.801, ep8), seed1337=0.6055 (0.431/0.780, ep2), seed7=0.6169 (KIE 0.493!/KFE 0.741, ep8). Mean ~0.609 ±0.006 = low seed variance (trustworthy ensemble). Diverse KIE/KFE balance (seed7 KIE-strong, seed42 KFE-strong) = good ensemble inputs. All 3 beat Phase3 (0.543) + paper-Kinetics (0.558); approaching paper-MD (0.626 test, ours val). D6 monitor was inverted (train/val), FIXED to val/train (commit 6f0587b) — validated by seed7's real late overfit (val/train 13x by ep16, after the ep<10 guard window so early-stop handled it). Base recipe (wd1e-4/dropout0.2) never tripped D6 -> locked across all 3 seeds. NEXT: Plan 04 — load 3 best.pt, aggregate_sigmoid_mean ensemble, per-head threshold_sweep (val), val-tuned TTA, TEST eval + 9 figures + P3-vs-P4 comparison chart. Eval primitives exist (ensemble.py/tta.py/metrics.py); the load->test-forward->ensemble->threshold->figures orchestration + notebook need building (Plan 04 not started). md_pretrain_v2 COMPLETE (ep0-19): probe ALSO peaks ep5 (0.5844, KIE 0.4554) then declines/oscillates (ep10 0.5443, ep15 0.5605); eff_rank held ~10 through ep5 then drifted to ~5-6 by ep11-19. Strong augs => higher peak (vs v1 0.5708) + better KIE (vs 0.380) + delayed collapse (ep11 vs ep6), but NOT a sustained plateau — peak-at-ep5 shape is robust across both runs. backbone.pt = v2-ep5 (probe-best, the SSL deliverable; later collapse does not touch saved ep5 weights). DECISION: SSL iteration DONE; proceed to Plan 03 = fine-tune 3 seeds (42/1337/7) from v2-ep5 backbone.pt (AdamW wd1e-4 dropout0.2 50ep/8-patience) -> Plan 04 ensemble(mean-sigmoid)+per-head threshold_sweep+val-TTA vs paper MD 0.6262. Pre-registered one-remediation if no lift = lower SSL LR (over-train-after-ep5 implicates LR). Plan 03 needs a new fine-tune notebook.
Stopped at: Plan 02 Tasks 1-6 all coded + pushed (origin ef40a2e). Probe resolved (per-clip JSON flat y-centers, 1:1 traj<->frame, SIGN bottom_is_argmax=False/ARGMIN, in-file NaNs -> interp). Dataset smoke green (len 4970, shapes ok, descent/ascent visually correct). VRAM/timing gate PASSED: batch 8 = 12.35 GB (fits L4), 38 min/epoch; user chose the 60-epoch FULL-EXTEND cap (~38h, multi-session, resume-safe). Collapse metric FIXED (256-sample probe; 8 capped eff_rank at ~7). NEXT: user runs Step 5 (md_pretrain_v1) across Colab sessions -> paste back the epoch-5 convergence checkpoint (ssl_loss down, embedding_std > 0.0044 no-collapse, first linear-probe > random) + the per-session linear-probe/emb_std trend; orchestrator flags the linear-probe plateau (§6 manual convergence stop). backbone.pt (linear-probe-best epoch) -> Plan 03 (3-seed fine-tune from it) -> Plan 04 (ensemble + val-tuned TTA + test eval + 9 figures + comparison chart).
Resume file: .planning/phases/04-squat-motion-disentangling-ssl/04-02-PLAN.md
