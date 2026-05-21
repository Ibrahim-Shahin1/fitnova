# Phase 04: Squat Motion-Disentangling SSL — Context

**Gathered:** 2026-05-21
**Status:** Ready for planning
**Source:** Interactive `/gsd:discuss-phase 4` (user elected discuss-phase first, unlike Phase 1/2/3 which were orchestrator-synthesized — Phase 4 is the research contribution and has strategic decisions to lock before planning). Grounded in the Phase 4 handoff (`~/.claude/plans/handoff-2026-05-20-phase-4-md-ssl.md`), Phase 3 SUMMARY/CONTEXT, ROADMAP Phase 4 success criteria, REQUIREMENTS SQUAT-04/SQUAT-05, master plan, and working-agreement memories. User authorized autonomous GSD operation ([[feedback_run_gsd_autonomously]]); routine gates decided from precedent, substantive design choices put to the user (this discussion).

<domain>
## Phase Boundary

**This phase delivers:**

1. **Motion-Disentangling (MD) SSL pretraining** of an R(2+1)D-18 backbone on the **4,970 unlabeled Squat clips** + shipped barbell trajectories, using the paper's half-cycle contrastive pretext task (§4). Output: an MD-pretrained backbone checkpoint on Drive. **(SQUAT-04)**
2. **Fine-tune** the MD-pretrained backbone on the labeled Squat set for joint KIE/KFE detection — same data pipeline + joint multi-label head as Phase 3, but starting from MD-SSL init instead of Kinetics-V1, with regularization (B) added. **(SQUAT-05)**
3. **3-seed ensemble** (D): one shared SSL pretrain → 3 fine-tunes (seeds 42, 1337, 7) → aggregated predictions.
4. **Test-time augmentation** (E) at inference, recipe val-tuned.
5. **F1 per error on the official Squat test split** (244 clips), threshold tuned on val, compared against Phase 3 baseline (KIE 0.286 / KFE 0.800 / macro 0.543) **and** the paper's MD row (KIE 0.419 / KFE 0.834 / macro ~0.626) and GYMetricPose, on identical metrics.
6. **Visualization deliverables** (per [[project_supervisor_visualizations]]): SSL pretraining loss curve, linear-probe F1-during-SSL curve, per-seed fine-tune training curves, ensemble-vs-single-seed comparison, **Phase 3-vs-Phase 4 headline comparison bar chart**, test confusion matrices (KIE/KFE), test PR curves (KIE/KFE), sample predictions, `results.pkl`.
7. **Phase 4 notebook** `backend/training/aqa/notebooks/04_squat_md_ssl.py` + `.ipynb` pair (Colab-runnable, Cell A bootstrap), disconnect-safe.
8. **New code modules:** SSL trainer (`harness/md_pretrain.py`), fine-tune trainer (`harness/md_finetune.py`), ensemble aggregation (`eval/ensemble.py`), TTA (`eval/tta.py`).
9. End-of-phase **`04-01-SUMMARY.md`** declaring the production checkpoint(s), inference protocol, per-error thresholds, and expected inference latency for the Phase 5 handoff.

**Stated goal (user):** beat the paper's MD row macro F1 (~0.626) or land slightly above, **without overfitting**. ROADMAP minimum bar: F1 ≥ Phase 3 baseline + a comparison chart.

**This phase does NOT deliver:**

- **F — auxiliary trajectory-prediction head: DROPPED** (25% regression risk, +12-24h re-SSL-pretrain cost, λ sweep required, paper doesn't do it — too risky for the schedule). Confirmed dropped by user 2026-05-20 and again at discuss-phase open.
- OHP / BarbellRow / Shallow-Squat models (Phases 6, 7).
- Backend inference integration / FastAPI changes (Phase 5).
- Polished Flutter form UI (later milestone).
- Graded severity output (out of scope — labels binary).

**Requirements satisfied:** **SQUAT-04** (MD-SSL pretraining on unlabeled Squat set), **SQUAT-05** (MD-pretrained fine-tuned for KIE/KFE; F1 vs baseline + vs published).

</domain>

<decisions>
## Implementation Decisions

### D1 — Scope: A + B + D + E (F dropped) [LOCKED]
- **A — Faithful MD-SSL reproduction** (the core contribution).
- **B — weight_decay + dropout on fine-tune** (mild regularization, paper-silent).
- **D — 3-seed fine-tune ensemble** (variance reduction; shared SSL pretrain).
- **E — Test-time augmentation** (val-tuned).
- **F — auxiliary trajectory head — NOT in scope.**

### D2 — MD-SSL pretraining recipe (A): faithful reproduction; details RESEARCHER-RESOLVES
The high-level approach is locked; the mechanics are delegated to `gsd-phase-researcher` because `Code_Release/motion_disentanglement/` is empty (1-byte README per Phase 2 [`CODE-RELEASE-NOTES.md`](../02-squat-data-pipeline-colab-harness/CODE-RELEASE-NOTES.md)) — reconstruct from paper §4 text + general SSL literature (SimCLR / MoCo / BYOL idioms).

**Locked premises:**
- Pretext task = the paper's **half-cycle contrast** on barbell-trajectory-split reps.
- **SimCLR-family contrastive loss** (NT-Xent / InfoNCE) is the default family; researcher confirms the exact formulation against paper §4 or labels `[ASSUMED]`.
- **AdamW** for the SSL optimizer (not Adam) — proper decoupled weight decay ([[reference_pytorch_persistent_workers]] sibling concern; see D3).
- **Linear-probe validation during SSL** as the convergence monitor — periodically freeze backbone, train linear head on a labeled subset, measure val F1; plateau ⇒ SSL has learned what it can.
- SSL pretrain budget is **12-24h on L4** per [[feedback_heavy_training_new_notebook]]; user is already on a fresh L4 notebook.

**Researcher MUST resolve (the open questions from the handoff, items 1-8):**
1. Half-cycle splitting algorithm (peak detection on which signal? window? smoothing? bottom-of-rep threshold? multi-rep clip handling — clips run to 13.5s / 2-3 reps).
2. Exact SSL loss formulation (NT-Xent vs triplet vs InfoNCE; stop-gradient?).
3. Positive/negative sampling (within-batch SimCLR vs memory-bank MoCo; hard negatives?).
4. Projection-head architecture (layers, output dim; discarded at fine-tune?).
5. SSL optimizer/scheduler/epochs/batch-size — adapt to L4 24 GB (SimCLR wants batch 256+; R(2+1)D-18 @ 32-frame 112² ≈ 1 GB/batch-16 → batch 256 ≈ 16 GB, tight but feasible — re-estimate VRAM, Phase 3 hit 15.22 GB backward at batch 16).
6. Linear-probe validation cadence (every 10 epochs? 50?).
7. SSL augmentation policy (temporal crop within rep, spatial crop, jitter — strong augs aid SSL invariance but may destroy form-error signal; balance).
8. Trajectory file format (JSON/CSV/npy? per-frame coords? smoothed/raw?) — quick `ls` of the unlabeled-set Drive folder at research time; Phase 1 didn't fully probe this.

### D3 — Regularization on fine-tune (B): MODERATE [LOCKED via discussion]
- **Optimizer: AdamW** (NOT Adam) — Phase 3 used `torch.optim.Adam` (L2-on-gradient ≠ true weight decay). Phase 4 uses `torch.optim.AdamW` for proper decoupled weight decay. One-line change, real semantic difference.
- **weight_decay = 1e-4**, **dropout = 0.2 head-only** (just before the final `Linear(512, 2)`). Moderate setting — defensible on small datasets, doesn't tend to underfit.
- **Epoch budget = 50, cosine annealing, 8-epoch val-macro-F1 patience early-stop** — matches Phase 3 recipe exactly for clean P3-vs-P4 comparison. **Deliberate deviation from paper §5's 20-epoch downstream budget**, justified by comparison parity + reliance on early-stop bounding actual cost (Phase 3 stopped at 11). Researcher notes the deviation.
- **Warmup OFF, fine-tune LR = 1e-4** matching Phase 3. Researcher MUST check paper §5 for the downstream fine-tune LR — if the paper specifies a lower LR for SSL→fine-tune transfer (common, to avoid disrupting SSL features), override with citation. Warmup is a deferred ablation if early epochs show AdamW+SSL-init instability.

### D4 — Multi-seed ensemble (D): 3 seeds, mean of sigmoids [LOCKED via discussion]
- **3 seeds: 42, 1337, 7.** (5 seeds rejected — +10h compute, diminishing variance reduction, would crowd Phases 5-8 in the budget.)
- **Shared SSL pretrain** — one MD-SSL backbone, 3 fine-tunes only. (SSL × seeds is prohibitive at 12-24h each.)
- **Aggregation = mean of sigmoid scores across seeds**, per error head (Claude's choice, user delegated). Rationale: continuous + calibrated → ensemble PR-AUC and threshold sweep work like Phase 3's; standard practice (Lakshminarayanan et al. 2017 deep ensembles); robust on the small 243-clip val (36 KIE-positives) where per-seed thresholding would overfit individually. Mean-of-logits and majority-vote rejected.
- **Single per-error decision threshold tuned on the ensemble val scores** (NOT per-seed thresholds then vote).
- **Per-seed F1 numbers also reported** in SUMMARY as variance evidence — proves the ensemble lift is real, not one lucky seed.

### D5 — Test-time augmentation (E): val-tuned; flip default CORRECTED [LOCKED via discussion]
- **TTA recipe is selected by val-tuning**, not fixed a priori. Candidate augmentations: **temporal jitter, spatial 5-crop (center + 4 corners), horizontal flip.** For each candidate combo, compute ensemble val macro-F1; keep the combo that maximizes it; apply to test.
- **Correction to the handoff's "flip + temporal" default:** Phase 3 trained with **horizontal flip OFF** (Phase 2/3 D5 — paper line 576 omits flip, CVCSPC commented it out). TTA only reduces variance for augmentations the model is *invariant* to — and invariance comes from training. Temporal jitter (trained ±2-frame) and spatial crop (trained random-crop) are safe; **flip is OOD** (model never saw flips) and could add noise. So flip is a *candidate to be validated on val*, not an assumed-good default. This is the [[feedback_ai_correctness]] "measure don't guess" stance.
- **Aggregation order:** per-seed → average its TTA copies → mean of sigmoids across the 3 seeds → val-tuned per-error threshold. (Mathematically a grand mean over all seed × TTA score variants.) TTA combo + threshold are *both* tuned on ensemble val; test stays held out. The val-test gap monitor (D6) guards against over-tuning val.
- **Phase 5 handoff:** Phase 4 reports the **full ensemble+TTA result as the headline F1** AND a **single-model / no-TTA number**, so Phase 5 can choose its production inference protocol against its latency budget (handoff requires this — note the TTA × ensemble forward-pass cost).

### D6 — Overfit safeguards: 3 monitors + abort/remediation policy [LOCKED via discussion]
Codify in PLAN as a `<determinism_checklist>` extension. Three monitors:

1. **Runtime — train-val BCE loss ratio per epoch.** If ratio **>10× before epoch 10** during a fine-tune: **abort that seed and bump regularization** (wd=5e-4, dropout=0.3). (Phase 3 hit 32× by epoch 8 with wd=0/no-dropout; moderate reg should keep Phase 4 well below — this likely won't fire.)
   - **Reg-bump policy (ensemble homogeneity):** fine-tunes run **sequentially on one L4**. Run **seed 42 first**; if it trips → abort, bump, restart 42 until clean; **lock that recipe for seeds 1337 & 7**. Keeps the ensemble homogeneous (3 seeds, 1 recipe) in the common case at zero wasted compute. A later-seed trip at the locked recipe is documented as genuine seed variance, not auto-bumped.
2. **Post-hoc — val-test F1 gap at final eval.** Target **< 0.05** (Phase 3 was 0.008). Larger ⇒ over-tuned to val (threshold + TTA selection). Report honestly; if violated, widen tuning to be less val-specific.
3. **Post-hoc — per-error KIE test F1 must NOT regress below Phase 3's 0.2857.** If ensemble macro improves but KIE drops below 0.286, that's a bad KFE-for-KIE trade — KIE is the paper's headline lift (0.297→0.419) and the harder, more important error. Flag prominently.

**No-lift policy:** if the final ensemble does NOT beat Phase 3 (macro 0.543): **one documented remediation pass** (the Area-1 regularization sweep, or an SSL-epoch/recipe adjustment informed by the linear-probe signal). If still no lift, close Phase 4 with **honest deviation analysis** ("faithfully implemented MD-SSL, didn't lift over Kinetics baseline in our regime") — a valid undergraduate result. Bounds the schedule hit; preserves Phases 5-8. (Open-ended iteration rejected.)

### D7 — Carry-forward landmines (must NOT re-surface) [LOCKED from Phase 3]
- `persistent_workers=True` on every DataLoader with `num_workers>0` ([[reference_pytorch_persistent_workers]]).
- `map_location='cpu'` for `load_latest_checkpoint` (Phase 3 fix `a0841b4`; CUDA ByteTensors break `set_rng_state_all`).
- `metrics_history` loaded from the `latest.txt`-pointed checkpoint, NOT `best.pt` (frozen at best epoch) for full training-curve viz ([[reference_best_pt_metrics_history_is_stale]]).
- Notebook delivered as paired `.py` + `.ipynb` with **Cell A bootstrap** (clone-or-pull + `sys.path.insert`) baked into the FIRST `.ipynb` ship ([[feedback_deliver_colab_as_ipynb]]).
- **One runnable unit at a time; NEVER pre-write task N+1 cells before task N's paste-back** ([[feedback_one_cell_at_a_time_strict]]).
- After `git pull` in a running Colab kernel, `sys.modules` is STALE — restart runtime or `importlib.reload` after every trainer fix ([[reference_colab_module_reload_after_git_pull]]). **Restart the runtime BEFORE any major trainer change.**
- `atomic_save_checkpoint` for `best.pt` must NOT clobber `latest.txt` (Phase 3 inline-saved/restored it; Phase 4 should consider an explicit `update_latest=False` kwarg).

### D8 — Visualizations [LOCKED — first-class per [[project_supervisor_visualizations]]]
Saved as PNG to `.planning/phases/04-squat-motion-disentangling-ssl/figures/` **before** any `plt.show()`; defensive parent-dir mkdir; backed up to Drive. The headline is the **Phase 3-vs-Phase 4 comparison bar chart** (baseline vs MD-SSL + ablations vs Parmar vs GYMetricPose on identical metrics). Full list in the Phase Boundary above.

### D9 — Reuse contracts (do NOT redefine) [LOCKED from Phase 2/3]
- `eval/metrics.py` (Phase 3) — `f1_per_error`, `pr_auc_per_error`, `threshold_sweep`, `confusion_matrix_per_error` — reused UNCHANGED. Phase 4's `eval/ensemble.py` + `eval/tta.py` call into it.
- `datasets/squat.py`, `datasets/transforms.py`, `datasets/splits.py` (Phase 2) — labeled pipeline reused for fine-tune. SSL needs a **NEW unlabeled dataset class** (no label JSON; barbell-trajectory pairing required) — Phase 2 deferred this explicitly.
- `harness/supervised_train.py` (Phase 3) — `_build_dataloaders`, `_val_pass`, `run_supervised_epoch`, `build_model`, `seed_worker` patterns transfer to `md_finetune.py`. Do NOT bloat `supervised_train.py`; new trainers are new modules.
- `harness/colab.py` (Phase 2) — `mount_drive`, `stage_squat_videos`, atomic checkpoint, RNG capture/restore, `prune_checkpoints` reused. SSL pretrain needs an unlabeled-clip staging path (analogous to `stage_squat_videos`).
- Notebook Cell A + Step 0 bootstrap pattern from `03_squat_supervised_baseline.py` transfers.

### Claude's Discretion
- **Ensemble aggregation method** — user delegated; chose mean of sigmoids (D4) with full rationale.
- **Most of D2 (SSL recipe mechanics)** — delegated to research-phase by design, not user abdication.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents (researcher, pattern-mapper, planner, plan-checker) MUST read these before producing their artefacts.**

### Phase 4 handoff + master plan
- [`~/.claude/plans/handoff-2026-05-20-phase-4-md-ssl.md`](file:///c/Users/tsh_x/.claude/plans/handoff-2026-05-20-phase-4-md-ssl.md) — **comprehensive Phase 4 handoff**; the 19 open questions (lines 110-164); deliverables target list; honest risks
- [`~/.claude/plans/what-pushed-me-back-iridescent-pnueli.md`](file:///c/Users/tsh_x/.claude/plans/what-pushed-me-back-iridescent-pnueli.md) — master plan; Phase 4 paragraph (lines 96-98); F1 comparison targets (lines 56-63)
- [`.planning/PROJECT.md`](../../PROJECT.md), [`.planning/ROADMAP.md`](../../ROADMAP.md) (Phase 4 success criteria, lines 59-67), [`.planning/REQUIREMENTS.md`](../../REQUIREMENTS.md) (SQUAT-04/05, lines 21-22)

### Phase 3 (the baseline Phase 4 must beat + code to extend)
- [`.planning/phases/03-squat-supervised-baseline/03-01-SUMMARY.md`](../03-squat-supervised-baseline/03-01-SUMMARY.md) — baseline numbers (KIE 0.286 / KFE 0.800 / macro 0.543), overfit pattern, deviations, Phase 4 handoff section, hyperparameter considerations
- [`.planning/phases/03-squat-supervised-baseline/03-CONTEXT.md`](../03-squat-supervised-baseline/03-CONTEXT.md) — D1-D14 locked decisions Phase 4 inherits
- [`.planning/phases/03-squat-supervised-baseline/03-01-PLAN.md`](../03-squat-supervised-baseline/03-01-PLAN.md) — executed task structure; **NOTE for plan-checker: Phase 3's PLAN had wrong F1/PR-AUC arithmetic in acceptance criteria (caught by user, fixed in `fecd795`) — verify Phase 4 plan's expected metric values are mathematically correct**
- [`.planning/phases/03-squat-supervised-baseline/figures/results.pkl`](../03-squat-supervised-baseline/figures/results.pkl) — frozen Phase 3 numbers + raw scores/labels/clip_ids for the comparison chart

### Phase 2 (data pipeline + harness Phase 4 reuses)
- [`.planning/phases/02-squat-data-pipeline-colab-harness/02-01-SUMMARY.md`](../02-squat-data-pipeline-colab-harness/02-01-SUMMARY.md) — pipeline interfaces, deviations
- [`.planning/phases/02-squat-data-pipeline-colab-harness/02-CONTEXT.md`](../02-squat-data-pipeline-colab-harness/02-CONTEXT.md) — MD-SSL unlabeled-loader deferral note (line 161); spatial/temporal aug rationale; flip-OFF rationale (line 60)
- [`.planning/phases/02-squat-data-pipeline-colab-harness/CODE-RELEASE-NOTES.md`](../02-squat-data-pipeline-colab-harness/CODE-RELEASE-NOTES.md) — **`Code_Release/motion_disentanglement/` is empty (1-byte README)** — SSL reconstructed from paper + literature
- [`.planning/phases/01-dataset-consolidation-eda/01-DATASET-REPORT.md`](../01-dataset-consolidation-eda/01-DATASET-REPORT.md) — verified counts; clip-length distribution (49-404 frames, multi-rep tail); barbell-trajectory probe; the 19 `traj_nan` clips (matter for MD-SSL pairing); 4,970 unlabeled count

### Phase 2/3 code modules (contracts — do NOT redefine)
- [`backend/training/aqa/datasets/`](../../../backend/training/aqa/datasets/) — `splits.py`, `transforms.py`, `squat.py`
- [`backend/training/aqa/harness/`](../../../backend/training/aqa/harness/) — `_envinit.py`, `colab.py`, `tiny_train.py`, `supervised_train.py`
- [`backend/training/aqa/eval/`](../../../backend/training/aqa/eval/) — `metrics.py`, `test_metrics.py`
- [`backend/training/aqa/notebooks/03_squat_supervised_baseline.py`](../../../backend/training/aqa/notebooks/03_squat_supervised_baseline.py) — notebook conventions, Cell A bootstrap, Step 0

### Fitness-AQA paper + dataset
- `Fitness-AQA Paper.pdf` (Parmar et al., ECCV 2022, [arXiv:2202.14019](https://arxiv.org/abs/2202.14019)) — **§4 MD-SSL is the pretext-task/loss/sampling ground truth for D2; §5 is the downstream fine-tune recipe ground truth for D3**
- Drive unlabeled set: `My Drive/Fitness-AQA_dataset_release/Squat/Unlabeled_Dataset/` (4,970 clips + trajectories) — researcher probes the trajectory file format (open question 8)
- GYMetricPose (Gallardo 2024) + LMM (Dibenedetto 2025) — independent comparison points for the headline chart

### Working-agreement memories (load all)
[[feedback_interactive_execution]], [[feedback_one_cell_at_a_time_strict]], [[feedback_dont_agree]], [[feedback_working_style]], [[feedback_run_gsd_autonomously]], [[feedback_ai_correctness]], [[feedback_heavy_training_new_notebook]], [[feedback_notebook_disconnect_safe]], [[feedback_deliver_colab_as_ipynb]], [[reference_pytorch_persistent_workers]], [[reference_colab_module_reload_after_git_pull]], [[reference_best_pt_metrics_history_is_stale]], [[project_supervisor_visualizations]], [[project_form_correction_status]]

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`eval/metrics.py`** — pure F1/PR-AUC/threshold-sweep/confusion functions, sklearn-backed, no torch/IO. Phase 4 ensemble + TTA aggregation feed cleaned-up score arrays into these unchanged.
- **`harness/supervised_train.py`** — `build_model` (R(2+1)D-18 + `Linear(512,2)` + `in_features==512` assert), `seed_worker`, `_build_dataloaders` (direct construction bypassing Phase 2's `num_workers=0` clamp; `persistent_workers=num_workers>0`), `_val_pass`, `run_supervised_epoch` with the D8 checkpoint schema + best.pt write contract + early-stop. `md_finetune.py` parallels this with AdamW + dropout + the D6 runtime monitor.
- **`harness/colab.py`** — atomic Drive checkpoint (tmp + round-trip verify + `os.replace` + `latest.txt`-last), 4-RNG capture/restore, `prune_checkpoints(keep_last=3, keep_best=True)`, `stage_squat_videos` (labeled). SSL needs an unlabeled-staging analog.
- **Notebook Cell A + Step 0** (`03_*.py`) — clone-or-pull, `sys.path.insert`, PyAV-before-torch, Drive mount, idempotent staging cache-hit.

### Established Patterns
- **Joint multi-label head over shared backbone** for KIE/KFE (Phase 1 co-occurrence: 192/232 KIE+ also KFE+). MD-SSL is hypothesized to give the backbone a joint motion prior that the Kinetics-init fine-tune couldn't recover (Phase 3 finding: heads decorrelated despite shared backbone).
- **`BCEWithLogitsLoss(pos_weight=dataset.pos_weight)`** for the fine-tune imbalance (KIE w≈6.10, KFE w≈0.45) — unchanged from Phase 3 D2.
- **Disconnect-safe by construction** — checkpoint every epoch; resume from `latest.txt`; idempotent staging. Applies doubly to the 12-24h SSL pretrain (most likely to span multiple Colab sessions).
- **Atomic commit per task** (`feat(04)/fix(04)/test(04)/chore(04)` with task # + D-ID inline).

### Integration Points
- New SSL trainer + unlabeled dataset are net-new under `backend/training/aqa/`. Fine-tune trainer extends Phase 3's harness patterns. Nothing touches the FastAPI runtime (Phase 5).
- Phase 4's production checkpoint declaration in SUMMARY is the input contract for Phase 5's inference service.

</code_context>

<specifics>
## Specific Ideas

- **F1 comparison targets** (master plan + paper Table 2): Phase 3 baseline KIE 0.286 / KFE 0.800 / macro 0.543; paper Kinetics row KIE 0.297 / KFE 0.818 / macro ~0.557; **paper MD target KIE 0.419 / KFE 0.834 / macro ~0.626**. User wants to match-or-beat the MD row.
- **Realistic outcome distribution** (handoff): macro 0.62-0.66 likely; ~35% beat by ≥+0.03, ~45% match ±0.02, ~15% slightly below, ~5% no meaningful lift.
- **Split sizes:** train 1,136 / val 243 / test 244; unlabeled 4,970 + trajectories.
- **Class balance (train):** KIE+ 14% (w≈6.10), KFE+ 68% (w≈0.45).
- **VRAM reality check:** Phase 3 hit 15.22 GB backward at batch 16 (RESEARCH estimated 8-12). SSL contrastive wants large batches — re-estimate before committing batch size; gradient accumulation if batch 256 doesn't fit 24 GB.
- **Drive checkpoint layout:** `My Drive/FitNova/checkpoints/phase04/md_pretrain_v1/backbone.pt` (SSL) + `md_finetune_seed{42,1337,7}/best.pt` (×3) + `results.pkl` backup.
- **Run-name convention:** `md_pretrain_v1` (SSL), `md_finetune_seed42` / `_seed1337` / `_seed7` (fine-tunes).
- **Wall-time anchor:** Phase 3 fine-tune ~5.3 min/epoch on L4. Phase 4 fine-tune (50ep × 3 seeds) ≈ 13-16h + SSL 12-24h. Resumable harness is mandatory.

</specifics>

<deferred>
## Deferred Ideas

- **F — auxiliary trajectory-prediction head** — explicitly dropped this phase (D1). Revisit only if a future milestone wants the full paper method and has the +12-24h re-pretrain budget.
- **Warmup ablation** — only if early fine-tune epochs show AdamW+SSL-init instability (D3).
- **Fine-tune LR ablation** — researcher checks paper §5 first; 1e-4 default (D3).
- **Regularization sweep (light/moderate/aggressive on seed 42)** — the designated **remediation pass** if the ensemble doesn't beat Phase 3 (D6 no-lift policy).
- **5-seed ensemble extension** — only if 3-seed variance is too wide and compute permits after Phase 4 close (D4).
- **TorchCodec decoder migration** — Phase 2/3 carry; only if `decode_clip` is the SSL throughput bottleneck on 4,970 clips (researcher flags).
- **OHP / BarbellRow / Shallow-Squat** (Phases 6/7), **API integration** (Phase 5).

### Reviewed Todos (not folded)
None — no pending todos matched Phase 4 scope.

</deferred>

---

*Phase: 04-squat-motion-disentangling-ssl*
*Context gathered: 2026-05-21 via interactive /gsd:discuss-phase 4*
