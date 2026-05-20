# Phase 03: Squat Supervised Baseline — Context

**Gathered:** 2026-05-20
**Status:** Ready for planning
**Source:** Orchestrator-synthesized from master plan (`~/.claude/plans/what-pushed-me-back-iridescent-pnueli.md`), Phase 2 SUMMARY (`02-01-SUMMARY.md`), Phase 2 PLAN's `<interfaces>` block, ROADMAP success criteria, REQUIREMENTS (SQUAT-03), and the user's working-agreement memories. Mirrors the Phase 1/2 precedent ([02-CONTEXT.md:5](../02-squat-data-pipeline-colab-harness/02-CONTEXT.md) — *"No discuss-phase pass — the master plan and Phase 1 closeout already discharge that function"*). User has explicitly authorized autonomous `/gsd:plan-phase 3` execution for this phase ([[feedback_run_gsd_autonomously]]).

<domain>
## Phase Boundary

**This phase delivers:**

1. A trained **R(2+1)D-18** supervised baseline for joint Squat KIE/KFE error detection. Backbone is `torchvision.models.video.r2plus1d_18` initialised from `R2Plus1D_18_Weights.KINETICS400_V1`. A two-output joint multi-label head shares the backbone — justified by Phase 1's co-occurrence finding (192/232 KIE+ clips also KFE+; [02-CONTEXT.md:43](../02-squat-data-pipeline-colab-harness/02-CONTEXT.md)).
2. **F1-score per error** on the official Squat **test** split (244 clips), with the decision threshold tuned on the official **val** split (243 clips). Targets per master plan line 58: **KFE ≈ 0.818** (paper Kinetics baseline), **KIE ≈ 0.297**. ROADMAP success criterion 2 accepts "in the range of the paper's supervised baseline" — interpreted as within roughly ±0.05 of paper.
3. **PR-AUC per error** on the test split as a threshold-free supplement (master plan line 54).
4. **Visualization deliverables**, saved as PNG to `.planning/phases/03-squat-supervised-baseline/figures/` AND embedded in the Phase 3 notebook (per `[[project_supervisor_visualizations]]` — defense 2026-06-03 weights visualizations heavily):
   - Train/val loss + per-epoch F1 curves (one figure, multiple axes).
   - Confusion matrix on test, KIE.
   - Confusion matrix on test, KFE.
   - PR curve on test, KIE.
   - PR curve on test, KFE.
   - Sample-prediction grid (true-positive, false-positive, false-negative frames with predicted scores) — proves the model isn't gaming the metric.
5. **Phase 3 Colab notebook** at `backend/training/aqa/notebooks/03_squat_supervised_baseline.py` (jupytext percent-format, following Phase 2 conventions verbatim), disconnect-safe by construction.
6. **Extended training-loop module** at `backend/training/aqa/harness/supervised_train.py` (full R(2+1)D-18 trainer; extends Phase 2's `tiny_train.py` patterns — does NOT replace it; `tiny_train.py` stays as the Phase 2 reference).
7. **Evaluation module** at `backend/training/aqa/eval/metrics.py` (F1 per error, PR-AUC, threshold sweep on val, confusion-matrix computation).
8. End-of-phase **`03-01-SUMMARY.md`** documenting trained-model artefacts (Drive paths to `best.pt` + final epoch checkpoint), deviations from PLAN.md, Phase 4 handoff (which weights file to load for SSL fine-tuning).

**This phase does NOT deliver:**

- Motion-Disentangling self-supervised pretraining (Phase 4).
- Image-modality models for Shallow-Squat (Phase 7).
- OHP or BarbellRow models (Phases 6, 7).
- Backend inference integration / FastAPI changes (Phase 5).
- Polished Flutter UI for form correction (later milestone).
- Graded severity output (out of scope — labels are binary).
- Multi-seed variance estimation as a headline number (paper reports point numbers; a second seed is optional if wall-time permits).

**Requirements satisfied:** **SQUAT-03** (R(2+1)D-18 baseline, F1 per error on the official Squat test split).

</domain>

<decisions>
## Implementation Decisions (Locked)

### D1 — Framework, target model, fine-tune strategy
- **PyTorch + torchvision R(2+1)D-18, Kinetics-400-V1 init.** Construct as `torchvision.models.video.r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)`. Replace the final fully-connected with `nn.Linear(in_features=512, out_features=2)` for joint KIE/KFE output. Verify `in_features=512` against the torchvision implementation at module-load time.
- **Fully fine-tuned end-to-end** — all backbone layers trainable. Paper § 4.1 supervised-baseline does end-to-end fine-tuning; backbone-frozen is faster but lower-fidelity and not what the paper reports. Researcher to confirm against the paper text and override D1 only with explicit citation.

### D2 — Loss function
- **`nn.BCEWithLogitsLoss(pos_weight=dataset.pos_weight)`** consumed directly from `SquatKIEKFEDataset.pos_weight` (train-derived `Tensor([w_KIE≈6.10, w_KFE≈0.45])` per Phase 2's [`squat.py:39-56`](../../../backend/training/aqa/datasets/squat.py)). One loss, joint over both outputs. Per Phase 2 D9, weighting handles class imbalance — NOT `WeightedRandomSampler`.
- Compute the loss on the **mean-reduced** (default) output so a single scalar enters `loss.backward()`.

### D3 — Data pipeline (consumed verbatim from Phase 2)
- Use [`backend.training.aqa.datasets.squat.build_loaders`](../../../backend/training/aqa/datasets/squat.py) with kwargs `drive_root`, `videos_root`, `num_frames=32`, `crop_size=112`, `train_aug=True`, `train_jitter_frames=2`.
- Output shape per item: `(3, 32, 112, 112)` float32, Kinetics-normalised. Per-batch: `(B, 3, 32, 112, 112)`.
- `dataset.pos_weight` is identical across train/val/test loader instances (computed once from train).
- Do **NOT** redefine splits, transforms, or the dataset — Phase 2 interfaces are the contract.

### D4 — `num_workers` lifted (resolves Phase 2 R10)
**Phase 2's `num_workers=0` constraint is no longer load-bearing.** Bitwise resume determinism was a Phase 2 guarantee; for real training, resume correctness comes from the checkpoint payload (model weights + optimizer + RNG state) + the atomic-write contract — NOT from byte-equality of loss trajectories.
- Set `num_workers >= 2` (concrete value picked by the researcher based on L4 GPU throughput; 2–4 is the expected range).
- Each worker re-seeds its torch/numpy/python RNG via a `worker_init_fn(worker_id)` that does `seed_offset = base_seed + worker_id; torch.manual_seed(seed_offset); np.random.seed(seed_offset); random.seed(seed_offset)`.
- The trainer constructs `DataLoader` directly (does NOT call `build_loaders` if `num_workers > 0` — or `build_loaders` is extended in this phase to accept a `worker_init_fn` kwarg; planner picks the cleaner option and documents inline).

### D5 — Augmentation policy
Locked defaults (Phase 2 D5):
- **Spatial:** short-side resize 128 → 112² **random** crop on train, **center** crop on val/test.
- **Temporal:** ±2-frame jitter on train, deterministic `round(linspace(0, F-1, 32))` on val/test.
- **Horizontal flip:** **OFF** by default (Phase 2 D5 / R5 — paper line 576 omits flip; CVCSPC code has it commented out).
- **Ablation hook:** trainer accepts `crop_size`, `train_jitter_frames`, and a `flip_aug` toggle. The PLAN.md must produce at least the **locked-default run**. Ablation runs (flip-on, crop=224) are nice-to-have if KIE F1 disappoints; out of scope otherwise.

### D6 — Optimizer / scheduler / training budget — RESEARCHER RESOLVES
Provisional defaults (researcher must confirm or override against paper § 4.1):
- **Optimizer:** Adam, `lr=1e-4` (CVCSPC convention used by Phase 2's tiny train; paper may specify SGD+momentum for the video baseline — researcher confirms).
- **Scheduler:** cosine annealing over the full epoch budget. Researcher confirms.
- **Epoch budget:** provisional 50 epochs. Early-stop on val-macro-F1 plateau (8-epoch patience). Researcher provides paper's reported convergence and a back-of-envelope wall-time estimate on **L4 GPU** (~16 GB VRAM; ~1.5–2× T4 throughput).
- **Batch size:** provisional 16 (R(2+1)D-18 + 32-frame 112² clips × 16 ≈ 7–8 GB VRAM on L4 — fits with headroom). Researcher confirms.

Whatever the researcher picks, every value is cited inline in 03-RESEARCH.md to either a paper section, the torchvision r2plus1d_18 fine-tune recipe, or readable portions of `Code_Release/pose_contrastive_learning/train_test.py` per [`CODE-RELEASE-NOTES.md`](../02-squat-data-pipeline-colab-harness/CODE-RELEASE-NOTES.md).

### D7 — Threshold tuning protocol
For each error head **independently**:
1. After training, evaluate per-clip sigmoid scores on the **val** split.
2. Sweep thresholds `0.05 → 0.95` step `0.01` (91 points) with a tqdm progress bar.
3. Pick the threshold that maximises F1 on val for that error.
4. Apply the val-tuned threshold to **test** and report F1.

Both per-error thresholds are stored in the final checkpoint as `best_thresholds = {"kie": float, "kfe": float}`. PR-AUC is computed on raw scores (threshold-free).

### D8 — Checkpoint payload schema (extends Phase 2 D10/F6 — does NOT replace)
Phase 2's payload schema ([`tiny_train.py:230-241`](../../../backend/training/aqa/harness/tiny_train.py)) is `{epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, rng_state, metrics_history, config_hash, config_repr, code_version}`. Phase 3 extends:
- `scheduler_state_dict` — actual state (Phase 2 set `None` for the tiny run).
- `best_f1_val` — best macro-F1 on val seen so far in this run; updated each epoch.
- `best_thresholds` — `{"kie": float, "kfe": float}`; set after the threshold sweep, `None` until then.
- `metrics_history[i]` — per-epoch dict gains keys `val_f1_kie`, `val_f1_kfe`, `val_pr_auc_kie`, `val_pr_auc_kfe`, `val_macro_f1` alongside Phase 2's loss fields.
- `code_version` — `"phase03-supervised-baseline"`.

A separate **`best.pt`** is written by [`atomic_save_checkpoint`](../../../backend/training/aqa/harness/colab.py) whenever **val macro-F1** improves; [`prune_checkpoints(keep_last=3, keep_best=True)`](../../../backend/training/aqa/harness/colab.py) runs after each epoch (Phase 2 already supports this).

### D9 — Evaluation harness
- F1: `sklearn.metrics.f1_score(y_true, y_pred, pos_label=1)` per error, independently. NO macro/micro averaging in the headline number — paper reports per-error.
- PR-AUC: `sklearn.metrics.average_precision_score(y_true, y_score)` per error.
- Confusion matrix: `sklearn.metrics.confusion_matrix` per error.
- All computations happen in CPU memory after gathering all val/test scores; no streaming necessary at 243 + 244 clip sizes.
- Evaluation is **disconnect-safe**: results dict pickled to Drive (`results.pkl`) and figures saved as PNG **before** any `plt.show()`. Defensive `Path(out).parent.mkdir(parents=True, exist_ok=True)` on every save.
- scikit-learn is already a Phase 2 dependency (`backend/requirements.txt:scikit-learn>=1.3.0`).

### D10 — Notebook + code-module discipline
- **Heavy-training notebook** = fresh Colab session on **L4 GPU** (per `[[feedback_heavy_training_new_notebook]]`). User has already moved to one before this conversation per the briefing.
- New notebook `backend/training/aqa/notebooks/03_squat_supervised_baseline.py` follows the [`02_squat_pipeline_harness.py`](../../../backend/training/aqa/notebooks/02_squat_pipeline_harness.py) jupytext percent-format conventions verbatim (`# %%` cell markers, `# %% [markdown]` for markdown, Step N markdown headers, paste-back verification pattern, one runnable unit per task).
- **Cell 0 first line is `from backend.training.aqa.harness import _envinit`** (F8 import-order constraint; sets `CUBLAS_WORKSPACE_CONFIG` before `import torch`).
- **Step 0** reuses Phase 2's PyAV-install-before-torch + Drive mount + idempotent `stage_squat_videos` cache-hit path (a fresh-session bootstrap is a cache hit if `/content/` survived, else re-downloads — both paths supported).
- **Trainer** goes in `backend/training/aqa/harness/supervised_train.py` (do NOT bloat `tiny_train.py`; keep it as the Phase 2 reference). Trainer mirrors `run_tiny_epoch`'s signature shape (`run_name`, `drive_root`, `videos_root`, `seed`, `batch_size`, hyperparameters, `resume`, `max_epochs`) extended with the new D6/D8 fields.
- **Metrics** live in `backend/training/aqa/eval/metrics.py` (new module, single responsibility: F1 + PR-AUC + threshold sweep + confusion matrix). Pure functions; no side effects.
- **tqdm progress bars** on: training epoch (per-batch), val pass (per-batch), test pass (per-batch), threshold sweep (per-threshold), any plotting loop that walks files.

### D11 — Working agreement (locked from memories)
- **Interactive execution** ([[feedback_interactive_execution]]): every PLAN.md task is **one runnable unit**; user pastes outputs back; no blind multi-cell runs. Tasks granular enough that a single paste-back surfaces failures immediately.
- **Disconnect-safe** ([[feedback_notebook_disconnect_safe]]): atomic checkpoint writes with `latest.txt`-last; full 4-RNG capture/restore at every epoch boundary; `stage_squat_videos` cache hit on every re-run; Drive scans cached (none expected in Phase 3 — Phase 2 already handled the expensive ones).
- **Heavy-training fresh notebook** ([[feedback_heavy_training_new_notebook]]) — already in effect for this phase per the user's briefing.
- **Progress bars** everywhere long-running (D9 + D10 above).
- **Visualizations first-class** ([[project_supervisor_visualizations]]): saved as PNG to `figures/` **before** `plt.show()`; defensive parent-dir mkdir.
- **AI correctness** ([[feedback_ai_correctness]]): every hyperparameter cited to paper §, torchvision docs, or sklearn docs; deviations called out inline with rationale.
- **No laziness** ([[feedback_working_style]]): work driven to verifiable conclusion; don't offload effort.
- **GSD discipline**: atomic commit per task (`feat(03)/fix(03)/chore(03)` with task # + D-ID reference inline); end-of-phase SUMMARY.md; ROADMAP / STATE / REQUIREMENTS updated.
- **No deadline-pressure framing** ([[feedback_working_style]]): don't justify shortcuts with "the 2026-06-03 defense".

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents (researcher, pattern-mapper, planner, plan-checker) MUST read these before producing their artefacts.**

### Master & milestone
- [`~/.claude/plans/what-pushed-me-back-iridescent-pnueli.md`](file:///c/Users/tsh_x/.claude/plans/what-pushed-me-back-iridescent-pnueli.md) — Master rebuild plan; Phase 3 paragraph at line 92–94; target F1 numbers at line 58
- [`.planning/PROJECT.md`](../../PROJECT.md) — Project framing
- [`.planning/ROADMAP.md`](../../ROADMAP.md) — Phase 3 success criteria (lines 46–54); `Depends on: Phase 2`
- [`.planning/REQUIREMENTS.md`](../../REQUIREMENTS.md) — SQUAT-03 at line 20

### Phase 2 (handoff — the surface Phase 3 builds on)
- [`.planning/phases/02-squat-data-pipeline-colab-harness/02-01-SUMMARY.md`](../02-squat-data-pipeline-colab-harness/02-01-SUMMARY.md) — What Phase 2 shipped; interfaces; deviations
- [`.planning/phases/02-squat-data-pipeline-colab-harness/02-01-PLAN.md`](../02-squat-data-pipeline-colab-harness/02-01-PLAN.md) — `<interfaces>` block (full schema for splits/transforms/squat/colab/tiny_train); risk register (R4 paper-vs-Code_Release gap; R8 long multi-rep clips; R9 crop-area loss; R10 num_workers — resolved by D4; R12 decode memory budget)
- [`.planning/phases/02-squat-data-pipeline-colab-harness/02-CONTEXT.md`](../02-squat-data-pipeline-colab-harness/02-CONTEXT.md) — Phase 2 locked decisions; spatial preprocessing rationale
- [`.planning/phases/02-squat-data-pipeline-colab-harness/02-RESEARCH.md`](../02-squat-data-pipeline-colab-harness/02-RESEARCH.md) — Paper + Code_Release findings; F11 decode windowing
- [`.planning/phases/02-squat-data-pipeline-colab-harness/CODE-RELEASE-NOTES.md`](../02-squat-data-pipeline-colab-harness/CODE-RELEASE-NOTES.md) — Upstream `Code_Release/` gap analysis (`motion_disentanglement/` is empty 2-byte README; CVCSPC has broken imports)
- [`.planning/phases/01-dataset-consolidation-eda/01-DATASET-REPORT.md`](../01-dataset-consolidation-eda/01-DATASET-REPORT.md) — Verified counts (1136/243/244 train/val/test; KIE+ 232, KFE+ 1109)

### Phase 2 code modules (contracts Phase 3 consumes — do NOT redefine)
- [`backend/training/aqa/datasets/splits.py`](../../../backend/training/aqa/datasets/splits.py) — `ClipRecord`, `index(split, ...) -> list[ClipRecord]`, `expected_counts()`
- [`backend/training/aqa/datasets/transforms.py`](../../../backend/training/aqa/datasets/transforms.py) — `KINETICS_MEAN/STD`, `uniform_sample_indices`, `decode_clip` (F11 windowed), `spatial_train`, `spatial_val`
- [`backend/training/aqa/datasets/squat.py`](../../../backend/training/aqa/datasets/squat.py) — `SquatKIEKFEDataset`, `.pos_weight`, `build_loaders` factory
- [`backend/training/aqa/harness/_envinit.py`](../../../backend/training/aqa/harness/_envinit.py) — F8 `CUBLAS_WORKSPACE_CONFIG` shim
- [`backend/training/aqa/harness/colab.py`](../../../backend/training/aqa/harness/colab.py) — `mount_drive`, `stage_squat_videos`, `capture_rng_state`, `restore_rng_state`, `hash_config`, `atomic_save_checkpoint`, `load_latest_checkpoint`, `prune_checkpoints`, `CheckpointConfigMismatchError`
- [`backend/training/aqa/harness/tiny_train.py`](../../../backend/training/aqa/harness/tiny_train.py) — Reference checkpoint-payload schema (lines 230–241) + multi-epoch resume pattern
- [`backend/training/aqa/notebooks/02_squat_pipeline_harness.py`](../../../backend/training/aqa/notebooks/02_squat_pipeline_harness.py) — Notebook conventions; Step 0 bootstrap; cell-by-cell paste-back pattern

### Phase 2 figures (visualization precedent for Phase 3)
- [`.planning/phases/02-squat-data-pipeline-colab-harness/figures/decoded_batch.png`](../02-squat-data-pipeline-colab-harness/figures/decoded_batch.png) — Phase 2 supervisor figure (2×8 grid + labels)
- [`.planning/phases/02-squat-data-pipeline-colab-harness/figures/tiny_train_loss.png`](../02-squat-data-pipeline-colab-harness/figures/tiny_train_loss.png) — Phase 2 bitwise-resume proof (overlay plot pattern Phase 3 reuses for train/val curves)

### Fitness-AQA paper + official repo
- `Fitness-AQA/` (git submodule cloned in Phase 2; not committed, listed in `.gitignore`) — `Code_Release/motion_disentanglement/` is empty; `Code_Release/pose_contrastive_learning/` is the CVCSPC image-side reference (broken imports per CODE-RELEASE-NOTES); `Code_Release/data_augmentations/` for any flip/crop reference
- `Fitness-AQA Paper.pdf` (Parmar et al., ECCV 2022, [arXiv:2202.14019](https://arxiv.org/abs/2202.14019)) — **§ 4.1 supervised baseline is the optimizer/scheduler/budget ground truth** for D6

### Working-agreement memories (load all)
[[feedback_interactive_execution]], [[feedback_notebook_disconnect_safe]], [[feedback_heavy_training_new_notebook]], [[feedback_ai_correctness]], [[feedback_working_style]], [[feedback_dont_agree]], [[feedback_run_gsd_autonomously]], [[project_supervisor_visualizations]], [[project_form_correction_status]], [[project_gsd_adoption]]

</canonical_refs>

<specifics>
## Specific Targets, Numbers, and Patterns

- **F1 targets (paper Kinetics supervised baseline, master plan line 58):** KFE ≈ 0.818, KIE ≈ 0.297. ±0.05 of paper = "in range" per ROADMAP success criterion 2.
- **Split sizes ([`splits.expected_counts()`](../../../backend/training/aqa/datasets/splits.py)):** train 1,136 / val 243 / test 244.
- **Class balance from Phase 2 (train, [`squat.py:39-56`](../../../backend/training/aqa/datasets/squat.py)):** KIE+ 160/1136 (14.08%) → `w_KIE ≈ 6.10`; KFE+ 782/1136 (68.84%) → `w_KFE ≈ 0.453`.
- **Input shape per batch:** `(B, 3, 32, 112, 112)` float32, Kinetics-400 normalised.
- **Output:** logits `[B, 2]` for `(KIE, KFE)`. Sigmoid on inference.
- **Best-checkpoint selection metric:** **macro-F1 on val** (mean of per-error F1s computed at threshold 0.5 during training). The independent per-error threshold sweep (D7) happens AFTER training completes and is used only for the final test report — does not feed back into training.
- **Notebook → script split:** notebook drives end-to-end; trainer + metrics live in importable, testable modules. Notebook cells call these.
- **Atomic commit template (per Phase 2 precedent):** `feat(03): <Task N description> (D-X, F-Y)` where D-X cites the relevant CONTEXT decision and F-Y cites the relevant PLAN finding/risk.
- **Drive checkpoint layout:** `My Drive/FitNova/checkpoints/phase03/{run_name}/epoch_{NNN}.pt`, `best.pt`, `latest.txt`. Same pattern as Phase 2.
- **Run name convention:** `r2plus1d18_squat_supervised_v1` for the locked-default run; ablation variants append `_flip` / `_crop224` / etc.

</specifics>

<deferred>
## Deferred Ideas

- **Motion-Disentangling SSL pretraining** — Phase 4. Phase 3's checkpoint `config_repr` + `code_version` give Phase 4 a clean fork point.
- **Multi-seed reporting (variance estimation)** — paper reports point numbers; Phase 3 produces a single-seed baseline as the headline number. A second seed is **optional** if wall-time budget permits and the researcher recommends it; not blocking for ROADMAP success criterion 2.
- **Crop-size ablation (224 vs 112)** — Phase 2 R9. Phase 3 baseline is 112 (D5). If KIE F1 disappoints relative to paper, the ablation belongs in a Phase 3 follow-up plan only; otherwise carries to Phase 4.
- **Horizontal flip ablation** — Phase 2 R5 / D5 default OFF. Same disposition as crop ablation — Phase 3 follow-up only if KIE F1 disappoints.
- **Backbone freezing experiments** — D1 locks full fine-tune. Frozen-backbone-with-trainable-head is much faster but lower-fidelity to paper; not in Phase 3 scope.
- **TorchCodec migration** — Phase 2 D6 swap target when Colab's torchvision crosses 0.24 (`read_video` removal). Not Phase 3 unless `decode_clip` is the throughput bottleneck (researcher to flag if so).
- **OHP / BarbellRow / Shallow-Squat** — Phases 6/7.
- **API integration** — Phase 5.

</deferred>

---

*Phase: 03-squat-supervised-baseline*
*Context synthesized: 2026-05-20 by orchestrator from master plan + Phase 2 SUMMARY + ROADMAP + REQUIREMENTS + working-agreement memories*
