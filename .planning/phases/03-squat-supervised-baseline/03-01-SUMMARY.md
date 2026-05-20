---
phase: 03-squat-supervised-baseline
plan: 01
status: complete
completed: 2026-05-20
---

# Phase 3 · Plan 01 — Summary

## What was planned

Sixteen tasks under `03-01-PLAN.md` covering: scaffold `eval/` package + figures dir (T1); implement `eval/metrics.py` — `f1_per_error`, `pr_auc_per_error`, `threshold_sweep` (sklearn.precision_recall_curve per RESEARCH §5), `confusion_matrix_per_error` (T2); 9 unit tests covering SQUAT-03-b/c/d (T3); Wave-0 pytest gate (T4); implement `harness/supervised_train.py` — `SupervisedConfig` (D6 paper-cited Adam lr=1e-4, 50 epochs / 8-patience, batch 16 / 4 workers, cosine annealing, no AMP), `build_model` (R(2+1)D-18 + Linear(512,2) head with in_features assertion), `seed_worker` (PyTorch official verbatim per RESEARCH §2), `_build_dataloaders` (direct DataLoader construction bypassing Phase 2's `num_workers=0` clamp), `_val_pass`, `run_supervised_epoch` with D8 checkpoint payload schema + D13 `best.pt` write contract + 8-epoch val-macro-F1 early-stop (T5); 2 unit tests covering SQUAT-03-a (head shape) + SQUAT-03-e (schema keys) (T6); scaffold notebook Step 0 with Cell A bootstrap + dep version probe + L4 GPU assertion + `stage_squat_videos` cache hit (T7); Step 1 dataset smoke (T8); Step 2 model + VRAM probe (T9); Step 3 epoch-0 timing gate as blocking `checkpoint:decision` (T10); Step 4 full training run (T11); Step 5 threshold sweep on val + `best_thresholds` written into `best.pt` with `latest.txt` preservation (T12); Step 6 test evaluation at val-tuned thresholds + `results.pkl` (T13); Step 7 all 7 supervisor figures (T14); blocking human-verify (T15); closeout SUMMARY (T16). Covers requirement SQUAT-03.

## What was done

Executed interactively in Colab on **L4 GPU (24 GB VRAM)** against the `Ibrahim-Shahin1/fitnova` repo (branch `fresh-start`). All 16 tasks shipped one runnable unit at a time per `[[feedback_interactive_execution]]`. Every code task got its own atomic commit (`feat(03)`, `fix(03)`, `test(03)`, `chore(03)`) referencing the task number and CONTEXT/PLAN decision IDs (D1–D14) inline. Notebook delivered as paired jupytext `.py` + Colab-runnable `.ipynb` per `[[feedback_deliver_colab_as_ipynb]]`.

Headline test results at val-tuned per-error thresholds (244-clip test split): **KIE F1 = 0.2857, KFE F1 = 0.8000, macro F1 = 0.5429**. PR-AUC KIE = 0.2645, PR-AUC KFE = 0.7640. Best epoch was 3 of 12 trained (early-stop at 11 on 8-epoch val-macro-F1 patience). The trained `best.pt` (358 MB) is at `My Drive/FitNova/checkpoints/phase03/r2plus1d18_squat_supervised_v1/best.pt` with `best_thresholds = {"kie": 0.363, "kfe": 0.571}`.

Wall-time on L4: ~5.3 min/epoch (Step 3 timing probe estimate of 4.44 h was conservative; full run took ~64 min before early-stop). VRAM at batch 16 fp32: 3.44 GB forward, 15.22 GB backward (within 20 GB gate, higher than RESEARCH §10's 8-12 GB estimate). R(2+1)D-18 param count: 31,301,151 (matches hand-calculation; the planner's 31,303,927 estimate was off by ~2,776, absorbed by my ±2% range check).

## Key findings (headlines)

- **End-to-end fine-tune overfits hard on 1136 train clips.** Train loss collapsed 0.83 → 0.07 (12×). Val loss diverged 0.81 → 2.27 (2.8×). Best val macro-F1 = 0.5454 at epoch 3; subsequent 8 epochs trained without improvement before early-stop fired. **This is precisely the overfit pattern the paper's domain-knowledge SSL is designed to prevent — the empirical motivation for Phase 4.**
- **Test F1 matches paper's Kinetics frozen-backbone row within 2% on every metric.** Paper Table 2 Kinetics: KIE 0.297 / KFE 0.818 / macro ~0.557. Us: 0.2857 / 0.8000 / 0.5429. Deltas: -0.011 / -0.018 / -0.014. ROADMAP success criterion 2 ("in the range of the paper's supervised baseline") **satisfied** — within 5% on every metric.
- **Did NOT reach the paper's MD end-to-end row.** Paper MD: KIE 0.419 / KFE 0.834 / macro ~0.626. Gap of 0.13 KIE-F1 and 0.083 macro-F1 — exactly the gap that domain-knowledge SSL is supposed to close in Phase 4.
- **KIE PR-AUC = 0.2645 is only 1.8× random** (positive rate 0.148). Model has weak KIE ranking ability. Heavy false-positive bias on KIE (precision 0.20, recall 0.47). KFE PR-AUC = 0.7640 vs random 0.693 is meaningful uplift.
- **KIE/KFE head decorrelation despite shared backbone.** Clip `48163_1` (and similar `47658_6`) score 1.00 on KIE (correct TP) but 0.10 on KFE (FN despite `label_kfe = 1`). Phase 1 measured 192/232 (83%) of KIE+ train clips also KFE+, so the joint structure exists in data. Model's two output heads learned per-error features without leveraging that joint structure — fine-tune from Kinetics init couldn't recover it from 1136 clips. Phase 4's MD pretraining is the candidate fix (gives backbone a joint motion representation prior).
- **Threshold tuning on val gave only +0.006 macro F1.** sklearn.precision_recall_curve sweep picked KIE threshold 0.363 (below 0.5 — expected for 14% positive class) and KFE threshold 0.571 (above 0.5 — expected for 68% majority). Model's sigmoid output distribution is already well-calibrated near 0.5; tuning didn't help much.
- **Atomic-write contract holds at 358 MB checkpoint scale on Drive FUSE.** RESEARCH §7 prediction (300-500 MB) was accurate. T-03-02 mitigation verified at production scale during Step 3 timing probe + every training epoch's atomic save.
- **`persistent_workers=True` is REQUIRED for multi-worker DataLoaders** to avoid PyTorch #40157 stderr spam at end-of-epoch shutdown. Surfaced in Phase 3 because Phase 2 hardcoded `num_workers=0`. Fix shipped; memory entry `[[reference_pytorch_persistent_workers]]` codified for Phase 4+.
- **`map_location='cpu'` is REQUIRED for `load_latest_checkpoint`** when restoring CUDA RNG state — `torch.cuda.set_rng_state_all` rejects CUDA ByteTensors with "RNG state must be a torch.ByteTensor" (the type system distinguishes CPU vs CUDA ByteTensors). Fix shipped.
- **`best.pt`'s `metrics_history` is frozen at the last best epoch**, NOT the full training run. For visualization, load `metrics_history` from the `latest.txt`-pointed checkpoint instead. Memory entry `[[reference_best_pt_metrics_history_is_stale]]` codified for Phase 4+.

## Deviations from PLAN

- **F1/PR-AUC arithmetic in PLAN Task 2/3 acceptance was wrong** (planner conflated values across thresholds). Correct values: `f1_per_error([1,0,1,0,1], [1,0,1,0,0]) = 0.8` (not 0.857), `threshold_sweep(...) = (0.4, 1.0)` (not (0.4, 0.857)), `pr_auc_per_error(...) = 1.0` (not "0.8-0.9"), confusion matrix correct. Implementation was right per sklearn; only the PLAN's expected values were wrong. Corrected in commit `fecd795`. The plan-checker didn't catch this — noted for future plan-checker improvements.
- **Tasks 8/9/10 cells over-committed in a batch before Step 0 paste-back** (working-agreement breach per `[[feedback_interactive_execution]]`). Reverted via three `git revert` commits (`e00b238`, `003da65`, `481e5fa`). Re-shipped per-task with paste-back gating. Memory entry `[[feedback_one_cell_at_a_time_strict]]` codified the rule for Phase 4+.
- **Notebook delivered as `.py` jupytext only on first ship** — user couldn't open it in Colab without conversion. Added `.ipynb` pair in commit `49c5ca6`. Memory entry `[[feedback_deliver_colab_as_ipynb]]` codified the rule.
- **`Cell A` repo bootstrap missing on first `.ipynb`** — user got `ModuleNotFoundError: No module named 'backend'`. Added Cell A (clone-or-pull + `sys.path.insert`) in commit `49c5ca6`.
- **DataLoader `persistent_workers=False` (default) produced PyTorch #40157 spam** at every end-of-epoch shutdown — user couldn't see tqdm bars or per-epoch metric lines for 34 min. Fix shipped in commit `5cec2f5` (`persistent_workers=config.num_workers > 0`). User had to restart Colab runtime to clear stale `sys.modules` cache and re-run Step 4; `resume=True` picked up from `epoch_005.pt` cleanly.
- **`load_latest_checkpoint(map_location='cuda')` corrupted RNG ByteTensors** on resume — `torch.cuda.set_rng_state_all` rejects CUDA ByteTensors. Fix shipped in commit `a0841b4` (`map_location='cpu'` — mirrors Phase 2's `tiny_train.py` default).
- **Module reload required after `git pull` in running Colab** — Python's `sys.modules` cache is stale after pulling code changes. Surfaced when user pulled the `map_location='cpu'` fix but kernel still ran the old code. Memory entry `[[reference_colab_module_reload_after_git_pull]]` codified for Phase 4+.
- **Step 6 initially saved `metrics_history` from `best.pt`** (which is frozen at the last best epoch) → `results.pkl` had only 4 entries → `training_curves.png` cut off at epoch 3 instead of showing the full overfit divergence over 12 epochs. Fix shipped in commit `e50fff9` (load `metrics_history` from `latest.txt`-pointed checkpoint).
- **`atomic_save_checkpoint(payload, best_path)` would overwrite `latest.txt` to `'best.pt'`** (Phase 2's atomic-write contract always updates latest.txt). Step 5 inline-saves and restores `latest.txt` around the `best.pt` update so the resume contract stays intact. Phase 4's `best.pt` write contract should encode this explicitly (consider adding `update_latest=False` kwarg to `atomic_save_checkpoint`).
- **VRAM at backward pass = 15.22 GB on L4** (RESEARCH §10 estimate: 8-12 GB). Within the 20 GB gate but higher than expected. Likely because the RESEARCH estimate didn't account for activation storage at 32-frame depth. Phase 4 should re-estimate before committing to batch sizes.
- **R(2+1)D-18 param count = 31,301,151** (RESEARCH §3 / planner's value: 31,303,927). Discrepancy of 2,776 absorbed by my ±2% sanity range. Origin of the off-by-2.7k is likely BN running-mean params shifting between torchvision Kinetics-V1 weight pubs.

## Acceptance criteria (ROADMAP success criteria + PLAN.md verification block)

- [x] **R(2+1)D-18 (Kinetics-V1-init) model trains to convergence on the labeled Squat set** — Task 11: 12 epochs trained (0-11), early-stop fired correctly at epoch 11 on 8-epoch val-macro-F1 patience. Best epoch = 3 with val_macro_f1 = 0.5454. Train loss converged from 0.8325 → 0.0654 (gradient flow confirmed); val loss diverged from 0.8107 → 2.2736 (overfit confirmed).
- [x] **F1 per error is reported on the official test split, in the range of the paper's supervised baseline** — Task 13: test_f1_kie = 0.2857 (paper Kinetics row 0.297, Δ -0.011, within 5%), test_f1_kfe = 0.8000 (paper 0.818, Δ -0.018, within 3%), test_macro_f1 = 0.5429 (paper macro ~0.557, Δ -0.014, within 3%). Decision thresholds tuned on val via `sklearn.precision_recall_curve` per RESEARCH §5: `best_thresholds = {"kie": 0.363, "kfe": 0.571}` written into `best.pt`.
- [x] **Training curves, confusion matrices and PR curves are produced** — Task 14: seven figures shipped — `training_curves.png` (118 KB; full 12-epoch trajectory with best-epoch marker and early-stop annotation), `confusion_kie.png` + `confusion_kfe.png` (33 KB each; 2x2 with values + F1/PR-AUC/threshold annotations), `pr_kie.png` + `pr_kfe.png` (~58 KB each; with val-tuned threshold point + random baseline), `sample_predictions_kie.png` + `sample_predictions_kfe.png` (~1.1 MB each; 3×4 grid of top-K TP/FP/FN clips × 8 frames per CONTEXT D14). Plus `results.pkl` (20.6 KB) containing all numeric results + raw scores/labels/clip_ids for downstream phase use.

**Requirements coverage:**
- **SQUAT-03** (R(2+1)D-18 supervised baseline trained; F1 per error on the official Squat test split): closed by Tasks 5-16.

## Phase 3 deliverable artifacts on disk

**Code modules (`backend/training/aqa/`):**
- `eval/__init__.py` — package marker
- `eval/metrics.py` — pure functions: `f1_per_error`, `pr_auc_per_error`, `threshold_sweep`, `confusion_matrix_per_error`. sklearn-backed, no torch, no I/O. Phase 4+ reuses unchanged.
- `eval/test_metrics.py` — 9 unit tests (SQUAT-03-b/c/d + 2x2 shape regression). Runs in ~1.5 s under pytest.
- `harness/supervised_train.py` — `SupervisedConfig` dataclass + `build_model` (R(2+1)D-18 + `Linear(512, 2)` head with `in_features == 512` assertion) + `seed_worker` (PyTorch official) + `_set_global_seed` (reuses Phase 2 `tiny_train._set_global_seed`) + `_build_dataloaders` (direct construction, bypasses Phase 2's `num_workers=0` clamp, `persistent_workers=config.num_workers > 0`) + `_val_pass` (gathers scores+labels on CPU, no per-batch F1) + `run_supervised_epoch` (full trainer with D6/D8/D12/D13 contracts + best.pt write contract + 8-epoch early-stop).
- `harness/test_supervised_train.py` — 2 unit tests (SQUAT-03-a head shape via `@pytest.mark.slow` + SQUAT-03-e checkpoint schema regression). Module-level `pytest.importorskip("torch")` so the file skips cleanly on torch-less environments.

**Notebook:**
- `notebooks/03_squat_supervised_baseline.py` — jupytext percent-format source (git-friendly diff)
- `notebooks/03_squat_supervised_baseline.ipynb` — Colab-runnable pair (delivered per `[[feedback_deliver_colab_as_ipynb]]`)

**Figures (`.planning/phases/03-squat-supervised-baseline/figures/`):**
- `training_curves.png` — train + val loss (left axis) + val F1 KIE/KFE/macro at threshold 0.5 (right axis) across all 12 epochs; best-epoch and early-stop annotations
- `confusion_kie.png` — 2×2 confusion matrix on test at val-tuned threshold 0.363
- `confusion_kfe.png` — 2×2 confusion matrix on test at val-tuned threshold 0.571
- `pr_kie.png` — precision-recall curve on test with val-tuned threshold point + random baseline (pos_rate 0.148)
- `pr_kfe.png` — precision-recall curve on test with val-tuned threshold point + random baseline (pos_rate 0.693)
- `sample_predictions_kie.png` — 3 rows (top-4 TP / FP / FN by score) × 4 clips × 8 frames per clip; val-tuned threshold annotation
- `sample_predictions_kfe.png` — same structure for KFE
- `results.pkl` — pickled dict: val/test scores+labels+clip_ids, per-error F1 + PR-AUC + confusion matrices, best_thresholds, full 12-epoch metrics_history, run provenance (run_name, best_epoch, final_epoch, config_hash, config_repr, code_version)

**Drive (`My Drive/FitNova/checkpoints/phase03/r2plus1d18_squat_supervised_v1/`):**
- `best.pt` (358 MB) — epoch 3 weights + best_thresholds + best_f1_val
- `epoch_NNN.pt` × up to 3 (per `prune_checkpoints(keep_last=3, keep_best=True)`)
- `latest.txt` → `epoch_011.pt`
- `results.pkl` — backup of figures/results.pkl
- `figures/*.png` × 7 — backup of figures/

**Commits (atomic per task, plus fixes and reverts):**
- Planning: `40911d1` (CONTEXT), `578ca07` (RESEARCH), `56ee757` (VALIDATION), `dec3a2e` (PATTERNS), `8721b51` (PLAN), `cac5efb` (STATE update)
- Tasks 1-7: `fb1aec1`, `1f7701b`, `e9d9cf5`, `17edaa2`, `156ca59`, `96678e9`
- Plan fix for arithmetic: `fecd795`
- Notebook .ipynb + Cell A: `49c5ca6`, `55865e1`
- Reverts of over-batched Tasks 8/9/10: `481e5fa`, `003da65`, `e00b238`
- Re-shipped Tasks 8-10: `772c065`, `f79c5d2`, `809e4ce`
- Trainer fixes: `5cec2f5` (persistent_workers), `a0841b4` (map_location='cpu')
- Steps 4-7: `27f302f`, `d4711b6`, `a39ad9b`, `fe44f61`, `c05dd14`, `6245514`, `cfea477`
- Step 6 metrics_history fix: `e50fff9`
- Closeout: this commit.

## Next

**Phase 4 — Squat Motion-Disentangling SSL.**

Goal: domain-knowledge self-supervised pretraining of R(2+1)D-18 on the 4,970-clip unlabeled Squat set, then fine-tune on labeled, compare F1 against Phase 3 baseline. Paper-claimed lift over Kinetics frozen-backbone row: KIE +0.122 (0.297 → 0.419), KFE +0.016 (0.818 → 0.834), macro +0.069 (~0.557 → ~0.626).

**Phase 3 → Phase 4 handoff:**
- **Reference fork point:** `My Drive/FitNova/checkpoints/phase03/r2plus1d18_squat_supervised_v1/best.pt` is the supervised baseline. Phase 4 produces a DIFFERENT checkpoint (MD-SSL-pretrained → fine-tuned) and compares F1 against Phase 3's KIE 0.286 / KFE 0.800 / macro 0.543 on the same official test split.
- **Reuse:** `eval/metrics.py` is unchanged for Phase 4. `harness/supervised_train.py`'s `_build_dataloaders` + `_val_pass` + `run_supervised_epoch` patterns transfer; SSL pretraining needs a NEW trainer (`harness/md_pretrain.py` likely). `notebooks/03_squat_supervised_baseline.py`'s Cell A + Step 0 bootstrap pattern transfers.
- **Carry-forward landmines (must NOT re-surface in Phase 4):**
  - `persistent_workers=True` on every DataLoader with `num_workers > 0`
  - `map_location='cpu'` for `load_latest_checkpoint`
  - `metrics_history` loaded from `latest.txt`-pointed checkpoint, not `best.pt`
  - Notebook delivered as paired `.py` + `.ipynb` with Cell A bootstrap baked in
  - One runnable unit at a time; no batch-write of cells before paste-back

**Hyperparameter considerations for Phase 4 PLAN:**
- Phase 3 overfit hard with `weight_decay=0`. Consider `weight_decay=1e-4` default for Phase 4 fine-tune.
- Paper §5 specified **20 epochs** for downstream. Phase 3 ran 50 with early-stop firing at 11. Consider 20-epoch budget for Phase 4 fine-tune.
- The SSL pretraining phase has its own optimizer/scheduler/budget — paper specifies them separately. RESEARCH for Phase 4 must resolve.

**Operational note for Phase 4 onward:** new Colab session per `[[feedback_heavy_training_new_notebook]]`. SSL pretraining on 4,970 unlabeled clips will be ~4× longer than Phase 3's labeled run. Plan for 12-24 h of compute.

Resume entry points for a fresh chat:
- `.planning/STATE.md` — current GSD state
- `.planning/ROADMAP.md` — Phase 4 success criteria
- `.planning/phases/03-squat-supervised-baseline/03-01-SUMMARY.md` (this file)
- `.planning/phases/03-squat-supervised-baseline/03-01-PLAN.md` — the executed plan
- `.planning/phases/03-squat-supervised-baseline/figures/results.pkl` — frozen Phase 3 baseline numbers for Phase 4 to beat
- `My Drive/FitNova/checkpoints/phase03/r2plus1d18_squat_supervised_v1/best.pt` — trained baseline weights (358 MB)
- `~/.claude/plans/what-pushed-me-back-iridescent-pnueli.md` — master plan, Phase 4 section
- Auto-memory: `[[feedback_interactive_execution]]`, `[[feedback_notebook_disconnect_safe]]`, `[[feedback_heavy_training_new_notebook]]`, `[[feedback_one_cell_at_a_time_strict]]`, `[[feedback_deliver_colab_as_ipynb]]`, `[[feedback_run_gsd_autonomously]]`, `[[reference_pytorch_persistent_workers]]`, `[[reference_colab_module_reload_after_git_pull]]`, `[[reference_best_pt_metrics_history_is_stale]]`, `[[project_supervisor_visualizations]]`, `[[project_form_correction_status]]`
