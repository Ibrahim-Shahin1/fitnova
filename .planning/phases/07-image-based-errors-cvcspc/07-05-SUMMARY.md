# Phase 7 — Image-Based Errors (Shallow-Squat CVCSPC) — SUMMARY

**Status:** COMPLETE (2026-05-30)
**Requirements:** IMG-01 ✓, IMG-02 ✓ (IMG-03 descoped — BarbellRow cancelled)
**Headline:** Shallow-Squat official-test-split F1 **0.8902** (3-seed CVCSPC ensemble) — **beats the
published CVCSPC 0.8694** by +0.021; **+0.0152 SSL lift** over our supervised baseline (0.8750);
val→test gap 0.0217 (no overfitting).

## What shipped

**Plan 01 (Wave 0, local CPU)** — the genuinely-new code + unit tests, gated green before any GPU:
- `datasets/shallow_squat.py` — single-image dataset (JPEG → 224² → ImageNet norm, flat {id:0/1}
  labels, [1] single-head pos_weight, official splits).
- `datasets/cvcspc_ssl.py` — the CVCSPC phase-matched cross-rep triplet dataset (`_traj2phase`
  degenerate-guarded, masking-only aug, traj_nan + `exclude_ids` holdout exclusion).
- `harness/cvcspc_pretrain.py` — self-contained CVCSPC SSL trainer (the 3-term loss, no-BN
  configurable projector, build_cvcspc_model, `_triplet_accuracy` monitor, run loop with phase-gap
  anneal). Does NOT import the R(2+1)D video trainer.
- `harness/image_supervised_train.py` — self-contained ResNet-18 2D-image trainer (`build_resnet18`
  + Linear(512,1), 4-D `_val_pass`, `run_image_epoch` with the `dataset_cls`/`model_builder`/
  `checkpoint_phase` seams). Does NOT import the video trainer.
- `harness/colab.py` additions — `stage_shallow_squat_images` (zipfile.extractall) +
  `extract_frames_for_ssl` (cv2 decode → per-clip frame dirs).
- 17 unit tests (3-term loss closed-form + directionality, phase-matched triplet, traj_nan +
  exclude_ids exclusion, masking, single-head shapes, no-BN projector, seams). Gate: **45 passed,
  4 deselected**, no Phase-3/4/6 regression.

**Plan 02 (Colab)** — ImageNet ResNet-18 supervised baseline (the SSL-lift control): 3-seed ensemble,
val-tuned threshold, official 540-crop test **F1 0.8750** (PR-AUC 0.951), val→test gap 0.010.

**Plan 03 (Colab)** — faithful CVCSPC pose-contrastive SSL pretrain: frame-extracted the 4,970
unlabeled clips (581,503 frames), excluded the 179 val/test holdout clips (leakage-free, 4,791-clip
SSL set), 100-epoch pretrain, triplet-accuracy **0.48 → 0.959** → `backbone.pt`.

**Plan 04 (Colab)** — fine-tune from the backbone via the `model_builder` seam (3 seeds) + ensemble
+ val-tuned threshold + test eval + `results.pkl`: **F1 0.8902** (PR-AUC 0.967), +0.0152 lift, all
3 seeds beat baseline, val→test gap 0.0217.

**Plan 05 (local)** — the deliverable pack: 3 notebooks (`docs/notebooks/09–11_shallow_squat_*`,
paired .py + executed .ipynb), 6 figures (`docs/figures/shallow_squat_*`), `FINDINGS_SHALLOW_SQUAT.md`,
`shallow_squat_test_scores.csv`, the README rows, and this SUMMARY — all from `results.pkl` + the
local archive, no fabrication.

## Production checkpoint declaration

- **CVCSPC backbone:** `My Drive/FitNova/checkpoints/phase07/shallow_squat_cvcspc_v1/backbone.pt`
  (the triplet-accuracy-best epoch, ep95, ResNet-18 fc=Identity).
- **Fine-tune seeds:** `phase07/shallow_squat_cvcspc_finetune_seed{42,1337,7}/best.pt`.
- **Ensemble protocol:** mean-of-sigmoids across the 3 seeds; single decision threshold **0.395**
  tuned on the ensemble val scores; F1 on the official test split.
- **Source of truth:** `.planning/phases/07-image-based-errors-cvcspc/figures/results.pkl` (committed).

## Deviations (documented honestly)

- **3-term loss:** the SSL backbone used the official code's 3-term distance-ratio loss
  (`train_test.py:68`), which adds a pos-neg term not in the paper's 2-term Eq.1. Implemented the
  as-shipped code version; documented in FINDINGS, not reconciled.
- **Adam (not AdamW)** in the SSL trainer — matches the official `train_test.py:157`; identical to
  AdamW at the default weight_decay=0.
- **traj_nan.json absent** from this dataset release — the Plan-01 degenerate-trajectory guard covers
  NaN trajectories instead (0 degenerate found).
- **val/test-clip exclusion added** to the SSL set (`exclude_ids`) — a faithfulness/leakage fix the
  plan missed; verified the labeled crops 97–99% overlap the unlabeled set, so excluding the 179
  holdout clips matches the official protocol and keeps the eval clean.
- **No projection-head ablation needed** — the [ASSUMED] 512→128→128 head converged cleanly
  (triplet-accuracy 0.959); the 512→512 escape hatch was not used.
- **Drive-FUSE fix:** `_atomic_write_text` hardened (read-back + retry) for the rapid ~5s-epoch
  `latest.txt` writes — a carry-fix benefiting all phases.

## Descopes (recorded)

- **IMG-03 (BarbellRow Lumbar/Torso)** — cancelled by the user (compute/time). Milestone
  form-correction scope = Squat + OHP (video) + Shallow-Squat (image).
- **ROADMAP SC3 ("served through the inference API")** — descoped (CONTEXT D1; same stance as OHP):
  the form-correction frontend is cancelled, so Shallow-Squat is not wired into the (UI-less) serving
  path. IMG-01/IMG-02 are met by the trained/evaluated models + the paper comparison + the
  visualization pack. The orphaned form backend (WS/SquatLiveSession) is intentionally kept for
  fallback, not stripped.

## Thesis contribution

Adds the **image modality** + the **2nd method (CVCSPC pose-contrastive SSL)** to the thesis,
alongside the video Motion-Disentangling SSL work (Squat KIE/KFE, OHP Elbows/Knees). The result is a
faithful reproduction that **exceeds the published CVCSPC number** on the identical metric/split, with
a measured, consistent SSL lift over a strong supervised control — defensible end to end.

## Phase 8 handoff

Phase 8 (Ensemble, Evaluation & Visualization Pack) — the Shallow-Squat CVCSPC result feeds the final
identical-metric comparison (ours vs Parmar / GYMetricPose / LMM) across all errors. `results.pkl` +
the 3-notebook pack are ready inputs. The MD+CVCSPC ensemble (where the paper applies it) and the
consolidated cross-exercise comparison are Phase 8's scope.
