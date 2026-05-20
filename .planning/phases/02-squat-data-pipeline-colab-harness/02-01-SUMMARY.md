---
phase: 02-squat-data-pipeline-colab-harness
plan: 01
status: complete
completed: 2026-05-20
---

# Phase 2 · Plan 01 — Summary

## What was planned

Sixteen tasks under `02-01-PLAN.md` covering: clone the official `Code_Release/` and document the gap (Tasks 1–2); scaffold the AQA module tree + jupytext notebook (Task 3); implement the PyTorch loader stack — splits, transforms, dataset, build_loaders (Tasks 4–6); build the resumable Colab harness in three slices — Drive mount + zip-stage, RNG capture/restore + cudnn determinism, atomic checkpoint primitives (Tasks 7, 9, 10); ship the supervisor-priority decoded-batch visualization (Task 8); wire the toy model + multi-epoch training loop (Task 11); produce the F1 acceptance evidence — fresh 2-epoch baseline, simulated-restart resume, bitwise-equivalence assertion + overlay plot (Tasks 12–14); human-verify both figures (Task 15); close out (Task 16). Covers requirements SQUAT-01, SQUAT-02.

## What was done

Executed interactively in Colab against the cloned `Ibrahim-Shahin1/fitnova` repo (branch `fresh-start`). All 14 implementation tasks shipped one runnable unit at a time with paste-back verification per the working agreement. Every code task got its own atomic commit (`feat(02)`, `fix(02)`, `chore(02)`) referencing the task number and PLAN.md decision/finding IDs (D-numbers, F-numbers) inline. Planning artifacts (`02-CONTEXT.md`, `02-RESEARCH.md`, `02-01-PLAN.md`, `02-01-PLAN-REVIEW.md`) produced via the GSD subagent loop (researcher → planner → plan-checker × 2; PASS on iteration 2) committed at `789a2ae`. The decoded-batch supervisor figure (`figures/decoded_batch.png`) rendered on a real train batch (records 0 and 1 in train_keys.json order); all 5 visual criteria passed inline review. The byte-equality proof (`figures/tiny_train_loss.png`) shows the baseline epoch-1 and resumed-from-epoch-0 epoch-1 loss trajectories as perfectly coincident curves — 8/8 batches identical to float64 precision.

## Key findings (headlines)

- **`Code_Release/motion_disentanglement/` is empty** (2-byte README only). The only complete code in the upstream repo is the CVCSPC image-side SSL trainer — and even it has broken imports between its own files (`opts_exercise_qa` missing, `apply_augmentations` referenced but undefined, `dataloader_eval` missing). Phase 2 reconstructs the data pipeline from paper text rather than adapting code; documented in `CODE-RELEASE-NOTES.md`.
- **Input resolution corrected from 224 → 112.** Paper's `320→224` is the Waseda 2DCNN baseline path; torchvision's `R2Plus1D_18_Weights.KINETICS400_V1` is native `112×112`. `crop_size` is parameterized so Phase 3 can ablate.
- **Multi-label imbalance via `BCEWithLogitsLoss(pos_weight=...)`** (not `WeightedRandomSampler`). Dataset computes `pos_weight = [6.10, 0.4527]` from train counts (160 KIE+ / 782 KFE+ / 1136 total) and exposes it; Phase 3 consumes.
- **Counts reconcile exactly to Phase 1** — train 1136 / val 243 / test 244; KIE+ totals 232; KFE+ totals 1109 — all matching the published Fitness-AQA figures and Phase 1's measured numbers.
- **Drive FUSE rename is non-atomic.** `atomic_save_checkpoint` writes to a `.tmp_{name}.{pid}` file, round-trips through `torch.load`-verify, `os.replace`s into final position, and writes `latest.txt` **last**. A crash at any step leaves the prior good `latest.txt` pointer untouched.
- **Bitwise resume proven, not claimed.** Baseline epoch_1 (8 floats) is `==`-identical to resumed-from-epoch-0 epoch_1 (8 floats). val_loss_mean identical to float64 precision (`0.6683911681175232`). `config_hash` stable across the resume cycle (`51d5de1e7a696843`). RNG capture/restore is functionally correct.
- **`stage_squat_videos` hardened to disconnect-by-default** through iterative user-driven feedback: tqdm progress bars on copy + extract, byte-level resume of the copy (seek + ab-append), per-member resume of the extract (skip files matching size), flat extraction (strips zip's `videos/` prefix), one-time migration of pre-existing nested mp4s on legacy-state sessions.

## Deviations from PLAN

- **F11 acceptance test indices `[0, 200, 403]` corrected to clustered `[100, 102, 104]`** during Task 5/7 — the original index pattern spans the full clip → windowing doesn't actually reduce memory because `start_pts=0, end_pts≈full` decodes the whole thing. Clustered indices are the meaningful test; the corrected version exercised real windowing and showed the intermediate decode buffer ratio < 0.25 vs full-clip decode. Documented inline.
- **Dataset's per-instance `self._generator` removed.** Task 6's spec called for a per-instance `torch.Generator(seed)` to drive jitter + spatial random-crop. This would have broken Task 14's bitwise-resume assertion because a resumed run creates a NEW dataset instance whose generator starts at seed-state, while the baseline run's dataset generator has already advanced through epoch-0 sampling by the time epoch-1 starts. Fix: dataset now passes `generator=None`, both `uniform_sample_indices` and `spatial_train` fall through to torch's default RNG — which IS what `capture_rng_state` / `restore_rng_state` snapshot and replay. The `seed=42` kwarg stays in the dataset signature for Phase 3+ API stability but is a no-op in Phase 2.
- **IPython 7.34.0 + `imp` monkey-patch** instead of upgrading IPython for autoreload. `google.colab 1.0.0` pins `ipython==7.34.0`; an upgrade to IPython 9.x would have broken Drive mounting. The `imp` shim (a `types.ModuleType` exposing `importlib.reload` under `imp.reload`) lets the autoreload extension load against IPython 7.34.0 on Python 3.12 without touching the pinned IPython version.
- **PyAV auto-install moved to Step 0** after Colab's torchvision 0.25 surprise — `torchvision.io.read_video` still exists (deprecated, not removed) but caches its `import av` failure at module-load time. The fix: install `av` BEFORE the first `import torch / torchvision` so the cached reference is a real module, not an Exception. Step 0 now does an idempotent `pip install -q av` guarded on a try/except `import av`.
- **Torchvision deprecation UserWarning filtered** at `colab.py` module load. The deprecation is real and documented in D6 (swap target is TorchCodec); until Colab's torchvision crosses 0.24 (removal), the warning was firing ~40 times per epoch on the tiny run. Filter is scoped by message regex so other UserWarnings still surface.
- **Step 0's video-existence probe replaced with a backend-only sanity check.** Original probe called `read_video_timestamps` on a hardcoded `/content/squat_videos/46777_1.mp4` — but Step 0 runs BEFORE Step 4 (stage), so the file wouldn't exist on fresh sessions. New check inspects `torchvision.io.video.av` directly (must be a real module, not an Exception), proving the PyAV load order is correct without needing any video file.

## Acceptance criteria (ROADMAP success criteria + PLAN.md verification block)

- [x] **Squat clips load as `(video tensor, KIE/KFE label)` batches using the official splits** — Task 6 verification: `next(iter(loaders['train']))` returns `(video[2, 3, 32, 112, 112] float32, labels[2, 2] float32)`; counts reconcile 1136/243/244; `pos_weight` identical across train/val/test loaders.
- [x] **Decoded and augmented sample frames visualized and confirmed correct** — Task 8 + Task 15: `figures/decoded_batch.png` (1013 KB, 1920×600) passes all 5 visual criteria (decode temporal order, sampling span, aspect preservation, label correctness, no horizontal flip).
- [x] **Colab harness checkpoints to Drive and resumes from latest checkpoint after restart** — Tasks 12/13/14: baseline 2-epoch run produced `epoch_000.pt` + `epoch_001.pt` + `latest.txt → epoch_001.pt` on Drive. Simulated restart (delete epoch_001 + rewind latest.txt + `sys.modules` purge + re-import + `run_tiny_epoch(resume=True, max_epochs=2)`) loaded epoch_000, restored all 4 RNGs, advanced `start_epoch=1`, trained one epoch in the F1 loop, wrote new `epoch_001.pt`. **Byte-equality of the 8 epoch-1 batch losses between baseline and resumed is the proof: `figures/tiny_train_loss.png` shows two perfectly coincident curves.**

**Requirements coverage:**
- **SQUAT-01** (PyTorch data pipeline — decode, 32-frame sampling, transforms, official-split loaders): closed by Tasks 4–6 + 8 (the decoded-batch viz exercising the full stack on a real batch).
- **SQUAT-02** (Resumable Colab training harness — checkpoints each epoch, auto-resumes after disconnect): closed by Tasks 7 + 9 + 10 + 11 + 12 + 13 + 14 (the full atomic-write + RNG capture/restore + tiny-train loop + byte-equality proof).

## Phase 2 deliverable artifacts on disk

- `backend/training/aqa/datasets/splits.py` — `ClipRecord` + `index(split, ...) -> list[ClipRecord]` + `expected_counts()`
- `backend/training/aqa/datasets/transforms.py` — `KINETICS_MEAN/STD`, `uniform_sample_indices`, `decode_clip` (F11 windowed), `spatial_train`, `spatial_val`
- `backend/training/aqa/datasets/squat.py` — `SquatKIEKFEDataset` + `build_loaders` factory
- `backend/training/aqa/harness/_envinit.py` — F8 `CUBLAS_WORKSPACE_CONFIG` shim
- `backend/training/aqa/harness/colab.py` — Drive mount, idempotent flat-extract zip-stage (with byte-resume + per-member resume + migration), RNG capture/restore, cudnn determinism config, `hash_config`, atomic checkpoint primitives, prune
- `backend/training/aqa/harness/tiny_train.py` — `TinyModel` + `run_tiny_epoch` (multi-epoch, resume-aware, F1 loop)
- `backend/training/aqa/notebooks/02_squat_pipeline_harness.py` — jupytext percent-format Colab notebook driving the full Phase 2 sequence (Step 0 bootstrap through Step 8c proof)
- `.planning/phases/02-squat-data-pipeline-colab-harness/CODE-RELEASE-NOTES.md` — upstream gap analysis (the `Code_Release/` reality check)
- `.planning/phases/02-squat-data-pipeline-colab-harness/figures/decoded_batch.png` — supervisor visualization
- `.planning/phases/02-squat-data-pipeline-colab-harness/figures/tiny_train_loss.png` — bitwise-resume proof
- `.planning/phases/02-squat-data-pipeline-colab-harness/figures/tiny_baseline_2epoch_losses.json` — baseline trajectory (Task 12 output)
- `.planning/phases/02-squat-data-pipeline-colab-harness/figures/tiny_resumed_epoch1_losses.json` — resumed trajectory (Task 13 output)

## Next

**Phase 3 — Squat Supervised Baseline (R(2+1)D-18, Kinetics-init).** Trains the real model on the labeled Squat set, evaluates F1 per error on the official test split, produces training curves + confusion matrices + PR curves.

**Operational note for Phase 3 onward:** real training will take hours per run. Per the user-locked working-agreement rule (`feedback_heavy_training_new_notebook.md`): start a fresh Colab notebook with the **L4 (or higher) GPU** before kicking off Phase 3 — do NOT continue in the Phase-2 tiny-run notebook. New session bootstrap is Cell A (clone-or-pull) → Cell B (Step 0 — installs PyAV, mounts Drive) → re-stage (cache hit if `/content/` persists, else re-download). Code already supports this — the harness is disconnect-by-default by construction (the very point of Phase 2).

Resume entry points for a fresh chat:
- `.planning/STATE.md` — current GSD state
- `.planning/ROADMAP.md` — Phase 3 success criteria
- `.planning/phases/02-squat-data-pipeline-colab-harness/02-01-SUMMARY.md` (this file)
- `.planning/phases/02-squat-data-pipeline-colab-harness/02-01-PLAN.md` — the executed plan
- `~/.claude/plans/what-pushed-me-back-iridescent-pnueli.md` — master implementation plan, Phase 3 section
- Auto-memory files (especially `feedback_heavy_training_new_notebook.md`)
