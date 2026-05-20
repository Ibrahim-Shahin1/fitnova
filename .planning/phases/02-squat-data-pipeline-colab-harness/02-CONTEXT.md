# Phase 02: Squat Data Pipeline & Colab Harness — Context

**Gathered:** 2026-05-20
**Status:** Ready for planning
**Source:** Orchestrator-synthesized from master plan + Phase 1 dataset report + REQUIREMENTS + working agreement. No discuss-phase pass — the master plan and Phase 1 closeout already discharge that function (mirrors the Phase 1 deviation precedent, logged in `01-01-SUMMARY.md`). User explicitly elected the full subagent loop for the planner/checker side.

<domain>
## Phase Boundary

**This phase delivers:**

1. A PyTorch data pipeline for **Squat KIE/KFE** that, given a Squat sample ID, yields `(video_tensor[C=3, T=32, H=224, W=224], labels[2])` where `labels[0]=KIE`, `labels[1]=KFE`, both binary, sourced from the **official** `train/val/test_keys.json` splits and the official `error_knees_inward.json` / `error_knees_forward.json` label files.
2. A **Colab harness** that:
   - Mounts Google Drive and copies the Squat labeled `videos.zip` from `My Drive/Fitness-AQA_dataset_release/Squat/Labeled_Dataset/` to `/content/` local once per session, extracts it, and trains from local disk (Drive FUSE is too slow per-epoch — Phase 1 measurement).
   - Checkpoints **after every epoch** to Drive: model weights, optimizer state, epoch number, RNG state (python / numpy / torch / CUDA), and metrics history.
   - On a fresh notebook restart (the default case in Colab), auto-resumes from the latest Drive checkpoint without manual intervention.
3. A **decoded-and-augmented batch visualization** — a frame grid sampled from the train loader that lets a human eye verify decode order, sampling spread, aspect handling, and augmentation behavior before any model is trained.
4. A **1-epoch tiny end-to-end validation run** through the harness (toy model, small subset, single GPU) proving DataLoaders → forward pass → loss → optimizer step → checkpoint → resume actually works as one unit.

**This phase does NOT deliver:**

- Any model architecture or training of R(2+1)D-18 (Phase 3).
- Class-weighted loss design or threshold tuning (Phase 3).
- Motion-Disentangling SSL pretraining or the unlabeled-set pipeline (Phase 4).
- Image-modality loaders for Shallow-Squat (Phase 7).
- Loaders for OHP or BarbellRow (Phases 6, 7).
- Test-set evaluation, F1, PR curves, or confusion matrices (Phase 3).
- Backend integration (Phase 5).

**Requirements satisfied:** SQUAT-01 (Squat KIE/KFE PyTorch pipeline), SQUAT-02 (resumable Colab harness).

</domain>

<decisions>
## Implementation Decisions (Locked)

### Framework & target model
- **PyTorch**, not TensorFlow. R(2+1)D-18 + the official Fitness-AQA `Code_Release/` are torch-native; matches the milestone constraint and the master plan.
- Phase 2 produces tensors **shaped for R(2+1)D-18** even though the model itself is Phase 3. Output shape is `(B, 3, 32, 224, 224)`, channels-first, RGB, float32, normalized with **Kinetics-400 mean/std** (`mean = [0.43216, 0.394666, 0.37645]`, `std = [0.22803, 0.22145, 0.216989]`) — these are the standard `torchvision.models.video.r2plus1d_18` weights' expected stats and what the paper builds on. Confirm against `Code_Release/` during the first task and update if the paper used different stats.

### Squat label semantics
- A clip is **KIE-positive** iff `error_knees_inward.json[clip_id]` is a **non-empty** list of `[start, end]` intervals; otherwise KIE-negative. Same for KFE via `error_knees_forward.json`.
- **Joint multi-label head** (KIE and KFE share a backbone, two sigmoid outputs). Justified in Phase 1 by 192/232 KIE+ co-occurring with KFE+. The DataLoader emits both labels per clip; the Phase 3 loss handles them jointly.
- The 1,623 official Squat clips drive everything. The 116 extra unlabeled clips in `videos.zip` (Phase 1 finding) are **dropped** by virtue of using the split files as the source of truth for IDs.
- The 19 `traj_nan.json` clips are **kept** for the supervised pipeline (they have valid KIE/KFE labels) — `traj_nan` only matters for Phase 4 MD-SSL.

### Splits
- Use `Squat/Labeled_Dataset/Splits/{train,val,test}_keys.json` **verbatim**. No reshuffle, no merge, no internal CV.
- The `val` split is used during the tiny run (loader smoke test); training-vs-val choices for the real run are Phase 3.

### Sampling — 32 frames per clip
- **Uniform sampling** across the **full clip span**: for a clip of `F` decoded frames, sample frames at indices `round(linspace(0, F-1, 32))`. This absorbs Phase 1's measured variability (49–404 frames, median ~111, long tail to 13.5s multi-rep). The paper's 32-frame target is the published number; the official `Code_Release/` will be inspected to confirm whether uniform vs. random-segment is the paper's exact sampler — and the planner will pin which we adopt for train vs. val/test.
- For train: optionally **temporal jitter** on the uniform grid (each picked index shifted by ±ε frames, clamped) — defer to the paper's `data_augmentations/` for the exact recipe.
- For val/test: deterministic uniform sampling, no jitter, no seed dependency.

### Spatial preprocessing
- **Aspect-aware resize → crop.** Phase 1 measured width pinned at 480 with heights varying across 270 / 324 / 480 / 584 / 592 / 600. The paper states a `320² resize → 224² crop` recipe; the literal reading is the **short side** is resized to 320 preserving aspect, then a 224×224 crop is taken. Confirm against `Code_Release/data_augmentations/`.
- **Train:** resize short-side to 320 → 224×224 **random crop** → horizontal flip with **caveat below** → Kinetics normalization.
- **Val/test:** resize short-side to 320 → 224×224 **center crop** → Kinetics normalization.
- **Horizontal flip caveat:** KIE is a *laterality* signal (knees moving toward the **inside** of the body) — but the label is "knees-inward", which is symmetric across the body's midline, so a left/right mirror does not change the label semantically. Confirm with `Code_Release/data_augmentations/` before turning flip on. **Default OFF until confirmed.**

### Decoding
- Start with `torchvision.io.read_video` for portability (no extra system deps in Colab). If it is the bottleneck on a measured profile, swap to `decord` or `pyav`. Decision is reversible — the planner specifies the decoder behind a thin function, the trainer is decoder-agnostic.
- **No frame-level caching to disk** in this phase. Local extraction of the zip is sufficient — disk → tensor on the fly is fast enough for R(2+1)D-18 on a single GPU per the paper's reported throughput.

### Data staging on Colab
- Per session, copy `My Drive/Fitness-AQA_dataset_release/Squat/Labeled_Dataset/videos.zip` → `/content/squat_videos.zip` → unzip to `/content/squat_videos/`. Drive layout stays exactly as-is.
- The copy step is **idempotent and cached** within a session — if the extracted tree exists and has the expected file count (1,739 mp4s, per Phase 1), skip the copy. This protects against re-running cells during interactive work.
- The split JSONs and label JSONs are small (~MB) and read directly from Drive without copying.

### Checkpoint + resume contract
- **Checkpoint payload** (single `.pt` file via `torch.save`):
  - `epoch` (int, the just-completed epoch)
  - `model_state_dict`
  - `optimizer_state_dict`
  - `scheduler_state_dict` (if scheduler used in Phase 3+)
  - `rng_state` = `{python, numpy, torch_cpu, torch_cuda}`
  - `metrics_history` (list of per-epoch dicts: train_loss, val_loss, per-error metrics)
  - `config_hash` (a hash of the training config — guards against silently resuming into a different recipe)
- **Layout on Drive:** `My Drive/FitNova/checkpoints/phase02/{run_name}/epoch_{NNN}.pt` plus a `latest.pt` symlink-equivalent (Drive doesn't symlink — keep a `latest.txt` pointing at the filename).
- **Resume on restart:** harness boot reads `latest.txt`, loads, restores RNGs, sets the start epoch, and the training loop picks up from `start_epoch + 1`. RNG restoration is what makes augmentation deterministic post-resume.
- **Failure mode:** if `config_hash` doesn't match the current run config, the harness **stops with a clear error** ("checkpoint config drift — start a new run name or reconcile config"). It does NOT silently re-init.
- **Retention:** keep the last 3 epoch checkpoints + the best-by-val metric. Older ones pruned on Drive after a successful new save. Phase 3 will tighten "best" definition; for Phase 2 tiny run, "last" is sufficient.

### Code organization
- New code under `backend/training/aqa/` — the home established in Phase 1 (`backend/training/aqa/notebooks/01_dataset_eda.py`).
- Library modules importable from the notebook (so notebook cells stay short and the logic is testable):
  - `backend/training/aqa/datasets/squat.py` — the `SquatKIEKFEDataset` class
  - `backend/training/aqa/datasets/transforms.py` — decode + sample + spatial preprocessing
  - `backend/training/aqa/datasets/splits.py` — official-split loaders, label index
  - `backend/training/aqa/harness/colab.py` — Drive mount, zip staging, checkpoint/resume
  - `backend/training/aqa/harness/tiny_train.py` — the 1-epoch smoke loop (toy model)
- Notebook: `backend/training/aqa/notebooks/02_squat_pipeline_harness.py` (jupytext percent format — Phase 1 convention).
- Code_Release study artifacts (notes, references) go in `.planning/phases/02-squat-data-pipeline-colab-harness/CODE-RELEASE-NOTES.md`.

### Code_Release access
- **First runnable unit** of Phase 2 is `git clone https://github.com/ParitoshParmar/Fitness-AQA Fitness-AQA-Code` at the repo root, with `Fitness-AQA-Code/` added to `.gitignore`. Read `Code_Release/motion_disentanglement/` and `Code_Release/data_augmentations/` for the supervised baseline's transforms, sampler, and (if present) any squat-specific dataloader. Findings update the locked decisions above where they conflict with the paper's literal description.
- If the official code has gaps (the README is "Work in Progress"), fall back to the paper. Diverge from neither without recording the divergence in `CODE-RELEASE-NOTES.md`.

### Working agreement
- **Interactive, one runnable unit at a time.** Each task in PLAN.md should hand the user exactly one runnable thing (a notebook cell, a script invocation, or a `pip install` line) and stop until output is pasted back. The orchestrator adapts to the actual output, not an assumed one.
- **No blind runs.** Any consequential step (especially anything writing to Drive, downloading models, or installing packages) is shown first, explained, and approved.
- **Colab is the default execution environment.** Local execution is allowed for pure-Python unit checks (no GPU, no MediaPipe), but the notebook must remain the single end-to-end driver.
- **Resumability is a build constraint, not a feature.** Every cell that writes durable state (a checkpoint, a cached zip extraction) must be re-runnable without corrupting earlier state.

</decisions>

<canonical_refs>
## Canonical References

Downstream agents MUST read these before planning or implementing.

### Phase scope and prior phase
- `.planning/ROADMAP.md` — Phase 2 goal + 3 success criteria
- `.planning/REQUIREMENTS.md` — SQUAT-01, SQUAT-02
- `.planning/phases/01-dataset-consolidation-eda/01-DATASET-REPORT.md` — verified dataset facts: clip lengths 49–404 (median ~111), 30fps uniform, width-480/variable-height, KIE×KFE co-occurrence, 1,623 labeled clips, traj_nan list, Drive-FUSE-too-slow finding
- `.planning/phases/01-dataset-consolidation-eda/01-01-SUMMARY.md` — Phase 1 closeout + deviation precedent

### Milestone spec
- `~/.claude/plans/what-pushed-me-back-iridescent-pnueli.md` — master implementation plan; Phase 2 section is authoritative
- `.planning/PROJECT.md` — milestone constraints (PyTorch form model, official splits, F1 per error, Colab compute)

### External
- `https://github.com/ParitoshParmar/Fitness-AQA` — official Code_Release (pointer at `Fitness-AQA/github-link.md`). Phase 2 first task clones this.
- Parmar et al., *Domain Knowledge-Informed Self-Supervised Representations for Workout Form Assessment*, ECCV 2022 (arXiv:2202.14019) — paper.

### Convention
- `CLAUDE.md` — backend Python conventions (snake_case modules, `from __future__ import annotations`, module docstring at top, type hints throughout, no enforced formatter but consistent style)
- `backend/training/aqa/notebooks/01_dataset_eda.py` — jupytext percent-format template for the Phase 2 notebook

### Dataset locations
- Drive: `My Drive/Fitness-AQA_dataset_release/Squat/Labeled_Dataset/` — Drive shortcut to Shared-with-me (verified accessible in Colab)
- Local dataset snapshot: `Fitness-AQA/Fitness-AQA_dataset_release-20260518T073812Z-3-001/Fitness-AQA_dataset_release/Squat/Labeled_Dataset/` — for offline inspection (NOT for training reads)

</canonical_refs>

<specifics>
## Specific Ideas

- **Run-name convention** for checkpoints: `squat_{stage}_{date}_{shortsha}` where `stage ∈ {tiny, baseline, ssl_pretrain, ssl_finetune}`. Phase 2 only produces `tiny` runs. Phase 3 picks up `baseline`.
- **Tiny-run config:** 16 train clips, 4 val clips, batch size 2, 1 epoch, toy model = a stride-1 `Conv3d(3, 4, 3) → AdaptiveAvgPool3d → Linear(4, 2)`. Goal is exercising the pipe end-to-end, not learning anything.
- **Determinism for the tiny run:** seed 42 everywhere; cudnn deterministic + no benchmark. The 1-epoch run, when re-run from scratch on the same machine, should produce bitwise-identical loss values. This is a regression guard for the harness, not a long-term policy.
- **Decoded-batch visualization:** N×T grid where N=2 train clips, T=8 evenly-spaced frames per clip (out of the 32). Save as a PNG to the phase dir and embed in the notebook. Side-by-side with each clip's KIE/KFE label printed underneath. This is the deliverable for Phase 2 success criterion 2.
- **Mark known footguns inline in the data pipeline:**
  - `read_video` returns `(video, audio, info)` — discard the latter two.
  - `torchvision.io` requires `pyav` installed.
  - Colab can hit a memory limit if you `read_video` a long clip and then crop, instead of indexing the wanted frames first.
- **Logging:** `logger = logging.getLogger("aqa.phase02")` per CLAUDE.md convention; JSONL metrics file appended per epoch alongside the checkpoint.

</specifics>

<deferred>
## Deferred Ideas

Carried forward to later phases:

- **Loss function design + class weighting** — Phase 3. The DataLoader exposes per-clip indices and per-class weight tensors as a hook, but does not compute the loss.
- **Augmentation richness beyond paper minimum** — Phase 3 if needed for the supervised baseline; Phase 4 may add SSL-specific augmentations from `motion_disentanglement/`.
- **Validation-split evaluation harness** — Phase 3. The Phase 2 tiny run just checks the loader works; it does not compute F1 or PR-AUC.
- **Decord / pyav decoder swap** — only if `torchvision.io.read_video` profiling shows it's the bottleneck (deferred to Phase 3).
- **MD-SSL unlabeled loader** — Phase 4, separate dataset class because no label JSON and barbell-trajectory pairing is required.
- **OHP / BarbellRow / Shallow-Squat loaders** — Phases 6, 7. Same code patterns, different sampling/transform sets.

</deferred>

<scope_fence>
## Scope Fence

**Inside the fence (Phase 2 will do):**
- DataLoader producing R(2+1)D-18-shaped tensors from official Squat splits + labels
- Colab resumable harness (Drive mount, zip staging, per-epoch checkpoint, auto-resume)
- Decoded-batch visualization
- 1-epoch tiny end-to-end smoke run
- Cloning + reading official Code_Release into project notes
- Modular library under `backend/training/aqa/datasets/` and `backend/training/aqa/harness/`
- A jupytext notebook `02_squat_pipeline_harness.py` driving the above

**Outside the fence (Phase 2 will NOT do):**
- Choose / train any real model
- Compute any F1 / accuracy / loss metric beyond "tiny run finishes"
- Touch OHP, BarbellRow, or Shallow-Squat data
- Modify the FastAPI backend or any inference path
- Modify or remove the existing legacy form-analysis code (Phase 5 task)
- Add MediaPipe — the new approach is raw-pixel CNN; pose is intentionally out

</scope_fence>

<risk_summary>
## Risk Summary

| Risk | Likelihood | Mitigation |
|---|---|---|
| `torchvision.io.read_video` decode bugs on the dataset's MP4 encoding | Low | Phase 1 didn't surface decode errors; tiny run exercises every split's first/last clip. Fallback decoder ready. |
| Colab disconnects mid-tiny-run | Certain (default) | Harness checkpoints after every epoch + auto-resume from `latest.txt`. The tiny run is 1 epoch — disconnect just means redo. |
| Drive write quota / latency causing checkpoint slowness | Medium | Checkpoint payload is small (< 100 MB at this stage). Retention keeps only 3 + best. |
| Spatial preprocessing diverges from paper because Code_Release is WIP | Medium | First task clones + reads it; any divergence is recorded in `CODE-RELEASE-NOTES.md` and called out in PLAN.md. |
| Horizontal flip changes KIE semantics in some edge cases | Low | Default OFF until confirmed against Code_Release. |
| RNG-resume drift produces non-deterministic post-resume runs | Medium | Tiny-run test explicitly verifies bitwise-identical loss on resume-from-epoch-0 vs. fresh-from-scratch on a fixed seed. |
| Loader yields wrong tensor shape for R(2+1)D-18 | Low | The toy model in the tiny run is shape-strict — wrong shape errors immediately, not at Phase 3. |

</risk_summary>

---

*Phase: 02-squat-data-pipeline-colab-harness*
*Context synthesized: 2026-05-20 — orchestrator-authored from existing canonical sources, no new discuss-phase pass*
