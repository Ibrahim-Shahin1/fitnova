# FitNova — Continuation Handoff (Phase 7 / Shallow-Squat CVCSPC) · 2026-05-30

## How to use this handoff
You are continuing the FitNova form-correction rebuild, working **exactly** as every prior session did.
**Before doing anything: read every document in "Read first" — in full, do not skip, do not get lazy,
do not work from this handoff alone.** This handoff orients you; the files are ground truth. **Verify
every claim against the actual code/files before building on it** — prior sessions caught real bugs
(and one wrong premise of mine) only by verifying. The user wants you to verify and push back, not agree.

## TL;DR
FitNova = AI fitness app (Flutter + FastAPI). Two features: a workout-plan generator (works, untouched)
and **exercise form correction (this milestone)** — Fitness-AQA dataset (Parmar et al., ECCV 2022),
faithful reproduction with **identical metrics (F1 per error on the official splits)**, PyTorch.

**Phases 1–6 COMPLETE.** Squat (video, MD-SSL) macro-F1 **0.6304** and OHP (video, MD-SSL) macro-F1
**0.6622** — both **match/edge the published paper** on the official test splits, both with full
visualization packs. **Phase 7 is PLANNED (5 plans, plan-check PASS + revision) and ready to execute.**
Branch `fresh-start`, **HEAD `38b2d91`**, pushed, in sync. 6/8 phases done.

**Phase 7 = Shallow-Squat depth error (the IMAGE modality, a NEW method = CVCSPC).** BarbellRow is
**CANCELLED** (IMG-03 descoped). The image SSL (CVCSPC pose-contrastive) is **faithfully reproducible
on the Back-Squat unlabeled set alone** — research-verified against the official code (an earlier
"CVCSPC needs BarbellRow" assumption was WRONG and corrected; see the CONTEXT D2). Approach:
ImageNet ResNet-18 **supervised baseline first**, then **faithful CVCSPC SSL** → fine-tune → F1 vs
paper **0.869** → deliverable pack. Same discipline as every prior phase.

## Read first (read ALL — verify, don't get lazy)
1. **This file.**
2. **`CLAUDE.md`** (repo root) — overview, stack, conventions, **GSD Workflow Enforcement** (drive all
   file-changing work through GSD).
3. **Your auto-memory** — `MEMORY.md` index + EVERY linked file (`feedback_*`, `project_*`, `reference_*`).
   The working contract. Especially: one-cell-at-a-time-STRICT, deliver-colab-as-ipynb,
   notebook-disconnect-safe, **progress-bars-required (tqdm on any Colab cell >2 min)**, heavy-training-new-notebook,
   colab-module-reload-after-git-pull, colab-notebook-cell-delivery, pytorch-persistent-workers,
   best-pt-metrics-history-stale, run-gsd-autonomously, dont-ask-to-push, interactive-execution,
   ai-correctness, dont-agree, working-style, **no-ai-comments-in-code**, full-extraction,
   supervisor-visualizations, realtime-demo-expectation, form-correction-status.
4. **`~/.claude/plans/what-pushed-me-back-iridescent-pnueli.md`** — the master plan. Phase 7 paragraph;
   the dataset table (Shallow-Squat 3,611 (2542/529/540), 44% erroneous); comparison target
   (**Shallow-Squat CVCSPC 0.869**, SimSiam 0.829); the method (baseline → SSL); the working agreement.
5. **`.planning/STATE.md`**, **`PROJECT.md`**, **`ROADMAP.md`** (Phase 7 section + success criteria),
   **`REQUIREMENTS.md`** (IMG-01, IMG-02; **IMG-03 DESCOPED**). NOTE the GSD state-handler counter
   quirk — trust ROADMAP + REQUIREMENTS + the SUMMARYs/CONTEXT over STATE frontmatter counters, and
   hand-reconcile STATE when the GSD tools lag.
6. **The Phase 7 planning artifacts (the authority for this phase):**
   `.planning/phases/07-image-based-errors-cvcspc/` — `07-CONTEXT.md` (locked decisions D1–D7; **D2 is
   the corrected CVCSPC-feasible decision — read it carefully**), `07-RESEARCH.md` (the full CVCSPC
   method reconstruction + the 3-term-vs-2-term loss finding + the standalone-feasibility verdict +
   Validation Architecture), `07-PATTERNS.md` (file→analog map + code excerpts), `07-VALIDATION.md`
   (Nyquist), and **`07-01-PLAN.md` … `07-05-PLAN.md`** (the 5 executable plans).
7. **The reusable pipeline CODE:** `backend/training/aqa/` — `eval/metrics.py` (reuse UNCHANGED:
   f1/pr_auc/threshold_sweep/confusion); `harness/colab.py` (staging/atomic-checkpoint/cpu-resume/
   RNG/prune — add image-zip staging + the unlabeled-Squat frame-extraction); `datasets/squat.py` /
   `ohp.py` (structural templates, minus video decode); `transforms.py` (the cv2 decode-robustness for
   tv 0.26). The video trainers (`supervised_train.py`, `md_pretrain.py`, `md_finetune.py`) are
   R(2+1)D-18-specific — Phase 7 creates a NEW lightweight ResNet-18 image trainer that mirrors their
   checkpoint/resume/early-stop/threshold-sweep + the `dataset_cls`/`checkpoint_phase` seams.
8. **The official CVCSPC code** (researcher already read it; you verify):
   `Fitness-AQA-Code/Code_Release/pose_contrastive_learning/self_supervised_learning/` —
   `dataloader.py` (phase-matched triplet via bar-trajectory phase; ImageNet norm; masking-only aug)
   and `train_test.py` (the 3-term loss at lines 64–68; Adam recipe; the AP<AN triplet-accuracy eval;
   phase-gap anneal). `CVC_SPC.py` / `opts*` / `linear_layers.py` / `dataloader_eval.py` are ABSENT —
   reconstruct from the present files + paper; record divergences.
9. **The deliverable pattern to REPLICATE for Shallow-Squat:** `docs/notebooks/05-08_ohp_*` (executed
   OHP pack), `docs/eval/FINDINGS_OHP.md`, `docs/figures/ohp_*.png`, and the
   `.planning/phases/06-…/figures/results.pkl` pattern (per-clip scores + curves → notebooks/figures
   built locally, no GPU). `backend/scripts/ohp_{eval,eda,finetune_curves,headline_chart}.py` are the
   viz-script analogs.

## The working contract (follow exactly)
1. **Verify, push back, stay correct.** Re-check every claim against code/data before acting. Disagree
   when warranted (the user explicitly wants this). Bar = academic correctness/defensibility (thesis;
   defense ~2026-06-03), faithful to the paper, identical metrics (F1-per-error, official splits). Cite
   or label `[ASSUMED]`. **Do NOT fabricate** any metric/curve — every number/plot from real data or a
   real run. macro/F1 computed in code, never hand-typed (a prior session shipped wrong literals; the
   user caught it).
2. **Interactive, one runnable unit at a time, no blind runs.** For Colab/training: hand ONE cell →
   WAIT for paste-back → validate → only then write the next. Never pre-write task N+1 before task N's
   paste-back is in (one-cell-at-a-time-STRICT). Trust-but-verify any subagent output. **Calibration the
   user set:** only gate (block for paste-back) when you NEED a cell's actual output to build the next
   one; for confirmation-only cells, state the expected output and tell them to ping only on a mismatch.
3. **Heavy training → fresh Colab notebook; prompt the user FIRST.** Deliver Colab notebooks as real
   `.ipynb` (not jupytext-only). Notebooks **disconnect-safe** (cache/skip-existing; resume from latest
   checkpoint; `map_location="cpu"`). **Any cell that runs >2 min MUST show in-loop progress (tqdm or
   per-batch prints)** — per-epoch summaries alone are not enough (the user made this a hard rule).
   `DataLoader(num_workers>0)` → `persistent_workers=True`. After `git pull` in a running runtime,
   **restart runtime or `importlib.reload`** (sys.modules is stale). Deliver new cells as **chat code
   blocks to paste** (Cell A's pull refreshes `.py` but not the open notebook). For `best.pt` training
   curves, load `metrics_history` from the `latest.txt`-pointed checkpoint, not `best.pt`. (Phase 7
   compute is LIGHT — ResNet-18 on small crops; the heavier bit is one-time frame-extraction of 4,970
   clips for CVCSPC.)
4. **Colab environment facts (verified this milestone, will recur):** Colab ships **torch 2.11.0+cu128,
   torchvision 0.26.0+cu128 — `read_video`/`read_video_timestamps` are REMOVED**. The training decode
   path already falls back to **cv2** (`transforms.decode_clip` / `count_frames`, `_HAS_TV_READ_VIDEO`
   guard); images use PIL/`torchvision.io.read_image` — confirm at Step 0. **`cv2.setNumThreads(0)` per
   DataLoader worker** (oversubscription fix — already in the worker hooks). PyAV 17 installed at Step 0
   before torch. A100 and L4 both available (the user pays per-unit) — **decode-bound work shows no A100
   speedup; use L4 unless a cache makes it GPU-bound.** Drive is **consolidated** at
   `My Drive/Fitness-AQA_dataset_release/` (one root; no -3-001/-3-002 split on Drive). When reading a
   `results.pkl` back from the Windows Drive mount (`G:\My Drive\...`), git-bash `cp`/`open` can read a
   stale/garbled copy — use **PowerShell `Copy-Item`** to force-materialize it first.
5. **Git:** push `fresh-start` to origin **without asking** (routine for the Colab Cell-A pull). NEVER
   force-push. Reconcile divergences by merge (`git merge origin/fresh-start`, prefer local on overlap),
   never force-overwrite. Stage files **by name only** — never `git add -A` (`Fitness-AQA/` ~5.8 GB,
   `Fitness-AQA-Code/`, weights, extracted media are untracked/gitignored). Commits:
   `feat(07)/fix(07)/test(07)/docs(07)`. End commit messages with
   `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
   **LESSON FROM THIS SESSION — heed it:** do git/STATE operations as **single Bash commands, not
   parallel batches** — a `grep` with no match exits 1 and cancels every other call in the same parallel
   block (this repeatedly aborted closeout mid-run). And `git show HEAD:file > file` racing with `Edit`
   on the same file corrupts STATE; restore-then-edit sequentially. The user runs a separate frontend
   chat on its own worktree/branch — never touch that branch.
6. **No AI comments in code.** Clean code; comment only a non-obvious WHY. No citation/narration
   comments (no "RESEARCH §", D-numbers, "mirrors X", "NEW kwarg"). That belongs in commit messages /
   SUMMARYs, not source. The user called this out explicitly. Applies to notebook cells too (section
   headers fine, paragraph narration not).
7. **GSD is how you operate.** Phase 7 is already PLANNED. Execute it with **`/gsd:execute-phase 7`**
   (or run the plans one at a time). Atomic commit per task; SUMMARY per plan; STATE/ROADMAP/REQUIREMENTS
   kept current (hand-reconcile counters). When the user authorizes autonomous GSD, decide routine gates
   from precedent (Phases 1–6) + memory — don't bounce settled questions. Genuine architectural forks /
   human-verify gates → ask via AskUserQuestion.
8. **Tone:** direct, evidence-based, terse. No sycophancy, no deadline-pressure framing, no unrequested
   scope.

## Current state (verify against files on arrival)
- **Branch `fresh-start` @ `38b2d91`**, pushed to origin (`Ibrahim-Shahin1/fitnova`), in sync.
- **Verify on arrival:** `python -c "import backend.app"` (exits 0); `python -m pytest
  backend/training/aqa/ -m "not slow" -q` (**expect 29 passed, 4 deselected** — Phase 7 Plan 01 adds new
  tests that grow this); `git log --oneline -8` (HEAD `38b2d91`); `git status -sb` (in sync). Push back
  if anything is off.
- **Pre-existing untracked/uncommitted (do NOT commit):** `Fitness-AQA/` (the ~5.8 GB dataset incl.
  extracted media), `Fitness-AQA-Code/` (cloned official code), `FLEX Dataset Paper.pdf`,
  `backend/data/distillation/`, `backend/_archive_form_v4_v6/`, the `*/flutter/generated_*` files,
  `.claude/scheduled_tasks.lock`, `docs/eval/.ipynb_checkpoints/`. There is also a long-standing benign
  `M .planning/phases/03-…/03-01-PLAN.md` (a CRLF/test-value doc tweak in a COMPLETE phase) — leave it.

## What the planning session did (so you don't redo or misread it)
- **Phase 6 (OHP) was completed end-to-end** — Plans 01–05, headline OHP test macro **0.6622**
  (Elbows 0.4474 / Knees 0.8770) on the official 339-clip split, **+0.050 SSL lift** over the baseline,
  **matches/edges paper Ours-MD 0.6502** (beats it on Knees), **val→test gap 0.011 (no overfitting)**.
  Deliverable shipped: `docs/figures/ohp_*.png` (7), `docs/notebooks/05-08_ohp_*` (executed),
  `docs/eval/FINDINGS_OHP.md`, `.planning/phases/06-…/figures/results.pkl`. OHP-01 complete; serving
  descoped (D1 — the form-correction frontend was cancelled earlier this milestone; benchmark eval is
  the deliverable). 2-seed ensemble (time-driven, vs Squat's 3); TTA not adopted (reverses on test).
- **Carry-fixes landed in the main pipeline code this milestone** (all on `fresh-start`, all green):
  cv2 decode fallback for tv 0.26; `drop_last=True` on the SSL loaders (size-1-batch BatchNorm crash);
  a decode-cache (`decode_clip_cached`) + `cv2.setNumThreads(0)` + SSL `num_workers=8`;
  `dataset_cls`/`checkpoint_phase` seams on `run_supervised_epoch`/`run_md_pretrain_epoch`/
  `run_md_finetune_epoch`; `best.pt` writes pass `update_latest=False`. Phase 7's image trainer should
  follow the same seam patterns.
- **Phase 7 was discussed + researched + planned** (this is where you start executing):
  - **IMG-03 (BarbellRow) DESCOPED** by the user (compute/time) — recorded in REQUIREMENTS + ROADMAP.
  - **CVCSPC feasibility CORRECTED:** an initial read concluded CVCSPC was inherently cross-exercise
    (needs BarbellRow) — **that was WRONG.** Verified against `dataloader.py` + the paper: CVCSPC
    pretrains on the **Back-Squat unlabeled set ALONE** (anchor/positive = phase-matched frames across
    two *Squat* reps via the per-rep bar-trajectory phase; negative = a frame ≥ phase-gap away). The
    4,970 unlabeled Squat clips + bar-trajectory JSONs (the Phase-4 SSL set) are exactly the input.
    **Decision (CONTEXT D2): do FAITHFUL CVCSPC**, baseline-first.
  - **Plan-check found 2 real blockers** (fixed in `abd6bde`): the video trainer hardcodes R(2+1)D-18
    5-D tensors with no model-injection seam → Phase 7 needs a **separate 2D ResNet-18 image trainer**;
    and the official dataloader applies a `traj_nan.json` exclusion → the SSL frame set must exclude it.
    Also: single-logit head (Shallow-Squat is ONE binary error → `Linear(512,1)` + BCE, not 2 outputs);
    loss-directionality unit test; phase-gap anneal; val→test gap reporting.

## Next — Phase 7: the 5 plans (the task)
**Goal (IMG-01 + IMG-02):** Shallow-Squat depth-error detector — ImageNet ResNet-18 supervised baseline
+ faithful CVCSPC pose-contrastive SSL, F1 on the official split, compared to the published CVCSPC 0.869.
Serving descoped (D1). The plans (read each `07-0N-PLAN.md` in full before executing it):
- **07-01 (Wave 0, LOCAL CPU, autonomous):** scaffold the genuinely-new code with pure-function cores
  implemented + unit-tested BEFORE any GPU — `datasets/shallow_squat.py` (image loader, ImageNet norm,
  official splits), `datasets/cvcspc_ssl.py` (phase-matched triplet + the 3-term loss),
  `harness/image_supervised_train.py` (ResNet-18 trainer), `harness/cvcspc_pretrain.py`, the `colab.py`
  image-zip staging + unlabeled-Squat frame-extraction helpers, the 4 test files, and the
  `09_shallow_squat_baseline` notebook scaffold. **Gate: `pytest -m "not slow"` green INCLUDING the new
  tests AND no Phase-3/4/6 regression.** This is the OHP/Squat Wave-0 pattern — run it locally exactly
  like the OHP Plan 01 was run (you run pytest yourself; no Colab).
- **07-02 (Colab):** ImageNet ResNet-18 supervised baseline → F1 on the official Shallow-Squat test
  split (the IMG-02 floor + the SSL control).
- **07-03 (Colab, the heavier one):** faithful CVCSPC SSL pretrain — frame-extract the 4,970 unlabeled
  Squat clips (one-time, skip-existing, tqdm), `traj_nan` excluded, phase-contrastive triplet + 3-term
  loss, AP<AN triplet-accuracy convergence monitor → `backbone.pt`. **Prompt the user to open a fresh
  notebook before this.**
- **07-04 (Colab):** fine-tune from the CVCSPC backbone + multi-seed ensemble + val-tuned threshold +
  test eval → baseline vs CVCSPC vs paper 0.869 + val→test gap → `results.pkl`.
- **07-05 (LOCAL):** the deliverable notebook pack (EDA + training/eval) + figures + `FINDINGS` +
  phase SUMMARY + reconcile ROADMAP/REQUIREMENTS/STATE; mark IMG-01/IMG-02 complete.

**Key technical facts (verified locally 2026-05-30):**
- **Shallow-Squat data:** `Fitness-AQA/Fitness-AQA_dataset_release-20260518T073812Z-3-001/Fitness-AQA_dataset_release/Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/`
  — `images.zip` (3,738 `crops_unaligned/{id}.jpg`), `labels_shallow_depth.json` (flat `{id: 0/1}`,
  3,611 entries, **43.9% positive**), `splits/{train,val,test}_ids.json` (lists, **2542/529/540**).
  ID format `{video}_{rep}_{frame}`. (Note: `splits/` filenames are `*_ids.json`, not `*_keys.json`.)
- **CVCSPC SSL set:** the unlabeled Back-Squat videos (4,970) + `bar_trajectories` JSONs — staged via
  `stage_unlabeled_squat_videos` (Phase 4); the SSL dataloader needs **extracted frames** per clip, so
  Plan 03 has a frame-extraction Step 0 (cv2, skip-existing, tqdm).
- **CVCSPC loss:** the official `train_test.py:64-68` implements a **3-term** denominator
  (anc-pos / (anc-pos + anc-neg + pos-neg)); the paper Eq.1 shows 2 terms. Implement the CODE version,
  document the deviation in FINDINGS. Features are L2-normalized; Adam.
- **Image transform:** resize→CenterCrop 224², **ImageNet** mean/std `[0.485,0.456,0.406]/[0.229,0.224,0.225]`
  (ResNet-18 native), aug = masking-only in the shipped code. NO video decode/temporal axis.
- **Comparison:** paper CVCSPC Shallow-Squat **0.869**, SimSiam 0.829. Baseline = our control (no
  published supervised-image row), narrative = baseline → CVCSPC → paper (same shape as OHP).
- **Checkpoints route to `phase07`** (the `checkpoint_phase` seam); use a multi-seed ensemble per D4.

## Loose ends / aware-don't-trip
- **Orphaned form backend** (WS `/ws/form-session`, `SquatLiveSession`, the dead Flutter form screens)
  is intentionally KEPT for fallback — do NOT strip it. The form-correction frontend tab is cancelled.
- **STATE counter quirk:** the GSD `state.*` tools miscount frontmatter progress on this repo and the
  planner subagent reverted them once — hand-reconcile (Phases 1–6 done = `completed_phases: 6`;
  `total_plans` includes the 5 Phase-7 plans; `percent` ≈ 75). Trust ROADMAP/REQUIREMENTS/SUMMARYs.
- **`Fitness-AQA-Code/`** (cloned official repo) is untracked/gitignored — it's the CVCSPC reference;
  don't commit it.

## Immediate next actions
1. Read the "Read first" list in full; verify Phase-6 state + arrival checks (`import backend.app`,
   29 tests, HEAD `38b2d91`, in sync).
2. Read `07-CONTEXT.md` (esp. D2), `07-RESEARCH.md`, and `07-01-PLAN.md` in full.
3. **Execute Phase 7 Plan 01 LOCALLY (CPU)** — scaffold + unit tests + the Wave-0 green gate, exactly
   like OHP Plan 01 (you run pytest; no Colab; commit per task; trust-but-verify if you spawn an executor
   subagent — the OHP run used `gsd-executor` on sonnet and it worked, but re-check the diffs yourself).
4. Then Plans 02–04 on Colab (cell-by-cell, paste-back, disconnect-safe, tqdm on long cells, `.ipynb`
   delivered, push `fresh-start` before handing cells). Plan 03 needs a fresh notebook — prompt first.
5. Plan 05 builds the deliverable pack locally from `results.pkl` (PowerShell-copy it off Drive first).
6. Save a `results.pkl`-style artifact in Plan 04 so the Plan-05 pack builds offline, exactly like Squat/OHP.
