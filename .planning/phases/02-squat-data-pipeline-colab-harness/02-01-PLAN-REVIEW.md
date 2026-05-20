# Phase 2 — PLAN.md Review

**Reviewed:** 2026-05-20
**Reviewer:** plan-checker (goal-backward verification)
**Plan file:** `.planning/phases/02-squat-data-pipeline-colab-harness/02-01-PLAN.md` (776 lines, 12 tasks)

---

## 1. Verdict

**REVISE.**

The plan is structurally strong: interfaces are pre-declared, research overrides are integrated with explicit rationale, every ROADMAP success criterion has a named task chain, and the supervisor-priority visualization has a blocking human-verify gate. However, several findings would degrade or invalidate the acceptance proof if executed as written. The most consequential is **Finding F1** (Task 10's "in-memory module purge" is not equivalent to the kernel-restart resume that the ROADMAP success criterion requires) and **Finding F2** (a hidden ordering dependency between Task 6 and Task 7 makes the declared task order unexecutable on a fresh Colab session). These are fixable inside the existing 12-task structure — no re-plan is needed.

Send back to planner with the fixes in Section 9. After revision, this is shippable.

---

## 2. Goal coverage assessment

ROADMAP Phase 2 success criteria (`.planning/ROADMAP.md` lines 37–40):

| # | Criterion | Tasks claimed | Coverage end-to-end? |
|---|-----------|---------------|----------------------|
| 1 | Squat clips load as `(video tensor, KIE/KFE label)` batches using the official splits | Tasks 3 (`splits.py`), 4 (`transforms.py`), 5 (`squat.py`) | **COVERED.** Task 5 acceptance ("`next(iter(loaders['train']))` returns `(video[B=2, 3, 32, 112, 112], labels[B=2, 2])` + counts reconcile 1136/243/244") closes the chain. |
| 2 | Decoded and augmented sample frames are visualized and confirmed correct | Task 6 (PNG generation), Task 11 (blocking human-verify) | **COVERED.** Task 11 is a `checkpoint:human-verify` gate with concrete visual criteria (a–e). |
| 3 | The Colab harness checkpoints to Google Drive and resumes from the latest checkpoint after a restart | Tasks 7 (atomic save + RNG), 9 (fresh write), 10 (resume read + bitwise determinism) | **PARTIAL — see F1.** Task 10 substitutes an in-session `sys.modules` purge for an actual kernel restart and admits this in line 628. The criterion says "after a restart"; the test does not exercise a restart, and the bitwise comparison it performs is between two fresh runs, not between a fresh run and a resumed run. |

Requirements:

- **SQUAT-01** (PyTorch data pipeline — decode, 32-frame sampling, transforms, official-split loaders): covered by Tasks 3–6. Every sub-claim (decode, sampling, transforms, official splits, KIE+KFE both labels per clip) maps to a task and an acceptance line.
- **SQUAT-02** (resumable Colab training harness — checkpoints each epoch, auto-resumes after a disconnect): structurally covered by Tasks 7–10, but the **auto-resume-after-disconnect** sub-claim is what F1 weakens.

Unmapped relevant requirements from `REQUIREMENTS.md`: none. SQUAT-03..05, IMG-*, OHP-*, EVAL-* belong to later phases.

---

## 3. Research integration check

The 6 research overrides flagged in the request:

| Override | Present in plan? | Citation present? | Notes |
|----------|------------------|-------------------|-------|
| `Code_Release/motion_disentanglement/` empty — Task 1 reframed around documenting the gap | YES (D19, Task 1 action lines 270–276) | YES — RESEARCH section 2 cited 4 times | Strong: Task 1 explicitly inverts the CONTEXT.md plan ("clone and adapt") into "clone and document the gap." |
| Input crop **112×112** (not 224) | YES (D1, line 237; interfaces `crop_size: int = 112`, line 148) | YES — RESEARCH section 1, 4 cited, with the rationale "paper's 320→224 was the 2DCNN/Waseda path, not the in-the-wild R(2+1)D path" | Strong override of CONTEXT.md line 39 (which said `(B, 3, 32, 224, 224)`). Correctly parameterized for Phase 3 ablation. |
| Decoder wrapped behind `decode_clip` | YES (D6 line 242, interfaces line 123–126, Task 4 action item 3) | YES — RESEARCH section 5 cited | Strong: explicit "single-function swap target for TorchCodec." |
| Horizontal flip **OFF** | YES (D5 line 241, Task 4 action item 4, interfaces line 135 "NO horizontal flip") | YES — RESEARCH section 4 + paper line 576 + CVCSPC commented-out aug | Strong, with the elevated certainty noted ("changed from CONTEXT.md's 'default OFF until confirmed' → confirmed"). |
| `pos_weight` (not `WeightedRandomSampler`) | YES (D9 line 245, Task 5 action item 1, interfaces lines 152) | YES — RESEARCH section 8 cited; paper line 413 quoted | Strong: contract is "dataset exposes `pos_weight`, Phase 2 tiny does not apply it, Phase 3 consumes." Concretely closes CONTEXT.md's vague "exposes weights as a hook." |
| Atomic Drive write (tmp + verify + replace + latest.txt last) | YES (D11 line 247, Task 7 action item 5) | YES — RESEARCH section 6 cited; Drive FUSE non-atomic-rename caveat called out | Strong: every step (a)–(d) of the contract is explicit. |

No PLAN.md decision contradicts RESEARCH.md without justification. **Decisions D1, D5, D9, D19 explicitly flag where they override CONTEXT.md** with the research-cited reason — exactly the trace requested.

One small gap: RESEARCH section 6 specifies `os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")` must be set **before CUDA initialization** for `torch.use_deterministic_algorithms(True)` to work. The plan (D13 + Task 7 item 3) sets it on module import, but Task 2 does not pin that `harness/colab.py` is imported before any other CUDA-touching module in the notebook. Minor — Finding F8.

---

## 4. Working-agreement compliance (per task)

The 4-clause working agreement at lines 197-204: one runnable unit per task, no blind runs, resumability as a build constraint, visualizations first-class.

| Task | Runnable unit | Compliance | Notes |
|------|---------------|------------|-------|
| 1 | clone + write CODE-RELEASE-NOTES.md | violation (mild) | Two-step compound: git clone, then author CODE-RELEASE-NOTES.md. The paste-back is `ls -la` + `wc -l` — fine, but the authoring step is itself multi-output. See F3. |
| 2 | scaffold 8 files + Step-0 cell | OK | One paste-back: `find ... wc -l` + Step-0 output. The 8 files are stubs; no consequential side effects. |
| 3 | implement splits.py + run unit-check | OK | One paste-back: per-split counts + (KIE+, KFE+) tuples from the notebook cell. |
| 4 | implement transforms.py + smoke | OK | One paste-back: `__main__` smoke + one decode on a real .mp4. |
| 5 | implement squat.py + build_loaders | OK | One paste-back: (batch.shape, label.shape, dtype, pos_weight). |
| 6 | visualization cell + PNG save | OK (but blocked by F2 ordering) | Single notebook cell producing PNG + display. Idempotent (line 467). |
| 7 | implement colab.py + 4-step demo | violation | Action item lists six implementation concerns, then the runnable unit is (a) mount Drive, (b) stage videos with timing, (c) round-trip a fake checkpoint, (d) demonstrates RNG capture/restore. Four distinct paste-backs in one task. See F4. |
| 8 | implement tiny_train.py + shape check | OK | One paste-back: stdout of `run_tiny_epoch(resume=False)`. But the real execution is Task 9 — Task 8 is implementation only. Slightly fuzzy boundary but acceptable. |
| 9 | cleanup + fresh tiny run | OK | Two cells (cleanup, run), one logical paste-back (per-batch losses + val loss + checkpoint path + config_hash). |
| 10 | three-cell sequence | violation | Three cells simulating a Colab restart cleanly — explicitly three runnable units with three paste-backs (purge, resume, bitwise re-run). See F5. |
| 11 | human verify | OK | Checkpoint gate, by design. |
| 12 | write SUMMARY.md | OK | One paste-back: `cat SUMMARY.md`. |

Resumability across cells: PNG-before-`plt.show()` (line 467, 741), idempotent `stage_squat_videos` (D15, Task 7 item 2), cleanup cell isolated in Task 9 as deliberately destructive (line 590). All consistent with the agreement.

Visualizations first-class: Tasks 6, 9, 10 all save PNGs before display per line 467 and 741. The two figures are itemized in the visualization_deliverables table (line 736).

---

## 5. Scope hygiene

Inside the fence (good):
- PyTorch loaders, decoded-batch viz, resumable harness, tiny 1-epoch run — all match CONTEXT.md scope fence (lines 169-177).
- Toy model is Conv3d(3,4,3) -> AdaptiveAvgPool3d -> Linear(4,2), exactly the minimal shape-strict model per CONTEXT.md line 142. Action item 1 in Task 8 even justifies the channel count.
- Code lives under `backend/training/aqa/{datasets,harness,notebooks}/` per D18 and Phase 1 convention.

Drift OUT of scope I checked for and did not find:
- No R(2+1)D-18 instantiation. No `torchvision.models.video.r2plus1d_18` import anywhere in the task actions. OK.
- No F1/PR-AUC computation. OK.
- No FastAPI, no `backend/services/*` modifications. OK.
- No MediaPipe. OK.
- No OHP, BarbellRow, Shallow-Squat loaders. OK.
- `pos_weight` is computed but NOT applied to the tiny run loss (Task 8 item 2: no pos_weight, per D9). OK.

Scope creep I would flag: none material. Two minor scope expansions:
- Task 1 also writes `.gitignore` (one line). Acceptable — D19 requires it.
- Task 10 third cell deletes `epoch_000.pt` and re-runs fresh to compare bitwise (line 643-645). This is additional to the documented load checkpoint and resume — it is a regression guard, not the resume test. The line 649 admission acknowledges this honestly. Borderline acceptable but conflates two tests. See F1.

Missing-from-scope check: CONTEXT.md does not promise a kernel-restart-resume test, only "auto-resumes from the latest Drive checkpoint without manual intervention" (line 16). The ROADMAP says "resumes from the latest checkpoint after a restart" (line 40). The plan Task 10 is the place this is tested — and it is where F1 lands.

---

## 6. Interface contract (Phase 3 readiness)

The `<interfaces>` block at lines 96-194 is the strongest part of this plan. Phase 3 inheritors will not have to spelunk.

What is named explicitly:
- `splits.index(split, *, drive_root, videos_root) -> list[ClipRecord]` with `ClipRecord(clip_id, video_path, label_kie, label_kfe)` (lines 100-112).
- `transforms.uniform_sample_indices`, `decode_clip`, `spatial_train`, `spatial_val`, `KINETICS_MEAN`, `KINETICS_STD` (lines 119-138).
- `SquatKIEKFEDataset.__init__` signature with every kwarg, `pos_weight` field, `build_loaders` factory (lines 141-162).
- `colab.{mount_drive, stage_squat_videos, capture_rng_state, restore_rng_state, hash_config, CheckpointConfigMismatchError, atomic_save_checkpoint, load_latest_checkpoint, prune_checkpoints}` (lines 165-184).
- `TinyModel` architecture line-by-line (line 187-188), `run_tiny_epoch` full signature with defaults (lines 190-193).

Output tensor shape is pinned in 4 places: D1, interfaces line 135-157, Task 5 acceptance (line 439), the verification mapping table (line 721). Hard to misread.

Checkpoint payload schema is fully specified at D10 (line 246): epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict (optional), rng_state (4-RNG dict), metrics_history, config_hash, config_repr, code_version. Matches RESEARCH section 6 verbatim.

Gaps Phase 3 might hit:
- The `metrics_history` field is declared but its per-entry schema is not (Task 8 says log each batch loss with logger.info — does the checkpoint store these in metrics_history per epoch, as a list of dicts, what keys?). See F6.
- `expected_config_hash` in `load_latest_checkpoint` is documented as raising on mismatch but the format of `config_repr` (the debug-readable companion to `config_hash`) is not fixed. Phase 3 will reinvent. See F6.
- `run_tiny_epoch` returns dict with keys epoch, train_loss, checkpoint_path, rng_capture (line 193) — but Task 9 acceptance refers to result_fresh.train_loss, val_loss, config_hash. The val_loss and config_hash keys are not in the declared return shape on line 193. Mild inconsistency; F7.

---

## 7. Visualization audit

Per `<visualization_deliverables>` (lines 733-742):

| Figure | File | Task | What it shows | Concrete enough? |
|--------|------|------|---------------|-------------------|
| Decoded-and-augmented batch grid | `figures/decoded_batch.png` | 6 | 2 clips x 8 frames, KIE/KFE labels under each row, dpi=120, figsize=(16, 5) | YES. Task 6 acceptance is explicit on dimensions (>=1600x400), file size (>200 KB), and visual criteria (a)-(e) at line 476. |
| Tiny-run loss trajectory (overlay) | `figures/tiny_train_loss.png` | 9, 10 | Two coincident fresh-run curves | YES. Task 10 acceptance requires the curves to coincide with bitwise-equal underlying data. |

The supervisor-priority visualization (the decoded grid) is the explicit subject of Task 11 (checkpoint:human-verify, blocking). The 5 visual criteria in Task 11 (line 676-681) are concrete and reviewable.

One concrete miss: the plan declares two PNGs but `figures/` does not exist in the repo yet (verified). Task 2 scaffolding action (lines 304-313) creates module files but does NOT create the `figures/` directory. Tasks 6 and 9 will write into a non-existent directory unless Python auto-creates it via the matplotlib save call (which it does NOT — `plt.savefig` to a missing dir fails). Need `os.makedirs` or `Path(...).parent.mkdir(parents=True, exist_ok=True)` in Tasks 6, 9, 10. See F9.

JSON sidecar: `tiny_fresh_loss.json` (line 597) is named and used in Task 10 bitwise check (line 645). Not in the `files_modified` frontmatter list (lines 7-19) — minor. See F10.

---

## 8. Risk register completeness

11 risks listed (R1-R11). Covers:

- OK: Decoder bugs (R1), Colab disconnect (R2), Drive write atomicity (R3), Code_Release gap (R4), flip semantics (R5), RNG-resume drift (R6), wrong tensor shape (R7), multi-rep clip handling (R8), aspect drop (R9), `num_workers=0` slowness (R10), `opts_exercise_qa` repo incompleteness (R11).

Specifically requested by review: Drive write atomicity, decoder deprecation, multi-rep clip handling, and Code_Release gaps — all four are present (R3, R1 + decoder-deprecation noted in D6 + R11 covers the bundled-config gap, R8, R4 + R11).

Notable gaps from RESEARCH that are NOT in the risk register:
- RESEARCH section 6 admits DataLoader worker RNGs ... restoring main-process RNG is insufficient. Phase 2 tiny run uses num_workers=0 — this is R10 essentially. OK.
- RESEARCH section 6 warning: `os.environ["CUBLAS_WORKSPACE_CONFIG"]` must be set before CUDA init. Not in risk register. See F8.
- RESEARCH section 5 footgun: Colab can hit a memory limit if you read_video a long clip and then crop, instead of indexing the wanted frames first. The plan `decode_clip` reads the whole video and then `out = video[indices]` (Task 4 item 3 line 374). For a 404-frame clip at width 480 x height 600 x 3 bytes x uint8 ~= 350 MB raw, that is a real Colab memory concern. The bigger clips will OOM. See F11.
- Task 9 assumes the user has 16 train clips loaded fast enough that the tiny run finishes — but cold-stage on Drive to local is ~10s zip copy + ~20s unzip on T4 per line 532. Not a risk per se, but the user-facing wall time budget is not surfaced anywhere. Low priority.

---

## 9. Concrete findings

### F1 — Task 10 in-memory module purge is not a kernel-restart resume test (HIGH; blocks ROADMAP criterion 3)

Where: lines 624-649, especially line 627-628 (A real Colab restart would do this plus drop GPU memory; for Phase 2 acceptance the module purge + RNG reset is the regression-equivalent. Document this honestly inline.) and line 643-645 (the bitwise check actually deletes `epoch_000.pt` and re-runs from resume=False).

Problem:
- The ROADMAP success criterion is resumes from the latest checkpoint after a restart. Task 10 cell 2 does read `latest.txt` and load the checkpoint, but because the tiny run is 1 epoch and the loop exits immediately (line 640), the resume path runs no training batches in the resume case. There is no observable proof that a resumed-from-checkpoint training step produces correct outputs. The acceptance is just stdout shows resumed from `epoch_000.pt`; max_epochs reached, exiting.
- Cell 3 bitwise test deletes the checkpoint and runs resume=False twice. This proves seed-42 determinism on the fresh path, not the resume path. Two cold runs being bitwise-equal does not prove RNG capture/restore works — only that the deterministic seed mechanism works.
- The actual RNG capture/restore code is only exercised in cell 2, which immediately exits without comparing anything.

Fix: Restructure Task 10 to actually test resume semantics:
- (a) run_tiny_epoch(resume=False, max_epochs=2) to baseline a 2-epoch trajectory (record per-batch losses across both epochs).
- (b) run_tiny_epoch(resume=False, max_epochs=1) fresh to write `epoch_000.pt`.
- (c) Module purge + run_tiny_epoch(resume=True, max_epochs=2) — this loads `epoch_000.pt`, restores RNG, and trains epoch 1.
- (d) Compare (c) epoch-1 batch losses to (a) epoch-1 batch losses — bitwise-equal proves RNG capture/restore is correct.

This requires extending `run_tiny_epoch` to accept `max_epochs` (currently 1-epoch only). Either widen it (and update D14 / interfaces line 192) or keep 1-epoch and rename the proof bitwise determinism under fixed seed and explicitly declare ROADMAP criterion 3 resume test as deferred to Phase 3 first real epoch.

Either fix is acceptable; the current plan does neither.

### F2 — Task 6 needs Task 7 staging but Task 6 runs before Task 7 (HIGH; blocks executable ordering)

Where: Task 6 dependencies (line 479): Tasks 3, 4, 5 (full loader stack); Task 7 must run first if videos are not already staged. Task 6 builds loaders and pulls a real batch (line 458). Loaders cannot resolve `videos_root='/content/squat_videos'` until `stage_squat_videos` has executed.

Problem: The dependency note is hedged (if videos are not already staged) but for a fresh Colab session — the default execution mode per CONTEXT.md line 103 — videos are not already staged. Task 7 must execute before Task 6. The plan task ordering puts Task 7 after Task 6.

Fix: Reorder. Either:
- Swap Task 6 and Task 7 so Task 7 (`colab.py` + mount_drive + stage_squat_videos) precedes Task 6 (visualization that consumes staged videos).
- Or split Task 7 into 7a (Drive mount + zip-stage) executed before Task 6, and 7b (checkpoint primitives + RNG) executed after Task 6.

The current plan also flags this in Task 6 dependency line — the writer was aware but did not apply the reorder. Apply it.

### F3 — Task 1 bundles two consequential operations into one runnable unit (MEDIUM)

Where: Task 1 action (lines 267-277).

Problem: Clone the official repo and author CODE-RELEASE-NOTES.md are two distinct deliverables. The paste-back at line 285 demands both `ls -la Fitness-AQA-Code/...` AND `wc -l CODE-RELEASE-NOTES.md` — two separate outputs from two separate operations. Working agreement clause 1 (one runnable thing per task) is technically met if you treat both as one shell + editor session, but the agreement spirit favors one cell at a time.

Fix: Split into Task 1a (clone + .gitignore + paste back `ls -la` + `git status`) and Task 1b (write CODE-RELEASE-NOTES.md + paste back `wc -l` + first 50 lines). Adds one task but keeps the loop tight.

### F4 — Task 7 has four runnable sub-units in one task (MEDIUM)

Where: Task 7 acceptance criteria (lines 530-535): paste-back is (a) mounts Drive, (b) stages videos with timing, (c) round-trips a fake checkpoint, (d) demonstrates RNG capture/restore — four distinct demonstrations.

Problem: Same as F3 but with four sub-units. Each sub-unit could fail independently and the user has to triage which one. Also, `stage_squat_videos` is the slowest step (~30s cold) and you do not want to re-run it because checkpoint round-trip code is wrong.

Fix: Split Task 7 into:
- Task 7a: implement + paste-back `mount_drive` + `stage_squat_videos` (the prerequisite-for-Task-6 work — see F2).
- Task 7b: implement + paste-back `capture_rng_state`/`restore_rng_state` + cudnn settings.
- Task 7c: implement + paste-back `hash_config` + `atomic_save_checkpoint` + `load_latest_checkpoint` + `prune_checkpoints`.

Three tighter tasks, each one runnable unit. Task 7a moves to before Task 6.

### F5 — Task 10 has three runnable cells listed as one task (MEDIUM)

Where: lines 624-651, explicitly labelled three cells.

Fix: Split into Task 10a (in-memory purge + resume cell + paste back), Task 10b (delete checkpoint + bitwise re-run + paste back + assertion), Task 10c (overlay plot + perturbation negative test). Also re-merge with F1 fix.

---
### F6 — Checkpoint metrics_history and config_repr schemas are unspecified (MEDIUM)

Where: D10 lists `metrics_history` and `config_repr` but does not pin their schema. Task 8 item 2 logs per-batch losses but is silent on how/whether they go into `metrics_history`.

Problem: Phase 3 will need to read prior epochs metrics to decide best and to compute the val-set early-stop signal. If Phase 2 leaves the schema implicit, Phase 3 will define it and Phase 4 will see drift.

Fix: Pin in the interfaces block:

    metrics_history: list[dict] = [
        {"epoch": int, "train_loss": float, "val_loss": float,
         "train_loss_per_batch": list[float]}
    ]
    config_repr: dict   # exact JSON-safe mirror of the hashed config (same keys as D12)

Add a line to Task 8 action that the checkpoint payload includes a `metrics_history` entry for epoch 0.

### F7 — run_tiny_epoch return signature mismatch between interfaces and acceptance (LOW)

Where: Interfaces line 193 declares `run_tiny_epoch` returns keys epoch, train_loss, checkpoint_path, rng_capture. Task 9 acceptance references `result_fresh['train_loss']` (ok), the run prints val loss, checkpoint path, config_hash (line 607) — val_loss and config_hash keys are not in the declared return shape.

Fix: Update line 193 to:

    def run_tiny_epoch(...) -> dict:
        # {"epoch": int, "train_loss": list[float], "val_loss": float,
        #  "checkpoint_path": str, "config_hash": str}

Drop rng_capture from the declared shape — Task 8 does not actually return it.

### F8 — CUBLAS_WORKSPACE_CONFIG must be set before CUDA init, not at harness/colab.py import (MEDIUM)

Where: D13 (line 249), Task 7 action item 3 (line 501).

Problem: Setting the env var on module import works only if `harness/colab.py` is the first thing imported in the notebook (before `import torch` triggers CUDA context creation). The Step-0 cell in Task 2 declares import sanity before specifying which order. If `torch` is imported before `colab` (the common case — notebook usually does `import torch` early for sanity checks), the env var arrives too late and `torch.use_deterministic_algorithms(True)` raises at runtime.

Fix: Either:
- Document in the Step-0 cell (Task 2 action) that `import backend.training.aqa.harness.colab` MUST be the first import before any torch import, OR
- Move the `os.environ["CUBLAS_WORKSPACE_CONFIG"]` write into a tiny `_envinit.py` that the notebook Cell 0 imports as line 1.

### F9 — figures/ directory is referenced but not created by any task (LOW; will cause Task 6 to fail at plt.savefig)

Where: Tasks 6, 9, 10 write to `.planning/phases/02-squat-data-pipeline-colab-harness/figures/`. The directory does not exist (verified). No task creates it.

Fix: Add to Task 2 action: Create the `.planning/phases/02-.../figures/.gitkeep` file — or inline `Path(...).parent.mkdir(parents=True, exist_ok=True)` in the visualization cells. Either fix is one-line.

### F10 — tiny_fresh_loss.json is named but not in files_modified frontmatter (LOW)

Where: line 597 (Task 9) writes `figures/tiny_fresh_loss.json`; not listed in `files_modified` (lines 7-19).

Fix: Add it to the frontmatter `files_modified` list.

---
### F11 — decode_clip reads full video then indexes — OOM risk on long clips (MEDIUM)

Where: Task 4 action item 3, line 372-375: video, _, _ = torchvision.io.read_video(path, output_format="TCHW", pts_unit="sec") — discard audio and info ... out = video[indices] — fancy-index along the T axis.

Problem: RESEARCH section 5 specifically warns: Long clips: gate by clip duration to avoid OOM at high res. Colab can hit a memory limit if you read_video a long clip and then crop, instead of indexing the wanted frames first. Phase 1 measured Squat clips up to 404 frames at width 480 x height 600 x 3 bytes uint8 ~= 350 MB per clip uncompressed. Batch of 2 = 700 MB raw — plus the model + decode buffers. On a Colab T4 (~15 GB) this is uncomfortable; on a free CPU runtime it is an OOM risk.

The Task 5 action uses `torchvision.io.read_video_timestamps` first to learn `num_frames` cheaply (line 418), then calls `decode_clip`. Good — but `decode_clip` itself still reads the whole video before indexing.

Fix: Use `torchvision.io.read_video(path, start_pts=..., end_pts=..., pts_unit="sec")` with start_pts = min(indices)/fps and end_pts = max(indices)/fps to bound the decode to the window containing the picked indices. Then re-base the indices. Or — simpler — accept the memory hit for Phase 2 (the tiny run only touches 16 clips), and add a TODO comment in `decode_clip` referencing RESEARCH section 5 and add a risk register entry. Either way, the inline B-fix comment-style warning in Task 4 should mention this.

### F12 — Tiny-run determinism preconditions are spread across D13, D14, Task 7, and Task 8 without a single audit list (LOW)

Where: D13 (RNG + cudnn), D14 (num_workers=0), Task 7 item 3 (cudnn at import), Task 8 item 2 (Set the global seed).

Problem: A reader trying to audit does this guarantee bitwise determinism? has to assemble: seed via all 4 RNG paths, cudnn.deterministic, no benchmark, use_deterministic_algorithms, CUBLAS_WORKSPACE_CONFIG (F8), num_workers=0, generator-driven uniform_sample_indices, generator-driven spatial_train random crop. That is 8 preconditions across 4 sections.

Fix: Add a Determinism checklist subsection to the plan (or to RESEARCH section 6) listing all 8 preconditions with the task that satisfies each. Phase 3 will need this when bumping num_workers > 0.

### F13 — torch.use_deterministic_algorithms warn on CPU-only hosts (LOW)

Where: D13 + Task 7 acceptance line 531 (All 9 names exported and importable on a non-Colab machine).

Problem: `torch.use_deterministic_algorithms(True)` on CPU-only is silent but the surrounding code that gates on `torch.cuda.is_available()` (mentioned at line 531) needs to be explicit. The plan says gate them with `torch.cuda.is_available()` but does not say which gates apply to which calls. `cudnn.deterministic` is harmless on CPU; the env var write is harmless; `get_rng_state_all` will return an empty list on CPU which is fine.

Fix: None required — note here for the executor.

---

## 10. Recommendation to orchestrator

Send back to planner with the F1-F11 fixes (especially F1 and F2 as blocking, F3-F8 and F11 as required revisions, F9 and F10 as one-line fixes, F12 as polish).

After revision:
- F1 fix forces Task 10 to actually test resume semantics — without it, ROADMAP criterion 3 resumes from the latest checkpoint after a restart is not proven.
- F2 fix makes the task DAG executable from a fresh Colab session.
- F3-F5 fixes bring the plan into compliance with the one runnable unit per task working agreement that CONTEXT.md and the request both call non-negotiable.
- F6-F8 close the contract gaps that Phase 3 would otherwise re-invent.
- F11 surfaces a real OOM risk that the plan currently ignores.

Estimated revision effort: ~30 minutes of plan editing. No re-research needed; every fix is sourced from RESEARCH.md or from a contract internal to the plan. The 12-task structure becomes ~15 tasks after the splits, which is still within scope sanity (tasks/plan target 2-3 does not strictly apply — this is one plan covering an entire phase, so the cap is loose; current 12, post-revision ~15 still readable).

Do not send for re-plan. The bones are right. The plan is one revision pass away from PASS.

---

*Reviewed against `.planning/ROADMAP.md` Phase 2, `.planning/REQUIREMENTS.md` SQUAT-01/02, `02-CONTEXT.md`, `02-RESEARCH.md`, `01-DATASET-REPORT.md`, `01-01-PLAN.md`, `CLAUDE.md`. 13 findings issued; severity distribution: 2 HIGH (blocking), 5 MEDIUM, 6 LOW.*

---

# Iteration 2 -- Revision Verification

**Reviewed:** 2026-05-20
**Reviewer:** plan-checker (delta verification)
**Plan file:** 02-01-PLAN.md (1054 lines, 16 tasks)

## 1. Verdict

**PASS.** All 11 fixes verified applied correctly. No new blockers introduced by the revision. Ship to executor.

## 2. Per-finding verification

| # | Status | Evidence |
|---|--------|----------|
| F1 | Verified | Task 12 (lines 768-812) writes baseline 2-epoch run and saves epoch_0/epoch_1 losses to JSON. Task 13 (lines 814-868) deletes epoch_001.pt, rewinds latest.txt to epoch_000.pt, purges sys.modules, then calls run_tiny_epoch(resume=True, max_epochs=2) -- actually trains epoch 1 from the restored RNG. Task 14 line 884 assigns baseline_epoch1 = baseline["epoch_1"], line 886 compares to resumed_epoch1 (the 8 floats from tiny_resumed_epoch1_losses.json). This is baseline-epoch-1-from-fresh-2-epoch vs. resumed-epoch-1, not fresh-vs-fresh. run_tiny_epoch signature line 230-231 carries max_epochs: int = 1. Verify automated (line 755) asserts max_epochs in sig.parameters. |
| F2 | Verified | Task ordering 7 (mount + stage, lines 557-592) precedes Task 8 (visualization, lines 594-627). Task 8 dependencies line 625 lists Tasks 4-7 explicitly. Task 7 dependencies line 590 is only Task 3. Executable from a fresh Colab session. |
| F3 | Verified | Task 1 (lines 327-349) = clone + .gitignore only, one paste-back (ls -la + git status). Task 2 (lines 351-377) = author notes, one paste-back (wc -l + first 50 lines). Each is a single runnable unit. |
| F4 | Verified | Original Task 7 split into Task 7 (mount+stage, 557-592), Task 9 (RNG capture/restore + cudnn config, 629-668), Task 10 (hash + atomic_save + load_latest + prune, 670-718). Each has one cohesive paste-back. |
| F5 | Verified | Subsumed by F1 restructure (Tasks 12/13/14) -- confirmed in revision log line 1043. |
| F6 | Verified -- schemas concrete | Interfaces lines 200-223 pin metrics_history: list[dict] with typed keys {epoch:int, train_loss_mean:float, train_loss_per_batch:list[float], val_loss_mean:float} and config_repr: dict (JSON-safe mirror of the hashed config, same keys as D12). D10 (line 295) mirrors this. Task 11 line 743 explicitly appends one entry per epoch matching the schema. Not vague. |
| F7 | Verified -- consistent | Interfaces line 236-241: {epoch, train_loss_per_batch, val_loss_mean, checkpoint_path, config_hash} (5 keys, no rng_capture). Task 11 acceptance line 760 enumerates the same 5 keys. Task 12 line 785 prints result_baseline; Task 13 line 851 accesses result_resumed["train_loss_per_batch"] (key exists in contract). No contradictions. |
| F8 | Verified | _envinit.py content pinned at interfaces lines 172-176 and Task 3 action lines 398-404 (exactly 2 lines + comment). Task 3 acceptance line 420 enforces Cell 0 first line is "from backend.training.aqa.harness import _envinit". Task 9 action line 647 + acceptance line 663 contain runtime assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" -- fails fast if import order is wrong. Belt and braces. |
| F9 | Verified | Task 3 creates figures/.gitkeep (lines 391, 406, 421). Defensive Path(out).parent.mkdir(parents=True, exist_ok=True) present in Task 8 line 609, Task 12 line 794, Task 13 line 850, Task 14 line 890. Created early (Task 3) so all downstream visualization tasks can write to the dir. |
| F10 | Verified | Frontmatter files_modified lines 7-23 includes _envinit.py (15), figures/.gitkeep (19), tiny_baseline_2epoch_losses.json (21), tiny_resumed_epoch1_losses.json (22), tiny_train_loss.png (23), decoded_batch.png (20). Complete. |
| F11 | Verified -- Option A applied | Interfaces lines 129-134 declares decode_clip is window-bounded with start_pts/end_pts. Task 5 action lines 472-482 implements the windowed-decode block with re-based indices and inline F11 (RESEARCH section 5) marker. Task 5 acceptance line 505 includes a memory-budget assertion: peak working-set increment < 200 MB on a >= 400-frame clip with indices=[0, 200, 403]. R12 (line 1015) carries the mitigation deadline. |
| F12 | Verified | determinism_checklist block lines 308-323 lists exactly 8 numbered preconditions covering env var (F8), global seed, cudnn deterministic + benchmark off, use_deterministic_algorithms, num_workers=0, sample-indices generator, spatial-train generator, RNG restore order. Referenced from R6 line 1009 and from Task 14 failure path line 886. |
| F13 | Verified -- no action needed | Marked as such in revision log line 1051. |

## 3. New issues introduced by the revision

None blocking. Two minor observations (informational, not blockers):

- **G1 (info, not blocking):** Task 6 dependency line 553 reads "Task 4, 5 (splits and transforms must exist); Task 7 informs nothing here". Task 6 acceptance line 547 reads "In Colab (after Task 7 stages videos), next(iter(loaders[train])) returns...". The DAG is correct (Task 6 implements the dataset class; Task 7 staging is needed only when you actually iterate it). The acceptance hedge "after Task 7" is honest. Leaving the dependency at Tasks 4, 5 is technically right (implementation-only) but a reader could be confused. The plan is consistent; no change needed.
- **G2 (info, not blocking):** Task 10 dependency line 716 reads "Task 9 (colab.py is the same module, but RNG primitives must land first)". Strictly speaking Task 10 only requires colab.py to exist; Task 9 ordering is a soft preference for cleaner diffs. The plan executes correctly either way. No change needed.

Cross-reference consistency confirmed:

- Risk register (lines 999-1018) renumbered: R6 cites Task 14 (not old Task 10) for the bitwise assertion; R3 cites Task 10 (not old Task 7); R12/R13 are new and tied to F11/F8.
- Verification mapping (lines 974-985) renumbered: criterion 3 lists Tasks 9, 10, 11, 12, 13, 14 -- matches the F1 restructure.
- visualization_deliverables (lines 991-994): Figure 1 built by Task 8, Figure 2 by Task 14 -- both new task IDs are right.
- success_criteria line 1027 references Task 14 bitwise-equality assertion correctly.
- files_modified frontmatter has no duplicates; the new JSONs + _envinit.py + .gitkeep are listed.

## 4. Goal coverage final check (ROADMAP success criterion 3)

Walking the proof:

1. **Baseline 2-epoch fresh produces an epoch-1 trajectory.** Task 12: run_tiny_epoch(seed=42, resume=False, max_epochs=2) writes epoch_000.pt and epoch_001.pt; the latter metrics_history[1]["train_loss_per_batch"] is 8 floats saved to tiny_baseline_2epoch_losses.json under key epoch_1. Confirmed (lines 783-796).
2. **Delete epoch-1 ckpt + restore latest.txt to epoch-0.** Task 13 cell 1 (lines 826-828): os.remove(run_dir + "epoch_001.pt"); open(latest.txt, "w").write("epoch_000.pt"). Confirmed.
3. **Module purge.** Task 13 cell 2 (lines 834-836): sys.modules purge + re-import. Confirmed.
4. **Resume.** Task 13 cell 3 (line 842): run_tiny_epoch(seed=42, resume=True, max_epochs=2) -- start_epoch=1 (loaded from epoch_000.pt payload["epoch"] + 1 per Task 11 line 738), restores RNG via restore_rng_state(payload["rng_state"]), runs the epoch-1 train loop. Per-batch losses saved to tiny_resumed_epoch1_losses.json. Confirmed.
5. **Bitwise equal to fresh epoch 1.** Task 14 line 886 + automated verify line 909: assert b["epoch_1"] == r. Confirmed.

**Answer: YES.** ROADMAP Phase 2 success criterion 3 is now proven by Tasks 12-14. The F1 fix is complete and correct.

## 5. Final recommendation

**Ship.** All 11 findings addressed. The Iteration-1 verdict (REVISE) is now PASS. Hand off to /gsd:execute-phase 02. Risk register and verification mapping are internally consistent. Determinism checklist is the single audit list Phase 3 will inherit. No re-plan, no additional revision pass required.

---
