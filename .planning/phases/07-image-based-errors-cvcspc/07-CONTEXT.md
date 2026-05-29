# Phase 7: Image-Based Errors (CVCSPC) — Context

**Gathered:** 2026-05-30
**Status:** Ready for planning

<domain>
## Phase Boundary

**This phase delivers:** Shallow-Squat (squat-depth) form-error detection — a binary image
classifier (2D ResNet-18) on the 3,611 labeled `crops_unaligned` JPEGs, evaluated by F1 on the
official split and compared to the published number (paper CVCSPC 0.869). A supervised ImageNet
baseline ships first; then **faithful CVCSPC pose-contrastive SSL** (pretrained on the unlabeled
Back-Squat set — feasible standalone, research-verified) is fine-tuned and the lift measured.
This adds the **image modality** and the **2nd method (CVCSPC)** to the thesis alongside the
video MD-SSL work (Squat, OHP).

**This phase does NOT deliver:**
- **BarbellRow Lumbar/Torso (IMG-03) — CANCELLED** by the user (compute/time). Descoped from the
  milestone, not deferred-but-pending. (CVCSPC does NOT need BarbellRow — see D2.)
- API/serving integration — descoped per CONTEXT D1 (the form-correction frontend is cancelled;
  same stance as OHP). IMG-01/02 are satisfied by trained/evaluated models + paper comparison +
  visualization pack.
- The video pipeline (R(2+1)D-18 / MD-SSL) — that's Phases 2–6; this phase is 2D images.

**Requirements satisfied:** IMG-01 (CVCSPC ResNet-18 image pipeline — baseline + self-supervised),
IMG-02 (Shallow-Squat detector; F1 on the official split). **IMG-03 is descoped** (see above).
</domain>

<decisions>
## Implementation Decisions (Locked)

### D1 — Scope: Shallow-Squat only; BarbellRow cancelled
**Choice:** Phase 7 covers only Shallow-Squat. IMG-03 (BarbellRow Lumbar/Torso) is descoped.
**Why:** User cancelled BarbellRow for compute/time. Milestone form-correction scope becomes
Squat + OHP (video) + Shallow-Squat (image) — every remaining requirement is actually delivered.

### D2 — Method: ImageNet ResNet-18 baseline FIRST, then FAITHFUL CVCSPC SSL [RESOLVED post-research 2026-05-30]
**Choice:** Ship the supervised **ImageNet-pretrained ResNet-18** detector first (guaranteed
IMG-02 result), THEN reproduce the paper's **faithful CVCSPC** pose-contrastive SSL and fine-tune
from it, measuring the lift. F1 on the official split vs paper CVCSPC 0.869.
**Why (RESOLVED — research corrected an earlier wrong assumption):** CVCSPC is **feasible as a
Squat-standalone** pretext task — the paper pretrained on the BackSquat Unlabeled set ALONE, NOT
multi-exercise. Verified in `Fitness-AQA-Code/.../pose_contrastive_learning/self_supervised_learning/dataloader.py`:
the triplet contrasts **phase-matched frames across two *Squat* reps** (anchor/positive at the same
bar-trajectory phase from two different reps; negative at a phase ≥`ssl_contrastive_phase_gap`
away), using the per-rep **bar-trajectory** as the phase signal. The "two roots" are train/val of
ONE exercise, not two exercises. The 4,970 unlabeled Squat videos + 4,970 bar-trajectory JSONs
(the Phase-4 SSL set) are exactly what's needed — already on disk/Drive. **BarbellRow is NOT
required.** (Correction: an earlier read mistakenly concluded CVCSPC was inherently cross-exercise;
the research + dataloader.py refuted that.)
**Loss (CITED `train_test.py:64-68`):** 3-term contrastive — `-log( e^-d(a,p) / (e^-d(a,p) +
e^-d(a,n) + e^-d(p,n)) )` on L2-normalized features (a 3-term denominator; the paper's Eq.1 shows
2 terms — implement the CODE version, note the deviation in FINDINGS).
**Pretext mechanics (CITED `dataloader.py`):** normalize each rep's bar-traj to [0,1]→×360°
("phase"); anchor+positive = the frame nearest a shared random phase in two different reps;
negative = a frame ≥ phase-gap away; augmentation = **masking only** (top 40–50% blackout, p≈0.5;
all other augs commented out in the shipped code). Phase-gap anneals 30°→ over epochs.

### D2a — CVCSPC requires a frame-extraction pre-step
**Choice:** A Step-0 cell decodes the 4,970 unlabeled Squat mp4s → per-clip JPEG frame dirs
(`{ssl_frames_dir}/{video_id}/*.jpg`) on Colab local disk before SSL (the dataloader indexes
extracted frames, not mp4s). Skip-existing/disconnect-safe; ~10–20 min one-time on L4 (use the
cv2 decode path — tv 0.26 removed read_video). The bar-trajectory JSONs give the phase signal.
**Why:** CVCSPC contrasts still frames selected by phase — it needs frame-addressable images.

### D3 — Image input pipeline: new `datasets/shallow_squat.py`, ImageNet norm
**Choice:** New `backend/training/aqa/datasets/shallow_squat.py` — load JPEG crop → 224² (the
code uses resize→CenterCrop) → **ImageNet mean/std** (`[0.485,0.456,0.406]/[0.229,0.224,0.225]`,
CITED `dataloader.py:96`) → train aug → tensor. NO frame sampling / video decode (still image).
**Why:** ResNet-18 is ImageNet-pretrained; ImageNet stats + 224² is the native contract (matches
the official code exactly).
**Rejected:** video `transforms.py` at T=1 + Kinetics norm — wrong stats, useless temporal axis.

### D4 — Rigor: multi-seed ensemble, matching the video phases
**Choice:** Multi-seed ResNet-18 ensemble (3-seed default; 2-seed acceptable) + mean-of-sigmoids
+ val-tuned threshold — same protocol as Squat (3-seed) / OHP (2-seed).
**Why:** Consistent methodology across all three exercises; ResNet-18 trains in minutes.

### D5 — Class balance + supervised loss
**Choice:** Shallow-Squat ~44% positive (1584/3611) — near-balanced. Class-weighted
`BCEWithLogitsLoss` with a train-derived `pos_weight`, but it matters far less than Squat KIE's 14%.
**Why:** Near-balanced → weighting is a minor correction, not load-bearing.

### D6 — Carry-forward contracts (reuse, do NOT redefine)
**Choice:** `eval/metrics.py` (f1/pr_auc/threshold_sweep/confusion) reused UNCHANGED. Colab harness
pattern (Drive stage, atomic per-epoch checkpoint, `latest.txt` resume, `map_location="cpu"`,
disconnect-safe) carried from Phases 2–6. `.ipynb` delivered; one cell at a time + paste-back;
**tqdm/in-loop progress on any cell >2 min**; `results.pkl` → deliverable notebook pack
(EDA + training/eval, planner sets final count) + FINDINGS + figures, matching Squat/OHP. New
lightweight ResNet-18 image trainer mirrors the supervised_train checkpoint/resume/early-stop/
threshold-sweep contracts + the `dataset_cls`/`checkpoint_phase` seam pattern.
**Why:** Proven; keeps the deliverable consistent for the supervisor.

### D7 — macro/F1 arithmetic computed in code; no fabricated numbers
**Choice:** Every metric computed in code from real scores; comparison uses the CITED paper CVCSPC
number (0.869). No hand-typed headline values (the Phase-3 precedent the user caught). Implement
the 3-term loss as in the code; note the code-vs-paper deviation honestly.
**Why:** [[feedback_ai_correctness]].

### Claude's Discretion / delegated
- Projection-head dims (the shipped `linear_layers.py` is absent — research [ASSUMED] 512→128→128);
  ablate if SSL triplet-accuracy is poor.
- Final notebook count (2 vs more) — planner decides from what's produced.
</decisions>

<canonical_refs>
## Canonical References

### Phase scope + requirements
- `.planning/ROADMAP.md` — Phase 7 goal + success criteria (criterion 3 "served through the API"
  DESCOPED per D1, mirroring OHP).
- `.planning/REQUIREMENTS.md` — IMG-01, IMG-02 (IMG-03 marked descoped).
- `~/.claude/plans/what-pushed-me-back-iridescent-pnueli.md` — master plan; Phase 7; CVCSPC 0.869.
- `.planning/phases/07-image-based-errors-cvcspc/07-RESEARCH.md` — the method reconstruction +
  feasibility verdict + Validation Architecture (researcher MUST-read; planner consumes).

### The method (researcher read; planner MUST honor)
- `Fitness-AQA-Code/Code_Release/pose_contrastive_learning/self_supervised_learning/dataloader.py`
  — the phase-contrastive triplet construction + masking aug + ImageNet norm.
- `.../self_supervised_learning/train_test.py` — the 3-term loss (lines 64-68), Adam recipe, the
  AP<AN triplet-accuracy eval, the phase-gap anneal.
- Parmar et al. (ECCV 2022, arXiv:2202.14019) — CVCSPC §; Shallow-Squat 0.869; SimSiam 0.829.

### Data (verified locally 2026-05-30)
- Labeled crops: `Fitness-AQA/...-3-001/.../Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/`
  — `images.zip` (3,738 `crops_unaligned/{id}.jpg`), `labels_shallow_depth.json` ({id:0/1}, 3611,
  43.9% pos), `splits/{train,val,test}_ids.json` (2542/529/540). ID = `{video}_{rep}_{frame}`.
- SSL set (for CVCSPC pretrain): the unlabeled Back-Squat videos (4,970) + `bar_trajectories`
  JSONs — the Phase-4 SSL set (`stage_unlabeled_squat_videos`); frame-extracted in D2a's Step 0.

### Reuse (do NOT redefine)
- `backend/training/aqa/eval/metrics.py`; `backend/training/aqa/harness/colab.py`;
  `backend/training/aqa/datasets/squat.py` (structural template, minus video decode).
- `docs/notebooks/05-08_ohp_*` + `docs/eval/FINDINGS_OHP.md` — deliverable-pack pattern.

### Working-agreement memory
[[feedback_interactive_execution]], [[feedback_one_cell_at_a_time_strict]],
[[feedback_deliver_colab_as_ipynb]], [[feedback_notebook_disconnect_safe]],
[[feedback_progress_bars_required]], [[feedback_heavy_training_new_notebook]],
[[feedback_ai_correctness]], [[feedback_no_ai_comments]], [[reference_pytorch_persistent_workers]],
[[reference_colab_module_reload_after_git_pull]], [[reference_best_pt_metrics_history_is_stale]],
[[project_supervisor_visualizations]].
</canonical_refs>

<code_context>
## Existing Code Insights

- **`eval/metrics.py`** — binary F1/PR-AUC/threshold-sweep/confusion; reused as-is.
- **`harness/colab.py`** — staging + atomic checkpoint + cpu-resume + RNG + prune; needs an
  image-zip staging fn + the unlabeled-Squat frame-extraction step (D2a).
- **`datasets/squat.py`** — split-index→records→`__getitem__`→loaders + train `pos_weight`
  transfers; swap video decode for single-image load + ImageNet transform.
- **`transforms.py` cv2 path** — the tv 0.26 read_video removal is video-only; images use
  `torchvision.io.read_image`/PIL (the official code uses PIL) — confirm at Step 0.
- **trainers** (`supervised_train.py`/`md_*`) are R(2+1)D-18-specific — new ResNet-18 image
  trainer mirrors their checkpoint/resume/early-stop/threshold-sweep + `dataset_cls`/
  `checkpoint_phase` seam.

## Integration Points
Net-new image dataset + image trainer + CVCSPC SSL trainer under `backend/training/aqa/`. Nothing
touches the FastAPI runtime (serving descoped). `results.pkl` feeds the deliverable pack.
</code_context>

<specifics>
## Specific Ideas

- Compute LIGHT for the supervised side (ResNet-18 on ~3.7k crops = minutes). CVCSPC adds a
  one-time frame-extraction (~10–20 min) + SSL pretrain (minutes-to-low-hours on the 4,970 set).
- Comparison: paper CVCSPC Shallow-Squat **0.869**, SimSiam 0.829. Our ImageNet baseline is the
  new control (no published supervised-image row) → narrative = baseline → CVCSPC → paper.
- Loss deviation: code is 3-term, paper Eq.1 is 2-term — implement code, document in FINDINGS.
- Projection head ([ASSUMED] 512→128→128) — `linear_layers.py` absent; ablatable.
</specifics>

<deferred>
## Deferred Ideas
- **BarbellRow Lumbar/Torso (IMG-03)** — cancelled this milestone (compute/time).
- **CVCSPC cross-exercise transfer** — moot without BarbellRow; not needed (CVCSPC runs on Squat alone).
</deferred>

<scope_fence>
## Scope Fence

**IN (Phase 7):**
- `datasets/shallow_squat.py` (image loader, ImageNet norm, official splits)
- ResNet-18 ImageNet **supervised** baseline trainer + multi-seed ensemble + val-tuned threshold
- **Faithful CVCSPC SSL** pretrain on the unlabeled Back-Squat set (frame-extract Step-0 +
  phase-contrastive triplet + 3-term loss) → fine-tune → measure the lift
- F1 on the official Shallow-Squat test split vs paper CVCSPC 0.869
- `results.pkl` + deliverable notebook pack + FINDINGS + figures

**OUT:**
- BarbellRow (IMG-03), any video-error work, API/serving, the polished Flutter UI
- Adapted/non-CVCSPC SSL (faithful CVCSPC is feasible, so no substitute method needed)
</scope_fence>

<risk_summary>
## Risk Summary

- **CVCSPC reconstruction [MEDIUM]** — 2 of the shipped files are present (dataloader, train_test);
  `CVC_SPC.py`/`opts`/`linear_layers.py`/`dataloader_eval.py` are absent. Loss + triplet +
  recipe are recoverable; projection head [ASSUMED]. Mitigate: the SSL triplet-accuracy eval
  (AP<AN, CITED train_test.py:130-134) is a cheap convergence monitor before fine-tune.
- **Frame-extraction cost/disk [LOW]** — 4,970 mp4s → frames on /content; skip-existing + tqdm.
- **Code-vs-paper loss deviation [LOW]** — 3-term vs 2-term; implement code, document.
- **Image-load API on tv 0.26 [LOW]** — PIL/`read_image`; confirm Step 0.
- **Low** — supervised baseline, data availability (verified), eval reuse.
</risk_summary>

---
*Phase: 07-image-based-errors-cvcspc*
*Context gathered: 2026-05-30 via /gsd:discuss-phase; D2 corrected post-research (CVCSPC feasible standalone)*
