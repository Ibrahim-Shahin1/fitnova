# Phase 7: Image-Based Errors (CVCSPC) — Context

**Gathered:** 2026-05-30
**Status:** Ready for planning

<domain>
## Phase Boundary

**This phase delivers:** Shallow-Squat (squat-depth) form-error detection — a binary image
classifier (2D ResNet-18) on the 3,611 labeled `crops_unaligned` JPEGs, evaluated by F1 on
the official split and compared to the published number (paper CVCSPC 0.869). A supervised
ImageNet baseline ships first; CVCSPC pose-contrastive SSL is then attempted as the lift
(research-gated). This adds the **image modality** and a **2nd method (CVCSPC)** to the
thesis alongside the video MD-SSL work (Squat, OHP).

**This phase does NOT deliver:**
- **BarbellRow Lumbar/Torso (IMG-03) — CANCELLED** by the user (compute/time). Descoped from
  the milestone, not deferred-but-pending.
- API/serving integration — descoped per CONTEXT D1 (the form-correction frontend is
  cancelled; same stance as OHP). IMG-02 is satisfied by trained/evaluated model + paper
  comparison + visualization pack.
- The video pipeline (R(2+1)D-18 / MD-SSL) — that's Phases 2–6; this phase is 2D images.

**Requirements satisfied:** IMG-01 (CVCSPC ResNet-18 image pipeline — baseline + SSL),
IMG-02 (Shallow-Squat detector; F1 on the official split). **IMG-03 is descoped** (see above).
</domain>

<decisions>
## Implementation Decisions (Locked)

### D1 — Scope: Shallow-Squat only; BarbellRow cancelled
**Choice:** Phase 7 covers only Shallow-Squat. IMG-03 (BarbellRow Lumbar/Torso) is descoped
from the milestone.
**Why:** User cancelled BarbellRow for compute/time. Milestone form-correction scope becomes
Squat + OHP (video) + Shallow-Squat (image) — every remaining requirement is actually delivered.

### D2 — Method: ImageNet ResNet-18 baseline FIRST, then research-gated CVCSPC SSL
**Choice:** Train the supervised ImageNet-pretrained ResNet-18 baseline first (the guaranteed
paper-comparable result). THEN the researcher resolves exactly how CVCSPC pose-contrastive SSL
applies to Shallow-Squat-alone and attempts the lift. If the SSL source can't be assembled
standalone, baseline + an honest write-up still satisfies IMG-02/IMG-01.
**Why:** CVCSPC faithfulness for Shallow-Squat-alone is genuinely uncertain — the paper's image
SSL may have leaned on the multi-exercise image set (which included the now-cancelled
BarbellRow), and the official `pose_contrastive_learning/` code is WIP. Baseline-first de-risks
the new method and guarantees a result.
**Rejected:** "Faithful CVCSPC SSL as a hard bar regardless" — too much risk/time on an
uncertain data assembly when the baseline already gives a defensible image-modality result.

### D3 — Image input pipeline: new `datasets/shallow_squat.py`, ImageNet norm
**Choice:** New module `backend/training/aqa/datasets/shallow_squat.py` — load the JPEG crop →
resize/center-crop to 224² → **ImageNet mean/std** normalization (ResNet-18 ImageNet weights'
native stats) → train-time aug (random crop / flip — confirm against the paper's image augs).
NO frame sampling / video decode (it is a still image).
**Why:** ResNet-18 is ImageNet-pretrained, so ImageNet stats + 224² is the native contract.
**Rejected:** Reusing the video `transforms.py` at T=1 + Kinetics norm — wrong norm stats for
an ImageNet backbone and a useless temporal axis.

### D4 — Rigor: multi-seed ensemble, matching the video phases
**Choice:** Multi-seed ResNet-18 ensemble (3-seed default; 2-seed acceptable) + mean-of-sigmoids
+ val-tuned per-error threshold — the same protocol as Squat (3-seed) / OHP (2-seed).
**Why:** Consistent methodology across all three exercises for the defense; ResNet-18 trains in
minutes so the ensemble is cheap.

### D5 — Class balance handling
**Choice:** Shallow-Squat is ~44% positive (1584/3611) — near-balanced. Use class-weighted
`BCEWithLogitsLoss` (or CE) with a train-derived weight, but expect it to matter far less than
Squat KIE's 14%. Confirm the exact loss against the paper / CVCSPC code.
**Why:** Near-balanced → weighting is a minor correction, not load-bearing.

### D6 — Carry-forward contracts (reuse, do NOT redefine)
**Choice:** `eval/metrics.py` (f1/pr_auc/threshold_sweep/confusion) reused UNCHANGED. The Colab
harness pattern (Drive stage, atomic per-epoch checkpoint, `latest.txt` resume,
`map_location="cpu"`, disconnect-safe) carried from Phases 2–6. `.ipynb` delivered; one cell at
a time + paste-back; **tqdm/in-loop progress on any cell >2 min**; `results.pkl` → 2-notebook
deliverable pack (EDA + training/eval) + FINDINGS + figures, matching Squat/OHP.
**Why:** Proven, and keeps the deliverable consistent for the supervisor.

### D7 — macro/F1 arithmetic computed in code; no fabricated numbers
**Choice:** Every metric computed in code from real scores; comparison uses the CITED paper
CVCSPC number (0.869). No hand-typed headline values (the Phase-3 precedent the user caught).
**Why:** [[feedback_ai_correctness]].
</decisions>

<canonical_refs>
## Canonical References

### Phase scope + requirements
- `.planning/ROADMAP.md` — Phase 7 goal + success criteria (criterion 3 "served through the
  API" is DESCOPED per D1, mirroring OHP).
- `.planning/REQUIREMENTS.md` — IMG-01, IMG-02 (IMG-03 to be marked descoped).
- `~/.claude/plans/what-pushed-me-back-iridescent-pnueli.md` — master plan; Phase 7 paragraph;
  comparison target (Shallow-Squat CVCSPC 0.869).

### The new method (researcher MUST read)
- `Fitness-AQA-Code/Code_Release/pose_contrastive_learning/self_supervised_learning/CVCSPC.py` —
  the CVCSPC SSL implementation (WIP but substantive, unlike the empty motion_disentanglement).
- `.../pose_contrastive_learning/self_supervised_learning/{opts.py,train_test.py}` — recipe + loop.
- `.../pose_contrastive_learning/README.md` — method overview.
- Parmar et al. (ECCV 2022, arXiv:2202.14019) — the CVCSPC §; the image-error eval protocol.

### Data (verified locally 2026-05-30)
- `Fitness-AQA/...-3-001/.../Squat/Labeled_Dataset/Shallow_Squat_Error_Dataset/` —
  `images.zip` (3,738 `crops_unaligned/{id}.jpg`), `labels_shallow_depth.json` (flat dict
  {id: 0/1}, 3611 entries, 43.9% pos), `splits/{train,val,test}_ids.json` (2542/529/540, flat
  ID lists). ID format `{video}_{rep}_{frame}`.

### Reuse (do NOT redefine)
- `backend/training/aqa/eval/metrics.py` — reused unchanged.
- `backend/training/aqa/harness/colab.py` — staging/checkpoint/resume primitives (add an
  image-zip staging analog).
- `backend/training/aqa/datasets/squat.py` — structural template for `shallow_squat.py` (minus
  video decode/sampling).
- `docs/notebooks/05-08_ohp_*` + `docs/eval/FINDINGS_OHP.md` — the deliverable-pack pattern to mirror.

### Working-agreement memory
[[feedback_interactive_execution]], [[feedback_one_cell_at_a_time_strict]],
[[feedback_deliver_colab_as_ipynb]], [[feedback_notebook_disconnect_safe]],
[[feedback_progress_bars_required]], [[feedback_heavy_training_new_notebook]],
[[feedback_ai_correctness]], [[feedback_no_ai_comments]], [[reference_pytorch_persistent_workers]],
[[reference_colab_module_reload_after_git_pull]], [[project_supervisor_visualizations]].
</canonical_refs>

<code_context>
## Existing Code Insights

- **`eval/metrics.py`** — pure F1/PR-AUC/threshold-sweep/confusion, sklearn-backed; reused as-is.
- **`harness/colab.py`** — `stage_*` (zip copy+extract+resume), `atomic_save_checkpoint`,
  `load_latest_checkpoint(map_location='cpu')`, RNG capture/restore, `prune_checkpoints`. Needs an
  image-set staging function (mirror `stage_ohp_videos` for `images.zip`).
- **`datasets/squat.py`** — class structure (split index → records → `__getitem__` → loaders +
  train-derived `pos_weight`) transfers; drop `decode_clip`/`uniform_sample_indices`, swap to a
  single-image PIL/torchvision load + ImageNet transform.
- **`transforms.py`** — the cv2/torchvision decode robustness (tv 0.26 removed read_video) is
  video-only; image loading uses `torchvision.io.read_image` / PIL — confirm availability on Colab.
- **The trainer harnesses** (`supervised_train.py`, `md_*`) are R(2+1)D-18-specific — Phase 7
  needs a new lightweight image trainer (ResNet-18), but can mirror their checkpoint/resume/
  early-stop/threshold-sweep contracts.

## Integration Points
Net-new image dataset + trainer under `backend/training/aqa/`. Nothing touches the FastAPI
runtime (serving descoped). `results.pkl` is the input to the Plan-N deliverable pack.
</code_context>

<specifics>
## Specific Ideas

- Compute is LIGHT (ResNet-18 on ~3.7k small crops trains in minutes) — unlike the multi-hour
  video phases. The risk is method (CVCSPC reconstruction), not compute. A fresh L4 notebook is
  fine but the heavy-training caution is lower here.
- Comparison target: paper CVCSPC Shallow-Squat **0.869** (master plan). Baseline (ImageNet
  fine-tune) has no separately published image-Kinetics-equivalent row — the baseline is our
  control, the CVCSPC number is the SSL target (same framing as OHP's baseline→MD story).
- Deliverable: a 2-notebook pack is likely sufficient here (EDA + training/eval) vs the video
  4-notebook pack, since there's no separate data-pipeline/SSL-curve story unless CVCSPC lands —
  planner decides final notebook count from what's produced.
</specifics>

<deferred>
## Deferred Ideas
- **BarbellRow Lumbar/Torso (IMG-03)** — cancelled this milestone (compute/time). Revisit only
  if a future milestone wants the full 3-exercise image coverage.
- **CVCSPC cross-exercise transfer** (the paper's BarbellRow-via-transfer trick) — moot without
  BarbellRow.
</deferred>

<scope_fence>
## Scope Fence

**IN (Phase 7):**
- `datasets/shallow_squat.py` (image loader, ImageNet norm, official splits)
- ResNet-18 ImageNet baseline trainer + multi-seed ensemble + val-tuned threshold
- CVCSPC SSL attempt (research-gated, after the baseline)
- F1 on the official Shallow-Squat test split vs paper CVCSPC 0.869
- `results.pkl` + deliverable notebook pack + FINDINGS + figures

**OUT:**
- BarbellRow (IMG-03), any video work, API/serving, the polished Flutter UI
</scope_fence>

<risk_summary>
## Risk Summary

- **CVCSPC SSL faithfulness for Shallow-Squat-alone [HIGH uncertainty]** → mitigated by D2
  (baseline-first, research-gated; baseline alone satisfies IMG-02). The researcher must
  determine from `CVCSPC.py` + the paper whether the image SSL needs the multi-exercise set.
- **Official CVCSPC code is WIP** → it's substantive (real `CVCSPC.py`/`opts.py`/`train_test.py`,
  unlike the empty motion_disentanglement), but may have broken imports (Phase-2 precedent).
  Reconstruct from paper + code; record divergences.
- **Image-load API on Colab** (tv 0.26 churn) → confirm `torchvision.io.read_image` / PIL at
  Step 0; low risk (images, not the removed video decode).
- **Low** — compute, data availability (verified present), eval reuse.
</risk_summary>

---
*Phase: 07-image-based-errors-cvcspc*
*Context gathered: 2026-05-30 via /gsd:discuss-phase*
