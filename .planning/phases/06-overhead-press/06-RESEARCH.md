# Phase 06: Overhead Press — Research

**Researched:** 2026-05-27
**Domain:** OHP form-error detection; MD-SSL recipe transfer; OHP trajectory format (raw BBox); code reuse boundary
**Confidence:** HIGH for paper-cited numbers (Table 4 read directly from PDF); HIGH for trajectory format (inspected 20 files + ReadMe.md.docx); HIGH for code coupling analysis (source read directly); MEDIUM for OHP-specific half-cycle sign inference (observation-based, gated probe still required at execution)

**Source files read:**
- `.planning/phases/06-overhead-press/06-CONTEXT.md` — locked decisions D1–D7, specifics
- `.planning/REQUIREMENTS.md` — OHP-01
- `.planning/STATE.md`
- `.planning/phases/04-squat-motion-disentangling-ssl/04-RESEARCH.md` — full MD-SSL recipe
- `.planning/phases/04-squat-motion-disentangling-ssl/04-02-SUMMARY.md` — probe results: argmin confirmed for Squat; v2 strong-aug fix
- `.planning/phases/03-squat-supervised-baseline/03-RESEARCH.md` — supervised baseline recipe
- `backend/training/aqa/datasets/squat.py`, `squat_ssl.py`, `splits.py`, `transforms.py`
- `backend/training/aqa/harness/supervised_train.py`, `md_finetune.py`, `colab.py`
- `Fitness-AQA/Domain Knowledge-Informed Self-Supervised.pdf` — Table 4 (p.13), §4, §5
- `Fitness-AQA/.../OHP/Unlabeled_Dataset/ReadMe.md.docx` — trajectory format documentation
- `Fitness-AQA/.../OHP/Unlabeled_Dataset/bar_trajectories_raw.zip` — 20 files inspected directly
- `Fitness-AQA/.../OHP/Labeled_Dataset/Labels/error_elbows.json`, `error_knees.json`
- `Fitness-AQA/.../OHP/Labeled_Dataset/Splits/{train,val,test}_keys.json`

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D1 — Scope: model + eval + visualization pack only; NO backend API integration.**
  Backend integration for OHP is descoped. Deliverable = OHP checkpoints + results.pkl + F1-vs-paper + 4-notebook pack + FINDINGS.md.

- **D2 — Detector structure: shared backbone, joint 2-output multi-label head.**
  R(2+1)D-18 MD-SSL backbone → `Dropout(0.2)+Linear(512,2)`, two sigmoid outputs (Elbows, Knees). Joint head is justified by shared MD-SSL representation + efficiency, not co-occurrence (OHP errors are near-independent — only 85/407 Elbow+ are also Knees+). BCEWithLogitsLoss multi-label handles independent labels.

- **D3 — Method and recipe: faithful reuse of P3 baseline + P4 MD-SSL pipeline.**
  Preprocessing, loss, supervised baseline, fine-tune, MD-SSL pretrain, ensemble, threshold sweep, TTA eval, eval harness — all carried from Phase 3/4 recipes unchanged.

- **D4 — OHP data contract [VERIFIED in discussion].**
  Labeled 2,260 clips — train 1,582 / val 339 / test 339. Labels under `OHP/Labeled_Dataset/Labels/error_{elbows,knees}.json`. Unlabeled 5,490 clips + 5,490 trajectory JSONs (`bar_trajectories_raw/{clip_id}.json`).

- **D5 — Trajectory format: gated probe before SSL [approach LOCKED; mechanics RESEARCHER/EXECUTE-RESOLVE].**
  Exact format + half-cycle sign confirmed by offline inspection this session (see §3 below) but the gated `checkpoint:human-verify` probe on 2–5 real Colab clips remains mandatory before finalizing `ohp_ssl._load_trajectory` and before the SSL GPU burn.

- **D6 — Code organization: new ohp.py/ohp_ssl.py from squat templates; reuse agnostic modules [LOCKED].**
  New: `datasets/ohp.py` (OHPElbowsKneesDataset), `datasets/ohp_ssl.py`. Reuse unchanged: `transforms.py`, `ssl_augs.py`, `eval/metrics.py`, `eval/ensemble.py`, `eval/tta.py`, training logic in harness modules. Needs OHP parameterization: `splits.py`, `colab.py` staging, `supervised_train._build_dataloaders` (and `md_finetune` reusing it).

- **D7 — Working agreement: interactive one-cell-at-a-time; disconnect-safe; notebooks as .ipynb; visualizations first-class.**

### Claude's Discretion
- Factoring of the shared split/staging/dataloader code (parameterize vs OHP variant).
- SSL/fine-tune batch sizing re-estimate on L4.
- Whether `ohp.py`'s `build_loaders` is on the training path (it is a convenience factory only — trainers build DataLoaders directly).

### Deferred Ideas (OUT OF SCOPE)
- OHP backend inference API integration.
- Image-based errors (Shallow-Squat, BarbellRow) — Phase 7.
- Cross-method ensemble + full 3-exercise comparison pack — Phase 8.
- TTA as a shipped default (evaluated per Phase-4 precedent, not adopted).
- 5-seed ensemble / weak-aug ablation re-run.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| OHP-01 | OHP Elbow/Knees baseline and MD-SSL models trained; F1 per error on the official OHP splits | §1 (paper numbers confirmed), §2 (MD method confirmed), §3 (trajectory format resolved offline), §4 (recipe transfer confirmed), §5 (code reuse boundary), §6 (validation architecture) |
</phase_requirements>

---

## Summary

Phase 6 is a faithful re-run of the proven Squat pipeline (Phases 2–4) on OHP data. This research session confirms five things that matter for planning:

**1. Paper numbers (Table 4, p.13):** The paper reports OHP results in Table 4 — **Ours MD: Elbow 0.4552, Knees 0.8452**. There is no Kinetics supervised-baseline row for OHP in the paper (unlike Squat which had a Kinetics row in Table 2). The CONTEXT's comparison targets (Knees 0.845 / Elbow 0.455) are confirmed correct (Table 4 rounds to 0.8452/0.4552). The paper also shows TemporalXform achieves Elbow 0.4138 / Knees 0.8416, which is the best non-MD general video baseline and a useful secondary comparison point.

**2. Same MD method, no OHP-specific deviations:** The paper applies the identical MD-SSL approach to OHP as to Squat. Section 5.2 (p.13) presents OHP results alongside Squat in the same evaluation framework, using the same method description. No OHP-specific changes are documented anywhere in §3–§5.

**3. Trajectory format — CRITICAL DELTA vs Squat:** OHP trajectories are **NOT flat y-lists**. They are per-frame bounding-box arrays in the format `[frame_0, frame_1, ...]` where each frame contains 3 detection regions: `[region_0, region_1, region_2]`, and each region is a list of `[x1, y1, x2, y2, confidence]` bounding boxes. The ReadMe.md.docx states: "Trajectory file format: .json files contain bounding boxes (BBoxes) with the following format: Bar_bbox = x1, y1, x2, y2, confidence_val. To get the height/amplitude of the center of the bounding box, you can use the following formula: y_center = (y1+y2)/2." This means `ohp_ssl._load_trajectory` must parse BBoxes and compute y_center, unlike `squat_ssl._load_trajectory` which loads a direct flat float list. Region 0 (wide horizontal bbox, full image width) consistently tracks the barbell. ~40% of files have some empty region-0 frames (detection gaps) requiring interpolation.

**4. Recipe transfers with one adjustment — BCEWithLogitsLoss handles independent multi-label correctly.** BCEWithLogitsLoss computes independent sigmoid cross-entropy per output, so the joint head for independent errors is mathematically sound. The expected behavior is that each head converges independently on its own signal; no co-occurrence lift is expected (and none should be claimed). Batch sizing and VRAM math carry from Phase 3/4 unchanged; both OHP error classes are moderate-minority (pos_weight 2.89/1.92) vs Squat's rare+majority (6.10/0.45), so the loss is better-balanced per-head for OHP.

**5. Code reuse boundary confirms D6 exactly:** `supervised_train._build_dataloaders` hardcodes `SquatKIEKFEDataset` (line 222–224). `md_finetune.run_md_finetune_epoch` imports `_build_dataloaders` from `supervised_train` and also imports `SquatKIEKFEDataset` directly (line 37). Both must be parameterized or wrapped to accept `OHPElbowsKneesDataset`. The cleanest factoring is a `dataset_cls` parameter on `_build_dataloaders` (plus `videos_root` rename to a generic name).

**Primary recommendation:** Proceed with the Phase 3/4 recipe for OHP, with one key engineering task: implement `ohp_ssl._load_trajectory` as BBox-parser (not flat-list loader), gate it behind a Colab probe cell, and parameterize `_build_dataloaders` to accept `OHPElbowsKneesDataset`. Everything else reuses verbatim.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| OHP BBox trajectory load + y_center extraction | SSL dataset (CPU, `datasets/ohp_ssl.py`) | ReadMe formula | Per-frame BBox → scalar y_center (differs from Squat) |
| Half-cycle split (press-up / lower) | SSL dataset (CPU) | scipy.ndimage | Same `split_half_cycles` from `squat_ssl.py`, sign TBD at probe |
| Triplet construction (anchor/pos/neg) | SSL dataset (CPU) | — | Same structure as `SquatSSLDataset` |
| SSL augmentations | SSL dataset (CPU workers) | `ssl_augs.py` | REUSE UNCHANGED — exercise-agnostic |
| MD forward + triplet loss | SSL trainer (GPU, `harness/md_pretrain.py`) | — | REUSE UNCHANGED — no exercise coupling |
| Linear-probe convergence monitor | SSL trainer | `eval/metrics.py` | REUSE UNCHANGED |
| OHP labeled dataset (Elbows/Knees) | Supervised dataset (`datasets/ohp.py`) | `splits.py` (OHP variant) | label fields renamed kie/kfe → elbows/knees |
| Supervised/fine-tune training | `harness/supervised_train.py` + `md_finetune.py` | Parameterize `_build_dataloaders` | Need `dataset_cls` param to accept OHP dataset |
| OHP staging (labeled + unlabeled + trajectory zip) | `harness/colab.py` (new OHP staging fns) | Drive FUSE | New `stage_ohp_videos`, `stage_unlabeled_ohp_videos` |
| Threshold sweep / F1 / PR-AUC / ensemble | `eval/metrics.py`, `eval/ensemble.py` | — | REUSE UNCHANGED — score-array operations |
| Visualization / results.pkl | Notebook (matplotlib) + scripts | `docs/figures/` | OHP analog of Squat pack |

---

## §1 — Paper's Published OHP Numbers (Confirm / Verify)

### Table 4 from Fitness-AQA paper (p.13) — VERIFIED by direct PDF read

The paper presents OHP results in Table 4 (§5.2 "Case Study 2: In-The-Wild Conditions"):

| Feature extraction model | Modality | Elbow Err. F1 | Knees Err. F1 |
|--------------------------|----------|----------------|----------------|
| OpenPose-TDM [2,27] | 2D Pose | 0.4265 | 0.7131 |
| SimSiam [5] | Image | 0.4145 | 0.5301 |
| Ours CVCSPC | Image | 0.4522 | 0.7203 |
| TemporalXform [17] | Video | 0.4138 | 0.8416 |
| **Ours MD** | **Video** | **0.4552** | **0.8452** |

[CITED: Parmar et al. ECCV 2022, Table 4, p.13]

**Key findings:**

1. **No Kinetics supervised-baseline row for OHP.** Table 4 does NOT include a "Kinetics [20]" row (unlike Table 2 for BackSquat which has Kinetics KIE=0.2970, KFE=0.8184). This is consistent with CONTEXT D3's claim: "No paper Kinetics (supervised) baseline exists for OHP." **CONFIRMED.**

2. **Paper targets (CONTEXT specifics lines 60–61):** CONTEXT says "OHP Knees 0.845 / Elbow 0.455." The table shows MD: Knees 0.8452 / Elbow 0.4552. Both match to 3 significant figures. **CONFIRMED.**

3. **MD outperforms all other methods on both errors.** TemporalXform (0.4138 / 0.8416) is the closest general video method. CVCSPC image method reaches 0.4522 / 0.7203.

4. **Elbow is notably harder than Knees for all methods.** The best Elbow F1 is only 0.4552 (MD), while Knees reaches 0.8452. This mirrors Squat's asymmetry (KIE hard / KFE easy). Defense framing: Elbow is an upper-body temporal error (subtle); Knees is a lower-body geometric error (more detectable).

5. **No "Kinetics + MD" combined row for OHP.** For Squat, the paper shows MD standalone (0.419/0.834) but also MD+CVCSPC ensemble (0.526/0.847). For OHP there is no such combination row in Table 4. The phase-end headline is MD standalone vs. our supervised baseline (our measured SSL-lift control).

**Headline comparison structure for the defense presentation:**

```
Our OHP pipeline:
  Supervised baseline (ours)     → [measured on our run]
  MD-SSL fine-tune, 3-seed ens.  → [measured on our run]
  Paper Ours MD                  → Elbow 0.4552 / Knees 0.8452  [CITED: Table 4]
  Paper TemporalXform (best gen) → Elbow 0.4138 / Knees 0.8416  [CITED: Table 4]
  Paper OpenPose-TDM             → Elbow 0.4265 / Knees 0.7131  [CITED: Table 4]
```

---

## §2 — Same MD-SSL Method for OHP? (Confirm / Verify)

### Verdict: Identical MD method, no OHP-specific deviations. CITED.

The paper's §5.2 introduces OHP evaluation with: "Further, we evaluated and compared approaches on a different exercise—OverheadPress." [CITED: §5.2 p.13]. All methods in Table 4 use the same approach descriptions as Table 2 (Squat). The MD model for OHP uses the same R(2+1)D-18 backbone, same triplet distance-ratio loss (Eq. 1), same half-cycle contrastive pretext on barbell trajectories, same "We randomly applied strong augmentations" instruction (§5 p.9 describes a single unified implementation).

**No OHP-specific method differences documented anywhere in §3–§5.**

The unlabeled set size (5,490 OHP vs 4,970 Squat) is close enough that wall-time estimates carry directly. Phase 4's 12–24h SSL estimate for Squat's 4,970 clips scales to ~13–25h for 5,490 OHP clips (a 10% longer run). This is within the same session-budget envelope.

**The only OHP-specific input to MD is the trajectory format (§3 below) — same algorithm, different input parsing.**

---

## §3 — Trajectory Format (D5 Offline Pre-Characterization)

### What the ReadMe.md.docx says [CITED: OHP/Unlabeled_Dataset/ReadMe.md.docx, word/document.xml]

> "Trajectory file format: .json files contain bounding boxes (BBoxes) with the following format: Bar_bbox = x1, y1, x2, y2, confidence_val. To get the height/amplitude of the center of the bounding box, you can use the following formula: y_center = (y1+y2)/2"

### What offline inspection of 20 sample files reveals [VERIFIED by direct archive inspection]

- **Archive layout:** `bar_trajectories_raw/{clip_id}.json`, flat directory, 5,490 files, 1:1 with unlabeled clip IDs. [VERIFIED: `zipfile.ZipFile` listing, 5490 JSON members]
- **Top-level structure:** each JSON is a Python `list` where each element is one video frame. The outer list length equals the number of video frames (length varies 43–427+; median ~106 across first 100 files). [VERIFIED: inspected 20 files]
- **Per-frame structure:** each frame element is a list of 3 region groups — `[region_0, region_1, region_2]`. Always exactly 3 regions per frame. [VERIFIED: `set(len(frame) for frame in data) == {3}` on tested files]
- **Per-region structure:** each region is a list of 0 or 1 bounding boxes. Each bounding box is `[x1, y1, x2, y2, confidence_val]` (5 integers, confidence in 0–100). An empty region `[]` means no detection for that region in that frame.
- **Region 0 = the barbell.** Region 0 has the widest horizontal bounding boxes (width ~400px out of 416px image width, height ~160px) — characteristic of a barbell plate visible across the full frame. Region 0 is the correct source for y_center. [VERIFIED: Region 0 bbox in 11681_3.json frame 0: x1=0, y1=138, x2=416, y2=302, width=416, height=164]
- **Region 0 completeness:** approximately 40% of files have some empty region-0 frames. Survey of 100 files: 40 had at least one empty frame; total 8.9% of frames across those 100 files were empty in region 0. This requires interpolation (same strategy as Squat's NaN interpolation in `squat_ssl._load_trajectory`). [VERIFIED: 100-file survey]
- **No in-file NaN values.** The BBox values are integers; "missing" is represented as an empty list `[]`, not `null`/`NaN`. The Squat loader's `np.isnan()` check should be replaced with an empty-list check for OHP. [VERIFIED: JSON content inspection]
- **Trajectory length vs frame count:** lengths range 43–427 (median ~106 for first 100 files). Assuming 1:1 with video frames (same as Squat — the Phase 4 probe found 1:1 for Squat). **This MUST be confirmed in the Colab probe — it cannot be determined offline without the videos.**

### Critical comparison to Squat trajectories

| Property | Squat (`bar_trajectories_raw/`) | OHP (`bar_trajectories_raw/`) |
|----------|--------------------------------|-------------------------------|
| Format | Flat `list[float]` of y-centers | Nested BBox list per frame |
| Parser in dataset | `np.asarray(json.load(...))` directly | Must compute `(y1+y2)/2` from region 0 |
| Missing data | `null`/NaN values → `np.isnan()` | Empty region list `[]` → custom fill |
| Smoothing needed | Yes (YOLO jitter on y-centers) | Yes (same reason) |
| Archive nesting | Nested subdirs (rglob required) | Flat directory (glob works) |
| 1:1 traj/frame | CONFIRMED (Phase 4 probe) | ASSUMED until Colab probe |

[VERIFIED: Squat format from 04-RESEARCH.md §8 + 04-02-SUMMARY.md probe notes; OHP format from direct inspection]

### Half-cycle sign for OHP (`bottom_is_argmax` parameter)

For OHP, the barbell starts at shoulder height (moderate y), presses UP overhead (y decreases in image coordinates = smaller y value), then lowers back down (y increases back to starting value). So:
- **Press-up phase (first half-cycle):** y_center DECREASES from shoulder to overhead.
- **Lower phase (second half-cycle):** y_center INCREASES back to shoulder.
- **Turning point = overhead = argMIN(y)** (the barbell is highest in the image = lowest y value).

This is **OPPOSITE to Squat**, where the squat bottom = argMIN(y) = deepest position was confirmed as the turning point with `bottom_is_argmax=False` in the Phase 4 probe. For OHP, the turning point is also argMIN (overhead = barbell at top of image = lowest y in image coordinates). **Both Squat and OHP use `bottom_is_argmax=False` (argMIN).** [ASSUMED — inferred from OHP exercise mechanics + image-coordinate convention; MUST be confirmed in the Colab probe by visual inspection of 5 clips]

Observation supporting this inference: in file `11681_3.json` (144 frames), region-0 y_center starts at 220.0 (shoulder height), decreases to 55.5 at frame 47 (overhead), then returns. The argMIN is at frame 47, correctly in the middle of the clip — consistent with a single-rep OHP. This suggests `bottom_is_argmax=False` is the correct setting, but the probe should confirm the visual alignment.

**Planner action:** gate `ohp_ssl._load_trajectory` finalization behind a `checkpoint:human-verify` probe on 2–5 OHP clips that: (1) inspects the raw BBox data, (2) computes y_center for region 0, (3) plots the trajectory with the argMIN split point marked, (4) visually confirms descent/ascent alignment. This mirrors Phase 4 Wave-1 Task 1.

### Recommended `_load_trajectory` implementation for `ohp_ssl.py` [ASSUMED — structure confirmed offline; 1:1 traj/frame needs Colab verification]

```python
def _load_trajectory(self, clip_id: str) -> np.ndarray:
    """Load OHP barbell trajectory from BBox JSON and return y_center array.

    OHP format (ReadMe.md.docx + offline inspection): each JSON is a list of
    frames; each frame = [region_0, region_1, region_2]; each region is a list
    of 0 or 1 bboxes [x1, y1, x2, y2, conf]. Region 0 = barbell (wide horizontal).
    y_center = (y1 + y2) / 2 per ReadMe formula.

    Missing frames (empty region-0 list) are linearly interpolated, analogous to
    squat_ssl's NaN interpolation.

    Returns:
        1-D float64 ndarray of length == number of frames in the clip.
        1:1 with video frames (confirmed at Colab probe — [ASSUMED until then]).
    """
    path = self._traj_paths[clip_id]
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)  # list of frames

    y = np.empty(len(raw), dtype=float)
    for i, frame in enumerate(raw):
        region_0 = frame[0] if frame else []
        if region_0:
            x1, y1, x2, y2, conf = region_0[0]
            y[i] = (y1 + y2) / 2.0
        else:
            y[i] = np.nan  # sentinel for empty detection; interpolated below

    nan_mask = np.isnan(y)
    if nan_mask.any():
        valid = ~nan_mask
        if int(valid.sum()) < 2:
            raise ValueError(f"trajectory {clip_id}: <2 non-NaN samples ({int(valid.sum())})")
        idx = np.arange(len(y))
        y[nan_mask] = np.interp(idx[nan_mask], idx[valid], y[valid])
    return y
```

---

## §4 — Recipe Transfer (Confirm with OHP-Specific Notes)

### Supervised Baseline (Phase 3 recipe)

**Transfers UNCHANGED:**
- R(2+1)D-18, Kinetics-V1 init, full end-to-end fine-tune. [CITED: Parmar §5 p.9]
- `BCEWithLogitsLoss(pos_weight=dataset.pos_weight)`, train-derived per-error.
- Adam lr=1e-4, weight_decay=0, CosineAnnealingLR, 50ep/8-patience. [Phase 3 D6]
- 32-frame uniform sampling, 128→112² random/center crop, Kinetics norm, flip OFF. [Phase 2 D1]
- `nn.Linear(512, 2)` head (no dropout for baseline). [Phase 3 D3]
- Threshold sweep on val, per-error independent, applied to test. [Phase 3 D7]

**OHP-specific note (D2 caveat):** `pos_weight = [w_elbows, w_knees]` where w_elbows ≈ 2.887, w_knees ≈ 1.924 (both >1, both moderate-minority). For Squat the weights were [6.10, 0.45] — a rare+majority pair. For OHP both weights are moderate, so the loss is better balanced per-head; the model should not have the Squat problem of `BCEWithLogitsLoss` with w<1 (which makes the majority KFE class essentially easier to get high F1 on). This is a favorable difference. [VERIFIED: label counts confirmed in discussion and re-confirmed by local label file inspection; error_elbows.json: 576/2260 = 25.5%; error_knees.json: 777/2260 = 34.4%]

**BCEWithLogitsLoss for independent multi-label:** The loss decomposes as `L = w_e * BCE(logit_e, y_e) + w_k * BCE(logit_k, y_k)` (sum of independent binary losses, each with its own pos_weight). Independent errors are handled correctly — the gradient for each head depends only on its own label, not the other head. No special handling is needed for near-independent labels. [CITED: PyTorch BCEWithLogitsLoss docs — each element of pos_weight scales the positive class loss for one output independently; ASSUMED that the multi-label factored form is correct, which is standard for all multi-label binary classification]

**No published OHP Kinetics baseline for comparison.** The phase-end report compares: (a) our supervised baseline [measured]; (b) our MD-SSL 3-seed ensemble [measured]; (c) paper Ours MD [0.4552 Elbow / 0.8452 Knees, CITED Table 4]. The measured SSL lift = (b) - (a). Defense framing: unlike Squat where the paper provided a comparable Kinetics row for our baseline to be checked against, for OHP the paper provides no such row — our supervised baseline is the only control.

### MD-SSL Pretrain (Phase 4 recipe)

**Transfers UNCHANGED:**
- Backbone init: Kinetics-V1 (same as fine-tune starting point for SSL, not the labeled-set fine-tune init). [CITED: §5 p.9]
- Loss: triplet distance-ratio (Eq. 1), 3-term denominator, squared L2. [CITED: Parmar Eq. 1 + official train_test.py:68]
- Optimizer: AdamW lr=1e-4, wd=1e-4. [Phase 4 D2/D3]
- Epochs: 20 baseline, linear-probe-gated. Linear-probe cadence every 5 epochs, freeze backbone, 32-frame val transform. [Phase 4 D2, RESEARCH §5/§6]
- Batch: 8 on L4 (paper-exact is 5; 8 recommended for throughput). [Phase 4 RESEARCH §5]
- Strong augs (v2 set — Phase 4 established this is required): translation/zoom/blur/channel-swap independently per branch; rotation OFF by default. [Phase 4 D2-augs, confirmed by 04-02-SUMMARY.md]
- Backbone saved as `backbone.pt` (backbone_state_dict only, discards projector). [Phase 4 D7]
- Wall-time anchor: 5,490 OHP clips ≈ 10% more than 4,970 Squat clips → ~15–26h (Squat was 12–24h at ep5 convergence). Expect same pattern: linear-probe peaks around ep3–7 then declines/collapses; use linear-probe-best backbone. [ASSUMED based on Phase 4 precedent; measure epoch-0 timing on Colab]

**Half-cycle split for OHP:** same `split_half_cycles` function from `squat_ssl.py`, but applied to y_center-extracted OHP trajectories. Sign parameter `bottom_is_argmax=False` (argMIN, same as Squat). [ASSUMED from §3 inference; confirmed at probe]

**Collapse guard:** same `_embedding_collapse_metrics` (effective_rank on 256-anchor probe set). Phase 4 found that ep5 convergence + strong augs maintained eff_rank ~10; the same is expected for OHP. If collapse recurs (eff_rank <3), the v2 strong-aug set is already wired — no additional remediation needed beyond verifying augs are active.

### Fine-Tune (Phase 4 recipe)

**Transfers UNCHANGED:**
- AdamW wd=1e-4, dropout=0.2 head-only, lr=1e-4, 50ep/8-patience cosine. [Phase 4 D3]
- `build_finetune_model`: loads MD backbone (strict=False), attaches `nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 2))`. **This is exercise-agnostic — no OHP modification needed.** [VERIFIED: md_finetune.build_finetune_model reads only backbone weights; head is always a fresh 2-output linear]
- D6 overfit monitor: val/train BCE ratio >10x before epoch 10 → abort seed + bump reg. [Phase 4 D6]
- 3-seed ensemble (42/1337/7), one shared SSL backbone, mean of sigmoids, single per-head threshold tuned on ensemble val scores. [Phase 4 D4]

**Dataset injection difference (the net-new engineering work):** `md_finetune.run_md_finetune_epoch` calls `supervised_train._build_dataloaders` which constructs `SquatKIEKFEDataset` directly. For OHP this must instead construct `OHPElbowsKneesDataset`. The cleanest fix is to add a `dataset_cls` kwarg to `_build_dataloaders` defaulting to `SquatKIEKFEDataset` (backward-compatible; OHP callers pass `OHPElbowsKneesDataset`). See §5.

### VRAM Re-Estimate for OHP (L4)

Phase 3 measured 15.22 GB at batch 16, 32-frame, 112², fp32. OHP has the same model + same input shape → same VRAM. Batch 16 still fits L4 24GB comfortably. No change needed. [ASSUMED: OHP video resolution is similar to Squat (both from social media, similar aspect ratios); VRAM probe cell in the first training notebook should confirm]

### pos_weight Handling for the Two Moderate-Minority Classes

For Squat, KFE had w<1 (majority class) which means the loss penalized false positives less than false negatives for KFE. For OHP, both w_elbows ≈ 2.89 and w_knees ≈ 1.92 are >1, so both heads penalize false negatives more. This should make convergence more symmetric. No code change needed — the same `_compute_pos_weight` logic from `squat.py` (reused in `ohp.py`) handles this automatically. [VERIFIED: formula is (N-pos)/max(pos,1); with 25.5% and 34.4% positive, both >1]

---

## §5 — Code Reuse Boundary (D6 Confirmation Against Actual Code)

### Exercise-Agnostic (REUSE UNCHANGED — confirmed by code read)

| Module | Evidence of agnosticism | Status |
|--------|------------------------|--------|
| `datasets/transforms.py` | `KINETICS_MEAN/STD`, `uniform_sample_indices`, `decode_clip`, `spatial_train`, `spatial_val` — no Squat/exercise coupling anywhere in file | CONFIRMED agnostic |
| `datasets/ssl_augs.py` | Operates on `[T,3,H,W]` tensors; no exercise-specific logic | CONFIRMED agnostic |
| `eval/metrics.py` | Operates on score arrays + label arrays; exercise-agnostic | CONFIRMED agnostic |
| `eval/ensemble.py` | `aggregate_sigmoid_mean` over score arrays | CONFIRMED agnostic |
| `eval/tta.py` | Per-clip augmented forward passes | CONFIRMED agnostic |
| `harness/md_pretrain.py` | `run_md_pretrain_epoch`, `_linear_probe`, `_embedding_collapse_metrics`, `ProjectionHead`, `md_triplet_loss`, `build_md_model` — operate on batch dicts from the SSL dataset without exercise coupling | CONFIRMED agnostic; accepts any SSL dataset that yields `{anchor,positive,negative}` dicts |
| `harness/supervised_train.py` — epoch loop, `build_model`, `_val_pass`, `_set_global_seed`, `seed_worker` | These functions are exercise-agnostic; only `_build_dataloaders` is Squat-coupled | MOSTLY agnostic; one function needs parameterization |
| `harness/colab.py` — `mount_drive`, atomic checkpoint, RNG, `prune_checkpoints` | No exercise-specific coupling | CONFIRMED agnostic |

### Squat-Hardcoded (NEEDS OHP VARIANT — confirmed by code read)

**1. `datasets/splits.py`** [VERIFIED: direct read]

```python
# Squat-hardcoded constants:
_EXPECTED_COUNTS = {"train": 1136, "val": 243, "test": 244}
_LABELS_SUBDIR   = "Fitness-AQA_dataset_release/Squat/Labeled_Dataset/Labels"
_SPLITS_SUBDIR   = "Fitness-AQA_dataset_release/Squat/Labeled_Dataset/Splits"
_LABEL_KIE_FILENAME = "error_knees_inward.json"
_LABEL_KFE_FILENAME = "error_knees_forward.json"
# ClipRecord has label_kie / label_kfe fields
```

OHP needs: different counts (1582/339/339), different subdir paths, different label filenames (`error_elbows.json`, `error_knees.json`), different `ClipRecord` fields (`label_elbows`/`label_knees`).

**Recommended factoring (Claude's Discretion):** Add an OHP-specific loader in `splits.py` OR create `ohp_splits.py` as a sibling. The simpler, safer approach for Phase 6 is to add an `index_ohp(split, *, drive_root, videos_root)` function directly in `splits.py` (alongside `index(...)`) with OHP-specific constants and a new `OHPClipRecord` dataclass. This avoids modifying the existing Squat `index` function (backward-safe) while keeping both loaders in one place. The interval-to-binary `_label_value` function is GENERIC and reused verbatim.

**2. `harness/supervised_train._build_dataloaders`** [VERIFIED: lines 187–260, constructs `SquatKIEKFEDataset` directly at lines 222–224]

The function imports `from backend.training.aqa.datasets.squat import SquatKIEKFEDataset` at the top of the file. Every call to `_build_dataloaders` uses `SquatKIEKFEDataset`.

**Recommended factoring:** Add a `dataset_cls` kwarg defaulting to `SquatKIEKFEDataset`:

```python
def _build_dataloaders(
    seed: int, config: SupervisedConfig, drive_root: str, videos_root: str,
    *, dataset_cls=SquatKIEKFEDataset,  # NEW kwarg -- backward-compatible
) -> dict[str, DataLoader]:
```

OHP callers pass `dataset_cls=OHPElbowsKneesDataset`. No other changes to the function body (it uses `common_kwargs` which are the shared kwargs `num_frames`, `crop_size`, `train_jitter_frames`, `seed` — all applicable to OHP). The `videos_root` default in `FinetuneConfig` (`"/content/squat_videos"`) becomes an OHP notebook override.

**3. `harness/md_finetune.run_md_finetune_epoch`** [VERIFIED: imports `_build_dataloaders` from `supervised_train`; line 193 calls it; line 37 also imports `SquatKIEKFEDataset` directly]

`SquatKIEKFEDataset` is imported at line 37 but used only in the type annotation comment "The labeled `SquatKIEKFEDataset` pipeline"; the actual construction is through `_build_dataloaders`. Once `_build_dataloaders` accepts `dataset_cls`, `md_finetune` needs only a `dataset_cls` kwarg forwarded to `_build_dataloaders`. No other changes.

**4. `harness/colab.py` staging** [VERIFIED: `stage_squat_videos` (line 317) and `stage_unlabeled_squat_videos` (line 449) are Squat-pathed]

`stage_squat_videos` hardcodes `Fitness-AQA_dataset_release/Squat/Labeled_Dataset/videos.zip` as the source path and `/content/squat_videos` as the local destination. `stage_unlabeled_squat_videos` hardcodes the unlabeled zip path and traj zip path.

**Required additions:** `stage_ohp_videos` and `stage_unlabeled_ohp_videos` (new functions in `colab.py`). These parallel `stage_squat_videos` / `stage_unlabeled_squat_videos` exactly, with OHP-specific paths:

- Labeled videos: `Fitness-AQA_dataset_release/OHP/Labeled_Dataset/videos.zip` → `/content/ohp_videos/` (2,367 mp4s expected)
- Unlabeled videos: `{drive_root}/...Fitness-AQA_dataset_release-20260518T073812Z-3-002/.../OHP/Unlabeled_Dataset/videos.zip` → `/content/ohp_unlabeled_videos/` (5,490 expected)
- Trajectory zip: `{drive_root}/...3-001/.../OHP/Unlabeled_Dataset/bar_trajectories_raw.zip` → `/content/ohp_trajectories/` (JSON per-member extract)

OHP archive split (two release folders) is the main staging complexity — the trajectory zip is in the `-3-001` folder while the unlabeled videos are in the `-3-002` folder. Staging must accept both source Drive paths as parameters (or derive them from `drive_root`).

**Note:** The trajectory archive uses `zipfile.extractall` (like the Squat unlabeled staging) — NOT the mp4-specific `_extract_with_resume_and_progress`. The OHP trajectory JSON members are under `bar_trajectories_raw/{clip_id}.json` (flat one-level, confirmed offline). `extractall` produces this layout directly.

### Summary of Net-New Code for Phase 6

| File | Action | Lines (estimate) |
|------|--------|-----------------|
| `datasets/ohp.py` | NEW — mirror `squat.py`: `OHPElbowsKneesDataset` + `build_loaders` + `pos_weight` | ~120 lines |
| `datasets/ohp_ssl.py` | NEW — mirror `squat_ssl.py`: `OHPSSLDataset` with BBox `_load_trajectory` | ~180 lines |
| `datasets/splits.py` | ADD `OHPClipRecord` dataclass + `index_ohp()` function | ~60 lines (additive, no changes to existing) |
| `harness/colab.py` | ADD `stage_ohp_videos()` + `stage_unlabeled_ohp_videos()` | ~100 lines (additive) |
| `harness/supervised_train.py` | ADD `dataset_cls` kwarg to `_build_dataloaders` | ~5 lines |
| `harness/md_finetune.py` | ADD `dataset_cls` kwarg to `run_md_finetune_epoch`, forwarded to `_build_dataloaders` | ~5 lines |
| Colab notebooks (OHP analog) | NEW — 3 training notebooks + 4 analysis notebooks | Per Phase 4 pattern |

---

## §6 — Carry-Forward Landmines (D7 — Must NOT Re-Surface)

All Phase 4 D7 landmines apply to Phase 6:

1. **`persistent_workers=True`** on every `DataLoader` with `num_workers > 0`. [VERIFIED: `supervised_train._build_dataloaders` already implements this; `ohp.py` must do the same in `build_loaders`]
2. **`map_location='cpu'`** for `load_latest_checkpoint`. [VERIFIED: colab.py already enforces this]
3. **`metrics_history` from `latest.txt`-pointed checkpoint**, NOT `best.pt`. [VERIFIED: colab.py enforces this; confirmed by `[[reference_best_pt_metrics_history_is_stale]]` memory]
4. **Deliver notebooks as `.py` + `.ipynb` paired.** [From `[[feedback_deliver_colab_as_ipynb]]`]
5. **One runnable unit at a time, paste-back before next cell.** [From `[[feedback_one_cell_at_a_time_strict]]`]
6. **Restart runtime / `importlib.reload` after `git pull`.** [From `[[reference_colab_module_reload_after_git_pull]]`]
7. **`update_latest=False`** when writing `backbone.pt` (must NOT clobber `latest.txt` of the SSL run). [Phase 4 D7 — confirmed in `colab.py`'s `atomic_save_checkpoint` signature]
8. **OHP-specific:** trajectory extraction uses `zipfile.extractall` (JSON members), NOT the mp4-only `_extract_with_resume_and_progress`.

---

## Standard Stack

### Core (all carried from Phase 2–4, no new installations)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | Colab-pinned (~2.x) | Training, loss, autograd | Project standard (CLAUDE.md) |
| torchvision | Colab-pinned | `r2plus1d_18` + Kinetics-V1 weights | Paper uses R(2+1)D-18 [CITED §5] |
| scipy | ≥1.11.0 | `gaussian_filter1d` for trajectory smoothing | Already in requirements.txt; Phase 4 carry |
| scikit-learn | ≥1.3.0 | F1, PR-AUC, threshold sweep via `eval/metrics.py` | Phase 3/4 contract |
| numpy | ≥1.24.0 | Trajectory arrays, score aggregation | Phase 2 contract |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| matplotlib | Colab | Visualizations, figures | All visualization tasks |
| tqdm | Colab | Progress bars (never-silent cells) | All training/eval loops |
| PyAV (`av`) | Colab Step 0 | FFmpeg backend for `read_video` decode | All training notebooks |

**Installation:** Zero new pip dependencies. All libraries are Colab-provided or already in `backend/requirements.txt`. Step 0 dep-version probe (carried from Phase 3) handles supply-chain guard.

## Package Legitimacy Audit

> This phase installs **no new external packages**. All libraries used are either Colab-provided or already pinned in `backend/requirements.txt`. The Phase 3/4 supply-chain guard (Step 0 dep-version probe) is carried forward unchanged.

| Package | Registry | Disposition |
|---------|----------|-------------|
| (none added) | — | N/A — no new installs this phase |

---

## Architecture Patterns

### System Architecture Diagram

```
OHP Unlabeled Dataset (5,490 clips + BBox trajectories)
        |
        v
[ohp_ssl._load_trajectory: BBox JSON -> y_center array]
        |
        v
[split_half_cycles (argMIN): descent_idx / ascent_idx]
        |
        +---- anchor (augmented descent)
        |
        +---- positive (re-augmented descent)
        |
        +---- negative (augmented ascent, optionally reversed)
        |
        v
[md_pretrain: triplet loss (Eq.1), AdamW, 20ep] --> backbone.pt (linear-probe-best)
        |
        v [3 seeds: 42, 1337, 7]
[md_finetune: OHPElbowsKneesDataset, AdamW wd=1e-4, Dropout(0.2), 50ep/8-pat]
        |
        v
[3 fine-tuned checkpoints: best.pt per seed]
        |
        v
[eval/ensemble: mean-of-sigmoids -> per-head val threshold sweep -> test F1/PR-AUC]
        |
        v
[F1 comparison: our baseline | our MD | paper Ours MD (0.4552 Elbow / 0.8452 Knees)]
        |
        v
[results.pkl + 4 OHP notebooks + FINDINGS.md + docs/figures/]
```

### Recommended Project Structure (Phase 6 additions)

```
backend/training/aqa/
├── datasets/
│   ├── ohp.py               # NEW: OHPElbowsKneesDataset + build_loaders
│   ├── ohp_ssl.py            # NEW: OHPSSLDataset (BBox _load_trajectory)
│   └── splits.py            # ADD: index_ohp() + OHPClipRecord (no edits to index())
├── harness/
│   ├── colab.py             # ADD: stage_ohp_videos, stage_unlabeled_ohp_videos
│   ├── supervised_train.py  # EDIT: dataset_cls kwarg on _build_dataloaders (5 lines)
│   └── md_finetune.py       # EDIT: dataset_cls kwarg on run_md_finetune_epoch (5 lines)
├── notebooks/
│   ├── 06_ohp_supervised_baseline.{py,ipynb}   # NEW: baseline training
│   ├── 07_ohp_md_ssl.{py,ipynb}                # NEW: SSL pretrain
│   └── 08_ohp_md_finetune.{py,ipynb}           # NEW: 3-seed fine-tune + ensemble
docs/
├── notebooks/
│   ├── 05_ohp_eda.{py,ipynb}                  # NEW: OHP EDA notebook
│   ├── 06_ohp_data_pipeline.{py,ipynb}         # NEW: staging + pipeline notebook
│   ├── 07_ohp_training.{py,ipynb}              # NEW: diagnostics notebook
│   └── 08_ohp_evaluation.{py,ipynb}            # NEW: eval + figures notebook
├── eval/
│   └── FINDINGS.md (OHP section)              # NEW: OHP findings
└── figures/ (OHP figures)                     # NEW: PNG figures
.planning/phases/06-overhead-press/
└── figures/results.pkl                        # NEW: results artifact
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| F1 per error | Custom metric | `eval/metrics.py:f1_per_error` | Already battle-tested on Squat |
| Threshold sweep | For-loop over linspace | `eval/metrics.py:best_threshold_from_val` (`precision_recall_curve`) | Exact thresholds, no discretization |
| Ensemble | Custom aggregation | `eval/ensemble.py:aggregate_sigmoid_mean` | Same math as Phase 4 |
| TTA | Custom per-clip aug | `eval/tta.py:select_tta_recipe/tta_forward` | Phase 4 verified |
| SSL triplet loss | NT-Xent or custom | `harness/md_pretrain.py:md_triplet_loss` | Paper Eq. 1, already implemented + tested |
| SSL augmentations | Custom aug set | `datasets/ssl_augs.py` | Phase 4 validated v2 set |
| Trajectory smoothing | Raw argmin on noisy BBox | `scipy.ndimage.gaussian_filter1d` + argmin | YOLO jitter makes raw extremum unreliable |

---

## Common Pitfalls

### Pitfall 1: Treating OHP Trajectories as Flat y-Lists (Like Squat)
**What goes wrong:** Copying `squat_ssl._load_trajectory` verbatim and calling `np.asarray(json.load(fh))` on an OHP file returns a 3D array of BBox data, not a 1D y-center array. The result: `split_half_cycles` receives the wrong input and crashes or silently produces meaningless indices.
**Why it happens:** The outer list structure (one element per frame) looks similar between Squat and OHP, but the inner content is completely different (float y-value vs nested BBox list).
**How to avoid:** `ohp_ssl._load_trajectory` extracts y_center from `frame[0][0]` (region 0, bbox 0) as `(y1+y2)/2` per the ReadMe formula. The Colab probe confirms this before finalizing the function.
**Warning signs:** `split_half_cycles` ValueError about array shape; `split_half_cycles` returning indices that span less than 2 frames; SSL dataset smoke test produces zero-length clips.

### Pitfall 2: Using argmax for OHP Half-Cycle Sign (Squat is argmin)
**What goes wrong:** The Phase 4 probe confirmed Squat uses `bottom_is_argmax=False` (argMIN) because the squat bottom = minimum y_center (deepest point). For OHP, the turning point is ALSO argMIN (the barbell is overhead = highest in the image = lowest y pixel value). If mistakenly set to argMAX, the split point lands at the start of the trajectory (argmax = shoulder-start position) and one half-cycle has 0 or 1 frames.
**Why it happens:** Confusion between "bottom of the rep" (squat terminology = deepest down, high y pixel) vs "turning point" (OHP = barbell at top, low y pixel). Both are argMIN of y_center, but for different physical reasons.
**How to avoid:** The Colab probe plots 5 OHP trajectories with the argMIN point marked. Visual inspection confirms it lands in the middle of the clip (mid-rep overhead position) — not at start/end.
**Warning signs:** ascent_idx or descent_idx spans only 1–2 frames; probe plot shows split at frame 0 or frame N-1.

### Pitfall 3: Forgetting Empty Region-0 Frames in OHP Trajectories
**What goes wrong:** ~40% of OHP trajectory files have detection gaps in region 0 (empty list for some frames). Attempting `frame[0][0]` on an empty frame crashes with an IndexError; missing y_center values break the smoothing and argmin.
**Why it happens:** OHP barbell detection occasionally fails (occluded by head/shoulders during the press). Squat had NaN values; OHP has empty lists — different representation of the same problem.
**How to avoid:** The `_load_trajectory` implementation writes `np.nan` for empty region-0 frames and interpolates. This mirrors squat_ssl's NaN interpolation exactly.
**Warning signs:** IndexError in `_load_trajectory`; suspiciously short or degenerate trajectory curves.

### Pitfall 4: Multi-Archive Staging (OHP Split Across Two Release Folders)
**What goes wrong:** `stage_unlabeled_ohp_videos` looks for the unlabeled videos in the same `-3-001` folder as the labeled data, but OHP unlabeled videos are in `-3-002`. Staging fails with FileNotFoundError.
**Why it happens:** The Fitness-AQA release was split into multiple archives by the dataset authors. Squat has all its data in one folder; OHP does not.
**How to avoid:** `stage_unlabeled_ohp_videos` accepts two `drive_root` parameters — one for the `-3-001` folder (trajectory zip source) and one for the `-3-002` folder (unlabeled videos source). Or, derive both paths from the parent Drive root and the known subfolder names.
**Warning signs:** FileNotFoundError on the unlabeled zip; only labeled OHP data staged correctly.

### Pitfall 5: Hardcoded videos_root in FinetuneConfig / SupervisedConfig
**What goes wrong:** `FinetuneConfig.videos_root` defaults to `"/content/squat_videos"` (line 139 of md_finetune.py). If the OHP notebook constructs `FinetuneConfig()` without overriding `videos_root`, training reads from the wrong (Squat) directory and fails or trains on Squat clips.
**Why it happens:** The default was set for Phase 4 convenience and never needed to be overridden within the Squat phases.
**How to avoid:** OHP notebooks always pass `videos_root="/content/ohp_videos"` explicitly. Consider adding an `exercise` discriminator field to `FinetuneConfig` that sets the default — but a simpler guard is an assertion in `_build_dataloaders` that `videos_root` points to a directory with >100 mp4s.
**Warning signs:** Training loss does not decrease (wrong clips); val F1 = 0 for Elbows/Knees (model trained on Squat clips with OHP labels).

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.0.0+ (backend/requirements.txt) |
| Config file | No pytest.ini — use inline `python -m pytest` |
| Quick run command | `python -m pytest backend/training/aqa/ -x -v -k "ohp"` |
| Full suite command | `python -m pytest backend/training/aqa/ -x -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| OHP-01-a | `OHPElbowsKneesDataset` yields `(clip[3,32,112,112], labels[2])` with correct pos_weight ≈ [2.89, 1.92] | unit | `python -m pytest backend/training/aqa/datasets/test_ohp.py::test_ohp_dataset_shape -x` | Wave 0 |
| OHP-01-b | `OHPSSLDataset._load_trajectory` extracts y_center correctly from BBox JSON | unit | `python -m pytest backend/training/aqa/datasets/test_ohp.py::test_trajectory_load -x` | Wave 0 |
| OHP-01-c | `split_half_cycles` with argMIN on synthetic OHP trajectory (decreasing then increasing y) produces correct descent/ascent indices | unit (reuses squat_ssl test) | `python -m pytest backend/training/aqa/datasets/test_ohp.py::test_split_half_cycles_ohp -x` | Wave 0 |
| OHP-01-d | `index_ohp(split, drive_root, videos_root)` returns correct counts (1582/339/339) | unit (offline, uses local archive) | `python -m pytest backend/training/aqa/datasets/test_ohp.py::test_ohp_splits -x` | Wave 0 |
| OHP-01-e | OHP supervised baseline training converges: val macro-F1 improves over 5 epochs | integration (Colab) | Colab paste-back of epoch-5 val metrics | Wave 2 |
| OHP-01-f | MD-SSL pretrain: linear-probe F1 > Kinetics baseline at some epoch | integration (Colab) | Colab paste-back of linear-probe curve | Wave 3 |
| OHP-01-g | Final OHP test F1 on official split: Knees F1 reported, Elbow F1 reported | integration (Colab) | Colab paste-back of final test report | Wave 5 |

### Wave 0 Gaps

- [ ] `backend/training/aqa/datasets/test_ohp.py` — OHP dataset + trajectory + splits unit tests
- [ ] `backend/training/aqa/datasets/ohp.py` — `OHPElbowsKneesDataset` + `build_loaders`
- [ ] `backend/training/aqa/datasets/ohp_ssl.py` — `OHPSSLDataset` with BBox `_load_trajectory`
- [ ] `datasets/splits.py` additions — `OHPClipRecord` + `index_ohp()`
- [ ] `harness/supervised_train.py` — `dataset_cls` kwarg on `_build_dataloaders`
- [ ] `harness/md_finetune.py` — `dataset_cls` kwarg on `run_md_finetune_epoch`
- [ ] `harness/colab.py` — `stage_ohp_videos` + `stage_unlabeled_ohp_videos`

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Squat-only form correction | OHP added as second exercise (same pipeline) | Phase 6 | Demonstrates generalization of the MD-SSL approach |
| Raw BBox trajectories (not processed) | Compute y_center = (y1+y2)/2 from region 0 | Phase 6 (first use) | OHP-specific preprocessing needed in _load_trajectory |
| Squat data verified (Phase 1–4) | OHP data separately verified (discuss-phase + this research) | 2026-05-27 | Confirmed 2260 clips, 5490 unlabeled, correct split files |

**Deprecated/outdated:**
- "OHP trajectories are the same flat y-list format as Squat": INCORRECT. OHP trajectories are raw BBox arrays requiring y_center extraction.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | OHP trajectories are 1:1 with video frames (same as Squat) | §3 | If not 1:1, the frame-index mapping needs a linear rescaling formula; changes _load_trajectory and split_half_cycles indexing. Colab probe confirms. |
| A2 | OHP half-cycle turning point = argMIN(y_center) — `bottom_is_argmax=False` | §3 | If argMAX: one half-cycle gets 0 frames; SSL dataset crashes. Visual probe immediately reveals this. |
| A3 | MD-SSL linear-probe will converge ~ep3–7 for OHP (same as Squat) | §4 | If convergence is slower or never occurs: extend run to 60 epochs; check strong-aug wiring; inspect linear-probe curve. No plan change needed — gated monitoring. |
| A4 | OHP video resolution similar to Squat → same VRAM envelope at batch 16 | §4 | If OHP videos are higher resolution: VRAM probe cell catches this before committing to full training; batch 8 as fallback. |
| A5 | `run_name` convention `ohp_supervised_v1` / `ohp_md_pretrain_v1` | CONTEXT specifics | Purely cosmetic; no functional impact. |
| A6 | All 3 region groups per frame have exactly the same length (3) across all 5,490 files | §3 | Inspected only 20 files; outer `{3}` region count. Defensive code: access `frame[0] if len(frame) > 0 else []` rather than `frame[0]` directly. |

---

## Environment Availability

All dependencies are Colab-provided or in `backend/requirements.txt`. No new tool installations required.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.10+ | Backend code | Available | 3.10 (system) | — |
| PyTorch + CUDA | Training | Colab L4 | ~2.x | — |
| scipy | `gaussian_filter1d` | requirements.txt ≥1.11.0 | — | — |
| Google Drive mount | Colab staging | Colab | — | — |
| OHP archive (Drive) | Training | User's Drive | — | No fallback — user must confirm Drive shortcut accessible |

---

## Open Questions

1. **Trajectory 1:1 frame alignment for OHP.** Squat was confirmed 1:1 at probe time. OHP lengths suggest the same (e.g., 11681_3.json has 144 frames which is a plausible clip length at 30fps = 4.8s). But cannot confirm without reading the actual OHP mp4 duration alongside the trajectory length in Colab.
   - What we know: trajectory lengths (43–427) are consistent with OHP clip durations at 30fps; no obvious discrepancy.
   - What's unclear: exact frame count of the videos vs trajectory lengths.
   - Recommendation: Colab probe cell measures `len(trajectory)` vs `read_video_timestamps(video)` frame count for 5 clips, as in Phase 4 Wave-1.

2. **Drive path for OHP unlabeled videos (split archive).** The unlabeled videos are in `…-3-002/…/OHP/Unlabeled_Dataset/videos.zip` and trajectories in `…-3-001/…/OHP/Unlabeled_Dataset/bar_trajectories_raw.zip`. The exact Drive shortcut layout (whether the user has shortcuts to both folders or one parent) needs to be confirmed at staging time.
   - What we know: both archives are locally available (confirmed in CONTEXT); staging will follow the same pattern as Squat's `stage_unlabeled_squat_videos`.
   - Recommendation: `stage_unlabeled_ohp_videos` accepts two explicit `drive_root_3001` and `drive_root_3002` kwargs (or constructs both paths from a common parent) — user provides the Drive paths as notebook variables.

---

## Sources

### Primary (HIGH confidence)
- Parmar, Gharat, Rhodin. ECCV 2022. `Fitness-AQA/Domain Knowledge-Informed Self-Supervised.pdf` — Table 4 p.13 (OHP Elbow/Knees F1 per method), §5.2 p.13 (OHP evaluation), §5 p.9 (unified implementation details for MD)
- `OHP/Unlabeled_Dataset/ReadMe.md.docx` — trajectory format documentation ("Bar_bbox = x1, y1, x2, y2, confidence_val; y_center = (y1+y2)/2")
- Direct inspection of 20 OHP trajectory JSON files from `bar_trajectories_raw.zip` — format structure, region count, empty frame frequency, y_center computation
- `backend/training/aqa/datasets/squat.py`, `squat_ssl.py`, `splits.py`, `transforms.py` — source read directly
- `backend/training/aqa/harness/supervised_train.py`, `md_finetune.py`, `colab.py` — source read directly
- `.planning/phases/04-squat-motion-disentangling-ssl/04-02-SUMMARY.md` — argmin confirmed for Squat (Task 1 probe); v2 strong-aug fix documented

### Secondary (MEDIUM confidence)
- `.planning/phases/04-squat-motion-disentangling-ssl/04-RESEARCH.md` — full MD-SSL recipe (§1–§16)
- `.planning/phases/03-squat-supervised-baseline/03-RESEARCH.md` — supervised baseline recipe (§1–§12)

---

## Metadata

**Confidence breakdown:**
- Paper numbers (§1): HIGH — Table 4 read directly from PDF, all values extracted verbatim
- MD method transfer (§2): HIGH — paper §5.2 presents identical method for OHP as Squat
- Trajectory format (§3): HIGH for structure (BBox JSON, ReadMe confirmed, 20 files inspected); MEDIUM for half-cycle sign (mechanically inferred, probe required); ASSUMED for 1:1 traj/frame
- Recipe transfer (§4): HIGH for unchanged parts; MEDIUM for BCEWithLogitsLoss independent-label behavior (standard PyTorch behavior, not paper-cited for OHP specifically)
- Code reuse boundary (§5): HIGH — source read line by line

**Research date:** 2026-05-27
**Valid until:** 2026-07-01 (stable dataset; paper published 2022; code unchanged)
