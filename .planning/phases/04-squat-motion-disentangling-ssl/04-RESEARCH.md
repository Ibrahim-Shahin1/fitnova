# Phase 04: Squat Motion-Disentangling SSL — Research

**Researched:** 2026-05-21
**Researcher:** gsd-phase-researcher (Opus 4.7)
**Domain:** Self-supervised contrastive video representation learning; triplet distance-ratio loss; R(2+1)D-18; barbell-trajectory half-cycle splitting; PyTorch fine-tune transfer; multi-seed ensembling + TTA
**Confidence:** HIGH for paper-cited recipe values (§3.2 method, §5 implementation details, official CVCSPC code); MEDIUM for SSL-literature-grounded `[ASSUMED]` details where the paper defers to "supplementary material"; LOW for VRAM/wall-time estimates (extrapolated from Phase 3's measured 15.22 GB)

**Source files read:**
- `.planning/phases/04-squat-motion-disentangling-ssl/04-CONTEXT.md` — locked decisions D1–D9 (full)
- `.planning/phases/04-squat-motion-disentangling-ssl/04-DISCUSSION-LOG.md`
- `.planning/REQUIREMENTS.md` — SQUAT-04, SQUAT-05
- `.planning/STATE.md`; `.planning/ROADMAP.md` (Phase 4 success criteria, lines 59–67)
- `.planning/phases/03-squat-supervised-baseline/03-01-SUMMARY.md` — baseline numbers + overfit pattern + carry-forward landmines + VRAM measurement
- `.planning/phases/03-squat-supervised-baseline/03-RESEARCH.md` — structure/depth template; paper §5 quote (p.9); Table 2 numbers
- `.planning/phases/02-squat-data-pipeline-colab-harness/CODE-RELEASE-NOTES.md` — `motion_disentanglement/` is empty (1-byte README); CVCSPC idioms
- `.planning/phases/01-dataset-consolidation-eda/01-DATASET-REPORT.md` — trajectory format, clip-length distribution, traj_nan
- `Fitness-AQA/Domain Knowledge-Informed Self-Supervised.pdf` — pages 1–18 read in full (Parmar et al., ECCV 2022). §3 Method (pp.4–8), §3.2 MD (pp.6–8), §5 Experiments (pp.9–14), Table 2 (p.12)
- `Fitness-AQA-Code/Code_Release/pose_contrastive_learning/self_supervised_learning/dataloader.py` — **the only extant SSL dataloader** (CVCSPC pose variant — confirms trajectory format + normalization)
- `Fitness-AQA-Code/Code_Release/pose_contrastive_learning/self_supervised_learning/train_test.py` — **the only extant SSL loss implementation** (Eq. 1 in code, lines 60–71)
- `Fitness-AQA-Code/Code_Release/data_augmentations/image_augmentations.py` — augmentation primitives the authors actually used
- `backend/training/aqa/datasets/squat.py`, `datasets/transforms.py` — dataset + decode/sample/spatial contracts to reuse
- `backend/training/aqa/harness/supervised_train.py` — full Phase 3 trainer (the `md_finetune.py` template; checkpoint payload schema; Adam→AdamW change point)
- `backend/training/aqa/harness/colab.py` — atomic checkpoint / RNG / staging contracts
- `backend/training/aqa/eval/metrics.py` — pure F1/PR-AUC/threshold-sweep functions (ensemble + TTA feed into these)
- `backend/training/aqa/notebooks/03_squat_supervised_baseline.py` — Cell A + Step 0 conventions
- `CLAUDE.md` — backend Python conventions
- PyTorch reproducibility + L4 scaling math (extrapolated from Phase 3's measured backward VRAM)
- WebSearch (verified): Hoffer & Ailon 2015 (arXiv:1412.6622) triplet distance-ratio loss; contrastive-collapse detection metrics

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D1 — Scope = A + B + D + E; F dropped.**
  - A — Faithful MD-SSL reproduction (core contribution).
  - B — weight_decay + dropout on fine-tune (mild regularization).
  - D — 3-seed fine-tune ensemble (shared SSL pretrain).
  - E — Test-time augmentation (val-tuned).
  - **F — auxiliary trajectory-prediction head is NOT in scope. Do not research or propose it.**
- **D2 — MD-SSL recipe: faithful reproduction; mechanics RESEARCHER-RESOLVES (this document).** Locked premises: pretext = the paper's half-cycle contrast on barbell-trajectory-split reps; SimCLR-family contrastive loss is the *provisional default* but researcher confirms against §4/§3.2 or labels `[ASSUMED]`; **AdamW** for the SSL optimizer; linear-probe validation during SSL as the convergence monitor; SSL budget 12–24h on L4. Eight open questions resolved in §1–§8 below.
- **D3 — Fine-tune regularization (B): MODERATE.** Optimizer = **AdamW** (NOT Adam); **weight_decay = 1e-4**; **dropout = 0.2 head-only** (before the final `Linear(512,2)`). Epoch budget = **50, cosine annealing, 8-epoch val-macro-F1 patience early-stop** (matches Phase 3 for P3-vs-P4 parity — a deliberate deviation from paper §5's 20-epoch downstream budget; researcher notes the deviation). **Warmup OFF, fine-tune LR = 1e-4.** Researcher's one fine-tune-hyperparam job: check paper §5 for a different downstream LR (see §9).
- **D4 — Ensemble (D): 3 seeds (42/1337/7), shared SSL pretrain, mean of sigmoids, single per-error threshold tuned on ensemble val scores; per-seed F1 also reported.**
- **D5 — TTA (E): val-tuned recipe.** Candidates: temporal jitter, spatial 5-crop, horizontal flip. **Flip is OOD** (Phase 3 trained flip-OFF) → must be val-validated, not assumed-good. Aggregation order: per-seed → average its TTA copies → mean of sigmoids across seeds → val-tuned per-error threshold.
- **D6 — Overfit safeguards: 3 monitors + abort/remediation policy.** (1) Runtime train/val BCE loss ratio > 10× before epoch 10 ⇒ abort seed + bump reg (wd=5e-4, dropout=0.3); run seed 42 first, lock recipe for 1337/7. (2) Post-hoc val-test F1 gap target < 0.05. (3) Per-error KIE test F1 must NOT regress below 0.2857. No-lift policy: one documented remediation pass, else honest deviation analysis.
- **D7 — Carry-forward landmines (must NOT re-surface):** `persistent_workers=True` on every DataLoader with `num_workers>0`; `map_location='cpu'` for `load_latest_checkpoint`; `metrics_history` from `latest.txt`-pointed checkpoint not `best.pt`; paired `.py`+`.ipynb` with Cell A bootstrap; one runnable unit at a time (never pre-write task N+1); restart runtime / `importlib.reload` after `git pull`; `atomic_save_checkpoint` for `best.pt`/`backbone.pt` must NOT clobber `latest.txt` (consider explicit `update_latest=False` kwarg).
- **D8 — Visualizations first-class.** PNG to `figures/` before `plt.show()`; headline = Phase 3-vs-Phase 4 comparison bar chart.
- **D9 — Reuse contracts (do NOT redefine):** `eval/metrics.py` reused UNCHANGED; `datasets/squat.py`/`transforms.py`/`splits.py` reused for fine-tune; SSL needs a NEW unlabeled dataset class; `harness/supervised_train.py` patterns transfer to `md_finetune.py` (new module, don't bloat); `harness/colab.py` primitives reused + new unlabeled-staging analog; notebook Cell A + Step 0 pattern transfers.

### Claude's Discretion
- Ensemble aggregation method (user delegated → mean of sigmoids, settled in D4).
- Most of D2 (SSL recipe mechanics) — delegated to this research-phase by design.

### Deferred Ideas (OUT OF SCOPE)
- F — auxiliary trajectory-prediction head (revisit only in a future full-method milestone).
- Warmup ablation (only if AdamW+SSL-init shows early-epoch instability).
- Fine-tune LR ablation (1e-4 default; researcher checks §5 first — see §9).
- Regularization sweep (the designated remediation pass under D6 no-lift policy).
- 5-seed ensemble extension.
- TorchCodec decoder migration (only if `decode_clip` is the SSL throughput bottleneck — flagged in §11).
- OHP / BarbellRow / Shallow-Squat (Phases 6/7); API integration (Phase 5).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SQUAT-04 | Motion-Disentangling self-supervised pretraining on the unlabeled Squat set | §1 (half-cycle splitter), §2 (loss — triplet distance-ratio, CITED from paper Eq. 1 + official code), §3 (sampling), §4 (projection head), §5 (optimizer/scheduler/epochs/batch + VRAM), §6 (linear-probe cadence), §7 (augmentation policy), §8 (trajectory format), §10 (disconnect-safety), §12 (collapse detection) |
| SQUAT-05 | MD-pretrained model fine-tuned for KIE/KFE; F1 vs baseline and vs published numbers | §9 (fine-tune transfer recipe + LR confirmation), §13 (ensemble), §14 (TTA), §15 (Validation Architecture), §16 (comparison-chart numbers + arithmetic) |

**ROADMAP Phase 4 Success Criteria (verbatim):**
1. Motion-Disentangling SSL pretraining runs on the unlabeled Squat set and converges
2. The MD-pretrained model, fine-tuned for KIE/KFE, reports F1 ≥ the Phase-3 baseline
3. A comparison table/chart places our F1 against Parmar and GYMetricPose on identical metrics
</phase_requirements>

---

## Executive Summary

The paper's Motion-Disentangling (MD) method (§3.2, pp.6–8) and its implementation details (§5, p.9) are **substantially more specified than CONTEXT.md's provisional premises assumed**, and in one important way they **override** the provisional default:

- **The loss is NOT NT-Xent / InfoNCE / SimCLR-family.** It is the **triplet distance-ratio loss of Hoffer & Ailon (2015)** (paper Eq. 1, ref [13], arXiv:1412.6622) — a softmax over the anchor–positive and anchor–negative L2 distances. The paper states this explicitly: *"We use a 3DCNN as the backbone for this model, and Eq. 1 as the loss function"* (p.7). The only extant official code (`train_test.py` lines 60–71, the CVCSPC variant) implements exactly this loss. **This is decisive and CITED — it removes the largest `[ASSUMED]` from the SSL recipe.** [CITED: Parmar §3.2 p.7, Eq. 1 p.6; Hoffer & Ailon 2015 arXiv:1412.6622]

- **MD uses explicit triplets, not in-batch negatives.** Anchor = first half-cycle; positive = augmented copy of the anchor; negative = second half-cycle (paper p.7). Because negatives are *constructed per sample* (not drawn from the batch), the SimCLR "needs batch 256+" intuition does **not** apply. The paper trained MD at **batch size 5** (§5 p.9). This is the single most important VRAM finding: **the paper's batch 5 fits L4 24 GB with enormous headroom (~8 GB est.), and batch 8–16 also fit** (§5 of this doc).

- **Half-cycles are split on the amplitude-normalized barbell y-trajectory; 16 frames sampled per half-cycle** (paper p.9: *"We sampled 16 frames from each half-cycle"*). The paper does not publish the exact peak-detection algorithm (it defers to YOLOv3 detection + amplitude normalization to −180→180, p.9, and the supplementary material). The split point is the trajectory's **global extremum** (bottom-of-rep for squat = max-depth = the single largest-amplitude point). The official CVCSPC `dataloader.py` confirms the on-disk trajectory format and min-max normalization. **The exact splitter is reconstructed in §1 and labeled `[ASSUMED]` with the algorithm grounded in the paper's described trajectory shape (a single parabola → single bottom).**

- **The paper is silent (defers to supplementary) on:** projection-head dimensions, augmentation magnitudes, scheduler, and a separate downstream fine-tune LR. The supplementary material is **not** in the available PDF. Every such value below is `[ASSUMED]` with an SSL-literature or official-code basis. The official `train_test.py` shows a `model_linear_layers` projection head with **L2-normalized** output before the loss — confirming a projection head exists and embeddings are L2-normalized (§4).

**The fine-tune LR question (D3's one assigned job):** §5 (p.9) gives **one** learning rate, **1e-4**, used for *both* the MD SSL pretraining *and* — by the paper's own statement that it *"simply finetuned the model end-to-end"* — there is **no separately specified downstream LR**. The paper does **not** prescribe a lower transfer LR. **Confirm CONTEXT.md's 1e-4 fine-tune LR is defensible and paper-consistent** (§9). [CITED: §5 p.9]

**Primary recommendation:** Reproduce MD faithfully with the **triplet distance-ratio loss (Eq. 1)**, **16-frame half-cycle clips split at the trajectory bottom**, **explicit (anchor, augmented-anchor, second-half) triplets**, **batch 8 on L4** (paper's 5 is safe; 8 gives throughput headroom), **AdamW lr=1e-4, 20-epoch baseline budget extended to a linear-probe-monitored convergence target**, a **2-layer MLP projection head (512→512→128) discarded at fine-tune**, and **conservative augmentations** (the paper's strong augs minus those that destroy form signal — see §7). Monitor convergence with linear-probe F1 every 5 epochs and **embedding-std collapse detection** every epoch. Then fine-tune from the MD backbone with AdamW wd=1e-4 / dropout=0.2 / lr=1e-4 / 50ep+8-patience (D3), 3 seeds, ensemble + val-tuned TTA.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Barbell-trajectory load + half-cycle split | SSL dataset (CPU, `datasets/squat_ssl.py`) | scipy.signal | Pure-signal peak detection; runs in DataLoader workers |
| Triplet construction (anchor/pos/neg + temporal reverse) | SSL dataset (CPU) | — | Per-`__getitem__`; reuses `decode_clip`/`uniform_sample_indices` |
| SSL augmentations (temporal/spatial/color) | SSL dataset (CPU workers) | — | Applied per-branch; NEW (Phase 2 transforms have none of these) |
| MD forward (3 branches) + triplet loss | SSL trainer (GPU, `harness/md_pretrain.py`) | — | R(2+1)D-18 backbone + projection head + Eq. 1 |
| Linear-probe convergence monitor | SSL trainer (CPU+GPU) | `eval/metrics.py` | Freeze backbone, train linear head on labeled subset, sklearn F1 |
| Collapse detection (embedding std / rank) | SSL trainer (CPU) | — | Per-epoch on a fixed probe batch |
| SSL checkpoint I/O (resume across sessions) | Harness (`colab.py`) | Drive FUSE | Atomic write; **must preserve `latest.txt`** (D7) |
| Unlabeled-clip staging | Harness (`colab.py`, new fn) | Drive FUSE | Analog of `stage_squat_videos` for the 4,970 unlabeled clips |
| Fine-tune (AdamW + dropout, from MD init) | Fine-tune trainer (GPU, `harness/md_finetune.py`) | Phase 3 patterns | Parallels `supervised_train.py`; Adam→AdamW + head dropout |
| Threshold sweep / F1 / PR-AUC / confusion | Eval (`eval/metrics.py`) | — | Reused UNCHANGED (D9) |
| Ensemble score aggregation | Eval (`eval/ensemble.py`, NEW) | `eval/metrics.py` | Mean of sigmoids across seeds; feeds into metrics |
| Test-time augmentation | Eval (`eval/tta.py`, NEW) | `transforms.py` | Per-clip augmented forward passes; averaged before ensemble |
| Visualization | Notebook (matplotlib) | `figures/` disk | PNG before show (D8) |

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | Colab-pinned (≈2.x) | Training, autograd, `F.normalize`, triplet loss | Project standard (CLAUDE.md: form model is PyTorch); Phase 3 used it |
| torchvision | Colab-pinned (≈0.17–0.22) | `r2plus1d_18` + `R2Plus1D_18_Weights.KINETICS400_V1`; `io.read_video` | Paper uses R(2+1)D-18; Phase 2/3 contract |
| scipy | ≥1.11.0 (already in `requirements.txt`) | `signal.find_peaks`, `signal.savgol_filter`, `ndimage.gaussian_filter1d` for trajectory smoothing + bottom detection | Already a dependency; no new install |
| scikit-learn | ≥1.3.0 (already in `requirements.txt`) | `f1_score`, `precision_recall_curve`, `average_precision_score` (via `eval/metrics.py`) | Phase 3 contract; reused unchanged |
| numpy | ≥1.24.0 | Trajectory arrays, score aggregation | Phase 2/3 contract |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| matplotlib | (Colab) | Visualizations (D8) | Figure tasks |
| tqdm | (Colab) | Progress bars (working-agreement: never-silent cells) | All training/eval loops |
| PyAV (`av`) | (Colab, installed in Step 0) | FFmpeg backend for `torchvision.io.read_video` | Decode — Phase 2/3 carry |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `scipy.signal.find_peaks` for bottom detection | `np.argmax` on the smoothed trajectory | For a single-rep squat the trajectory is one parabola → the bottom is the **global** extremum → `np.argmax`/`np.argmin` is simpler and more robust than peak-finding. **Use `np.argmax` for single-rep; reserve `find_peaks` for the multi-rep tail** (§1). |
| Triplet distance-ratio loss (Eq. 1) | NT-Xent / InfoNCE (SimCLR), BYOL stop-gradient | **Rejected — the paper explicitly uses Eq. 1 (CITED).** NT-Xent would be a deviation from the published method, undermining the "faithful reproduction" defense ([[feedback_ai_correctness]]). Only consider if Eq. 1 fails to converge (document as deviation). |
| `torchvision.io.read_video` decode | TorchCodec | Phase 2/3 carry: one-line swap target if decode is the SSL throughput bottleneck on 4,970 clips (flagged §11; deferred per CONTEXT). |

**Installation:**
```bash
# Zero new pip dependencies. scipy / scikit-learn / numpy already in backend/requirements.txt.
# torch / torchvision are Colab-provided (same as Phase 3 — "no new pip installs" model).
# Step 0 installs PyAV (av) before the first torch import, exactly as Phase 3.
```

**Version verification:** No PyPI install is added by this phase, so no slopcheck/registry audit applies (see Package Legitimacy Audit). The notebook's Step 0 dep-version probe (carried from Phase 3) prints `torch / torchvision / scipy / scikit-learn / numpy / matplotlib / tqdm` versions for the supply-chain guard and reproducibility record.

## Package Legitimacy Audit

> This phase installs **no new external packages**. All libraries used (`torch`, `torchvision`, `scipy`, `scikit-learn`, `numpy`, `matplotlib`, `tqdm`, `av`) are either Colab-provided runtime packages or already pinned in `backend/requirements.txt` at Phase 2 close. The Phase 3 supply-chain guard (Step 0 dep-version probe that hard-fails on a missing package) is carried forward unchanged.

| Package | Registry | Disposition |
|---------|----------|-------------|
| (none added) | — | N/A — no new installs this phase |

**Packages removed due to slopcheck [SLOP] verdict:** none (no new packages).
**Packages flagged as suspicious [SUS]:** none (no new packages).

---

## §1 — Half-Cycle Splitting Algorithm (Open Question 1)

### What the paper specifies (CITED)

- The barbell/weight is detected over time with a **custom YOLOv3** to trace a motion trajectory; plotted against time it is *"an approximately parabolic curve"* (Fig. 3, §3 preliminary, p.5). [CITED: §3 p.5; §5 p.9]
- Trajectories are **amplitude-normalized to −180→180** *"simply for a resemblance to a circle"* (§5 p.9). [CITED: §5 p.9]
- For a squat, the **first half-cycle = squatting down (descent)**, the **second half-cycle = getting up (ascent)** (§3.2 "Useful property 1", p.6; Fig. 3 labels "first half-cycle" / "second half-cycle"). [CITED: §3.2 p.6]
- MD samples **16 frames from each half-cycle** (§5 p.9). [CITED: §5 p.9]
- The split is implicitly at the trajectory's **turning point** — the bottom of the rep (max depth), which for a single parabola is the **single global extremum** of the y-center curve.

### What the paper does NOT specify (defers to supplementary — `[ASSUMED]`)

The exact peak-detection signal (raw y vs. velocity), smoothing window, and multi-rep handling are **not in the main text**; "Further details provided in the supplementary material" (§5 p.9), and the supplementary is not in the available PDF. `Code_Release/motion_disentanglement/` is empty (CODE-RELEASE-NOTES.md). So the splitter is reconstructed below.

### What the official CVCSPC code reveals about the trajectory representation (CITED-from-code)

`Code_Release/.../dataloader.py` (the **pose** variant, not MD, but it loads the *same trajectory files*) shows:
- Trajectories are **per-clip JSON files** named `{video_id}.json`, each a **flat list of y-centers** (`json.load(open(ssl_trajectories_dir + file))`, lines 107–109). [CITED: dataloader.py:107]
- Normalization is **min-max to [0,1]**: `(traj[i] - min(traj)) / (max(traj) - min(traj))` (lines 111–114), then `traj2phase` multiplies by 360 to get degrees `[0,360]` (lines 35–39). The paper's text says −180→180; the code uses 0→360 — **equivalent circle parameterizations** (a constant offset). [CITED: dataloader.py:35,111]
- This **confirms Phase 1's finding**: "Squat: processed flat list of y-centers per clip ... Length 26–441 frames, 0% NaN" (01-DATASET-REPORT.md).

### Recommended splitter algorithm `[ASSUMED — grounded in the paper's described single-parabola shape + Phase 1's clip-length data]`

```python
# Source: reconstructed from Parmar §3.2 (descent=first half, ascent=second half) +
# §5 (16 frames/half-cycle) + Phase 1 trajectory shape. NOT from official code (MD code empty).
def split_half_cycles(traj_y: np.ndarray, frames_per_half: int = 16,
                      smooth_sigma: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (descent_indices[16], ascent_indices[16]) frame indices.

    traj_y: 1-D barbell y-center per trajectory sample (already a flat list per Phase 1).
    """
    # 1. Smooth to suppress YOLO jitter before extremum detection.
    sm = scipy.ndimage.gaussian_filter1d(traj_y.astype(float), sigma=smooth_sigma)
    # 2. Bottom-of-rep = the global extremum of the y-curve. Convention check at
    #    notebook time: in image coords y INCREASES downward, so "bottom of squat"
    #    (deepest) is the MAXIMUM y. Verify the sign empirically on 5 clips in the
    #    first execution task (a CHECKPOINT) — Fig. 3 plots amplitude, not raw pixel y.
    bottom = int(np.argmax(sm))          # or argmin — resolve sign empirically (see note)
    # 3. Descent = start..bottom ; Ascent = bottom..end.
    descent = uniform_sample_indices(num_frames=bottom + 1, target=frames_per_half)
    ascent  = uniform_sample_indices(num_frames=len(sm) - bottom, target=frames_per_half) + bottom
    return descent.numpy(), ascent.numpy()
```

**Design decisions and their justification:**

1. **Signal = smoothed raw y-center, not velocity.** The paper splits on the *amplitude* trajectory (Fig. 3 is amplitude vs. time), and the bottom-of-rep is an amplitude extremum, not a velocity zero-crossing. Velocity zero-crossings are noisier (differentiation amplifies YOLO jitter). `[ASSUMED — paper plots amplitude; velocity differentiation is a known noise amplifier]`

2. **Smoothing = Gaussian σ≈2 frames** (`scipy.ndimage.gaussian_filter1d`). Light smoothing removes single-frame YOLO detection jitter without displacing the extremum. Alternative `scipy.signal.savgol_filter(window=7, poly=2)` preserves the parabola peak slightly better. **Recommend trying both on 5 sample trajectories in the first execution task and picking by visual inspection** ([[feedback_interactive_execution]] — measure, don't guess). `[ASSUMED — σ=2 is conservative; verify visually]`

3. **Bottom = global extremum (`argmax`/`argmin`), NOT `find_peaks`,** for the median single-rep clip. Phase 1: median clip = 111 frames ≈ a single rep → the trajectory is one parabola → one extremum. `find_peaks` is overkill and can return spurious peaks on a monotone-ish curve. **The sign (argmax vs argmin) MUST be resolved empirically** because Fig. 3 plots "amplitude" (could be inverted) and image-coordinate y increases downward — make this a **`checkpoint:human-verify`** in the plan (the planner should gate the splitter behind a 5-clip visual check). `[ASSUMED — sign resolved at execution time]`

4. **Multi-rep clip handling (the long tail).** Phase 1: Squat clips run 49–404 frames (median 111, up to 13.5 s / 2–3 reps); trajectory length 26–441. The paper claims samples *"were automatically processed to contain a single repetition"* (§4 p.8) — so the **dataset is nominally single-rep**, and `argmax`/`argmin` is correct for the bulk. For the genuine multi-rep tail:
   - **Primary recommendation:** detect multi-rep via `scipy.signal.find_peaks(sm, prominence=0.3*amplitude_range, distance=min_rep_frames)`; if **>1 prominent minimum** is found, **take the first full rep** (first descent + its ascent up to the next peak) and ignore the rest. This matches the paper's "single repetition" intent and keeps every triplet a clean one-rep contrast. `[ASSUMED — paper says single-rep; first-rep extraction is the conservative reconstruction]`
   - **Fallback:** if peak detection is unreliable, clamp to the global extremum split (treats a 2-rep clip's deepest point as the boundary — slightly impure but rarely fires given the median is single-rep).
   - **Make multi-rep frequency a measured quantity:** the first execution task should report how many of the 4,970 trajectories have >1 prominent extremum, so the planner/user knows whether the tail matters at all.

5. **Frame-index mapping.** Trajectory length (26–441) differs from clip frame count (49–404) per Phase 1 — they are **not 1:1**. The trajectory is sampled at the barbell-detection cadence, not every video frame. **The mapping from trajectory index → video frame index must be probed in-notebook** (likely linear: `frame = round(traj_idx * (n_video_frames / n_traj_samples))`, or the trajectory may already be per-video-frame). **Flag as the first execution probe** alongside the format check (§8). `[ASSUMED — exact mapping unknown until probed]`

**Reuse:** the descent/ascent index sets feed directly into the existing `decode_clip(path, indices)` + `spatial_*` pipeline (datasets/transforms.py) — no new decode path. The half-cycle splitter is the **only genuinely new sampling logic**.

---

## §2 — SSL Loss Formulation (Open Question 2)

### Verdict: Triplet distance-ratio loss (Hoffer & Ailon 2015), NOT NT-Xent. **CITED.**

This **overrides** CONTEXT.md D2's provisional "SimCLR-family NT-Xent / InfoNCE default."

**Paper Eq. 1 (§3.1 p.6, reused for MD per §3.2 p.7):**

> *"Following [43], we optimize the parameters of f during the self-supervised training, by minimizing the distance ratio loss [13]"* — and for MD: *"We use a 3DCNN as the backbone for this model, and Eq. 1 as the loss function for this self-supervision task."* [CITED: §3.1 p.6; §3.2 p.7]

The LaTeX (extracted from the PDF, p.6):

```
L = -log( e^(-||φ_anc - φ_pos||₂) / ( e^(-||φ_anc - φ_pos||₂) + e^(-||φ_anc - φ_neg||₂) ) )
```

Reference [13] = **Hoffer & Ailon, "Deep Metric Learning Using Triplet Network," 2015 (arXiv:1412.6622)** — verified via WebSearch: *"A SoftMax function is applied on both outputs to create a ratio measure."* This is a **triplet network with a softmax-over-distances objective**, distinct from NT-Xent (which uses cosine similarity / temperature over a batch of negatives). [CITED: Hoffer & Ailon 2015, arXiv:1412.6622]

### What the official code actually implements (CITED-from-code) — and a discrepancy to flag

`train_test.py` lines 60–71 (the CVCSPC variant, same loss family):

```python
# Source: Fitness-AQA-Code/.../train_test.py:60-71 (VERBATIM, official author code)
loss = 0
current_batch_size = anchor_im_feats.shape[0]
for sample in range(current_batch_size):
    temp_loss_anc_pos = torch.exp(-1 * sum(((anchor_im_feats[sample,:] - positive_im_feats[sample,:])**2)))
    temp_loss_anc_neg = torch.exp(-1 * sum(((anchor_im_feats[sample,:] - negative_im_feats[sample,:])**2)))
    temp_loss_pos_neg = torch.exp(-1 * sum(((positive_im_feats[sample,:] - negative_im_feats[sample,:])**2)))
    # temp_loss = -1*torch.log(temp_loss_anc_pos / (temp_loss_anc_pos + temp_loss_anc_neg))                       # <-- 2-term (matches Eq. 1), COMMENTED OUT
    temp_loss = -1*torch.log(temp_loss_anc_pos / (temp_loss_anc_pos + temp_loss_anc_neg + temp_loss_pos_neg))     # <-- 3-term, ACTIVE
    temp_loss = temp_loss / current_batch_size
    loss += temp_loss
```

**Two discrepancies between the paper's Eq. 1 and the released code — document both:**

1. **Squared vs. non-squared distance.** Eq. 1 (paper) uses `||·||₂` (Euclidean). The code uses `sum((a-p)**2)` = **squared** Euclidean. `[ASSUMED — the released code is the more authoritative artifact for implementation; recommend matching the code's squared form, noting the paper's Eq. 1 shows non-squared]`
2. **2-term vs. 3-term denominator.** Eq. 1 has a **2-term** denominator (`anc-pos + anc-neg`). The **active** code line has a **3-term** denominator adding `pos-neg` repulsion. The 2-term line is commented out. `[ASSUMED — the active code uses 3-term; recommend matching the code (3-term) as the as-shipped author intent, but note Eq. 1 is 2-term]`

**Recommendation for the planner:** implement the loss to **match the released code** (squared distance, 3-term denominator) as the primary, since it is the authors' actual artifact, but **expose a `denominator: Literal["2term","3term"]` config flag and a `squared: bool` flag** so the exact-Eq.1 form can be ablated if convergence is poor. Cite both the paper Eq. 1 and `train_test.py:68` in the docstring. This is the [[feedback_ai_correctness]] "defensible to examiners" stance: the implementation traces to the authors' own code, and the paper-vs-code discrepancy is documented, not hidden.

```python
# Recommended PyTorch implementation (batched, vectorized — avoids the code's per-sample loop)
def md_triplet_loss(phi_anc, phi_pos, phi_neg, *, squared=True, three_term=True, eps=1e-9):
    # phi_* are L2-normalized embeddings [B, D] (see §4).
    d_ap = ((phi_anc - phi_pos) ** 2).sum(-1)          # squared L2 per official code
    d_an = ((phi_anc - phi_neg) ** 2).sum(-1)
    if not squared:
        d_ap, d_an = d_ap.sqrt(), d_an.sqrt()
    num = torch.exp(-d_ap)
    den = num + torch.exp(-d_an)
    if three_term:
        d_pn = ((phi_pos - phi_neg) ** 2).sum(-1)
        if not squared:
            d_pn = d_pn.sqrt()
        den = den + torch.exp(-d_pn)
    return (-torch.log(num / (den + eps))).mean()
```

- **No stop-gradient (NOT BYOL/SimSiam-style).** The paper's loss is symmetric metric learning over a triplet; gradients flow through all branches. There is no predictor head, no stop-gradient, no momentum/target encoder. `[CITED — Eq. 1 has no stop-gradient term; official code backprops through all three branches]`
- **No temperature parameter.** Eq. 1 has no temperature τ (unlike NT-Xent). The "temperature" is effectively fixed at 1 inside `exp(-d)`. `[CITED — Eq. 1]`

---

## §3 — Positive / Negative Sampling (Open Question 3)

### Verdict: explicit per-sample triplet, within-instance negative. No memory bank, no hard-negative mining. **CITED.**

Paper §3.2 (p.7), verbatim:

> *"The first half-cycle serves as the anchor; an augmented copy of the anchor serves as the positive input. The second half-cycle serves as the negative input. As discussed previously, we randomly temporally-reverse either the {anchor, positive} pair or the {negative} input to make the global motion of all three identical."* [CITED: §3.2 p.7]

So, **per training sample (one rep)**:
- **Anchor** = first half-cycle (descent), 16 frames.
- **Positive** = augmented copy of the **same** anchor half-cycle (a different augmentation of the descent).
- **Negative** = second half-cycle (ascent), 16 frames — from the **same rep/clip**.
- **Temporal reversal**: randomly reverse **either** {anchor + positive} **or** {negative}, so the global (down-vs-up) motion is identical across all three and only the **local** (anomalous) motion distinguishes them. This is the "accentuating local motion" mechanism (§3.2 p.6–7). [CITED: §3.2 p.7]

**Consequences for the planner:**
- This is **NOT** SimCLR/MoCo in-batch contrast. There is **one negative per anchor**, constructed from the same clip. **No memory bank (MoCo), no in-batch negatives (SimCLR), no hard-negative mining** — none are in the paper. `[CITED — paper constructs exactly one (anc, pos, neg) triplet per sample]`
- This is **why batch size 5 works** (§5): the loss does not need many negatives in the batch. The negative is the structural opposite half-cycle. **Do not "fix" this by adding in-batch negatives** — that would deviate from the published method.
- **Implication for the SSL dataset class:** each `__getitem__` returns a **dict of 3 clips** `{anchor[3,16,H,W], positive[3,16,H,W], negative[3,16,H,W]}` (mirroring `train_test.py`'s `data['anchor_im']/'positive_im']/'negative_im']`), not a single clip. The collate produces three `[B,3,16,H,W]` batches.

---

## §4 — Projection Head Architecture (Open Question 4)

### What the official code shows (CITED-from-code)

`train_test.py` lines 51–53 apply a `model_linear_layers` head and **L2-normalize** the output before the loss:

```python
# Source: train_test.py:51 (VERBATIM)
anchor_im_feats = F.normalize(model_linear_layers(model_CNN(anchor_im)), dim=-1, p=2)
```

So a projection head (`model_linear_layers`) **does exist**, and embeddings are **L2-normalized** (`F.normalize(..., dim=-1, p=2)`) before the distance-ratio loss. **The exact dims of `model_linear_layers` are NOT in the repo** (`ssl_contrastive_image_cleaned.models.linear_layers` is one of the missing modules per CODE-RELEASE-NOTES.md §(b)). The main paper does not state projection dims either. `[CITED that a head exists + L2-norm applied; ASSUMED for exact dims]`

### Recommendation `[ASSUMED — SimCLR/SimSiam-standard MLP head; L2-norm is CITED from official code]`

```python
# Source: SimCLR (Chen et al. 2020) standard 2-layer projector convention;
#         L2-norm before loss is CITED from train_test.py:51.
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, hidden=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):           # x = backbone global-pooled features [B, 512]
        return F.normalize(self.net(x), dim=-1, p=2)   # L2-norm CITED from official code
```

- **2-layer MLP, 512 → 512 → 128**, BN+ReLU between. This is the SimCLR-standard projector (Chen et al. 2020); SimSiam uses 3 layers but 2 is the common minimal and matches "linear_layers" (plural but small). `[ASSUMED — SimCLR convention]`
- **Output dim 128.** SimCLR/MoCo use 128; this is the field default. `[ASSUMED — SimCLR/MoCo default]`
- **L2-normalize the projection output** before the triplet loss. **CITED** from `train_test.py:51`.
- **Backbone features = the 512-dim global-average-pooled output** of R(2+1)D-18 (the same `in_features=512` Phase 3 asserts at the `fc` layer — `supervised_train.py:129`). Take features **before** the classification `fc`. The cleanest construction: build `r2plus1d_18`, set `model.fc = nn.Identity()`, attach the projector separately. `[CITED in_features=512 from torchvision; ASSUMED Identity-fc construction]`
- **DISCARD the projection head at fine-tune time (standard).** Fine-tune loads only the backbone weights and attaches a fresh `Linear(512, 2)` + Dropout(0.2) head (D3). This is universal SSL practice (SimCLR §B; the projector is a pretext-only artifact). The official MD model "simply finetuned the model end-to-end" with a fresh error head (§5 p.9). `[CITED discard-projector via §5 end-to-end fine-tune; ASSUMED that the discard follows SimCLR convention]`

**Checkpoint implication:** `md_pretrain.py`'s `backbone.pt` payload should save **both** `backbone_state_dict` and `projector_state_dict` (so SSL can resume mid-pretrain), but `md_finetune.py` loads **only** `backbone_state_dict` into the R(2+1)D-18 (then attaches its own fresh head). See §10.

---

## §5 — SSL Optimizer / Scheduler / Epochs / Batch Size + VRAM (Open Question 5)

### Paper-specified values (CITED)

> *"Motion Disentanglement approach (MD). We used R(2+1)D-18 as our backbone CNN. We sampled 16 frames from each half-cycle. We randomly applied strong augmentations. We initialized our backbone CNN with Kinetics pretrained weights. We optimized our models using ADAM optimizer with an initial learning rate of 1e-4 for 20 epochs with a batch size of 5."* [CITED: §5 p.9]

| Parameter | Paper value | Recommendation for Phase 4 | Basis |
|-----------|-------------|----------------------------|-------|
| Backbone | R(2+1)D-18, Kinetics-V1 init | **Same** (matches Phase 3 init for clean comparison) | [CITED §5 p.9] |
| Frames per half-cycle | 16 | **16** (32-frame triplet total span: 16 anchor + 16 negative) | [CITED §5 p.9] |
| Optimizer | ADAM, lr=1e-4 | **AdamW lr=1e-4** (CONTEXT D2 overrides to AdamW for decoupled weight decay; wd=1e-4) | [CITED lr; CONTEXT D2 for AdamW] |
| Epochs | 20 | **20 baseline budget, extended to linear-probe convergence (≤ ~60)** — see below | [CITED 20; extension ASSUMED] |
| Batch size | 5 | **8 on L4** (5 is paper-faithful and safe; 8 adds throughput; both fit 24 GB) | [CITED 5; 8 ASSUMED from VRAM math] |
| Scheduler | not stated | **cosine annealing over the epoch budget** (paper silent; cosine is the SSL default) | [ASSUMED — SimCLR convention] |
| Weight decay (SSL) | not stated | **1e-4** (AdamW decoupled) — modest, standard | [ASSUMED] |
| Warmup | not stated | **OFF** (short run, Kinetics-init backbone is already warm) | [ASSUMED] |
| LARS | — | **NO** (LARS is for batch ≥1024; we use batch 5–8) | [ASSUMED — LARS unnecessary at small batch] |

### Epoch budget: 20 vs. convergence-monitored

CONTEXT D2 sets the SSL budget at **12–24h on L4**. The paper's 20 epochs is on the full unlabeled set (5,000+ clips ≈ our 4,970). **Recommendation: run to the paper's 20 epochs as the baseline, but use linear-probe F1 (§6) as the convergence signal and allow extension up to ~60 epochs if the linear-probe is still rising.** Rationale: the paper's 20 may have been schedule-bounded, not convergence-bounded; the linear-probe plateau is the principled stop. Cap at ~60 to bound the 12–24h budget. Document the actual epoch count and the linear-probe curve in SUMMARY. `[CITED 20 baseline; ASSUMED extension policy]`

### VRAM — the critical re-estimate (extrapolated from Phase 3's MEASURED 15.22 GB)

Phase 3 **measured** R(2+1)D-18 at **batch 16, 32-frame, 112², fp32 = 15.22 GB backward** (03-01-SUMMARY.md). Extrapolating activation memory linearly (≈0.0278 GB per sample×frame; ~1 GB fixed for weights+Adam states+grads+workspace):

| MD config (3 branches × 16 frames = 48 frame-equiv/sample) | Estimated peak VRAM (fp32, training) | Fits L4 24 GB? |
|-------------------------------------------------------------|--------------------------------------|----------------|
| **batch 5** (paper-faithful) | **~7.7 GB** | ✅ huge headroom |
| **batch 8** (recommended) | **~11.7 GB** | ✅ comfortable |
| batch 16 | ~22.3 GB | ⚠️ tight — likely OOM with fragmentation |
| batch 24 | ~33 GB | ❌ |

**Key finding: gradient accumulation is NOT needed.** CONTEXT D2 worried that "SimCLR wants batch 256+." That intuition applies to **in-batch-negative** methods (NT-Xent). **The paper's MD uses explicit per-sample triplets with one structural negative (§3) — it does not need large batches, and the paper trained at batch 5.** Batch 8 on L4 is comfortable and faster than 5. **Recommend batch 8; fall back to 5 (paper-exact) if any OOM.** A VRAM probe cell (like Phase 3's Step 2) should measure actual peak before committing — Phase 3's estimate was off (8–12 GB est. vs 15.22 GB measured), so **measure, don't trust the table** ([[feedback_ai_correctness]]). `[MEASURED basis = Phase 3 15.22 GB; ASSUMED linear extrapolation — re-probe in-notebook]`

### Wall-time anchor

- Phase 3 fine-tune: ~5.3 min/epoch on L4 (1,136 clips, batch 16, 32-frame).
- SSL: 4,970 clips ≈ 4.4× more data; but each clip yields one triplet (3×16 frames ≈ 48 frame-equiv vs Phase 3's 32) and batch 8 vs 16. Net throughput is hard to predict; **estimate ~15–35 min/epoch**. At 20–60 epochs → **5–35h**, consistent with CONTEXT's 12–24h expectation. **The epoch-0 timing probe (carry from Phase 3 Step 3, a BLOCKING `checkpoint:decision`) is mandatory** before authorizing the full pretrain. `[ASSUMED — extrapolated; measure epoch 0]`
- **Decode is the likely bottleneck on 4,970 clips.** If epoch-0 timing is dominated by decode (GPU < 50% util), flag the TorchCodec swap (deferred per CONTEXT, but `decode_clip` is already a one-line swap point per transforms.py docstring).

### No AMP/fp16 (carry Phase 3 §11)

Same reasoning as Phase 3: paper fidelity (no mixed precision mentioned), determinism conflict (`torch.use_deterministic_algorithms(True, warn_only=True)` is set in `colab.py`), and VRAM is not constrained at batch 5–8. **Keep fp32.** `[ASSUMED — carries Phase 3 §11 rationale]`

---

## §6 — Linear-Probe Validation Cadence During SSL (Open Question 6)

### Purpose

The linear-probe is the **convergence monitor** for the 12–24h pretrain (CONTEXT D2) and is the principled early-stop / extend signal. The paper does not describe a linear-probe cadence (it reports only final fine-tune F1). This is `[ASSUMED — standard SSL practice (SimCLR §B.2 evaluates linear-probe periodically); cadence is a judgment call]`.

### Protocol

Every **N epochs** (recommend **N = 5**):
1. **Freeze** the SSL backbone (no grad).
2. Extract 512-dim **backbone** features (NOT projection-head output — linear-probe evaluates the backbone that fine-tune will reuse) for the labeled **train** subset, using the **val/test spatial transform** (deterministic, no SSL augs) and **32-frame uniform sampling** (the fine-tune sampling, not the 16-frame half-cycle — the probe should mirror downstream use).
3. Train a **single `Linear(512, 2)`** head (logistic regression) on those frozen features for a few epochs (or sklearn `LogisticRegression`), with `BCEWithLogitsLoss(pos_weight=...)`.
4. Compute **val macro-F1** (KIE + KFE) at threshold 0.5 via `eval/metrics.py`.
5. Log `linear_probe_f1_kie / kfe / macro` into the SSL `metrics_history`.

**Cadence rationale:** N=5 over a 20–60 epoch budget gives 4–12 probe points — enough to see the plateau without dominating wall-time (each probe is cheap: feature extraction on ~1,136 clips once + a tiny head fit, ≈ a few minutes). N=10 (CONTEXT's other suggestion) is acceptable if probe cost is high; N=5 is the recommendation for a clearer convergence curve (a defense visualization — D8 lists "linear-probe F1-during-SSL curve"). `[ASSUMED — N=5 balances signal vs cost]`

**Convergence criterion:** linear-probe macro-F1 plateaus (no improvement over 2 consecutive probes = 10 epochs) ⇒ SSL has learned what it can ⇒ stop. This is separate from the SSL loss decreasing (loss can keep dropping while features stop improving for the downstream task). **Both signals (loss ↓ AND probe-F1 plateau) are reported.**

**Important:** the linear-probe is a **monitor**, not the deliverable. The deliverable is the **fine-tuned** (end-to-end) model (§9). A rising-then-plateauing linear-probe F1 is the evidence that "SSL pretraining converged" (ROADMAP success criterion 1). A **flat-at-random** linear-probe F1 is the early warning of representation collapse (§12) or a broken pretext task.

---

## §7 — SSL Augmentation Policy (Open Question 7)

### What the paper specifies (CITED)

> *"In practice, we randomly and independently applied the following augmentations on the triplets: image horizontal flipping, partial image masking, image translation, image rotation, image blurring, image zooming, color channel swapping, temporal shifting."* (§3.2 p.7) — and *"We randomly applied strong augmentations"* (§5 p.9). [CITED: §3.2 p.7; §5 p.9]

So the paper's MD augmentation set is **8 augmentations, applied randomly and independently per triplet branch**: (1) horizontal flip, (2) partial masking, (3) translation, (4) rotation, (5) blur, (6) zoom, (7) color-channel swap, (8) temporal shift. The exact magnitudes/probabilities are in the supplementary (not available). The official `image_augmentations.py` confirms the primitives exist (`hori_flip`, `masking`, `masking_checker_*`) but in the CVCSPC `dataloader.py` **only `masking` is active** (lines 219–231; flip/translation/rotation/blur/zoom/color are all **commented out** in the released config). `[CITED augmentation list; magnitudes ASSUMED]`

### The form-signal tension and the recommendation

CONTEXT D2 flags the core tension: *"strong augs aid SSL invariance but may destroy form-error signal."* This is real — **rotation and aggressive zoom/translation can destroy the very knee-angle/depth cues that KIE/KFE detection depends on**. But the paper's whole point is that the local (anomalous) motion survives the augmentations because the **half-cycle reversal** (not the augmentation) is what isolates it (§3.2). The positive is an *augmented copy of the anchor* — augmentation defines the invariances; the descent-vs-reversed-ascent contrast defines what's discriminative.

**Recommendation: a conservative subset of the paper's 8, justified per-augmentation, with magnitudes verified by the linear-probe.** `[ASSUMED — paper lists augs but not magnitudes; conservatism justified by form-signal preservation + [[feedback_ai_correctness]]]`

| Augmentation | Include? | Magnitude (start) | Justification |
|--------------|----------|-------------------|---------------|
| Temporal shift | **YES** | ±2 frames (matches Phase 3 jitter) | Core to MD; the half-cycle is temporal; safe |
| Horizontal flip | **YES** | p=0.5 | Body is bilaterally ~symmetric for KIE/KFE; paper includes it; **note this differs from Phase 3 fine-tune (flip OFF) — that's fine, SSL augs ≠ fine-tune augs** |
| Partial masking | **YES** | top-mask 0.4–0.5 (per `image_augmentations.masking`) | Occlusion robustness (the paper's only *active* CVCSPC aug); preserves lower-body knee region if masking the top |
| Color-channel swap / jitter | **YES (mild)** | channel permute OR brightness/contrast ±0.2 | Appearance invariance (clothing/lighting) — the paper's stated goal; does not move joints |
| Spatial translation | **MILD** | ±10 px (per code's commented `translation`) | Small only; large translation crops out the knees |
| Zoom | **MILD** | 0.9–1.1× | Small only; aggressive zoom changes apparent depth (KFE cue) |
| Rotation | **OPTIONAL / SMALL** | ±10° max | **Riskiest** — rotation changes apparent knee-valgus angle (the KIE signal). Start OFF or ≤10°; **only enable if linear-probe F1 improves with it** |
| Blur | **OPTIONAL** | σ 0–1.2 (per code) | Low risk but low value for motion; include if cheap |

**Decision procedure (measure, don't guess):** start with the **safe core** (temporal shift, flip, masking, mild color) for the first SSL run. After the first linear-probe reading, if convergence is weak, **add** mild translation/zoom. **Treat rotation as an ablation toggle**, not a default. Document the final augmentation set in SUMMARY with the linear-probe evidence. This honors [[feedback_ai_correctness]]: every augmentation choice is either paper-cited (in the set) or justified by a measured linear-probe delta (magnitude). `[ASSUMED magnitudes — verified empirically]`

**Implementation note:** these augmentations are **all NEW** — Phase 2's `transforms.py` has only resize/crop/normalize (no flip, rotation, masking, color, temporal-shift-as-aug beyond the index jitter). Build them as a small `datasets/ssl_augs.py` (or inline in the SSL dataset) operating on the decoded `[T,3,H,W]` uint8 clip **before** the existing `spatial_*` normalize step. Apply **independently per branch** (anchor, positive, negative each get their own random aug draw — matching `train_test.py:219` `augmentations = [{}, {}, {}]`).

---

## §8 — Trajectory File Format (Open Question 8)

### What we know (CITED from Phase 1 + official code)

- **Squat unlabeled trajectories ship as `bar_trajectories_raw.zip`** (4,970 entries) in `My Drive/Fitness-AQA_dataset_release/Squat/Unlabeled_Dataset/` (01-DATASET-REPORT.md line 30). [CITED: Phase 1]
- Phase 1 processed them to a **"flat list of y-centers per clip ... Length 26–441 frames, 0% NaN"** (01-DATASET-REPORT.md line 90). [CITED: Phase 1]
- The official `dataloader.py` loads trajectories as **per-clip JSON files** named `{video_id}.json`, each a flat list, via `json.load(open(ssl_trajectories_dir + file))` (lines 66, 107). [CITED: dataloader.py:66,107]
- **`traj_nan.json` lists 19 clips with NaN trajectories — but that is for the LABELED set** (01-DATASET-REPORT.md line 107: "lists 19 labeled clips"). The CVCSPC `dataloader.py` excludes `traj_nan` from the SSL set (lines 67–68). For the **unlabeled** Squat set, Phase 1 measured **0% NaN** — so no exclusion may be needed, but **verify there is no separate unlabeled `traj_nan` list at probe time.** [CITED: Phase 1 + dataloader.py:67]

### What MUST be probed in-notebook (the FIRST execution task) — `[ASSUMED until probed]`

The exact on-disk layout cannot be confirmed from this machine (the files are on the user's Drive, not in the repo). **The first execution task (a `checkpoint:human-verify` probe) must `ls` and load 2–3 trajectory entries and report:**

1. **Archive internal structure**: one JSON per clip (`{id}.json`, per the CVCSPC code), or one big JSON/CSV/`.npy`? (CVCSPC code → expect per-clip JSON.)
2. **Value type**: is each entry a flat `list[float]` of y-centers (Phase 1 says yes), or raw YOLO boxes `[[x1,y1,x2,y2,conf]]` like the **OHP** set (01-DATASET-REPORT.md line 92 — OHP is raw, Squat is processed)? **Squat is expected processed/flat; confirm.**
3. **Raw vs. smoothed**: Phase 1 says "clean parabolic curves" → likely already smooth, but the splitter (§1) applies its own light smoothing defensively.
4. **Trajectory length vs. video frame count mapping** (§1 point 5): is the trajectory per-video-frame or per-detection-sample? Probe `len(traj)` vs `read_video_timestamps(video)` count for the same clip ID.
5. **ID alignment**: do trajectory file stems match the unlabeled `videos.zip` clip IDs 1:1? (CVCSPC code assumes `{video_id}.json` ↔ `{video_id}/` frame dir.)
6. **Unlabeled `traj_nan`**: is there a NaN-exclusion list for the unlabeled set, or is 0% NaN (Phase 1) sufficient to skip exclusion?

**Planner action:** make the trajectory-format probe the **first SSL task**, gated as a `checkpoint:human-verify`, BEFORE the SSL dataset class is written. The dataset class's trajectory-loading code is `[ASSUMED]` (per-clip JSON, flat float list) until this probe confirms. This is the highest-uncertainty item in the phase and the cheapest to resolve (one `ls` + 3 `json.load`s).

---

## §9 — Fine-Tune Transfer Recipe + LR Confirmation (Open Question / D3's assigned job)

### The assigned LR question: does §5 specify a different downstream fine-tune LR?

**Answer: NO. The paper specifies exactly one learning rate (1e-4), used for the MD SSL pretraining. For the downstream task it states only that it *"simply finetuned the model end-to-end on the labeled dataset for error detection"* (§5 p.9) with no separate LR.** A full-text search of the paper for learning-rate mentions returns only the two `1e-4` occurrences (CVCSPC and MD pretraining, both §5 p.9). There is **no lower transfer LR** prescribed. [CITED: §5 p.9 — only `1e-4` appears]

**Verdict: CONTEXT.md D3's fine-tune LR = 1e-4 is defensible and paper-consistent.** The common SSL practice of using a *lower* downstream LR (to avoid disrupting pretrained features) is a reasonable instinct, but **the paper does not do it**, and matching the paper preserves the "faithful reproduction" defense. **Keep LR = 1e-4.** If early fine-tune epochs show AdamW+SSL-init instability (D3's deferred warmup ablation trigger), a 5-epoch linear warmup is the first remedy — but that is a documented deviation, not the default. `[CITED 1e-4; the "keep it" verdict is the assigned confirmation]`

### Fine-tune recipe (D3, with the AdamW + dropout deltas vs Phase 3)

`md_finetune.py` parallels `supervised_train.py` (do NOT bloat the Phase 3 module — D9). The deltas:

| Aspect | Phase 3 (`supervised_train.py`) | Phase 4 (`md_finetune.py`) | Change source |
|--------|--------------------------------|----------------------------|---------------|
| Init | `R2Plus1D_18_Weights.KINETICS400_V1` | **MD `backbone.pt` from §5 pretrain** | D2 |
| Optimizer | `torch.optim.Adam` (supervised_train.py:420) | **`torch.optim.AdamW`** | D3 (decoupled weight decay) |
| weight_decay | 0.0 | **1e-4** | D3 |
| Head | `nn.Linear(512, 2)` | **`nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 2))`** | D3 (dropout head-only) |
| LR | 1e-4 | **1e-4** (unchanged) | §9 (paper-confirmed) |
| Epochs / patience | 50 / 8 | **50 / 8** (unchanged — P3 parity) | D3 |
| Scheduler | CosineAnnealingLR | **CosineAnnealingLR** (unchanged) | D3 |
| Loss | `BCEWithLogitsLoss(pos_weight)` | **same** | D3/Phase 3 D2 |
| Sampling | 32-frame, jitter ±2, crop 112, flip OFF | **same** (fine-tune uses the labeled pipeline, NOT 16-frame half-cycles) | D9 |
| Warmup | OFF | **OFF** (deferred ablation) | D3 |

**Backbone loading detail:** `md_finetune.build_model()` constructs `r2plus1d_18(weights=None)` (architecture only), loads the MD `backbone_state_dict` (NOT the projector), asserts `fc.in_features == 512`, then sets `model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 2))`. **Do NOT load Kinetics weights** — the MD backbone *is* the init. Verify the state-dict keys match (R(2+1)D-18 layer names) and that the projector keys are absent/ignored. `[CITED end-to-end fine-tune from §5; construction ASSUMED]`

**Deviation to NOTE in SUMMARY (per D3):** the paper used **20 downstream epochs**; Phase 4 uses **50 with 8-epoch early-stop** for clean Phase-3-vs-Phase-4 comparison parity. Early-stop bounds actual cost (Phase 3 stopped at 11). This is a *deliberate, justified* deviation. `[CITED 20-epoch paper budget; deviation per CONTEXT D3]`

**AdamW vs Adam semantic note (carry handoff risk #5):** Phase 3's `Adam(weight_decay=0)` had no regularization. AdamW decouples weight decay from the gradient-based update (Loshchilov & Hutter 2019), so `wd=1e-4` is *true* weight decay, not L2-on-gradient. This is a one-line change (`torch.optim.Adam` → `torch.optim.AdamW`) with a real semantic difference — the planner must use `AdamW` explicitly. `[ASSUMED AdamW citation = Loshchilov & Hutter 2019; the change itself is CONTEXT D3]`

---

## §10 — Disconnect-Safety for the 12–24h Pretrain

The SSL pretrain is the **longest run in the project** and **will span multiple Colab sessions** (CONTEXT specifics: ~12–24h; [[feedback_heavy_training_new_notebook]] + [[feedback_notebook_disconnect_safe]]). The Phase 2 atomic-write + `latest.txt` contract (`colab.py`) is the base; the SSL trainer extends it.

### SSL checkpoint payload schema (mirror `supervised_train.py`'s payload, with SSL-specific additions)

```python
payload = {
    "epoch": epoch,
    "backbone_state_dict": backbone.state_dict(),         # R(2+1)D-18 (fc=Identity)
    "projector_state_dict": projector.state_dict(),       # discarded at fine-tune, kept for resume
    "optimizer_state_dict": optimizer.state_dict(),        # AdamW state — REQUIRED for correct resume
    "scheduler_state_dict": scheduler.state_dict(),
    "rng_state": capture_rng_state(),                      # 4-RNG (colab.py) — covers SSL aug + triplet sampling
    "metrics_history": metrics_history,                    # per-epoch SSL loss + linear-probe F1 + embedding-std
    "linear_probe_history": linear_probe_history,          # the convergence curve (D8 viz)
    "config_hash": config_hash_str,                        # MDConfig hashed (D12 analog)
    "config_repr": config_repr,
    "code_version": "phase04-md-pretrain",
}
```

### Resume mechanics

- Auto-resume from `{drive_root}/FitNova/checkpoints/phase04/md_pretrain_v1/latest.txt` via `load_latest_checkpoint(run_dir, expected_config_hash, map_location="cpu")` — **`map_location='cpu'` is mandatory** (D7; Phase 3 fix `a0841b4` — CUDA ByteTensors break `set_rng_state_all`).
- Restore backbone + projector + AdamW + scheduler + 4-RNG; recompute `start_epoch = prior["epoch"] + 1`; copy `metrics_history` and `linear_probe_history`.
- Checkpoint **every epoch** via `atomic_save_checkpoint(payload, f"{run_dir}/epoch_{epoch:03d}.pt")` + `prune_checkpoints(keep_last=3, keep_best=True)`. A disconnect loses at most one epoch.

### The `latest.txt` clobber landmine (D7 — must fix)

`atomic_save_checkpoint` **always** writes `latest.txt` last (colab.py:589–590) — there is **no `update_latest=False` kwarg**. Phase 3 worked around this by inline save/restore of `latest.txt` around the `best.pt` write (03-01-SUMMARY.md deviation). **For Phase 4 SSL, the "best" checkpoint concept differs:** SSL has no val-F1-best in the supervised sense; the deliverable is the **final** backbone (or the linear-probe-best epoch). **Recommendation:**
- Define `backbone.pt` = the **linear-probe-macro-F1-best** epoch's backbone (analogous to Phase 3's `best.pt` selection but on linear-probe F1).
- **Add an explicit `update_latest: bool = True` kwarg to `atomic_save_checkpoint`** (the clean fix CONTEXT D7 suggests) so the `backbone.pt` write passes `update_latest=False` and never clobbers the epoch-resume pointer. This is a small, safe, well-isolated change to `colab.py` that also benefits Phase 3's pattern retroactively. `[ASSUMED — the kwarg is the cleanest fix; CONTEXT D7 endorses it]`
- The fine-tune `best.pt` writes (×3 seeds) inherit Phase 3's exact `best.pt` contract via `md_finetune.py`.

### Unlabeled-clip staging (new `colab.py` function)

`stage_squat_videos` is hardcoded to the **labeled** `videos.zip` (`expect_count=1739`, colab.py:59,317). SSL needs an analog for the **4,970 unlabeled** clips:
- New `stage_unlabeled_squat_videos(drive_root, *, expect_count=4970)` copying `Squat/Unlabeled_Dataset/videos.zip` → `/content/squat_unlabeled_videos/` with the same 3-layer resume (cache-hit / byte-resume copy / per-member extract). **Plus** stage `bar_trajectories_raw.zip` → a local trajectories dir.
- **Reuse the existing `_copy_with_resume_and_progress` + `_extract_with_resume_and_progress` helpers** (they're generic). The new function is ~30 lines parameterizing the existing primitives. Do NOT duplicate the resume logic. `[ASSUMED count=4970 from Phase 1; staging pattern reused]`
- **Disk budget check:** 4,970 mp4 clips — Phase 1 didn't report total size, but at ~1–3 MB/clip ≈ 5–15 GB. Colab `/content/` has ~70–100 GB. **Probe the unlabeled `videos.zip` size at staging time**; if it threatens disk, the 3-layer resume already handles partial extraction. `[ASSUMED size; probe at stage time]`

---

## §11 — Reuse Map (Planning-Critical)

### Modules reused UNCHANGED

| Module | What transfers | Constraint |
|--------|----------------|------------|
| `eval/metrics.py` | `f1_per_error`, `pr_auc_per_error`, `threshold_sweep`, `confusion_matrix_per_error` | **UNCHANGED** (D9). `ensemble.py`/`tta.py` feed cleaned score arrays in. Note: `pr_auc_per_error` **asserts scores ∈ [0,1]** (metrics.py:72) — ensemble/TTA must output **sigmoid means, not logit means** (D4 mean-of-sigmoids satisfies this). |
| `datasets/splits.py` | `index(split, ...)`, `ClipRecord` | UNCHANGED — fine-tune uses labeled splits |
| `datasets/squat.py` | `SquatKIEKFEDataset`, `build_loaders`, `_compute_pos_weight` | UNCHANGED — fine-tune reuses the labeled pipeline + `pos_weight` |
| `datasets/transforms.py` | `decode_clip`, `uniform_sample_indices`, `spatial_train`, `spatial_val`, `KINETICS_MEAN/STD` | UNCHANGED — **SSL reuses `decode_clip` + `uniform_sample_indices` + `spatial_*` for each half-cycle**; SSL augs are a NEW layer applied before `spatial_*` |
| `harness/colab.py` | `mount_drive`, `_copy_*`/`_extract_*` resume helpers, `capture_rng_state`, `restore_rng_state`, `hash_config`, `load_latest_checkpoint`, `prune_checkpoints` | Reused; **`atomic_save_checkpoint` gets a new `update_latest` kwarg** (§10); **new `stage_unlabeled_squat_videos`** added |
| `harness/_envinit.py` | `CUBLAS_WORKSPACE_CONFIG` set before torch | UNCHANGED — notebook Step 0 imports it first |
| `harness/supervised_train.py` | `seed_worker`, `_set_global_seed`, `_build_dataloaders` pattern, `_val_pass`, checkpoint payload schema, early-stop logic, resume branch | **Patterns transfer to `md_finetune.py` — do NOT edit `supervised_train.py`** (D9) |
| Notebook Cell A + Step 0 | clone-or-pull, `sys.path.insert`, PyAV-before-torch, dep probe, L4 assert, Drive mount, idempotent staging | Pattern transfers to `04_squat_md_ssl.py` |

### Genuinely NEW code

| New module | Responsibility | Built from |
|------------|----------------|------------|
| `datasets/squat_ssl.py` (or `datasets/ssl_squat.py`) | Unlabeled SSL dataset: load trajectory → `split_half_cycles` (§1) → build (anchor, positive, negative) triplet with temporal reversal (§3) → per-branch SSL augs (§7) → `decode_clip` + `spatial_*`. Returns `{anchor, positive, negative}` dict of 3 clips. No labels. Excludes val/test keys + traj_nan (per `dataloader.py:67–74`). | NEW; reuses `decode_clip`/`uniform_sample_indices`/`spatial_*` |
| `datasets/ssl_augs.py` (or inline) | The 8-augmentation set (§7) on `[T,3,H,W]` uint8, applied per-branch independently | NEW; primitives informed by `image_augmentations.py` |
| `harness/md_pretrain.py` | `MDConfig`, `build_md_model` (backbone fc=Identity + `ProjectionHead`), `md_triplet_loss` (§2), `run_md_pretrain_epoch` (3-branch forward, Eq.1 loss, atomic checkpoint, resume, linear-probe every N epochs, collapse detection), `_linear_probe` helper | NEW; mirrors `supervised_train.py` structure + the official `train_test.py` loss/loop |
| `harness/md_finetune.py` | `FinetuneConfig` (AdamW, wd=1e-4, dropout=0.2, lr=1e-4, 50ep/8-patience), `build_finetune_model` (load MD backbone, fresh Dropout+Linear head), `run_md_finetune_epoch` (parallels `run_supervised_epoch` + D6 runtime overfit monitor) | NEW; parallels `supervised_train.py` with the D3 deltas |
| `eval/ensemble.py` | `aggregate_sigmoid_mean(per_seed_scores)` → mean-of-sigmoids across seeds; single per-error threshold on ensemble val scores (D4) | NEW; feeds `eval/metrics.py` |
| `eval/tta.py` | `tta_forward(model, clip, recipe)` → augmented copies (temporal jitter / spatial 5-crop / flip), average sigmoids; `select_tta_recipe(val_scores)` → val-tuned combo (D5) | NEW; reuses `transforms.py` + `spatial_*` |
| `notebooks/04_squat_md_ssl.py` + `.ipynb` | The phase notebook (Cell A + Step 0 + SSL steps + fine-tune steps + ensemble/TTA + figures) | NEW; Cell A/Step 0 pattern from `03_*.py` |

### Decode throughput flag

The SSL pretrain decodes **4,970 clips × ~20–60 epochs** = far more decode than Phase 3. If the epoch-0 timing probe (§5) shows GPU < 50% util (decode-bound), the **TorchCodec swap** (`decode_clip` is a documented one-line swap point, transforms.py docstring) is the deferred remedy (CONTEXT defers it; flag if it bites). `[ASSUMED — flagged per CONTEXT deferral]`

---

## §12 — Representation Collapse Detection (Validation-Critical)

Representation collapse (all clips → same embedding) is the classic contrastive failure (CONTEXT "Validation Architecture" requirement; handoff risk #1). The triplet distance-ratio loss is **less collapse-prone than NT-Xent** (the explicit negative provides a repulsion signal), but collapse can still occur if augmentations are too weak or the projector saturates. **Detection is mandatory.**

### Metrics (computed every SSL epoch on a fixed probe batch)

1. **Embedding standard deviation** (the SimSiam/BYOL-standard collapse monitor): for a fixed batch of N clips, compute the L2-normalized **backbone** embeddings `z ∈ [N, 512]`, then `std_per_dim = z.std(dim=0)`; report `mean(std_per_dim)`. For L2-normalized d-dim embeddings, healthy ≈ `1/√d` per dim; **collapse ⇒ → 0**. Log `embedding_std` per epoch. [VERIFIED: collapse-detection literature — embedding variance is the standard collapse signal; arXiv:2209.15007, arXiv:2110.09348]
2. **Effective rank / dimensional collapse:** compute the singular values of the embedding covariance; a sharp drop (most variance in few dims) signals **dimensional collapse** (a partial collapse where embeddings live in a low-rank subspace). Report `effective_rank = exp(entropy(normalized_singular_values))` or simply the number of singular values above a threshold. [VERIFIED: arXiv:2110.09348 "Understanding Dimensional Collapse in Contrastive SSL"]
3. **Linear-probe F1 (§6):** the **downstream-relevant** collapse signal — if probe-F1 is flat at the random baseline (KIE ≈ pos-rate, KFE ≈ pos-rate F1) while loss decreases, the representation is collapsed *for the task*. This is the most actionable signal for a defense.

### Abort/remediation

- `embedding_std → 0` (e.g., < 0.1/√512) **before** epoch 10 ⇒ collapse ⇒ **abort**, increase augmentation strength (§7) or reduce LR. Document.
- This is the SSL analog of D6's fine-tune overfit monitors. The planner should add it to the `<determinism_checklist>` / `<ssl_health_checklist>` extension.

---

## §13 — Ensemble (D4) — Implementation Notes

CONTEXT D4 fully settles the policy. Implementation specifics for `eval/ensemble.py`:

- **3 seeds (42/1337/7), shared MD `backbone.pt`, 3 fine-tunes** (`md_finetune_seed{42,1337,7}/best.pt`). Run sequentially on one L4 (D6 reg-bump policy: seed 42 first, lock recipe).
- **Aggregation = mean of sigmoid scores per error head across seeds.** `ensemble_score[clip, head] = mean_over_seeds(sigmoid(logit_seed[clip, head]))`. This is in [0,1] → satisfies `pr_auc_per_error`'s assertion. [D4]
- **Single per-error threshold tuned on ensemble val scores** via `threshold_sweep` (eval/metrics.py) — NOT per-seed thresholds then vote. [D4]
- **Per-seed F1 also reported** (variance evidence) — proves the ensemble lift is real (Lakshminarayanan et al. 2017 deep-ensembles rationale, per D4). [D4]
- **Aggregation order with TTA (D5):** per-seed → average its TTA copies → mean-of-sigmoids across seeds → val-tuned per-error threshold. Mathematically a grand mean over all seed×TTA score variants.

---

## §14 — TTA (D5) — Implementation Notes

CONTEXT D5 settles the policy (val-tuned; flip OOD-corrected). Specifics for `eval/tta.py`:

- **Candidate augmentations:** temporal jitter (±2 frames — *in-distribution*, Phase 3 trained jitter ±2), spatial 5-crop (center + 4 corners — *in-distribution*, Phase 3 trained random-crop), horizontal flip (**OOD** — Phase 3 trained flip OFF). [D5]
- **Recipe selection = val-tuned:** for each candidate combo, compute ensemble val macro-F1; keep the combo that maximizes it; apply to test. **Flip is a candidate to validate, not an assumed-good default** — TTA only reduces variance for augmentations the model is invariant to, and invariance comes from training (D5). [D5]
- **Aggregation = mean of sigmoids across TTA copies** (per seed), then ensemble across seeds.
- **Report BOTH** the full ensemble+TTA headline F1 AND a single-model/no-TTA number, so Phase 5 can pick its inference protocol against its latency budget (D5 + handoff prerequisite). Note the TTA × ensemble forward-pass cost (e.g., 3 seeds × N_tta copies = up to 3×(1+jitter+5crop+flip) forward passes per clip).
- **Spatial 5-crop note:** `spatial_val` does a center crop; 5-crop needs 4 corner crops added — extend in `tta.py`, do NOT modify `transforms.py` (D9). `[ASSUMED — 5-crop extension is standard; reuses resize + manual corner offsets]`

---

## §15 — Validation Architecture (Nyquist)

**Required — `workflow.nyquist_validation: true` in `.planning/config.json`.**

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.0.0+ (already in `backend/requirements.txt`) |
| Config file | none — inline `python -m pytest` (Phase 3 convention) |
| Quick run command | `python -m pytest backend/training/aqa/harness/test_md_pretrain.py backend/training/aqa/eval/test_ensemble.py -x -q` |
| Full suite command | `python -m pytest backend/training/aqa/ -x -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SQUAT-04-a | `split_half_cycles` returns two 16-frame index sets, descent ends ≤ bottom ≤ ascent start, on a synthetic parabola | unit | `pytest .../test_md_pretrain.py::test_half_cycle_split -x` | ❌ Wave 0 |
| SQUAT-04-b | `md_triplet_loss` matches a hand-computed value on known embeddings (both 2-term and 3-term, squared) | unit | `pytest .../test_md_pretrain.py::test_triplet_loss_known -x` | ❌ Wave 0 |
| SQUAT-04-c | `md_triplet_loss` is lower when anchor≈positive and far from negative than the reverse (directionality) | unit | `pytest .../test_md_pretrain.py::test_triplet_loss_direction -x` | ❌ Wave 0 |
| SQUAT-04-d | `ProjectionHead` output is L2-normalized (norm ≈ 1) and shape `[B, 128]` | unit | `pytest .../test_md_pretrain.py::test_projector_l2norm -x` | ❌ Wave 0 |
| SQUAT-04-e | `build_md_model` backbone has `fc==Identity`, projector attached, 512-dim features | unit (slow) | `pytest .../test_md_pretrain.py::test_md_model_build -x` | ❌ Wave 0 |
| SQUAT-04-f | SSL checkpoint payload has all required keys (backbone/projector/optimizer/scheduler/rng/metrics/linear_probe) | unit | `pytest .../test_md_pretrain.py::test_ssl_checkpoint_schema -x` | ❌ Wave 0 |
| SQUAT-04-g | Trajectory-format probe: per-clip JSON loads as flat float list; len mapping reported | integration (Colab) | Colab paste-back (the first `checkpoint:human-verify` probe) | ❌ Wave 1 |
| SQUAT-04-h | SSL loss decreases over first 5 epochs; `embedding_std` stays > 0.1/√512 (no collapse) | integration (Colab) | Colab paste-back of epoch-5 SSL metrics | ❌ Wave 2 |
| SQUAT-04-i | Linear-probe macro-F1 rises above random baseline by epoch 10 (SSL is learning) | integration (Colab) | Colab paste-back of linear-probe curve | ❌ Wave 2 |
| SQUAT-05-a | `build_finetune_model` loads MD backbone (not Kinetics), head = `Dropout(0.2)+Linear(512,2)` | unit (slow) | `pytest .../test_md_finetune.py::test_finetune_model_build -x` | ❌ Wave 0 |
| SQUAT-05-b | `aggregate_sigmoid_mean` averages per-seed sigmoids correctly; output ∈ [0,1] | unit | `pytest .../test_ensemble.py::test_sigmoid_mean -x` | ❌ Wave 0 |
| SQUAT-05-c | `select_tta_recipe` returns the val-F1-maximizing combo on synthetic scores | unit | `pytest .../test_tta.py::test_recipe_selection -x` | ❌ Wave 0 |
| SQUAT-05-d | Fine-tune val macro-F1 ≥ Phase 3 best (0.5454) — SSL transferred | integration (Colab) | Colab paste-back of fine-tune curve | ❌ Wave 3 |
| SQUAT-05-e | Ensemble+TTA test macro-F1 ≥ Phase 3 (0.543); KIE F1 NOT < 0.2857 (D6 monitor 3) | integration (Colab) | Colab paste-back of final test report | ❌ Wave 4 |
| SQUAT-05-f | val-test F1 gap < 0.05 (D6 monitor 2) | integration (Colab) | Colab paste-back | ❌ Wave 4 |

**Note on granularity:** unit tests (pytest) cover pure functions (half-cycle splitter on synthetic data, triplet loss arithmetic, projector L2-norm, ensemble/TTA aggregation, checkpoint schema). Training convergence, collapse, linear-probe, and test-F1 assertions happen **interactively in Colab via paste-back** (D7 one-runnable-unit-at-a-time). The triplet-loss and half-cycle-splitter unit tests are **load-bearing** — they let us verify the reconstructed-from-paper logic *before* burning 12–24h of GPU on it.

### Sampling Rate

- **Per SSL batch:** triplet loss logged.
- **Per SSL epoch:** mean SSL loss + `embedding_std` + `effective_rank` (collapse monitor §12) into `metrics_history`.
- **Every 5 SSL epochs:** linear-probe macro-F1 (§6) into `linear_probe_history`.
- **Per fine-tune epoch:** train/val BCE loss + val F1/PR-AUC + **D6 runtime monitor (train/val loss ratio)**.
- **Post-fine-tune:** threshold sweep on val; per-seed test F1.
- **Phase gate:** ensemble+TTA test F1, val-test gap, KIE-no-regress check, all 9 figures, `results.pkl` present before `/gsd:verify-work`.

### Aliasing Guards

- **SSL `embedding_std` / linear-probe F1:** computed on a **fixed probe batch / the full labeled subset** at once — no per-batch streaming (same anti-aliasing reasoning as Phase 3 §8; 14% KIE positive rate aliases on small batches).
- **Fine-tune val/test F1:** full-pass gather then single sklearn call (carry `_val_pass` from `supervised_train.py`).
- **Ensemble/TTA:** all scores gathered before threshold sweep — no streaming.

### Wave 0 Gaps

- [ ] `backend/training/aqa/datasets/squat_ssl.py` — SSL dataset + half-cycle splitter
- [ ] `backend/training/aqa/datasets/ssl_augs.py` (or inline) — SSL augmentations
- [ ] `backend/training/aqa/harness/md_pretrain.py` — SSL trainer + triplet loss + projector + linear-probe + collapse detection
- [ ] `backend/training/aqa/harness/md_finetune.py` — fine-tune trainer (AdamW + dropout, MD init)
- [ ] `backend/training/aqa/eval/ensemble.py` — mean-of-sigmoids aggregation
- [ ] `backend/training/aqa/eval/tta.py` — TTA forward + recipe selection
- [ ] `backend/training/aqa/harness/test_md_pretrain.py` — unit tests (splitter, loss, projector, schema)
- [ ] `backend/training/aqa/harness/test_md_finetune.py` — unit tests (model build)
- [ ] `backend/training/aqa/eval/test_ensemble.py` + `test_tta.py` — unit tests
- [ ] `backend/training/aqa/harness/colab.py` — add `update_latest` kwarg to `atomic_save_checkpoint` + `stage_unlabeled_squat_videos`

---

## §16 — Comparison-Chart Numbers (Headline Deliverable, D8) + Arithmetic Check

The headline is the Phase-3-vs-Phase-4 bar chart (D8) placing our F1 against Parmar and GYMetricPose on identical metrics. **The plan-checker MUST verify these expected values are arithmetically correct** (Phase 3's plan had wrong F1 arithmetic — CONTEXT canonical_refs note).

**Verified from paper Table 2 (p.12), Squat KIE/KFE F-score (read directly from the PDF):**

| Model (Table 2) | KIE F1 | KFE F1 | macro F1 = (KIE+KFE)/2 |
|-----------------|--------|--------|------------------------|
| OpenPose-TDM [2,27] | 0.4143 | 0.8123 | **0.6133** |
| ImageNet | 0.1923 | 0.7725 | **0.4824** |
| SimSiam | 0.2270 | 0.7868 | **0.5069** |
| Ours CVCSPC (image) | 0.5195 | 0.8286 | **0.6741** |
| **Kinetics (video, R(2+1)D-18) — Phase 3's comparison row** | **0.2970** | **0.8184** | **0.5577** |
| TemporalXform | 0.3414 | 0.8319 | **0.5867** |
| **Ours MD (video) — Phase 4's target row** | **0.4186** | **0.8338** | **0.6262** |
| Ours MD + CVCSPC | 0.5263 | 0.8468 | **0.6866** |

[CITED: Parmar Table 2 p.12 — values read directly from the PDF]

**Arithmetic verification (the macro = mean of the two):**
- Kinetics macro = (0.2970 + 0.8184)/2 = **0.5577** ✓ (CONTEXT's "~0.557" ✓)
- MD macro = (0.4186 + 0.8338)/2 = **0.6262** ✓ (CONTEXT's "~0.626" ✓)
- Phase 3 result macro = (0.2857 + 0.8000)/2 = **0.5429** ✓ (matches 03-SUMMARY's 0.543 ✓)

**Phase 4 success arithmetic (the bars to beat):**
- ROADMAP minimum: Phase 4 macro **≥ 0.5429** (Phase 3 baseline).
- Paper MD target: Phase 4 macro **≈ 0.6262** (KIE 0.4186 / KFE 0.8338).
- D6 monitor 3: Phase 4 KIE F1 **≥ 0.2857** (no regression below Phase 3 KIE).
- The paper's MD lift over its OWN Kinetics row: KIE +0.122 (0.2970→0.4186), KFE +0.015 (0.8184→0.8338), macro +0.069 (0.5577→0.6262). **Our Phase 3 Kinetics ≈ paper's Kinetics; so if MD-SSL works, expect a similar lift over our 0.5429.**

**GYMetricPose (Gallardo 2024) + LMM (Dibenedetto 2025)** are independent comparison points for the chart (CONTEXT canonical_refs). Their exact KIE/KFE numbers are **not in this repo's paper PDFs** — the `FLEX Dataset Paper.pdf` and `Fine-Tuning Large Multimodal Models...pdf` at repo root may carry them. **Flag for the planner:** if the chart needs GYMetricPose/LMM bars, those numbers must be sourced from those papers (a separate small read task) — they are NOT in the Parmar paper. `[ASSUMED — GYMetricPose numbers not yet extracted; flag as a chart-data task]`

---

## Architecture Patterns

### System Architecture Diagram (SSL pretrain → fine-tune → ensemble+TTA → eval)

```
                        ┌─────────────────────────────────────────────────┐
 Drive: Unlabeled       │  stage_unlabeled_squat_videos (NEW, colab.py)    │
 videos.zip (4,970)  ──►│  + bar_trajectories_raw.zip → /content/ local    │
 + trajectories         └───────────────────────┬─────────────────────────┘
                                                 ▼
                  ┌──────────────────────────────────────────────────────────┐
                  │  squat_ssl.py  __getitem__  (NEW)                          │
                  │   trajectory(JSON) → split_half_cycles (§1, scipy)         │
                  │   → (anchor=descent, neg=ascent); positive=aug(anchor)     │
                  │   → temporal-reverse {anc,pos} OR {neg} (§3)               │
                  │   → per-branch SSL augs (§7) → decode_clip → spatial_*     │
                  │   returns {anchor[3,16,H,W], positive[...], negative[...]} │
                  └───────────────────────┬──────────────────────────────────┘
                                          ▼ (batch 8)
       ┌──────────────────────────────────────────────────────────────────────┐
       │  md_pretrain.run_md_pretrain_epoch (NEW)                               │
       │   R(2+1)D-18(fc=Identity) → 512-d feat → ProjectionHead → L2-norm      │
       │   md_triplet_loss(φ_anc, φ_pos, φ_neg)  [Eq.1, §2]  → AdamW            │
       │   per epoch: SSL loss, embedding_std + rank (collapse §12)             │
       │   every 5 ep: linear-probe macro-F1 (§6) ── convergence signal        │
       │   atomic checkpoint every epoch (resume; map_location=cpu; §10)        │
       └───────────────────────┬──────────────────────────────────────────────┘
                                ▼  backbone.pt (linear-probe-best)
       ┌──────────────────────────────────────────────────────────────────────┐
       │  md_finetune.run_md_finetune_epoch (NEW, ×3 seeds 42/1337/7)           │
       │   load MD backbone (NOT Kinetics) + fresh Dropout(0.2)+Linear(512,2)   │
       │   labeled pipeline (32-frame, jitter±2, crop112, flip OFF)             │
       │   AdamW lr=1e-4 wd=1e-4, BCEWithLogitsLoss(pos_weight), 50ep/8-patience│
       │   D6 runtime monitor: train/val loss ratio; abort+bump on seed 42      │
       └───────────────────────┬──────────────────────────────────────────────┘
                                ▼  best.pt × 3
       ┌──────────────────────────────────────────────────────────────────────┐
       │  tta.py (NEW): per-seed → val-tuned augs (jitter/5-crop/flip) → mean σ │
       │  ensemble.py (NEW): mean-of-sigmoids across 3 seeds → single threshold │
       │  eval/metrics.py (UNCHANGED): threshold_sweep / f1 / pr_auc / confusion│
       └───────────────────────┬──────────────────────────────────────────────┘
                                ▼
       figures/ (D8): SSL loss curve, linear-probe curve, per-seed curves,
       ensemble-vs-single, **Phase3-vs-Phase4 bar chart**, confusion, PR,
       sample predictions, results.pkl
```

### Recommended Project Structure (additions only)

```
backend/training/aqa/
├── datasets/
│   ├── squat_ssl.py          # NEW — SSL triplet dataset + half-cycle splitter
│   └── ssl_augs.py           # NEW (or inline in squat_ssl) — 8 SSL augmentations
├── harness/
│   ├── md_pretrain.py        # NEW — SSL trainer + triplet loss + projector + linear-probe + collapse
│   ├── md_finetune.py        # NEW — fine-tune trainer (AdamW+dropout, MD init)
│   ├── colab.py              # EDIT — add update_latest kwarg + stage_unlabeled_squat_videos
│   ├── test_md_pretrain.py   # NEW
│   └── test_md_finetune.py   # NEW
├── eval/
│   ├── ensemble.py           # NEW — mean-of-sigmoids
│   ├── tta.py                # NEW — TTA forward + recipe selection
│   ├── test_ensemble.py      # NEW
│   └── test_tta.py           # NEW
└── notebooks/
    ├── 04_squat_md_ssl.py    # NEW — jupytext source
    └── 04_squat_md_ssl.ipynb # NEW — Colab pair (Cell A baked in)
```

### Pattern 1: Triplet forward (3 branches share one backbone)

```python
# Source: official train_test.py:51-53 (one backbone applied 3×) + §2/§4 of this doc
def forward_triplet(backbone, projector, anchor, positive, negative):
    feat = lambda x: projector(backbone(x))     # backbone fc=Identity → 512-d → projector → L2-norm
    return feat(anchor), feat(positive), feat(negative)   # each [B,128], L2-normed
loss = md_triplet_loss(*forward_triplet(backbone, projector, a, p, n))
```

### Pattern 2: Backbone-only transfer to fine-tune (discard projector)

```python
# Source: §4/§9 — load only backbone, fresh head, NOT Kinetics
ckpt = torch.load(".../md_pretrain_v1/backbone.pt", map_location="cpu", weights_only=False)
model = r2plus1d_18(weights=None)
assert model.fc.in_features == 512
model.load_state_dict(ckpt["backbone_state_dict"], strict=False)  # projector keys absent
model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 2))      # D3 head
```

### Anti-Patterns to Avoid

- **Replacing Eq. 1 with NT-Xent "because it's modern."** The paper uses Eq. 1; deviating breaks the faithful-reproduction defense. NT-Xent is only a documented fallback if Eq. 1 fails to converge.
- **Adding in-batch negatives / a memory bank.** MD uses one structural negative per anchor (§3). In-batch negatives are a *different method*.
- **Using large batches "because contrastive needs them."** That's NT-Xent's requirement. MD trains at batch 5 (§5). Batch 8 is the recommendation; gradient accumulation is unnecessary.
- **Loading Kinetics weights into the fine-tune model.** The MD backbone IS the init (§9). Loading Kinetics would discard the SSL contribution.
- **Bloating `supervised_train.py`.** `md_finetune.py` is a NEW module (D9).
- **Letting `atomic_save_checkpoint(backbone.pt)` clobber `latest.txt`** — would corrupt SSL resume (§10; D7).
- **Mean-of-logits in the ensemble.** D4 locks mean-of-sigmoids (and `pr_auc_per_error` asserts [0,1] — logit means would fail the assertion).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| F1 / PR-AUC / threshold sweep | Custom metric math | `eval/metrics.py` (UNCHANGED, D9) | Phase 3 already verified against sklearn; the ensemble/TTA arrays feed straight in |
| Video decode + frame sampling | New decoder for SSL | `decode_clip` + `uniform_sample_indices` (transforms.py) | Window-bounded, OOM-safe, TorchCodec swap point already wired |
| Trajectory smoothing / peak detection | Hand-rolled smoothing/argmax loops | `scipy.ndimage.gaussian_filter1d`, `scipy.signal.find_peaks`/`savgol_filter`, `np.argmax` | scipy is already a dependency; battle-tested |
| L2 normalization | Manual `x / x.norm()` | `torch.nn.functional.normalize(x, dim=-1, p=2)` | Numerically stable; matches official code exactly (train_test.py:51) |
| Atomic checkpoint + resume + RNG | New save/load | `colab.py` primitives (+ `update_latest` kwarg) | Phase 2/3 contract; Drive-FUSE-safe; carries the `map_location='cpu'` fix |
| Per-session clip staging | New copy/extract loop | `_copy_with_resume_and_progress` + `_extract_with_resume_and_progress` (parameterize) | 3-layer disconnect-safe resume already built |
| AdamW weight decay | Manual L2 in the loss | `torch.optim.AdamW(weight_decay=1e-4)` | Decoupled weight decay is the whole point of the Adam→AdamW change (D3) |
| Cosine LR schedule | Manual LR math | `torch.optim.lr_scheduler.CosineAnnealingLR` | Phase 3 contract |

**Key insight:** the **only** genuinely novel logic in this phase is (1) the half-cycle splitter (§1) and (2) the triplet loss + 3-branch forward (§2). Both are small, both have unit-testable pure-function cores, and both must be verified on synthetic data **before** the 12–24h GPU burn. Everything else is reuse or parameterization of Phase 2/3 contracts.

## Common Pitfalls

### Pitfall 1: Representation collapse (the classic contrastive failure)
**What goes wrong:** all clips map to (nearly) the same embedding; SSL loss looks like it's decreasing but the backbone learns nothing.
**Why it happens:** augmentations too weak (positive ≈ anchor trivially), projector saturates, or LR too high early.
**How to avoid:** monitor `embedding_std` and linear-probe F1 every epoch/5-epochs (§12, §6). The triplet's explicit negative makes full collapse less likely than NT-Xent, but dimensional collapse is still possible.
**Warning signs:** `embedding_std → 0`; linear-probe F1 flat at random baseline while loss drops.

### Pitfall 2: Half-cycle split sign error (argmax vs argmin)
**What goes wrong:** descent and ascent are swapped, or the split lands mid-motion, because the y-coordinate sign convention (image-y-down vs amplitude-up) is guessed wrong.
**Why it happens:** Fig. 3 plots "amplitude" (possibly inverted from raw pixel y); the trajectory could be either sign.
**How to avoid:** resolve the sign **empirically on 5 sample trajectories** in the first execution task (a `checkpoint:human-verify`), before writing the dataset class (§1 point 3).
**Warning signs:** split point not near the visual rep-bottom; descent/ascent frames look reversed in a sample-frame grid.

### Pitfall 3: Trajectory↔video frame-index mismatch
**What goes wrong:** trajectory indices (26–441 samples) are used directly as video frame indices (49–404 frames) — they're not 1:1 (Phase 1).
**Why it happens:** assuming the trajectory is per-video-frame when it's per-detection-sample.
**How to avoid:** probe `len(traj)` vs `read_video_timestamps(video)` count for the same clip; derive the mapping (§1 point 5, §8 point 4).
**Warning signs:** decoded half-cycle frames don't match the trajectory's descent/ascent phase.

### Pitfall 4: `latest.txt` clobbered by the backbone-best write
**What goes wrong:** SSL resume after a disconnect jumps to the wrong epoch or fails, because `atomic_save_checkpoint(backbone.pt)` overwrote `latest.txt` to point at `backbone.pt`.
**Why it happens:** `atomic_save_checkpoint` always writes `latest.txt` last (colab.py:589) with no opt-out.
**How to avoid:** add `update_latest=False` kwarg and use it for the backbone-best write (§10; D7).
**Warning signs:** `latest.txt` contains `backbone.pt` instead of `epoch_NNN.pt`; resume loads the best epoch's state but `start_epoch` is wrong.

### Pitfall 5: Module-stale-after-git-pull (carry Phase 3)
**What goes wrong:** a trainer fix is pulled into a running Colab kernel but the old code still runs (`sys.modules` cached).
**Why it happens:** `git pull` doesn't reload imported modules ([[reference_colab_module_reload_after_git_pull]]).
**How to avoid:** **restart the Colab runtime BEFORE any major trainer change** (D7); or `importlib.reload`. This is doubly important for the SSL trainer (a stale loss = 12–24h wasted).
**Warning signs:** a fix doesn't take effect; behavior matches the pre-fix code.

### Pitfall 6: Strong augmentations destroy the form-error signal
**What goes wrong:** aggressive rotation/zoom makes the knee-valgus (KIE) or knee-forward (KFE) cue invariant, so the SSL representation can't distinguish good from bad form, and linear-probe F1 stays low.
**Why it happens:** "strong augs help SSL" applied without the form-signal caveat (CONTEXT D2's explicit tension).
**How to avoid:** conservative augmentation subset (§7); treat rotation as an ablation toggle; verify magnitudes by linear-probe delta.
**Warning signs:** linear-probe KIE F1 doesn't rise above ~0.15 (the positive rate) even as SSL loss drops.

## Runtime State Inventory

> N/A for the SSL/training portion (greenfield code under `backend/training/aqa/`). The only stateful artifacts are Drive checkpoints (new run names under `phase04/`, no collision with `phase03/`) and locally-staged clips (`/content/`, ephemeral). No rename/refactor/migration. **Nothing to inventory — verified: this phase adds new modules + new Drive run dirs, touches no existing runtime state, and does not modify the FastAPI runtime (Phase 5).**

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| L4 GPU (24 GB) | SSL pretrain + fine-tune | ✓ (user on fresh L4 per CONTEXT) | 24 GB | None — required ([[feedback_heavy_training_new_notebook]]) |
| PyTorch + torchvision | All training | ✓ (Colab) | Colab-pinned | None |
| scipy | Half-cycle splitter | ✓ | ≥1.11.0 (requirements.txt) | numpy `argmax` for the extremum (no smoothing) |
| scikit-learn | metrics + linear-probe | ✓ | ≥1.3.0 | None |
| PyAV | decode | ✓ (Step 0 installs) | Colab | None |
| Drive: Squat `Unlabeled_Dataset/videos.zip` (4,970) | SSL pretrain | ⚠ on user's Drive — NOT verifiable from this machine | — | None — **probe at session start** (§8) |
| Drive: `bar_trajectories_raw.zip` (4,970) | half-cycle split | ⚠ on user's Drive — NOT verifiable here | — | None — **probe format at session start** (§8) |

**Missing dependencies with no fallback:** the unlabeled videos + trajectories are on the user's Drive and **cannot be verified from this machine** — the first execution task MUST `ls` and probe them (§8). This is the phase's single hard external precondition.

## State of the Art

| Old Approach (CONTEXT provisional) | Current Approach (paper-confirmed) | Source | Impact |
|------------------------------------|-------------------------------------|--------|--------|
| "SimCLR-family NT-Xent / InfoNCE default" | **Triplet distance-ratio loss (Eq. 1, Hoffer & Ailon 2015)** | Paper §3.2 p.7 + train_test.py:68 | Removes the largest `[ASSUMED]`; defines exact loss |
| "in-batch / memory-bank negatives" | **One structural negative per anchor (the opposite half-cycle)** | Paper §3.2 p.7 | Batch 5 works; no large-batch need |
| "SimCLR batch 256+, gradient accumulation" | **Batch 5 (paper) / 8 (recommended) — fits L4** | Paper §5 p.9 + Phase 3 VRAM math | No grad-accum; simpler |
| "LARS optimizer + warmup" | **AdamW lr=1e-4, no LARS, no warmup** | Paper §5 (Adam 1e-4) + CONTEXT D2 (AdamW) | Standard small-batch optimizer |
| "lower downstream fine-tune LR" | **1e-4 (paper specifies no separate transfer LR)** | Paper §5 p.9 | Keep 1e-4 (§9) |

**Deprecated/outdated for this phase:**
- The CONTEXT.md provisional "NT-Xent default" — superseded by the paper's Eq. 1 (this is expected; CONTEXT D2 explicitly delegated the confirmation to research).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Half-cycle split = global extremum (`argmax`/`argmin`) of smoothed y; sign resolved empirically | §1 | If sign wrong, descent/ascent swap — caught by the 5-clip `checkpoint:human-verify` before any GPU burn |
| A2 | Multi-rep handling = take first rep via `find_peaks`; dataset is mostly single-rep (paper claim) | §1 | If many multi-rep clips, triplets impure — measured in first task (count >1-extremum trajectories) |
| A3 | Trajectory↔video frame mapping is linear / per-frame | §1, §8 | If wrong, half-cycle frames misaligned — probed in first task |
| A4 | Loss = squared L2, 3-term denominator (matches official code, not exactly Eq. 1) | §2 | Both forms exposed as config flags; ablatable; documented discrepancy |
| A5 | Projection head = 2-layer MLP 512→512→128, BN+ReLU, L2-norm output | §4 | Exact dims unknown (supplementary); L2-norm is CITED; dims are SimCLR-standard; low risk |
| A6 | Projector discarded at fine-tune (backbone-only transfer) | §4, §9 | Universal SSL practice + paper's end-to-end fine-tune; very low risk |
| A7 | Batch 8 on L4 (5 paper-faithful); ~7.7–11.7 GB; no grad-accum | §5 | VRAM extrapolated from Phase 3's measured 15.22 GB — re-probe in-notebook (Phase 3 estimate was off) |
| A8 | SSL epochs 20 baseline, extend to linear-probe convergence ≤60 | §5, §6 | If 20 is enough, faster; linear-probe plateau is the principled stop |
| A9 | Cosine scheduler, no warmup, wd=1e-4 (SSL) | §5 | Paper silent; SimCLR-standard; low risk |
| A10 | Linear-probe every 5 epochs on backbone features, 32-frame val transform | §6 | Cadence is a judgment call; N=10 acceptable; affects only monitoring granularity |
| A11 | Conservative augmentation subset (rotation as ablation toggle) | §7 | Magnitudes unknown (supplementary); justified by form-signal preservation; verified by linear-probe |
| A12 | Trajectory format = per-clip JSON, flat float list (per CVCSPC code + Phase 1) | §8 | Probed in first task; CVCSPC code strongly suggests this |
| A13 | Fine-tune LR = 1e-4 (paper specifies no separate transfer LR) | §9 | CITED — paper has only 1e-4; the "keep it" verdict is the assigned confirmation |
| A14 | `update_latest=False` kwarg added to `atomic_save_checkpoint` | §10 | Small isolated change; CONTEXT D7 endorses it |
| A15 | Unlabeled set = 4,970 clips, ~5–15 GB on disk | §10 | Count CITED (Phase 1); size probed at stage time; 3-layer resume handles partial |
| A16 | GYMetricPose/LMM chart numbers must be sourced from their own papers (not in Parmar) | §16 | Flagged as a separate chart-data task; absence doesn't block the core P3-vs-P4 bar |
| A17 | `embedding_std` collapse threshold ≈ 0.1/√512 before epoch 10 | §12 | Heuristic; the linear-probe F1 is the more actionable co-signal |

**These `[ASSUMED]` items signal to the planner and execute-phase which decisions need confirmation. A1/A3/A12 are resolved by the first execution probe (cheap, gated as `checkpoint:human-verify`). A4/A5/A11 are exposed as config flags / ablation toggles. A7 is re-probed by the VRAM cell. None should be presented to examiners as paper-verified.**

## Open Questions for the Planner

1. **Trajectory format probe is task #1 (gated).** The SSL dataset class cannot be finalized until the format (§8) and the half-cycle sign (§1) are confirmed on 2–5 real clips. **Sequence the plan so the probe + sign-check is a `checkpoint:human-verify` BEFORE `squat_ssl.py` is written.** This is the single highest-leverage sequencing decision.
2. **Loss form (2-term vs 3-term, squared vs not):** recommend matching the official code (3-term, squared) as primary, with config flags. The planner should make this a `MDConfig` field, not a hard-coded choice, so the exact-Eq.1 form is ablatable if convergence is poor (A4).
3. **`backbone.pt` selection metric:** linear-probe-macro-F1-best epoch (§6/§10) vs final epoch. Recommend linear-probe-best (the principled "SSL learned the most" point). Planner confirms.
4. **GYMetricPose/LMM chart bars:** if the headline chart needs them (CONTEXT canonical_refs lists them), add a small read task against `FLEX Dataset Paper.pdf` / the LMM paper at repo root — those numbers are NOT in Parmar (A16). The core P3-vs-P4-vs-Parmar bar does not need them.
5. **SSL run on a fresh notebook:** per [[feedback_heavy_training_new_notebook]], confirm the user is on a fresh L4 before the pretrain task. (CONTEXT says they are.)

## Code Examples

### Verified: L2-normalized triplet embeddings + distance-ratio loss
```python
# Source: train_test.py:51-68 (official author code) + paper Eq. 1 §3.2 p.6
import torch, torch.nn.functional as F
def md_triplet_loss(phi_a, phi_p, phi_n, *, squared=True, three_term=True, eps=1e-9):
    d_ap = ((phi_a - phi_p) ** 2).sum(-1); d_an = ((phi_a - phi_n) ** 2).sum(-1)
    if not squared: d_ap, d_an = d_ap.sqrt(), d_an.sqrt()
    num = torch.exp(-d_ap); den = num + torch.exp(-d_an)
    if three_term:
        d_pn = ((phi_p - phi_n) ** 2).sum(-1)
        den = den + torch.exp(-(d_pn if squared else d_pn.sqrt()))
    return (-torch.log(num / (den + eps))).mean()
# phi_* are F.normalize(projector(backbone(x)), dim=-1, p=2)  # L2-norm CITED train_test.py:51
```

### Verified: trajectory min-max normalize → phase (official code)
```python
# Source: train_test.py / dataloader.py:111-117 (official author code)
normed = [(t - min(traj)) / (max(traj) - min(traj)) for t in traj]   # [0,1]
phase  = [v * 360 for v in normed]                                    # degrees (paper: -180..180)
```

### Verified: R(2+1)D-18 backbone with fc=Identity (feature extractor)
```python
# Source: torchvision r2plus1d_18 (in_features=512 CITED, supervised_train.py:129)
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
assert backbone.fc.in_features == 512
backbone.fc = torch.nn.Identity()   # 512-d global-pooled features
```

## Security Domain

> `security_enforcement: true` in config, but this phase is **offline ML training in Colab** with **no authentication, no network-facing surface, no user input, no session/access-control, no cryptography, and no PII**. The data is the research-use Fitness-AQA dataset (non-commercial license — already governed by the milestone constraint). ASVS categories V2–V6 are **N/A** for this phase. The only adjacent concern is **supply-chain integrity** (no new pip packages; the Step 0 dep-version probe is the guard — see Package Legitimacy Audit) and **research-use licensing** (Fitness-AQA is non-commercial; this is training-only, no redistribution). Backend inference security is a **Phase 5** concern (when the model is served via FastAPI). No security controls are added or required in Phase 4.

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | N/A — offline training |
| V3 Session Management | no | N/A |
| V4 Access Control | no | N/A |
| V5 Input Validation | no | N/A (inputs are the fixed research dataset) |
| V6 Cryptography | no | N/A |
| Supply-chain | yes | Step 0 dep-version probe (no new packages) |

## Sources

### Primary (HIGH confidence)
- **Parmar, Gharat, Rhodin. *Domain Knowledge-Informed Self-Supervised Representations for Workout Form Assessment.* ECCV 2022. arXiv:2202.14019.** `Fitness-AQA/Domain Knowledge-Informed Self-Supervised.pdf` — read pp.1–18. §3.2 MD method (pp.6–8: half-cycle, anchor/pos/neg, temporal reverse, Eq.1 loss, 8 augmentations); §5 implementation details (p.9: R(2+1)D-18, Kinetics init, 16 frames/half-cycle, ADAM lr=1e-4, 20 epochs, batch 5, amplitude norm −180→180, end-to-end fine-tune); Table 2 (p.12: MD KIE 0.4186 / KFE 0.8338, Kinetics 0.2970/0.8184).
- **`Fitness-AQA-Code/Code_Release/.../train_test.py`** (official author code) — lines 51–68: L2-normalized embeddings + distance-ratio triplet loss (3-term denominator active, squared distance); Adam lr=1e-4.
- **`Fitness-AQA-Code/Code_Release/.../dataloader.py`** (official author code) — lines 35–39, 66–74, 107–117: per-clip JSON trajectory format, min-max normalize → ×360 phase, traj_nan + val/test exclusion.
- **`backend/training/aqa/{datasets,harness,eval}/*.py`** — Phase 2/3 contracts (decode_clip, spatial_*, atomic checkpoint, RNG, metrics, supervised_train trainer pattern, checkpoint payload schema).
- **`docs.pytorch.org` torchvision r2plus1d_18** — `in_features=512` (confirmed in supervised_train.py:129 assertion).
- **Hoffer & Ailon 2015, "Deep Metric Learning Using Triplet Network," arXiv:1412.6622** (paper ref [13]) — the distance-ratio loss provenance; WebSearch-verified ("SoftMax on both outputs to create a ratio measure").

### Secondary (MEDIUM confidence)
- **Phase 1 DATASET-REPORT** — trajectory format ("flat list of y-centers, 0% NaN, length 26–441"), clip-length distribution (49–404, median 111, multi-rep tail), 4,970 unlabeled count, traj_nan (19, labeled set).
- **Phase 3 SUMMARY** — measured VRAM (15.22 GB backward, batch 16/32-frame), baseline F1 (0.286/0.800/0.543), overfit pattern, carry-forward landmines.
- **Contrastive-collapse detection literature** (WebSearch-verified): embedding std + effective rank as collapse signals — arXiv:2110.09348 (dimensional collapse), arXiv:2209.15007 (non-contrastive collapse).
- **SimCLR (Chen et al. 2020)** — 2-layer projector + 128-d + discard-at-fine-tune conventions (for the `[ASSUMED]` projector dims).
- **Loshchilov & Hutter 2019, AdamW** — decoupled weight decay (for the Adam→AdamW rationale).

### Tertiary (LOW confidence — estimates / probe-pending)
- VRAM extrapolation (batch 5≈7.7 GB / 8≈11.7 GB): linear scaling from Phase 3's measured 15.22 GB — **re-probe in-notebook**.
- Wall-time (~15–35 min/epoch SSL): extrapolated from Phase 3's 5.3 min/epoch — **measure epoch 0**.
- Trajectory on-disk format / frame mapping / half-cycle sign — **probe-pending (task #1)**.
- Unlabeled set disk size (~5–15 GB) — **probe at stage time**.

## Metadata

**Confidence breakdown:**
- SSL loss (§2): **HIGH** — CITED from paper Eq. 1 + official code (the largest provisional assumption is now resolved).
- Sampling / triplet structure (§3): **HIGH** — CITED verbatim from §3.2 p.7.
- Half-cycle splitter (§1): **MEDIUM** — algorithm reconstructed; sign/format/mapping probe-pending (task #1).
- Optimizer/batch/VRAM (§5): **HIGH** for paper values (batch 5, lr 1e-4, 20 ep); **MEDIUM** for the batch-8 recommendation (VRAM extrapolated, re-probe).
- Projection head (§4): **MEDIUM** — L2-norm CITED; dims SimCLR-standard `[ASSUMED]`.
- Augmentation policy (§7): **MEDIUM** — set CITED; magnitudes `[ASSUMED]`, verified by linear-probe.
- Fine-tune transfer + LR (§9): **HIGH** — paper specifies only 1e-4; the confirmation is decisive.
- Trajectory format (§8): **MEDIUM** — strongly suggested by CVCSPC code + Phase 1; **probe-pending**.
- Reuse map (§11): **HIGH** — read every relevant module.
- Validation / collapse (§12, §15): **HIGH** — standard SSL practice + literature-verified metrics.

**Research date:** 2026-05-21
**Valid until:** ~14 days (the paper is stable ground truth; the only volatile items are the in-notebook probes, which resolve at execution time, and Colab's torch/torchvision versions).

## RESEARCH COMPLETE
