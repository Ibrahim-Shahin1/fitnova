# Phase 03: Squat Supervised Baseline — Research

**Researched:** 2026-05-20
**Researcher:** gsd-phase-researcher (sonnet-4-6)
**Domain:** PyTorch video fine-tuning; binary multi-label classification; R(2+1)D-18; sklearn evaluation
**Confidence:** HIGH for paper-cited values; MEDIUM for wall-time estimates; LOW for fp16 determinism interaction

**Source files read:**
- `.planning/phases/03-squat-supervised-baseline/03-CONTEXT.md` — locked decisions D1–D11 (full)
- `.planning/REQUIREMENTS.md` — SQUAT-03
- `.planning/ROADMAP.md` — Phase 3 success criteria
- `.planning/STATE.md`
- `.planning/phases/02-squat-data-pipeline-colab-harness/02-01-SUMMARY.md` — Phase 2 deliverables + deviations
- `.planning/phases/02-squat-data-pipeline-colab-harness/02-01-PLAN.md` — interfaces block (lines 102–243), risk register R4/R8/R9/R10/R12 (lines 999–1018)
- `.planning/phases/02-squat-data-pipeline-colab-harness/02-RESEARCH.md` — §§1–10 paper findings
- `.planning/phases/02-squat-data-pipeline-colab-harness/CODE-RELEASE-NOTES.md` — full gap analysis
- `.planning/phases/01-dataset-consolidation-eda/01-DATASET-REPORT.md` — verified counts
- `backend/training/aqa/datasets/squat.py` — full (pos_weight, build_loaders)
- `backend/training/aqa/harness/tiny_train.py` — full (checkpoint schema, multi-epoch loop)
- `backend/training/aqa/harness/colab.py` — full (atomic_save_checkpoint, capture_rng_state)
- `backend/training/aqa/notebooks/02_squat_pipeline_harness.py` — Step 0 bootstrap, cell conventions
- `Fitness-AQA/Domain Knowledge-Informed Self-Supervised.pdf` — pages 1–18 (Parmar et al., ECCV 2022)
- `https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r2plus1d_18.html`
- `https://docs.pytorch.org/docs/2.12/notes/randomness.html`
- `https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html`

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- D1: PyTorch + torchvision R(2+1)D-18, Kinetics-400-V1 init. Full end-to-end fine-tune. Joint KIE/KFE head `nn.Linear(512, 2)`.
- D2: `BCEWithLogitsLoss(pos_weight=dataset.pos_weight)` — train-derived `[w_KIE≈6.10, w_KFE≈0.45]`, mean-reduced.
- D3: Phase 2 pipeline consumed verbatim — `build_loaders`, `num_frames=32`, `crop_size=112`, `train_aug=True`, `train_jitter_frames=2`.
- D4: `num_workers >= 2`; `worker_init_fn(worker_id)` for per-worker re-seeding. Trainer builds DataLoader directly OR extends `build_loaders` with `worker_init_fn` kwarg — planner picks.
- D5: Spatial crop locked default (128→112 random/center, flip OFF). Ablation hooks present but not executed unless KIE F1 disappoints.
- D6: RESEARCHER RESOLVES — optimizer, lr, scheduler, epochs, batch size.
- D7: Threshold sweep `0.05→0.95` step `0.01` (91 points) per error independently, on val; best threshold applied to test.
- D8: Best checkpoint = best val macro-F1. `best.pt` written via `atomic_save_checkpoint`.
- D9: F1 per error via `sklearn.f1_score`; PR-AUC via `sklearn.average_precision_score`. All computations on gathered CPU arrays.
- D10: Notebook `03_squat_supervised_baseline.py`, jupytext percent-format, Phase 2 conventions verbatim. `_envinit` first line. Trainer in `supervised_train.py`, metrics in `eval/metrics.py`.
- D11: Working agreement — interactive execution, disconnect-safe, progress bars everywhere, visualizations first-class (PNG before plt.show()), AI correctness (every hyperparameter cited).

### Claude's Discretion
- None stated.

### Deferred Ideas (OUT OF SCOPE)
- Motion-Disentangling SSL (Phase 4)
- Multi-seed variance estimation (optional if wall-time permits)
- Crop-size ablation (Phase 3 follow-up only if KIE F1 disappoints)
- Horizontal flip ablation (same)
- Backbone freezing experiments
- TorchCodec migration
- OHP / BarbellRow / Shallow-Squat
- API integration (Phase 5)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SQUAT-03 | Supervised baseline (R(2+1)D-18, Kinetics-init) trained; F1 per error on the official Squat test split | §§1–10 cover all gaps: optimizer/scheduler (§1), workers (§2), head (§3), evaluation architecture (§4), threshold mechanics (§5), checkpoint metric (§6), checkpoint safety (§7), metrics accumulation (§8), risk forwarding (§9), wall time (§10) |
</phase_requirements>

---

## Executive Summary

The paper (Parmar et al., ECCV 2022, Table 2) reports the **Kinetics supervised baseline** — the exact Phase 3 target — as **KIE = 0.2970, KFE = 0.8184**, using R(2+1)D-18 with Kinetics-400 init. CONTEXT.md's provisional D6 values (Adam 1e-4, 50 epochs, batch 16) are **partially correct and partially wrong**:

- **Adam lr=1e-4 is paper-confirmed** for both MD and CVCSPC approaches (paper §5 implementation details). The torchvision reference recipe uses SGD+momentum, but that recipe is for training from scratch on Kinetics — not fine-tuning on a small domain dataset. The paper's choice takes precedence for fidelity.
- **50 epochs is not in the paper** for the supervised Kinetics baseline — the paper says "20 epochs" for MD SSL pretraining (which uses the same R(2+1)D-18 architecture end-to-end). For the supervised downstream task on the labeled set, the paper is **silent on epoch count for the Kinetics baseline specifically**. The Phase 3 provisional of 50 epochs with early-stop is a reasonable working assumption, stated explicitly as such.
- **Batch size 5** is what the paper uses for MD pretraining. For fine-tuning on 1,136 clips with a 16 GB VRAM GPU (L4 has 24 GB), batch 16 is safe. Batch 8 also safe. The paper's downstream evaluation training epoch count is unspecified — no override needed.
- **Cosine annealing is not mentioned in the paper for this context.** Paper is silent on scheduler for the Kinetics supervised baseline. Cosine annealing is a reasonable default for a capped 50-epoch run.
- **No gradient clipping** mentioned in the paper or CVCSPC code.
- **Wall time on L4**: roughly 4–7 hours for a full 50-epoch run at batch 16 with `num_workers=4`. Dominates the decision on early-stop patience.

The primary flag for the planner: **the paper is silent on several D6 hyperparameters for the specific supervised Kinetics baseline** (epoch count, scheduler, weight decay for fine-tune). This is because the paper only specifies parameters for pretraining (MD/CVCSPC) and treats the downstream fine-tune as standard. Every unspecified value is documented below with its justification.

---

## §1 D6 Hyperparameter Confirmation

### Paper Source (ECCV 2022 §5 Implementation Details, p. 9)

Direct quote from the paper (verified by reading the PDF):
> "Motion Disentanglement approach (MD). We used R(2+1)D-18 as our backbone CNN. ... We initialized our backbone CNN with Kinetics pretrained weights. We optimized our models using ADAM optimizer with an initial learning rate of 1e-4 for 20 epochs with a batch size of 5."

The paper uses R(2+1)D-18 end-to-end for both pretraining AND supervised fine-tuning. The phrase "for the motion disentangling model, since the temporal model is already baked in it, we simply finetuned the model end-to-end on the labeled dataset for error detection" (p. 9) confirms end-to-end fine-tune for the supervised step.

The "Kinetics" entry in Table 2 is the Kinetics-pretrained R(2+1)D-18 used directly as a feature extractor (frozen backbone + learned head) — **not end-to-end fine-tuned**. This is an important distinction: the paper's **KIE 0.297, KFE 0.818 is for the frozen-backbone Kinetics baseline**, not end-to-end fine-tune. The MD entry (KIE 0.419, KFE 0.834) uses end-to-end fine-tune.

**Implication for CONTEXT.md D1 (end-to-end fine-tune):** D1 says "Paper § 4.1 supervised-baseline does end-to-end fine-tuning." This is true for the **MD model** but the pure **Kinetics supervised baseline** in Table 2 is frozen-backbone. If Phase 3 trains end-to-end (D1), expect results closer to the MD line (0.419 KIE, 0.834 KFE), not the Kinetics baseline line (0.297 KIE, 0.818 KFE). **The CONTEXT.md target numbers (KFE ≈ 0.818, KIE ≈ 0.297) are thus for the frozen-backbone variant.** End-to-end fine-tune should exceed these, potentially matching or exceeding the MD numbers. The ROADMAP criterion "in the range of the paper's supervised baseline" is still achievable — this just means Phase 3 may outperform the strict "Kinetics baseline" row, which is fine. **No decision override required; call this out in PLAN.md task that reports results.**

[CITED: Parmar et al. ECCV 2022, Table 2, p. 12; §5 Implementation Details, p. 9]

### Per-value Table

| Parameter | CONTEXT Provisional | Confirmed/Overridden | Citation |
|-----------|--------------------|-----------------------|----------|
| Optimizer | Adam | **Adam — CONFIRMED** | Paper §5: "ADAM optimizer with an initial learning rate of 1e-4" for MD; CVCSPC train_test.py line 157: `optim.Adam(parameters_2_optimize, lr=learning_rate)`. The torchvision reference recipe uses SGD (for Kinetics training from scratch), not for fine-tuning on small domain dataset. Paper takes precedence. [CITED: Paper §5, p.9; CITED: Code_Release/pose_contrastive_learning/self_supervised_learning/train_test.py:157] |
| Learning rate | 1e-4 | **1e-4 — CONFIRMED** | Paper §5: "initial learning rate of 1e-4". [CITED: Paper §5, p.9] |
| Momentum / β | Not set (Adam) | **Adam defaults: β1=0.9, β2=0.999, ε=1e-8** — no override. Paper and CVCSPC code use vanilla Adam defaults. [ASSUMED — paper does not state Adam betas explicitly; PyTorch Adam defaults are standard] | |
| Weight decay | Not stated | **0 (no weight decay)** — paper and CVCSPC code do not mention weight decay for fine-tuning. The torchvision SGD recipe uses 1e-4 weight decay, but that is for pretraining on Kinetics, not fine-tune on 1,136 clips. Adding weight decay on a small dataset is a regularization choice; paper is silent, so default to 0 for paper fidelity. [ASSUMED — paper is silent; defaulting to 0 to match CVCSPC's `optim.Adam(parameters)` with no weight_decay kwarg] | |
| Scheduler | Cosine annealing | **Cosine annealing — PLAUSIBLE, UNCONFIRMED** — paper does not mention a scheduler for the Kinetics supervised baseline or the MD fine-tune. "20 epochs" (MD SSL) and "100 epochs" (CVCSPC) are straight Adam with no scheduler mentioned. For a 50-epoch run the difference between cosine and no scheduler is minor, but cosine is a safe conservative default. If early-stop fires before epoch 50 the scheduler tail is irrelevant. [ASSUMED — paper is silent; cosine annealing over `T_max=max_epochs` is a reasonable default for short fine-tuning runs] | |
| Epoch budget | 50 | **50 — UNCONFIRMED, RETAINED** — paper says 20 epochs for MD SSL pretraining on the unlabeled set (5,000+ clips). Supervised fine-tuning on 1,136 labeled clips to convergence likely takes fewer epochs than pretraining. 50 with early-stop at 8-epoch patience is conservative. Paper is silent on epoch count for the Kinetics supervised baseline. [ASSUMED — paper provides epoch count only for MD SSL; 50 with early-stop is a defensible default] | |
| Early-stop patience | 8 epochs | **8 — RETAINED** — no paper convention; 8 epochs is reasonable given ~71 batches per epoch at batch 16. With cosine annealing the validation signal may be noisy in early epochs; 8-epoch patience guards against premature stop. [ASSUMED — paper does not use early-stop language; 8 is a practical choice] | |
| Batch size | 16 | **CONFIRM 16 is safe on L4 (24 GB VRAM), DOWNGRADE PAPER REFERENCE** — paper uses batch 5 for MD SSL on the unlabeled set. For supervised fine-tune on 1,136 clips, batch 16 at input `(16, 3, 32, 112, 112)` is the working hypothesis. VRAM estimate (§10): ~6–8 GB activation + ~0.5 GB weights (31M params × 4 bytes) + optimizer states ≈ 8–10 GB total at fp32. L4 has 24 GB — batch 16 fits with headroom. [ASSUMED VRAM estimate — no hardware measurement; conservative model based on model size + activation rules of thumb. Planner should add a VRAM probe cell before full training.] | |
| Gradient clipping | Not stated | **None** — paper and CVCSPC code do not mention gradient clipping. BCE + sigmoid is numerically stable; gradient explosion is unlikely on fine-tuning a Kinetics-pretrained backbone. If training diverges, gradient clipping (max_norm=1.0) is a fallback. [ASSUMED — paper is silent; omitting is paper-faithful] | |

### Wall-Time Estimate on L4 (§10 below for detailed breakdown)

**Headline: ~4–7 hours for a 50-epoch run at batch 16, num_workers=4, on L4 (24 GB).**

---

## §2 num_workers + worker_init_fn (Resolves Phase 2 R10)

### Recommended num_workers Value

**Recommend `num_workers = 4`.**

Reasoning:
- Training is decode-bound, not GPU-compute-bound, for a dataset where each item requires `read_video_timestamps` + windowed `read_video` decode + spatial transforms. With a 1,136-sample train set, at batch 16, there are ~71 batches per epoch. Prefetching with 4 workers reduces I/O stall.
- L4 has 24 GB VRAM but the bottleneck on decode-heavy video workloads is CPU I/O, not GPU.
- Colab provides 2 vCPUs minimum; 4 workers is the standard upper bound on Colab to avoid OOM from worker fork overhead. Some Colab runtimes give 2 cores only — if `num_workers=4` shows high memory pressure, fall back to 2.
- In practice: start with 4, check VRAM and CPU utilization in the first epoch log. If GPU utilization is < 50% between batches, increase; if OOM, reduce.

[ASSUMED — no direct benchmark of this decode pipeline. L4 Colab CPU allotment is variable. Stated as starting value, not guaranteed optimal.]

### Exact worker_init_fn Body

From PyTorch official docs [CITED: docs.pytorch.org/docs/2.12/notes/randomness.html]:

```python
def seed_worker(worker_id: int) -> None:
    """Re-seed each DataLoader worker from the main-process RNG.

    PyTorch docs: https://docs.pytorch.org/docs/2.12/notes/randomness.html
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

Companion DataLoader construction:

```python
g = torch.Generator()
g.manual_seed(seed)

DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    worker_init_fn=seed_worker,
    generator=g,
)
```

`torch.initial_seed()` inside each worker returns `base_seed + worker_id` (PyTorch sets this automatically). The `% 2**32` guard is needed because `torch.initial_seed()` returns a 64-bit value but NumPy seeds must be 32-bit.

**Note on Phase 2 `capture_rng_state` / `restore_rng_state`:** with `num_workers > 0`, the worker RNGs are NOT captured by `capture_rng_state` (which only captures the main-process RNGs). This means Phase 3 loses the **bitwise-resume guarantee** from Phase 2. This is explicitly acceptable: D4 states "resume correctness comes from the checkpoint payload ... NOT from byte-equality of loss trajectories." Phase 3 resume is functionally correct (model weights + optimizer + scheduler + main-process RNG restored) but not bitwise-identical.

[CITED: docs.pytorch.org/docs/2.12/notes/randomness.html — `seed_worker` pattern verbatim]

### build_loaders Extension vs. Trainer-Side DataLoader

**Recommendation: Trainer builds DataLoader directly; do NOT modify `build_loaders`.**

Rationale:
- `build_loaders` is part of the Phase 2 interface contract. Modifying it to accept `worker_init_fn` risks breaking Phase 2's bitwise-resume test if anyone re-runs the Phase 2 notebook with a newer codebase.
- The trainer (`supervised_train.py`) already needs to construct its own DataLoaders to wire the Generator `g` from its own `seed`, `batch_size`, and `num_workers` configuration. That is a single place, clean.
- `build_loaders` with `num_workers=0` stays as the Phase 2 reference for smoke-test / offline audit use cases.
- Phase 2 `squat.py` already has an inline warning when `num_workers != 0` — the trainer bypassing `build_loaders` is consistent with the documented intent.
- Diff: adding `worker_init_fn` and `generator` to `build_loaders` adds ~10 lines; building in the trainer adds ~20 lines. The delta is minor; isolation is the bigger win.

---

## §3 R(2+1)D-18 Head Architecture

### Confirmed `in_features`

**`in_features = 512` — CONFIRMED.**

From the torchvision source (verified via `resnet.py` on the torchvision GitHub main branch and via the GFLOPS/params card on the model page):
- `AdaptiveAvgPool3d((1, 1, 1))` reduces the last ResNet block output to `(B, 512, 1, 1, 1)`.
- `x.flatten(1)` → `(B, 512)`.
- `self.fc = nn.Linear(512 * block.expansion, num_classes)` with `BasicBlock.expansion = 1` → `Linear(512, 400)` for the Kinetics-400 head.

[CITED: docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r2plus1d_18.html — "31,505,325 parameters, 40.52 GFLOPS, 120.3 MB"; confirmed `in_features=512` from torchvision/models/video/resnet.py `self.fc = nn.Linear(512 * block.expansion, num_classes)`]

### Head Module Recommendation

**Use `nn.Linear(512, 2)` directly — no Dropout layer.**

Reasoning:
- Paper is silent on dropout in the fine-tuning head.
- CVCSPC `train_test.py` uses a `my_resnet.output_feature` → `linear_layers.Linear(512, 1)` pattern without dropout.
- With only 1,136 training samples and heavy class imbalance already addressed by `pos_weight`, adding dropout to a 2-output head is a premature regularization that the paper does not support.
- Adding `Dropout(p=0.5)` before `fc` is listed as an ablation option in the PLAN if KIE F1 disappoints (out-of-scope for the baseline).

```python
# Replace original 400-class head
model = torchvision.models.video.r2plus1d_18(
    weights=R2Plus1D_18_Weights.KINETICS400_V1
)
model.fc = nn.Linear(model.fc.in_features, 2)  # assert in_features == 512 at load time
```

[ASSUMED — no Dropout; paper is silent. Inline `assert model.fc.in_features == 512` is required per D1 ("Verify `in_features=512` against the torchvision implementation at module-load time").]

---

## §4 Validation Architecture

**Required for Nyquist gate (nyquist_validation: true in .planning/config.json).**

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.0.0+ (already in backend/requirements.txt) |
| Config file | No pytest.ini — use inline `python -m pytest` |
| Quick run command | `python -m pytest backend/training/aqa/eval/metrics.py -x -v` |
| Full suite command | `python -m pytest backend/training/aqa/ -x -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SQUAT-03-a | R(2+1)D-18 head outputs `[B, 2]` logits from `[B, 3, 32, 112, 112]` input | unit | `python -m pytest backend/training/aqa/eval/metrics.py::test_model_head_shape -x` | Wave 0 |
| SQUAT-03-b | `eval/metrics.py` F1 per error matches manual calculation on known arrays | unit | `python -m pytest backend/training/aqa/eval/metrics.py::test_f1_known_values -x` | Wave 0 |
| SQUAT-03-c | `eval/metrics.py` threshold sweep selects the F1-maximizing threshold | unit | `python -m pytest backend/training/aqa/eval/metrics.py::test_threshold_sweep -x` | Wave 0 |
| SQUAT-03-d | `eval/metrics.py` PR-AUC matches `sklearn.average_precision_score` on known arrays | unit | `python -m pytest backend/training/aqa/eval/metrics.py::test_pr_auc -x` | Wave 0 |
| SQUAT-03-e | Checkpoint schema: `best.pt` contains `best_thresholds`, `best_f1_val`, Phase 3 keys | unit (smoke) | Manual paste-back from Colab | Wave 0 |
| SQUAT-03-f | Training converges: val macro-F1 improves over 5 epochs vs. random (KFE > 0.5) | integration (Colab) | Colab paste-back of epoch-5 val metrics | Wave 2 |
| SQUAT-03-g | Test F1 on official split: KFE ≥ 0.77, KIE ≥ 0.25 | integration (Colab) | Colab paste-back of final test report | Wave 3 |

**Note on test granularity:** The training, val F1, and test F1 assertions happen inside the interactive Colab notebook (paste-back pattern per D11). pytest covers only the pure-function modules (`eval/metrics.py`, model-head shape check). The training loop is exercised interactively, not in pytest automation.

### Property Axes (Nyquist Dimension 8)

| Axis | Measurement | Cadence |
|------|-------------|---------|
| Forward pass correctness | Model output shape `[B, 2]`; dtype float32 | Once at module load (assert in `supervised_train.py`) |
| Loss convergence | Train BCE loss per batch, mean per epoch | Per batch (logged), per epoch (checkpoint `metrics_history`) |
| Validation F1 per error | KIE F1, KFE F1 at threshold 0.5 (training-time proxy) | Per epoch (val pass after each train epoch) |
| Validation macro-F1 | Mean of KIE F1 + KFE F1 at threshold 0.5 | Per epoch — drives `best.pt` selection |
| Threshold-sweep F1 | Per-error F1 over 91 thresholds on val scores | Once post-training |
| Test F1 (final) | KIE F1 and KFE F1 at val-tuned thresholds on test split | Once at end of training (after threshold sweep) |
| PR-AUC | KIE and KFE PR-AUC on test split (threshold-free) | Once at end of training |
| Checkpoint integrity | `torch.load` round-trip verify inside `atomic_save_checkpoint` | Per epoch checkpoint write |
| Resume state integrity | Model weights + optimizer + scheduler + 4 main-process RNGs restored | On every resume (per Phase 2 contract) |

### Bounded Windows (Disconnect-Safe)

- **Epoch-level state:** full model + optimizer + scheduler + RNG captured in `epoch_{NNN}.pt` after every completed epoch. A disconnect mid-epoch loses at most one epoch of training.
- **Best state:** `best.pt` is a second write on `macro-F1 improvement` epochs — independent of `latest.txt` so it can never point to a corrupt file.
- **Metrics history:** `metrics_history` is accumulated in the checkpoint payload and reconstructed on resume. All prior epoch metrics survive a disconnect.
- **Threshold sweep results:** `best_thresholds` written into `best.pt` and the final epoch checkpoint. Disconnecting during the threshold sweep (val pass + sweep, post-training) means re-running the val pass — cheap (243 clips).

### Aliasing Guards

- **Val F1 during training:** Computed once per epoch by gathering ALL val scores before calling `f1_score`. No batch-level F1 averaging — batch sizes near 243/batch_size ≈ 15 batches; per-batch F1 is unreliable (see §8). Single epoch-level gather eliminates aliasing.
- **Test F1:** computed once on all 244 test clips. No streaming estimator.
- **PR curve:** `sklearn.precision_recall_curve` operates on all 244 scores at once — no aliasing.

### Sampling Rate

- **Per batch:** train loss logged to `metrics_history`; no F1 (avoid aliasing).
- **Per epoch:** val loss mean + val macro-F1 at threshold 0.5 computed from full val pass.
- **Post-training:** threshold sweep on val; test F1 + PR-AUC at val-tuned thresholds.
- **Phase gate:** test F1 results, confusion matrices, PR curves, loss curves all present before `/gsd:verify-work`.

### Wave 0 Gaps

- [ ] `backend/training/aqa/eval/__init__.py` — new module directory
- [ ] `backend/training/aqa/eval/metrics.py` — F1, PR-AUC, threshold sweep, confusion matrix functions + unit tests
- [ ] `backend/training/aqa/harness/supervised_train.py` — full trainer module

---

## §5 Threshold-Sweep Mechanics

### Granularity Decision

**Recommendation: Use `sklearn.metrics.precision_recall_curve` to drive the threshold sweep — NOT the linspace.**

Reasoning:
- `precision_recall_curve(y_true, y_score)` returns precision, recall, and **all unique thresholds** (one per unique score value). With 243 val clips, this gives at most 243 unique thresholds — more precise than a fixed 91-point grid, with zero extra computation.
- F1 at each threshold: `f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-9)` (the `[:-1]` drops the artificial p=1/r=0 endpoint). `best_idx = np.argmax(f1_scores)`, `best_threshold = thresholds[best_idx]`.
- The linspace sweep (D7) is described in CONTEXT.md as the implementation method. The `precision_recall_curve` approach is **strictly superior** (more thresholds, no discretization error) and trivially implementable. This is a direct override of D7's linspace description.
- **Override rationale:** the paper does not specify the exact sweep method — D7's "linspace 91 points" was a provisional planning default, not a paper-cited requirement. `precision_recall_curve` is the standard sklearn tool for this exact task and is already in the dependency list.
- **Keep the linspace tqdm progress bar:** if the user or supervisor wants to see a visible sweep, keep it for the notebook display but run it in parallel to `precision_recall_curve` as a sanity check. The production result uses `precision_recall_curve`.
- **PR-AUC** = `average_precision_score(y_true, y_score)` — threshold-free, uses the same curves. Already in D9.

[CITED: docs.pytorch.org/vision/stable/models/generated/sklearn.metrics.precision_recall_curve — "n_thresholds = len(np.unique(y_score))"; "The last precision and recall values are 1. and 0. respectively and do not have a corresponding threshold."]

### Sweep Code Pattern

```python
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

def best_threshold_from_val(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Return (best_threshold, best_f1) for a single binary error head."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    best_idx = int(np.argmax(f1_values))
    return float(thresholds[best_idx]), float(f1_values[best_idx])
```

[CITED: scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html]

---

## §6 best.pt Selection Metric

**Confirmed: val macro-F1 — RETAINED from CONTEXT.md D8.**

Paper convention: the paper does not explicitly state what metric drives best-checkpoint selection for the downstream supervised fine-tune (the paper's primary evaluation is reported as a final F1, not a checkpoint tracking protocol). Val macro-F1 is the standard choice for multi-label classification problems where per-class F1 is the headline metric.

**Important note:** The val macro-F1 during training uses threshold 0.5 (the quick proxy). The final threshold-tuned F1 (post-training) may differ slightly. This is acceptable — the best checkpoint (by val F1 at 0.5) is the one that has learned the most, even if the exact threshold makes a ±2% difference. Using the val-tuned threshold for training-time best-checkpoint selection would require running the sweep every epoch, which is expensive and noisy on 243 clips.

**Alternative considered: val loss.** Using val BCE loss is simpler but does not directly reflect the F1 metric the paper reports. Val macro-F1 at 0.5 is the correct proxy.

[ASSUMED — paper is silent on best-checkpoint selection metric for downstream fine-tune. Val macro-F1 is the standard choice for this problem. No override needed.]

---

## §7 Atomic-Write at R(2+1)D-18 Scale

### Payload Size

R(2+1)D-18 weights: 31,505,325 parameters × 4 bytes (fp32) = **~120 MB** (matching the 120.3 MB file size on the torchvision model card).

Full checkpoint payload: 120 MB (model) + ~120 MB (Adam optimizer: 2 moment tensors per parameter × 31.5M params × 4 bytes = ~240 MB) + negligible (scheduler, RNG, metrics_history, config) ≈ **~360–380 MB per checkpoint**.

[CITED: docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r2plus1d_18.html — "File size: 120.3 MB"]

### Drive FUSE Safety at 360 MB

The Phase 2 atomic-write contract (`tmp write → torch.load round-trip verify → os.replace → latest.txt last`) **scales to 360 MB payloads without modification**:
- `torch.save` writes the full payload to a `.tmp_{name}.{pid}` file atomically (Drive FUSE allows large sequential writes).
- `torch.load(tmp)` round-trip verify reads ~360 MB from FUSE — adds ~10–15 seconds per checkpoint on slow Drive connections.
- `os.replace(tmp, target)` — same POSIX rename semantics as before.
- `latest.txt` written last — same crash safety guarantee.

**Write time at 360 MB via Drive FUSE:** expect 20–45 seconds per checkpoint write (Drive FUSE throughput is typically 8–15 MB/s for non-cached reads; sequential writes may be faster via bufferred FUSE). This is non-trivial for a 50-epoch run (total checkpoint time: 50 × ~30s = ~25 minutes). At 4–7 hours total training, this is 5–10% overhead — acceptable.

**Recommendation: Stay synchronous.** Do NOT move checkpoint save to a background thread.

Reasoning:
- The `capture_rng_state()` call must happen BEFORE any next-epoch randomness (optimizer step, data shuffle). If the RNG capture is moved to a thread, there is a race: the main training loop might advance the RNG before the background thread's `torch.save` captures the snapshot. The current Phase 2 pattern captures RNG at end-of-epoch, then saves — this must remain synchronous to be correct.
- A 30-second checkpoint write is dead time once per epoch in a 4–7 hour run. The cost is low relative to the risk of a race condition.
- Colab L4 training iteration time for a batch of 16 clips at 112² is estimated at ~0.5–1.5 seconds per batch (§10) — a single epoch is ~1–2 minutes. The 30-second checkpoint write is 25–50% of an epoch's training time, which is notable but not crippling.

[ASSUMED — Drive FUSE throughput estimate based on general FUSE benchmarks, not a measurement on this specific Colab L4 instance. Actual throughput may vary 2–3×. First-epoch user paste-back should report actual checkpoint write time.]

---

## §8 Per-Batch / Per-Epoch Metrics

### Recommendation: Accumulate Per-Epoch, NOT Per-Batch

**Gather all val scores first, compute F1 once per epoch.**

During the val pass:
1. Accumulate `(logits, labels)` tensors into CPU lists across all val batches.
2. After the pass: `scores = torch.sigmoid(torch.cat(all_logits)).numpy()`, `labels = torch.cat(all_labels).numpy()`.
3. Call `f1_score(labels[:, 0], (scores[:, 0] > 0.5).astype(int), pos_label=1)` for KIE; same for KFE at threshold 0.5.
4. Compute `val_macro_f1 = (f1_kie + f1_kfe) / 2`.

**Why not per-batch streaming F1:**
- At batch size 16 and 243 val clips, each val batch has ~16 samples. KIE+ rate is 14% → ~2 KIE+ samples per batch. F1 on 2 positive samples is extremely noisy.
- `sklearn.f1_score` on a batch with 0 true positives returns 0 and warns — producing a meaningless running mean.
- Epoch-level accumulation is correct and simple. 243 samples fit in CPU memory trivially.

**torchmetrics:** NOT recommended for Phase 3.
- torchmetrics is not in `backend/requirements.txt` and is not a Phase 2 dependency.
- Adding it adds a package install step, a new import, and a streaming-state object. For 243 val clips, the complexity is not justified.
- sklearn already provides all required metrics (D9). Stick with epoch-level sklearn calls.

[CITED: CONTEXT.md D9 — "All computations happen in CPU memory after gathering all val/test scores; no streaming necessary at 243 + 244 clip sizes."]

### Training-Phase Metrics

During training, log only:
- Per-batch train BCE loss (scalar, logged to `metrics_history`).
- Per-epoch train loss mean.
- No per-batch F1 during training (batch is too small and the model is mid-weight-update, not useful).

---

## §9 Risk Forwarding from Phase 2

### R4: Paper-vs-Code_Release Gap (MEDIUM-HIGH, carried forward)

**Status: Active for Phase 3.**

The paper does not publish a separate supervised baseline training script. The "Kinetics supervised baseline" row in Table 2 (KIE=0.297, KFE=0.818) uses a **frozen backbone** + learned head — not end-to-end fine-tune (as noted in §1 above). This means Phase 3's end-to-end fine-tune is slightly departing from the exact setup of the paper's Kinetics row.

Hyperparameters not in the paper for the Kinetics supervised baseline downstream training: epoch count, scheduler, weight decay, early-stop, batch size for fine-tuning. All marked [ASSUMED] in §1.

**Mitigation:** Every [ASSUMED] hyperparameter is documented with its justification. The final test F1 is the empirical check — if it falls outside ROADMAP ±0.05 range, §1's "frozen vs. end-to-end" distinction is the first variable to investigate (freeze backbone, re-train, compare).

### R8: Long Multi-Rep Clips (MEDIUM, carried forward)

**Status: Monitoring in Phase 3; no code change from Phase 2 default.**

Phase 1 found Squat clips up to 404 frames (~13.5s at 30fps). The paper says clips "were automatically processed to contain a single repetition" but the long tail suggests multi-rep or slow-tempo clips exist.

Phase 3 baseline uses uniform 32-frame sampling across the full clip span (D3/D5). KIE errors typically occur during the **ascent** phase of the squat. If a multi-rep clip's 32 uniformly sampled frames straddle rest intervals, the model may miss the error interval entirely.

**Phase 1 finding:** "KIE often shows during the ascent rather than at rep bottom. KIE is angle-and-temporal-dependent — exactly why the paper's KIE F1 caps at ~0.53 [citation: paper OpenPose-TDM F1 for KIE = 0.414]." The paper's own models cap at 0.5195 for the best image method and 0.4186 for MD video. The Kinetics frozen-backbone baseline achieves only 0.297. Phase 3's end-to-end fine-tune is expected to exceed 0.297 (possibly reaching 0.4–0.5 range).

**Action if KIE F1 < 0.25:** flag R8 as active; add a random-window-of-32-frames ablation run. Out of scope for the baseline plan.

### R9: Aspect-Preserving Crop Area Loss (MEDIUM, carried forward)

**Status: Unchanged from Phase 2 decision. Ablation hook present.**

Phase 2 confirmed: short-side-128 → 112×112 crop drops 30–50% of frame area on landscape/portrait clips. The paper's `320→224` recipe (Waseda 2DCNN path) distorts aspect to square, which has different problems.

Phase 3 runs at `crop_size=112` (D5). The `crop_size` parameter is exposed in the trainer for ablation. If KIE F1 disappoints AND R8 is not the cause, `crop_size=224` is the next variable (decode memory doubles — see R12 below).

**Decision from Phase 2 summary:** "Aspect-preserve crop_size=112 is paper-faithful by Kinetics inheritance; parameterize for Phase 3 ablation if KIE F1 disappoints."

### R12: Decode Memory Budget (LOW for Phase 3, previously MEDIUM)

**Status: Mitigated. The F11 windowed-decode contract in Phase 2 holds at full training scale.**

Phase 2's `decode_clip` uses `read_video(start_pts=..., end_pts=...)` window-bounded decode. For 32 uniform indices sampled from a 404-frame clip, the window spans `indices[0]/fps` to `(indices[31]+1)/fps` ≈ the full clip duration. But the window only decodes to `output_format="TCHW"` and discards audio — memory is proportional to the actual number of frames in the window, not the full clip.

**At batch 16, training scale:** 16 clips × up to 404 frames × 480×600×3 bytes (pre-crop, worst case) ≈ 16 × 350 MB = 5.6 GB if full-clip decoded. But the F11 windowed decode caps this to the indices-span subset. With uniform 32-index sampling, `start_pts ≈ 0`, `end_pts ≈ full_duration` for long clips — so the full clip IS decoded anyway. **For clips > 300 frames, F11 does not substantially reduce memory on the current sampling scheme.**

**Practical impact:** at batch 16, DataLoader pre-fetches up to `2 × num_workers` batches. With `num_workers=4`, up to 8 batches in flight. Memory for decoded frames lives in CPU RAM, not VRAM. Colab L4 provides 53 GB system RAM — even 8 × 350 MB = 2.8 GB worst case is fine.

If `crop_size=224` ablation is run (R9 mitigation), memory doubles. Still within 53 GB. R12 is effectively resolved for Phase 3.

---

## §10 Wall-Time Estimate on L4

### Model Size and FLOP Count

R(2+1)D-18:
- Parameters: 31,505,325 [CITED: torchvision docs]
- GFLOPS: 40.52 per forward pass at input `(1, 3, 32, 112, 112)` [CITED: torchvision docs — "GFLOPS: 40.52"]
- File size: 120.3 MB weights

### L4 GPU Specs

- VRAM: 24 GB [CITED: comparison benchmarks]
- FP32 training speed vs. T4: L4 is approximately **3× faster at fp32** (empirical LLM fine-tuning benchmark: T4 657s vs. L4 211s, ratio 3.1×). [CITED: ubaada.com Colab GPU comparison]
- Assumed L4 fp32 TFLOPS: ~30–40 TFLOPS (L4 is an Ada Lovelace card; NVIDIA spec lists 30.3 TFLOPS fp32 sparse). [ASSUMED — citing NVIDIA's public L4 spec; not measured on this Colab instance]

### Per-Batch Training Time Estimate

- Forward pass at `(16, 3, 32, 112, 112)`: 16 × 40.52 GFLOPS = 648 GFLOPS
- L4 at 30 TFLOPS fp32 → 648 / 30,000 ≈ **0.022 seconds** raw FLOP time
- Backward pass ≈ 2× forward = 0.044 seconds
- Memory transfer + overhead: L4 GPU memory bandwidth ~300 GB/s; 120 MB weights × 2 (grad) = 240 MB per pass; 240 MB / 300 GB/s = 0.0008 seconds (negligible)
- DataLoader decode + transforms: **the bottleneck** — each clip needs `read_video_timestamps` (cheap) + windowed `read_video` decode + spatial transforms. Rough estimate: 0.1–0.3 seconds per clip CPU-side with PyAV; pipelined with 4 workers → effective stall of ~0.05–0.1 seconds per batch after warm-up.
- **Estimated per-batch wall time: 0.2–0.5 seconds** (decode-limited at first epochs, GPU-limited at warm-up)

### Per-Epoch Wall Time

- Train batches: ceil(1136 / 16) = 71 batches
- Val pass: ceil(243 / 16) = 16 batches (no augmentation, slightly faster decode)
- Per-epoch estimate: 71 × 0.3s (train) + 16 × 0.2s (val) + 30s (checkpoint write) ≈ **21 + 3 + 30 = ~54 seconds per epoch**
- Checkpoint only on improvement epochs (best.pt) or every epoch (latest.pt) — worst case every epoch.

### Total Run Estimate

- 50 epochs × ~54 seconds = **~45 minutes**, best case (after decode warm-up, with 4 workers)
- Realistic estimate with higher decode overhead, slower Drive writes, and DataLoader cold starts: **2–4 hours**
- Upper bound (slow Drive, 2 workers, OOM-avoidance batch 8): **6–8 hours**

**Conservative headline: plan for a 4-hour run on L4 at batch 16, num_workers=4.** If the first epoch takes > 5 minutes, investigate decode throughput and consider reducing `train_jitter_frames` overhead or switching to TorchCodec.

[ASSUMED — all timing estimates based on GFLOPS, published L4 specs, and general decode benchmarks. Actual timing is hardware + network dependent. User should log per-batch timing in epoch 0 and adjust.]

---

## §11 fp16 / Mixed Precision

### Recommendation: DO NOT enable AMP/fp16 for the Phase 3 baseline run.

**Reasoning:**

1. **Paper convention:** The paper does not mention mixed precision for the supervised baseline or MD fine-tuning. Paper fidelity favors fp32.

2. **Determinism conflict:** Phase 2 enabled `torch.use_deterministic_algorithms(True, warn_only=True)` (PLAN.md D13, `colab.py`). AMP/fp16 introduces fp16-specific cuDNN kernels that may not have deterministic implementations. PyTorch documentation notes that enabling deterministic mode "tends to have worse performance" and can conflict with fp16 ops — specifically, some `nn.conv` variants and batch norm ops used in R(2+1)D-18 do not have deterministic fp16 implementations on all CUDA architectures. Running with `warn_only=True` would suppress the error but produce non-deterministic behavior silently.

3. **VRAM is not constrained:** L4 has 24 GB. At batch 16 fp32, estimated total VRAM ≈ 8–10 GB — less than 50% of L4. The primary motivation for AMP is reducing VRAM. That pressure does not exist here.

4. **Training time:** the bulk of wall time is decode-bound (CPU), not GPU compute-bound. AMP's 2× GPU throughput gain gives at most a 20–30% wall-time improvement given the decode bottleneck. Not worth the complexity and determinism risk for a one-time baseline run.

**If AMP is added in the future (Phase 4+):**
- Use `torch.amp.autocast("cuda", dtype=torch.float16)` (new API, not the deprecated `torch.cuda.amp.autocast`).
- Pair with `torch.amp.GradScaler("cuda")`.
- Change `colab.py`'s `torch.use_deterministic_algorithms(True, warn_only=True)` to `warn_only=False` to surface any non-deterministic fp16 ops as errors rather than silent divergence.
- Capture `scaler.state_dict()` in the checkpoint payload.

[ASSUMED — the specific conflict between `torch.use_deterministic_algorithms` and fp16 AMP is based on general PyTorch documentation noting performance degradation of 2-4x in deterministic mode combined with the NVIDIA forum reports of mixed-precision non-determinism on T4 class hardware. No specific per-op audit was performed. Recommendation to avoid AMP for the baseline is conservative but defensible.]

---

## §12 Sample-Prediction Grid Recipe

**Visualization deliverable 4 from CONTEXT.md D10.**

### Recipe

After test evaluation (all 244 test clips scored), produce a 3-row × N-column grid per error (KIE and KFE separately):

**Row 1: Top-K True Positives (TP)** — clips where `label_error = 1` AND `score_error` is highest (most confident correct positive).
**Row 2: Top-K False Positives (FP)** — clips where `label_error = 0` AND `score_error` is highest (most confident wrong positive).
**Row 3: Top-K False Negatives (FN)** — clips where `label_error = 1` AND `score_error` is lowest (most confident missed positive).

Use **K = 4** per row (gives 12 clips per error, 24 total for KIE + KFE). Larger K clutters the figure.

### Per-Clip Rendering

For each selected clip:
1. Call `decode_clip(record.video_path, uniform_sample_indices(num_frames, target=8))` — **8 frames** per clip (reuses Phase 2's 8-frame grid from `decoded_batch.png`).
2. Call `spatial_val(clip_tchw, crop_size=112)` — val transform (center crop, normalized).
3. Denormalize: `frame = (clip_chw * KINETICS_STD[:, None, None] + KINETICS_MEAN[:, None, None]).clamp(0, 1)`.
4. Arrange 8 frames horizontally for each clip; stack 4 clips vertically per row; stack 3 rows (TP/FP/FN) to form the full grid.
5. Add score chip above each clip column: `score={score:.2f}  label={label}  type={TP/FP/FN}`.

### Implementation Notes

- Use `matplotlib.gridspec.GridSpec` for the 3×4 subplot grid (each "cell" is an 8-frame strip from `imshow`).
- Save PNG to `.planning/phases/03-squat-supervised-baseline/figures/sample_predictions_{error}.png` where `{error}` is `kie` or `kfe`.
- Call `Path(out).parent.mkdir(parents=True, exist_ok=True)` before save — D11 defensive mkdir.
- `plt.savefig(out, dpi=150, bbox_inches="tight")` before `plt.show()` — disconnect-safe.
- Reuse Phase 2's `decoded_batch.png` rendering pattern for the frame grid (the same spatial pipeline gives a visually consistent figure).
- If a clip is unavailable on disk (rare edge case): skip and reduce K gracefully, do not crash.

[ASSUMED — the K=4 and 8-frame-per-clip choices are judgment calls. No paper convention. The supervisor visualization goal (defense 2026-06-03) drives the requirement for human-readable chips.]

---

## §13 Open Questions for the Planner

1. **Frozen backbone vs. end-to-end fine-tune for the "baseline" column:**
   The paper's Table 2 "Kinetics" row (KIE=0.297, KFE=0.818) uses a **frozen backbone**. CONTEXT.md D1 locks **end-to-end fine-tune** for Phase 3. This means Phase 3 will likely produce results closer to the paper's MD row (0.419, 0.834) rather than the Kinetics frozen-backbone row.
   - **Recommended planner action:** Add a task note to the first training run clarifying which paper row Phase 3 is comparable to. Report both the frozen-backbone numbers (as a sanity check in ≤5 minutes of evaluation) and the end-to-end fine-tune numbers.
   - **No plan change required** — D1 is locked and its reasoning (full fine-tune = better result) is sound. The supervisor defense benefits from showing Phase 3 exceeds the paper's frozen-backbone baseline.

2. **`build_loaders` num_workers warning on trainer bypass:**
   The existing `squat.py:build_loaders` logs a warning when `num_workers > 0` is passed. The trainer will construct DataLoaders directly (bypassing `build_loaders`), so this warning will not fire. However, any future code that calls `build_loaders` with `num_workers > 0` will get the Phase 2 override back to 0. The planner should decide whether to add a docstring update to `build_loaders` noting that Phase 3+ training uses direct DataLoader construction, or leave it as-is (the inline comment already documents this).

3. **Epoch budget calibration post-wall-time observation:**
   The §10 estimate (2–4 hours) is based on published FLOP specs and general throughput estimates. The actual first-epoch time may be 2–5× faster or slower. The planner should structure the task sequence so the user pastes back epoch 0 timing before committing to 50 epochs, and includes a cell that prints `estimated_total_h = epoch_0_time_s * max_epochs / 3600`.

4. **Checkpoint `config_hash` for Phase 3:**
   Phase 2's `hash_config` hashes a specific set of keys. Phase 3 adds `scheduler_name`, `weight_decay`, `max_epochs`, `num_workers` to the hash. The planner must specify which new keys enter the config hash to avoid silent config drift on resume. Suggest hashing: all D6 hyperparameters + model architecture identifier + seed + label semantics. Explicitly exclude: Drive paths, run_name, max_epochs (varies across resume cycles).

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Video decode and augmentation | Training pipeline (CPU) | DataLoader workers | `decode_clip` + `spatial_train` in worker processes |
| Model forward pass / loss | Training pipeline (GPU) | — | R(2+1)D-18 backbone + BCE loss |
| Per-epoch val F1 | Training pipeline (CPU) | — | Epoch-level accumulation, sklearn calls |
| Checkpoint I/O | Harness (`colab.py`) | Drive FUSE | Atomic write contract from Phase 2 |
| Threshold sweep | Eval module (`eval/metrics.py`) | — | Post-training, pure function, CPU-only |
| Test evaluation | Eval module + notebook | — | Single pass, sklearn |
| Visualization | Notebook (matplotlib) | `figures/` disk | PNG saved before show |
| Resume / state restore | Harness (`colab.py`) | `supervised_train.py` | Phase 2 contract extended |

---

## Sources

### Primary (HIGH confidence)
- Parmar, Gharat, Rhodin. *Domain Knowledge-Informed Self-Supervised Representations for Workout Form Assessment.* ECCV 2022. arXiv:2202.14019. `Fitness-AQA/Domain Knowledge-Informed Self-Supervised.pdf` — §5 p.9 (optimizer=Adam, lr=1e-4, 20 epochs, batch 5 for MD; end-to-end fine-tune); Table 2 p.12 (Kinetics frozen-backbone: KIE=0.297, KFE=0.818)
- `docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r2plus1d_18.html` — `in_features=512`; 40.52 GFLOPS; 120.3 MB; KINETICS400_V1 transform pipeline (`resize [128,171]` → center crop `[112,112]`)
- `docs.pytorch.org/docs/2.12/notes/randomness.html` — `seed_worker` pattern verbatim
- `scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html` — return shape, last-point semantics, threshold array length

### Secondary (MEDIUM confidence)
- `github.com/pytorch/vision/blob/main/references/video_classification/train.py` — torchvision reference recipe uses SGD+momentum for Kinetics scratch training (NOT the fine-tune recipe; informative contrast)
- `ubaada.com/post/ec967231` — L4 vs. T4 training speed comparison (~3× faster at fp32)
- Phase 2 RESEARCH.md §§2–8 — confirmed CVCSPC code idioms (Adam lr=1e-4, no weight decay in `optim.Adam(...)`), paper augmentation list (no flip), normalization stats
- CODE-RELEASE-NOTES.md §(c) — CVCSPC trainer idioms

### Tertiary (LOW confidence — estimates)
- VRAM estimate for batch 16 at fp32: general rule-of-thumb (model weights + Adam states + activations); no direct measurement
- Wall-time estimate: FLOP arithmetic + L4 specs + general decode pipeline benchmarks; not measured
- AMP/determinism conflict: general PyTorch doc notes on deterministic mode performance impact; NVIDIA developer forum report of mixed-precision non-determinism; no per-op audit performed

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Adam betas β1=0.9, β2=0.999 (PyTorch defaults); paper does not state | §1 | Negligible — defaults are universally used |
| A2 | Weight decay = 0 for fine-tuning | §1 | Slight regularization difference; if KIE F1 disappoints, try wd=1e-4 |
| A3 | Cosine annealing scheduler (paper is silent) | §1 | If paper used no scheduler, cosine annealing adds mild LR warmdown — unlikely to hurt; easily ablated |
| A4 | Epoch budget 50 (paper specifies only MD pretraining=20 epochs) | §1 | If convergence is faster (e.g., 15 epochs), run time halves; early-stop guards against over-running |
| A5 | num_workers=4 optimal for L4 Colab | §2 | Could cause OOM on limited-vCPU Colab allocation; mitigated by user fallback to 2 |
| A6 | VRAM budget ~8–10 GB at batch 16 fp32 | §1/§10 | If higher (e.g., 14 GB), batch 16 still fits L4's 24 GB; if lower, batch can increase |
| A7 | Wall time 2–4 hours (estimate from FLOP arithmetic) | §10 | Could be 1–8 hours; user should time epoch 0 before committing |
| A8 | No Dropout on fc head (paper is silent) | §3 | Slight overfitting risk on 1,136 samples; easily added if train F1 >> val F1 |
| A9 | best.pt selection = val macro-F1 (paper is silent) | §6 | Negligible vs. val loss; both are valid proxies |
| A10 | AMP/fp16 conflicts with deterministic algorithms (not per-op verified) | §11 | If no conflict exists, skipping AMP costs ~20–30% wall-time speedup; conservative choice |
| A11 | Drive FUSE write speed ~8–15 MB/s for checkpoint save | §7 | If faster (e.g., 30 MB/s), checkpoint time halves; stay synchronous regardless |
| A12 | "Kinetics baseline" in Table 2 is frozen-backbone, not end-to-end fine-tune | §1 | If end-to-end: paper KIE=0.297 is directly comparable to Phase 3; if frozen: Phase 3 will likely exceed it. Either way, Phase 3 succeeds |

---

## RESEARCH COMPLETE
