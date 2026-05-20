---
phase: 3
slug: squat-supervised-baseline
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-20
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution. Derived from `03-RESEARCH.md` §4 Validation Architecture; refined when the planner produces Task IDs.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.0.0+ (already in `backend/requirements.txt`) |
| **Config file** | None — inline `python -m pytest` invocation |
| **Quick run command** | `python -m pytest backend/training/aqa/eval/ -x -v` |
| **Full suite command** | `python -m pytest backend/training/aqa/ -x -v` |
| **Estimated runtime** | ~5 seconds (unit-only — no GPU paths in pytest) |

Pytest covers **only the pure-function modules** (`eval/metrics.py`, model-head shape check). The training loop, val F1 curve, threshold sweep, and test F1 evaluation are exercised interactively in the Colab notebook with paste-back per CONTEXT D11.

---

## Sampling Rate

- **Per batch (training):** train BCE loss logged to `metrics_history`; **no batch-level F1** (avoids aliasing on the ~15-batch val pass — see RESEARCH §4 Aliasing Guards).
- **Per epoch:** val loss mean + val macro-F1 (threshold 0.5) computed from the **full val pass** (243 clips gathered once before `f1_score`). Drives `best.pt` selection.
- **Per task commit:** Run `python -m pytest backend/training/aqa/eval/ -x -v` (quick suite).
- **Per plan wave:** Run `python -m pytest backend/training/aqa/ -x -v` (full suite).
- **Post-training:** Threshold sweep on val scores (using `sklearn.precision_recall_curve`); test F1 + PR-AUC at val-tuned thresholds.
- **Before `/gsd:verify-work`:** Full suite must be green AND every visualization deliverable (CONTEXT domain item 4) must be on disk in `figures/`.
- **Max feedback latency:** ~5 seconds for unit suite; one epoch (~3 min on L4 per RESEARCH §10) for training-loop feedback.

---

## Per-Task Verification Map

> **Note:** Task IDs are placeholder until `gsd-planner` produces `03-01-PLAN.md`. The planner MUST update this table with concrete `{N}-{plan}-{task}` IDs and wave assignments. The behaviors and commands below are stable — derived from RESEARCH §4.

| Behavior (Req sub-ID) | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| **SQUAT-03-a** R(2+1)D-18 head outputs `[B, 2]` logits from `[B, 3, 32, 112, 112]` input | 01 | 1 | SQUAT-03 | T-03-01 (model-construction integrity) | Head replacement preserves backbone weights; no silent in-place mutation of Kinetics-V1 init | unit | `python -m pytest backend/training/aqa/eval/test_metrics.py::test_model_head_shape -x` | ❌ W0 | ⬜ pending |
| **SQUAT-03-b** `eval/metrics.py` F1 per error matches manual calculation on known arrays | 01 | 1 | SQUAT-03 | — | N/A | unit | `python -m pytest backend/training/aqa/eval/test_metrics.py::test_f1_known_values -x` | ❌ W0 | ⬜ pending |
| **SQUAT-03-c** `eval/metrics.py` threshold sweep selects the F1-maximizing threshold via `sklearn.precision_recall_curve` | 01 | 1 | SQUAT-03 | — | N/A | unit | `python -m pytest backend/training/aqa/eval/test_metrics.py::test_threshold_sweep -x` | ❌ W0 | ⬜ pending |
| **SQUAT-03-d** `eval/metrics.py` PR-AUC matches `sklearn.average_precision_score` on known arrays | 01 | 1 | SQUAT-03 | — | N/A | unit | `python -m pytest backend/training/aqa/eval/test_metrics.py::test_pr_auc -x` | ❌ W0 | ⬜ pending |
| **SQUAT-03-e** Checkpoint schema extension: `best.pt` payload contains `best_thresholds`, `best_f1_val`, `scheduler_state_dict`, and Phase 3 metrics_history keys | 01 | 2 | SQUAT-03 | T-03-02 (checkpoint integrity) | `atomic_save_checkpoint` round-trip verify on every write; `latest.txt` last; resume loads exact byte-identical state | unit (smoke) + Colab paste-back | `python -c "import torch; ck=torch.load('best.pt', map_location='cpu', weights_only=False); assert {'best_thresholds','best_f1_val','scheduler_state_dict'} <= ck.keys()"` | ❌ W0 | ⬜ pending |
| **SQUAT-03-f** Training converges: val macro-F1 strictly improves over the first 5 epochs vs. random-init baseline (KFE val F1 > 0.5 by epoch 5) | 01 | 2 | SQUAT-03 | — | N/A | integration (Colab interactive) | Colab paste-back of epoch-5 `metrics_history` row | N/A | ⬜ pending |
| **SQUAT-03-g** Test F1 on official split within ±0.05 of paper's end-to-end fine-tune row: KFE ≥ 0.78, KIE ≥ 0.36 (paper MD row 0.834/0.419 per RESEARCH §1) | 01 | 3 | SQUAT-03 | — | N/A | integration (Colab interactive) | Colab paste-back of final `results.pkl` | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `backend/training/aqa/eval/__init__.py` — new module directory
- [ ] `backend/training/aqa/eval/metrics.py` — `f1_per_error`, `pr_auc_per_error`, `threshold_sweep`, `confusion_matrix_per_error` functions
- [ ] `backend/training/aqa/eval/test_metrics.py` — unit tests for SQUAT-03-a through -d
- [ ] `backend/training/aqa/harness/supervised_train.py` — `SupervisedConfig`, `build_model`, `run_supervised_epoch` (extends Phase 2's `run_tiny_epoch` patterns)
- [ ] `backend/training/aqa/eval/test_supervised_train.py` — unit test for SQUAT-03-a (model head shape) + checkpoint schema (SQUAT-03-e)

---

## Property Axes (Nyquist Dimension 8)

Verbatim from `03-RESEARCH.md` §4:

| Axis | Measurement | Cadence |
|------|-------------|---------|
| Forward pass correctness | Model output shape `[B, 2]`; dtype float32 | Once at module load (assert in `supervised_train.py`) |
| Loss convergence | Train BCE loss per batch, mean per epoch | Per batch (logged), per epoch (checkpoint `metrics_history`) |
| Validation F1 per error | KIE F1, KFE F1 at threshold 0.5 (training-time proxy) | Per epoch (val pass after each train epoch) |
| Validation macro-F1 | Mean of KIE F1 + KFE F1 at threshold 0.5 | Per epoch — drives `best.pt` selection |
| Threshold-sweep F1 | Per-error F1 via `sklearn.precision_recall_curve` on val scores | Once post-training |
| Test F1 (final) | KIE F1 and KFE F1 at val-tuned thresholds on test split | Once at end of training (after threshold sweep) |
| PR-AUC | KIE and KFE PR-AUC on test split (threshold-free) | Once at end of training |
| Checkpoint integrity | `torch.load` round-trip verify inside `atomic_save_checkpoint` | Per epoch checkpoint write |
| Resume state integrity | Model weights + optimizer + scheduler + 4 main-process RNGs restored | On every resume (per Phase 2 contract) |

---

## Bounded Windows (Disconnect-Safe)

- **Epoch-level state:** full model + optimizer + scheduler + 4-RNG snapshot captured in `epoch_{NNN}.pt` after every completed epoch. A disconnect mid-epoch loses at most one epoch of training.
- **Best state:** `best.pt` is a second atomic write on val-macro-F1 improvement — independent of `latest.txt` so it can never point to a corrupt file.
- **Metrics history:** accumulated in the checkpoint payload and reconstructed on resume. All prior epoch metrics survive a disconnect.
- **Threshold sweep results:** `best_thresholds` written into `best.pt` and the final-epoch checkpoint. Disconnecting during the threshold sweep means re-running the val pass — cheap (243 clips).

---

## Aliasing Guards

- **Val F1 during training:** computed once per epoch by gathering ALL val scores before calling `f1_score`. No batch-level F1 averaging — per-batch F1 is unreliable on ~15-batch val passes (see RESEARCH §4 / §8).
- **Test F1:** computed once on all 244 test clips. No streaming estimator.
- **PR curve:** `sklearn.precision_recall_curve` operates on all 244 scores at once — no aliasing.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Loss curves visually decrease across epochs without divergence | SQUAT-03 ROADMAP criterion 3 | Visual gestalt — automated curve-shape test would over-engineer | Open `figures/training_curves.png`, confirm train+val loss trend down and val F1 trends up |
| Confusion matrices match the F1 numerator/denominator | SQUAT-03 ROADMAP criterion 3 | Cross-check between two independent measurements | Open `figures/confusion_kie.png` + `figures/confusion_kfe.png`, verify TP/FP/FN/TN counts reconcile with reported F1 |
| PR curves show realistic precision–recall tradeoffs (not pathological flat lines) | SQUAT-03 ROADMAP criterion 3 | Visual sanity — automated curve-shape test would over-engineer | Open `figures/pr_kie.png` + `figures/pr_kfe.png`, confirm curves bow toward upper-right |
| Sample-prediction grid shows recognisable lifters with predicted scores | Supervisor visualization mandate ([[project_supervisor_visualizations]]) | Human eye check that the model isn't gaming the metric on degenerate clips | Open `figures/sample_predictions.png`, confirm TP / FP / FN examples are visually distinct and labels match obvious form errors |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies (planner enforces)
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify (planner enforces)
- [ ] Wave 0 covers all MISSING references (5 files above)
- [ ] No watch-mode flags (CI-compatible)
- [ ] Feedback latency < 5 s for unit suite
- [ ] `nyquist_compliant: true` set in frontmatter when planner completes Task ID assignment

**Approval:** pending (awaiting planner's Task ID assignment, then planner self-verifies + plan-checker re-verifies)
