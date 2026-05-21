---
phase: 4
slug: squat-motion-disentangling-ssl
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-21
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from 04-RESEARCH.md §15 (Validation Architecture) + §16 (verified comparison arithmetic).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.0.0+ (already in `backend/requirements.txt`) |
| **Config file** | none — inline `python -m pytest` (Phase 3 convention) |
| **Quick run command** | `python -m pytest backend/training/aqa/harness/test_md_pretrain.py backend/training/aqa/eval/test_ensemble.py -x -q` |
| **Full suite command** | `python -m pytest backend/training/aqa/ -x -q` |
| **Estimated runtime** | ~5-15 s (pure-function unit tests; training/convergence assertions are Colab paste-back) |

---

## Sampling Rate

- **After every code task commit:** Run the quick command (`pytest test_md_pretrain.py test_ensemble.py -x -q`)
- **After every Wave:** Run the full suite (`pytest backend/training/aqa/ -x -q`)
- **Per SSL batch:** triplet loss logged
- **Per SSL epoch:** mean SSL loss + `embedding_std` + `effective_rank` (collapse monitor, RESEARCH §12) into `metrics_history`
- **Every 5 SSL epochs:** linear-probe macro-F1 (RESEARCH §6) into `linear_probe_history`
- **Per fine-tune epoch:** train/val BCE loss + val F1/PR-AUC + **D6 runtime monitor (train/val loss ratio, abort if >10× before epoch 10)**
- **Before `/gsd:verify-work`:** full pytest suite green; ensemble+TTA test F1, val-test gap, KIE-no-regress check, all 9 figures + `results.pkl` present
- **Max feedback latency:** ~15 s for unit tests (training feedback is per-epoch Colab paste-back, D7 one-runnable-unit-at-a-time)

---

## Per-Task Verification Map

| Req ID | Behavior | Test Type | Automated Command | File Exists |
|--------|----------|-----------|-------------------|-------------|
| SQUAT-04-a | `split_half_cycles` returns two 16-frame index sets; descent ends ≤ bottom ≤ ascent start, on a synthetic parabola | unit | `pytest .../test_md_pretrain.py::test_half_cycle_split -x` | ❌ W0 |
| SQUAT-04-b | `md_triplet_loss` matches hand-computed value (both 2-term and 3-term, squared) | unit | `pytest .../test_md_pretrain.py::test_triplet_loss_known -x` | ❌ W0 |
| SQUAT-04-c | `md_triplet_loss` lower when anchor≈positive & far from negative than reverse (directionality) | unit | `pytest .../test_md_pretrain.py::test_triplet_loss_direction -x` | ❌ W0 |
| SQUAT-04-d | `ProjectionHead` output L2-normalized (norm ≈ 1), shape `[B, 128]` | unit | `pytest .../test_md_pretrain.py::test_projector_l2norm -x` | ❌ W0 |
| SQUAT-04-e | `build_md_model` backbone `fc==Identity`, projector attached, 512-dim features | unit (slow) | `pytest .../test_md_pretrain.py::test_md_model_build -x` | ❌ W0 |
| SQUAT-04-f | SSL checkpoint payload has all keys (backbone/projector/optimizer/scheduler/rng/metrics/linear_probe) | unit | `pytest .../test_md_pretrain.py::test_ssl_checkpoint_schema -x` | ❌ W0 |
| SQUAT-04-g | Trajectory-format probe: per-clip JSON loads as flat float list; frame→traj len mapping reported | integration (Colab) | Colab paste-back (FIRST `checkpoint:human-verify` probe) | ❌ W1 |
| SQUAT-04-h | SSL loss decreases over first 5 epochs; `embedding_std` > 0.1/√512 (no collapse) | integration (Colab) | Colab paste-back of epoch-5 SSL metrics | ❌ W2 |
| SQUAT-04-i | Linear-probe macro-F1 rises above random baseline by epoch 10 (SSL is learning) | integration (Colab) | Colab paste-back of linear-probe curve | ❌ W2 |
| SQUAT-05-a | `build_finetune_model` loads MD backbone (NOT Kinetics); head = `Dropout(0.2)+Linear(512,2)` | unit (slow) | `pytest .../test_md_finetune.py::test_finetune_model_build -x` | ❌ W0 |
| SQUAT-05-b | `aggregate_sigmoid_mean` averages per-seed sigmoids; output ∈ [0,1] | unit | `pytest .../test_ensemble.py::test_sigmoid_mean -x` | ❌ W0 |
| SQUAT-05-c | `select_tta_recipe` returns val-F1-maximizing combo on synthetic scores | unit | `pytest .../test_tta.py::test_recipe_selection -x` | ❌ W0 |
| SQUAT-05-d | Fine-tune val macro-F1 ≥ Phase 3 **val-best** (0.5454, epoch 3) — SSL transferred. *(0.5454 = VAL bar for fine-tune monitoring; the 0.5429 TEST bar is SQUAT-05-e — both correct, not an arithmetic discrepancy)* | integration (Colab) | Colab paste-back of fine-tune curve | ❌ W3 |
| SQUAT-05-e | Ensemble+TTA test macro-F1 ≥ Phase 3 (0.543); KIE F1 NOT < 0.2857 (D6 monitor 3) | integration (Colab) | Colab paste-back of final test report | ❌ W4 |
| SQUAT-05-f | val-test F1 gap < 0.05 (D6 monitor 2) | integration (Colab) | Colab paste-back | ❌ W4 |

*Status legend: ❌ W{n} = not yet created, target wave n · ✅ green · ⚠️ flaky*

**Granularity note:** unit tests (pytest) cover pure functions — half-cycle splitter on synthetic parabola, triplet-loss arithmetic, projector L2-norm, ensemble/TTA aggregation, checkpoint schema. The triplet-loss and half-cycle-splitter unit tests are **load-bearing**: they verify the reconstructed-from-paper logic *before* burning 12–24h of GPU. Training convergence, collapse, linear-probe, and test-F1 assertions happen interactively in Colab via paste-back (D7).

---

## Wave 0 Requirements

- [ ] `backend/training/aqa/datasets/squat_ssl.py` — SSL dataset + half-cycle splitter
- [ ] `backend/training/aqa/datasets/ssl_augs.py` (or inline) — SSL augmentations (§7)
- [ ] `backend/training/aqa/harness/md_pretrain.py` — SSL trainer + triplet loss + projector + linear-probe + collapse detection
- [ ] `backend/training/aqa/harness/md_finetune.py` — fine-tune trainer (AdamW + dropout, MD init)
- [ ] `backend/training/aqa/eval/ensemble.py` — mean-of-sigmoids aggregation
- [ ] `backend/training/aqa/eval/tta.py` — TTA forward + recipe selection
- [ ] `backend/training/aqa/harness/test_md_pretrain.py` — unit tests (splitter, loss, projector, schema)
- [ ] `backend/training/aqa/harness/test_md_finetune.py` — unit tests (model build)
- [ ] `backend/training/aqa/eval/test_ensemble.py` + `test_tta.py` — unit tests
- [ ] `backend/training/aqa/harness/colab.py` — add `update_latest` kwarg to `atomic_save_checkpoint` + `stage_unlabeled_squat_videos`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Trajectory file format + half-cycle bottom-detection sign (argmax vs argmin) | SQUAT-04-g | Files live on Google Drive (unreachable from CI); sign depends on real data | Probe 2–5 real unlabeled clips in Colab; plot trajectory + overlay split point; human confirms descent/ascent before writing `squat_ssl.py`. **Gated `checkpoint:human-verify`, task #1, before any GPU burn.** |
| SSL convergence (loss ↓) + no representation collapse | SQUAT-04-h | Requires GPU training run | Colab paste-back of epoch-5 SSL `metrics_history`: loss decreasing, `embedding_std` > threshold, `effective_rank` not ≈1 |
| Linear-probe macro-F1 rising | SQUAT-04-i | Requires GPU + labeled subset | Colab paste-back of linear-probe curve every 5 epochs; rises above random by epoch 10 |
| Fine-tune transfer (val F1 ≥ Phase 3) | SQUAT-05-d | Requires GPU fine-tune | Colab paste-back of per-seed fine-tune curves |
| Ensemble+TTA test F1 + D6 gates (macro ≥ 0.543, KIE ≥ 0.286, val-test gap < 0.05) | SQUAT-05-e/f | Requires full pipeline | Colab paste-back of final test report + comparison chart |

---

## Aliasing Guards

- **SSL `embedding_std` / linear-probe F1:** computed on a fixed probe batch / full labeled subset at once — no per-batch streaming (14% KIE positive rate aliases on small batches; carries Phase 3 §8 reasoning).
- **Fine-tune val/test F1:** full-pass gather then single sklearn call (carry `_val_pass` from `supervised_train.py`).
- **Ensemble/TTA:** all scores gathered before threshold sweep — no streaming.

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify (Colab integration tasks gated by paste-back)
- [ ] Wave 0 covers all MISSING references (10 new files above)
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s for unit tests
- [ ] `nyquist_compliant: true` set in frontmatter (after Wave 0 tests authored)

**Approval:** pending
