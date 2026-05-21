---
phase: 04-squat-motion-disentangling-ssl
plan: 01
status: complete
completed: 2026-05-21
---

# Phase 4 · Plan 01 (Wave 0) — Summary

## What was planned

13 tasks under `04-01-PLAN.md`: scaffold all 6 new SSL modules + 4 test files (T1); implement the pure-function cores reconstructed from the paper — `split_half_cycles` (T2), `ssl_augs` 5 augmentations (T3), `ProjectionHead` (T4), `md_triplet_loss` (T5), `build_md_model` (T6), the SSL checkpoint schema (T7), `build_finetune_model` (T8), `aggregate_sigmoid_mean` (T9), `select_tta_recipe` + `tta_forward` (T10) — plus the `colab.py` `update_latest` kwarg + `stage_unlabeled_squat_videos` (T11) and its behavioral test (T12), then the Wave 0 gate (T13). Wave 0 per RESEARCH §15: verify the from-paper logic on synthetic data **before** the 12-24h GPU burn. Covers SQUAT-04 + SQUAT-05 machinery.

## What was done

Executed **locally on the Windows machine** (CPU-only torch 2.12.0+cpu installed for this purpose, with user consent) — Wave 0 is pure-function code with no GPU/Colab/Drive dependency, so each task was implemented, unit-tested via `pytest`, and atomically committed one runnable unit at a time per `[[feedback_one_cell_at_a_time_strict]]`. The two slow model-build tests download R(2+1)D-18 Kinetics weights once (cached thereafter).

**Wave 0 gate (T13): GREEN.** `python -m pytest backend/training/aqa/ -m "not slow"` → **21 passed, 0 failures** (full Phase 3 regression — `test_metrics` 9 + `test_supervised_train` — STILL green, proving the `colab.py` edit is backward-compatible; plus all fast Phase 4 tests). The two slow model-build tests pass standalone (2 passed). The reconstructed-from-paper machinery is locked behind a green pure-function suite; the 12-24h SSL GPU burn (Plan 02) is authorized.

## Key verifications (the "no mistakes before GPU" gate)

- **`split_half_cycles` (T2):** descent[-1] ≤ bottom ≤ ascent[0] contract holds on synthetic concave-up + concave-down parabolas; flipping `bottom_is_argmax` moves the detected bottom (sign is parameterized, not hard-coded — T-04-02); <2-sample ValueError guard.
- **`md_triplet_loss` (T5):** matches the closed form (computed independently via `math`) for all 4 squared×three_term combos + the exact log(2)/log(3) degenerate case; directionality holds (anchor≈pos & far-from-neg ⇒ strictly lower loss) — T-04-01. This is the single highest-leverage check: a wrong sign/denominator would have wasted 12-24h.
- **`ProjectionHead` (T4):** output (4,128), per-row L2 norm = 1.0.
- **`build_md_model` (T6):** backbone `fc=Identity`, forward (2,512), projector (2,128) L2-normed; Kinetics-V1 init.
- **`build_finetune_model` (T8):** loads MD backbone (NOT Kinetics, `strict=False`), head = `Dropout(0.2)+Linear(512,2)`, forward zeros[2,3,32,112,112] → (2,2).
- **SSL checkpoint schema (T7):** `build_ssl_checkpoint_payload` assembles exactly the 11 keys; `code_version="phase04-md-pretrain"`.
- **`aggregate_sigmoid_mean` (T9):** mean-of-sigmoids in [0,1]; raises on empty/shape-mismatch.
- **`select_tta_recipe` (T10):** returns the val-macro-F1-maximizing recipe with deterministic shortest-first tiebreak; `tta_forward` returns mean-logits (2,).
- **`update_latest` kwarg (T12):** `update_latest=False` writes backbone.pt but leaves `latest.txt` byte-unchanged — D7 / Pitfall 4 closed.

## Deviations from PLAN

- **Test-label correction (T2):** the plan's test description swapped the argmax/argmin parabola labels (`-(x²)` is concave-DOWN → its middle is the argmax, not argmin). Implemented the test correctly (each parabola paired with the sign that detects its middle). No code impact — only the plan's parenthetical was wrong.
- **`tta.py` torch import (T10):** switched the module-top `import torch` to a `TYPE_CHECKING` guard + lazy import inside `tta_forward`, so `select_tta_recipe` is importable + testable torch-free (honoring the plan's explicit "test_tta torch-free except torch tests" design; the interfaces' module-top `import torch` would have blocked it).
- **Trajectory extraction (T11):** the plan said to reuse `_extract_with_resume_and_progress` for both zips, but that helper is **mp4-only** (skips non-mp4 members). The trajectory JSONs therefore use copy-resume + `zipfile.extractall`; the videos use the full resume path. The exact trajectory on-disk layout is confirmed by Plan 02's gated probe (§8) — `stage_unlabeled_squat_videos`' behavioral correctness is verified on Colab in Plan 02.
- **Execution venue:** Wave 0 ran locally (CPU torch) rather than on Colab, with user consent, for fast pre-GPU verification. Colab Step 0 re-runs pytest in the real env (Plan 02).
- **`slow` marker warning:** `@pytest.mark.slow` is unregistered (no pytest config) → benign `PytestUnknownMarkWarning`, same as Phase 3's `test_supervised_train.py`. Filtering via `-m "not slow"` works regardless.

## Artifacts (committed on `fresh-start`)

**New modules** (`backend/training/aqa/`): `datasets/squat_ssl.py` (+ `split_half_cycles`), `datasets/ssl_augs.py`, `harness/md_pretrain.py` (`MDConfig`, `ProjectionHead`, `md_triplet_loss`, `build_md_model`, `_SSL_CHECKPOINT_KEYS`, `build_ssl_checkpoint_payload`), `harness/md_finetune.py` (`FinetuneConfig`, `build_finetune_model`), `eval/ensemble.py`, `eval/tta.py`. **Edited:** `harness/colab.py` (`update_latest` kwarg + `stage_unlabeled_squat_videos`). **Tests:** `test_md_pretrain.py`, `test_md_finetune.py`, `eval/test_ensemble.py`, `eval/test_tta.py`, `test_colab_update_latest.py`.

Training-loop bodies (`run_md_pretrain_epoch`, `_linear_probe`, `run_md_finetune_epoch`, `SquatSSLDataset.__getitem__`/`_load_trajectory`, `build_ssl_loader`) are documented stubs (`NotImplementedError` naming the finalizing plan) — they need GPU + probe-confirmed real data and are finalized in Plans 02/03.

**Commits:** Task 1 `chore(04)` scaffold; Tasks 2-12 `feat(04)`/`test(04)` per task; Task 13 gate (no files).

## Next

**Plan 02 (Waves 1-2) — on a fresh L4 Colab notebook** (`[[feedback_heavy_training_new_notebook]]`):
- **Wave 1 (gated `checkpoint:human-verify`):** the trajectory-format + half-cycle-sign probe on 2-5 real unlabeled clips — resolves `bottom_is_argmax` (argmax vs argmin), the trajectory↔video frame mapping, and the on-disk JSON format BEFORE `squat_ssl._load_trajectory` is finalized and BEFORE any GPU burn.
- **Wave 2:** finalize the SSL dataset + `run_md_pretrain_epoch`, then the 12-24h MD-SSL pretrain (linear-probe convergence monitoring + representation-collapse detection), producing `backbone.pt`.

Then Plan 03 (3-seed fine-tune) and Plan 04 (ensemble + val-tuned TTA + test eval + 9 figures + the Phase 3-vs-Phase 4 comparison chart).
