# Archive: Old TF + MediaPipe Form Subsystem (v4 – v6)

**Archived:** 2026-05-26
**Reason:** Replaced by the PyTorch Squat-specific service in Phase 5 (API-01, D-06).
**Status:** NOT loaded at runtime. Kept for dissertation-appendix evidence and ablation.

---

## Why This Exists

The original form-correction pipeline (v4 through v6) used TensorFlow MT-TCN / ST-GCN models
backed by a MediaPipe pose-estimation layer. Phase 5 replaced this entire stack with a PyTorch
R(2+1)D-18 model trained on the Fitness-AQA dataset, exposing `backend/services/squat_form_service.py`
as the new inference contract.

The old pipeline is archived here — not deleted — so it can:
- Be cited in the dissertation appendix (training history, confusion matrices, model configs).
- Support ablation comparisons between the TF+MediaPipe approach and the PyTorch approach.
- Be restored if needed by reverting the `git mv` operations that created this directory.

**Reversibility:** All files here were moved with `git mv` and preserve full git history.
To restore any file: `git mv backend/_archive_form_v4_v6/<path> backend/<path>`.

---

## Manifest

### Services (backend/services/ → backend/_archive_form_v4_v6/services/)

| Archived Path | Original Path | Role |
|---------------|--------------|------|
| services/form_analyzer.py | backend/services/form_analyzer.py | TF model loader + MediaPipe per-frame inference |
| services/form_session.py | backend/services/form_session.py | Sliding-window session, rep-counting, EMA |
| services/form_geometry.py | backend/services/form_geometry.py | Angular feature computation (22 angles) |
| services/mediapipe_config.py | backend/services/mediapipe_config.py | Pinned MediaPipe 0.10.33 options |
| services/exercise_sanity.py | backend/services/exercise_sanity.py | Exercise mismatch detector |
| services/test_form_analyzer_v6.py | backend/services/test_form_analyzer_v6.py | Unit tests for FormAnalyzer |
| services/test_form_session_v6_routing.py | backend/services/test_form_session_v6_routing.py | Unit tests for FormSession routing |
| services/test_form_geometry.py | backend/services/test_form_geometry.py | Unit tests for form geometry |

### Tests (backend/tests/ → backend/_archive_form_v4_v6/tests/)

| Archived Path | Original Path | Role |
|---------------|--------------|------|
| tests/test_exercise_sanity.py | backend/tests/test_exercise_sanity.py | Integration tests for ExerciseMismatchDetector |
| tests/validate_user_squats.py | backend/tests/validate_user_squats.py | Phase-F user-clip validation via FormSession |

### Training Models (backend/training/models/ → backend/_archive_form_v4_v6/training/models/)

| Archived Path | Original Path | Role |
|---------------|--------------|------|
| training/models/mt_tcn.py | backend/training/models/mt_tcn.py | v4 MT-TCN architecture |
| training/models/st_gcn.py | backend/training/models/st_gcn.py | v5 ST-GCN architecture |
| training/models/st_gcn_v6.py | backend/training/models/st_gcn_v6.py | v6 ST-GCN architecture |
| training/models/st_gcn_v6_1.py | backend/training/models/st_gcn_v6_1.py | v6.1 ST-GCN architecture |
| training/models/test_st_gcn_v6.py | backend/training/models/test_st_gcn_v6.py | v6 architecture tests |
| training/models/test_st_gcn_v6_1.py | backend/training/models/test_st_gcn_v6_1.py | v6.1 architecture tests |

### Training Scripts (backend/training/ → backend/_archive_form_v4_v6/training/train/)

| Archived Path | Original Path | Role |
|---------------|--------------|------|
| training/train/train_form_model.py | backend/training/train_form_model.py | v4 MT-TCN training |
| training/train/train_form_model_v6.py | backend/training/train_form_model_v6.py | v6 QEVD ST-GCN training |
| training/train/train_form_model_v6_1.py | backend/training/train_form_model_v6_1.py | v6.1 multi-label training |
| training/train/train_ssl_pretrain.py | backend/training/train_ssl_pretrain.py | SSL pre-training (v5 pipeline) |

### Model Weight Directories (backend/models/ → backend/_archive_form_v4_v6/models/)

| Archived Path | Original Path | Notes |
|---------------|--------------|-------|
| models/form_model/ | backend/models/form_model/ | v4 MT-TCN weights + metadata |
| models/form_model_v1_quality_broken_20260420_154019/ | backend/models/form_model_v1_quality_broken_20260420_154019/ | Broken v1 (quality-score bug) |
| models/form_model_v5/ | backend/models/form_model_v5/ | v5 ST-GCN + SSL encoder |
| models/form_model_v5_1/ | backend/models/form_model_v5_1/ | v5.1 ST-GCN |
| models/form_model_v5_2/ | backend/models/form_model_v5_2/ | v5.2 ST-GCN (knee-aware 5-group) |
| models/form_model_v6/ | backend/models/form_model_v6/ | v6 QEVD ST-GCN |
| models/form_model_v6_smoke/ | backend/models/form_model_v6_smoke/ | v6 smoke-train dev artifact (gitignored) |
| models/form_model_v6_1_smoke/ | backend/models/form_model_v6_1_smoke/ | v6.1 smoke-train dev artifact (gitignored) |
| models/form_model_v2_backup_20260424_154358/ | backend/models/form_model_v2_backup_20260424_154358/ | Timestamped backup (gitignored) |
| models/form_model_mediapipe_backup_20260420_032708/ | backend/models/form_model_mediapipe_backup_20260420_032708/ | MediaPipe-era backup (gitignored) |
| models/pose_landmarker_full.task | backend/models/pose_landmarker_full.task | MediaPipe landmarker asset (gitignored, ~29 MB) |

**Note:** The `.weights.h5` files in the weight directories are gitignored in the archive
(`.gitignore` rules under `backend/_archive_form_v4_v6/models/**/*.weights.h5`).
Only metadata files (`.json`, `.npy`, `.png`) are tracked as evidence.

---

## What Was NOT Archived

The following are **still active** and must NOT be archived:

- `backend/training/aqa/` — shared inference contract; `SquatFormService` imports from it
  (transforms.py, eval/ensemble.py, eval/metrics.py, harness/md_finetune.py, datasets/squat.py)
- `backend/models/form_model_squat_md/` — the new PyTorch Phase-4/5 weights (staged from Drive)
- All recommender services: `recommender.py`, `content_filter.py`, `neumf_ranker.py`,
  `llm_adapter.py`, `chat_service.py`, and `backend/models/neumf_*.keras` / `*.pkl`
- New frontend/auth/coach services merged in commit `adc9c33`:
  `coach_service.py`, `coach_tools.py`, `crew_runner.py`, `plan_crew.py`,
  `plan_validators.py`, `supabase_auth.py`, `backend/config/supabase.py`

---

## Replacement

The new form-correction entry point is `backend/services/squat_form_service.py`
(`SquatFormService`), loaded by `backend/app.py` at startup via the `lifespan` context.
It uses PyTorch R(2+1)D-18 models trained on Fitness-AQA and reports KIE / KFE binary
error detection per D-05.
