# `_archive_v5_2/` — v5.2 evidence pack

**Frozen on:** 2026-05-06
**Reason:** v5.2 form-analysis model failed its primary success metric on 2026-05-05. v6 (QEVD-trained) supersedes it. Files here are preserved verbatim for the dissertation defence committee.

---

## Why this archive exists

v5.2 was an ST-GCN trained on Fit3D pose data with **synthetic angular perturbation labels** (`synthetic_perturbation.py`, `corruption.py`). The model was supposed to learn human form errors but instead learned to recognise our hand-crafted noise.

**Reality-check on 2026-05-05** (4 user-recorded squat videos, single-camera, real phone):

| File | Mean Quality | Verdict |
|------|-------------:|---------|
| `Good_Squats.mp4`  | 0.365 | (good form) |
| `Good_Squats2.mp4` | 0.429 | (good form) |
| `Bad_Squats.mp4`   | 0.431 | (bad form)  |
| `Bad_Squats2.mp4`  | 0.459 | (bad form)  |

**Quality gap = mean(Good) − mean(Bad) = -0.048.** Required: ≥ +0.15. The model rated bad squats *higher* than good squats. Per-joint-error channels saturated near 1.0 indiscriminately. Boundary head saturated. See `models/form_model_v5_2/phase6_reality_check.json` for the raw numbers.

Two layers of fallback in `backend/services/form_session.py:38-93` (time-based rep counter + disabled quality-cycle detector) hid the symptoms behind tolerable rep-counting behaviour, but the underlying form-quality signal was non-functional.

---

## What this archive contains

```
_archive_v5_2/
├── README.md                           ← you are here
├── models/
│   └── form_model_v5_2/                ← all v5.2 weights + metrics + reality-check JSON
│       ├── v5_2_supervised.weights.h5  ← trained weights (~25 MB)
│       ├── model_config.json
│       ├── exercise_labels.json
│       ├── phase6_reality_check.json   ← THE SMOKING GUN
│       ├── v5_2_history.json           ← training curves
│       ├── v5_2_test_metrics.json      ← held-out test set numbers
│       ├── v5_2_val_perturbed_metrics.json
│       ├── confusion_matrix_test.npy
│       └── v5_2_confusion_matrix_test.npy
└── training/
    ├── preprocessing/
    │   ├── fit3d_loader.py             ← Fit3D MoCap loader
    │   ├── aifit_features.py           ← AIFit's 22 angular features (reused in v6)
    │   ├── dataset_builder.py          ← v4 builder (synthetic-corruption variant)
    │   ├── dataset_builder_v5.py       ← v5.0 builder
    │   ├── dataset_builder_v5_1.py     ← v5.1 builder
    │   ├── synthetic_perturbation.py   ← THE DESIGN FLAW: noise-as-label
    │   ├── corruption.py               ← v4 corruption (predecessor to perturbation)
    │   └── extract_fit3d_videos.py     ← Fit3D RGB → MediaPipe extraction
    └── evaluation/
        └── reality_check_v5.py         ← the script that produced the failure JSON above
```

The originals are also still present in the live tree (`backend/models/form_model_v5_2/`, `backend/training/preprocessing/*`) until v6 ships. This is a defensive copy; the live tree provides the rollback path during v6 development.

After D9 PASSES (target: 2026-05-31), the live-tree duplicates will be deleted in one commit and `_archive_v5_2/` becomes the sole copy.

---

## The lesson (defence story spine)

> *"v5.2's labels were ground truth of the corruption, not of human form. There is no path from 'Fit3D + perturb' to a model that distinguishes Good from Bad squats on a body the model has never seen."*

v6 replaces this with **real human exercise video annotated by humans**: Qualcomm's QEVD dataset (303,117 clips, 1,800+ subjects, natural-language coaching feedback). The encoder architecture (ST-GCN, 5M params, 15-joint skeleton graph) is unchanged from v5.2 — *the data is what changed*.

See:
- Master plan: `C:\Users\tsh_x\.claude\plans\i-am-now-on-zazzy-brooks.md` (Part I = v5 history, Part II = v6 QEVD)
- v6 supersession plan: `C:\Users\tsh_x\.claude\plans\phase-1-complete-critical-snappy-flurry.md`
- v6 success metric: D9 reality-check `mean(Good) − mean(Bad) ≥ +0.15` on the same 4 videos that failed v5.2.

---

## Files NOT in this archive

These belong to v5.2 in spirit but are **kept in the live tree** and reused unchanged by v6:

- `backend/training/preprocessing/normalize.py` — `AngleNormalizer`, `resample_sequence` are model-agnostic
- `backend/training/preprocessing/joint_mapping.py` — 33→15 canonical joint map
- `backend/training/preprocessing/angular_features.py` — the 22 angular features (kept; v6 uses them too)
- `backend/training/preprocessing/mediapipe_extractor.py` — Fit3D-targeted extractor; v6 has its own QEVD-targeted version (`qevd_extractor.py`) that shares MediaPipe config via `backend/services/mediapipe_config.py`
- `backend/training/models/st_gcn.py` — v5.2 model code; v6 forks it as `st_gcn_v6.py` with two head changes (10-channel joint-err native, 25-class action) but the trunk is identical
- `backend/services/form_analyzer.py`, `form_session.py`, `app.py`, all Flutter — v5.2's **service contracts** are preserved end-to-end by v6
