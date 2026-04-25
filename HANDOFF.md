# FitNova — Form Analysis Handoff

> **One-paragraph elevator summary.** FitNova is a graduation-project fitness app with three AI features: (1) **NeuMF fitness planning** — done, working, untouched; (2) **Form analysis** — synthetic-pretrained MT-TCN that *passes academic gates on held-out MoCap but fails on real phone video*; (3) **Conversational chat / nutrition** — done, working. This document is the handoff for **the form-analysis problem**, which is the one outstanding thing to fix before defence.

> **TL;DR for the next chat.** The form-analysis pipeline is end-to-end functional (camera → MediaPipe → MT-TCN → Flutter UI). The model holds the academic gates on Fit3D held-out subject s11 (exercise_acc=0.883, joint_F1=0.607, rep_MAE=0.033). On real user-recorded phone video it collapses: classifier predicts "burpees" on squats, quality head saturates near 1.0 regardless of form, knee/hip joint-error channels flatline to 0, rep boundary head detects 0 reps. This is the synthetic-to-real domain gap. Two paths forward are documented at the bottom of this file.

---

## 0. Where we are RIGHT NOW (date: 2026-04-25)

| Layer | State | Notes |
|---|---|---|
| Backend FastAPI | Running on `localhost:8000` | All routes work — `/api/exercises`, WS `/ws/form-session`, REST `/analyze-form-video`. |
| Flutter app (BlueStacks) | Installed + running | New flow: Home → Exercise Selection → Guidelines → Live Feedback / Upload Video. |
| MT-TCN v4 weights | Deployed to `backend/models/form_model/` | Test metrics in §3. v2/v1/mediapipe-baseline backups exist. |
| Smart-mode mismatch detector | Active | `confusion_matrix_test.npy` loaded; squat top-3 confusions = `[deadlift, diamond_pushup, pushup]`. |
| Selected-exercise lock | Patched both in `visualize_pipeline.py` and live `FormSession` | Classifier output is silenced from UI; joint-error display masked to `key_errors_detected` from `exercises.json`. |
| User-recorded validation (Phase F3) | **FAILED** | See §5. This is what the next chat needs to fix. |
| NeuMF / fitness planning | Untouched | `git diff HEAD -- backend/services/recommender.py backend/tests/test_recommender.py` is empty. |

The defence-blocking question: **how do we make the form analyser produce a meaningful good-vs-bad signal on real phone video?**

---

## 1. What this codebase contains (in order of relevance)

### 1.1 Form-analysis backend (the focus area)
```
backend/
├── services/
│   ├── form_analyzer.py        # MediaPipe → 15 canonical joints → 22 angular features → MT-TCN.predict_window()
│   ├── form_session.py         # WebSocket session, EMA smoothing, rep counting, end-of-session summary
│   └── exercise_sanity.py      # Confusion-aware mismatch detector — 8-frame buffer, 5/8 mismatch threshold,
│                               # smart-mode (matrix loaded) vs degraded-mode (tighter conf floor)
├── models/form_model/          # v4 weights — mt_tcn_weights.weights.h5, exercise_labels.json,
│                               # angle_stats.npz, model_config.json, confusion_matrix_{val,test}.{npy,png},
│                               # training_metrics.json, test_metrics.json, training_history.json
├── config/
│   ├── exercises.json          # SSOT for 27 exercises: idx, display_name, camera_view, distance_m,
│   │                           # phone_height_cm, orientation, key_errors_detected, guidelines
│   └── exercises.py            # Loader + relevant_joint_indices(name) → indices into 10-joint head
├── training/
│   ├── train_form_model.py     # SSL pretrain → supervised 2-phase fine-tune → val + test eval +
│   │                           # confusion matrix save + reality_check artifacts
│   ├── train_ssl_pretrain.py   # Masked-autoencoder pretraining on the joints stream
│   ├── models/mt_tcn.py        # 5-head TCN: exercise (27), quality (scalar), boundary (per-frame),
│   │                           # rep_count (scalar), joint_errors (10 per frame)
│   └── preprocessing/
│       ├── fit3d_loader.py     # 8 subjects × 27 exercises × rep_ann.json segmentation
│       ├── normalize.py        # Pelvis-centring, torso-scaling, 64-frame resampling
│       ├── joint_mapping.py    # 25 Fit3D joints → 15 canonical → MediaPipe-compatible
│       ├── angular_features.py # 22 angular features (joint flexions + spine + ratios)
│       ├── corruption.py       # Synthetic bad-form generation (3 severity tiers × 6 joint groups)
│       ├── augmentation.py     # Y-rotation + anisotropic noise + temporal jitter + mirror
│       └── dataset_builder.py  # Wires it all into (X_angles, X_joints, y_*) windows
└── tests/
    ├── test_exercises_consistency.py            # 6 tests — SSOT vs trained-label drift check
    ├── test_exercise_sanity.py                  # 6 tests — smart/degraded/disabled mode behaviour
    ├── test_angular_features_rotation_invariance.py  # 5 tests — Y-rotation invariance assertion
    ├── check_noise_budget.py                    # Standalone — clean-MAE budget after augmentation
    └── validate_user_squats.py                  # Standalone — end-to-end Phase-F3 validation harness
```

### 1.2 Flutter (form-analysis flow)
```
lib/
├── screens/
│   ├── exercise_selection_screen.dart   # Grid of 27 exercises grouped Side / Front / Either
│   ├── guidelines_screen.dart           # Per-exercise camera diagram + distance/height/orientation
│   ├── form_check_screen.dart           # Live camera feedback (WS streaming)
│   ├── form_results_screen.dart         # Post-session summary + LLM coaching
│   └── video_upload_screen.dart         # image_picker → POST /analyze-form-video
├── widgets/
│   ├── mismatch_banner.dart             # Amber dismissible "you selected X, model thinks Y" banner
│   └── skeleton_painter.dart            # 14-bone skeleton overlay
├── providers/form_session_provider.dart # State for live session + mismatch banner
├── models/exercise_meta.dart            # Mirror of exercises.json schema
└── services/
    ├── api_service.dart                 # fetchExercises(), uploadFormVideo()
    └── form_session_service.dart        # WS handshake — sends 'selected_exercise'
```

### 1.3 Top-level project files
- `PROJECT_OVERVIEW.md` — high-level project description (mostly NeuMF-era)
- `TECHNICAL_DEEP_DIVE.md` — 1267-line deep dive (mostly NeuMF + planning)
- `COLAB_RUNBOOK.md` — instructions to retrain MT-TCN on Colab Pro A100
- `FitNova_Colab.ipynb` — the actual notebook (10 cells)
- `FIT3D_AIFIT_USAGE.md` — what we used/ignored from Fit3D + AIFit (sibling to this file)
- `HANDOFF.md` — *this file*
- `_build_fitnova_src_zip.py`, `_build_fit3d_annotations_zip.py` — Drive-upload bundle scripts

### 1.4 Untracked / gitignored large artifacts
- `videos.zip` (2.4 GB) — Fit3D camera-60457274 video clips
- `fit3d_annotations.zip` (138 MB) — joints3d_25 + rep_ann.json
- `fitnova_mediapipe-*.zip` (117 MB) — pre-extracted MediaPipe joints (legacy v1 path)
- `backend/data/form_dataset/` (421 MB), `form_dataset_mediapipe/` (1.3 GB), `mediapipe_fit3d/` (132 MB)
- `backend/models/pose_landmarker_full.task` — Google MediaPipe asset (download from CDN)
- All these are now in `.gitignore`. To rebuild, follow `COLAB_RUNBOOK.md`.

---

## 2. What we tried this sprint — Phase A through Phase F (the recovery plan)

The plan-of-record was `~/.claude/plans/glistening-inventing-steele.md`. Phases:

### Phase A — Fix the training pipeline (DONE, all gates passed)
- **A1.** Made angular features 16–17 (`spine_vert_sag`, `spine_vert_front`) Y-rotation invariant via radial projection. Test: `test_angular_features_rotation_invariance.py` asserts <1e-5 delta under rotations of 30°/60°/90°/180°. **Passes.**
- **A2.** Fixed the dataset_builder bug at lines 455–461 — augmentation was being applied to the *clean* `rep` instead of the *corrupted* `canon`, destroying the corruption that defined the quality label. Pairs are now (corrupted-rotated-pose, corrupted-quality-label).
- **A3.** Added `check_noise_budget.py` — verifies median clean-MAE after augmentation < 0.05 rad (well below the 0.30-rad quality ceiling). **Passes.**
- **A4.** Fixed the `training_metrics.json` / `test_metrics.json` collision — they used to be the same file via a rogue `shutil.copy`. Now `evaluate_model(filename=...)` writes them separately.
- **A5.** Made `weight_probe.py` and `offline_eval.py` data-path-agnostic — they now read from `model_config.json["source"]` instead of hardcoded paths.
- **A6.** Added confusion-matrix saving to `evaluate_model` — `confusion_matrix_{val,test}.{npy,png}` ship alongside metrics.

### Phase B — Rebuild dataset + retrain on Colab (DONE, partial gate pass)
- Rebuilt `fitnova_src.zip` (192 KB), purged stale Drive state, ran all 10 notebook cells.
- Three retrains across the sprint (v2 → v3 → v4); v4 is the deployed weights.
- **Gate result:** 3 of 4 academic gates pass on s11 held-out test. Quality Pearson r ceiling at 0.31.

### Phase C — Per-exercise metadata SSOT (DONE)
- Wrote `backend/config/exercises.json` (27 entries with full metadata).
- Built `backend/config/exercises.py` loader + helpers (`EXERCISE_TO_IDX`, `relevant_joint_indices`, `joint_group_names`).
- Added `GET /api/exercises` route to `backend/app.py`.
- 6 consistency tests in `test_exercises_consistency.py` (count=27, contiguous indices, required fields, enum values, training-label match). **All pass.**

### Phase D — Flutter UX rebuild (DONE)
- New screens: `exercise_selection_screen.dart`, `guidelines_screen.dart`, `video_upload_screen.dart`.
- New widget: `mismatch_banner.dart`.
- New model: `exercise_meta.dart`.
- Wired routes in `main.dart`. `image_picker: ^1.0.7` added to `pubspec.yaml` (user ran `flutter pub get`).
- Old direct-tap-to-camera flow replaced — every tile now routes through the picker → guidelines → live/upload.

### Phase E — Backend `selected_exercise` + confusion-aware sanity check (DONE)
- `FormSession.__init__` accepts both `selected_exercise` and legacy `exercise_hint`.
- `ExerciseMismatchDetector` loads `confusion_matrix_test.npy`, computes per-class top-3 confusable neighbours, fires a one-shot warning when ≥5/8 windows disagree with selection AND the predicted class is not a known confusion.
- Modes: **smart** (matrix present, conf floor 0.70, neighbour suppression), **degraded** (matrix missing, conf floor 0.80, no suppression), **disabled** (labels missing).
- 6 sanity tests in `test_exercise_sanity.py`. **All pass.**
- Flutter renders `MismatchBanner` via `Consumer<FormSessionProvider>` on `FormCheckScreen`.

### Phase F — Validation + defence artifacts (PARTIAL FAIL — this is the open problem)
- **F1.** `reality_check.py --subject s11` — not yet run on v4 weights.
- **F2.** `visualize_pipeline.py` — patched to accept `--selected-exercise squat`; produces annotated MP4s with locked label and joint-error filtering.
- **F3.** User-recorded validation — **FAILED.** Details in §5.
- **F4.** Latency: ~62 ms/sampled-frame on user laptop, comfortably under the 80 ms budget. **Passes.**
- **F5.** Confusion matrix slide ready (`confusion_matrix_test.png`).

---

## 3. Model performance (what passes, what doesn't)

### v4 test_metrics.json (held-out subject s11, MoCap path)
```json
{
  "exercise_accuracy":   0.8830,   // ✓ gate ≥ 0.80
  "exercise_macro_f1":   0.8570,
  "quality_mse":         0.0315,
  "quality_mae":         0.1469,
  "quality_pearson_r":   0.3128,   // ✗ gate ≥ 0.65
  "quality_acc_d03":     0.9139,
  "quality_acc_d05":     0.9974,
  "quality_acc_d07":     1.0,
  "joint_error_f1":      0.6073,   // ✓ gate ≥ 0.60
  "rep_count_mae":       0.0330,   // ✓ gate ≤ 0.10
  "rep_count_obo":       0.0       // (zero off-by-one — perfect on synthetic windows)
}
```

### v4 training_metrics.json (validation subjects s09, s10)
```json
{
  "exercise_accuracy":   0.8236,
  "exercise_macro_f1":   0.7966,
  "quality_pearson_r":   0.2886,
  "joint_error_f1":      0.5508,
  "rep_count_mae":       0.0354
}
```

### Critical academic interpretation
- 3 of 4 G3 gates **pass** on synthetic held-out → defensible MoCap-domain story.
- Quality Pearson r is **stuck at 0.28–0.33 across 3 retrains** (v2/v3/v4). This is not a bug — it is the synthetic-corruption-label ceiling. Quality scalar in the literature is known to be noisy; the **per-joint-error head (F1=0.61) is the substantive form-feedback signal**.
- All zero-shot generalisation to real phone video collapses (see §5).

---

## 4. The user's 4 recorded test videos and what the model said about them

Files in `C:\Users\tsh_x\Downloads\`:
- `Good_Squats.mp4`, `Bad_Squats.mp4` — full side-view, portrait 576×1024, 12.8s / 15.6s
- `Good_Squats2.mp4`, `Bad_Squats2.mp4` — angled (not fully side-on), portrait 576×1024, ~17 s each

Ran through `validate_user_squats.py` with `selected_exercise="squat"` and joint-error mask `[L Knee, R Knee, L Hip, R Hip, Trunk/Spine]`. Annotated outputs at `*_LOCKED.mp4`.

| Joint (squat-relevant) | Good_Side | Bad_Side | Good_Angled | Bad_Angled |
|---|---|---|---|---|
| Left Knee   | 0.000 | 0.000 | 0.000 | 0.000 |
| Right Knee  | 0.000 | 0.000 | 0.000 | 0.000 |
| Left Hip    | 0.000 | 0.000 | 0.000 | 0.000 |
| Right Hip   | 0.000 | 0.000 | 0.000 | 0.000 |
| Trunk/Spine | 0.406 | 0.388 | **0.746** | 0.522 |
| Quality (per-frame mean) | 0.834 | 0.864 | 0.866 | 0.871 |
| Reps detected | 0 | 0 | 0 | 0 |
| Raw classifier guess (silenced from UI) | burpees | burpees | pushup | burpees |

Latency: ~62 ms/sampled-frame — under the 80 ms budget. Pose detection: 100% of frames had pose.

Pre-fix annotated MP4s: `*_annotated.mp4` (showed classifier as "burpees").
Post-fix annotated MP4s: `*_LOCKED.mp4` (now show "Squat (selected by user)" — display fix only, model output unchanged).

---

## 5. WHY IT FAILED — three distinct failure modes

### 5.1 Quality head saturates near 1.0 regardless of form
**Symptom:** mean quality 0.83–0.87 on every video, perfect-form / bad-form / camera-angle-off, all the same.
**Cause:** Synthetic corruption labels in `corruption.py` define "bad form" as a constant geometric perturbation applied to one joint group across the whole rep (e.g. "rotate forearm by 30°"). This is structurally different from real bad form, which is mid-rep dynamic deviations driven by fatigue/weakness. The 0.31 Pearson r on test_metrics.json *is* this ceiling — the head is not learning what real bad form looks like, it's learning what a constant rotation looks like.
**Why this surprised us:** test_metrics.acc_d03=0.91 made it look like the head was working. But that's measured on synthetic data with the same corruption distribution as training. On real video, the corruption fingerprint isn't there.

### 5.2 Knee + hip joint-error channels flatline at 0 on real video
**Symptom:** L/R Knee and L/R Hip predictions are exactly 0.000 on all 4 user videos. Only Trunk/Spine fires.
**Cause:** During training, the corruption sampler hits each joint group with equal probability. The model learned a per-joint detection signature that depends on the *specific synthetic corruption pattern* — small constant offsets in the spinal sagittal-plane projection are the easiest signal to memorise (because we used to have the rotation-non-invariance bug there too — A1 fixed it but the head's reliance on that signal didn't transfer). On phone-MediaPipe joints the noise distribution is different and the per-joint detectors collapse.
**Why this surprised us:** joint_error_f1=0.61 on s11 looked like it generalised. But s11 is still MoCap.

### 5.3 Rep-boundary head detects 0 reps on real video
**Symptom:** All 4 videos report `total_reps=1` (the end-of-session flush; no real boundaries fire).
**Cause:** Boundary head was trained on the multi-rep concatenation windows added in B3 (`MULTI_REP_FRACTION=0.70`). The boundary signal is a sharp probability spike at the rep transition frame in MoCap. On phone-MediaPipe joints, the joints stream is ~5× noisier — the spike is buried in noise and never crosses the 0.55 threshold. The per-rep variants we trained on were also single-camera Fit3D MoCap, never sliding-window MediaPipe.
**Why this surprised us:** rep_count_mae=0.033 on s11 looked perfect. Same MoCap-only generalisation issue.

### 5.4 Exercise classifier predicts "burpees" on phone squats
**Symptom:** All 4 user videos → classifier says burpees (or pushup once).
**Cause:** Same domain gap. The classifier learned MoCap-specific geometry. The mismatch detector handles it correctly — fires a warning — but it confirms the model is unusable as a primary signal on phone video. **This is the reason we built the user-pre-selection UX in the first place.**

### 5.5 The unifying root cause
All four failures are the **synthetic-pretrain-only domain gap**. The model has never seen a single real MediaPipe-from-phone-camera training sample. There is no transfer step in the training pipeline.

The previous chat surfaced two roads to fix this. They are reproduced verbatim in §7.

---

## 6. What we used and what we ignored from external sources

This section gets its own file: **`FIT3D_AIFIT_USAGE.md`**. Read that next.

---

## 7. The two roads forward (this is the next decision)

### Road A — MediaPipe-domain pseudo-supervised retrain (~$3, ~3–4 hr Colab A100)
1. Render Fit3D's 200-ish clean clips back into RGB images via the camera-60457274 angle.
2. Run MediaPipe on those rendered images to extract phone-style 3D joints (with realistic noise).
3. Use the existing Fit3D ground-truth labels (clean → quality=1.0, joint_errors=0, plus the existing synthetic corruption variants) but with the **MediaPipe-derived joints stream** as input.
4. Warm-start from v4, fine-tune for 10–15 epochs on this MediaPipe-domain data.
5. **Expected outcome:** the classifier collapse on real video should disappear; the joint-error head should wake up because it'll see real noise during training.
6. **Risk:** rendering Fit3D back to images is non-trivial. If we can't access the original RGB videos in the right resolution, this won't work cleanly. Camera-60457274 is in `videos.zip` — check there first.
7. **Code that already exists for this:** `backend/training/preprocessing/mediapipe_loader.py`, `backend/data/mediapipe_fit3d/` (132 MB of pre-extracted MediaPipe-on-Fit3D joints — this might already be enough to skip step 1–2). **The next chat should look at this directory first** — if it has all 8 subjects pre-extracted, Road A is nearly free.

### Road B — Real-world fine-tune on user-recorded squats (~30–50 clips, evening shoot + 1 hr scripting)
1. User records 30–50 squat clips on phone, varying quality intentionally (clean form, knees-cave, back-rounds, shallow ROM, hip-shift, etc.).
2. Manual labelling: each clip gets a quality scalar ∈ [0,1] and joint-error tags (knees / hips / back / good).
3. Scaffold a `backend/training/fine_tune_real.py` that warm-starts from v4 weights, freezes lower TCN layers, fine-tunes the heads on this small dataset for 10 epochs with early stopping.
4. **Expected outcome:** the model finally sees real bad form during training. The quality scalar should differentiate good/bad meaningfully on out-of-sample real video.
5. **Academic story:** "synthetic pretrain + real-world fine-tune" — a respected pattern (cf. domain-adaptation literature).
6. **Risk:** 30–50 clips is small. Per-joint-error labels are subjective. This can overfit hard.

### Recommended order
**Road A first** (cheap, no extra data collection, leverages already-downloaded MediaPipe-on-Fit3D extractions). If Road A doesn't move the needle on quality_pearson_r above ~0.45, layer **Road B** on top.

---

## 8. Smoke-test commands the next chat should run first

```powershell
# Confirm backend imports and v4 weights load
cd "C:\Users\tsh_x\Desktop\FitNova Application"
python -c "from backend.services.form_analyzer import FormAnalyzer; a=FormAnalyzer(model_dir='backend/models/form_model'); print('ready=', a.model_ready)"

# Run the 17-test green suite
python -m pytest backend/tests/test_exercises_consistency.py backend/tests/test_exercise_sanity.py backend/tests/test_angular_features_rotation_invariance.py -q

# Re-run the failing user-video validation to confirm baseline
python backend/tests/validate_user_squats.py
# Expected: per-joint knee/hip = 0.000 across all 4 videos, quality delta < 0.05.

# Inspect the locked-overlay annotated MP4s if they don't exist
python -m backend.training.evaluation.visualize_pipeline "C:\Users\tsh_x\Downloads\Good_Squats.mp4" --selected-exercise squat --out "C:\Users\tsh_x\Downloads\Good_Squats_LOCKED.mp4" --every 3

# Check if MediaPipe-on-Fit3D extractions already exist (Road A pre-req)
ls backend/data/mediapipe_fit3d/
ls backend/data/form_dataset_mediapipe/

# Backend smoke test
uvicorn backend.app:app --host 0.0.0.0 --port 8000
# Then in another shell:  curl http://localhost:8000/api/exercises | head
```

---

## 9. Things the next chat should NOT do without explicit user approval

- Do not modify NeuMF / `recommender.py` / `test_recommender.py`. The fitness planning feature is locked.
- Do not delete v1/v2/v3 model backups under `backend/models/form_model_v*_backup_*/` — they are evidence of the iteration history for the defence committee.
- Do not retrain quality-only without retraining classifier + joint-error heads — they share the TCN backbone and a quality-only retrain will silently regress the others.
- Do not ship academic claims that conflate test_metrics (s11 MoCap) with real-world generalisation. The two numbers are not interchangeable.
- Do not introduce another "augmentation" that operates on the wrong tensor — the bug we hunted in A2 was exactly that. Always confirm augmentations apply to the *corrupted* tensor when the label depends on the corruption.

---

## 10. Useful artifacts the next chat should know exist

- `backend/models/form_model/reality_check/user_squat_validation.json` — the failing F3 report.
- `backend/models/form_model/confusion_matrix_test.png` — block-diagonal visual for the slide deck.
- `~/.claude/plans/glistening-inventing-steele.md` — the full Phase A–F plan-of-record.
- `~/.claude/projects/C--Users-tsh-x-Desktop-FitNova-Application/memory/MEMORY.md` — user profile + project notes.
- `~/.claude/projects/.../50c56bf1-c838-4530-85c3-ecff44e0ed50.jsonl` — full transcript of this sprint, if context-recovery is needed.

---

## 11. Outstanding TODOs (in priority order)

1. **Decide Road A vs Road B vs both.** This is the only thing actively blocking defence.
2. Run `reality_check.py --subject s11` on v4 weights — the F1 artifact is missing.
3. (Low priority) Patch Cell 10 of `FitNova_Colab.ipynb` to include `confusion_matrix_*.{npy,png}` in its Drive-sync whitelist — last retrain we had to manually rescue these files.
4. (Low priority) Update `PROJECT_OVERVIEW.md` and `TECHNICAL_DEEP_DIVE.md` to mention form-analysis (currently they are NeuMF-era).
