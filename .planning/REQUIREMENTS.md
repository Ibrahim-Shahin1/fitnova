# Requirements: FitNova — Form-Correction Rebuild (Fitness-AQA)

**Defined:** 2026-05-19
**Core Value:** A user can record/upload a lift and get trustworthy, plain-language form-error feedback grounded in a published dataset and method.

## v1 Requirements

Requirements for this milestone. Each maps to a roadmap phase.

### Dataset

- [ ] **DATA-01**: Fitness-AQA dataset fully extracted (including nested video/image archives) and consolidated into one tree
- [ ] **DATA-02**: Dataset sample counts reconciled against the official train/val/test split JSONs
- [ ] **DATA-03**: EDA report produced with visualizations (class balance, clip properties, barbell trajectories, error co-occurrence)

### Squat Model

- [ ] **SQUAT-01**: PyTorch data pipeline for Squat KIE/KFE — decode, 32-frame sampling, transforms, official-split loaders
- [ ] **SQUAT-02**: Resumable Colab training harness — checkpoints to Google Drive each epoch, auto-resumes after a disconnect
- [x] **SQUAT-03**: Supervised baseline (R(2+1)D-18, Kinetics-init) trained; F1 per error on the official Squat test split  *(complete 2026-05-20 — test KIE 0.286 / KFE 0.800 / macro 0.543 at val-tuned thresholds; within 2% of paper Kinetics row)*
- [ ] **SQUAT-04**: Motion-Disentangling self-supervised pretraining on the unlabeled Squat set
- [ ] **SQUAT-05**: MD-pretrained model fine-tuned for KIE/KFE; F1 vs baseline and vs published numbers

### Overhead Press Model

- [x] **OHP-01**: OHP Elbow/Knees baseline and MD-SSL models trained; F1 per error on the official OHP splits  *(complete 2026-05-30 — official OHP test: Elbows 0.447 / Knees 0.877 / macro 0.662; matches/edges paper MD 0.650, +0.050 over baseline, val→test gap 0.011; serving descoped per CONTEXT D1)*

### Image-Based Errors

- [x] **IMG-01**: CVCSPC (ResNet-18) image pipeline — baseline and self-supervised  *(complete 2026-05-30 — ImageNet ResNet-18 supervised baseline + faithful CVCSPC pose-contrastive SSL; triplet-accuracy 0.48→0.959; met by model+eval+viz)*
- [x] **IMG-02**: Shallow-Squat detector trained; F1 on the official split  *(complete 2026-05-30 — official 540-crop test F1 0.8902, 3-seed CVCSPC ensemble; beats paper CVCSPC 0.8694, +0.0152 SSL lift over baseline 0.8750, val→test gap 0.0217)*
- [~] **IMG-03**: Barbell Row Lumbar and Torso-Angle detectors trained; F1 on the official splits  *(DESCOPED 2026-05-30 — BarbellRow cancelled by user for compute/time; milestone form-correction scope is Squat + OHP (video) + Shallow-Squat (image))*

### Inference API

- [x] **API-01**: Old form-analysis subsystem removed; PyTorch form-inference service loaded in FastAPI
- [x] **API-02**: Rep-segmentation component for the live and upload inference paths
- [x] **API-03**: Video-upload REST endpoint returning binary error detections with timing
- [x] **API-04**: Live WebSocket endpoint returning per-rep error feedback

### Evaluation & Visualization

- [x] **EVAL-01**: Identical-metric comparison (ours vs Parmar / GYMetricPose / LMM) across all errors  *(complete 2026-05-31 -- master 5-error comparison table vs Parmar; all values recomputed-in-code from results.pkl; GYMetricPose + LMM as labeled context with caveats)*
- [x] **EVAL-02**: MD + CVCSPC ensemble evaluated where the paper applies it  *(complete 2026-05-31 -- evaluated -> not-applicable; paper applies MD+CVCSPC only to BackSquat KIE/KFE (Table 2: 0.5263 / 0.8468); our scope has exactly one method per error so no error has both an MD and a CVCSPC score)*
- [x] **EVAL-03**: Consolidated visualization / results pack  *(complete 2026-05-31 -- docs/notebooks/12_full_evaluation.py + .ipynb (pre-executed) + docs/eval/FINDINGS_FULL.md + 4 master figures in docs/figures/)*

## v2 Requirements

Deferred to a future milestone.

### Form UI

- **UI-01**: Polished Flutter live-camera form screen
- **UI-02**: Polished Flutter video-upload and results screens

### Feedback

- **FB-01**: Graded error severity as a measured target (requires an added labeling/data strategy)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Recommendation / plan-generation feature | Already working — untouched this milestone |
| Reviving v4/v5/v6/v6.1/v7/QEVD form-model work | Scrapped entirely; produced no defensible result |
| LLaVA-Video LMM approach | Needs 4×A100 and underperforms the chosen method |
| MediaPipe pose pipeline for the new model | Fitness-AQA method is raw-pixel CNN; pose is unreliable in-the-wild per the paper |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| SQUAT-01 | Phase 2 | Pending |
| SQUAT-02 | Phase 2 | Pending |
| SQUAT-03 | Phase 3 | Complete |
| SQUAT-04 | Phase 4 | Pending |
| SQUAT-05 | Phase 4 | Pending |
| API-01 | Phase 5 | Complete |
| API-02 | Phase 5 | Complete |
| API-03 | Phase 5 | Complete |
| API-04 | Phase 5 | Complete |
| OHP-01 | Phase 6 | Complete |
| IMG-01 | Phase 7 | Complete |
| IMG-02 | Phase 7 | Complete |
| IMG-03 | Phase 7 | Descoped |
| EVAL-01 | Phase 8 | Complete |
| EVAL-02 | Phase 8 | Complete |
| EVAL-03 | Phase 8 | Complete |

**Coverage:**

- v1 requirements: 19 total
- Mapped to phases: 19
- Unmapped: 0 ✓

---
*Requirements defined: 2026-05-19*
*Last updated: 2026-05-19 after initialization*
