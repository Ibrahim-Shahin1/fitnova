# Roadmap: FitNova — Form-Correction Rebuild (Fitness-AQA)

## Overview

This milestone rebuilds FitNova's form-correction feature from scratch on the Fitness-AQA dataset. Phases 1–5 deliver a complete Squat vertical slice — dataset → pipeline → supervised baseline → domain-knowledge SSL → backend inference API. Phases 6–8 extend the same pipeline to Overhead Press, add the image-based error method for Shallow-Squat and Barbell Row, and finish with the ensemble, full evaluation, and visualization pack. Every phase produces visualizations and, from Phase 3 on, F1 scores measured on the dataset's official splits.

## Phases

- [ ] **Phase 1: Dataset Consolidation & EDA** - Extract, merge and verify the dataset; full exploratory analysis
- [ ] **Phase 2: Squat Data Pipeline & Colab Harness** - PyTorch loaders + a resumable Colab training harness
- [ ] **Phase 3: Squat Supervised Baseline** - R(2+1)D-18 baseline for KIE/KFE, F1 on the official split
- [ ] **Phase 4: Squat Motion-Disentangling SSL** - Self-supervised pretraining + fine-tune + comparison
- [ ] **Phase 5: Backend Inference Integration (Squat)** - Replace the old subsystem; live + upload API
- [ ] **Phase 6: Overhead Press** - Extend the video pipeline to OHP Elbow/Knees errors
- [ ] **Phase 7: Image-Based Errors (CVCSPC)** - Shallow-Squat and Barbell Row detectors
- [ ] **Phase 8: Ensemble, Evaluation & Visualization Pack** - Final integration and the comparison deliverable

## Phase Details

### Phase 1: Dataset Consolidation & EDA
**Goal**: One clean, verified Fitness-AQA dataset tree and a complete characterization of it.
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03
**Success Criteria** (what must be TRUE):
  1. The nested video/image archives are extracted and the 4 split-folders merged into one consistent tree
  2. Sample counts per exercise/error reconcile with the official train/val/test split JSONs
  3. An EDA report exists with visualizations: class balance, clip-length/property distributions, sample frames, barbell-trajectory curves, error co-occurrence
**Plans**: TBD

### Phase 2: Squat Data Pipeline & Colab Harness
**Goal**: A tested PyTorch data pipeline for Squat KIE/KFE and a Colab training harness that survives disconnects.
**Depends on**: Phase 1
**Requirements**: SQUAT-01, SQUAT-02
**Success Criteria** (what must be TRUE):
  1. Squat clips load as (video tensor, KIE/KFE label) batches using the official splits
  2. Decoded and augmented sample frames are visualized and confirmed correct
  3. The Colab harness checkpoints to Google Drive and resumes from the latest checkpoint after a restart
**Plans**: TBD

### Phase 3: Squat Supervised Baseline
**Goal**: A working supervised baseline detector for Squat KIE and KFE with F1 on the official test split.
**Depends on**: Phase 2
**Requirements**: SQUAT-03
**Success Criteria** (what must be TRUE):
  1. An R(2+1)D-18 (Kinetics-init) model trains to convergence on the labeled Squat set
  2. F1 per error is reported on the official test split, in the range of the paper's supervised baseline
  3. Training curves, confusion matrices and PR curves are produced
**Plans**: TBD

### Phase 4: Squat Motion-Disentangling SSL
**Goal**: Domain-knowledge self-supervised pretraining that improves Squat error detection over the baseline.
**Depends on**: Phase 3
**Requirements**: SQUAT-04, SQUAT-05
**Success Criteria** (what must be TRUE):
  1. Motion-Disentangling SSL pretraining runs on the unlabeled Squat set and converges
  2. The MD-pretrained model, fine-tuned for KIE/KFE, reports F1 ≥ the Phase-3 baseline
  3. A comparison table/chart places our F1 against Parmar and GYMetricPose on identical metrics
**Plans**: TBD

### Phase 5: Backend Inference Integration (Squat)
**Goal**: The Squat form detector served through the backend in live and video-upload modes.
**Depends on**: Phase 4
**Requirements**: API-01, API-02, API-03, API-04
**Success Criteria** (what must be TRUE):
  1. The old form-analysis subsystem is removed and a PyTorch form-inference service loads in FastAPI
  2. Uploading a squat clip returns binary error detections with timing
  3. A live WebSocket session segments reps and returns per-rep error feedback
**Plans**: TBD

### Phase 6: Overhead Press
**Goal**: Form-error detection for Overhead Press (Elbows, Knees) using the established video pipeline.
**Depends on**: Phase 5
**Requirements**: OHP-01
**Success Criteria** (what must be TRUE):
  1. OHP baseline and MD-SSL models train on the official OHP splits
  2. F1 per error is reported and compared to the published numbers
  3. OHP is selectable through the inference API
**Plans**: TBD

### Phase 7: Image-Based Errors (CVCSPC)
**Goal**: Detection of the static (image-based) errors — Shallow-Squat and Barbell Row Lumbar/Torso.
**Depends on**: Phase 6
**Requirements**: IMG-01, IMG-02, IMG-03
**Success Criteria** (what must be TRUE):
  1. A CVCSPC (ResNet-18) image pipeline trains baseline and self-supervised models
  2. Shallow-Squat and Barbell Row Lumbar/Torso detectors report F1 on the official splits
  3. These errors are served through the inference API
**Plans**: TBD

### Phase 8: Ensemble, Evaluation & Visualization Pack
**Goal**: The complete 3-exercise form-correction subsystem with a defensible evaluation deliverable.
**Depends on**: Phase 7
**Requirements**: EVAL-01, EVAL-02, EVAL-03
**Success Criteria** (what must be TRUE):
  1. The MD + CVCSPC ensemble is evaluated where the paper applies it
  2. A full identical-metric comparison covers every error vs Parmar / GYMetricPose / LMM
  3. A consolidated visualization and results pack is assembled
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Dataset Consolidation & EDA | 0/TBD | Not started | - |
| 2. Squat Data Pipeline & Colab Harness | 0/TBD | Not started | - |
| 3. Squat Supervised Baseline | 0/TBD | Not started | - |
| 4. Squat Motion-Disentangling SSL | 0/TBD | Not started | - |
| 5. Backend Inference Integration (Squat) | 0/TBD | Not started | - |
| 6. Overhead Press | 0/TBD | Not started | - |
| 7. Image-Based Errors (CVCSPC) | 0/TBD | Not started | - |
| 8. Ensemble, Evaluation & Visualization Pack | 0/TBD | Not started | - |
