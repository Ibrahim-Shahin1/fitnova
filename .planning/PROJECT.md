# FitNova

## What This Is

FitNova is an AI-powered fitness mobile app (Flutter client + FastAPI/Python backend) with two features: a **workout plan generator** (conversational intake + a layered recommender that produces a personalized 7-day plan) and **exercise form correction** (camera- or video-based detection of form errors with feedback). This milestone rebuilds the form-correction feature from scratch on the Fitness-AQA dataset; the plan generator already works and stays as-is.

## Core Value

A user can record or upload themselves doing a lift and get trustworthy, plain-language feedback on specific form errors — feedback grounded in a published dataset and method, not guesswork.

**Current state — v1.0 shipped 2026-05-31:** the form-correction model + benchmark evaluation are complete for 5 errors across 3 exercises (Squat KIE/KFE + OHP Elbows/Knees via video MD-SSL; Shallow-Squat depth via image CVCSPC), matching/beating Parmar et al. (ECCV 2022) on the official splits with identical F1 metrics. The serving API was rebuilt in PyTorch (UI-less); the polished Flutter form UI and graded severity are deferred to v2.

## Requirements

### Validated

<!-- Shipped / working in the existing codebase. -->

- ✓ Conversational fitness-intake (GPT-4o-mini) extracting user profile parameters — existing
- ✓ Layered workout recommender (content filter → NeuMF → LLM plan adapter) producing a 7-day plan — existing
- ✓ Flutter app shell: registration, chat, plan screens, theming, state management — existing
- ✓ FastAPI backend with REST + WebSocket transport and static media serving — existing
- ✓ Fitness-AQA dataset consolidated, verified, characterized (EDA + report) — v1.0 (Phase 1)
- ✓ Squat form-error detection (KIE/KFE): supervised baseline → Motion-Disentangling SSL, official-split F1 — v1.0 (macro 0.6304, matches paper MD 0.6262)
- ✓ Overhead Press form-error detection (Elbows/Knees): baseline → MD-SSL, official-split F1 — v1.0 (macro 0.6622, matches/edges paper 0.6502)
- ✓ Image-based error (Shallow-Squat depth) via faithful CVCSPC — v1.0 (F1 0.8902 > paper CVCSPC 0.8694; BarbellRow/IMG-03 descoped)
- ✓ Backend inference API — live (WebSocket) + video-upload (REST), binary + timing — v1.0 (PyTorch SquatFormService; UI-less — the form frontend was cancelled)
- ✓ Identical-metric comparison vs Parmar (+ GYMetricPose/LMM as caveated context), visualizations throughout — v1.0 (FINDINGS_FULL + notebook 12 + master figures)

### Active

<!-- Next milestone not yet defined. The form-correction model + benchmark evaluation shipped in v1.0. -->

- [ ] (v2) Polished Flutter form-correction UI — live-camera + video-upload + results screens (UI-01 / UI-02)
- [ ] (v2) Graded error severity as a measured target — needs an added labeling/data strategy (FB-01)

### Out of Scope

- The recommendation / plan-generation feature — already working; untouched this milestone
- Polished Flutter form-correction UI — a later milestone; this milestone delivers the model + inference API
- Graded error severity as a measured target — dataset labels are binary; severity wording is a confidence-derived UX layer only
- The LLaVA-Video LMM approach — needs 4×A100 and underperforms the chosen method
- Reviving any v4/v5/v6/v6.1/v7/QEVD form-model work — scrapped entirely

## Context

- The prior form-correction line (MediaPipe pose → ST-GCN, across v4–v7 / QEVD / VLM-distillation) produced no defensible result and is discarded. Root cause of the failure: building before the data and method were understood.
- New foundation: the **Fitness-AQA** dataset and paper (Parmar, Gharat, Rhodin — ECCV 2022, arXiv:2202.14019). In-the-wild gym video for 3 exercises, expert error labels + large unlabeled sets for self-supervised pretraining. Official code at `github.com/ParitoshParmar/Fitness-AQA` (`Code_Release/`).
- The Fitness-AQA method is raw-pixel CNN-based — the paper's thesis is that pose estimation is unreliable in-the-wild — so the existing MediaPipe pose pipeline does not transfer.
- Heavy training runs on Colab (the user has a Colab subscription). Notebooks must be checkpoint/resume-safe by construction.
- Approved implementation plan: `~/.claude/plans/what-pushed-me-back-iridescent-pnueli.md`.
- Visualizations are a stated priority and a deliverable in every phase.

## Constraints

- **Tech stack**: Form model in PyTorch (R(2+1)D-18 / ResNet-18 — official code and paper architectures are torch-native) — Why: faithful reproduction, lower risk. The existing recommender stays TensorFlow; FastAPI loads both.
- **Method fidelity**: use the dataset's official train/val/test splits verbatim and report F1-score per error — Why: identical metrics to published work for a defensible comparison.
- **Compute**: heavy training on Colab; every notebook checkpoints to Google Drive and resumes after a disconnect — Why: Colab runtimes are not durable.
- **Working mode**: interactive — one runnable unit at a time, user runs and pastes outputs back, no blind runs — Why: the prior failure came from building on un-understood data.
- **Data**: Fitness-AQA is non-commercial / research-use only.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Rebuild form-correction on Fitness-AQA | Published dataset + method → reproducible, academically defensible; replaces a failed line | ✓ Shipped v1.0 |
| Scrap the entire v4–v7 / QEVD form line | Produced no defensible result; not worth measuring against | ✓ Shipped v1.0 |
| Squat-first vertical slice | Richest data (labeled + unlabeled + trajectories); proves the full pipeline before scaling | ✓ Shipped v1.0 |
| Supervised baseline before domain-knowledge SSL | Guarantees an early paper-comparable result; the SSL lift becomes the measured contribution | ✓ Shipped v1.0 |
| Binary detection + timing (no graded severity) | Matches what the dataset labels support; honest about measured vs estimated | ✓ Shipped v1.0 |
| PyTorch for the form model | Official code and R(2+1)D / ResNet are torch-native | ✓ Shipped v1.0 |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition:**
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone:**
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-31 after v1.0 (Form-Correction Rebuild) milestone*
