# Roadmap: FitNova — Form-Correction Rebuild (Fitness-AQA)

## Milestones

- ✅ **v1.0 Form-Correction Rebuild** — Phases 1–8 (shipped 2026-05-31)

## Phases

<details>
<summary>✅ v1.0 Form-Correction Rebuild (Phases 1–8) — SHIPPED 2026-05-31</summary>

Form-correction rebuilt from scratch on the Fitness-AQA dataset (Parmar et al., ECCV 2022) — faithful methods, identical F1 metrics on the official splits. Matches/beats the paper on all 5 errors across 3 exercises (Squat KIE/KFE + OHP Elbows/Knees via video MD-SSL; Shallow-Squat depth via image CVCSPC). Full phase detail: [`milestones/v1.0-ROADMAP.md`](milestones/v1.0-ROADMAP.md). Consolidated evaluation: [`../docs/eval/FINDINGS_FULL.md`](../docs/eval/FINDINGS_FULL.md).

- [x] Phase 1: Dataset Consolidation & EDA (1/1) — 2026-05-20
- [x] Phase 2: Squat Data Pipeline & Colab Harness (1/1) — 2026-05-20
- [x] Phase 3: Squat Supervised Baseline (1/1) — 2026-05-20
- [x] Phase 4: Squat Motion-Disentangling SSL (4/4) — 2026-05-25
- [x] Phase 5: Backend Inference Integration — Squat (5/5) — 2026-05-26
- [x] Phase 6: Overhead Press (5/5) — 2026-05-29
- [x] Phase 7: Image-Based Errors — CVCSPC (5/5) — 2026-05-30 *(BarbellRow/IMG-03 descoped)*
- [x] Phase 8: Ensemble, Evaluation & Visualization Pack (2/2) — 2026-05-31

</details>

### 📋 Next milestone — not yet defined

Candidate v2 scope (deferred): polished Flutter form-correction UI (UI-01/UI-02) + graded error severity (FB-01). Run `/gsd:new-milestone` to scope it.

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Dataset Consolidation & EDA | v1.0 | 1/1 | Complete | 2026-05-20 |
| 2. Squat Data Pipeline & Colab Harness | v1.0 | 1/1 | Complete | 2026-05-20 |
| 3. Squat Supervised Baseline | v1.0 | 1/1 | Complete | 2026-05-20 |
| 4. Squat Motion-Disentangling SSL | v1.0 | 4/4 | Complete | 2026-05-25 |
| 5. Backend Inference Integration (Squat) | v1.0 | 5/5 | Complete | 2026-05-26 |
| 6. Overhead Press | v1.0 | 5/5 | Complete | 2026-05-29 |
| 7. Image-Based Errors (CVCSPC) | v1.0 | 5/5 | Complete | 2026-05-30 |
| 8. Ensemble, Evaluation & Visualization Pack | v1.0 | 2/2 | Complete | 2026-05-31 |

**Total:** 8/8 phases, 24/24 plans complete. v1.0 shipped 2026-05-31.
