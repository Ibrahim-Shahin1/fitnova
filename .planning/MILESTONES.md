# Milestones

## v1.0 Form-Correction Rebuild (Shipped: 2026-05-31)

**Scope:** 8 phases, 24 plans. Form-correction feature rebuilt from scratch on the **Fitness-AQA** dataset (Parmar et al., ECCV 2022) — faithful methods, identical metrics (F1 per error on the official splits), PyTorch. Replaces the scrapped v4–v7 / QEVD ST-GCN line. (The plan-generation / recommender feature was untouched.)

**Delivered:** A defensible exercise-form-error detector for **5 errors across 3 exercises**, evaluated on the official Fitness-AQA test splits with identical F1 metrics — **matching or beating Parmar et al. (ECCV 2022) on every error.**

**Key accomplishments:**

- **Squat KIE/KFE** (video · Motion-Disentangling SSL · R(2+1)D-18): macro-F1 **0.6304** (KIE 0.420 / KFE 0.841) — matches the paper's MD-SSL (0.6262) on the official 244-clip split; **+0.087** over the supervised baseline.
- **Overhead Press Elbows/Knees** (video · MD-SSL): macro-F1 **0.6622** (Elbows 0.447 / Knees 0.877) — matches/edges the paper MD (0.6502), **beats it on Knees** (0.877 vs 0.845); +0.050 SSL lift; val→test gap 0.011.
- **Shallow-Squat depth** (image · faithful CVCSPC · ResNet-18): F1 **0.8902** — **beats the published CVCSPC (0.8694)** by +0.021 and SimSiam/OpenPose-TDM by ~0.06; +0.0152 SSL lift over the supervised baseline; triplet-acc 0.48→0.959.
- **Backend inference API rebuilt in PyTorch** (SquatFormService: REST upload + live WebSocket, motion-energy rep-segmentation, binary+timing schema); the old TF/MediaPipe subsystem reversibly archived. (The form-correction frontend was cancelled — D1 — so this serves as the API-level deliverable, not a live UI.)
- **Consolidated evaluation + visualization pack** (defense-weighted): 12 reproducible pre-executed notebooks, per-exercise + cross-exercise FINDINGS, the EVAL-01 master comparison table + 4 master figures incl. a two-method methodology diagram — **every number recomputed-in-code from the committed `results.pkl` and asserted (no fabrication)**. MD+CVCSPC ensemble honestly reported as not-applicable to the single-method-per-error scope; GYMetricPose (Gallardo 2024) + LMM (Dibenedetto 2025) sourced as caveated related-work context, not comparison columns.

**Requirements:** all 19 v1 requirements complete (IMG-03 BarbellRow descoped — compute/time).

**Scope notes / deferred:**
- **Descoped this milestone:** BarbellRow Lumbar/Torso (IMG-03); the form-correction frontend + UI-less serving integration for OHP/Shallow (the offline benchmark is the defensible deliverable; in-the-wild/phone-camera transfer is the known domain-shift limit).
- **v2 (future milestone):** polished Flutter form UI (UI-01/UI-02); graded error severity as a measured target (FB-01).
- **Known deferred items at close: 2** (Phase-05 HUMAN-UAT 2 live-demo scenarios + quick-task `260527-2bt` unit-3 upload-replay) — both superseded by the frontend cancellation (D1), not coverage gaps. See STATE.md `## Deferred Items`.

**Full detail:** [`milestones/v1.0-ROADMAP.md`](milestones/v1.0-ROADMAP.md) · [`milestones/v1.0-REQUIREMENTS.md`](milestones/v1.0-REQUIREMENTS.md). Consolidated evaluation: [`../docs/eval/FINDINGS_FULL.md`](../docs/eval/FINDINGS_FULL.md).

---
