# FitNova — Living Retrospective

## Milestone: v1.0 — Form-Correction Rebuild

**Shipped:** 2026-05-31
**Phases:** 8 | **Plans:** 24

### What Was Built

The form-correction feature rebuilt from scratch on the Fitness-AQA dataset (Parmar et al., ECCV 2022) after the prior v4–v7/QEVD pose-based line produced no defensible result. Five errors across three exercises, faithful methods, identical F1 metrics on the official splits:

- Squat KIE/KFE — video Motion-Disentangling SSL (R(2+1)D-18): macro-F1 0.6304 (matches paper MD 0.6262).
- OHP Elbows/Knees — video MD-SSL: macro-F1 0.6622 (matches/edges paper 0.6502; beats on Knees).
- Shallow-Squat depth — image CVCSPC (ResNet-18): F1 0.8902 (beats paper CVCSPC 0.8694).
- PyTorch inference API (upload + live WebSocket) replacing the TF/MediaPipe subsystem.
- Consolidated evaluation + visualization pack (12 notebooks, FINDINGS, master figures), every number recomputed-in-code.

### What Worked

- **Published-dataset + published-method foundation.** Anchoring to Parmar et al. with identical splits/metrics made every result defensible and gave concrete targets to match. The root-cause fix for the prior failure ("built before understanding the data") held: each phase characterized its inputs before implementing.
- **Supervised-baseline-first, then SSL.** Guaranteed an early paper-comparable number per exercise and made the SSL contribution a measured lift, not a hope.
- **Resumable-by-construction Colab harness.** Atomic Drive checkpoints + auto-resume survived disconnects across multi-hour SSL runs.
- **Structural no-fabrication enforcement.** Computing every headline F1 in code from the committed `results.pkl` and asserting it (≤1e-6) — rather than hand-typing — caught the class of bug that bit Phase 3 and made the final deliverable auditable.
- **Verify-and-push-back discipline.** Independent verification caught real issues each phase (val/test leakage in Phase 7; the trajectory-format/half-cycle-sign probes; the notebook-12 `backend`-import-under-nbconvert bug in Phase 8).

### What Was Inefficient

- **SSL contrastive collapse on weak augmentations** (Phase 4) cost a re-run before the strong-aug recipe was locked — though it produced a useful ablation datapoint.
- **Planning-doc tracking lag.** Several phases completed without their REQUIREMENTS traceability rows / STATE counters being flipped, requiring hand-reconciliation at milestone close. The GSD state handlers' counters drifted from reality more than once.
- **In-the-wild domain shift** surfaced late relative to expectations — offline benchmark F1 did not transfer to home/phone camera, which (with the cancelled frontend) reframed the deliverable as benchmark-only.

### Patterns Established

- `results.pkl` (per-clip ensemble scores + curves + paper targets) as the single committed source of truth feeding offline notebooks/figures — decouples viz from GPU.
- One deliverable shape per exercise: results.pkl → notebook pack + FINDINGS + figures; consolidated at milestone end.
- Reuse the exercise-agnostic eval primitives (`eval/metrics.py`) unchanged; mirror established figure scripts.
- Honest framing carried in FINDINGS prose (no-paper-baseline notes, loss deviations, TTA-not-adopted, benchmark-only caveat) rather than silently reconciled.

### Key Lessons

- A published dataset + method + identical metrics is the cheapest path to a defensible ML thesis result.
- Make correctness structural (assert numbers from artifacts), not a promise — it survives context resets and catches drift.
- Offline F1 ≠ live quality; surface domain-shift early and scope the deliverable honestly around what's defensible.
- Hand-reconcile planning tracking at phase/milestone boundaries — don't trust the auto-counters blindly.

### Cost Observations

- Heavy training (R(2+1)D-18 MD-SSL, CVCSPC) ran on Colab L4/A100; offline assembly + eval ran locally (no GPU).
- Phase 8 (this consolidation) was fully local from committed pickles — no re-runs, no Colab.

---

## Cross-Milestone Trends

*(First milestone — trends accumulate from v1.1 onward.)*
