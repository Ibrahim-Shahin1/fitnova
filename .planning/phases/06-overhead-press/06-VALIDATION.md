---
phase: 6
slug: overhead-press
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-27
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution. Mirrors the Phase 3/4 pytest setup; the per-task map is populated by the planner.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x (already in `backend/requirements.txt`; Phases 2–4 precedent) |
| **Config file** | none — markers used ad hoc (`@pytest.mark.slow` for model-build tests) |
| **Quick run command** | `python -m pytest backend/training/aqa/ -m "not slow" -q` |
| **Full suite command** | `python -m pytest backend/training/aqa/ -q` |
| **Estimated runtime** | ~5–10 s quick (pure-function unit tests); model-build `slow` tests download R(2+1)D-18 weights once |

Pure-function OHP code (split loader, trajectory bbox-parser, half-cycle splitter, dataset wiring) is verified **locally on CPU** before any Colab GPU burn — the Phase-4 Wave-0 pattern. Training-run claims (convergence, F1) are measured on Colab L4 and captured in `results.pkl`.

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest backend/training/aqa/ -m "not slow" -q`
- **After every plan wave:** Run the full suite (`python -m pytest backend/training/aqa/ -q`)
- **Before phase verification:** Full suite green; the gated Colab trajectory probe signed off; F1-on-official-test-split recorded in `results.pkl`
- **Max feedback latency:** ~10 s (local unit suite)

---

## Per-Task Verification Map

*Populated by the planner once plan/task IDs exist. Each pure-function task gets an `<automated>` pytest assertion (Phase-4 Wave-0 pattern); each training task is validated by a measured artifact (linear-probe macro-F1, per-epoch curves, official test-split F1 in `results.pkl`).*

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 06-01-01 | 01 | 0 | OHP-01 | — | N/A (offline research-data pipeline) | unit | `python -m pytest backend/training/aqa/datasets/test_ohp.py -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `backend/training/aqa/datasets/test_ohp.py` — OHP split loader + dataset wiring + train-derived `pos_weight` (Elbows ≈ 2.89 / Knees ≈ 1.92) for OHP-01
- [ ] `backend/training/aqa/datasets/test_ohp_ssl.py` — OHP trajectory **bbox-parser** (region-0 `y_center=(y1+y2)/2`, empty-frame interpolation) + half-cycle splitter sign (`bottom_is_argmax=False`, OHP argMIN) on synthetic + sample-file inputs
- [ ] Extend the splits test for `index_ohp()` + `OHPClipRecord` (counts reconcile to 1582/339/339)

*The MD-SSL math primitives (triplet loss, projector, ensemble, TTA) are already covered by the Phase-4 tests and reused unchanged.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Trajectory format + half-cycle sign correctness | OHP-01 | Requires visual inspection of a real OHP clip overlaid with the parsed bar y-trajectory + detected mid-rep split point; cannot be asserted offline (Phase-4 Wave-1 precedent) | Gated Colab probe on 5 OHP clips: extract region-0 `y_center`, plot over frames, confirm argMIN lands at the visually correct turning point, confirm 1:1 traj↔frame alignment. **Blocks SSL GPU burn.** |
| F1-per-error on the official OHP test split vs paper | OHP-01 | The headline academic result; eyeballed against paper Table 4 (Elbow 0.455 / Knees 0.845) | After ensemble eval: confirm `results.pkl` F1 values + the comparison figure; human reviews the 4-notebook OHP pack |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
