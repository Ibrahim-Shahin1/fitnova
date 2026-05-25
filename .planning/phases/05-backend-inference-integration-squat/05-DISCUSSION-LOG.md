# Phase 5: Backend Inference Integration (Squat) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-25
**Phase:** 5-backend-inference-integration-squat
**Areas discussed:** Rep segmentation, Ensemble vs single-seed, API output schema, Feedback source

---

## How this discussion went

Claude initially presented the four gray areas below as an `AskUserQuestion` multiSelect. The user corrected this as an over-eager question-dump: *"Its supposed you have access and answers to all these questions and that we have a fucking documents that get updated and wrote and planned. If you have any questions and want me to get back to the agents in the prior chat then give it to me."*

The miss: Claude had not yet read the **master plan** (`~/.claude/plans/what-pushed-me-back-iridescent-pnueli.md`) or the Phase 4 handoff — the documents that carry the Phase 5 design. After reading them, the master plan was found to **specify the Phase 5 design and API contract verbatim**, resolving nearly all four areas. The rest are engineering calls Claude made and labeled for researcher confirmation. **No blocking questions remained for prior sessions.**

---

## Rep segmentation

| Option | Description | Selected |
|--------|-------------|----------|
| Whole-clip-as-one-rep | Treat upload as a single rep; one prediction | partial |
| Fixed sliding-window | Classify a rolling 32-frame window on a cadence | ✓ (live) |
| No-pose motion-energy counter | Segment reps via motion signal, then per-rep classify | ✓ (candidate) |
| Keep MediaPipe for rep boundaries only | Pose used only for segmentation | rejected |

**Resolution (from master plan lines 100-101, 126-131, 140):** Dedicated rep-segmentation component, **no pose** (pose explicitly not reused). Upload segments server-side and ships first; live uses lightweight sliding-window on-the-fly ("sliding-window for timing"). Single-rep ~3 s clip = simplest correct case. Exact no-pose algorithm → researcher resolves + measure on a real clip (CONTEXT D-03/D-04).

## Ensemble vs single-seed

| Option | Description | Selected |
|--------|-------------|----------|
| 3-seed ensemble | Test 0.6304, 3× R(2+1)D-18 forward | ✓ (default) |
| Single-seed | ~0.612, 1× forward, faster | ✓ (fallback) |
| Configurable | Seed-count as a config knob | ✓ |

**Resolution (Claude's discretion — not fixed by docs):** Default to the ensemble; **measure CPU latency first** on the target machine (server is CPU-only); fall back to single-seed for live if 3× is too slow; upload can afford the ensemble. Seed-count is a config knob (CONTEXT D-08).

## API output schema

| Option | Description | Selected |
|--------|-------------|----------|
| New Squat-error schema | `{exercise, errors:[{type,detected,confidence,severity_word,intervals}]}` | ✓ |
| Shim into old FormFrameResult | Keep landmarks/joint_errors[10]/quality for current-app compat | rejected |

**Resolution (LOCKED by master plan lines 126-131):** The new binary+timing schema is already specified in the master plan; old `FormFrameResult`/`FormSessionSummary` are replaced, not shimmed. No percentages shown to the user; `severity_word` is a confidence-derived estimate. Flutter UI rework is explicitly deferred (later milestone); the bar is "verified via API" (CONTEXT D-05).

## Feedback source

| Option | Description | Selected |
|--------|-------------|----------|
| GPT-4o-mini coaching layer | Reuse existing LLM coach for plain-language feedback | optional |
| Deterministic templated text | Confidence→severity_word → template, no API cost | ✓ |

**Resolution (Claude's discretion):** Deterministic templated feedback ("Knees inward — detected"), matching the master plan's example and "no percentages." Existing GPT-4o-mini coach is optional session-level polish, not required (CONTEXT D-10).

---

## Claude's Discretion

- Rep-segmentation exact algorithm (D-04) — researcher resolves.
- Ensemble-vs-single default + latency probe (D-08).
- Weights transfer mechanism Drive→server (D-09).
- Deterministic feedback (D-10).

## Deferred Ideas

- Polished Flutter form UI + any current-app compatibility shim (v2).
- OHP (Phase 6), image errors (Phase 7), cross-method ensemble + full comparison (Phase 8).
- Graded-severity ground truth (v2 FB-01).
- Backend security hardening (upload-size cap, rate limiting, CORS allowlist) — out of scope unless trivial.
- GPU/ONNX/TorchServe serving — only if CPU latency proves unworkable.
