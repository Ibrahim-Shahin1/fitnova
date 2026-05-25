---
phase: 5
slug: backend-inference-integration-squat
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-25
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Detailed validation architecture (what proves each success criterion) lives in
> `05-RESEARCH.md` § Validation Architecture. The per-task map below is populated
> once `*-PLAN.md` tasks exist.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (`pytest>=8.0.0`, per CLAUDE.md) |
| **Config file** | none — tests live in `backend/services/test_*.py` and `backend/tests/` |
| **Quick run command** | `python -m pytest backend/services/ backend/tests/ -m "not slow" -q` |
| **Full suite command** | `python -m pytest backend/ -m "not slow"` |
| **Estimated runtime** | ~30–60 s (torch ensemble load dominates; API tests via FastAPI TestClient are fast) |

---

## Sampling Rate

- **After every task commit:** Run the quick run command (scoped to the new/modified test file).
- **After every plan wave:** Run the full suite command.
- **Before `/gsd:verify-work`:** Full suite must be green.
- **Max feedback latency:** ~60 s.

---

## Per-Task Verification Map

> Populated during/after planning (tasks do not exist yet). Each task's
> `<acceptance_criteria>` must map to an automated assertion or a Wave 0 stub.
> Source of truth for WHAT to verify: `05-RESEARCH.md` § Validation Architecture +
> the ROADMAP success criteria (old subsystem gone + PyTorch service loads;
> upload → binary+timing; live → per-rep feedback).

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| _(populated after planning)_ | | | API-01..04 | | | | | | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Test fixtures for the inference service (a sample squat clip + expected schema shape)
- [ ] FastAPI `TestClient` / httpx fixtures for the REST + WebSocket endpoints

*Refined when the planner assigns Wave 0 tasks.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Real-camera domain-shift check | API-04 (D-11) | Needs a real phone-camera squat clip from the user; offline F1 ≠ live (domain shift) | User records/uploads a real squat clip early; paste back the model's KIE/KFE confidences; compare against expected behavior (strong KFE, modest KIE) |

*Other behaviors target automated API verification (httpx REST + WS test client).*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60 s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
