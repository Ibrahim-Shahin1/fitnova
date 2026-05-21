# Phase 04: Squat Motion-Disentangling SSL - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-21
**Phase:** 04-squat-motion-disentangling-ssl
**Areas discussed:** Regularization (B), Ensemble (D), TTA (E), Overfit-abort thresholds

User elected to discuss all four areas explicitly: *"We will discuss all. I said we will gather all the information needed before we implement anything to be able to implement at the highest quality possible."*

---

## Regularization (B)

### Strength (wd + dropout combo)

| Option | Description | Selected |
|--------|-------------|----------|
| Moderate (rec.) | AdamW wd=1e-4, dropout=0.2 head-only | ✓ |
| Light | AdamW wd=1e-5, dropout=0.1; closer to Phase 3 + minor protection | |
| Aggressive | AdamW wd=5e-4, dropout=0.3; risk underfit on 1136 clips | |
| Sweep on val | Run all 3 on seed 42, pick best val macro-F1, ensemble that | |

**User's choice:** Moderate (rec.)
**Notes:** All options use AdamW (not Adam) for proper decoupled weight decay — locked regardless of strength.

### Epoch budget + LR schedule

| Option | Description | Selected |
|--------|-------------|----------|
| 20 + warmup (rec.) | Paper §5 budget, 5-epoch linear warmup → cosine | |
| 30 + warmup | Extra headroom, 8-epoch patience | |
| 20 hard cap | Strictly 20, cosine-only | |
| 50 + patience (P3 recipe) | Same as Phase 3: 50 epochs cosine + 8-epoch patience | ✓ |

**User's choice:** 50 + patience (P3 recipe)
**Notes:** Heaviest compute option but easiest P3-vs-P4 comparison. Reliance on early-stop + best.pt bounds the actual cost (Phase 3 stopped at 11). Logged as deliberate deviation from paper §5's 20-epoch budget. Warmup OFF, LR=1e-4 matching Phase 3 (researcher may override per paper §5).

---

## Ensemble (D)

### Aggregation method

| Option | Description | Selected |
|--------|-------------|----------|
| Mean sigmoids (rec.) | Average sigmoid scores across 3 seeds, per error head | ✓ (Claude's choice) |
| Mean logits | Sum logits then sigmoid (geometric mean of probs); more pessimistic | |
| Majority vote | Each seed binarizes at own threshold; 2/3 vote; no PR curve | |
| Best-of-3 (oracle) | Single best-val seed; not a true ensemble | |

**User's choice:** *"You know better. Think carefully and then choose."* → Claude chose **Mean of sigmoids**.
**Notes:** Rationale: continuous + calibrated → ensemble PR-AUC + threshold sweep work like Phase 3; standard practice (Lakshminarayanan et al. 2017 deep ensembles); robust on the small 243-clip val (36 KIE-positives) where per-seed thresholding overfits. 3 seeds (42/1337/7), shared SSL pretrain. Per-seed numbers reported as variance evidence. Single per-error threshold tuned on ensemble val scores.

---

## TTA (E)

### Recipe selection approach

| Option | Description | Selected |
|--------|-------------|----------|
| Val-tune TTA (rec.) | Sweep {temporal jitter, 5-crop, h-flip} on val; keep best-val combo | ✓ |
| Fixed safe recipe | Lock crop + temporal only (match training augs); skip flip | |
| Temporal only | Just temporal jitter; leanest for Phase 5 latency | |
| Include flip anyway | Handoff default (flip + temporal) without validation | |

**User's choice:** Val-tune TTA (rec.)
**Notes:** Claude pushed back on the handoff's "flip + temporal" default — Phase 3 trained flip-OFF, so flipped clips are OOD and flip-TTA could add noise. Temporal jitter + spatial crop match Phase 3 training augs (invariant by construction); flip becomes a candidate validated on val rather than an assumed-good default. [[feedback_ai_correctness]] "measure don't guess." Aggregation: per-seed TTA mean → mean across seeds → val-tuned threshold. Phase 4 also reports single-model/no-TTA number for Phase 5.

---

## Overfit-abort thresholds

### Runtime monitor behavior (train-val BCE ratio >10× before epoch 10)

| Option | Description | Selected |
|--------|-------------|----------|
| Flag + continue (rec.) | Log warning, finish run (best.pt+early-stop protect result) | |
| Abort that seed + bump reg | Stop seed, re-run at wd=5e-4/dropout=0.3 | ✓ |
| Hard abort + re-plan | Stop all, return to plan-phase | |

**User's choice:** Abort that seed + bump reg
**Notes:** User prioritizes fighting overfit over compute economy, consistent with the "without overfitting" goal.

### Reg-bump ensemble-homogeneity policy

| Option | Description | Selected |
|--------|-------------|----------|
| Seq, lock on seed 42 (rec.) | Run 42 first; trip → bump+restart until clean; lock recipe for 1337/7 | ✓ |
| Re-run all 3 at stronger reg | Any trip → bump all, re-run all (wastes clean seeds) | |
| Bump only tripped seed | Heterogeneous ensemble; weakest variance story | |

**User's choice:** Seq, lock on seed 42 (rec.)
**Notes:** Fine-tunes run sequentially on one L4 anyway, so this keeps the ensemble homogeneous in the common case at zero wasted compute. Later-seed trip at the locked recipe documented as genuine seed variance.

### No-lift policy (ensemble < Phase 3 baseline)

| Option | Description | Selected |
|--------|-------------|----------|
| One remediation pass (rec.) | One documented fix (reg sweep / SSL-epoch adjust); else close honestly | ✓ |
| Document immediately | Close with honest deviation analysis, no remediation | |
| Iterate until lift | Open-ended; could consume the budget | |

**User's choice:** One remediation pass (rec.)
**Notes:** Bounds the schedule hit; honest deviation analysis if remediation also fails to lift. Preserves Phases 5-8.

## Claude's Discretion

- **Ensemble aggregation method** — user explicitly delegated ("you know better"); Claude chose mean of sigmoids with full rationale.
- **MD-SSL recipe mechanics (CONTEXT D2 open questions 1-8)** — delegated to research-phase by design (Code_Release/motion_disentanglement is empty; reconstruct from paper §4 + SSL literature).

## Deferred Ideas

- Auxiliary trajectory head (F) — dropped this phase.
- Warmup ablation — only if AdamW+SSL-init instability appears.
- Fine-tune LR ablation — researcher checks paper §5; 1e-4 default.
- Regularization sweep (light/moderate/aggressive on seed 42) — the designated remediation pass.
- 5-seed extension — only if 3-seed variance too wide + compute permits.
- TorchCodec decoder migration — only if decode is the SSL throughput bottleneck.
