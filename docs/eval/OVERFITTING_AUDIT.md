# Overfitting Audit — FitNova Form-Correction Models

**Audited:** every trained model in the form-correction milestone — Squat (video, R(2+1)D-18),
OHP (video, R(2+1)D-18), Shallow-Squat (image, ResNet-18), each with a **supervised baseline** and
an **SSL fine-tune**, every seed, every error head. Official Fitness-AQA train/val/test splits;
F1 per error (identical metric/splits to Parmar et al., ECCV 2022).

**Independent, evidence-driven.** Nothing here trusts a cached headline. Every F1, threshold,
confusion matrix and gap is **re-derived** — from the committed per-clip scores with the same
`backend/training/aqa/eval/metrics.py` that produced the originals, and (the load-bearing pass)
from the **actual `best.pt` weights on Drive**, reloaded and re-run through the official eval.

Reproduce: `docs/audit/overfitting_audit_local.py` (committed-data pass, no Drive),
`docs/audit/reeval_shallow.py` + `docs/audit/reeval_video.py` (local re-eval from weights),
`docs/audit/overfitting_audit_colab.py` (GPU twin). Figures: `docs/figures/audit_*.png`.

---

## Bottom line

**No deployed model shows validation-to-test (deployment) overfitting.** For every deployed
(multi-seed SSL ensemble) error head, held-out **test** matches **validation** within sampling
noise — at the tuned-threshold F1, the threshold-free PR-AUC, and AUC-ROC. `best.pt` is taken at or
before any validation degradation, per-seed variance is small, and the SSL pretraining did not
collapse. Re-evaluating from the actual weights reproduces the committed numbers exactly, so the
headline metrics are genuinely the models' outputs, not a stale or hand-edited cache.

The audit also *detects* overfitting where it exists: the OHP supervised **baseline** (a control,
not deployed) over-fits the hard Elbows class (val→test +0.098, beyond sampling noise), and the
deployed MD-SSL model closes that gap to +0.031. The same method flagging the baseline but clearing
the deployed models is evidence the clean calls are real, not a rubber stamp.

**The honest limitation — surfaced by the train-to-test view (Pillar 7).** The two rarest, hardest
classes show large *train*-to-test gaps. Squat Knees-Inward, a **deployed** head, has train F1 0.93
vs test 0.42 (gap **+0.51**); the unregularized baselines are worse (OHP-baseline Elbows +0.53). The
model fits training examples of these rare errors far better than unseen ones — **train-memorization**.
This caps their absolute test F1 (~0.42–0.45, the published paper's level) but does **not** break
deployment generalization: model selection is on validation, validation tracks test, and the test F1
on these classes equals the published state of the art. SSL measurably shrinks the gap on OHP Elbows
(baseline +0.53 → deployed +0.13); on Squat-KIE it raises test F1 without closing the gap. The
easy/balanced classes (Squat KFE, OHP Knees, Shallow depth) have small train-to-test gaps (≤0.13).

This audit makes the rigorous, supportable claim — *the deployed operating point generalizes to
held-out test; the weights reproduce the cache; the rare classes sit at the published difficulty
ceiling with honest train-memorization* — not a literal "0% overfitting," which no finite test set
can prove.

---

## Scope & method

| Exercise | Modality | Baseline | SSL fine-tune (deployed) | Errors |
|---|---|---|---|---|
| Squat | video R(2+1)D-18 | `r2plus1d18_squat_supervised_v1` (1 seed) | MD-SSL ensemble, seeds 42/1337/7 | KIE, KFE |
| OHP | video R(2+1)D-18 | `ohp_supervised_v1` (1 seed) | MD-SSL ensemble, seeds 42/1337 | Elbows, Knees |
| Shallow-Squat | image ResNet-18 | 3-seed ensemble (42/1337/7) | CVCSPC ensemble, seeds 42/1337/7 | depth |

**Seven evidence pillars** (each model passes all that apply):
1. **Cache integrity** — the stored headline F1 reproduce from the stored per-clip scores.
2. **Re-eval from the weights** — reload every `best.pt`, re-run official eval, match the cache.
3. **Learning curves** — `best.pt` taken at/before any val degradation (early-stop discipline).
4. **Held-out val→test gap** — small / within noise, at F1 *and* threshold-free PR-AUC.
5. **Confusion + score-distribution consistency** — val and test fail the same way.
6. **Per-seed variance + SSL health** — stable across seeds; no representation collapse.
7. **Train→test gap** — train vs val/test F1; large only on the rare classes (train-memorization).

**Why two generalization metrics.** The operating threshold is tuned on validation, so val F1 is
the threshold-*maximised* value and is upward-biased by selection — a modest val>test F1 gap is
expected from threshold selection alone, independent of overfitting. We therefore also report the
**threshold-free PR-AUC** gap, which isolates the model's ranking generalization from the threshold
choice, with **bootstrap CIs on both splits** (rare classes carry tens-of-positives noise). The
overfitting test is **directional**: only val *exceeding* test beyond ~0.03 *and* beyond the test
bootstrap CI counts; test ≥ val is never overfitting.

---

## Pillar 1 — Cache integrity (results.pkl reproduces from per-clip scores)

Recomputing F1 at the stored threshold from the stored per-clip ensemble scores, and re-running
`threshold_sweep` on the val scores, reproduces **every** stored headline number exactly
(|Δ| < 1e-6). A hand-edited or stale headline would disagree here.

| Model · error | stored test F1 | re-derived from scores | stored threshold | re-derived sweep |
|---|---|---|---|---|
| Squat MD-SSL · KIE | 0.41975 | 0.41975 ✓ | 0.61449 | 0.61449 ✓ |
| Squat MD-SSL · KFE | 0.84103 | 0.84103 ✓ | 0.38465 | 0.38465 ✓ |
| OHP MD-SSL · Elbows | 0.44737 | 0.44737 ✓ | 0.35736 | 0.35736 ✓ |
| OHP MD-SSL · Knees | 0.87705 | 0.87705 ✓ | 0.47648 | 0.47648 ✓ |
| Shallow CVCSPC · depth | 0.89022 | 0.89022 ✓ | 0.39537 | 0.39537 ✓ |
| Shallow baseline · depth | 0.87500 | 0.87500 ✓ | 0.47338 | — |
| Squat baseline · KIE/KFE | 0.28571 / 0.80000 | ✓ / ✓ | 0.36308 / 0.57124 | — |

**Verdict: cache internally consistent — the headline F1 are derived from the per-clip scores, not
asserted.**

---

## Pillar 2 — Re-eval from the actual weights (the load-bearing certainty pass)

The committed scores reproduce the headline (Pillar 1) — but do the **scores themselves** come from
the trained weights? We reload every `best.pt` and re-run the official val+test eval through the
exact committed dataset + model code.

### Shallow-Squat (image) — `reeval_shallow.py`, local CPU, **12/12 checks PASS**

Every number re-derived from the 6 reloaded `best.pt` matches the committed `results.pkl` to
**Δ = 0.0000**, and the per-clip ensemble scores reproduce the cache to float noise:

| check | re-eval | results.pkl | Δ |
|---|---|---|---|
| CVCSPC ensemble test F1 | 0.8902 | 0.8902 | 0.0000 |
| CVCSPC ensemble val F1 | 0.9119 | 0.9119 | 0.0000 |
| CVCSPC val-tuned threshold | 0.3954 | 0.3954 | 0.0000 |
| CVCSPC test PR-AUC | 0.9668 | 0.9668 | 0.0000 |
| CVCSPC per-seed test F1 (42/1337/7) | 0.8961 / 0.8902 / 0.8825 | identical | 0.0000 |
| baseline ensemble test F1 | 0.8750 | 0.8750 | 0.0000 |
| baseline ensemble val F1 | 0.8942 | 0.8942 | 0.0000 |
| test confusion `[[262,42],[13,223]]` | exact match | exact match | — |

Per-clip ensemble-score `max|Δ|` vs cache: **CVCSPC test 7.6e-4, val 1.0e-3, baseline test 6.4e-4**
(CPU-vs-GPU float noise; no clip sits within that of the threshold, so F1/confusion are exact).

**→ The committed Shallow-Squat cache is faithfully the weights' output. No fabrication, no staleness.**

### Squat + OHP (video) — `reeval_video.py`, local CPU — **all deployed-model checks PASS, Δ = 0.0000**

Reloading every video `best.pt` and re-running the official eval reproduces the committed
`results.pkl` **exactly** on F1, with per-clip ensemble scores matching the cache to float noise
(local tv 0.27 and the Colab tv 0.26 that produced the cache both decode via the same cv2 fallback —
tight parity):

| model · error | re-eval test F1 | results.pkl | Δ F1 | re-eval threshold | Δ thr | per-clip score max\|Δ\| |
|---|---|---|---|---|---|---|
| Squat MD-SSL · KIE | 0.4198 | 0.4198 | 0.0000 | 0.6145 | 0.0000 | test 4.8e-4 / val 5.9e-4 |
| Squat MD-SSL · KFE | 0.8410 | 0.8410 | 0.0000 | 0.3846 | −0.0000 | (ensemble, both heads) |
| Squat baseline · KIE | 0.2857 | 0.2857 | 0.0000 | — | — | test 1.1e-3 |
| Squat baseline · KFE | 0.8000 | 0.8000 | 0.0000 | — | — | |
| OHP MD-SSL · Elbows | 0.4474 | 0.4474 | 0.0000 | 0.3574 | +0.0001 | test 6.3e-4 / val 6.2e-4 |
| OHP MD-SSL · Knees | 0.8770 | 0.8770 | 0.0000 | 0.4766 | +0.0001 | (ensemble, both heads) |

**→ The committed video caches are faithfully the weights' output too** — not stale, not fabricated
(12/12 video checks within tolerance, all deployed F1 at Δ0).

The OHP supervised baseline (a control — the committed pkl stored only its summary F1) was re-eval'd
from its weights to recover its val→test gap, and it is the one place overfitting **is** present:
**Elbows val_F1 0.5146 → test_F1 0.4167 (gap +0.098, beyond the test bootstrap CI)** — the
unregularized single-seed control over-fits the hard upper-body class (Knees is fine, −0.009). The
**deployed** OHP MD-SSL closes that Elbows gap to **+0.031** (PR-AUC gap −0.001) — the measured
effect of MD-SSL pretraining + weight decay + head dropout + the 2-seed ensemble + early-stop. That
the same audit flags the baseline but clears the deployed model is direct evidence the deployed
"no-overfit" calls are a real signal, not a blanket pass.

> **Decode-parity note.** torchvision ≥ 0.26 removed `read_video`; both the local machine (tv 0.27)
> and the Colab env that produced the video `results.pkl` (tv 0.26) decode via the **cv2** fallback,
> so re-eval matches the cache closely. Any residual per-clip difference is cv2-version/CPU-GPU float
> noise; the threshold-free PR-AUC (ranking) is invariant to it, so the generalization verdict does
> not depend on exact-bit decode reproduction.

---

## Pillar 3 — Learning curves: `best.pt` is taken before any validation degradation

Overfitting within a run looks like `train_loss ↓` while `val_loss ↑` *past* the best epoch, with
the checkpoint taken after the rise. For every run, `best.pt` is the argmax-val checkpoint and the
early-stop (patience 8 on val-F1) terminates the run — the **deployed weights predate the
divergence**. `Δval_loss(best→final)` below is the post-best rise that early-stopping discards.

| run | best epoch | final epoch | Δval_loss (best→final) | Δtrain_loss |
|---|---|---|---|---|
| squat / baseline / s42 | 3 | 11 | **+1.003** | −0.182 |
| squat / MD-SSL / s42 | 8 | 16 | +0.187 | −0.195 |
| squat / MD-SSL / s1337 | 2 | 10 | +0.510 | −0.425 |
| squat / MD-SSL / s7 | 8 | 16 | +0.681 | −0.211 |
| ohp / MD-SSL / s42 | 6 | 14 | +0.642 | −0.373 |
| ohp / MD-SSL / s1337 | 8 | 16 | +0.536 | −0.293 |
| shallow / baseline / s42 | 10 | 18 | +0.081 | −0.106 |
| shallow / baseline / s1337 | 12 | 20 | +0.114 | −0.025 |
| shallow / baseline / s7 | 15 | 23 | +0.092 | −0.036 |
| shallow / CVCSPC / s42 | 12 | 20 | **−0.007** | −0.013 |
| shallow / CVCSPC / s1337 | 13 | 21 | +0.079 | −0.018 |
| shallow / CVCSPC / s7 | 5 | 13 | +0.054 | −0.047 |

Reading: the models *would* over-fit if trained past the best epoch (val loss rises while train loss
keeps falling — clearest in the Squat baseline, +1.00) — and that is exactly why the early-stop +
best.pt contract exists. The **deployed** checkpoint is always the pre-divergence one. (BCE val loss
and val F1 partially decouple under class imbalance, so the held-out F1 gap in Pillar 4 — not the val
loss alone — is the deployment-relevant test.) Figures: `audit_squat_learning_curves.png`,
`audit_ohp_learning_curves.png`, `audit_shallow_learning_curves.png`.

---

## Pillar 4 — Held-out val→test gap (the decisive generalization check)

Per error, at the val-tuned threshold, with bootstrap CIs and the threshold-free PR-AUC gap.
**Every audited error generalizes** (directional, noise-aware criterion). Figure:
`audit_val_test_gap_summary.png`.

| model · error | val F1 | test F1 | F1 gap | val PR-AUC | test PR-AUC | PR-AUC gap | test PR-AUC 95% CI | verdict |
|---|---|---|---|---|---|---|---|---|
| Squat MD-SSL · KIE | 0.5000 | 0.4198 | +0.0802 | 0.5195 | 0.3875 | +0.1320 | [0.267, 0.557] | **generalizes** (within noise) |
| Squat MD-SSL · KFE | 0.8128 | 0.8410 | −0.0282 | 0.8343 | 0.8812 | −0.0469 | [0.830, 0.924] | generalizes |
| Squat baseline · KIE | 0.3040 | 0.2857 | +0.0183 | 0.1938 | 0.2645 | −0.0707 | [0.151, 0.404] | generalizes (test ≥ val) |
| Squat baseline · KFE | 0.7979 | 0.8000 | −0.0021 | 0.7019 | 0.7640 | −0.0621 | [0.695, 0.838] | generalizes (test ≥ val) |
| OHP MD-SSL · Elbows | 0.4780 | 0.4474 | +0.0306 | 0.3682 | 0.3694 | −0.0012 | [0.288, 0.472] | generalizes |
| OHP MD-SSL · Knees | 0.8683 | 0.8770 | −0.0088 | 0.9387 | 0.9372 | +0.0015 | [0.906, 0.962] | generalizes |
| Shallow CVCSPC · depth | 0.9119 | 0.8902 | +0.0217 | 0.9620 | 0.9668 | −0.0048 | [0.949, 0.980] | generalizes |
| Shallow baseline · depth | 0.8942 | 0.8750 | +0.0192 | — | 0.9506 | — | [0.930, 0.968] | generalizes |

**Reading.** Five of eight rows have a near-zero or *negative* gap (test ≥ val) at both F1 and
PR-AUC — unambiguous. OHP Elbows (+0.031 F1) is the hard upper-body class; its **PR-AUC gap is
−0.001** (ranking identical val↔test), so the small F1 gap is threshold-selection, not overfitting.

**The honest KIE caveat.** Squat MD-SSL KIE shows the largest gaps (F1 +0.080, PR-AUC +0.132). It is
the rarest class — **36 test positives** — where both estimates carry ±0.13 sampling noise. The
val PR-AUC (0.520) sits **inside** the test PR-AUC 95% bootstrap CI [0.267, 0.557], so the gap is
not statistically distinguishable from noise. The threshold-free **ROC-AUC**, which is far more
stable than PR-AUC on a 36-positive class, puts KIE at validation 0.790 versus test 0.784
(gap +0.006): the model ranks KIE positives essentially as well on test as on validation, so the
large PR-AUC point-gap reflects PR-AUC's sensitivity on few positives, not a ranking failure.
Corroboration that this is intrinsic difficulty, not
overfit degradation: our **test KIE F1 (0.420) equals the published Parmar MD-SSL KIE F1 (0.419)** —
we match the paper on the same hard class on the same split. The deployed mitigations (3-seed
ensemble, weight decay, head dropout, early-stop) are exactly what holds KIE at the ceiling. This is
the single weakest-supported row in the audit and is flagged as such; it is *within noise*, not an
overfit.

---

## Pillar 5 — Confusion + score-distribution consistency across splits

A generalizing model fails the same way on val and test. Confusion matrices at the production
threshold and positive/negative score histograms are structurally similar across the two splits for
every error — including the honest overlap on the rare/hard classes (KIE, Elbows), which overlaps
*the same way* on val and test. Figures: `audit_squat_ohp_confusion.png`, `audit_shallow_confusion.png`,
`audit_squat_ohp_score_dist.png`, `audit_shallow_score_dist.png`.

---

## Pillar 6 — Per-seed variance + SSL health

**Per-seed variance** (low ⇒ not a lucky-seed artifact). Figure: `audit_per_seed_variance.png`.

- Squat MD-SSL test macro-F1: seeds 0.6115 / 0.6121 / 0.5691 (std ≈ 0.020; seed 7 the soft outlier),
  ensemble **0.6304** — the ensemble exceeds every single seed.
- OHP MD-SSL test F1: Elbows 0.4417 / 0.4158 (std ≈ 0.013), Knees 0.8594 / 0.8548 (std ≈ 0.002);
  ensemble Elbows 0.4474, Knees 0.8770.
- Shallow CVCSPC test F1: 0.8961 / 0.8902 / 0.8825 (std ≈ 0.006), ensemble 0.8902.

**SSL pretraining did not collapse.** Figures: `audit_ssl_health.png`, `audit_ssl_linear_probe.png`.

- Video MD-SSL backbones: `effective_rank` stays well above the collapse floor of 1 (Squat v2 and
  OHP both in the ~12–14 band), `embedding_std` stays > 0, and the frozen-backbone linear probe
  reaches macro-F1 ≈ 0.51–0.55 (well above chance) — the representation is diverse and linearly
  useful, the opposite of collapse.
- Image CVCSPC backbone: the triplet-accuracy monitor rises 0.75 → **0.959** over pretraining
  (chance ≈ 0.5) — the pose-contrastive task is learned, not collapsed.

---

## Pillar 7 — Train→test F1 view (train-memorization on the rare classes)

Each model is run on the official **train**, validation and test splits at its val-tuned threshold,
so the train→validation→test generalization is explicit. Train F1 above test is normal (the model
has seen the training data); what matters is the *size* of the gap and whether validation tracks
test. Figure: `audit_train_val_test_f1.png` (`train_eval.py`).

| model · error | train F1 | val F1 | test F1 | train−test | val−test | role |
|---|---|---|---|---|---|---|
| Squat MD-SSL · KIE | 0.933 | 0.500 | 0.420 | **+0.513** | +0.080 | deployed |
| Squat MD-SSL · KFE | 0.854 | 0.813 | 0.841 | +0.012 | −0.028 | deployed |
| OHP MD-SSL · Elbows | 0.576 | 0.478 | 0.447 | +0.129 | +0.031 | deployed |
| OHP MD-SSL · Knees | 0.984 | 0.868 | 0.877 | +0.107 | −0.009 | deployed |
| Shallow CVCSPC · depth | 0.971 | 0.912 | 0.890 | +0.081 | +0.022 | deployed |
| Squat baseline · KIE | 0.711 | 0.304 | 0.286 | +0.425 | +0.018 | control |
| Squat baseline · KFE | 0.928 | 0.798 | 0.800 | +0.128 | −0.002 | control |
| OHP baseline · Elbows | 0.942 | 0.515 | 0.417 | **+0.525** | +0.098 | control |
| OHP baseline · Knees | 0.991 | 0.798 | 0.807 | +0.184 | −0.009 | control |
| Shallow baseline · depth | 0.982 | 0.894 | 0.875 | +0.107 | +0.019 | control |

**Reading (honest).**
- **Deployment generalization holds everywhere:** the `val−test` column is small for every deployed
  error (≤ +0.08, mostly within noise). The operating point selected on validation transfers to test.
- **Rare classes show large train→test gaps:** Squat-KIE (deployed) **+0.51** and OHP-baseline-Elbows
  **+0.53** — the model fits training rare-class examples far better than unseen ones. This is
  train-memorization, the expected behaviour of a high-capacity model on a class with only a few
  hundred training positives; it caps absolute test F1 on those classes, which nonetheless equals the
  published paper.
- **SSL regularizes the gap where it can:** OHP Elbows train→test falls from **+0.53 (baseline) to
  +0.13 (deployed MD-SSL)**. On Squat-KIE the SSL lifts test F1 (0.29 → 0.42) but the train→test gap
  stays ≈ 0.5 — reported, not smoothed over.
- **Easy/balanced classes generalize tightly:** Squat-KFE +0.01, OHP-Knees +0.11, Shallow-depth +0.08.

The train→test gap on the rare classes is a real, honest limitation: those classes lean partly on
training-set memorization and reach only the task's published ceiling. It does not contradict the
no-deployment-overfitting verdict (which rests on val ≈ test), but it is the right thing to disclose.

---

## Per-model verdict

| model | val→test (F1) | train→test (per error) | re-eval | **verdict** |
|---|---|---|---|---|
| **Shallow CVCSPC (deployed)** | depth +0.022 | depth +0.08 | **yes (12/12, Δ0)** | **no val→test overfit; small train-test gap** |
| Shallow baseline | depth +0.019 | depth +0.11 | yes (Δ0) | no val→test overfit; small train-test gap |
| **Squat MD-SSL (deployed)** | KIE +0.080; KFE −0.028 | **KIE +0.51**; KFE +0.01 | **yes (Δ0)** | **no val→test overfit; KIE train-memorization** (val≈test, at paper ceiling) |
| Squat baseline | KIE +0.018; KFE −0.002 | KIE +0.43; KFE +0.13 | yes (Δ0) | no val→test overfit; KIE train-memorization |
| **OHP MD-SSL (deployed)** | Elbows +0.031; Knees −0.009 | Elbows +0.13; Knees +0.11 | **yes (Δ0)** | **no val→test overfit; Elbows train-test gap SSL-cut from +0.53** |
| OHP baseline (control) | **Elbows +0.098**; Knees −0.009 | **Elbows +0.53**; Knees +0.18 | yes (Δ0) | **OVERFITS Elbows** (val→test *and* train→test) |

---

## Honest caveats — what would change the verdict, and what this is *not*

- **KIE rare-class noise.** With 36 test positives, KIE F1/PR-AUC carry ±0.13 noise; the audit's
  KIE conclusion rests on CI overlap + the paper-ceiling match, not a tight gap. A larger labeled
  KIE test set is the only way to tighten it; nothing in the current data indicates val→test overfitting.
- **Rare-class train-memorization (Pillar 7).** Squat-KIE (deployed) and the Elbows/KIE baselines fit
  training rare-class examples far better than test (train→test up to +0.53). This is a real
  generalization gap on those classes, bounded by the deployment check (val ≈ test) and the
  paper-ceiling match; it means the rare-class accuracy relies partly on memorized training patterns
  and would benefit most from more labeled positives.
- **Video re-eval reproduced the headline exactly (Δ0); per-clip scores to ~5e-4.** Every video F1
  and threshold re-derived from the weights matched the cache to **Δ0.0000**; the per-clip ensemble
  scores matched to max|Δ| ≈ 5e-4–1e-3 (cv2-version / CPU-GPU float noise, below any threshold
  margin — hence the exact F1). Local tv 0.27 and the Colab tv 0.26 that built the cache share the
  cv2 decode fallback, so parity is tight; the threshold-free PR-AUC is invariant to the residual.
- **Domain shift is a separate boundary, not overfitting.** These models generalize **within the
  benchmark domain** (in-the-wild, side-on gym video). They do **not** transfer to home/front-facing
  phone clips — a documented in-the-wild limit. For this reason a live, record-yourself
  form-correction frontend is not deployed; instead the models are surfaced through a benchmark
  inspector that runs them on the dataset's own held-out test clips beside the expert ground truth.
  That distribution gap is a train/serve gap, distinct from train/test overfitting, which is what
  this audit certifies as sound.

---

## Figures

Committed-data pass (`docs/figures/audit_*.png`): `squat_learning_curves`, `ohp_learning_curves`,
`shallow_learning_curves`, `val_test_gap_summary`, `roc_curves`, `train_val_test_f1`,
`squat_ohp_confusion`, `shallow_confusion`, `squat_ohp_score_dist`, `shallow_score_dist`,
`per_seed_variance`, `ssl_health`, `ssl_linear_probe`, `headline_vs_paper`, `reeval_certainty`.
Live-from-weights (Colab twin): `audit_live_val_test_gap.png`,
`audit_live_confusion_scoredist.png`.

## Overall verdict

**No deployed FitNova form-correction model shows validation-to-test (deployment) overfitting.**
Held-out test matches validation within sampling noise at F1, PR-AUC and AUC-ROC; `best.pt` is taken
pre-divergence with val-F1 early-stop; per-seed variance is small; the SSL backbones did not collapse;
and re-evaluating from the actual weights reproduces the committed numbers **exactly** (Shallow and
video F1 both Δ0.0000; per-clip scores to ≤1e-3). The deployed SSL ensembles match or beat the
published Parmar et al. results on the identical metric and splits.

The honest limitation, disclosed in Pillar 7: the two rarest hard classes — Squat-KIE (deployed) and
the OHP/Squat baselines on Elbows/KIE — show large **train**-to-test gaps (up to +0.53), i.e.
train-memorization on classes with only a few hundred training positives. This caps their absolute
test F1 at the published ceiling but does not break deployment (val→test) generalization, and SSL
measurably reduces the gap where it can (OHP Elbows +0.53 → +0.13). The supportable claim is
therefore: *the deployed models generalize to held-out test, the weights confirm the cache, and the
rare classes sit at the field's difficulty ceiling with honest train-memorization* — not a blanket
"0% overfitting."
