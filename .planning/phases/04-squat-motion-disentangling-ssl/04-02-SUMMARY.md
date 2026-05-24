---
phase: 04-squat-motion-disentangling-ssl
plan: 02
status: complete
completed: 2026-05-22
---

# Phase 4 · Plan 02 (MD-SSL pretrain) — Summary

> Reconciled retroactively (2026-05-25): Plan 02 executed interactively on Colab over several
> sessions; this SUMMARY records the outcome that the interactive flow didn't capture at the time.

## What was planned

6 tasks under `04-02-PLAN.md`: the gated trajectory-format + half-cycle-sign probe (T1, blocking
`checkpoint:human-verify`), finalize `squat_ssl.py` `_load_trajectory`/`__getitem__`/`build_ssl_loader`
(T2), the SSL dataset smoke + triplet sanity grid (T3), finalize `md_pretrain.run_md_pretrain_epoch`
+ `_linear_probe` + collapse detection (T4), the VRAM + epoch-0 timing gate (T5, blocking
`checkpoint:decision`), and the full MD-SSL pretrain with linear-probe convergence monitoring (T6).
Produces `backbone.pt` for Plan 03.

## What was done

Executed interactively on a fresh L4 Colab, one cell at a time with paste-back
(`[[feedback_one_cell_at_a_time_strict]]`).

- **Probe (T1) resolved:** trajectories are per-clip JSON flat y-lists, 1:1 with video frames, in-file
  NaNs linearly interpolated, nested under subdirs (`rglob`, not `glob` — the `len(ds)==0` fix).
  Half-cycle sign = `bottom_is_argmax=False` (ARGMIN), confirmed on real clips (argmax degenerates the ascent).
- **`md_pretrain_v1` (safe-core 4 augs, 60-ep cosine ceiling):** SSL loss fell, but the linear-probe
  peaked at epoch 5 (macro 0.5708, KIE 0.380) then declined (ep10 0.566, ep15 0.551) with
  `effective_rank` collapsing 11.8→3.3 — a weak-augmentation contrastive collapse. A frozen-probe
  control confirmed SSL *worked*: ep5 macro 0.5708 vs pure-Kinetics 0.4297 = **+0.141** (KIE 0.169→0.380,
  mirroring the paper's KIE-concentrated MD lift) — it just over-trained past epoch 5.
- **`md_pretrain_v2` (strong augs — the pre-registered D2-augs iteration):** the paper (§3.2 + the Fig.5
  caption "in practice we apply much stronger augmentations") requires strong augs; v1 ran 4 mild ones.
  Wired the full set (added translation/zoom/blur/channel-swap; rotation gated OFF — KIE-angle risk),
  applied independently per branch; 20-ep cosine. A 5-epoch verification gate PASSED (ep5 probe 0.5844 >
  v1's 0.5708, KIE 0.4554, `eff_rank` 10.4 vs v1's 5.8 = collapse prevented). The full 20-ep run peaked
  again at ep5 (0.5844) then eroded (`eff_rank` →~5-6 by ep11-19): strong augs raised the peak + delayed
  collapse but didn't move the convergence point. **`backbone.pt` = v2-ep5** (linear-probe-best; the later
  collapse never touches the saved weights).

## Key verifications

- Collapse metric corrected mid-flight (commit `922ac0d`): the 8-sample `effective_rank` capped at ~7;
  fixed to a ~256-anchor probe set forwarded in chunks of 16 (diverse→199.5, collapsed→1.0).
- DataLoader teardown noise (`AssertionError: can only test a child process`) confirmed benign
  (GC teardown of short-lived worker pools; runs + checkpoints intact).
- Disconnect-safety proven repeatedly: multiple Colab disconnects mid-run, each resumed from the
  `latest.txt`-pointed Drive checkpoint with no lost epochs.

## Deviations from PLAN

- **Strong-aug v2 (headline deviation, pre-registered):** D2-augs said "if linear-probe convergence is
  weak, add translation/zoom." v1's safe-core collapsed; v2 strong augs are the paper-faithful fix. v1
  `backbone.pt` retained as the weak-aug ablation datapoint (for the Plan 04 linear-probe figure).
- **Epoch budget:** the paper's 20 was treated as a linear-probe-gated ceiling (≤60). Both v1 and v2
  converged by epoch ~5, so neither rode to 60 — our linear-probe stop rule (the paper has none) fired early.
- **`MDConfig` gained `strong_augs`/`use_rotation`/`aug_prob`**, threaded to the dataset + recorded in `config_repr`.

## Artifacts (committed on `fresh-start`)

- `datasets/ssl_augs.py` (+ `zoom`/`gaussian_blur`/`channel_swap`/`rotation`), `datasets/squat_ssl.py`
  (`_augment` strong/safe-core branches + finalized `__getitem__`/`_load_trajectory`), `harness/md_pretrain.py`
  (finalized `run_md_pretrain_epoch`/`_linear_probe`/`_embedding_collapse_metrics` + the aug config),
  notebook `04_squat_md_ssl.{py,ipynb}` (Cell A, Steps 0-5).
- **On Drive:** `phase04/md_pretrain_v1/backbone.pt` (weak-aug ep5, ablation) + `md_pretrain_v2/backbone.pt`
  (strong-aug ep5 — **the SSL deliverable into Plan 03**).
- Key commits: `922ac0d` (collapse-metric fix), `e5b5050` (strong-aug v2).

## Next

Plan 03 — 3-seed fine-tune from `md_pretrain_v2/backbone.pt`.
