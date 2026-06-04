# Vendored upstream — provenance, license, and patches

**Source:** [github.com/majumderb/recipe-personalization](https://github.com/majumderb/recipe-personalization) at commit `7ba0624` (2019-12-03).
**License:** GNU GPLv3 (per the per-file headers preserved verbatim).
**Original authors:** Bodhisattwa Prasad Majumder, Shuyang Li, Jianmo Ni, Julian McAuley — UCSD.
**Paper:** *Generating Personalized Recipes from Historical User Preferences*, EMNLP 2019.

## Why vendored

FitNova reproduces the paper on the **official Food.com splits** with **modern PyTorch**. Upstream is unmaintained and targets PyTorch 1.1 / Python 3.6 (no pinned `requirements.txt`). Vendoring keeps the reproduction:

- **Defensible** — we run their actual published code, not a from-scratch rewrite.
- **Self-contained** — Colab notebook pulls one branch, no second-repo clone.
- **Patchable** — modernization fixes are clearly diff-ed against upstream.

## Modifications (FitNova, 2026-05-29)

All patches are flagged with `# FitNova patch:` inline comments. Diff vs. upstream is intentionally minimal — *modernize, don't rewrite*.

- `language.py` — `pytorch_pretrained_bert.OpenAIGPTTokenizer` → modern `transformers.OpenAIGPTTokenizer`. Tokenizer load wrapped in `try/except` and `GPT_VOCAB_SIZE` hardcoded to **40,478** (documented OpenAI GPT-1 BPE vocab size) so the module imports cleanly even without `transformers` installed. `TECHNIQUE_TOKENS` guarded the same way.
- `pipeline/eval.py` — eval/preprocessing deps (`tqdm`, `nltk.bleu`, `pyrouge`, `spacy`) made lazy/optional via `try/except`. The decode helpers (`top_k_logits`, `top_p_logits`, `sample_next_token`) now import without those deps; Colab installs them before eval runs.

**CPU smoke test (2026-05-29):** `baseline.create_model(...)` + a `(B=2, T=8)` forward pass on torch 2.12 returns `(2, 8, 40483)` log-probs, all finite, row-sum 1.0000, NLL ≈ 10.59 ≈ `ln(40483)` — the expected entropy of an untrained model on uniform targets. The vendored package + patches genuinely run on modern PyTorch before any Colab heavy run.

## Out of scope (faithfully reported, not reimplemented)

- BERT-based step-coherence + entailment scorers (Sections 5 of the paper) — separate model artefacts, not core to PPL/BLEU/ROUGE/Distinct/UMA/MRR.
- Human pairwise evaluation (Table 2 PP column) — requires raters.
