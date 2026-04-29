# Phase C — Domain SFT (v1 ship target)

**Goal:** ship Gemma 4 E4B and E2B adapters that beat stock by **≥ +2 COMET-22** with **≥ 7/8 canary** and no regression.

**Wall clock:** ~1 week.

## C0. Prerequisite — rebuild the broken holdout

`bible_data/aligned/verse_pairs_test.jsonl` is a 2-line stub today (the S-sweep eval bug from 2026-03 was never actually fixed — only documented). Before any C2/C3 eval will produce trustworthy numbers, rebuild it:

```bash
python tools/build_eval_sets.py --verse-count 500 --seed 42
sha256sum bible_data/aligned/verse_pairs_test.jsonl
# expect sha256 e314def9a9383f3d6a953c7788705744caadb664471555c30f34c125f9bbc0b8 per eval_registry.json
```

If the sha doesn't match, `tools/build_eval_sets.py` may have changed since 2026-03-22 — investigate before continuing.

## C1. Reconstruct the training corpus

Critical changes from the S-series:

### Drop archaic Spanish from the primary mix

RVR1909 and Platense bias the model toward archaic register that's wrong for modern sermon speech. Keep them only as ≤ 10% replay buffer. The constants `MODERN_TRANSLATIONS` / `ARCHAIC_TRANSLATIONS` already exist at `training/train_gemma.py:77-78` — reuse them.

### Add a Gemini-2.5-Pro distillation pass

Top 5% hardest sermon chunks (those where existing DeepL pairs scored lowest CometKiwi-XL). ~1k chunks. Estimated cost: $50–150. GPT-4o is the alternative; both beat NLLB/MADLAD on colloquial sermon English by 3–5 COMET per recent benchmarks.

### Round-trip filter all synthetic pairs

Translate EN→ES with the candidate teacher, then ES→EN with a second teacher (DeepL ↔ Gemini), compute COMET on the round-trip. Keep only pairs where COMET ≥ 0.85.

### CometKiwi-XL filter all pairs

- General domain: threshold ≥ 0.80.
- Sermon domain: threshold ≥ 0.85.

Run via `training/qe_filter.py` (built in Phase A4).

### Final mixture (target ~15–25K pairs total)

| Component | Share | Source |
|---|---|---|
| Modern Bible verses (WEB / Español Sencillo / BBE pairs) | ~30% | `bible_data/aligned/verse_pairs_train.jsonl` filtered to modern only |
| Sermon (filtered, hybrid teachers) | ~30% | `bible_data/synthetic/*` after QE filter + new Gemini pass |
| Glossary-tagged pairs (`<g>EN||ES</g>` annotation) | ~30% | Either bible or sermon pairs, post-processed via `training/glossary_annotate.py` |
| OPUS-100 general-domain replay | ~10% | streaming, anti-forgetting anchor |

Research consensus: **diminishing returns past ~30K** for LLM MT fine-tuning; do not push past 25K in v1.

## C2. Train E4B v1

```bash
python training/train_gemma4.py \
    --base unsloth/gemma-4-E4B-it \
    --train-data bible_data/v1_corpus/mixed_25k.jsonl \
    --epochs 2 \
    --output fine_tuned_gemma4_e4b_v1
```

**Monitor:** loss curve must land in 1–3 by step ~500. If it stays above 5, something is wrong with the chat template or prompt format — stop and recheck `training/train_gemma4.py` against `engines/llamacpp_engine.py:115-138`.

**Estimated wall clock:** 6–10 hrs (similar to TranslateGemma 4B QLoRA, faster per step due to Unsloth + packing).

## C3. E4B eval gate

Run via `scripts/benchmarks/bench_translate_t1_t4.py`:

```bash
python training/export_gguf.py --adapter fine_tuned_gemma4_e4b_v1 --output models/gemma-4-e4b-it-q4km-v1.gguf
~/llama.cpp/build/bin/llama-server -m models/gemma-4-e4b-it-q4km-v1.gguf --port 8092 &

python scripts/benchmarks/bench_translate_t1_t4.py --config t3_v1 --server-url http://localhost:8092 --n-sermon 125 --out metrics/v1_e4b.json
python scripts/benchmarks/bench_translate_t1_t4.py --config merge --inputs metrics/phase1a_t3.json metrics/v1_e4b.json --out metrics/v1_e4b_diff.json
```

### Required (all four must pass)

- [ ] **≥ +2 COMET-22** vs. stock E4B on 500-verse holdout AND on 422-chunk sermon eval (use **xCOMET-XL or COMET-22**, NOT CometKiwi-XL — the latter was the training filter).
- [ ] **≥ 7/8 canary**, no regression below 7. Target 8/8.
- [ ] **Hallucination ratio** ≤ stock baseline.
- [ ] **Latency p50 ≤ baseline + 5%**, peak VRAM (nvidia-smi) ≤ baseline + 0.3 GB. QLoRA-merged Q4_K_M is the same size — should match exactly.

If any gate fails: re-examine data pipeline, do **not** patch by extending epochs.

## C4. Repeat for E2B

Same data, same script, separate run:

```bash
python training/train_gemma4.py \
    --base unsloth/gemma-4-E2B-it \
    --train-data bible_data/v1_corpus/mixed_25k.jsonl \
    --epochs 2 \
    --output fine_tuned_gemma4_e2b_v1
```

### E2B eval gate (relaxed — smaller model, less capacity)

- [ ] **≥ +1 COMET-22** vs. stock E2B.
- [ ] **≥ 6/8 canary** (stock E2B is 6/8; aim for 7/8 but accept maintain).
- [ ] **No latency / VRAM regression.**

## Gotchas / debugging hints

- **Loss stuck high:** prompt format mismatch. Diff `training/train_gemma4.py` against `engines/llamacpp_engine.py:115-138`.
- **Theological terms regress:** glossary annotation rate too low or `tier2_master.json` not loaded. Verify `glossary_annotate.py` output by sampling.
- **Modern register but Bible verses sound archaic:** the modern-translation filter at `training/train_gemma.py:77-78` may have included `asv` (which is in `ARCHAIC_TRANSLATIONS`). Recheck.
- **Sermon chunks worse than verses:** synthetic-data quality. Increase CometKiwi-XL threshold or add the Gemini distillation pass.
- **Hallucination ratio up:** packing may have crossed example boundaries without proper EOS. Verify packing config in Unsloth.
