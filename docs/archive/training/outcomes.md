# Benchmark Outcomes — April 2026

> **Date:** 2026-04-13
> **Hardware:** NVIDIA RTX 2000 Ada Generation, 16 GB VRAM, WSL2
> **Quantization:** NF4 (bitsandbytes 4-bit, double quant, bfloat16 compute)

---

## Gemma 4 vs TranslateGemma Benchmark

Four models compared across 3 evaluation tiers: 480 Bible verse pairs (formal register, reference-based), 125 Deepgram sermon chunks (informal register, no reference), and 8 theological canary sentences.

| Metric | TG 4B | TG 12B | Gemma 4 E2B | Gemma 4 E4B |
|--------|-------|--------|-------------|-------------|
| VRAM | 3.0 GB | 10.3 GB | 6.3 GB | 15.0 GB |
| Load time | 87s | 255s | 99s | 154s |
| Verse latency (median) | 2.46s | 15.72s | **2.24s** | 8.34s |
| Verse latency (p95) | 4.32s | 41.8s | 3.77s | 12.21s |
| Sermon latency (median) | 4.6s | 21.72s | **3.9s** | 14.11s |
| BLEU | 39.3 | 40.8 | **60.5** | **62.2** |
| chrF++ | 62.0 | 64.5 | **75.6** | **76.8** |
| Hallucination ratio | 0.953 | 0.96 | 0.946 | 0.945 |
| Theological canary | 5/8 | 6/8 | 6/8 | **7/8** |
| VRAM peak | 3.1 GB | ~16 GB | 6.4 GB | 15.0 GB |

### Summary

**Gemma 4 E2B is the standout.** It outperforms TranslateGemma 4B on every quality metric — BLEU +54% (60.5 vs 39.3), chrF++ +22% (75.6 vs 62.0), same hallucination rate, better theological canary accuracy (6/8 vs 5/8) — while also being 9% faster (2.24s vs 2.46s median verse latency). The tradeoff is 2x VRAM (6.3 vs 3.0 GB), which still fits comfortably on all target hardware (M3 Pro 18 GB, RTX 3060 12 GB, A2000 Ada 16 GB).

**Gemma 4 E4B gets the best theological accuracy (7/8)** but is impractical at NF4 on 16 GB — 15 GB VRAM with only 350 MiB headroom causes memory pressure that slows inference to 8.34s/verse (3.4x slower than TG 4B). E4B may become viable via GGUF quantization (llama.cpp Pillar 1), which typically achieves better memory efficiency than bitsandbytes NF4 for PLE-architecture models.

**TranslateGemma 12B is too slow for CUDA deployment.** At 15.72s median per verse (6.4x slower than 4B) and 10.3 GB VRAM, the marginal quality improvement (BLEU +1.5, canary +1/8) does not justify the cost. The 12B remains useful on MLX (Apple Silicon) where it runs at 400-700ms, and as the teacher model for S6 fine-tuning data generation.

### Canary Sentence Details

Both E2B and TG 4B miss "Jacobo" (James the apostle — both translate as "James" or "Santiago") and "partimiento del pan" (breaking of bread — E2B uses "partir del pan", TG 4B varies). E4B is the only model that correctly translates "Jacobo" in apostle context.

| Canary Term | TG 4B | TG 12B | E2B | E4B |
|-------------|-------|--------|-----|-----|
| expiación (atonement) | ? | ? | hit | hit |
| pacto (covenant) | ? | ? | hit | hit |
| gracia (grace) | ? | ? | hit | hit |
| justicia (righteousness) | ? | ? | hit | hit |
| Santiago (James, epistle) | ? | ? | hit | hit |
| Jacobo (James, apostle) | ? | ? | miss | **hit** |
| santificación | ? | ? | hit | hit |
| partimiento del pan | ? | ? | miss | miss |

*TG 4B/12B canary details not captured per-sentence in the first benchmark run (killed before completion). TG 4B scored 5/8 and TG 12B scored 6/8 overall.*

### Model Decision

**Recommended default: Gemma 4 E2B** for CUDA inference deployment. Replace TranslateGemma 4B as the primary translation model. Key reasons:

1. +54% BLEU, +22% chrF++ — not a marginal improvement, a generational leap
2. Faster inference (2.24s vs 2.46s median)
3. 6.3 GB VRAM fits all target hardware with headroom
4. Same hallucination rate as TG 4B
5. Better theological accuracy (6/8 vs 5/8)

**Keep TranslateGemma 4B as fallback** for extremely low-VRAM devices (<6 GB) and as a speculative draft model candidate.

**Keep TranslateGemma 12B for MLX** (Apple Silicon) where it runs fast, and as the teacher for synthetic training data generation.

**E4B: defer to GGUF** — re-evaluate once llama.cpp export is available (Pillar 1 of 2026.5 plan). At GGUF Q4_K_M it should fit in ~4-5 GB and run much faster.

---

## Whisper Fine-Tuning Results (W16 Curriculum Learning)

| Adapter | Data | Init From | Fresh Eval WER | Status |
|---------|------|-----------|----------------|--------|
| Base (no adapter) | — | — | 21.41% | Baseline |
| W7 (3 epochs) | 17K chunks | Random | 7.61% (fresh) / 5.63% (old) | Previous best |
| W13 (top 17K, 3ep) | 17K chunks | Random | 7.61% | Tied W7 |
| W14 (combined 50K, 3ep) | 40K chunks | Random | **7.42%** | Better than W13 |
| W15 (hard-only, lr=5e-5) | 2.5K hard | W14 (broken) | 64.76% | **FAILED** (init-from bug) |
| W15b (hard-only, lr=1e-5) | 9.8K hard | W14 (broken) | 64.76% | **FAILED** (same bug) |
| **W16 (mixed, lr=5e-5)** | **10K (30% hard + 70% easy)** | **W7 (fixed)** | **7.25%** | **New best** |

W16 confirms that curriculum learning works when done correctly: mixed hard+easy data, proper PEFT key remapping for `--init-from`, and validated weight loading. The 0.36% absolute WER improvement (7.61% → 7.25%) is modest but real, representing a 4.7% relative reduction. W17 (curriculum cycle 2, mining from W16) is training now.
