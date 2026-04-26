# Phase 1A — Translation Benchmark Results

> **Status:** complete
> **Date:** 2026-04-25
> **Branch:** main (extends v2026.5 release plan)
> **Author:** auto-mode benchmark run + writeup

---

## TL;DR

- **Gate 1A: ✅ PASS by 8.9× margin.** llama.cpp-served Gemma 4 E2B Q4_K_M (T2) hits **66.15 tok/s** on sermon translation vs HF E2B NF4's 7.44 tok/s — ~9× speedup, no canary regression (6/8 == 6/8), 4× less VRAM (3.5 GB vs 14.2 GB).
- **Spec decode (T4) underperforms target alone (T3) on this hardware.** E4B+E2B speculative decoding is 36 tok/s vs E4B alone at 41 tok/s, with 2× the VRAM and same canary score. **Recommendation: ship llama.cpp engine in plain (non-spec) mode for now.** Acceptance rate was 64% — not enough to overcome single-GPU draft/target context-switch overhead.
- **E4B is the quality winner** (7/8 canary across both HF and GGUF runtimes). It's the only model in the matrix that disambiguates *Jacobo* (apostle James) from *Santiago* (the epistle). 1 GB more VRAM than E2B; 50% more latency.
- **TG12B (HF) is the most expensive config** at 15.6 GB VRAM and 6.15 tok/s — and it doesn't beat E4B on quality (6/8 vs 7/8). PR #46's decision to drop TranslateGemma was correct; this benchmark confirms it on production-domain data.
- **Production fit on 12 GB RTX 3060:** all HF NF4 configs are too big; E4B HF / TG12B HF need the 16 GB card (or a smaller one with offload). T2 and T3 (llama.cpp GGUF) fit comfortably alongside Whisper STT. **llama.cpp is required for production deployment, not just preferred.**
- **Two universal-miss canary terms** (no model gets them): *partimiento del pan* (all 7 say *partir/romper del pan*). *Jacobo* is also missed by 4/7 — only E4B family handles it.

---

## Hardware

| Field | Value |
|---|---|
| GPU | NVIDIA RTX 2000 Ada Generation (A2000 Ada) |
| VRAM total | 16,379 MiB (16 GB) |
| Compute capability | 8.9 |
| Host | WSL2 on Windows 11, Ubuntu 24.04 |
| llama.cpp | b8782 (commit `e97492369`), CUDA build |
| PyTorch | 2.9.1+cu126 |
| transformers | 5.5.3 |
| bitsandbytes | NF4 default (double quant, bf16 compute) |
| Python venv | `/home/wbell/stt_train_env/` |

> **Hardware note:** the v2026.5 plan specified RTX 3060 12 GB (deployment target). Current run is on A2000 Ada 16 GB, same compute capability. tok/s numbers should transfer cleanly. **VRAM headroom on the 12 GB card will fail for all HF NF4 configs except TG4B**; the GGUF configs all fit.

## Test set

| Tier | N | Source | Metric |
|---|---|---|---|
| Sermon | 125 | first 125 of `bible_data/synthetic/deepl_sermon_benchmark_500.jsonl` (DeepL synthetic Spanish refs) | tok/s, latency p50/p95, hallucination ratio (output_words / input_words) |
| Canary | 8 | `training/benchmark_gemma4.CANARY_SENTENCES` | substring match for expected Spanish term, target ≥6/8 for Gate 1A |

> **Why no Bible-verse tier:** verses test formal-register translation against human-quality references — useful for publication-grade quality, not for a Gate 1A latency-and-regression check. Sermon + canary covers the live-demo domain better. (See feedback memo `feedback_eval_design.md`.)

## Configurations

| Config | Engine | Target | Draft | Quant | Notes |
|---|---|---|---|---|---|
| `tg4b_hf` | HF + bnb | `google/translategemma-4b-it` | — | NF4 | Pre-#46 default |
| `t1` (E2B HF) | HF + bnb | `google/gemma-4-e2b-it` | — | NF4 | Post-#46 default; baseline for Gate 1A |
| `tg12b_hf` | HF + bnb | `google/translategemma-12b-it` | — | NF4 | Quality reference |
| `e4b_hf` | HF + bnb | `google/gemma-4-e4b-it` | — | NF4 | Best HF quality |
| `t2` | llama.cpp | E2B GGUF | — | Q4_K_M | Drop-in replacement for `t1` |
| `t3` | llama.cpp | E4B GGUF | — | Q4_K_M | Drop-in replacement for `e4b_hf` |
| `t4` | llama.cpp | E4B GGUF | E2B GGUF | Q4_K_M (both) | Spec decode, `--draft 16 --draft-min 5` |

llama-server flags: `-ngl 999 -c 512 -ctk q8_0 --host 127.0.0.1 --port 8090`. T4 adds `-md <draft.gguf> --draft 16 --draft-min 5`.

---

## Results — full matrix (sorted by sermon tok/s)

| config | tok/s | latency p50 | latency p95 | VRAM peak | model resident | Canary | Halluc. | compl. tok | speedup vs T1 |
|---|---|---|---|---|---|---|---|---|---|
| **t2** | **66.15** | **281 ms** | 390 ms | **3.53 GB** | ~2.0 GB | 6/8 | 0.937 | 18 | **8.89×** |
| **t3** | **40.59** | **469 ms** | 640 ms | **4.94 GB** | ~3.5 GB | **7/8** | 0.951 | 19 | **5.46×** |
| t4 | 36.39 | 559 ms | 1023 ms | 8.52 GB | ~7.1 GB | 7/8 | 0.952 | 21 | 4.89× |
| tg4b_hf | 8.27 | 2381 ms | 3864 ms | 7.22 GB | 6.15 GB | 5/8 | 0.971 | 19 | 1.11× |
| t1 (E2B HF) | 7.44 | 2379 ms | 3781 ms | **14.23 GB** | 12.78 GB | 6/8 | 0.951 | 19 | 1.00× (baseline) |
| tg12b_hf | 6.15 | 3304 ms | 4763 ms | **15.59 GB** | 14.54 GB | 6/8 | 0.972 | 20 | 0.83× |
| e4b_hf | 5.83 | 3190 ms | 4507 ms | **15.58 GB** | 14.09 GB | 7/8 | 0.937 | 19 | 0.78× |

> **Reading the table:** `model resident` is `post_load_mib − baseline_mib` for HF runs; for llama.cpp runs the server was pre-loaded so the JSON shows 0. The values shown above are computed from observed server-up VRAM minus idle baseline (~1500 MiB).

### Per-canary hit/miss matrix

Models sorted by canary score (lowest → highest):

| Term (expected Spanish) | tg4b_hf | t1 (E2B HF) | tg12b_hf | t2 (E2B GGUF) | e4b_hf | t3 (E4B GGUF) | t4 (E4B+spec) |
|---|---|---|---|---|---|---|---|
| expiación (atonement) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| pacto (covenant) | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| gracia (grace) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| justicia (righteousness) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Santiago (epistle of James) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Jacobo (apostle James)** | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| santificación (sanctification) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **partimiento del pan (breaking of bread)** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Total** | **5/8** | **6/8** | **6/8** | **6/8** | **7/8** | **7/8** | **7/8** |

**Two findings:**
1. ***Jacobo* requires E4B.** The apostle/epistle disambiguation (Mark 3:17 vs the epistle) is only correctly translated by the E4B family. E2B, TG4B, and TG12B all default to *Santiago* even when the surrounding context names "James and John" as fishermen. This is a strong argument for shipping E4B as default.
2. ***Partimiento del pan* is a universal miss.** All 7 configs translate "breaking of bread" as *partir del pan*, *romper del pan*, etc. This Brethren-tradition idiom requires explicit fine-tuning. **Add to active learning data.** (See "Action items" below.)

### Per-model VRAM detail

| config | baseline | post-load | peak (inference) | model resident | inference overhead | load time |
|---|---|---|---|---|---|---|
| tg4b_hf | 1014 MiB | 7309 MiB | 7397 MiB | **6.15 GB** | 88 MiB | 89.7 s |
| t1 (E2B HF) | 1483 MiB | 14570 MiB | 14575 MiB | **12.78 GB** | 5 MiB | 77.8 s¹ |
| tg12b_hf | 1058 MiB | 15946 MiB | 15969 MiB | **14.54 GB** | 23 MiB | 232.6 s |
| e4b_hf | 1518 MiB | 15942 MiB | 15958 MiB | **14.09 GB** | 16 MiB | 172.5 s |
| t2 (E2B GGUF) | 1521 MiB² | 3558 MiB | 3623 MiB | ~2.0 GB | 65 MiB | server pre-load |
| t3 (E4B GGUF) | 1521 MiB² | 5042 MiB | 5056 MiB | ~3.5 GB | 14 MiB | server pre-load |
| t4 (E4B + E2B draft) | 1521 MiB² | 8649 MiB | 8721 MiB | ~7.1 GB | 72 MiB | server pre-load |

¹ T1 was loaded with HF cache warm (model was already on disk after smoke test). Cold-load time would be similar to E2B's first run (~115 s).
² For llama.cpp configs, "baseline" was sampled while the server was already running; resident is computed against system idle (~1500 MiB).

**Surprising: Gemma 4 E2B HF (12.78 GB) is 2× the resident VRAM of TG4B (6.15 GB)** despite both being 4B-class. The cause is Gemma 4's Per-Layer Embeddings (PLE) — large embedding tables that stay in bf16 even under NF4. This is also why the prior `metrics/gemma4_benchmark/comparison.json` was misleading: it used `torch.cuda.max_memory_allocated()` which only counts PyTorch tensor allocations and missed the bnb scratch + bf16 PLE. **GGUF Q4_K_M quantizes the embeddings too**, so E2B GGUF is ~2 GB resident — nearly 7× smaller than HF.

### Server-side timings (llama.cpp T2/T3/T4)

Parsed from `/tmp/llama_t{2,3,4}.log` via `parse_server_timings()`:

| config | prompt_eval_tps median | gen_eval_tps median | prompt_ms median | gen_ms median |
|---|---|---|---|---|
| t2 | 938 tok/s | 84 tok/s | 46 ms | 312 ms |
| t3 | (TBD — T3 server log gone) | | | |
| t4 | 736 tok/s | 47 tok/s | 63 ms | 530 ms |

Note: the server-side `gen_tps` differs from the client-side `tokens_per_second` because:
- Client measures wall-clock around HTTP, includes serialization + network overhead
- Server's "eval" excludes prompt processing time
- Gemma 4 SWA forces full prompt re-evaluation per request (no KV cache reuse), adding ~50 ms

### Spec decode (T4) detail

```
n_requests:                    135
mean_acceptance_per_request:   0.6445
weighted_acceptance:           0.5683 (152 / 268 across all rounds)
draft_max_per_round:           16
draft_min:                     5
```

**Why T4 is slower than T3 despite ~64% acceptance:** with 1 GPU, draft (E2B) and target (E4B) execute sequentially, not in parallel. For each round of D=16 draft tokens at α=0.65, E4B spends ~D × `gen_ms_per_token` to verify, but the savings from skipping individual E4B forwards is eaten by:
- E2B's draft pass (sequential, ~260 ms for 16 tokens)
- Bookkeeping + tree verification logic
- VRAM contention (8.5 GB peak vs 5 GB for T3)

The headline speedup math (assumes parallel draft+verify on multi-GPU): `1 / (1/D + (1-α)) ≈ 2.4×`. **Single-GPU reality: 0.9×**, i.e. small slowdown.

**Spec decode could still win on a multi-GPU host** (e.g. dual A2000s with the draft on one card, target on the other) where draft and verify overlap. Not relevant for the current single-card deployment target.

---

## Gate 1A verdict: ✅ PASS

> Criterion: best llama.cpp config tok/s ≥ 1.30× T1 baseline AND canary ≥ 6/8.

| config | speedup vs T1 | canary | gate |
|---|---|---|---|
| t2 | **8.89×** | 6/8 | ✅ PASS |
| t3 | **5.46×** | 7/8 | ✅ PASS |
| t4 | **4.89×** | 7/8 | ✅ PASS |

**Recommendation:** ship `LlamaCppEngine` in **non-spec-decode mode**. Use **T3 (E4B Q4_K_M alone)** as the production default — best quality (7/8 canary), 5× faster than HF baseline, fits in 5 GB VRAM. Keep T2 (E2B Q4_K_M) as the low-VRAM fallback (3.5 GB) for hardware-constrained deployments.

---

## Bugs found and fixed in this PR

### 1. `engines/llamacpp_engine.py` — Gemma 4 reasoning mode (production blocker)

**Symptom:** First T2 run produced **canary 0/8** with all `es_hyp` empty strings and `completion_tokens_median: 256` (max).

**Root cause:** Gemma 4's chat template defaults to chain-of-thought "reasoning" mode. llama-server parses responses into `content` (the answer) and `reasoning_content` (the meta-commentary). For non-trivial translation prompts the model spends all 256 tokens "thinking" (e.g., *"1. Analyze the English text... 2. Identify the tone... 3. Break down the sentence structure..."*) and the actual translation never makes it into `content`. Short, simple inputs (the first sermon chunk worked) skip reasoning; anything more complex triggers it.

**Fix:** add `chat_template_kwargs: {"enable_thinking": False}` to the request payload when `model_family == "gemma4"`. Verified: response goes from `finish_reason="length"` + 256 tokens + empty content to `finish_reason="stop"` + 18 tokens + clean Spanish. **14× latency improvement** for the affected canary alone.

**This was a real production bug**, not a benchmark artifact. Patched in `engines/llamacpp_engine.py:124-138` and `scripts/benchmarks/bench_translate_t1_t4.py`.

### 2. VRAM measurement — `torch.cuda.max_memory_allocated` undercounts on Gemma 4

**Symptom:** prior `metrics/gemma4_benchmark/comparison.json` reported E2B at 6.3 GB. Live nvidia-smi shows 14.5 GB. 2× discrepancy.

**Root cause:** `torch.cuda.max_memory_allocated()` only counts PyTorch tensor allocations. It misses:
- bitsandbytes' NF4 dequantization scratch buffers
- Gemma 4's Per-Layer Embeddings (PLE) staying in bf16 even under NF4 quantization
- PyTorch caching allocator's reserved-but-unallocated pool

**Fix:** continuous `nvidia-smi` sampler thread (poll `memory.used` every 0.5 s, track running max). Works for out-of-process llama-server too. Implementation in `scripts/benchmarks/bench_translate_t1_t4.py:VramSampler`.

**Implication:** all prior VRAM numbers in `metrics/gemma4_benchmark/comparison.json` should be considered lower bounds, not actual usage. **Update the file or annotate it.** (See action items.)

### 3. Hardware fit on RTX 3060 12 GB — design implication

The supposed "post-#46 default" T1 (E2B HF NF4) occupies 14.23 GB peak. On the 12 GB RTX 3060 deployment target, T1 doesn't fit even alone — let alone alongside Whisper STT (~0.9 GB) plus OS/driver overhead. This benchmark is the first time this has been measured correctly. **Conclusion: llama.cpp engine must ship for v2026.5 production deployment**, not just as a faster path.

---

## Notes & caveats

- **Gemma 4 sliding-window attention (SWA)** prevents llama.cpp from reusing KV cache across requests. Server log shows *"forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory)"* on every request, costing ~50 ms of prompt-eval per call. The HF path avoids this via `CUDAGemmaStreamingEngine`'s pre-computed prompt cache. Even with this disadvantage, llama.cpp still wins by 5–9×.
- **llama-server `--reasoning-budget` defaults to 2,147,483,647 tokens** (effectively unlimited) and engages reasoning mode for any model declaring it in chat-template metadata. Per-request `chat_template_kwargs` is the cleanest way to opt out. A server-level `--reasoning-format none` flag also works.
- T1's VRAM was patched post-hoc from manual nvidia-smi observations because T1 was started before the `VramSampler` was added to the script. T2/T3/T4/E4B-HF/TG4B-HF/TG12B-HF used the in-process sampler.
- Server-side timings for T3 are missing — its log was overwritten when T4 started. T3's client-side numbers are unaffected.
- Hallucination ratio = `output_words / input_words`. All configs sit in the 0.93–0.97 range — Spanish translations of English sermons are typically slightly shorter, so this is a healthy band. None of the configs exceeded the 1.5 hallucination flag threshold.

---

## Action items

| # | Item | Owner | Phase |
|---|---|---|---|
| 1 | Ship `LlamaCppEngine` in non-spec mode as default for v2026.5 CUDA path; T3 (E4B) preferred, T2 (E2B) as low-VRAM fallback | TBD | 1B (already merged in PR #48) |
| 2 | Wire `engines/factory.py` to default `engine_type="llamacpp"` on CUDA when `models/gemma-4-*-q4km.gguf` exists | TBD | 1D |
| 3 | Re-run `metrics/gemma4_benchmark/comparison.json` with `VramSampler` so its VRAM numbers are correct, OR annotate it as PyTorch-only | TBD | 1A follow-up |
| 4 | Add *partimiento del pan* and other Brethren-specific terms to active learning data + glossary Tier 1 | TBD | Phase 6 |
| 5 | Skip Phase 1C EAGLE-3 — spec decode is a single-GPU loss in this hardware bracket. Revisit only if/when multi-GPU deployment is on the roadmap | TBD | 1C |
| 6 | Update `engines/cuda_engine.py` to verify if HF Gemma 4 is also affected by reasoning mode (via `apply_chat_template`); patch if so | TBD | 1D |

---

## Reproducibility

```bash
source /home/wbell/stt_train_env/bin/activate

# T1 (HF E2B baseline) and other HF configs — no llama-server needed
python scripts/benchmarks/bench_translate_t1_t4.py --config t1 --n-sermon 125 --out metrics/phase1a_t1.json
python scripts/benchmarks/bench_translate_t1_t4.py --config tg4b_hf --n-sermon 125 --out metrics/phase1a_tg4b_hf.json
python scripts/benchmarks/bench_translate_t1_t4.py --config tg12b_hf --n-sermon 125 --out metrics/phase1a_tg12b_hf.json
python scripts/benchmarks/bench_translate_t1_t4.py --config e4b_hf --n-sermon 125 --out metrics/phase1a_e4b_hf.json

# T2 (E2B GGUF)
/home/wbell/llama.cpp/build/bin/llama-server \
  -m models/gemma-4-e2b-it-q4km.gguf \
  --host 127.0.0.1 --port 8090 -ngl 999 -c 512 -ctk q8_0 \
  > /tmp/llama_t2.log 2>&1 &
python scripts/benchmarks/bench_translate_t1_t4.py --config t2 --server-log /tmp/llama_t2.log \
  --n-sermon 125 --out metrics/phase1a_t2.json
pkill -f llama-server

# T3 (E4B GGUF)
/home/wbell/llama.cpp/build/bin/llama-server \
  -m models/gemma-4-e4b-it-q4km.gguf \
  --host 127.0.0.1 --port 8090 -ngl 999 -c 512 -ctk q8_0 \
  > /tmp/llama_t3.log 2>&1 &
python scripts/benchmarks/bench_translate_t1_t4.py --config t3 --server-log /tmp/llama_t3.log \
  --n-sermon 125 --out metrics/phase1a_t3.json
pkill -f llama-server

# T4 (E4B + E2B draft, spec decode)
/home/wbell/llama.cpp/build/bin/llama-server \
  -m models/gemma-4-e4b-it-q4km.gguf \
  -md models/gemma-4-e2b-it-q4km.gguf \
  --draft 16 --draft-min 5 \
  --host 127.0.0.1 --port 8090 -ngl 999 -c 512 -ctk q8_0 \
  > /tmp/llama_t4.log 2>&1 &
python scripts/benchmarks/bench_translate_t1_t4.py --config t4 --server-log /tmp/llama_t4.log \
  --n-sermon 125 --out metrics/phase1a_t4.json
pkill -f llama-server

# Final merge + table
python scripts/benchmarks/bench_translate_t1_t4.py --config merge \
  --inputs metrics/phase1a_t1.json metrics/phase1a_t2.json metrics/phase1a_t3.json \
           metrics/phase1a_t4.json metrics/phase1a_e4b_hf.json \
           metrics/phase1a_tg4b_hf.json metrics/phase1a_tg12b_hf.json \
  --out metrics/phase1a_benchmark.json
```

Run wall-clock: ~70 minutes total (HF chain dominates due to NF4 cold loads).
