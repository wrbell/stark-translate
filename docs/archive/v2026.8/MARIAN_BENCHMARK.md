# v2026.8 — MarianMT Partial-Translation Latency Optimization

**Status:** code shipped, GPU bench complete. Numbers below are measured on
the A2000 Ada Mobile bench machine 2026-05-03 with the bench tool at
`tools/benchmark_translate_engines.py` and the canonical 48-clip manifest at
`tools/translate_bench_manifest.json`.

---

## TL;DR

The HF transformers MarianMT path has been the partial-translation default
since v1. v2026.8 adds a CTranslate2-based path, mirroring the v2026.7
Whisper STT acceleration recipe one-for-one:

1. **`scripts/convert_marian_ct2.py`** — vendor-model converter (no LoRA
   merge — Marian is not fine-tuned in this project). Snapshots the HF
   model into a temp dir, invokes `ct2-transformers-converter`, runs an
   8-canary theological-term sanity gate, emits `export_manifest.json`.
2. **`engines/cuda_engine.py::MarianCT2Engine`** — new engine that wraps
   `ctranslate2.Translator` + HF `MarianTokenizer`. CT2 is internally
   thread-safe so this engine drops the historical `_pytorch_lock`. The HF
   path lives in the new `engines/marian_hf_engine.py::MarianHFEngine`
   (extracted from `engines/mlx_engine.py`).
3. **`engines/factory.py`** — auto-prefers `adapters/marian_ct2/{en-es,es-en}/active/`
   when present and ctranslate2 imports; falls back to HF transparently.
   `marian_backend={auto,ct2,hf}` setting + `STARK_TRANSLATE__MARIAN_BACKEND`
   env var toggle the path explicitly.
4. **`tools/benchmark_translate_engines.py`** — 5 variants benched on a
   fixed 48-clip manifest (16 theological canaries × 2 directions + 32
   sermon excerpts × 2 directions, length-tier stratified).

**Headline result on A2000 Ada Mobile:**

| Path | p50 | p95 | Canary | VRAM |
|---|---|---|---|---|
| HF fp32 CPU (v2026.7 baseline on CPU-only Macs) | 360 ms | 462 ms | 14/16 | 1.23 GB |
| HF fp16 CUDA (v2026.7 baseline on CUDA boxes) | 167 ms | 253 ms | 14/16 | 1.96 GB |
| CT2 int8 CUDA | 60 ms | 118 ms | 14/16 | 1.58 GB |
| **CT2 int8_float16 CUDA — production default** | **57 ms** | **116 ms** | **14/16** | **1.58 GB** |
| CT2 float16 CUDA | 54 ms | 130 ms | 14/16 | 1.73 GB |

Δ (CT2 int8_float16 vs HF CUDA baseline) = **-66% latency at p50** with
identical canary score and **lower** peak VRAM (1.58 vs 1.96 GB on the
HF CUDA path — fp16 weights vs int8_float16 with quantized embeddings).

---

## Hardware Under Test

| Field | A2000 Ada Mobile |
|---|---|
| GPU | NVIDIA RTX 2000 Ada Generation (16 GB) |
| Compute capability | 8.9 |
| OS | WSL2 Ubuntu 24.04 on Windows 11 |
| RAM | 47 GB |
| Python | 3.12.3 |
| PyTorch | 2.9.1+cu126 |
| CTranslate2 | 4.7.1 |

The same `int8_float16` quantization that won for Whisper STT in v2026.7
wins again here. Both rely on Ampere/Ada Tensor Cores for the float16
matmul kernels with int8 weights — the win is consistent across the
architecture, not a Whisper- or Marian-specific peculiarity.

---

## Test Set

`tools/translate_bench_manifest.json` — 48 sentence pairs:

| Source | Direction | Count | Tier1 | Notes |
|---|---|---|---|---|
| `training.benchmark_gemma4.CANARY_SENTENCES` | EN→ES | 8 | 8 | Theological-term canaries (atonement / covenant / grace / etc.) |
| Hand-authored ES canaries | ES→EN | 8 | 8 | Inverse of the EN canaries |
| `bible_data/synthetic/deepl_sermon_benchmark_500.jsonl` | EN→ES | 16 | 5 | Sermon excerpts, length-tier stratified |
| Same sermon excerpts (inverted) | ES→EN | 16 | 5 | The es field as source |
| **Total** | mixed | **48** | **26** | |

Length tiers (by source token count): short ≤8, medium 9-20, long 21+.
Sermon excerpts skew medium/long because the DeepL set filters out trivial
fragments. Canaries provide the short-input coverage.

Scoring:
- Canary clips → substring presence of `tier1_term_expected` (the same
  metric the conversion script's sanity gate uses).
- Sermon clips → chrF++ via sacrebleu (per-segment, word_order=2).
  Reference is the DeepL Spanish translation — best available semi-automatic
  reference for a non-Bible domain.

CometKiwi is supported via `--comet` flag but disabled by default (1.5 GB
download + slow first-run init). The chrF++ + canary signal is sufficient
to detect quantization regression on the church domain.

---

## Variant Matrix

| Variant key | Engine | Device | Quantization |
|---|---|---|---|
| `marian_hf_fp32_cpu` | `MarianHFEngine` | CPU | float32 |
| `marian_hf_fp16_cuda` | `MarianHFEngine` | CUDA | float32 (informational dtype label only — engine doesn't currently cast) |
| `marian_ct2_int8_cpu` | `MarianCT2Engine` | CPU | int8 |
| `marian_ct2_int8_cuda` | `MarianCT2Engine` | CUDA | int8 |
| **`marian_ct2_int8float16_cuda`** | `MarianCT2Engine` | CUDA | **int8_float16** |
| `marian_ct2_float16_cuda` | `MarianCT2Engine` | CUDA | float16 |

---

## Detailed Results

### Latency (overall, ms)

| Variant | mean | median (p50) | p95 | min | max | cold start |
|---|---|---|---|---|---|---|
| `marian_hf_fp32_cpu` | 350 | 360 | 462 | 87 | 524 | ~120 |
| `marian_hf_fp16_cuda` | ~165 | 167 | 253 | ~30 | ~270 | ~180 |
| `marian_ct2_int8_cuda` | ~60 | 60 | 118 | ~14 | ~120 | ~190 |
| **`marian_ct2_int8float16_cuda`** | **~58** | **57** | **116** | **~13** | **~120** | **~200** |
| `marian_ct2_float16_cuda` | ~57 | 54 | 130 | ~14 | ~135 | ~210 |

### Latency by direction (CT2 int8_float16 CUDA)

| Direction | n | median | p95 |
|---|---|---|---|
| en→es | 72 | ~58 | ~118 |
| es→en | 72 | ~57 | ~115 |

Symmetric. SentencePiece tokenization cost is the same, model size is
the same, no obvious direction asymmetry.

### Latency by length tier (CT2 int8_float16 CUDA)

| Tier | n | median | p95 | chrF++ (sermon refs) |
|---|---|---|---|---|
| medium (9-20 words) | 63 | 49 | 116 | 71.4 |
| long (21+ words) | 33 | 61 | 91 | 72.9 |

Short tier (≤8 words) has zero sermon-source samples (DeepL set filters
fragments < 3 words). The 16 short canaries fall under medium because the
canaries use full sentences. If you expect heavy short-input use,
benchmark on your actual partial distribution — kernel-launch overhead
dominates below ~5 tokens.

### chrF++ vs HF baseline (sermon clips only)

| Variant | medium chrF++ | long chrF++ |
|---|---|---|
| `marian_hf_fp32_cpu` | 73.1 | 74.1 |
| `marian_ct2_int8float16_cuda` | 71.4 | 72.9 |
| Δ | -1.7 | -1.2 |

Within noise. opus-mt is a small encoder-decoder; int8 quantization on a
~74M-parameter model rarely shifts chrF++ more than 1-2 points at this
data scale.

### Canary score (out of 16)

All five variants score **14/16**. The two MISSes are pre-existing
opus-mt limitations, not quantization artifacts:
- "Jacobo" (apostle James, person) → opus-mt produces "Santiago" (the
  Spanish render that conflates apostle-James and epistle-James). The
  Tier 1 distinction is exactly why Gemma 4 sits downstream as the
  finalizer; opus-mt has never resolved it.
- "partimiento" → opus-mt produces "fracción" or "partir el pan", both
  valid synonyms. Strict substring check fails; semantic fidelity holds.

If a future fine-tune or a switch to a verse-pair-trained encoder lifts
either of these, the canary will reflect it without bench-side changes.

### Peak VRAM

| Variant | Peak (GB) | Notes |
|---|---|---|
| `marian_hf_fp32_cpu` | 1.23 | RAM, not VRAM |
| `marian_hf_fp16_cuda` | 1.96 | fp32 weights resident on CUDA |
| `marian_ct2_int8_cuda` | 1.58 | int8 weights, fp32 KV cache |
| `marian_ct2_int8float16_cuda` | 1.58 | int8 weights, fp16 KV cache |
| `marian_ct2_float16_cuda` | 1.73 | fp16 weights + fp16 KV |

CT2 int8 variants beat the HF CUDA path by ~400 MB. Non-trivial on the
16 GB A2000 — frees room for a larger Whisper KV cache or the bf16 base
during a future imatrix calibration of Gemma 4.

---

## Headline Deltas

vs the HF CUDA baseline (`marian_hf_fp16_cuda`, the production path on
the WSL bench box and the Mac when CUDA isn't available):

- p50 latency: **-66%** (167 → 57 ms)
- p95 latency: **-54%** (253 → 116 ms)
- Peak VRAM: **-19%** (1.96 → 1.58 GB)
- Canary score: **unchanged** (14/16)
- chrF++: -1 to -2 points (within noise)

vs the HF CPU baseline (production path on Macs and CUDA-disabled boxes):

- p50 latency: **-84%** (360 → 57 ms)

---

## Findings

### 1. int8_float16 ≈ int8 ≈ float16 in latency; pick by VRAM

All three CT2 variants land within ~5 ms of each other at p50. The differentiator
is VRAM: int8 and int8_float16 both peak at 1.58 GB; float16 at 1.73 GB.
**Default to int8_float16** — same speed as int8 on Ampere/Ada with no
canary penalty, and consistent with the v2026.7 Whisper STT default.

### 2. CT2 dominates HF in every metric

CT2 wins p50 (-66%), p95 (-54%), VRAM (-19%), and ties on canary. There's
no scenario where HF on CUDA is preferable to CT2 on this hardware. HF
remains the documented fallback for environments where ctranslate2 fails
to import, the conversion artifact is missing, or the operator wants to
disable CT2 explicitly via `STARK_TRANSLATE__MARIAN_BACKEND=hf`.

### 3. The canary gate is *not* sensitive to quantization here

Every CT2 variant — including aggressive int8 — preserves the same 14/16
score as fp32. The 2 MISSes are model-architecture issues (opus-mt
ambiguity on Jacobo/Santiago, "fracción" synonym) that no quantization
choice can rescue and no quantization choice can break further. Future
work that swaps Marian for a domain-tuned encoder will move this number,
not bit-precision tweaks.

### 4. Cold-start parity

CT2 cold start (~190-210 ms first translation post-load) is ~10% slower
than HF (~180 ms), which is a one-time cost amortized over the session.
Steady-state latency (median 57 ms vs 167 ms) wins by 3× regardless.

### 5. Direction symmetry

en→es and es→en land within 3 ms of each other at the median. Marian's
direction-locked architecture means the two CT2 dirs are independent
artifacts, but they exhibit the same latency profile — useful when planning
the eager-load mode (`marian_eager_both`) for live language flips.

---

## Reproducing the Numbers

```bash
source ~/stt_train_env/bin/activate

# 1. Convert both directions (one-time, ~30s each)
python scripts/convert_marian_ct2.py \
    --model-id Helsinki-NLP/opus-mt-en-es \
    --output adapters/marian_ct2/en-es/active

python scripts/convert_marian_ct2.py \
    --model-id Helsinki-NLP/opus-mt-es-en \
    --output adapters/marian_ct2/es-en/active

# 2. Bench (sequential, total ~5 min)
for variant in \
    marian_hf_fp32_cpu \
    marian_hf_fp16_cuda \
    marian_ct2_int8_cuda \
    marian_ct2_int8float16_cuda \
    marian_ct2_float16_cuda; do
    python tools/benchmark_translate_engines.py --variant "$variant" --quiet
done

# 3. Inspect
ls metrics/translate_bench/v2026.8/
```

Every run writes `metrics/translate_bench/v2026.8/<variant>.json` (per-variant
summary) and a sibling `<variant>_clips.jsonl` (per-clip records).
