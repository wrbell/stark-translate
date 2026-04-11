# TurboQuant + vLLM Integration Planning

Research and implementation plan for integrating TurboQuant KV-cache quantization
with vLLM serving into the stark-translate NVIDIA backend.

---

## TurboQuant Research

**TurboQuant** is a KV-cache quantization technique from Google Research
(ICLR 2026, arXiv 2504.19874). It compresses the key-value cache to
3--3.5 bits per value during autoregressive generation.

Key properties:

- **6x KV memory reduction** compared to FP16 KV cache
- **Up to 8x faster attention** on NVIDIA GPUs (fused dequant + FlashAttention)
- **Zero accuracy loss** — no perplexity or downstream quality degradation
- **Training-free** — applied at inference time, no fine-tuning required
- **Data-oblivious** — no calibration dataset needed
- **Online** — quantizes on the fly as KV pairs are generated

### Community Implementations

| Repo | Description |
|------|-------------|
| [0xSero/turboquant](https://github.com/0xSero/turboquant) | Triton kernels for fused quantized attention |
| [mitkox/vllm-turboquant](https://github.com/mitkox/vllm-turboquant) | vLLM 0.18+ fork with TurboQuant KV support |
| [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) | Pure PyTorch reference (no custom kernels) |

---

## Why It Fits stark-translate

### Current NVIDIA Pipeline

- **STT:** Whisper Large-V3-Turbo via faster-whisper (INT8/FP16)
- **Partial translation:** MarianMT PyTorch (~750 ms)
- **Final translation:** TranslateGemma 4B or 12B, bitsandbytes 4-bit + transformers/accelerate

### Bottleneck

The main VRAM and latency bottleneck is the **KV cache during TranslateGemma
autoregressive generation**. A typical sermon sentence (30--80 tokens input,
40--100 tokens output) allocates a KV cache that scales linearly with sequence
length and model depth. For the 12B model this cache alone can consume 2--3 GB
at FP16.

### Target Hardware

- **A2000 Ada 16 GB** — current training/inference card (WSL2)
- **RTX 3060 12 GB** — target deployment card for church setup

Both are VRAM-constrained. Any KV savings translate directly into either
(a) fitting the 12B model where it previously did not fit, or (b) freeing
headroom for longer sequences and batch-of-one latency improvements.

---

## VRAM Budget Comparison

| Component | Current (bnb 4-bit) | With TurboQuant KV |
|-----------|---------------------|---------------------|
| Whisper Turbo INT8 | 2--3 GB | 2--3 GB (unchanged) |
| TranslateGemma 4B (weights + KV) | ~2.5 GB | ~1.5--1.8 GB |
| TranslateGemma 12B (weights + KV) | ~7.3 GB | ~4--5 GB |
| 12B + Whisper combined | ~10 GB | ~7 GB |

The savings come entirely from compressing the KV cache (weights remain 4-bit).
On the RTX 3060 (12 GB), running 12B + Whisper simultaneously becomes feasible
with ~5 GB of headroom instead of running right at the limit.

---

## Recommended Path: vLLM + TurboQuant

vLLM provides several features that compound with TurboQuant:

- **PagedAttention** — eliminates KV-cache fragmentation, pairs naturally with
  compressed KV pages
- **Continuous batching** — not critical for batch-of-one inference but enables
  future multi-user serving
- **Built-in FlashAttention-2** — already fused; TurboQuant adds fused dequant
  before the attention kernel
- **Native speculative decoding** — use 4B as draft model, 12B as verifier,
  with both benefiting from compressed KV
- **Existing quantization support** — loads AWQ, GPTQ, and bitsandbytes weights
  natively

The combination of 4-bit weights (existing) + 3-bit KV cache (TurboQuant)
gives near-optimal memory efficiency without any quality trade-off.

---

## Implementation Plan

### Phase 0 — Benchmark (1 day)

1. Clone the vLLM-TurboQuant fork:
   ```bash
   pip install git+https://github.com/mitkox/vllm-turboquant.git
   ```
2. Load TranslateGemma 4B and 12B with `kv_cache_dtype="turbo3"`.
3. Measure:
   - Peak VRAM (via `nvidia-smi` polling)
   - Tokens/second for typical sermon sentence lengths (50--120 tokens)
   - Time-to-first-token and total generation latency
4. Compare against current bitsandbytes path in `cuda_engine.py`.
5. Verify translation quality on the 500-segment verse holdout set
   (COMET/chrF should be within noise of current scores).

### Phase 1 — New Engine (2--3 days)

Create `engines/vllm_turbo_engine.py` implementing the `TranslationEngine` ABC:

```python
from vllm import LLM, SamplingParams

class VllmTurboEngine(TranslationEngine):
    def __init__(self, model_id, kv_dtype="turbo3", ...):
        self.llm = LLM(
            model=model_id,
            quantization="bitsandbytes",   # or "awq"
            kv_cache_dtype=kv_dtype,
            max_model_len=512,
            gpu_memory_utilization=0.85,
        )
        self.eos_token_id = 106  # <end_of_turn> for TranslateGemma

    def translate(self, text, src_lang, tgt_lang):
        prompt = self._build_prompt(text, src_lang, tgt_lang)
        params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
            stop_token_ids=[self.eos_token_id],
        )
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text.strip()
```

Registration and wiring:

- Add `engine_type == "vllm"` branch in `engines/factory.py`
- Add `vllm_model_id`, `vllm_kv_dtype`, `vllm_gpu_util` fields to
  `TranslationSettings` in `settings.py`

### Phase 2 — Pipeline Integration (1--2 days)

- Swap final translation calls in `dry_run_ab.py` to use the vLLM engine
  when `--engine vllm` is passed.
- Enable **speculative decoding** via vLLM's built-in support:
  - Draft model: TranslateGemma 4B
  - Verifier: TranslateGemma 12B
  - Both share TurboQuant KV compression
- Add fallback: if vLLM import fails or `--engine cuda` is passed, use the
  existing bitsandbytes path in `cuda_engine.py`. Controlled via `settings.py`.

### Phase 3 — Polish (3--5 days)

- **Docker image** with vLLM pre-installed for reproducible deployment.
- **`--low-vram` mode:** forces 4B only + turbo3 KV + `max_model_len=256`.
  Targets 6--8 GB cards.
- **Benchmarks** using existing `tools/` evaluation scripts (COMET, chrF, WER)
  against the verse holdout and sermon eval sets.
- **AWQ weights** on top of TurboQuant KV: if AWQ checkpoints become available
  for TranslateGemma, swap from bitsandbytes to AWQ for faster weight loading
  and slightly better throughput.

---

## Alternative: Pure PyTorch (Minimal Code Change)

For a lighter-touch integration, use
[turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) to wrap
the `past_key_values` cache directly inside `cuda_engine.py`.

Estimated effort: 1--2 days of patching. Replace the default HF
`DynamicCache` with TurboQuant's `QuantizedKVCache` before calling
`model.generate()`.

Trade-offs vs the vLLM path:

| | Pure PyTorch | vLLM |
|---|---|---|
| Code change | ~50 lines in cuda_engine.py | New engine file + factory wiring |
| VRAM savings | Yes (KV only) | Yes (KV + PagedAttention) |
| Latency improvement | Moderate (no paged attention) | Significant (fused kernels) |
| Speculative decoding | Manual implementation | Built-in |
| Maintenance burden | Low | Medium (vLLM version tracking) |

Recommendation: start with Phase 0 benchmarks. If the vLLM path shows >30%
latency improvement over pure PyTorch, commit to the full vLLM integration.
Otherwise, the pure PyTorch path is a pragmatic win with minimal risk.

---

## Expected Results on Minimal Hardware

| Card | Current State | With TurboQuant + vLLM |
|------|--------------|------------------------|
| RTX 3060 12 GB | 12B barely fits, no Whisper concurrency | 12B + Whisper with ~5 GB headroom |
| A2000 Ada 16 GB | 12B + Whisper fits but tight | 12B + Whisper with ~9 GB headroom |
| RTX 4060 8 GB | 4B only | 12B feasible (~7 GB total) |
| RTX 3050 4--6 GB | MarianMT only | 4B with TurboQuant KV fits |

Additional benefits:

- **Faster finals** — lower latency means the final (high-quality) translation
  replaces the partial sooner, improving the real-time feel for viewers.
- **Zero quality regression** — TurboQuant is lossless at 3-bit KV; all COMET
  and chrF scores should remain within noise of current results.

---

## Dependencies

```
vllm>=0.18
# TurboQuant fork — install from GitHub:
# pip install git+https://github.com/mitkox/vllm-turboquant.git
```

These are NVIDIA-only dependencies and should be added to
`requirements-nvidia.txt`, not the base `requirements.txt`.

---

## Files to Create/Modify

| File | Change |
|------|--------|
| `engines/vllm_turbo_engine.py` | **NEW** — vLLM engine with TurboQuant KV cache |
| `engines/factory.py` | Add `engine_type == "vllm"` branch |
| `settings.py` | Add `vllm_model_id`, `vllm_kv_dtype`, `vllm_gpu_util` to `TranslationSettings` |
| `requirements-nvidia.txt` | Add `vllm>=0.18` |
| `tests/conftest.py` | Add `"vllm"` to `_MOCK_MODULES` |
| `docs/turbo_planning.md` | This document |
