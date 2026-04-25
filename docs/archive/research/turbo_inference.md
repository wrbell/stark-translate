# Turbo Inference — Acceleration Roadmap for stark-translate

## Priority Order (Hard-Coded)

1. **Latency** — real-time captions for live church services
2. **Accuracy** — theological/church domain, zero regression on COMET/WER/glossary
3. **Memory** — Apple Silicon 8-18 GB / NVIDIA 6+ GB

**Targets:** STT < 400 ms, Translation final < 400 ms, A/B peak memory -30-50%.

Every code change must include a measurable faster/better inference path (benchmark script update + before/after numbers in PR description).

---

## Current State (Baseline)

### Apple Silicon (M3 Pro, 18 GB)

| Component | Current Latency | Current Memory | Engine |
|-----------|----------------|----------------|--------|
| STT (Whisper Turbo) | 300-500 ms | ~1.5 GB | `mlx-whisper` (greedy only) |
| Translation 4B (with cache) | 350-550 ms | ~2.5 GB | `mlx_lm` + prompt cache |
| Translation 12B (with 4B draft) | 400-700 ms | ~7 GB | `mlx_lm` + speculative |
| MarianMT fallback | 50-80 ms | ~0.3 GB | PyTorch CPU |
| **End-to-end (4B only)** | **~650-1050 ms** | **~4.3 GB** | |
| **End-to-end (12B + draft)** | **~700-1200 ms** | **~9.5 GB** | |

### NVIDIA CUDA (A2000 Ada 16 GB / RTX 2070 8 GB)

| Component | Current Latency | Current Memory | Engine |
|-----------|----------------|----------------|--------|
| STT (faster-whisper Turbo) | 1.0-1.5 s | ~0.9 GB (INT8) | CTranslate2 |
| Translation 4B | 2.0-3.5 s | ~3.0 GB (NF4) | transformers + bitsandbytes |
| Translation 12B | 2.5-4.0 s | ~7.3 GB (NF4) | transformers + bitsandbytes |
| MarianMT fallback | 50-80 ms | ~0.3 GB | PyTorch |
| **End-to-end (4B only)** | **~3.0-5.0 s** | **~4.7 GB** | |

### Existing Optimizations Already Shipped

| Optimization | Status | Savings | Location |
|-------------|--------|---------|----------|
| Prompt cache (KV prefix) | Shipped (MLX + CUDA) | 50-80 ms | `mlx_engine.py:552`, `cuda_engine.py:886` |
| Speculative decoding (12B) | Benchmarked, not in main pipeline | 15-25% | `benchmark_latency.py:352` |
| EOS token fix | Shipped | avoids 5s padding | `mlx_engine.py:426`, `cuda_engine.py:436` |
| Quality-based STT fallback | Shipped | N/A (accuracy) | both engines |
| Dynamic token limiting | Shipped | avoids over-generation | both engines |
| VRAM tier auto-detection | Shipped (CUDA) | N/A (reliability) | `cuda_engine.py:533` |

---

## Technology Readiness Assessment

Research conducted April 2026. Technologies graded by shipping status.

| Technology | Status | Platform | Expected Impact | Risk |
|-----------|--------|----------|----------------|------|
| **lightning-whisper-mlx** | Shipped (PyPI) | Apple Silicon | STT 4x faster | LOW |
| **mlx-optiq (TurboQuant KV)** | Shipped (PyPI) | Apple Silicon | KV cache 4.6x smaller, +5-15% tok/s | LOW |
| **torch.compile (Whisper)** | Shipped (PyTorch) | CUDA | 4.5-6x overall, decoder > encoder | MEDIUM |
| **CTranslate2 (faster-whisper)** | Already used | CUDA | Already at limit | N/A |
| **vLLM + Gemma 4** | Buggy (Apr 2026) | CUDA | Would replace transformers | HIGH |
| **llama.cpp GGUF** | Shipped | CUDA + CPU | ~25-40 tok/s (vs 15-18) | MEDIUM |
| **Gemma 4 SFP8/2-bit** | Does NOT exist | — | — | N/A |
| **MLX compile (inference)** | Does NOT exist | — | — | N/A |

### Key Reality Checks

1. **MLX compile is training-only.** The `mx.compile()` API compiles forward+backward+update for `mlx.nn.Module` + `mlx.optimizers.Optimizer`. There is no general inference compilation API in MLX. Use `mlx-optiq` and `lightning-whisper-mlx` instead.

2. **Gemma 4 native SFP8/2-bit does not exist.** No pre-quantized 2-bit models ship. Post-training quantization to 2-bit is theoretically possible but unvalidated for translation quality.

3. **vLLM Gemma 4 quantization is buggy.** FP8 dynamic quant produces gibberish (issue #39049). MoE MXFP4 crashes during weight loading (issue #39000). Use NVFP4 only, or stick to transformers + bitsandbytes.

4. **torch.compile helps Whisper overall but the encoder isn't the bottleneck.** The encoder runs once per chunk; the decoder runs once per output token. Compilation helps the decoder loop most. On Ada GPUs: ~4.5x overall (non-quantized), ~6x (quantized).

5. **TurboQuant is real and has multiple competing implementations.** `mlx-optiq` (official, PyPI) bundles TurboQuantKVCache. Also: `mlx-turboquant` (rachittshah), `turboquant-mlx` (arozanov), `turboquant_mlx` (helgklaizar). Use `mlx-optiq` as the canonical choice.

6. **lightning-whisper-mlx is real and fast.** Shipped PyPI package, 4x faster than standard mlx-whisper, supports quantization (4-bit, 8-bit), all Whisper model sizes including large-v3.

---

## Phase 0: Benchmark Infrastructure (1 file, ~30 min)

**Status:** `tools/benchmark_latency.py` already exists (1,428 lines, 8 benchmark suites). Extend it.

### Changes

1. Add `--model-family gemma4` flag to `benchmark_latency.py`:
   - New `bench_gemma4_suite()` that tests Gemma 4 E2B and E4B as translation alternatives
   - Reuses existing `bench_translate_suite()` structure

2. Add Gemma 4 models to `setup_models.py`:
   - `--model-family gemma4 --size e2b` → downloads `google/gemma-4-e2b-it`
   - `--model-family gemma4 --size e4b` → downloads `google/gemma-4-e4b-it`
   - Default remains `translategemma-4b`

3. Add before/after latency tracking:
   - JSON output includes `optimization_applied` field
   - Diff mode: `--compare metrics/baseline.json` prints delta table

### Files to edit

- `tools/benchmark_latency.py` — add `bench_gemma4_suite()`
- `setup_models.py` — add Gemma 4 download paths
- `engines/factory.py` — add `Gemma4Engine` dispatch

### Validation gate

Run full benchmark suite on both platforms before any optimization PR. Save as `metrics/baseline_YYYYMMDD.json`.

---

## Phase 1: Gemma 4 Model Comparison — PR `feat/gemma4-support`

**Goal:** Drop-in support for Gemma 4 E2B/E4B as translation alternative while keeping TranslateGemma as default.

### Implementation

#### `engines/mlx_engine.py` — New `MLXGemma4Engine(TranslationEngine)`

```python
class MLXGemma4Engine(TranslationEngine):
    """Gemma 4 general-purpose model for translation via instruct prompt."""

    def __init__(self, model_id="google/gemma-4-e4b-it", quantization="4bit"):
        self._model_id = model_id
        self._quantization = quantization

    def translate(self, text, source_lang="en", target_lang="es"):
        messages = [{"role": "user", "content":
            f"Translate the following {LANG_NAMES[source_lang]} text to "
            f"{LANG_NAMES[target_lang]}. Output only the translation, "
            f"nothing else.\n\n{text}"
        }]
        # ... generate via mlx_lm.generate()
        return _clean_instruct_output(raw_output)
```

Key differences from TranslateGemma engine:
- Generic instruct prompt (no `source_lang_code`/`target_lang_code` template)
- Output cleaning: strip preambles ("Translation:", "Here is the translation:")
- May need different EOS handling (verify `<end_of_turn>` behavior)

#### `engines/cuda_engine.py` — New `CUDAGemma4Engine(TranslationEngine)`

Same pattern, using transformers + bitsandbytes NF4. Inherits prompt cache logic if template prefix is stable.

#### `engines/factory.py` — Updated dispatch

```python
def create_translation_engine(backend="auto", engine_type="gemma", ...):
    if engine_type == "gemma4":
        if backend == "mlx":
            return MLXGemma4Engine(model_id=model_id)
        else:
            return CUDAGemma4Engine(model_id=model_id)
    # ... existing TranslateGemma path
```

#### `settings.py` — New settings

```python
class TranslationSettings(BaseModel):
    # ... existing fields ...
    gemma4_model_e2b: str = "google/gemma-4-e2b-it"
    gemma4_model_e4b: str = "google/gemma-4-e4b-it"
    model_family: str = "translategemma"  # or "gemma4"
```

### Validation gate

Run `benchmark_gemma4.py` on 480 sermon pairs + 100 Deepgram chunks:
- Gemma 4 E4B must achieve ≥ 95% of TranslateGemma 4B COMET score
- Gemma 4 E4B latency must be ≤ TranslateGemma 4B latency
- If both pass: update default in README
- If latency wins but accuracy loses: keep as optional `--model-family gemma4`

### Estimated impact

| Metric | TranslateGemma 4B | Gemma 4 E4B (est.) | Gemma 4 E2B (est.) |
|--------|-------------------|---------------------|---------------------|
| COMET | 0.7516 | 0.69-0.74 (?) | 0.62-0.68 (?) |
| Latency | 350-550 ms (MLX) | 300-500 ms (?) | 200-350 ms (?) |
| VRAM | 2.5 GB | 2.5 GB | 1.5 GB |

**Unknown until benchmarked.** Gemma 4 is a general model, not translation-specialized. It may lose on accuracy but win on latency due to newer architecture.

---

## Phase 2: TurboQuant KV-Cache Compression — PR `feat/turboquant-kv`

**Goal:** 4-6x smaller KV cache, 5-15% higher tok/s, zero accuracy regression. This is the biggest single win for memory-constrained devices.

### Apple Silicon (MLX) — Primary target (80% of users)

#### Dependencies

```bash
pip install mlx-optiq  # includes TurboQuantKVCache
```

#### Implementation in `MLXGemmaEngine.__init__`

```python
from mlx_optiq import TurboQuantKVCache

class MLXGemmaEngine(TranslationEngine):
    def load(self):
        self._model, self._tokenizer = mlx_lm.load(self._model_id)

        if self._use_turboquant:
            self._model.kv_cache = TurboQuantKVCache(
                self._model,
                key_bits=3,      # near-lossless (within 2.7x of information-theoretic lower bound)
                val_bits=4,
                rotate=True,     # orthogonal rotation preserves attention inner products
            )
```

Generation loop unchanged — `mlx_lm.generate()` works identically with the replaced KV cache.

#### Settings

```python
class TranslationSettings(BaseModel):
    turboquant: bool = True          # enabled by default for new models
    turboquant_key_bits: int = 3
    turboquant_val_bits: int = 4
```

#### CUDA path — Deferred

vLLM TurboQuant integration does not exist yet (MLX-only as of April 2026). CUDA benefits from Phase 4 (llama.cpp) instead, which handles KV cache differently via PagedAttention.

### Validation gate

Run benchmark with/without TurboQuant on 100 church translation pairs:
- BLEU delta ≤ 0.5 points (zero regression threshold)
- COMET delta ≤ 0.005
- KV cache memory ≤ 25% of baseline
- Decode speed ≥ baseline (no regression)

### Estimated impact

| Metric | Without TurboQuant | With TurboQuant |
|--------|--------------------|-----------------|
| KV cache memory | 100% | **~22%** (4.6x compression) |
| Translation latency | 350-550 ms | **330-520 ms** (5-10% faster) |
| 12B feasibility on 8 GB Mac | No | **Yes** (KV cache fits) |
| Accuracy (COMET) | 0.7516 | ~0.7510 (near-lossless) |

**Biggest win: enables 12B quality on 8 GB Apple Silicon devices** where KV cache currently exceeds memory.

---

## Phase 3: Whisper Ultra-Fast Path — PR `feat/whisper-ultra`

**Goal:** STT < 400 ms on both platforms.

### Apple Silicon — Replace `mlx-whisper` with `lightning-whisper-mlx`

#### Dependencies

```bash
pip install lightning-whisper-mlx  # 4x faster than mlx-whisper
```

#### Implementation in `MLXWhisperEngine`

```python
# Before (current):
import mlx_whisper
result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=self._model_id)

# After (Phase 3):
from lightning_whisper_mlx import WhisperMLX
model = WhisperMLX("large-v3-turbo", quantize=True)  # 4-bit quantized
result = model.transcribe(audio_path)
```

Key changes:
- `lightning-whisper-mlx` supports batched decoding and 4-bit quantization
- Supports all Whisper model sizes including large-v3 (check large-v3-turbo support)
- Greedy decoding same as current `mlx-whisper`

**Risk:** `lightning-whisper-mlx` may not support Whisper Large-V3-**Turbo** specifically (it lists large-v3 but turbo is a different architecture with 4 decoder layers). **Must verify before committing.**

Fallback if turbo not supported: use `large-v3` at 4-bit quant (larger but still faster than current unquantized turbo via mlx-whisper).

#### CUDA — `torch.compile` on decoder loop

```python
# In FasterWhisperEngine or a new TorchWhisperEngine:
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
model.model.decoder = torch.compile(model.model.decoder, mode="reduce-overhead")
```

Expected: 4.5-6x overall speedup on Ada architecture (A2000 Ada = compute capability 8.9).

**Caveat:** faster-whisper (CTranslate2) is already highly optimized. `torch.compile` applies to the HuggingFace transformers path, not CTranslate2. To use torch.compile, we'd need a separate `TorchWhisperEngine` class that loads via transformers instead of faster-whisper. Benchmark both to determine winner.

#### Shared — VAD tuning

- Profile current Silero VAD threshold (default 0.3)
- Test lower thresholds (0.1, 0.15, 0.2) for earlier speech detection
- Risk: lower threshold = more false positives (starts transcribing silence/noise)
- Measure: end-of-utterance detection latency (silence_trigger currently 0.5s)

### Validation gate

- STT latency < 400 ms on 10-second church audio (both platforms)
- No WER regression vs current engine (test on fresh eval set, 2,706 chunks)
- Fallback path still works if lightning-whisper-mlx fails

### Estimated impact

| Platform | Current STT | After Phase 3 |
|----------|------------|---------------|
| MLX (M3 Pro) | 300-500 ms | **75-150 ms** (4x from lightning-whisper) |
| CUDA (A2000 Ada) | 1.0-1.5 s | **250-400 ms** (torch.compile on decoder) |

---

## Phase 4: llama.cpp / exllamav2 CUDA Translation — PR `feat/llamacpp-translate`

**Goal:** Replace transformers + bitsandbytes translation path on CUDA with a compiled C++ runtime for 3-5x speedup.

### Why this is necessary

The Python-level autoregressive loop in HuggingFace transformers is the CUDA bottleneck. Each `model.generate()` call involves Python→CUDA→Python round-trips per token. llama.cpp eliminates this with a fused C++ generation loop.

### Implementation

#### New engine: `engines/llamacpp_engine.py`

```python
class LlamaCppGemmaEngine(TranslationEngine):
    """Translation via llama-cpp-python with GGUF-quantized models."""

    def __init__(self, model_path, n_gpu_layers=-1, n_ctx=512):
        from llama_cpp import Llama
        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,  # -1 = all layers on GPU
            n_ctx=n_ctx,
            verbose=False,
        )

    def translate(self, text, source_lang="en", target_lang="es"):
        prompt = self._build_prompt(text, source_lang, target_lang)
        output = self._llm(prompt, max_tokens=256, temperature=0)
        return self._clean_output(output["choices"][0]["text"])
```

#### Model conversion pipeline

```bash
# Convert TranslateGemma 4B to GGUF Q4_K_M
python scripts/export_gguf.py \
    --model google/translategemma-4b-it \
    --output models/translategemma-4b-Q4_K_M.gguf \
    --quantization Q4_K_M

# Convert Gemma 4 E4B to GGUF Q4_K_M
python scripts/export_gguf.py \
    --model google/gemma-4-e4b-it \
    --output models/gemma-4-e4b-Q4_K_M.gguf \
    --quantization Q4_K_M
```

#### Factory dispatch

```python
# settings.py
class CUDASettings(BaseModel):
    translation_runtime: str = "transformers"  # or "llamacpp"

# factory.py
if cuda_settings.translation_runtime == "llamacpp":
    return LlamaCppGemmaEngine(model_path=gguf_path)
```

### Validation gate

- Translation latency < 800 ms (4B) / < 1100 ms (12B) on A2000 Ada
- BLEU/COMET within 1% of transformers NF4 baseline
- VRAM ≤ transformers NF4 usage

### Estimated impact

| Metric | transformers + bnb | llama.cpp GGUF |
|--------|--------------------|----------------|
| Translation latency (4B) | 2.0-3.5 s | **500-800 ms** |
| Translation latency (12B) | 2.5-4.0 s | **700-1100 ms** |
| Tokens per second | 15-18 tok/s | **25-40 tok/s** |
| VRAM (4B Q4_K_M) | ~3.0 GB | ~2.5 GB |
| End-to-end | ~3.0-5.0 s | **~1.5-2.5 s** |

---

## Phase 5: Advanced Accelerations — PR `feat/speculative-accel`

### 5A. Speculative decoding integration (MLX → main pipeline)

**Status:** Benchmarked in `benchmark_latency.py` but NOT integrated into `dry_run_ab.py`.

**Task:** Wire speculative decoding into the live A/B comparison pipeline:

```python
# In dry_run_ab.py or pipeline orchestrator:
if ab_mode and model_12b_loaded:
    result_12b = engine_12b.translate(
        text,
        draft_model=engine_4b._model,
        num_draft_tokens=settings.translation.num_draft_tokens,
    )
```

**Expected:** 15-25% faster 12B generation (already benchmarked, just needs wiring).

### 5B. Speculative decoding with Gemma 4 E2B as draft

If Phase 1 benchmarks show Gemma 4 E2B has fast inference (~200-350 ms) with reasonable accuracy:

```python
# E2B as speculative draft for TranslateGemma 4B target
result = engine_4b.translate(
    text,
    draft_model=gemma4_e2b_model,
    num_draft_tokens=4,
)
```

**Expected:** E2B generates 4 candidate tokens → 4B verifies in one forward pass → 1.2-1.3x speedup on 4B path. Acceptance rate target: 60-80%.

**Risk:** E2B may produce poor Spanish translations (general model, not translation-specialized), leading to low acceptance rate and no speedup.

### 5C. Full async pipeline

```python
# End-to-end with asyncio:
async def process_utterance(audio_chunk):
    stt_task = asyncio.create_task(stt_engine.transcribe(audio_chunk))

    # While STT runs, prepare translation engine (warm KV cache)
    translation_engine.warm_cache()

    stt_result = await stt_task
    translation = await translation_engine.translate_async(stt_result.text)

    if tts_enabled:
        tts_task = asyncio.create_task(tts_engine.synthesize(translation.text))
        # Start displaying translation immediately
        display(translation)
        await tts_task
```

On CUDA with 2-worker thread pool, STT and translation can overlap (STT on GPU while translation prompt cache is being prepared on CPU).

### 5D. GGUF export pipeline

New script: `scripts/export_gguf.py`
- Converts any HuggingFace model to GGUF Q4_K_M
- Validates output against original (10-sentence comparison)
- Registers in `models/manifest.json`

---

## PR Roadmap & Estimated Timeline

| PR | Branch | Dependencies | Duration | Impact |
|----|--------|-------------|----------|--------|
| **Phase 0** | `feat/benchmark-gemma4` | None | 0.5 day | Baseline numbers |
| **Phase 1** | `feat/gemma4-support` | Phase 0 | 1-2 days | Gemma 4 as translation option |
| **Phase 2** | `feat/turboquant-kv` | None (independent) | 1 day | MLX KV cache 4.6x smaller |
| **Phase 3** | `feat/whisper-ultra` | None (independent) | 1-2 days | STT 4x faster (MLX) |
| **Phase 4** | `feat/llamacpp-translate` | Phase 1 (model comparison) | 2-3 days | CUDA translation 3-5x faster |
| **Phase 5A** | `feat/speculative-pipeline` | None | 0.5 day | 12B 15-25% faster in live demo |
| **Phase 5B** | `feat/e2b-draft` | Phase 1 | 1 day | 4B 20-30% faster (if E2B quality sufficient) |

**Critical path:** Phase 0 → Phase 1 → Phase 4 (CUDA speedup is the biggest user-facing win)

**Independent (can parallelize):** Phase 2 (MLX KV), Phase 3 (STT), Phase 5A (speculative wiring)

---

## Target End State

### Apple Silicon (M3 Pro, 18 GB)

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| STT | 300-500 ms | **75-150 ms** | 3-4x |
| Translation 4B | 350-550 ms | **300-450 ms** | 1.1-1.2x |
| Translation 12B | 400-700 ms | **350-600 ms** | 1.1-1.2x |
| KV cache memory | 100% | **22%** | 4.6x smaller |
| **End-to-end (4B)** | **650-1050 ms** | **375-600 ms** | **1.7x** |
| **End-to-end (12B)** | **700-1200 ms** | **425-750 ms** | **1.6x** |

### NVIDIA CUDA (A2000 Ada 16 GB)

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| STT | 1.0-1.5 s | **250-400 ms** | 3-4x |
| Translation 4B | 2.0-3.5 s | **500-800 ms** | 3-5x |
| Translation 12B | 2.5-4.0 s | **700-1100 ms** | 3-4x |
| **End-to-end (4B)** | **3.0-5.0 s** | **750-1200 ms** | **3-4x** |
| **End-to-end (12B)** | **3.5-5.5 s** | **950-1500 ms** | **3-4x** |

### Go/No-Go Gate Per PR

Every PR must demonstrate:
1. Latency improvement (median, p95) on the benchmark suite
2. No accuracy regression (BLEU ± 0.5, COMET ± 0.005, WER ± 0.5%)
3. Memory within budget (no new OOM on target hardware)
4. Before/after table in PR description

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| lightning-whisper-mlx doesn't support large-v3-turbo | MEDIUM | Blocks Phase 3 (MLX STT) | Fall back to large-v3 at 4-bit quant |
| Gemma 4 E4B loses badly on translation accuracy | MEDIUM | Phase 1 yields no improvement | Keep TranslateGemma as default; E4B becomes optional |
| TurboQuant introduces subtle translation errors | LOW | Phase 2 regression | key_bits=4 (more conservative) as fallback |
| llama.cpp GGUF conversion loses quality | LOW | Phase 4 regression | Test Q5_K_M and Q6_K as alternatives to Q4_K_M |
| vLLM Gemma 4 bugs not fixed | HIGH | Blocks vLLM path | Skip vLLM entirely; use llama.cpp instead |
| torch.compile incompatible with CTranslate2 | CERTAIN | Can't apply to faster-whisper | Create separate TorchWhisperEngine for compile path |
| E2B speculative draft has low acceptance rate | MEDIUM | Phase 5B yields no speedup | Keep 4B-as-draft-for-12B (already benchmarked) |
