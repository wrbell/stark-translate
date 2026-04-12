# engines/ — STT + Translation + TTS Engine Layer

## Architecture

`base.py` defines ABCs (`STTEngine`, `TranslationEngine`, `TTSEngine`) and result dataclasses. `mlx_engine.py` and `cuda_engine.py` provide platform implementations. `factory.py` auto-detects backend and returns the right engine. `active_learning.py` logs fallback events to JSONL.

**Callers must call `.load()` before `.transcribe()` or `.translate()`** — `create_stt_engine()` does not auto-load.

## CUDA Inference Runtime

`CUDAGemmaStreamingEngine` provides full feature parity with the MLX path:

- **Streaming**: Token-by-token output via `TextIteratorStreamer`, batched every 3 tokens
- **Prompt cache**: Pre-computed `past_key_values` for fixed chat template prefix (~50-80ms savings)
- **Speculative decoding**: `assistant_model=` parameter for 4B drafting 12B tokens
- **VRAM tier detection**: `detect_vram_tier()` auto-selects `full_ab` / `4b_only` / `marian`

### CUDA Thread Model

CUDA **is** thread-safe (unlike MLX). The pipeline uses `ThreadPoolExecutor(max_workers=2)` to enable true STT/Translation overlap — STT(N) runs concurrently with Translation(N-1) on separate CUDA streams. No translation lock needed on CUDA.

### VRAM Tiers

| Tier | VRAM | Config |
|------|------|--------|
| `full_ab` | ≥15 GB | Whisper + 4B + 12B (~12 GB) |
| `4b_only` | ≥5.5 GB | Whisper + 4B (~4.7 GB) |
| `marian` | <5.5 GB | MarianMT only (~1.3 GB) |

### CUDA Model IDs

| Role | Model ID | Size (NF4) |
|------|----------|------------|
| CUDA Translation A | `google/translategemma-4b-it` | ~3 GB |
| CUDA Translation B | `google/translategemma-12b-it` | ~7 GB |

Settings: `STARK_TRANSLATE__CUDA_MODEL_4B`, `STARK_TRANSLATE__CUDA_MODEL_12B`, `STARK_CUDA__*`.

### CUDASettings (`STARK_CUDA_` env prefix)

| Field | Default | Description |
|-------|---------|-------------|
| `vram_tier` | `auto` | `auto` / `full_ab` / `4b_only` / `marian` |
| `use_prompt_cache` | `True` | Pre-compute KV cache (~50-80ms savings/call) |
| `use_speculative` | `True` | 4B drafts 12B tokens (A/B mode only) |
| `pipeline_workers` | `2` | Thread pool workers (2 = STT/Translation overlap) |
| `streaming_batch_size` | `3` | Tokens per WebSocket batch during streaming |
| `compute_type` | `int8` | faster-whisper CTranslate2 compute type |

### CUDA Engine Classes

| Class | Role |
|-------|------|
| `FasterWhisperEngine` | STT via faster-whisper (CUDA/CPU), quality-based fallback retry |
| `CUDAGemmaEngine` | Basic translation with bitsandbytes 4-bit, no streaming |
| `CUDAGemmaStreamingEngine` | Full-featured: streaming, prompt cache, speculative decoding |

## MLX Thread Safety (CRITICAL)

Metal is NOT thread-safe. All MLX inference (Whisper STT + TranslateGemma translation) must run on a **single thread** via `ThreadPoolExecutor(max_workers=1)`. Concurrent MLX on different threads causes SIGSEGV.

PyTorch operations (MarianMT, Silero VAD) use a separate `_pytorch_lock`. VAD runs inline on the asyncio thread (<1ms) — never on a separate thread (heap corruption risk from concurrent PyTorch).

## Model IDs

| Role | Model ID | Size |
|------|----------|------|
| STT primary | `mlx-community/whisper-large-v3-turbo` | ~1.5 GB |
| STT fallback | `wbell7/distil-whisper-large-v3.5-mlx` | ~1.5 GB |
| Translation A | `mlx-community/translategemma-4b-it-4bit` | ~2.5 GB |
| Translation B | `mlx-community/translategemma-12b-it-4bit` | ~7 GB |
| Partial translate | `Helsinki-NLP/opus-mt-en-es` / `es-en` (MarianMT) | ~298 MB |
| TTS (EN) | Piper `en_US-lessac-high` | ~63 MB |
| TTS (ES) | Piper `es_MX-claude-high` | ~63 MB |

**STT fallback note:** `wbell7/distil-whisper-large-v3.5-mlx` was self-converted from `distil-whisper/distil-large-v3.5` using `mlx-examples/whisper/convert.py` + rename `model.safetensors` → `weights.safetensors`. mlx-whisper can't auto-convert HF transformers format due to `_name_or_path` key in config.

## Memory Budget (M3 Pro 18GB)

| Mode | RAM |
|------|-----|
| 4B-only | ~4.3 GB |
| A/B (4B + 12B) | ~11.3 GB |
| Peak (both loaded) | ~9 GB |

Use `mx.set_cache_limit(100 * 1024 * 1024)` to prevent Metal cache growth with `word_timestamps=True`.

## TranslateGemma EOS Fix

Must add `<end_of_turn>` (id=106) to `tokenizer._eos_token_ids`. Default EOS is `<eos>` (id=1) which the model never generates. Without this fix, generates 256 pad tokens (~5s wasted).

## Confidence-Based Flagging

Whisper exposes three segment-level quality signals:

| Metric | Good | Flag for review | Auto-reject |
|--------|------|----------------|-------------|
| `avg_logprob` | > -0.3 | < -0.5 | < -1.0 |
| `no_speech_prob` | < 0.1 | > 0.3 | > 0.6 (with low logprob) |
| `compression_ratio` | < 1.8 | > 2.0 | > 2.4 (hallucination) |

Flag any word with probability < 0.3. Route bottom 5–15% of segments to human review queue.

**Caveat:** Token confidence mixes language model and acoustic signals. High-frequency function words may score high even when misrecognized. Use segment-level aggregation over individual word scores.

## Speculative Decoding

4B serves as draft model for 12B via `mlx_lm.generate(draft_model=)`, configurable via `--num-draft-tokens`.

## Prompt Caching

Pre-computed KV cache is deep-copied per request. Instrumented for >20ms warnings.

## Other Critical Fixes

- **numba libomp conflict**: Set `os.environ["NUMBA_THREADING_LAYER"] = "workqueue"` before any imports.
- **PyTorch fp16 on MPS**: Causes inf/nan with TranslateGemma (logit collapse). Use bfloat16 or MLX instead.
- **`device_map="auto"` on MPS**: Can offload params to disk causing inf/nan. Use `device_map={"": "mps"}`.
- **`torch_dtype` deprecated**: Use `dtype=` parameter in transformers pipeline.

---

## Adding a New Engine

Follow this 4-step pattern (derived from `base.py`, `mlx_engine.py`, `factory.py`):

1. **Implement the ABC** from `base.py` — subclass `STTEngine`, `TranslationEngine`, or `TTSEngine`. Required methods: `load()`, `transcribe()`/`translate()`/`synthesize()`, `unload()`. Required properties: `model_id`, `backend`.
2. **Create a new file** — e.g., `engines/llamacpp_engine.py` for a llama.cpp backend. Keep one engine class per file.
3. **Register in `factory.py`** — add a branch in `create_stt_engine()`, `create_translation_engine()`, or `create_tts_engine()` with a lazy import (import inside the branch, not at module level).
4. **Add mock modules to `tests/conftest.py`** — append any new native dependencies (e.g., `llama_cpp`) to the `_MOCK_MODULES` list so CI tests pass without the library installed.

**Example — llama.cpp translation engine:**
- Create `LlamaCppGemmaEngine(TranslationEngine)` in `engines/llamacpp_engine.py`
- Register as `engine_type == "llamacpp"` in `create_translation_engine()` (see `docs/optimized.md` for full implementation plan)
- Add `"llama_cpp"` to `_MOCK_MODULES` in `tests/conftest.py`

## Adding a New Language

The engine interfaces already support new languages — no ABC changes needed:

- **TranslateGemma**: `translate()` accepts `source_lang`/`target_lang` params. Same model, pass different `source_lang_code`/`target_lang_code` in the chat template (e.g., `"hi"`, `"zh-Hans"`).
- **MarianMT**: Install the right `opus-mt-{src}-{tgt}` model via `create_translation_engine(engine_type="marian", model_id="Helsinki-NLP/opus-mt-en-hi")`. Each language direction is a separate model.
- **TTS**: Add a voice entry to `settings.tts.voices` dict (already has `hi` and `zh` defaults in the settings schema).
- **Whisper prompt**: Create a language-specific initial prompt (e.g., `whisper_prompt_hi`) in settings, or pass via `initial_prompt=` to `transcribe()`.
- **Hindi partial caveat**: Hindi is SOV — partials will be garbled. Show English partial + Hindi final only (no Hindi partial). See `docs/multi_lingual.md` Phase 4B.

## Adapter Loading

Fine-tuned LoRA adapters integrate as follows:

- **MLX**: `mlx_lm.load(model_path, adapter_path=)` — point `adapter_path` to `adapters/{model}/active/`
- **CUDA**: Merge LoRA into base model offline, export GGUF/quantized, load merged model (no runtime adapter swapping)
- **Hot-reload (Mac)**: Pipeline receives SIGUSR1, re-runs `load()` within the single-thread executor (Metal thread safety preserved). During reload (~2-3s), VAD + STT + MarianMT partials continue — only TranslateGemma finals pause.
- **Directory convention**: `adapters/{model}/active/` (current) + `adapters/{model}/previous/` (one-step rollback). See `docs/deploy.md` for the full 6-phase deployment pipeline.

## Settings Integration

When adding engine config, follow this pattern from `settings.py`:

```python
class MySettings(BaseSettings):
    my_param: str = Field(default="value", description="...")
    model_config = SettingsConfigDict(env_prefix="STARK_MY_")
```

Add `my: MySettings = Field(default_factory=MySettings)` to `PipelineSettings`. All settings use the `STARK_` env prefix convention.
