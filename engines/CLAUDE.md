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

> **Tier breakpoints above were derived from `torch.cuda.max_memory_allocated()` — that metric undercounts on Gemma 4 by ~2× (misses bnb scratch + bf16 PLE embeddings). For accurate VRAM budgeting use the nvidia-smi-measured numbers in `docs/archive/v2026.5/BENCHMARK.md`. The `detect_vram_tier()` helper still works on its own thresholds, but those thresholds must be raised when serving Gemma 4 via HF NF4 — E2B alone occupies ~14 GB total.**

### CUDA Model IDs (NF4 via bitsandbytes — measured peak VRAM)

| Role | Model ID | Peak VRAM (nvidia-smi) | Notes |
|------|----------|------------------------|-------|
| CUDA Translation (Gemma 4 E2B, post-#46) | `google/gemma-4-e2b-it` | **~14.2 GB** | PLE keeps embeddings in bf16; large vs param count |
| CUDA Translation (Gemma 4 E4B) | `google/gemma-4-e4b-it` | **~15.6 GB** | Best HF quality; tight on 16 GB |
| CUDA Translation (TG4B legacy) | `google/translategemma-4b-it` | ~7.2 GB | Pre-#46; lower canary score (5/8) |
| CUDA Translation (TG12B legacy) | `google/translategemma-12b-it` | ~15.6 GB | Largest; same canary as E2B (6/8) |

> **For deployment on RTX 3060 12 GB, all HF NF4 configs except TG4B are too big.** Use `engines/llamacpp_engine.py` instead — Gemma 4 E2B Q4_K_M peaks at ~3.5 GB, E4B Q4_K_M at ~4.9 GB. See `docs/archive/v2026.5/BENCHMARK.md` Phase 1A results.

### llama.cpp engine (recommended for CUDA, v2026.5+)

| Role | GGUF | Peak VRAM | tok/s on A2000 Ada |
|------|------|-----------|---------------------|
| Gemma 4 E2B Q4_K_M | `models/gemma-4-e2b-it-q4km.gguf` (3.2 GB on disk) | **3.5 GB** | **66 tok/s** (canary 6/8) |
| Gemma 4 E4B Q4_K_M | `models/gemma-4-e4b-it-q4km.gguf` (5.0 GB on disk) | **4.9 GB** | **41 tok/s** (canary 7/8) |
| E4B + E2B draft (spec decode) | both | 8.5 GB | 36 tok/s — slower than E4B alone on single GPU |

Caller starts `llama-server` (see `start_server.sh`), then constructs via `create_translation_engine(engine_type="llamacpp")`. Must pass `chat_template_kwargs: {"enable_thinking": false}` for Gemma 4 — already handled inside `LlamaCppEngine`.

Settings: `STARK_TRANSLATE__CUDA_MODEL_4B`, `STARK_TRANSLATE__CUDA_MODEL_12B`, `STARK_CUDA__*`.

### CUDASettings (`STARK_CUDA_` env prefix)

| Field | Default | Description |
|-------|---------|-------------|
| `vram_tier` | `auto` | `auto` / `full_ab` / `4b_only` / `marian` |
| `use_prompt_cache` | `True` | Pre-compute KV cache (~50-80ms savings/call) |
| `use_speculative` | `True` | 4B drafts 12B tokens (A/B mode only) |
| `pipeline_workers` | `2` | Thread pool workers (2 = STT/Translation overlap) |
| `streaming_batch_size` | `3` | Tokens per WebSocket batch during streaming |
| `compute_type` | `int8_float16` | faster-whisper CTranslate2 compute type. Default bumped from `int8` in v2026.7 (CTranslate2 docs recommend `int8_float16` for Ampere/Ada; v2026.7 bench saw 0% latency / 0% VRAM delta on A2000 Ada — set back to `int8` if your hardware shows VRAM growth). |
| `engine` | `auto` | Translation engine (auto / llamacpp / hf) |

### TranslationSettings (`STARK_TRANSLATE_` env prefix) — Marian (v2026.8)

| Field | Default | Description |
|-------|---------|-------------|
| `marian_backend` | `auto` | `auto` (CT2 if `adapters/marian_ct2/<dir>/active/model.bin` present, else HF), `ct2` (force CT2; raises if missing), `hf` (force HF). Set via `STARK_TRANSLATE__MARIAN_BACKEND`. |
| `marian_compute_type` | `int8_float16` | CT2 compute type for `MarianCT2Engine`. Mirrors v2026.7 STT default. Set `int8` for VRAM-constrained 6 GB cards (no measurable latency penalty). |
| `marian_max_new_tokens` | `128` | Max decoding length. Marian rarely emits beyond ~80 tokens; 128 is a safe ceiling. |
| `marian_warmup_passes` | `2` | Warmup translations at engine load. Pass 1: `"Hello"`; pass 2: `"Lord, have mercy on us."` (theological subword path). |
| `marian_eager_both` | `False` | Pre-load both en-es and es-en at startup. Only meaningful with `--allow-flip`; lazy by default. |

### CUDA Engine Classes

| Class | Role |
|-------|------|
| `FasterWhisperEngine` | STT via faster-whisper (CUDA/CPU), quality-based fallback retry. Default `compute_type` since v2026.7 is `int8_float16` (~20% faster than `int8`, +30% VRAM). Auto-loads `adapters/whisper_turbo_ct2/active/` (W16 fine-tune, 7.25% fresh-eval WER) when present; off-the-shelf `large-v3-turbo` is the fallback model_id. |
| `ParakeetEngine` | Optional EN-only STT via NVIDIA NeMo Parakeet TDT (`stt_backend="parakeet"`). Not bilingual — keep Whisper for ES. Requires `nemo_toolkit[asr]`. Bench with `tools/benchmark_parakeet_en.py` before adopting for `--lang en`. |
| `HFWhisperEngine` | STT via HF transformers Whisper. Supports `compile_mode` (torch.compile w/ CUDA graphs) + `warmup_seconds` constructor args (v2026.7) and `assistant_model` for spec decode. **Spec-decode default draft removed in v2026.7** — distil-large-v3.5 + whisper-turbo is broken (different decoder layer counts → 10× slower with hallucinated output, see `docs/archive/v2026.5/spec_decode_research.md`). Caller must supply a verified-compatible draft. Faster-whisper is the default everywhere else. |
| `CUDAGemmaEngine` | Basic translation with bitsandbytes 4-bit, no streaming |
| `CUDAGemmaStreamingEngine` | Full-featured: streaming, prompt cache, speculative decoding |
| `MarianCT2Engine` | Fast partial translator via CTranslate2 (v2026.8). 57 ms p50 / 116 ms p95 on A2000 Ada at `int8_float16` — 2.9× faster than the HF CUDA path with identical canary score and lower peak VRAM. Auto-loads `adapters/marian_ct2/{en-es,es-en}/active/` when present; CT2 `Translator` is internally thread-safe so this engine drops the historical `_pytorch_lock` (live pipeline can call `translate()` concurrently). See `docs/archive/v2026.8/MARIAN_BENCHMARK.md`. |
| `MarianHFEngine` (in `engines/marian_hf_engine.py`) | Fast partial translator via HF transformers — fallback path when CT2 isn't available (Mac MLX, missing `ctranslate2`, or explicit `STARK_TRANSLATE__MARIAN_BACKEND=hf`). Acquires the shared `_pytorch_lock` from `engines/_locks.py` (Silero VAD uses the same lock — pre-v2026.8 they had separate locks, a latent thread-safety bug). |

## MLX Thread Safety (MLX >= 0.31.2)

As of **mlx 0.31.2**, independent models may run concurrently on separate threads via **thread-local streams** (see ml-explore/mlx#3078). The live pipeline uses `ThreadPoolExecutor(max_workers=2)` on Mac — same as CUDA — so **STT(N) can overlap Translation(N−1)**.

Rules that still apply:

- Materialize weights on the load thread (`mx.eval(model.parameters())` / `mx.synchronize()` after Whisper warmup) before pool workers use them. Lazy arrays are bound to the creating thread's stream (#3529).
- Do not share one stream across threads without serialization (`mx.new_thread_unsafe_stream`).
- `--multiprocess` remains an optional escape hatch (separate OS processes / Metal contexts) for debugging or older mlx builds — not required for overlap on 0.31.2+.
- Companion: **mlx-lm >= 0.31.3** (thread-local generation stream).

PyTorch operations (MarianMT, Silero VAD) use a separate `_pytorch_lock`. VAD runs inline on the asyncio thread (<1ms) — never on a separate thread (heap corruption risk from concurrent PyTorch).

## Model IDs

| Role | Model ID | Size |
|------|----------|------|
| STT primary (Mac) | `mlx-community/whisper-large-v3-turbo` | ~1.5 GB |
| STT primary (CUDA, v2026.7) | merged W16 LoRA at `adapters/whisper_turbo_ct2/active/` (~777 MB CT2 int8_float16) — falls back to off-the-shelf `large-v3-turbo` (downloaded by faster-whisper into the cache) | ~777 MB |
| STT fallback (Mac) | `wbell7/distil-whisper-large-v3.5-mlx` | ~1.5 GB |
| STT fallback (CUDA) | off-the-shelf `large-v3` via faster-whisper, lazy-loaded by `FasterWhisperEngine` on low-confidence retry | ~3 GB |
| Translation A (Mac default) | `mlx-community/gemma-4-e4b-it-OptiQ-4bit` | OptiQ 4-bit |
| Translation A (TG opt-out) | `mlx-community/translategemma-4b-it-4bit` | ~2.5 GB |
| Translation B (TG A/B) | `mlx-community/translategemma-12b-it-4bit` | ~7 GB |
| Translation CUDA prod | Gemma 4 E4B Q4_K_M via llama.cpp | ~4.9 GB VRAM |
| Partial translate | `Helsinki-NLP/opus-mt-en-es` / `es-en` (MarianMT) | ~298 MB |
| TTS (EN) | Piper `en_US-lessac-high` | ~63 MB |
| TTS (ES) | Piper `es_MX-claude-high` | ~63 MB |

**Shared prompts:** All Gemma 4 / TranslateGemma chat strings and cleanup live in [`translation_prompts.py`](./translation_prompts.py). See [`docs/mlx_cuda_parity.md`](../docs/mlx_cuda_parity.md).

**STT fallback note (Mac):** `wbell7/distil-whisper-large-v3.5-mlx` was self-converted from `distil-whisper/distil-large-v3.5` using `mlx-examples/whisper/convert.py` + rename `model.safetensors` → `weights.safetensors`. mlx-whisper can't auto-convert HF transformers format due to `_name_or_path` key in config.

**STT spec-decode caveat:** distil-large-v3.5 cannot draft for whisper-large-v3-turbo (target = 4 decoder layers, draft = 32). Tested 2026-04-13: 10× slower with hallucinated output. The factory raises `ValueError` when `spec_decode=True` is passed without an explicit `draft_model_id`. Verified-compatible pairing: turbo (4 layers) drafts for large-v3 (32 layers). See `docs/archive/v2026.5/spec_decode_research.md`.

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

Flag any word with probability < 0.5. Route bottom 5–15% of segments to human review queue.

**Caveat:** Token confidence mixes language model and acoustic signals. High-frequency function words may score high even when misrecognized. Use segment-level aggregation over individual word scores.

## Speculative Decoding / MTS

- **TranslateGemma A/B:** 4B drafts for 12B via `mlx_lm.generate(draft_model=)`, `--num-draft-tokens` (default 3).
- **Gemma 4 MTS:** assistant drafter (`--mts` / `STARK_TRANSLATE_MLX_MTS`) with gamma=1 on Metal. Accel matrix: `tools/benchmark_mlx_accel.py`.

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
- Register as `engine_type == "llamacpp"` in `create_translation_engine()` (see `docs/archive/research/optimized.md` for full implementation plan)
- Add `"llama_cpp"` to `_MOCK_MODULES` in `tests/conftest.py`

## Adding a New Language

The engine interfaces already support new languages — no ABC changes needed:

- **TranslateGemma**: `translate()` accepts `source_lang`/`target_lang` params. Same model, pass different `source_lang_code`/`target_lang_code` in the chat template (e.g., `"hi"`, `"zh-Hans"`).
- **MarianMT**: Install the right `opus-mt-{src}-{tgt}` model via `create_translation_engine(engine_type="marian", model_id="Helsinki-NLP/opus-mt-en-hi")`. Each language direction is a separate model.
- **TTS**: Add a voice entry to `settings.tts.voices` dict (already has `hi` and `zh` defaults in the settings schema).
- **Whisper prompt**: Create a language-specific initial prompt (e.g., `whisper_prompt_hi`) in settings, or pass via `initial_prompt=` to `transcribe()`.
- **Hindi partial caveat**: Hindi is SOV — partials will be garbled. Show English partial + Hindi final only (no Hindi partial). See `docs/archive/research/multi_lingual.md` Phase 4B.

## Adapter Loading

Fine-tuned LoRA adapters integrate as follows:

- **MLX**: `mlx_lm.load(model_path, adapter_path=)` — point `adapter_path` to `adapters/{model}/active/`
- **CUDA Whisper STT (v2026.7)**: Use `training/export_ct2.py` to merge the LoRA into Whisper bf16 and convert to CTranslate2. Output goes to `whisper_ct2/{run}/`; register via `tools/manage_adapters.py register --model whisper_turbo_ct2 --adapter whisper_ct2/{run}`. The factory auto-loads `adapters/whisper_turbo_ct2/active/` when present (no engine-side change required). Override via `model_id=` kwarg or set `STARK_STT__WHISPER_CUDA_MODEL`.
- **CUDA Gemma**: Merge LoRA into base model offline, export GGUF/quantized, load merged model (no runtime adapter swapping)
- **Hot-reload (Mac)**: Pipeline receives SIGUSR1, re-runs `load()` on the pipeline pool (materialize weights after reload). During reload (~2-3s), VAD + STT + MarianMT partials continue — only TranslateGemma finals pause.
- **Directory convention**: `adapters/{model}/active/` (current) + `adapters/{model}/previous/` (one-step rollback). See `docs/deploy.md` for the full 6-phase deployment pipeline.

## Settings Integration

When adding engine config, follow this pattern from `settings.py`:

```python
class MySettings(BaseSettings):
    my_param: str = Field(default="value", description="...")
    model_config = SettingsConfigDict(env_prefix="STARK_MY_")
```

Add `my: MySettings = Field(default_factory=MySettings)` to `PipelineSettings`. All settings use the `STARK_` env prefix convention.
