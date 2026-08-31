# MLX ↔ CUDA Model Parity

Living checklist for keeping Mac (MLX) and CUDA production paths semantically aligned
after pipeline parallelism (#168) and the Gemma 4 CUDA cutover (v2026.5).

> **Mac default is Gemma 4 OptiQ E4B** (`model_family=gemma4`). Opt out with
> `--model-family translategemma` / `STARK_TRANSLATE_MODEL_FAMILY=translategemma`.

## Checklist

| Dimension | CUDA (prod) | MLX (Mac) | Parity status |
|-----------|-------------|-----------|---------------|
| Finals MT family | Gemma 4 E4B Q4_K_M (`LlamaCppEngine`, `model_family=gemma4`) | **Default:** Gemma 4 OptiQ E4B; TranslateGemma opt-out | Prompt/cleanup shared; Mac default flipped 2026-08-30 |
| Instruct prompt | `engines/translation_prompts.gemma4_user_content` | Same helper | **Aligned** |
| Thinking flag | `chat_template_kwargs.enable_thinking=false` (llama.cpp) | `apply_chat_template(..., enable_thinking=False)` via `chat_template_extra_kwargs` | **Aligned** |
| EOS / preamble | Strip `<end_of_turn>` + “Here is the translation:” | Same `clean_translation()` | **Aligned** |
| Spec / MTS | llama.cpp `-md` (bench: loss on single GPU) | Gemma-4 `-assistant` drafter via `--mts` (gamma=1) | Wired; measure on Mac |
| TurboQuant KV | N/A (use `-ctk q8_0`) | `--turboquant` requested → soft-disabled on mlx-optiq 0.4.x (no drop-in `TurboQuantKVCache` for `mlx_lm.generate`; OptiQ KV lives in serve/runtime) | Unavailable on Mac live path today |
| STT model | W16 CT2 turbo (`adapters/whisper_turbo_ct2/active/`) | Stock `mlx-community/whisper-large-v3-turbo` | Same size family; **fine-tune CUDA-only** |
| STT confidence | avg_logprob / compression_ratio / no_speech | Same on `MLXWhisperEngine` | **Aligned** |
| Partials timestamps | `word_timestamps=False` | Same | **Aligned** |
| Pipeline overlap | `max_workers=2` | #168 in-process / `--multiprocess` escape | Pipeline parity in flight |

## Shared code

- [`engines/translation_prompts.py`](../engines/translation_prompts.py) — prompt builders + cleanup + `dynamic_max_tokens`
- Consumers: `MLXGemmaEngine`, `LlamaCppEngine`, `CUDAGemmaStreamingEngine`, `dry_run_ab.translate_mlx`

## Opt-in TranslateGemma / overrides

```bash
# Mac default is already Gemma 4 OptiQ E4B:
python dry_run_ab.py --backend mlx

# Opt out to TranslateGemma
python dry_run_ab.py --backend mlx --model-family translategemma

# + assistant-drafter MTS (gamma=1 on Metal)
python dry_run_ab.py --backend mlx --mts

# Explicit model override
python dry_run_ab.py --backend mlx --model-family gemma4 \
  --mlx-model mlx-community/gemma-4-e4b-it-OptiQ-4bit \
  --mlx-drafter mlx-community/gemma-4-e4b-it-assistant-bf16 --mts
```

**Do not** use naïve `mlx-community/gemma-4-*-4bit` without PLE-safe / OptiQ packaging — PLE quantization produces garbage.

## Accel matrix (Mac)

```bash
python tools/benchmark_mlx_accel.py --quick
# or:
python tools/benchmark_latency.py --only mlx-accel --quick
```

Configs: `tg4b`, `e4b`, `e2b`, `e4b_mts`, `e4b_tq`, `e4b_mts_tq`.

**Gates before claiming Mac≈CUDA latency:** medium p50 competitive with ~470 ms CUDA finals (Mac E4B OptiQ is slower today — quality-first parity). Canary ≥ 7/8 and no PLE garbage are required for any OptiQ soak.

## STT notes

- W16 LoRA → CT2 is **CUDA-only** (mlx-whisper has no LoRA load path).
- Keep mlx-whisper turbo; do **not** re-adopt lightning-whisper-mlx ([`fast_stt_options.md`](archive/research/fast_stt_options.md)).
- Probe quantized mlx Whisper via `--stt-model` on the accel bench if a community turbo quant appears.

## Related

- [`docs/mac_pipeline_refresh.md`](mac_pipeline_refresh.md)
- [`docs/archive/v2026.5/BENCHMARK.md`](archive/v2026.5/BENCHMARK.md)
- [`engines/CLAUDE.md`](../engines/CLAUDE.md)
