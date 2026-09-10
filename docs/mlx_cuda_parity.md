# MLX ↔ CUDA Model Parity

Living checklist for keeping Mac (MLX) and CUDA production paths semantically aligned
after pipeline parallelism (#168) and the Gemma 4 CUDA cutover (v2026.5).
Current contracts and evidence are in [the architecture](current_architecture.md)
and [evaluation index](evaluation/README.md); shared code does not establish
equal performance or transcription confidence across backends.

> **Mac default is Gemma 4 OptiQ E4B** (`model_family=gemma4`). Opt out with
> `--model-family translategemma` / `STARK_TRANSLATE_MODEL_FAMILY=translategemma`.

## Checklist

| Dimension | CUDA (prod) | MLX (Mac) | Parity status |
|-----------|-------------|-----------|---------------|
| Finals MT family | Gemma 4 E4B Q4_K_M (`LlamaCppEngine`, `model_family=gemma4`) | **Default:** Gemma 4 OptiQ E4B; TranslateGemma opt-out | Prompt/cleanup shared; Mac default flipped 2026-08-30 |
| Instruct prompt | `engines/translation_prompts.gemma4_user_content` | Same helper | **Aligned** |
| Thinking flag | `chat_template_kwargs.enable_thinking=false` (llama.cpp) | `apply_chat_template(..., enable_thinking=False)` via `chat_template_extra_kwargs` | **Aligned** |
| EOS / preamble | Model-family stop and cleanup rules | `ensure_stop_tokens()` preserves Gemma 4 native stops; shared `clean_translation()` | Shared family contract; do not apply TranslateGemma EOS replacement to Gemma 4 |
| Spec / MTS | Drafting off by default; CUDA proposal remains separate | Live `--mts` is rejected before loading | Negative Mac experiment; speed gate failed, stays off (#177) |
| TurboQuant KV | N/A (use `-ctk q8_0`) | `--turboquant` requested → soft-disabled on mlx-optiq 0.4.x (no drop-in `TurboQuantKVCache` for `mlx_lm.generate`; OptiQ KV lives in serve/runtime) | Unavailable on Mac live path today |
| STT model | W16 CT2 turbo (`adapters/whisper_turbo_ct2/active/`) | Parakeet MLX for EN; Whisper turbo for ES | Separate STT engines; historical roundtrip evidence is in the v2026.13 archive |
| STT confidence | Whisper log probability / compression / no-speech fields | Whisper fields for ES; Parakeet decoder confidence for EN | Do not assume confidence calibration or equal routing behavior across engines |
| Partials timestamps | `word_timestamps=False` | Same | **Aligned** |
| Pipeline overlap | `max_workers=2` | In-process overlap; optional `--multiprocess` uses shared family prompts | Contracts shared; hardware performance measured separately |

## Shared code

- [`engines/translation_prompts.py`](../engines/translation_prompts.py) — prompt builders + cleanup + `dynamic_max_tokens`
- Consumers: `MLXGemmaEngine`, `LlamaCppEngine`, `CUDAGemmaStreamingEngine`; the live MLX path and multiprocess workers delegate to the shared MLX engine.

## Opt-in TranslateGemma / overrides

```bash
# Mac default is already Gemma 4 OptiQ E4B:
python dry_run_ab.py --backend mlx

# Opt out to TranslateGemma
python dry_run_ab.py --backend mlx --model-family translategemma

# Explicit model override
python dry_run_ab.py --backend mlx --model-family gemma4 \
  --mlx-model mlx-community/gemma-4-e4b-it-OptiQ-4bit
```

Use the manifest-pinned OptiQ weights. Other quantizations need separate output
and canary validation. The historical EOS/thinking failure must not be relabeled
as evidence that every other quantization produces bad output.

## Accel matrix (Mac)

```bash
python tools/benchmark_mlx_accel.py --quick
# or:
python tools/benchmark_latency.py --only mlx-accel --quick
```

Configs: `tg4b`, `e4b`, `e2b`, `e4b_mts`, `e4b_tq`, `e4b_mts_tq`.
These are offline probes; an experimental configuration name does not make the
corresponding live flag available. Retain unavailable/failed probe results.

Compare identical transcripts, prompts, token workloads and metric definitions
before claiming Mac/CUDA parity. [Historical measurements](archive/v2026.13/MAC_LATENCY.md)
and the [current 96-run EN→ES screen](evaluation/overnight_screen_20260910/README.md)
are separate cohorts. The current screen promoted no optimization and did not
meet the sub-second final-delivery goal. Natural references and bilingual review
remain quality gates.

## STT notes

- mlx-whisper has no direct Whisper LoRA load path. The specified W16/v2-cpo Mac
  transfer and A/B gate remains open (#135); stock Parakeet results do not satisfy it.
- Keep mlx-whisper turbo; do **not** re-adopt lightning-whisper-mlx ([`fast_stt_options.md`](archive/research/fast_stt_options.md)).
- Probe quantized mlx Whisper via `--stt-model` on the accel bench if a community turbo quant appears.

## Related

- [`docs/mac_pipeline_refresh.md`](mac_pipeline_refresh.md)
- [`docs/archive/v2026.5/BENCHMARK.md`](archive/v2026.5/BENCHMARK.md)
- [`engines/CLAUDE.md`](../engines/CLAUDE.md)
