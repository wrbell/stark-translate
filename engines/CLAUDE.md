# engines/ — STT + Translation + TTS Engine Layer

> Paired with [`AGENTS.md`](./AGENTS.md) (agent constraints and navigation).
> Statements below describe v2026.14 source tracked by [PR #192](https://github.com/wrbell/stark-translate/pull/192).
> The last published release recorded here is v2026.13. Historical benchmark numbers live under dated links in
> [`docs/archive/`](../docs/archive/) and are not repeated here.

Backend-agnostic ABCs with MLX (Apple Silicon), CUDA (NVIDIA) and CPU implementations.
`create_stt_engine()` / `create_translation_engine()` in [`factory.py`](./factory.py)
return **unloaded** engines — call `.load()` before `.transcribe()` / `.translate()`.

## Module map

| File | Role |
|------|------|
| `base.py` | `STTEngine`, `TranslationEngine`, `TTSEngine` ABCs; `STTResult` / `TranslationResult` dataclasses; `text_compression_ratio()` |
| `factory.py` | Hardware tier detection (`mlx` > `cuda` > `cpu`), STT backend selection, Marian CT2/HF routing, MLX Gemma model-id resolution |
| `mlx_engine.py` | `MLXWhisperEngine`, `MLXGemmaEngine` (Gemma 4 OptiQ or TranslateGemma), `PiperTTSEngine`, `warm_mlx_model()` |
| `parakeet_mlx_engine.py` | `ParakeetMLXEngine` — Parakeet TDT 0.6B v3 via `parakeet-mlx` (Mac EN default) |
| `parakeet_engine.py` | `ParakeetEngine` — NeMo Parakeet, EN-only CUDA accelerator; never the bilingual default |
| `cuda_engine.py` | `FasterWhisperEngine` (CT2), `MarianCT2Engine`, `CUDAGemmaEngine`, `CUDAGemmaStreamingEngine` (HF NF4, legacy) |
| `llamacpp_engine.py` | `LlamaCppEngine` — Gemma 4 GGUF through `llama-server` (CUDA production finals) |
| `marian_hf_engine.py` | `MarianHFEngine` — PyTorch fallback for partials, guarded by `_pytorch_lock` |
| `hf_whisper_engine.py` | `HFWhisperEngine` — transformers Whisper; only path with `torch.compile` / spec decode |
| `translation_prompts.py` | `build_chat_messages()`, `gemma4_user_content()`, `ensure_stop_tokens()`, Gemma 4 output cleanup |
| `model_paths.py` | Pure offline lookup (`resolve_model_path`, `resolve_marian_ct2`, `resolve_piper_voice`) plus explicit pinned acquisition at loading boundaries (`resolve_model_for_loading`) |
| `_locks.py`, `mlx_generation_lock.py` | Shared PyTorch lock; per-model MLX generation locks so distinct models overlap |
| `audio_devices.py` | Lazy output-device discovery and per-language TTS routing with hotplug re-resolution |
| `mlx_spec.py`, `spec_decode_logger.py` | Experimental mlx-optiq spec-decode wrapper and its telemetry (see #177) |
| `active_learning.py` | JSONL logger for STT fallback events (quality layer 6) |

Live Mac MLX wrappers resolve an existing explicit local override or a complete
cached/downloaded snapshot before calling upstream loaders. Unknown uncached remote IDs require a full manifest revision or an
explicit local path; setup/preflight lookup stays offline. Existing local caches
and defaults are preserved. [Exact pinning scope and remaining download sites](../docs/evaluation/mac_followup_20260910/live-hf-pinning.md)
keep native validation and broad B615 findings separate.

## Current defaults (Mac, `--backend auto` → `mlx`)

| Role | Selection | Source |
|------|-----------|--------|
| STT EN | `ParakeetMLXEngine`, `mlx-community/parakeet-tdt-0.6b-v3` | `dry_run_ab.py` picks `parakeet-mlx` for `--lang en` when available; `--stt-backend mlx` forces Whisper |
| STT ES | `MLXWhisperEngine`, `mlx-community/whisper-large-v3-turbo`; Spanish never retries the English-only Distil fallback | `settings.stt` |
| Partials | `MarianCT2Engine` int8 on CPU (`intra_threads=4`) from `adapters/marian_ct2/<dir>/active` or the managed setup cache; `MarianHFEngine` if no CT2 artifact | `factory.create_translation_engine(engine_type="marian")` |
| Finals | `MLXGemmaEngine`, `mlx-community/gemma-4-e4b-it-OptiQ-4bit`; `--gemma4-size e2b` → E2B OptiQ | `settings.translation.model_family = "gemma4"` |
| Opt-out | `--model-family translategemma` → `mlx-community/translategemma-4b-it-4bit` (+12B with `--ab`) | `resolve_mlx_translation_model_id()` |
| TTS | `PiperTTSEngine`, voices `en_US-lessac-high` / `es_MX-claude-high` (`settings.tts.voices`) | `create_tts_engine()` |
| VAD | Packaged Silero 6.2.1 through `tools/vad_runtime.py` (no Torch Hub); ONNX via `--vad-backend onnx` | `settings.vad.backend` |

Naïve uniform 4-bit `mlx-community` Gemma 4 quants are known-broken (PLE layers
quantized); only the OptiQ mixed-precision repos are supported.

**CUDA (`--backend cuda`):** `FasterWhisperEngine` prefers `adapters/whisper_turbo_ct2/active`
(W16 fine-tune) when present, else `large-v3-turbo`; `MarianCT2Engine` int8_float16;
finals through `LlamaCppEngine` when `start_server.sh` is running (`--engine auto`
probes `http://127.0.0.1:8090`). HF NF4 Gemma 4 is not recommended on 16 GB cards —
PLE embeddings stay bf16 and occupy 14–15 GB; see
[`docs/archive/v2026.5/BENCHMARK.md`](../docs/archive/v2026.5/BENCHMARK.md).

## Gemma 4 stop tokens and prompts (do not regress #172)

`ensure_stop_tokens(tokenizer, model_family=...)` runs after every load. It **adds**
the family's stop tokens to `tokenizer._eos_token_ids` and preserves every EOS id the
loader supplied. For Gemma 4 that means `<eos>` (id 1), `<turn|>` (id 106) and the
`<|tool_response>` id (50); the historical TranslateGemma fix of adding
`<end_of_turn>` (id 106 in the Gemma 3 vocabulary) must not be applied to Gemma 4,
and replacing the set with `{1, 3}` truncates output. Gemma 4 receives the plain
instruct prompt from `gemma4_user_content()` with thinking disabled; TranslateGemma
receives the structured `source_lang_code` / `target_lang_code` template. The
`--multiprocess` worker in [`workers.py`](../workers.py) goes through
`MLXGemmaEngine` with the parent-selected `model_family`, so both paths share
these rules (#176).

## MLX thread safety

- **mlx ≥ 0.31.2 / mlx-lm ≥ 0.31.3:** thread-local streams allow `ThreadPoolExecutor(max_workers=2)` overlap of STT(N+1) with translation(N) in one process. `--multiprocess` (separate Metal contexts) remains an escape hatch only.
- **Materialize on the load thread:** `warm_mlx_model()` runs `mx.eval` on weights and the first Gemma forward on the loading thread before pool handoff; skipping it reproduces the first-forward crash covered by `tests_gpu/test_mlx_worker_first_forward.py`.
- **One lock per model:** `mlx_generation_lock.py` serializes generation on a single model while distinct models overlap.
- **PyTorch lock:** `MarianHFEngine` and the Silero VAD share `_pytorch_lock`; concurrent PyTorch forwards from different threads corrupt the Metal heap. Default VAD remains on the asyncio thread; serialized worker VAD is an opt-in experiment.
- **Metal cache:** engines set `mx.set_cache_limit(256 MB)` (`cache_limit_mb`).

## Confidence and fallback (quality layer 3)

`MLXWhisperEngine` defaults: `fallback_threshold=-1.2` (avg_logprob below → retry
with the fallback model), `hallucination_threshold=2.4` (compression ratio above →
retry), `fallback_on_low_conf=True`. Automatic fallback is English-only. Spanish keeps
the primary result on low confidence and fails visibly if its selected multilingual
model cannot load. Engine callers can pass `session_language="es"` to guard startup;
Spanish inference also rejects an engine that already selected an English fallback.
Custom fallback IDs do not establish multilingual support. Fallback events are logged by
`active_learning.py`. `ParakeetMLXEngine` derives `avg_logprob` / word probabilities
from TDT token confidences — a proxy, not a calibrated Whisper probability — and does
not use the Whisper fallback chain.

## Speculative decoding — status

- **Whisper HF spec decode:** requires an explicit, verified draft. distil-large-v3.5 → whisper-large-v3-turbo is incompatible (10× slower, hallucinations; [`docs/archive/v2026.5/spec_decode_research.md`](../docs/archive/v2026.5/spec_decode_research.md)). Off by default.
- **TranslateGemma 4B → 12B drafting (`--ab --num-draft-tokens`):** legacy A/B path only.
- **Gemma 4 assistant drafter (`--mts`, #177):** **off, and rejected before any model loads.** `dry_run_ab.py` `validate_live_mts()` raises `LIVE_MTS_UNAVAILABLE` ("Live --mts is unavailable: the supported mlx-lm loader cannot load the Gemma 4 assistant drafter… run with --no-mts") when `--mts` or `STARK_TRANSLATE_MLX_MTS` is set; the live pipeline never continues silently after a drafter failure. The working offline probe is `engines/mlx_spec.py` over mlx-optiq (`tools/mts_acceptance_probe.py`). Results and the rejected RoPE hypothesis: [`docs/archive/v2026.13/MAC_LATENCY.md`](../docs/archive/v2026.13/MAC_LATENCY.md), [`docs/mlx_mtp_notes.md`](../docs/mlx_mtp_notes.md).
- **CUDA MTP:** `start_server.sh --mtp` opt-in on llama.cpp `b10883`; unbenchmarked on hardware ([`docs/cuda_latency_proposal.md`](../docs/cuda_latency_proposal.md)).
- **Latency experiments** (`tools/latency_experiments.py`, `tools/latency_scheduler.py`, `tools/incremental_stt.py`, `tools/preview_candidates.py`): opt-in research controls validated before session startup, never promoted by selecting a backend; evidence is collected by `tools/overnight_bench.py` (parent-owned).

## Deployment profiles (`stark_translate/profiles.py`)

`apply_profile()` runs once after CLI overrides and before model loads. `standard` leaves
the factory's platform selection alone. The Lite profiles (`lite-cpu`, `lite-cpu-quality`,
`lite-cuda-8gb`) force: `stt.backend = faster-whisper` with the pinned CT2 model
(`whisper-small` int8 on CPU; `whisper-large-v3-turbo` int8_float16 on CUDA), 3 CT2 threads
and one STT worker, `local_files_only`, no low-confidence fallback model; `vad.backend =
onnx`; `marian_backend = ct2` on CPU int8 with one intra-thread; `model_family = gemma4`
with E2B; finals via `MarianCT2Engine` (`lite-cpu`, `low_vram`) or `LlamaCppEngine`
against a session-owned `llama-server` started by `tools/llama_runtime.py`
(`lite-cpu-quality`, `lite-cuda-8gb`); `run_ab`, `multiprocess`, `use_speculative` and
`mlx_mts` off. Lite never inherits the standard path's W16 CT2 adapter preference. E2B or
native-runtime failures fail the session; no HF NF4 fallback. Admission floors and
evidence: [`docs/lite_profiles.md`](../docs/lite_profiles.md).

## Environment variables (pydantic-settings, `settings.py`)

Nested groups use the group prefix directly (`STARK_STT_`, `STARK_TRANSLATE_`,
`STARK_VAD_`, `STARK_TTS_`, `STARK_CUDA_`); the top-level `PipelineSettings` also
accepts `STARK_<GROUP>__<FIELD>` with the double-underscore delimiter. Common keys:

| Variable | Effect |
|----------|--------|
| `STARK_STT_BACKEND` / `--stt-backend` | `auto`, `mlx`, `parakeet-mlx`, `faster-whisper`, `hf`, `parakeet` |
| `STARK_TRANSLATE_MODEL_FAMILY` | `gemma4` (default) or `translategemma` |
| `STARK_TRANSLATE_MARIAN_BACKEND` | `auto` (CT2 when artifact present), `ct2` (strict), `hf` |
| `STARK_TRANSLATE_MLX_MTS` | Drafter flag (#177); when set, `validate_live_mts` aborts the live session before load — leave unset |
| `STARK_PROFILE` / `--profile` | `standard` (default), `lite-cpu`, `lite-cpu-quality`, `lite-cuda-8gb`; `stark-translate-lite` defaults to `lite-cpu` |
| `STARK_VAD_BACKEND` | `torch` (default) or `onnx` |
| `STARK_TTS_OUTPUT_DEVICES` | Per-language local TTS output map (index, name substring or null) |
| `STARK_MODELS_DIR` | Model cache used by setup and inference (`engines/model_paths.py`) |

## Adapters

- **MLX Gemma:** `--adapter-dir` (primary) / `--adapter-dir-b` (12B in `--ab`) pass `adapter_path=` to `mlx_lm.load`. Registry and health gate: [`tools/CLAUDE.md`](../tools/CLAUDE.md), [`docs/deploy.md`](../docs/deploy.md) (8-canary `health_check.py`).
- **Whisper W16 CT2:** consumed by `FasterWhisperEngine` on CUDA/CPU only; mlx-whisper and Parakeet do not load LoRA adapters.
- **Marian CT2:** `scripts/convert_marian_ct2.py --quantization int8` for custom directories; `stark-translate setup` builds the managed copy and never touches `adapters/marian_ct2/*/active`.

## Adding a New Engine

1. Subclass the ABC in `base.py` and return the shared result dataclasses.
2. Add the implementation file (one backend per file) and gate heavy imports inside `load()`.
3. Add the selection branch in `factory.py` (validate incompatible combinations early, as `parakeet-mlx` does).
4. Register the module in `tests/conftest.py` `_MOCK_MODULES` so the CPU suite imports without the runtime.
5. Add the model to `models.lock.json` and the setup profile so `stark-translate setup` / `doctor` know about it.

## Adding a New Language

- STT: mlx-whisper is multilingual; Parakeet TDT v3 covers EN/ES/others but is only wired for EN dispatch. Add the `--lang` choice in `dry_run_ab.py` and a Marian direction in `factory._marian_direction_from_langs()` (only `en-es` / `es-en` have CT2 adapters today).
- Translation: Gemma 4 prompts take language names; TranslateGemma takes codes. Hindi/Chinese integration requires a later user decision. The [offline Hindi baseline](../docs/evaluation/overnight_hindi/README.md) is completed R&D with no live integration or approved references; EN↔ES remains the latency priority.
- TTS: add a Piper voice to `settings.tts.voices` and the setup `tts` profile.

## Related

- [`docs/current_architecture.md`](../docs/current_architecture.md) — pipeline contracts and schema 2 timing
- [`docs/mlx_cuda_parity.md`](../docs/mlx_cuda_parity.md) — MLX ↔ CUDA semantic parity checklist
- [`docs/packaging/models.md`](../docs/packaging/models.md) — model manifest, managed Marian CT2 artifacts
- [`docs/lite_profiles.md`](../docs/lite_profiles.md) — Lite profile contract, pinned artifacts, Mac CPU smoke evidence
- [`docs/backlog.json`](../docs/backlog.json) — current engine status (#176 closed completed; #177 closed not planned, off with promotion pending; Lite implemented / hardware pending)
