# engines/ — STT + Translation + TTS Engine Layer

> Paired with [`AGENTS.md`](./AGENTS.md) (constraint checklist). Describes v2026.14 source on
> `main`; measured numbers live in [`README.md`](../README.md) § Measured performance and the
> evidence they cite, never here. Contracts: [`docs/current_architecture.md`](../docs/current_architecture.md).

Backend-agnostic ABCs with MLX (Apple Silicon), CUDA (NVIDIA) and CPU implementations.
`create_stt_engine()` / `create_translation_engine()` in [`factory.py`](./factory.py)
return **unloaded** engines — call `.load()` before `.transcribe()` / `.translate()`.

## Module map

| File | Role |
|------|------|
| `base.py` | `STTEngine`, `TranslationEngine`, `TTSEngine` ABCs; `STTResult` / `TranslationResult` dataclasses; `text_compression_ratio()` |
| `factory.py` | Hardware tier detection (`mlx` > `cuda` > `cpu`), STT backend selection, Marian CT2/HF routing, MLX Gemma model-id resolution |
| `mlx_engine.py` | `MLXWhisperEngine`, `MLXGemmaEngine` (Gemma 4 OptiQ or TranslateGemma; streaming with `first_batch_size`), `warm_mlx_model()`; re-exports `PiperTTSEngine` |
| `tts_engine.py` | `PiperTTSEngine` (ONNX Piper voices) |
| `parakeet_mlx_engine.py` | `ParakeetMLXEngine` — Parakeet TDT 0.6B v3 via `parakeet-mlx` (Mac EN default); installs the pinned joint decode at load |
| `parakeet_joint_decode.py` | `install_qualified_joint_decode()` — hash-pinned scalar joint decode (source and transformed-AST SHA-256); falls back to stock decode on drift; restored on unload |
| `parakeet_engine.py` | `ParakeetEngine` — NeMo Parakeet, EN-only CUDA accelerator; never the bilingual default |
| `mlx_memory.py` | `apply_wired_limit()` — opt-in Metal wired limit (`STARK_MLX_WIRED_LIMIT`); off by default because the series-4 identity screen showed no speed effect and a large peak-RSS increase |
| `prefix_cache.py` | `shared_token_prefix()`, `FixedPrefixStore` — fixed-prefix cache for the Gemma prompt (opt-in experiment) |
| `stt_fallback.py` | `require_mlx_fallback_language()` — the English-only fallback rule; Spanish never retries on an English-only model |
| `cuda_engine.py` | `FasterWhisperEngine` (CT2), `MarianCT2Engine`, `CUDAGemmaEngine`, `CUDAGemmaStreamingEngine` (HF NF4, legacy) |
| `llamacpp_engine.py` | `LlamaCppEngine` — Gemma 4 GGUF through `llama-server` (CUDA and Lite finals) |
| `marian_hf_engine.py` | `MarianHFEngine` — PyTorch fallback for partials, guarded by `_pytorch_lock` |
| `hf_whisper_engine.py` | `HFWhisperEngine` — transformers Whisper; only path with `torch.compile` / spec decode |
| `translation_prompts.py` | `build_chat_messages()`, `gemma4_user_content()`, `ensure_stop_tokens()`, Gemma 4 output cleanup |
| `model_paths.py` | Pure offline lookup (`resolve_model_path`, `resolve_marian_ct2`, `resolve_piper_voice`) plus explicit pinned acquisition at loading boundaries (`resolve_model_for_loading`) |
| `_locks.py`, `mlx_generation_lock.py` | Shared PyTorch lock; per-model MLX generation locks so distinct models overlap |
| `audio_devices.py` | Lazy output-device discovery and per-language TTS routing with hotplug re-resolution |
| `mlx_spec.py`, `spec_decode_logger.py` | Experimental mlx-optiq spec-decode wrapper and its telemetry (#177, offline only) |
| `active_learning.py` | JSONL logger for STT fallback events (quality layer 6) |
| `__init__.py` | Lazy `__getattr__` exports so importing the package loads no runtime |

Live Mac MLX wrappers resolve an existing explicit local override or a complete cached/downloaded
snapshot before calling upstream loaders. Unknown uncached remote IDs require a manifest revision
or an explicit local path; setup/preflight lookup stays offline. Pinning scope and the remaining
download sites: [live HF pinning](../docs/evaluation/mac_followup_20260910/live-hf-pinning.md).

## Current defaults (Mac, `--backend auto` → `mlx`)

| Role | Selection | Source |
|------|-----------|--------|
| STT EN | `ParakeetMLXEngine`, `mlx-community/parakeet-tdt-0.6b-v3` | `dry_run_ab.py` and `operator_app/preflight.py` pick `parakeet-mlx` for `--lang en` when available; `--stt-backend mlx` forces Whisper |
| STT ES | `MLXWhisperEngine`, `mlx-community/whisper-large-v3-turbo`; Spanish never retries the English-only Distil fallback | `settings.stt` |
| Partials | `MarianCT2Engine` from the int8 artifact in `adapters/marian_ct2/<dir>/active` or the managed setup cache (`intra_threads=4`, `marian_compute_type` default `int8_float16`, resolved by CT2 on CPU); `MarianHFEngine` if no CT2 artifact | `factory.create_translation_engine(engine_type="marian")` |
| Finals | `MLXGemmaEngine`, `mlx-community/gemma-4-e4b-it-OptiQ-4bit`; `--gemma4-size e2b` → E2B OptiQ; `routing_policy` (`legacy` default) may send short high-confidence finals to Marian | `settings.translation.model_family = "gemma4"` |
| Opt-out | `--model-family translategemma` → `mlx-community/translategemma-4b-it-4bit` (+12B with `--ab`) | `resolve_mlx_translation_model_id()` |
| TTS | `PiperTTSEngine`, voices `en_US-lessac-high` / `es_MX-claude-high` (`settings.tts.voices`) | `create_tts_engine()` |
| VAD | Packaged Silero 6.2.1 through `tools/vad_runtime.py` (no Torch Hub); ONNX via `--vad-backend onnx` | `settings.vad.backend` |

Naïve uniform 4-bit `mlx-community` Gemma 4 quants are known-broken (PLE layers quantized); only
the OptiQ mixed-precision repos are supported.

**CUDA (`--backend cuda`):** `FasterWhisperEngine` prefers `adapters/whisper_turbo_ct2/active` (W16
fine-tune) when present, else `large-v3-turbo`; `MarianCT2Engine` int8_float16; finals through
`LlamaCppEngine` when `start_server.sh` is running (`--engine auto` probes `http://127.0.0.1:8090`).
HF NF4 Gemma 4 is not recommended on 16 GB cards — PLE embeddings stay bf16; see
[`docs/archive/v2026.5/BENCHMARK.md`](../docs/archive/v2026.5/BENCHMARK.md).

## Gemma 4 stop tokens and prompts (do not regress #172)

`ensure_stop_tokens(tokenizer, model_family=...)` runs after every load. It **adds** the family's
stop tokens to `tokenizer._eos_token_ids` and preserves every EOS id the loader supplied. For
Gemma 4 that means `<eos>` (id 1), `<turn|>` (id 106) and the `<|tool_response>` id (50); the
historical TranslateGemma fix of adding `<end_of_turn>` (id 106 in the Gemma 3 vocabulary) must
not be applied to Gemma 4, and replacing the set with `{1, 3}` truncates output. Gemma 4 receives
the plain instruct prompt from `gemma4_user_content()` with thinking disabled; TranslateGemma
receives the structured `source_lang_code` / `target_lang_code` template. The `--multiprocess`
worker in [`workers.py`](../workers.py) goes through `MLXGemmaEngine` with the parent-selected
`model_family`, so both paths share these rules (#176).

## Streaming and warm-up (series 4, on by default)

- `translate_streaming(..., batch_size=3, first_batch_size=1)`: the first callback fires at token 1
  (`STARK_STREAM_FIRST_TOKEN_BATCH_SIZE`, default 1), later batches every 3 tokens. The pipeline
  gives the first `translation_stream` batch a deterministic event id so displays can acknowledge
  it (`first_stream`, see [`displays/CLAUDE.md`](../displays/CLAUDE.md)).
- The pipeline re-arms its idle warm-up right after each final's translation finishes
  (`STARK_WARMUP_AFTER_FINAL`, default on) instead of after `process_final` is submitted.
- Parakeet loads with the hash-pinned scalar joint decode (`STARK_PARAKEET_JOINT_EVAL`, default
  on); on a hash mismatch it logs and keeps stock decode. Re-qualify with
  `tools/parakeet_joint_eval.py` before changing the pins.
- All three were merged on the paired identity screen in
  [series 4 P1](../docs/evaluation/series4_20260912/P1-runtime/README.md) (byte-identical output).

## MLX thread safety

- **Thread-local streams (mlx ≥ 0.31.2; pinned 0.32.2 / mlx-lm 0.31.3):** `ThreadPoolExecutor(max_workers=2)` overlaps STT(N+1) with translation(N) in one process. `--multiprocess` (separate Metal contexts) remains an escape hatch only.
- **Materialize on the load thread:** `warm_mlx_model()` runs `mx.eval` on weights and the first Gemma forward on the loading thread before pool handoff; skipping it reproduces the first-forward crash covered by `tests_gpu/test_mlx_worker_first_forward.py`.
- **One lock per model:** `mlx_generation_lock.py` serializes generation on a single model while distinct models overlap.
- **PyTorch lock:** `MarianHFEngine` and the Silero VAD share `_pytorch_lock`; concurrent PyTorch forwards from different threads corrupt the Metal heap. Default VAD remains on the asyncio thread; serialized worker VAD is an opt-in experiment.
- **Metal cache:** engines set `mx.set_cache_limit(256 MB)` (`cache_limit_mb`); `STARK_EXPERIMENT_MLX_CACHE_MB` overrides it process-wide. mlx-lm sets the wired limit per generation call itself, which is why `apply_wired_limit` at load is opt-in.

## Confidence and fallback (quality layer 3)

`MLXWhisperEngine` defaults: `fallback_threshold=-1.2` (avg_logprob below → retry with the
fallback model), `hallucination_threshold=2.4` (compression ratio above → retry),
`fallback_on_low_conf=True`. Automatic fallback is English-only (`stt_fallback.py`). Spanish keeps
the primary result on low confidence and fails visibly if its selected multilingual model cannot
load. Engine callers can pass `session_language="es"` to guard startup; Spanish inference also
rejects an engine that already selected an English fallback. Custom fallback IDs do not establish
multilingual support. Fallback events are logged by `active_learning.py`. `ParakeetMLXEngine`
derives `avg_logprob` / word probabilities from TDT token confidences — a proxy, not a calibrated
Whisper probability — and does not use the Whisper fallback chain.

## Speculative decoding — status

- **Whisper HF spec decode:** requires an explicit, verified draft. distil-large-v3.5 → whisper-large-v3-turbo is incompatible ([`docs/archive/v2026.5/spec_decode_research.md`](../docs/archive/v2026.5/spec_decode_research.md)). Off by default.
- **TranslateGemma 4B → 12B drafting (`--ab --num-draft-tokens`):** legacy A/B path only.
- **Gemma 4 assistant drafter (`--mts`, #177):** **off, and rejected before any model loads.** `dry_run_ab.py` `validate_live_mts()` raises `LIVE_MTS_UNAVAILABLE` when `--mts` or `STARK_TRANSLATE_MLX_MTS` is set; the live pipeline never continues silently after a drafter failure. The working offline probe is `engines/mlx_spec.py` over mlx-optiq (`tools/mts_acceptance_probe.py`). Results: [`docs/archive/v2026.13/MAC_LATENCY.md`](../docs/archive/v2026.13/MAC_LATENCY.md), [`docs/mlx_mtp_notes.md`](../docs/mlx_mtp_notes.md).
- **E2B full-model draft for E4B (`STARK_EXPERIMENT_DRAFT_MODEL_ID`):** rejected in the live pipeline (STT starvation) — [follow-up tail screen](../docs/evaluation/followup_20260911/X-tail-screen/README.md).
- **CUDA MTP:** `start_server.sh --mtp` opt-in on llama.cpp `b10883`; unbenchmarked on hardware ([`docs/cuda_latency_proposal.md`](../docs/cuda_latency_proposal.md)).
- **Latency experiments** (`tools/latency_experiments.py`, `tools/latency_scheduler.py`, `tools/incremental_stt.py`, `tools/preview_candidates.py`): opt-in research controls validated before session startup, never promoted by selecting a backend; every screened arm and its outcome is in [`docs/latency_next_experiments.md`](../docs/latency_next_experiments.md).

## Deployment profiles (`stark_translate/profiles.py`)

`apply_profile()` runs once after CLI overrides and before model loads. `standard` leaves the
factory's platform selection alone. The Lite profiles (`lite-cpu`, `lite-cpu-quality`,
`lite-cuda-8gb`) force: `stt.backend = faster-whisper` with the pinned CT2 model (`whisper-small`
int8 on CPU; `whisper-large-v3-turbo` int8_float16 on CUDA), 3 CT2 threads and one STT worker,
`local_files_only`, no low-confidence fallback model; `vad.backend = onnx`; `marian_backend = ct2`
on CPU int8 with one intra-thread; `model_family = gemma4` with E2B; finals via `MarianCT2Engine`
(`lite-cpu`) or `LlamaCppEngine` against a session-owned `llama-server` started by
`tools/llama_runtime.py` (`lite-cpu-quality`, `lite-cuda-8gb`); `run_ab`, `multiprocess`,
`use_speculative` and `mlx_mts` off. Lite never inherits the standard path's W16 CT2 adapter
preference. E2B or native-runtime failures fail the session; no HF NF4 fallback. Admission floors
and evidence: [`docs/lite_profiles.md`](../docs/lite_profiles.md).

## Environment variables (pydantic-settings, `settings.py`)

Nested groups use the group prefix directly (`STARK_STT_`, `STARK_TRANSLATE_`, `STARK_VAD_`,
`STARK_TTS_`, `STARK_CUDA_`); the top-level `PipelineSettings` also accepts
`STARK_<GROUP>__<FIELD>` with the double-underscore delimiter. Common keys:

| Variable | Effect |
|----------|--------|
| `STARK_STT_BACKEND` / `--stt-backend` | `auto`, `mlx`, `parakeet-mlx`, `faster-whisper`, `hf`, `parakeet` |
| `STARK_TRANSLATE_MODEL_FAMILY` | `gemma4` (default) or `translategemma` |
| `STARK_TRANSLATE_MARIAN_BACKEND` | `auto` (CT2 when artifact present), `ct2` (strict), `hf` |
| `STARK_TRANSLATE_ROUTING_POLICY` / `--routing-policy` | `legacy` (default), `conservative`, `off` — when a short, high-confidence final may take the Marian route |
| `STARK_TRANSLATE_MLX_MTS` | Drafter flag (#177); when set, `validate_live_mts` aborts the live session before load — leave unset |
| `STARK_PROFILE` / `--profile` | `standard` (default), `lite-cpu`, `lite-cpu-quality`, `lite-cuda-8gb`; `stark-translate-lite` defaults to `lite-cpu` |
| `STARK_VAD_BACKEND` | `torch` (default) or `onnx` |
| `STARK_TTS_OUTPUT_DEVICES` | Per-language local TTS output map (index, name substring or null) |
| `STARK_MODELS_DIR` | Model cache used by setup and inference (`engines/model_paths.py`) |
| `STARK_WARMUP_AFTER_FINAL` | Keep-warm re-armed after each final's translation (default on; `0` disables) |
| `STARK_STREAM_FIRST_TOKEN_BATCH_SIZE` | Tokens before the first stream callback (default 1; later batches every 3) |
| `STARK_PARAKEET_JOINT_EVAL` | Hash-pinned Parakeet joint decode at load (default on; `0` keeps stock decode) |
| `STARK_MLX_WIRED_LIMIT` | Opt-in Metal wired limit at load (`1`/`true`/`on`); rejected as a default for peak RSS |
| `STARK_EXPERIMENT_*` | Research controls in `tools/latency_experiments.py` (`serial_finals`, `partial_recheck_translation`, `partial_reuse_ms`, `partial_reuse_keep_confidence`, `mlx_cache_mb`, `gemma_prefix_cache`, `draft_model_id` / `draft_tokens`, `trace`); validated before startup; all closed or opt-in arms |

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

- STT: mlx-whisper is multilingual; Parakeet TDT v3 is the English automatic default, with explicit Spanish selection available for experiments. Additional languages need a `--lang` choice in `dry_run_ab.py` and a Marian direction in `factory._marian_direction_from_langs()` (only `en-es` / `es-en` have CT2 adapters today).
- Translation: Gemma 4 prompts take language names; TranslateGemma takes codes. Hindi/Chinese integration requires a user decision. The [offline Hindi baseline](../docs/evaluation/overnight_hindi/README.md) is completed R&D with no live integration.
- TTS: add a Piper voice to `settings.tts.voices` and the setup `tts` profile.

## Related

- [`docs/current_architecture.md`](../docs/current_architecture.md) — pipeline contracts and schema 2 timing
- [`docs/mlx_cuda_parity.md`](../docs/mlx_cuda_parity.md) — MLX ↔ CUDA semantic parity checklist
- [`docs/packaging/models.md`](../docs/packaging/models.md) — model manifest, managed Marian CT2 artifacts
- [`docs/lite_profiles.md`](../docs/lite_profiles.md) — Lite profile contract, pinned artifacts, Mac CPU smoke evidence
- [`docs/latency_next_experiments.md`](../docs/latency_next_experiments.md) — closed-arm registry
- [`docs/backlog.json`](../docs/backlog.json) — engine-related items (`issue-135-mac-ab`, `issue-177-mtp`, `lite-cpu-inference`, `rtx2070-native-validation`, `issue-138-hindi-zero-shot`)
