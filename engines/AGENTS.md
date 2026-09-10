# engines/AGENTS.md — STT + Translation + TTS (Agent Guide)

> Paired with [`CLAUDE.md`](./CLAUDE.md), which holds the module map, defaults table,
> env-var table and the full stop-token/thread-safety rationale. This file is the
> constraint checklist an agent must hold while editing `engines/`.

## Hard constraints

- Factory returns **unloaded** engines; call `.load()` first. Heavy imports belong inside `load()` so the CPU test suite (with `tests/conftest.py` `_MOCK_MODULES`) imports cleanly.
- **Gemma 4 stop tokens:** always run `ensure_stop_tokens(tokenizer, model_family=...)` after load; it *adds* `<turn|>` and preserves loader EOS ids. Never replace the set (the `{1, 3}` regression, #172) and never apply the TranslateGemma `<end_of_turn>` fix to Gemma 4.
- **Prompts:** Gemma 4 → `gemma4_user_content()` (plain instruct, thinking off); TranslateGemma → structured lang-code template. Both MLX and CUDA paths, and `workers.py`, must use `translation_prompts.py` — no inline prompts.
- **MLX threads:** mlx ≥ 0.31.2 thread-local streams; `max_workers=2` overlap is the production path. Materialize weights and run the first Gemma forward on the load thread (`warm_mlx_model`). One generation lock per model.
- **PyTorch:** `MarianHFEngine` and Silero VAD share `_pytorch_lock`; VAD runs on the asyncio thread by default. The experimental worker path remains opt-in and uses the same lock.
- **Quantization:** only OptiQ mixed-precision Gemma 4 repos; uniform 4-bit quants break PLE.
- **Downloads:** keep `resolve_model_path` purely offline for setup/preflight. Live MLX wrappers use `resolve_model_for_loading`: existing local copy or registered full-commit snapshot, never an uncached bare ID. Model identity and explicit local overrides remain in provenance. Do not add unpinned live-path downloads; [pinning scope and residual inventory](../docs/evaluation/mac_followup_20260910/live-hf-pinning.md) retain the B615 limitations.
- Do not recreate `stt_env`; do not run model loads in unit tests.

## Current Mac defaults (verify in `settings.py` / `factory.py` before citing)

| Role | Engine / model |
|------|----------------|
| STT EN | `ParakeetMLXEngine` — `mlx-community/parakeet-tdt-0.6b-v3` (auto for `--lang en`) |
| STT ES | `MLXWhisperEngine` — `mlx-community/whisper-large-v3-turbo` |
| Partials | `MarianCT2Engine` int8 CPU (adapter dir or managed cache); `MarianHFEngine` fallback |
| Finals | `MLXGemmaEngine` — `mlx-community/gemma-4-e4b-it-OptiQ-4bit`; E2B via `--gemma4-size e2b` |
| TTS | `PiperTTSEngine` — `en_US-lessac-high`, `es_MX-claude-high` |
| VAD | packaged Silero 6.2.1 (`tools/vad_runtime.py`); ONNX opt-in |
| MTP / `--mts` | **off** (#177) — `validate_live_mts` rejects it before any model loads; `--no-mts` is the explicit off switch |
| Profile | `standard` default; `lite-cpu` / `lite-cpu-quality` / `lite-cuda-8gb` via `--profile` or `STARK_PROFILE` (`stark_translate/profiles.py`) |

CUDA finals: `LlamaCppEngine` + `start_server.sh` (`--no-draft` default, `--mtp` opt-in, pin `b10883`).
W16 Whisper CT2 auto-preferred at `adapters/whisper_turbo_ct2/active` on the standard path only.

Lite profiles force faster-whisper CT2 (`whisper-small` int8 CPU or turbo int8_float16 CUDA),
ONNX Silero, Marian CT2 int8 partials, and finals from Marian (`lite-cpu`) or Gemma 4 **E2B**
Q4_K_M through a session-owned `llama-server` (`tools/llama_runtime.py`); A/B, drafting,
multiprocess, STT fallback models and live diarization are off. Contract and evidence:
[`docs/lite_profiles.md`](../docs/lite_profiles.md).

## Env vars

Group prefixes are single-underscore (`STARK_TRANSLATE_MARIAN_BACKEND`,
`STARK_STT_BACKEND`, `STARK_VAD_BACKEND`, `STARK_TTS_OUTPUT_DEVICES`); the
top-level settings object also accepts `STARK_<GROUP>__<FIELD>`. Full table in
[`CLAUDE.md`](./CLAUDE.md#environment-variables-pydantic-settings-settingspy).

## Status of engine-related backlog items

| Item | State |
|------|-------|
| #176 `--multiprocess` shared prompts/stop rules | merged in `workers.py` (wraps `MLXGemmaEngine`); closed completed ([actual closeout](../docs/evaluation/overnight_closeout_20260910/README.md)) |
| #177 Gemma 4 assistant drafter | closed not planned; deferred with promotion pending. Off; live `--mts` rejected before load; `engines/mlx_spec.py` offline probe only |
| Lite CPU profile / RTX 2070 | **implemented and integrated**; isolated Mac CPU synthetic EN+ES caption/TTS and optional CPU E2B inference smokes passed; x86 CPU, native Windows and 2070 performance **pending hardware** ([evidence](../docs/lite_profiles.md)) |
| W16 + v2-cpo Mac A/B (#135) | pending artifact transfer from WSL |
| Hindi (#138) | [Offline church-audio baseline](../docs/evaluation/overnight_hindi/README.md) completed as separate R&D; no live integration or further EN↔ES-program work; human review/language decision pending |

Canonical list: [`docs/backlog.json`](../docs/backlog.json).

## Adding backends / languages

Follow the five steps in [`CLAUDE.md`](./CLAUDE.md#adding-a-new-engine): ABC → file →
`factory.py` branch → `tests/conftest.py` mock → `models.lock.json` + setup profile.
Hindi/Chinese remain pending user decisions; do not wire a new `--lang` without one.
