# engines/AGENTS.md — STT + Translation + TTS (Agent Guide)

> Paired with [`CLAUDE.md`](./CLAUDE.md), which holds the module map, defaults table,
> env-var table and the full stop-token/thread-safety rationale. This file is the
> constraint checklist an agent must hold while editing `engines/`.

## Hard constraints

- Factory returns **unloaded** engines; call `.load()` first. Heavy imports belong inside `load()` so the CPU test suite (with `tests/conftest.py` `_MOCK_MODULES`) imports cleanly.
- **Gemma 4 stop tokens:** always run `ensure_stop_tokens(tokenizer, model_family=...)` after load; it *adds* `<turn|>` and preserves loader EOS ids. Never replace the set (the `{1, 3}` regression, #172) and never apply the TranslateGemma `<end_of_turn>` fix to Gemma 4.
- **Prompts:** Gemma 4 → `gemma4_user_content()` (plain instruct, thinking off); TranslateGemma → structured lang-code template. Both MLX and CUDA paths, and `workers.py`, must use `translation_prompts.py` — no inline prompts.
- **MLX threads:** mlx ≥ 0.31.2 thread-local streams; `max_workers=2` overlap is the production path. Materialize weights and run the first Gemma forward on the load thread (`warm_mlx_model`). One generation lock per model.
- **PyTorch:** `MarianHFEngine` and Silero VAD share `_pytorch_lock`; VAD stays on the asyncio thread.
- **Quantization:** only OptiQ mixed-precision Gemma 4 repos; uniform 4-bit quants break PLE.
- **Downloads:** identity is an HF id, resolution is local (`model_paths.py`, `STARK_MODELS_DIR`, `models.lock.json`). Do not add unpinned live-path downloads (B615 followups are tracked in the backlog).
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
| MTP / `--mts` | **off**, experimental (#177) — never loads through mlx-lm |

CUDA finals: `LlamaCppEngine` + `start_server.sh` (`--no-draft` default, `--mtp` opt-in, pin `b10883`).
W16 Whisper CT2 auto-preferred at `adapters/whisper_turbo_ct2/active`.

## Env vars

Group prefixes are single-underscore (`STARK_TRANSLATE_MARIAN_BACKEND`,
`STARK_STT_BACKEND`, `STARK_VAD_BACKEND`, `STARK_TTS_OUTPUT_DEVICES`); the
top-level settings object also accepts `STARK_<GROUP>__<FIELD>`. Full table in
[`CLAUDE.md`](./CLAUDE.md#environment-variables-pydantic-settings-settingspy).

## Status of engine-related backlog items

| Item | State |
|------|-------|
| #176 `--multiprocess` shared prompts/stop rules | implemented in `workers.py` (wraps `MLXGemmaEngine`); root closes after merge |
| #177 Gemma 4 assistant drafter | experimental, off; `engines/mlx_spec.py` probe only |
| Lite CPU profile / RTX 2070 | implementation in progress in the lite worktree; certification on hardware pending |
| W16 + v2-cpo Mac A/B (#135) | pending artifact transfer from WSL |

Canonical list: [`docs/backlog.json`](../docs/backlog.json).

## Adding backends / languages

Follow the five steps in [`CLAUDE.md`](./CLAUDE.md#adding-a-new-engine): ABC → file →
`factory.py` branch → `tests/conftest.py` mock → `models.lock.json` + setup profile.
Hindi/Chinese remain pending user decisions; do not wire a new `--lang` without one.
