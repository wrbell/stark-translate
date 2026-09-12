# engines/AGENTS.md — STT + Translation + TTS (Agent Guide)

> Paired with [`CLAUDE.md`](./CLAUDE.md), which holds the module map, defaults table, env-var
> table and the stop-token / thread-safety rationale. This file is the checklist an agent must
> hold while editing `engines/`. Repo-wide constraints: [`../AGENTS.md`](../AGENTS.md).

## Hard constraints

- Factory returns **unloaded** engines; call `.load()` first. Heavy imports belong inside `load()` so the CPU test suite (with `tests/conftest.py` `_MOCK_MODULES`) imports cleanly. Never load a model in a unit test.
- **Gemma 4 stop tokens:** always run `ensure_stop_tokens(tokenizer, model_family=...)` after load; it *adds* `<turn|>` and preserves loader EOS ids. Never replace the set (the `{1, 3}` regression, #172) and never apply the TranslateGemma `<end_of_turn>` fix to Gemma 4.
- **Prompts:** Gemma 4 → `gemma4_user_content()` (plain instruct, thinking off); TranslateGemma → structured lang-code template. Both MLX and CUDA paths, and `workers.py`, must use `translation_prompts.py` — no inline prompts.
- **MLX threads:** thread-local streams (pinned mlx 0.32.2); `max_workers=2` overlap is the production path. Materialize weights and run the first Gemma forward on the load thread (`warm_mlx_model`). One generation lock per model.
- **PyTorch:** `MarianHFEngine` and Silero VAD share `_pytorch_lock`; VAD runs on the asyncio thread by default.
- **Quantization:** only OptiQ mixed-precision Gemma 4 repos; uniform 4-bit quants break PLE.
- **Downloads:** keep `resolve_model_path` purely offline for setup/preflight. Live MLX wrappers use `resolve_model_for_loading`: existing local copy or registered full-commit snapshot, never an uncached bare ID. Do not add unpinned live-path downloads ([pinning scope](../docs/evaluation/mac_followup_20260910/live-hf-pinning.md)).
- **Parakeet joint decode** is hash-pinned (`parakeet_joint_decode.py`). Do not edit the pins or the transform without re-running `tools/parakeet_joint_eval.py` (output-exact) and a paired identity screen; the engine must keep falling back to stock decode on drift. `tools/parakeet_profile.py` loads the engine with `joint_scalar_eval=False`.
- **Streaming / warm-up defaults** (`first_batch_size`, `STARK_WARMUP_AFTER_FINAL`) were merged on an identity screen; changing them is a behaviour change that needs a declared screen. `apply_wired_limit` stays opt-in.
- **Timing objects:** `ChunkTiming.relative_stages()` subtracts floats from every undeclared attribute; add new fields as declared dataclass fields (and exclude non-numeric ones), never ad-hoc attributes.
- Any change to defaults (models, thresholds, cadence, routing) goes through a declared screen plus human review; see [`../AGENTS.md`](../AGENTS.md).

## Current Mac defaults

Verify in `settings.py` / `factory.py` before citing; full table in
[`CLAUDE.md`](./CLAUDE.md#current-defaults-mac---backend-auto--mlx). EN Parakeet TDT v3 (MLX), ES
mlx-whisper large-v3-turbo, Marian CT2 int8 previews on CPU, Gemma 4 E4B OptiQ finals (E2B
opt-in), packaged Silero VAD, Piper TTS off, profile `standard`. `--mts` is rejected before load.
CUDA finals: `LlamaCppEngine` + `start_server.sh` (pin `b10883`). Lite profiles: see
[`CLAUDE.md`](./CLAUDE.md#deployment-profiles-stark_translateprofilespy).

## Env vars

Group prefixes are single-underscore (`STARK_TRANSLATE_MARIAN_BACKEND`, `STARK_STT_BACKEND`,
`STARK_VAD_BACKEND`, `STARK_TTS_OUTPUT_DEVICES`); the top-level settings object also accepts
`STARK_<GROUP>__<FIELD>`. Series-4 runtime switches (`STARK_WARMUP_AFTER_FINAL`,
`STARK_STREAM_FIRST_TOKEN_BATCH_SIZE`, `STARK_PARAKEET_JOINT_EVAL`, `STARK_MLX_WIRED_LIMIT`) and the
`STARK_EXPERIMENT_*` family are listed in
[`CLAUDE.md`](./CLAUDE.md#environment-variables-pydantic-settings-settingspy).

## Adding backends / languages

Follow the five steps in [`CLAUDE.md`](./CLAUDE.md#adding-a-new-engine): ABC → file →
`factory.py` branch → `tests/conftest.py` mock → `models.lock.json` + setup profile.
Hindi/Chinese remain pending user decisions; do not wire a new `--lang` without one.
Engine-related backlog items live in [`docs/backlog.json`](../docs/backlog.json).
