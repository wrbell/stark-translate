# engines/AGENTS.md — STT + Translation + TTS (Agent Guide)

> Paired with [`CLAUDE.md`](./CLAUDE.md) (imports this file for Claude Code) and with the
> reference guide [`docs/agents/engines.md`](../docs/agents/engines.md), which holds the module map, defaults table, env-var
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
[`docs/agents/engines.md`](../docs/agents/engines.md#current-defaults-mac---backend-auto--mlx). EN Parakeet TDT v3 (MLX), ES
mlx-whisper large-v3-turbo, Marian CT2 int8 previews on CPU, Gemma 4 E4B OptiQ finals (E2B
opt-in), packaged Silero VAD, Piper TTS off, profile `standard`. `--mts` is rejected before load.
CUDA finals: `LlamaCppEngine` + `start_server.sh` (pin `b10883`). Lite profiles: see
[`docs/agents/engines.md`](../docs/agents/engines.md#deployment-profiles-stark_translateprofilespy).

## Env vars

Group prefixes are single-underscore (`STARK_TRANSLATE_MARIAN_BACKEND`, `STARK_STT_BACKEND`,
`STARK_VAD_BACKEND`, `STARK_TTS_OUTPUT_DEVICES`); the top-level settings object also accepts
`STARK_<GROUP>__<FIELD>`. Series-4 runtime switches (`STARK_WARMUP_AFTER_FINAL`,
`STARK_STREAM_FIRST_TOKEN_BATCH_SIZE`, `STARK_PARAKEET_JOINT_EVAL`, `STARK_MLX_WIRED_LIMIT`) and the
`STARK_EXPERIMENT_*` family are listed in
[`docs/agents/engines.md`](../docs/agents/engines.md#environment-variables-pydantic-settings-settingspy).

## Adding backends / languages

Follow the five steps in [`docs/agents/engines.md`](../docs/agents/engines.md#adding-a-new-engine): ABC → file →
`factory.py` branch → `tests/conftest.py` mock → `models.lock.json` + setup profile.
Hindi/Chinese remain pending user decisions; do not wire a new `--lang` without one.
Engine-related backlog items live in [`docs/backlog.json`](../docs/backlog.json).

<!-- standards:begin -->
## Collection standards

Every project under `/Users/willem/Code` follows the shared standards in
`/Users/willem/Code/standards/` (index: `standards/STANDARDS.md`; future
standards: `standards/ROADMAP.md`).

- **Presentations:** build every deck from
  `standards/powerpoint template/Willem-Default.potx` (theme "Helena": Neue Haas
  Grotesk Text Pro, 16:9, teal/orange/red accent palette). Spec:
  `standards/powerpoint template/STANDARD.md`. Generate with
  `standards/powerpoint template/house_style.py` (open
  `Willem-Default-Base.pptx`, never the `.potx`) and gate with
  `standards/powerpoint template/deck_checks.py` before calling a deck done.
- **Deck rules:** no speaker notes in submitted decks; editable shapes, not
  chart images; numbered, linked superscript citations with a final References
  slide; no bottom rules, citation strips, or page counters; footer text only
  when a course or client requires it (for example `ME460 HWx`), which overrides
  the default of no footer; export the deliverable PDF with native PowerPoint
  and use LibreOffice renders only for QA.
- **Everything else:** do not invent facts, dates, or numbers; mark unknowns TBD
  and point at the source. Keep copyrighted course material out of git. This
  block is managed by `standards/tools/apply_standards.py`; edit
  `standards/ai-files/BLOCK-root.md`, not this copy.
- **AI use (school work):** no AI-generated or AI-modified images in any school
  deliverable; AI-written deliverable text only with written adviser
  pre-clearance (`docs/ai-clearances/`); never cite an AI tool as a source;
  never edit graded report text (the repo's `protected-paths.txt`; example:
  `standards/enforcement/senior-design-repo/sd-protected-paths.txt`). Log AI use
  in `docs/ai-use-log.md` and disclose it per
  `standards/standards/ai-use-disclosure/ai-use-disclosure.md`.
- **AI files:** one `AGENTS.md` (≤ 200 lines, Clarity verbatim); `CLAUDE.md` is
  `@AGENTS.md`. Gates: `standards/tools/agents_md_lint.py`, `ai_file_lint.py`.
<!-- standards:end -->
