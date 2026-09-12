# AGENTS.md — Short agent guide

Read this before touching the repo as a coding agent (Codex, Grok, Claude). The full developer
guide is [`CLAUDE.md`](CLAUDE.md); the product overview, measured performance and quick start are in
[`README.md`](README.md); every document is listed in [`docs/README.md`](docs/README.md).
Current source is v2026.14 (last release v2026.14.0.0, 2026-09-11). Contracts:
[`docs/current_architecture.md`](docs/current_architecture.md) · evidence:
[`docs/mac_implementation_status.md`](docs/mac_implementation_status.md) · remaining work:
[`docs/backlog.json`](docs/backlog.json).

## Standing constraints

- **No microphone, no speakers, unattended.** Use `--audio-file` replay and `--dry-run-text`.
  Live-microphone and playback checks are attended sessions the owner runs.
- **Interpreters.** `.stark-python` → `venv/bin/python` (Torch 2.13.0, runtime only: no
  pytest/ruff/mypy). `stt_env` (Torch 2.10) is the rollback: never install into, upgrade or
  recreate it (`stt_env/bin/python -m pip freeze | shasum -a 256` starts `a09be842`). Run checks
  with `stt_env/bin/python -m pytest|ruff|mypy` (read-only use). Do not load models in unit tests.
- **Models.** Google Gemma 4 (and its official drafters/QAT), NVIDIA Parakeet, OpenAI Whisper,
  Helsinki-NLP opus-mt, mlx-community re-quantizations of those. No Chinese-origin models. Pinned
  revisions in `models.lock.json`; no unpinned live-path downloads.
- **Defaults never change by a screen alone.** E4B OptiQ finals, 0.5 s silence, 0.6 s cadence,
  Parakeet EN / mlx-whisper turbo ES, Marian CT2 previews change only after a declared screen passes
  its gates and a human reviews. Rejected arms are never re-run as confirmations or combined
  ([registry](docs/latency_next_experiments.md)). Output-identical engineering changes merge on a
  paired identity screen.
- **Evidence.** `docs/evaluation/**` is immutable once written; corrections are dated appends.
  Never fabricate references, labels or numbers; cite session ids and files. Numbers stay in
  evidence documents (the README quotes and links them; guides link only).
- **One inference process at a time** on the Mac GPU.
- **Guides** keep durable rules and links: one state line, no per-PR banners, no session-scoped
  statements. Dated narrative goes in the boards under `docs/evaluation/`.

## Checks

```bash
ruff check . && ruff format --check .
mypy engines/ settings.py
pytest tests/ -v --cov=engines --cov=tools --cov=features --cov-fail-under=65
python tools/render_backlog.py validate && python tools/render_backlog.py render --check
python tools/render_backlog.py check-links
pytest tests/test_documentation.py -v
```

10 GitHub Actions workflow files in `.github/workflows/`; required checks for merge: `lint`,
`test (3.11)`, `test (3.12)`, `security`. Lint also runs bandit and HTML Tidy on `displays/*.html`.
`tests/test_documentation.py` checks the workflow count, required links and forbidden stale claims
in the guides.

## Working in this repo

- One branch or worktree per task (`git worktree add ../SRTranslate-wt-<task> -b <branch> main`);
  never work on `main`. Delegated agents do not commit; the orchestrator reviews, commits and merges.
- Commitlint: header ≤ 100 chars, conventional type, lower-case subject
  (`feat(mlx): gemma …`, not `Gemma …`).
- Branch protection: squash-merge with auto-merge and branch deletion; `gh pr update-branch` when
  BEHIND; after every merge verify the previous PRs' hunks are still on `main`.
- Rebuild a branch with `git reset --hard origin/main && git cherry-pick <sha>` (never
  `reset --soft`); check `git diff --stat origin/main` before a force-push.
- Published tags never move. Evidence under `docs/evaluation/**` gets dated appends only.
- Specs for delegated work name the interpreter, the test commands, exact file seams, the files
  that are off-limits, and "do not install packages or load models".

## Current Mac defaults (verify in `settings.py` / `engines/factory.py` before citing)

| Role | Engine / model |
|------|----------------|
| STT EN / ES | `ParakeetMLXEngine` `mlx-community/parakeet-tdt-0.6b-v3` / `MLXWhisperEngine` `mlx-community/whisper-large-v3-turbo` |
| Previews | `MarianCT2Engine` int8 on CPU (adapter dir or managed cache); `MarianHFEngine` fallback |
| Finals | `MLXGemmaEngine` `mlx-community/gemma-4-e4b-it-OptiQ-4bit`; E2B via `--gemma4-size e2b` |
| VAD / TTS | Packaged Silero 6.2.1 (`tools/vad_runtime.py`, ONNX opt-in) / Piper, off by default |
| Profile | `standard`; `lite-cpu`, `lite-cpu-quality`, `lite-cuda-8gb` via `--profile` or `STARK_PROFILE` |

`--mts` is rejected before any model loads; `STARK_EXPERIMENT_*` controls are opt-in research arms.

## Where things are

| Directory | Guide pair |
|-----------|-----------|
| `engines/` — STT, translation, TTS engines | [`engines/AGENTS.md`](engines/AGENTS.md) · [`engines/CLAUDE.md`](engines/CLAUDE.md) |
| `tools/` — evaluation, screens, review, adapters | [`tools/AGENTS.md`](tools/AGENTS.md) · [`tools/CLAUDE.md`](tools/CLAUDE.md) |
| `displays/` — displays, protocol, operator page | [`displays/AGENTS.md`](displays/AGENTS.md) · [`displays/CLAUDE.md`](displays/CLAUDE.md) |
| `features/` — diarization, verses, summary | [`features/AGENTS.md`](features/AGENTS.md) · [`features/CLAUDE.md`](features/CLAUDE.md) |
| `training/` — WSL training and data | [`training/AGENTS.md`](training/AGENTS.md) · [`training/CLAUDE.md`](training/CLAUDE.md) |

Machine guides: [`CLAUDE-macbook.md`](CLAUDE-macbook.md) (inference, screens, troubleshooting),
[`CLAUDE-windows.md`](CLAUDE-windows.md) (WSL training, native Lite). Latest boards:
[series 3](docs/evaluation/series3_20260912/STATUS.md), [series 4](docs/evaluation/series4_20260912/STATUS.md).
Runbooks: [`docs/operator_runbook.md`](docs/operator_runbook.md),
[`docs/packaging/macos.md`](docs/packaging/macos.md), [`docs/lite_profiles.md`](docs/lite_profiles.md).

## Extension patterns

- New engine → `engines/CLAUDE.md` § Adding a New Engine (ABC → file → `factory.py` branch →
  `tests/conftest.py` mock → `models.lock.json` + setup profile)
- New language → `engines/CLAUDE.md` + `training/CLAUDE.md`; Hindi/Chinese need a user decision first
- New display → `displays/CLAUDE.md` § Adding a display
- Adapter deploy / active learning → `tools/CLAUDE.md`
- New deployment profile → `stark_translate/profiles.py` + `operator_app/lite_preflight.py` + `models.lock.json`
- New latency experiment → `tools/latency_experiments.py`, protocol declared before run 1, gates from
  `tools/tail_screen_report.py`, outcome row in `docs/latency_next_experiments.md`
