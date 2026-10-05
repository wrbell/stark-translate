# AGENTS.md

## Project

On-device live EN↔ES speech-to-text and translation for Stark Road Gospel Hall
(Farmington Hills, MI). Python 3.11; MLX on Apple Silicon for inference,
CUDA/WSL for training. Release line v2026.14 (exact version in `pyproject.toml`,
history in [`CHANGELOG.md`](CHANGELOG.md)). Overview and quick start:
[`README.md`](README.md). Contracts:
[`docs/current_architecture.md`](docs/current_architecture.md). Evidence:
[`docs/mac_implementation_status.md`](docs/mac_implementation_status.md).
Remaining work: [`docs/backlog.json`](docs/backlog.json). Every document:
[`docs/README.md`](docs/README.md). Developer guide (pipeline, evidence map,
phases): [`docs/agents/project-guide.md`](docs/agents/project-guide.md).

## Commands

Run checks with `stt_env/bin/python -m …` (read-only); `venv` is runtime only.

- Setup (Mac): `venv/bin/stark-translate setup --backend mlx` (all steps: README
  § Quick start)
- Install check: `venv/bin/stark-translate doctor --backend mlx --lang en`
- Lint: `ruff check . && ruff format --check .`
- Types: `mypy engines/ settings.py`
- Test: `pytest tests/ -v --cov=engines --cov=tools --cov=features
  --cov-fail-under=65`
- Backlog: `python tools/render_backlog.py validate`; then the same script with
  `render --check` and with `check-links`
- Guide contract: `pytest tests/test_documentation.py -v`
- Standards: `python3 tools/agents_md_lint.py`, `python3 tools/check_docs.py`,
  `pre-commit run --files <changed files>`

11 GitHub Actions workflow files in `.github/workflows/`; required checks for
merge: `lint`, `test (3.11)`, `test (3.12)`, `security`. Lint also runs bandit
and HTML Tidy on `displays/*.html`. `standards.yml` (inlined from
`wrbell/standards`) is not a required check.

## Code style

Follow the config files. Do not paste a style guide into this file.

- Python: `pyproject.toml` (`[tool.ruff]`, `[tool.mypy]`)
- Editor defaults: `.editorconfig`
- Markdown: `enforcement/markdownlint/.markdownlint-cli2.yaml`
- Spelling: `enforcement/cspell/cspell.json`
- YAML: `enforcement/yamllint/.yamllint.yml`
- Shell: `enforcement/shellcheck/.shellcheckrc`
- Secrets: `enforcement/gitleaks/.gitleaks.toml`

`enforcement/`, `tools/agents_md_lint.py` and `tools/check_docs.py` are
byte-identical copies from the private repository `wrbell/standards`, which holds
the collection standards (source commit in `.standards.json`); do not edit them
here. When a standards file and this file disagree, this file wins. Say that in
the pull request body.

## Tests

Give each task a check you can run. Write or update the failing test first. Show
that the test fails, then make it pass. A test checks correctness. It does not
define the solution. Do not hard-code a value or a special case to pass a test.
If a test is wrong, say so. For numeric code, keep the simple correct version as
a reference test. Then optimize only while that test passes. Do not load models
in unit tests; `tests/conftest.py` mocks the heavy modules.

## Standing constraints

These apply to every session, attended or not.

- **No microphone, no speakers, unattended.** Automation uses `--audio-file`
  replay and `--dry-run-text`. Live-microphone and playback checks are attended
  sessions Willem runs.
- **Interpreters.** `.stark-python` points at `venv/bin/python` (Torch 2.13.0,
  runtime only: no pytest/ruff/mypy). `stt_env` (Torch 2.10) is the rollback
  environment: never install into, upgrade or recreate it (`stt_env/bin/python
  -m pip freeze | shasum -a 256` starts `a09be842`).
- **Models.** Google Gemma 4 (and its official drafters/QAT), NVIDIA Parakeet,
  OpenAI Whisper, Helsinki-NLP opus-mt, and mlx-community re-quantizations of
  those. No Chinese-origin models. Revisions are pinned in `models.lock.json`;
  no unpinned live-path downloads.
- **Defaults never change by a screen alone.** E4B OptiQ finals, 0.5 s silence,
  0.6 s cadence, Parakeet EN / mlx-whisper turbo ES, Marian CT2 previews change
  only after a declared screen passes its gates and a human reviews. Rejected
  arms are never re-run as confirmations or combined
  ([registry](docs/latency_next_experiments.md)). Output-identical engineering
  changes merge on a paired identity screen.
- **Evidence.** `docs/evaluation/**` is immutable once written; corrections are
  dated appends. Never fabricate references, labels or numbers; cite session ids
  and files. Numbers live in the evidence documents; the README's performance
  section quotes and links them, guides link only.
- **One inference process at a time** on the Mac GPU.
- **Guides** keep durable rules and links: one state line, no per-PR banners, no
  session-scoped statements, no dated narrative (that belongs in the boards
  under `docs/evaluation/`).

## Pull requests and commits

- Use a Conventional Commit subject. Commitlint: header ≤ 100 chars, lower-case
  subject (`feat(mlx): gemma …`, not `Gemma …`).
- One branch or worktree per task (`git worktree add ../SRTranslate-wt-<task> -b
  <branch> main`); never work on `main`. Delegated agents do not commit; the
  orchestrator reviews and commits.
- Open the pull request as a draft. Wait for CI to pass. List each assumption
  and each open tradeoff in the pull request body. A new dependency needs a
  reason in the pull request.
- Keep each change near 100 lines. Do not edit files outside the task.
- Specs for delegated work name the interpreter, the test commands, exact file
  seams, the files that are off-limits, and "do not install packages or load
  models".
- Do not force-push, rewrite history, delete a branch, merge, or publish unless
  Willem asks. When he does: squash-merge with auto-merge and branch deletion;
  `gh pr update-branch` when BEHIND; after every merge verify the previous PRs'
  hunks are still on `main`. Rebuild a branch with `git reset --hard origin/main
  && git cherry-pick <sha>` (never `reset --soft`) and check `git diff --stat
  origin/main` before a force-push. Published tags never move.

## Security

Do not commit a secret. Do not put a secret in a prompt or a log. Do not use a
production credential. Use a test credential with a budget limit. Run unattended
mode only in a sandbox.

## Where things are

Each folder keeps its own `AGENTS.md` (its `CLAUDE.md` imports it) and a
reference guide under `docs/agents/`.

| Area | Rules | Reference |
| --- | --- | --- |
| `engines/` — STT, translation, TTS engines | [`engines/AGENTS.md`](engines/AGENTS.md) | [`docs/agents/engines.md`](docs/agents/engines.md) |
| `tools/` — evaluation, screens, review, adapters | [`tools/AGENTS.md`](tools/AGENTS.md) | [`docs/agents/tools.md`](docs/agents/tools.md) |
| `displays/` — displays, protocol, operator page | [`displays/AGENTS.md`](displays/AGENTS.md) | [`docs/agents/displays.md`](docs/agents/displays.md) |
| `features/` — diarization, verses, summary | [`features/AGENTS.md`](features/AGENTS.md) | [`docs/agents/features.md`](docs/agents/features.md) |
| `training/` — WSL training and data | [`training/AGENTS.md`](training/AGENTS.md) | [`docs/agents/training.md`](docs/agents/training.md) |
| Mac inference, screens, troubleshooting | this file | [`docs/agents/platform-macbook.md`](docs/agents/platform-macbook.md) (old name [`CLAUDE-macbook.md`](CLAUDE-macbook.md) points there) |
| WSL training, native Lite | this file | [`docs/agents/platform-windows.md`](docs/agents/platform-windows.md) (old name [`CLAUDE-windows.md`](CLAUDE-windows.md) points there) |
| Mac defaults table, extension patterns | this file | [`docs/agents/project-guide.md`](docs/agents/project-guide.md) |
| Runbooks | this file | [`docs/operator_runbook.md`](docs/operator_runbook.md), [`docs/packaging/macos.md`](docs/packaging/macos.md), [`docs/lite_profiles.md`](docs/lite_profiles.md) |
| Dated evaluation boards | this file | [`docs/evaluation/README.md`](docs/evaluation/README.md) |
| Vendored STE skill copy | Clarity below | `.agents/skills/simplified-technical-english/` (`.claude/skills/` links to it) |

## Clarity

Write explanations to Willem in Simplified Technical English. Use the same style
for a pull request description, a commit message body, and prose in a README or
another doc. Aim for about 80 percent of the rules. Full compliance with
ASD-STE100 is not the goal.

The rules are in the standards repository at `standards/writing-ste/ste.md`.
The upstream skill is
[simplified-technical-english](https://github.com/0xpili/simplified-technical-english/tree/1e148d670cba46685ad2b4c3f2354a637a7fdbbe)
(MIT, commit `1e148d670cba46685ad2b4c3f2354a637a7fdbbe`). Link to that skill.
Do not copy the skill into this repository again.

Code, identifiers, math, command-line output, and quoted error text are exempt.

When structure, flow, or architecture is the point, use a mermaid diagram.
For a complex result, offer a self-contained HTML page.
That page is a throwaway file.
Do not commit it unless Willem asks.

Make a video only when Willem asks for a video.
Do not add an API key or a secret.

`scripts/ste_check.py` in the upstream skill is an optional check on docs.
Do not use it as a CI gate.

<!-- standards:begin -->
## Collection standards

Every project under `/Users/willem/Code` follows the shared standards in
`/Users/willem/Code/standards/` (index: `standards/STANDARDS.md`; future
standards: `standards/ROADMAP.md`).

- **Presentations:** build every deck from
  `standards/powerpoint template/Willem-Default.potx` (theme "Helena": Neue Haas
  Grotesk Text Pro, 16:9, black on white with a gray ramp, template v2). Spec:
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
  `standards/enforcement/senior-design-repo/sd-protected-paths.txt`). AI-use
  logging and attestation are opt-in per repo via `ai-attestation-roots.txt`;
  see `standards/standards/ai-use-disclosure/ai-use-disclosure.md`.
- **AI files:** one `AGENTS.md` (≤ 200 lines, Clarity verbatim); `CLAUDE.md` is
  `@AGENTS.md`. Gates: `standards/tools/agents_md_lint.py`, `ai_file_lint.py`.
<!-- standards:end -->
