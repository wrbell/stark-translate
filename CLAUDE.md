@AGENTS.md

Claude Code loads this file and imports `AGENTS.md` above; the rules live there.
`tests/test_documentation.py` checks this file for the release line v2026.14,
links to [`docs/backlog.json`](docs/backlog.json) and
[`docs/current_architecture.md`](docs/current_architecture.md), and the count of
11 GitHub Actions workflow files. Keep them current.

<!-- standards:begin -->
Collection standards (presentations, sources, git hygiene) live in
`/Users/willem/Code/standards/STANDARDS.md`; decks are built from
`standards/powerpoint template/` and gated with its `deck_checks.py`. Managed
block: edit `standards/ai-files/BLOCK-nested.md`, not this copy.
AI use in school work: no AI images, adviser pre-clearance for AI-written
text, never edit graded text; see
`standards/standards/ai-use-disclosure/ai-use-disclosure.md`.
AI files: one `AGENTS.md` per repo; `CLAUDE.md` is `@AGENTS.md`.
<!-- standards:end -->
