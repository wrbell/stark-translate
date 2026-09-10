# Overnight documentation status — `codex/overnight-docs`

> **Worktree role:** Documentation and backlog consolidation only. No model loads,
> installs, GPU work, or GitHub delivery from this branch. Root owns merge, PR,
> issue closure, and publication.

**Branch base:** `5154fb9` on `codex/mac-reliability-roadmap` · **Version:** `2026.14.0.0`
**Main:** v2026.13 (`09e4679`) — integration pending

## What this worktree delivered

| Artifact | Purpose |
|----------|---------|
| [`backlog.json`](./backlog.json) | Canonical machine-readable remaining tasks |
| [`backlog.md`](./backlog.md) | Rendered human view (`tools/render_backlog.py`) |
| [`current_architecture.md`](./current_architecture.md) | Current inference/operator contracts |
| Consolidated `CLAUDE.md` / `AGENTS.md` | Concise agent guidance + archival links |
| Subdirectory `AGENTS.md` guides | Paired with existing `CLAUDE.md` files |
| `tools/render_backlog.py` | Validate, render, link-check helpers |
| `tests/test_documentation.py` | Backlog schema, render drift, local links |

## Evidence boundaries (do not over-claim)

**Implemented and validated locally** (see [`mac_implementation_status.md`](./mac_implementation_status.md)):

- Operator reliability, schema 2 timing, setup/resolver, Review/export, opt-in experiments
- 48-run English screening (no combined winner), 24 synthetic routing probes
- Final CPU suite, packaging artifact checks, controlled operator browser rehearsal
- Test counts and coverage: cite `mac_implementation_status.md` only — not duplicated in every guide

**Still open** (tracked in [`backlog.json`](./backlog.json)):

- Natural Spanish references, two-speaker labels, bilingual blinded review
- Physical second output and church-hardware service rehearsal (#134)
- Visible-browser timing with non-zero ACK coverage
- WSL Phase 4 / W17 / CUDA latency execution
- Lite CPU and RTX 2070 validation (equal priority; separate agents/hardware)
- PyPI publication (explicitly pending by user choice)

## User decisions captured

- **Publishing:** pending; merge to main authorized when integration complete
- **Mac tonight:** documentation reflects Parakeet EN, Whisper ES, Marian CT2 CPU, E4B finals
- **Partials vs finals:** fast revisable Marian previews; careful Gemma finals (optional E2B only after review)
- **Targets:** lite CPU and 2070 equally targeted alongside Mac

## Root integration checklist

1. Merge `codex/mac-reliability-roadmap` documentation commits
2. Reconcile open issues #131–#138, #176, #177 against [`backlog.json`](./backlog.json) statuses
3. Run `python tools/render_backlog.py validate && python tools/render_backlog.py render --check`
4. Run `pytest tests/test_documentation.py -v`
5. Update release tag / PyPI when user approves publication
6. Refresh [`operator_runbook.md`](./operator_runbook.md) with final UI evidence (root-owned)

## Parallel overnight agents (out of scope here)

| Agent / owner | Owns |
|---------------|------|
| overnight-lite | `docs/packaging/*`, CPU/2070 install paths |
| overnight-operator-ui | Operator runbook UI evidence |
| root | GitHub delivery, merge, tags, issue closure |
