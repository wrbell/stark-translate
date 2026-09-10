# Overnight documentation status — `codex/overnight-docs`

> **Worktree role:** documentation, backlog and doc-validation tooling only. No model
> loads, installs, GPU work or GitHub delivery from this branch. Root owns merge, PR,
> issue closure and publication; the sibling overnight worktrees were read for
> orientation only and are never edited from here.

**Branch base:** `5154fb9` on `codex/mac-reliability-roadmap` · **Version:** `2026.14.0.0`
**Main:** v2026.13 (`09e4679`) · **Draft [PR #192](https://github.com/wrbell/stark-translate/pull/192):**
open, base `main` ← `codex/mac-reliability-roadmap`, **not merged**. Source and issue
publishing and the final merge are authorized; PyPI/package artifacts/release tags remain
pending by user choice.

**Commits on this branch:** `0095491` (first pass) plus the follow-up correction commit
recorded in `git log codex/overnight-docs`.

## What this worktree delivered

| Artifact | State |
|----------|-------|
| [`backlog.json`](./backlog.json) (schema 2) + rendered [`backlog.md`](./backlog.md) | 38 items with separate `status` and `certification`; `in_progress` vocabulary; issue acceptance quoted per GitHub issue |
| `tools/render_backlog.py` | `validate` (schema, status/certification consistency, source paths exist), `render [--check]`, `check-links` |
| `tests/test_documentation.py` | Backlog schema/render drift, local links, guide pairs, workflow count derived from `.github/workflows/`, known-stale-claim guard |
| Root `CLAUDE.md` / `AGENTS.md` | Same content, cross-linked; release table with dated archive links; CI list |
| `engines/`, `features/`, `displays/`, `tools/`, `training/` guides | `CLAUDE.md` rewritten from current source; `AGENTS.md` paired constraint summaries (training `CLAUDE.md` edited in place — long historical sections kept and marked) |
| `CLAUDE-macbook.md` | Rewritten: setup CLI install, `models.lock.json` model set, Gemma 4 stop-token semantics, mic-stall known issue, no static latency/memory tables |
| `CLAUDE-windows.md` | Targeted: status banner, runbook, Phase 4 labels marked historical (Deepgram oracle), TranslateGemma section marked historical → Gemma 4 program, Model Transfer rewritten around exported artifacts |
| `README.md` | Removed linear GPU-scaling claim and unsourced latency/size numbers; models table by role with archive links; 10 workflow files; project structure refreshed; PR #192 and mic-stall status |
| [`docs/roadmap.md`](./roadmap.md) | Rewritten: current state, active work, historical sections with dated sources |
| [`current_architecture.md`](./current_architecture.md) | PR #192 status, mic-stall known bug, in-progress worktrees pending integration, CI section |

## Evidence boundaries (do not over-claim)

**Implemented and validated locally** — cite [`mac_implementation_status.md`](./mac_implementation_status.md)
for counts; not duplicated in guides:

- Operator reliability, schema 2 timing, setup/resolver, Review/export, opt-in experiments
- 48-run English screening (no combined winner), 24 synthetic routing probes
- Final CPU suite, packaging artifact checks, controlled operator browser rehearsal
- Tonight's controlled **file-replay** EN and ES caption sessions
  (`20260909_233546_027169_en`, `20260909_233823_034893_es`, `audio_source: file`,
  exit 0) and review-draft recovery passed

**Not proven tonight:**

- **Live built-in microphone:** session `20260909_233204_799019_en` (`audio_source: mic`)
  loaded models, printed "Listening...", served the audience page and then received no
  audio frames; lifecycle stayed `status: running`; operator showed RUNNING from the CSV
  header; audience display stayed disconnected. A standalone `sounddevice` record probe
  stalled too. File replay does not prove live mic. → `mac-live-mic-stall` (P0,
  `in_progress`), `issue-131-smoke` (`in_progress`)
- **Physical devices:** second output / TTS routing acceptance (#132), church or
  laptop stand-in dry run (#134) — user deferred to tomorrow
- **Correction loop:** `merge_corrections.py` validated on fixtures only; no approved
  human correction data yet (#137)
- **Hindi/Chinese:** offline text Hindi runs exist but are not the requested
  church-audio/live baseline (#138); language expansion remains a pending user choice
- **Sub-second caption delivery:** active Mac engineering (`caption-delivery-goal`,
  `overnight-latency-scheduling`) with external certification gates; independent
  experiments proceed without Spanish references

**Overnight implementations in sibling worktrees (read, not integrated):** latency
scheduling (`overnight-latency` commit `1a8470c` + uncommitted incremental STT/preview
candidates), lite CPU profile/preflight/llama runtime (`overnight-lite`, uncommitted),
operator caption/QR widgets (`overnight-operator-ui`, uncommitted), reliability
helpers — process supervision, work lease, capture worker, isolated audio, pipeline
health (`overnight-reliability`, uncommitted). They are recorded as `in_progress` with
`certification: pending`; nothing here claims they are merged or validated.

## User decisions captured

- **Publishing:** source/issue publishing and final merge authorized; PyPI, package
  artifacts and release tags pending
- **Mac defaults:** Parakeet EN, Whisper turbo ES, Marian CT2 CPU partials, Gemma 4 E4B
  OptiQ finals, 0.6 s partial cadence / 0.5 s silence — frozen for the candidate
- **Partials vs finals:** fast revisable previews; careful finals; E2B only after review
- **Targets:** lite CPU and RTX 2070 equal priority alongside Mac; implementation and
  certification tracked separately
- **Tomorrow:** live microphone and physical-device checks

## Root integration checklist

1. Merge the `codex/overnight-docs` commits into the integration branch behind PR #192
2. Reconcile issues #131–#138, #176, #177 against [`backlog.json`](./backlog.json)
   statuses (do not close #131/#132/#134 on file-replay evidence)
3. `python tools/render_backlog.py validate && python tools/render_backlog.py render --check`
4. `python tools/render_backlog.py check-links` and `pytest tests/test_documentation.py -v`
5. After integrating other worktrees, move their items from `in_progress` and add
   evidence links; re-render `backlog.md`
6. Refresh [`operator_runbook.md`](./operator_runbook.md) with final UI evidence (root-owned)
7. Release tag / PyPI only when the user approves publication

## Unfinished

Exact areas this worktree did **not** finish; root or a later pass should cover them:

- `training/CLAUDE.md` (edited in place, ~400 lines) still carries design-era sections
  (Phase 4 preprocessing steps, TranslateGemma QLoRA config, pseudo-labeling, WER
  tables). They are marked historical with pointers but were not condensed or verified
  line by line against the current `training/*.py` scripts; the numbers inside them are
  original design/benchmark notes, not re-sourced.
- `CLAUDE-windows.md` (~1,300 lines) was edited in targeted places only (banner,
  runbook, Phase 4 label source, TranslateGemma section header, Model Transfer). The
  long embedded code samples, WSL install commands (CUDA 12.6 toolkit,
  `requirements-windows.txt`, FlashAttention), storage layout, Phase 4 step details and
  troubleshooting were not re-verified on the WSL machine tonight — nothing ran there.
- `docs/mac_implementation_status.md`, `docs/operator_runbook.md`, `docs/packaging/*`
  and `docs/evaluation/*` were not edited (root/lite ownership); they were only
  link-checked. Any stale wording there is out of scope for this worktree.
- Backlog items for the sibling worktrees describe their file inventories as observed
  tonight; acceptance text for them should be tightened by their owners at integration.
- No benchmark numbers were added anywhere; a few historical numbers remain in
  `training/CLAUDE.md` and the archives with their original context.
- `docs/deploy.md`, `docs/wsl_pipeline_refresh.md`, `docs/cuda_latency_proposal.md`
  and `docs/gemma4_tuning/*` are linked but were not reviewed for stale statements.

## Parallel overnight agents (out of scope here)

| Agent / worktree | Owns |
|------------------|------|
| overnight-lite | `docs/packaging/*`, CPU profile, lite preflight, 2070 install paths |
| overnight-latency | scheduler / caption delivery / latency trace experiments |
| overnight-operator-ui | operator caption + QR widgets, runbook UI evidence |
| overnight-reliability | process supervision, capture worker, pipeline health |
| root | GitHub delivery, PR #192, merge, tags, issue closure |
