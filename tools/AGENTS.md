# tools/AGENTS.md — Monitoring & Validation (Agent Guide)

> Paired with [`CLAUDE.md`](./CLAUDE.md).

## Agent constraints

- **YouTube alignment:** Use `find_global_offset_by_text()` for large offsets; check `_wer is None`.
- **Health check:** 8 canaries in `training/theological_canaries.py` — run before adapter activate.
- **Review/export:** Only completed sessions with explicit approvals export training data.
- **Holdout:** Evaluation manifest v2 rebinds by source text; do not pool mismatched cohorts.

## Key tools

| Tool | Use |
|------|-----|
| `health_check.py` | Pre-deploy adapter verification |
| `replay_bench.py` | Real-audio pipeline latency matrix |
| `merge_corrections.py` | Active learning merge |
| `manage_adapters.py` / `deploy_adapters.py` | Lifecycle |
| `render_backlog.py` | Validate/render [`docs/backlog.json`](../docs/backlog.json) |

## Measurement

- Schema 2 fields in session CSV — see [`docs/current_architecture.md`](../docs/current_architecture.md)
- Archived `e2e_latency_ms` ≠ speech-end-to-display

## Full reference

YouTube compare, QE tiers, deployment pipeline, eval sets:
[`CLAUDE.md`](./CLAUDE.md)
