# features/AGENTS.md — Post-Processing (Agent Guide)

> Paired with [`CLAUDE.md`](./CLAUDE.md).

## Live vs offline

| Feature | Live | Offline/batch |
|---------|------|----------------|
| Diarization | `--diarize` + `live_diarize.py` (default **off**) | `diarize.py` pyannote |
| Summary | Operator trigger post-session | `summarize_sermon.py` |
| Verses | Operator `/api/features/verses` | `extract_verses.py` |

## Diarization gate (#133)

Code shipped locally; **gate pending:** two-speaker natural clip, ≤+50 ms final p95 vs off.
Design: [`docs/live_diarization.md`](../docs/live_diarization.md)

## Active learning

Speaker labels and verse hits feed correction prioritization — export only after human approval.

## Full reference

Integration with `dry_run_ab.py`, CSV columns, MLX contention notes:
[`CLAUDE.md`](./CLAUDE.md)
