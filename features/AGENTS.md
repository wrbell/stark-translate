# features/AGENTS.md — Diarization, Verses, Summary (Agent Guide)

> Paired with [`CLAUDE.md`](./CLAUDE.md) (status table, data flow, dependencies).

## Constraints

- Nothing in `features/` may run inside the live pipeline's MLX GPU pool. Verse extraction is regex-only and runs inline; summary runs as a subprocess; live diarization is a separate daemon (`features/live_diarize.py`) whose labels are read from `metrics/diarization_<session>.jsonl` by cheap overlap lookup.
- Diarization is **off by default** (`--diarize`). Budget from `docs/live_diarization.md`: final p95 within +50 ms of baseline; do not move STT/translation code to satisfy a diarization change.
- `speaker_labels.py` must stay free of model imports (it is unit-tested on CPU).
- Summary model ids follow the Gemma 4 stop-token rules in `engines/translation_prompts.py`; never inline prompts.
- Human-facing claims: verse highlights and summary have workflow evidence in the operator rehearsal only; bilingual accuracy approval is pending. Live diarization has **no** two-speaker evidence — the #133 gate is unmet.

## Where features surface

| Surface | Route / flag |
|---------|--------------|
| Operator UI verses | `GET /api/features/verses` (`operator_app/features.py`) |
| Operator UI summary | `POST /api/features/summary`, `GET /api/features/summary/{task_id}` |
| Live speaker labels | `dry_run_ab.py --diarize --diarize-mode embed\|pyannote --diarize-interval-s N` |
| Batch tools | `python features/diarize.py`, `features/extract_verses.py`, `features/summarize_sermon.py` |

## Backlog

`issue-133-diarize-gate`, `natural-two-speaker` in [`docs/backlog.json`](../docs/backlog.json).
Runbook UI evidence (`docs/operator_runbook.md`) is owned by root.
