# tools/AGENTS.md — Evaluation, Monitoring, Review (Agent Guide)

> Paired with [`CLAUDE.md`](./CLAUDE.md) (grouped inventory, measurement rules, contracts).

## Constraints

- Importing a `tools/` module must not load a model. Anything needing inference spawns `dry_run_ab.py` or an isolated worker (`mac_evaluation.py` pattern), one model process at a time.
- **Measurement:** report schema 2 `speech_end_to_final_ms` (server, payload-ready) separately from `speech_end_to_ack_upper_bound_ms` (visible browser, includes return network). Legacy `e2e_latency_ms` ≠ speech-end-to-display. Never pool cohorts with different manifest hashes.
- **References:** predicted transcripts never count as references. Use `mac_evaluation.py annotate` for human-reviewed text; natural Spanish and two-speaker references do not exist yet.
- **Review/export:** corrections are revisioned sidecars; only completed sessions with explicit approvals export; evaluation/training splits never cross. Fixtures are not approved correction data (#137 evidence pending).
- **Health check:** `health_check.py --backend mlx` runs 8 of the 18 canaries by default; run before `manage_adapters.py activate`.
- **YouTube alignment:** `find_global_offset_by_text()` for large offsets; treat `_wer is None` as no data. Cross-system WER is disagreement, not accuracy.
- **Replay argv** must match `dry_run_ab.py` flags exactly; on Mac `--stt-backend parakeet-mlx` is valid and is the EN default.
- Do not add benchmark numbers to guides; link `docs/evaluation/` or `docs/archive/` documents with dates.

## Key entry points

| Task | Command |
|------|---------|
| Validate frozen manifest | `python tools/mac_evaluation.py validate --manifest docs/evaluation/mac_v2026_14_manifest_v2.json` |
| Frozen experiment screen | `python tools/mac_evaluation.py experiments --manifest docs/evaluation/mac_v2026_14_screening.json --spec docs/evaluation/mac_v2026_14_experiments.json --output metrics/mac_roadmap/experiments --tag <tag> --runs 3` |
| Replay matrix | `python tools/replay_bench.py --configs-file <json> --tag <tag>` |
| Adapter gate | `python tools/health_check.py --backend mlx --adapter <dir>` |
| Merge corrections | `python tools/merge_corrections.py translation --corrections <bundle> --train-jsonl <jsonl> --output <jsonl>` |
| Backlog docs | `python tools/render_backlog.py validate && python tools/render_backlog.py render --check && python tools/render_backlog.py check-links` |

## Evidence discipline

A status claim needs `metrics/` artifacts (`session_*.log`, `session_lifecycle_*.json`,
`session_metadata_*.json` with `audio_source`). Tonight: mic session
`20260909_233204_799019_en` stalled (lifecycle stuck `running`); file replays
`..._233546_027169_en` / `..._233823_034893_es` completed. Cite the session id, never
"tests passed", when describing live behavior.

## Backlog

[`docs/backlog.json`](../docs/backlog.json): `caption-delivery-goal` (in progress),
`visible-browser-timing-run`, `issue-137-active-learning`, `issue-135-mac-ab`.
