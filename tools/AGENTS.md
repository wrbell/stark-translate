# tools/AGENTS.md — Evaluation, Screens, Monitoring, Review (Agent Guide)

> Paired with [`CLAUDE.md`](./CLAUDE.md) (imports this file for Claude Code) and with the
> reference guide [`docs/agents/tools.md`](../docs/agents/tools.md) (grouped inventory, measurement rules, harness rules,
> contracts). Repo-wide constraints: [`../AGENTS.md`](../AGENTS.md).

## Constraints

- Importing a `tools/` module must not load a model. Anything needing inference spawns `dry_run_ab.py` or an isolated worker (`mac_evaluation.py` pattern), one model process at a time.
- **Measurement:** report schema 2 `speech_end_to_final_ms` (server, payload-ready) separately from `speech_end_to_ack_upper_bound_ms` (visible browser, includes return network) and from the first-token measure; legacy `e2e_latency_ms` ≠ speech-end-to-display. Never pool cohorts with different manifest hashes. Definitions in [`docs/agents/tools.md`](../docs/agents/tools.md#measurement-rules).
- **Screens:** protocol declared before run 1; gates computed by `tail_screen_report.py`; rejected arms are never re-run or combined; the outcome goes in [`docs/latency_next_experiments.md`](../docs/latency_next_experiments.md); evidence README with source commit, commands, hashes, result and a "does not certify" sentence.
- **References:** predicted transcripts never count as references. Use `mac_evaluation.py annotate` for human-reviewed text; the public EN/ES read-speech references in `docs/evaluation/mac_followup_20260910/public_data/` are evaluation-only, not local approval. Church Spanish and verified two-speaker references remain pending.
- **Review/export:** corrections are revisioned sidecars; only completed sessions with explicit approvals export; evaluation/training splits never cross. Fixtures are not approved correction data (#137 evidence pending).
- **Health check:** `health_check.py --backend mlx` runs 8 of the 18 canaries by default; run before `manage_adapters.py activate`.
- **YouTube alignment:** `find_global_offset_by_text()` for large offsets; treat `_wer is None` as no data. Cross-system WER is disagreement, not accuracy.
- **Replay argv** must match `dry_run_ab.py` flags exactly; on Mac `--stt-backend parakeet-mlx` is valid and is the EN default. Run `python -m tools.replay_bench`; it inherits `STARK_*` and writes `metrics/` under the current checkout.
- No microphone or playback from automation; no numbers in guides (link the dated evidence).

## Key entry points

| Task | Command |
|------|---------|
| Validate frozen manifest | `python tools/mac_evaluation.py validate --manifest docs/evaluation/mac_v2026_14_manifest_v2.json` |
| Frozen experiment screen | `python tools/mac_evaluation.py experiments --manifest docs/evaluation/mac_v2026_14_screening.json --spec docs/evaluation/mac_v2026_14_experiments.json --output metrics/mac_roadmap/experiments --tag <tag> --runs 3` |
| Replay screen | `python -m tools.replay_bench --manifest <runs.json> --tag <tag> --configs "ctl=<argv>" --configs "arm=<argv>"` then `python tools/tail_screen_report.py --runs <runs.jsonl> --output <report.json> --markdown <report.md>` |
| Stage attribution | `python tools/silence_final_stages.py …`, `python tools/stt_overlap_attribution.py …` (see `--help`) |
| Endurance | `python tools/endurance_monitor.py --pid <pipeline pid> --session <id> --output <json>` |
| Adapter gate | `python tools/health_check.py --backend mlx --adapter <dir>` |
| Merge corrections | `python tools/merge_corrections.py translation --corrections <bundle> --train-jsonl <jsonl> --output <jsonl>` |
| Runtime audit check | `python tools/check_dependency_audit.py --runtime mac <report.json>` |
| Backlog docs | `python tools/render_backlog.py validate && python tools/render_backlog.py render --check && python tools/render_backlog.py check-links` |

## Evidence discipline

A status claim needs `metrics/` artifacts (`session_*.log`, `session_lifecycle_*.json`,
`session_metadata_*.json` with `audio_source`); cite the session id, never "tests passed". Browser
`visible:true` ACKs are client reports, not native visibility certification. Evidence under
`docs/evaluation/` is immutable once written (dated appends only). Microphone and device evidence
and its open Spanish capture-loss result: [`docs/agents/tools.md`](../docs/agents/tools.md#recording-evidence).

## Backlog

[`docs/backlog.json`](../docs/backlog.json): `caption-delivery-goal`, `visible-browser-timing-run`,
`mac-live-mic-stall`, `issue-137-active-learning`, `issue-135-mac-ab`.

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
