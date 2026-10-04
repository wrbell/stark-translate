# features/AGENTS.md — Diarization, Verses, Summary (Agent Guide)

> Paired with [`CLAUDE.md`](./CLAUDE.md) (imports this file for Claude Code) and with the
> reference guide [`docs/agents/features.md`](../docs/agents/features.md) (status table, data flow, dependencies). Repo-wide
> constraints: [`../AGENTS.md`](../AGENTS.md).

## Constraints

- Nothing in `features/` may run inside the live pipeline's MLX GPU pool. Verse extraction is regex-only and runs inline; summary runs as a subprocess; live diarization is a separate daemon (`features/live_diarize.py`) whose labels are read from `metrics/diarization_<session>.jsonl` by cheap overlap lookup.
- Diarization is **off by default** (`--diarize`). Budget from `docs/live_diarization.md`: final p95 within +50 ms of baseline; do not move STT/translation code to satisfy a diarization change.
- `speaker_labels.py` must stay free of model imports (it is unit-tested on CPU).
- Verse extraction recognizes English book names and number phrases, including whole spaced/hyphenated compounds. Resolve context in text order; a changed book clears the old chapter. Hymn/stanza markers clear Bible context; unresolved numbers, unfinished references and unsupported lists are omitted rather than guessed.
- Validate chapter/verse bounds before emitting a reference or committing chapter context. [`bible_reference_bounds.py`](./bible_reference_bounds.py) bundles metadata from the existing KJV corpus; see [`docs/agents/features.md`](../docs/agents/features.md#verse-extraction) for its source hash. Valid bounds prove structural existence only, not what was spoken. Spanish reference grammar and alternative versifications are not implemented.
- Summary model ids follow the Gemma 4 stop-token rules in `engines/translation_prompts.py`; never inline prompts. Summary and live diarization loaders use pinned local resolution (details and receipts in [`docs/agents/features.md`](../docs/agents/features.md#dependencies)); do not turn the standalone offline diarization loader into an implicit live download path.
- Human-facing claims: verse highlights and summary have workflow evidence in the operator rehearsal only; bilingual accuracy approval is pending. Live diarization has **no** two-speaker evidence — the #133 gate is unmet.

## Where features surface

| Surface | Route / flag |
|---------|--------------|
| Operator UI verses | `GET /api/features/verses` (`operator_app/features.py`) |
| Operator UI summary | `POST /api/features/summary`, `GET /api/features/summary/{task_id}` |
| Live speaker labels | `dry_run_ab.py --diarize --diarize-mode embed\|pyannote --diarize-interval-s N [--diarize-python <interpreter>]` |
| Two-voice latency fixture | `python tools/synthetic_two_voice_clip.py` (synthetic, silent to device; not a natural two-speaker clip) |
| Batch tools | `python features/diarize.py`, `features/extract_verses.py`, `features/summarize_sermon.py` |

## Backlog

`issue-133-diarize-gate`, `natural-two-speaker` in [`docs/backlog.json`](../docs/backlog.json).

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
