# features/ — Diarization, Verse Extraction, Sermon Summary

> Paired with [`AGENTS.md`](./AGENTS.md). Describes v2026.14 source on `main`. Acceptance is
> stated per feature; "implemented" does not mean "certified for a service". Evidence:
> [`docs/mac_implementation_status.md`](../docs/mac_implementation_status.md).

Three post-processing capabilities plus a live diarization path. Two are exposed to the
operator UI through [`operator_app/features.py`](../operator_app/features.py); live
diarization is a separate daemon started by `dry_run_ab.py --diarize`. None of them
touches the MLX GPU pool used by STT/translation.

## Status

| Feature | Module(s) | How it is reached today | Certification |
|---------|-----------|-------------------------|---------------|
| Live verse highlights | `extract_verses.py` (`VerseExtractor`) | `GET /api/features/verses` tails the session CSV every few seconds (regex only, inline) | Workflow exercised in the operator rehearsal; parsing safety has recorded-fragment regressions. Spoken-reference accuracy is not certified; no LLM involved |
| Post-session summary | `summarize_sermon.py` | `POST /api/features/summary` spawns the script out-of-process against the finished CSV; `GET /api/features/summary/{task_id}` polls | Runs on Gemma 4 E4B OptiQ by default (`settings.translation.mlx_model_gemma4_e4b`); output quality not human-reviewed |
| Offline diarization | `diarize.py` | Batch CLI over a WAV or a `stark_data/live_sessions/<id>/` directory (pyannote 3.1) | Historical; needs HF token and pyannote agreement |
| Live diarization (9.6.1 / #133) | `live_diarize.py`, `rolling_buffer.py`, `speaker_labels.py` | `dry_run_ab.py --diarize [--diarize-mode embed\|pyannote] [--diarize-interval-s] [--diarize-python <interpreter>]`; daemon writes `metrics/diarization_<session>.jsonl`, pipeline attaches `speaker` to finals/CSV/JSONL/WebSocket | **Implemented; natural two-speaker gate not run.** Issue acceptance needs a two-speaker dry run with distinct labels and final p95 within +50 ms. `tools/synthetic_two_voice_clip.py` builds a silent-to-device Piper fixture for the latency half-gate; no natural two-speaker clip exists yet (`natural-two-speaker`) |

Default is diarization **off**. Design and the p95 budget:
[`docs/live_diarization.md`](../docs/live_diarization.md).

## Live diarization data flow

```
mic → VAD → STT ∥ translation                 (GPU pool, unchanged)
                 │ final chunk audio
                 ▼
rolling_buffer.py  keeps ~25 s of speech as rolling.wav + chunks.jsonl (disk)
                 │
                 ▼
live_diarize.py    separate process: ECAPA embeddings ("embed", SpeechBrain,
                   online cosine clustering) or pyannote ("pyannote", needs HF_TOKEN);
                   appends {start, end, speaker} to metrics/diarization_<session>.jsonl
                 │
                 ▼
speaker_labels.assign_speaker_from_jsonl()  cheap overlap lookup on the
                   asyncio thread → result_data["speaker"] → CSV/JSONL/WS
```

Pure helpers in `speaker_labels.py` (JSONL parsing, overlap assignment, cosine
clustering) have no model imports and are unit-tested
(`tests/test_speaker_labels.py`, `tests/test_phase9_6_1_live_diarize.py`).

## Verse extraction

[`extract_verses.py`](./extract_verses.py) recognizes English-form references such as
`Romans 8:28`, `Luke twenty three and verse thirty two`, and contextual `verse 28`.
Spaced and hyphenated compound numbers are matched whole, without backtracking from
`twenty three` to 20:3. Pattern priority resolves overlaps; accepted matches and book
mentions update context in original text order. An explicit standalone book mention
can establish pending context, but changing books never carries the old chapter.
Hymn/stanza markers clear Bible context. Recognized unfinished number/reference forms, ambiguous
numbered books and unsupported verse lists are omitted rather than completed by guesswork.

[`bible_reference_bounds.py`](./bible_reference_bounds.py) checks structural bounds
before reference emission or chapter-context commit. Its metadata-only table covers
66 books and 1,189 chapters, derived from the existing public-domain
`bible_data/scrollmapper/formats/sqlite/KJV.db`, SHA-256
`c208b439188880c442dd77bef936926487d98ef101e11d65759b283368077234`.
The installed extractor imports this bundled Python module relative to its package;
no database or repository working directory is needed at runtime. Valid KJV bounds
do not establish that the reference was spoken or correctly transcribed.

Input remains pipeline CSV or diarized JSONL; original predictions are not rewritten.
The CSV's historical `english` field is read as supplied, so Spanish-source sessions
do not gain Spanish reference grammar from the language-direction switch. Other
versification systems are not implemented. Tests:
[`test_verse_extraction.py`](../tests/test_verse_extraction.py) and
[`test_verse_safety.py`](../tests/test_verse_safety.py), including actual recorded
fragments, the real operator watcher, and imports outside the checkout. Training
verse pairs still come from `tools/rebuild_verse_pairs.py`, not this extractor.

## Sermon summary

Reads the session CSV or diarized JSONL and produces EN + ES summaries (five sentences
with speakers, three without). Model family is inferred from the model id
(`_summary_model_family`), so the Gemma 4 stop-token rules from
[`engines/CLAUDE.md`](../engines/CLAUDE.md) apply. Spanish comes from the same model or
`--translate-with-gemma` (TranslateGemma 4B). The operator runs the post-session
summary in a subprocess after live work releases the exclusive work lease.
Both model loading paths now use the shared pinned load-time resolver;
an uncached custom model requires a manifest revision or local path. Tests:
`tests/test_summarize_sermon.py`, `tests/test_phase9_6_features.py`.

## Dependencies

- Live diarization `embed` mode: `.[mlx,diarization]` (SpeechBrain ECAPA, torchaudio, scikit-learn).
- `pyannote` mode and `diarize.py`: `pyannote.audio ≥ 3.1` plus authorized HF access to both `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`.
- Summary: the MLX runtime already installed for finals.

Live ECAPA pins its YAML and secondary weights. Full Pyannote waits for a valid
nonempty rolling WAV, resolves pinned configuration and nested checkpoint sources,
and retains the original pipeline parameters. The available credential currently
lacks access to the segmentation model; unavailable optional models fail gracefully
and are attempted once per daemon run. [Source and access receipts](../docs/evaluation/mac_followup_20260910/live-hf-pinning.md)
do not certify native speaker accuracy or latency.

## Open items

- `issue-133-diarize-gate` and `natural-two-speaker` in [`docs/backlog.json`](../docs/backlog.json).
- Summary/verse UI evidence lives in the operator runbook and rehearsal records, not here.
