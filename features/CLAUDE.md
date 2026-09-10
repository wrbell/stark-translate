# features/ — Diarization, Verse Extraction, Sermon Summary

> Paired with [`AGENTS.md`](./AGENTS.md). Describes the local branch (base `5154fb9`,
> v2026.14 candidate); main is v2026.13. Integration status is stated per feature —
> do not read "implemented" as "certified for a service".

Three post-processing capabilities plus a live diarization path. Two are exposed to the
operator UI through [`operator_app/features.py`](../operator_app/features.py); live
diarization is a separate daemon started by `dry_run_ab.py --diarize`. None of them
touches the MLX GPU pool used by STT/translation.

## Status

| Feature | Module(s) | How it is reached today | Certification |
|---------|-----------|-------------------------|---------------|
| Live verse highlights | `extract_verses.py` (`VerseExtractor`) | `GET /api/features/verses` tails the session CSV every few seconds (regex only, inline) | Works in the operator rehearsal (`docs/evaluation/mac_v2026_14_rehearsal.md`); no LLM involved |
| Post-session summary | `summarize_sermon.py` | `POST /api/features/summary` spawns the script out-of-process against the finished CSV; `GET /api/features/summary/{task_id}` polls | Runs on Gemma 4 E4B OptiQ by default (`settings.translation.mlx_model_gemma4_e4b`); output quality not human-reviewed |
| Offline diarization | `diarize.py` | Batch CLI over a WAV or a `stark_data/live_sessions/<id>/` directory (pyannote 3.1) | Historical; needs HF token and pyannote agreement |
| Live diarization (9.6.1 / #133) | `live_diarize.py`, `rolling_buffer.py`, `speaker_labels.py` | `dry_run_ab.py --diarize [--diarize-mode embed\|pyannote] [--diarize-interval-s]`; daemon writes `metrics/diarization_<session>.jsonl`, pipeline attaches `speaker` to finals/CSV/JSONL/WebSocket | **Implemented, gate not run.** Issue acceptance needs a two-speaker dry run with distinct labels and final p95 within +50 ms; no natural two-speaker clip exists yet |

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

Two regex passes — explicit citations (`Romans 8:28`) and spoken forms (`turn to
Romans chapter eight`, bare `verse 28` resolved against the last book/chapter) — over
all 66 book-name variants. Input: pipeline CSV or diarized JSONL. Unit tests:
`tests/test_verse_extraction.py`. Verse pairs for training come from
`tools/rebuild_verse_pairs.py`, not from this extractor.

## Sermon summary

Reads the session CSV or diarized JSONL and produces EN + ES summaries (five sentences
with speakers, three without). Model family is inferred from the model id
(`_summary_model_family`), so the Gemma 4 stop-token rules from
[`engines/CLAUDE.md`](../engines/CLAUDE.md) apply. Spanish comes from the same model or
`--translate-with-gemma` (TranslateGemma 4B). Always run out-of-process next to a live
pipeline. Tests: `tests/test_summarize_sermon.py`, `tests/test_phase9_6_features.py`.

## Dependencies

- Live diarization `embed` mode: `.[mlx,diarization]` (SpeechBrain ECAPA, torchaudio, scikit-learn).
- `pyannote` mode and `diarize.py`: `pyannote.audio ≥ 3.1` plus an HF token with the accepted model agreement.
- Summary: the MLX runtime already installed for finals.

## Open items

- `issue-133-diarize-gate` and `natural-two-speaker` in [`docs/backlog.json`](../docs/backlog.json).
- Summary/verse UI evidence in the operator runbook is root-owned and refreshed after PR #192 integration.
