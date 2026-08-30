# features/ — Post-Processing Features

All three features are implemented but not yet integrated with the live pipeline. Ready for integration testing with live session data.

## Speaker Diarization (`diarize.py`)

Uses `pyannote-audio` to identify and separate speakers. Two-speaker assumption enforced via `min_speakers=2, max_speakers=2`. Runs on CPU (Mac) or CUDA (WSL). Required by both summarization and verse extraction.

## Post-Sermon Summary (`summarize_sermon.py`)

Generates a structured 5-sentence summary from the full diarized transcript:

- **Sentence 1:** Overall gist covering both speakers' themes
- **Sentences 2–3:** Summary of Speaker 1's key points
- **Sentences 4–5:** Summary of Speaker 2's key points

Uses a local LLM (Gemma 3 4B or similar) with a structured prompt. Outputs both English and Spanish summaries. Runs as a batch job post-service — no real-time constraint.

## Verse Reference Extraction (`extract_verses.py`)

Returns a per-speaker list of every Bible verse referenced during the sermon.

**Two-pass approach:**
1. Regex + rule-based extraction for explicit citations ("Romans 8:28")
2. LLM-assisted extraction for spoken-form references ("turn to Romans chapter eight")

Outputs per-speaker JSON with reference, timestamp, and context snippet. Stretch goal: cross-reference extracted verses against translation output for theological accuracy checks.

---

## Integration with Live Pipeline

How features connect to `dry_run_ab.py` for future live/batch integration:

- **Input**: Features read session CSV from `metrics/ab_metrics_{SESSION_ID}.csv` (columns: `chunk_id`, `timestamp`, `english`, `spanish_a`, `spanish_b`, `stt_ms`, `translate_ms_a`, `confidence`, `low_confidence_words`)
- **Diarization**: Currently standalone. Will be called with `--diarize-on-demand` flag to process live session audio post-service. Output feeds into both summary and verse extraction.
- **Summary + Verses**: Run as batch jobs after session ends, reading the session CSV. No real-time constraint.
- **Future live integration**: Register as post-final callback in `dry_run_ab.py` pipeline — verse extraction could run on each final translation in real-time. Summary remains batch-only (needs full transcript).

## Active Learning Connection

Features feed into and benefit from the active learning cycle:

- **Diarization → fine-tuning**: Speaker-labeled segments improve per-speaker WER tracking. Enables targeted correction of worst-performing speaker segments.
- **Verse extraction → quality checks**: If a verse reference is detected, spot-check the translation against glossary entries for that verse's key theological terms (e.g., "atonement" → "expiación").
- **Summary generation → MLX thread affinity**: Uses the same MLX pipeline pool as live translation. Prefer queuing summaries after the live session ends so they do not contend with STT∥translation overlap. MLX >= 0.31.2 allows independent concurrent eval, but session-end still keeps the live path predictable.
