# Release Plan: Sunday Morning Church Test (2026-03-01)

## Context

The live bilingual pipeline is functionally complete — STT ~600ms, Gemma 4B ~700ms, QE 0.85-1.0, zero hallucinations in the last 46-chunk session. Recent fixes (truncation, near-miss detection, marian latency race, short-phrase dampening) landed in `b0db5c9`. No fine-tuning time available. Projector display will be `audience_display.html` (white background, full streaming support, QR code).

**Goal:** Ship the highest-impact, lowest-risk changes before Sunday morning.

## Changes in this release (v2026.3.1.0)

### 1. Music/Hymn Auto-Muting

- Tracks consecutive non-speech VAD frames where RMS is above a "music" threshold (default 0.02)
- After ~2s of music-like audio (no speech + high energy), enters music-hold mode
- Skips submitting audio to STT — no more garbage text during hymns
- Broadcasts `music_hold` WebSocket message to displays
- Subtle music note indicator on audience_display.html
- Exits music-hold when speech resumes for ~0.5s
- CLI flags: `--music-threshold` (default: 0.02 RMS), `--music-holdoff` (default: 2.0s)
- Session summary includes music hold count and total duration

### 2. Context-Aware Theological STT Correction

- `correct_stt_output(text)` applies three correction layers:
  1. **PHRASE_CORRECTIONS** — always applied (multi-word, unambiguous)
  2. **NEAR_MISS_CORRECTIONS** — always applied (unambiguous misspellings)
  3. **HOMOPHONE_FLAGS** — only when theological context is present
- Context gate prevents false corrections: "It started to rain" stays unchanged, but "the rain of Christ" becomes "the reign of Christ"
- Runs between STT and translation in the pipeline coordinator
- Corrections logged to diagnostics JSONL and session summary
- Inline console output: `>> STT CORRECTED (homophone): 'rain' → 'reign'`

### 3. Launch Script (`run_church.sh`)

- One-command start: `./run_church.sh`
- Activates virtualenv, prints display URLs with LAN IP
- Passes all CLI args through to dry_run_ab.py

### 4. Church Environment (`.env.church`)

- Pre-configured settings for church service:
  - `STARK_VAD_THRESHOLD=0.25` (slightly more sensitive)
  - `STARK_VAD_SILENCE_TRIGGER=1.0` (longer pause before final)
  - `STARK_AUDIO_CHUNK_DURATION=3.0` (more context per chunk)
- Usage: `cp .env.church .env` before service

## Pre-Service Checklist

1. `cp .env.church .env`
2. `./run_church.sh`
3. Open `http://localhost:8080/displays/audience_display.html` on projector
4. Open `http://localhost:8080/displays/ab_display.html` on operator laptop
5. Verify WebSocket connected (green dot)
6. Test with a spoken sentence — confirm EN/ES appears
7. Test music muting — play a hymn, confirm no garbage text

## Verification

1. **Music muting:** Play a hymn/music into mic — verify no garbage, pipeline resumes on speech
2. **STT correction:** Say "the media between God and man" — verify "mediator" in translation input
3. **STT correction safety:** Say "check the media for updates" — verify "media" is NOT corrected (no theological context)
4. **Launch script:** `./run_church.sh` starts correctly
5. **Full dry run:** 5+ minutes of mixed speech — clean output
6. **Tests:** `pytest tests/ -v` — all pass
7. **Lint:** `ruff check . && ruff format --check .` — clean
