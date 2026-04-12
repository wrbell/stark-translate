# displays/ — Browser Display Clients

## Display Modes

| File | Use Case | Key Features |
|------|----------|--------------|
| `audience_display.html` | Projector | EN/ES side-by-side, fading context, fullscreen toggle, QR code overlay |
| `ab_display.html` | Operator | Gemma 4B / MarianMT / 12B comparison with latency stats, rolling averages |
| `mobile_display.html` | Phones/tablets | Responsive, model toggle, Spanish-only mode, accessible via LAN QR |
| `church_display.html` | Church | Simplified layout for non-technical environments |
| `obs_overlay.html` | Streaming | Transparent overlay for OBS Studio integration |

## Serving

- **HTTP**: `0.0.0.0:8080` serves static HTML. Phones connect via LAN IP (QR code on audience display).
- **WebSocket (text)**: `0.0.0.0:8765` for transcription/translation data.
- **WebSocket (TTS audio)**: `0.0.0.0:8766` for binary PCM audio when `--tts` is enabled.
- **URL parameters**: `?port=` overrides the default WebSocket port.

## WebSocket Protocol

### Message Types

**`translation`** — Main transcription/translation updates:
- `stage`: `partial` (italic, MarianMT), `translation_a` (Gemma 4B final), `complete` (includes 12B if A/B mode)
- Key fields: `en_text`, `es_text`, `approach`, `latency_ms`, `confidence`, `stt_ms`, `translate_ms`

**`translation_stream`** — Streaming token-by-token output from TranslateGemma.

**`music_hold`** — Indicates music detection / speech pause.

**`rolling_stats`** — 5-minute rolling averages for operator displays (latency, WER, segment counts).

**`lang_config`** — Language direction change notification (EN→ES or ES→EN).

### TTS Audio WebSocket (port 8766)

Binary PCM frames (16kHz, 16-bit mono). Clients play audio directly via Web Audio API.

## Auto-Reconnect

All display clients auto-reconnect on WebSocket drop. No manual refresh needed.

## Development Notes

- Displays are pure static HTML/CSS/JS — no build step, no bundler.
- Test changes by opening files directly or via the HTTP server.
- WebSocket message format is defined in `dry_run_ab.py` (`broadcast_translation()` and related functions).

### Recent Fixes

- **Scroll behavior**: Changed from `scrollTop` assignment to `scrollIntoView()` for reliable auto-scroll. Removed CSS `scroll-behavior: smooth` which caused animation conflicts with rapid updates.
- **Orphaned partials**: Fixed via CSS class targeting instead of ID mismatch — partial elements now correctly replaced by finals.

---

## Multi-Language Protocol Extension

Planned WebSocket protocol changes for Hindi/Chinese support (full spec in `docs/multi_lingual.md`):

**Current format** (hardcoded EN/ES):
```json
{"en_text": "...", "es_text": "...", "is_final": true}
```

**Future format** (multi-language):
```json
{
  "source_text": "...",
  "translations": {"es": "...", "hi": "...", "zh-Hans": "..."},
  "is_final": true
}
```

- Pipeline sends all active translations over the same WebSocket connection
- Each display client filters by its configured language
- Mobile display stores language preference in `localStorage`
- Language selector tabs: `[EN] [ES] [HI] [ZH]` in `mobile_display.html`
- Audience display: configurable second language via URL param (`?lang2=hi`)

## Font & Typography by Script

| Script | Font | Line-Height | Notes |
|--------|------|-------------|-------|
| Latin (EN/ES) | System fonts | 1.4 | No extra downloads |
| Devanagari (Hindi) | Noto Sans Devanagari (~200KB) | 1.6 | Set `lang="hi"` attribute |
| CJK (Chinese) | System CJK fonts (PingFang SC, Microsoft YaHei, etc.) | 1.5 | No 16MB download — use `lang="zh-Hans"` |

- Minimum font size: 16px+ for readability on projector
- Hindi partials: show English-only partial + Hindi final (SOV word order makes Hindi partials garbled)
- Chinese partials: display normally (SVO, similar to English)

## WebSocket Message Field Reference

Key fields in `translation` messages not fully documented above:

| Field | Type | Description |
|-------|------|-------------|
| `confidence` | float 0.0–1.0 | Mapped from Whisper `avg_logprob` |
| `stt_ms` | float | STT inference time in milliseconds |
| `translate_ms` | float | Translation inference time in milliseconds |
| `chunk_id` | int | Monotonically increasing per session |
| `low_confidence_words` | list | `[{"word": str, "probability": float}, ...]` |
