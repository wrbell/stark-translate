# displays/AGENTS.md — Displays & WebSocket Protocol (Agent Guide)

> Paired with [`CLAUDE.md`](./CLAUDE.md) (full message table, timing semantics, SPA routes).

## Constraints

- Displays are static HTML/JS — no build step, no framework. Include `display_connection.js` and wrap socket handlers with `caption_telemetry.js` so `caption_rendered` ACKs keep flowing.
- Never add post-send durations to a broadcast payload; ACK timing is computed server-side by `RenderTracker` (`tools/pipeline_timing.py`) and written to `metrics/display_metrics_<session>.jsonl`.
- `speech_end_to_ack_upper_bound_ms` is recorded only for visible tabs and includes return-network time. Hidden tabs and accelerated replay cannot satisfy caption-delivery gates.
- Handle `lang_config` first: it carries `session_id`; a changed id resets history. `english` / `spanish_a` are source/target slots, not languages.
- `utterance_discarded` removes only the matching session/utterance's provisional caption and suppresses late matching partials. Final chunk numbers are a separate identity; never remove a final or final stream because its number matches a discarded utterance.
- An authoritative `translation` complete closes its explicit session/utterance identity against late partials. Preserve previews for newer utterances when an older final or `translation_start` arrives. Publication closure is always active; experimental closure at final admission remains opt-in.
- Legacy `e2e_latency_ms` / `true_e2e_ms` are processing measurements; do not label them speech-end-to-display in any UI or doc.
- Operator SPA state must not infer RUNNING from file presence. Since 2026-09-10 `ready` comes from the `tools/pipeline_health.py` channel (`phase`, `stale`) surfaced by `/api/session/status`; keep it that way. Real microphone readiness and synthetic acoustic tests ran; the Spanish retest retained upstream sample loss. #131 remains unvalidated, and the current user instruction prohibits further microphone/output tests this session.
- Review/support endpoints live under `/api/review/...`, `/api/support/...`, `/api/storage/...`, `/api/audio/test-input|test-output`; the overnight widgets (`widgets/captions.js`, `qr.js`, `sparkline.js`) are integrated.

## Message types

`lang_config`, `translation` (`stage: partial|complete`), `translation_start`,
`translation_stream`, `utterance_discarded`, `speaker_update`, `music_hold`, `rolling_stats`, `text`.
Every broadcast has `session_id` and an opaque, session-scoped `event_id`; finals also
carry provenance (`session_kind` live/replay/synthetic, `audio_source`, input hash) and
schema 2 sample metadata.

## Ports

8080 HTTP displays · 8765 caption WebSocket · 9000 operator control plane
(`/ws/control`, `/ws/audio/ingest`, `/ws/audio/subscribe`).

## Validation

HTML5 Tidy (zero warnings) is part of the recorded validation; run it after editing
any display. Physical projector / second-screen checks remain human gates
(`docs/backlog.json`: `visible-browser-timing-run`); #134 separately permits a laptop stand-in.
