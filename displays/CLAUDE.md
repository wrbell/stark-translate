# displays/ — Browser Displays, WebSocket Protocol, Operator SPA

> Paired with [`AGENTS.md`](./AGENTS.md). Protocol statements come from
> `dry_run_ab.py` (`broadcast()`, `_session_provenance()`), `tools/pipeline_timing.py`
> and the display JS on the local branch (base `5154fb9`). Main is v2026.13.

Static HTML/JS served by the pipeline's HTTP server (`--http-port 8080`) with captions
pushed over WebSocket (`--ws-port 8765`). The operator control plane is a separate
FastAPI app on port 9000 (`run_operator.sh` / `stark-translate operator`) that serves
[`operator/`](./operator/) and proxies session control over `/ws/control`.

## Files

| File | Purpose |
|------|---------|
| `audience_display.html` | Projector: source/target side by side, fading history, fullscreen, QR overlay, `music_hold` banner, speaker labels |
| `ab_display.html` | Operator comparison: A (Gemma finals) / Marian partial / B (A/B mode) with latency stats |
| `mobile_display.html` | Phone/tablet: responsive, model toggle, target-only mode |
| `church_display.html` | Simplified church layout |
| `obs_overlay.html` | Transparent overlay for OBS Studio |
| `display_connection.js` | Shared connection settings: `?port=` override, direct/replay/HTTPS host resolution, session guard that clears history when `lang_config.session_id` changes |
| `caption_telemetry.js` | Wraps the socket handler; after a double animation frame sends `caption_rendered` with `event_id`, `receive_to_render_ms`, `visible` |
| `operator/` | Control-plane SPA: `index.html`, `app.js`, `style.css`, `review.js` (live/post-session Review), `widgets/sparkline.js` |

## Connection

Displays resolve the WebSocket host from the page location (same LAN host, port 8765
unless `?port=` is given) and reconnect automatically. On connect the server sends
`lang_config`; a new `session_id` in that message resets the caption history so chunk
ids from a previous session cannot collide with the new one.

## Message types (server → display)

| `type` | When | Key fields |
|--------|------|------------|
| `lang_config` | On connect and on language flip | `session_id`, `source_lang`, `target_lang`, `source_label`, `target_label` |
| `translation` (`stage: "partial"`) | Every ~0.6 s of new speech | `chunk_id` (utterance id), `english`, `spanish_a` (Marian), `spanish_b: null`, `stt_latency_ms`, `latency_a_ms`, `marian_pt_ms`, `timing_schema_version: 2`, sample bounds |
| `translation_start` | Final STT done, translation starting | `chunk_id`, `english`, `stage: "final"`, `stt_latency_ms`, `stt_confidence` |
| `translation_stream` | Token batches while a final translation streams (CUDA streaming engine; batch size `settings.cuda.streaming_batch_size`) | `chunk_id`, `partial_spanish_a`, `tokens_so_far` |
| `translation` (`stage: "complete"`) | Final ready | `chunk_id`, `english`, `spanish_a`, `spanish_b`, `stt_latency_ms`, `latency_a_ms`, `latency_b_ms`, legacy `e2e_latency_ms` / `true_e2e_ms` / `silence_delay_ms`, `queue_wait_ms`, `stt_confidence`, `tps_a`, `qe_a`, `word_stability_pct`, `speaker` (with `--diarize`), session provenance, schema 2 sample metadata |
| `speaker_update` | Diarization label arrives after a final (`--diarize`) | `chunk_id`, `speaker`, `session_id` |
| `music_hold` | Hymn/music detection toggles STT mute | `active: true\|false` |
| `rolling_stats` | Periodic session stats for the A/B display | `chunks`, `stt_avg_ms`, `a_avg_ms`, `true_e2e_avg_ms` |
| `text` | `--dry-run-text` and test markers | `text`, lang codes |

Every broadcast carries `session_id` and a monotonically increasing
`event_id = "<session_id>:<sequence>"`. Provenance fields (`session_kind` = `live`,
`replay` or `synthetic`; `audio_source`; `input_audio_path`; `input_audio_sha256`) are
attached to finals so downstream review and evaluation can separate live audio from
file replay.

The "spanish_*" field names are historical: in `--lang es` sessions `english` carries
the Spanish source and `spanish_a` the English target; use `lang_config` labels for UI.

## Timing semantics (schema 2)

Producer-side fields come from `tools/pipeline_timing.py` (`ChunkTiming.metrics()`):
`speech_end_to_final_ms` (last VAD-positive frame → final payload ready; `null` for
non-real-time replay), `vad_wait_ms`, `stt_queue_wait_ms`, `translation_queue_wait_ms`,
`finalization_overhead_ms`, `broadcast_ms`, `endpoint_reason`, `timing_source`.
Legacy `e2e_latency_ms` (submission → processing done) and `true_e2e_ms` (first
speech observed → processing done) remain for continuity and are **not**
speech-end-to-display.

Display-side: `caption_telemetry.js` replies `caption_rendered` after two animation
frames (a render opportunity, not a measured photon). `RenderTracker.acknowledge()`
records `receive_to_render_ms`, `send_to_ack_ms` and — only when the tab was visible —
`speech_end_to_ack_upper_bound_ms`, which includes the return network hop. Records land
in `metrics/display_metrics_<session>.jsonl`. Hidden tabs and accelerated replay cannot
satisfy caption-delivery gates; see
[`docs/evaluation/README.md`](../docs/evaluation/README.md).

## Operator SPA (port 9000)

`app.js` drives session start/stop/pause/resume, language flip, VAD threshold, fallback
toggle, preflight (`/api/preflight`), device lists (`/api/devices`,
`/api/audio/output-devices`), metrics (`/api/metrics`) and feature panels
(`/api/features/verses`, `/api/features/summary`). `review.js` edits finalized chunks
during or after a session through `/{session}/segments` and exports approved corrections
via `/{session}/export` — original predictions and audio are never rewritten.

**Known gap (2026-09-09):** session state showed RUNNING once the pipeline printed the
CSV header while the built-in microphone never delivered frames and the audience display
stayed disconnected. Status must derive from live audio/health signals; tracked as
`mac-live-mic-stall` and `overnight-reliability` in
[`docs/backlog.json`](../docs/backlog.json). Overnight operator-UI widgets
(captions/QR) are uncommitted in another worktree and not yet integrated.

## Access

```bash
open http://localhost:8080/displays/audience_display.html   # operator machine
open http://localhost:9000/operator/                         # control plane
# Phones: scan the QR on the audience display, or http://<LAN-IP>:8080/displays/mobile_display.html
```

Ports 8080/8765/9000 must be reachable on the LAN; displays reconnect on drops.

## Adding a display

1. Include `display_connection.js` and `caption_telemetry.js`; wrap your socket handler with the telemetry helper so ACKs keep working.
2. Handle `lang_config` first (labels + history reset), then `translation` partial/complete, `translation_start`, `translation_stream`, `speaker_update`, `music_hold`.
3. Treat `english` / `spanish_a` as source/target per `lang_config`, not by language.
4. Keep the page static (no build step); add it to `dry_run_ab.py`'s printed "Local displays" list if operators should see it.
5. Run the HTML5 Tidy check recorded in [`docs/mac_implementation_status.md`](../docs/mac_implementation_status.md) (zero warnings expected) before committing.
