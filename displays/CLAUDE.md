# displays/ — Browser Displays, WebSocket Protocol, Operator SPA

> Paired with [`AGENTS.md`](./AGENTS.md). Protocol statements come from
> `dry_run_ab.py` (`broadcast()`, `_session_provenance()`), `tools/pipeline_timing.py`
> and the display JS. Integration and release state are recorded in
> [`Mac implementation status`](../docs/mac_implementation_status.md).

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
| `utterance_discarded` | A provisional utterance is abandoned | `session_id`, `utterance_id`, `reason`; remove only its partial and suppress late matching partials, never a final or final stream |
| `translation_start` | Final STT done, translation starting | `chunk_id`, `utterance_id`, `english`, `stage: "final"`, `stt_latency_ms`, `stt_confidence` |
| `translation_stream` | Token batches while a final translation streams (CUDA streaming engine; batch size `settings.cuda.streaming_batch_size`) | `chunk_id`, `partial_spanish_a`, `tokens_so_far` |
| `translation` (`stage: "complete"`) | Final ready | `chunk_id`, `english`, `spanish_a`, `spanish_b`, `stt_latency_ms`, `latency_a_ms`, `latency_b_ms`, legacy `e2e_latency_ms` / `true_e2e_ms` / `silence_delay_ms`, `queue_wait_ms`, `stt_confidence`, `tps_a`, `qe_a`, `word_stability_pct`, `speaker` (with `--diarize`), session provenance, schema 2 sample metadata |
| `speaker_update` | Diarization label arrives after a final (`--diarize`) | `chunk_id`, `speaker`, `session_id` |
| `music_hold` | Energy/VAD heuristic hold state; can miss singing, not a music classification | `active: true\|false` |
| `rolling_stats` | Periodic session stats for the A/B display | `chunks`, `stt_avg_ms`, `a_avg_ms`, `true_e2e_avg_ms` |
| `text` | `--dry-run-text` and test markers | `text`, lang codes |

Every broadcast carries `session_id` and a producer-assigned `event_id` scoped to
that session. Preview IDs include a `:partial:` component; consumers treat event
IDs as opaque identities rather than parsing them as chunk numbers. Provenance fields (`session_kind` = `live`,
`replay` or `synthetic`; `audio_source`; `input_audio_path`; `input_audio_sha256`) are
attached to finals so downstream review and evaluation can separate live audio from
file replay.

The first `translation_stream` batch per chunk uses the deterministic event ID
`<session_id>:stream:<chunk_id>`; later batches retain ordinary counter IDs.
The telemetry wrapper acknowledges that first batch with `stage: "first_stream"`
only when its handler synchronously changes the DOM in a visible tab, using the
same double-animation-frame render opportunity. This once-per-client/chunk
measurement is reported separately from `complete` and is not a gate. A first
batch dropped by delivery coalescing, or ignored by a display, produces no ACK.

Capture `utterance_id` is distinct from final `chunk_id`. At authoritative final
publication the producer, delivery queue, health inventory and displays close the
explicit session/utterance identity against late partials. Ordinary previews can
still arrive while the final computes. Replacing that utterance's preview must
preserve a newer utterance already on screen; never substitute a final chunk
counter for a missing utterance identity. Legacy events without the identity keep
their historical display fallback. Experimental closure at final admission is a
separate opt-in scheduling policy.

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

`index.html` is organized for lay volunteers (start/stop captions, microphone and voice
choice, audience display / phone link with an inline QR, health list, caption preview,
troubleshooting `<details>`, support export). `app.js` drives session start/stop/pause/resume,
language flip, VAD threshold, fallback toggle, preflight (`/api/preflight`), capability
gating (`/api/capabilities`), device lists (`/api/devices`, `/api/audio/output-devices`),
idle-only device probes (`/api/audio/test-input`, `/api/audio/test-output` — native calls
run in disposable processes), status (`/api/session/status`, which now carries the pipeline
health `phase`/`ready`/`stale` fields), metrics (`/api/metrics`) and feature panels
(`/api/features/verses`, `/api/features/summary`). `widgets/captions.js`, `widgets/qr.js`
and `widgets/sparkline.js` are the integrated overnight widgets. `review.js` edits finalized
chunks during or after a session through `/api/review/{session}/segments` and exports
approved corrections via `/api/review/{session}/export` — original predictions and audio
are never rewritten. `/api/support/preview|export` builds scoped support bundles and
`/api/storage/cleanup` removes only regenerable completed-session logs
(`operator_app/support.py`).

**Readiness (source repaired 2026-09-10, full live acceptance pending):** on 2026-09-09 the SPA showed RUNNING
once the pipeline printed the CSV header while the built-in microphone never delivered
frames. `PipelineRunner` now reads `tools/pipeline_health.py` (`loading → listening →
ready` on the first input frame, `paused`, `input_error`; `stale` after 3 s without a
heartbeat) and reports `ready` only from that channel; the pipeline itself fails a capture
that delivers no samples within 5 s (`tools/isolated_audio.py`). Polls that return an
unchanged status do not churn the live region. Real quiet-room EN/ES readiness and
synthetic acoustic checks subsequently ran. A Spanish retest retained 160 ms of
upstream sample loss; the later device-identity probe did not run STT. See
[actual device evidence](../docs/evaluation/tts_routing_20260910/README.md) and
`mac-live-mic-stall` / `issue-131-smoke` in [`docs/backlog.json`](../docs/backlog.json).
The current user instruction prohibits further microphone/output tests this session.

## Access

```bash
open http://localhost:8080/displays/audience_display.html   # operator machine
open http://localhost:9000/operator/                         # control plane
# Phones: scan the QR on the audience display, or http://<LAN-IP>:8080/displays/mobile_display.html
```

Audience phones need LAN access to ports 8080/8765; displays reconnect on drops.
The operator control plane on port 9000 binds to loopback by default and does not
need LAN exposure for audience captions.

## Adding a display

1. Include `display_connection.js` and `caption_telemetry.js`; wrap your socket handler with the telemetry helper so ACKs keep working.
2. Handle `lang_config` first (labels + history reset), then `translation` partial/complete, `utterance_discarded`, `translation_start`, `translation_stream`, `speaker_update`, `music_hold`.
3. Treat `english` / `spanish_a` as source/target per `lang_config`, not by language.
4. Keep the page static (no build step); add it to `dry_run_ab.py`'s printed "Local displays" list if operators should see it.
5. Run the HTML5 Tidy check recorded in [`docs/mac_implementation_status.md`](../docs/mac_implementation_status.md) (zero warnings expected) before committing.
