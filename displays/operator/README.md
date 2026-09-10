# Operator page (`/operator/`)

Vanilla HTML/CSS/JS served by `operator_app/main.py`. No build step. The page is
organised for a non-technical volunteer:

| Tab | What it holds |
|---|---|
| **Prepare** | Speaker language, microphone, optional spoken translation (TTS routing), microphone/speaker tests, readiness checks in plain words, performance profile, the Start button. |
| **Live** | Large Pause / Resume / Stop, "switch speaker language" (restarts captions), sound level, health chips, caption preview, audience-display open / phone link / QR, Bible references heard. |
| **Sessions** | The review/corrections workspace (unchanged semantics: drafts, revisions, approvals, export), session summary as bilingual text, disk space and log cleanup. |
| **Help** | Troubleshooting in plain language and the support-bundle builder. |
| **Advanced** | Backend, engine, VAD threshold, A/B, live diarization, HF fallback, raw metrics sparklines, raw status / readiness / capabilities / summary JSON, other display pages. |

Files: `index.html`, `style.css`, `app.js` (page controller, exported as
`StarkOperator`), `review.js` (review workspace, unchanged), `widgets/sparkline.js`,
`widgets/qr.js` (offline QR encoder, byte mode ECC-L v1–9), `widgets/captions.js`
(read-only caption model + reconnecting client).

## Behaviour rules

- **Readiness**: a failed or errored `/api/preflight` clears the ready flag at once;
  Start is disabled until a fresh check passes. Readiness is not polled while a
  session is active; a language switch, a stop, a profile change or newly
  advertised capabilities trigger a recheck. The preflight query carries the
  same `profile` the start request will send, so both agree.
- **Connection**: a failed or missing `/api/session/status` poll (or silence for
  more than 6 s) shows **Not connected** with the last known state and age. It
  never leaves a green **Live**. Pause/Resume/Switch/Start are disabled while
  stale; **Stop** stays enabled so a live process can still be recovered.
- **Errors**: `state == "error"` shows a persistent banner with the error text and a
  **Stop and reset** button (Stop stays enabled in the error state). Failed
  requests are reported with the server's detail; nothing is shown as success
  without a server response saying so.
- **Acknowledged controls**: pause/resume answer with the unchanged state and a
  `last_event`; the page says "Pausing…/Resuming… waiting for the caption process
  to confirm", disables the button, and only flips when a later poll reports the
  new state (after 15 s it adds "No confirmation yet").
- **Ordering**: every poll and probe carries a request counter; a slower, older
  response never overwrites a newer selection, status, listing, or preview.
- **Caption preview**: the caption WebSocket (`ws://<page host>:<display_ports.websocket>`,
  default 8765) is used read-only when open; otherwise `health.captions` from the
  status poll is shown. It never sends `caption_rendered` acknowledgments, so it
  cannot affect audience-display timing, and it does not depend on diarization.
- **Profiles**: the empty option follows the server's `default_profile`
  (`STARK_PROFILE`, e.g. a Lite launcher). An explicit persisted choice
  (`stark-operator-profile`; legacy `full` maps to `standard`) overrides it only
  while the server still lists that profile. A Lite profile forces backend `auto`
  and disables backend/engine/A/B/diarization so no contradictory (e.g. MLX)
  request is sent. Selecting a profile is not a claim its models are installed:
  readiness decides.
- **Audio tests** are real device probes behind the work lease: Start is disabled
  while one is pending, the measured `peak`/`rms` is shown as a level, and a
  speaker test only says the tone was played and asks the operator to confirm
  they heard it.
- **Summary**: bilingual text plus format/notice; an excerpt (`metadata.content_mode
  == "excerpt"`) is labelled as recorded text, not a summary, and every result
  notes it was not reviewed by a person. Failures stay visible with **Try again**;
  a running task can be cancelled. Raw JSON only appears under Advanced.
- **Persistence**: TTS choices (`stark-translate-tts-outputs`), the selected tab,
  the chosen profile and the review selection/drafts live in `localStorage`.
- **Live regions**: identical text is never rewritten, so polling does not make
  screen readers re-announce the status pill or readiness summary.

## Backend contract

Required (v2026.6 contract, unchanged): `/api/preflight`, `/api/devices`,
`/api/audio/output-devices`, `/api/session/{status,start,stop}`,
`/api/control/{pause,resume,lang_flip,fallback}`, `/api/features/*`,
`/api/review/*`, `/ws/control`.

Everything below is optional. A 404/405/501, a `false` flag, or a missing field
degrades to explicit "not available" text.

### `GET /api/capabilities`

```json
{
  "profiles": ["standard", "lite-cpu", "lite-cpu-quality", "lite-cuda-8gb"],
  "default_profile": "standard",
  "preflight_required": true,
  "audio_tests": true,
  "audio_tests_require_idle": true,
  "audio_devices_validated": false,
  "support": true,
  "storage": true,
  "display_ports": {"http": 8080, "websocket": 8765},
  "audience_urls": {"audience": "http://host:8080/displays/audience_display.html?port=8765",
                    "church": "...", "mobile": "...", "obs": "..."}
}
```

`profiles` are ids the installed code can select (objects with
`{id, label, description, available}` are also accepted). `audience_urls` are
built from the request host, so a page opened as `localhost` still gets a
non-shareable link and no QR code. `audio_devices_validated: false` shows a note
that the devices have not been validated yet. The same object embedded as
`capabilities` in the status snapshot is honoured.

### `GET /api/session/status` additive fields

| Field | Rendering |
|---|---|
| `outcome` (`null`/`completed`/`interrupted`/`failed`) | Status pill detail and Live subtitle after a session ends. |
| `readiness {phase, ready, reason, updated_at, age_s, stale}` | "Getting ready: <phase>" (phases such as `loading`, `listening`, `ready`, `paused`, `input_error`); `stale` becomes "has not reported for …". |
| `health` (`tools/pipeline_health.py`): `input_seen`, `input_age_s`, `caption_age_s`, `input_level`, `captions[]`, `errors[]`/`error_count`, `recording {audio_enabled, required_failures, ok}`, `persistence {ok,…}`, `storage {free_bytes, low_space}`, `queues{}`, `clients`, `publish_failures` | Sound level meter, health chips, status-feed caption preview. Legacy booleans/numbers still render. |
| `effective_profile {name}` (or a string) | "profile: …" next to the language on Live. |
| `work {kind, id}` | "Busy" pill; Start, audio tests and summary wait while set. |
| `last_event` | Small line under the Live title while active or in error. |

### Start errors

- `422 {detail: {code: "preflight_failed", message, checks}}` clears the ready
  flag, renders `checks` in Prepare, switches to Prepare.
- `422 {detail: {code: "profile_unavailable", message}}` "This profile is not installed".
- `422 {detail: "<string>"}` (profile/backend conflict from `session_overrides`)
  "The selected settings can't be used together".
- `409 {detail: {code: "work_busy", message, work}}` and plain-string 409s are shown as notices.

### Audio tests (`operator_app/audio_tests.py`)

`POST /api/audio/test-input {device, duration_s: 2}` →
`{ok, device, duration_s, rms, peak, samples, recorded: false}`;
`POST /api/audio/test-output {device, duration_s: 0.4}` → `{ok, device, duration_s}`.
`409 {code: work_busy}` and `422 {code: audio_unavailable, message}` are shown
verbatim. `device` is the selected index or `null` for the computer default.

### Summary (`/api/features/summary`)

Task `{task_id, state, return_code, error, result}`; `result` is
`{english, spanish, format, translation_method, notice?, metadata {content_mode,
human_reviewed, total_words, …}}`. `POST /api/features/summary/{id}/cancel` returns
the task. `409 {code: work_busy}` is explained.

### Storage and support (`operator_app/support.py`)

- `GET /api/storage -> {free_bytes, used_bytes, total_bytes, low_space, sessions:
  [{session_id, status, cleanup_bytes, originals_preserved}], cleanup_scope}`.
  Only sessions with `status == "completed"` can be selected for cleanup.
- `POST /api/storage/cleanup/preview {session_ids} -> {preview_id, files, bytes,
  originals_preserved}` then `POST /api/storage/cleanup {preview_id} ->
  {removed_bytes, originals_preserved}`; the delete button only enables after a
  successful preview.
- `POST /api/support/preview {session_id, include_text, include_audio} ->
  {preview_id, files, bytes, privacy {text_included, audio_included, message}}`
  then `POST /api/support/export {preview_id} -> {bundle_id, download_url}`.
  `session_id` is required: the page uses the selected session, else the most
  recent review session, else the current status session. Changing the session
  or the include boxes invalidates the preview; a late export reply cannot
  re-enable an outdated link.

## Tests

`tests/test_operator_layperson_ui.py`, `tests/test_operator_language_ui.py`,
`tests/test_operator_contracts.py::test_startup_buttons_allow_stop_but_not_pause`
and `tests/test_operator_review_browser.py` run the shipped JavaScript under Node
(`tests/frontend/fake_dom.js` parses the real `index.html`;
`tests/frontend/operator_harness.js` routes fetch/WebSocket fakes and provides
`realCapabilities()` / `realHealth()` fixtures matching the current backend).
Browser rendering, real audio devices and real inference are not covered here.
