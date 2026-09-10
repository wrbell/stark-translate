# Operator page (`/operator/`)

Vanilla HTML/CSS/JS served by `operator_app/main.py`. No build step. The page is
organised for a non-technical volunteer:

| Tab | What it holds |
|---|---|
| **Prepare** | Speaker language, microphone, optional spoken translation (TTS routing), readiness checks in plain words, optional performance profile, the Start button. |
| **Live** | Large Pause / Resume / Stop, "switch speaker language" (restarts captions), caption preview, audience-display open / phone link / QR, Bible references heard, plain status and health lines. |
| **Sessions** | The review/corrections workspace (unchanged semantics: drafts, revisions, approvals, export), session summary, disk space and cleanup. |
| **Help** | Troubleshooting in plain language and the support-bundle builder. |
| **Advanced** | Backend, engine, VAD threshold, A/B, live diarization, HF fallback, raw metrics sparklines, raw status / readiness / capabilities JSON. |

Files: `index.html`, `style.css`, `app.js` (page controller, exported as
`StarkOperator`), `review.js` (review workspace, unchanged), `widgets/sparkline.js`,
`widgets/qr.js` (offline QR encoder, byte mode ECC-L v1–9), `widgets/captions.js`
(read-only caption model + reconnecting client).

## Behaviour rules

- **Readiness**: a failed or errored `/api/preflight` clears the ready flag at once;
  Start is disabled until a fresh check passes. Readiness is not polled while a
  session is active; a language switch or a stop triggers a recheck.
- **Connection**: a failed or missing `/api/session/status` poll (or silence for
  more than 6 s) shows **Not connected** with the last known state and age. It
  never leaves a green **Live**. Pause/Resume/Switch/Start are disabled while
  stale; **Stop** stays enabled so a live process can still be recovered.
- **Errors**: `state == "error"` shows a persistent banner with the error text and a
  **Stop and reset** button (Stop stays enabled in the error state). Failed
  requests are reported with the server's detail; nothing is shown as success
  without a server response saying so.
- **Ordering**: every poll and probe carries a request counter; a slower, older
  response never overwrites a newer selection, status, listing, or preview.
- **Caption preview** connects read-only to the caption WebSocket
  (`ws://<page host>:8765`, override via `?caption_port=` or capabilities). It
  never sends `caption_rendered` acknowledgments, so it cannot affect the
  audience-display timing measurements. It does not depend on diarization.
- **Persistence**: TTS choices (`stark-translate-tts-outputs`), the selected tab,
  the chosen profile and the review selection/drafts live in `localStorage`.

## Backend contract

Required (v2026.6 contract, unchanged): `/api/preflight`, `/api/devices`,
`/api/audio/output-devices`, `/api/session/{status,start,stop}`,
`/api/control/{pause,resume,lang_flip,fallback}`, `/api/features/*`,
`/api/review/*`, `/ws/control`.

Optional. Every item below is capability-gated: a 404/405/501 or a missing field
degrades to explicit "not available" text.

### `GET /api/session/status` additive fields

| Field | Rendering |
|---|---|
| `outcome` (`null`/`completed`/`interrupted`/`failed`) | Status pill detail and Live subtitle after a session ends. |
| `readiness {phase, ready, reason, updated_at, age_s, stale}` | "Getting ready: <phase>" line on Live; `stale` prefixes the line. |
| `health {queues, input_age_s, caption_age_s, clients, errors, persistence, recording}` | Health chips on Live; only present keys are shown. |
| `effective_profile` | Shown next to the language on Live. |
| `work {kind, id}` | "Busy" pill; Start disabled with an explanation while set. |
| `capabilities` (same shape as `GET /api/capabilities`) | Applied without a separate probe. |

### Start errors

- `422 {detail: {code: "preflight_failed", message, checks}}` — clears the ready
  flag, renders `checks` in Prepare, switches to Prepare.
- `409 {detail: {code: "work_busy", message, work}}` — "operator service is busy" notice.
- Legacy `409 {detail: "<string>"}` and framework `422 {detail: [...]}` still render.

### `GET /api/capabilities` (optional)

```json
{
  "profiles": ["full", "lite-cpu", "lite-cuda-8gb"],
  "audience": {"lan_host": "192.168.1.20", "http_port": 8080, "ws_port": 8765,
               "audience_url": "...", "mobile_url": "..."},
  "audio": {"input_test": true, "output_test": {"url": "/api/audio/output-test"}},
  "support": {"preview": true}, "storage": {"cleanup": true}
}
```

- `profiles`: strings or `{id, label, description, available}`; only `full`,
  `lite-cpu`, `lite-cuda-8gb` are offered. A chosen profile is sent as
  `profile` in the start body only when the server advertises profiles.
- `audience`: overrides the page-derived audience/mobile links. Without it the
  links use the page hostname; `localhost` is flagged as not shareable and no
  QR is drawn.
- `audio.input_test` / `audio.output_test`: `true` enables the default URL
  (`/api/audio/input-test`, `/api/audio/output-test`); an object with `url`
  chooses the endpoint. Requests: `POST {device, seconds}` /
  `POST {device, lang}`. Expected reply: `{state: "done"|"failed"|"pending",
  level_peak?, level_rms?, message?, poll_url?}`. Missing levels are reported as
  "did not report a level", never as a pass. This is the requested contract;
  until the server advertises it the buttons stay disabled/hidden.

### Storage and support (optional)

- `GET /api/storage -> {free_bytes, used_bytes, sessions}` (`sessions` may be a
  count or a list of `{session_id|session, bytes, status}`).
- `POST /api/storage/cleanup/preview {session_ids} -> {preview_id, files, bytes}`
  then `POST /api/storage/cleanup {preview_id}`; the delete button only enables
  after a successful preview.
- `POST /api/support/preview {session_id, include_text, include_audio} ->
  {preview_id, files, bytes, privacy}` then `POST /api/support/export {preview_id}
  -> {download_url}`. Changing the session or the include boxes invalidates the
  preview; a late export reply cannot re-enable an outdated link.

## Tests

`tests/test_operator_layperson_ui.py`, `tests/test_operator_language_ui.py`,
`tests/test_operator_contracts.py::test_startup_buttons_allow_stop_but_not_pause`
and `tests/test_operator_review_browser.py` run the shipped JavaScript under Node
(`tests/frontend/fake_dom.js` parses the real `index.html`;
`tests/frontend/operator_harness.js` routes fetch/WebSocket fakes). Browser
rendering, real audio devices and real inference are not covered here.
