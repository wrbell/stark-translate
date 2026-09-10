# Operator reliability and support

This describes the overnight reliability implementation. The integrated UI uses these APIs; this document does not claim physical microphone, speaker, reconnect, or Windows device acceptance. Those checks remain a separate rehearsal.

## Before a service

Select a runtime profile and language, then run preflight. Profiles listed by `GET /api/capabilities` are selectable software configurations; the list does not establish that their models or dependencies are installed. Preflight reports the effective configuration, and Start runs fresh server-side preflight again. A failed preflight returns HTTP 422 with `detail.code = preflight_failed` and the current checks. A stale green browser result cannot bypass it.

Use the input test while idle, with the actual microphone selected. It reports sample count, RMS and peak level; it does not save a recording. The output test plays a short quiet tone on the selected device. These tests, a live session, and a summary share an exclusive work reservation within one operator process. A conflicting request returns HTTP 409. The reservation is released after the owned subprocess exits, including cancellation and timeout.

A native audio-device open runs in a disposable child process. An unresponsive permission dialog or device cannot block the operator request indefinitely or freeze the pipeline's event loop. Input tests are limited to five seconds of requested audio, with three seconds of startup allowance; output tests request at most one second. Live capture reports a startup error after five seconds without samples and detects later missing samples. The operator cannot grant macOS microphone permission: allow the application in macOS settings, reconnect the selected device if needed, then retry.

## Starting, pausing and stopping

Starting shows loading/listening progress. `running` readiness requires actual capture samples and fresh pipeline health; creating a CSV header is insufficient. During a session, input level comes from the existing capture stream, so the dashboard does not open a second microphone. Ordinary caption preview is supplied even when speaker identification is off.

Pause and Resume use a session-scoped command file with an acknowledged sequence. Pause closes the live capture child. Resume opens a fresh child and retains the original capture sample coordinate system. Stop requests cooperative shutdown before escalating to process termination; the runner reaps owned children, including capture and runtime-server descendants.

Stop and Pause submit a valid buffered utterance before the translation coordinator drains. The existing minimum utterance length is retained. If queued capture audio must be discarded, or capture disconnects while speech is buffered, the recording is marked incomplete. Uncaught TTS synthesis/dispatch errors are visible in health without declaring otherwise saved captions lost. Device fallback/skip details remain in operational logs.

The final outcome has a specific meaning:

| Outcome | Meaning | Review/export |
| --- | --- | --- |
| `completed` | Captured input was observed, required writes drained successfully, and no required capture/STT/translation failure was recorded. The final diagnostics hash matches the completion marker. | Review drafts and export are available. |
| `interrupted` | The operator requested a stop but completion could not be proved, including stopping before input arrived. | Draft review remains available; export is blocked. |
| `failed` | The process failed, required recording/inference work failed, or a running process disappeared without final completion evidence. | Draft review remains available; export is blocked. |
| unknown legacy completion | No trustworthy lifecycle evidence exists, or diagnostics changed after completion. | Draft review remains available; export requires explicit verified migration. |

Do not treat exit code zero by itself as proof of a complete recording. Required asynchronous writes have a bounded ledger; failed, cancelled, or refused writes prevent completion. A failure cannot disappear merely because the short visible error history has filled up.

## Live health and recovery

`GET /api/session/status` and the existing control WebSocket expose `readiness`, `health`, `outcome`, `effective_profile`, and the active work reservation. Health is published about twice per second and becomes stale after three seconds. The UI must show stale readiness even if its last session state was running.

Health contains actual input/caption ages, level, recent captions, connected display count, capture/final/token queue depths, recording status, bounded recent errors, and cumulative failure counts. Queue depth is unknown until the pipeline supplies it. Low disk space is visible below 1 GiB free. Operational logging separately reports dropped log records and write failures.

Capture is bounded at the child, parent handoff, and asyncio queue. Only one parent drain is scheduled at a time. File replay backpressures its producer. A live microphone cannot be slowed indefinitely, so overflow is counted and makes the recording incomplete; original sample gaps remain visible in timing metadata. Device ADC time is mapped onto the host monotonic clock rather than replaced with a child wall-clock estimate.

If input stops, inspect the health error and selected device; pause/stop, correct the permission or connection, and retry. If recording is incomplete, retain the session for review and support. Starting a new session creates a new identity and does not reuse old corrections.

## Logs, storage and privacy

Application operational logs are asynchronous JSON lines with timestamp, logger, level, session/task correlation fields, event/stage, and a redacted message. Each file rotates at 20 MiB with five backups. Its queue is bounded at 2,048 records; overload is counted instead of blocking capture. Repeated logger configuration does not add duplicate file handlers. These operational logs may contain spoken text from normal pipeline output, so treat the local files as private. Correlation fields may be null for process-wide events.

Thirty-day pruning applies to rotated application logs and operational session logs belonging to proven completed sessions. It does not delete original audio, predictions, diagnostics, corrections, or exports. Legacy launchd/service stdout and stderr files have separate service-manager behavior; the application rotation limit is not a cap on those files or on total recording storage.

Storage inspection reports disk capacity/free space and per-session removable operational-log bytes. Cleanup is explicit: preview completed sessions, inspect the file list, then confirm that preview. Completion and file hashes are checked again before deletion. Unknown, failed and running session logs cannot be removed by this cleanup API. Originals are always preserved.

`record_audio:false` disables saving original chunk WAVs. It does not mean no audio is stored: enabled speaker identification still uses a rolling WAV, and TTS `wav`/`both` modes save synthesized audio. Use the setting label “Save original chunk audio”.

Support exports are local ZIP downloads. By default they contain allowlisted metadata, model revision/hash evidence, storage state, and bounded error codes. They exclude transcript text, audio, raw operational log messages, environment variables, local model paths, and user home paths. Text and original chunk audio require explicit opt-in followed by a preview. Attachment totals are limited to 100 MiB; changed attachments invalidate the preview. Preview tokens expire after ten minutes and are bound to the installation and action. Symlink paths are rejected.

A support bundle is not sent anywhere automatically. Review its preview before downloading or sharing it. The in-process work reservation does not coordinate unrelated CLI sessions or multiple operator server instances; run one operator per installation.

## API contract

| Request | Body / response |
| --- | --- |
| `GET /api/capabilities` | Selectable `profiles`, `default_profile`, `preflight_required`, `audio_tests`, `audio_tests_require_idle`, `audio_devices_validated:false`, `support`, `storage`, effective `display_ports` and `audience_urls`. |
| `GET /api/preflight` | Query selected `profile`, `backend`, `lang`, `tts`, `diarize`, `input_device`; returns checks and effective profile/configuration. |
| `POST /api/session/start` | Selected configuration, including `profile` and `record_audio`. Explicit profile selection overrides inherited `STARK_PROFILE`. |
| `POST /api/audio/test-input` | `{device, duration_s}`; returns `ok`, `rms`, `peak`, `samples`, `recorded:false`. |
| `POST /api/audio/test-output` | `{device, duration_s}`; returns `ok`. |
| `GET /api/storage` | Capacity, free space, session status, removable operational-log bytes and cleanup scope. |
| `POST /api/support/preview` | `{session_id, include_text:false, include_audio:false}`; returns `preview_id`, file list, attachment bytes and privacy description. |
| `POST /api/support/export` | `{preview_id}`; returns local `download_url`. |
| `POST /api/storage/cleanup/preview` | `{session_ids:[...]}`; returns preview ID, file list, bytes and originals-preserved flag. |
| `POST /api/storage/cleanup` | `{preview_id}`; returns removed bytes and originals-preserved flag. |

## Validation boundary

The focused CPU checks use fake device children and real small files/processes. They cover blocked native startup, timestamp/sample gaps, bounded producer handoff, writer failure and overload, Stop/Pause/disconnect outcomes, child-group cleanup, lease cancellation, fresh preflight, support opt-in/symlink/changed-file guards, log rotation, and installed-source provenance. They do not load models or exercise physical audio devices.

Runtime provenance records a checkout SHA only when the executing `dry_run_ab.py` is tracked at that repository's root. A wheel installed under a checkout's cache records its own Python source hash and package/artifact metadata, with `git_sha:null`; the ancestor checkout's HEAD is not treated as the wheel's revision.
