# Operator runbook

This guide is for the volunteer running English/Spanish captions after the setup
owner has installed and rehearsed the system. The normal event workflow uses a
browser. The setup owner's commands are kept at the end.

## Before people arrive

1. Plug in the microphone, projector, and any speakers used for spoken translation.
   Use the connections that were rehearsed. Keep the computer connected to power.
2. Open the installed operator shortcut, or its saved browser bookmark. The usual
   address on that computer is `http://localhost:9000/operator/`.
3. In **Prepare**, choose **Speaker's language** and the microphone by name. English
   speakers produce Spanish captions; Spanish speakers produce English captions.
4. Keep the performance profile selected by the setup owner. **Standard** is the
   normal Apple Silicon configuration. Lite choices are for installations prepared
   for those models; selecting a profile does not install them.
5. Read **Is everything ready?** Clear red failures before starting. A warning can
   mean base models are in use rather than custom vocabulary; it is not necessarily
   a fault. **Check again** repeats the checks after a setup change.
6. Use **Test microphone** while captions are stopped. Speak normally and check the
   measured level. If macOS asks for microphone permission, allow the installed
   launcher and retry. A successful level test does not check translation quality.
7. If spoken translation is wanted, expand **Spoken translation (optional)**. Select
   its destination and the rehearsed language/output devices. **Test speakers**
   checks playback; someone must confirm the intended speakers actually made sound.
   A reported playback success alone cannot identify what people heard.
8. Click **Start captions**. Keep watching the startup explanation until audio is
   arriving, then say a complete test sentence and wait for its final translation.

The target is under one minute of volunteer interaction once installed; model
loading and the spoken rehearsal take additional time. Cold startup may take
longer. Preflight checks local prerequisites without loading every model, so green
checks do not replace this rehearsal.

## Projector and phones

In **Live**, use **Open audience display** for the projector. Put that browser on
the projector and enter full screen using the browser's controls.

The **Projector and phones** area provides the phone link and QR code. Phones must
be on a network that can reach the caption computer. A link containing `localhost`
or `127.0.0.1` only works on that computer. Use the LAN audience-display bookmark
prepared by the setup owner, then click that display's header for its phone QR code.
The setup owner can replace the loopback host in the audience URL with the
computer's LAN address, retaining its display port and `?port=` query. Keep the
operator controls on their default local address. Custom ports come from the service.
The caption display reconnects automatically between sessions.

The operator's **What the audience sees** preview helps check content. Also inspect
one actual audience display. The preview is not proof of projector or phone output.

## While the speaker talks

- **Live** shows input activity, recent captions, session state and any required
  action. Italic text is a revisable preview. A final caption replaces it.
- **Pause** stops accepting new speech for captions and finishes valid buffered
  speech. Use it during a hymn or an intentional break. **Resume** continues in
  the same session. Do not speak essential content during the pause.
- **Switch to Spanish speaker** (or English) restarts the pipeline in the other
  direction. Wait through loading and test the first sentence. The new direction
  gets a new session identity; old captions and correction records stay separate.
- **Bible references heard** shows detected verses. Treat detection as an aid;
  confirm the passage before presenting it as authoritative.
- Leave technical controls in **Advanced** to the setup owner. Changing a backend
  or profile in the middle of an event can require a new model load.

**Starting** means preparation is still underway. **Listening** requires actual
input frames. The first translated sentence confirms that audio, recognition,
translation and delivery have worked together. A file or CSV header does not make
a session ready. If input becomes stale, the page shows the fault and recovery
step rather than continuing to claim healthy operation.

## End the session

Click **Stop captions** and wait for completion. Buffered speech and final writes
must drain before the session can be marked completed. Do not quit the launcher
while it says it is stopping.

A completed session has persisted completion evidence. An interrupted or failed
session remains available for diagnosis and text review, but cannot be exported as
successfully completed training data. Closing the browser does not by itself stop
captions; use **Stop captions** first.

## Review and corrections

Open **Sessions** and choose the session to review. During recording, only finalized
segments whose diagnostic records are already saved are shown. New arrivals do not
replace an edit in progress. Use the priority filters, neighboring context and audio
player to find and understand a segment.

Edit the transcript and translation independently. Transcript approval means you
checked the spoken words; translation approval means you checked the bilingual
meaning. Neither implies the other. Mark unsuitable material excluded. Use **Save
draft** or **Save and next**, and check that the save succeeded.

A local draft can survive a browser reload. **Restore my local draft** lets you
recover it; the server save remains the durable shared record. A revision conflict
means somebody saved a newer version: reload and reconcile it instead of silently
replacing their work.

Corrections are separate revisioned records. They do not rewrite original
predictions, change audience captions, or trigger a new translation. Missing audio
still permits text review, but prevents an STT training sample.

### Export reviewed material

After a session completes, select **Training — reviewed live audio** or
**Evaluation — held apart from training**, then **Export completed session** and
download the bundle. Training requires known live provenance. Replay, synthetic
and unknown sessions remain evaluation material. A session's assigned split is
preserved across repeated exports.

An approved transcript plus retained audio is required for STT export. A bilingual
pair requires both transcript and translation approval; missing audio does not
prevent that text-only pair. Excluded chunks are omitted. The portable ZIP includes
copied audio, language direction, original predictions, correction revisions and
provenance. Repeated unchanged exports and imports preserve identity without adding
duplicates. Keep Spanish STT material separate from an English-only trainer.

### Session summary

With captions stopped, choose **Create summary**. Read the English and Spanish text
before sharing it. Summaries are model output, not human approval. Very short
sessions show the actual recorded text as a labeled excerpt rather than inventing
a longer sermon. Long transcripts use their beginning and end; the result explicitly
states when the middle was omitted. Failed generation remains visible and can be
retried or canceled.

## When something goes wrong

| What you see | What to do |
|---|---|
| Start is unavailable | Read the failed readiness check. Reconnect the required device or ask the setup owner to repair missing software/models, then **Check again**. |
| Starting never reaches input readiness | Read the displayed reason. Check the operating system's microphone permission, then stop and retry. Do not assume a running process can hear you. |
| Microphone unplugged or input stale | Pause or stop, reconnect and reselect the intended device, test it, then restart and say a test sentence. |
| Operator disconnected or status stale | Wait for reconnect; reload if needed. Avoid repeated Start clicks. Check the launcher if reconnect fails. |
| Audience display reconnecting | Confirm the session is listening, the link/ports are correct, and the display can reach the caption computer. Reload that display if needed. |
| Captions fall behind | Pause at a suitable break and close competing compute-heavy applications. Ask the setup owner to inspect queues and timing; do not switch models blindly. |
| Recording incomplete or a write failed | Keep the session and its original files. Create a support bundle; do not claim it completed or delete evidence. |
| Low disk space | Stop before storage is exhausted. Use the explicit cleanup preview for completed operational logs. Original recordings need the setup owner's separate retention decision. |

**Help** contains these recovery instructions and support controls. Choose the
session, leave text/audio unchecked for a metadata-only bundle, click **Preview
bundle**, inspect the listed files, then **Create bundle** and download it. Nothing
is sent automatically. Opting into caption text or audio makes the bundle more
sensitive; share only what the setup owner needs.

**Sessions → Disk space** offers a cleanup preview. **Delete the listed files**
removes only eligible completed-session operational logs that still match the
preview. It never automatically deletes original audio, predictions, corrections,
or exports. Keep original data until its retention has been agreed.

## Setup owner reference

Before an event, rehearse a complete hymn, spoken material including a reference
such as John 3:16, a pause and a language switch. Time setup to the first actual
caption, check the intended audience screen, and record anything that failed.
The [API rehearsal script](../scripts/dry_run_rehearsal.sh) walks the control
endpoints; a successful script run does not replace that audio/browser rehearsal.
It starts a session, so use it only with the intended, rehearsed input source.

The `/healthz` route checks service liveness. The usual audience path is
`/displays/audience_display.html`; use the service-provided URL for custom ports.
Local operational evidence is under `metrics/`. Liveness and log-file existence
do not prove that captions are being generated.

- [Mac installation and launchd](packaging/macos.md): selected environment, model
  setup, installed entry points and explicit service install/uninstall.
- [Lite profiles](lite_profiles.md): CPU Marian, optional CPU E2B, and 8 GB NVIDIA
  setup; actual x86/RTX2070 certification is separate from Mac CPU tests.
- [Reliability and support contracts](operator_reliability.md): capture isolation,
  work ownership, health, logging limits, privacy, and API payloads.
- [Evaluation definitions](evaluation/README.md): current measurements and dated
  evidence. [Backlog](backlog.md) distinguishes implementation from validation.

Run one operator service for an installation. It coordinates sessions, summaries
and audio tests, but cannot reserve resources used by unrelated command-line jobs.
An installed package starts with `stark-translate operator`; a checkout starts with
`./run_operator.sh`. The setup owner should supply a working shortcut/service so
volunteers do not need a terminal. Do not replace a working `stt_env` to test an
installation. A dedicated `STARK_PROJECT_ROOT` holds writable session data when
running an installed package outside the checkout.

Operator controls and private recordings are local by default (`127.0.0.1:9000`).
The separate audience ports remain reachable on the LAN and serve only public
display assets and live captions. Browser-origin and Host checks protect the
operator from foreign pages; they are not account authentication. Explicit
`stark-translate operator --host 0.0.0.0` enables unauthenticated remote operator
access and prints its exposure warning. Use a restricted network or authenticated
proxy for that configuration; custom proxy hostnames must be listed explicitly in
`STARK_OPERATOR_ALLOWED_HOSTS`. Audience phones do not require remote operator access.

Operational application logs rotate at 20 MiB with five backups. Thirty-day
pruning applies only to eligible operational logs, not originals; service-manager
stdout/stderr files have their own policy. This is not a bound on total audio or
disk use. The recording switch means **Save original chunk audio**: optional
speaker identification can still create a rolling WAV, and TTS file modes save
synthesized audio.

Current schema-2 server latency measures estimated speech end to final payload
readiness, including endpoint wait, queues, recognition and translation. Browser
receipt-to-render is reported separately. Speech-end-to-acknowledgement is an upper
bound including return-network time. Hidden tabs and accelerated replay cannot
pass the live delivery gate. A dash means unavailable, not zero. Historical
processing timings must not be pooled with these measurements.

After extracting a portable review bundle, the training owner can use:

```bash
python tools/merge_corrections.py whisper --corrections /path/to/bundle --train-dir /path/to/english/train
python tools/merge_corrections.py whisper --corrections /path/to/bundle --train-dir /path/to/spanish/train --language es
python tools/merge_corrections.py translation --corrections /path/to/bundle/translation/train.jsonl --train-jsonl /path/to/pairs_train.jsonl
```

Evaluation bundles are rejected by training mergers. Retain the bundle manifest
with any training run. Older sessions without successful completion evidence remain
reviewable; do not invent completion or human approval to export them.

Live microphone permission, output routing/unplug/replug, a real two-speaker
recording, natural Spanish references and bilingual approval remain explicit
rehearsal gates. Controlled file replay does not certify these physical or human
steps. The September 10 overnight work defers live microphone and physical outputs
until the user returns.
