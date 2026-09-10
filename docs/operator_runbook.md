# Operator Runbook — stark-translate

> **Audience:** the volunteer running live translation at Stark Road Gospel Hall (Farmington Hills, MI) or a coffee-shop outreach event. Assumes you can open a web browser; no command line required.

---

## What you're running

The **stark-translate operator** turns a microphone in the room into live English/Spanish subtitles on a projector and any phones connected to the local Wi-Fi. Two screens matter:

1. **Operator screen** (your laptop or the church PC): `http://localhost:9000/operator/`
2. **Audience display** (projector / TV): `http://<server-ip>:8080/audience_display.html`

You only have to interact with the operator screen.

---

## Before the event (10 minutes)

### 1. Power on the church PC and the projector

- Wait ~30 seconds after boot for the operator service to come up automatically (systemd starts it at boot).
- If the service didn't auto-start, open a terminal and run:

  ```bash
  cd /opt/stark-translate
  ./run_operator.sh
  ```

  Leave the terminal window open during the event.

### 2. Open the operator UI

- On the church PC, open Firefox or Chrome and go to **http://localhost:9000/operator/**.
- You should see **Pre-flight**, **Session**, **Live observability**, **Features**, and **Review and corrections** panels.

### 3. Pre-flight: all green or yellow

Choose the **Language direction**, **Backend**, microphone, and optional TTS or diarization settings first. Pre-flight checks the selected configuration without loading models. `auto` selects MLX on Apple Silicon and CUDA on the NVIDIA machine.

| Check | Green means | Yellow means | Red means |
|---|---|---|---|
| GPU | CUDA or Apple Silicon detected | running on CPU (slow but works) | n/a |
| Runtime dependencies | The selected backend and optional features have compatible packages | n/a | a package is missing or incompatible |
| Models | The selected STT, translation, and Marian models are available locally | n/a | a required local model or TTS voice is missing |
| Microphone | Audio input is available and the selected device supports input | n/a | unavailable audio runtime or input device |
| Adapter manifest | `adapters/manifest.json` parses | manifest absent (using base models) | invalid JSON |
| llama-server (CUDA/CPU only) | server reachable on the configured port | server not running; the selected setup needs attention | n/a |

**Red means STOP — fix before the event.** Most common red:

- *Microphone* red: USB mic not plugged in. Plug it into the same port you used last time.

On a Mac, pre-flight checks MLX model caches and does not require GGUF files or a llama-server. Missing MLX models, Python packages, or audio input are red checks. The Start button stays disabled until red checks clear. A green pre-flight confirms local prerequisites; it does not replace a short audio rehearsal. Treat a fallback warning as something to rehearse before the event.

The setup owner can install the matching backend dependencies, then run `stark-translate setup --backend mlx` and `stark-translate doctor --backend mlx --lang en` on the Mac. Add `--include tts` to setup when voices are needed. See [Mac packaging and service setup](packaging/macos.md) for installation and launchd commands. The repository launcher, `./run_operator.sh`, uses the configured Python environment rather than requiring a fixed checkout location.

### 4. Pick the right mic

In the **Session** panel, the **Mic** dropdown lists every input device. Pick the USB lavaliere mic by name (e.g. *"Yeti Stereo Microphone"* — not *"Built-in Microphone"*).

### 5. (Optional) Open the audience display on the projector

- On the projector PC: `http://<church-pc-ip>:8080/audience_display.html`
- Press **F11** for full-screen.
- The display will say *"Connecting…"* until you start a session — that's normal.

---

## During the event

### Start the session

1. Confirm **Language direction**: usually `EN → ES` for the speaker.
2. Click **Start session**. The state pill at the top right turns yellow (`STARTING`) then green (`RUNNING`).
3. Wait for model startup and speak a short test sentence. `RUNNING` means the subprocess launched; the first subtitle confirms audio and inference are ready. A cold start can take longer than later sessions.

### Watch the dashboard while the speaker talks

- **VRAM / CPU** sparklines show how hard the machine is working. Steady is fine; spiking is fine.
- **Latency p50 / p95** for current sessions measures the last detected speech frame to the final caption being ready to send. It includes endpoint waiting, queueing, recognition, and translation. It does not include browser/network delay. Hover over the value to see whether the data uses this definition or a historical pipeline timing definition. Historical and current measurements are kept apart; a new session clears the history.
- **An em dash (`—`) means unavailable**, including missing historical timing or confidence fields. It is not a zero-millisecond result. Compare delays only within the same timing definition and replay/live setup. Low latency targets are goals; confirm the actual measured result before relying on them at an event.
- **Confidence mean** can help identify difficult audio, but is not an accuracy score. Listen to the source before deciding whether to move the mic or flag a correction.
- The **VRAM** counter uses NVIDIA monitoring. A zero value on a Mac does not establish how much Apple unified memory the models use.
- **Recent verses** under Features shows Bible references the system caught (Romans 8:28, John 3:16, etc.). Useful to confirm coverage.
- **Live diarization** is off by default. Check **Live diarization** on the session form only when you want `Speaker A:` / `Speaker B:` prefixes on the projector. It must not be used on a latency-sensitive Sunday until the two-speaker gate in [`live_diarization.md`](./live_diarization.md) has been run.

### Mid-session controls

The controls row (just below Start / Stop) is for the speaker pausing or switching languages mid-event. **You don't need them for a normal sermon.**

| Button | When to press |
|---|---|
| **Pause** | Speaker is taking a long break (>1 minute). Suspends pipeline processing on macOS/Linux. |
| **Resume** | Press after Pause. State pill goes back to green. |
| **Flip EN↔ES** | A different-language speaker is taking the mic. Restarts the session with the selected direction and matching STT; allow time for model startup. |
| **Fallback to HF** | For a CUDA setup whose llama-server failed. Restarts with the HF engine. This is not a Mac MLX latency or memory control. |

### Stop the session

Click **Stop session**. State pill goes back to gray (`IDLE`).

### Generate a post-session summary (optional)

After **Stop**, click **Generate summary** under Features. The background task reads the saved session transcript and displays the English/Spanish result or an error. It uses local inference, so wait until live translation and benchmarking are finished. Completion time depends on the session and model. A failed task shows its error rather than a success message; an empty transcript cannot produce a summary.

## Review and correct a session

The **Review and corrections** panel defaults to the newest completed session. You can select a session marked **LIVE** while it runs. Only finalized chunks already saved to diagnostics are available; partial captions and unfinished writes are not editable. The list refreshes while you are not editing.

1. Select a session. **Pending only** and **Flagged only** are enabled initially. Clear **Flagged only** to inspect other finalized chunks. Higher-priority flags appear first; use **Previous page** and **Next page** for a large queue.
2. Select a chunk and listen to its audio. The previous and next source lines provide context. **Original predictions and flags** shows the model text, confidence, QE, and detected vocabulary issues.
3. Check **Spoken language**, especially for older sessions. The transcript always represents the language spoken in the audio. English source maps to Spanish translation; Spanish source maps to English translation. Unknown language must be resolved before approval.
4. Correct the transcript and translation independently. Select **Transcript checked against audio** only after checking the source text, and **Translation checked by a bilingual reviewer** only after checking the translation. Editing either text clears the affected approval; editing source text or language also clears translation approval.
5. Use **Save draft** to save a revision without approving it, or **Save and next** to continue. You can save revisions during a live session. **Exclude this segment from exports** retains its history while removing it from training/evaluation output.

Draft text is also kept in this browser while you type. Navigation tries to save first. A failed save keeps your local draft and shows the error. If another reviewer saved a newer revision, click **Refresh** to load the current saved version, inspect it, then use **Restore my local draft** if you want to recover your own text and reconcile the difference. Save again to create a new revision. Local browser drafts are not a substitute for a successful server save.

Saved corrections live in revisioned sidecar files under `stark_data/corrections/`. They do not rewrite the original diagnostic text or change audience captions. If audio was not retained or has been removed, text review remains available; the panel explains that STT export will skip that chunk.

### Export a completed session

1. Stop the session before exporting. Live review is allowed, but **Export completed session** is unavailable until that session finishes.
2. Choose **Training — reviewed live audio** or **Evaluation — held apart from training**. Training requires known live provenance. Replay, synthetic, and unknown sessions can be reviewed and exported for evaluation; they cannot silently become training data. Once a session is assigned to a split, later exports keep that assignment.
3. Click **Export completed session**, then **Download review bundle**. The ZIP contains copied WAVs, Whisper metadata separated by spoken language, EN/ES translation pairs, correction revisions, and a manifest. Repeating an unchanged export produces the same bundle.

A transcript approval plus retained audio is required for a Whisper sample. **Both text approvals** are required for a translation pair. Missing audio still permits an approved translation pair. Excluded chunks are omitted. Keep Spanish speech samples in a separate Spanish corpus; the existing English Whisper trainer is configured for English.

For the setup/training owner, extract the ZIP on the destination machine, then use the existing merger commands:

```bash
python tools/merge_corrections.py whisper --corrections /path/to/bundle --train-dir /path/to/english/train
python tools/merge_corrections.py whisper --corrections /path/to/bundle --train-dir /path/to/spanish/train --language es
python tools/merge_corrections.py translation --corrections /path/to/bundle/translation/train.jsonl --train-jsonl /path/to/pairs_train.jsonl
```

Merges preserve sample identity and do not add duplicates when repeated. Evaluation bundles are rejected by training mergers. Retain the ZIP and manifest with the training run for provenance.

Completion is checked against a persisted lifecycle record, so an operator restart cannot make a still-active session look exportable. Older sessions without evidence of successful completion remain reviewable. For an older successful file replay, the setup owner can migrate its real benchmark report with `python -m tools.session_lifecycle --root . --session SESSION --migrate-successful-replay REPORT_JSON`. The report must identify the same successful pipeline run; this does not turn replay data into live training audio.

The historical `prepare_finetune_data.py` commands also read review sidecars. Use `--eval-only` for unknown/replay evaluation data and `--source-lang en` or `es` when an old session has no language tag. For an older recording you can independently verify was live training audio, `--confirm-live-session SESSION` records that explicit provenance; it cannot override replay or evaluation flags. Choosing a source language alone does not establish live provenance. The historical TSV `apply-corrections` command still writes diagnostics; the browser workflow uses sidecars.

---

## First-time setup

Have the setup owner install the runtime and models for the machine before handing it to a volunteer. A working launcher or installed wheel alone does not establish that the inference dependencies, cached models, microphone, or optional voices are ready.

- **Apple Silicon:** follow [Mac installation and launchd setup](packaging/macos.md), then check the selected MLX configuration in Pre-flight.
- **Windows/NVIDIA:** use an installer supplied and verified by the setup owner for the intended release. Confirm the installed version, CUDA dependencies, model cache, and audio input before an event. The [Windows packaging notes](packaging/windows.md) include development plans and are not evidence that a particular downloaded installer has passed an end-to-end test.
- **Repository checkout:** `./run_operator.sh` selects the configured environment. Installed packages can launch with `stark-translate operator`; their static assets are resolved from the package even when the working directory is elsewhere. Set `STARK_PROJECT_ROOT` to the writable session-data directory when needed.

If microphone permission was denied, enable it for the launcher/terminal in the operating system's privacy settings, then restart the operator. Run a short spoken rehearsal after setup changes.

---

## When something goes wrong

### Audience display says "Disconnecting" / no subtitles

- Refresh the audience display browser tab.
- If still broken, click **Stop** then **Start session** in the operator UI.

### Operator UI says state="error"

- Click **Stop**. Wait until the pill says `IDLE`.
- Click **Start session** again.
- Read the error shown in Session and the per-session log under `metrics/`. Startup failures now include the last log lines. Missing models, incompatible packages, audio failures, and memory pressure need different fixes; share the actual error with the setup owner.

### USB mic gets unplugged mid-session

- A yellow toast appears on the operator UI: *"Audio devices changed — confirm your mic is still selected."*
- Plug the mic back in. The dropdown will refresh automatically.
- Re-select the mic in the dropdown if it cleared.
- Click **Stop** then **Start session** to resume cleanly.

### "VRAM" reads >90% of the card and stays there

- Stop the session and close competing GPU applications. HF fallback is not a guarantee of lower memory use.
- If memory remains high, ask the setup owner to inspect the process and selected model. The NVIDIA VRAM counter does not measure Mac unified memory.

### Latency sparkline keeps creeping up

- Usually means another process is competing for the GPU (someone left a game open?).
- Click **Stop**, close other GPU apps, **Start** again.

---

## End-of-event checklist (2 minutes)

1. Click **Stop session** if not already stopped.
2. (Optional) Click **Generate summary** and wait for it to finish — the JSON lands in `metrics/`.
3. Save any review drafts. Export completed reviewed data now or leave it for the reviewer after the event.
4. Close the browser tabs.
5. Power off the projector. Leave the church PC powered on; systemd will keep the operator service running for next time.

---

## Route TTS to a second output (9.4.1)

Before starting a session, enable **TTS audio**, choose **local (selected speakers)**
as the TTS output mode, then select **English TTS output** and **Spanish TTS output**
independently. For example, send English to MacBook Pro Speakers and Spanish to
BlackHole 2ch or a hearing-assist transmitter. Each selector routes the spoken
translation language. **Use fallback output** uses the common TTS fallback choice,
which defaults to the system output. Choices are saved in this browser. Stop and
start the session to apply changed routes.

The device lists refresh on USB hotplug. A missing selection stays visible as
unavailable. Named routes are resolved again after a playback error; if the retry
fails, TTS warns in the log and tries the system default. Numeric indices can
change after hotplug, so prefer names for USB devices.

CLI equivalent (device names match the first case-insensitive substring):

```bash
python dry_run_ab.py --tts --tts-output local \
  --tts-device-en "MacBook Pro Speakers" --tts-device-es "BlackHole 2ch"
```

Or set `STARK_TTS_OUTPUT_DEVICES='{"en":"MacBook Pro Speakers","es":"BlackHole 2ch"}'`
and run with `--tts --tts-output local`. Per-language CLI flags override that map;
unlisted languages use `--tts-device <index>` / `STARK_TTS_OUTPUT_DEVICE`, or the
system default. A JSON `null` explicitly selects the system default for a language.
The output list is available at `/api/audio/output-devices`. WebSocket and WAV
modes keep their existing behavior and do not use these device choices.

## Glossary for non-technical operators

| Term | Plain English |
|---|---|
| **Pre-flight** | The checklist that confirms the system can run a session. |
| **Session** | One run of the live translation, from Start to Stop. |
| **VAD** | "Voice Activity Detection" — the system noticing when someone starts/stops talking. Don't change unless told to. |
| **Backend** | Which inference path to use (CUDA = NVIDIA card, MLX = Apple Silicon). Leave on `auto`. |
| **Engine** | CUDA translation runtime. Leave on `auto`; MLX uses its own runtime on a Mac. |
| **A/B comparison** | Runs two translation models side-by-side. Only useful for development; **leave unchecked for live events**. |
| **TTS** | Text-to-speech (the system reading translations aloud). Leave unchecked unless you have headphones routed for it. |
| **VRAM** | Memory on the graphics card. The sparkline shows how full it is. |
| **Latency** | For current sessions, speech end to the final caption being ready to send. Browser delivery is measured separately. |
| **p50 / p95** | Median and 95th percentile of measured delays; these describe the observed sample, not a guaranteed maximum. |
| **Confidence** | A model-specific signal to help prioritize review; it does not guarantee the transcript is correct. |
| **Adapter** | A small fine-tuning patch that improves accuracy on church-specific vocabulary. The pre-flight check tells you if one is loaded. |

---

## Quick health probe

A quick way to check the service is alive without opening the operator UI:

```
curl http://localhost:9000/healthz
```

Expect a 200 with JSON containing `"status": "ok"`. If that fails, the operator service is down — see "When something goes wrong" above.

## Who to call

| Problem | Who |
|---|---|
| Operator UI won't load at all | The dev who set up the church PC. |
| Pre-flight has red items you can't fix | Same dev. |
| Audience display projector not showing the laptop | The A/V volunteer. Hardware-only, not a software issue. |
| Wrong translation of a specific term | Select the finalized chunk in Review, listen to the audio, and save the corrected text and note. A bilingual reviewer can approve it for export. |

---

## Pre-event dry-run

Run this once a week before a real event, not the day of.

**Automated** (preferred): run the rehearsal script, which walks every endpoint
and exits 0 on green, 1 on any red:

```
./scripts/dry_run_rehearsal.sh
```

**Manual**:

1. Open the operator UI.
2. Select the intended language, backend, mic, and optional features; resolve every red pre-flight check.
3. Click **Start session** with the default settings.
4. Speak a test sentence into the mic ("This is a test — Romans 8:28 says God is good.").
5. Confirm an accurate Spanish subtitle appears on the audience display and inspect the measured delay.
6. Confirm "Romans 8:28" appears in the Recent verses panel.
7. In Review, confirm a saved final chunk appears and its audio can be played; save a draft without approval.
8. Click **Stop session**. Confirm the state pill returns to `IDLE`; completed-session export should now be available when its completion record is present.

If any step fails, file an issue with the dev.
