# Reserved-window TTS and synthetic microphone checks

Prepared on 2026-09-10. **Not executed and not acceptance evidence.** The primary
agent must reserve the model/audio window after the timed translation jobs.
No operator setting, audio stream, synthesis, playback or volume was changed
during this preparation. The ordinary operator last remained idle on port 9000.

The prior [actual device inventory](../attended_mic_20260910/raw/audio-device-inventory.json)
reported MacBook Pro Speakers (output 2), Microsoft Teams Audio (output/input 3),
and MacBook Pro Microphone (input 1). Rediscover exact current indices first;
device presence is not successful routing. Microsoft Teams Audio is a virtual
route candidate. Physical second-device audibility and unplug/hotplug gates stay
separate. No volume adjustment or human-heard claim is authorized by this plan.

## 1. Freeze the actual runtime and exercise operator route persistence

Record HEAD, hashes of `dry_run_ab.py`, `engines/tts_engine.py`,
`engines/audio_devices.py`, `engines/model_paths.py`, `models.lock.json`,
`operator_app/pipeline_manager.py`, `displays/operator/app.js`, the interpreter,
package versions, `/api/session/status`, `/api/capabilities` and
`/api/audio/output-devices`. Use a new ignored output directory beneath
`.cache/mac-en-es-closeout/tts-routing-20260910/`. Preserve failures and exact
stdout/stderr; do not replace an unsuccessful attempt with a favorable receipt.

On the existing Chrome operator tab, capture the initial controls and storage
choices through the documented CUA interface. In Advanced, select local TTS,
English output **MacBook Pro Speakers** and Spanish output **Microsoft Teams
Audio**. The per-language selectors persist device names in the existing
`stark-translate-tts-outputs` local-storage record. Enable TTS only for this
reserved routing test. Reload the page and verify both named selections and
local mode survive. Restart only the owned, idle port-9000 operator service
after recording its actual PID/command; return to the same browser profile and
origin and verify those selections again. Capture AX state and native window
screenshots, with source/version labels. Do not start a microphone pipeline with
TTS enabled during the acoustic check below.

The production start configuration corresponding to these selections is:

```json
{"tts":true,"tts_output_mode":"local","tts_device_en":"MacBook Pro Speakers","tts_device_es":"Microsoft Teams Audio"}
```

`PipelineRunner._build_argv` must retain `--tts --tts-output local`,
`--tts-device-en 'MacBook Pro Speakers'` and
`--tts-device-es 'Microsoft Teams Audio'`. During an actual controlled session,
capture its resolved argv/config rather than calling a reconstructed argv actual
runtime evidence. Language flip must preserve both route selections.

## 2. Bounded production Piper and native output checks

[probe.py](probe.py) is a prepared operational harness, not a production change.
It uses `PiperTTSEngine.load/synthesize/play` and
`OutputDeviceResolver.play`, existing local voices only, with no auto-download.
EN uses `en_US-lessac-high`; ES uses `es_MX-claude-high`. Missing voices or devices
fail before playback. It records exact model/config/source hashes, synthesis
time, generated PCM identity, actual newly opened PortAudio stream device/rate,
channels, reported latency, callback status, wait completion and closed state.
Fallback attempts are retained and cannot count as success on the selected route.

The two prepared phrases are synthetic acoustic-test material, each limited to
15 seconds. They also supply pre-generated source files for phase 3. Generated
WAVs stay in the ignored cache, not committed evidence. A 120-second parent
timeout bounds the entire process; no model process may run concurrently.

From the repository root, after reservation:

```bash
STARK_MODELS_DIR=/Users/willem/Code/vibes/SRTranslate/.cache/mac-roadmap/ct2-setup-validation/managed \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.cache/package-smoke/bin/python - <<'PY'
import subprocess, sys
subprocess.run([
    sys.executable, 'docs/evaluation/tts_routing_20260910/probe.py',
    '--execute-playback',
    '--output', '.cache/mac-en-es-closeout/tts-routing-20260910/routes-r0',
    '--device-en', 'MacBook Pro Speakers',
    '--device-es', 'Microsoft Teams Audio',
], timeout=120, check=True)
PY
```

This establishes the actual engine/resolver/native-stream path, independently
of caption timing. Production `tts_timing.playback_call_ms` ends at the host
enqueue call and does **not** mean playback completed. The harness's public
`sounddevice.wait(ignore_errors=False)` boundary is native host completion; it
still does not establish acoustic onset, hearing, or receipt at the far end of
the Teams virtual device. Preserve these distinctions in the final report.

## 3. Optional synthetic acoustic microphone loopback

Only after successful bounded routing, with a separately reserved inference
window, open a temporary operator on port 9002. Leave the user's primary operator
idle. Use current source and the exact chosen interpreter, offline model cache,
`STARK_AUDIO_SOURCE=mic`, **`STARK_SESSION_KIND=synthetic`**,
`STARK_SERVER_HTTP_PORT=9080`, `STARK_SERVER_WS_PORT=9765`. An example service
command, to be wrapped by the parent supervisor and captured to a new log, is:

```bash
STARK_PROJECT_ROOT=/Users/willem/Code/vibes/SRTranslate \
STARK_PYTHON=/Users/willem/Code/vibes/SRTranslate/.cache/package-smoke/bin/python \
STARK_MODELS_DIR=/Users/willem/Code/vibes/SRTranslate/.cache/mac-roadmap/ct2-setup-validation/managed \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 STARK_AUDIO_SOURCE=mic \
STARK_SESSION_KIND=synthetic STARK_SERVER_HTTP_PORT=9080 STARK_SERVER_WS_PORT=9765 \
.cache/package-smoke/bin/python -m operator_app.cli operator --no-browser --host 127.0.0.1 --port 9002
```

Start through the operator API so `record_audio:false` is explicit; the current
UI has no equivalent recording toggle. Capture this request and response:

```json
{"lang":"en","profile":"standard","backend":"mlx","stt_backend":"parakeet-mlx","model_family":"gemma4","gemma4_size":"e4b","mic_device":1,"tts":false,"record_audio":false,"run_ab":false}
```

POST it to `http://127.0.0.1:9002/api/session/start` only after refreshing the mic
inventory. Observe the actual operator and audience pages; the audience URL is
`http://127.0.0.1:9080/displays/audience_display.html?port=9765`. Wait at most 90
seconds for health `ready`; record input frames/RMS without inferring speech.
Then play the previously generated English WAV through the built-in speakers:

```bash
.cache/package-smoke/bin/python - <<'PY'
import subprocess, sys
subprocess.run([
    sys.executable, 'docs/evaluation/tts_routing_20260910/probe.py',
    '--execute-playback',
    '--play-existing', '.cache/mac-en-es-closeout/tts-routing-20260910/routes-r0/receipt.json',
    '--language', 'en', '--device-en', 'MacBook Pro Speakers',
    '--output', '.cache/mac-en-es-closeout/tts-routing-20260910/acoustic-en-r0',
], timeout=30, check=True)
PY
```

`--play-existing` loads no Piper/STT/translation model. Capture any actual
partials/finals and visible-browser ACKs for at most 30 seconds after playback.
Silence/no detection remains a failed or unassessable acoustic check; do not
raise volume or repeat indefinitely. Stop cleanly, then start Spanish with
`lang:es`, `stt_backend:mlx`, TTS still false and recording still false. Play
the Spanish source using the same command with `--language es`,
`--device-es 'MacBook Pro Speakers'` and a fresh `acoustic-es-r0` output directory.

Retain actual session IDs, source/runtime hashes, `audio_source=mic` **and**
`session_kind=synthetic`, `record_audio=false`, source WAV hash/voice/text,
native playback receipts, capture health and final caption/ACK artifacts.
Attach explicit `training_eligible:false` and `human_reviewed:false` to the
evidence manifest; do not export, approve, or copy microphone chunks. This is
synthetic room/speaker/mic functional coverage, not human/natural bilingual
quality, clean latency, or completion of unrelated #131 adapter requirements.

## Cleanup and retained boundaries

For every started session, request `/api/session/stop`, confirm lifecycle
completed/failed honestly, required writes and owned capture/model processes
terminated, and record failures. Terminate only the owned temporary port-9002
operator and close its test tabs; verify 9002/9080/9765 close. Restore the primary
operator's original route/mode settings and TTS-off state, without changing
system volume or other applications. The harness always calls `sounddevice.stop`
and unloads its own voices. Record timeout cleanup explicitly if a supervisor
had to terminate it. Do not claim physical audibility, virtual far-end reception,
hotplug recovery, human approval or a caption-latency improvement from these checks.
