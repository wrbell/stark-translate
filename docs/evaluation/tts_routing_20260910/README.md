# TTS routes and synthetic acoustic microphone checks — September 10

The English and Spanish Piper host-output checks passed on **MacBook Pro
Speakers** and **Microsoft Teams Audio**, respectively. The English synthetic
speaker-to-built-in-microphone session produced partials and a final and completed
cleanly. Spanish produced partials and a final but **failed capture reliability**:
the handoff dropped samples. Its failed lifecycle and blocked export remain
authoritative even though the captions looked complete.

This is bounded functional evidence, not a natural/human bilingual quality
cohort, a latency promotion gate, human-heard confirmation, physical second-output
acceptance, or virtual far-end reception. The Mac was locked again at CUA
inventory. Browser accessibility and tab rendering worked; ACK `visible:true` is
**client-reported visibility only**, not native display certification.

The later traced Spanish retest on `a8511ee` also **failed**: the parent handoff
lost no frames, but the native child reported 160 ms of upstream loss. It also
received incidental room speech and is therefore not a controlled synthetic
comparison. Only structural data from that retest is shared below; incidental
transcript content stays in local private artifacts.

## Runtime and controls

The [prepared plan](PLAN.md) was executed in the reserved model/device window;
its preparation-time nonexecution label describes the original plan. The
[initial runtime receipt](raw/initial-runtime.json) recorded HEAD `08935a6` and
source hashes. The actual microphone lifecycles record `e644fe0` and pipeline
SHA-256 `af6ff9a69a3adf747ef106265e6eab4c93d8ada904d7232d5f43d2be25c32359`:
the parent committed source/report work before their startup. These were current
checkout pipeline processes using `.cache/package-smoke/bin/python` and its
installed dependency/artifact metadata, not a claim that a newly built package
was installed. Package version was `2026.14.0.0`; actual dependencies included
Piper 1.8.0, ONNX Runtime 1.29.0, sounddevice 0.5.6, Torch/TorchAudio 2.10.0.
`stt_env` and model defaults were untouched.

CUA observed the port-9000 operator initially with spoken translation off,
browser output, default general device and both language routes “Same as above.”
It selected local output, English MacBook Pro Speakers, Spanish Microsoft Teams
Audio and enabled spoken translation while idle. Both named routes, local mode
and the enabled checkbox survived page reload and an owned idle operator restart
(PID 6564 → 23727). The primary operator was restored to spoken translation off,
browser output, default/inherited routes, English and built-in microphone.
These are agent observations of actual AX controls, not reconstructed settings.

The actual [restart command](raw/primary-operator-restart.json) and
[isolated microphone operator command](raw/synthetic-operator.json) are retained.
Route UI persistence and production engine/resolver playback were exercised
separately. No caption-triggered TTS pipeline ran; that integrated boundary is
not established by combining the two checks.

## Actual Piper/native-stream results

| Language / selected route | Synthesis ms | Generated duration s | Host open call ms | Host completion ms |
|---|---:|---:|---:|---:|
| EN / MacBook Pro Speakers, device 2 | 867.5 | 5.956 | 3010.5 | 9083.2 |
| ES / Microsoft Teams Audio, device 3 | 231.4 | 7.628 | 61.7 | 7815.0 |

[Successful native receipts](raw/routes-r1-receipt.json) use production
`PiperTTSEngine.load/synthesize/play` and `OutputDeviceResolver.play`. Each opened
a new native stream on the selected device at 22050 Hz, completed public
`sounddevice.wait(ignore_errors=False)`, and was stopped/closed with no callback
error or fallback. Host completion includes open/enqueue plus playback; neither
that nor reported device latency measures acoustic onset or human hearing.
The stock voices were the existing pinned snapshot
`1162a9173d0ce503555aed757976b7a9912eae4c`; exact model/config/WAV hashes are in the
receipt. No voices were downloaded and system volume was not changed.

The [first attempt](raw/routes-r0-receipt.json) failed before synthesis/playback:
passing an explicit HF snapshot symlink into `resolve_piper_voice()` resolved it
to the blob filename, losing the adjacent `.onnx.json` lookup. The harness was
changed to the ordinary stock-name path after confirming both pinned local
voices existed. This failure is retained; explicit symlink override handling
was fixed in `38f5199`. The failed and successful probe hashes differ and are
recorded in their receipts. The exact original failing harness is archived as
[routes-r0-probe.py](raw/routes-r0-probe.py), verified against its original hash.

The [explicit-path retest](raw/routes-r2-explicit-receipt.json) on clean `a8511ee`
then passed both voices and selected native streams, with no callback errors or
fallback. English synthesis/open/completion took 1056.1/58.9/6092.7 ms; Spanish
210.4/26.2/7603.9 ms. No download or system-volume change occurred.

## Synthetic acoustic microphone results

Only the previously generated WAVs were played through the real MacBook speakers
while the real built-in microphone (device 1) captured audio. No synthesis model
was loaded concurrently with STT/translation. Both operator API start bodies
explicitly set `record_audio:false`, `tts:false`, and `run_ab:false`; environment
`STARK_AUDIO_SOURCE=mic` and `STARK_SESSION_KIND=synthetic` persisted in actual
session metadata. Generated WAVs remain in ignored cache; microphone audio was
not saved, corrections were not approved, and no training export was attempted.

| Source | Session ID | Partials / finals | Lifecycle | Payload-ready ms | Client-reported visible ACK ms |
|---|---|---:|---|---:|---:|
| Synthetic English via speakers → built-in mic | `20260910_102239_515548_en` | 7 / 1 | completed, exit 0 | 3412.6 | 3427.8 |
| Synthetic Spanish via speakers → built-in mic | `20260910_102356_408494_es` | 6 / 1 | **failed, exit 1** | 5510.9 | 5542.5 |

These individual timing observations are not clean latency measurements or
comparative estimates. Full final ACK identities and span bounds are retained in
the [derived summary](raw/derived-session-summary.json), alongside raw
`display_metrics`, diagnostics, source metadata, logs and lifecycles. The
English final event is `20260910_102239_515548_en:11`; Spanish is
`20260910_102356_408494_es:9`. Browser screenshots were emitted to the task via CUA
and showed the matching bilingual text. They are tab renders while the OS was
locked, not a saved native-screen artifact.

English completed 19/19 required writes with no capture gaps. Spanish recorded
16 `handoff_overflow` errors at 14:24:50.592–50.634 UTC, then
`capture_overflow:6144`; lifecycle persistence records 17 capture failures.
Coverage records gaps `[1571328,1595904)` and `[1641984,1648128)` at 48 kHz:
30,720 samples / 640 ms total. They occur after the retained final source span
`[1107456,1489920)`, but their missing acoustic content is unknown. Reporting a
successful final does not establish lossless Spanish capture. Live capture has
no file EOF, so `source_coverage.complete:false` alone is not a replay failure;
the actual gap/error records establish this failure.

## Cleanup and remaining work

Both pipelines and their capture children exited. The temporary operator was
terminated; ports 9002/9080/9765 closed and its test tabs were closed. The primary
operator remains idle on 9000 with original controls restored. The initial
cleanup snapshot caught PID 24024 finishing termination; the later snapshot
confirms it is gone. No owned model/capture/probe process remained when the
reserved window was released. See [cleanup](raw/cleanup.json).

## Traced post-fix Spanish retest

The capture change preserves 32-frame native and handoff queues and the 64-frame
asyncio queue. It backpressures the isolated transport reader, retains native
callback drop/error policy, and distinguishes prolonged consumer backpressure
from no microphone input at the same timeout threshold. Seven focused transport
tests and an 83-test capture/reliability/replay batch passed before this retest.
Source hashes and the empty dirty-patch hash are in
[retest-start-source.json](raw/retest-start-source.json).

The first start request received HTTP 422 because the temporary operator's
PortAudio inventory had MacBook microphone index 0 and speakers index 1, while a
fresh process had the WR17.1 continuity microphone at 0 and MacBook microphone at
1. The [rejected start](raw/acoustic-es-retest-start.json) is retained. Restarting
only the owned idle operator aligned its inventory; the resolved start used
built-in microphone index 1. This exposes a device-identity boundary: integer
indices can differ across the operator and newly spawned capture process.
Selected names should be bound and resolved again in the opening process before
claiming robust input routing.

Session `20260910_105554_710324_es` used trace capacity 131,072, the same original
Spanish synthetic WAV hash, recording off, TTS off and synthetic provenance.
The audience browser also exposed incidental room speech, so capture was stopped
and no incidental text was copied into this evidence directory. Its local
original artifact paths/hashes are recorded in the
[structural summary](raw/acoustic-es-retest-structural-summary.json). No audio
binary was saved or committed. This run is not approved training data or a
controlled pre/post comparison.

The closed handoff admitted/dequeued 1,681/1,681 frames with zero handoff drops,
zero terminal pending/waiting producers and no timeout. Its high-water was 32;
maximum producer wait was 903.8 ms, handoff wait 1592.0 ms, and capture age
2201.5 ms. Pipe maximum callback-to-read age was 1508.1 ms. The child reported
7,680 missing samples / 160 ms at 48 kHz, with gaps `[436224,442368)` and
`[2537472,2539008)`. Lifecycle remains failed/exit 1. This validates the narrow
handoff repair in one observational run while leaving upstream capture loss
unresolved.

The original trace retained 8,585 events with zero discarded entries. The shared
[structural trace](raw/retest-structural-trace.jsonl) deliberately excludes text
and nonessential event types. At the second gap, final STT overlapped a Gemma
warmup, and an inline VAD span measured 747.0 ms. These measured concurrent stages
justify further scheduling analysis, not an assertion about GIL or GPU causation.
See [capture analysis](CAPTURE_ANALYSIS.md). The retest pipeline, capture child,
both temporary operator PIDs and all test ports were stopped before releasing
the device/model window; [cleanup](raw/retest-cleanup.json) confirms it.

Retained work: resolve input identity within the process opening the device;
diagnose upstream Spanish capture loss before claiming reliable EN↔ES live capture; exercise
caption-triggered route dispatch and virtual far-end receipt if required by
#132; retain human-heard/physical output and unrelated #131 adapter gates.
No acceptance issue was closed. The [raw manifest](raw/manifest.json) hashes
the retained text evidence and explicitly marks it unreviewed and ineligible for
training; it includes no audio binaries.
