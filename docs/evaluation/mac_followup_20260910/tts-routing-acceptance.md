# Issue #132 routing acceptance audit — September 10, 2026

The existing operator, native Piper and test evidence supports the original
routing acceptance of [#132: P1: 9.4.1 Multi-channel TTS
routing](https://github.com/wrbell/stark-translate/issues/132). This audit supports
closing that issue after the final source merge. It does not certify a complete
caption-triggered audio session. The issue was still open when this audit was
prepared; no new device testing was performed.

The original acceptance is:

> Operator can send TTS to a chosen output device; EN and ES can be routed independently. Tests cover the new engine path.

The issue explicitly permits virtual cables. Its
[last implementation comment](https://github.com/wrbell/stark-translate/issues/132#issuecomment-5609268563)
records implementation in #188 and leaves a hands-on virtual-cable or second-output
check outstanding. The retained native runs below supply that bounded host-output
check. Physical unplug/replug and far-end reception are separate service gates.

## Acceptance and retained evidence

| Original requirement | Existing evidence | Scope |
|---|---|---|
| Operator selects real output devices | Actual operator accessibility observations selected local output, English **MacBook Pro Speakers**, Spanish **Microsoft Teams Audio**, and enabled spoken translation. All survived page reload and the owned idle operator restart, PID 6564 → 23727. | Recorded in the [original report](../tts_routing_20260910/README.md#runtime-and-controls); selections were observed through the UI. The [restart receipt](../tts_routing_20260910/raw/primary-operator-restart.json) records the command, not the selected configuration. Original settings were restored. |
| EN and ES route independently | Production `PiperTTSEngine.load/synthesize/play` and `OutputDeviceResolver.play` opened actual English device **2** and Spanish device **3**, respectively, at 22050 Hz. Both completed `sounddevice.wait(ignore_errors=False)` and closed without callback errors or fallback. | [Stock-voice run](../tts_routing_20260910/raw/routes-r1-receipt.json), 14:21:35–14:21:55 UTC; [explicit-path retest](../tts_routing_20260910/raw/routes-r2-explicit-receipt.json), 14:52:05–14:52:22 UTC. These were sequential language-specific probes, not simultaneous duplicate output. |
| Tests cover the engine path | Actual pipeline routing, Piper device forwarding, operator request/argv fields, output listing, name resolution, retry/fallback and UI persistence have focused mocked tests. | Both Python 3.11 and 3.12 test jobs [passed on `4b0144d`](https://github.com/wrbell/stark-translate/actions/runs/34502608307). Those tests establish software contracts; native host routing is established separately by the receipts above. |

The exact commands are retained for
[run r1](../tts_routing_20260910/raw/routes-r1-command.json) and
[run r2](../tts_routing_20260910/raw/routes-r2-command.json). Both used
`.cache/package-smoke/bin/python` and the tracked
[`probe.py`](../tts_routing_20260910/probe.py), with
`--device-en "MacBook Pro Speakers" --device-es "Microsoft Teams Audio"`;
r2 additionally used `--explicit-voice-paths`. Both exited 0 with empty stderr.
They are probe identifiers, **not pipeline sessions**, and have no session IDs.
The installed dependencies included Piper 1.8.0, ONNX Runtime 1.29.0 and
sounddevice 0.5.6. This is checkout-source evidence using the isolated installed
dependencies, not a new package-installation certification.

## Source chain and regression coverage

The audited source is
[`a9dcd064852c1b528ac1fb65de8a1c457f888e6d`](https://github.com/wrbell/stark-translate/commit/a9dcd064852c1b528ac1fb65de8a1c457f888e6d).
The production connection is explicit:

1. [`displays/operator/app.js`](../../../displays/operator/app.js)
   `readForm` carries the independent language routes; `persistTtsChoices`
   retains the local controls.
2. [`operator_app/pipeline_manager.py`](../../../operator_app/pipeline_manager.py)
   `SessionConfig` and `_build_argv` carry `--tts-device-en` and
   `--tts-device-es` into a local-TTS session.
3. [`dry_run_ab.py`](../../../dry_run_ab.py) loads the target-language voice,
   sends finalized translated text and `TARGET_LANG` to `_run_tts`, and resolves
   `settings.tts.output_devices[language]` with the configured general fallback.
   Thus EN→ES uses the Spanish output, and ES→EN uses the English output.
4. [`OutputDeviceResolver.play`](../../../engines/audio_devices.py) selects the
   named device and handles retry/fallback;
   [`PiperTTSEngine.play`](../../../engines/tts_engine.py) passes that device to
   `sounddevice.play`. The retained probe observes the newly opened native
   stream and waits for host completion through public sounddevice APIs.

The focused tests are
[`test_tts_multichannel.py`](../../../tests/test_tts_multichannel.py)
(`test_run_tts_routes_languages_and_logs_once`, operator/CLI contracts and
retry/fallback),
[`test_phase9_4_1_tts_device.py`](../../../tests/test_phase9_4_1_tts_device.py)
(Piper forwarding and request/argv validation), and
[`test_operator_language_ui.py`](../../../tests/test_operator_language_ui.py)
(the actual JavaScript controls, submitted fields and reload persistence).
All three files existed at the successful CI revision cited above.

The r2 source record is clean
[`a8511ee`](../tts_routing_20260910/raw/retest-start-source.json).
The engine and resolver have no diff from that revision to the audited revision.
Their SHA-256 values, already recorded in both native receipts, are:

| File | Retained SHA-256 |
|---|---|
| `engines/tts_engine.py` | `c7a92e13390044b07f8b5101f2ccab12771f9d7712f4843eec96a4f0993513a3` |
| `engines/audio_devices.py` | `994706f0d9dde96bfde3ad4d653a66cd4f43bd8eefe0be6ea2de7c4f7966efe5` |

The [initial runtime record](../tts_routing_20260910/raw/initial-runtime.json)
records `08935a6`; it must not be treated as an exact revision stamp for every
later check. The original explicit-voice-path failure is preserved in
[r0](../tts_routing_20260910/raw/routes-r0-receipt.json): the snapshot symlink
lost its adjacent configuration lookup. Fix `38f5199` preceded the successful
r2 retest. Voice snapshot `1162a9173d0ce503555aed757976b7a9912eae4c` and existing
voice/config hashes are retained in the receipts; this audit neither rehashed
audio/model bytes nor downloaded models.

## Separate pending validation

The operator and native engine checks were separate. No full
caption-triggered native-TTS session, virtual far-end reception, human-heard
confirmation, physical second-output rehearsal or physical unplug/replug test
is established. The nearby synthetic microphone sessions explicitly had TTS
disabled and do not fill that gap. Native host completion is not acoustic onset,
intelligibility, sustained service reliability or caption-latency evidence.

These remaining checks should retain their own acceptance gates without
extending the original routing issue after its implementation and allowed
virtual-output check have evidence. The later user prohibition on microphone
and output-device testing applies for the rest of this session. Device rehearsals
remain deferred to a later session under the user's then-current instructions.
