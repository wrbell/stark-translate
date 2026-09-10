# Capture loss accounting, 2026-09-10

The capture accounting repair distinguishes **measured worker FIFO loss** from
**PortAudio input overflow with an unknown sample count**. It also reconciles
losses at shutdown even when their notifications have not reached the audio
consumer. This is an accounting and failure-reporting repair. Sustained live
capture and spoken EN↔ES service acceptance remain pending; no microphone or
output-device test was performed for this change.

## What the retained failure establishes

The [structural receipt](../tts_routing_20260910/raw/acoustic-es-retest-structural-summary.json)
for `20260910_105554_710324_es`, source `a8511ee`, remains a failed observation.
Its original `upstream_dropped_samples=7680` is five 1,536-sample callbacks, or
160 ms at 48 kHz. The source gaps are `[436224,442368)` and
`[2537472,2539008)`. The parent handoff admitted and dequeued 1,681 frames with
zero handoff drops. The recorded counter increments at the worker's Python
`queue.Full` branch. Thus these five callbacks establish **worker callback FIFO
loss**, not a measured PortAudio driver overflow.

The [prior analysis](../tts_routing_20260910/CAPTURE_ANALYSIS.md) records a maximum
callback-to-pipe interval of 1,508.147 ms and a maximum parent producer wait of
903.845 ms. Those spans combine queue residence and scheduling with possible
pipe backpressure. There are no separate child writer start/end timestamps, so
neither is an isolated measurement of a blocked write. Warmup/STT/VAD overlaps
are observations; they do not prove GIL, GPU, or driver causation. Original
receipts and measurements are unchanged. No incidental room text or audio is
copied into this note or its tests.

## Production contract

- [`capture_worker.py`](../../../tools/capture_worker.py) retains its 32-frame,
  nonblocking callback FIFO and original sample/ADC coordinates. Frames append
  schema-2 metadata: the existing cumulative `dropped` counter has an explicit
  `worker_fifo_dropped_samples` alias; native overflow gets its own boolean and
  cumulative callback count. An overflow flag on a FIFO-dropped callback can
  therefore reach the parent in the next admitted frame.
- [`capture_protocol.py`](../../../tools/capture_protocol.py) keeps the historical
  `capture_overflow:N` display string while retaining independent FIFO/native
  flags. The reader rejects contradictory flags, decreasing/invalid counters,
  and disagreeing aliases. Plain `input overflow` now causes a required capture
  failure. Its lost-sample count is **null**, because the flag supplies no count.
- [`isolated_audio.py`](../../../tools/isolated_audio.py) preserves
  `upstream_dropped_samples` as the last counter observed on the framed pipe.
  It does not replace that historical value with a larger terminal total.
  Terminal receipts and explicit availability are additive snapshot fields.
- [`dry_run_ab.py`](../../../dry_run_ab.py), `audio_callback` and
  `_record_capture_transport`, route detected loss and expected-but-unverified
  accounting into the real required persistence/health ledger before lifecycle
  completion. Parsed statuses can still be queued when Stop discards a tail;
  close reconciliation therefore checks observed counters even if they equal
  the terminal counters. A live and close notification may describe the same
  loss. **Failure notifications must not be summed as lost-sample counts.**
- [`CaptureTransportSummary`](../../../tools/capture_handoff.py) retains cumulative
  observed/terminal counters and affirmative loss flags across all pause/resume
  segments, including when its 16-detail limit truncates older segments. A
  terminal total is null unless every segment has a verified receipt. An
  observed native-overflow total is null if any segment lacks that observation;
  availability counts and affirmative known-loss flags remain available.

The session-summary JSONL `capture_transport` field retains the receipt after
the temporary side channel is removed. Existing summary/trace fields retain
their meanings. `portaudio_input_overflow_lost_samples` is always null. False
`*_loss_detected` / `*_overflow_detected` flags mean no affirmative detection in
the available counters; they do not override unavailable accounting.

## Bounded POSIX terminal receipt

A production POSIX isolated stream owns a private temporary directory and a
random capture token. On SIGTERM the worker unwinds the input stream context
and atomically writes a small **metadata-only** receipt outside its potentially
congested stdout pipe. The parent retains its existing 0.5-second graceful wait
and bounded kill/wait fallback. It validates the closed-stream receipt only
after worker/reader shutdown and removes the temporary directory. Receipt
identity, schema, integer counters, callback/sample bounds, conservation, and
consistency with already parsed frames are checked.

The receipt partitions samples returned to callbacks:

```text
callback_samples = worker_fifo_dropped_samples + worker_fifo_admitted_samples
worker_fifo_admitted_samples = pipe_written_samples
                            + worker_fifo_pending_samples
                            + writer_unfinished_samples
```

`pipe_written_samples` means the writer completed its flush/bookkeeping; it
**does not mean the parent consumed those samples**. `writer_unfinished_samples`
includes a dequeued, partially written, or flushed-but-not-yet-bookkept frame.
Already parsed samples cannot all remain pending in the worker FIFO. These
counts describe callback-returned samples and cannot reconstruct audio omitted
by PortAudio before a callback.

A deliberate Stop may leave admitted samples in the worker, writer, or pipe.
Those tail counts are informational and do not, by themselves, turn an ordinary
Stop into a capture failure. This repair does not certify every captured Stop
tail as captioned, or claim full acoustic completeness from a verified receipt.
Actual FIFO drops and native input-overflow flags still fail capture. Missing,
invalid, or incomplete **expected POSIX** receipts cause
`capture_terminal_accounting_unverified`, which is an accounting failure, not
a diagnosis of microphone hardware failure.

On Windows, `Popen.terminate()` does not guarantee Python `finally` execution.
Terminal accounting is explicitly `unsupported_platform`; the existing bounded
termination behavior and observed loss reporting remain. Explicit legacy/fake
worker commands are `unavailable_legacy_worker`; direct/legacy streams are
unavailable; file replay is `not_applicable_file_replay`. These scopes do not
create a false missing-receipt failure for every file replay or Windows Stop.

## Validation and limits

The new [capture accounting regressions](../../../tests/test_capture_accounting.py)
execute the real protocol, worker loop with replaced native modules, framed
reader, shutdown methods, transport accumulator, and persistence ledger.
Source-definition extraction avoids the application's model/native import graph
for the standalone suite. Fixtures cover simultaneous FIFO/native loss,
terminal-only loss with no later admitted frame, parsed-but-unconsumed status,
receipt contradiction/absence, shutdown errors, pending ordinary Stop, legacy
and Windows availability, source offsets, and more than 16 pause/resume segments.

Recorded local checks:

```text
python3 -m unittest tests.test_capture_accounting -q
21 tests passed (stdlib/fake dependencies; no native audio or models)

stt_env/bin/python -m pytest tests/test_capture_accounting.py tests/test_capture_transport.py tests/test_input_device_identity.py tests/test_music_capture_resume.py -q
45 passed, 20 subtests passed in 0.86s (mocked/scripted capture)
```

Ruff check/format passed for the changed Python files. These checks did not open
input/output devices, play audio, execute a model, prove native SIGTERM cleanup,
or establish a lower real capture-loss rate. The original failed observation
remains failed. The user prohibited microphone and output-device testing for
the rest of this session; any future native rehearsal needs fresh authorization.

## Remaining Mac-local work

The current evidence justifies accounting fixes, not larger queues or a new
default. A later silent file-input comparison can isolate the existing VAD-worker
and idle/coalesced-warmup toggles with identical EN↔ES inputs and source spans.
A model-loaded pre-input VAD priming experiment could test first-call delay only
if VAD state is reset afterward. These are proposed bounded experiments, not
implemented or selected optimizations. File replay cannot certify PortAudio
capture. A later authorized native rehearsal must separately verify callback
flow, observed and terminal accounting, consumer delivery, and clean shutdown.

## Existing fixture compatibility follow-up

The first full CI run on `d7ed43d` exposed six failures in older capture/file
fixtures. Their unrestricted `MagicMock` streams invented `capture_snapshot()`
and truthy loss counters; production shutdown correctly reported those fabricated
losses. The fixtures now expose only their scripted context-manager interface
and use the real per-session transport accumulator. They do not pretend to be
`FileAudioStream` instances or weaken production loss checks.

The combined capture-accounting, utterance-discard and file-stream checks then
passed **73 tests and 20 subtests** in 11.34 seconds, with devices/models mocked.
The original CI failure remains retained; full CI on the fixture repair is a
separate validation step.
