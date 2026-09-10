# Spanish capture loss: bounded source analysis

This analysis uses the failed synthetic microphone session
`20260910_102356_408494_es`, current source and a model-free reproduction. It does
not relabel that session successful or establish that a proposed change fixes
the real microphone path.

## Established pre-fix loss path

1. `tools/capture_worker.py` receives 1,536 samples at 48 kHz per callback
   (32 ms). Its native callback uses a bounded 32-frame `queue.Queue`, drops a
   **new** frame when full, and includes cumulative dropped samples and original
   ADC/current-time/host-callback `received` stamps in later pipe frames.
2. `tools/isolated_audio.py:IsolatedInputStream._read` reads that framed pipe on
   the ordinary `capture-reader` thread. It anchors `CaptureStamp` to the child's
   original callback receipt, not delayed pipe receipt. It invokes the pipeline
   callback synchronously for each parsed frame, so buffered pipe frames can
   arrive as a fast burst.
3. `dry_run_ab.py:audio_loop` constructs `CaptureHandoff` with capacity 32 and
   `wait_for_space` **only for files**. For this isolated live reader, the
   handoff therefore drops its **oldest** frame when full, even when the downstream
   64-frame asyncio audio queue has room. The event loop drains up to eight
   handoff frames per turn, resamples them on the loop, and schedules another
   drain after 5 ms. Thirty-two frames represent 1.024 seconds; the next queue's
   64 frames represent 2.048 seconds. These capacities are unchanged.
4. The event-loop audio consumer runs packaged Torch Silero inline under
   `_pytorch_lock` by default. STT is submitted to a worker pool. This session's
   partial translator was CT2, so attributing a stall to Marian HF holding the
   PyTorch lock is unsupported here.

The [CPU reproduction](raw/capture-burst-cpu-repro.json) feeds 48 frames before
the scheduled event-loop drain, with `can_accept()` always true. The existing
helper drops exactly 16 and keeps frames 16–47. It establishes avoidable
transport-burst loss independent of microphone hardware, actual sample rate,
model inference, downstream capacity or sustained producer overload. Its shape
matches the real handoff's 16 overflow notifications, but a shape match is not
proof of the initiating stall.

The real receipt also contains `capture_overflow:6144`: four additional native
child frames / 128 ms were already missing upstream. Coverage gaps total 640 ms,
consistent with 16 × 32 ms handoff loss plus 4 × 32 ms upstream loss. Preventing
the handoff's drop policy alone cannot prove the child loss repaired.

## Timing and uncertainty

Final diagnostics retain session-clock stage times: requested 47,212.078 ms,
worker started 47,212.210 ms, finished 50,639.707 ms (3,427.497 ms), translation
requested 50,641.789 ms, finished 52,147.852 ms. The handoff's 16 notifications
occurred at 14:24:50.592–50.634 UTC, immediately before the final STT result was
logged at 14:24:50.815 UTC; the child overflow was logged at 14:24:50.668 UTC.

The main cancellation handler also did not log its cancelled partial until
14:24:50.636, after the cancellation request at 14:24:47.303. That is consistent
with delayed event-loop progress around the final worker interval, but does not
identify whether GIL retention, inline VAD, resampling, memory pressure or
another wait caused it. The latency trace was disabled. Original per-frame
callback metadata exists in the transport protocol but was not persisted;
the retained span stamps cannot reconstruct the receipt-time lag of every frame.
There is no defensible measured maximum VAD wait or pipe delivery lag for this
run. Do not invent one or conclude that GPU execution alone blocked capture.

## Narrow remediation and required validation

The following transport change is implemented with seven focused model-free
regressions in `tests/test_capture_transport.py`. The real microphone retest is
recorded below; the original failed run remains pre-fix evidence.

Use bounded backpressure for the isolated **reader thread**, as already used for
file transport, with the native child callback remaining nonblocking. Select the
policy from the actual stream capability/type rather than assuming an arbitrary
`mic` environment source is safe to block. Preserve queue capacities, ordering,
original sample/ADC stamps, no-input timeouts and fail-closed capture errors.
Closing the handoff must unblock a waiting reader before child/thread joins.
Backpressure is bounded in memory; sustained inability to consume may still
overflow the child's bounded native queue and must still fail the session.

Add a regression for a buffered child-reader burst exceeding handoff capacity
that drains without drops, a downstream-full ordering case, and close/shutdown
unblocking while the producer waits. Keep the direct/native callback mode's
nonblocking drop semantics covered. Add bounded per-stage telemetry for original
callback receipt → pipe parse → handoff dequeue → audio dequeue, plus original
sample spans on drop and stage high-water marks. Existing `audio_dequeued`
capture-age and `vad_complete` spans require trace enabled and cannot separately
measure the pipe/handoff boundaries.

The implemented summary key `capture_transport` snapshots closed transports
after handoff close and child/reader shutdown, including pause/error exits. It
retains 16 segment details and cumulative drop totals, with explicit truncation;
each handoff/pipe retains at most 32 dropped source spans. Existing bounded
session trace events now include `capture_pipe_received`, `capture_pipe_gap`,
`capture_handoff_enqueued`, `capture_handoff_dequeued`,
`capture_handoff_dropped`, and source sample coordinates on `audio_dequeued`.
Trace clocks retain the existing session origin and do not include PCM.

The reserved traced microphone retest required zero child/handoff/input-queue
drops and a completed lifecycle. Neither widening queues nor asserting the
event loop is “nonblocking” from its async syntax is sufficient evidence.

## Traced retest: upstream loss remains

Session `20260910_105554_710324_es` used clean source `a8511ee` with trace capacity
131,072 and retained all 8,585 events. It played the original generated Spanish
WAV, but incidental room speech also entered the real microphone. That makes
this an uncontrolled functional/reliability observation, not a controlled
synthetic comparison or a quality reference. Shared evidence keeps only
[structural timing](raw/retest-structural-trace.jsonl) and a
[structural summary](raw/acoustic-es-retest-structural-summary.json); no room
speech text or audio is included.

The parent handoff admitted and dequeued 1,681 frames with zero drops, zero
terminal pending frames and no waiting producers. The native worker still
lost 7,680 samples (160 ms), so the lifecycle failed with two required capture
failures. Its queue remains 32 frames, approximately 1.024 seconds. This
demonstrates the new handoff policy can preserve the reader burst in this run;
it does not repair or certify the upstream capture path.

All times below use the original pipeline monotonic trace origin. Gap receipt
time is when the parent parsed the first later frame reporting loss; it is not
the exact instant a callback was discarded.

| Source gap at 48 kHz | Gap parsed at ms | Measured overlapping work / transport delay |
|---|---:|---|
| `[436224,442368)` (128 ms) | 23424.918 | Partial STT ran 20009.754–21707.605; Gemma warmup ran 21814.926–24432.010. The frame beginning 368640 was received by the child callback at 21888.799 and parsed at 23396.972, a measured callback-to-pipe delay of 1508.147 ms. |
| `[2537472,2539008)` (32 ms) | 68251.343 | Final STT ran 65664.785–68468.289 and Gemma warmup ran 65752.197–68458.595. Inline MainThread VAD occupied 746.953 ms ending at 66498.485. The frame beginning 2469888 had callback-to-pipe delay 1414.656 ms; frame 2445312 reached handoff dequeue with capture age 2201.490 ms. |

Handoff high-water was 32. Maximum producer wait was 903.845 ms; maximum
handoff wait was 1592.030 ms. The measured inline VAD span establishes delayed
event-loop progress within that call, including any unmeasured scheduling or
lock waits. The trace does not separate those causes, prove GIL retention,
identify the exact child write stall, or assign either gap solely to a model.
Its source coordinates and receipt clocks establish accumulated transport
delay, not the acoustic content of the missing samples.

The existing opt-in `latency.vad_worker` already submits `is_speech` to a
dedicated one-thread pool and awaits each result, preserving frame order and
the existing PyTorch lock. No duplicate worker or queue enlargement is needed.
Both Torch and ONNX worker variants were already unselected in the English
45-second [September 10 screen](../overnight_screen_20260910/README.md), including
tail-guard failures. If a later reserved experiment tests that existing toggle,
its distinct question is whether the traced Spanish microphone capture losses
and event-loop delays change. It needs bounded trace, identical source/defaults
apart from the toggle, zero input loss and shutdown accounting; it would not
reopen the prior speed result or justify changing the default by itself.

## Input device identity follow-up

The first retest start was correctly rejected with HTTP 422: index 1 meant
MacBook speakers in the old operator process, but built-in microphone in a new
native process after the continuity microphone reappeared. Restarting the
temporary operator refreshed its list and allowed the original retest. This is
separate from the subsequent capture loss.

Source `d8c3f05` binds explicit microphone selection to exact name and
host API through operator preflight, session restarts, live capture and the
idle microphone test. The native child resolves the current index immediately
before opening input, fails if the identity is missing or ambiguous, and
returns its actual identity in frame/probe metadata. Legacy integer-only CLI
callers retain process-local index behavior; automatic input remains explicit
automatic selection. CPU regressions exercise drift, missing and duplicate
names, cross-API identity, actual worker options and metadata, bounded UI
placeholders and test-input propagation.

The [native identity probe](raw/identity/microphone-identity-probe-20260910.json)
at 15:32 UTC on `eddb0ad` requested stale index 2 with exact name
`MacBook Pro Microphone` and host API `Core Audio`. The child opened current
index 1, captured 96,000 samples over two seconds and discarded them; no audio
file or STT run was produced. A missing identity was rejected. This validates
native identity resolution and rejection in that bounded probe, not sustained
capture, the complete operator pipeline, caption quality or upstream loss repair.

## Native inference drain at EOF and shutdown

The later Standard screen v1 exposed a separate shutdown integrity defect.
Spanish E4B `early_2s_160ms` repeat 0 exited with code 0 but its frozen trace
contained an unmatched native partial-STT start/finish. Cancelling the asyncio
executor wrapper did not stop the native call, and the old cleanup could write
the summary before that call finished. The 96-run cohort was aborted after 13
completed process runs and remains failed technical evidence.

In `d8c3f05`, `_drain_inference_workers` cancels publication tasks and queued
inference that has not started, then joins the actual executors off the event
loop before model unload and summary persistence. Already-promised queued TTS
is allowed to finish. This applies at replay EOF and ordinary shutdown; the
process supervisor retains the outer deadline if native work never returns.
It does not change native microphone queue capacity or repair upstream capture
loss by itself.

The distinct v4 integrity pilot used `eddb0ad` for three Spanish E4B runs:
opening control, `early_2s_160ms`, closing control. All completed integrity and
source-ledger checks. The [follow-up status](../mac_followup_20260910/README.md)
keeps this functional result separate from the aborted v1 cohort and the
96-run Standard v2 cohort started at 15:33 UTC on `eddb0ad`, whose completion
and results remain pending. No optimization or microphone certification follows
from these three file-replay runs.
