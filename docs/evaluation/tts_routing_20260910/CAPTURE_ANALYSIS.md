# Spanish capture loss: bounded source analysis

This analysis uses the failed synthetic microphone session
`20260910_102356_408494_es`, current source and a model-free reproduction. It does
not relabel that session successful or establish that a proposed change fixes
the real microphone path.

## Established loss path

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

The following transport change is now implemented with five focused model-free
regressions in `tests/test_capture_transport.py`; a real microphone retest is
still required. The original failed run remains pre-fix evidence.

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

Then run one separately reserved, traced Spanish synthetic acoustic retest.
Require zero child/handoff/input-queue drops and a completed lifecycle; preserve
the failed original. If upstream loss persists, use measured stage delays to
choose an isolation/scheduling change. Neither widening queues nor asserting
the event loop is “nonblocking” from its async syntax is sufficient evidence.
