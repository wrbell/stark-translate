"""Real framed pipe, bounded handoff and shutdown, without devices or models."""

import io
import json
import queue
import struct
import sys
import threading
import time
from contextlib import ExitStack
from types import SimpleNamespace

from tools.capture_handoff import CaptureHandoff, CaptureTransportSummary
from tools.isolated_audio import IsolatedInputStream
from tools.latency_trace import LatencyTrace


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "bounded capture condition timed out"
        time.sleep(0.001)


def pipe_script(tmp_path, count=48):
    script = tmp_path / "framed_capture.py"
    script.write_text(f"""import json, struct, sys, time
for index in range({count}):
    metadata = dict(frames=1536, channels=1, inputBufferAdcTime=90.0,
        currentTime=90.032, received=time.perf_counter()-1,
        sample_start=index*1536, dropped=0, status='')
    header=json.dumps(metadata).encode()
    sys.stdout.buffer.write(struct.pack('!I',len(header))+header+bytes(1536*4))
    sys.stdout.buffer.flush()
time.sleep(60)
""")
    return [sys.executable, str(script)]


def setup_pipe(tmp_path, *, capacity=32, can_accept=lambda: True):
    callbacks, consumed = queue.Queue(), []
    loop = SimpleNamespace(call_soon_threadsafe=callbacks.put, call_later=lambda delay, fn: callbacks.put(fn))
    trace = LatencyTrace(enabled=True)
    drops = []
    handoff = CaptureHandoff(
        loop,
        lambda audio, frames, stamp, status: consumed.append(stamp),
        can_accept,
        lambda: drops.append(1),
        capacity=capacity,
        wait_for_space=True,
        trace=trace,
        frame_metadata=lambda item: {
            "sample_start": item[2].sample_start,
            "sample_end": item[2].sample_end,
            "sample_rate": item[2].sample_rate,
            "capture_end_perf_counter_s": item[2].end,
        },
    )
    stream = IsolatedInputStream(
        callback=handoff.put,
        samplerate=48000,
        channels=1,
        dtype="float32",
        blocksize=1536,
        device=None,
        argv=pipe_script(tmp_path),
    )
    stream.trace = trace
    return callbacks, consumed, drops, handoff, stream, trace


def test_buffered_child_pipe_preserves_all_frames_with_bounded_handoff(tmp_path):
    callbacks, consumed, drops, handoff, stream, trace = setup_pipe(tmp_path)
    summary = CaptureTransportSummary()
    with ExitStack() as stack:
        stack.callback(summary.record, stream, handoff)
        stack.enter_context(stream)
        stack.enter_context(handoff)
        wait_until(lambda: handoff.qsize() == 32)
        assert callbacks.qsize() == 1
        assert stream._reader.is_alive()
        deadline = time.monotonic() + 2
        while len(consumed) < 48:
            assert time.monotonic() < deadline
            callbacks.get(timeout=1)()
            time.sleep(0.001)
        assert not drops and not handoff.dropped
    assert [s.sample_start for s in consumed] == [n * 1536 for n in range(48)]
    assert all(s.sample_rate == 48000 and s.source == "portaudio_adc" for s in consumed)
    assert not stream._reader.is_alive()
    assert stream._proc.poll() is not None
    segment = summary.snapshot()["segments"][0]
    assert segment["handoff"]["closed"] and segment["handoff"]["pending"] == 0
    assert segment["handoff"]["high_water"] == 32
    assert segment["handoff"]["max_producer_wait_ms"] > 0
    assert segment["pipe"]["max_callback_to_pipe_ms"] >= 999
    pipe = [e for e in trace.snapshot()["events"] if e["event"] == "capture_pipe_received"]
    dequeue = [e for e in trace.snapshot()["events"] if e["event"] == "capture_handoff_dequeued"]
    assert len(pipe) == len(dequeue) == 48
    assert [e["sample_start"] for e in pipe] == [e["sample_start"] for e in dequeue]
    assert all(e["at_ms"] - e["callback_received_at_ms"] >= 999 for e in pipe)


def test_downstream_full_preserves_order_and_close_releases_waiting_reader(tmp_path):
    accept = threading.Event()
    callbacks, consumed, drops, handoff, stream, _ = setup_pipe(tmp_path, capacity=1, can_accept=accept.is_set)
    with ExitStack() as stack:
        stack.enter_context(stream)
        stack.enter_context(handoff)
        wait_until(lambda: handoff.qsize() == 1)
        callbacks.get(timeout=1)()
        assert not consumed and handoff.qsize() == 1
        accept.set()
        callbacks.get(timeout=1)()
        assert consumed[0].sample_start == 0
        wait_until(lambda: handoff.qsize() == 1)
    assert not stream._reader.is_alive() and stream._proc.poll() is not None
    assert handoff.snapshot()["closed"] and handoff.snapshot()["pending"] == 0
    assert drops  # closing with undelivered source frames remains a required loss


def test_native_nonblocking_overflow_keeps_original_drop_spans_bounded():
    callbacks = []
    loop = SimpleNamespace(call_soon_threadsafe=callbacks.append)
    trace = LatencyTrace(enabled=True, origin=10)
    handoff = CaptureHandoff(
        loop,
        lambda x: None,
        lambda: False,
        lambda: None,
        capacity=1,
        trace=trace,
        clock=lambda: 12,
        frame_metadata=lambda item: {"sample_start": item[0], "sample_end": item[0] + 1},
    )
    for value in range(100):
        handoff.put(value)
    snap = handoff.snapshot()
    assert snap["wait_for_space"] is False and snap["dropped_frames"] == 99
    assert snap["drop_spans_truncated"] and len(snap["drop_spans"]) == 32
    assert snap["drop_spans"][-1] == {
        "at_ms": 2000,
        "sample_start": 98,
        "sample_end": 99,
        "reason": "overflow",
        "wait_ms": 0,
    }
    handoff.close()


def test_segment_summary_retains_totals_when_old_detail_is_dropped():
    summary = CaptureTransportSummary()
    stream = SimpleNamespace(capture_snapshot=lambda: {"upstream_dropped_samples": 1536})
    handoff = SimpleNamespace(snapshot=lambda: {"dropped_frames": 1, "closed": True})
    for _ in range(20):
        summary.record(stream, handoff)
    snap = summary.snapshot()
    assert snap["segments_completed"] == 20 and len(snap["segments"]) == 16
    assert snap["segments_truncated"] and snap["handoff_dropped_frames"] == 20
    assert snap["upstream_dropped_samples"] == 30720


def test_pipe_gap_telemetry_retains_original_source_positions_and_callback_clock():
    trace = LatencyTrace(enabled=True)
    received = []
    wire = bytearray()
    callback_at = time.perf_counter() - 2
    for start, dropped in ((0, 0), (960, 480)):
        header = json.dumps(
            dict(
                frames=480,
                channels=1,
                inputBufferAdcTime=90,
                currentTime=90.01,
                received=callback_at,
                sample_start=start,
                dropped=dropped,
                status="",
            )
        ).encode()
        wire.extend(struct.pack("!I", len(header)) + header + bytes(480 * 4))
    stream = IsolatedInputStream(
        callback=lambda *frame: received.append(frame),
        samplerate=48000,
        channels=1,
        dtype="float32",
        blocksize=480,
        device=None,
    )
    stream.sample_offset = 100
    stream.trace = trace
    stream._proc = SimpleNamespace(stdout=io.BytesIO(wire))
    stream._read()
    snap = stream.capture_snapshot()
    assert snap["upstream_dropped_samples"] == 480
    assert snap["source_gaps"] == [{"sample_start": 580, "sample_end": 1060, "sample_rate": 48000}]
    assert snap["max_callback_to_pipe_ms"] >= 1999
    assert received[1][2].sample_start == 1060 and received[1][3] == "capture_overflow:480"
    pipe = [e for e in trace.snapshot()["events"] if e["event"] == "capture_pipe_received"]
    assert all(e["callback_received_at_ms"] == (callback_at - trace.origin) * 1000 for e in pipe)


def test_prolonged_backpressure_is_bounded_consumer_failure_not_false_microphone_idle(tmp_path):
    _, _, _, handoff, stream, _ = setup_pipe(tmp_path, capacity=1)
    stream.idle_timeout = 0.1
    with ExitStack() as stack:
        stack.enter_context(stream)
        stack.enter_context(handoff)
        wait_until(lambda: stream.finished.is_set())
        assert "processing backpressure" in str(stream.error)
        assert "Microphone delivered no samples" not in str(stream.error)
        assert stream.capture_snapshot()["timeout_stage"] == "consumer_backpressure"
    assert not stream._reader.is_alive() and stream._proc.poll() is not None
    spans = handoff.snapshot()["drop_spans"]
    assert [(row["sample_start"], row["sample_end"]) for row in spans] == [(0, 1536), (1536, 3072)]


def test_intentional_replay_close_does_not_mark_waiting_prefetch_as_lost():
    loop = SimpleNamespace(call_soon_threadsafe=lambda fn: None)
    handoff = CaptureHandoff(loop, lambda *a: None, lambda: False, lambda: None, capacity=1, wait_for_space=True)
    handoff.put(0)
    started = threading.Event()

    def producer():
        started.set()
        handoff.put(1)

    thread = threading.Thread(target=producer)
    thread.start()
    assert started.wait(timeout=1)
    wait_until(lambda: handoff.snapshot()["waiting_producers"] == 1)
    handoff.close(record_discard=False)
    handoff.close()  # ExitStack's repeated close must retain the first policy.
    thread.join(timeout=1)
    assert not thread.is_alive() and handoff.dropped == 0
