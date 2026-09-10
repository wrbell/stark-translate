"""Queue telemetry observes real asyncio backpressure without importing models."""

import asyncio
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from tools.final_queue_pressure import FinalQueuePressure
from tools.latency_trace import LatencyTrace
from tools.pipeline_timing import ChunkTiming


def test_capacity_blocked_producer_and_actual_enqueue_events(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("tools.latency_trace.time.perf_counter", lambda: clock[0])
    trace = LatencyTrace(True, origin=99.0)
    ledger = FinalQueuePressure(origin=trace.origin, on_event=trace.record, clock=lambda: clock[0])
    first, second = object(), object()

    async def exercise():
        queue = asyncio.Queue(maxsize=1)
        await ledger.put(queue, first, utterance_id=7)
        clock[0] = 101.0
        producer = asyncio.create_task(ledger.put(queue, second, utterance_id=7))
        await asyncio.sleep(0)
        assert not producer.done()
        snapshot = ledger.snapshot()
        assert (snapshot["submitted"], snapshot["terminal_queued"], snapshot["terminal_producer_waiting"]) == (1, 1, 1)
        assert snapshot["max_pending"] == 2
        clock[0] = 103.0
        item = await queue.get()
        assert item is first
        ledger.dequeued(item)
        await producer
        clock[0] = 105.0
        item = await queue.get()
        assert item is second
        ledger.dequeued(item)
        assert queue.empty() and queue.maxsize == 1

    asyncio.run(exercise())
    snapshot = ledger.snapshot()
    assert snapshot["max_queued"] == 1
    assert snapshot["max_producer_waiting"] == 1
    assert snapshot["max_wait_ms"] == 4000
    assert snapshot["max_queue_wait_ms"] == 3000
    assert snapshot["max_producer_wait_ms"] == 2000
    assert snapshot["terminal_outstanding"] == 0
    assert snapshot["bookkeeping_truncated"] is False
    events = trace.snapshot()["events"]
    assert [row["event"] for row in events] == [
        "final_queue_submit_started",
        "final_queue_submitted",
        "final_queue_submit_started",
        "final_queue_dequeued",
        "final_queue_submitted",
        "final_queue_dequeued",
    ]
    assert [row["submission_id"] for row in events if row["event"] == "final_queue_dequeued"] == [1, 2]
    assert all(row["utterance_id"] == 7 for row in events)
    assert all(row["at_ms"] == row["queue_at_ms"] for row in events)
    assert events[-1]["at_ms"] == 6000  # same session trace origin, not a second stopwatch


def test_cancelled_producer_retains_queued_work_and_terminal_age():
    clock = [0.0]
    ledger = FinalQueuePressure(clock=lambda: clock[0])

    async def exercise():
        queue = asyncio.Queue(maxsize=1)
        first = object()
        await ledger.put(queue, first)
        producer = asyncio.create_task(ledger.put(queue, object()))
        await asyncio.sleep(0)
        clock[0] = 2.0
        producer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await producer
        snapshot = ledger.snapshot()
        assert snapshot["put_failed"] == 1
        assert snapshot["submitted"] == 1
        assert snapshot["terminal_outstanding"] == snapshot["terminal_queued"] == 1
        assert snapshot["terminal_producer_waiting"] == 0
        assert snapshot["terminal_oldest_wait_ms"] == 2000
        assert await queue.get() is first
        ledger.dequeued(first)

    asyncio.run(exercise())
    assert ledger.snapshot()["terminal_outstanding"] == 0
    assert ledger.snapshot()["bookkeeping_active"] == 0


def test_failed_put_propagates_and_new_session_starts_empty():
    class BrokenQueue:
        maxsize = 8

        async def put(self, item):
            raise RuntimeError("put failed")

    ledger = FinalQueuePressure()
    with pytest.raises(RuntimeError, match="put failed"):
        asyncio.run(ledger.put(BrokenQueue(), object()))
    snapshot = ledger.snapshot()
    assert snapshot["submit_attempts"] == snapshot["put_failed"] == 1
    assert snapshot["submitted"] == snapshot["terminal_outstanding"] == 0
    assert snapshot["bookkeeping_active"] == 0
    fresh = FinalQueuePressure(origin=500).snapshot()
    assert fresh["origin_perf_counter_s"] == 500
    assert fresh["submit_attempts"] == fresh["max_pending"] == fresh["put_failed"] == 0
    assert fresh["first_window_wait_ms"] == fresh["last_window_wait_ms"] == []


def test_bounded_wait_windows_show_growth_and_ledger_overflow_is_explicit():
    clock = [0.0]
    ledger = FinalQueuePressure(capacity=2, window_size=2, clock=lambda: clock[0])

    async def exercise():
        queue = asyncio.Queue()
        for duration in (1, 2, 3, 4, 5):
            item = object()
            await ledger.put(queue, item)
            clock[0] += duration
            ledger.dequeued(await queue.get())
        snapshot = ledger.snapshot()
        assert snapshot["first_window_wait_ms"] == [1000, 2000]
        assert snapshot["last_window_wait_ms"] == [4000, 5000]
        assert snapshot["first_window_mean_wait_ms"] == 1500
        assert snapshot["last_window_mean_wait_ms"] == 4500
        assert snapshot["wait_windows_overlap"] is False
        assert snapshot["max_oldest_wait_ms"] == 5000
        assert snapshot["bookkeeping_truncated"] is False
        for _ in range(3):
            await ledger.put(queue, object())
        snapshot = ledger.snapshot()
        assert snapshot["bookkeeping_active"] == 2
        assert snapshot["terminal_outstanding"] == 3
        assert snapshot["bookkeeping_dropped"] == 1
        assert snapshot["bookkeeping_truncated"] is True
        while not queue.empty():
            ledger.dequeued(await queue.get())

    asyncio.run(exercise())
    assert ledger.snapshot()["terminal_outstanding"] == 0
    assert ledger.snapshot()["bookkeeping_truncated"] is True


@pytest.mark.parametrize("capacity,window_size", [(0, 1), (1, 0)])
def test_invalid_bounds_rejected(capacity, window_size):
    with pytest.raises(ValueError, match="positive"):
        FinalQueuePressure(capacity=capacity, window_size=window_size)


def test_real_pipeline_dequeues_before_stt_errors_and_excludes_sentinel(monkeypatch):
    import dry_run_ab as pipeline

    ledger = FinalQueuePressure(origin=pipeline._SESSION_CLOCK_ORIGIN)
    monkeypatch.setattr(pipeline, "_final_queue_pressure", ledger)
    monkeypatch.setattr(pipeline, "_chunk_timings", {})
    monkeypatch.setattr(pipeline, "_stt_scheduler", None)
    monkeypatch.setattr(pipeline, "_health", None)
    monkeypatch.setattr(pipeline, "MULTIPROCESS", False)
    for name in ("chunk_id", "_pipeline_total", "_pipeline_overlaps", "_chunks_attempted"):
        monkeypatch.setattr(pipeline, name, 0)
    monkeypatch.setattr(pipeline, "_run_tracked_stt", Mock(side_effect=RuntimeError("synthetic STT failure")))

    async def exercise():
        queue = asyncio.Queue(maxsize=8)
        monkeypatch.setattr(pipeline, "_pipeline_chunk_queue", queue)
        # Same/absent utterance IDs cannot alias submissions. A real filter and
        # downstream failure both have already left the input queue.
        await pipeline.pipeline_submit(np.zeros(32), timing=ChunkTiming())
        await pipeline.pipeline_submit(np.ones(32), timing=ChunkTiming())
        await queue.put(None)
        await pipeline._pipeline_coordinator()
        assert queue.empty()

    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(pipeline, "_pipeline_pool", pool)
        asyncio.run(exercise())
    snapshot = ledger.snapshot()
    assert snapshot["submitted"] == snapshot["dequeued"] == 2
    assert snapshot["terminal_outstanding"] == 0
    assert snapshot["bookkeeping_truncated"] is False
    monkeypatch.setattr(pipeline, "all_results", [])
    writer = Mock()
    monkeypatch.setattr(pipeline, "_io_pool", writer)
    pipeline.print_summary()
    assert writer.submit.call_args.args[1]["final_queue_pressure"]["dequeued"] == 2


def test_failed_get_does_not_invent_dequeue(monkeypatch):
    import dry_run_ab as pipeline

    class BrokenQueue(asyncio.Queue):
        async def get(self):
            raise RuntimeError("get failed")

    ledger = FinalQueuePressure()
    queue = BrokenQueue(maxsize=8)
    asyncio.run(ledger.put(queue, object()))
    monkeypatch.setattr(pipeline, "_pipeline_chunk_queue", queue)
    monkeypatch.setattr(pipeline, "_final_queue_pressure", ledger)
    with pytest.raises(RuntimeError, match="get failed"):
        asyncio.run(pipeline._pipeline_coordinator())
    assert ledger.snapshot()["dequeued"] == 0
    assert ledger.snapshot()["terminal_outstanding"] == 1


def test_helper_import_is_standard_library_only():
    source = Path(__file__).parents[1] / "tools" / "final_queue_pressure.py"
    script = """
import runpy, sys
before = set(sys.modules)
runpy.run_path(sys.argv[1])
assert not ({'torch', 'mlx', 'numpy', 'transformers', 'sounddevice'} & (set(sys.modules) - before))
"""
    subprocess.run([sys.executable, "-I", "-c", script, str(source)], check=True, timeout=10)
