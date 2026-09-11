"""Experiments observe physical work, offload Marian, and drain all finalizers."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from tools.pipeline_timing import ChunkTiming


@pytest.mark.parametrize("kind", ["partial", "final"])
def test_idle_tracking_waits_for_worker_after_asyncio_future_is_cancelled(monkeypatch, kind):
    import dry_run_ab as pipeline

    started, release = threading.Event(), threading.Event()
    monkeypatch.setattr(pipeline, "_active_stt_workers", {"partial": 0, "final": 0})
    monkeypatch.setattr(pipeline, "_experiment_counters", {})
    monkeypatch.setattr(pipeline, "_translation_active", threading.Event())
    monkeypatch.setattr(pipeline, "_final_pending", threading.Event())
    monkeypatch.setattr(pipeline, "_pipeline_chunk_queue", None)
    monkeypatch.setattr(pipeline, "_active_partial_future", None)

    def work():
        started.set()
        assert release.wait(2)

    async def exercise(pool):
        future = asyncio.get_running_loop().run_in_executor(pool, pipeline._run_tracked_stt, kind, work)
        await asyncio.to_thread(started.wait, 1)
        assert started.is_set()
        future.cancel()
        assert future.done()
        assert not pipeline._inference_idle()
        release.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        asyncio.run(exercise(pool))
    assert pipeline._inference_idle()
    counts = pipeline._experiment_snapshot()
    assert counts[f"{kind}_stt_started"] == counts[f"{kind}_stt_finished"] == 1


def test_opt_in_partial_suppression_is_counted_without_starting_inference(monkeypatch):
    import dry_run_ab as pipeline

    monkeypatch.setattr(pipeline, "_experiment_counters", {})
    active = threading.Event()
    active.set()
    monkeypatch.setattr(pipeline, "_translation_active", active)
    monkeypatch.setattr(pipeline.settings.translation, "final_aware_partials", True)
    asyncio.run(pipeline.process_partial(np.ones(16000), 1))
    snapshot = pipeline._experiment_snapshot()
    assert snapshot["partial_suppressed_final_decode"] == 1
    assert snapshot["partial_stt_started"] == 0


def test_marian_final_runs_on_cpu_pool_and_keeps_routing_count(monkeypatch):
    import dry_run_ab as pipeline

    loop_thread = threading.get_ident()
    monkeypatch.setattr(pipeline, "_experiment_counters", {})
    monkeypatch.setattr(pipeline, "BACKEND", "mlx")
    monkeypatch.setattr(pipeline, "MULTIPROCESS", False)
    monkeypatch.setattr(pipeline, "_RUN_AB", False)
    monkeypatch.setattr(pipeline, "mlx_a_model", object())
    monkeypatch.setattr(pipeline, "mlx_b_model", None)
    monkeypatch.setattr(pipeline, "DIARIZE_ENABLED", False)
    monkeypatch.setattr(pipeline, "tts_engine", None)
    monkeypatch.setattr(pipeline, "_io_pool", Mock())
    monkeypatch.setattr(pipeline, "all_results", [])
    monkeypatch.setattr(pipeline, "partial_translations", {})
    monkeypatch.setattr(pipeline, "partial_latencies", {})
    monkeypatch.setattr(pipeline, "_last_gen_stats", {})
    monkeypatch.setattr(pipeline, "should_use_marian_only", lambda *args: True)
    monkeypatch.setattr(pipeline, "qe_score", lambda *args: 1)
    monkeypatch.setattr(pipeline, "get_resource_snapshot", lambda: {})
    monkeypatch.setattr(pipeline, "broadcast", AsyncMock())
    for name in ("check_homophones", "check_bad_split", "check_near_miss"):
        monkeypatch.setattr(pipeline, name, lambda *args: None)

    def marian(text):
        assert threading.get_ident() != loop_thread
        return "Buenos días", 40

    monkeypatch.setattr(pipeline, "translate_marian", marian)

    async def exercise():
        monkeypatch.setattr(pipeline, "_pipeline_translation_lock", asyncio.Lock())
        await pipeline._pipeline_translate_and_finalize(1, "Good morning", 100, 0.9, [], [], np.ones(16000), 1)

    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(pipeline, "_pytorch_pool", pool)
        asyncio.run(exercise())
    assert pipeline.all_results[0]["spanish_a"] == "Buenos días"
    assert pipeline._experiment_snapshot()["final_marian_routes"] == 1
    assert pipeline._experiment_snapshot()["final_gemma_requests"] == 0


def test_coordinator_shutdown_waits_for_an_older_finalizer(monkeypatch):
    import dry_run_ab as pipeline

    monkeypatch.setattr(pipeline, "MULTIPROCESS", False)
    monkeypatch.setattr(pipeline, "_active_stt_workers", {"partial": 0, "final": 0})
    monkeypatch.setattr(pipeline, "_experiment_counters", {})
    monkeypatch.setattr(pipeline, "chunk_id", 0)
    monkeypatch.setattr(pipeline, "prev_text", "")
    monkeypatch.setattr(pipeline, "_last_final_text", "")
    monkeypatch.setattr(pipeline, "_chunk_timings", {})
    monkeypatch.setattr(pipeline, "_final_pending", threading.Event())
    monkeypatch.setattr(pipeline, "broadcast", AsyncMock())
    monkeypatch.setattr(pipeline, "_is_garbage_text", lambda *args: False)
    monkeypatch.setattr(pipeline, "_should_suppress", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "correct_stt_output", lambda text: (text, []))
    monkeypatch.setattr(pipeline, "_run_stt", lambda audio, prompt: (str(audio[0]), 1, 0.9, [], []))

    async def exercise():
        first_release, second_done = asyncio.Event(), asyncio.Event()

        async def finalize(cid, *args, **kwargs):
            if cid == 1:
                await first_release.wait()
            else:
                second_done.set()

        monkeypatch.setattr(pipeline, "_pipeline_translate_and_finalize", finalize)
        queue = asyncio.Queue()
        for number in (1, 2):
            queue.put_nowait((np.ones(16000) * number, 1, 1, ChunkTiming(utterance_id=number)))
        queue.put_nowait(None)
        monkeypatch.setattr(pipeline, "_pipeline_chunk_queue", queue)
        coordinator = asyncio.create_task(pipeline._pipeline_coordinator())
        await asyncio.wait_for(second_done.wait(), 1)
        await asyncio.sleep(0)
        assert not coordinator.done()
        first_release.set()
        await asyncio.wait_for(coordinator, 1)

    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(pipeline, "_pipeline_pool", pool)
        asyncio.run(exercise())


def test_zero_output_summary_still_records_experiment_execution(monkeypatch):
    import dry_run_ab as pipeline

    io = Mock()
    monkeypatch.setattr(pipeline, "_io_pool", io)
    monkeypatch.setattr(pipeline, "all_results", [])
    monkeypatch.setattr(pipeline, "_experiment_counters", {"warmup_suppressed_busy": 3})
    pipeline.print_summary()
    record = io.submit.call_args.args[1]
    assert record["event"] == "session_summary"
    assert record["latency_experiment_counters"]["warmup_suppressed_busy"] == 3
    assert record["latency_experiment_counters"]["warmup_executed"] == 0


def test_session_drain_joins_cancelled_native_stt_before_summary(monkeypatch):
    import dry_run_ab as pipeline
    from tools.latency_trace import LatencyTrace

    started, release = threading.Event(), threading.Event()
    trace = LatencyTrace(True)
    monkeypatch.setattr(pipeline, "_latency_trace", trace)
    monkeypatch.setattr(pipeline, "_active_stt_workers", {"partial": 0, "final": 0})
    monkeypatch.setattr(pipeline, "_experiment_counters", {})
    monkeypatch.setattr(pipeline, "_partial_tasks", set())
    for name in ("_pytorch_pool", "_stt_comm_pool", "_trans_comm_pool", "_vad_pool", "_tts_pool"):
        monkeypatch.setattr(pipeline, name, None)
    pool = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(pipeline, "_pipeline_pool", pool)
    queued_ran = threading.Event()

    def native():
        started.set()
        assert release.wait(3)
        return "result"

    async def exercise():
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(pool, pipeline._run_tracked_stt, "partial", native)
        assert await asyncio.to_thread(started.wait, 1)
        future.cancel()
        assert future.done() and pipeline._active_stt_workers["partial"] == 1
        queued = pool.submit(queued_ran.set)
        drain = asyncio.create_task(pipeline._drain_inference_workers())
        # Event loop remains available to release native work; the canceled
        # wrapper alone must not let the drain return or the summary freeze.
        for _ in range(100):
            if queued.cancelled():
                break
            await asyncio.sleep(0.001)
        assert queued.cancelled() and not drain.done()
        release.set()
        await asyncio.wait_for(drain, 2)
        events = trace.snapshot()["events"]
        assert [e["event"] for e in events] == ["physical_stt_started", "physical_stt_finished"]
        assert pipeline._active_stt_workers == {"partial": 0, "final": 0}
        assert not queued_ran.is_set()

    try:
        asyncio.run(exercise())
    finally:
        release.set()
        pool.shutdown(wait=True, cancel_futures=True)


def test_session_drain_preserves_queued_tts_for_published_finals(monkeypatch):
    import dry_run_ab as pipeline

    for name in ("_pipeline_pool", "_pytorch_pool", "_stt_comm_pool", "_trans_comm_pool", "_vad_pool"):
        monkeypatch.setattr(pipeline, name, None)
    monkeypatch.setattr(pipeline, "_partial_tasks", set())
    started, release = threading.Event(), threading.Event()
    played = []
    pool = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(pipeline, "_tts_pool", pool)

    def first():
        started.set()
        assert release.wait(3)
        played.append(1)

    async def exercise():
        pool.submit(first)
        assert await asyncio.to_thread(started.wait, 1)
        second = pool.submit(played.append, 2)
        drain = asyncio.create_task(pipeline._drain_inference_workers())
        await asyncio.sleep(0.02)
        assert not drain.done() and not second.cancelled()
        release.set()
        await asyncio.wait_for(drain, 2)
        assert played == [1, 2]

    try:
        asyncio.run(exercise())
    finally:
        release.set()
        pool.shutdown(wait=True)
