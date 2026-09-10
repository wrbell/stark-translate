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
