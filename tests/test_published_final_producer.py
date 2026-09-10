"""Run real producer coroutines with deterministic model-free worker boundaries."""

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from engines.base import STTEngine, STTResult
from tools.latency_experiments import LatencyExperiments
from tools.pipeline_timing import ChunkTiming


@pytest.fixture
def pipeline(monkeypatch):
    import dry_run_ab as d

    monkeypatch.setattr(d, "SESSION_ID", "publication-test-session")
    for name in (
        "_partial_emitted_sequence",
        "_partial_source_text",
        "partial_translations",
        "partial_latencies",
        "_rolling_previews",
        "_chunk_timings",
        "_last_gen_stats",
    ):
        monkeypatch.setattr(d, name, {})
    monkeypatch.setattr(d, "_published_final_utterances", set())
    monkeypatch.setattr(d, "_published_final_session", None)
    monkeypatch.setattr(d, "_published_final_floor", 0)
    monkeypatch.setattr(d, "_discarded_utterance_id", 0)
    monkeypatch.setattr(d, "_closed_utterances", set())
    monkeypatch.setattr(d, "_latency", LatencyExperiments())
    for name in (
        "_stt_scheduler",
        "_incremental_stt",
        "_health",
        "_caption_delivery",
        "_active_partial_future",
        "_pipeline_chunk_queue",
        "_final_pending_utterance_id",
        "tts_engine",
        "MLX_DRAFT_MODEL",
    ):
        monkeypatch.setattr(d, name, None)
    monkeypatch.setattr(d, "_final_pending", threading.Event())
    monkeypatch.setattr(d, "_translation_active", threading.Event())
    monkeypatch.setattr(d, "_latency_event", Mock())
    monkeypatch.setattr(d, "_caption_before_send", Mock())
    monkeypatch.setattr(d, "_io_pool", Mock())
    monkeypatch.setattr(d, "_is_garbage_text", lambda text: False)
    monkeypatch.setattr(d, "_should_suppress", lambda *a, **k: None)
    monkeypatch.setattr(d, "_confirmed_speculation", lambda *a: None)
    monkeypatch.setattr(d, "should_use_marian_only", lambda *a: False)
    monkeypatch.setattr(d, "qe_score", lambda *a: 0.9)
    monkeypatch.setattr(d, "get_resource_snapshot", lambda: {})
    for name in ("check_homophones", "check_bad_split", "check_near_miss", "check_marian_divergence"):
        monkeypatch.setattr(d, name, Mock())
    monkeypatch.setattr(d, "diag_durations", [])
    monkeypatch.setattr(d, "all_results", [])
    monkeypatch.setattr(d, "MULTIPROCESS", False)
    monkeypatch.setattr(d, "BACKEND", "mlx")
    monkeypatch.setattr(d, "DIARIZE_ENABLED", False)
    monkeypatch.setattr(d, "mlx_a_model", object())
    monkeypatch.setattr(d, "mlx_b_model", None)
    monkeypatch.setattr(d.settings.translation, "final_aware_partials", False)
    monkeypatch.setattr(d, "ws_clients", set())
    assert not d._preview_ordering_enabled()
    return d


async def final(d, uid=7):
    await d._pipeline_translate_and_finalize(
        3,
        "Careful final source.",
        10,
        0.9,
        [],
        [],
        np.ones(16000),
        time.perf_counter(),
        timing=ChunkTiming(utterance_id=uid),
    )


def test_coordinator_translation_start_preserves_capture_identity_distinct_from_chunk_id(pipeline, monkeypatch):
    d = pipeline
    messages = []
    producer_messages = []
    actual_broadcast = d.broadcast
    monkeypatch.setattr(d, "chunk_id", 2)
    monkeypatch.setattr(d, "prev_text", "")
    monkeypatch.setattr(d, "_last_final_text", "")
    monkeypatch.setattr(d, "correct_stt_output", lambda text: (text, []))
    monkeypatch.setattr(d, "_run_stt", lambda *args: ("Actual coordinator source.", 10, 0.9, [], []))
    finalize = AsyncMock()
    monkeypatch.setattr(d, "_pipeline_translate_and_finalize", finalize)

    async def run():
        async def send(raw):
            messages.append(json.loads(raw))

        async def observe_producer(data):
            # Assert the coordinator payload before broadcast can enrich it
            # from the chunk timing lookup; otherwise that fallback masks a
            # missing producer field (including when no clients are connected).
            producer_messages.append(dict(data))
            await actual_broadcast(data)

        timing = ChunkTiming(utterance_id=17)
        queue = asyncio.Queue()
        queue.put_nowait((np.ones(16000), time.perf_counter(), None, timing))
        queue.put_nowait(None)
        monkeypatch.setattr(d, "_pipeline_chunk_queue", queue)
        monkeypatch.setattr(d, "ws_clients", {Mock(send=send)})
        monkeypatch.setattr(d, "broadcast", observe_producer)
        await asyncio.wait_for(d._pipeline_coordinator(), 1)
        assert len(producer_messages) == 1
        assert producer_messages[0]["chunk_id"] == 3
        assert producer_messages[0]["utterance_id"] == 17
        assert len(messages) == 1
        event = messages[0]
        assert event["type"] == "translation_start"
        assert event["chunk_id"] == 3
        assert event["utterance_id"] == 17
        assert event["english"] == "Actual coordinator source."
        assert finalize.await_args.args[0] == 3
        assert finalize.await_args.kwargs["timing"] is timing

    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(d, "_pipeline_pool", pool)
        asyncio.run(run())


@pytest.mark.parametrize("ab", [False, True])
@pytest.mark.parametrize("blocked_stage", ["stt", "marian"])
def test_late_worker_result_cannot_republish_or_recreate_preview_caches(pipeline, monkeypatch, blocked_stage, ab):
    d = pipeline
    release = threading.Event()
    messages = []

    async def run():
        entered = asyncio.Event()
        loop = asyncio.get_running_loop()

        def block(stage):
            if stage == blocked_stage:
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(3)

        def stt(*args, **kwargs):
            block("stt")
            return STTResult("Provisional source.", 10, confidence=0.9)

        def marian(text):
            block("marian")
            return "Traducción provisional.", 10

        async def send(raw):
            messages.append(json.loads(raw))

        engine = Mock(spec=STTEngine)
        engine.transcribe.side_effect = stt
        monkeypatch.setattr(d, "stt_pipe", engine)
        monkeypatch.setattr(d, "translate_marian", marian)
        monkeypatch.setattr(d, "translate_mlx_streaming", lambda *a, **k: ("Final A.", 20, 10))
        monkeypatch.setattr(d, "translate_mlx", lambda *a, **k: ("Final B.", 25, 10))
        monkeypatch.setattr(d, "mlx_b_model", object() if ab else None)
        monkeypatch.setattr(d, "ws_clients", {Mock(send=send)})
        monkeypatch.setattr(d, "_pipeline_translation_lock", asyncio.Lock())
        task = asyncio.create_task(d.process_partial(np.ones(16000), 7))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await final(d)
            assert messages[-1]["stage"] == "complete"
            assert messages[-1]["chunk_id"] == 3 and messages[-1]["utterance_id"] == 7
            assert messages[-1]["spanish_b"] == ("Final B." if ab else None)
            release.set()
            await asyncio.wait_for(task, 1)
            assert not any(m["stage"] == "partial" for m in messages)
            assert not d.partial_translations and not d.partial_latencies
            assert not d._partial_source_text and not d._partial_emitted_sequence
            assert all(call.args[0] is not d._write_partial_record for call in d._io_pool.submit.call_args_list)
            before = engine.transcribe.call_count
            await d.process_partial(np.ones(16000), 7)
            assert engine.transcribe.call_count == before
            # An unrelated utterance still gets an ordinary preview.
            await d.process_partial(np.ones(16000), 8)
            assert messages[-1]["stage"] == "partial" and messages[-1]["utterance_id"] == 8
        finally:
            release.set()
            await task

    with ThreadPoolExecutor(max_workers=3) as pool:
        monkeypatch.setattr(d, "_pipeline_pool", pool)
        monkeypatch.setattr(d, "_pytorch_pool", pool)
        asyncio.run(run())


def test_default_preview_remains_allowed_while_final_translation_computes(pipeline, monkeypatch):
    d = pipeline
    release = threading.Event()
    messages = []

    async def run():
        entered = asyncio.Event()
        loop = asyncio.get_running_loop()

        def translate(*args, **kwargs):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(3)
            return "Final A.", 20, 10

        async def send(raw):
            messages.append(json.loads(raw))

        engine = Mock(spec=STTEngine)
        engine.transcribe.return_value = STTResult("Provisional source.", 10, confidence=0.9)
        monkeypatch.setattr(d, "stt_pipe", engine)
        monkeypatch.setattr(d, "translate_marian", lambda text: ("Preview.", 10))
        monkeypatch.setattr(d, "translate_mlx_streaming", translate)
        monkeypatch.setattr(d, "ws_clients", {Mock(send=send)})
        monkeypatch.setattr(d, "_pipeline_translation_lock", asyncio.Lock())
        task = asyncio.create_task(final(d))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            assert not d._preview_was_finalized(7)
            await d.process_partial(np.ones(16000), 7)
            assert [m["stage"] for m in messages] == ["partial"]
            release.set()
            await asyncio.wait_for(task, 1)
            assert [m["stage"] for m in messages] == ["partial", "complete"]
            assert not d.partial_translations and not d.partial_latencies
        finally:
            release.set()
            await task

    with ThreadPoolExecutor(max_workers=3) as pool:
        monkeypatch.setattr(d, "_pipeline_pool", pool)
        monkeypatch.setattr(d, "_pytorch_pool", pool)
        asyncio.run(run())


def test_no_client_final_is_session_scoped_bounded_and_never_uses_final_chunk_id(pipeline, monkeypatch):
    d = pipeline

    def message(uid, **changes):
        return {
            "type": "translation",
            "stage": "complete",
            "session_id": d.SESSION_ID,
            "utterance_id": uid,
            "chunk_id": 9000,
            **changes,
        }

    async def run():
        for bad in (None, True, 0, -1, 2.5, "7", 2**53):
            await d.broadcast(message(bad))
        await d.broadcast(message(7, session_id="other-session"))
        await d.broadcast(message(7, stage="translation_a"))
        assert not d._published_final_utterances
        for uid in range(1, 260):
            await d.broadcast(message(uid))
        assert len(d._published_final_utterances) == 128
        assert d._preview_was_finalized(1) and d._preview_was_finalized(259)
        assert not d._preview_was_finalized(260) and not d._preview_was_finalized(9000)
        health = Mock()
        monkeypatch.setattr(d, "_health", health)
        await d.broadcast(message(7, stage="partial"))
        health.caption.assert_not_called()
        monkeypatch.setattr(d, "SESSION_ID", "new-session")
        assert not d._preview_was_finalized(7)
        await d.broadcast(message(7, stage="partial"))
        health.caption.assert_called_once()
        await d.broadcast(message(7))
        assert d._published_final_utterances == {7} and d._published_final_floor == 0

    asyncio.run(run())


def test_old_session_worker_cannot_publish_into_new_session_with_reused_utterance(pipeline, monkeypatch):
    d = pipeline
    release = threading.Event()
    messages = []

    async def run():
        entered = asyncio.Event()
        loop = asyncio.get_running_loop()

        def translate(text):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(3)
            return "Preview.", 10

        async def send(raw):
            messages.append(json.loads(raw))

        engine = Mock(spec=STTEngine)
        engine.transcribe.return_value = STTResult("Provisional source.", 10, confidence=0.9)
        monkeypatch.setattr(d, "stt_pipe", engine)
        monkeypatch.setattr(d, "translate_marian", translate)
        monkeypatch.setattr(d, "ws_clients", {Mock(send=send)})
        task = asyncio.create_task(d.process_partial(np.ones(16000), 7))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            monkeypatch.setattr(d, "SESSION_ID", "new-session")
            release.set()
            await asyncio.wait_for(task, 1)
            assert not messages and not d.partial_translations and not d.partial_latencies
            d._io_pool.submit.assert_not_called()
            await d.process_partial(np.ones(16000), 7)
            assert len(messages) == 1 and messages[0]["session_id"] == "new-session"
            assert messages[0]["stage"] == "partial" and messages[0]["utterance_id"] == 7
        finally:
            release.set()
            await task

    with ThreadPoolExecutor(max_workers=2) as pool:
        monkeypatch.setattr(d, "_pipeline_pool", pool)
        monkeypatch.setattr(d, "_pytorch_pool", pool)
        asyncio.run(run())
