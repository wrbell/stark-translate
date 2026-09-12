"""First-stream producer and before-send telemetry without loading models."""

import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from tools.caption_delivery import CaptionDelivery
from tools.latency_experiments import LatencyExperiments
from tools.pipeline_timing import SAMPLE_COLUMNS, ChunkTiming, RenderTracker


def test_broadcaster_marks_only_first_batch_of_interleaved_chunks(monkeypatch):
    import dry_run_ab as pipeline

    broadcaster = AsyncMock()
    monkeypatch.setattr(pipeline, "broadcast", broadcaster)
    monkeypatch.setattr(pipeline, "SESSION_ID", "s")
    monkeypatch.setattr(pipeline, "ws_clients", set())

    async def run():
        queue = asyncio.Queue()
        monkeypatch.setattr(pipeline, "_stream_token_queue", queue)
        for item in [
            ("token", 1, "Hola", 3),
            ("token", 2, "Dos", 3),
            ("token", 1, "Hola mundo", 6),
            ("token", 2, "Dos veces", 6),
            None,
        ]:
            queue.put_nowait(item)
        await pipeline.stream_token_broadcaster()

    asyncio.run(run())
    payloads = [call.args[0] for call in broadcaster.await_args_list]
    assert [p.get("event_id") for p in payloads] == ["s:stream:1", "s:stream:2", None, None]
    assert [p["chunk_id"] for p in payloads] == [1, 2, 1, 2]
    assert [p["tokens_so_far"] for p in payloads] == [3, 3, 6, 6]
    assert payloads[2]["partial_spanish_a"] == "Hola mundo"


@pytest.mark.parametrize("queued", [False, True])
def test_broadcaster_through_actual_delivery_registers_first_batch_only(monkeypatch, queued):
    import dry_run_ab as pipeline

    tracker = RenderTracker()
    monkeypatch.setattr(pipeline, "SESSION_ID", "s")
    monkeypatch.setattr(pipeline, "_render_tracker", tracker)
    monkeypatch.setattr(pipeline, "_chunk_timings", {1: ChunkTiming(utterance_id=17, speech_end=19)})
    monkeypatch.setattr(pipeline, "_latency", replace(LatencyExperiments(), async_captions=queued))
    monkeypatch.setattr(pipeline, "_caption_delivery", None)
    monkeypatch.setattr(pipeline, "_broadcast_sequence", 0)
    monkeypatch.setattr(pipeline.time, "perf_counter", lambda: 20)

    async def run():
        queue = asyncio.Queue()
        monkeypatch.setattr(pipeline, "_stream_token_queue", queue)
        messages = []

        class Client:
            async def send(self, payload):
                messages.append(json.loads(payload))
                if len(messages) == 1:
                    queue.put_nowait(("token", 1, "Hola mundo", 6))
                else:
                    queue.put_nowait(None)

        client = Client()
        monkeypatch.setattr(pipeline, "ws_clients", {client})
        queue.put_nowait(("token", 1, "Hola", 3))
        await asyncio.wait_for(pipeline.stream_token_broadcaster(), 2)
        if pipeline._caption_delivery:
            await pipeline._caption_delivery.close()
        assert [message["event_id"] for message in messages] == ["s:stream:1", "s:1"]
        assert [message["tokens_so_far"] for message in messages] == [3, 6]
        assert len(tracker.pending) == 1
        ack = tracker.acknowledge(client, {"event_id": "s:stream:1", "visible": True, "receive_to_render_ms": 32}, 20.1)
        assert ack["stage"] == "first_stream"
        assert ack["speech_end_to_ack_upper_bound_ms"] == 1100
        assert ack["utterance_id"] == 17
        assert ack["caption_delivery_mode"] == ("queued" if queued else "awaited")

    asyncio.run(run())


@pytest.mark.parametrize("source", ["portaudio_adc", "replay_nonrealtime", None])
def test_before_send_first_stream_uses_chunk_timing_and_only_special_ids(monkeypatch, source):
    import dry_run_ab as pipeline

    tracker, client = RenderTracker(), object()
    timing = ChunkTiming(
        utterance_id=17,
        timing_source=source,
        speech_end=19,
        sample_start=16000,
        sample_end=32000,
        sample_rate=16000,
        speech_end_sample=30000,
    )
    monkeypatch.setattr(pipeline, "_chunk_timings", {1: timing} if source else {})
    monkeypatch.setattr(pipeline, "_render_tracker", tracker)
    payload = {
        "type": "translation_stream",
        "event_id": "s:stream:1",
        "chunk_id": 1,
        "tokens_so_far": 3,
        "caption_delivery_mode": "queued",
        "utterance_id": -1,  # producer timing, not payload overrides, owns these fields
        "timing_source": "wrong",
        "sample_end": -1,
    }
    for event_id in [None, 42, "s:2"]:
        pipeline._caption_before_send(client, {**payload, "event_id": event_id}, 20, 1)
    assert not tracker.pending
    pipeline._caption_before_send(client, payload, 20, 12.3456)
    ack = tracker.acknowledge(client, {"event_id": "s:stream:1", "visible": True, "receive_to_render_ms": 32}, 20.1)
    assert ack["stage"] == "first_stream"
    assert ack["speech_end_to_ack_upper_bound_ms"] == (1100 if source == "portaudio_adc" else None)
    assert ack["utterance_id"] == (17 if source else None)
    assert ack["timing_source"] == (source or "unknown")
    assert ack["caption_delivery_mode"] == "queued"
    assert ack["caption_queue_wait_ms"] == 12.346
    assert ack["tokens_so_far"] == 3
    assert {key: ack[key] for key in SAMPLE_COLUMNS} == (
        timing.sample_metadata() if source else dict.fromkeys(SAMPLE_COLUMNS)
    )


def test_coalesced_first_stream_never_registers_a_render_event(monkeypatch):
    import dry_run_ab as pipeline

    tracker = RenderTracker()
    monkeypatch.setattr(pipeline, "_render_tracker", tracker)

    class Client:
        def __init__(self):
            self.messages = []

        async def send(self, message):
            self.messages.append(json.loads(message))

    async def run():
        client = Client()
        delivery = CaptionDelivery(before_send=pipeline._caption_before_send)
        payload = {"type": "translation_stream", "session_id": "s", "chunk_id": 1, "tokens_so_far": 3}
        # Publish before the writer runs, so the second batch supersedes the first.
        delivery.publish(client, {**payload, "event_id": "s:stream:1"})
        delivery.publish(client, {**payload, "event_id": "s:2", "tokens_so_far": 6})
        await delivery.close()
        assert [row["event_id"] for row in client.messages] == ["s:2"]
        assert not tracker.pending

    asyncio.run(run())
