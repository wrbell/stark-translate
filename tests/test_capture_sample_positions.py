"""Absolute capture coordinates survive buffering, resampling, cuts and captions."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import numpy as np
import pytest

from tools.pipeline_timing import (
    SAMPLE_COLUMNS,
    AudioFrame,
    AudioTimeline,
    CaptureSampleClock,
    CaptureStamp,
    ChunkTiming,
)


def test_capture_counter_advances_even_when_intervening_frames_are_not_buffered():
    clock = CaptureSampleClock()
    first = clock.capture(1536, 48000, None, received=10)
    clock.capture(1536, 48000, None, received=10.032)  # muted/discarded frame
    last = clock.capture(1536, 48000, None, received=10.064)
    assert (first.sample_start, first.sample_end, first.sample_rate) == (0, 1536, 48000)
    assert (last.sample_start, last.sample_end) == (3072, 4608)
    frame = AudioFrame(np.ones(512), last)  # downstream 48k -> 16k resampling
    assert (frame.sample_start, frame.sample_end, frame.sample_rate) == (3072, 4608, 48000)
    replay = CaptureStamp(20, 20.032, "replay_realtime", 12000, 12100, 16000, 412)
    assert clock.capture(512, 16000, replay) is replay


def test_smart_cuts_preserve_original_rate_positions_through_resampling():
    timeline = AudioTimeline()
    timeline.append(512, CaptureStamp(10, 10.032, "portaudio_adc", 48000, 49536, 48000), True)
    timeline.append(512, CaptureStamp(10.032, 10.064, "portaudio_adc", 49536, 51072, 48000), False)
    first = timeline.split(600)
    second = timeline.split(100)
    assert first.sample_metadata() == {
        "sample_start": 48000,
        "sample_end": 49800,
        "sample_rate": 48000,
        "speech_end_sample": 49536,
        "padding_samples": 0,
    }
    assert second.sample_metadata()["sample_start"] == first.sample_metadata()["sample_end"]
    assert second.sample_metadata()["sample_end"] == timeline.sample_metadata()["sample_start"] == 50100
    assert timeline.sample_metadata()["sample_end"] == 51072
    timing = ChunkTiming.from_timeline(first, 1, "smart_cut")
    assert all(timing.metrics()[name] == first.sample_metadata()[name] for name in SAMPLE_COLUMNS)
    assert set(SAMPLE_COLUMNS).isdisjoint(timing.relative_stages(9))


def test_eof_padding_is_not_spread_across_real_audio_when_a_frame_is_cut():
    timeline = AudioTimeline()
    timeline.append(512, CaptureStamp(10, 10.032, "replay_realtime", 16000, 16300, 16000, 212), True)
    real = timeline.split(200)
    assert (real.sample_metadata()["sample_start"], real.sample_metadata()["sample_end"]) == (16000, 16200)
    assert real.sample_metadata()["padding_samples"] == 0
    mixed = timeline.split(150)
    assert mixed.sample_metadata()["sample_start"] == 16200
    assert mixed.sample_metadata()["sample_end"] == 16300
    assert mixed.sample_metadata()["padding_samples"] == 50
    assert mixed.spans[-1][1].audio_end == pytest.approx(10 + 300 / 16000)
    assert mixed.speech_end == pytest.approx(10 + 350 / 16000)  # existing VAD frame-end clock semantics
    assert timeline.sample_metadata()["sample_start"] == timeline.sample_metadata()["sample_end"] == 16300
    assert timeline.sample_metadata()["padding_samples"] == 162
    assert timeline.speech_end == pytest.approx(10.032)
    # A false VAD positive on padding never advances the source sample boundary.
    assert timeline.sample_metadata()["speech_end_sample"] is None


def test_legacy_timing_has_unknown_sample_positions():
    timeline = AudioTimeline()
    timeline.append(512, CaptureStamp(1, 1.032, "legacy"), True)
    first = timeline.split(128)
    assert first.sample_metadata()["sample_start"] is None
    assert first.sample_metadata()["sample_end"] is None
    assert first.sample_metadata()["sample_rate"] is None
    assert first.last == pytest.approx(1.008)


def test_actual_partial_logs_and_broadcasts_the_snapshot_sample_bounds(monkeypatch):
    import dry_run_ab as pipeline

    monkeypatch.setattr(pipeline, "MULTIPROCESS", True)
    monkeypatch.setattr(pipeline, "_final_pending", threading.Event())
    monkeypatch.setattr(pipeline, "_pipeline_chunk_queue", None)
    monkeypatch.setattr(pipeline, "_active_partial_future", None)
    monkeypatch.setattr(pipeline.settings.translation, "final_aware_partials", False)
    monkeypatch.setattr(pipeline, "partial_translations", {})
    monkeypatch.setattr(pipeline, "partial_latencies", {})
    monkeypatch.setattr(pipeline, "_run_partial_stt_via_worker", lambda audio: ("God loves every person", 100))
    monkeypatch.setattr(pipeline, "translate_marian", lambda text: ("Dios ama a cada persona", 40))
    monkeypatch.setattr(pipeline, "_is_garbage_text", lambda text: False)
    monkeypatch.setattr(pipeline, "_should_suppress", lambda *args, **kwargs: None)
    records, captions = [], []
    io = Mock()
    io.submit.side_effect = lambda fn, record: records.append(record)
    monkeypatch.setattr(pipeline, "_io_pool", io)

    async def broadcast(data):
        captions.append(data)

    monkeypatch.setattr(pipeline, "broadcast", broadcast)
    bounds = {
        "sample_start": 32000,
        "sample_end": 48000,
        "sample_rate": 16000,
        "speech_end_sample": 47000,
        "padding_samples": 0,
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        monkeypatch.setattr(pipeline, "_stt_comm_pool", pool)
        monkeypatch.setattr(pipeline, "_pytorch_pool", pool)
        asyncio.run(pipeline.process_partial(np.ones(16000, np.float32) * 0.1, 7, sample_bounds=bounds))
    assert len(records) == len(captions) == 1
    for result in (records[0], captions[0]):
        assert {name: result[name] for name in SAMPLE_COLUMNS} == bounds


def test_final_caption_and_display_ack_share_source_sample_bounds(monkeypatch):
    import dry_run_ab as pipeline
    from tools.pipeline_timing import RenderTracker

    sent = []

    class Socket:
        async def send(self, text):
            sent.append(json.loads(text))

    client = Socket()
    tracker = RenderTracker()
    timing = ChunkTiming(utterance_id=2, sample_start=100, sample_end=16100, sample_rate=16000, speech_end_sample=16000)
    monkeypatch.setattr(pipeline, "ws_clients", {client})
    monkeypatch.setattr(pipeline, "_chunk_timings", {3: timing})
    monkeypatch.setattr(pipeline, "_render_tracker", tracker)
    asyncio.run(pipeline.broadcast({"type": "translation", "stage": "complete", "chunk_id": 3}))
    assert sent[0]["sample_start"] == 100 and sent[0]["sample_end"] == 16100
    import time

    ack = tracker.acknowledge(
        client, {"event_id": sent[0]["event_id"], "receive_to_render_ms": 32, "visible": True}, time.perf_counter()
    )
    assert {name: ack[name] for name in SAMPLE_COLUMNS} == timing.sample_metadata()
