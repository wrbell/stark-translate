"""Timing boundaries exercised with actual producer and finalizer functions."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tools.pipeline_timing import AudioTimeline, CaptureStamp, ChunkTiming, RenderTracker, capture_stamp


def test_adc_clock_mapping_does_not_include_callback_dispatch_delay():
    stamp = capture_stamp(480, 48000, SimpleNamespace(inputBufferAdcTime=41.5, currentTime=41.53), 100)
    assert stamp.start == pytest.approx(99.97)
    assert stamp.end == pytest.approx(99.98)
    assert stamp.source == "portaudio_adc"


def test_cut_retains_samples_and_internal_silence_timestamps():
    timeline = AudioTimeline()
    timeline.append(16000, CaptureStamp(10, 11, "replay_realtime"), True)
    timeline.append(8000, CaptureStamp(11, 11.5, "replay_realtime"), False)
    timeline.append(16000, CaptureStamp(11.5, 12.5, "replay_realtime"), True)
    first = timeline.split(24000)
    assert first.first == 10
    assert first.last == 11.5
    assert first.speech_end == 11
    assert timeline.first == 11.5
    assert timeline.speech_end == 12.5
    remainder_prefix = timeline.split(4000)
    assert remainder_prefix.last == 11.75
    assert timeline.first == 11.75


def test_stage_arithmetic_keeps_utterance_duration_out_of_silence():
    timing = ChunkTiming(
        captured_start=1,
        speech_end=13,
        vad_finalized=13.5,
        submitted=13.5,
        stt_started=13.7,
        translation_requested=14,
        translation_started=14.1,
        translation_finished=14.5,
        final_ready=14.6,
        broadcast_finished=14.65,
    )
    assert timing.metrics()["speech_end_to_final_ms"] == 1600
    assert timing.metrics()["vad_wait_ms"] == 500
    assert timing.metrics()["stt_queue_wait_ms"] == 200
    assert timing.metrics()["translation_queue_wait_ms"] == 100
    assert timing.metrics()["finalization_overhead_ms"] == 100
    assert timing.metrics()["broadcast_ms"] == 50
    assert ChunkTiming().metrics()["speech_end_to_final_ms"] is None


def test_render_ack_upper_bound_is_server_clock_only():
    tracker = RenderTracker(ttl_s=10)
    client = object()
    tracker.sent(client, "a:1", 20, 18, "complete")
    # The protocol accepts a browser duration, never its absolute clock.
    record = tracker.acknowledge(
        client, {"event_id": "a:1", "receive_to_render_ms": 32, "visible": True, "browser_time": 9999999}, 20.1
    )
    assert record["speech_end_to_ack_upper_bound_ms"] == 2100
    assert record["send_to_ack_ms"] == 100
    assert tracker.acknowledge(client, {"event_id": "a:1"}, 21) is None
    tracker.sent(client, "a:2", 20, 18, "complete")
    assert tracker.acknowledge(object(), {"event_id": "a:2"}, 21) is None
    assert tracker.acknowledge(client, {"event_id": "a:2", "receive_to_render_ms": float("nan")}, 21) is None
    tracker.sent(client, "a:3", 20, 18, "complete")
    hidden = tracker.acknowledge(client, {"event_id": "a:3", "receive_to_render_ms": 32, "visible": False}, 21)
    assert hidden["speech_end_to_ack_upper_bound_ms"] is None
    tracker.sent(client, "a:4", 20, 18, "complete")
    tracker.disconnect(client)
    assert not tracker.pending


def test_real_finalizer_times_payload_after_diagnostics_and_before_broadcast(monkeypatch):
    import dry_run_ab as pipeline

    clock = [20.0]
    monkeypatch.setattr(pipeline.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(pipeline, "BACKEND", "mlx")
    monkeypatch.setattr(pipeline, "MULTIPROCESS", False)
    monkeypatch.setattr(pipeline, "mlx_a_model", object())
    monkeypatch.setattr(pipeline, "mlx_a_tokenizer", object())
    monkeypatch.setattr(pipeline, "mlx_b_model", None)
    monkeypatch.setattr(pipeline, "MLX_DRAFT_MODEL", None)
    monkeypatch.setattr(pipeline, "DIARIZE_ENABLED", False)
    monkeypatch.setattr(pipeline, "tts_engine", None)
    monkeypatch.setattr(pipeline, "_io_pool", Mock())
    monkeypatch.setattr(pipeline, "all_results", [])
    monkeypatch.setattr(pipeline, "get_resource_snapshot", lambda: {})
    monkeypatch.setattr(pipeline, "should_use_marian_only", lambda *args: False)
    monkeypatch.setattr(pipeline, "partial_translations", {})
    monkeypatch.setattr(pipeline, "partial_latencies", {})
    monkeypatch.setattr(pipeline, "_last_gen_stats", {})

    def translation(*args, **kwargs):
        clock[0] += 0.4
        return "La gracia de Dios", 400, 10

    def diagnostics(*args):
        clock[0] += 0.2

    monkeypatch.setattr(pipeline, "translate_mlx_streaming", translation)
    monkeypatch.setattr(pipeline, "check_homophones", diagnostics)
    monkeypatch.setattr(pipeline, "check_bad_split", lambda *args: None)
    monkeypatch.setattr(pipeline, "check_near_miss", lambda *args: None)
    messages = []

    async def broadcast(data):
        messages.append(dict(data))
        clock[0] += 0.05

    monkeypatch.setattr(pipeline, "broadcast", broadcast)
    timing = ChunkTiming(utterance_id=4, speech_end=19, vad_finalized=19.5, submitted=19.5)

    async def exercise():
        monkeypatch.setattr(pipeline, "_pipeline_translation_lock", asyncio.Lock())
        await pipeline._pipeline_translate_and_finalize(
            7, "God's grace", 100, 0.9, [], [], np.ones(16000), 19.5, utterance_start=17, timing=timing
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(pipeline, "_pipeline_pool", pool)
        asyncio.run(exercise())
    assert len(messages) == 1
    final = messages[0]
    assert final["speech_end_to_final_ms"] == 1600
    assert final["e2e_latency_ms"] == 900  # unchanged legacy boundary
    assert final["finalization_overhead_ms"] == 200
    assert final["utterance_id"] == 4
    assert final["broadcast_ms"] is None  # cannot know before sending
    assert pipeline.all_results[0]["broadcast_ms"] == 50


def test_delayed_speaker_update_uses_final_chunk_id(monkeypatch):
    import dry_run_ab as pipeline

    messages = []
    monkeypatch.setattr(pipeline, "_speaker_pending", {42: (100, 102, None)})
    monkeypatch.setattr(pipeline, "_lookup_speaker", lambda *args: "Speaker B")
    monkeypatch.setattr(pipeline, "DIARIZE_INTERVAL_S", 0.001)
    monkeypatch.setattr(pipeline, "_write_jsonl_record", lambda *args: None)

    async def broadcast(message):
        messages.append(message)

    monkeypatch.setattr(pipeline, "broadcast", broadcast)

    async def exercise():
        task = asyncio.create_task(pipeline._speaker_update_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(pipeline, "_io_pool", pool)
        asyncio.run(exercise())
    assert messages[0]["chunk_id"] == 42
    assert messages[0]["type"] == "speaker_update"
    assert messages[0]["speaker"] == "Speaker B"


def test_kpi_does_not_count_long_speech_as_a_stall_or_mix_legacy():
    from tools.kpi_report import compute_latency_kpis

    rows = [
        {
            "timing_schema_version": "2",
            "speech_end_to_final_ms": "800",
            "true_e2e_ms": "12800",
            "e2e_latency_ms": "300",
        },
        {"e2e_latency_ms": "9999", "true_e2e_ms": "14000"},
    ]
    result = compute_latency_kpis(rows)
    assert result["e2e_p95"] == 800
    assert result["e2e_count"] == 1
    assert result["stall_count"] == 0
    assert compute_latency_kpis([])["e2e_p50"] is None


def test_output_ack_rejects_malformed_messages_without_affecting_connection(monkeypatch):
    import dry_run_ab as pipeline

    class Socket:
        async def send(self, value):
            pass

        def __aiter__(self):
            return self.items()

        async def items(self):
            for item in ("{", "null", json.dumps({"type": "caption_rendered", "event_id": []}), "[]"):
                yield item

    monkeypatch.setattr(pipeline, "ws_clients", set())
    asyncio.run(pipeline.ws_handler(Socket()))
    assert not pipeline.ws_clients


def test_tts_records_synthesis_separately_from_playback_request(monkeypatch):
    import dry_run_ab as pipeline

    clock = [10.0]
    monkeypatch.setattr(pipeline.time, "perf_counter", lambda: clock[0])
    records = []
    pool = Mock()
    pool.submit.side_effect = lambda fn, record: records.append(record)
    monkeypatch.setattr(pipeline, "_io_pool", pool)
    result = SimpleNamespace(audio=np.zeros(16000), sample_rate=16000, latency_ms=250)

    def synthesize(*args, **kwargs):
        clock[0] += 0.25
        return result

    def play(*args, **kwargs):
        clock[0] += 0.02

    engine = SimpleNamespace(synthesize=synthesize, play=Mock())
    monkeypatch.setattr(pipeline, "_tts_device_resolver", SimpleNamespace(play=play))
    pipeline._run_tts(engine, "Hola", "es", 42, "local", None, speech_end=9, submitted=9.8)
    record = records[0]
    assert record["tts_queue_wait_ms"] == 200
    assert record["tts_synthesis_ms"] == 250
    assert record["speech_end_to_synthesis_ms"] == 1250
    assert record["speech_end_to_playback_request_ms"] == 1250
    assert record["playback_call_ms"] == 20
    assert record["playback_boundary"] == "host_call_not_acoustic_onset"
