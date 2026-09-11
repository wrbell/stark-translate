"""Exercise the capture loop and late provisional work, without models/devices."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, Mock

import numpy as np
import pytest

from tools.capture_handoff import CaptureTransportSummary
from tools.latency_experiments import LatencyExperiments
from tools.latency_scheduler import PartialRuntimePredictor
from tools.latency_trace import LatencyTrace
from tools.pipeline_timing import AudioFrame, CaptureStamp
from tools.source_coverage import SourceCoverage


@pytest.fixture
def pipeline(monkeypatch):
    import dry_run_ab as d

    for name in (
        "_utterance_start_times",
        "_utterance_timings",
        "_partial_emitted_sequence",
        "_partial_source_text",
        "_pause_epochs",
        "_speculative_candidates",
        "_speculation_attempts",
        "_rolling_previews",
        "partial_translations",
        "partial_latencies",
    ):
        monkeypatch.setattr(d, name, {})
    for name in ("_partial_tasks", "_speculation_tasks", "_closed_utterances"):
        monkeypatch.setattr(d, name, set())
    monkeypatch.setattr(d, "_discarded_utterance_id", 0)
    monkeypatch.setattr(d, "_last_capture_utterance_id", 0)
    monkeypatch.setattr(d, "_latency", LatencyExperiments())
    monkeypatch.setattr(d, "_source_coverage", SourceCoverage())
    monkeypatch.setattr(d, "_capture_transport", CaptureTransportSummary())
    monkeypatch.setattr(d, "_stt_scheduler", None)
    monkeypatch.setattr(d, "_vad_pool", None)
    monkeypatch.setattr(d, "_health", None)
    monkeypatch.setattr(d, "_caption_delivery", None)
    monkeypatch.setattr(d, "_active_partial_future", None)
    monkeypatch.setattr(d, "_final_pending", threading.Event())
    monkeypatch.setattr(d, "_latency_event", Mock())
    monkeypatch.setattr(d, "_test_broadcast_impl", d.broadcast, raising=False)
    monkeypatch.setattr(d, "broadcast", AsyncMock())
    monkeypatch.setattr(d, "SAMPLE_RATE", 16000)
    monkeypatch.setattr(d.settings.vad, "silence_trigger", 0.5)
    monkeypatch.setattr(d.settings.vad, "partial_interval", 0.6)
    monkeypatch.setattr(d.settings.vad, "max_utterance", 8)
    monkeypatch.setattr(d, "mlx_a_model", None)
    monkeypatch.setattr(d, "_schedule_warmup", Mock())
    monkeypatch.setattr(d, "diag_music_holds", [])
    monkeypatch.setattr(d.settings.translation, "final_aware_partials", False)
    monkeypatch.setattr(d, "vad_model", Mock())
    return d


def frame(sample, speech=True, loud=False):
    # Source/capture is 48 kHz; the loop receives already-resampled 16 kHz PCM.
    return (
        AudioFrame(
            np.full(512, 0.3 if loud else 0.1, np.float32),
            CaptureStamp(sample / 48000, (sample + 1536) / 48000, "replay_realtime", sample, sample + 1536, 48000),
        ),
        speech,
    )


async def capture(pipeline, monkeypatch, frames, queue_type=asyncio.Queue, resumed_frames=()):
    from tools import audio_bridge_client

    d = pipeline
    queue = queue_type()
    all_frames = [*frames, *resumed_frames]
    decisions = iter(speech for _, speech in all_frames)
    for audio, _ in frames:
        queue.put_nowait(audio)
    # Scripted context only: never invent capture_snapshot() or loss counters.
    stream = MagicMock(spec=["__enter__", "__exit__"])
    stream.finished = threading.Event()
    stream.finished.set()
    stream.error = None
    stream.source_sample_count = all_frames[-1][0].sample_end if all_frames else 0
    stream.samplerate = 48000
    if resumed_frames:
        entries = 0

        def reopen(*_args):
            nonlocal entries
            if entries:
                for audio, _ in resumed_frames:
                    queue.put_nowait(audio)
            entries += 1
            return stream

        stream.__enter__.side_effect = reopen
    monkeypatch.setattr(audio_bridge_client, "open_audio_stream", lambda **kwargs: stream)
    monkeypatch.setattr(d, "audio_queue", queue)
    monkeypatch.setattr(d, "_pipeline_chunk_queue", asyncio.Queue())
    monkeypatch.setattr(d, "EXIT_AFTER_REPLAY", True)
    monkeypatch.setattr(d, "is_speech", lambda *args: next(decisions))
    monkeypatch.setattr(d, "process_partial", AsyncMock())
    finals = []

    async def save(audio, uid):
        finals.append((audio, uid, d._utterance_timings[uid]))

    monkeypatch.setattr(d, "process_final", save)
    await asyncio.wait_for(d.audio_loop(), 2)
    return finals


def test_music_trace_explains_actual_streak_resets_without_reclassifying_frames(pipeline, monkeypatch):
    trace = LatencyTrace(enabled=True, capacity=1000)
    monkeypatch.setattr(pipeline, "_latency_trace", trace)
    monkeypatch.setattr(pipeline, "MUSIC_HOLDOFF", 0.096)
    monkeypatch.setattr(pipeline, "MUSIC_THRESHOLD", 0.15)
    kinds = [(False, True), (False, True), (False, False), (False, True), (True, True)]
    kinds += [(False, True)] * 3 + [(True, False)] * 25 + [(False, False)] * 20
    frames = [frame(i * 1536, speech, loud) for i, (speech, loud) in enumerate(kinds)]
    finals = asyncio.run(capture(pipeline, monkeypatch, frames))
    events = [row for row in trace.snapshot()["events"] if row["event"] == "vad_complete"]
    # The actual capture loop consumes each controlled VAD decision once. Its
    # source-bound trace distinguishes energy resets from speech resets.
    assert len(events) == len(frames)
    assert [row["music_nonspeech_frames"] for row in events[:8]] == [1, 2, 0, 1, 0, 1, 2, 3]
    assert [row["vad_positive"] for row in events] == [speech for speech, _ in kinds]
    assert [row["frame_rms"] for row in events] == pytest.approx([0.3 if loud else 0.1 for _, loud in kinds])
    for index, row in enumerate(events):
        assert (row["sample_start"], row["sample_end"], row["sample_rate"]) == (index * 1536, (index + 1) * 1536, 48000)
        assert row["processed_samples"] == 512 and row["processing_sample_rate"] == 16000
        assert row["padding_samples"] == 0 and row["elapsed_ms"] >= 0
        assert row["music_holdoff_frames"] == 3 and row["music_resume_frames"] == 15
        assert row["music_threshold"] == 0.15
    assert events[7]["music_hold_active_before_transition"] is False
    assert events[8]["music_hold_active_before_transition"] is True
    assert events[22]["music_speech_frames"] == 15
    assert events[22]["music_hold_active_before_transition"] is True
    assert events[23]["music_hold_active_before_transition"] is False
    assert len(finals) == 1 and finals[0][2].sample_start == 8 * 1536


@pytest.mark.parametrize("pause_frames,threshold_ms", [(5, 160), (8, 240)])
def test_early_clause_uses_actual_pause_and_preserves_remainder(pipeline, monkeypatch, pause_frames, threshold_ms):
    monkeypatch.setattr(
        pipeline, "_latency", replace(LatencyExperiments(), early_clause_s=2, early_clause_pause_ms=threshold_ms)
    )
    stop = 70 + pause_frames + 30
    frames = [frame(i * 1536, i < 70 or 70 + pause_frames <= i < stop) for i in range(stop + 20)]
    finals = asyncio.run(capture(pipeline, monkeypatch, frames))
    assert len(finals) == 2
    first, second = finals
    assert first[2].endpoint_reason == "early_clause"
    assert second[2].endpoint_reason == "silence"
    assert first[2].sample_end == second[2].sample_start == 70 * 1536
    assert first[1] != second[1]
    assert first[2].speech_end_sample == 70 * 1536
    for audio, _uid, timing in finals:
        assert timing.sample_end - timing.sample_start == len(audio) * 3
    assert not pipeline._source_coverage.snapshot()["capture_gaps"]


def test_early_clause_waits_for_configured_minimum(pipeline, monkeypatch):
    monkeypatch.setattr(
        pipeline, "_latency", replace(LatencyExperiments(), early_clause_s=4, early_clause_pause_ms=160)
    )
    frames = [frame(i * 1536, i < 70 or 75 <= i < 135) for i in range(155)]
    finals = asyncio.run(capture(pipeline, monkeypatch, frames))
    assert len(finals) == 1
    assert finals[0][2].endpoint_reason == "early_clause"  # final pause after 4s of buffered speech
    assert finals[0][2].sample_start == 0
    assert finals[0][2].sample_end == 135 * 1536


def test_deadline_deferral_never_calls_stt_or_logs_empty_prediction(pipeline, monkeypatch):
    d = pipeline
    predictor = PartialRuntimePredictor()
    for _ in range(3):
        predictor.observe(1, 300)
    monkeypatch.setattr(d, "_partial_runtime", predictor)
    monkeypatch.setattr(d, "_latency", replace(LatencyExperiments(), partial_deadline_margin_ms=100))
    monkeypatch.setattr(d, "MULTIPROCESS", True)
    recognize, drop = Mock(), Mock()
    monkeypatch.setattr(d, "_run_partial_stt_via_worker", recognize)
    monkeypatch.setattr(d, "_log_stt_drop", drop)
    monkeypatch.setattr(d, "_pipeline_chunk_queue", None)
    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(d, "_stt_comm_pool", pool)
        asyncio.run(d.process_partial(np.full(16000, 0.1), 1, captured_start=0))
    recognize.assert_not_called()
    drop.assert_not_called()
    assert any(call.args == ("partial_deferred_deadline",) for call in d._latency_event.call_args_list)


def test_short_vad_blip_expires_before_post_hymn_speech(pipeline, monkeypatch):
    # Recorded failure shape: one positive VAD frame leaves ~0.5s buffered;
    # later speech was assigned its stale 441.6s source origin across the hymn.
    old_start, resumed_start = 21196800, 34460160
    frames = [frame(old_start)] + [frame(old_start + i * 1536, False) for i in range(1, 22)]
    frames += [frame(resumed_start + i * 1536, i < 25) for i in range(45)]
    finals = asyncio.run(capture(pipeline, monkeypatch, frames))
    assert len(finals) == 1
    audio, uid, timing = finals[0]
    assert uid == 2
    assert timing.sample_start == resumed_start
    assert timing.captured_start == resumed_start / 48000
    assert timing.speech_end_sample == resumed_start + 25 * 1536
    assert (timing.sample_end - timing.sample_start) / 48000 == pytest.approx(len(audio) / 16000)
    assert pipeline._discarded_utterance_id == 1
    assert 1 not in pipeline._utterance_start_times
    assert 1 not in pipeline._utterance_timings
    assert not pipeline.diag_music_holds  # ordinary silence, not an observed hold
    assert any(call.kwargs.get("reason") == "short_silence" for call in pipeline._latency_event.call_args_list)


def test_one_silence_deadline_frame_does_not_become_hidden_gap_in_next_utterance(pipeline, monkeypatch):
    # Chunk141's retained WAV matched source blocks0..15, then17..22 exactly:
    # the deadline frame16 was absent, despite one continuous reported interval.
    start = 86387712
    frames = [frame(start + i * 1536, i < 2) for i in range(17)]
    frames += [frame(start + i * 1536, i < 42) for i in range(17, 62)]
    finals = asyncio.run(capture(pipeline, monkeypatch, frames))
    assert len(finals) == 1
    audio, uid, timing = finals[0]
    assert uid == 2 and timing.sample_start == start + 17 * 1536
    assert (timing.sample_end - timing.sample_start) / 48000 == pytest.approx(len(audio) / 16000)


def test_music_hold_clears_buffer_timeline_and_resumes_with_new_identity(pipeline, monkeypatch):
    monkeypatch.setattr(pipeline, "MUSIC_HOLDOFF", 0.064)
    monkeypatch.setattr(pipeline, "diag_music_holds", [])
    frames = [frame(i * 1536) for i in range(35)]
    frames += [frame(i * 1536, False, True) for i in range(35, 40)]
    # The existing 15-frame decision now retains the accepted speech onset.
    frames += [frame(i * 1536, i < 79) for i in range(40, 99)]
    finals = asyncio.run(capture(pipeline, monkeypatch, frames))
    assert len(finals) == 1
    audio, uid, timing = finals[0]
    assert uid == 2 and timing.sample_start == 40 * 1536
    np.testing.assert_array_equal(audio, np.concatenate([f.samples for f, _ in frames[40:93]]))
    assert timing.speech_end_sample == 79 * 1536
    assert (timing.sample_end - timing.sample_start) / 48000 == pytest.approx(len(audio) / 16000)
    assert pipeline._discarded_utterance_id == 1
    assert pipeline.diag_music_holds
    assert [
        call.args[0]["active"] for call in pipeline.broadcast.call_args_list if call.args[0]["type"] == "music_hold"
    ] == [True, False]
    coverage = pipeline._source_coverage.snapshot()
    assert coverage["duplicate_observed_samples"] == 0
    assert not coverage["capture_gaps"]
    assert any(
        row["state"] == "music_resume_recovered" and row["start"] == 40 * 1536 and row["end"] == 55 * 1536
        for row in coverage["outcomes"]
    )
    assert not coverage["complete"]  # a recovered buffer is not yet a final disposition
    pipeline._source_coverage.outcome(timing.sample_metadata(), "final_ready", uid)
    assert pipeline._source_coverage.snapshot()["complete"]


def test_music_recovery_discards_short_burst_before_recovering_new_onset(pipeline, monkeypatch):
    monkeypatch.setattr(pipeline, "MUSIC_HOLDOFF", 0.064)
    frames = [frame(i * 1536, False, True) for i in range(2)]
    frames += [frame(i * 1536) for i in range(2, 16)]  # 14 frames: unchanged rejection
    frames += [frame(i * 1536, False) for i in range(16, 18)]
    frames += [frame(i * 1536, i < 43) for i in range(18, 63)]
    finals = asyncio.run(capture(pipeline, monkeypatch, frames))
    assert len(finals) == 1
    audio, _, timing = finals[0]
    assert timing.sample_start == 18 * 1536
    np.testing.assert_array_equal(audio, np.concatenate([f.samples for f, _ in frames[18:57]]))
    outcomes = pipeline._source_coverage.snapshot()["outcomes"]
    assert any(
        row["state"] == "music_resume_suppressed_vad_non_speech" and (row["start"], row["end"]) == (2 * 1536, 16 * 1536)
        for row in outcomes
    )


@pytest.mark.parametrize("speech_frames", [1, 14])
def test_music_recovery_unaccepted_eof_tail_stays_suppressed(pipeline, monkeypatch, speech_frames):
    monkeypatch.setattr(pipeline, "MUSIC_HOLDOFF", 0.064)
    frames = [frame(i * 1536, False, True) for i in range(2)]
    frames += [frame(i * 1536) for i in range(2, 2 + speech_frames)]
    assert not asyncio.run(capture(pipeline, monkeypatch, frames))
    coverage = pipeline._source_coverage.snapshot()
    assert coverage["complete"]
    assert coverage["duplicate_observed_samples"] == 0
    assert any(
        row["state"] == "music_resume_suppressed_eof"
        and (row["start"], row["end"]) == (2 * 1536, (2 + speech_frames) * 1536)
        for row in coverage["outcomes"]
    )


@pytest.mark.parametrize("terminal_error,reason", [(asyncio.CancelledError, "stop"), (RuntimeError, "capture_closed")])
def test_music_recovery_resolves_tentative_spans_on_stop_or_failure(pipeline, monkeypatch, terminal_error, reason):
    class EndedQueue(asyncio.Queue):
        async def get(self):
            if self.empty():
                raise terminal_error()
            return await super().get()

    monkeypatch.setattr(pipeline, "MUSIC_HOLDOFF", 0.064)
    monkeypatch.setattr(pipeline, "_session_stop_requested", True)
    frames = [frame(i * 1536, False, True) for i in range(2)]
    frames += [frame(i * 1536) for i in range(2, 7)]
    with pytest.raises(terminal_error):
        asyncio.run(capture(pipeline, monkeypatch, frames, EndedQueue))
    coverage = pipeline._source_coverage.snapshot()
    assert not coverage["unclassified_intervals"]
    assert not coverage["complete"]  # stop/failure is not replay EOF
    assert any(row["state"] == "music_resume_suppressed_" + reason for row in coverage["outcomes"])


def test_music_recovery_pause_clears_tentative_spans(pipeline, monkeypatch):
    class PauseOnce:
        requested = False
        phase = Mock()
        discard_utterance = Mock()

        @property
        def paused(self):
            paused, self.requested = self.requested, False
            return paused

    health = PauseOnce()

    class PausingQueue(asyncio.Queue):
        async def get(self):
            value = await super().get()
            if self.empty():
                health.requested = True
            return value

    monkeypatch.setattr(pipeline, "MUSIC_HOLDOFF", 0.064)
    monkeypatch.setattr(pipeline, "_health", health)
    frames = [frame(i * 1536, False, True) for i in range(2)]
    frames += [frame(i * 1536) for i in range(2, 7)]
    assert not asyncio.run(capture(pipeline, monkeypatch, frames, PausingQueue))
    coverage = pipeline._source_coverage.snapshot()
    assert coverage["complete"]
    assert any(row["state"] == "music_resume_suppressed_pause" for row in coverage["outcomes"])


def test_music_hold_pause_logging_separates_consumed_frames_from_wall_time(pipeline, monkeypatch):
    clock = [100.0]

    class PauseOnce:
        requested = False
        phase = Mock()
        discard_utterance = Mock()

        @property
        def paused(self):
            paused, self.requested = self.requested, False
            if paused:
                clock[0] += 3600
            return paused

    health = PauseOnce()

    class PausingQueue(asyncio.Queue):
        paused_once = False

        async def get(self):
            value = await super().get()
            if self.empty() and not self.paused_once:
                health.requested = self.paused_once = True
            return value

    trace = Mock()
    monkeypatch.setattr(pipeline, "MUSIC_HOLDOFF", 0.064)
    monkeypatch.setattr(pipeline, "_health", health)
    monkeypatch.setattr(pipeline, "_latency_trace", trace)
    monkeypatch.setattr(pipeline.time, "perf_counter", lambda: clock[0])
    before = [frame(i * 1536, False, True) for i in range(2)]
    before += [frame(i * 1536) for i in range(2, 7)]
    after = [frame(i * 1536, i < 32) for i in range(7, 52)]
    finals = asyncio.run(capture(pipeline, monkeypatch, before, PausingQueue, after))
    assert len(finals) == 1 and finals[0][2].sample_start == 7 * 1536
    assert pipeline.diag_music_holds == [(2, 22, 0.6)]
    event = next(c.kwargs for c in trace.record.call_args_list if c.args == ("music_hold_finished",))
    assert event == {
        "captured_frame_duration_ms": 640.0,
        "captured_frame_time_excludes_operator_pause": True,
        "wall_elapsed_ms": 3600000.0,
        "wall_time_includes_operator_pause": True,
    }
    # This fixture supplies explicit contiguous source stamps across restart.
    # The actual isolated-reader offset mapping has a separate framed-input test.
    coverage = pipeline._source_coverage.snapshot()
    assert not coverage["capture_gaps"] and not coverage["duplicate_observed_samples"]


@pytest.mark.parametrize("speech_frames", [15, 20])
def test_stop_after_music_recovery_below_final_minimum_has_a_discard_disposition(pipeline, monkeypatch, speech_frames):
    class StopQueue(asyncio.Queue):
        async def get(self):
            if self.empty():
                raise asyncio.CancelledError()
            return await super().get()

    monkeypatch.setattr(pipeline, "MUSIC_HOLDOFF", 0.064)
    monkeypatch.setattr(pipeline, "_session_stop_requested", True)
    frames = [frame(i * 1536, False, True) for i in range(2)]
    frames += [frame(i * 1536) for i in range(2, 2 + speech_frames)]
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(capture(pipeline, monkeypatch, frames, StopQueue))
    coverage = pipeline._source_coverage.snapshot()
    assert not coverage["unclassified_intervals"]
    assert not coverage["capture_gaps"] and not coverage["duplicate_observed_samples"]
    assert any(
        row["state"] == "discarded_stop" and (row["start"], row["end"]) == (2 * 1536, (2 + speech_frames) * 1536)
        for row in coverage["outcomes"]
    )


def test_music_recovery_never_joins_speech_across_source_gap(pipeline, monkeypatch):
    monkeypatch.setattr(pipeline, "MUSIC_HOLDOFF", 0.064)
    frames = [frame(i * 1536, False, True) for i in range(2)]
    frames += [frame(i * 1536) for i in range(2, 12)]
    frames += [frame(i * 1536, i < 47) for i in range(22, 67)]
    finals = asyncio.run(capture(pipeline, monkeypatch, frames))
    assert len(finals) == 1
    assert finals[0][2].sample_start == 22 * 1536
    coverage = pipeline._source_coverage.snapshot()
    assert coverage["capture_gaps"] == [[12 * 1536, 22 * 1536]]
    assert not coverage["complete"]
    assert any(row["state"] == "music_resume_suppressed_source_discontinuity" for row in coverage["outcomes"])


def test_capture_reentry_does_not_reuse_discarded_utterance_ids(pipeline, monkeypatch):
    async def run():
        first = [frame(i * 1536, i == 0) for i in range(22)]
        assert not await capture(pipeline, monkeypatch, first)
        assert pipeline._discarded_utterance_id == 1
        second = [frame((100 + i) * 1536, i < 25) for i in range(45)]
        finals = await capture(pipeline, monkeypatch, second)
        assert [uid for _, uid, _ in finals] == [2]
        assert not pipeline._preview_was_discarded(2)
        assert pipeline._preview_was_discarded(1)

    asyncio.run(run())


def test_actual_loop_sends_scoped_discard_event_to_socket(pipeline, monkeypatch):
    client = Mock(send=AsyncMock())
    monkeypatch.setattr(pipeline, "broadcast", pipeline._test_broadcast_impl)
    monkeypatch.setattr(pipeline, "ws_clients", {client})
    monkeypatch.setattr(pipeline, "SESSION_ID", "discard-session")
    asyncio.run(capture(pipeline, monkeypatch, [frame(i * 1536, i == 0) for i in range(22)]))
    messages = [json.loads(call.args[0]) for call in client.send.await_args_list]
    assert len(messages) == 1
    assert messages[0]["type"] == "utterance_discarded"
    assert messages[0]["session_id"] == "discard-session"
    assert messages[0]["utterance_id"] == 1
    assert messages[0]["reason"] == "short_silence"
    assert "chunk_id" not in messages[0]


def test_capture_error_discards_identity_and_short_buffer_before_retry(pipeline, monkeypatch):
    from tools import audio_bridge_client
    from tools.isolated_audio import AudioCaptureError

    d = pipeline
    monkeypatch.setattr(d.sd, "PortAudioError", RuntimeError)
    monkeypatch.setattr(d, "_io_pool", Mock())
    monkeypatch.setattr(d, "process_partial", AsyncMock())
    monkeypatch.setattr(d, "EXIT_AFTER_REPLAY", True)
    monkeypatch.setattr(d, "is_speech", lambda *args: True)
    final = AsyncMock()
    monkeypatch.setattr(d, "process_final", final)
    calls = []

    async def run():
        queue = asyncio.Queue()
        monkeypatch.setattr(d, "audio_queue", queue)
        monkeypatch.setattr(d, "_pipeline_chunk_queue", asyncio.Queue())
        # Replace only the retry sleep, not queue timeouts or task scheduling.
        original_sleep = asyncio.sleep

        async def sleep(delay):
            assert delay == 2
            assert d._preview_was_discarded(1)
            assert 1 not in d._utterance_start_times
            await original_sleep(0)

        monkeypatch.setattr(d.asyncio, "sleep", sleep)

        def open_stream(**kwargs):
            stream = MagicMock(spec=["__enter__", "__exit__"])
            stream.finished = threading.Event()
            stream.finished.set()
            stream.error = AudioCaptureError("device lost") if not calls else None
            count, offset = (3, 0) if not calls else (25, 10000)
            for index in range(count):
                queue.put_nowait(frame(offset + index * 1536)[0])
            calls.append(stream)
            return stream

        monkeypatch.setattr(audio_bridge_client, "open_audio_stream", open_stream)
        await asyncio.wait_for(d.audio_loop(), 1)
        assert len(calls) == 2
        assert final.await_args.args[1] == 2
        assert d._utterance_timings[2].sample_start == 10000
        d._io_pool.record_failure.assert_called_once_with("audio_capture", "interrupted_utterance")
        assert any(call.kwargs.get("reason") == "capture_error" for call in d._latency_event.call_args_list)

    asyncio.run(run())


@pytest.mark.parametrize("operation", ["pause", "eof", "stop"])
def test_subminimum_capture_end_invalidates_inflight_preview_without_final(pipeline, monkeypatch, tmp_path, operation):
    from tools import audio_bridge_client
    from tools.pipeline_health import PipelineHealth

    d = pipeline
    monkeypatch.setattr(d, "MULTIPROCESS", True)
    monkeypatch.setattr(d, "_run_partial_stt_via_worker", lambda audio: ("Only an interim.", 10))
    monkeypatch.setattr(d, "_is_garbage_text", lambda text: False)
    monkeypatch.setattr(d, "_should_suppress", lambda *args, **kwargs: None)
    monkeypatch.setattr(d, "_io_pool", Mock())
    monkeypatch.setattr(d, "process_final", AsyncMock())
    monkeypatch.setattr(d, "_session_stop_requested", operation == "stop")
    monkeypatch.setattr(d, "EXIT_AFTER_REPLAY", operation == "eof")
    monkeypatch.setattr(d, "is_speech", lambda *args: True)
    health = PipelineHealth(tmp_path, d.SESSION_ID)
    monkeypatch.setattr(d, "_health", health)
    stream = MagicMock(spec=["__enter__", "__exit__"])
    stream.finished = threading.Event()
    stream.error = None
    monkeypatch.setattr(audio_bridge_client, "open_audio_stream", lambda **kwargs: stream)
    release = threading.Event()

    async def run():
        queue = asyncio.Queue()
        # The default 0.6s preview can fire, but the buffer is below 0.7s.
        for index in range(20):
            queue.put_nowait(frame(index * 1536)[0])
        monkeypatch.setattr(d, "audio_queue", queue)
        monkeypatch.setattr(d, "_pipeline_chunk_queue", asyncio.Queue())
        loop = asyncio.get_running_loop()
        started = asyncio.Event()

        def translate(text):
            loop.call_soon_threadsafe(started.set)
            assert release.wait(2)
            return "Solo un avance.", 10

        monkeypatch.setattr(d, "translate_marian", translate)
        task = asyncio.create_task(d.audio_loop())
        try:
            await asyncio.wait_for(started.wait(), 1)
            if operation == "pause":
                health.paused = True
            elif operation == "eof":
                stream.finished.set()
            else:
                task.cancel()
            for _ in range(100):
                if d._preview_was_discarded(1):
                    break
                await asyncio.sleep(0.005)
            assert d._preview_was_discarded(1)
            release.set()
            await asyncio.gather(*list(d._partial_tasks))
            d.process_final.assert_not_awaited()
            assert [call.args[0]["type"] for call in d.broadcast.await_args_list] == ["utterance_discarded"]
            assert d.broadcast.await_args.args[0]["reason"] == operation
            assert not d.partial_translations
            if operation == "eof":
                await asyncio.wait_for(task, 1)
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        monkeypatch.setattr(d, "_stt_comm_pool", pool)
        monkeypatch.setattr(d, "_pytorch_pool", pool)
        asyncio.run(run())


@pytest.mark.parametrize("blocked_stage", ["stt", "marian"])
def test_discarded_partial_cannot_recreate_state_or_publish_after_worker_returns(pipeline, monkeypatch, blocked_stage):
    d = pipeline
    release = threading.Event()
    monkeypatch.setattr(d, "MULTIPROCESS", True)
    monkeypatch.setattr(d, "_pipeline_chunk_queue", None)
    monkeypatch.setattr(d, "_is_garbage_text", lambda text: False)
    monkeypatch.setattr(d, "_should_suppress", lambda *a, **k: None)
    monkeypatch.setattr(d, "_io_pool", Mock())
    assert not d._preview_ordering_enabled()  # default runtime also needs discard safety

    async def run():
        started = asyncio.Event()
        loop = asyncio.get_running_loop()

        def block(stage):
            if stage == blocked_stage:
                loop.call_soon_threadsafe(started.set)
                assert release.wait(2)

        def stt(audio):
            block("stt")
            return "Discard this noise.", 10

        def translate(text):
            block("marian")
            return "Descarta este ruido.", 10

        monkeypatch.setattr(d, "_run_partial_stt_via_worker", stt)
        monkeypatch.setattr(d, "translate_marian", translate)
        task = asyncio.create_task(d.process_partial(np.ones(16000), 7))
        try:
            await asyncio.wait_for(started.wait(), 1)
            d._discard_utterance(7, "music_hold", 1, {})
            release.set()
            await task
            assert not d._partial_source_text
            assert not d._rolling_previews
            assert not d.partial_translations
            d.broadcast.assert_not_awaited()
            d._io_pool.submit.assert_not_called()
            # Even a previously scheduled coroutine starting after discard
            # must not submit STT work or publish a caption.
            monkeypatch.setattr(d, "_run_partial_stt_via_worker", Mock(side_effect=AssertionError("stale STT")))
            await d.process_partial(np.ones(16000), 7)
            d._run_partial_stt_via_worker.assert_not_called()
        finally:
            release.set()
            await task

    with ThreadPoolExecutor(max_workers=2) as pool:
        monkeypatch.setattr(d, "_stt_comm_pool", pool)
        monkeypatch.setattr(d, "_pytorch_pool", pool)
        asyncio.run(run())


def test_discard_clears_exact_speculation_and_queued_work_without_closing_worker_owned_stream(pipeline, monkeypatch):
    d = pipeline
    monkeypatch.setattr(d, "_stt_scheduler", Mock())
    monkeypatch.setattr(d, "_incremental_stt", Mock())
    d._speculative_candidates[7] = object()
    d._speculation_attempts[7] = 1
    d._pause_epochs[7] = 0
    d._partial_source_text[7] = "old"
    d._rolling_previews[7] = object()
    d._discard_utterance(7, "short_silence", 0.5, {})
    assert d._confirmed_speculation(7, "old", None) is None
    assert not d._speculative_candidates and not d._rolling_previews
    assert not d._pause_epochs and not d._partial_source_text
    d._stt_scheduler.cancel_partial.assert_called_once_with(7)
    d._incremental_stt.close.assert_not_called()
    d._discard_utterance(4, "music_hold", 0, {})
    assert d._preview_was_discarded(7) and not d._preview_was_discarded(8)
