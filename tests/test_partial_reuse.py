"""Exercise emitted-preview reuse through the real coordinator with stubbed inference."""

import asyncio
import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from tools.latency_experiments import LatencyExperiments
from tools.latency_trace import LatencyTrace
from tools.pipeline_timing import ChunkTiming


@pytest.fixture
def pipeline(monkeypatch):
    import dry_run_ab as module

    monkeypatch.setattr(module, "_latency", LatencyExperiments(partial_reuse_ms=300))
    monkeypatch.setattr(module, "_latency_trace", LatencyTrace(True))
    for name in (
        "_partial_reuse_candidates",
        "_experiment_counters",
        "_chunk_timings",
        "_partial_emitted_sequence",
        "_partial_source_text",
        "partial_translations",
        "partial_latencies",
    ):
        monkeypatch.setattr(module, name, {})
    for name in ("_published_final_utterances", "_closed_utterances"):
        monkeypatch.setattr(module, name, set())
    for name in (
        "chunk_id",
        "_partial_sequence",
        "_discarded_utterance_id",
        "_published_final_floor",
        "_pipeline_total",
        "_pipeline_overlaps",
    ):
        monkeypatch.setattr(module, name, 0)
    for name in (
        "_stt_scheduler",
        "_active_partial_future",
        "_pipeline_chunk_queue",
        "_health",
        "_published_final_session",
        "_final_pending_utterance_id",
    ):
        monkeypatch.setattr(module, name, None)
    monkeypatch.setattr(module, "_active_stt_workers", {"partial": 0, "final": 0})
    monkeypatch.setattr(module, "_final_pending", threading.Event())
    monkeypatch.setattr(module, "_translation_active", threading.Event())
    monkeypatch.setattr(module, "_last_final_text", "")
    monkeypatch.setattr(module, "prev_text", "")
    monkeypatch.setattr(module, "MULTIPROCESS", False)
    monkeypatch.setattr(module.settings.translation, "final_aware_partials", False)
    monkeypatch.setattr(module, "broadcast", AsyncMock())
    monkeypatch.setattr(module, "_io_pool", Mock())
    monkeypatch.setattr(module, "_is_garbage_text", Mock(return_value=False))
    monkeypatch.setattr(module, "_should_suppress", Mock(return_value=None))
    monkeypatch.setattr(module, "correct_stt_output", Mock(side_effect=lambda text: (text, [])))
    return module


@pytest.mark.parametrize(
    "enabled,endpoint,gap,rate,has_candidate,reuse",
    [
        (300, "silence", 4799, 16000, True, True),
        (300, "silence", 4800, 16000, True, True),
        (300, "silence", 4801, 16000, True, False),
        (300, "silence", 14400, 48000, True, True),
        (300, "silence", 14401, 48000, True, False),
        (1, "silence", 44, 44100, True, True),
        (1, "silence", 45, 44100, True, False),
        (300, "silence", -100, 48000, True, True),
        (300, "smart_cut", 0, 48000, True, False),
        (300, "max_utterance", 0, 48000, True, False),
        (300, "hard", 0, 48000, True, False),
        (300, "silence", 0, 48000, False, False),
        (0, "silence", 0, 48000, True, False),
        (300, "silence", None, 48000, True, False),
        (300, "silence", 0, None, True, False),
    ],
)
@pytest.mark.parametrize("scheduled", [False, True])
def test_final_worker_reuse_or_full_stt(
    pipeline, monkeypatch, enabled, endpoint, gap, rate, has_candidate, reuse, scheduled
):
    monkeypatch.setattr(pipeline, "_latency", LatencyExperiments(partial_reuse_ms=enabled))
    candidate = {"sample_end": 48000, "speech_end_sample": 47500, "request_sequence": 23, "english": "Good morning"}
    if has_candidate:
        pipeline._partial_reuse_candidates[7] = candidate
    timing = ChunkTiming(
        utterance_id=7,
        endpoint_reason=endpoint,
        sample_rate=rate,
        speech_end_sample=48000 + gap if gap is not None else None,
    )
    stt = Mock(return_value=("The full transcription", 440, 0.9, [{"confidence": 0.9}], [{"word": "full"}]))
    monkeypatch.setattr(pipeline, "_run_stt", stt)
    finalized = AsyncMock()
    monkeypatch.setattr(pipeline, "_pipeline_translate_and_finalize", finalized)
    scheduler = Mock()

    def submit(kind, function, **kwargs):
        result = Future()
        result.set_result(function())
        return result

    scheduler.submit.side_effect = submit
    if scheduled:
        monkeypatch.setattr(pipeline, "_stt_scheduler", scheduler)
    pipeline._final_pending.set()
    monkeypatch.setattr(pipeline, "_final_pending_utterance_id", 7)

    async def run():
        queue = asyncio.Queue()
        queue.put_nowait((np.full(16000, 0.1), 1, 1, timing))
        queue.put_nowait(None)
        monkeypatch.setattr(pipeline, "_pipeline_chunk_queue", queue)
        await pipeline._pipeline_coordinator()

    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(pipeline, "_pipeline_pool", pool)
        asyncio.run(run())
    finalized.assert_awaited_once()
    args = finalized.call_args.args
    expected_text = candidate["english"] if reuse else "The full transcription"
    assert args[1] == expected_text
    pipeline._is_garbage_text.assert_called_once_with(expected_text)
    pipeline.correct_stt_output.assert_called_once_with(expected_text)
    pipeline._should_suppress.assert_called_once()
    assert not pipeline._final_pending.is_set()
    assert pipeline._final_pending_utterance_id is None
    assert finalized.call_args.kwargs["timing"].final_stt_route == ("partial_reuse" if reuse else "full")
    assert stt.call_count == int(not reuse)
    assert scheduler.submit.call_count == int(scheduled and not reuse)
    if reuse:
        assert args[2:6] == (0.0, None, [], [])
        assert timing.stt_requested == timing.stt_started == timing.stt_finished
        assert not pipeline.should_use_marian_only(args[1], args[3])
    else:
        assert args[2:6] == stt.return_value[1:]
    counts = pipeline._experiment_snapshot()
    if enabled:
        assert counts["final_stt_reused_partial"] == int(reuse)
        assert counts["final_stt_full"] == int(not reuse)
        assert 7 not in pipeline._partial_reuse_candidates
    else:
        assert "final_stt_reused_partial" not in counts
        assert "final_stt_full" not in counts
        assert pipeline._partial_reuse_candidates[7] is candidate
    events = [e for e in pipeline._latency_trace.snapshot()["events"] if e["event"] == "final_stt_reused_partial"]
    assert len(events) == int(reuse)
    if reuse:
        assert {key: events[0][key] for key in ("chunk_id", "utterance_id", "gap_samples", "request_sequence")} == {
            "chunk_id": 1,
            "utterance_id": 7,
            "gap_samples": gap,
            "request_sequence": 23,
        }


@pytest.mark.parametrize("enabled", [0, 300])
def test_only_emitted_partials_record_exact_source_and_bound_candidates(pipeline, monkeypatch, enabled):
    monkeypatch.setattr(pipeline, "_latency", LatencyExperiments(partial_reuse_ms=enabled))
    monkeypatch.setattr(pipeline, "MULTIPROCESS", True)
    source = "God loves every person"
    monkeypatch.setattr(pipeline, "_run_partial_stt_via_worker", Mock(return_value=(source, 100)))
    targets = iter(["Dios ama a cada persona", "", "Dios ama a cada persona"])
    monkeypatch.setattr(pipeline, "translate_marian", lambda text: (next(targets), 40))
    # Prime a full cache to check bounded insertion without 257 STT requests.
    initial = {uid: {"english": "old"} for uid in range(100, 356)}
    monkeypatch.setattr(pipeline, "_partial_reuse_candidates", dict(initial))

    async def preview(end):
        await pipeline.process_partial(
            np.full(16000, 0.1),
            7,
            sample_bounds={"sample_end": end, "speech_end_sample": end - 100, "sample_rate": 48000},
        )

    async def run():
        await preview(48000)
        if enabled:
            assert pipeline._partial_reuse_candidates[7] == {
                "sample_end": 48000,
                "speech_end_sample": 47900,
                "request_sequence": 1,
                "english": source,
                "stt_confidence": None,  # the worker path returns no confidence
            }
            assert 100 not in pipeline._partial_reuse_candidates
        before = dict(pipeline._partial_reuse_candidates)
        await preview(49000)  # Empty translation must not replace the emitted candidate.
        assert pipeline._partial_reuse_candidates == before
        await preview(50000)

    with ThreadPoolExecutor(max_workers=2) as pool:
        monkeypatch.setattr(pipeline, "_stt_comm_pool", pool)
        monkeypatch.setattr(pipeline, "_pytorch_pool", pool)
        asyncio.run(run())
    assert pipeline.broadcast.await_count == 2
    assert len(pipeline._partial_reuse_candidates) == 256
    if enabled:
        candidate = pipeline._partial_reuse_candidates[7]
        assert candidate["sample_end"] == 50000
        assert candidate["request_sequence"] == 3
        assert candidate["english"] == pipeline.broadcast.call_args.args[0]["english"] == source
    else:
        assert pipeline._partial_reuse_candidates == initial


def test_final_route_survives_finalizer_and_diagnostics(pipeline, monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "MULTIPROCESS", True)
    monkeypatch.setattr(pipeline, "_RUN_AB", False)
    monkeypatch.setattr(pipeline, "_translate_via_worker", Mock(return_value=("Buenos días", 100, 10, None, 0, 0)))
    monkeypatch.setattr(pipeline, "translate_marian", Mock(side_effect=AssertionError("must route to Gemma")))
    monkeypatch.setattr(pipeline, "_confirmed_speculation", lambda *args: None)
    monkeypatch.setattr(pipeline, "qe_score", lambda *args: None)
    monkeypatch.setattr(pipeline, "get_resource_snapshot", lambda: {})
    monkeypatch.setattr(pipeline, "DIARIZE_ENABLED", False)
    monkeypatch.setattr(pipeline, "tts_engine", None)
    monkeypatch.setattr(pipeline, "all_results", [])
    monkeypatch.setattr(pipeline, "_last_gen_stats", {})
    for name in ("check_homophones", "check_bad_split", "check_near_miss"):
        monkeypatch.setattr(pipeline, name, Mock())
    timing = ChunkTiming(utterance_id=7)
    timing.final_stt_route = "partial_reuse"

    async def run():
        monkeypatch.setattr(pipeline, "_pipeline_translation_lock", asyncio.Lock())
        await pipeline._pipeline_translate_and_finalize(
            1, "Good morning", 0.0, None, [], [], np.ones(16000), 1, timing=timing
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(pipeline, "_trans_comm_pool", pool)
        asyncio.run(run())
    result = pipeline.all_results[0]
    assert result["final_stt_route"] == "partial_reuse"
    assert result["final_translation_route"] == "gemma"
    assert result["stt_confidence"] is None
    path = tmp_path / "diagnostics.jsonl"
    monkeypatch.setattr(pipeline, "DIAG_PATH", str(path))
    pipeline.write_diag_jsonl(result, None, segment_meta=[], low_conf_words=[])
    record = json.loads(path.read_text())
    assert record["final_stt_route"] == "partial_reuse"
    assert record["final_translation_route"] == "gemma"
    assert record["stt_confidence"] is None


def test_final_stt_route_is_not_a_stage_stamp():
    timing = ChunkTiming(utterance_id=3, endpoint_reason="silence", stt_started=10.0, final_stt_route="partial_reuse")
    stages = timing.relative_stages(9.0)
    assert stages == {"stt_started": 1000.0}
    assert ChunkTiming(utterance_id=3).final_stt_route is None


@pytest.mark.parametrize("keep", [False, True])
def test_reused_final_keeps_partial_confidence_only_when_enabled(pipeline, monkeypatch, keep):
    monkeypatch.setattr(
        pipeline, "_latency", LatencyExperiments(partial_reuse_ms=300, partial_reuse_keep_confidence=keep)
    )
    pipeline._partial_reuse_candidates[7] = {
        "sample_end": 48000,
        "speech_end_sample": 47500,
        "request_sequence": 23,
        "english": "Good morning",
        "stt_confidence": 0.93,
    }
    timing = ChunkTiming(utterance_id=7, endpoint_reason="silence", sample_rate=48000, speech_end_sample=48000)
    monkeypatch.setattr(pipeline, "_run_stt", Mock(side_effect=AssertionError("full STT must not run")))
    finalized = AsyncMock()
    monkeypatch.setattr(pipeline, "_pipeline_translate_and_finalize", finalized)
    pipeline._final_pending.set()
    monkeypatch.setattr(pipeline, "_final_pending_utterance_id", 7)

    async def run():
        queue = asyncio.Queue()
        queue.put_nowait((np.full(16000, 0.1), 1, 1, timing))
        queue.put_nowait(None)
        monkeypatch.setattr(pipeline, "_pipeline_chunk_queue", queue)
        await pipeline._pipeline_coordinator()

    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(pipeline, "_pipeline_pool", pool)
        asyncio.run(run())
    args = finalized.call_args.args
    assert args[1] == "Good morning"
    assert args[3] == (0.93 if keep else None)
    assert finalized.call_args.kwargs["timing"].final_stt_route == "partial_reuse"
