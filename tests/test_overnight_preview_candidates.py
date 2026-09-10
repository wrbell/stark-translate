"""Earlier work cannot bypass exact final confirmation or streaming cleanup."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from tools.incremental_stt import StreamingPreview
from tools.latency_experiments import LatencyExperiments
from tools.latency_scheduler import LatestSTTWorker
from tools.preview_candidates import RollingPreview, TranslationCandidate, common_prefix_words


@pytest.fixture(autouse=True)
def fresh_session_discard_state(monkeypatch):
    # These tests start fresh utterance IDs, independent of prior audio-loop
    # tests in the same interpreter. The production guard remains active.
    import dry_run_ab as pipeline

    monkeypatch.setattr(pipeline, "_discarded_utterance_id", 0)


def test_candidate_confirmation_does_not_normalize_negation_case_or_request_identity():
    identity = ("en", "es", "model", "He is not guilty.")
    candidate = TranslationCandidate(identity, object(), 2, {})
    assert candidate.confirmed(identity, 2)
    for changed in (
        ("en", "es", "model", "He is guilty."),
        ("en", "es", "model", "he is not guilty."),
        ("es", "en", "model", "He is not guilty."),
        ("en", "es", "other-model", "He is not guilty."),
    ):
        assert not candidate.confirmed(changed, 2)
    assert not candidate.confirmed(identity, 3)


def test_rolling_preview_requires_multiword_overlap_and_can_revise_draft_tail():
    preview = RollingPreview()
    assert preview.update("We know that he is guilty", 0) == ("We know that he is guilty", True)
    assert preview.update("that he is not guilty", 16000) == ("We know that he is not guilty", True)
    assert preview.update("a different uncertain clause", 32000) == ("a different uncertain clause", False)
    assert common_prefix_words("He is guilty", "He is not guilty") == "He is"


class FakeStream:
    def __init__(self):
        self.feeds = []
        self.closed = False
        self.text = "A preview"
        self.tokens = [SimpleNamespace(confidence=0.95)]

    def add_audio(self, audio):
        self.feeds.append(list(audio))

    def close(self):
        self.closed = True


def test_stream_preview_feeds_only_new_samples_and_closes_when_utterance_changes():
    streams = []

    class Engine:
        def open_stream(self):
            stream = FakeStream()
            streams.append(stream)
            return stream

    preview = StreamingPreview(Engine())
    _, processed = preview.transcribe(np.array([1, 2, 3]), 1)
    assert processed == 3
    _, processed = preview.transcribe(np.array([1, 2, 3, 4, 5]), 1)
    assert processed == 2 and streams[0].feeds == [[1, 2, 3], [4, 5]]
    preview.transcribe(np.array([6, 7]), 2)
    assert streams[0].closed
    preview.close()
    assert streams[1].closed


def test_final_batch_restores_model_mode_before_calling_recognizer(monkeypatch):
    import dry_run_ab as pipeline

    calls = []
    monkeypatch.setattr(pipeline, "_incremental_stt", SimpleNamespace(close=lambda: calls.append("closed")))
    monkeypatch.setattr(pipeline, "BACKEND", "mlx")

    def batch(*args):
        assert calls == ["closed"]
        return "final"

    monkeypatch.setattr(pipeline, "_run_stt_mlx", batch)
    assert pipeline._run_stt(np.ones(10), "prompt") == "final"


def test_unidentified_final_does_not_poison_later_bounded_history_cleanup(monkeypatch):
    import dry_run_ab as pipeline

    monkeypatch.setattr(pipeline, "_stt_scheduler", Mock())
    monkeypatch.setattr(pipeline, "_closed_utterances", set())
    monkeypatch.setattr(pipeline, "pipeline_submit", AsyncMock())

    async def run():
        await pipeline.process_final(np.ones(10))
        assert None not in pipeline._closed_utterances
        await pipeline.process_final(np.ones(10), 200)
        assert pipeline._closed_utterances == {200}

    try:
        asyncio.run(run())
    finally:
        pipeline._final_pending.clear()


@pytest.mark.parametrize(
    "experiment",
    [
        {"first_preview_s": 0.32},
        {"speculate_pause_ms": 128},
        {"pause_preview_ms": 128},
        {"clause_preview_s": 4},
        {"incremental_stt": "rolling"},
        {"incremental_stt": "stream"},
        {"latest_partial": True},
    ],
)
def test_preview_finishing_marian_after_final_never_repaints_closed_utterance(monkeypatch, experiment):
    import dry_run_ab as pipeline

    monkeypatch.setattr(pipeline, "_latency", replace(LatencyExperiments(), **experiment))
    monkeypatch.setattr(pipeline, "_stt_scheduler", None)  # policy must not depend on startup side effects
    monkeypatch.setattr(pipeline, "_closed_utterances", set())
    monkeypatch.setattr(pipeline, "_final_pending", threading.Event())
    monkeypatch.setattr(pipeline, "_partial_emitted_sequence", {})
    monkeypatch.setattr(pipeline, "_partial_source_text", {})
    monkeypatch.setattr(pipeline, "_rolling_previews", {})
    monkeypatch.setattr(pipeline, "MULTIPROCESS", True)
    monkeypatch.setattr(pipeline, "_pipeline_chunk_queue", None)
    monkeypatch.setattr(pipeline, "_active_partial_future", None)
    monkeypatch.setattr(pipeline.settings.translation, "final_aware_partials", False)
    monkeypatch.setattr(pipeline, "_run_partial_stt_via_worker", lambda audio: ("He is not guilty.", 10))
    monkeypatch.setattr(pipeline, "_is_garbage_text", lambda text: False)
    monkeypatch.setattr(pipeline, "_should_suppress", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "pipeline_submit", AsyncMock())
    monkeypatch.setattr(pipeline, "broadcast", AsyncMock())
    monkeypatch.setattr(pipeline, "_io_pool", Mock())
    release = threading.Event()

    async def run():
        started = asyncio.Event()
        loop = asyncio.get_running_loop()

        def marian(text):
            loop.call_soon_threadsafe(started.set)
            assert release.wait(2)
            return "Él no es culpable.", 10

        monkeypatch.setattr(pipeline, "translate_marian", marian)
        task = asyncio.create_task(pipeline.process_partial(np.ones(16000), 7))
        try:
            await asyncio.wait_for(started.wait(), 1)
            await pipeline.process_final(np.ones(16000), 7)
            pipeline._final_pending.clear()  # the final STT/translation has already completed
            release.set()
            await task
            assert 7 in pipeline._closed_utterances
            pipeline.broadcast.assert_not_awaited()
            pipeline._io_pool.submit.assert_not_called()
        finally:
            release.set()
            await task

    with ThreadPoolExecutor(max_workers=2) as pool:
        monkeypatch.setattr(pipeline, "_stt_comm_pool", pool)
        monkeypatch.setattr(pipeline, "_pytorch_pool", pool)
        asyncio.run(run())


@pytest.mark.parametrize("resume", [False, True])
def test_real_pause_speculation_is_private_and_requires_matching_final_source(monkeypatch, resume):
    import dry_run_ab as pipeline
    import engines.mlx_engine as mlx_engine
    from engines.base import TranslationResult

    worker = LatestSTTWorker()
    pool = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(pipeline, "_latency", replace(LatencyExperiments(), speculate_pause_ms=128))
    monkeypatch.setattr(pipeline, "_stt_scheduler", worker)
    monkeypatch.setattr(pipeline, "_pipeline_pool", pool)
    monkeypatch.setattr(pipeline, "BACKEND", "mlx")
    monkeypatch.setattr(pipeline, "MULTIPROCESS", False)
    monkeypatch.setattr(pipeline, "MODEL_FAMILY", "gemma4")
    monkeypatch.setattr(pipeline, "SOURCE_LANG", "en")
    monkeypatch.setattr(pipeline, "TARGET_LANG", "es")
    monkeypatch.setattr(pipeline, "mlx_a_model", object())
    monkeypatch.setattr(pipeline, "mlx_a_tokenizer", object())
    monkeypatch.setattr(pipeline, "mlx_b_model", None)
    monkeypatch.setattr(pipeline, "MLX_DRAFT_MODEL", None)
    monkeypatch.setattr(pipeline, "_partial_source_text", {1: "He is not guilty."})
    monkeypatch.setattr(pipeline, "_pause_epochs", {1: 2})
    monkeypatch.setattr(pipeline, "_closed_utterances", set())
    monkeypatch.setattr(pipeline, "_speculative_candidates", {})
    monkeypatch.setattr(pipeline, "_speculation_attempts", {})
    monkeypatch.setattr(pipeline, "_should_suppress", lambda *a, **k: False)
    monkeypatch.setattr(pipeline, "should_use_marian_only", lambda *a: False)
    monkeypatch.setattr(pipeline, "_run_stt", lambda *a: ("He is not guilty.", 12, 0.95, [], []))
    pipeline._final_pending.clear()
    pipeline._translation_active.clear()

    def generate(*args, **kwargs):
        assert "token_callback" not in kwargs  # no speculative final reaches displays
        if resume:
            pipeline._pause_epochs[1] += 1
        return TranslationResult("Él no es culpable.", 42)

    monkeypatch.setattr(mlx_engine, "translate_loaded_model", generate)

    async def run():
        monkeypatch.setattr(pipeline, "_pipeline_translation_lock", asyncio.Lock())
        await pipeline._speculate_pause(np.ones(16000), 1, 2, {"sample_start": 0, "sample_end": 16000})
        if resume:
            assert not pipeline._speculative_candidates
        else:
            candidate = pipeline._speculative_candidates[1]
            assert pipeline._confirmed_speculation(1, "He is guilty.", 0.95) is None
            pipeline._speculative_candidates[1] = candidate
            assert pipeline._confirmed_speculation(1, "He is not guilty.", 0.95).text == "Él no es culpable."

    try:
        asyncio.run(run())
    finally:
        worker.shutdown()
        pool.shutdown()
