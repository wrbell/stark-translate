"""Opt-in experiments preserve production defaults and safe routing boundaries."""

from concurrent.futures import Future
from unittest.mock import Mock

import pytest

from engines.translation_prompts import build_chat_messages, conservative_marian_route
from settings import TranslationSettings, VADSettings


def test_experiments_are_opt_in():
    translation = TranslationSettings(_env_file=None)
    assert translation.idle_warmup_only is False
    assert translation.final_aware_partials is False
    assert translation.terminology_prompt == "none"
    assert translation.routing_policy == "legacy"
    assert VADSettings(_env_file=None).backend == "torch"


@pytest.mark.parametrize("lang,text", [("en", "Thank you!"), ("es", "¡Buenos días!")])
def test_operational_routing_is_bilingual_and_confidence_gated(lang, text):
    assert conservative_marian_route(text, lang, 0.9)
    assert not conservative_marian_route(text, lang, None)
    assert not conservative_marian_route(text, lang, 0.79)
    assert not conservative_marian_route(text, lang, float("nan"))


@pytest.mark.parametrize(
    "lang,text",
    [
        ("en", "The covenant of grace"),
        ("es", "El pacto de la gracia"),
        ("en", "Thank you Jesus"),
        ("es", "Gracias a Dios"),
    ],
)
def test_routing_never_infers_safety_from_high_confidence_alone(lang, text):
    assert not conservative_marian_route(text, lang, 0.999)


def test_terminology_policy_does_not_rewrite_default_or_spanish_input_prompts():
    baseline = build_chat_messages("James", model_family="gemma4")
    church = build_chat_messages("James", model_family="gemma4", terminology_prompt="church")
    assert "Jacobo" not in baseline[0]["content"]
    assert "Jacobo" in church[0]["content"] and "partimiento" in church[0]["content"]
    es = build_chat_messages("Jacobo", model_family="gemma4", source_lang="es", target_lang="en")
    assert es == build_chat_messages(
        "Jacobo", model_family="gemma4", source_lang="es", target_lang="en", terminology_prompt="church"
    )


def test_idle_warmup_is_coalesced_and_skips_final_work(monkeypatch):
    import dry_run_ab as pipeline

    monkeypatch.setattr(pipeline.settings.translation, "idle_warmup_only", True)
    monkeypatch.setattr(pipeline, "_inference_idle", lambda: False)
    loop = Mock()
    pipeline._schedule_warmup(loop)
    loop.run_in_executor.assert_not_called()
    monkeypatch.setattr(pipeline, "_inference_idle", lambda: True)
    future = Future()
    monkeypatch.setattr(pipeline, "_warmup_future", future)
    pipeline._schedule_warmup(loop)
    loop.run_in_executor.assert_not_called()
    future.set_result(None)
    pipeline._schedule_warmup(loop)
    loop.run_in_executor.assert_called_once()


@pytest.mark.parametrize("same_model", [True, False])
def test_generation_guard_serializes_only_shared_models(same_model):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from engines.mlx_generation_lock import generation_guard

    class Model:
        pass

    first_model = Model()
    second_model = first_model if same_model else Model()
    first_entered, second_entered, release = threading.Event(), threading.Event(), threading.Event()

    def first():
        with generation_guard(first_model):
            first_entered.set()
            assert release.wait(2)

    def second():
        with generation_guard(second_model):
            second_entered.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(first)
        assert first_entered.wait(1)
        two = pool.submit(second)
        if same_model:
            assert not second_entered.wait(0.03)
        else:
            assert second_entered.wait(1)
        release.set()
        one.result(timeout=1)
        two.result(timeout=1)
    assert second_entered.is_set()


@pytest.mark.parametrize("cache_env,expected", [("512", 512), (None, 256), ("unset_latency", 256)])
def test_stt_and_worker_cache_limits_share_experiment(monkeypatch, cache_env, expected):
    import dry_run_ab as pipeline
    import engines.parakeet_mlx_engine as parakeet
    from tools.latency_experiments import LatencyExperiments

    monkeypatch.delenv("STARK_EXPERIMENT_MLX_CACHE_MB", raising=False)
    if cache_env == "512":
        monkeypatch.setenv("STARK_EXPERIMENT_MLX_CACHE_MB", cache_env)
    latency = None if cache_env == "unset_latency" else LatencyExperiments.from_env()
    monkeypatch.setattr(pipeline, "_latency", latency)
    constructor = Mock()
    monkeypatch.setattr(parakeet, "ParakeetMLXEngine", constructor)
    monkeypatch.setattr(pipeline, "_resolve_mlx_stt_backend", lambda: "parakeet-mlx")
    assert pipeline.load_whisper("mlx") is constructor.return_value
    assert constructor.call_args.kwargs["cache_limit_mb"] == expected
    constructor.return_value.load.assert_called_once()

    process = Mock()
    parent, child = Mock(), Mock()
    parent.recv.return_value = "ready"
    monkeypatch.setattr(pipeline.multiprocessing, "Pipe", lambda: (parent, child))
    monkeypatch.setattr(pipeline.multiprocessing, "Process", process)
    monkeypatch.setattr(pipeline, "ThreadPoolExecutor", Mock())
    for name in (
        "_stt_worker_proc",
        "_trans_worker_proc",
        "_stt_worker_conn",
        "_trans_worker_conn",
        "_stt_comm_pool",
        "_trans_comm_pool",
    ):
        monkeypatch.setattr(pipeline, name, None)
    pipeline._start_workers()
    assert len(process.call_args_list) == 2
    assert [call.kwargs["kwargs"]["cache_limit_mb"] for call in process.call_args_list] == [expected, expected]


@pytest.mark.parametrize("after_final", [True, False])
@pytest.mark.parametrize("translation_fails", [False, True])
def test_keep_warm_waits_for_final_translation(monkeypatch, after_final, translation_fails):
    import asyncio
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    from unittest.mock import ANY, AsyncMock

    import numpy as np

    import dry_run_ab as pipeline
    from tools.final_queue_pressure import FinalQueuePressure

    for name, value in {
        "WARMUP_AFTER_FINAL": after_final,
        "_warmup_pending": False,
        "_last_warmup_time": float("inf"),
        "_active_partial_future": None,
        "_stt_scheduler": None,
        "_final_pending": threading.Event(),
        "_final_pending_utterance_id": None,
        "_translation_active": threading.Event(),
        "_first_stream_chunks": {7},
        "_source_coverage": Mock(),
        "_health": None,
        "BACKEND": "mlx",
        "MULTIPROCESS": False,
        "mlx_a_model": object(),
        "mlx_a_tokenizer": object(),
        "mlx_b_model": None,
        "MLX_DRAFT_MODEL": None,
        "DIARIZE_ENABLED": False,
        "tts_engine": None,
        "_io_pool": Mock(),
        "all_results": [],
        "_chunks_completed": 0,
        "partial_translations": {},
        "partial_latencies": {},
        "_last_gen_stats": {},
        "diag_durations": [],
        "_utterance_start_times": {},
        "_utterance_timings": {},
    }.items():
        monkeypatch.setattr(pipeline, name, value)
    monkeypatch.setattr(pipeline, "_final_queue_pressure", FinalQueuePressure(origin=0))
    monkeypatch.setattr(pipeline, "_confirmed_speculation", lambda *args: None)
    monkeypatch.setattr(pipeline, "should_use_marian_only", lambda *args: False)
    monkeypatch.setattr(pipeline, "get_resource_snapshot", lambda: {})
    for name in ("check_homophones", "check_bad_split", "check_near_miss"):
        monkeypatch.setattr(pipeline, name, lambda *args: None)
    broadcast = AsyncMock()
    monkeypatch.setattr(pipeline, "broadcast", broadcast)
    started, release = threading.Event(), threading.Event()

    def translate(*args, **kwargs):
        started.set()
        assert release.wait(2)
        if translation_fails:
            raise ValueError("fake translation failure")
        return "La gracia de Dios", 100, 10

    monkeypatch.setattr(pipeline, "translate_mlx_streaming", translate)

    async def exercise():
        queue = asyncio.Queue()
        monkeypatch.setattr(pipeline, "_pipeline_chunk_queue", queue)
        monkeypatch.setattr(pipeline, "_pipeline_translation_lock", asyncio.Lock())
        audio = np.zeros(16000, dtype=np.float32)
        await pipeline.process_final(audio)
        assert queue.qsize() == 1
        assert pipeline._warmup_pending is False
        task = asyncio.create_task(
            pipeline._pipeline_translate_and_finalize(7, "God's grace", 0, None, [], [], audio, time.perf_counter())
        )
        try:
            deadline = time.monotonic() + 2
            while not started.is_set():
                assert time.monotonic() < deadline
                await asyncio.sleep(0.001)
            assert pipeline._translation_active.is_set()
            assert pipeline._warmup_pending is False
        finally:
            release.set()
            await task
        assert pipeline._warmup_pending is after_final
        assert not pipeline._translation_active.is_set()
        assert not pipeline._pipeline_translation_lock.locked()
        assert 7 not in pipeline._first_stream_chunks

    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(pipeline, "_pipeline_pool", pool)
        asyncio.run(exercise())
    if translation_fails:
        broadcast.assert_not_awaited()
        pipeline._source_coverage.outcome.assert_called_with(ANY, "translation_error", None)
    else:
        broadcast.assert_awaited_once()
        assert broadcast.await_args.args[0]["spanish_a"] == "La gracia de Dios"


@pytest.mark.parametrize("after_final", [True, False])
@pytest.mark.parametrize("hard_cut", [True, False])
def test_audio_loop_submit_warmup_kill_switch(monkeypatch, after_final, hard_cut):
    import asyncio
    import threading
    from unittest.mock import AsyncMock, MagicMock

    import numpy as np

    import dry_run_ab as pipeline
    from tools import audio_bridge_client
    from tools.capture_handoff import CaptureTransportSummary

    # Each fake capture is a separate session; discard watermarks must not leak.
    monkeypatch.setattr(pipeline, "_discarded_utterance_id", 0)
    monkeypatch.setattr(pipeline, "_last_capture_utterance_id", 0)
    monkeypatch.setattr(pipeline, "_source_coverage", Mock())
    monkeypatch.setattr(pipeline, "_capture_transport", CaptureTransportSummary())
    monkeypatch.setattr(pipeline, "diag_force_cuts", [])
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
        monkeypatch.setattr(pipeline, name, {})
    monkeypatch.setattr(pipeline, "WARMUP_AFTER_FINAL", after_final)
    monkeypatch.setattr(pipeline, "_warmup_pending", False)
    monkeypatch.setattr(pipeline, "_last_warmup_time", float("inf"))
    monkeypatch.setattr(pipeline, "EXIT_AFTER_REPLAY", True)
    monkeypatch.setattr(pipeline, "_partial_tasks", set())
    monkeypatch.setattr(pipeline, "_active_partial_future", None)
    monkeypatch.setattr(pipeline, "is_speech", lambda chunk, *args: bool(chunk.any()))
    monkeypatch.setattr(pipeline, "process_final", AsyncMock())
    monkeypatch.setattr(pipeline, "process_partial", AsyncMock())
    monkeypatch.setattr(pipeline, "vad_model", Mock())
    schedule = Mock()
    monkeypatch.setattr(pipeline, "_schedule_warmup", schedule)
    monkeypatch.setattr(pipeline.settings.vad, "max_utterance", 0.8 if hard_cut else 8.0)
    monkeypatch.setattr(pipeline, "_compute_force_cut", lambda size, *args: ("hard", size))
    stream = MagicMock(spec=["__enter__", "__exit__"])
    stream.finished = threading.Event()
    stream.finished.set()
    stream.error = None
    monkeypatch.setattr(audio_bridge_client, "open_audio_stream", lambda **kwargs: stream)

    async def exercise():
        queue = asyncio.Queue()
        for _ in range(25):
            queue.put_nowait(np.full(512, 0.1, np.float32))
        for _ in range(20):
            queue.put_nowait(np.zeros(512, np.float32))
        monkeypatch.setattr(pipeline, "audio_queue", queue)
        monkeypatch.setattr(pipeline, "_pipeline_chunk_queue", asyncio.Queue())
        await asyncio.wait_for(pipeline.audio_loop(), 2)
        pipeline.process_final.assert_awaited_once()
        assert pipeline._warmup_pending is (not after_final)
        assert bool(schedule.call_count) is (not after_final)

    asyncio.run(exercise())
