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
