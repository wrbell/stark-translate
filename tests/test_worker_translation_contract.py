"""Exercise the actual worker Pipe loop and canonical prompts without weights."""

import multiprocessing
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("family", ["gemma4", "translategemma"])
@pytest.mark.parametrize("source,target", [("en", "es"), ("es", "en")])
def test_pipe_worker_uses_shared_engine_prompt_direction_and_ab_contract(monkeypatch, family, source, target):
    import mlx_lm

    import engines.mlx_engine as implementation
    from workers import translation_worker_main

    loaded, prompts = [], []
    monkeypatch.setattr(implementation, "MLX_AVAILABLE", True)
    monkeypatch.delenv("STARK_EXPERIMENT_GEMMA_PREFIX_CACHE", raising=False)

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            prompts.append((messages, kwargs))
            return [1, 2, 3]

    def load(engine):
        engine._model = object()
        engine._tokenizer = Tokenizer()
        engine._loaded = True
        loaded.append(engine)

    monkeypatch.setattr(implementation.MLXGemmaEngine, "load", load)
    monkeypatch.setattr(
        mlx_lm,
        "stream_generate",
        lambda *a, **k: iter([SimpleNamespace(text="translated", generation_tokens=1, generation_tps=42.0)]),
    )
    parent, child = multiprocessing.Pipe()
    thread = threading.Thread(
        target=translation_worker_main,
        args=(child, "model-a", "model-b"),
        kwargs={"source_lang": source, "target_lang": target, "model_family": family, "adapter_path": "adapter-a"},
    )
    thread.start()
    try:
        assert parent.poll(2) and parent.recv() == "ready"
        parent.send(("translate", "A test", True))
        assert parent.poll(2)
        result = parent.recv()
        assert len(result) == 6 and result[0] == result[3] == "translated"
        assert result[2] == result[5] == 42.0
        parent.send(("translate", "Another test", False))
        assert parent.poll(2)
        result = parent.recv()
        assert result[3:] == (None, 0.0, 0.0)
        assert loaded[0]._adapter_path == "adapter-a"
        for messages, options in prompts:
            if family == "gemma4":
                assert isinstance(messages[0]["content"], str)
                assert options["enable_thinking"] is False
                assert ("English text to Spanish" if source == "en" else "Spanish text to English") in messages[0][
                    "content"
                ]
            else:
                content = messages[0]["content"][0]
                assert (content["source_lang_code"], content["target_lang_code"]) == (source, target)
                assert "enable_thinking" not in options
    finally:
        parent.send(None)
        thread.join(2)
        parent.close()
        child.close()
    assert not thread.is_alive()
    assert all(not engine._loaded for engine in loaded)


def test_live_mts_rejects_before_any_target_model_load(monkeypatch):
    import dry_run_ab as pipeline

    for requested, disabled, configured in ((True, False, False), (None, False, True)):
        with pytest.raises(ValueError, match="will not silently continue"):
            pipeline.validate_live_mts(requested, disabled, configured)
    pipeline.validate_live_mts(True, True, True)
    pipeline.validate_live_mts(None, False, False)
    monkeypatch.setattr(pipeline, "USE_MTS", True)
    loader = Mock()
    monkeypatch.setattr(pipeline, "load_mlx_gemma", loader)
    with pytest.raises(RuntimeError, match="offline experiment"):
        pipeline.load_translation_models()
    loader.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("load_b", [False, True])
def test_live_full_model_draft_load_order_and_ab_models(monkeypatch, enabled, load_b):
    import dry_run_ab as pipeline
    from tools.latency_experiments import LatencyExperiments

    draft_id = "mlx-community/gemma-4-e2b-it-OptiQ-4bit"
    monkeypatch.setattr(
        pipeline,
        "_latency",
        LatencyExperiments.from_env(
            {
                "STARK_EXPERIMENT_DRAFT_MODEL_ID": draft_id,
                "STARK_EXPERIMENT_DRAFT_TOKENS": "2",
            }
            if enabled
            else {}
        ),
    )
    for name, value in {
        "USE_MTS": False,
        "MODEL_FAMILY": "gemma4",
        "MLX_MODEL_A": "target-e4b",
        "ADAPTER_DIR_A": None,
        "MLX_DRAFT_MODEL": None,
        "MLX_DRAFT_MODEL_ID": None,
        "NUM_DRAFT_TOKENS": 3,
        "mlx_a_prompt_cache": None,
        "mlx_a_suffix_tokens": None,
        "mlx_b_prompt_cache": None,
        "mlx_b_suffix_tokens": None,
    }.items():
        monkeypatch.setattr(pipeline, name, value)
    target, draft, secondary = object(), object(), object()
    models = [(target, "target-tokenizer")]
    if enabled:
        models.append((draft, "draft-tokenizer"))
    if load_b:
        models.append((secondary, "secondary-tokenizer"))
    loader = Mock(side_effect=models)
    monkeypatch.setattr(pipeline, "load_mlx_gemma", loader)
    monkeypatch.setattr("engines.factory.resolve_mlx_translation_model_id", lambda **kw: "ab-e2b")
    result = pipeline.load_translation_models(load_b=load_b)
    assert [call.args[0] for call in loader.call_args_list] == [
        "target-e4b",
        *([draft_id] if enabled else []),
        *(["ab-e2b"] if load_b else []),
    ]
    assert result == (
        target,
        "target-tokenizer",
        secondary if load_b else None,
        "secondary-tokenizer" if load_b else None,
    )
    assert pipeline.MLX_DRAFT_MODEL is (draft if enabled else None)
    assert (2 if enabled else 3) == pipeline.NUM_DRAFT_TOKENS
    assert pipeline._draft_metadata() == {"draft": draft_id if enabled else None, "draft_tokens": 2 if enabled else 0}
    # Reject the old --mts request even when the independent experiment is set.
    monkeypatch.setattr(pipeline, "USE_MTS", True)
    loader.reset_mock()
    with pytest.raises(RuntimeError, match="offline experiment"):
        pipeline.load_translation_models(load_b=load_b)
    loader.assert_not_called()


def test_live_draft_reaches_shared_translation(monkeypatch):
    import dry_run_ab as pipeline
    from engines.base import TranslationResult

    draft = object()
    monkeypatch.setattr(pipeline, "NUM_DRAFT_TOKENS", 2)
    shared = Mock(return_value=TranslationResult("translated", 10, 20))
    monkeypatch.setattr("engines.mlx_engine.translate_loaded_model", shared)
    assert pipeline.translate_mlx(object(), object(), "source", draft_model=draft) == ("translated", 10, 20)
    assert shared.call_args.kwargs["draft_model"] is draft
    assert shared.call_args.kwargs["num_draft_tokens"] == 2
