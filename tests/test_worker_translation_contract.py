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
