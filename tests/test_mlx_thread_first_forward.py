"""warm_mlx_model: the first forward pass runs on the load thread (mlx thread-local streams)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from engines import mlx_engine
from engines.translation_prompts import gemma4_user_content


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, add_generation_prompt=True, **kwargs):
        self.calls.append((messages, kwargs))
        return [2, 3, 4]


def test_warm_mlx_model_runs_one_token_generation_with_family_prompt():
    tok = _FakeTokenizer()
    fake_generate = MagicMock(return_value="x")
    with patch.object(sys.modules["mlx_lm"], "generate", fake_generate):
        mlx_engine.warm_mlx_model("model", tok, model_family="gemma4", label="unit")
    assert fake_generate.call_count == 1
    assert fake_generate.call_args.kwargs["max_tokens"] == 1
    messages, kwargs = tok.calls[0]
    assert messages[0]["content"] == gemma4_user_content("Hello.", "en", "es")
    assert kwargs == {"enable_thinking": False}


def test_warm_mlx_model_translategemma_prompt_has_no_thinking_kwarg():
    tok = _FakeTokenizer()
    with patch.object(sys.modules["mlx_lm"], "generate", MagicMock()):
        mlx_engine.warm_mlx_model("model", tok, model_family="translategemma")
    messages, kwargs = tok.calls[0]
    assert messages[0]["content"][0]["source_lang_code"] == "en"
    assert kwargs == {}


def test_warm_mlx_model_swallows_errors():
    tok = _FakeTokenizer()
    with patch.object(sys.modules["mlx_lm"], "generate", MagicMock(side_effect=RuntimeError("boom"))):
        mlx_engine.warm_mlx_model("model", tok, model_family="gemma4")  # must not raise


def test_warm_mlx_model_noop_without_model_or_tokenizer():
    fake_generate = MagicMock()
    with patch.object(sys.modules["mlx_lm"], "generate", fake_generate):
        mlx_engine.warm_mlx_model(None, _FakeTokenizer(), model_family="gemma4")
        mlx_engine.warm_mlx_model("model", None, model_family="gemma4")
    fake_generate.assert_not_called()


def test_engine_load_calls_warm_forward_on_load_thread():
    tok = _FakeTokenizer()
    tok.eos_token_id = 1
    tok.unk_token_id = 3
    tok._eos_token_ids = {1, 106, 50}
    tok.convert_tokens_to_ids = lambda s: 106 if s == "<turn|>" else 3
    with (
        patch.object(sys.modules["mlx_lm"], "load", return_value=("model", tok)),
        patch.object(mlx_engine, "materialize_mlx_model") as materialize,
        patch.object(mlx_engine, "warm_mlx_model") as warm,
    ):
        eng = mlx_engine.MLXGemmaEngine(model_family="gemma4", use_prompt_cache=False)
        eng.load()
    materialize.assert_called_once_with("model")
    warm.assert_called_once()
    assert warm.call_args.kwargs["model_family"] == "gemma4"
