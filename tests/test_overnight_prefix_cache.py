"""Exact token boundaries and immutable prefix state; no actual model forward."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from engines.prefix_cache import FixedPrefixStore, shared_token_prefix


def test_actual_prompt_prefix_must_match_and_every_request_gets_independent_state():
    store = FixedPrefixStore()
    fills = []

    def prefill(tokens, cache):
        fills.append(tokens)
        cache[0]["tokens"] = list(tokens)

    first = store.prepare((1, 2), [1, 2, 3, 4], lambda: [{}], prefill)
    assert not first.hit and first.cached_tokens == 2 and first.prompt == [3, 4]
    first.cache[0]["tokens"].extend([3, 4, 99])  # simulate generating an assistant suffix
    second = store.prepare((1, 2), [1, 2, 7, 8], lambda: [{}], prefill)
    assert second.hit and second.cache == [{"tokens": [1, 2]}]
    assert second.prompt == [7, 8] and fills == [[1, 2]]
    merged_boundary = store.prepare((1, 2), [1, 23, 4], lambda: [{}], prefill)
    assert merged_boundary.cache is None and merged_boundary.prompt == [1, 23, 4]


def test_failed_prefill_is_never_reused_and_templates_are_bounded():
    store = FixedPrefixStore(capacity=1)

    def fail(*args):
        raise RuntimeError("prefill failure")

    with pytest.raises(RuntimeError, match="prefill failure"):
        store.prepare((1,), [1, 2], dict, fail)
    assert not store.prepare((1,), [1, 2], dict, lambda *a: None).hit
    assert not store.prepare((3,), [3, 2], dict, lambda *a: None).hit
    assert not store.prepare((1,), [1, 2], dict, lambda *a: None).hit


def test_shared_prefix_stops_before_differing_or_merged_source_tokens():
    assert shared_token_prefix([[1, 2, 3, 7], [1, 2, 4, 8], [1, 2, 5]]) == (1, 2)
    assert shared_token_prefix([[1, 23], [1, 2, 3]]) == (1,)


def test_prompt_tokenization_uses_same_bos_rule_as_installed_stream_generate():
    from engines.mlx_engine import _prompt_tokens

    tokenizer = SimpleNamespace(bos_token="<bos>", encode=MagicMock(return_value=[1, 2]))
    assert _prompt_tokens(tokenizer, "<bos>Hi") == [1, 2]
    tokenizer.encode.assert_called_with("<bos>Hi", add_special_tokens=False)
    _prompt_tokens(tokenizer, "Hi")
    tokenizer.encode.assert_called_with("Hi", add_special_tokens=True)


def test_prefix_preparation_time_is_counted_in_first_token_and_model_time(monkeypatch):
    import engines.mlx_engine as engine
    from engines.base import TranslationResult
    from engines.prefix_cache import PreparedPrefix

    monkeypatch.setattr(
        engine, "_prepare_gemma_prefix", lambda *a: (PreparedPrefix([3], [{"state": 1}], 20, False), 21)
    )
    monkeypatch.setattr(
        engine, "_generate_translation", lambda *a, **k: TranslationResult("Hola", 100, ttft_ms=30, decode_ms=70)
    )
    with patch.object(engine.time, "perf_counter", side_effect=[1, 1.01, 1.01, 1.03]):
        result = engine.generate_translation(
            object(), object(), model_family="gemma4", gen_kwargs={"prompt": [1, 2, 3], "_stark_prefix_options": {}}
        )
    assert result.latency_ms == pytest.approx(120)
    assert result.ttft_ms == pytest.approx(50)
    assert result.decode_ms == 70
    assert result.generation_lock_wait_ms == pytest.approx(10)
    assert result.prompt_tokens == 21 and result.cached_prompt_tokens == 20


def test_prefix_and_mtp_combination_fails_explicitly(monkeypatch):
    from engines.mlx_engine import translate_loaded_model

    monkeypatch.setenv("STARK_EXPERIMENT_GEMMA_PREFIX_CACHE", "true")
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: [1, 2])
    with pytest.raises(ValueError, match="cannot be combined"):
        translate_loaded_model(object(), tokenizer, "Hello", draft_model=object())
