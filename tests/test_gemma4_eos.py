"""Family-specific EOS regression tests; never load real weights."""

from unittest.mock import MagicMock, patch

import pytest

from engines.translation_prompts import ensure_stop_tokens, stop_token_strings


@pytest.fixture(autouse=True)
def _local_model_source(monkeypatch):
    # These suites isolate MLX behavior; real pinned resolution has its own tests.
    monkeypatch.setattr("engines.mlx_engine.resolve_model_for_loading", lambda model: "/installed/" + model)


class FakeTokenizer:
    eos_token_id = 1
    unk_token_id = 3

    def __init__(self, family="gemma4"):
        self.family = family
        self._eos_token_ids = {1, 106, 50} if family == "gemma4" else {1}

    def convert_tokens_to_ids(self, token):
        turn = "<turn|>" if self.family == "gemma4" else "<end_of_turn>"
        return 106 if token == turn else 3


@pytest.mark.parametrize("family,expected", [("gemma4", {1, 106, 50}), ("translategemma", {1, 106})])
def test_preserves_family_eos(family, expected):
    tok = FakeTokenizer(family)
    original_set = tok._eos_token_ids
    original = tok._eos_token_ids.copy()
    assert ensure_stop_tokens(tok, model_family=family) == expected
    assert tok._eos_token_ids == expected
    assert tok._eos_token_ids is original_set
    assert original <= tok._eos_token_ids
    assert 3 not in tok._eos_token_ids
    assert ensure_stop_tokens(tok, model_family=family) == expected


def test_unknown_translategemma_token_is_not_added_to_gemma4():
    tok = FakeTokenizer()
    assert tok.convert_tokens_to_ids("<end_of_turn>") == 3
    assert ensure_stop_tokens(tok, model_family="translategemma") == {1, 106, 50}


@pytest.mark.parametrize("invalid", [None, -1, 3])
def test_invalid_token_is_skipped(invalid):
    tok = FakeTokenizer()
    tok.convert_tokens_to_ids = lambda _: invalid
    assert ensure_stop_tokens(tok, model_family="gemma4") == {1, 106, 50}


def test_add_eos_method_and_missing_attribute():
    tok = FakeTokenizer("translategemma")
    del tok._eos_token_ids
    tok.add_eos_token = MagicMock(side_effect=lambda token: tok._eos_token_ids.add(tok.convert_tokens_to_ids(token)))
    assert ensure_stop_tokens(tok, model_family="translategemma") == {1, 106}
    tok.add_eos_token.assert_called_once_with("<end_of_turn>")
    ensure_stop_tokens(tok, model_family="translategemma")
    assert tok.add_eos_token.call_count == 1


def test_missing_attribute_without_add_method():
    tok = FakeTokenizer()
    del tok._eos_token_ids
    assert ensure_stop_tokens(tok, model_family="gemma4") == {1, 106}
    assert tok._eos_token_ids == {1, 106}


def test_stop_strings():
    assert stop_token_strings("gemma4") == ("<turn|>",)
    assert stop_token_strings("translategemma") == ("<end_of_turn>",)


@pytest.mark.parametrize("family", ["gemma4", "translategemma"])
def test_engine_load_preserves_stop_tokens(family):
    from engines.mlx_engine import MLXGemmaEngine

    tok = FakeTokenizer(family)
    with (
        patch("engines.mlx_engine.MLX_AVAILABLE", True),
        patch("engines.mlx_engine.mx", MagicMock()),
        patch("engines.mlx_engine.materialize_mlx_model"),
        # These tests isolate EOS setup; first-forward behavior has its own suite.
        patch("engines.mlx_engine.warm_mlx_model"),
        patch("mlx_lm.load", return_value=(object(), tok)),
    ):
        engine = MLXGemmaEngine(model_family=family, use_prompt_cache=False)
        engine.load()
    assert {106} <= tok._eos_token_ids
    assert 3 not in tok._eos_token_ids
    if family == "gemma4":
        assert tok._eos_token_ids == {1, 106, 50}


@pytest.mark.parametrize("family", [None, "translategemma"])
def test_pipeline_load_uses_global_or_explicit_family(monkeypatch, family):
    import dry_run_ab as d

    tok = FakeTokenizer(family or "gemma4")
    monkeypatch.setattr(d, "MODEL_FAMILY", "gemma4")
    monkeypatch.setattr(d, "USE_TURBOQUANT", False)
    with (
        patch("mlx_lm.load", return_value=(MagicMock(), tok)),
        patch("engines.mlx_engine.warm_mlx_model"),
    ):
        d.load_mlx_gemma("fake", "test", model_family=family)
    assert tok._eos_token_ids == ({1, 106} if family else {1, 106, 50})


def test_health_check_gemma4_preserves_config_eos():
    from tools.health_check import _load_mlx

    tok = FakeTokenizer()
    with patch("mlx_lm.load", return_value=(object(), tok)):
        _load_mlx("fake", None, "gemma4")
    assert tok._eos_token_ids == {1, 106, 50}
