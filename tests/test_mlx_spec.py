"""Tests for engines.mlx_spec — Gemma-4 MTS wrapper (optiq mocked, no GPU)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from engines.mlx_spec import (
    SpecToken,
    compute_spec_stats,
    load_gemma4_drafter,
    spec_stream,
)


@dataclass
class FakeEvent:
    kind: str
    token_id: int = -1
    text: str = ""
    from_draft: bool = False


@dataclass
class FakeSpecConfig:
    gamma: int = 1
    max_tokens: int = 256
    eos_token_id: int | None = None
    accept_temp: float = 0.0


class FakeDetokenizer:
    """Streaming detokenizer stub: last_segment is the piece since last read."""

    def __init__(self, pieces: dict[int, str] | None = None):
        self.pieces = pieces or {}
        self.tokens: list[int] = []
        self.text = ""
        self._unread = ""
        self.finalized = False

    def reset(self) -> None:
        self.tokens = []
        self.text = ""
        self._unread = ""
        self.finalized = False

    def add_token(self, token_id: int) -> None:
        self.tokens.append(token_id)
        piece = self.pieces.get(token_id, f"<{token_id}>")
        self.text += piece
        self._unread += piece

    def finalize(self) -> None:
        self.finalized = True

    @property
    def last_segment(self) -> str:
        segment = self._unread
        self._unread = ""
        return segment


class FakeTokenizer:
    def __init__(
        self,
        *,
        eos_ids: set[int] | None = None,
        pieces: dict[int, str] | None = None,
        use_public_eos_attr: bool = True,
    ):
        self.bos_token_id = 2
        self.bos_token = "<bos>"
        self.eos_token_id = 1
        self._eos_token_ids = eos_ids if eos_ids is not None else {1, 106, 50}
        if use_public_eos_attr:
            self.eos_token_ids = set(self._eos_token_ids)
        self._detok = FakeDetokenizer(pieces)

    @property
    def detokenizer(self) -> FakeDetokenizer:
        return self._detok

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        if not ids:
            return ""
        if ids[0] == self.bos_token_id and not skip_special_tokens:
            return self.bos_token + "PROMPT"
        return "PROMPT"

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        add_special = kwargs.get("add_special_tokens", True)
        body = [7, 8, 9]
        if add_special:
            return [self.bos_token_id, *body]
        return body


def _install_fake_spec(monkeypatch, events, captured: dict[str, Any] | None = None):
    def spec_generate(target, drafter, tokenizer, prompt, cfg=None):
        if captured is not None:
            captured["target"] = target
            captured["drafter"] = drafter
            captured["tokenizer"] = tokenizer
            captured["prompt"] = prompt
            captured["cfg"] = cfg
            captured["encode_result"] = tokenizer.encode(prompt)
        yield from events

    def _import_spec():
        return MagicMock(), spec_generate, FakeSpecConfig

    monkeypatch.setattr("engines.mlx_spec._import_spec", _import_spec)


class TestModuleImport:
    def test_no_top_level_optiq_binding(self):
        import engines.mlx_spec as ms

        assert "GemmaAssistantDrafter" not in vars(ms)
        assert "spec_generate" not in vars(ms)


class TestComputeSpecStats:
    def test_gamma1_formula(self):
        stats = compute_spec_stats(n_tokens=5, n_from_draft=2, gamma=1)
        assert stats.n_verify_steps == 2  # 5 - 2 - 1
        assert stats.accept_rate == pytest.approx(1.0)

    def test_gamma1_partial_accept(self):
        stats = compute_spec_stats(n_tokens=5, n_from_draft=1, gamma=1)
        assert stats.n_verify_steps == 3
        assert stats.accept_rate == pytest.approx(1 / 3)

    def test_gamma3_generalization(self):
        # 1 prefill + 4 drafts + 3 corrections = 8 tokens, 3 verify steps
        stats = compute_spec_stats(n_tokens=8, n_from_draft=4, gamma=3)
        assert stats.n_verify_steps == 3
        assert stats.accept_rate == pytest.approx(4 / 9)

    def test_zero_tokens(self):
        stats = compute_spec_stats(n_tokens=0, n_from_draft=0, gamma=1)
        assert stats.n_verify_steps == 0
        assert stats.accept_rate == 0.0


class TestLoadGemma4Drafter:
    def test_import_error_becomes_runtime_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "optiq", None)
        monkeypatch.setitem(sys.modules, "optiq.runtime", None)
        monkeypatch.setitem(sys.modules, "optiq.runtime.spec", None)
        with pytest.raises(RuntimeError, match="mlx-optiq"):
            load_gemma4_drafter("mlx-community/gemma-4-e4b-it-assistant-bf16")

    def test_from_pretrained_then_eval(self, monkeypatch):
        drafter_cls = MagicMock()
        instance = MagicMock()
        params = {"w": object()}
        instance.parameters.return_value = params
        drafter_cls.from_pretrained.return_value = instance
        fake_mx = MagicMock()

        monkeypatch.setattr(
            "engines.mlx_spec._import_spec",
            lambda: (drafter_cls, MagicMock(), FakeSpecConfig),
        )
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

        out = load_gemma4_drafter("some/repo")
        assert out is instance
        drafter_cls.from_pretrained.assert_called_once_with("some/repo")
        fake_mx.eval.assert_called_once_with(params)


class TestSpecStream:
    def test_spec_stream_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "optiq", None)
        monkeypatch.setitem(sys.modules, "optiq.runtime", None)
        monkeypatch.setitem(sys.modules, "optiq.runtime.spec", None)
        with pytest.raises(RuntimeError, match="mlx-optiq"):
            spec_stream(MagicMock(), FakeTokenizer(), MagicMock(), [2, 3])

    def test_stop_on_eos_id(self, monkeypatch):
        events = [
            FakeEvent("token", 10, from_draft=False),
            FakeEvent("token", 11, from_draft=True),
            FakeEvent("token", 1, from_draft=False),  # <eos>
            FakeEvent("token", 99, from_draft=False),
            FakeEvent("done"),
        ]
        _install_fake_spec(monkeypatch, events)
        tok = FakeTokenizer(pieces={10: "Hel", 11: "lo"})
        stream = spec_stream(MagicMock(), tok, MagicMock(), [2, 7], gamma=1)
        tokens = list(stream)
        assert [t.token_id for t in tokens] == [10, 11]
        assert 99 not in [t.token_id for t in tokens]
        assert tok.detokenizer.finalized is True

    def test_stop_on_end_of_turn(self, monkeypatch):
        events = [
            FakeEvent("token", 10, from_draft=False),
            FakeEvent("token", 106, from_draft=True),  # <turn|>
            FakeEvent("token", 11, from_draft=False),
            FakeEvent("done"),
        ]
        _install_fake_spec(monkeypatch, events)
        tok = FakeTokenizer(eos_ids={1, 106, 50}, pieces={10: "x"})
        tokens = list(spec_stream(MagicMock(), tok, MagicMock(), [2, 7]))
        assert [t.token_id for t in tokens] == [10]

    def test_stop_on_private_eos_attr(self, monkeypatch):
        events = [
            FakeEvent("token", 10, from_draft=False),
            FakeEvent("token", 50, from_draft=False),  # <|tool_response>
            FakeEvent("token", 11, from_draft=False),
            FakeEvent("done"),
        ]
        _install_fake_spec(monkeypatch, events)
        tok = FakeTokenizer(eos_ids={1, 106, 50}, pieces={10: "x"}, use_public_eos_attr=False)
        tokens = list(spec_stream(MagicMock(), tok, MagicMock(), [2, 7]))
        assert [t.token_id for t in tokens] == [10]

    def test_text_assembled_through_detokenizer(self, monkeypatch):
        events = [
            FakeEvent("token", 10, text="WRONG", from_draft=False),
            FakeEvent("token", 11, text="WRONG", from_draft=True),
            FakeEvent("token", 12, text="WRONG", from_draft=False),
            FakeEvent("done"),
        ]
        _install_fake_spec(monkeypatch, events)
        tok = FakeTokenizer(pieces={10: "Hello", 11: " world", 12: "!"})
        tokens = list(spec_stream(MagicMock(), tok, MagicMock(), [2, 7]))
        assert "".join(t.text for t in tokens) == "Hello world!"
        assert all(t.text != "WRONG" for t in tokens)
        assert tokens[0] == SpecToken(10, False, "Hello")
        assert tokens[1].from_draft is True

    def test_accept_rate_gamma1(self, monkeypatch):
        # prefill + accepted draft + correction  →  n_verify = 3-1-1 = 1, rate = 1.0
        events = [
            FakeEvent("token", 10, from_draft=False),
            FakeEvent("token", 11, from_draft=True),
            FakeEvent("token", 12, from_draft=False),
            FakeEvent("done"),
        ]
        _install_fake_spec(monkeypatch, events)
        stream = spec_stream(MagicMock(), FakeTokenizer(), MagicMock(), [2, 7], gamma=1)
        tokens = list(stream)
        assert len(tokens) == 3
        assert stream.stats is not None
        assert stream.stats.n_tokens == 3
        assert stream.stats.n_from_draft == 1
        assert stream.stats.n_verify_steps == 1
        assert stream.stats.accept_rate == pytest.approx(1.0)
        assert stream.stats.gamma == 1

    def test_accept_rate_gamma1_zero_drafts(self, monkeypatch):
        events = [
            FakeEvent("token", 10, from_draft=False),
            FakeEvent("token", 12, from_draft=False),
            FakeEvent("done"),
        ]
        _install_fake_spec(monkeypatch, events)
        stream = spec_stream(MagicMock(), FakeTokenizer(), MagicMock(), [2, 7], gamma=1)
        list(stream)
        assert stream.stats is not None
        assert stream.stats.n_from_draft == 0
        assert stream.stats.n_verify_steps == 1
        assert stream.stats.accept_rate == pytest.approx(0.0)

    def test_gamma_passthrough(self, monkeypatch):
        captured: dict[str, Any] = {}
        events = [FakeEvent("token", 10, from_draft=False), FakeEvent("done")]
        _install_fake_spec(monkeypatch, events, captured=captured)
        list(spec_stream(MagicMock(), FakeTokenizer(), MagicMock(), [2, 7, 8], gamma=3, max_tokens=32))
        cfg = captured["cfg"]
        assert cfg.gamma == 3
        assert cfg.max_tokens == 32
        assert cfg.accept_temp == 0.0
        assert cfg.eos_token_id == 1

    def test_gamma_gt1_accept_rate(self, monkeypatch):
        # 1 prefill + 2 accepted drafts + 1 bonus  (one verify step at gamma=2)
        events = [
            FakeEvent("token", 10, from_draft=False),
            FakeEvent("token", 11, from_draft=True),
            FakeEvent("token", 12, from_draft=True),
            FakeEvent("token", 13, from_draft=False),
            FakeEvent("done"),
        ]
        _install_fake_spec(monkeypatch, events)
        stream = spec_stream(MagicMock(), FakeTokenizer(), MagicMock(), [2, 7], gamma=2)
        list(stream)
        assert stream.stats is not None
        assert stream.stats.n_tokens == 4
        assert stream.stats.n_from_draft == 2
        assert stream.stats.n_verify_steps == 1
        assert stream.stats.accept_rate == pytest.approx(2 / 2)  # 2 drafts / (1 * gamma=2)

    def test_prompt_ids_avoid_double_bos(self, monkeypatch):
        captured: dict[str, Any] = {}
        events = [FakeEvent("token", 10, from_draft=False), FakeEvent("done")]
        _install_fake_spec(monkeypatch, events, captured=captured)
        prompt_ids = [2, 7, 8, 9]
        list(spec_stream(MagicMock(), FakeTokenizer(), MagicMock(), prompt_ids, gamma=1))
        assert captured["prompt"] == "PROMPT"  # leading BOS stripped before decode
        assert not str(captured["prompt"]).startswith("<bos>")
        # First encode() is pinned to the original ids (no extra BOS).
        assert captured["encode_result"] == prompt_ids

    def test_unpack_spec_token(self, monkeypatch):
        events = [FakeEvent("token", 10, from_draft=False), FakeEvent("done")]
        _install_fake_spec(monkeypatch, events)
        tok = next(iter(spec_stream(MagicMock(), FakeTokenizer(pieces={10: "Hi"}), MagicMock(), [2])))
        token_id, from_draft, text = tok
        assert token_id == 10
        assert from_draft is False
        assert text == "Hi"
