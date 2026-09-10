"""Live Gemma 4 OptiQ regression for the EOS/stop-token fix (#172).

Loads the real Mac default model once, verifies the Gemma stop set, and checks
that canary translations stop before ``max_tokens`` without template markers.
The eight-item lexical slice must retain the observed stock floor of 6/8; this is
a regression check, not a 7/8 quality-promotion claim or the full 18-item review.
"""

from __future__ import annotations

import pytest

from engines.translation_prompts import dynamic_max_tokens
from training.theological_canaries import THEOLOGICAL_CANARIES

MODEL_ID = "mlx-community/gemma-4-e4b-it-OptiQ-4bit"
MARKERS = ("<turn|>", "<|channel>", "<|turn>", "<end_of_turn>")


@pytest.fixture(scope="module")
def engine():
    from engines.mlx_engine import MLXGemmaEngine

    eng = MLXGemmaEngine(model_id=MODEL_ID, model_family="gemma4", use_prompt_cache=False)
    eng.load()
    eng.translate("Warm up.", source_lang="en", target_lang="es")
    yield eng
    eng.unload()


def test_stop_set_contains_gemma4_turn_token(engine):
    ids = engine._tokenizer._eos_token_ids
    assert 106 in ids, ids  # <turn|>
    assert 3 not in ids, ids  # <unk> must never be a stop token


def test_canaries_stop_before_max_tokens(engine):
    passed = 0
    for canary in THEOLOGICAL_CANARIES[:8]:
        text = canary["en"]
        expected = [s.lower() for s in canary["expected_substrings"]]
        res = engine.translate(text, source_lang="en", target_lang="es")
        max_tok = dynamic_max_tokens(text)
        assert res.finish_reason == "stop", (text, res.finish_reason, res.text)
        assert res.generated_tokens is not None and res.generated_tokens < max_tok - 8, (
            text,
            res.generated_tokens,
            max_tok,
        )
        assert not any(m in res.text for m in MARKERS), res.text
        assert res.latency_ms < 1500, (text, res.latency_ms)
        if all(sub in res.text.lower() for sub in expected):
            passed += 1
    # health_check's 8-sentence slice scores 6/8 on stock E4B (Santiago / partimiento lexical gaps)
    assert passed >= 6, f"canary {passed}/8"
