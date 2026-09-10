"""Shared translation prompt + cleanup helpers for MLX / CUDA / llama.cpp.

Keeps Gemma 4 instruct and TranslateGemma structured prompts identical across
backends so Mac and CUDA production paths stay semantically aligned.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

GEMMA4_STOP_TOKENS = ("<turn|>",)
TRANSLATEGEMMA_STOP_TOKENS = ("<end_of_turn>",)


def stop_token_strings(model_family: str) -> tuple[str, ...]:
    """Return the turn terminators for the model's chat template."""
    return GEMMA4_STOP_TOKENS if model_family == "gemma4" else TRANSLATEGEMMA_STOP_TOKENS


def ensure_stop_tokens(tokenizer, *, model_family: str) -> set[int]:
    """Add valid family stop tokens, preserving every EOS id supplied by load()."""
    existing = getattr(tokenizer, "_eos_token_ids", None)
    eos_ids = set(existing) if existing else {tokenizer.eos_token_id}
    original = eos_ids.copy()
    # Initialize before add_eos_token, which expects the backing set to exist.
    if existing is None:
        tokenizer._eos_token_ids = eos_ids.copy()
    for token in stop_token_strings(model_family):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0 or token_id == getattr(tokenizer, "unk_token_id", None):
            continue
        if token_id not in eos_ids:
            add_eos = getattr(tokenizer, "add_eos_token", None)
            if callable(add_eos):
                add_eos(token)
            eos_ids.add(token_id)
    eos_ids.update(getattr(tokenizer, "_eos_token_ids", None) or ())
    tokenizer._eos_token_ids.update(eos_ids)
    logger.info("%s EOS ids: added=%s already present=%s", model_family, sorted(eos_ids - original), sorted(original))
    return eos_ids


# Canonical language display names for Gemma 4 instruct prompts.
LANG_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "hi": "Hindi",
    "zh": "Chinese",
}

# Preamble strings Gemma 4 sometimes emits before the actual translation.
GEMMA4_PREAMBLES: tuple[str, ...] = (
    "Here is the translation:\n",
    "Here is the translation:",
)

# Default max-token ratio: Spanish ~1.3x English words; subword tokenization
# adds overhead. Engines historically used 1.8 (dry_run_ab / CUDA) or 3.0 (MLX
# engine). Prefer the more generous ratio for Gemma 4 finals to avoid truncation.
DEFAULT_MAX_TOK_RATIO = 3.0
DEFAULT_MAX_TOK_FLOOR = 64


def lang_display_name(code: str) -> str:
    """Return a human-readable language name, falling back to the raw code."""
    return LANG_NAMES.get(code, code)


def gemma4_user_content(text: str, source_lang: str = "en", target_lang: str = "es") -> str:
    """Plain-text instruct prompt for Gemma 4 (and llama.cpp / OptiQ paths)."""
    src_name = lang_display_name(source_lang)
    tgt_name = lang_display_name(target_lang)
    return (
        f"Translate the following {src_name} text to {tgt_name}. Output only the translation, nothing else.\n\n{text}"
    )


def gemma4_chat_messages(
    text: str,
    source_lang: str = "en",
    target_lang: str = "es",
) -> list[dict[str, Any]]:
    """Chat-template messages for Gemma 4 instruct (HF / mlx-lm)."""
    return [
        {
            "role": "user",
            "content": gemma4_user_content(text, source_lang, target_lang),
        }
    ]


def translategemma_chat_messages(
    text: str,
    source_lang: str = "en",
    target_lang: str = "es",
) -> list[dict[str, Any]]:
    """Structured TranslateGemma chat-template messages with lang codes."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source_lang,
                    "target_lang_code": target_lang,
                    "text": text,
                }
            ],
        }
    ]


def build_chat_messages(
    text: str,
    *,
    source_lang: str = "en",
    target_lang: str = "es",
    model_family: str = "translategemma",
    terminology_prompt: str = "none",
) -> list[dict[str, Any]]:
    """Dispatch to Gemma 4 or TranslateGemma message builders."""
    if terminology_prompt not in {"none", "church"}:
        raise ValueError("terminology_prompt must be none or church")
    if model_family == "gemma4":
        messages = gemma4_chat_messages(text, source_lang, target_lang)
        if terminology_prompt == "church" and target_lang == "es":
            messages[0]["content"] = (
                "For biblical names use Jacobo for James the person and Santiago for the epistle. "
                "Use partimiento del pan for breaking of bread and pacto for covenant. " + messages[0]["content"]
            )
        return messages
    return translategemma_chat_messages(text, source_lang, target_lang)


def llamacpp_user_content(
    text: str,
    *,
    source_lang: str = "en",
    target_lang: str = "es",
    model_family: str = "gemma4",
) -> str:
    """User content string for llama-server / OptiQ OpenAI-compatible APIs.

    TranslateGemma structured blocks are not supported by llama-server, so the
    non-gemma4 path falls back to a plain instruction.
    """
    if model_family == "gemma4":
        return gemma4_user_content(text, source_lang, target_lang)
    return f"Translate from {source_lang} to {target_lang}: {text}"


def clean_translation(
    raw: str,
    *,
    model_family: str = "translategemma",
) -> str:
    """Strip EOS markers and Gemma 4 preamble / thinking leakage from raw output."""
    clean = raw.split("<end_of_turn>")[0].strip()
    if model_family == "gemma4":
        # Thinking / channel markers (Gemma 4 chat template) — keep text before them.
        # Longer markers first so "<turn|><|channel>" wins over "<|channel>".
        for marker in ("<turn|><|channel>", "<|channel>", "<|turn>thought"):
            if marker in clean:
                clean = clean.split(marker)[0].strip()
                break
        while clean.endswith("<turn|>"):
            clean = clean[: -len("<turn|>")].strip()
        for prefix in GEMMA4_PREAMBLES:
            if clean.startswith(prefix):
                clean = clean[len(prefix) :].strip()
                break
    return clean


def chat_template_extra_kwargs(*, model_family: str = "translategemma") -> dict[str, Any]:
    """Extra kwargs for ``tokenizer.apply_chat_template`` (MLX / HF parity).

    Gemma 4 defaults to chain-of-thought unless ``enable_thinking=False``.
    llama.cpp passes the same via ``chat_template_kwargs``; mlx-lm accepts the
    flag as a top-level ``apply_chat_template`` keyword.
    """
    if model_family == "gemma4":
        return {"enable_thinking": False}
    return {}


def dynamic_max_tokens(
    text: str,
    *,
    ratio: float = DEFAULT_MAX_TOK_RATIO,
    floor: int = DEFAULT_MAX_TOK_FLOOR,
) -> int:
    """Cap generation length from input word count (stops early on EOS anyway)."""
    input_words = len(text.split())
    return max(floor, int(input_words * ratio))


# Small reviewed operational phrase set, rather than a confidence-only expansion.
_OPERATIONAL_PHRASES = {
    "en": {
        "good morning",
        "good afternoon",
        "good evening",
        "thank you",
        "please sit down",
        "you may be seated",
        "would you please take your seats",
        "please turn to the next page",
    },
    "es": {
        "buenos días",
        "buenas tardes",
        "buenas noches",
        "gracias",
        "muchas gracias",
        "pueden sentarse",
        "por favor tomen asiento",
        "por favor pasen a la siguiente página",
    },
}


def conservative_marian_route(text: str, source_lang: str, confidence: float | None) -> bool:
    """Opt-in bilingual routing; unknown confidence and non-allowlisted text use Gemma."""
    import math
    import re

    if confidence is None or not math.isfinite(confidence) or confidence < 0.8:
        return False
    normalized = re.sub(r"[^\w\s]", "", text.casefold())
    normalized = " ".join(normalized.split())
    return normalized in _OPERATIONAL_PHRASES.get(source_lang, set())
