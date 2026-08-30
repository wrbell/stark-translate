"""Shared translation prompt + cleanup helpers for MLX / CUDA / llama.cpp.

Keeps Gemma 4 instruct and TranslateGemma structured prompts identical across
backends so Mac and CUDA production paths stay semantically aligned.
"""

from __future__ import annotations

from typing import Any

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
        f"Translate the following {src_name} text to {tgt_name}. "
        f"Output only the translation, nothing else.\n\n{text}"
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
) -> list[dict[str, Any]]:
    """Dispatch to Gemma 4 or TranslateGemma message builders."""
    if model_family == "gemma4":
        return gemma4_chat_messages(text, source_lang, target_lang)
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
    """Strip EOS markers and Gemma 4 preamble from a raw generation string."""
    clean = raw.split("<end_of_turn>")[0].strip()
    if model_family == "gemma4":
        for prefix in GEMMA4_PREAMBLES:
            if clean.startswith(prefix):
                clean = clean[len(prefix) :].strip()
                break
    return clean


def dynamic_max_tokens(
    text: str,
    *,
    ratio: float = DEFAULT_MAX_TOK_RATIO,
    floor: int = DEFAULT_MAX_TOK_FLOOR,
) -> int:
    """Cap generation length from input word count (stops early on EOS anyway)."""
    input_words = len(text.split())
    return max(floor, int(input_words * ratio))
