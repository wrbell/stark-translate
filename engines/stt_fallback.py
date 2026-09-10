"""Language contract for automatic MLX Whisper fallback; no model imports.

The configured Distil-Whisper fallback is English-trained. Custom fallback IDs
do not implicitly establish support for another language. Until a multilingual
fallback has an explicit, evaluated capability contract, retries are EN-only.
"""


def require_mlx_fallback_language(language: str, model_id: str) -> None:
    """Fail before loading or transcribing with an unvalidated fallback."""
    if language != "en":
        raise RuntimeError(
            f"Automatic MLX Whisper fallback {model_id!r} is validated only for English; "
            f"refusing fallback for language={language!r}. Restore the selected multilingual "
            "primary Whisper model and retry; changing the fallback ID does not enable other languages."
        )
