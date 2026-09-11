"""engines -- Backend-agnostic STT and translation inference.

Provides a factory pattern for loading models on MLX (Apple Silicon)
or CUDA (NVIDIA) backends with a unified API.
"""

__all__ = ["create_stt_engine", "create_translation_engine", "create_tts_engine"]


def __getattr__(name):
    # Keep standard-library engine helpers importable without NumPy or ML APIs.
    if name in __all__:
        from engines import factory

        value = getattr(factory, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
