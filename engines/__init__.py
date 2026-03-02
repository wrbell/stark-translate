"""engines -- Backend-agnostic STT and translation inference.

Provides a factory pattern for loading models on MLX (Apple Silicon)
or CUDA (NVIDIA) backends with a unified API.
"""

from engines.factory import create_stt_engine, create_translation_engine, create_tts_engine

__all__ = ["create_stt_engine", "create_translation_engine", "create_tts_engine"]
