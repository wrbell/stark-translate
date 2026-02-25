"""Shared fixtures for the test suite.

All tests run on Ubuntu CI without GPU, torch, mlx, or model downloads.
Heavy dependencies are mocked at the module level.
"""

import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock heavy optional dependencies before any project imports
# ---------------------------------------------------------------------------

_MOCK_MODULES = [
    "mlx",
    "mlx.core",
    "mlx_lm",
    "mlx_lm.utils",
    "mlx_whisper",
    "torch",
    "torch.cuda",
    "transformers",
    "faster_whisper",
    "ctranslate2",
    "sounddevice",
    "websockets",
    "pyannote.audio",
    "silero_vad",
    "noisereduce",
    "demucs",
    "scipy",
    "scipy.io",
    "scipy.io.wavfile",
    "jiwer",
    "youtube_transcript_api",
    "bitsandbytes",
    "yt_dlp",
    "language_tool_python",
    "sentence_transformers",
    "streamlink",
    "soundfile",
]


@pytest.fixture(autouse=True)
def _mock_heavy_deps(monkeypatch):
    """Auto-mock heavy ML dependencies that aren't installed on CI."""
    for mod_name in _MOCK_MODULES:
        if mod_name not in sys.modules:
            monkeypatch.setitem(sys.modules, mod_name, MagicMock())


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_stt_result():
    """Create a sample STTResult for testing."""
    from engines.base import STTResult

    return STTResult(
        text="For God so loved the world",
        latency_ms=150.5,
        confidence=0.92,
        avg_logprob=-0.25,
        compression_ratio=1.4,
    )


@pytest.fixture
def sample_stt_result_low_conf():
    """Create a low-confidence STTResult for testing fallback scenarios."""
    from engines.base import STTResult

    return STTResult(
        text="For God so loved the word",
        latency_ms=180.3,
        confidence=0.45,
        avg_logprob=-1.5,
        compression_ratio=2.8,
    )


@pytest.fixture
def sample_translation_result():
    """Create a sample TranslationResult for testing."""
    from engines.base import TranslationResult

    return TranslationResult(
        text="Porque de tal manera amó Dios al mundo",
        latency_ms=650.0,
        tokens_per_second=45.2,
        qe_score=0.88,
    )
