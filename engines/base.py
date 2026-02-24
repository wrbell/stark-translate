"""Abstract base classes for STT and translation engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class STTResult:
    """Result from speech-to-text inference."""

    text: str
    latency_ms: float
    confidence: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    segments: list | None = field(default_factory=list)
    low_confidence_words: list | None = field(default_factory=list)
    used_fallback: bool = False


@dataclass
class TranslationResult:
    """Result from translation inference."""

    text: str
    latency_ms: float
    tokens_per_second: float = 0.0
    qe_score: float | None = None


class STTEngine(ABC):
    """Abstract STT engine interface."""

    @abstractmethod
    def load(self) -> None:
        """Load model into memory."""
        ...

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str = "en",
        initial_prompt: str | None = None,
        word_timestamps: bool = False,
    ) -> STTResult:
        """Transcribe audio to text."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model from memory."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the model identifier."""
        ...

    @property
    @abstractmethod
    def backend(self) -> str:
        """Return backend name ('mlx', 'cuda', 'cpu')."""
        ...


class TranslationEngine(ABC):
    """Abstract translation engine interface."""

    @abstractmethod
    def load(self) -> None:
        """Load model into memory."""
        ...

    @abstractmethod
    def translate(self, text: str, *, source_lang: str = "en", target_lang: str = "es") -> TranslationResult:
        """Translate text between languages."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model from memory."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the model identifier."""
        ...

    @property
    @abstractmethod
    def backend(self) -> str:
        """Return backend name ('mlx', 'cuda', 'cpu')."""
        ...
