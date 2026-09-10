"""Versioned preview helpers. None of these hypotheses is a final transcript."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TranslationCandidate:
    identity: tuple
    result: Any
    pause_epoch: int
    sample_bounds: dict

    def confirmed(self, identity: tuple, pause_epoch: int) -> bool:
        return self.identity == identity and self.pause_epoch == pause_epoch


def common_prefix_words(first: str, second: str) -> str:
    """Exact agreement is a visual stability proxy, never a quality score."""
    left, right = first.split(), second.split()
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return " ".join(right[:count])


class RollingPreview:
    """Join overlapping rolling hypotheses conservatively for previews only.

    A tail without an exact multiword overlap is displayed as a new uncertain
    tail, rather than guessed into the prior prefix. Final STT always sees the
    complete original audio and does not consume this text.
    """

    def __init__(self):
        self.text = ""
        self.window_start = 0

    def update(self, text: str, window_start: int) -> tuple[str, bool]:
        if not self.text or window_start == 0:
            self.text, self.window_start = text, window_start
            return text, True
        previous, current = self.text.split(), text.split()
        for length in range(min(len(previous), len(current), 24), 1, -1):
            # Prefix of the new window can occur before the previous trailing
            # words: those words are draft and are allowed to be revised.
            for start in range(max(0, len(previous) - 32), len(previous) - length + 1):
                if previous[start : start + length] == current[:length]:
                    self.text = " ".join(previous[:start] + current)
                    self.window_start = window_start
                    return self.text, True
        self.text, self.window_start = text, window_start
        return text, False
