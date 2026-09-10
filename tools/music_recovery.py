"""Bounded, source-contiguous speech staging during an existing music hold.

This preserves samples after a caller accepts recovery. It makes no music or
speech-classification decision and never reconstructs missing source frames.
"""

from __future__ import annotations

from tools.pipeline_timing import AudioFrame, AudioTimeline


class MusicSpeechRecovery:
    def __init__(self, capacity: int):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Music recovery capacity must be a positive frame count")
        self.capacity = capacity
        self._frames: list[AudioFrame] = []

    def __len__(self):
        return len(self._frames)

    def contiguous(self, frame: AudioFrame) -> bool:
        if not self._frames:
            return True
        previous = self._frames[-1].stamp
        return (
            previous.sample_end is not None
            and previous.sample_end == frame.stamp.sample_start
            and previous.sample_rate == frame.stamp.sample_rate
            and previous.source == frame.stamp.source
            and not previous.padding_samples
        )

    def append(self, frame: AudioFrame):
        if len(self._frames) >= self.capacity:
            raise RuntimeError("Music recovery must be resolved at its frame limit")
        if not self.contiguous(frame):
            raise ValueError("Music recovery frames must have contiguous source positions")
        self._frames.append(frame)

    def take(self) -> tuple[AudioFrame, ...]:
        result = tuple(self._frames)
        self._frames.clear()
        return result

    @staticmethod
    def timeline(frames: tuple[AudioFrame, ...]) -> AudioTimeline:
        result = AudioTimeline()
        for frame in frames:
            result.append(len(frame.samples), frame.stamp, True)
        return result
