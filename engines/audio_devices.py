"""Lazy output-device discovery and resilient local TTS routing."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypedDict

logger = logging.getLogger(__name__)
DeviceSpec = int | str | None


class OutputDevice(TypedDict):
    index: int
    name: str
    channels: int
    default: bool


def list_output_devices() -> list[OutputDevice]:
    """Return output-capable devices without opening an audio stream."""
    import sounddevice as sd

    devices = sd.query_devices()
    default = sd.default.device[1]
    return [
        OutputDevice(
            index=index,
            name=str(device["name"]),
            channels=int(device["max_output_channels"]),
            default=index == default,
        )
        for index, device in enumerate(devices)
        if device.get("max_output_channels", 0) > 0
    ]


def _match_output_device(spec: int | str, devices: list[OutputDevice]) -> OutputDevice:
    for device in devices:
        if isinstance(spec, int):
            if device["index"] == spec:
                return device
        elif spec.strip() and spec.casefold() in device["name"].casefold():
            return device
    raise ValueError(f"No output device matches {spec!r}")


def resolve_output_device(spec: DeviceSpec) -> int | None:
    """Validate an index, match the first name substring, or use the default."""
    if spec is None:
        return None
    return _match_output_device(spec, list_output_devices())["index"]


class OutputDeviceResolver:
    """Cache routes on the TTS worker; discard stale indices on playback errors."""

    def __init__(self) -> None:
        self._cache: dict[int | str, OutputDevice] = {}
        self._logged_languages: set[str] = set()

    def resolve(self, spec: DeviceSpec) -> int | None:
        if spec is None:
            return None
        if spec not in self._cache:
            self._cache[spec] = _match_output_device(spec, list_output_devices())
        return self._cache[spec]["index"]

    def invalidate(self) -> None:
        """Renumbering can affect every cached language, not just the failed one."""
        self._cache.clear()

    def play(
        self,
        play: Callable[..., None],
        audio: Any,
        sample_rate: int,
        *,
        language: str,
        spec: DeviceSpec,
    ) -> None:
        """Retry a PortAudio failure once, then try the system default safely."""
        import sounddevice as sd

        for attempt in range(2):
            try:
                device = self.resolve(spec)
                play(audio, sample_rate, device=device)
                if language not in self._logged_languages:
                    name = self._cache[spec]["name"] if spec is not None else "system default"
                    logger.info("TTS %s output: %s (device=%s)", language, name, device)
                    self._logged_languages.add(language)
                return
            except sd.PortAudioError as exc:
                self.invalidate()
                logger.warning("TTS %s output %r failed: %s", language, spec, exc)
                if attempt == 0:
                    continue
            except Exception as exc:
                logger.warning("TTS %s output %r unavailable: %s", language, spec, exc)
            break

        logger.warning("TTS %s falling back to system default output", language)
        try:
            play(audio, sample_rate, device=None)
        except Exception as exc:
            logger.warning("TTS %s default output failed; skipping playback: %s", language, exc)
