"""Optional benchmark barrier; no file producer or capture clock runs here."""

from __future__ import annotations

import asyncio
import math
import time


def validate_replay_client_wait(seconds: float, source: str) -> None:
    if not math.isfinite(seconds) or not 0 <= seconds <= 60:
        raise ValueError("--replay-wait-client-seconds must be finite and between 0 and 60")
    if seconds and source != "file":
        raise ValueError("--replay-wait-client-seconds requires a file audio source")


class ReplayClientBarrier:
    """Wait for any caption socket, without claiming that its page is visible."""

    def __init__(self, seconds: float = 0, *, clock=time.perf_counter, sleep=asyncio.sleep):
        validate_replay_client_wait(seconds, "file")
        self.seconds = seconds
        self._clock, self._sleep = clock, sleep
        self._started = None
        self._elapsed = 0.0
        self._clients = 0
        self._status = "pending" if seconds else "disabled"

    def snapshot(self) -> dict:
        elapsed = self._clock() - self._started if self._status == "waiting" else self._elapsed
        return {
            "requested_seconds": self.seconds,
            "actual_wait_seconds": max(0.0, elapsed),
            "status": self._status,
            "connected_clients": self._clients,
            "visibility_confirmed": False,
        }

    async def wait(self, client_count) -> None:
        if not self.seconds:
            return
        started = self._clock()
        self._started = started
        self._status = "waiting"
        try:
            while True:
                self._clients = client_count()
                if self._clients:
                    self._status = "connected"
                    return
                remaining = started + self.seconds - self._clock()
                if remaining <= 0:
                    self._status = "timed_out"
                    raise TimeoutError(
                        f"No caption client connected within {self.seconds:g}s; replay audio was not started"
                    )
                await self._sleep(min(0.05, remaining))
        except asyncio.CancelledError:
            self._status = "cancelled"
            raise
        finally:
            self._elapsed = self._clock() - started
