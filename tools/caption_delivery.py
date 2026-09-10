"""Bounded per-client caption writers, isolated from inference admission."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


def _replace_key(message: dict) -> tuple | None:
    kind, stage = message.get("type"), message.get("stage")
    if kind == "translation_stream" or (kind == "translation" and stage == "partial"):
        return (message.get("session_id"), kind, message.get("utterance_id", message.get("chunk_id")))
    return None


@dataclass
class _Client:
    pending: deque = field(default_factory=deque)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    closing: bool = False


class CaptionDelivery:
    def __init__(
        self,
        *,
        capacity: int = 32,
        send_timeout: float = 2.0,
        before_send: Callable | None = None,
        on_failure: Callable | None = None,
        on_event: Callable | None = None,
    ):
        if capacity < 1 or send_timeout <= 0:
            raise ValueError("Caption capacity and timeout must be positive")
        self.capacity, self.send_timeout = capacity, send_timeout
        self.before_send = before_send or (lambda *args: None)
        self.on_failure = on_failure or (lambda *args: None)
        self.on_event = on_event or (lambda *args: None)
        self._clients: dict[Any, _Client] = {}

    def add(self, client):
        if client in self._clients:
            return
        state = _Client()
        self._clients[client] = state
        state.task = asyncio.create_task(self._write(client, state))

    def publish(self, client, message: dict):
        self.add(client)
        state = self._clients[client]
        if state.closing:
            return
        key = _replace_key(message)
        # Only coalesce within the suffix after the last reliable event. An
        # update must not jump across a final or a language/session boundary.
        if key is not None:
            for index in range(len(state.pending) - 1, -1, -1):
                previous, _, previous_key = state.pending[index]
                if previous_key is None:
                    break
                if previous_key == key:
                    del state.pending[index]
                    self.on_event("caption_coalesced")
                    break
        if len(state.pending) >= self.capacity:
            discard = next((i for i, item in enumerate(state.pending) if item[2] is not None), None)
            if discard is not None:
                del state.pending[discard]
                self.on_event("caption_partial_evicted")
            elif key is not None:
                self.on_event("caption_partial_evicted")
                return
            else:
                state.closing = True
                state.pending.clear()
                state.ready.set()
                self.on_failure(client, RuntimeError("Caption client exhausted reliable queue capacity"))
                self.on_event("caption_client_overflow")
                return
        state.pending.append((dict(message), time.perf_counter(), key))
        state.ready.set()

    async def _write(self, client, state):
        try:
            while True:
                await state.ready.wait()
                while state.pending:
                    message, queued, _ = state.pending.popleft()
                    started = time.perf_counter()
                    self.before_send(client, message, started, (started - queued) * 1000)
                    await asyncio.wait_for(client.send(json.dumps(message)), self.send_timeout)
                    self.on_event("caption_sent")
                state.ready.clear()
                if state.closing:
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.on_failure(client, exc)
            self.on_event("caption_client_failed")
        finally:
            self._clients.pop(client, None)
            try:
                await asyncio.wait_for(client.close(), self.send_timeout)
            except (Exception, asyncio.CancelledError):
                pass

    async def remove(self, client):
        state = self._clients.pop(client, None)
        if state is not None and state.task is not None:
            state.task.cancel()
            await asyncio.gather(state.task, return_exceptions=True)
            try:
                await asyncio.wait_for(client.close(), self.send_timeout)
            except Exception:
                pass

    async def close(self, drain_timeout: float = 2.0):
        tasks = []
        for state in list(self._clients.values()):
            state.closing = True
            state.ready.set()
            tasks.append(state.task)
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=drain_timeout)
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
