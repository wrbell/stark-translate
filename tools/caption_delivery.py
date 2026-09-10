"""Bounded per-client caption writers, isolated from inference admission."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


def valid_utterance_id(value):
    return isinstance(value, int) and not isinstance(value, bool) and 0 < value <= 2**53 - 1


def partial_utterance_id(message):
    uid = message.get("utterance_id")
    return message.get("chunk_id") if uid is None else uid


class FinalCaptionHistory:
    """Bound exact published identities; expired older previews stay obsolete."""

    def __init__(self, capacity=128):
        self.capacity = capacity
        self.ids: set[int] = set()
        self.retired_through = 0

    def observe(self, utterance_id):
        if not valid_utterance_id(utterance_id) or utterance_id <= self.retired_through:
            return
        self.ids.add(utterance_id)
        while len(self.ids) > self.capacity:
            oldest = min(self.ids)
            self.ids.remove(oldest)
            self.retired_through = max(self.retired_through, oldest)

    def blocks(self, utterance_id):
        return valid_utterance_id(utterance_id) and (utterance_id <= self.retired_through or utterance_id in self.ids)


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
    finals: dict[str, FinalCaptionHistory] = field(default_factory=dict)


class CaptionDelivery:
    def __init__(
        self,
        *,
        capacity: int = 32,
        send_timeout: float = 2.0,
        before_send: Callable | None = None,
        on_failure: Callable | None = None,
        on_event: Callable | None = None,
        allow_message: Callable | None = None,
    ):
        if capacity < 1 or send_timeout <= 0:
            raise ValueError("Caption capacity and timeout must be positive")
        self.capacity, self.send_timeout = capacity, send_timeout
        self.before_send = before_send or (lambda *args: None)
        self.on_failure = on_failure or (lambda *args: None)
        self.on_event = on_event or (lambda *args: None)
        self.allow_message = allow_message or (lambda message: True)
        self._clients: dict[Any, _Client] = {}

    def add(self, client):
        if client in self._clients:
            return
        state = _Client()
        self._clients[client] = state
        state.task = asyncio.create_task(self._write(client, state))

    def publish(self, client, message: dict):
        if not self.allow_message(message):
            self.on_event("caption_discarded_partial_suppressed")
            return
        self.add(client)
        state = self._clients[client]
        if state.closing:
            return
        session = message.get("session_id")
        if (
            message.get("type") == "translation"
            and message.get("stage", "complete") == "complete"
            and isinstance(session, str)
            and session
            and valid_utterance_id(message.get("utterance_id"))
        ):
            history = state.finals.setdefault(session, FinalCaptionHistory())
            history.observe(message["utterance_id"])
            # A delivery instance belongs to one pipeline; retain a small
            # allowance for callers that explicitly change session namespaces.
            while len(state.finals) > 8:
                state.finals.pop(next(iter(state.finals)))
            state.pending = deque(item for item in state.pending if not self._finalized_partial(state, item[0]))
        if self._finalized_partial(state, message):
            self.on_event("caption_finalized_partial_suppressed")
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

    @staticmethod
    def _finalized_partial(state, message):
        session = message.get("session_id")
        history = state.finals.get(session) if isinstance(session, str) else None
        return (
            history is not None
            and message.get("type") == "translation"
            and message.get("stage") == "partial"
            and history.blocks(partial_utterance_id(message))
        )

    def discard_utterance(self, session_id, utterance_id):
        """Remove only queued partials for this exact session/utterance.

        A send already in progress cannot be recalled; consumers receive the
        subsequent discard event and suppress any late matching partial too.
        """
        for state in self._clients.values():
            before = len(state.pending)
            state.pending = deque(
                item
                for item in state.pending
                if not (
                    item[0].get("type") == "translation"
                    and item[0].get("stage") == "partial"
                    and item[0].get("session_id") == session_id
                    and item[0].get("utterance_id", item[0].get("chunk_id")) == utterance_id
                )
            )
            for _ in range(before - len(state.pending)):
                self.on_event("caption_discarded_partial_removed")

    async def _write(self, client, state):
        try:
            while True:
                await state.ready.wait()
                while state.pending:
                    message, queued, _ = state.pending.popleft()
                    if not self.allow_message(message) or self._finalized_partial(state, message):
                        self.on_event("caption_discarded_partial_suppressed")
                        continue
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
