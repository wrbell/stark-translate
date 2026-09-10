"""Discard only the identified interim caption, including queued delivery."""

import asyncio
import json
from unittest.mock import AsyncMock, Mock

from tools.caption_delivery import CaptionDelivery
from tools.pipeline_health import PipelineHealth


def partial(session, uid):
    return {"type": "translation", "stage": "partial", "session_id": session, "utterance_id": uid, "chunk_id": uid}


def test_pending_discard_removes_only_matching_partial_and_rechecks_before_send():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        sent = []
        invalid = set()

        async def send(raw):
            message = json.loads(raw)
            if not sent:
                entered.set()
                await release.wait()
            sent.append(message)

        client = Mock(send=send, close=AsyncMock())
        delivery = CaptionDelivery(
            allow_message=lambda m: (
                m.get("stage") != "partial" or (m.get("session_id"), m.get("utterance_id")) not in invalid
            )
        )
        first = {"type": "translation", "stage": "complete", "session_id": "s", "chunk_id": 1}
        delivery.publish(client, first)
        await entered.wait()
        delivery.publish(client, partial("s", 7))
        delivery.publish(client, partial("other-session", 7))
        final = {"type": "translation", "stage": "complete", "session_id": "s", "chunk_id": 7}
        delivery.publish(client, final)
        invalid.add(("s", 7))
        delivery.discard_utterance("s", 7)
        event = {"type": "utterance_discarded", "session_id": "s", "utterance_id": 7}
        delivery.publish(client, event)
        delivery.publish(client, partial("s", 7))  # delayed publisher cannot requeue discarded work
        delivery.publish(client, partial("s", 8))
        invalid.add(("s", 8))  # writer guard checks again even without explicit queue pruning
        release.set()
        await delivery.close()
        assert sent == [first, partial("other-session", 7), final, event]

    asyncio.run(run())


def test_health_snapshot_discard_is_session_and_partial_specific(tmp_path):
    health = PipelineHealth(tmp_path, "s")
    health.caption(partial("s", 7))
    final = {"type": "translation", "stage": "complete", "session_id": "s", "chunk_id": 7}
    health.caption(final)
    health.caption(partial("s", 8))
    health.discard_utterance("other", 7)
    assert len(health.snapshot()["captions"]) == 3
    health.discard_utterance("s", 7)
    assert [(r["stage"], r["chunk_id"]) for r in health.snapshot()["captions"]] == [("complete", 7), ("partial", 8)]


def test_delayed_broadcast_cannot_restore_discarded_health_snapshot(tmp_path, monkeypatch):
    import dry_run_ab as d

    health = PipelineHealth(tmp_path, "s")
    health.caption(partial("s", 7))
    monkeypatch.setattr(d, "SESSION_ID", "s")
    monkeypatch.setattr(d, "_health", health)
    monkeypatch.setattr(d, "_caption_delivery", None)
    monkeypatch.setattr(d, "_stt_scheduler", None)
    monkeypatch.setattr(d, "_discarded_utterance_id", 0)
    monkeypatch.setattr(d, "ws_clients", set())
    event = d._discard_utterance(7, "short_silence", 0.64, {})
    assert event["utterance_id"] == 7
    asyncio.run(d.broadcast(partial("s", 7)))
    assert health.snapshot()["captions"] == []
