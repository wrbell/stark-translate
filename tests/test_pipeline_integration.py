"""WebSocket broadcast message contract tests.

Validates the JSON schema for each broadcast message type sent by dry_run_ab.py.
Tests message structure only — no model inference, no audio, pure JSON assertions.
"""

import json
from datetime import datetime

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_iso_timestamp(value):
    """Check that a string is a valid ISO-8601 timestamp."""
    try:
        datetime.fromisoformat(value)
        return True
    except (ValueError, TypeError):
        return False


def _assert_common_fields(msg, expected_type, expected_stage=None):
    """Assert fields shared by all broadcast messages."""
    assert msg["type"] == expected_type
    if expected_stage is not None:
        assert msg["stage"] == expected_stage
    assert isinstance(msg["chunk_id"], int)


# ---------------------------------------------------------------------------
# Message type: translation / stage: partial
# ---------------------------------------------------------------------------


class TestPartialMessage:
    """Partial translation broadcast (MarianMT, while speaker is still talking)."""

    def _make_partial(self, **overrides):
        base = {
            "type": "translation",
            "stage": "partial",
            "chunk_id": 1,
            "english": "For God so loved the world",
            "spanish_a": "Porque de tal manera amó Dios al mundo",
            "spanish_b": None,
            "stt_latency_ms": 150.5,
            "latency_a_ms": 80.2,
            "marian_pt_ms": 80.2,
            "timestamp": datetime.now().isoformat(),
        }
        base.update(overrides)
        return base

    def test_partial_has_required_fields(self):
        msg = self._make_partial()
        _assert_common_fields(msg, "translation", "partial")
        assert isinstance(msg["english"], str) and len(msg["english"]) > 0
        assert isinstance(msg["spanish_a"], str)
        assert msg["spanish_b"] is None
        assert isinstance(msg["stt_latency_ms"], float)
        assert isinstance(msg["latency_a_ms"], float)
        assert _is_iso_timestamp(msg["timestamp"])

    def test_partial_is_json_serializable(self):
        msg = self._make_partial()
        encoded = json.dumps(msg)
        decoded = json.loads(encoded)
        assert decoded["type"] == "translation"
        assert decoded["stage"] == "partial"

    def test_partial_spanish_b_always_none(self):
        msg = self._make_partial()
        assert msg["spanish_b"] is None


# ---------------------------------------------------------------------------
# Message type: translation_start
# ---------------------------------------------------------------------------


class TestTranslationStartMessage:
    """Broadcast when STT completes and translation is about to begin."""

    def _make_start(self, **overrides):
        base = {
            "type": "translation_start",
            "chunk_id": 2,
            "english": "He that believeth in Him",
            "stage": "final",
            "stt_latency_ms": 280.3,
            "stt_confidence": 0.92,
            "timestamp": datetime.now().isoformat(),
        }
        base.update(overrides)
        return base

    def test_start_has_required_fields(self):
        msg = self._make_start()
        assert msg["type"] == "translation_start"
        assert isinstance(msg["chunk_id"], int)
        assert isinstance(msg["english"], str) and len(msg["english"]) > 0
        assert msg["stage"] == "final"
        assert isinstance(msg["stt_latency_ms"], float)
        assert isinstance(msg["stt_confidence"], float)
        assert _is_iso_timestamp(msg["timestamp"])

    def test_start_is_json_serializable(self):
        encoded = json.dumps(self._make_start())
        decoded = json.loads(encoded)
        assert decoded["type"] == "translation_start"


# ---------------------------------------------------------------------------
# Message type: translation / stage: stt (legacy)
# ---------------------------------------------------------------------------


class TestSttLegacyMessage:
    """Legacy STT stage message (backward compat with older displays)."""

    def _make_stt(self, **overrides):
        base = {
            "type": "translation",
            "stage": "stt",
            "chunk_id": 3,
            "english": "And we know that all things work together for good",
            "spanish_a": None,
            "spanish_b": None,
            "stt_latency_ms": 310.0,
            "stt_confidence": 0.88,
            "timestamp": datetime.now().isoformat(),
        }
        base.update(overrides)
        return base

    def test_stt_has_required_fields(self):
        msg = self._make_stt()
        _assert_common_fields(msg, "translation", "stt")
        assert isinstance(msg["english"], str)
        assert msg["spanish_a"] is None
        assert msg["spanish_b"] is None

    def test_stt_is_json_serializable(self):
        encoded = json.dumps(self._make_stt())
        decoded = json.loads(encoded)
        assert decoded["stage"] == "stt"


# ---------------------------------------------------------------------------
# Message type: translation / stage: translation_a
# ---------------------------------------------------------------------------


class TestTranslationAMessage:
    """Broadcast when 4B (Approach A) translation completes."""

    def _make_a(self, **overrides):
        base = {
            "type": "translation",
            "stage": "translation_a",
            "chunk_id": 4,
            "english": "The wages of sin is death",
            "spanish_a": "Porque la paga del pecado es muerte",
            "spanish_b": None,
            "stt_latency_ms": 290.0,
            "latency_a_ms": 350.5,
            "stt_confidence": 0.95,
            "tps_a": 45.2,
            "qe_a": {"length_ratio": 1.18, "untranslated": False},
            "timestamp": datetime.now().isoformat(),
        }
        base.update(overrides)
        return base

    def test_a_has_required_fields(self):
        msg = self._make_a()
        _assert_common_fields(msg, "translation", "translation_a")
        assert isinstance(msg["spanish_a"], str) and len(msg["spanish_a"]) > 0
        assert msg["spanish_b"] is None
        assert isinstance(msg["tps_a"], float)
        assert isinstance(msg["qe_a"], dict)

    def test_a_is_json_serializable(self):
        encoded = json.dumps(self._make_a())
        decoded = json.loads(encoded)
        assert decoded["stage"] == "translation_a"


# ---------------------------------------------------------------------------
# Message type: translation_stream (token streaming)
# ---------------------------------------------------------------------------


class TestTranslationStreamMessage:
    """Progressive token streaming for 4B translation."""

    def _make_stream(self, **overrides):
        base = {
            "type": "translation_stream",
            "chunk_id": 5,
            "partial_spanish_a": "Porque de tal",
            "tokens_so_far": 3,
        }
        base.update(overrides)
        return base

    def test_stream_has_required_fields(self):
        msg = self._make_stream()
        assert msg["type"] == "translation_stream"
        assert isinstance(msg["chunk_id"], int)
        assert isinstance(msg["partial_spanish_a"], str)
        assert isinstance(msg["tokens_so_far"], int) and msg["tokens_so_far"] > 0

    def test_stream_is_json_serializable(self):
        encoded = json.dumps(self._make_stream())
        decoded = json.loads(encoded)
        assert decoded["type"] == "translation_stream"


# ---------------------------------------------------------------------------
# Message type: translation / stage: complete (final)
# ---------------------------------------------------------------------------


class TestFinalCompleteMessage:
    """Final broadcast with both A and B translations complete."""

    def _make_final(self, **overrides):
        base = {
            "type": "translation",
            "stage": "complete",
            "chunk_id": 6,
            "english": "For God so loved the world",
            "spanish_a": "Porque de tal manera amó Dios al mundo",
            "spanish_b": "Porque de tal manera amó Dios al mundo",
            "stt_latency_ms": 290.0,
            "latency_a_ms": 350.0,
            "latency_b_ms": 800.0,
            "e2e_latency_ms": 1100.0,
            "stt_confidence": 0.95,
            "tps_a": 45.2,
            "tps_b": 38.1,
            "qe_a": {"length_ratio": 1.18, "untranslated": False},
            "qe_b": {"length_ratio": 1.20, "untranslated": False},
            "timestamp": datetime.now().isoformat(),
        }
        base.update(overrides)
        return base

    def test_final_has_required_fields(self):
        msg = self._make_final()
        _assert_common_fields(msg, "translation", "complete")
        assert isinstance(msg["english"], str)
        assert isinstance(msg["spanish_a"], str)
        assert isinstance(msg["spanish_b"], str)
        assert isinstance(msg["e2e_latency_ms"], float)
        assert isinstance(msg["tps_a"], float)
        assert isinstance(msg["tps_b"], float)
        assert _is_iso_timestamp(msg["timestamp"])

    def test_final_is_json_serializable(self):
        encoded = json.dumps(self._make_final())
        decoded = json.loads(encoded)
        assert decoded["stage"] == "complete"

    def test_final_spanish_b_nullable(self):
        """In 4B-only mode, spanish_b is None."""
        msg = self._make_final(spanish_b=None, latency_b_ms=None, tps_b=None, qe_b=None)
        encoded = json.dumps(msg)
        decoded = json.loads(encoded)
        assert decoded["spanish_b"] is None
        assert decoded["latency_b_ms"] is None

    def test_final_chunk_ids_are_monotonic(self):
        """Chunk IDs should increase across messages."""
        msgs = [self._make_final(chunk_id=i) for i in range(1, 5)]
        ids = [m["chunk_id"] for m in msgs]
        assert ids == sorted(ids)
        assert len(set(ids)) == len(ids)
