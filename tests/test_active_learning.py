"""Tests for engines/active_learning.py — JSONL fallback event logging."""

import json

import pytest

from engines.active_learning import log_fallback_event
from engines.base import STTResult


@pytest.fixture
def original_result():
    return STTResult(
        text="For God so loved the word",
        latency_ms=120.0,
        confidence=0.45,
        avg_logprob=-1.5,
        compression_ratio=2.1,
    )


@pytest.fixture
def retry_result():
    return STTResult(
        text="For God so loved the world",
        latency_ms=250.0,
        confidence=0.92,
        avg_logprob=-0.25,
        compression_ratio=1.4,
    )


class TestLogFallbackEvent:
    def test_creates_file(self, tmp_path, original_result, retry_result):
        output = tmp_path / "events.jsonl"
        log_fallback_event(original_result, retry_result, "hash123", output_path=str(output))

        assert output.exists()
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_record_fields(self, tmp_path, original_result, retry_result):
        output = tmp_path / "events.jsonl"
        log_fallback_event(
            original_result,
            retry_result,
            "hash456",
            output_path=str(output),
            primary_model="whisper-turbo",
            fallback_model="whisper-large",
        )

        record = json.loads(output.read_text().strip())
        assert record["audio_hash"] == "hash456"
        assert record["primary_model"] == "whisper-turbo"
        assert record["fallback_model"] == "whisper-large"
        assert record["original_text"] == "For God so loved the word"
        assert record["retry_text"] == "For God so loved the world"
        assert record["which_was_chosen"] == "retry"
        assert "timestamp" in record

    def test_append_behavior(self, tmp_path, original_result, retry_result):
        output = tmp_path / "events.jsonl"
        log_fallback_event(original_result, retry_result, "hash1", output_path=str(output))
        log_fallback_event(original_result, retry_result, "hash2", output_path=str(output))

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["audio_hash"] == "hash1"
        assert json.loads(lines[1])["audio_hash"] == "hash2"

    def test_creates_parent_directories(self, tmp_path, original_result, retry_result):
        output = tmp_path / "nested" / "dir" / "events.jsonl"
        log_fallback_event(original_result, retry_result, "hash789", output_path=str(output))
        assert output.exists()

    def test_chosen_field(self, tmp_path, original_result, retry_result):
        output = tmp_path / "events.jsonl"
        log_fallback_event(original_result, retry_result, "hash_orig", output_path=str(output), chosen="original")

        record = json.loads(output.read_text().strip())
        assert record["which_was_chosen"] == "original"

    def test_latency_rounded(self, tmp_path, original_result, retry_result):
        output = tmp_path / "events.jsonl"
        log_fallback_event(original_result, retry_result, "hash_lat", output_path=str(output))

        record = json.loads(output.read_text().strip())
        assert record["original_latency_ms"] == 120.0
        assert record["retry_latency_ms"] == 250.0
