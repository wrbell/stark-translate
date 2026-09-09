"""Unit tests for features/speaker_labels.py and rolling_buffer.py."""

from __future__ import annotations

import json

import numpy as np
import pytest


class TestNormalizeLabel:
    def test_pyannote_speaker_00(self):
        from features.speaker_labels import normalize_speaker_label

        assert normalize_speaker_label("SPEAKER_00") == "Speaker A"
        assert normalize_speaker_label("SPEAKER_01") == "Speaker B"

    def test_already_pretty(self):
        from features.speaker_labels import normalize_speaker_label

        assert normalize_speaker_label("Speaker A") == "Speaker A"
        assert normalize_speaker_label("Speaker B") == "Speaker B"

    def test_empty_defaults_to_a(self):
        from features.speaker_labels import normalize_speaker_label

        assert normalize_speaker_label("") == "Speaker A"
        assert normalize_speaker_label(None) == "Speaker A"


class TestOverlap:
    def test_partial_overlap(self):
        from features.speaker_labels import overlap_seconds

        assert overlap_seconds(0.0, 2.0, 1.0, 3.0) == pytest.approx(1.0)

    def test_no_overlap(self):
        from features.speaker_labels import overlap_seconds

        assert overlap_seconds(0.0, 1.0, 2.0, 3.0) == 0.0

    def test_point_inside_counts(self):
        from features.speaker_labels import overlap_seconds

        assert overlap_seconds(0.0, 2.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_point_outside_is_zero(self):
        from features.speaker_labels import overlap_seconds

        assert overlap_seconds(0.0, 1.0, 5.0, 5.0) == 0.0


class TestParseJsonl:
    def test_legacy_record_without_timestamps(self):
        from features.speaker_labels import parse_jsonl_record

        seg = parse_jsonl_record({"chunk_id": 1, "speaker": "Speaker A", "confidence": 0.9, "ts": 10.0})
        assert seg is not None
        assert seg.speaker == "Speaker A"
        assert seg.start_ts == 10.0
        assert seg.end_ts == 10.0
        assert seg.timestamp == 10.0

    def test_current_record_with_interval(self):
        from features.speaker_labels import parse_jsonl_record

        seg = parse_jsonl_record(
            {
                "chunk_id": 2,
                "speaker": "SPEAKER_01",
                "confidence": 0.8,
                "ts": 12.0,
                "start_ts": 10.0,
                "end_ts": 12.0,
            }
        )
        assert seg is not None
        assert seg.speaker == "Speaker B"
        assert seg.start_ts == 10.0
        assert seg.end_ts == 12.0

    def test_metadata_header_skipped(self):
        from features.speaker_labels import parse_jsonl_record

        assert parse_jsonl_record({"_metadata": {"session": "x"}}) is None

    def test_missing_speaker_skipped(self):
        from features.speaker_labels import parse_jsonl_record

        assert parse_jsonl_record({"chunk_id": 1, "ts": 1.0}) is None


class TestAssignSpeaker:
    def test_max_overlap_wins(self):
        from features.speaker_labels import SpeakerSegment, assign_speaker

        segs = [
            SpeakerSegment("Speaker A", start_ts=0.0, end_ts=2.0, confidence=1.0),
            SpeakerSegment("Speaker B", start_ts=1.5, end_ts=4.0, confidence=1.0),
        ]
        # [1.6, 4.0] overlaps B more
        assert assign_speaker(1.6, 4.0, segs) == "Speaker B"

    def test_legacy_point_inside_utterance(self):
        from features.speaker_labels import SpeakerSegment, assign_speaker

        segs = [SpeakerSegment("Speaker A", start_ts=1.0, end_ts=1.0, timestamp=1.0, confidence=0.5)]
        assert assign_speaker(0.0, 2.0, segs, carry_forward=False) == "Speaker A"

    def test_no_overlap_without_carry_forward(self):
        from features.speaker_labels import SpeakerSegment, assign_speaker

        segs = [SpeakerSegment("Speaker A", start_ts=0.0, end_ts=1.0)]
        assert assign_speaker(5.0, 6.0, segs, carry_forward=False) is None

    def test_carry_forward_uses_latest_prior_segment(self):
        from features.speaker_labels import SpeakerSegment, assign_speaker

        segs = [
            SpeakerSegment("Speaker A", start_ts=0.0, end_ts=1.0),
            SpeakerSegment("Speaker B", start_ts=2.0, end_ts=3.0),
        ]
        assert assign_speaker(10.0, 12.0, segs, carry_forward=True) == "Speaker B"

    def test_from_jsonl_file(self, tmp_path):
        from features.speaker_labels import assign_speaker_from_jsonl

        path = tmp_path / "diarize.jsonl"
        path.write_text(
            json.dumps({"chunk_id": 1, "speaker": "Speaker A", "confidence": 1.0, "start_ts": 0.0, "end_ts": 2.0})
            + "\n"
            + json.dumps({"chunk_id": 2, "speaker": "Speaker B", "confidence": 1.0, "start_ts": 2.0, "end_ts": 4.0})
            + "\n"
        )
        assert assign_speaker_from_jsonl(path, 2.1, 3.5) == "Speaker B"

    def test_missing_file(self, tmp_path):
        from features.speaker_labels import assign_speaker_from_jsonl

        assert assign_speaker_from_jsonl(tmp_path / "missing.jsonl", 0.0, 1.0) is None


class TestSpeakerFieldForResult:
    def test_omitted_when_disabled(self):
        from features.speaker_labels import speaker_field_for_result

        assert speaker_field_for_result(False, "Speaker A") == {}

    def test_present_when_enabled(self):
        from features.speaker_labels import speaker_field_for_result

        assert speaker_field_for_result(True, "Speaker A") == {"speaker": "Speaker A"}
        assert speaker_field_for_result(True, None) == {"speaker": None}


class TestUtteranceWallclock:
    def test_end_is_start_plus_duration(self):
        from features.speaker_labels import utterance_wallclock

        start, end = utterance_wallclock(100.0, 2.5, now_perf=103.0, now_wall=1000.0)
        # utterance began 3s ago in perf time → wall 997, end 999.5
        assert start == pytest.approx(997.0)
        assert end == pytest.approx(999.5)

    def test_none_start_uses_duration(self):
        from features.speaker_labels import utterance_wallclock

        start, end = utterance_wallclock(None, 2.0, now_perf=10.0, now_wall=50.0)
        assert start == pytest.approx(48.0)
        assert end == pytest.approx(50.0)


class TestOnlineCluster:
    def test_assigns_a_then_b(self):
        from features.speaker_labels import OnlineSpeakerCluster

        cluster = OnlineSpeakerCluster(threshold=0.65, max_speakers=4)
        a, conf_a = cluster.assign([1.0, 0.0])
        b, conf_b = cluster.assign([0.0, 1.0])
        assert a == "Speaker A"
        assert conf_a == pytest.approx(1.0)
        assert b == "Speaker B"
        assert conf_b == pytest.approx(1.0)
        assert cluster.assign([0.99, 0.01])[0] == "Speaker A"
        assert cluster.assign([0.01, 0.99])[0] == "Speaker B"

    def test_threshold_same_speaker(self):
        from features.speaker_labels import OnlineSpeakerCluster, cosine_similarity

        cluster = OnlineSpeakerCluster(threshold=0.6, max_speakers=4)
        cluster.assign([1.0, 0.0])
        # cosine([1,0], [0.5,0.5]) = 0.707 > 0.6 → same speaker
        assert cosine_similarity([1.0, 0.0], [0.5, 0.5]) == pytest.approx(0.7071, rel=1e-3)
        assert cluster.assign([0.5, 0.5])[0] == "Speaker A"

    def test_threshold_new_speaker(self):
        from features.speaker_labels import OnlineSpeakerCluster

        cluster = OnlineSpeakerCluster(threshold=0.8, max_speakers=4)
        cluster.assign([1.0, 0.0])
        # 0.707 < 0.8 → new speaker
        assert cluster.assign([0.5, 0.5])[0] == "Speaker B"

    def test_max_speakers_forces_nearest(self):
        from features.speaker_labels import OnlineSpeakerCluster

        cluster = OnlineSpeakerCluster(threshold=0.99, max_speakers=2)
        cluster.assign([1.0, 0.0, 0.0])
        cluster.assign([0.0, 1.0, 0.0])
        label, _ = cluster.assign([0.0, 0.0, 1.0])
        assert label in ("Speaker A", "Speaker B")


class TestLoadRecentCaptions:
    def test_reads_speaker_and_english(self, tmp_path):
        from features.speaker_labels import load_recent_captions

        path = tmp_path / "ab.csv"
        path.write_text("chunk_id,english,spanish_a,speaker\n1,Hello,Hola,Speaker A\n2,World,Mundo,Speaker B\n")
        caps = load_recent_captions(path, limit=8)
        assert len(caps) == 2
        assert caps[0]["speaker"] == "Speaker A"
        assert caps[0]["english"] == "Hello"
        assert caps[1]["speaker"] == "Speaker B"

    def test_legacy_csv_without_speaker(self, tmp_path):
        from features.speaker_labels import load_recent_captions

        path = tmp_path / "legacy.csv"
        path.write_text("chunk_id,english,spanish_a\n1,Hello,Hola\n")
        caps = load_recent_captions(path)
        assert caps[0]["speaker"] is None
        assert caps[0]["english"] == "Hello"

    def test_missing_file(self, tmp_path):
        from features.speaker_labels import load_recent_captions

        assert load_recent_captions(tmp_path / "nope.csv") == []


class TestRollingBuffer:
    def test_writes_rolling_wav_and_index(self, tmp_path):
        from features.rolling_buffer import RollingSpeechWindow

        window = RollingSpeechWindow(tmp_path, max_seconds=5.0, sample_rate=16000)
        a = np.zeros(16000, dtype=np.float32)
        b = np.ones(16000, dtype=np.float32) * 0.1
        window.append(a, 1, 100.0, 101.0)
        window.append(b, 2, 101.0, 102.0)
        assert window.rolling_wav.exists()
        assert window.meta_path.exists()
        lines = window.chunks_jsonl.read_text().strip().splitlines()
        assert len(lines) == 2
        rec = json.loads(lines[0])
        assert rec["chunk_id"] == 1
        assert rec["wav"] == "chunk_0001.wav"
        meta = json.loads(window.meta_path.read_text())
        assert meta["window_start_ts"] == 100.0
        assert len(meta["chunks"]) == 2

    def test_trims_to_max_seconds(self, tmp_path):
        from features.rolling_buffer import RollingSpeechWindow

        window = RollingSpeechWindow(tmp_path, max_seconds=2.5, sample_rate=16000)
        for i in range(5):
            audio = np.zeros(16000, dtype=np.float32)
            window.append(audio, i, float(i), float(i + 1))
        # 5 x 1s, cap 2.5s → keep last 3 (can't drop below covering max)
        total = sum(c["duration_s"] for c in window._chunks)
        assert total <= 3.0 + 1e-6
        assert len(window._chunks) <= 3
