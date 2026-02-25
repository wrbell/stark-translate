"""Tests for pure utility functions in features/diarize.py."""

import json

# ===================================================================
# _format_timestamp (diarize version — includes milliseconds)
# ===================================================================


class TestFormatTimestamp:
    def test_zero(self):
        from features.diarize import _format_timestamp

        assert _format_timestamp(0) == "00:00:00.000"

    def test_90_seconds(self):
        from features.diarize import _format_timestamp

        result = _format_timestamp(90.123)
        assert result == "00:01:30.123"

    def test_over_one_hour(self):
        from features.diarize import _format_timestamp

        result = _format_timestamp(3661.5)
        assert result == "01:01:01.500"


# ===================================================================
# write_jsonl
# ===================================================================


class TestWriteJsonl:
    def test_basic_write(self, tmp_path):
        from features.diarize import write_jsonl

        output = str(tmp_path / "out.jsonl")
        results = [
            {"speaker": "A", "start": 0.0, "end": 5.0, "duration": 5.0, "text": "Hello", "confidence": 0.9},
        ]
        write_jsonl(results, output)
        with open(output) as f:
            lines = f.read().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["speaker"] == "A"

    def test_metadata_header(self, tmp_path):
        from features.diarize import write_jsonl

        output = str(tmp_path / "out.jsonl")
        results = [
            {"speaker": "A", "start": 0.0, "end": 5.0, "duration": 5.0, "text": "Hello", "confidence": 0.9},
        ]
        metadata = {"source": "test", "duration": 60.0}
        write_jsonl(results, output, metadata=metadata)
        with open(output) as f:
            lines = f.read().strip().split("\n")
        assert len(lines) == 2
        header = json.loads(lines[0])
        assert "_metadata" in header
        assert header["_metadata"]["source"] == "test"

    def test_empty_results(self, tmp_path):
        from features.diarize import write_jsonl

        output = str(tmp_path / "empty.jsonl")
        write_jsonl([], output)
        with open(output) as f:
            content = f.read().strip()
        assert content == ""

    def test_creates_parent_dir(self, tmp_path):
        from features.diarize import write_jsonl

        output = str(tmp_path / "nested" / "deep" / "out.jsonl")
        write_jsonl([{"speaker": "A", "start": 0, "end": 1, "duration": 1, "text": "Hi", "confidence": 1}], output)
        with open(output) as f:
            assert json.loads(f.readline())["speaker"] == "A"

    def test_speaker_stats_printed(self, tmp_path, capsys):
        from features.diarize import write_jsonl

        output = str(tmp_path / "stats.jsonl")
        results = [
            {"speaker": "A", "start": 0, "end": 10, "duration": 10.0, "text": "Hello", "confidence": 0.9},
            {"speaker": "B", "start": 10, "end": 15, "duration": 5.0, "text": "World", "confidence": 0.8},
        ]
        write_jsonl(results, output)
        captured = capsys.readouterr()
        assert "Speakers: 2" in captured.out
        assert "Total speech:" in captured.out
