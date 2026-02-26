"""Tests for CSV/JSONL I/O functions in dry_run_ab.py."""

import csv
import json

import pytest

# ---------------------------------------------------------------------------
# Fixture: clear diagnostic accumulators and redirect paths to tmp_path
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _setup_io(tmp_path, monkeypatch):
    """Redirect CSV/DIAG output to tmp_path, clear diagnostics."""
    import dry_run_ab as d

    monkeypatch.setattr(d, "CSV_PATH", str(tmp_path / "metrics" / "test.csv"))
    monkeypatch.setattr(d, "DIAG_PATH", str(tmp_path / "metrics" / "diag.jsonl"))
    monkeypatch.setattr(d, "SESSION_ID", "test_session")
    monkeypatch.setattr(d, "MIC_GAIN", 1.0)
    d.diag_homophones.clear()
    d.diag_bad_splits.clear()
    d.diag_marian_diverge.clear()
    d.diag_durations.clear()
    d.diag_low_confidence.clear()
    d.partial_latencies.clear()
    yield
    d.diag_homophones.clear()
    d.diag_bad_splits.clear()
    d.diag_marian_diverge.clear()
    d.diag_durations.clear()
    d.diag_low_confidence.clear()
    d.partial_latencies.clear()


# ===================================================================
# init_csv
# ===================================================================


class TestInitCsv:
    def test_creates_file(self):
        import dry_run_ab as d

        d.init_csv()
        assert open(d.CSV_PATH).read() != ""

    def test_header_has_19_columns(self):
        import dry_run_ab as d

        d.init_csv()
        with open(d.CSV_PATH, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert len(header) == 20

    def test_creates_parent_dirs(self, tmp_path):
        import dry_run_ab as d

        deep = str(tmp_path / "a" / "b" / "c" / "test.csv")
        d.CSV_PATH = deep
        d.init_csv()
        assert open(deep).read() != ""


# ===================================================================
# write_csv_row
# ===================================================================


class TestWriteCsvRow:
    def _make_data(self, cid=1, **overrides):
        data = {
            "chunk_id": cid,
            "timestamp": "2026-02-24T10:00:00",
            "english": "Hello world",
            "spanish_a": "Hola mundo",
            "spanish_b": "",
            "stt_latency_ms": 150.0,
            "latency_a_ms": 650.0,
            "latency_b_ms": 0.0,
            "e2e_latency_ms": 800.0,
        }
        data.update(overrides)
        return data

    def test_appends_row(self):
        import dry_run_ab as d

        d.init_csv()
        d.write_csv_row(self._make_data())
        with open(d.CSV_PATH, newline="") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 2  # header + 1 data row

    def test_handles_missing_optional_keys(self):
        import dry_run_ab as d

        d.init_csv()
        data = self._make_data()
        # Don't set stt_confidence, qe_a, etc.
        d.write_csv_row(data)
        with open(d.CSV_PATH, newline="") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 2

    def test_includes_homophone_flags(self):
        import dry_run_ab as d

        d.init_csv()
        d.diag_homophones.append((1, "rain", "reign", "the rain fell"))
        d.write_csv_row(self._make_data(cid=1))
        with open(d.CSV_PATH, newline="") as f:
            rows = list(csv.reader(f))
        # homophone_flags column (index 15)
        assert "rain->reign" in rows[1][15]

    def test_includes_bad_split_flag(self):
        import dry_run_ab as d

        d.init_csv()
        d.diag_bad_splits.append((1, "the", "He said the"))
        d.write_csv_row(self._make_data(cid=1))
        with open(d.CSV_PATH, newline="") as f:
            rows = list(csv.reader(f))
        # bad_split column (index 17, after near_miss_flags at 16)
        assert rows[1][17] == "Y"


# ===================================================================
# write_diag_jsonl
# ===================================================================


class TestWriteDiagJsonl:
    def _make_data(self, cid=1, **overrides):
        data = {
            "chunk_id": cid,
            "timestamp": "2026-02-24T10:00:00",
            "english": "Hello world",
            "spanish_a": "Hola mundo",
        }
        data.update(overrides)
        return data

    def test_basic_record(self):
        import dry_run_ab as d

        d.write_diag_jsonl(self._make_data(), "/tmp/chunk_0001.wav")
        with open(d.DIAG_PATH) as f:
            record = json.loads(f.readline())
        assert record["chunk_id"] == 1
        assert record["english"] == "Hello world"
        assert record["audio_path"] == "/tmp/chunk_0001.wav"

    def test_priority_low_confidence(self):
        import dry_run_ab as d

        data = self._make_data(stt_confidence=0.4)
        d.write_diag_jsonl(data, "/tmp/chunk.wav")
        with open(d.DIAG_PATH) as f:
            record = json.loads(f.readline())
        assert record["review_priority"] >= 3

    def test_priority_hallucination(self):
        import dry_run_ab as d

        data = self._make_data()
        segment_meta = [{"compression_ratio": 3.0}]
        d.write_diag_jsonl(data, "/tmp/chunk.wav", segment_meta=segment_meta)
        with open(d.DIAG_PATH) as f:
            record = json.loads(f.readline())
        assert record["is_hallucination"] is True
        assert record["review_priority"] >= 5

    def test_priority_homophones(self):
        import dry_run_ab as d

        d.diag_homophones.append((1, "rain", "reign", "the rain"))
        data = self._make_data(cid=1)
        d.write_diag_jsonl(data, "/tmp/chunk.wav")
        with open(d.DIAG_PATH) as f:
            record = json.loads(f.readline())
        assert record["review_priority"] >= 2
        assert len(record["homophone_flags"]) == 1

    def test_creates_parent_dir(self, tmp_path, monkeypatch):
        import dry_run_ab as d

        deep = str(tmp_path / "x" / "y" / "diag.jsonl")
        monkeypatch.setattr(d, "DIAG_PATH", deep)
        d.write_diag_jsonl(self._make_data(), "/tmp/chunk.wav")
        assert open(deep).read() != ""
