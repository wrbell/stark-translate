"""Tests for tools/prepare_finetune_data.py — fine-tuning data bridge."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is on path so we can import the tool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from tools.prepare_finetune_data import (
    _get_segment_metadata,
    _passes_whisper_filter,
    build_parser,
    cmd_apply_corrections,
    cmd_export_translation,
    cmd_export_whisper,
    cmd_extract_review_queue,
    cmd_summary,
    load_diagnostics,
    resolve_diagnostics_paths,
)

# ---------------------------------------------------------------------------
# Test data factory
# ---------------------------------------------------------------------------


def _make_diag_record(**overrides) -> dict:
    """Create a complete diagnostics record with sensible defaults."""
    rec = {
        "chunk_id": 1,
        "session": "20260301_113532",
        "timestamp": "2026-03-01T11:37:01.960449",
        "audio_path": "stark_data/live_sessions/20260301_113532/chunk_0001.wav",
        "mic_gain": 8.48,
        "english": "For God so loved the world.",
        "spanish_gemma": "Porque de tal manera amó Dios al mundo.",
        "spanish_marian": "Porque Dios amó tanto al mundo.",
        "stt_confidence": 0.92,
        "qe_a": 0.95,
        "utterance_dur": 3.84,
        "segment_metadata": [
            {
                "avg_logprob": -0.18,
                "no_speech_prob": 0.0,
                "compression_ratio": 0.96,
            }
        ],
        "low_confidence_words": [],
        "is_hallucination": False,
        "homophone_flags": [],
        "near_miss_flags": [],
        "bad_split": False,
        "marian_similarity": 0.38,
        "review_priority": 0,
        "marian_backend_latency": {"pt_ms": 121.4},
        "resources": {"cpu_percent": 6.0, "ram_used_gb": 9.83},
        "stt_corrections": [],
        "corrected_english": None,
        "corrected_spanish": None,
    }
    rec.update(overrides)
    return rec


def _write_diag_jsonl(path: Path, records: list[dict]) -> Path:
    """Write records as JSONL to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def _write_corrections_tsv(path: Path, rows: list[dict]) -> Path:
    """Write a corrections TSV file."""
    fieldnames = [
        "chunk_id",
        "session",
        "recorded_english",
        "corrected_english",
        "spanish_gemma",
        "spanish_marian",
        "corrected_spanish",
        "stt_confidence",
        "qe_a",
        "review_priority",
        "audio_path",
        "correction_source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            full = {k: "" for k in fieldnames}
            full.update(row)
            writer.writerow(full)
    return path


# ---------------------------------------------------------------------------
# JSONL loading tests
# ---------------------------------------------------------------------------


class TestResolveDiagnosticsPaths:
    def test_finds_all_files(self, tmp_path):
        metrics = tmp_path / "metrics"
        _write_diag_jsonl(metrics / "diagnostics_20260301_113532.jsonl", [])
        _write_diag_jsonl(metrics / "diagnostics_20260301_120127.jsonl", [])

        paths = resolve_diagnostics_paths(metrics_dir=str(metrics))
        assert len(paths) == 2

    def test_filters_by_session(self, tmp_path):
        metrics = tmp_path / "metrics"
        _write_diag_jsonl(metrics / "diagnostics_20260301_113532.jsonl", [])
        _write_diag_jsonl(metrics / "diagnostics_20260301_120127.jsonl", [])

        paths = resolve_diagnostics_paths(session_ids=["20260301_113532"], metrics_dir=str(metrics))
        assert len(paths) == 1
        assert "113532" in paths[0].name

    def test_returns_empty_for_missing_dir(self, tmp_path):
        paths = resolve_diagnostics_paths(metrics_dir=str(tmp_path / "nonexistent"))
        assert paths == []

    def test_ignores_non_diagnostics_files(self, tmp_path):
        metrics = tmp_path / "metrics"
        _write_diag_jsonl(metrics / "diagnostics_20260301_113532.jsonl", [])
        _write_diag_jsonl(metrics / "ab_metrics_20260301.jsonl", [])

        paths = resolve_diagnostics_paths(metrics_dir=str(metrics))
        assert len(paths) == 1


class TestLoadDiagnostics:
    def test_loads_valid_records(self, tmp_path):
        p = _write_diag_jsonl(
            tmp_path / "d.jsonl",
            [_make_diag_record(), _make_diag_record(chunk_id=2)],
        )
        records = load_diagnostics([p])
        assert len(records) == 2
        assert records[0]["chunk_id"] == 1
        assert records[1]["chunk_id"] == 2

    def test_filters_event_records(self, tmp_path):
        p = tmp_path / "d.jsonl"
        with open(p, "w") as f:
            f.write(json.dumps({"event": "session_start", "ts": "2026-01-01"}) + "\n")
            f.write(json.dumps(_make_diag_record()) + "\n")
            f.write(json.dumps({"event": "session_end", "ts": "2026-01-01"}) + "\n")
        records = load_diagnostics([p])
        assert len(records) == 1
        assert records[0]["chunk_id"] == 1

    def test_skips_invalid_json(self, tmp_path):
        p = tmp_path / "d.jsonl"
        with open(p, "w") as f:
            f.write(json.dumps(_make_diag_record()) + "\n")
            f.write("not valid json\n")
            f.write(json.dumps(_make_diag_record(chunk_id=2)) + "\n")
        records = load_diagnostics([p])
        assert len(records) == 2

    def test_skips_records_without_chunk_id(self, tmp_path):
        p = tmp_path / "d.jsonl"
        with open(p, "w") as f:
            f.write(json.dumps({"some": "metadata"}) + "\n")
            f.write(json.dumps(_make_diag_record()) + "\n")
        records = load_diagnostics([p])
        assert len(records) == 1

    def test_multi_session_merge(self, tmp_path):
        p1 = _write_diag_jsonl(
            tmp_path / "d1.jsonl",
            [_make_diag_record(session="sess_a", chunk_id=1)],
        )
        p2 = _write_diag_jsonl(
            tmp_path / "d2.jsonl",
            [_make_diag_record(session="sess_b", chunk_id=1)],
        )
        records = load_diagnostics([p1, p2])
        assert len(records) == 2
        sessions = {r["session"] for r in records}
        assert sessions == {"sess_a", "sess_b"}

    def test_attaches_source_file(self, tmp_path):
        p = _write_diag_jsonl(tmp_path / "d.jsonl", [_make_diag_record()])
        records = load_diagnostics([p])
        assert records[0]["_source_file"] == str(p)

    def test_empty_lines_skipped(self, tmp_path):
        p = tmp_path / "d.jsonl"
        with open(p, "w") as f:
            f.write(json.dumps(_make_diag_record()) + "\n")
            f.write("\n")
            f.write("   \n")
            f.write(json.dumps(_make_diag_record(chunk_id=2)) + "\n")
        records = load_diagnostics([p])
        assert len(records) == 2

    def test_empty_file(self, tmp_path):
        p = tmp_path / "d.jsonl"
        p.write_text("")
        records = load_diagnostics([p])
        assert records == []


# ---------------------------------------------------------------------------
# Segment metadata / whisper filter tests
# ---------------------------------------------------------------------------


class TestSegmentMetadata:
    def test_extracts_aggregates(self):
        rec = _make_diag_record(
            segment_metadata=[
                {"avg_logprob": -0.2, "compression_ratio": 1.0, "no_speech_prob": 0.01},
                {"avg_logprob": -0.4, "compression_ratio": 1.5, "no_speech_prob": 0.05},
            ]
        )
        meta = _get_segment_metadata(rec)
        assert abs(meta["avg_logprob"] - (-0.3)) < 1e-6
        assert meta["max_compression_ratio"] == 1.5
        assert meta["max_no_speech_prob"] == 0.05

    def test_empty_metadata(self):
        rec = _make_diag_record(segment_metadata=[])
        meta = _get_segment_metadata(rec)
        assert meta == {}

    def test_none_metadata(self):
        rec = _make_diag_record(segment_metadata=None)
        meta = _get_segment_metadata(rec)
        assert meta == {}


class TestWhisperFilter:
    def test_passes_good_record(self):
        rec = _make_diag_record()
        assert _passes_whisper_filter(rec) is True

    def test_rejects_high_compression(self):
        rec = _make_diag_record(
            segment_metadata=[{"avg_logprob": -0.2, "compression_ratio": 2.5, "no_speech_prob": 0.0}]
        )
        assert _passes_whisper_filter(rec) is False

    def test_rejects_low_logprob(self):
        rec = _make_diag_record(
            segment_metadata=[{"avg_logprob": -1.5, "compression_ratio": 1.0, "no_speech_prob": 0.0}]
        )
        assert _passes_whisper_filter(rec) is False

    def test_rejects_high_no_speech_with_low_logprob(self):
        rec = _make_diag_record(
            segment_metadata=[{"avg_logprob": -0.6, "compression_ratio": 1.0, "no_speech_prob": 0.7}]
        )
        assert _passes_whisper_filter(rec) is False

    def test_passes_high_no_speech_with_good_logprob(self):
        rec = _make_diag_record(
            segment_metadata=[{"avg_logprob": -0.2, "compression_ratio": 1.0, "no_speech_prob": 0.7}]
        )
        assert _passes_whisper_filter(rec) is True

    def test_passes_no_metadata(self):
        rec = _make_diag_record(segment_metadata=None)
        assert _passes_whisper_filter(rec) is True


# ---------------------------------------------------------------------------
# extract-review-queue tests
# ---------------------------------------------------------------------------


class TestExtractReviewQueue:
    def _run(self, tmp_path, records, **kwargs):
        """Helper: write records and run extract-review-queue."""
        metrics = tmp_path / "metrics"
        _write_diag_jsonl(metrics / "diagnostics_20260301_113532.jsonl", records)
        out = tmp_path / "review.tsv"
        defaults = {
            "session": None,
            "min_priority": 1,
            "top_n": 0,
            "output": str(out),
            "metrics_dir": str(metrics),
        }
        defaults.update(kwargs)
        args = argparse.Namespace(**defaults)
        cmd_extract_review_queue(args)
        return out

    def test_writes_tsv_with_flagged_segments(self, tmp_path):
        records = [
            _make_diag_record(chunk_id=1, review_priority=3),
            _make_diag_record(chunk_id=2, review_priority=0),
            _make_diag_record(chunk_id=3, review_priority=2),
        ]
        out = self._run(tmp_path, records)

        with open(out, encoding="utf-8") as f:
            reader = list(csv.DictReader(f, delimiter="\t"))
        assert len(reader) == 2  # Only priority >= 1
        # Sorted by priority desc
        assert reader[0]["chunk_id"] == "3" or reader[0]["review_priority"] == "3"

    def test_priority_sorting(self, tmp_path):
        records = [
            _make_diag_record(chunk_id=1, review_priority=1),
            _make_diag_record(chunk_id=2, review_priority=5),
            _make_diag_record(chunk_id=3, review_priority=3),
        ]
        out = self._run(tmp_path, records)

        with open(out, encoding="utf-8") as f:
            reader = list(csv.DictReader(f, delimiter="\t"))
        priorities = [int(r["review_priority"]) for r in reader]
        assert priorities == [5, 3, 1]

    def test_min_priority_filter(self, tmp_path):
        records = [
            _make_diag_record(chunk_id=1, review_priority=1),
            _make_diag_record(chunk_id=2, review_priority=3),
            _make_diag_record(chunk_id=3, review_priority=5),
        ]
        out = self._run(tmp_path, records, min_priority=3)

        with open(out, encoding="utf-8") as f:
            reader = list(csv.DictReader(f, delimiter="\t"))
        assert len(reader) == 2

    def test_top_n_limit(self, tmp_path):
        records = [_make_diag_record(chunk_id=i, review_priority=i) for i in range(1, 11)]
        out = self._run(tmp_path, records, top_n=3)

        with open(out, encoding="utf-8") as f:
            reader = list(csv.DictReader(f, delimiter="\t"))
        assert len(reader) == 3

    def test_skips_already_corrected(self, tmp_path):
        records = [
            _make_diag_record(
                chunk_id=1,
                review_priority=5,
                corrected_english="Fixed text",
            ),
            _make_diag_record(chunk_id=2, review_priority=3),
        ]
        out = self._run(tmp_path, records)

        with open(out, encoding="utf-8") as f:
            reader = list(csv.DictReader(f, delimiter="\t"))
        assert len(reader) == 1
        assert reader[0]["chunk_id"] == "2"

    def test_tsv_has_empty_correction_columns(self, tmp_path):
        records = [_make_diag_record(chunk_id=1, review_priority=2)]
        out = self._run(tmp_path, records)

        with open(out, encoding="utf-8") as f:
            reader = list(csv.DictReader(f, delimiter="\t"))
        assert reader[0]["corrected_english"] == ""
        assert reader[0]["corrected_spanish"] == ""
        assert reader[0]["correction_source"] == ""


# ---------------------------------------------------------------------------
# apply-corrections tests
# ---------------------------------------------------------------------------


class TestApplyCorrections:
    def _setup_and_run(self, tmp_path, records, corrections, **kwargs):
        """Write diagnostics + corrections, run apply-corrections."""
        metrics = tmp_path / "metrics"
        diag_path = metrics / "diagnostics_20260301_113532.jsonl"
        _write_diag_jsonl(diag_path, records)

        tsv_path = tmp_path / "corrections.tsv"
        _write_corrections_tsv(tsv_path, corrections)

        defaults = {
            "corrections": str(tsv_path),
            "auto_correct": False,
            "dry_run": False,
            "session": None,
            "metrics_dir": str(metrics),
        }
        defaults.update(kwargs)
        args = argparse.Namespace(**defaults)
        cmd_apply_corrections(args)
        return diag_path

    def test_applies_human_corrections(self, tmp_path):
        records = [_make_diag_record(chunk_id=1)]
        corrections = [
            {
                "chunk_id": "1",
                "session": "20260301_113532",
                "corrected_english": "For God so loved the world.",
                "corrected_spanish": "Porque Dios amó tanto al mundo.",
            }
        ]
        diag_path = self._setup_and_run(tmp_path, records, corrections)

        with open(diag_path) as f:
            updated = json.loads(f.readline())
        assert updated["corrected_english"] == "For God so loved the world."
        assert updated["corrected_spanish"] == "Porque Dios amó tanto al mundo."
        assert updated["correction_source"] == "human"
        assert "correction_timestamp" in updated

    def test_partial_correction_english_only(self, tmp_path):
        records = [_make_diag_record(chunk_id=1)]
        corrections = [
            {
                "chunk_id": "1",
                "session": "20260301_113532",
                "corrected_english": "Corrected English.",
            }
        ]
        diag_path = self._setup_and_run(tmp_path, records, corrections)

        with open(diag_path) as f:
            updated = json.loads(f.readline())
        assert updated["corrected_english"] == "Corrected English."
        assert updated["corrected_spanish"] is None  # Unchanged

    def test_dry_run_does_not_modify(self, tmp_path):
        records = [_make_diag_record(chunk_id=1)]
        corrections = [
            {
                "chunk_id": "1",
                "session": "20260301_113532",
                "corrected_english": "Fixed.",
            }
        ]
        diag_path = self._setup_and_run(tmp_path, records, corrections, dry_run=True)

        with open(diag_path) as f:
            original = json.loads(f.readline())
        assert original["corrected_english"] is None  # Unchanged

    def test_auto_correct_applies_rules(self, tmp_path):
        records = [
            _make_diag_record(
                chunk_id=1,
                english="Damn you, if you will, is one of my heroes.",
            )
        ]
        diag_path = self._setup_and_run(tmp_path, records, [], auto_correct=True)

        with open(diag_path) as f:
            updated = json.loads(f.readline())
        assert updated["corrected_english"] is not None
        assert "damn" not in updated["corrected_english"].lower()
        assert updated["correction_source"] == "auto"

    def test_auto_correct_skips_already_corrected(self, tmp_path):
        records = [
            _make_diag_record(
                chunk_id=1,
                english="Damn you said hello.",
                corrected_english="Daniel said hello.",
            )
        ]
        diag_path = self._setup_and_run(tmp_path, records, [], auto_correct=True)

        with open(diag_path) as f:
            updated = json.loads(f.readline())
        # Should not overwrite existing correction
        assert updated["corrected_english"] == "Daniel said hello."

    def test_human_takes_precedence_over_auto(self, tmp_path):
        records = [
            _make_diag_record(
                chunk_id=1,
                english="Damn you, if you will, is one of my heroes.",
            )
        ]
        corrections = [
            {
                "chunk_id": "1",
                "session": "20260301_113532",
                "corrected_english": "Human override.",
            }
        ]
        diag_path = self._setup_and_run(tmp_path, records, corrections, auto_correct=True)

        with open(diag_path) as f:
            updated = json.loads(f.readline())
        assert updated["corrected_english"] == "Human override."
        assert updated["correction_source"] == "human"

    def test_idempotent_rerun(self, tmp_path):
        records = [_make_diag_record(chunk_id=1)]
        corrections = [
            {
                "chunk_id": "1",
                "session": "20260301_113532",
                "corrected_english": "Fixed.",
            }
        ]
        diag_path = self._setup_and_run(tmp_path, records, corrections)

        # Run again
        tsv_path = tmp_path / "corrections.tsv"
        args = argparse.Namespace(
            corrections=str(tsv_path),
            auto_correct=False,
            dry_run=False,
            session=None,
            metrics_dir=str(tmp_path / "metrics"),
        )
        cmd_apply_corrections(args)

        with open(diag_path) as f:
            updated = json.loads(f.readline())
        assert updated["corrected_english"] == "Fixed."

    def test_no_correction_leaves_record_unchanged(self, tmp_path):
        records = [_make_diag_record(chunk_id=1)]
        diag_path = self._setup_and_run(tmp_path, records, [])

        with open(diag_path) as f:
            updated = json.loads(f.readline())
        assert updated["corrected_english"] is None
        assert updated["corrected_spanish"] is None

    def test_auto_correct_no_match_leaves_record(self, tmp_path):
        """Auto-correct with no rule matches should not set corrected_english."""
        records = [
            _make_diag_record(
                chunk_id=1,
                english="For God so loved the world.",
            )
        ]
        diag_path = self._setup_and_run(tmp_path, records, [], auto_correct=True)

        with open(diag_path) as f:
            updated = json.loads(f.readline())
        # No corrections triggered, should remain None
        assert updated["corrected_english"] is None


# ---------------------------------------------------------------------------
# export-whisper tests
# ---------------------------------------------------------------------------


class TestExportWhisper:
    def _run(self, tmp_path, records, **kwargs):
        """Helper: write records and run export-whisper."""
        metrics = tmp_path / "metrics"
        _write_diag_jsonl(metrics / "diagnostics_20260301_113532.jsonl", records)

        # Create fake audio files
        for rec in records:
            audio_path = Path(rec["audio_path"])
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(b"\x00" * 100)

        out = tmp_path / "whisper_out"
        defaults = {
            "session": None,
            "output": str(out),
            "accent": "church_live",
            "eval_ratio": 0.05,
            "metrics_dir": str(metrics),
        }
        defaults.update(kwargs)
        args = argparse.Namespace(**defaults)
        cmd_export_whisper(args)
        return out

    def test_creates_audiofolder_structure(self, tmp_path):
        records = [_make_diag_record(chunk_id=i) for i in range(1, 21)]
        # Give unique audio paths
        for i, r in enumerate(records, 1):
            r["audio_path"] = str(tmp_path / f"stark_data/live_sessions/sess/chunk_{i:04d}.wav")
        out = self._run(tmp_path, records)

        assert (out / "train" / "metadata.csv").exists()
        assert (out / "eval" / "metadata.csv").exists()
        assert (out / "train" / "church_live").is_dir()
        assert (out / "eval" / "church_live").is_dir()

    def test_metadata_csv_format(self, tmp_path):
        rec = _make_diag_record(chunk_id=1)
        rec["audio_path"] = str(tmp_path / "stark_data/live_sessions/sess/chunk_0001.wav")
        out = self._run(tmp_path, [rec], eval_ratio=0.0)

        meta_path = out / "train" / "metadata.csv"
        with open(meta_path) as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert reader[0]["file_name"] == "church_live/chunk_0001.wav"
        assert reader[0]["transcription"] == "For God so loved the world."
        assert reader[0]["accent"] == "church_live"

    def test_prefers_corrected_text(self, tmp_path):
        rec = _make_diag_record(
            chunk_id=1,
            english="Original.",
            corrected_english="Corrected.",
        )
        rec["audio_path"] = str(tmp_path / "stark_data/live_sessions/sess/chunk_0001.wav")
        out = self._run(tmp_path, [rec], eval_ratio=0.0)

        meta_path = out / "train" / "metadata.csv"
        with open(meta_path) as f:
            reader = list(csv.DictReader(f))
        assert reader[0]["transcription"] == "Corrected."

    def test_filters_by_confidence(self, tmp_path):
        good = _make_diag_record(chunk_id=1)
        good["audio_path"] = str(tmp_path / "stark_data/live_sessions/sess/chunk_0001.wav")

        bad = _make_diag_record(
            chunk_id=2,
            segment_metadata=[{"avg_logprob": -0.2, "compression_ratio": 3.0, "no_speech_prob": 0.0}],
        )
        bad["audio_path"] = str(tmp_path / "stark_data/live_sessions/sess/chunk_0002.wav")
        out = self._run(tmp_path, [good, bad], eval_ratio=0.0)

        meta_path = out / "train" / "metadata.csv"
        with open(meta_path) as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 1  # Only the good record

    def test_skips_empty_transcription(self, tmp_path, capsys):
        rec = _make_diag_record(chunk_id=1, english="", corrected_english=None)
        rec["audio_path"] = str(tmp_path / "stark_data/live_sessions/sess/chunk_0001.wav")
        out = self._run(tmp_path, [rec], eval_ratio=0.0)

        # No eligible records → function returns early, no output created
        captured = capsys.readouterr()
        assert "No eligible records" in captured.out
        assert not (out / "train" / "metadata.csv").exists()

    def test_eval_ratio_split(self, tmp_path):
        records = []
        for i in range(1, 21):
            rec = _make_diag_record(chunk_id=i)
            rec["audio_path"] = str(tmp_path / f"stark_data/live_sessions/sess/chunk_{i:04d}.wav")
            records.append(rec)

        out = self._run(tmp_path, records, eval_ratio=0.2)

        with open(out / "train" / "metadata.csv") as f:
            train_rows = list(csv.DictReader(f))
        with open(out / "eval" / "metadata.csv") as f:
            eval_rows = list(csv.DictReader(f))
        # 20 records * 0.2 = 4 eval, 16 train
        assert len(eval_rows) == 4
        assert len(train_rows) == 16


# ---------------------------------------------------------------------------
# export-translation tests
# ---------------------------------------------------------------------------


class TestExportTranslation:
    def _run(self, tmp_path, records, **kwargs):
        """Helper: write records and run export-translation."""
        metrics = tmp_path / "metrics"
        _write_diag_jsonl(metrics / "diagnostics_20260301_113532.jsonl", records)
        out = tmp_path / "pairs.jsonl"
        defaults = {
            "session": None,
            "output": str(out),
            "min_qe": 0.6,
            "metrics_dir": str(metrics),
        }
        defaults.update(kwargs)
        args = argparse.Namespace(**defaults)
        cmd_export_translation(args)
        return out

    def test_writes_jsonl_format(self, tmp_path):
        records = [_make_diag_record(chunk_id=1, qe_a=0.95)]
        out = self._run(tmp_path, records)

        with open(out) as f:
            pair = json.loads(f.readline())
        assert pair["en"] == "For God so loved the world."
        assert pair["es"] == "Porque de tal manera amó Dios al mundo."

    def test_spanish_cascade_corrected_first(self, tmp_path):
        records = [
            _make_diag_record(
                chunk_id=1,
                qe_a=0.95,
                corrected_spanish="Human Spanish.",
                spanish_gemma="Gemma Spanish.",
                spanish_marian="Marian Spanish.",
            )
        ]
        out = self._run(tmp_path, records)

        with open(out) as f:
            pair = json.loads(f.readline())
        assert pair["es"] == "Human Spanish."

    def test_spanish_cascade_gemma_second(self, tmp_path):
        records = [
            _make_diag_record(
                chunk_id=1,
                qe_a=0.95,
                corrected_spanish=None,
                spanish_gemma="Gemma Spanish.",
                spanish_marian="Marian Spanish.",
            )
        ]
        out = self._run(tmp_path, records)

        with open(out) as f:
            pair = json.loads(f.readline())
        assert pair["es"] == "Gemma Spanish."

    def test_spanish_cascade_marian_third(self, tmp_path):
        records = [
            _make_diag_record(
                chunk_id=1,
                qe_a=0.95,
                corrected_spanish=None,
                spanish_gemma=None,
                spanish_marian="Marian Spanish.",
            )
        ]
        out = self._run(tmp_path, records)

        with open(out) as f:
            pair = json.loads(f.readline())
        assert pair["es"] == "Marian Spanish."

    def test_qe_filter(self, tmp_path):
        records = [
            _make_diag_record(chunk_id=1, qe_a=0.3),
            _make_diag_record(chunk_id=2, qe_a=0.8),
        ]
        out = self._run(tmp_path, records, min_qe=0.6)

        with open(out) as f:
            pairs = [json.loads(line) for line in f]
        assert len(pairs) == 1

    def test_human_correction_bypasses_qe_filter(self, tmp_path):
        records = [
            _make_diag_record(
                chunk_id=1,
                qe_a=0.2,
                correction_source="human",
                corrected_english="Human English.",
                corrected_spanish="Human Spanish.",
            )
        ]
        out = self._run(tmp_path, records, min_qe=0.6)

        with open(out) as f:
            pairs = [json.loads(line) for line in f]
        assert len(pairs) == 1
        assert pairs[0]["en"] == "Human English."

    def test_skips_missing_english(self, tmp_path):
        records = [_make_diag_record(chunk_id=1, english="", qe_a=0.95)]
        out = self._run(tmp_path, records)

        assert not out.exists() or out.stat().st_size == 0

    def test_skips_missing_spanish(self, tmp_path):
        records = [
            _make_diag_record(
                chunk_id=1,
                qe_a=0.95,
                spanish_gemma=None,
                spanish_marian=None,
                corrected_spanish=None,
            )
        ]
        out = self._run(tmp_path, records)

        assert not out.exists() or out.stat().st_size == 0

    def test_prefers_corrected_english(self, tmp_path):
        records = [
            _make_diag_record(
                chunk_id=1,
                qe_a=0.95,
                english="Original.",
                corrected_english="Corrected.",
            )
        ]
        out = self._run(tmp_path, records)

        with open(out) as f:
            pair = json.loads(f.readline())
        assert pair["en"] == "Corrected."


# ---------------------------------------------------------------------------
# summary tests
# ---------------------------------------------------------------------------


class TestSummary:
    def _run(self, tmp_path, records, capsys, **kwargs):
        """Helper: write records and run summary."""
        metrics = tmp_path / "metrics"
        _write_diag_jsonl(metrics / "diagnostics_20260301_113532.jsonl", records)
        defaults = {
            "session": None,
            "metrics_dir": str(metrics),
        }
        defaults.update(kwargs)
        args = argparse.Namespace(**defaults)
        # Mock dry_run_ab import for auto-correction potential
        with patch(
            "tools.prepare_finetune_data.apply_auto_corrections_to_record",
            return_value=("text", []),
        ):
            cmd_summary(args)
        return capsys.readouterr()

    def test_shows_total_chunks(self, tmp_path, capsys):
        records = [_make_diag_record(chunk_id=i) for i in range(1, 6)]
        captured = self._run(tmp_path, records, capsys)
        assert "Total chunks: 5" in captured.out

    def test_shows_correction_breakdown(self, tmp_path, capsys):
        records = [
            _make_diag_record(chunk_id=1, correction_source="human"),
            _make_diag_record(chunk_id=2, correction_source="auto"),
            _make_diag_record(chunk_id=3),
        ]
        captured = self._run(tmp_path, records, capsys)
        assert "Human-corrected: 1" in captured.out
        assert "Auto-corrected:  1" in captured.out
        assert "Uncorrected:     1" in captured.out

    def test_shows_priority_distribution(self, tmp_path, capsys):
        records = [
            _make_diag_record(chunk_id=1, review_priority=0),
            _make_diag_record(chunk_id=2, review_priority=3),
            _make_diag_record(chunk_id=3, review_priority=3),
        ]
        captured = self._run(tmp_path, records, capsys)
        assert "priority 0: 1" in captured.out
        assert "priority 3: 2" in captured.out

    def test_empty_input(self, tmp_path, capsys):
        metrics = tmp_path / "metrics"
        _write_diag_jsonl(metrics / "diagnostics_20260301_113532.jsonl", [])
        args = argparse.Namespace(session=None, metrics_dir=str(metrics))
        with patch(
            "tools.prepare_finetune_data.apply_auto_corrections_to_record",
            return_value=("text", []),
        ):
            cmd_summary(args)
        captured = capsys.readouterr()
        assert "No chunk records found" in captured.out


# ---------------------------------------------------------------------------
# CLI parser tests
# ---------------------------------------------------------------------------


class TestCLIParser:
    def test_extract_review_queue_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["extract-review-queue"])
        assert args.command == "extract-review-queue"
        assert args.min_priority == 1
        assert args.top_n == 0

    def test_apply_corrections_requires_file(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["apply-corrections"])

    def test_export_whisper_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["export-whisper"])
        assert args.accent == "church_live"
        assert args.eval_ratio == 0.05

    def test_export_translation_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["export-translation"])
        assert args.min_qe == 0.6

    def test_summary_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["summary"])
        assert args.session is None
