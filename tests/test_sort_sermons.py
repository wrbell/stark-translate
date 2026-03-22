"""Tests for tools/sort_sermons.py — sermon classification, date parsing, manifest building."""

from __future__ import annotations

import json
import os
import sys
from datetime import date

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.sort_sermons import (
    build_manifest,
    classify_type,
    parse_date,
    parse_speakers,
    tag_chunks,
    target_dir_for_entry,
)

# ---------------------------------------------------------------------------
# classify_type
# ---------------------------------------------------------------------------


class TestClassifyType:
    def test_gospel_message(self):
        assert classify_type("Gospel Message (10/12/25)") == "gospel"

    def test_conference(self):
        assert classify_type("Conference 2022") == "conference"

    def test_baptism_is_gospel(self):
        assert classify_type("Baptism (5/3/25)") == "gospel"

    def test_ministry_meeting(self):
        assert classify_type("Ministry Meeting (3/1/26)") == "ministry"

    def test_default_is_ministry(self):
        assert classify_type("Some other title") == "ministry"

    def test_gospel_in_title(self):
        assert classify_type("Sunday Gospel") == "gospel"

    def test_conference_takes_priority_over_gospel(self):
        # "Conference" is checked first in the function
        assert classify_type("Conference Gospel Special") == "conference"


# ---------------------------------------------------------------------------
# parse_date
# ---------------------------------------------------------------------------


class TestParseDate:
    def test_parenthesized_date(self):
        assert parse_date("Gospel Message (10/12/25)") == date(2025, 10, 12)

    def test_parenthesized_date_2026(self):
        assert parse_date("Ministry Meeting (2/8/26)") == date(2026, 2, 8)

    def test_inline_date(self):
        assert parse_date("Something 10/18/25 title") == date(2025, 10, 18)

    def test_no_date_returns_none(self):
        assert parse_date("No date here") is None

    def test_invalid_date_returns_none(self):
        # Month 13 is invalid
        assert parse_date("Title (13/45/25)") is None

    def test_old_year_2000(self):
        assert parse_date("Throwback (7/12/00)") == date(2000, 7, 12)

    def test_old_year_1999(self):
        # yr >= 50 maps to 1900+yr
        assert parse_date("Old (3/15/99)") == date(1999, 3, 15)

    def test_year_49_maps_to_2049(self):
        assert parse_date("Future (1/1/49)") == date(2049, 1, 1)

    def test_year_50_maps_to_1950(self):
        assert parse_date("Vintage (6/15/50)") == date(1950, 6, 15)

    def test_slash_separator(self):
        assert parse_date("Title (3/14/26)") == date(2026, 3, 14)

    def test_underscore_separator(self):
        assert parse_date("Title (3_14_26)") == date(2026, 3, 14)

    def test_parenthesized_preferred_over_inline(self):
        # The function checks parenthesized first
        result = parse_date("Title (1/1/25) extra 6/6/26")
        assert result == date(2025, 1, 1)


# ---------------------------------------------------------------------------
# parse_speakers
# ---------------------------------------------------------------------------


class TestParseSpeakers:
    def test_from_speakers_field(self):
        meta = {"speakers": "John Smith", "description": ""}
        assert parse_speakers(meta) == "John Smith"

    def test_from_description_speakers_prefix(self):
        meta = {"description": "Speakers: John Smith, Jane Doe. Some other info."}
        assert parse_speakers(meta) == "John Smith, Jane Doe"

    def test_from_description_speaker_singular(self):
        meta = {"description": "Speaker: Bob Jones. Rest of description."}
        assert parse_speakers(meta) == "Bob Jones"

    def test_empty_fallback(self):
        meta = {"description": "Just a plain description."}
        assert parse_speakers(meta) == ""

    def test_empty_meta(self):
        assert parse_speakers({}) == ""

    def test_speakers_field_takes_precedence(self):
        meta = {"speakers": "Field Value", "description": "Speakers: Description Value."}
        assert parse_speakers(meta) == "Field Value"


# ---------------------------------------------------------------------------
# target_dir_for_entry
# ---------------------------------------------------------------------------


class TestTargetDirForEntry:
    def test_gospel_with_year(self):
        entry = {"type": "gospel", "year": "2021"}
        result = target_dir_for_entry(entry, "/data")
        assert result == os.path.join("/data", "gospel", "2021")

    def test_ministry_with_year(self):
        entry = {"type": "ministry", "year": "2025"}
        result = target_dir_for_entry(entry, "/data")
        assert result == os.path.join("/data", "ministry", "2025")

    def test_throwback_flat_path(self):
        entry = {"type": "throwback", "year": "2020"}
        result = target_dir_for_entry(entry, "/data")
        assert result == os.path.join("/data", "throwback")

    def test_unknown_year(self):
        entry = {"type": "gospel", "year": None}
        result = target_dir_for_entry(entry, "/data")
        assert result == os.path.join("/data", "gospel", "unknown")

    def test_conference(self):
        entry = {"type": "conference", "year": "2022"}
        result = target_dir_for_entry(entry, "/base")
        assert result == os.path.join("/base", "conference", "2022")


# ---------------------------------------------------------------------------
# build_manifest
# ---------------------------------------------------------------------------


class TestBuildManifest:
    def test_builds_from_json_files(self, tmp_path):
        # Create two fake metadata JSONs
        meta1 = {
            "title": "Gospel Message (10/12/25)",
            "video_id": "abc123",
            "duration_seconds": 3600,
        }
        meta2 = {
            "title": "Ministry Meeting (3/1/26)",
            "video_id": "def456",
            "duration_seconds": 2700,
        }
        (tmp_path / "sermon1.json").write_text(json.dumps(meta1))
        (tmp_path / "sermon2.json").write_text(json.dumps(meta2))

        cutoff = date(2026, 1, 1)
        manifest = build_manifest(tmp_path, cutoff)

        assert len(manifest) == 2
        types = {e["type"] for e in manifest}
        assert "gospel" in types
        assert "ministry" in types

    def test_train_eval_split_by_cutoff(self, tmp_path):
        meta_train = {"title": "Gospel Message (6/15/25)", "video_id": "t1"}
        meta_eval = {"title": "Gospel Message (3/15/26)", "video_id": "e1"}
        (tmp_path / "train_sermon.json").write_text(json.dumps(meta_train))
        (tmp_path / "eval_sermon.json").write_text(json.dumps(meta_eval))

        cutoff = date(2026, 1, 1)
        manifest = build_manifest(tmp_path, cutoff)

        splits = {e["video_id"]: e["split"] for e in manifest}
        assert splits["t1"] == "train"
        assert splits["e1"] == "eval"

    def test_unknown_split_when_no_date(self, tmp_path):
        meta = {"title": "No date title", "video_id": "nd1"}
        (tmp_path / "nodate.json").write_text(json.dumps(meta))

        manifest = build_manifest(tmp_path, date(2026, 1, 1))
        assert manifest[0]["split"] == "unknown"

    def test_empty_directory(self, tmp_path):
        manifest = build_manifest(tmp_path, date(2026, 1, 1))
        assert manifest == []

    def test_wav_path_none_when_no_wav(self, tmp_path):
        meta = {"title": "Gospel Message (1/1/25)", "video_id": "v1"}
        (tmp_path / "test.json").write_text(json.dumps(meta))

        manifest = build_manifest(tmp_path, date(2026, 1, 1))
        assert manifest[0]["wav_path"] is None


# ---------------------------------------------------------------------------
# tag_chunks
# ---------------------------------------------------------------------------


class TestTagChunks:
    def test_tags_matching_chunks(self, tmp_path):
        chunks = [
            {"source": "sermon1", "text": "Hello"},
            {"source": "sermon2", "text": "World"},
            {"source": "unknown_src", "text": "Other"},
        ]
        chunks_path = tmp_path / "chunks.json"
        chunks_path.write_text(json.dumps(chunks))

        manifest = [
            {"source": "sermon1", "split": "train", "type": "gospel", "year": "2025"},
            {"source": "sermon2", "split": "eval", "type": "ministry", "year": "2026"},
        ]

        tagged = tag_chunks(chunks_path, manifest)
        assert tagged == 2

        # Verify the file was updated
        result = json.loads(chunks_path.read_text())
        assert result[0]["split"] == "train"
        assert result[0]["sermon_type"] == "gospel"
        assert result[1]["split"] == "eval"
        assert result[1]["sermon_type"] == "ministry"
        assert result[2]["split"] == "unknown"
        assert result[2]["sermon_type"] == "unknown"

    def test_tags_to_separate_output(self, tmp_path):
        chunks = [{"source": "s1", "text": "Hi"}]
        chunks_path = tmp_path / "input.json"
        chunks_path.write_text(json.dumps(chunks))
        output_path = tmp_path / "output.json"

        manifest = [{"source": "s1", "split": "train", "type": "gospel", "year": "2025"}]
        tag_chunks(chunks_path, manifest, output_path=output_path)

        assert output_path.exists()
        result = json.loads(output_path.read_text())
        assert result[0]["split"] == "train"

    def test_no_matching_sources(self, tmp_path):
        chunks = [{"source": "no_match", "text": "X"}]
        chunks_path = tmp_path / "chunks.json"
        chunks_path.write_text(json.dumps(chunks))

        tagged = tag_chunks(chunks_path, [])
        assert tagged == 0
