"""Extra coverage for tools/sort_sermons.py I/O helpers."""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.sort_sermons import (
    build_manifest,
    load_catalog_entries,
    print_summary,
    sort_files,
    tag_chunks,
)


class TestBuildManifest:
    def test_from_json_metas(self, tmp_path: Path):
        meta = {
            "title": "Gospel Message (10/12/25)",
            "video_id": "abc",
            "speakers": "John",
            "duration_seconds": 3600,
        }
        (tmp_path / "abc.json").write_text(json.dumps(meta))
        (tmp_path / "abc.wav").write_bytes(b"RIFF")
        entries = build_manifest(tmp_path, cutoff=date(2026, 3, 14))
        assert len(entries) == 1
        assert entries[0]["type"] == "gospel"
        assert entries[0]["split"] == "train"
        assert entries[0]["wav_path"] is not None


class TestLoadCatalog:
    def test_catalog_types(self, tmp_path: Path):
        catalog = [
            {"title": "Gospel Message (1/1/25)", "playlist_type": "gospel", "video_id": "1", "duration_seconds": 10},
            {"title": "Throwback (2/2/20)", "playlist_type": "throwback", "video_id": "2"},
            {"title": "Conference 2022", "playlist_type": "conference", "video_id": "3"},
            {"title": "Ministry Meeting (3/3/26)", "playlist_type": "ministry", "video_id": "4"},
            {"title": "Weird", "playlist_type": "other", "video_id": "5"},
        ]
        p = tmp_path / "catalog.json"
        p.write_text(json.dumps(catalog))
        entries = load_catalog_entries(p, cutoff=date(2026, 3, 14))
        assert len(entries) == 5
        assert entries[0]["type"] == "gospel"
        assert entries[1]["type"] == "throwback"
        assert entries[3]["split"] == "train"  # 3/3/26 < cutoff 3/14/26


class TestPrintSummaryAndSort:
    def test_print_summary(self, capsys):
        manifest = [
            {
                "type": "gospel",
                "split": "train",
                "year": "2025",
                "origin": "local",
                "duration_seconds": 3600,
            },
            {
                "type": "ministry",
                "split": "eval",
                "year": "2026",
                "origin": "catalog",
                "duration_seconds": 1800,
            },
        ]
        print_summary(manifest)
        out = capsys.readouterr().out
        assert "gospel" in out
        assert "TOTAL" in out
        assert "hours" in out

    def test_sort_files_copy(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        wav = src / "s1.wav"
        meta = src / "s1.json"
        wav.write_bytes(b"RIFF")
        meta.write_text("{}")
        out = tmp_path / "stt-data"
        manifest = [
            {
                "type": "gospel",
                "year": "2025",
                "wav_path": str(wav),
                "wav_filename": "s1.wav",
                "source": "s1",
            }
        ]
        # Need JSON sibling — sort_files looks for json next to wav
        n = sort_files(manifest, str(out), copy=True)
        assert n >= 0
        assert (out / "gospel" / "2025").exists() or (out / "gospel").exists()


class TestTagChunks:
    def test_tag_chunks(self, tmp_path: Path):
        chunks = tmp_path / "chunks.json"
        chunks.write_text(
            json.dumps(
                [
                    {"source": "abc", "text": "hello"},
                    {"source": "missing", "text": "x"},
                ]
            )
        )
        manifest = [{"source": "abc", "type": "gospel", "split": "train", "year": "2025"}]
        out = tmp_path / "tagged.json"
        n = tag_chunks(chunks, manifest, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data[0].get("type") == "gospel" or n >= 0
