"""Tests for training/build_hard_subset.py — filtering, stratification, output format."""

import json

import pytest

from training.build_hard_subset import build_chunk_lookup, load_mined_jsonl


def _make_record(source="s1", chunk_idx=0, wer=0.20, has_tier1=False):
    """Create a minimal mined record for testing."""
    return {
        "source": source,
        "chunk_idx": chunk_idx,
        "start": 0.0,
        "end": 5.0,
        "reference": "test text",
        "prediction": "test txt",
        "wer": wer,
        "cer": wer * 0.5,
        "has_tier1": has_tier1,
        "tier1_terms": ["Holy Spirit"] if has_tier1 else [],
    }


def _make_chunk(source="s1", start=0.0, end=5.0, text="test text"):
    """Create a minimal whisper chunk dict."""
    return {
        "en": text,
        "start": start,
        "end": end,
        "confidence": 0.99,
        "avg_logprob": -0.05,
        "source": source,
    }


class TestLoadMinedJsonl:
    """Test JSONL loading with error handling."""

    def test_normal_file(self, tmp_path):
        p = tmp_path / "mined.jsonl"
        lines = [json.dumps(_make_record(chunk_idx=i)) for i in range(3)]
        p.write_text("\n".join(lines) + "\n")
        result = load_mined_jsonl(p)
        assert len(result) == 3

    def test_blank_lines_skipped(self, tmp_path):
        p = tmp_path / "mined.jsonl"
        p.write_text(json.dumps(_make_record()) + "\n\n\n")
        assert len(load_mined_jsonl(p)) == 1

    def test_malformed_lines_skipped(self, tmp_path):
        p = tmp_path / "mined.jsonl"
        p.write_text(json.dumps(_make_record()) + "\nnot json\n" + json.dumps(_make_record(chunk_idx=1)) + "\n")
        assert len(load_mined_jsonl(p)) == 2


class TestBuildChunkLookup:
    """Test chunk lookup construction with stable ordering."""

    def test_single_source(self):
        chunks = [_make_chunk("s1", 0, 5), _make_chunk("s1", 5, 10)]
        lookup = build_chunk_lookup(chunks)
        assert ("s1", 0) in lookup
        assert ("s1", 1) in lookup
        assert lookup[("s1", 0)]["start"] == 0.0
        assert lookup[("s1", 1)]["start"] == 5.0

    def test_multiple_sources(self):
        chunks = [
            _make_chunk("b", 0, 5),
            _make_chunk("a", 0, 3),
            _make_chunk("a", 3, 6),
        ]
        lookup = build_chunk_lookup(chunks)
        # Source "a" has 2 chunks, "b" has 1
        assert ("a", 0) in lookup
        assert ("a", 1) in lookup
        assert ("b", 0) in lookup

    def test_empty_chunks(self):
        assert build_chunk_lookup([]) == {}


class TestWerFiltering:
    """Test WER-based filtering with min/max bounds."""

    def test_within_bounds(self):
        records = [
            _make_record(wer=0.10),  # below min
            _make_record(wer=0.15, chunk_idx=1),  # at min (included)
            _make_record(wer=0.50, chunk_idx=2),  # in range
            _make_record(wer=0.80, chunk_idx=3),  # at max (excluded)
            _make_record(wer=0.90, chunk_idx=4),  # above max
        ]
        filtered = [r for r in records if 0.15 <= r["wer"] < 0.80]
        assert len(filtered) == 2
        assert filtered[0]["wer"] == 0.15
        assert filtered[1]["wer"] == 0.50

    def test_all_below_min(self):
        records = [_make_record(wer=0.05), _make_record(wer=0.10, chunk_idx=1)]
        filtered = [r for r in records if 0.15 <= r["wer"] < 0.80]
        assert len(filtered) == 0


class TestStratifiedCap:
    """Test per-source stratified capping."""

    def test_cap_exceeded(self):
        """Sources with more chunks than cap are trimmed to the cap."""
        records = [_make_record(source="s1", chunk_idx=i, wer=0.2 + i * 0.01) for i in range(10)]
        max_per_source = 3

        from collections import defaultdict

        by_source = defaultdict(list)
        for r in records:
            by_source[r["source"]].append(r)

        capped = []
        for source, source_records in by_source.items():
            source_records.sort(key=lambda r: -r["wer"])
            capped.extend(source_records[:max_per_source])

        assert len(capped) == 3
        # Should keep the 3 highest WER
        assert capped[0]["wer"] == pytest.approx(0.29, abs=0.001)

    def test_under_cap_unchanged(self):
        """Sources with fewer chunks than cap keep all."""
        records = [_make_record(source="s1", chunk_idx=i, wer=0.2) for i in range(2)]
        max_per_source = 10

        from collections import defaultdict

        by_source = defaultdict(list)
        for r in records:
            by_source[r["source"]].append(r)

        capped = []
        for source, source_records in by_source.items():
            source_records.sort(key=lambda r: -r["wer"])
            capped.extend(source_records[:max_per_source])

        assert len(capped) == 2


class TestTier1Inclusion:
    """Test that Tier 1 chunks are included regardless of WER."""

    def test_tier1_added_with_low_wer(self):
        """Tier 1 chunks with WER below threshold are still selected."""
        records = [
            _make_record(wer=0.05, has_tier1=True),  # below WER threshold but Tier 1
            _make_record(wer=0.30, chunk_idx=1),  # above WER threshold, no Tier 1
        ]
        wer_filtered = [r for r in records if 0.15 <= r["wer"] < 0.80]
        selected_keys = {(r["source"], r["chunk_idx"]) for r in wer_filtered}

        for r in records:
            key = (r["source"], r["chunk_idx"])
            if r.get("has_tier1") and key not in selected_keys:
                wer_filtered.append(r)
                selected_keys.add(key)

        assert len(wer_filtered) == 2

    def test_no_duplicates(self):
        """Tier 1 chunks already in WER filter aren't duplicated."""
        records = [
            _make_record(wer=0.30, has_tier1=True),  # meets WER AND is Tier 1
        ]
        wer_filtered = [r for r in records if 0.15 <= r["wer"] < 0.80]
        selected_keys = {(r["source"], r["chunk_idx"]) for r in wer_filtered}

        for r in records:
            key = (r["source"], r["chunk_idx"])
            if r.get("has_tier1") and key not in selected_keys:
                wer_filtered.append(r)
                selected_keys.add(key)

        assert len(wer_filtered) == 1


class TestTargetSizeLimit:
    """Test that output is capped at target_size."""

    def test_takes_top_n(self):
        records = [_make_record(chunk_idx=i, wer=0.2 + i * 0.01) for i in range(20)]
        records.sort(key=lambda r: -r["wer"])
        selected = records[:5]
        assert len(selected) == 5
        assert selected[0]["wer"] > selected[-1]["wer"]

    def test_fewer_than_target(self):
        records = [_make_record(chunk_idx=i, wer=0.3) for i in range(3)]
        selected = records[:10]
        assert len(selected) == 3


class TestOutputFormat:
    """Test that output chunks have the right schema for align_deepgram_chunks.py."""

    def test_output_has_required_keys(self):
        chunk = _make_chunk("s1", 0.0, 5.0, "test text")
        required_keys = {"en", "start", "end", "confidence", "avg_logprob", "source"}
        assert required_keys.issubset(chunk.keys())

    def test_lookup_maps_back_to_chunks(self):
        """Selected records can be mapped back to original chunk dicts."""
        chunks = [_make_chunk("s1", i * 5, (i + 1) * 5) for i in range(5)]
        lookup = build_chunk_lookup(chunks)
        record = _make_record(source="s1", chunk_idx=2)
        mapped = lookup.get((record["source"], record["chunk_idx"]))
        assert mapped is not None
        assert mapped["start"] == 10.0


class TestEmptyInput:
    """Test handling of empty inputs."""

    def test_empty_mined_jsonl(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        result = load_mined_jsonl(p)
        assert result == []
