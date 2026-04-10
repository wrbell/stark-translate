"""Tests for training/mine_hard_examples.py — Tier 1 detection and resume support."""

import json

from training.mine_hard_examples import find_tier1_terms, load_resume_set


class TestFindTier1Terms:
    """Test Tier 1 theological term detection in text."""

    TERMS = [
        "Holy Spirit",
        "substitutionary atonement",
        "born again",
        "kingdom of God",
    ]

    def test_exact_match(self):
        result = find_tier1_terms("the Holy Spirit descends", self.TERMS)
        assert result == ["Holy Spirit"]

    def test_case_insensitive(self):
        result = find_tier1_terms("THE HOLY SPIRIT descends", self.TERMS)
        assert result == ["Holy Spirit"]

    def test_multi_word_term(self):
        result = find_tier1_terms("the doctrine of substitutionary atonement is central", self.TERMS)
        assert result == ["substitutionary atonement"]

    def test_no_match(self):
        result = find_tier1_terms("a regular sentence about weather", self.TERMS)
        assert result == []

    def test_multiple_matches(self):
        result = find_tier1_terms("the Holy Spirit works in the kingdom of God", self.TERMS)
        assert "Holy Spirit" in result
        assert "kingdom of God" in result
        assert len(result) == 2

    def test_empty_text(self):
        assert find_tier1_terms("", self.TERMS) == []

    def test_empty_terms(self):
        assert find_tier1_terms("the Holy Spirit", []) == []

    def test_returns_original_casing(self):
        """Matching is case-insensitive but returned terms keep glossary casing."""
        result = find_tier1_terms("holy spirit", self.TERMS)
        assert result == ["Holy Spirit"]


class TestLoadResumeSet:
    """Test resume set loading from existing JSONL."""

    def test_nonexistent_file(self, tmp_path):
        result = load_resume_set(tmp_path / "nonexistent.jsonl")
        assert result == set()

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        result = load_resume_set(p)
        assert result == set()

    def test_partial_file(self, tmp_path):
        p = tmp_path / "partial.jsonl"
        lines = [
            json.dumps({"source": "sermon_a", "chunk_idx": 0, "wer": 0.1}),
            json.dumps({"source": "sermon_a", "chunk_idx": 5, "wer": 0.3}),
            json.dumps({"source": "sermon_b", "chunk_idx": 2, "wer": 0.5}),
        ]
        p.write_text("\n".join(lines) + "\n")
        result = load_resume_set(p)
        assert result == {("sermon_a", 0), ("sermon_a", 5), ("sermon_b", 2)}

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "blanks.jsonl"
        p.write_text(
            json.dumps({"source": "s", "chunk_idx": 0}) + "\n\n\n" + json.dumps({"source": "s", "chunk_idx": 1}) + "\n"
        )
        result = load_resume_set(p)
        assert len(result) == 2

    def test_skips_malformed_json(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text(
            json.dumps({"source": "s", "chunk_idx": 0})
            + "\n"
            + "not json at all\n"
            + json.dumps({"source": "s", "chunk_idx": 1})
            + "\n"
        )
        result = load_resume_set(p)
        assert result == {("s", 0), ("s", 1)}

    def test_skips_missing_keys(self, tmp_path):
        p = tmp_path / "missing.jsonl"
        p.write_text(json.dumps({"source": "s", "chunk_idx": 0}) + "\n" + json.dumps({"other_key": "value"}) + "\n")
        result = load_resume_set(p)
        assert result == {("s", 0)}
