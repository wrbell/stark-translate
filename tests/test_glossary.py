"""Tests for tools/glossary.py — tiered glossary helper for Deepgram keyterms."""

from __future__ import annotations

import json
import os
import sys

import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.glossary import (
    BIBLE_BOOKS,
    BOOST_PRIORITY,
    MAX_BOOST_TERMS,
    PROPER_NAMES,
    build_tiers,
    count_tokens,
    load_boost_keyterms,
    load_tier,
    validate_boost,
)

# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_single_word_terms(self):
        assert count_tokens(["hello", "world"]) == 2

    def test_multi_word_terms(self):
        assert count_tokens(["the Lamb of God"]) == 4

    def test_mixed_single_and_multi(self):
        assert count_tokens(["rapture", "born again", "kingdom of heaven"]) == 1 + 2 + 3

    def test_empty_list(self):
        assert count_tokens([]) == 0

    def test_single_term(self):
        assert count_tokens(["sanctification"]) == 1


# ---------------------------------------------------------------------------
# validate_boost
# ---------------------------------------------------------------------------


class TestValidateBoost:
    def test_passes_within_limits(self):
        terms = ["rapture", "born again", "tribulation"]
        # Should not raise
        validate_boost(terms)

    def test_raises_on_too_many_terms(self):
        terms = [f"term{i}" for i in range(MAX_BOOST_TERMS + 1)]
        with pytest.raises(ValueError, match="terms"):
            validate_boost(terms)

    def test_raises_on_too_many_tokens(self):
        # Each 8-word phrase = 8 tokens; 60 terms * 8 = 480 > 420 limit
        terms = [f"a b c d e f g h{i}" for i in range(MAX_BOOST_TERMS)]
        with pytest.raises(ValueError, match="tokens"):
            validate_boost(terms)

    def test_custom_limits(self):
        terms = ["one", "two", "three"]
        # Passes with generous limits
        validate_boost(terms, max_terms=10, max_tokens=100)
        # Fails with tight term limit
        with pytest.raises(ValueError):
            validate_boost(terms, max_terms=2, max_tokens=100)
        # Fails with tight token limit
        with pytest.raises(ValueError):
            validate_boost(terms, max_terms=10, max_tokens=2)


# ---------------------------------------------------------------------------
# build_tiers
# ---------------------------------------------------------------------------


class TestBuildTiers:
    def _make_glossary(self, extra_terms: int = 0) -> dict[str, str]:
        """Build a glossary that includes all BOOST_PRIORITY terms."""
        glossary: dict[str, str] = {}
        for term in BOOST_PRIORITY:
            glossary[term] = f"es_{term}"
        # Add extra filler terms (not in boost priority, books, or names)
        for i in range(extra_terms):
            glossary[f"extra_term_{i}"] = f"es_extra_{i}"
        return glossary

    def test_returns_boost_and_master_keys(self):
        glossary = self._make_glossary()
        tiers = build_tiers(glossary, boost_size=50)
        assert "boost" in tiers
        assert "master" in tiers

    def test_boost_respects_size(self):
        glossary = self._make_glossary(extra_terms=100)
        tiers = build_tiers(glossary, boost_size=20)
        assert len(tiers["boost"]) == 20

    def test_master_contains_all_terms(self):
        glossary = self._make_glossary(extra_terms=10)
        tiers = build_tiers(glossary, boost_size=50, master_size=5000)
        assert len(tiers["master"]) == len(glossary)

    def test_master_capped_by_size(self):
        glossary = self._make_glossary(extra_terms=200)
        tiers = build_tiers(glossary, boost_size=50, master_size=100)
        assert len(tiers["master"]) == 100

    def test_boost_excludes_bible_books(self):
        glossary = {b: f"es_{b}" for b in list(BIBLE_BOOKS)[:10]}
        glossary["rapture"] = "rapto"
        tiers = build_tiers(glossary, boost_size=50)
        for book in BIBLE_BOOKS:
            assert book not in tiers["boost"]

    def test_boost_excludes_proper_names(self):
        glossary = {n: f"es_{n}" for n in list(PROPER_NAMES)[:10]}
        glossary["rapture"] = "rapto"
        tiers = build_tiers(glossary, boost_size=50)
        for name in PROPER_NAMES:
            assert name not in tiers["boost"]

    def test_boost_priority_terms_included_first(self):
        glossary = self._make_glossary(extra_terms=100)
        tiers = build_tiers(glossary, boost_size=50)
        # All BOOST_PRIORITY terms that fit should be in boost
        for term in BOOST_PRIORITY[:50]:
            if term in glossary:
                assert term in tiers["boost"]


# ---------------------------------------------------------------------------
# load_tier — file not found
# ---------------------------------------------------------------------------


class TestLoadTier:
    def test_boost_returns_empty_list_when_missing(self, tmp_path):
        result = load_tier("boost", path=str(tmp_path / "nonexistent.json"))
        assert result == []

    def test_master_returns_empty_dict_when_missing(self, tmp_path):
        result = load_tier("master", path=str(tmp_path / "nonexistent.json"))
        assert result == {}

    def test_unknown_tier_raises(self):
        with pytest.raises(ValueError, match="Unknown tier"):
            load_tier("invalid")

    def test_boost_loads_from_file(self, tmp_path):
        f = tmp_path / "boost.json"
        terms = ["rapture", "tribulation"]
        f.write_text(json.dumps(terms))
        result = load_tier("boost", path=str(f))
        assert result == terms

    def test_master_loads_from_file(self, tmp_path):
        f = tmp_path / "master.json"
        data = {"faith": "fe", "grace": "gracia"}
        f.write_text(json.dumps(data))
        result = load_tier("master", path=str(f))
        assert result == data


# ---------------------------------------------------------------------------
# load_boost_keyterms
# ---------------------------------------------------------------------------


class TestLoadBoostKeyterms:
    def test_returns_empty_list_when_missing(self, tmp_path):
        result = load_boost_keyterms(path=str(tmp_path / "no_such_file.json"))
        assert result == []

    def test_loads_and_validates(self, tmp_path):
        f = tmp_path / "boost.json"
        terms = ["rapture", "born again"]
        f.write_text(json.dumps(terms))
        result = load_boost_keyterms(path=str(f))
        assert result == terms

    def test_raises_if_terms_exceed_limits(self, tmp_path):
        f = tmp_path / "boost.json"
        terms = [f"term{i}" for i in range(MAX_BOOST_TERMS + 5)]
        f.write_text(json.dumps(terms))
        with pytest.raises(ValueError):
            load_boost_keyterms(path=str(f))


# ---------------------------------------------------------------------------
# Sanity checks on constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_boost_priority_terms_are_strings(self):
        for term in BOOST_PRIORITY:
            assert isinstance(term, str)
            assert len(term) > 0

    def test_bible_books_count(self):
        assert len(BIBLE_BOOKS) == 66

    def test_bible_books_not_in_boost_priority(self):
        for book in BIBLE_BOOKS:
            assert book not in BOOST_PRIORITY

    def test_proper_names_not_in_boost_priority(self):
        for name in PROPER_NAMES:
            assert name not in BOOST_PRIORITY
