"""Tests for tools/translation_qe.py — Tier 1 pure QE functions."""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.translation_qe import qe_length_ratio, qe_untranslated, tier1_score


class TestQELengthRatio:
    def test_good_ratio(self):
        # Spanish is typically 15-25% longer
        score = qe_length_ratio("Hello world", "Hola mundo amigos")
        assert score == 1.0

    def test_equal_length(self):
        score = qe_length_ratio("hello", "hola!")
        assert score == 1.0

    def test_moderately_longer(self):
        # Ratio around 1.8 (within 0.7-2.0 band)
        score = qe_length_ratio("Hello friends", "Hola queridos amigos míos")
        assert score == 0.7

    def test_very_short_translation(self):
        score = qe_length_ratio("This is a long sentence", "Si")
        assert score <= 0.4

    def test_empty_source(self):
        assert qe_length_ratio("", "hola") == 0.0

    def test_empty_translation(self):
        assert qe_length_ratio("hello", "") == 0.0

    def test_both_empty(self):
        assert qe_length_ratio("", "") == 0.0


class TestQEUntranslated:
    def test_clean_spanish(self):
        score = qe_untranslated(
            "God so loved the world",
            "Dios amó tanto al mundo",
        )
        assert score == 1.0

    def test_heavily_untranslated(self):
        # Many English stopwords left in "translation"
        score = qe_untranslated(
            "The Lord is my shepherd",
            "The Lord is my shepherd",  # not translated at all
        )
        assert score < 1.0

    def test_empty_translation(self):
        assert qe_untranslated("hello", "") == 0.0

    def test_few_english_words_ok(self):
        # A single "the" in Spanish text shouldn't flag heavily
        score = qe_untranslated(
            "source text",
            "Esta es una buena traducción the final",  # 1 stopword in 7 words
        )
        assert score >= 0.7


class TestTier1Score:
    def test_good_translation(self):
        result = tier1_score(
            "For God so loved the world",
            "Porque de tal manera amó Dios al mundo",
        )
        assert "length_ratio" in result
        assert "untranslated" in result
        assert "tier1" in result
        assert result["tier1"] >= 0.5

    def test_bad_translation(self):
        # Very short + contains English stopwords = both scores low
        result = tier1_score(
            "This is a very long English sentence with many words",
            "the and with",  # short and untranslated
        )
        assert result["tier1"] < 0.5

    def test_combined_score_is_average(self):
        result = tier1_score("hello", "hola mundo")
        expected = round((result["length_ratio"] + result["untranslated"]) / 2, 3)
        assert result["tier1"] == expected
