"""Tests for diagnostic and QE pure functions in dry_run_ab.py."""

import pytest

# ---------------------------------------------------------------------------
# Fixture: clear diagnostic accumulators before each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_diagnostics():
    """Reset module-level diagnostic lists before each test."""
    import dry_run_ab as d

    d.diag_homophones.clear()
    d.diag_bad_splits.clear()
    d.diag_marian_diverge.clear()
    d.diag_durations.clear()
    d.diag_low_confidence.clear()
    yield
    d.diag_homophones.clear()
    d.diag_bad_splits.clear()
    d.diag_marian_diverge.clear()
    d.diag_durations.clear()
    d.diag_low_confidence.clear()


# ===================================================================
# check_homophones
# ===================================================================


class TestCheckHomophones:
    def test_flags_known_homophone(self):
        import dry_run_ab as d

        d.check_homophones(1, "The rain came down")
        assert len(d.diag_homophones) == 1
        assert d.diag_homophones[0][1] == "rain"
        assert d.diag_homophones[0][2] == "reign"

    def test_no_flag_for_normal_word(self):
        import dry_run_ab as d

        d.check_homophones(1, "God is sovereign")
        assert len(d.diag_homophones) == 0

    def test_strips_punctuation(self):
        import dry_run_ab as d

        d.check_homophones(1, 'the "rain," fell')
        assert len(d.diag_homophones) == 1
        assert d.diag_homophones[0][1] == "rain"

    def test_multiple_flags_in_one_text(self):
        import dry_run_ab as d

        d.check_homophones(1, "The rain and the prey")
        assert len(d.diag_homophones) == 2
        flagged_words = {h[1] for h in d.diag_homophones}
        assert flagged_words == {"rain", "prey"}


# ===================================================================
# check_bad_split
# ===================================================================


class TestCheckBadSplit:
    def test_flags_function_word_ending(self):
        import dry_run_ab as d

        d.check_bad_split(1, "He said that the")
        assert len(d.diag_bad_splits) == 1
        assert d.diag_bad_splits[0][1] == "the"

    def test_no_flag_for_content_word_ending(self):
        import dry_run_ab as d

        d.check_bad_split(1, "God is sovereign")
        assert len(d.diag_bad_splits) == 0

    def test_empty_text(self):
        import dry_run_ab as d

        d.check_bad_split(1, "")
        assert len(d.diag_bad_splits) == 0

    def test_strips_trailing_punctuation(self):
        import dry_run_ab as d

        d.check_bad_split(1, "He went to the,")
        assert len(d.diag_bad_splits) == 1
        assert d.diag_bad_splits[0][1] == "the"


# ===================================================================
# check_marian_divergence
# ===================================================================


class TestCheckMarianDivergence:
    def test_identical_text(self):
        import dry_run_ab as d

        d.check_marian_divergence(1, "el amor de Dios", "el amor de Dios")
        assert len(d.diag_marian_diverge) == 1
        assert d.diag_marian_diverge[0][3] == 1.0

    def test_disjoint_text(self):
        import dry_run_ab as d

        d.check_marian_divergence(1, "hola mundo", "adiós tierra")
        assert len(d.diag_marian_diverge) == 1
        assert d.diag_marian_diverge[0][3] == 0.0

    def test_partial_overlap(self):
        import dry_run_ab as d

        d.check_marian_divergence(1, "el amor de Dios", "el poder de Cristo")
        assert len(d.diag_marian_diverge) == 1
        sim = d.diag_marian_diverge[0][3]
        assert 0.0 < sim < 1.0

    def test_empty_marian_returns_early(self):
        import dry_run_ab as d

        d.check_marian_divergence(1, "", "some text")
        assert len(d.diag_marian_diverge) == 0

    def test_empty_gemma_returns_early(self):
        import dry_run_ab as d

        d.check_marian_divergence(1, "some text", "")
        assert len(d.diag_marian_diverge) == 0


# ===================================================================
# qe_length_ratio
# ===================================================================


class TestQeLengthRatio:
    def test_perfect_range(self):
        import dry_run_ab as d

        # Spanish ~20% longer → ratio ~1.2, within [0.9, 1.6]
        score = d.qe_length_ratio("Hello world", "Hola mundo amigo")
        assert score == 1.0

    def test_short_translation(self):
        import dry_run_ab as d

        # Translation much shorter than source
        score = d.qe_length_ratio("A very long English sentence here", "Hola")
        assert score < 1.0

    def test_very_long_translation(self):
        import dry_run_ab as d

        # Translation way too long
        score = d.qe_length_ratio("Hi", "Una oración extremadamente larga en español que no tiene sentido")
        assert score <= 0.4

    def test_empty_source(self):
        import dry_run_ab as d

        score = d.qe_length_ratio("", "algo")
        assert score == 0.0

    def test_empty_translation(self):
        import dry_run_ab as d

        score = d.qe_length_ratio("hello", "")
        assert score == 0.0


# ===================================================================
# qe_untranslated
# ===================================================================


class TestQeUntranslated:
    def test_clean_spanish(self):
        import dry_run_ab as d

        score = d.qe_untranslated(
            "God so loved the world",
            "Dios amó tanto al mundo",
        )
        assert score == 1.0

    def test_english_heavy_translation(self):
        import dry_run_ab as d

        # Translation is mostly English words
        score = d.qe_untranslated(
            "The Lord is my shepherd",
            "The Lord is my shepherd I shall not want",
        )
        assert score < 0.5

    def test_empty_translation(self):
        import dry_run_ab as d

        score = d.qe_untranslated("Hello", "")
        assert score == 0.0

    def test_moderate_english(self):
        import dry_run_ab as d

        # Some English leaks but mostly Spanish
        score = d.qe_untranslated(
            "The covenant of grace",
            "El pacto de the gracia",
        )
        # "the" is 1 EN word out of 5 → 20% → score 0.4
        assert 0.0 < score < 1.0


# ===================================================================
# qe_score
# ===================================================================


class TestQeScore:
    def test_good_pair(self):
        import dry_run_ab as d

        score = d.qe_score("God is love", "Dios es amor")
        assert 0.5 <= score <= 1.0

    def test_bad_pair(self):
        import dry_run_ab as d

        score = d.qe_score(
            "A very long sentence about the covenant",
            "X",
        )
        assert score < 0.8

    def test_empty_inputs(self):
        import dry_run_ab as d

        score = d.qe_score("", "")
        assert score == 0.0
