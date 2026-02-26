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
    d.diag_empty_stt.clear()
    d.diag_force_cuts.clear()
    d.diag_near_misses.clear()
    yield
    d.diag_homophones.clear()
    d.diag_bad_splits.clear()
    d.diag_marian_diverge.clear()
    d.diag_durations.clear()
    d.diag_low_confidence.clear()
    d.diag_empty_stt.clear()
    d.diag_force_cuts.clear()
    d.diag_near_misses.clear()


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


# ===================================================================
# _compute_force_cut
# ===================================================================


class TestComputeForceCut:
    def test_smart_cut_at_silence_boundary(self):
        from dry_run_ab import _compute_force_cut

        # 8s buffer at 16kHz, silence at 5s mark
        sr = 16000
        buf_len = 8 * sr
        silence_at = 5 * sr
        cut_type, pos = _compute_force_cut(buf_len, sr, silence_at)
        assert cut_type == "smart"
        assert pos == silence_at

    def test_hard_cut_when_no_silence(self):
        from dry_run_ab import _compute_force_cut

        sr = 16000
        buf_len = 8 * sr
        cut_type, pos = _compute_force_cut(buf_len, sr, 0)
        assert cut_type == "hard"
        assert pos == buf_len

    def test_hard_cut_when_silence_too_early(self):
        from dry_run_ab import _compute_force_cut

        sr = 16000
        buf_len = 8 * sr
        # Silence at 0.3s — below min_cut_s default of 0.5
        silence_at = int(0.3 * sr)
        cut_type, pos = _compute_force_cut(buf_len, sr, silence_at)
        assert cut_type == "hard"
        assert pos == buf_len

    def test_smart_cut_at_minimum_boundary(self):
        from dry_run_ab import _compute_force_cut

        sr = 16000
        buf_len = 8 * sr
        # Silence at exactly 0.5s + 1 sample — just above min_cut_s
        silence_at = int(0.5 * sr) + 1
        cut_type, pos = _compute_force_cut(buf_len, sr, silence_at)
        assert cut_type == "smart"
        assert pos == silence_at

    def test_custom_min_cut(self):
        from dry_run_ab import _compute_force_cut

        sr = 16000
        buf_len = 8 * sr
        silence_at = int(0.8 * sr)
        # With min_cut_s=1.0, 0.8s boundary is too early
        cut_type, _ = _compute_force_cut(buf_len, sr, silence_at, min_cut_s=1.0)
        assert cut_type == "hard"


# ===================================================================
# _log_stt_drop
# ===================================================================


class TestLogSttDrop:
    def test_appends_to_accumulator(self, monkeypatch):
        import dry_run_ab as d

        # Prevent actual file I/O
        monkeypatch.setattr(d._io_pool, "submit", lambda fn, *a, **kw: None)

        d._log_stt_drop("partial", 42, 1.5)
        assert len(d.diag_empty_stt) == 1
        assert d.diag_empty_stt[0] == ("partial", 42, 1.5)

    def test_records_stage_correctly(self, monkeypatch):
        import dry_run_ab as d

        monkeypatch.setattr(d._io_pool, "submit", lambda fn, *a, **kw: None)

        d._log_stt_drop("final", 7, 3.2)
        assert d.diag_empty_stt[0][0] == "final"

    def test_submits_io_task(self, monkeypatch):
        import dry_run_ab as d

        submitted = []
        monkeypatch.setattr(d._io_pool, "submit", lambda fn, *a, **kw: submitted.append(fn))

        d._log_stt_drop("partial", 1, 0.8)
        assert len(submitted) == 1


# ===================================================================
# _is_garbage_text
# ===================================================================


class TestIsGarbageText:
    """Tests for the hallucination/garbage filter."""

    # --- True positives: actual garbage that should be caught ---

    def test_single_char_repetition(self):
        from dry_run_ab import _is_garbage_text

        assert _is_garbage_text("aaaaaaa") is True

    def test_single_char_run_in_word(self):
        from dry_run_ab import _is_garbage_text

        assert _is_garbage_text("hellllllo world") is True

    def test_repeating_ngram_word(self):
        from dry_run_ab import _is_garbage_text

        assert _is_garbage_text("gagagagagaga") is True

    def test_repeating_trigram(self):
        from dry_run_ab import _is_garbage_text

        assert _is_garbage_text("abcabcabcabc") is True

    # --- True negatives: legitimate text that must NOT be filtered ---

    def test_short_text_passes(self):
        from dry_run_ab import _is_garbage_text

        assert _is_garbage_text("Amen.") is False

    def test_normal_sentence(self):
        from dry_run_ab import _is_garbage_text

        assert _is_garbage_text("God is love.") is False

    def test_long_sermon_sentence(self):
        from dry_run_ab import _is_garbage_text

        text = "James is going to describe this man who's double-minded, unstable in all his ways."
        assert _is_garbage_text(text) is False

    def test_long_sermon_sentence_2(self):
        from dry_run_ab import _is_garbage_text

        text = "But he doesn't have the works, he doesn't have the fruit to show for the faith that he claims to have."
        assert _is_garbage_text(text) is False

    def test_long_sermon_sentence_3(self):
        from dry_run_ab import _is_garbage_text

        text = "With patience for the rain that comes down from above, the early and the latter rain."
        assert _is_garbage_text(text) is False

    def test_wisdom_sentence(self):
        from dry_run_ab import _is_garbage_text

        text = "Wisdom from above. It's not earthly wisdom, it's not sensual, it's not devilish."
        assert _is_garbage_text(text) is False

    def test_repeated_legitimate_words(self):
        """Legitimate repeated words like 'no no no' should not be garbage."""
        from dry_run_ab import _is_garbage_text

        assert _is_garbage_text("no no no no no") is False

    def test_empty_string(self):
        from dry_run_ab import _is_garbage_text

        assert _is_garbage_text("") is False

    def test_whitespace_only(self):
        from dry_run_ab import _is_garbage_text

        assert _is_garbage_text("   ") is False


# ===================================================================
# _last_final_text (consecutive duplicate suppression)
# ===================================================================


class TestConsecutiveDedupState:
    """Verify the _last_final_text global exists and behaves correctly."""

    def test_initial_value(self):
        import dry_run_ab as d

        assert hasattr(d, "_last_final_text")
        # Should be a string (empty at module load)
        assert isinstance(d._last_final_text, str)


# ===================================================================
# check_near_miss
# ===================================================================


class TestCheckNearMiss:
    """Tests for near-miss STT detection."""

    def test_exact_correction(self):
        import dry_run_ab as d

        d.check_near_miss(1, "The self-exhitation of the wicked")
        assert len(d.diag_near_misses) == 1
        assert d.diag_near_misses[0][1] == "self-exhitation"
        assert d.diag_near_misses[0][2] == "self-exaltation"
        assert d.diag_near_misses[0][3] == "exact"

    def test_phrase_correction(self):
        import dry_run_ab as d

        d.check_near_miss(2, "He began to waver in the beach")
        assert any(nm[3] == "phrase" for nm in d.diag_near_misses)
        phrase_match = next(nm for nm in d.diag_near_misses if nm[3] == "phrase")
        assert phrase_match[2] == "wavering in speech"

    def test_fuzzy_theological_match(self):
        import dry_run_ab as d

        # "reconciliaton" is a close misspelling of "reconciliation"
        d.check_near_miss(3, "The reconciliaton of man")
        fuzzy = [nm for nm in d.diag_near_misses if nm[3] == "fuzzy"]
        assert len(fuzzy) == 1
        assert fuzzy[0][2] == "reconciliation"

    def test_no_flag_for_normal_words(self):
        import dry_run_ab as d

        d.check_near_miss(4, "God is sovereign and good")
        assert len(d.diag_near_misses) == 0

    def test_no_flag_for_exact_theological_term(self):
        import dry_run_ab as d

        # Exact match of a theological term should NOT flag (it's correct)
        d.check_near_miss(5, "The atonement of Christ")
        fuzzy = [nm for nm in d.diag_near_misses if nm[3] == "fuzzy"]
        assert len(fuzzy) == 0

    def test_skips_short_words(self):
        import dry_run_ab as d

        # Short words (< 5 chars) should not trigger fuzzy matching
        d.check_near_miss(6, "The man ran to God")
        assert len(d.diag_near_misses) == 0

    def test_homophone_not_double_flagged(self):
        import dry_run_ab as d

        # Words already in HOMOPHONE_FLAGS should be skipped by check_near_miss
        d.check_near_miss(7, "The rain came down")
        assert len(d.diag_near_misses) == 0

    def test_boosting_homophone(self):
        import dry_run_ab as d

        # "boosting" was added to HOMOPHONE_FLAGS, so check_near_miss should skip it
        d.check_near_miss(8, "boosting about his works")
        assert len(d.diag_near_misses) == 0


# ===================================================================
# Short-phrase whitelist dampening
# ===================================================================


class TestShortPhraseDampening:
    """Tests for _SHORT_PHRASE_WHITELIST priority dampening."""

    def test_whitelist_exists(self):
        import dry_run_ab as d

        assert hasattr(d, "_SHORT_PHRASE_WHITELIST")
        assert "amen" in d._SHORT_PHRASE_WHITELIST
        assert "thank you" in d._SHORT_PHRASE_WHITELIST

    def test_amen_in_whitelist(self):
        import dry_run_ab as d

        assert "amen" in d._SHORT_PHRASE_WHITELIST

    def test_hallelujah_in_whitelist(self):
        import dry_run_ab as d

        assert "hallelujah" in d._SHORT_PHRASE_WHITELIST
