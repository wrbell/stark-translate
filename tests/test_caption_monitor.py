"""Tests for pure utility functions in tools/live_caption_monitor.py.

All functions under test are pure (no I/O, no models, no network).
jiwer is the only external dependency and is installed in CI.
"""

import sys
from unittest.mock import MagicMock

# jiwer is a real dependency for these tests (installed in CI).
# Remove the mock so the real jiwer is used.
if "jiwer" in sys.modules and isinstance(sys.modules["jiwer"], MagicMock):
    del sys.modules["jiwer"]

from tools.live_caption_monitor import (
    STOPWORDS,
    TimedSegment,
    compute_wer_cer,
    detect_youtube_repetition,
    extract_video_id,
    find_global_offset_by_text,
    get_content_words,
    interpret_wer,
    normalize_text,
    segments_in_window,
)

# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_none_returns_empty(self):
        assert normalize_text(None) == ""

    def test_lowercases(self):
        assert normalize_text("Hello World") == "hello world"

    def test_removes_brackets(self):
        assert normalize_text("word [Music] more") == "word more"

    def test_removes_parens(self):
        assert normalize_text("word (inaudible) more") == "word more"

    def test_keeps_apostrophes(self):
        assert "don't" in normalize_text("Don't stop")

    def test_removes_punctuation(self):
        result = normalize_text("Hello, world! How are you?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_collapses_whitespace(self):
        assert normalize_text("too   many   spaces") == "too many spaces"


# ---------------------------------------------------------------------------
# get_content_words
# ---------------------------------------------------------------------------


class TestGetContentWords:
    def test_filters_stopwords(self):
        words = get_content_words("the lord is my shepherd")
        assert "the" not in words
        assert "is" not in words
        assert "lord" in words
        assert "shepherd" in words

    def test_filters_short_words(self):
        words = get_content_words("a b c word")
        assert "a" not in words
        assert "b" not in words
        assert "word" in words

    def test_empty_string(self):
        assert get_content_words("") == set()

    def test_all_stopwords(self):
        words = get_content_words("the and is are")
        assert len(words) == 0

    def test_stopwords_frozenset(self):
        assert isinstance(STOPWORDS, frozenset)
        assert "the" in STOPWORDS


# ---------------------------------------------------------------------------
# detect_youtube_repetition
# ---------------------------------------------------------------------------


class TestDetectYoutubeRepetition:
    def test_normal_text_not_degraded(self):
        is_deg, details = detect_youtube_repetition("the lord is my shepherd i shall not want")
        assert is_deg is False

    def test_short_text_not_degraded(self):
        is_deg, details = detect_youtube_repetition("hi there")
        assert is_deg is False
        assert details["word_repeat_ratio"] == 0.0

    def test_heavy_word_repetition(self):
        text = "the the the the the the the the"
        is_deg, details = detect_youtube_repetition(text)
        assert is_deg is True
        assert details["word_repeat_ratio"] > 0.25

    def test_phrase_repetition_youtube_stutter(self):
        text = "but what's it mean what's it mean to but what's it mean what's it mean to"
        is_deg, details = detect_youtube_repetition(text)
        assert is_deg is True
        assert details["phrase_repeat_count"] > 0

    def test_returns_repeated_phrases(self):
        text = "god is good god is good god is good and he is faithful"
        _, details = detect_youtube_repetition(text)
        assert isinstance(details["repeated_phrases"], list)

    def test_empty_text(self):
        is_deg, details = detect_youtube_repetition("")
        assert is_deg is False


# ---------------------------------------------------------------------------
# compute_wer_cer
# ---------------------------------------------------------------------------


class TestComputeWerCer:
    def test_identical_strings(self):
        wer, cer, ops = compute_wer_cer("hello world", "hello world")
        assert wer == 0.0
        assert cer == 0.0
        assert ops["hits"] == 2

    def test_both_empty(self):
        wer, cer, ops = compute_wer_cer("", "")
        assert wer == 0.0
        assert cer == 0.0

    def test_empty_reference(self):
        wer, cer, ops = compute_wer_cer("", "some words here")
        assert wer == 1.0
        assert ops["insertions"] == 3

    def test_empty_hypothesis(self):
        wer, cer, ops = compute_wer_cer("some words here", "")
        assert wer == 1.0
        assert ops["deletions"] == 3

    def test_partial_match(self):
        wer, cer, ops = compute_wer_cer("the cat sat", "the dog sat")
        assert 0 < wer < 1.0
        assert ops["substitutions"] == 1
        assert ops["hits"] == 2

    def test_ops_dict_keys(self):
        _, _, ops = compute_wer_cer("a", "b")
        assert set(ops.keys()) == {"substitutions", "deletions", "insertions", "hits"}


# ---------------------------------------------------------------------------
# interpret_wer
# ---------------------------------------------------------------------------


class TestInterpretWer:
    def test_excellent(self):
        assert interpret_wer(0.03) == "excellent"

    def test_comparable(self):
        assert interpret_wer(0.10) == "comparable_or_better"

    def test_potential_issues(self):
        assert interpret_wer(0.20) == "potential_issues"

    def test_degradation(self):
        assert interpret_wer(0.30) == "likely_degradation"

    def test_boundary_excellent(self):
        assert interpret_wer(0.05) == "excellent"

    def test_boundary_comparable(self):
        assert interpret_wer(0.15) == "comparable_or_better"

    def test_boundary_potential(self):
        assert interpret_wer(0.25) == "potential_issues"


# ---------------------------------------------------------------------------
# extract_video_id
# ---------------------------------------------------------------------------


class TestExtractVideoId:
    def test_bare_id(self):
        assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_watch_url(self):
        assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self):
        assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_live_url(self):
        assert extract_video_id("https://www.youtube.com/live/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_url_with_extra_params(self):
        result = extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120")
        assert result == "dQw4w9WgXcQ"


# ---------------------------------------------------------------------------
# TimedSegment + segments_in_window
# ---------------------------------------------------------------------------


class TestTimedSegment:
    def test_basic_creation(self):
        seg = TimedSegment(0.0, 5.0, "hello world")
        assert seg.start == 0.0
        assert seg.end == 5.0
        assert seg.text == "hello world"

    def test_strips_whitespace(self):
        seg = TimedSegment(0, 1, "  padded  ")
        assert seg.text == "padded"

    def test_repr(self):
        seg = TimedSegment(1.5, 3.5, "test text")
        r = repr(seg)
        assert "TimedSegment" in r
        assert "1.5" in r


class TestSegmentsInWindow:
    def _segments(self):
        return [
            TimedSegment(0.0, 10.0, "first"),
            TimedSegment(10.0, 20.0, "second"),
            TimedSegment(20.0, 30.0, "third"),
        ]

    def test_full_overlap(self):
        result = segments_in_window(self._segments(), 0.0, 30.0)
        assert "first" in result
        assert "second" in result
        assert "third" in result

    def test_partial_overlap(self):
        result = segments_in_window(self._segments(), 5.0, 15.0)
        assert "first" in result
        assert "second" in result
        assert "third" not in result

    def test_no_overlap(self):
        result = segments_in_window(self._segments(), 35.0, 45.0)
        assert result == ""

    def test_empty_segments(self):
        result = segments_in_window([], 0.0, 30.0)
        assert result == ""


# ---------------------------------------------------------------------------
# find_global_offset_by_text
# ---------------------------------------------------------------------------


class TestFindGlobalOffsetByText:
    """Tests for the text-based anchor alignment function."""

    def _make_segments(self, texts_and_times):
        """Build TimedSegments from list of (start, end, text) tuples."""
        return [TimedSegment(s, e, t) for s, e, t in texts_and_times]

    def test_empty_local_returns_zero(self):
        yt = self._make_segments([(0, 10, "hello world from the stream")])
        offset, confidence = find_global_offset_by_text([], yt)
        assert offset == 0.0
        assert confidence == 0.0

    def test_empty_yt_returns_zero(self):
        local = self._make_segments([(0, 10, "hello world from the sermon")])
        offset, confidence = find_global_offset_by_text(local, [])
        assert offset == 0.0
        assert confidence == 0.0

    def test_identical_text_zero_offset(self):
        """When local and YouTube have the same text at the same times, offset should be ~0."""
        segs = [
            (0, 10, "the lord is my shepherd i shall not want"),
            (10, 20, "he makes me lie down in green pastures"),
            (20, 30, "he leads me beside still waters"),
            (30, 40, "he restores my soul and leads me"),
        ]
        local = self._make_segments(segs)
        yt = self._make_segments(segs)
        offset, confidence = find_global_offset_by_text(local, yt)
        assert abs(offset) < 5.0
        assert confidence > 0.0

    def test_known_positive_offset(self):
        """Local starts at t=600 (wall clock), YouTube has same text at t=0."""
        base_segs = [
            (0, 10, "salvation through faith alone not works"),
            (15, 25, "the righteousness of christ imputed freely"),
            (30, 40, "sanctification proceeds from justification always"),
            (45, 55, "covenant mercy shown to those redeemed"),
            (60, 70, "the gospel proclaimed throughout all nations"),
            (75, 85, "atonement accomplished at calvary completely"),
            (90, 100, "propitiation satisfying divine wrath fully"),
            (105, 115, "mediator between god and men forever"),
            (120, 130, "redemption purchased through precious blood"),
            (135, 145, "intercession made continuously for believers"),
        ]
        time_offset = 600.0  # 10 minutes
        local = self._make_segments([(s + time_offset, e + time_offset, t) for s, e, t in base_segs])
        yt = self._make_segments(base_segs)

        offset, confidence = find_global_offset_by_text(local, yt)
        # Offset should be close to 600s (local_time - yt_time = 600)
        assert abs(offset - time_offset) < 5.0
        assert confidence >= 0.3

    def test_known_negative_offset(self):
        """YouTube starts later than local (unusual but possible)."""
        base_segs = [
            (0, 10, "repentance and faith required for salvation"),
            (15, 25, "predestination according to sovereign purpose"),
            (30, 40, "glorification the final step completed"),
            (45, 55, "perseverance of saints through tribulation"),
            (60, 70, "adoption into the family established firmly"),
        ]
        time_offset = -300.0  # local is 5 minutes before YouTube
        local = self._make_segments([(s + 300, e + 300, t) for s, e, t in base_segs])
        yt = self._make_segments([(s + 600, e + 600, t) for s, e, t in base_segs])

        offset, confidence = find_global_offset_by_text(local, yt)
        # local_time - yt_time = 300 - 600 = -300
        assert abs(offset - time_offset) < 5.0
        assert confidence >= 0.3

    def test_partial_word_mismatch_still_aligns(self):
        """STT differences between local and YouTube shouldn't prevent alignment."""
        time_offset = 1200.0

        local = self._make_segments(
            [
                (time_offset + 0, time_offset + 10, "the lord is my shepherd i shall not want"),
                (time_offset + 15, time_offset + 25, "he makes me to lie down in green pastures"),
                (time_offset + 30, time_offset + 40, "he leads me beside the still waters here"),
                (time_offset + 45, time_offset + 55, "he restores my soul and guides my path"),
                (time_offset + 60, time_offset + 70, "surely goodness and mercy shall follow me"),
            ]
        )
        # YouTube has slight STT differences
        yt = self._make_segments(
            [
                (0, 10, "the lord is my shepherd i shall not want"),
                (15, 25, "he makes me lie down in green pastures"),  # missing "to"
                (30, 40, "he leads me beside still waters"),  # missing "the", "here"
                (45, 55, "he restores my soul and guides my path"),
                (60, 70, "surely goodness and mercy shall follow me"),
            ]
        )

        offset, confidence = find_global_offset_by_text(local, yt)
        assert abs(offset - time_offset) < 5.0
        assert confidence > 0.0

    def test_no_matching_text_returns_zero(self):
        """Completely different texts should yield zero offset with zero confidence."""
        local = self._make_segments(
            [
                (0, 10, "the weather today is sunny and warm"),
                (15, 25, "tomorrow will bring rain and clouds"),
            ]
        )
        yt = self._make_segments(
            [
                (0, 10, "salvation through faith alone by grace"),
                (15, 25, "the righteousness of christ imputed freely"),
            ]
        )

        offset, confidence = find_global_offset_by_text(local, yt)
        assert confidence < 0.3

    def test_short_segments_skipped(self):
        """Segments with fewer than 4 words should be skipped as anchors."""
        local = self._make_segments(
            [
                (0, 5, "yes"),
                (5, 10, "okay then"),
                (10, 15, "uh huh"),
            ]
        )
        yt = self._make_segments(
            [
                (100, 110, "completely different sermon content here"),
            ]
        )

        offset, confidence = find_global_offset_by_text(local, yt)
        assert confidence == 0.0

    def test_large_offset_1800s(self):
        """Test with a 30-minute offset (common for church streams)."""
        time_offset = 1800.0
        base_segs = [
            (0, 10, "the apostle paul wrote to the corinthians"),
            (15, 25, "concerning spiritual gifts and their proper use"),
            (30, 40, "love is patient and love is always kind"),
            (45, 55, "prophecy and tongues will eventually cease"),
            (60, 70, "faith hope and love these three remain"),
        ]
        local = self._make_segments([(s + time_offset, e + time_offset, t) for s, e, t in base_segs])
        yt = self._make_segments(base_segs)

        offset, confidence = find_global_offset_by_text(local, yt)
        assert abs(offset - time_offset) < 5.0
        assert confidence >= 0.3
