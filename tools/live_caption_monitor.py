#!/usr/bin/env python3
"""
live_caption_monitor.py — YouTube Caption Comparison System (Layer 4)

Compares local Whisper STT output against YouTube's auto-captions to track
whether the local pipeline is outperforming YouTube over time.

Two modes:
  1. Live mode:   Monitor a YouTube livestream in real-time. Capture audio via
                  streamlink, transcribe locally with mlx-whisper, and compare
                  against YouTube's live auto-captions.
  2. Post-stream: After a stream ends, download YouTube auto-subs and compare
                  against saved local transcription CSVs from dry_run_ab.py.

Alignment uses 30-second windowed comparison with sliding-window offset search
(+/-5s) to handle latency drift between YouTube and local transcription.

Interpreting cross-system WER (from CLAUDE.md):
  5-15%:  Comparable or better -- normal disagreement range
  15-25%: Potential issues or genuinely ambiguous segments
  >25%:   Likely real degradation -- flag for review

Output: JSONL to metrics/caption_comparison.jsonl

Usage:
    # Post-stream: compare saved CSV against YouTube auto-subs
    python live_caption_monitor.py post \\
        --video-id VIDEO_ID \\
        --csv metrics/ab_metrics_20260208_183356.csv

    # Live mode: monitor a livestream in real-time
    python live_caption_monitor.py live \\
        --video-id VIDEO_ID \\
        --duration 3600

    # Trend report from existing JSONL data
    python live_caption_monitor.py report
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime

import numpy as np

# ---------------------------------------------------------------------------
# Required dependency: jiwer
# ---------------------------------------------------------------------------
try:
    import jiwer
except ImportError:
    print("ERROR: jiwer is required. Install with: pip install jiwer", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Optional dependencies with helpful error messages
# ---------------------------------------------------------------------------


def _import_youtube_transcript_api():
    """Lazy import for youtube-transcript-api."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        return YouTubeTranscriptApi
    except ImportError:
        print(
            "ERROR: youtube-transcript-api is required for caption fetching.\n"
            "  Install with: pip install youtube-transcript-api",
            file=sys.stderr,
        )
        sys.exit(1)


def _import_mlx_whisper():
    """Lazy import for mlx_whisper."""
    try:
        import mlx.core as mx
        import mlx_whisper

        mx.set_cache_limit(100 * 1024 * 1024)  # prevent Metal cache growth
        return mlx_whisper
    except ImportError:
        print(
            "ERROR: mlx_whisper is required for local transcription.\n  Install with: pip install mlx-whisper",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WHISPER_MODEL = "wbell7/distil-whisper-large-v3.5-mlx"
SAMPLE_RATE = 16000
WINDOW_SECONDS = 30.0  # comparison window size
OFFSET_SEARCH_RANGE = 5.0  # +/- seconds for latency drift alignment
OFFSET_SEARCH_STEP = 0.5  # step size for offset search
POLL_INTERVAL = 7.0  # seconds between YouTube caption polls (live mode)
JSONL_PATH = "metrics/caption_comparison.jsonl"
SESSION_ID = f"{datetime.now():%Y%m%d_%H%M%S}"

# Whisper initial_prompt -- same theological vocabulary bias as dry_run_ab.py
WHISPER_PROMPT = (
    "Sermon at Stark Road Gospel Hall. "
    "God is always capitalized. Christ Jesus, the Holy Spirit, God the Father. "
    "Topics: salvation, atonement, propitiation, mediator, covenant, "
    "righteousness, sanctification, justification, redemption, Savior, "
    "reconciliation, intercession, predestination, sovereignty, "
    "sin, repentance, forgiveness, reign, bow, glory, grace, mercy, "
    "the Gospel, epistle, apostle, tabernacle, congregation, "
    "First Timothy, Second Timothy, First Corinthians, Ephesians, "
    "Romans, Hebrews, Galatians, Colossians, Thessalonians, Philippians, "
    "Genesis, Exodus, Leviticus, Deuteronomy, Isaiah, Jeremiah, Ezekiel, "
    "Scripture, the Lord, the Word of God."
)


# ---------------------------------------------------------------------------
# Text Normalization
# ---------------------------------------------------------------------------


def normalize_text(text):
    """Normalize text for WER/CER comparison.

    Applies the same transformations to both local and YouTube captions
    so that formatting differences don't inflate error rates.
    """
    if not text:
        return ""
    t = text.lower()
    # Remove common transcription artifacts
    t = re.sub(r"\[.*?\]", "", t)  # [Music], [Applause], etc.
    t = re.sub(r"\(.*?\)", "", t)  # (inaudible), etc.
    t = re.sub(r"[^\w\s\']", " ", t)  # keep apostrophes for contractions
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------------------------------------------------------------------
# English Stopwords (for content-word overlap checking)
# ---------------------------------------------------------------------------

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "must",
        "it",
        "its",
        "i",
        "me",
        "my",
        "we",
        "us",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "they",
        "them",
        "their",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "why",
        "not",
        "no",
        "nor",
        "so",
        "if",
        "then",
        "than",
        "too",
        "very",
        "just",
        "about",
        "up",
        "out",
        "all",
        "also",
        "as",
        "into",
        "over",
        "after",
        "before",
        "between",
        "through",
        "there",
        "here",
        "some",
        "any",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "such",
        "only",
        "own",
        "same",
        "now",
    }
)


def get_content_words(text):
    """Extract content words (non-stopwords) from normalized text."""
    return {w for w in text.split() if w not in STOPWORDS and len(w) > 1}


# ---------------------------------------------------------------------------
# YouTube Caption Quality Detection
# ---------------------------------------------------------------------------


def detect_youtube_repetition(text):
    """Detect word-level and phrase-level repetition in YouTube captions.

    YouTube's live auto-caption engine sometimes enters repetition loops,
    producing text like "but what's it mean what's it mean to but what's
    it mean what's it mean to".

    Returns:
        (is_degraded, details_dict) where is_degraded is True if the text
        shows significant repetition patterns.
    """
    words = text.split()
    n_words = len(words)
    details = {
        "word_repeat_ratio": 0.0,
        "phrase_repeat_count": 0,
        "repeated_phrases": [],
    }

    if n_words < 4:
        return False, details

    # --- Word-level repetition: consecutive duplicate words ---
    consecutive_repeats = 0
    for i in range(1, n_words):
        if words[i] == words[i - 1]:
            consecutive_repeats += 1
    word_repeat_ratio = consecutive_repeats / (n_words - 1) if n_words > 1 else 0.0
    details["word_repeat_ratio"] = round(word_repeat_ratio, 3)

    # --- Phrase-level repetition: 3+ word n-grams appearing multiple times ---
    # Check trigrams (3-word sequences)
    phrase_coverage = 0.0
    has_adjacent_repeats = False
    if n_words >= 6:
        trigrams = {}
        trigram_positions = {}  # track positions for adjacency check
        for i in range(n_words - 2):
            gram = " ".join(words[i : i + 3])
            trigrams[gram] = trigrams.get(gram, 0) + 1
            if gram not in trigram_positions:
                trigram_positions[gram] = []
            trigram_positions[gram].append(i)

        repeated_phrases = {g: c for g, c in trigrams.items() if c >= 2}
        details["phrase_repeat_count"] = len(repeated_phrases)
        # Keep the top 3 worst offenders for debugging
        top_repeated = sorted(repeated_phrases.items(), key=lambda x: -x[1])[:3]
        details["repeated_phrases"] = [f"{g} (x{c})" for g, c in top_repeated]

        # Compute what fraction of the text is covered by repeated trigrams
        # A high ratio means the text is mostly repetition
        repeated_word_positions = set()
        for gram, count in repeated_phrases.items():
            if count >= 2:
                gram_words = gram.split()
                for i in range(n_words - 2):
                    if words[i : i + 3] == gram_words:
                        repeated_word_positions.update([i, i + 1, i + 2])
        phrase_coverage = len(repeated_word_positions) / n_words if n_words > 0 else 0.0

        # Check for near-adjacent repeated 4+ word sequences (YouTube stutter signature).
        # Natural text repeats a prefix like "the lord is" but follows with DIFFERENT
        # words ("the lord is my shepherd" vs "the lord is gracious"). YouTube stutters
        # repeat the same 4+ word sequence: "the lord is my the lord is my".
        # We check 4-grams for adjacency to distinguish the two cases.
        if n_words >= 8:
            fourgram_positions = {}
            for i in range(n_words - 3):
                gram4 = " ".join(words[i : i + 4])
                if gram4 not in fourgram_positions:
                    fourgram_positions[gram4] = []
                fourgram_positions[gram4].append(i)

            for gram4, positions in fourgram_positions.items():
                if len(positions) >= 2:
                    for j in range(1, len(positions)):
                        gap = positions[j] - positions[j - 1]
                        if gap <= 8:
                            has_adjacent_repeats = True
                            break
                if has_adjacent_repeats:
                    break

    # --- Decision: flag as degraded if either criterion triggers ---
    # Word-level: >25% of consecutive word pairs are repeats
    # Phrase-level: >50% of words are part of repeated trigrams (general)
    # OR: >35% coverage WITH adjacent repeated trigrams (YouTube stutter pattern)
    is_degraded = (
        word_repeat_ratio > 0.25 or phrase_coverage > 0.50 or (phrase_coverage > 0.35 and has_adjacent_repeats)
    )

    return is_degraded, details


# ---------------------------------------------------------------------------
# WER/CER Computation with Offset Search
# ---------------------------------------------------------------------------


def compute_wer_cer(reference, hypothesis):
    """Compute WER and CER between two normalized text strings.

    Returns (wer, cer, word_diff_ops) where word_diff_ops is a dict of
    {substitutions, deletions, insertions, hits} from jiwer.
    """
    if not reference and not hypothesis:
        return 0.0, 0.0, {"substitutions": 0, "deletions": 0, "insertions": 0, "hits": 0}
    if not reference:
        # All insertions
        n_words = len(hypothesis.split())
        return 1.0, 1.0, {"substitutions": 0, "deletions": 0, "insertions": n_words, "hits": 0}
    if not hypothesis:
        # All deletions
        n_words = len(reference.split())
        return 1.0, 1.0, {"substitutions": 0, "deletions": n_words, "insertions": 0, "hits": 0}

    wer_result = jiwer.process_words(reference, hypothesis)
    wer = wer_result.wer

    cer_result = jiwer.process_characters(reference, hypothesis)
    cer = cer_result.cer

    ops = {
        "substitutions": wer_result.substitutions,
        "deletions": wer_result.deletions,
        "insertions": wer_result.insertions,
        "hits": wer_result.hits,
    }

    return wer, cer, ops


def interpret_wer(wer):
    """Return a human-readable interpretation of cross-system WER."""
    if wer <= 0.05:
        return "excellent"
    elif wer <= 0.15:
        return "comparable_or_better"
    elif wer <= 0.25:
        return "potential_issues"
    else:
        return "likely_degradation"


# ---------------------------------------------------------------------------
# Timed Text Segments
# ---------------------------------------------------------------------------


class TimedSegment:
    """A text segment with a start and end time (seconds from stream start)."""

    __slots__ = ("end", "start", "text")

    def __init__(self, start, end, text):
        self.start = float(start)
        self.end = float(end)
        self.text = text.strip()

    def __repr__(self):
        return f"TimedSegment({self.start:.1f}-{self.end:.1f}: {self.text[:40]})"


def segments_in_window(segments, window_start, window_end):
    """Return concatenated text of all segments overlapping a time window."""
    texts = []
    for seg in segments:
        # Segment overlaps window if it starts before window ends and ends after window starts
        if seg.start < window_end and seg.end > window_start:
            texts.append(seg.text)
    return " ".join(texts)


def find_best_offset(
    local_segments,
    yt_segments,
    window_start,
    window_end,
    search_range=OFFSET_SEARCH_RANGE,
    search_step=OFFSET_SEARCH_STEP,
):
    """Find the time offset that minimizes WER between local and YouTube captions.

    Searches offsets in [-search_range, +search_range] at search_step intervals.
    The offset is applied to YouTube timestamps: yt_time + offset = local_time.

    After finding the best offset, validates alignment by checking content-word
    overlap. If fewer than 5 content words are shared, marks alignment as
    uncertain and falls back to offset=0.

    Returns (best_offset, best_wer, local_text, yt_text_at_best_offset, alignment_uncertain).
    """
    local_text = normalize_text(segments_in_window(local_segments, window_start, window_end))
    if not local_text:
        return 0.0, None, "", "", False

    best_wer = float("inf")
    best_offset = 0.0
    best_yt_text = ""

    offset = -search_range
    while offset <= search_range + 1e-9:
        yt_text = normalize_text(
            segments_in_window(
                yt_segments,
                window_start - offset,
                window_end - offset,
            )
        )
        if yt_text:
            wer, _, _ = compute_wer_cer(local_text, yt_text)
            if wer < best_wer:
                best_wer = wer
                best_offset = offset
                best_yt_text = yt_text
        offset += search_step

    if best_wer == float("inf"):
        return 0.0, None, local_text, "", False

    # --- Alignment validation: check content-word overlap ---
    local_content = get_content_words(local_text)
    yt_content = get_content_words(best_yt_text)
    shared_content = local_content & yt_content
    alignment_uncertain = len(shared_content) < 5

    if alignment_uncertain and abs(best_offset) > 0.01:
        # Fall back to 0 offset since the matched offset looks like a false alignment
        yt_text_zero = normalize_text(segments_in_window(yt_segments, window_start, window_end))
        if yt_text_zero:
            wer_zero, _, _ = compute_wer_cer(local_text, yt_text_zero)
            best_offset = 0.0
            best_wer = wer_zero
            best_yt_text = yt_text_zero

    return best_offset, best_wer, local_text, best_yt_text, alignment_uncertain


def find_global_offset_by_text(
    local_segments,
    yt_segments,
    n_anchors=10,
    phrase_len=6,
    tolerance=3.0,
):
    """Find global time offset using text-based anchor matching.

    When the live session and YouTube video have very different time bases
    (e.g. the live recording started 20 minutes into the stream), the narrow
    ±5s offset search in find_best_offset() will fail. This function searches
    the entire YouTube timeline by matching distinctive phrases from the live
    session against the YouTube transcript.

    Args:
        local_segments: list of TimedSegment from the live session
        yt_segments: list of TimedSegment from YouTube captions
        n_anchors: number of anchor phrases to extract from local segments
        phrase_len: number of words per anchor phrase
        tolerance: seconds within which candidate offsets are clustered

    Returns:
        (offset, confidence) where offset is seconds to add to YouTube
        timestamps so they align with local timestamps, and confidence
        is 0.0-1.0 (fraction of anchors that agreed on the winning offset).
    """
    if not local_segments or not yt_segments:
        return 0.0, 0.0

    # --- Step 1: Pick anchor segments evenly spaced through local session ---
    # Filter to segments with enough words to be distinctive
    viable = [
        s for s in local_segments
        if len(normalize_text(s.text).split()) >= 4
    ]
    if not viable:
        return 0.0, 0.0

    step = max(1, len(viable) // n_anchors)
    anchors = viable[::step][:n_anchors]

    # --- Step 2: Build word-level index from YouTube segments ---
    yt_words = []  # list of (word, timestamp)
    for seg in yt_segments:
        words = normalize_text(seg.text).split()
        if not words:
            continue
        # Interpolate timestamps evenly across words in the segment
        duration = seg.end - seg.start
        for i, w in enumerate(words):
            t = seg.start + duration * (i + 0.5) / len(words)
            yt_words.append((w, t))

    if not yt_words:
        return 0.0, 0.0

    # --- Step 3: For each anchor, find the best match in YouTube ---
    candidate_offsets = []

    for anchor_seg in anchors:
        anchor_text = normalize_text(anchor_seg.text)
        anchor_words = anchor_text.split()
        anchor_time = (anchor_seg.start + anchor_seg.end) / 2.0

        # Use the middle phrase_len words (most distinctive part)
        if len(anchor_words) > phrase_len:
            start_idx = (len(anchor_words) - phrase_len) // 2
            anchor_phrase = anchor_words[start_idx : start_idx + phrase_len]
        else:
            anchor_phrase = anchor_words

        anchor_set = set(anchor_phrase)
        window_len = len(anchor_phrase)
        best_sim = 0.0
        best_yt_time = 0.0

        # Slide through YouTube words
        for i in range(len(yt_words) - window_len + 1):
            window_set = {yt_words[i + j][0] for j in range(window_len)}
            # Jaccard similarity
            intersection = len(anchor_set & window_set)
            union = len(anchor_set | window_set)
            sim = intersection / union if union > 0 else 0.0

            if sim > best_sim:
                best_sim = sim
                # Use midpoint of matched window
                best_yt_time = (
                    yt_words[i][1] + yt_words[i + window_len - 1][1]
                ) / 2.0

        if best_sim > 0.6:
            candidate_offsets.append(anchor_time - best_yt_time)

    if not candidate_offsets:
        return 0.0, 0.0

    # --- Step 4: Cluster candidate offsets and pick the best ---
    candidate_offsets.sort()
    best_cluster = []
    for i, off in enumerate(candidate_offsets):
        cluster = [off]
        for j in range(i + 1, len(candidate_offsets)):
            if abs(candidate_offsets[j] - off) <= tolerance:
                cluster.append(candidate_offsets[j])
            else:
                break
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    confidence = len(best_cluster) / n_anchors
    offset = float(np.median(best_cluster))

    return offset, confidence


# ---------------------------------------------------------------------------
# YouTube Caption Fetching
# ---------------------------------------------------------------------------


def fetch_youtube_captions_api(video_id, language="en"):
    """Fetch captions using youtube-transcript-api (v1.x API).

    Returns a list of TimedSegment objects.
    """
    YouTubeTranscriptApi = _import_youtube_transcript_api()
    api = YouTubeTranscriptApi()

    print(f"  Fetching YouTube captions for {video_id} via transcript API...")
    try:
        transcript_list = api.list(video_id=video_id)
    except Exception as e:
        print(f"  WARNING: Could not list transcripts: {e}", file=sys.stderr)
        return []

    # Prefer auto-generated captions (that is what we are comparing against)
    transcript = None
    try:
        transcript = transcript_list.find_generated_transcript([language])
    except Exception:
        # Fall back to manually created captions if auto-generated not available
        try:
            transcript = transcript_list.find_transcript([language])
        except Exception as e:
            print(f"  WARNING: No {language} transcript found: {e}", file=sys.stderr)
            return []

    fetched = transcript.fetch()
    segments = []
    for entry in fetched:
        start = entry.start
        duration = entry.duration
        text = entry.text
        if text.strip():
            segments.append(TimedSegment(start, start + duration, text))

    print(
        f"  Fetched {len(segments)} YouTube caption segments ({segments[0].start:.1f}s - {segments[-1].end:.1f}s)"
        if segments
        else "  No segments returned"
    )
    return segments


def fetch_youtube_captions_ytdlp(video_id, language="en"):
    """Fetch captions using yt-dlp --write-auto-subs as a fallback.

    Downloads the VTT subtitle file, parses it, and returns TimedSegment objects.
    """
    print(f"  Fetching YouTube captions for {video_id} via yt-dlp...")
    url = f"https://www.youtube.com/watch?v={video_id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_template = os.path.join(tmpdir, "subs")
        cmd = [
            "yt-dlp",
            "--write-auto-subs",
            "--sub-lang",
            language,
            "--sub-format",
            "vtt",
            "--skip-download",
            "-o",
            out_template,
            url,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                print(f"  WARNING: yt-dlp failed: {result.stderr[:200]}", file=sys.stderr)
                return []
        except FileNotFoundError:
            print("  WARNING: yt-dlp not found. Install with: pip install yt-dlp", file=sys.stderr)
            return []
        except subprocess.TimeoutExpired:
            print("  WARNING: yt-dlp timed out", file=sys.stderr)
            return []

        # Find the VTT file
        vtt_files = [f for f in os.listdir(tmpdir) if f.endswith(".vtt")]
        if not vtt_files:
            print("  WARNING: No VTT file produced by yt-dlp", file=sys.stderr)
            return []

        vtt_path = os.path.join(tmpdir, vtt_files[0])
        return parse_vtt(vtt_path)


def parse_vtt(vtt_path):
    """Parse a WebVTT file into TimedSegment objects."""
    segments = []
    timestamp_re = re.compile(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})")

    with open(vtt_path, encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = timestamp_re.match(line)
        if m:
            h1, m1, s1, ms1 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            h2, m2, s2, ms2 = int(m.group(5)), int(m.group(6)), int(m.group(7)), int(m.group(8))
            start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
            end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000

            # Collect text lines until blank line
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip():
                # Remove VTT formatting tags
                clean = re.sub(r"<[^>]+>", "", lines[i].strip())
                if clean:
                    text_lines.append(clean)
                i += 1
            text = " ".join(text_lines)
            if text.strip():
                segments.append(TimedSegment(start, end, text))
        else:
            i += 1

    print(f"  Parsed {len(segments)} segments from VTT")
    return segments


def fetch_youtube_captions(video_id, language="en"):
    """Fetch YouTube captions, trying youtube-transcript-api first, then yt-dlp."""
    segments = fetch_youtube_captions_api(video_id, language)
    if not segments:
        print("  Falling back to yt-dlp for caption fetching...")
        segments = fetch_youtube_captions_ytdlp(video_id, language)
    return segments


# ---------------------------------------------------------------------------
# Local Transcription CSV Loading (Post-Stream Mode)
# ---------------------------------------------------------------------------


def load_local_csv(csv_path):
    """Load local Whisper transcription from a dry_run_ab.py CSV.

    The CSV has columns: chunk_id, timestamp, english, ...
    We parse the timestamp to compute seconds-from-start, treating the first
    chunk's timestamp as t=0.

    Returns a list of TimedSegment objects.
    """
    segments = []
    rows = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("english", "").strip()
            ts_str = row.get("timestamp", "")
            if not text or not ts_str:
                continue
            # Parse ISO timestamp
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                continue
            rows.append((ts, text))

    if not rows:
        print(f"  WARNING: No valid rows in {csv_path}", file=sys.stderr)
        return segments

    # Sort by timestamp
    rows.sort(key=lambda x: x[0])
    t0 = rows[0][0]

    for i, (ts, text) in enumerate(rows):
        start = (ts - t0).total_seconds()
        # Estimate duration from gap to next segment, or use a default
        if i + 1 < len(rows):
            end = (rows[i + 1][0] - t0).total_seconds()
        else:
            # Last segment: estimate ~3 seconds
            end = start + 3.0
        segments.append(TimedSegment(start, end, text))

    print(f"  Loaded {len(segments)} local segments from CSV ({segments[0].start:.1f}s - {segments[-1].end:.1f}s)")
    return segments


# ---------------------------------------------------------------------------
# Live Audio Capture and Transcription
# ---------------------------------------------------------------------------


def capture_and_transcribe_live(video_id, duration_seconds, whisper_prompt=WHISPER_PROMPT):
    """Capture live audio from a YouTube stream via streamlink and transcribe.

    Uses streamlink to pipe audio from the livestream, converts to 16kHz mono
    via ffmpeg, and feeds chunks to mlx-whisper for transcription.

    Returns a list of TimedSegment objects.
    """
    mlx_whisper = _import_mlx_whisper()

    # Check that streamlink is available
    try:
        subprocess.run(["streamlink", "--version"], capture_output=True, timeout=5)
    except FileNotFoundError:
        print(
            "ERROR: streamlink is required for live mode.\n  Install with: pip install streamlink",
            file=sys.stderr,
        )
        sys.exit(1)

    url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"  Opening live stream: {url}")
    print(f"  Duration: {duration_seconds}s ({duration_seconds / 60:.1f} min)")

    # streamlink pipes audio to stdout; ffmpeg converts to raw 16kHz mono PCM
    streamlink_cmd = [
        "streamlink",
        "--stdout",
        url,
        "audio_only",  # or "worst" if audio_only not available
    ]

    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-f",
        "s16le",  # raw signed 16-bit little-endian
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-loglevel",
        "error",
        "pipe:1",
    ]

    segments = []
    chunk_size_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
    bytes_per_sample = 2  # 16-bit
    chunk_bytes = chunk_size_samples * bytes_per_sample

    print("  Starting capture pipeline: streamlink | ffmpeg | mlx-whisper")
    print(f"  Chunk size: {WINDOW_SECONDS}s ({chunk_size_samples} samples)")

    # Warm up Whisper
    print("  Warming up Whisper...")
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    mlx_whisper.transcribe(
        silence,
        path_or_hf_repo=WHISPER_MODEL,
        language="en",
        condition_on_previous_text=False,
    )
    print("  Whisper ready")

    try:
        streamlink_proc = subprocess.Popen(streamlink_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=streamlink_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Allow streamlink_proc to receive SIGPIPE if ffmpeg exits
        streamlink_proc.stdout.close()

        t_start = time.time()
        elapsed = 0.0
        chunk_idx = 0

        while elapsed < duration_seconds:
            raw_bytes = ffmpeg_proc.stdout.read(chunk_bytes)
            if not raw_bytes:
                print("  Stream ended (no more audio data)")
                break

            audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            chunk_start = chunk_idx * WINDOW_SECONDS
            chunk_end = chunk_start + len(audio) / SAMPLE_RATE

            # Transcribe
            t0 = time.perf_counter()
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=WHISPER_MODEL,
                language="en",
                condition_on_previous_text=False,
                initial_prompt=whisper_prompt,
            )
            stt_ms = (time.perf_counter() - t0) * 1000
            text = result["text"].strip()

            if text:
                segments.append(TimedSegment(chunk_start, chunk_end, text))
                print(f"  [{chunk_start:.0f}s-{chunk_end:.0f}s] ({stt_ms:.0f}ms) {text[:80]}")

            chunk_idx += 1
            elapsed = time.time() - t_start

    except KeyboardInterrupt:
        print("\n  Capture interrupted by user")
    finally:
        try:
            ffmpeg_proc.terminate()
        except Exception:
            pass
        try:
            streamlink_proc.terminate()
        except Exception:
            pass

    print(f"  Captured {len(segments)} local segments over {elapsed:.0f}s")
    return segments


# ---------------------------------------------------------------------------
# Windowed Comparison
# ---------------------------------------------------------------------------


def compare_windowed(local_segments, yt_segments, window_size=WINDOW_SECONDS):
    """Compare local and YouTube segments using sliding windows.

    Divides the timeline into windows of window_size seconds and computes
    WER/CER for each window, using offset search to handle latency drift.

    Each window record includes a 'quality_status' field:
      - 'comparable':          Both sides have sufficient content and align well
      - 'youtube_degraded':    YouTube caption shows repetition/stuttering
      - 'alignment_uncertain': Offset search found no meaningful content overlap
      - 'too_few_words':       One or both sides have < 3 words (skipped)

    Only 'comparable' windows contribute to the headline "clean WER" metric.

    Returns a list of per-window comparison records.
    """
    if not local_segments or not yt_segments:
        print("  WARNING: Cannot compare -- one or both segment lists are empty")
        return []

    # Determine the time range to cover
    local_end = max(s.end for s in local_segments)
    yt_end = max(s.end for s in yt_segments)
    total_duration = max(local_end, yt_end)

    records = []
    window_start = 0.0
    n_skipped_few_words = 0

    while window_start < total_duration:
        window_end = window_start + window_size

        best_offset, wer, local_text, yt_text, alignment_uncertain = find_best_offset(
            local_segments, yt_segments, window_start, window_end
        )

        if wer is not None and local_text:
            local_words = len(local_text.split())
            yt_words = len(yt_text.split())

            # --- Window rejection: too few words ---
            if local_words < 3 or yt_words < 3:
                n_skipped_few_words += 1
                window_start += window_size
                continue

            # Also compute CER at the best offset
            _, cer, ops = compute_wer_cer(local_text, yt_text)
            interpretation = interpret_wer(wer)

            # --- Determine quality status ---
            yt_degraded, yt_rep_details = detect_youtube_repetition(yt_text)

            if yt_degraded:
                quality_status = "youtube_degraded"
            elif alignment_uncertain:
                quality_status = "alignment_uncertain"
            else:
                quality_status = "comparable"

            record = {
                "window_start": round(window_start, 1),
                "window_end": round(window_end, 1),
                "wer": round(wer, 4),
                "cer": round(cer, 4),
                "offset_seconds": round(best_offset, 1),
                "local_words": local_words,
                "yt_words": yt_words,
                "ops": ops,
                "interpretation": interpretation,
                "quality_status": quality_status,
                "local_text": local_text[:200],
                "yt_text": yt_text[:200],
            }

            # Attach repetition details for degraded windows (debugging aid)
            if yt_degraded:
                record["yt_repetition"] = yt_rep_details

            records.append(record)

        window_start += window_size

    if n_skipped_few_words:
        print(f"  Skipped {n_skipped_few_words} window(s) with < 3 words on one/both sides")

    return records


# ---------------------------------------------------------------------------
# JSONL Logging
# ---------------------------------------------------------------------------


def write_jsonl(records, video_id, mode, csv_path=None):
    """Append comparison results to the JSONL log file.

    Computes separate aggregates for comparable, youtube_degraded, and
    alignment_uncertain windows. The headline metric is 'clean_wer' which
    only includes comparable windows.
    """
    os.makedirs(os.path.dirname(JSONL_PATH), exist_ok=True)

    session_record = {
        "session_id": SESSION_ID,
        "timestamp": datetime.now(UTC).isoformat(),
        "video_id": video_id,
        "mode": mode,
        "csv_path": csv_path,
        "window_size_seconds": WINDOW_SECONDS,
        "num_windows": len(records),
        "windows": records,
    }

    # Compute session-level aggregates split by quality status
    if records:
        # Partition records by quality status
        comparable = [r for r in records if r.get("quality_status") == "comparable"]
        yt_degraded = [r for r in records if r.get("quality_status") == "youtube_degraded"]
        align_uncertain = [r for r in records if r.get("quality_status") == "alignment_uncertain"]

        # Raw metrics (all windows, for backward compat)
        all_wers = [r["wer"] for r in records]
        all_cers = [r["cer"] for r in records]

        agg = {
            # --- Headline: clean metrics (comparable windows only) ---
            "num_comparable": len(comparable),
            "num_youtube_degraded": len(yt_degraded),
            "num_alignment_uncertain": len(align_uncertain),
        }

        if comparable:
            clean_wers = [r["wer"] for r in comparable]
            clean_cers = [r["cer"] for r in comparable]
            agg["clean_mean_wer"] = round(float(np.mean(clean_wers)), 4)
            agg["clean_median_wer"] = round(float(np.median(clean_wers)), 4)
            agg["clean_std_wer"] = round(float(np.std(clean_wers)), 4)
            agg["clean_min_wer"] = round(float(np.min(clean_wers)), 4)
            agg["clean_max_wer"] = round(float(np.max(clean_wers)), 4)
            agg["clean_p95_wer"] = round(float(np.percentile(clean_wers, 95)), 4)
            agg["clean_mean_cer"] = round(float(np.mean(clean_cers)), 4)
            agg["clean_median_cer"] = round(float(np.median(clean_cers)), 4)

        # --- Secondary: raw metrics (all windows) ---
        agg["raw_mean_wer"] = round(float(np.mean(all_wers)), 4)
        agg["raw_median_wer"] = round(float(np.median(all_wers)), 4)
        agg["raw_std_wer"] = round(float(np.std(all_wers)), 4)
        agg["raw_min_wer"] = round(float(np.min(all_wers)), 4)
        agg["raw_max_wer"] = round(float(np.max(all_wers)), 4)
        agg["raw_p95_wer"] = round(float(np.percentile(all_wers, 95)), 4)
        agg["raw_mean_cer"] = round(float(np.mean(all_cers)), 4)
        agg["raw_median_cer"] = round(float(np.median(all_cers)), 4)

        # WER interpretation buckets (on comparable windows only)
        if comparable:
            clean_wers = [r["wer"] for r in comparable]
            agg["windows_comparable_low_wer"] = sum(1 for w in clean_wers if w <= 0.15)
            agg["windows_comparable_mid_wer"] = sum(1 for w in clean_wers if 0.15 < w <= 0.25)
            agg["windows_comparable_high_wer"] = sum(1 for w in clean_wers if w > 0.25)

        # Backward compat aliases (map to clean when available, raw otherwise)
        if comparable:
            agg["mean_wer"] = agg["clean_mean_wer"]
            agg["median_wer"] = agg["clean_median_wer"]
        else:
            agg["mean_wer"] = agg["raw_mean_wer"]
            agg["median_wer"] = agg["raw_median_wer"]

        session_record["aggregate"] = agg

    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps(session_record) + "\n")

    print(f"  Results written to {JSONL_PATH}")
    return session_record


# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------


def print_session_summary(session_record):
    """Print a detailed summary of a comparison session.

    Shows the quality-filtered breakdown with clean WER as headline metric
    and raw WER as secondary. Separates comparable, YouTube-degraded, and
    alignment-uncertain windows.
    """
    records = session_record.get("windows", [])
    agg = session_record.get("aggregate", {})

    print(f"\n{'=' * 70}")
    print("  CAPTION COMPARISON SUMMARY")
    print(f"  Session: {session_record.get('session_id', 'N/A')}")
    print(f"  Video:   {session_record.get('video_id', 'N/A')}")
    print(f"  Mode:    {session_record.get('mode', 'N/A')}")
    print(f"{'=' * 70}")

    if not records:
        print("  No comparison windows produced.")
        print(f"{'=' * 70}")
        return

    # Partition by quality status
    comparable = [r for r in records if r.get("quality_status") == "comparable"]
    yt_degraded = [r for r in records if r.get("quality_status") == "youtube_degraded"]
    align_uncertain = [r for r in records if r.get("quality_status") == "alignment_uncertain"]
    n = len(records)

    # --- Window quality breakdown ---
    print(f"\n  Window Quality Breakdown ({n} total):")
    if comparable:
        clean_mean = agg.get("clean_mean_wer", 0)
        print(
            f"    Comparable windows:    {len(comparable):>3}/{n} "
            f"({len(comparable) / n:.0%}) -- Mean WER: {clean_mean:.1%}"
        )
    else:
        print(f"    Comparable windows:    {len(comparable):>3}/{n} ({len(comparable) / n:.0%}) -- no clean metrics")
    print(f"    YouTube degraded:      {len(yt_degraded):>3}/{n} ({len(yt_degraded) / n:.0%}) -- excluded from metrics")
    print(
        f"    Alignment uncertain:   {len(align_uncertain):>3}/{n} "
        f"({len(align_uncertain) / n:.0%}) -- excluded from metrics"
    )

    # --- Headline: clean metrics (comparable windows only) ---
    if comparable:
        print(f"\n  Clean Metrics ({len(comparable)} comparable windows):")
        print(f"    Mean WER:   {agg.get('clean_mean_wer', 0):.1%}")
        print(f"    Median WER: {agg.get('clean_median_wer', 0):.1%}")
        print(f"    Std WER:    {agg.get('clean_std_wer', 0):.1%}")
        print(f"    Min WER:    {agg.get('clean_min_wer', 0):.1%}")
        print(f"    Max WER:    {agg.get('clean_max_wer', 0):.1%}")
        print(f"    P95 WER:    {agg.get('clean_p95_wer', 0):.1%}")
        print(f"    Mean CER:   {agg.get('clean_mean_cer', 0):.1%}")
        print(f"    Median CER: {agg.get('clean_median_cer', 0):.1%}")

        # Interpretation breakdown for comparable windows
        n_low = agg.get("windows_comparable_low_wer", 0)
        n_mid = agg.get("windows_comparable_mid_wer", 0)
        n_high = agg.get("windows_comparable_high_wer", 0)
        nc = len(comparable)

        print("\n  WER Interpretation (comparable windows):")
        print(f"    Comparable/better (<=15%): {n_low}/{nc} ({n_low / nc:.0%})")
        print(f"    Potential issues (15-25%): {n_mid}/{nc} ({n_mid / nc:.0%})")
        print(f"    Likely degradation (>25%): {n_high}/{nc} ({n_high / nc:.0%})")
    else:
        print("\n  WARNING: No comparable windows -- all excluded by quality filters")

    # --- Secondary: raw metrics (all windows) ---
    print(f"\n  Raw Metrics ({n} windows, all included):")
    print(f"    Mean WER:   {agg.get('raw_mean_wer', 0):.1%}")
    print(f"    Median WER: {agg.get('raw_median_wer', 0):.1%}")
    print(f"    P95 WER:    {agg.get('raw_p95_wer', 0):.1%}")
    print(f"    Mean CER:   {agg.get('raw_mean_cer', 0):.1%}")

    # --- Per-window detail: best/worst comparable windows ---
    if comparable:
        sorted_comp = sorted(comparable, key=lambda r: r["wer"])

        if len(sorted_comp) > 5:
            print("\n  Best 5 Comparable Windows:")
            for r in sorted_comp[:5]:
                print(
                    f"    [{r['window_start']:.0f}s-{r['window_end']:.0f}s] "
                    f"WER={r['wer']:.1%} CER={r['cer']:.1%} "
                    f"offset={r['offset_seconds']:+.1f}s "
                    f"({r['local_words']}w local, {r['yt_words']}w YT)"
                )

            print("\n  Worst 5 Comparable Windows:")
            for r in sorted_comp[-5:]:
                print(
                    f"    [{r['window_start']:.0f}s-{r['window_end']:.0f}s] "
                    f"WER={r['wer']:.1%} CER={r['cer']:.1%} "
                    f"offset={r['offset_seconds']:+.1f}s "
                    f"({r['local_words']}w local, {r['yt_words']}w YT)"
                )
                if r["wer"] > 0.25:
                    print(f"      Local: {r['local_text'][:100]}")
                    print(f"      YT:    {r['yt_text'][:100]}")
        else:
            print("\n  All Comparable Windows:")
            for r in sorted_comp:
                print(
                    f"    [{r['window_start']:.0f}s-{r['window_end']:.0f}s] "
                    f"WER={r['wer']:.1%} CER={r['cer']:.1%} "
                    f"offset={r['offset_seconds']:+.1f}s"
                )

    # --- Detail on excluded windows for debugging ---
    if yt_degraded:
        print(f"\n  YouTube Degraded Windows ({len(yt_degraded)}):")
        for r in yt_degraded[:3]:  # show first 3 for brevity
            rep = r.get("yt_repetition", {})
            phrases = ", ".join(rep.get("repeated_phrases", [])[:2])
            print(
                f"    [{r['window_start']:.0f}s-{r['window_end']:.0f}s] "
                f"word_repeat={rep.get('word_repeat_ratio', 0):.0%} "
                f"phrase_repeats={rep.get('phrase_repeat_count', 0)}"
            )
            if phrases:
                print(f"      Repeated: {phrases}")
            print(f"      YT text: {r['yt_text'][:120]}")
        if len(yt_degraded) > 3:
            print(f"    ... and {len(yt_degraded) - 3} more")

    if align_uncertain:
        print(f"\n  Alignment Uncertain Windows ({len(align_uncertain)}):")
        for r in align_uncertain[:3]:
            print(
                f"    [{r['window_start']:.0f}s-{r['window_end']:.0f}s] "
                f"offset={r['offset_seconds']:+.1f}s "
                f"({r['local_words']}w local, {r['yt_words']}w YT)"
            )
        if len(align_uncertain) > 3:
            print(f"    ... and {len(align_uncertain) - 3} more")

    # Offset drift analysis (comparable windows only)
    offset_records = comparable if comparable else records
    offsets = [r["offset_seconds"] for r in offset_records]
    label = "comparable" if comparable else "all"
    print(f"\n  Latency Drift Analysis ({label} windows):")
    print(f"    Mean offset:   {np.mean(offsets):+.1f}s")
    print(f"    Std offset:    {np.std(offsets):.1f}s")
    print(f"    Range:         [{min(offsets):+.1f}s, {max(offsets):+.1f}s]")
    if np.std(offsets) > 2.0:
        print("    WARNING: High offset variance -- latency drift detected")

    print(f"\n{'=' * 70}")


def print_trend_report(jsonl_path=JSONL_PATH):
    """Load all historical JSONL data and print a trend analysis."""
    if not os.path.exists(jsonl_path):
        print(f"  No historical data found at {jsonl_path}")
        return

    sessions = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    sessions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not sessions:
        print("  No valid sessions in JSONL")
        return

    print(f"\n{'=' * 70}")
    print(f"  TREND ANALYSIS ({len(sessions)} sessions)")
    print(f"{'=' * 70}")

    # Table header -- show clean WER and quality breakdown
    print(
        f"\n  {'Session':<20} {'Video':<15} {'Mode':<6} {'Win':>4} "
        f"{'Clean WER':>10} {'Raw WER':>8} {'Comp':>5} {'YTDeg':>5} {'Unc':>4}"
    )
    print(f"  {'-' * 20} {'-' * 15} {'-' * 6} {'-' * 4} {'-' * 10} {'-' * 8} {'-' * 5} {'-' * 5} {'-' * 4}")

    clean_wers = []
    for s in sessions:
        sid = s.get("session_id", "?")[:20]
        vid = s.get("video_id", "?")[:15]
        mode = s.get("mode", "?")[:6]
        agg = s.get("aggregate", {})
        n = s.get("num_windows", 0)

        # New-format fields (clean vs raw)
        clean_wer = agg.get("clean_mean_wer", None)
        raw_wer = agg.get("raw_mean_wer", agg.get("mean_wer", None))
        n_comp = agg.get("num_comparable", agg.get("windows_comparable", 0))
        n_ytdeg = agg.get("num_youtube_degraded", 0)
        n_unc = agg.get("num_alignment_uncertain", 0)

        clean_str = f"{clean_wer:.1%}" if clean_wer is not None else "N/A"
        raw_str = f"{raw_wer:.1%}" if raw_wer is not None else "N/A"

        print(
            f"  {sid:<20} {vid:<15} {mode:<6} {n:>4} {clean_str:>10} {raw_str:>8} {n_comp:>5} {n_ytdeg:>5} {n_unc:>4}"
        )

        # Use clean WER for trend if available, otherwise raw
        wer_for_trend = clean_wer if clean_wer is not None else raw_wer
        if wer_for_trend is not None:
            clean_wers.append(wer_for_trend)

    if len(clean_wers) >= 2:
        # Simple trend: is the WER going down over time?
        first_half = clean_wers[: len(clean_wers) // 2]
        second_half = clean_wers[len(clean_wers) // 2 :]
        avg_first = np.mean(first_half)
        avg_second = np.mean(second_half)
        delta = avg_second - avg_first

        print("\n  Trend (using clean WER where available, raw otherwise):")
        print(f"    First half avg WER:  {avg_first:.1%} ({len(first_half)} sessions)")
        print(f"    Second half avg WER: {avg_second:.1%} ({len(second_half)} sessions)")
        if delta < -0.01:
            print(f"    Direction: IMPROVING (WER down {abs(delta):.1%})")
        elif delta > 0.01:
            print(f"    Direction: DEGRADING (WER up {delta:.1%})")
        else:
            print(f"    Direction: STABLE (WER delta {delta:+.1%})")

        # Overall stats
        print(f"\n  Overall ({len(clean_wers)} sessions):")
        print(f"    Mean of means: {np.mean(clean_wers):.1%}")
        print(f"    Best session:  {min(clean_wers):.1%}")
        print(f"    Worst session: {max(clean_wers):.1%}")

    print(f"\n{'=' * 70}")


# ---------------------------------------------------------------------------
# Live Mode: Poll YouTube Captions Periodically
# ---------------------------------------------------------------------------


def poll_youtube_captions_live(video_id, duration_seconds, language="en"):
    """Poll YouTube captions periodically during a livestream.

    Returns accumulated TimedSegment objects. Since YouTube auto-captions
    for a live stream may not be available via the transcript API until
    after the stream, this function polls and accumulates what it can.

    Falls back to post-stream fetching if live captions are not available.
    """
    YouTubeTranscriptApi = _import_youtube_transcript_api()
    api = YouTubeTranscriptApi()

    print(f"  Polling YouTube captions every {POLL_INTERVAL}s for {duration_seconds}s...")
    all_segments = []
    seen_texts = set()  # deduplicate
    t_start = time.time()

    while (time.time() - t_start) < duration_seconds:
        try:
            transcript_list = api.list(video_id=video_id)
            transcript = transcript_list.find_generated_transcript([language])
            fetched = transcript.fetch()

            new_count = 0
            for entry in fetched:
                text = entry.text.strip()
                start = entry.start
                key = f"{start:.1f}:{text}"
                if key not in seen_texts and text:
                    seen_texts.add(key)
                    duration = entry.duration
                    all_segments.append(TimedSegment(start, start + duration, text))
                    new_count += 1

            if new_count > 0:
                print(f"  [{time.time() - t_start:.0f}s] +{new_count} new YT segments (total: {len(all_segments)})")

        except Exception as e:
            print(f"  [{time.time() - t_start:.0f}s] YT poll failed: {e}")

        time.sleep(POLL_INTERVAL)

    # Sort by start time
    all_segments.sort(key=lambda s: s.start)
    print(f"  Collected {len(all_segments)} total YouTube segments")
    return all_segments


# ---------------------------------------------------------------------------
# Mode Handlers
# ---------------------------------------------------------------------------


def run_post_stream(video_id, csv_path, language="en"):
    """Post-stream mode: compare saved CSV against YouTube auto-subs."""
    print("\n  Mode: POST-STREAM")
    print(f"  Video: {video_id}")
    print(f"  CSV:   {csv_path}")

    # Load local segments from CSV
    local_segments = load_local_csv(csv_path)
    if not local_segments:
        print("  ERROR: No local segments loaded from CSV", file=sys.stderr)
        return

    # Fetch YouTube captions
    yt_segments = fetch_youtube_captions(video_id, language)
    if not yt_segments:
        print("  ERROR: No YouTube captions fetched", file=sys.stderr)
        return

    # Compare
    print(f"\n  Comparing {len(local_segments)} local segments against {len(yt_segments)} YouTube segments...")
    records = compare_windowed(local_segments, yt_segments)

    # Log and report
    session = write_jsonl(records, video_id, "post", csv_path=csv_path)
    print_session_summary(session)

    # Check for historical trend data
    if os.path.exists(JSONL_PATH):
        with open(JSONL_PATH) as f:
            n_sessions = sum(1 for line in f if line.strip())
        if n_sessions > 1:
            print_trend_report()


def run_live(video_id, duration_seconds, language="en"):
    """Live mode: capture audio, transcribe locally, compare against YT captions."""
    print("\n  Mode: LIVE")
    print(f"  Video: {video_id}")
    print(f"  Duration: {duration_seconds}s")

    # Run local transcription and YouTube polling in parallel
    # Since live YouTube captions can be tricky, we capture audio + transcribe first,
    # then fetch whatever YouTube captions are available
    print("\n  Phase 1: Capturing and transcribing live audio...")
    local_segments = capture_and_transcribe_live(video_id, duration_seconds)

    if not local_segments:
        print("  ERROR: No local segments captured", file=sys.stderr)
        return

    # Give YouTube a moment to finalize captions, then fetch
    print("\n  Phase 2: Fetching YouTube captions...")
    print("  Waiting 10s for YouTube to finalize captions...")
    time.sleep(10)
    yt_segments = fetch_youtube_captions(video_id, language)

    if not yt_segments:
        # Try polling approach
        print("  Transcript API returned nothing, trying live poll...")
        yt_segments = poll_youtube_captions_live(video_id, min(60, duration_seconds), language)

    if not yt_segments:
        print(
            "  WARNING: No YouTube captions available yet. Saving local transcription for later comparison.",
            file=sys.stderr,
        )
        # Save local segments to a temporary CSV so they can be compared later
        tmp_csv = f"metrics/live_local_{SESSION_ID}.csv"
        os.makedirs("metrics", exist_ok=True)
        with open(tmp_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["chunk_id", "timestamp", "english"])
            t0 = datetime.now()
            for i, seg in enumerate(local_segments):
                writer.writerow([i + 1, t0.isoformat(), seg.text])
        print(f"  Local transcription saved to {tmp_csv}")
        print("  Re-run in post mode once YouTube captions are available:")
        print(f"    python live_caption_monitor.py post --video-id {video_id} --csv {tmp_csv}")
        return

    # Compare
    print(f"\n  Phase 3: Comparing {len(local_segments)} local segments against {len(yt_segments)} YouTube segments...")
    records = compare_windowed(local_segments, yt_segments)

    # Log and report
    session = write_jsonl(records, video_id, "live")
    print_session_summary(session)

    # Check for historical trend data
    if os.path.exists(JSONL_PATH):
        with open(JSONL_PATH) as f:
            n_sessions = sum(1 for line in f if line.strip())
        if n_sessions > 1:
            print_trend_report()


def run_wav(video_id, wav_path, language="en", _chunk_seconds=30.0):
    """WAV mode: transcribe a WAV file with Whisper and compare against YouTube captions."""
    print("\n  Mode: WAV TRANSCRIBE + COMPARE")
    print(f"  Video: {video_id}")
    print(f"  WAV:   {wav_path}")

    # Transcribe WAV locally with mlx-whisper
    import mlx.core as mx
    import mlx_whisper

    mx.set_cache_limit(256 * 1024 * 1024)

    print("\n  Phase 1: Transcribing WAV with mlx-whisper...")
    t0 = time.perf_counter()
    result = mlx_whisper.transcribe(
        wav_path,
        path_or_hf_repo=WHISPER_MODEL,
        language="en",
        word_timestamps=False,
        condition_on_previous_text=True,
        initial_prompt="church sermon gospel Bible God Jesus Christ faith grace prayer",
    )
    stt_ms = (time.perf_counter() - t0) * 1000

    local_segments = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if text:
            local_segments.append(TimedSegment(seg["start"], seg["end"], text))

    audio_duration = local_segments[-1].end if local_segments else 0
    print(
        f"  Transcribed {len(local_segments)} segments in {stt_ms / 1000:.1f}s "
        f"({audio_duration:.0f}s audio, {audio_duration / (stt_ms / 1000):.1f}x realtime)"
    )

    if not local_segments:
        print("  ERROR: No segments transcribed", file=sys.stderr)
        return

    # Fetch YouTube captions
    print("\n  Phase 2: Fetching YouTube captions...")
    yt_segments = fetch_youtube_captions(video_id, language)
    if not yt_segments:
        print("  ERROR: No YouTube captions fetched", file=sys.stderr)
        return

    # Compare
    print(f"\n  Phase 3: Comparing {len(local_segments)} local segments against {len(yt_segments)} YouTube segments...")
    records = compare_windowed(local_segments, yt_segments)

    # Log and report
    session = write_jsonl(records, video_id, "wav", csv_path=wav_path)
    print_session_summary(session)


def run_report():
    """Report mode: show trend analysis from existing JSONL data."""
    print_trend_report()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    global WINDOW_SECONDS, JSONL_PATH

    parser = argparse.ArgumentParser(
        description="YouTube Caption Comparison System (Layer 4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Post-stream comparison
  python live_caption_monitor.py post --video-id dQw4w9WgXcQ \\
      --csv metrics/ab_metrics_20260208_183356.csv

  # Live monitoring (1 hour)
  python live_caption_monitor.py live --video-id dQw4w9WgXcQ --duration 3600

  # Trend report
  python live_caption_monitor.py report

Cross-system WER interpretation:
   5-15%%:  Comparable or better
  15-25%%:  Potential issues
    >25%%:  Likely degradation
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", help="Operating mode")

    # Post-stream mode
    post_parser = subparsers.add_parser("post", help="Post-stream comparison")
    post_parser.add_argument("--video-id", required=True, help="YouTube video ID (e.g., dQw4w9WgXcQ)")
    post_parser.add_argument("--csv", required=True, help="Path to local transcription CSV from dry_run_ab.py")
    post_parser.add_argument("--language", default="en", help="Caption language code (default: en)")
    post_parser.add_argument(
        "--window-size",
        type=float,
        default=WINDOW_SECONDS,
        help=f"Comparison window size in seconds (default: {WINDOW_SECONDS})",
    )
    post_parser.add_argument("--output", default=JSONL_PATH, help=f"Output JSONL path (default: {JSONL_PATH})")

    # Live mode
    live_parser = subparsers.add_parser("live", help="Live stream monitoring")
    live_parser.add_argument("--video-id", required=True, help="YouTube video ID or URL")
    live_parser.add_argument(
        "--duration", type=int, default=3600, help="Duration to monitor in seconds (default: 3600)"
    )
    live_parser.add_argument("--language", default="en", help="Caption language code (default: en)")
    live_parser.add_argument(
        "--window-size",
        type=float,
        default=WINDOW_SECONDS,
        help=f"Comparison window size in seconds (default: {WINDOW_SECONDS})",
    )
    live_parser.add_argument("--output", default=JSONL_PATH, help=f"Output JSONL path (default: {JSONL_PATH})")

    # WAV mode
    wav_parser = subparsers.add_parser("wav", help="Transcribe WAV + compare against YouTube")
    wav_parser.add_argument("--video-id", required=True, help="YouTube video ID for the same sermon")
    wav_parser.add_argument("--wav", required=True, help="Path to WAV file (16kHz mono)")
    wav_parser.add_argument("--language", default="en", help="Caption language code (default: en)")
    wav_parser.add_argument(
        "--window-size",
        type=float,
        default=WINDOW_SECONDS,
        help=f"Comparison window size in seconds (default: {WINDOW_SECONDS})",
    )
    wav_parser.add_argument("--output", default=JSONL_PATH, help=f"Output JSONL path (default: {JSONL_PATH})")

    # Report mode
    report_parser = subparsers.add_parser("report", help="Trend analysis report")
    report_parser.add_argument("--input", default=JSONL_PATH, help=f"Input JSONL path (default: {JSONL_PATH})")

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        sys.exit(1)

    # Apply shared options
    if args.mode in ("post", "live", "wav"):
        WINDOW_SECONDS = args.window_size
        JSONL_PATH = args.output

    print(f"{'=' * 70}")
    print("  Live Caption Monitor — YouTube vs Local Whisper")
    print(f"  Session: {SESSION_ID}")
    print(f"  Output:  {JSONL_PATH}")
    print(f"{'=' * 70}")

    if args.mode == "post":
        if not os.path.exists(args.csv):
            print(f"  ERROR: CSV file not found: {args.csv}", file=sys.stderr)
            sys.exit(1)
        # Extract video ID from URL if full URL was provided
        video_id = extract_video_id(args.video_id)
        run_post_stream(video_id, args.csv, args.language)

    elif args.mode == "live":
        video_id = extract_video_id(args.video_id)
        run_live(video_id, args.duration, args.language)

    elif args.mode == "wav":
        if not os.path.exists(args.wav):
            print(f"  ERROR: WAV file not found: {args.wav}", file=sys.stderr)
            sys.exit(1)
        video_id = extract_video_id(args.video_id)
        run_wav(video_id, args.wav, args.language)

    elif args.mode == "report":
        JSONL_PATH = args.input
        run_report()


def extract_video_id(video_id_or_url):
    """Extract a YouTube video ID from a URL or return as-is if already an ID."""
    # Handle full YouTube URLs
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/live/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        m = re.search(pattern, video_id_or_url)
        if m:
            return m.group(1)

    # Assume it is already a bare video ID
    return video_id_or_url


if __name__ == "__main__":
    main()
