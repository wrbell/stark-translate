#!/usr/bin/env python3
"""
validate_session.py — Post-sermon session validation pipeline.

Compares live pipeline results against batch re-processing and YouTube audio
to identify STT/translation quality issues after a session.

5-step pipeline:
  1. Re-translate live session transcription (MarianMT + Gemma 12B)
  2. Download & transcribe YouTube audio (mlx-whisper + VAD)
  3. Translate YouTube transcription (MarianMT + Gemma 4B + Gemma 12B)
  4. Compare results (WER, QE scores, consistency)
  5. Generate findings report (markdown)
  6. (Optional) Incorporate human feedback screenshots

Usage:
    python tools/validate_session.py \\
        --session-csv metrics/ab_metrics_20260301_120127.csv \\
        --video-id VIDEO_ID

    python tools/validate_session.py \\
        --session-csv metrics/ab_metrics_20260301_120127.csv \\
        --video-id VIDEO_ID \\
        --feedback-dir human_feedback_screenshots/26-3-1/ \\
        --output metrics/validation_20260301.md
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Project imports (lazy where needed for heavy ML deps)
# ---------------------------------------------------------------------------

# Add project root to path so we can import engines/tools
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.live_caption_monitor import (
    TimedSegment,
    compare_windowed,
    compute_wer_cer,
    find_best_offset,
    find_global_offset_by_text,
    normalize_text,
    segments_in_window,
)
from tools.translation_qe import tier1_score

logger = logging.getLogger(__name__)

# Whisper prompt — same theological vocabulary bias as dry_run_ab.py
WHISPER_PROMPT = (
    "Sermon at Stark Road Gospel Hall. "
    "Christ Jesus, the Holy Spirit, God the Father. "
    "Atonement, propitiation, mediator, covenant, righteousness, "
    "sanctification, justification, redemption, reconciliation, "
    "repentance, reign, grace, mercy, the Gospel, epistle, apostle, "
    "Scripture, the Lord, the Word of God."
)

SAMPLE_RATE = 16000

# Screenshot filename pattern: "Screenshot YYYY-MM-DD at HH.MM.SS.png"
_SCREENSHOT_RE = re.compile(r"Screenshot\s+(\d{4})-(\d{2})-(\d{2})\s+at\s+(\d{1,2})\.(\d{2})\.(\d{2})")


@dataclass
class AudioSegment:
    """An audio segment with position tracking for temporal alignment."""

    audio: np.ndarray
    start_sample: int
    end_sample: int

    @property
    def start_seconds(self) -> float:
        return self.start_sample / SAMPLE_RATE

    @property
    def end_seconds(self) -> float:
        return self.end_sample / SAMPLE_RATE


# ---------------------------------------------------------------------------
# Step 0: Read session CSV
# ---------------------------------------------------------------------------


def read_session_csv(csv_path: str) -> list[dict]:
    """Read a dry_run_ab.py session CSV and return rows as dicts."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        logger.warning("Session CSV is empty: %s", csv_path)
    return rows


# ---------------------------------------------------------------------------
# Step 1: Re-translate live session transcription
# ---------------------------------------------------------------------------


def retranslate_session(
    rows: list[dict],
    marian_engine: Any,
    gemma_engine: Any | None,
    mlx_executor: ThreadPoolExecutor | None,
) -> list[dict]:
    """Re-translate each English row through MarianMT and optionally Gemma 12B.

    Returns enriched rows with retranslate_marian, retranslate_12b, and QE scores.
    """
    results = []
    total = len(rows)
    for i, row in enumerate(rows):
        english = row.get("english", "").strip()
        if not english:
            results.append(row)
            continue

        logger.info("Re-translating %d/%d: %.60s...", i + 1, total, english)
        enriched = dict(row)

        # MarianMT re-translation (PyTorch, can run on any thread)
        try:
            marian_result = marian_engine.translate(english)
            enriched["retranslate_marian"] = marian_result.text
            enriched["retranslate_marian_ms"] = round(marian_result.latency_ms, 1)
            qe = tier1_score(english, marian_result.text)
            enriched["retranslate_marian_qe"] = qe["tier1"]
        except Exception:
            logger.exception("MarianMT re-translation failed for chunk %s", row.get("chunk_id"))
            enriched["retranslate_marian"] = ""
            enriched["retranslate_marian_ms"] = 0.0
            enriched["retranslate_marian_qe"] = 0.0

        # Gemma 12B re-translation (MLX, must run on single thread)
        if gemma_engine is not None and mlx_executor is not None:
            try:
                future = mlx_executor.submit(gemma_engine.translate, english)
                gemma_result = future.result(timeout=30)
                enriched["retranslate_12b"] = gemma_result.text
                enriched["retranslate_12b_ms"] = round(gemma_result.latency_ms, 1)
                qe = tier1_score(english, gemma_result.text)
                enriched["retranslate_12b_qe"] = qe["tier1"]
            except Exception:
                logger.exception("Gemma 12B re-translation failed for chunk %s", row.get("chunk_id"))
                enriched["retranslate_12b"] = ""
                enriched["retranslate_12b_ms"] = 0.0
                enriched["retranslate_12b_qe"] = 0.0
        else:
            enriched["retranslate_12b"] = ""
            enriched["retranslate_12b_ms"] = 0.0
            enriched["retranslate_12b_qe"] = 0.0

        results.append(enriched)

    return results


# ---------------------------------------------------------------------------
# Step 2: Download & transcribe YouTube audio
# ---------------------------------------------------------------------------


def download_youtube_audio(video_id: str, output_dir: str) -> str | None:
    """Download YouTube audio as 16kHz mono WAV via yt-dlp.

    Returns path to the WAV file, or None on failure.
    """
    if not shutil.which("yt-dlp"):
        logger.error("yt-dlp not found on PATH")
        return None
    if not shutil.which("ffmpeg"):
        logger.error("ffmpeg not found on PATH")
        return None

    url = f"https://www.youtube.com/watch?v={video_id}"
    output_path = os.path.join(output_dir, f"{video_id}.wav")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-x",
        "--audio-format",
        "wav",
        "--postprocessor-args",
        f"ffmpeg:-ar {SAMPLE_RATE} -ac 1",
        "-o",
        output_path,
        url,
    ]
    logger.info("Downloading YouTube audio: %s", video_id)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    except subprocess.CalledProcessError as e:
        logger.error("yt-dlp failed: %s", e.stderr)
        return None
    except subprocess.TimeoutExpired:
        logger.error("yt-dlp timed out after 10 minutes")
        return None

    if os.path.exists(output_path):
        return output_path
    logger.error("Expected output not found: %s", output_path)
    return None


def load_audio_wav(path: str) -> np.ndarray:
    """Load a WAV file as a float32 numpy array at 16kHz."""
    import scipy.io.wavfile as wav

    sr, data = wav.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)

    # Convert stereo to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if sr != SAMPLE_RATE:
        from scipy.signal import resample

        num_samples = int(len(data) * SAMPLE_RATE / sr)
        data = resample(data, num_samples).astype(np.float32)

    return np.clip(data, -1.0, 1.0)


def segment_audio_vad(audio: np.ndarray, min_duration: float = 1.0, max_duration: float = 30.0) -> list[AudioSegment]:
    """Segment audio using Silero VAD into speech chunks.

    Returns list of AudioSegment objects with position tracking for temporal alignment.
    """
    import torch

    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    get_speech_ts = utils[0]

    # Silero VAD expects torch tensor
    audio_tensor = torch.from_numpy(audio)
    timestamps = get_speech_ts(audio_tensor, model, sampling_rate=SAMPLE_RATE)

    segments: list[AudioSegment] = []
    for ts in timestamps:
        start = ts["start"]
        end = ts["end"]
        duration = (end - start) / SAMPLE_RATE
        if duration < min_duration:
            continue
        if duration > max_duration:
            # Split long segments
            chunk_samples = int(max_duration * SAMPLE_RATE)
            for offset in range(start, end, chunk_samples):
                seg_end = min(offset + chunk_samples, end)
                seg = audio[offset:seg_end]
                if len(seg) >= int(min_duration * SAMPLE_RATE):
                    segments.append(AudioSegment(seg, offset, seg_end))
        else:
            segments.append(AudioSegment(audio[start:end], start, end))

    logger.info("VAD found %d speech segments", len(segments))
    return segments


def transcribe_segments(
    segments: list[AudioSegment],
    stt_engine: Any,
    mlx_executor: ThreadPoolExecutor,
) -> list[dict]:
    """Transcribe audio segments using the STT engine.

    Returns list of dicts with 'english', 'confidence', 'segment_index',
    'start_seconds', and 'end_seconds' for temporal alignment.
    """
    results = []
    total = len(segments)
    for i, seg in enumerate(segments):
        audio = seg.audio
        logger.info("Transcribing YouTube segment %d/%d (%.1fs)", i + 1, total, len(audio) / SAMPLE_RATE)
        try:
            future = mlx_executor.submit(
                stt_engine.transcribe,
                audio,
                language="en",
                initial_prompt=WHISPER_PROMPT,
            )
            stt_result = future.result(timeout=60)
            text = stt_result.text.strip()
            if text:
                results.append(
                    {
                        "segment_index": i,
                        "english": text,
                        "confidence": stt_result.confidence,
                        "latency_ms": round(stt_result.latency_ms, 1),
                        "start_seconds": round(seg.start_seconds, 2),
                        "end_seconds": round(seg.end_seconds, 2),
                    }
                )
        except Exception:
            logger.exception("Failed to transcribe segment %d", i)

    logger.info("Transcribed %d/%d segments", len(results), total)
    return results


# ---------------------------------------------------------------------------
# Step 3: Translate YouTube transcription
# ---------------------------------------------------------------------------


def translate_segments(
    yt_segments: list[dict],
    marian_engine: Any,
    gemma_engine: Any | None,
    mlx_executor: ThreadPoolExecutor | None,
    gemma_4b_engine: Any | None = None,
) -> list[dict]:
    """Translate YouTube transcription segments.

    Adds yt_marian, yt_4b, yt_12b, and QE scores to each segment dict.
    """
    total = len(yt_segments)
    for i, seg in enumerate(yt_segments):
        english = seg["english"]
        logger.info("Translating YT segment %d/%d: %.60s...", i + 1, total, english)

        # MarianMT
        try:
            marian_result = marian_engine.translate(english)
            seg["yt_marian"] = marian_result.text
            qe = tier1_score(english, marian_result.text)
            seg["yt_marian_qe"] = qe["tier1"]
        except Exception:
            logger.exception("MarianMT translation failed for YT segment %d", i)
            seg["yt_marian"] = ""
            seg["yt_marian_qe"] = 0.0

        # Gemma 4B
        if gemma_4b_engine is not None and mlx_executor is not None:
            try:
                future = mlx_executor.submit(gemma_4b_engine.translate, english)
                gemma_4b_result = future.result(timeout=30)
                seg["yt_4b"] = gemma_4b_result.text
                qe = tier1_score(english, gemma_4b_result.text)
                seg["yt_4b_qe"] = qe["tier1"]
            except Exception:
                logger.exception("Gemma 4B translation failed for YT segment %d", i)
                seg["yt_4b"] = ""
                seg["yt_4b_qe"] = 0.0
        else:
            seg["yt_4b"] = ""
            seg["yt_4b_qe"] = 0.0

        # Gemma 12B
        if gemma_engine is not None and mlx_executor is not None:
            try:
                future = mlx_executor.submit(gemma_engine.translate, english)
                gemma_result = future.result(timeout=30)
                seg["yt_12b"] = gemma_result.text
                qe = tier1_score(english, gemma_result.text)
                seg["yt_12b_qe"] = qe["tier1"]
            except Exception:
                logger.exception("Gemma 12B translation failed for YT segment %d", i)
                seg["yt_12b"] = ""
                seg["yt_12b_qe"] = 0.0
        else:
            seg["yt_12b"] = ""
            seg["yt_12b_qe"] = 0.0

    return yt_segments


# ---------------------------------------------------------------------------
# Step 4: Compare results
# ---------------------------------------------------------------------------


def compute_wer(reference: str, hypothesis: str) -> float | None:
    """Compute WER between two texts using jiwer."""
    try:
        import jiwer
    except ImportError:
        logger.warning("jiwer not available, skipping WER computation")
        return None

    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return None
    try:
        return round(jiwer.wer(ref, hyp), 4)
    except ValueError:
        return None


def local_rows_to_timed_segments(
    rows: list[dict],
    time_offset: float = 0.0,
) -> list[TimedSegment]:
    """Convert CSV rows from a session to TimedSegment objects.

    Computes seconds-from-start using ISO timestamps in each row.
    time_offset shifts all segments (used to place Speaker 2 after Speaker 1).
    """
    if not rows:
        return []

    # Parse the first timestamp as the reference point
    first_ts = None
    for row in rows:
        ts_str = row.get("timestamp", "")
        if ts_str:
            try:
                first_ts = datetime.fromisoformat(ts_str)
                break
            except ValueError:
                continue
    if first_ts is None:
        return []

    segments = []
    for i, row in enumerate(rows):
        text = row.get("english", "").strip()
        if not text:
            continue
        ts_str = row.get("timestamp", "")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            continue

        start = (ts - first_ts).total_seconds() + time_offset

        # Use utterance_dur if available, otherwise estimate from gap to next row
        dur = None
        try:
            dur = float(row.get("utterance_dur", 0) or 0)
        except (ValueError, TypeError):
            pass
        if not dur or dur <= 0:
            # Estimate from gap to next row with a timestamp
            for nxt in rows[i + 1 :]:
                nxt_str = nxt.get("timestamp", "")
                if nxt_str:
                    try:
                        nxt_ts = datetime.fromisoformat(nxt_str)
                        dur = (nxt_ts - ts).total_seconds()
                        break
                    except ValueError:
                        continue
        if not dur or dur <= 0:
            dur = 3.0  # fallback: assume 3 seconds

        segments.append(TimedSegment(start, start + dur, text))

    return segments


def yt_segments_to_timed_segments(yt_segments: list[dict]) -> list[TimedSegment]:
    """Convert YouTube transcription dicts (with start/end_seconds) to TimedSegment objects."""
    segments = []
    for seg in yt_segments:
        text = seg.get("english", "").strip()
        if not text:
            continue
        start = seg.get("start_seconds")
        end = seg.get("end_seconds")
        if start is not None and end is not None:
            segments.append(TimedSegment(start, end, text))
    return segments


def compare_stt(
    live_rows: list[dict],
    yt_segments: list[dict],
) -> dict:
    """Compare live STT against YouTube re-transcription (legacy full-text mode).

    Uses full text concatenation. Kept for backward compatibility when
    YouTube segments lack timing info.
    """
    live_text = " ".join(r.get("english", "") for r in live_rows if r.get("english"))
    yt_text = " ".join(s.get("english", "") for s in yt_segments if s.get("english"))

    if not live_text or not yt_text:
        return {"stt_wer": None, "live_word_count": 0, "yt_word_count": 0}

    wer = compute_wer(yt_text, live_text)
    return {
        "stt_wer": wer,
        "live_word_count": len(normalize_text(live_text).split()),
        "yt_word_count": len(normalize_text(yt_text).split()),
    }


def compare_stt_aligned(
    speaker_sessions: list[tuple[str, list[dict]]],
    yt_segments: list[dict],
    window_size: float = 30.0,
) -> dict:
    """Compare live STT against YouTube using time-windowed alignment.

    Merges all speakers' rows into TimedSegments with proper time offsets,
    then uses compare_windowed() from live_caption_monitor for per-window WER.

    Args:
        speaker_sessions: list of (label, rows) tuples, one per speaker/session.
        yt_segments: YouTube transcription dicts with start_seconds/end_seconds.
        window_size: comparison window in seconds.

    Returns dict with combined_wer, per_speaker_wer, per_window breakdown,
    and word counts.
    """
    # --- Build local TimedSegments for each speaker, offset by session gaps ---
    all_local_segments: list[TimedSegment] = []
    speaker_ranges: list[tuple[str, float, float]] = []  # (label, start, end)

    cumulative_offset = 0.0
    for label, rows in speaker_sessions:
        segs = local_rows_to_timed_segments(rows, time_offset=cumulative_offset)
        if segs:
            speaker_start = segs[0].start
            speaker_end = segs[-1].end
            speaker_ranges.append((label, speaker_start, speaker_end))
            all_local_segments.extend(segs)
            # Next speaker starts after this one ends (plus any gap)
            cumulative_offset = speaker_end

    if not all_local_segments:
        return {
            "combined_wer": None,
            "per_speaker_wer": {},
            "per_window": [],
            "live_word_count": 0,
            "yt_word_count": 0,
        }

    # --- Build YouTube TimedSegments ---
    yt_timed = yt_segments_to_timed_segments(yt_segments)

    # If YouTube segments lack timing, fall back to full-text comparison
    if not yt_timed:
        all_rows = []
        for _, rows in speaker_sessions:
            all_rows.extend(rows)
        legacy = compare_stt(all_rows, yt_segments)
        return {
            "combined_wer": legacy["stt_wer"],
            "per_speaker_wer": {},
            "per_window": [],
            "live_word_count": legacy["live_word_count"],
            "yt_word_count": legacy["yt_word_count"],
        }

    # --- Find global time offset between local and YouTube ---
    local_end = max(s.end for s in all_local_segments)

    # Always compute text-based anchor offset (cheap, <1s)
    text_offset, text_confidence = find_global_offset_by_text(all_local_segments, yt_timed)
    logger.info(
        "Text-anchor alignment: offset=%.1fs (confidence=%.0f%%)",
        text_offset,
        text_confidence * 100,
    )

    # Try narrow ±5s search (works when timelines already close)
    _offset, _wer, _lt, _yt, _uncertain = find_best_offset(all_local_segments, yt_timed, 0.0, min(local_end, 60.0))
    logger.info(
        "Narrow ±5s search: offset=%.1fs, WER=%s, uncertain=%s",
        _offset,
        _wer,
        _uncertain,
    )

    # Decide which offset to use
    narrow_ok = not _uncertain and _wer is not None and _wer <= 0.6
    text_ok = text_confidence >= 0.3

    if narrow_ok and text_ok and abs(_offset - text_offset) > 10.0:
        # Both methods claim success but disagree — verify on a mid-session sample
        mid = local_end / 2
        sample_start, sample_end = mid - 15, mid + 15

        # WER using narrow offset (YouTube shifted by narrow offset)
        yt_narrow = [TimedSegment(s.start + _offset, s.end + _offset, s.text) for s in yt_timed]
        _, wer_n, _, _, _ = find_best_offset(
            all_local_segments,
            yt_narrow,
            sample_start,
            sample_end,
        )
        # WER using text-anchor offset (YouTube shifted by text offset)
        yt_text = [TimedSegment(s.start + text_offset, s.end + text_offset, s.text) for s in yt_timed]
        _, wer_t, _, _, _ = find_best_offset(
            all_local_segments,
            yt_text,
            sample_start,
            sample_end,
        )
        logger.info(
            "Offset conflict: narrow=%.1fs (mid-WER=%s), text=%.1fs (mid-WER=%s)",
            _offset,
            f"{wer_n * 100:.0f}%" if wer_n is not None else "N/A",
            text_offset,
            f"{wer_t * 100:.0f}%" if wer_t is not None else "N/A",
        )
        if wer_t is not None and (wer_n is None or wer_t < wer_n):
            _offset = text_offset
            logger.info("Validation: text-anchor wins at mid-session")
    elif not narrow_ok:
        if text_ok:
            _offset = text_offset
            logger.info("Using text-anchor offset: %.1fs", _offset)
        # else: both failed, _offset stays at 0 or whatever narrow returned

    # Apply offset to YouTube segments
    if abs(_offset) > 0.01:
        yt_timed = [TimedSegment(s.start + _offset, s.end + _offset, s.text) for s in yt_timed]
        logger.info("Applied global time offset: %.1fs", _offset)

    # --- Per-window comparison ---
    windows = compare_windowed(all_local_segments, yt_timed, window_size=window_size)

    # --- Assign windows to speakers ---
    for w in windows:
        w_mid = (w["window_start"] + w["window_end"]) / 2
        w["speaker"] = None
        for label, sp_start, sp_end in speaker_ranges:
            if sp_start <= w_mid <= sp_end:
                w["speaker"] = label
                break

    # --- Compute per-speaker WER ---
    per_speaker_wer: dict[str, dict] = {}
    for label, sp_start, sp_end in speaker_ranges:
        sp_local = segments_in_window(all_local_segments, sp_start, sp_end)
        sp_yt = segments_in_window(yt_timed, sp_start, sp_end)
        sp_local_norm = normalize_text(sp_local)
        sp_yt_norm = normalize_text(sp_yt)
        if sp_local_norm and sp_yt_norm:
            wer, cer, ops = compute_wer_cer(sp_yt_norm, sp_local_norm)
            per_speaker_wer[label] = {
                "wer": round(wer, 4),
                "cer": round(cer, 4),
                "local_words": len(sp_local_norm.split()),
                "yt_words": len(sp_yt_norm.split()),
            }
        else:
            per_speaker_wer[label] = {
                "wer": None,
                "cer": None,
                "local_words": len(sp_local_norm.split()) if sp_local_norm else 0,
                "yt_words": len(sp_yt_norm.split()) if sp_yt_norm else 0,
            }

    # --- Combined WER ---
    comparable = [w for w in windows if w.get("quality_status") == "comparable"]
    if comparable:
        total_local_words = sum(w["local_words"] for w in comparable)
        # Weighted average WER across comparable windows
        if total_local_words > 0:
            combined_wer = sum(w["wer"] * w["local_words"] for w in comparable) / total_local_words
        else:
            combined_wer = None
    else:
        combined_wer = None

    total_local = sum(len(normalize_text(r.get("english", "")).split()) for _, rows in speaker_sessions for r in rows)
    total_yt = sum(len(normalize_text(s.get("english", "")).split()) for s in yt_segments)

    return {
        "combined_wer": round(combined_wer, 4) if combined_wer is not None else None,
        "per_speaker_wer": per_speaker_wer,
        "per_window": windows,
        "live_word_count": total_local,
        "yt_word_count": total_yt,
        "global_offset": round(_offset, 1),
        "comparable_windows": len(comparable) if comparable else 0,
        "total_windows": len(windows),
    }


def compare_translations(live_rows: list[dict]) -> dict:
    """Compare live translations against re-translations.

    Computes average QE scores and consistency metrics.
    """
    live_qe_a_scores = []
    retranslate_marian_qe_scores = []
    retranslate_12b_qe_scores = []
    marian_agreement = 0
    total_compared = 0

    for row in live_rows:
        english = row.get("english", "").strip()
        if not english:
            continue

        # Live QE scores
        try:
            qe_a = float(row.get("qe_a", 0) or 0)
            if qe_a > 0:
                live_qe_a_scores.append(qe_a)
        except (ValueError, TypeError):
            pass

        # Re-translation QE scores
        try:
            rt_marian_qe = float(row.get("retranslate_marian_qe", 0) or 0)
            if rt_marian_qe > 0:
                retranslate_marian_qe_scores.append(rt_marian_qe)
        except (ValueError, TypeError):
            pass

        try:
            rt_12b_qe = float(row.get("retranslate_12b_qe", 0) or 0)
            if rt_12b_qe > 0:
                retranslate_12b_qe_scores.append(rt_12b_qe)
        except (ValueError, TypeError):
            pass

        # MarianMT consistency (live marian vs re-translate marian)
        live_spanish = row.get("spanish_a", "").strip()
        retranslate_spanish = row.get("retranslate_marian", "").strip()
        if live_spanish and retranslate_spanish:
            total_compared += 1
            # Simple word overlap as consistency proxy
            live_words = set(normalize_text(live_spanish).split())
            rt_words = set(normalize_text(retranslate_spanish).split())
            if live_words and rt_words:
                overlap = len(live_words & rt_words) / max(len(live_words), len(rt_words))
                if overlap > 0.5:
                    marian_agreement += 1

    def _safe_mean(scores: list[float]) -> float | None:
        return round(sum(scores) / len(scores), 3) if scores else None

    return {
        "live_qe_a_mean": _safe_mean(live_qe_a_scores),
        "live_qe_a_count": len(live_qe_a_scores),
        "retranslate_marian_qe_mean": _safe_mean(retranslate_marian_qe_scores),
        "retranslate_12b_qe_mean": _safe_mean(retranslate_12b_qe_scores),
        "marian_consistency_rate": round(marian_agreement / total_compared, 3) if total_compared else None,
        "marian_consistency_count": total_compared,
    }


def find_flagged_segments(live_rows: list[dict], qe_threshold: float = 0.6) -> list[dict]:
    """Find segments with low QE scores or high WER divergence."""
    flagged = []
    for row in live_rows:
        english = row.get("english", "").strip()
        if not english:
            continue

        reasons = []
        chunk_id = row.get("chunk_id", "?")

        # Low live QE
        try:
            qe_a = float(row.get("qe_a", 1) or 1)
            if qe_a < qe_threshold:
                reasons.append(f"low live QE ({qe_a:.2f})")
        except (ValueError, TypeError):
            pass

        # Low re-translation QE
        try:
            rt_qe = float(row.get("retranslate_marian_qe", 1) or 1)
            if rt_qe < qe_threshold:
                reasons.append(f"low MarianMT re-translate QE ({rt_qe:.2f})")
        except (ValueError, TypeError):
            pass

        try:
            rt_12b_qe = float(row.get("retranslate_12b_qe", 1) or 1)
            if rt_12b_qe < qe_threshold:
                reasons.append(f"low Gemma 12B re-translate QE ({rt_12b_qe:.2f})")
        except (ValueError, TypeError):
            pass

        # Low STT confidence
        try:
            conf = float(row.get("stt_confidence", 1) or 1)
            if conf < 0.6:
                reasons.append(f"low STT confidence ({conf:.2f})")
        except (ValueError, TypeError):
            pass

        if reasons:
            flagged.append(
                {
                    "chunk_id": chunk_id,
                    "timestamp": row.get("timestamp", ""),
                    "english": english,
                    "spanish_a": row.get("spanish_a", ""),
                    "retranslate_marian": row.get("retranslate_marian", ""),
                    "retranslate_12b": row.get("retranslate_12b", ""),
                    "reasons": reasons,
                }
            )

    return flagged


# ---------------------------------------------------------------------------
# Step 5: Generate findings report
# ---------------------------------------------------------------------------


def _wer_assessment(wer: float) -> str:
    """Return a human-readable WER assessment string."""
    if wer < 0.15:
        return "Good — within normal disagreement range"
    elif wer < 0.25:
        return "Moderate — some divergence, review flagged segments"
    else:
        return "High divergence — likely real issues"


def _confidence_lines(rows: list[dict]) -> list[str]:
    """Generate confidence distribution lines for a set of rows."""
    confidences = []
    for row in rows:
        try:
            c = float(row.get("stt_confidence", 0) or 0)
            if c > 0:
                confidences.append(c)
        except (ValueError, TypeError):
            pass
    lines: list[str] = []
    if confidences:
        lines.append(
            f"- **Confidence:** mean={np.mean(confidences):.2f}, "
            f"min={np.min(confidences):.2f}, "
            f"max={np.max(confidences):.2f}"
        )
        low_conf = sum(1 for c in confidences if c < 0.6)
        if low_conf:
            lines.append(f"- **Low confidence chunks (<0.6):** {low_conf}/{len(confidences)}")
    return lines


def _translation_lines(tc: dict) -> list[str]:
    """Generate translation quality lines from a comparison dict."""
    lines: list[str] = []
    if tc.get("live_qe_a_mean") is not None:
        lines.append(f"- **Live Gemma QE mean:** {tc['live_qe_a_mean']:.3f} (n={tc['live_qe_a_count']})")
    if tc.get("retranslate_marian_qe_mean") is not None:
        lines.append(f"- **Re-translate MarianMT QE mean:** {tc['retranslate_marian_qe_mean']:.3f}")
    if tc.get("retranslate_12b_qe_mean") is not None:
        lines.append(f"- **Re-translate Gemma 12B QE mean:** {tc['retranslate_12b_qe_mean']:.3f}")
    if tc.get("marian_consistency_rate") is not None:
        lines.append(
            f"- **MarianMT consistency (live vs re-translate):** "
            f"{tc['marian_consistency_rate']:.1%} "
            f"(n={tc['marian_consistency_count']})"
        )
    return lines


def _flagged_lines(flagged_segments: list[dict]) -> list[str]:
    """Generate flagged segment lines."""
    lines: list[str] = []
    if flagged_segments:
        lines.append(f"**{len(flagged_segments)} segments flagged for review:**")
        lines.append("")
        for seg in flagged_segments[:20]:
            lines.append(f"### Chunk {seg['chunk_id']} ({seg['timestamp']})")
            lines.append("")
            lines.append(f"- **English:** {seg['english']}")
            lines.append(f"- **Live Spanish:** {seg['spanish_a']}")
            if seg.get("retranslate_marian"):
                lines.append(f"- **Re-translate (MarianMT):** {seg['retranslate_marian']}")
            if seg.get("retranslate_12b"):
                lines.append(f"- **Re-translate (Gemma 12B):** {seg['retranslate_12b']}")
            lines.append(f"- **Reasons:** {', '.join(seg['reasons'])}")
            lines.append("")
        if len(flagged_segments) > 20:
            lines.append(f"*...and {len(flagged_segments) - 20} more flagged segments.*")
            lines.append("")
    else:
        lines.append("No segments flagged — all passed QE thresholds.")
        lines.append("")
    return lines


def generate_report(
    csv_path: str | list[str],
    live_rows: list[dict],
    stt_comparison: dict,
    translation_comparison: dict | list[dict],
    flagged_segments: list[dict] | list[list[dict]],
    yt_segments: list[dict] | None,
    feedback_matches: list[dict] | None,
    video_id: str | None,
    *,
    speaker_sessions: list[tuple[str, list[dict]]] | None = None,
) -> str:
    """Generate a markdown validation report.

    Supports both single-session (backward compatible) and multi-session modes.
    Multi-session mode is activated when speaker_sessions is provided.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    multi = speaker_sessions is not None and len(speaker_sessions) > 1

    # --- Header ---
    if multi:
        csv_paths = csv_path if isinstance(csv_path, list) else [csv_path]
        lines = [
            "# Multi-Speaker Session Validation Report",
            "",
            f"**Generated:** {now}",
            f"**Sessions:** {len(speaker_sessions)}",
        ]
        total_chunks = 0
        for i, (label, rows) in enumerate(speaker_sessions):
            csv_name = csv_paths[i] if i < len(csv_paths) else "?"
            chunk_count = len(rows)
            total_chunks += chunk_count
            # Determine time range from timestamps
            timestamps = []
            for r in rows:
                ts_str = r.get("timestamp", "")
                if ts_str:
                    try:
                        timestamps.append(datetime.fromisoformat(ts_str))
                    except ValueError:
                        pass
            if timestamps:
                t_start = min(timestamps).strftime("%H:%M:%S")
                t_end = max(timestamps).strftime("%H:%M:%S")
                lines.append(f"- **{label}:** `{csv_name}` - {chunk_count} chunks, {t_start}-{t_end}")
            else:
                lines.append(f"- **{label}:** `{csv_name}` — {chunk_count} chunks")
        lines.append(f"- **Total chunks:** {total_chunks}")
    else:
        csv_str = csv_path if isinstance(csv_path, str) else csv_path[0]
        session_name = Path(csv_str).stem
        chunk_count = len(live_rows)
        lines = [
            f"# Session Validation Report: {session_name}",
            "",
            f"**Generated:** {now}",
            f"**Session CSV:** `{csv_str}`",
            f"**Chunks:** {chunk_count}",
        ]

    if video_id:
        lines.append(f"**YouTube Video:** `{video_id}`")
    lines.append("")

    # --- STT Quality ---
    lines.append("## STT Quality")
    lines.append("")

    # Multi-speaker: per-speaker WER sections + combined
    if multi and stt_comparison.get("per_speaker_wer"):
        per_sp = stt_comparison["per_speaker_wer"]
        for label, sp_data in per_sp.items():
            lines.append(f"### {label}")
            lines.append("")
            if sp_data.get("wer") is not None:
                wer = sp_data["wer"]
                lines.append(f"- **WER:** {wer:.1%}")
                lines.append(f"- **Local words:** {sp_data['local_words']}")
                lines.append(f"- **YouTube words:** {sp_data['yt_words']}")
                lines.append(f"- Assessment: {_wer_assessment(wer)}")
            else:
                lines.append("- WER not available (insufficient aligned text)")
            lines.append("")

        # Find the per-speaker rows for confidence
        for label, rows in speaker_sessions:
            conf_lines = _confidence_lines(rows)
            if conf_lines:
                lines.append(f"**{label} confidence:**")
                lines.extend(conf_lines)
                lines.append("")

        lines.append("### Combined")
        lines.append("")

    combined_wer = stt_comparison.get("combined_wer") or stt_comparison.get("stt_wer")
    if combined_wer is not None:
        lines.append(f"- **Live vs YouTube WER:** {combined_wer:.1%}")
        lines.append(f"- **Live word count:** {stt_comparison.get('live_word_count', 0)}")
        lines.append(f"- **YouTube word count:** {stt_comparison.get('yt_word_count', 0)}")
        if stt_comparison.get("comparable_windows") is not None:
            lines.append(
                f"- **Comparable windows:** {stt_comparison['comparable_windows']}/{stt_comparison['total_windows']}"
            )
        if stt_comparison.get("global_offset") is not None:
            lines.append(f"- **Global time offset:** {stt_comparison['global_offset']:.1f}s")
        lines.append(f"- Assessment: {_wer_assessment(combined_wer)}")
    else:
        lines.append("- STT comparison not available (no YouTube transcription)")

    if not multi:
        conf_lines = _confidence_lines(live_rows)
        if conf_lines:
            lines.append("")
            lines.extend(conf_lines)

    lines.append("")

    # --- Translation Quality ---
    if multi:
        lines.append("## Translation Quality")
        lines.append("")
        # Per-speaker translation quality
        tc_list = translation_comparison if isinstance(translation_comparison, list) else [translation_comparison]
        for i, (label, _rows) in enumerate(speaker_sessions):
            tc = tc_list[i] if i < len(tc_list) else {}
            lines.append(f"### {label}")
            lines.append("")
            tl = _translation_lines(tc)
            if tl:
                lines.extend(tl)
            else:
                lines.append("- No translation quality data available")
            lines.append("")
    else:
        lines.append("## Translation Quality")
        lines.append("")
        tc = translation_comparison if isinstance(translation_comparison, dict) else translation_comparison[0]
        lines.extend(_translation_lines(tc))
        lines.append("")

    # --- YouTube Translation Quality ---
    if yt_segments:
        yt_qe_data: dict[str, list[float]] = {
            "MarianMT": [],
            "Gemma 4B": [],
            "Gemma 12B": [],
        }
        for seg in yt_segments:
            for key, label in [("yt_marian_qe", "MarianMT"), ("yt_4b_qe", "Gemma 4B"), ("yt_12b_qe", "Gemma 12B")]:
                try:
                    val = float(seg.get(key, 0) or 0)
                    if val > 0:
                        yt_qe_data[label].append(val)
                except (ValueError, TypeError):
                    pass

        lines.append("## YouTube Translation Quality")
        lines.append("")
        lines.append("QE scores for translations of YouTube-transcribed audio:")
        lines.append("")
        for label, scores in yt_qe_data.items():
            if scores:
                lines.append(
                    f"- **{label}:** mean={np.mean(scores):.3f}, "
                    f"min={np.min(scores):.3f}, max={np.max(scores):.3f} (n={len(scores)})"
                )
            else:
                lines.append(f"- **{label}:** not available")
        lines.append("")

    # --- Flagged Segments ---
    if multi:
        lines.append("## Flagged Segments")
        lines.append("")
        flagged_list = (
            flagged_segments
            if isinstance(flagged_segments, list) and flagged_segments and isinstance(flagged_segments[0], list)
            else [flagged_segments]
        )
        for i, (label, _rows) in enumerate(speaker_sessions):
            fl = flagged_list[i] if i < len(flagged_list) else []
            lines.append(f"### {label}")
            lines.append("")
            lines.extend(_flagged_lines(fl))
    else:
        lines.append("## Flagged Segments")
        lines.append("")
        fl = flagged_segments if isinstance(flagged_segments, list) else []
        lines.extend(_flagged_lines(fl))

    # --- Human Feedback ---
    if feedback_matches:
        lines.append("## Human-Flagged Segments")
        lines.append("")
        lines.append(f"**{len(feedback_matches)} screenshots correlated with session chunks:**")
        lines.append("")
        for fm in feedback_matches:
            lines.append(f"### {fm['screenshot']}")
            lines.append("")
            lines.append(f"- **Screenshot time:** {fm['screenshot_time']}")
            if fm.get("matched_chunk"):
                mc = fm["matched_chunk"]
                lines.append(f"- **Matched chunk:** {mc.get('chunk_id', '?')} ({mc.get('timestamp', '')})")
                lines.append(f"- **English:** {mc.get('english', '')}")
                lines.append(f"- **Spanish:** {mc.get('spanish_a', '')}")
                try:
                    qe_a = float(mc.get("qe_a", 0) or 0)
                    lines.append(f"- **QE score:** {qe_a:.2f}")
                except (ValueError, TypeError):
                    pass
            else:
                lines.append("- *No matching chunk found by timestamp*")
            lines.append(f"- ![screenshot]({fm['screenshot_path']})")
            lines.append("")

    # --- Recommendations ---
    lines.append("## Recommendations")
    lines.append("")
    recs = []

    # Collect all flagged segments for recommendation counting
    all_flagged = []
    if multi:
        flagged_list = (
            flagged_segments
            if isinstance(flagged_segments, list) and flagged_segments and isinstance(flagged_segments[0], list)
            else [flagged_segments]
        )
        for fl in flagged_list:
            all_flagged.extend(fl)
    else:
        all_flagged = flagged_segments if isinstance(flagged_segments, list) else []

    if all_flagged:
        recs.append(f"- Review {len(all_flagged)} flagged segments for STT/translation errors")
    if combined_wer is not None and combined_wer > 0.25:
        recs.append("- High STT divergence from YouTube — investigate audio quality or model issues")

    # Check translation quality across all speakers
    tc_items = translation_comparison if isinstance(translation_comparison, list) else [translation_comparison]
    for tc_item in tc_items:
        if isinstance(tc_item, dict) and tc_item.get("live_qe_a_mean") is not None and tc_item["live_qe_a_mean"] < 0.7:
            recs.append("- Translation QE below 0.7 — consider model fine-tuning or glossary updates")
            break
    for tc_item in tc_items:
        if (
            isinstance(tc_item, dict)
            and tc_item.get("marian_consistency_rate") is not None
            and tc_item["marian_consistency_rate"] < 0.5
        ):
            recs.append("- Low MarianMT consistency — live inference may have had transient issues")
            break

    if not recs:
        recs.append("- No major issues detected. Session quality looks good.")
    lines.extend(recs)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 6 (Optional): Incorporate human feedback screenshots
# ---------------------------------------------------------------------------


def parse_screenshot_timestamp(filename: str) -> datetime | None:
    """Extract timestamp from screenshot filename.

    Expected format: "Screenshot YYYY-MM-DD at HH.MM.SS.png"
    """
    m = _SCREENSHOT_RE.search(filename)
    if not m:
        return None
    year, month, day, hour, minute, second = m.groups()
    try:
        return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    except ValueError:
        return None


def match_screenshots_to_chunks(
    feedback_dir: str,
    live_rows: list[dict],
    max_offset_seconds: float = 30.0,
) -> list[dict]:
    """Correlate screenshot timestamps with CSV chunk timestamps.

    Returns list of dicts with screenshot info and matched chunk data.
    """
    feedback_path = Path(feedback_dir)
    if not feedback_path.is_dir():
        logger.warning("Feedback directory not found: %s", feedback_dir)
        return []

    screenshots = sorted(feedback_path.glob("*.png"))
    if not screenshots:
        logger.info("No PNG screenshots found in %s", feedback_dir)
        return []

    # Parse chunk timestamps
    chunk_times = []
    for row in live_rows:
        ts_str = row.get("timestamp", "")
        if not ts_str:
            continue
        try:
            # ISO format: "2026-03-01T12:01:47.285906"
            ts = datetime.fromisoformat(ts_str)
            chunk_times.append((ts, row))
        except ValueError:
            continue

    matches = []
    for screenshot_path in screenshots:
        screenshot_time = parse_screenshot_timestamp(screenshot_path.name)
        if screenshot_time is None:
            logger.debug("Could not parse timestamp from: %s", screenshot_path.name)
            continue

        # Find closest chunk by timestamp
        best_match = None
        best_offset = float("inf")
        for chunk_ts, row in chunk_times:
            offset = abs((screenshot_time - chunk_ts).total_seconds())
            if offset < best_offset:
                best_offset = offset
                best_match = row

        match_entry: dict[str, Any] = {
            "screenshot": screenshot_path.name,
            "screenshot_path": str(screenshot_path),
            "screenshot_time": screenshot_time.strftime("%H:%M:%S"),
        }
        if best_match is not None and best_offset <= max_offset_seconds:
            match_entry["matched_chunk"] = best_match
            match_entry["offset_seconds"] = round(best_offset, 1)
        else:
            match_entry["matched_chunk"] = None
            match_entry["offset_seconds"] = None

        matches.append(match_entry)

    logger.info("Matched %d/%d screenshots to chunks", sum(1 for m in matches if m["matched_chunk"]), len(matches))
    return matches


# ---------------------------------------------------------------------------
# KPI scorecard integration
# ---------------------------------------------------------------------------


def _generate_kpi_section(csv_paths: list[str]) -> str | None:
    """Generate a KPI scorecard section for the validation report.

    Imports kpi_report lazily so the validation pipeline doesn't fail if
    the module has issues. Returns markdown string or None on failure.
    """
    try:
        from tools.kpi_report import (
            compute_latency_kpis,
            compute_reliability_kpis,
            compute_stability_kpis,
            compute_translation_kpis,
            compute_wer_kpis,
            find_diagnostics_for_csv,
            generate_scorecard,
            load_csv,
            load_diagnostics,
        )
    except ImportError:
        logger.warning("Could not import kpi_report, skipping KPI scorecard")
        return None

    try:
        all_rows: list[dict] = []
        all_diag: list[dict] = []
        for path in csv_paths:
            all_rows.extend(load_csv(path))
            diag_path = find_diagnostics_for_csv(path)
            if diag_path:
                all_diag.extend(load_diagnostics(diag_path))

        if not all_rows:
            return None

        latency = compute_latency_kpis(all_rows)
        wer = compute_wer_kpis(all_rows, all_diag)
        translation = compute_translation_kpis(all_rows)
        stability = compute_stability_kpis(all_rows, all_diag)
        reliability = compute_reliability_kpis(all_rows, all_diag)
        scorecard = generate_scorecard(latency, wer, translation, stability, reliability)

        lines = [
            "---",
            "",
            "## KPI Scorecard",
            "",
            scorecard,
            "",
        ]
        return "\n".join(lines)
    except Exception:
        logger.warning("KPI scorecard generation failed", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_validation(
    session_csv: str | list[str],
    video_id: str | None = None,
    audio_dir: str | None = None,
    feedback_dir: str | None = None,
    output: str | None = None,
    skip_youtube: bool = False,
    skip_gemma: bool = False,
) -> str:
    """Run the full validation pipeline.

    Accepts one or more session CSVs. Multiple CSVs are treated as separate
    speakers in the same service for time-aligned comparison.

    Returns the path to the generated report.
    """
    from engines.factory import create_stt_engine, create_translation_engine

    # Normalize to list
    csv_paths = [session_csv] if isinstance(session_csv, str) else list(session_csv)

    # --- Step 0: Read all session CSVs ---
    speaker_sessions: list[tuple[str, list[dict]]] = []
    for i, csv_p in enumerate(csv_paths):
        logger.info("Step 0: Reading session CSV: %s", csv_p)
        rows = read_session_csv(csv_p)
        if not rows:
            logger.error("No data in session CSV: %s", csv_p)
            sys.exit(1)
        label = f"Speaker {i + 1}" if len(csv_paths) > 1 else "Speaker"
        logger.info("Read %d chunks from %s (%s)", len(rows), csv_p, label)
        speaker_sessions.append((label, rows))

    multi = len(speaker_sessions) > 1

    # Flatten all rows for operations that need the full set
    all_rows = []
    for _, rows in speaker_sessions:
        all_rows.extend(rows)

    # --- Load engines ---
    logger.info("Loading MarianMT engine...")
    marian_engine = create_translation_engine(engine_type="marian")
    marian_engine.load()

    mlx_executor = ThreadPoolExecutor(max_workers=1)

    gemma_4b_engine = None
    gemma_engine = None
    if not skip_gemma:
        logger.info("Loading Gemma 4B engine...")
        try:
            gemma_4b_engine = create_translation_engine(model_id="mlx-community/translategemma-4b-it-4bit")
            gemma_4b_engine.load()
        except Exception:
            logger.warning("Could not load Gemma 4B, skipping 4B translation", exc_info=True)
            gemma_4b_engine = None

        logger.info("Loading Gemma 12B engine...")
        try:
            gemma_engine = create_translation_engine(model_id="mlx-community/translategemma-12b-it-4bit")
            gemma_engine.load()
        except Exception:
            logger.warning("Could not load Gemma 12B, skipping re-translation", exc_info=True)
            gemma_engine = None

    stt_engine = None
    if video_id and not skip_youtube:
        logger.info("Loading STT engine...")
        stt_engine = create_stt_engine()
        stt_engine.load()

    # --- Step 1: Re-translate each speaker ---
    logger.info("Step 1: Re-translating live session(s)...")
    retranslated_sessions: list[tuple[str, list[dict]]] = []
    for label, rows in speaker_sessions:
        logger.info("Re-translating %s (%d chunks)...", label, len(rows))
        enriched = retranslate_session(rows, marian_engine, gemma_engine, mlx_executor)
        retranslated_sessions.append((label, enriched))
    speaker_sessions = retranslated_sessions

    # Rebuild flat list after re-translation
    all_rows = []
    for _, rows in speaker_sessions:
        all_rows.extend(rows)

    # --- Step 2 & 3: YouTube download, transcribe, translate (once) ---
    yt_segments: list[dict] | None = None
    stt_comparison: dict = {
        "combined_wer": None,
        "stt_wer": None,
        "per_speaker_wer": {},
        "per_window": [],
        "live_word_count": 0,
        "yt_word_count": 0,
    }

    if video_id and stt_engine and not skip_youtube:
        with tempfile.TemporaryDirectory() as tmpdir:
            download_dir = audio_dir or tmpdir
            logger.info("Step 2: Downloading YouTube audio...")
            wav_path = download_youtube_audio(video_id, download_dir)
            if wav_path:
                logger.info("Loading audio from %s", wav_path)
                audio = load_audio_wav(wav_path)
                logger.info("Audio loaded: %.1f seconds", len(audio) / SAMPLE_RATE)

                logger.info("Segmenting with VAD...")
                segments = segment_audio_vad(audio)

                logger.info("Transcribing %d segments...", len(segments))
                yt_segments = transcribe_segments(segments, stt_engine, mlx_executor)

                if yt_segments:
                    logger.info("Step 3: Translating YouTube transcription...")
                    yt_segments = translate_segments(
                        yt_segments, marian_engine, gemma_engine, mlx_executor, gemma_4b_engine
                    )
            else:
                logger.warning("YouTube download failed, skipping steps 2-3")

    # --- Step 4: Compare ---
    logger.info("Step 4: Comparing results...")
    if yt_segments:
        stt_comparison = compare_stt_aligned(speaker_sessions, yt_segments)

    # Per-speaker translation comparison and flagged segments
    translation_comparisons: list[dict] = []
    flagged_per_speaker: list[list[dict]] = []
    for label, rows in speaker_sessions:
        translation_comparisons.append(compare_translations(rows))
        flagged_per_speaker.append(find_flagged_segments(rows))

    # --- Step 6 (optional): Human feedback ---
    feedback_matches = None
    if feedback_dir:
        logger.info("Step 6: Matching human feedback screenshots...")
        feedback_matches = match_screenshots_to_chunks(feedback_dir, all_rows)

    # --- Step 5: Generate report ---
    logger.info("Step 5: Generating report...")
    if multi:
        report = generate_report(
            csv_paths,
            all_rows,
            stt_comparison,
            translation_comparisons,
            flagged_per_speaker,
            yt_segments,
            feedback_matches,
            video_id,
            speaker_sessions=speaker_sessions,
        )
    else:
        # Single session — backward compatible format
        report = generate_report(
            csv_paths[0],
            all_rows,
            stt_comparison,
            translation_comparisons[0],
            flagged_per_speaker[0],
            yt_segments,
            feedback_matches,
            video_id,
        )

    # Determine output path
    if not output:
        if multi:
            # Use date from first CSV
            first_name = Path(csv_paths[0]).stem.replace("ab_metrics_", "")
            # Extract date portion (YYYYMMDD)
            date_part = first_name.split("_")[0] if "_" in first_name else first_name
            output = f"metrics/validation_{date_part}_combined.md"
        else:
            session_name = Path(csv_paths[0]).stem.replace("ab_metrics_", "")
            output = f"metrics/validation_{session_name}.md"

    # --- KPI Scorecard ---
    logger.info("Generating KPI scorecard...")
    kpi_section = _generate_kpi_section(csv_paths)
    if kpi_section:
        report += "\n" + kpi_section

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Report written to %s", output)

    # Cleanup
    mlx_executor.shutdown(wait=False)
    if stt_engine:
        stt_engine.unload()
    if gemma_4b_engine:
        gemma_4b_engine.unload()
    if gemma_engine:
        gemma_engine.unload()
    marian_engine.unload()

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Post-sermon session validation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--session-csv",
        required=True,
        nargs="+",
        help="Path(s) to session CSV(s) from dry_run_ab.py. "
        "Multiple paths are treated as separate speakers in the same service.",
    )
    parser.add_argument(
        "--video-id",
        default=None,
        help="YouTube video ID for audio comparison",
    )
    parser.add_argument(
        "--audio-dir",
        default=None,
        help="Directory to save/load YouTube audio (default: temp dir)",
    )
    parser.add_argument(
        "--feedback-dir",
        default=None,
        help="Directory with timestamped screenshot PNGs for human feedback correlation",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for markdown report (default: metrics/validation_{session}.md)",
    )
    parser.add_argument(
        "--skip-youtube",
        action="store_true",
        help="Skip YouTube download/transcription (re-translate only)",
    )
    parser.add_argument(
        "--skip-gemma",
        action="store_true",
        help="Skip Gemma 12B re-translation (MarianMT only)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    for csv_p in args.session_csv:
        if not os.path.isfile(csv_p):
            logger.error("Session CSV not found: %s", csv_p)
            sys.exit(1)

    # Single CSV → pass as string for backward compat; multiple → pass as list
    session_input = args.session_csv[0] if len(args.session_csv) == 1 else args.session_csv

    output_path = run_validation(
        session_csv=session_input,
        video_id=args.video_id,
        audio_dir=args.audio_dir,
        feedback_dir=args.feedback_dir,
        output=args.output,
        skip_youtube=args.skip_youtube,
        skip_gemma=args.skip_gemma,
    )
    print(f"\nValidation report: {output_path}")


if __name__ == "__main__":
    main()
