#!/usr/bin/env python3
"""
prepare_piper_dataset.py — Convert Church Audio to Piper/LJSpeech Training Format

Converts cleaned church audio (WAVs + transcripts) into Piper-compatible LJSpeech
format for TTS voice fine-tuning. Supports multiple input sources:

  1. Preprocessed chunks: stark_data/cleaned/chunks/ + stark_data/transcripts/
  2. Live session output: stark_data/live_sessions/{session}/ (chunk WAVs + diagnostics JSONL)

Output: LJSpeech-style directory ready for Piper training:
    stark_data/piper_dataset/{lang}/
    +-- wav/           # 22050 Hz mono 16-bit PCM audio clips
    +-- metadata.csv   # filename|transcription|normalized_transcription

Piper expects 22050 Hz mono 16-bit PCM WAVs with clips in the 1-15 second range.
Longer clips degrade attention alignment during VITS training.

Usage:
    python prepare_piper_dataset.py
    python prepare_piper_dataset.py --lang en --input stark_data/cleaned/chunks
    python prepare_piper_dataset.py --lang es --live-sessions stark_data/live_sessions
    python prepare_piper_dataset.py --min-confidence 0.7 --max-duration 10
    python prepare_piper_dataset.py --lang en --input stark_data/cleaned/chunks --live-sessions stark_data/live_sessions
"""

import argparse
import csv
import json
import logging
import os
import struct
import wave
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Piper's default sample rate for VITS TTS training
PIPER_SAMPLE_RATE = 22050

# Minimum text length to include (very short utterances hurt alignment)
MIN_TEXT_LENGTH = 5


def read_wav_info(wav_path):
    """Read WAV file metadata without loading full audio into memory.

    Returns (sample_rate, n_channels, n_frames, sample_width) or None on error.
    """
    try:
        with wave.open(str(wav_path), "rb") as wf:
            return (
                wf.getframerate(),
                wf.getnchannels(),
                wf.getnframes(),
                wf.getsampwidth(),
            )
    except Exception:
        return None


def get_wav_duration(wav_path):
    """Get duration in seconds from a WAV file."""
    info = read_wav_info(wav_path)
    if info is None:
        return 0.0
    sr, _, n_frames, _ = info
    return n_frames / sr if sr > 0 else 0.0


def resample_wav(input_path, output_path, target_sr=PIPER_SAMPLE_RATE):
    """Resample a WAV file to target sample rate, mono, 16-bit PCM.

    Uses scipy.signal.resample for high-quality resampling without requiring
    ffmpeg or librosa as dependencies. Falls back to simple linear interpolation
    if scipy is not available.
    """
    with wave.open(str(input_path), "rb") as wf:
        src_sr = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        sample_width = wf.getsampwidth()
        raw_data = wf.readframes(n_frames)

    # Decode raw bytes to numpy array
    if sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    elif sample_width == 1:
        dtype = np.uint8
    else:
        logger.warning(f"Unsupported sample width {sample_width} in {input_path}")
        return False

    audio = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)

    # Convert to mono if stereo
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    # Normalize to [-1, 1] range for processing
    if dtype == np.int16:
        audio /= 32768.0
    elif dtype == np.int32:
        audio /= 2147483648.0
    elif dtype == np.uint8:
        audio = (audio - 128.0) / 128.0

    # Resample if needed
    if src_sr != target_sr:
        try:
            from scipy.signal import resample as scipy_resample

            target_length = int(len(audio) * target_sr / src_sr)
            audio = scipy_resample(audio, target_length)
        except ImportError:
            # Fallback: linear interpolation (lower quality but no extra deps)
            target_length = int(len(audio) * target_sr / src_sr)
            x_old = np.linspace(0, 1, len(audio))
            x_new = np.linspace(0, 1, target_length)
            audio = np.interp(x_new, x_old, audio)

    # Clip and convert back to 16-bit PCM
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    # Write output WAV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(target_sr)
        wf.writeframes(audio_int16.tobytes())

    return True


def normalize_text(text):
    """Basic text normalization for the normalized_transcription column.

    Piper's LJSpeech format has a third column for normalized text (numbers
    expanded, abbreviations spelled out, etc.). We apply conservative
    normalization here; Piper's own preprocessing handles phonemization.
    """
    text = text.strip()
    # Remove multiple spaces
    text = " ".join(text.split())
    # Remove leading/trailing quotes if wrapping the whole string
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    return text


# ---------------------------------------------------------------------------
# Source 1: Preprocessed chunks + transcript JSONs
# ---------------------------------------------------------------------------

def load_from_chunks(chunks_dir, transcripts_dir, min_confidence, min_dur, max_dur):
    """Load entries from preprocessed audio chunks and their transcript JSONs.

    Mirrors the loading logic in prepare_whisper_dataset.py: reads transcript
    JSONs from transcripts_dir, resolves audio paths in chunks_dir, and
    applies confidence + duration filtering.

    Returns list of dicts: {audio_path, text, confidence}
    """
    chunks_path = Path(chunks_dir)
    transcripts_path = Path(transcripts_dir)
    entries = []

    if not transcripts_path.exists():
        logger.warning(f"Transcripts directory not found: {transcripts_path}")
        return entries

    # Collect all transcript JSONs (including accent subdirs)
    json_files = sorted(transcripts_path.glob("*.json"))
    json_files += sorted(transcripts_path.glob("*/*.json"))
    json_files = [f for f in json_files if not f.name.startswith("_")]

    logger.info(f"Found {len(json_files)} transcript files in {transcripts_path}")

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                transcript = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Skipping {json_path.name}: {e}")
            continue

        text = transcript.get("text", "").strip()
        if not text or len(text) < MIN_TEXT_LENGTH:
            continue

        # Resolve audio path
        audio_path_str = transcript.get("audio_path", "")
        audio_path = None
        if audio_path_str:
            audio_path = Path(audio_path_str)
            if not audio_path.exists():
                # Try relative to chunks_dir
                for pattern in [f"*/{audio_path.name}", audio_path.name]:
                    candidates = list(chunks_path.glob(pattern))
                    if candidates:
                        audio_path = candidates[0]
                        break
                else:
                    audio_path = None

        if audio_path is None:
            wav_name = json_path.stem + ".wav"
            candidates = list(chunks_path.glob(f"*/{wav_name}"))
            candidates += list(chunks_path.glob(wav_name))
            if candidates:
                audio_path = candidates[0]
            else:
                continue

        if not audio_path.exists():
            continue

        # Duration filter
        duration = get_wav_duration(audio_path)
        if duration < min_dur or duration > max_dur:
            continue

        # Confidence filter from segment metadata
        segments = transcript.get("segments", [])
        avg_logprob = None
        if segments and isinstance(segments[0], dict):
            logprobs = [s.get("avg_logprob") for s in segments
                        if s.get("avg_logprob") is not None]
            if logprobs:
                avg_logprob = sum(logprobs) / len(logprobs)

        if avg_logprob is not None and avg_logprob < min_confidence:
            continue

        entries.append({
            "audio_path": str(audio_path.resolve()),
            "text": text,
            "confidence": avg_logprob,
        })

    return entries


# ---------------------------------------------------------------------------
# Source 2: Live session output (chunk WAVs + diagnostics JSONL)
# ---------------------------------------------------------------------------

def load_from_live_sessions(sessions_dir, lang, min_confidence, min_dur, max_dur):
    """Load entries from live pipeline output directories.

    Live sessions are stored as:
        stark_data/live_sessions/{session_id}/
            chunk_0001.wav, chunk_0002.wav, ...

    Diagnostics are stored in:
        metrics/diagnostics_{session_id}.jsonl

    Each diagnostics record contains:
        - audio_path: path to the chunk WAV
        - english: STT transcription
        - spanish_gemma: Gemma translation (for es lang)
        - stt_confidence: float 0-1

    Returns list of dicts: {audio_path, text, confidence}
    """
    sessions_path = Path(sessions_dir)
    entries = []

    if not sessions_path.exists():
        logger.warning(f"Live sessions directory not found: {sessions_path}")
        return entries

    # Find all session directories
    session_dirs = sorted([
        d for d in sessions_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not session_dirs:
        logger.info(f"No session directories found in {sessions_path}")
        return entries

    # Project root for resolving diagnostics paths
    project_root = Path(__file__).resolve().parent.parent

    for session_dir in session_dirs:
        session_id = session_dir.name

        # Find corresponding diagnostics JSONL
        diag_path = project_root / "metrics" / f"diagnostics_{session_id}.jsonl"
        if not diag_path.exists():
            logger.info(f"  No diagnostics found for session {session_id}, skipping")
            continue

        # Load all diagnostics records for this session
        records = []
        try:
            with open(diag_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"  Error reading diagnostics for {session_id}: {e}")
            continue

        logger.info(f"  Session {session_id}: {len(records)} records")

        for record in records:
            audio_path_str = record.get("audio_path", "")
            if not audio_path_str:
                continue

            audio_path = Path(audio_path_str)
            if not audio_path.exists():
                # Try relative to session dir
                candidate = session_dir / audio_path.name
                if candidate.exists():
                    audio_path = candidate
                else:
                    continue

            # Select text based on language
            if lang == "en":
                text = record.get("english", "").strip()
            elif lang == "es":
                # Prefer corrected Spanish, fall back to Gemma translation
                text = (record.get("corrected_spanish")
                        or record.get("spanish_gemma", "")).strip()
            elif lang == "hi":
                text = record.get("hindi", "").strip()
            elif lang == "zh":
                text = record.get("chinese", "").strip()
            else:
                text = record.get("english", "").strip()

            if not text or len(text) < MIN_TEXT_LENGTH:
                continue

            # Duration filter
            duration = get_wav_duration(audio_path)
            if duration < min_dur or duration > max_dur:
                continue

            # Confidence filter
            confidence = record.get("stt_confidence")
            # Convert stt_confidence (0-1 scale) to avg_logprob-equivalent threshold
            # stt_confidence = min(1.0, max(0.0, 1.0 + mean_logprob))
            # So avg_logprob = stt_confidence - 1.0
            if confidence is not None:
                equiv_logprob = confidence - 1.0
                if equiv_logprob < min_confidence:
                    continue

            entries.append({
                "audio_path": str(audio_path.resolve()),
                "text": text,
                "confidence": confidence,
            })

    return entries


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def build_piper_dataset(entries, output_dir, lang):
    """Convert entries to Piper/LJSpeech format with resampled audio.

    Creates:
        output_dir/
        +-- wav/         # 22050 Hz mono 16-bit PCM
        +-- metadata.csv # filename|transcription|normalized_transcription
    """
    output_path = Path(output_dir)
    wav_dir = output_path / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    total_duration = 0.0
    resample_errors = 0

    for i, entry in enumerate(entries):
        src_path = entry["audio_path"]
        # Generate clean filename: {lang}_{index:06d}.wav
        wav_name = f"{lang}_{i:06d}.wav"
        dest_path = wav_dir / wav_name

        # Resample to Piper format (22050 Hz, mono, 16-bit PCM)
        if not resample_wav(src_path, str(dest_path)):
            resample_errors += 1
            continue

        # Track duration from output file
        duration = get_wav_duration(dest_path)
        total_duration += duration

        text = entry["text"]
        normalized = normalize_text(text)

        metadata_rows.append({
            "filename": wav_name,
            "transcription": text,
            "normalized_transcription": normalized,
        })

        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{len(entries)} clips...")

    # Write metadata.csv in LJSpeech pipe-delimited format
    csv_path = output_path / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        for row in metadata_rows:
            # LJSpeech format: filename|transcription|normalized_transcription
            f.write(f"{row['filename']}|{row['transcription']}|{row['normalized_transcription']}\n")

    return len(metadata_rows), total_duration, resample_errors


def main():
    parser = argparse.ArgumentParser(
        description="Convert church audio to Piper/LJSpeech training format"
    )
    parser.add_argument("--lang", default="en", choices=["en", "es", "hi", "zh"],
                        help="Target language (default: en)")
    parser.add_argument("--input", "-i", default=None,
                        help="Input directory with cleaned audio chunks "
                        "(default: stark_data/cleaned/chunks)")
    parser.add_argument("--transcripts", default=None,
                        help="Transcripts directory "
                        "(default: stark_data/transcripts)")
    parser.add_argument("--live-sessions", default=None,
                        help="Live session directory to also include "
                        "(default: stark_data/live_sessions)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory "
                        "(default: stark_data/piper_dataset/{lang})")
    parser.add_argument("--min-confidence", type=float, default=-0.5,
                        help="Minimum avg_logprob to include (default: -0.5)")
    parser.add_argument("--min-duration", type=float, default=1.0,
                        help="Minimum clip duration in seconds (default: 1.0)")
    parser.add_argument("--max-duration", type=float, default=15.0,
                        help="Maximum clip duration in seconds (default: 15.0)")
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent

    chunks_dir = Path(args.input) if args.input else project_root / "stark_data" / "cleaned" / "chunks"
    if not chunks_dir.is_absolute():
        chunks_dir = project_root / chunks_dir

    transcripts_dir = Path(args.transcripts) if args.transcripts else project_root / "stark_data" / "transcripts"
    if not transcripts_dir.is_absolute():
        transcripts_dir = project_root / transcripts_dir

    live_sessions_dir = Path(args.live_sessions) if args.live_sessions else project_root / "stark_data" / "live_sessions"
    if not live_sessions_dir.is_absolute():
        live_sessions_dir = project_root / live_sessions_dir

    output_dir = Path(args.output) if args.output else project_root / "stark_data" / "piper_dataset" / args.lang
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    logger.info(f"Preparing Piper dataset for language: {args.lang}")
    logger.info(f"  Chunks dir:        {chunks_dir}")
    logger.info(f"  Transcripts dir:   {transcripts_dir}")
    logger.info(f"  Live sessions dir: {live_sessions_dir}")
    logger.info(f"  Output dir:        {output_dir}")
    logger.info(f"  Confidence >= {args.min_confidence}")
    logger.info(f"  Duration: {args.min_duration}-{args.max_duration}s")

    all_entries = []

    # Source 1: Preprocessed chunks + transcript JSONs
    logger.info("\n--- Loading from preprocessed chunks ---")
    chunk_entries = load_from_chunks(
        str(chunks_dir), str(transcripts_dir),
        min_confidence=args.min_confidence,
        min_dur=args.min_duration,
        max_dur=args.max_duration,
    )
    logger.info(f"  Loaded {len(chunk_entries)} entries from chunks")
    all_entries.extend(chunk_entries)

    # Source 2: Live session output
    logger.info("\n--- Loading from live sessions ---")
    session_entries = load_from_live_sessions(
        str(live_sessions_dir), args.lang,
        min_confidence=args.min_confidence,
        min_dur=args.min_duration,
        max_dur=args.max_duration,
    )
    logger.info(f"  Loaded {len(session_entries)} entries from live sessions")
    all_entries.extend(session_entries)

    if not all_entries:
        logger.error("No entries found. Check input paths and filters.")
        logger.error(f"  Chunks dir exists: {chunks_dir.exists()}")
        logger.error(f"  Transcripts dir exists: {transcripts_dir.exists()}")
        logger.error(f"  Live sessions dir exists: {live_sessions_dir.exists()}")
        return

    # Deduplicate by audio path (live sessions and chunks may overlap)
    seen_paths = set()
    unique_entries = []
    for entry in all_entries:
        if entry["audio_path"] not in seen_paths:
            seen_paths.add(entry["audio_path"])
            unique_entries.append(entry)
    if len(unique_entries) < len(all_entries):
        logger.info(f"  Removed {len(all_entries) - len(unique_entries)} duplicate entries")
    all_entries = unique_entries

    # Build the dataset
    logger.info(f"\n--- Building Piper dataset ({len(all_entries)} clips) ---")
    n_clips, total_dur, n_errors = build_piper_dataset(all_entries, str(output_dir), args.lang)

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Piper Dataset Summary ({args.lang})")
    logger.info(f"{'='*50}")
    logger.info(f"  Total clips:       {n_clips}")
    logger.info(f"  Total duration:    {total_dur:.1f}s ({total_dur/60:.1f} min)")
    logger.info(f"  Avg clip duration: {total_dur/max(n_clips,1):.1f}s")
    logger.info(f"  Resample errors:   {n_errors}")
    logger.info(f"  Filtered out:      {len(all_entries) - n_clips - n_errors}")
    logger.info(f"  Output:            {output_dir}")
    logger.info(f"  metadata.csv:      {output_dir / 'metadata.csv'}")
    logger.info(f"  WAV directory:     {output_dir / 'wav'}")


if __name__ == "__main__":
    main()
