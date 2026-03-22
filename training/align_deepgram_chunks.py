#!/usr/bin/env python3
"""
align_deepgram_chunks.py — Align Whisper Chunks with Deepgram Word Timestamps

Aligns faster-whisper chunk boundaries with Deepgram word-level timestamps to
build a Whisper fine-tuning dataset. For each chunk, finds overlapping Deepgram
words (with small tolerance for alignment drift), concatenates them as ground-truth
text, and extracts the audio segment from the source WAV.

Output is HuggingFace audiofolder format:
    {output}/train/metadata.csv  (file_name,transcription)
    {output}/train/*.wav
    {output}/eval/metadata.csv
    {output}/eval/*.wav

Usage:
    python training/align_deepgram_chunks.py
    python training/align_deepgram_chunks.py --eval-sources "Gospel_Message_(10_12_25)_I6dy6o_ewDk"
    python training/align_deepgram_chunks.py --min-chars 30 --output stark_data/whisper_ds_dg
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Ensure project root is on path for tools/settings imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Tolerance (seconds) for matching Deepgram words to chunk boundaries
ALIGNMENT_TOLERANCE = 0.1


def load_whisper_chunks(chunks_path: Path) -> list[dict]:
    """Load faster-whisper chunks from expanded JSON."""
    logger.info("Loading whisper chunks from %s", chunks_path)
    with open(chunks_path) as f:
        chunks = json.load(f)
    logger.info("Loaded %d chunks", len(chunks))
    return chunks


def group_chunks_by_source(chunks: list[dict]) -> dict[str, list[dict]]:
    """Group chunks by their source field (filename stem)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        source = chunk.get("source", "")
        if source:
            groups[source].append(chunk)
    logger.info("Found %d unique sources", len(groups))
    return groups


def load_deepgram_transcript(deepgram_dir: Path, source: str) -> dict | None:
    """Find and load the Deepgram transcript matching a source stem.

    Tries several filename patterns to find the matching transcript.
    """
    candidates = [
        deepgram_dir / f"{source}.deepgram.json",
        deepgram_dir / f"{source}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)
    # Try fuzzy match — source stem may be a substring of the filename
    for path in deepgram_dir.glob("*.deepgram.json"):
        if source in path.stem:
            with open(path) as f:
                return json.load(f)
    return None


def find_deepgram_words_for_chunk(
    words: list[dict], chunk_start: float, chunk_end: float, tolerance: float
) -> list[dict]:
    """Find Deepgram words that fall within the chunk boundaries.

    A word is included if:
        word.start >= chunk_start - tolerance
        word.end   <= chunk_end   + tolerance
    """
    matched = []
    for w in words:
        w_start = w.get("start", 0.0)
        w_end = w.get("end", 0.0)
        if w_start >= chunk_start - tolerance and w_end <= chunk_end + tolerance:
            matched.append(w)
    return matched


def extract_audio_chunk(source_wav: Path, output_wav: Path, start: float, end: float) -> bool:
    """Extract an audio segment from source WAV using ffmpeg.

    The source WAVs are already 16kHz mono, so we use -c copy for speed.
    """
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_wav),
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-c",
        "copy",
        str(output_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(
            "ffmpeg failed for %s [%.2f-%.2f]: %s",
            source_wav.name,
            start,
            end,
            result.stderr.strip(),
        )
        return False
    return True


def find_audio_file(audio_dir: Path, source: str) -> Path | None:
    """Find the WAV file for a given source stem."""
    candidates = [
        audio_dir / f"{source}.wav",
        audio_dir / f"{source}.WAV",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Glob fallback
    matches = list(audio_dir.glob(f"{source}.*"))
    wav_matches = [m for m in matches if m.suffix.lower() == ".wav"]
    if wav_matches:
        return wav_matches[0]
    return None


def determine_eval_sources(chunks: list[dict], eval_sources_arg: str | None) -> set[str]:
    """Determine which sources belong to the eval split.

    If --eval-sources is provided, use those. Otherwise, auto-detect from
    chunks that have split == 'eval'.
    """
    if eval_sources_arg:
        sources = {s.strip() for s in eval_sources_arg.split(",") if s.strip()}
        logger.info("Using %d eval sources from CLI: %s", len(sources), sources)
        return sources

    # Auto-detect from chunk metadata
    eval_sources = set()
    for chunk in chunks:
        if chunk.get("split") == "eval":
            source = chunk.get("source", "")
            if source:
                eval_sources.add(source)
    if eval_sources:
        logger.info(
            "Auto-detected %d eval sources from chunk metadata: %s",
            len(eval_sources),
            eval_sources,
        )
    else:
        logger.info("No eval sources detected; all data goes to train split")
    return eval_sources


def write_metadata_csv(output_dir: Path, rows: list[dict]) -> None:
    """Write metadata.csv for a HuggingFace audiofolder split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "transcription"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="Align Whisper chunks with Deepgram word timestamps to build a Whisper fine-tuning dataset."
    )
    parser.add_argument(
        "--whisper-chunks",
        type=Path,
        default=Path("ablation/sermon_whisper_chunks_expanded.json"),
        help="Path to faster-whisper chunks JSON (default: ablation/sermon_whisper_chunks_expanded.json)",
    )
    parser.add_argument(
        "--deepgram-dir",
        type=Path,
        default=Path("stark_data/deepgram_transcripts"),
        help="Directory containing Deepgram transcript JSONs (default: stark_data/deepgram_transcripts)",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("stark_data/raw/midwest"),
        help="Directory containing source WAV files (default: stark_data/raw/midwest)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stark_data/whisper_dataset_deepgram"),
        help="Output dataset directory (default: stark_data/whisper_dataset_deepgram)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=20,
        help="Skip chunks where Deepgram text has fewer characters (default: 20)",
    )
    parser.add_argument(
        "--eval-sources",
        type=str,
        default=None,
        help="Comma-separated source stems to hold out for eval (default: auto-detect from chunks with split=='eval')",
    )
    args = parser.parse_args()

    # Load and group whisper chunks
    chunks = load_whisper_chunks(args.whisper_chunks)
    grouped = group_chunks_by_source(chunks)
    eval_sources = determine_eval_sources(chunks, args.eval_sources)

    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    stats = {
        "total_chunks": 0,
        "matched": 0,
        "skipped_short": 0,
        "skipped_no_words": 0,
        "skipped_no_deepgram": 0,
        "skipped_no_audio": 0,
        "skipped_existing": 0,
        "ffmpeg_errors": 0,
    }

    for source, source_chunks in sorted(grouped.items()):
        # Load Deepgram transcript
        dg = load_deepgram_transcript(args.deepgram_dir, source)
        if dg is None:
            logger.warning("No Deepgram transcript found for source: %s", source)
            stats["skipped_no_deepgram"] += len(source_chunks)
            continue

        # Find audio file
        audio_path = find_audio_file(args.audio_dir, source)
        if audio_path is None:
            logger.warning("No audio file found for source: %s", source)
            stats["skipped_no_audio"] += len(source_chunks)
            continue

        dg_words = dg.get("words", [])
        if not dg_words:
            logger.warning("No words in Deepgram transcript for: %s", source)
            stats["skipped_no_words"] += len(source_chunks)
            continue

        is_eval = source in eval_sources
        split_name = "eval" if is_eval else "train"

        logger.info(
            "Processing %s: %d chunks, %d DG words [%s]",
            source,
            len(source_chunks),
            len(dg_words),
            split_name,
        )

        for i, chunk in enumerate(source_chunks):
            stats["total_chunks"] += 1

            chunk_start = chunk.get("start", 0.0)
            chunk_end = chunk.get("end", 0.0)

            # Find matching Deepgram words
            matched_words = find_deepgram_words_for_chunk(dg_words, chunk_start, chunk_end, ALIGNMENT_TOLERANCE)

            if not matched_words:
                stats["skipped_no_words"] += 1
                continue

            # Build ground-truth text from Deepgram words
            text = " ".join(w.get("word", "") for w in matched_words).strip()

            if len(text) < args.min_chars:
                stats["skipped_short"] += 1
                continue

            # Output WAV filename
            wav_name = f"{source}_{i:05d}.wav"
            out_dir = args.output / split_name
            out_wav = out_dir / wav_name

            # Idempotency — skip if already extracted
            if out_wav.exists():
                stats["skipped_existing"] += 1
                row = {"file_name": wav_name, "transcription": text}
                if is_eval:
                    eval_rows.append(row)
                else:
                    train_rows.append(row)
                continue

            # Extract audio chunk
            ok = extract_audio_chunk(audio_path, out_wav, chunk_start, chunk_end)
            if not ok:
                stats["ffmpeg_errors"] += 1
                continue

            stats["matched"] += 1
            row = {"file_name": wav_name, "transcription": text}
            if is_eval:
                eval_rows.append(row)
            else:
                train_rows.append(row)

            # Progress logging every 500 chunks
            processed = stats["matched"] + stats["skipped_existing"]
            if processed % 500 == 0:
                logger.info(
                    "Progress: %d chunks processed (%d matched, %d existing)",
                    processed,
                    stats["matched"],
                    stats["skipped_existing"],
                )

    # Write metadata CSVs
    if train_rows:
        write_metadata_csv(args.output / "train", train_rows)
    if eval_rows:
        write_metadata_csv(args.output / "eval", eval_rows)

    # Summary
    logger.info("=" * 60)
    logger.info("Alignment complete")
    logger.info("  Total chunks:        %d", stats["total_chunks"])
    logger.info("  Matched (new):       %d", stats["matched"])
    logger.info("  Skipped (existing):  %d", stats["skipped_existing"])
    logger.info("  Skipped (short):     %d", stats["skipped_short"])
    logger.info("  Skipped (no words):  %d", stats["skipped_no_words"])
    logger.info("  Skipped (no DG):     %d", stats["skipped_no_deepgram"])
    logger.info("  Skipped (no audio):  %d", stats["skipped_no_audio"])
    logger.info("  FFmpeg errors:       %d", stats["ffmpeg_errors"])
    logger.info("  Train rows:          %d", len(train_rows))
    logger.info("  Eval rows:           %d", len(eval_rows))
    logger.info("  Output:              %s", args.output)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
