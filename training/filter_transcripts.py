#!/usr/bin/env python3
"""
filter_transcripts.py — Filter pseudo-labeled transcripts by confidence

Removes low-confidence segments before Whisper LoRA training. Used when
baseline WER assessment (Phase F) indicates WER > 10%.

Filters on avg_logprob, no_speech_prob, and compression_ratio thresholds
from engines/CLAUDE.md. Copies passing segments to a new directory.

Usage:
    python filter_transcripts.py \
      --input /mnt/d/Data/stt-data/stark_data/transcripts \
      --min-logprob -0.7 \
      --output /mnt/d/Data/stt-data/stark_data/transcripts_filtered

    # Stricter filtering (keep only high-confidence segments)
    python filter_transcripts.py \
      --input /mnt/d/Data/stt-data/stark_data/transcripts \
      --min-logprob -0.5 \
      --max-no-speech 0.3 \
      --max-compression 2.0 \
      --output /mnt/d/Data/stt-data/stark_data/transcripts_strict

    # Dry run to see what would be kept/dropped
    python filter_transcripts.py \
      --input /mnt/d/Data/stt-data/stark_data/transcripts \
      --dry-run
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default thresholds aligned with engines/CLAUDE.md "flag" level
DEFAULT_MIN_LOGPROB = -0.7
DEFAULT_MAX_NO_SPEECH = 0.3
DEFAULT_MAX_COMPRESSION = 2.4


def filter_segments(segments, min_logprob, max_no_speech, max_compression):
    """Filter segments by confidence thresholds.

    Returns (kept, dropped) lists of segments.
    """
    kept = []
    dropped = []

    for seg in segments:
        avg_logprob = seg.get("avg_logprob")
        no_speech_prob = seg.get("no_speech_prob")
        compression_ratio = seg.get("compression_ratio")

        drop = False

        if avg_logprob is not None and avg_logprob < min_logprob:
            drop = True
        if no_speech_prob is not None and no_speech_prob > max_no_speech:
            drop = True
        if compression_ratio is not None and compression_ratio > max_compression:
            drop = True

        if drop:
            dropped.append(seg)
        else:
            kept.append(seg)

    return kept, dropped


def filter_transcript_file(
    input_path,
    min_logprob,
    max_no_speech,
    max_compression,
):
    """Filter segments in a single transcript JSON file.

    Returns (original_count, kept_count, filtered_data) or None if file is empty.
    """
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    if not segments:
        return 0, 0, data

    kept, dropped = filter_segments(segments, min_logprob, max_no_speech, max_compression)

    filtered_data = dict(data)
    filtered_data["segments"] = kept
    filtered_data["_filter_stats"] = {
        "original_segments": len(segments),
        "kept_segments": len(kept),
        "dropped_segments": len(dropped),
        "min_logprob": min_logprob,
        "max_no_speech": max_no_speech,
        "max_compression": max_compression,
    }

    return len(segments), len(kept), filtered_data


def filter_transcripts(
    input_dir,
    output_dir,
    min_logprob=DEFAULT_MIN_LOGPROB,
    max_no_speech=DEFAULT_MAX_NO_SPEECH,
    max_compression=DEFAULT_MAX_COMPRESSION,
    dry_run=False,
    copy_audio=False,
):
    """Filter all transcript JSONs in a directory by confidence thresholds.

    Args:
        input_dir: Directory containing transcript JSON files
        output_dir: Directory for filtered transcripts (created if needed)
        min_logprob: Minimum avg_logprob (more negative = less confident)
        max_no_speech: Maximum no_speech_prob
        max_compression: Maximum compression_ratio
        dry_run: If True, just report what would be filtered
        copy_audio: If True, also copy associated audio files
    """
    input_path = Path(input_dir)
    json_files = sorted(input_path.glob("*.json"))
    json_files = [f for f in json_files if not f.name.startswith("_")]

    if not json_files:
        logger.error(f"No JSON files found in {input_dir}")
        return

    logger.info(f"Filtering {len(json_files)} transcript files")
    logger.info(
        f"  Thresholds: logprob >= {min_logprob}, no_speech <= {max_no_speech}, compression <= {max_compression}"
    )

    if not dry_run and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total_original = 0
    total_kept = 0
    files_with_segments = 0
    files_empty_after_filter = 0

    for jf in json_files:
        original_count, kept_count, filtered_data = filter_transcript_file(
            jf, min_logprob, max_no_speech, max_compression
        )

        total_original += original_count
        total_kept += kept_count

        if original_count > 0:
            files_with_segments += 1
            if kept_count == 0:
                files_empty_after_filter += 1

        if not dry_run and output_dir and kept_count > 0:
            out_path = Path(output_dir) / jf.name
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(filtered_data, f, ensure_ascii=False, indent=2)

            # Optionally copy associated audio file
            if copy_audio:
                audio_path = filtered_data.get("audio_path")
                if audio_path and os.path.exists(audio_path):
                    audio_dest = Path(output_dir) / Path(audio_path).name
                    if not audio_dest.exists():
                        shutil.copy2(audio_path, audio_dest)

    total_dropped = total_original - total_kept
    drop_pct = (total_dropped / total_original * 100) if total_original > 0 else 0

    logger.info("")
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Filter results:")
    logger.info(f"  Files processed:          {len(json_files)}")
    logger.info(f"  Files with segments:      {files_with_segments}")
    logger.info(f"  Files empty after filter: {files_empty_after_filter}")
    logger.info(f"  Total segments:           {total_original}")
    logger.info(f"  Kept:                     {total_kept} ({100 - drop_pct:.1f}%)")
    logger.info(f"  Dropped:                  {total_dropped} ({drop_pct:.1f}%)")

    if not dry_run and output_dir:
        logger.info(f"  Output directory:         {output_dir}")

    return {
        "total_segments": total_original,
        "kept": total_kept,
        "dropped": total_dropped,
        "drop_pct": round(drop_pct, 1),
        "files_processed": len(json_files),
        "files_empty_after_filter": files_empty_after_filter,
    }


def main():
    parser = argparse.ArgumentParser(description="Filter pseudo-labeled transcripts by confidence")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Directory with transcript JSON files",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory for filtered transcripts (required unless --dry-run)",
    )
    parser.add_argument(
        "--min-logprob",
        type=float,
        default=DEFAULT_MIN_LOGPROB,
        help=f"Minimum avg_logprob threshold (default: {DEFAULT_MIN_LOGPROB})",
    )
    parser.add_argument(
        "--max-no-speech",
        type=float,
        default=DEFAULT_MAX_NO_SPEECH,
        help=f"Maximum no_speech_prob threshold (default: {DEFAULT_MAX_NO_SPEECH})",
    )
    parser.add_argument(
        "--max-compression",
        type=float,
        default=DEFAULT_MAX_COMPRESSION,
        help=f"Maximum compression_ratio threshold (default: {DEFAULT_MAX_COMPRESSION})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report filtering stats without writing output",
    )
    parser.add_argument(
        "--copy-audio",
        action="store_true",
        help="Also copy associated audio files to output directory",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.output:
        parser.error("--output is required unless --dry-run is specified")

    filter_transcripts(
        input_dir=args.input,
        output_dir=args.output,
        min_logprob=args.min_logprob,
        max_no_speech=args.max_no_speech,
        max_compression=args.max_compression,
        dry_run=args.dry_run,
        copy_audio=args.copy_audio,
    )


if __name__ == "__main__":
    main()
