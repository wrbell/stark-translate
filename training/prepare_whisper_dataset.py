#!/usr/bin/env python3
"""
prepare_whisper_dataset.py — Accent-Balanced Whisper Dataset Builder

Reads transcripts from stark_data/transcripts/, looks up accent tags from
companion metadata, applies confidence filtering, performs temperature-based
accent balancing, and outputs HuggingFace audiofolder format with train/eval
splits stratified by accent.

Output format:
    stark_data/whisper_dataset/
    ├── train/
    │   ├── metadata.csv   (file_name,transcription,accent)
    │   └── {accent}/*.wav (symlinked or copied)
    └── eval/
        ├── metadata.csv
        └── {accent}/*.wav

Usage:
    python prepare_whisper_dataset.py
    python prepare_whisper_dataset.py --chunks-dir stark_data/cleaned/chunks
    python prepare_whisper_dataset.py --temperature 0.5 --eval-ratio 0.05
    python prepare_whisper_dataset.py --no-balance   # Skip accent balancing
    python prepare_whisper_dataset.py --copy          # Copy WAVs instead of symlinking
"""

import argparse
import csv
import json
import logging
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Confidence thresholds for filtering (from CLAUDE.md Layer 3)
MAX_COMPRESSION_RATIO = 2.4
MIN_AVG_LOGPROB = -1.0
MAX_NO_SPEECH_PROB = 0.6


def find_accent_for_chunk(chunk_path, transcripts_dir):
    """Determine accent label for a chunk by checking its directory structure.

    Accent subdirectories: stark_data/cleaned/chunks/{accent}/chunk.wav
    Falls back to 'general' if not in an accent subdirectory.
    """
    chunk = Path(chunk_path)
    # Check if parent directory is an accent label
    parent = chunk.parent.name
    # If parent is 'chunks', there's no accent subdir
    if parent == "chunks":
        return "general"
    return parent


def load_transcripts(transcripts_dir, chunks_dir):
    """Load all transcript JSONs and pair with audio chunks.

    Returns list of dicts: {audio_path, text, accent, segments_meta}
    """
    transcripts_path = Path(transcripts_dir)
    chunks_path = Path(chunks_dir)
    entries = []

    # Collect all transcript JSONs (including in accent subdirs)
    json_files = sorted(transcripts_path.glob("*.json"))
    json_files += sorted(transcripts_path.glob("*/*.json"))
    # Exclude summary/meta files
    json_files = [f for f in json_files if not f.name.startswith("_")]

    logger.info(f"Found {len(json_files)} transcript files")

    for json_path in json_files:
        try:
            with open(json_path, encoding="utf-8") as f:
                transcript = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Skipping {json_path.name}: {e}")
            continue

        text = transcript.get("text", "").strip()
        if not text:
            continue

        audio_path_str = transcript.get("audio_path", "")
        if not audio_path_str:
            # Try to reconstruct from filename
            wav_name = json_path.stem + ".wav"
            # Search in chunks dir and accent subdirs
            candidates = list(chunks_path.glob(f"*/{wav_name}"))
            candidates += list(chunks_path.glob(wav_name))
            if candidates:
                audio_path_str = str(candidates[0])
            else:
                logger.warning(f"No audio found for {json_path.name}")
                continue

        audio_path = Path(audio_path_str)
        if not audio_path.exists():
            # Try relative to chunks_dir
            for pattern in [f"*/{audio_path.name}", audio_path.name]:
                candidates = list(chunks_path.glob(pattern))
                if candidates:
                    audio_path = candidates[0]
                    break
            else:
                logger.warning(f"Audio not found: {audio_path_str}")
                continue

        accent = find_accent_for_chunk(audio_path, transcripts_dir)

        # Extract segment-level confidence metadata (from faster-whisper backend)
        segments = transcript.get("segments", [])
        seg_meta = {}
        if segments and isinstance(segments[0], dict):
            avg_logprobs = [s.get("avg_logprob", 0) for s in segments if "avg_logprob" in s]
            compression_ratios = [s.get("compression_ratio", 0) for s in segments if "compression_ratio" in s]
            no_speech_probs = [s.get("no_speech_prob", 0) for s in segments if "no_speech_prob" in s]
            if avg_logprobs:
                seg_meta["avg_logprob"] = np.mean(avg_logprobs)
            if compression_ratios:
                seg_meta["max_compression_ratio"] = max(compression_ratios)
            if no_speech_probs:
                seg_meta["max_no_speech_prob"] = max(no_speech_probs)

        entries.append(
            {
                "audio_path": str(audio_path.resolve()),
                "text": text,
                "accent": accent,
                "segments_meta": seg_meta,
            }
        )

    return entries


def filter_by_confidence(entries):
    """Remove low-quality segments based on Whisper confidence metrics.

    Thresholds from CLAUDE.md Layer 3:
    - compression_ratio > 2.4 → auto-reject (hallucination)
    - avg_logprob < -1.0 → auto-reject
    - no_speech_prob > 0.6 (with low logprob) → auto-reject
    """
    kept = []
    rejected = Counter()

    for entry in entries:
        meta = entry.get("segments_meta", {})
        if not meta:
            # No confidence data available — keep the entry
            kept.append(entry)
            continue

        cr = meta.get("max_compression_ratio", 0)
        lp = meta.get("avg_logprob", 0)
        nsp = meta.get("max_no_speech_prob", 0)

        if cr > MAX_COMPRESSION_RATIO:
            rejected["high_compression_ratio"] += 1
            continue
        if lp < MIN_AVG_LOGPROB:
            rejected["low_avg_logprob"] += 1
            continue
        if nsp > MAX_NO_SPEECH_PROB and lp < -0.5:
            rejected["high_no_speech_prob"] += 1
            continue

        kept.append(entry)

    if rejected:
        logger.info(f"Confidence filtering rejected {sum(rejected.values())} segments:")
        for reason, count in rejected.most_common():
            logger.info(f"  {reason}: {count}")

    return kept


def balance_accents(entries, temperature=0.5, seed=42):
    """Temperature-based accent balancing.

    Weight per accent: count^(1/T) where T is temperature.
    T=1.0: proportional (no change), T→0: uniform, T=0.5: square-root smoothing.

    Oversamples minority accents, downsamples majority accents.
    """
    rng = np.random.RandomState(seed)

    # Group by accent
    by_accent = defaultdict(list)
    for entry in entries:
        by_accent[entry["accent"]].append(entry)

    counts = {accent: len(items) for accent, items in by_accent.items()}
    total = sum(counts.values())

    logger.info("Accent distribution before balancing:")
    for accent, count in sorted(counts.items()):
        logger.info(f"  {accent}: {count} ({count / total:.1%})")

    if len(counts) <= 1:
        logger.info("Only one accent present, skipping balancing")
        return entries

    # Compute sampling weights via temperature scaling
    raw_weights = {accent: count ** (1.0 / temperature) for accent, count in counts.items()}
    weight_sum = sum(raw_weights.values())
    target_fractions = {accent: w / weight_sum for accent, w in raw_weights.items()}

    # Target total size = original total size
    target_counts = {accent: max(1, int(total * frac)) for accent, frac in target_fractions.items()}

    balanced = []
    for accent, target in target_counts.items():
        pool = by_accent[accent]
        if target <= len(pool):
            # Downsample
            indices = rng.choice(len(pool), size=target, replace=False)
            balanced.extend(pool[i] for i in indices)
        else:
            # Oversample: include all originals + sample additional
            balanced.extend(pool)
            extra = target - len(pool)
            indices = rng.choice(len(pool), size=extra, replace=True)
            balanced.extend(pool[i] for i in indices)

    rng.shuffle(balanced)

    # Log results
    new_counts = Counter(e["accent"] for e in balanced)
    new_total = len(balanced)
    logger.info("Accent distribution after balancing:")
    for accent in sorted(new_counts):
        count = new_counts[accent]
        logger.info(f"  {accent}: {count} ({count / new_total:.1%})")

    return balanced


def stratified_split(entries, eval_ratio=0.05, seed=42):
    """Split entries into train/eval, stratified by accent."""
    rng = np.random.RandomState(seed)

    by_accent = defaultdict(list)
    for entry in entries:
        by_accent[entry["accent"]].append(entry)

    train, eval_ = [], []
    for accent, items in by_accent.items():
        n_eval = max(1, int(len(items) * eval_ratio))
        indices = rng.permutation(len(items))
        eval_indices = set(indices[:n_eval])
        for i, item in enumerate(items):
            if i in eval_indices:
                eval_.append(item)
            else:
                train.append(item)

    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def write_audiofolder(entries, output_dir, copy_files=False):
    """Write entries as HuggingFace audiofolder format.

    Creates:
        output_dir/
        ├── metadata.csv  (file_name,transcription,accent)
        └── {accent}/chunk.wav  (symlinked or copied)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata_rows = []

    for entry in entries:
        accent = entry["accent"]
        audio_src = Path(entry["audio_path"])
        accent_dir = output_path / accent
        accent_dir.mkdir(parents=True, exist_ok=True)

        dest = accent_dir / audio_src.name
        # Handle filename collisions
        if dest.exists() and dest.resolve() != audio_src.resolve():
            stem = audio_src.stem
            suffix = audio_src.suffix
            i = 1
            while dest.exists():
                dest = accent_dir / f"{stem}_{i}{suffix}"
                i += 1

        if not dest.exists():
            if copy_files:
                shutil.copy2(str(audio_src), str(dest))
            else:
                try:
                    dest.symlink_to(audio_src.resolve())
                except OSError:
                    # Symlinks may not be supported; fall back to copy
                    shutil.copy2(str(audio_src), str(dest))

        # Relative path from output_dir for metadata.csv
        rel_path = dest.relative_to(output_path)

        metadata_rows.append(
            {
                "file_name": str(rel_path),
                "transcription": entry["text"],
                "accent": accent,
            }
        )

    # Write metadata.csv
    csv_path = output_path / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "transcription", "accent"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    logger.info(f"Wrote {len(metadata_rows)} entries to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Build accent-balanced Whisper training dataset from transcripts")
    parser.add_argument(
        "--chunks-dir",
        default="stark_data/cleaned/chunks",
        help="Directory containing cleaned audio chunks (with accent subdirs)",
    )
    parser.add_argument(
        "--transcripts-dir", default="stark_data/transcripts", help="Directory containing transcript JSONs"
    )
    parser.add_argument(
        "--output", "-o", default="stark_data/whisper_dataset", help="Output directory for audiofolder dataset"
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.5,
        help="Temperature for accent balancing (0→uniform, 1→proportional, default: 0.5)",
    )
    parser.add_argument(
        "--eval-ratio", type=float, default=0.05, help="Fraction of data for evaluation split (default: 0.05)"
    )
    parser.add_argument("--no-balance", action="store_true", help="Skip accent balancing (use raw distribution)")
    parser.add_argument("--no-filter", action="store_true", help="Skip confidence-based filtering")
    parser.add_argument("--copy", action="store_true", help="Copy WAV files instead of symlinking")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    chunks_dir = Path(args.chunks_dir)
    if not chunks_dir.is_absolute():
        chunks_dir = project_root / chunks_dir
    transcripts_dir = Path(args.transcripts_dir)
    if not transcripts_dir.is_absolute():
        transcripts_dir = project_root / transcripts_dir
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    # Step 1: Load transcripts and pair with audio
    logger.info("Loading transcripts...")
    entries = load_transcripts(str(transcripts_dir), str(chunks_dir))
    logger.info(f"Loaded {len(entries)} transcript-audio pairs")

    if not entries:
        logger.error("No entries found. Check --chunks-dir and --transcripts-dir paths.")
        return

    # Step 2: Confidence filtering
    if not args.no_filter:
        logger.info("Applying confidence filtering...")
        entries = filter_by_confidence(entries)
        logger.info(f"{len(entries)} entries after filtering")

    # Step 3: Accent balancing
    if not args.no_balance:
        logger.info(f"Balancing accents (temperature={args.temperature})...")
        entries = balance_accents(entries, temperature=args.temperature, seed=args.seed)
        logger.info(f"{len(entries)} entries after balancing")

    # Step 4: Stratified train/eval split
    logger.info(f"Creating train/eval split (eval_ratio={args.eval_ratio})...")
    train_entries, eval_entries = stratified_split(entries, eval_ratio=args.eval_ratio, seed=args.seed)
    logger.info(f"Train: {len(train_entries)}, Eval: {len(eval_entries)}")

    # Step 5: Write audiofolder format
    train_dir = output_dir / "train"
    eval_dir = output_dir / "eval"

    logger.info(f"Writing train split to {train_dir}...")
    write_audiofolder(train_entries, str(train_dir), copy_files=args.copy)

    logger.info(f"Writing eval split to {eval_dir}...")
    write_audiofolder(eval_entries, str(eval_dir), copy_files=args.copy)

    # Summary
    train_accents = Counter(e["accent"] for e in train_entries)
    eval_accents = Counter(e["accent"] for e in eval_entries)

    logger.info("\n=== Dataset Summary ===")
    logger.info(f"Total: {len(train_entries) + len(eval_entries)} entries")
    logger.info(f"Train: {len(train_entries)}")
    for accent, count in sorted(train_accents.items()):
        logger.info(f"  {accent}: {count}")
    logger.info(f"Eval: {len(eval_entries)}")
    for accent, count in sorted(eval_accents.items()):
        logger.info(f"  {accent}: {count}")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
