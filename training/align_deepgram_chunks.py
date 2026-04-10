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
import gc
import json
import logging
import os
import shutil
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


_audio_file_cache: dict[str, Path | None] = {}


def find_audio_file(audio_dir: Path, source: str) -> Path | None:
    """Find the WAV file for a given source stem.

    Searches flat directory first, then falls back to recursive search.
    Results are cached to avoid repeated rglob calls.
    """
    if source in _audio_file_cache:
        return _audio_file_cache[source]

    # Fast path: flat directory
    candidates = [
        audio_dir / f"{source}.wav",
        audio_dir / f"{source}.WAV",
    ]
    for c in candidates:
        if c.exists():
            _audio_file_cache[source] = c
            return c

    # Glob fallback (flat)
    matches = list(audio_dir.glob(f"{source}.*"))
    wav_matches = [m for m in matches if m.suffix.lower() == ".wav"]
    if wav_matches:
        _audio_file_cache[source] = wav_matches[0]
        return wav_matches[0]

    # Recursive fallback for nested directory structures (e.g., stt-data/{type}/{year}/)
    rmatches = list(audio_dir.rglob(f"{source}.wav"))
    if rmatches:
        _audio_file_cache[source] = rmatches[0]
        return rmatches[0]

    _audio_file_cache[source] = None
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


def _generate_examples(rows, grouped_chunks, audio_dir, processor):
    """Yield preprocessed examples one at a time from source audio + Deepgram text.

    Each example contains a mel spectrogram (~960KB) and tokenized labels.
    Yielding (not accumulating) keeps memory flat regardless of dataset size.
    """
    import soundfile as sf

    for i, row in enumerate(rows):
        fname = row["file_name"]
        text = row["transcription"]

        # Parse source and chunk index from filename: {source}_{index:05d}.wav
        parts = fname.rsplit("_", 1)
        if len(parts) != 2:
            continue
        source = parts[0]
        chunk_idx = int(parts[1].replace(".wav", ""))

        # Find the chunk to get start/end timestamps
        source_chunks = grouped_chunks.get(source, [])
        if chunk_idx >= len(source_chunks):
            continue
        chunk = source_chunks[chunk_idx]
        start = chunk.get("start", 0.0)
        end = chunk.get("end", 0.0)

        # Find source audio
        audio_path = find_audio_file(audio_dir, source)
        if audio_path is None:
            continue

        # Read audio segment directly (no ffmpeg, no temp file)
        try:
            info = sf.info(str(audio_path))
            sr = info.samplerate
            start_frame = int(start * sr)
            end_frame = int(end * sr)
            audio_data, _ = sf.read(str(audio_path), start=start_frame, stop=end_frame, dtype="float32")

            # Resample to 16kHz if needed
            if sr != 16000:
                import librosa

                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

            # Compute mel spectrogram
            input_features = processor(audio_data, sampling_rate=16000).input_features[0]

            # Tokenize transcription
            labels = processor.tokenizer(text).input_ids

            yield {
                "input_features": input_features,
                "labels": labels,
            }
        except Exception as e:
            logger.warning("Failed to process %s chunk %d: %s", source, chunk_idx, e)
            continue


def _get_rss_gb():
    """Return current process RSS in GB, or None if psutil unavailable."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e9
    except ImportError:
        return None


# Number of rows per Arrow shard — caps peak RAM at ~960MB of mel data
_SHARD_SIZE = 1000


def _build_split_sharded(rows, split_name, grouped_chunks, audio_dir, processor, cache_dir):
    """Build one split (train or test) via batched Arrow shards.

    Processes rows in batches of _SHARD_SIZE, saves each batch as an Arrow
    shard to disk. Then streams all shards through a generator into the final
    dataset — never holding more than one shard in memory.

    Completed shards are skipped on re-run (crash-resume support).
    """
    from datasets import Dataset

    if not rows:
        return Dataset.from_dict({"input_features": [], "labels": []})

    shard_dir = cache_dir / f"_shards_{split_name}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    total_shards = (len(rows) + _SHARD_SIZE - 1) // _SHARD_SIZE
    logger.info(
        "  [%s] %d rows -> %d shards of %d",
        split_name,
        len(rows),
        total_shards,
        _SHARD_SIZE,
    )

    for batch_start in range(0, len(rows), _SHARD_SIZE):
        shard_idx = batch_start // _SHARD_SIZE
        shard_path = shard_dir / f"shard_{batch_start:06d}"

        # Resume: skip already-completed shards
        if shard_path.exists():
            logger.info(
                "  [%s] Shard %d/%d already exists, skipping",
                split_name,
                shard_idx + 1,
                total_shards,
            )
            continue

        batch_rows = rows[batch_start : batch_start + _SHARD_SIZE]

        shard_ds = Dataset.from_generator(
            _generate_examples,
            gen_kwargs={
                "rows": batch_rows,
                "grouped_chunks": grouped_chunks,
                "audio_dir": audio_dir,
                "processor": processor,
            },
        )
        shard_ds.save_to_disk(str(shard_path))

        rss = _get_rss_gb()
        rss_str = f" | RSS: {rss:.1f} GB" if rss is not None else ""
        logger.info(
            "  [%s] Shard %d/%d saved (%d examples)%s",
            split_name,
            shard_idx + 1,
            total_shards,
            len(shard_ds),
            rss_str,
        )

        # Free shard from memory before next batch
        del shard_ds
        gc.collect()

    # Merge shards by streaming one at a time — hard memory cap.
    # Never loads more than one shard into memory at once.
    shard_paths = sorted(shard_dir.iterdir())
    if not shard_paths:
        return Dataset.from_dict({"input_features": [], "labels": []})

    # Capture features from the first shard so Dataset.from_generator preserves
    # the original Arrow types (float32 mel spectrograms, not float64 default).
    # Without this, train_whisper.py's interleave_datasets() crashes on dtype
    # mismatch with the LibriSpeech replay buffer.
    first_shard = Dataset.load_from_disk(str(shard_paths[0]))
    shard_features = first_shard.features
    del first_shard
    gc.collect()

    def _stream_from_shards():
        """Yield examples from each shard sequentially, loading one at a time."""
        for i, sp in enumerate(shard_paths):
            shard = Dataset.load_from_disk(str(sp))
            yield from shard
            del shard
            gc.collect()
            if (i + 1) % 10 == 0:
                rss = _get_rss_gb()
                rss_str = f" | RSS: {rss:.1f} GB" if rss is not None else ""
                logger.info(
                    "  [%s] Streamed %d/%d shards%s",
                    split_name,
                    i + 1,
                    len(shard_paths),
                    rss_str,
                )

    logger.info("  [%s] Streaming %d shards into final dataset...", split_name, len(shard_paths))
    final_ds = Dataset.from_generator(_stream_from_shards, features=shard_features)
    logger.info("  [%s] Final dataset: %d examples", split_name, len(final_ds))

    return final_ds


def _build_preprocessed_cache(
    train_rows,
    eval_rows,
    grouped_chunks,
    deepgram_dir,
    audio_dir,
    output_dir,
    whisper_model,
    min_chars,
):
    """Build preprocessed Arrow cache directly from source audio + Deepgram text.

    Instead of extracting 75K+ small WAV files to HDD (slow, painful audiofolder scan),
    this reads audio segments directly from source WAVs, computes mel spectrograms,
    and saves as Arrow format that train_whisper.py can load instantly.

    Processes rows in batches of 1000 to cap peak RAM at ~1-2 GB regardless of
    dataset size. Completed shards are skipped on re-run for crash-resume support.

    Hard memory cap: 12 GB RSS. Process aborts cleanly if exceeded.
    """
    import resource

    # Hard cap at 12 GB virtual memory to prevent OOM-kill.
    # Sharded streaming should stay under 5 GB; 12 GB gives headroom.
    _HARD_MEM_CAP_GB = 12
    _hard_bytes = _HARD_MEM_CAP_GB * 1024 * 1024 * 1024
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (_hard_bytes, hard))
        logger.info("Set hard memory cap: %d GB virtual", _HARD_MEM_CAP_GB)
    except (ValueError, OSError) as e:
        logger.warning("Could not set memory cap: %s", e)

    os.environ["USE_TF"] = "0"
    from datasets import DatasetDict
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(whisper_model)
    logger.info("Loaded WhisperProcessor from %s", whisper_model)

    cache_dir = output_dir / ".preprocessed_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processing train split (%d rows)...", len(train_rows))
    train_ds = _build_split_sharded(train_rows, "train", grouped_chunks, audio_dir, processor, cache_dir)

    logger.info("Processing eval split (%d rows)...", len(eval_rows))
    eval_ds = _build_split_sharded(eval_rows, "test", grouped_chunks, audio_dir, processor, cache_dir)

    # Save final DatasetDict
    ds = DatasetDict({"train": train_ds, "test": eval_ds})
    logger.info("Built DatasetDict: %s", ds)
    ds.save_to_disk(str(cache_dir))
    logger.info("Saved preprocessed cache to %s", cache_dir)

    # Clean up temporary shard directories
    for shard_dir in cache_dir.glob("_shards_*"):
        shutil.rmtree(shard_dir)
        logger.info("Cleaned up shard dir: %s", shard_dir)

    # Also write JSONL metadata for compatibility
    jsonl_dir = output_dir / "train"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    with open(jsonl_dir / "metadata.jsonl", "w") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    eval_jsonl_dir = output_dir / "eval"
    eval_jsonl_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_jsonl_dir / "metadata.jsonl", "w") as f:
        for row in eval_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    parser.add_argument(
        "--preprocess-cache",
        action="store_true",
        help="Build preprocessed Arrow cache directly (skips WAV extraction + audiofolder scan). Much faster for large datasets.",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="Whisper model for processor (mel spectrogram extraction). Only used with --preprocess-cache.",
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

            # Output WAV filename (used as ID even in cache mode)
            wav_name = f"{source}_{i:05d}.wav"
            stats["matched"] += 1
            row = {"file_name": wav_name, "transcription": text}

            # Extract WAV only if not in preprocess-cache mode
            if not args.preprocess_cache:
                out_dir = args.output / split_name
                out_wav = out_dir / wav_name

                if out_wav.exists():
                    stats["skipped_existing"] += 1
                    if is_eval:
                        eval_rows.append(row)
                    else:
                        train_rows.append(row)
                    continue

                ok = extract_audio_chunk(audio_path, out_wav, chunk_start, chunk_end)
                if not ok:
                    stats["ffmpeg_errors"] += 1
                    stats["matched"] -= 1
                    continue
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

    # Output mode: either WAV files + metadata.csv, or preprocessed Arrow cache
    if args.preprocess_cache:
        logger.info("Building preprocessed Arrow cache directly (no WAV extraction)...")
        _build_preprocessed_cache(
            train_rows,
            eval_rows,
            grouped,
            args.deepgram_dir,
            args.audio_dir,
            args.output,
            args.whisper_model,
            args.min_chars,
        )
    else:
        # Write metadata CSVs (original audiofolder mode)
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
