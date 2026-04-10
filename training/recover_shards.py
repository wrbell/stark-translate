#!/usr/bin/env python3
"""
recover_shards.py — Rebuild a DatasetDict from completed shards after OOM.

If align_deepgram_chunks.py dies during the concat/save step, the shards
are still on disk. This script streams them one-at-a-time into a final
DatasetDict, never holding more than one shard in memory.

Usage:
    python training/recover_shards.py \
        --cache-dir /mnt/d/Data/stt-data/whisper_dataset_sttdata/.preprocessed_cache

    # With custom memory cap (default: 12 GB)
    python training/recover_shards.py \
        --cache-dir /path/to/.preprocessed_cache \
        --mem-cap-gb 8
"""

import argparse
import gc
import logging
import resource
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_rss_gb():
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e9
    except ImportError:
        return None


def stream_shards(shard_dir: Path):
    """Yield examples from each shard sequentially, one shard in memory at a time."""
    from datasets import Dataset

    shard_paths = sorted(shard_dir.iterdir())
    logger.info("Streaming %d shards from %s", len(shard_paths), shard_dir)

    for i, sp in enumerate(shard_paths):
        shard = Dataset.load_from_disk(str(sp))
        yield from shard
        del shard
        gc.collect()
        if (i + 1) % 10 == 0:
            rss = get_rss_gb()
            rss_str = f" | RSS: {rss:.1f} GB" if rss is not None else ""
            logger.info("  Streamed %d/%d shards%s", i + 1, len(shard_paths), rss_str)


def main():
    parser = argparse.ArgumentParser(description="Rebuild DatasetDict from completed shards after OOM")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Path to .preprocessed_cache directory containing _shards_* dirs",
    )
    parser.add_argument(
        "--mem-cap-gb",
        type=int,
        default=12,
        help="Hard virtual memory cap in GB (default: 12)",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Don't delete shard directories after successful merge",
    )
    args = parser.parse_args()

    # Set hard memory cap
    hard_bytes = args.mem_cap_gb * 1024 * 1024 * 1024
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (hard_bytes, hard))
        logger.info("Hard memory cap: %d GB", args.mem_cap_gb)
    except (ValueError, OSError) as e:
        logger.warning("Could not set memory cap: %s", e)

    import shutil

    from datasets import Dataset, DatasetDict

    cache_dir = args.cache_dir

    # Find shard directories
    train_shard_dir = cache_dir / "_shards_train"
    test_shard_dir = cache_dir / "_shards_test"

    splits = {}

    for split_name, shard_dir in [("train", train_shard_dir), ("test", test_shard_dir)]:
        if not shard_dir.exists():
            logger.info("No %s shards found, skipping", split_name)
            continue

        shard_count = len(list(shard_dir.iterdir()))
        if shard_count == 0:
            logger.info("Empty %s shard dir, creating empty split", split_name)
            splits[split_name] = Dataset.from_dict({"input_features": [], "labels": []})
            continue

        logger.info("Rebuilding %s from %d shards...", split_name, shard_count)
        ds = Dataset.from_generator(stream_shards, gen_kwargs={"shard_dir": shard_dir})
        splits[split_name] = ds
        logger.info("  %s: %d examples", split_name, len(ds))

    if not splits:
        logger.error("No shard directories found in %s", cache_dir)
        sys.exit(1)

    # Save final DatasetDict
    dd = DatasetDict(splits)
    logger.info("Saving DatasetDict: %s", dd)
    dd.save_to_disk(str(cache_dir))
    logger.info("Saved to %s", cache_dir)

    # Cleanup shards
    if not args.keep_shards:
        for shard_dir in cache_dir.glob("_shards_*"):
            shutil.rmtree(shard_dir)
            logger.info("Cleaned up %s", shard_dir)

    rss = get_rss_gb()
    rss_str = f" (RSS: {rss:.1f} GB)" if rss is not None else ""
    logger.info("Done%s", rss_str)


if __name__ == "__main__":
    main()
