#!/usr/bin/env python3
"""
sort_sermons.py — Classify Sermons by Type/Date and Apply Training Cutoff

Reads metadata JSONs from downloaded sermon audio, parses type and date
from the title field, and produces a manifest with train/eval split.
Optionally sorts WAV files into a structured stt-data/ directory.

Usage:
    # Manifest only (no file moves)
    python tools/sort_sermons.py --input stark_data/raw/midwest

    # Sort existing WAVs into stt-data/ structure
    python tools/sort_sermons.py --input stark_data/raw/midwest --output-dir stt-data

    # Include playlist catalog entries (not yet downloaded)
    python tools/sort_sermons.py --input stark_data/raw/midwest --output-dir stt-data \
        --catalog stark_data/playlist_catalog.json

    # Tag existing chunks with split field
    python tools/sort_sermons.py --input stark_data/raw/midwest \
        --tag-chunks ablation/sermon_whisper_chunks_expanded.json
"""

import argparse
import json
import logging
import os
import re
import shutil
from datetime import date
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Date regex patterns for title field
_DATE_PAREN = re.compile(r"\((\d{1,2})[/_](\d{1,2})[/_](\d{2})\)")
_DATE_INLINE = re.compile(r"(\d{1,2})[/_](\d{1,2})[/_](\d{2})")


def classify_type(title: str) -> str:
    """Classify sermon type from title string."""
    if "Conference" in title:
        return "conference"
    if "Baptism" in title:
        return "gospel"  # baptism services are gospel meetings
    if "Gospel" in title:
        return "gospel"
    return "ministry"


def parse_date(title: str) -> date | None:
    """Extract date from title string. Returns None if no date found."""
    m = _DATE_PAREN.search(title)
    if not m:
        m = _DATE_INLINE.search(title)
    if not m:
        return None
    month, day, yr = int(m.group(1)), int(m.group(2)), int(m.group(3))
    year = 2000 + yr if yr < 50 else 1900 + yr
    try:
        return date(year, month, day)
    except ValueError:
        logger.warning(f"Invalid date in title: {title!r} (parsed {year}-{month}-{day})")
        return None


def parse_speakers(meta: dict) -> str:
    """Extract speakers from metadata, falling back to description field."""
    if meta.get("speakers"):
        return meta["speakers"]
    desc = meta.get("description", "")
    if desc.startswith("Speakers:"):
        return desc.split(".")[0].replace("Speakers:", "").strip()
    if "Speaker:" in desc:
        return desc.split(".")[0].replace("Speaker:", "").strip()
    return ""


def target_dir_for_entry(entry: dict, base_dir: str) -> str:
    """Compute the target directory for a sermon entry."""
    sermon_type = entry["type"]
    year = entry.get("year") or "unknown"
    if sermon_type == "throwback":
        # Throwback has mixed/unknown dates, flat dir
        return os.path.join(base_dir, "throwback")
    return os.path.join(base_dir, sermon_type, str(year))


def build_manifest(input_dir: Path, cutoff: date) -> list[dict]:
    """Build sermon manifest from metadata JSONs."""
    entries = []
    for json_path in sorted(input_dir.glob("*.json")):
        meta = json.loads(json_path.read_text())
        title = meta.get("title", "")
        sermon_type = classify_type(title)
        sermon_date = parse_date(title)
        split = "unknown"
        if sermon_date:
            split = "train" if sermon_date < cutoff else "eval"

        wav_name = json_path.stem + ".wav"
        wav_path = input_dir / wav_name
        year = str(sermon_date.year) if sermon_date else None

        entries.append(
            {
                "video_id": meta.get("video_id", ""),
                "title": title,
                "type": sermon_type,
                "date": sermon_date.isoformat() if sermon_date else None,
                "year": year,
                "split": split,
                "speakers": parse_speakers(meta),
                "duration_seconds": meta.get("duration_seconds", 0),
                "accent": meta.get("accent", ""),
                "channel": meta.get("channel", ""),
                "wav_path": str(wav_path) if wav_path.exists() else None,
                "wav_filename": wav_name,
                "source": json_path.stem,
                "origin": "local",
            }
        )

    return entries


def load_catalog_entries(catalog_path: Path, cutoff: date) -> list[dict]:
    """Load playlist catalog entries and classify them."""
    catalog = json.loads(catalog_path.read_text())
    entries = []
    for item in catalog:
        title = item.get("title", "")
        playlist_type = item.get("playlist_type", "")

        # Use playlist_type directly for catalog entries
        if playlist_type in ("gospel", "gospel_2020"):
            sermon_type = "gospel"
        elif playlist_type == "throwback":
            sermon_type = "throwback"
        elif playlist_type == "conference":
            sermon_type = "conference"
        elif playlist_type == "ministry":
            sermon_type = "ministry"
        else:
            sermon_type = classify_type(title)

        sermon_date = parse_date(title)
        year = item.get("year") or (str(sermon_date.year) if sermon_date else "unknown")
        split = "unknown"
        if sermon_date:
            split = "train" if sermon_date < cutoff else "eval"

        entries.append(
            {
                "video_id": item.get("video_id", ""),
                "title": title,
                "type": sermon_type,
                "date": sermon_date.isoformat() if sermon_date else None,
                "year": year,
                "split": split,
                "speakers": "",
                "duration_seconds": item.get("duration_seconds", 0),
                "accent": "",
                "channel": "",
                "wav_path": None,  # not yet downloaded
                "wav_filename": None,
                "source": item.get("video_id", ""),
                "url": item.get("url", ""),
                "origin": "catalog",
                "playlist_type": playlist_type,
            }
        )

    return entries


def sort_files(manifest: list[dict], output_dir: str, copy: bool = False) -> int:
    """Sort WAV + JSON files into type/year directory structure.

    Only moves/copies files that have a local wav_path.
    Returns count of files sorted.
    """
    sorted_count = 0
    for entry in manifest:
        wav_path = entry.get("wav_path")
        if not wav_path or not Path(wav_path).exists():
            continue

        target = target_dir_for_entry(entry, output_dir)
        os.makedirs(target, exist_ok=True)

        wav_src = Path(wav_path)
        wav_dst = Path(target) / wav_src.name

        if wav_dst.exists():
            logger.debug(f"Already exists: {wav_dst}")
            continue

        # Copy or symlink the WAV
        if copy:
            shutil.copy2(wav_src, wav_dst)
        else:
            os.symlink(wav_src.resolve(), wav_dst)

        # Also copy/link the metadata JSON if it exists
        json_src = wav_src.with_suffix(".json")
        if json_src.exists():
            json_dst = Path(target) / json_src.name
            if not json_dst.exists():
                if copy:
                    shutil.copy2(json_src, json_dst)
                else:
                    os.symlink(json_src.resolve(), json_dst)

        # Update entry with new path
        entry["sorted_path"] = str(wav_dst)
        entry["target_dir"] = target
        sorted_count += 1

    return sorted_count


def tag_chunks(chunks_path: Path, manifest: list[dict], output_path: Path | None = None) -> int:
    """Tag chunks with split and type fields based on source → manifest mapping."""
    source_to_meta = {}
    for entry in manifest:
        source_to_meta[entry["source"]] = {
            "split": entry["split"],
            "type": entry["type"],
            "year": entry.get("year"),
        }

    chunks = json.loads(chunks_path.read_text())
    tagged = 0
    for chunk in chunks:
        source = chunk.get("source", "")
        if source in source_to_meta:
            chunk["split"] = source_to_meta[source]["split"]
            chunk["sermon_type"] = source_to_meta[source]["type"]
            chunk["year"] = source_to_meta[source].get("year")
            tagged += 1
        else:
            chunk["split"] = "unknown"
            chunk["sermon_type"] = "unknown"

    out = output_path or chunks_path
    out.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    logger.info(f"Tagged {tagged}/{len(chunks)} chunks with split + type, saved to {out}")
    return tagged


def print_summary(manifest: list[dict]) -> None:
    """Print summary table of manifest entries."""
    from collections import Counter

    type_counts = Counter()
    split_counts = Counter()
    type_year_counts = Counter()
    origin_counts = Counter()
    total_duration = 0

    for entry in manifest:
        t = entry["type"]
        s = entry["split"]
        y = entry.get("year", "?")
        o = entry.get("origin", "local")
        type_counts[t] += 1
        split_counts[s] += 1
        type_year_counts[(t, y)] += 1
        origin_counts[o] += 1
        total_duration += entry.get("duration_seconds", 0)

    # By type
    print()
    print(f"{'Type':<18} {'Count':>6}")
    print("-" * 26)
    for t, c in sorted(type_counts.items()):
        print(f"{t:<18} {c:>6}")
    print(f"{'TOTAL':<18} {len(manifest):>6}")
    print()

    # By type x year
    print(f"{'Type/Year':<25} {'Count':>6}")
    print("-" * 33)
    for (t, y), c in sorted(type_year_counts.items()):
        print(f"{t}/{y:<19} {c:>6}")
    print()

    # By split
    print(f"{'Split':<18} {'Count':>6}")
    print("-" * 26)
    for s, c in sorted(split_counts.items()):
        print(f"{s:<18} {c:>6}")
    print()

    # By origin
    print(f"{'Origin':<18} {'Count':>6}")
    print("-" * 26)
    for o, c in sorted(origin_counts.items()):
        print(f"{o:<18} {c:>6}")
    print()

    print(f"Total duration: {total_duration / 3600:.1f} hours")
    print()


def main():
    parser = argparse.ArgumentParser(description="Classify sermons by type/date and apply training cutoff")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("stark_data/raw/midwest"),
        help="Directory containing sermon metadata JSONs",
    )
    parser.add_argument(
        "--cutoff",
        type=lambda s: date.fromisoformat(s),
        default=date(2026, 3, 14),
        help="Training cutoff date (ISO format, default: 2026-03-14)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("stt-data/manifest.json"),
        help="Output manifest JSON path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Sort WAV files into this directory (e.g., stt-data). Creates type/year subdirs.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Path to playlist_catalog.json to include not-yet-downloaded videos in manifest",
    )
    parser.add_argument(
        "--tag-chunks",
        type=Path,
        default=None,
        help="Path to sermon_whisper_chunks JSON to tag with split + type fields",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinking when sorting",
    )
    args = parser.parse_args()

    logger.info(f"Scanning {args.input} for metadata JSONs...")
    logger.info(f"Training cutoff: {args.cutoff}")

    # Build manifest from local files
    manifest = build_manifest(args.input, args.cutoff)
    logger.info(f"Found {len(manifest)} local sermon files")

    # Optionally merge playlist catalog entries
    if args.catalog and args.catalog.exists():
        catalog_entries = load_catalog_entries(args.catalog, args.cutoff)
        # Deduplicate by video_id
        existing_ids = {e["video_id"] for e in manifest}
        new_entries = [e for e in catalog_entries if e["video_id"] not in existing_ids]
        manifest.extend(new_entries)
        logger.info(
            f"Added {len(new_entries)} catalog entries ({len(catalog_entries) - len(new_entries)} duplicates skipped)"
        )

    if not manifest:
        logger.error("No entries found")
        return

    # Optionally sort files into output dir
    if args.output_dir:
        logger.info(f"Sorting files into {args.output_dir}/...")
        # Set target_dir for all entries
        for entry in manifest:
            entry["target_dir"] = target_dir_for_entry(entry, str(args.output_dir))
        sorted_count = sort_files(manifest, str(args.output_dir), copy=args.copy)
        logger.info(f"Sorted {sorted_count} files into {args.output_dir}/")

    # Save manifest
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info(f"Saved manifest with {len(manifest)} entries to {args.output}")

    print_summary(manifest)

    # Optionally tag chunks
    if args.tag_chunks:
        if args.tag_chunks.exists():
            tag_chunks(args.tag_chunks, manifest)
        else:
            logger.warning(f"Chunks file not found: {args.tag_chunks}")


if __name__ == "__main__":
    main()
