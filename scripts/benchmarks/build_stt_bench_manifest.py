#!/usr/bin/env python3
"""build_stt_bench_manifest.py — Stratified bench manifest for STT engine comparison.

Reads the pre-chunked Deepgram-aligned eval set at
``stark_data/whisper_dataset_deepgram/eval/``, computes per-chunk duration +
SHA-256 + tier-1 theological term flags, then samples a fixed stratified subset
(default: 30 short + 20 medium + 10 long = 60 clips) prioritising clips that
contain at least one Tier 1 boost term so quality regressions surface.

The output manifest is committed to git. Re-running this script with the same
source corpus is deterministic (sort by SHA-256 then duration), so the bench
remains apples-to-apples across machines.

Usage:
    python scripts/benchmarks/build_stt_bench_manifest.py
    python scripts/benchmarks/build_stt_bench_manifest.py --output tools/stt_bench_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import struct
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.glossary import load_tier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_stt_bench_manifest")

DEFAULT_SOURCE = PROJECT_ROOT / "stark_data" / "whisper_dataset_deepgram" / "eval"
DEFAULT_OUTPUT = PROJECT_ROOT / "tools" / "stt_bench_manifest.json"

TIER_BREAKS = {"short": (1.0, 6.0), "medium": (6.0, 15.0), "long": (15.0, 30.0)}
DEFAULT_COUNTS = {"short": 30, "medium": 20, "long": 10}


def wav_duration_seconds(path: Path) -> float:
    """Read a 16-bit PCM WAV header and return duration in seconds.

    Avoids loading the full sample buffer — cheap enough to call on hundreds
    of chunks. Falls back to scipy if the header is non-standard.
    """
    with path.open("rb") as fh:
        riff = fh.read(12)
        if riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise ValueError(f"not a RIFF/WAVE file: {path}")
        sample_rate = 16000
        bits_per_sample = 16
        num_channels = 1
        data_size = 0
        while True:
            chunk_hdr = fh.read(8)
            if len(chunk_hdr) < 8:
                break
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_hdr)
            if chunk_id == b"fmt ":
                fmt = fh.read(chunk_size)
                _audio_fmt, num_channels, sample_rate, _byte_rate, _block_align, bits_per_sample = struct.unpack(
                    "<HHIIHH", fmt[:16]
                )
            elif chunk_id == b"data":
                data_size = chunk_size
                break
            else:
                fh.seek(chunk_size, os.SEEK_CUR)
    if data_size == 0 or sample_rate == 0:
        return 0.0
    bytes_per_sample = max(bits_per_sample // 8, 1)
    n_samples = data_size // (num_channels * bytes_per_sample)
    return n_samples / float(sample_rate)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def tier_for_duration(d: float) -> str | None:
    for name, (lo, hi) in TIER_BREAKS.items():
        if lo <= d < hi:
            return name
    return None


def detect_tier1_terms(text: str, tier1_terms: list[str]) -> list[str]:
    lo = text.lower()
    hits = []
    for term in tier1_terms:
        if term.lower() in lo:
            hits.append(term)
    return hits


def build_manifest(
    source: Path = DEFAULT_SOURCE,
    counts: dict[str, int] = DEFAULT_COUNTS,
    output: Path = DEFAULT_OUTPUT,
) -> dict:
    metadata_path = source / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing metadata.jsonl at {metadata_path}")

    tier1_raw = load_tier("boost")
    tier1_terms: list[str] = tier1_raw if isinstance(tier1_raw, list) else []
    if not tier1_terms:
        log.warning("Tier 1 boost terms unavailable — proceeding without tier1 detection.")

    log.info("scanning %s", source)
    candidates: dict[str, list[dict]] = {"short": [], "medium": [], "long": []}
    skipped = 0
    with metadata_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            audio = source / row["file_name"]
            if not audio.exists():
                skipped += 1
                continue
            try:
                duration = wav_duration_seconds(audio)
            except ValueError as exc:
                log.warning("skipping %s: %s", audio.name, exc)
                skipped += 1
                continue
            tier = tier_for_duration(duration)
            if tier is None:
                continue
            transcription = row.get("transcription", "").strip()
            if not transcription:
                continue
            tier1_hits = detect_tier1_terms(transcription, tier1_terms)
            candidates[tier].append(
                {
                    "id": audio.stem,
                    "audio_path_relative": str(audio.relative_to(PROJECT_ROOT)),
                    "duration_s": round(duration, 3),
                    "tier": tier,
                    "ground_truth": transcription,
                    "tier1_terms_present": tier1_hits,
                }
            )

    log.info(
        "candidates: short=%d medium=%d long=%d (skipped %d)",
        len(candidates["short"]),
        len(candidates["medium"]),
        len(candidates["long"]),
        skipped,
    )

    selected: list[dict] = []
    for tier_name, want in counts.items():
        pool = candidates[tier_name]
        # Sort: tier1-bearing first, then deterministically by ground_truth then duration
        pool_sorted = sorted(pool, key=lambda c: (-len(c["tier1_terms_present"]), c["ground_truth"], c["duration_s"]))
        chosen = pool_sorted[:want]
        if len(chosen) < want:
            log.warning("tier %s has only %d clips (wanted %d) — using all", tier_name, len(chosen), want)
        selected.extend(chosen)

    log.info("computing SHA-256 for %d selected clips", len(selected))
    for clip in selected:
        clip["audio_sha256"] = sha256_file(PROJECT_ROOT / clip["audio_path_relative"])

    manifest = {
        "version": "1.0",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_dir_relative": str(source.relative_to(PROJECT_ROOT)),
        "source_metadata_relative": str(metadata_path.relative_to(PROJECT_ROOT)),
        "tier_breaks_seconds": TIER_BREAKS,
        "counts_per_tier": counts,
        "total_clips": len(selected),
        "tier1_clip_count": sum(1 for c in selected if c["tier1_terms_present"]),
        "notes": (
            "Pre-chunked Deepgram-aligned eval split from W7-era dataset. Suitable for "
            "RELATIVE backend comparison (latency, RTF, WER deltas). For ABSOLUTE WER "
            "claims on out-of-domain audio, use the fresh-eval set described in "
            "CLAUDE.md (whisper_ablation/W16_mixed_w7 reports 7.25% on that set)."
        ),
        "clips": selected,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    log.info("wrote %s (%d clips, %d with tier1 terms)", output, manifest["total_clips"], manifest["tier1_clip_count"])
    return manifest


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help=f"chunk dir (default: {DEFAULT_SOURCE})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"manifest path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--short", type=int, default=DEFAULT_COUNTS["short"])
    p.add_argument("--medium", type=int, default=DEFAULT_COUNTS["medium"])
    p.add_argument("--long", type=int, default=DEFAULT_COUNTS["long"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    counts = {"short": args.short, "medium": args.medium, "long": args.long}
    build_manifest(source=args.source, counts=counts, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
