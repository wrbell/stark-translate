#!/usr/bin/env python3
"""Import reviewed training exports without duplicates or evaluation leakage.

Whisper accepts metadata.jsonl or the historical metadata.csv audiofolder.
Pass a language directory (or a review bundle root); Spanish imports require
--language es and must target a separate Spanish training corpus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.review_data import atomic_jsonl, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_jsonl(path, rows)


def _training_only(path: Path, rows: list[dict]) -> None:
    if {"eval", "test", "holdout"} & set(path.parts):
        raise ValueError("Evaluation data must not be imported into a training corpus")
    if any(row.get("split", "train") != "train" or row.get("dataset_split", "train") != "train" for row in rows):
        raise ValueError("Export contains evaluation/unknown split records; import training records only")
    if any(row.get("is_eval") or row.get("session_kind") in ("replay", "synthetic") for row in rows):
        raise ValueError("Evaluation/replay records must not be imported into a training corpus")


def _pair(row: dict) -> tuple[str, str]:
    if row.get("en") or row.get("es"):
        return (row.get("en") or "").strip(), (row.get("es") or "").strip()
    source = (row.get("source_text") or row.get("corrected_english") or "").strip()
    target = (row.get("target_text") or row.get("corrected_spanish") or "").strip()
    return (target, source) if row.get("source_lang") == "es" else (source, target)


def merge_translation(
    corrections_path: Path,
    train_jsonl: Path,
    *,
    output: Path | None = None,
    dedupe: bool = True,
) -> dict:
    corrections = _load_jsonl(corrections_path)
    _training_only(corrections_path, corrections)
    _training_only(output or train_jsonl, [])
    existing = _load_jsonl(train_jsonl)
    before = len(existing)
    ids = {row.get("sample_id"): index for index, row in enumerate(existing) if row.get("sample_id")}
    seen = {tuple(part.lower() for part in _pair(row)) for row in existing}
    added = updated = 0
    for row in corrections:
        en, es = _pair(row)
        if not en or not es:
            continue
        pair_key = (en.lower(), es.lower())
        item = {**row, "en": en, "es": es, "source": "active_learning", "split": "train"}
        sample = row.get("sample_id")
        if sample in ids:
            index = ids[sample]
            if existing[index] != item:
                existing[index] = item
                updated += 1
            continue
        if dedupe and pair_key in seen:
            continue
        existing.append(item)
        if sample:
            ids[sample] = len(existing) - 1
        seen.add(pair_key)
        added += 1
    out = output or train_jsonl
    _write_jsonl(out, existing)
    return {
        "mode": "translation",
        "before": before,
        "added": added,
        "updated": updated,
        "after": len(existing),
        "output": str(out),
        "dedupe": dedupe,
    }


def _read_metadata(path: Path) -> list[dict]:
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as source:
            return list(csv.DictReader(source))
    return _load_jsonl(path)


def _metadata(directory: Path, language: str, *, required: bool = True) -> Path:
    # Explicit supported layouts only; never choose an arbitrary nested eval file.
    for base in (
        directory,
        directory / "whisper" / "train" / language,
        directory / "train" / language,
        directory / "train",
    ):
        for name in ("metadata.jsonl", "metadata.csv"):
            if (base / name).is_file():
                return base / name
    if required:
        raise FileNotFoundError(f"No training metadata.jsonl or metadata.csv under {directory}")
    return directory / "metadata.jsonl"


def merge_whisper(
    corrections_dir: Path,
    train_dir: Path,
    *,
    output_dir: Path | None = None,
    replay_ratio: float = 0.3,
    seed: int = 42,
    language: str = "en",
) -> dict:
    if language not in ("en", "es"):
        raise ValueError("language must be en or es")
    manifest = corrections_dir / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()).get("split") != "train":
        raise ValueError("Evaluation bundles cannot be merged into training")
    meta_path = _metadata(corrections_dir, language)
    corr_rows = _read_metadata(meta_path)
    _training_only(meta_path, corr_rows)
    out_dir = output_dir or train_dir
    _training_only(out_dir, [])
    train_meta = out_dir / "metadata.jsonl"
    if not train_meta.exists() and (out_dir / "metadata.csv").exists():
        train_meta = out_dir / "metadata.csv"
    existing = _read_metadata(train_meta) if train_meta.exists() else []
    if any((row.get("source_lang") or "en") != language for row in [*corr_rows, *existing]):
        raise ValueError("Audio language does not match --language; use a separate corpus per language")
    before = len(existing)
    # IDs are preserved across exports. Content hashes dedupe legacy exports.
    ids = {row.get("sample_id"): index for index, row in enumerate(existing) if row.get("sample_id")}
    prepared = []
    for row in corr_rows:
        source_name = row.get("file_name") or row.get("audio") or ""
        transcript = (row.get("transcription") or row.get("sentence") or row.get("text") or "").strip()
        if not source_name or not transcript:
            continue
        src = (meta_path.parent / source_name).resolve()
        if not src.is_relative_to(meta_path.parent.resolve()):
            raise ValueError("Correction audio must stay within its export directory")
        if not src.is_file():
            raise FileNotFoundError(f"Correction audio missing: {source_name}")
        audio_hash = sha256_file(src)
        if row.get("audio_sha256") and row["audio_sha256"] != audio_hash:
            raise ValueError(f"Audio hash mismatch: {source_name}")
        sample = row.get("sample_id") or f"audio-{audio_hash}-{language}"
        name = "al_" + hashlib.sha256(sample.encode()).hexdigest()[:24] + ".wav"
        item = {
            **row,
            "file_name": name,
            "transcription": transcript,
            "sample_id": sample,
            "audio_sha256": audio_hash,
            "source_lang": language,
            "split": "train",
            "source": "active_learning",
        }
        prepared.append((src, item))
    out_dir.mkdir(parents=True, exist_ok=True)
    added = updated = 0
    for src, item in prepared:
        sample = item["sample_id"]
        if sample in ids:
            index = ids[sample]
            same = all(
                str(existing[index].get(key) or "") == str(item.get(key) or "")
                for key in existing[index].keys() | item.keys()
            )
            if same and (out_dir / item["file_name"]).exists():
                continue
            existing[index] = item
            updated += 1
        else:
            ids[sample] = len(existing)
            existing.append(item)
            added += 1
        shutil.copy2(src, out_dir / item["file_name"])
    if train_meta.suffix == ".csv":
        # Retain historical corpus metadata format rather than create two competing files.
        fields = list(dict.fromkeys(key for row in existing for key in row))
        import io
        import os
        import tempfile

        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)
        fd, name = tempfile.mkstemp(dir=train_meta.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as dest:
            dest.write(stream.getvalue())
        os.replace(name, train_meta)
    else:
        _write_jsonl(train_meta, existing)
    _ = seed  # Compatibility only; replay sampling belongs to the training stage.
    return {
        "mode": "whisper",
        "before": before,
        "added": added,
        "updated": updated,
        "after": len(existing),
        "output": str(out_dir),
        "replay_ratio": replay_ratio,
        "language": language,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="cmd", required=True)
    translation = commands.add_parser("translation", help="Merge EN/ES correction pairs into train JSONL")
    translation.add_argument("--corrections", type=Path, required=True)
    translation.add_argument("--train-jsonl", type=Path, required=True)
    translation.add_argument("--output", type=Path, default=None)
    translation.add_argument("--no-dedupe", action="store_true")
    whisper = commands.add_parser("whisper", help="Merge corrections audiofolder into a language-specific train dir")
    whisper.add_argument("--corrections", type=Path, required=True)
    whisper.add_argument("--train-dir", type=Path, required=True)
    whisper.add_argument("--output-dir", type=Path, default=None)
    whisper.add_argument(
        "--replay-ratio", type=float, default=0.3, help="Recorded for compatibility; no downsampling occurs"
    )
    whisper.add_argument("--language", choices=["en", "es"], default="en")
    args = parser.parse_args(argv)
    if args.cmd == "translation":
        result = merge_translation(args.corrections, args.train_jsonl, output=args.output, dedupe=not args.no_dedupe)
    else:
        result = merge_whisper(
            args.corrections,
            args.train_dir,
            output_dir=args.output_dir,
            replay_ratio=args.replay_ratio,
            language=args.language,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
