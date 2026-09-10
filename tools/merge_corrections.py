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


def _training_only(path: Path, rows: list[dict], *, require_live: bool = False) -> None:
    if {"eval", "test", "holdout"} & (set(path.parts) | set(path.resolve().parts)):
        raise ValueError("Evaluation data must not be imported into a training corpus")
    if any(row.get("split", "train") != "train" or row.get("dataset_split", "train") != "train" for row in rows):
        raise ValueError("Export contains evaluation/unknown split records; import training records only")
    if any(
        str(row.get("is_eval", "")).lower() in ("true", "1", "yes")
        or row.get("session_kind") in ("replay", "synthetic")
        for row in rows
    ):
        raise ValueError("Evaluation/replay records must not be imported into a training corpus")
    if require_live and any(row.get("session_kind") != "live" for row in rows):
        raise ValueError(
            "Correction training imports require explicit live-session provenance. "
            "Re-export with prepare_finetune_data --confirm-live-session SESSION after verifying the source, "
            "or keep unknown data in evaluation."
        )


def _revision(row: dict) -> int:
    try:
        value = int(str(row.get("revision") or 0))
        if value < 0:
            raise ValueError
    except (ValueError, TypeError) as exc:
        raise ValueError("Correction revision must be a nonnegative integer") from exc
    return value


def _accept_revision(existing: dict, incoming: dict, fields: tuple[str, ...]) -> bool:
    """Ignore older exports and reject identity/content conflicts before writing."""
    old_revision, new_revision = _revision(existing), _revision(incoming)
    sample = incoming.get("sample_id", "unknown")
    old_hash, new_hash = existing.get("audio_sha256"), incoming.get("audio_sha256")
    if old_hash and new_hash and old_hash != new_hash:
        raise ValueError(f"Conflicting audio identity for sample {sample}; use a distinct session/sample ID")
    if new_revision < old_revision:
        return False
    if (
        new_revision == old_revision
        and "revision" in existing
        and "revision" in incoming
        and any(str(existing.get(key) or "") != str(incoming.get(key) or "") for key in fields)
    ):
        raise ValueError(f"Conflicting content at revision {new_revision} for sample {sample}; re-export latest")
    return True


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
    _training_only(corrections_path, corrections, require_live=True)
    _training_only(output or train_jsonl, [])
    existing = _load_jsonl(train_jsonl)
    _training_only(train_jsonl, existing)
    before = len(existing)
    ids = {row.get("sample_id"): index for index, row in enumerate(existing) if row.get("sample_id")}
    seen = {tuple(part.lower() for part in _pair(row)) for row in existing}
    added = updated = stale = 0
    for row in corrections:
        _revision(row)
        en, es = _pair(row)
        if not en or not es:
            continue
        pair_key = (en.lower(), es.lower())
        item = {**row, "en": en, "es": es, "source": "active_learning", "split": "train"}
        sample = row.get("sample_id")
        if sample in ids:
            index = ids[sample]
            if not _accept_revision(existing[index], item, ("en", "es", "source_lang")):
                stale += 1
                continue
            if existing[index] != item:
                existing[index] = item
                seen = {tuple(part.lower() for part in _pair(saved)) for saved in existing}
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
        "stale_skipped": stale,
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
    _training_only(meta_path, corr_rows, require_live=True)
    out_dir = output_dir or train_dir
    _training_only(out_dir, [])
    train_meta = out_dir / "metadata.jsonl"
    if not train_meta.exists() and (out_dir / "metadata.csv").exists():
        train_meta = out_dir / "metadata.csv"
    source_meta = train_meta
    if not train_meta.exists() and output_dir is not None and train_dir.resolve() != out_dir.resolve():
        source_meta = _metadata(train_dir, language, required=False)
        train_meta = out_dir / source_meta.name
    existing = _read_metadata(source_meta) if source_meta.exists() else []
    _training_only(source_meta, existing)
    if any((row.get("source_lang") or "en") != language for row in [*corr_rows, *existing]):
        raise ValueError("Audio language does not match --language; use a separate corpus per language")
    before = len(existing)
    base_copies = []
    if source_meta.parent.resolve() != out_dir.resolve():
        for row in existing:
            name = row.get("file_name") or row.get("audio") or ""
            source_audio = (source_meta.parent / name).resolve()
            destination = (out_dir / name).resolve()
            if not name or not source_audio.is_relative_to(source_meta.parent.resolve()):
                raise ValueError("Existing corpus audio must stay within its directory")
            if not destination.is_relative_to(out_dir.resolve()):
                raise ValueError("Existing corpus audio must stay within the output directory")
            if not source_audio.is_file():
                raise FileNotFoundError(f"Existing corpus audio missing: {name}")
            base_copies.append((source_audio, destination))
    # IDs are preserved across exports. Content hashes dedupe legacy exports.
    ids = {row.get("sample_id"): index for index, row in enumerate(existing) if row.get("sample_id")}
    prepared = []
    for row in corr_rows:
        _revision(row)
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
        if not (out_dir / name).resolve().is_relative_to(out_dir.resolve()):
            raise ValueError("Correction destination audio must stay within the training corpus")
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
    added = updated = stale = 0
    copies = []
    for src, item in prepared:
        sample = item["sample_id"]
        if sample in ids:
            index = ids[sample]
            if not _accept_revision(existing[index], item, ("transcription", "audio_sha256", "source_lang")):
                stale += 1
                continue
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
        copies.append((src, out_dir / item["file_name"]))
    # Validate every revision before any audio or metadata can be changed.
    out_dir.mkdir(parents=True, exist_ok=True)
    for src, destination in [*base_copies, *copies]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, destination)
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
        "stale_skipped": stale,
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
