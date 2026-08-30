#!/usr/bin/env python3
"""Merge human/auto corrections into Whisper and translation training sets.

Closes the Phase 8 active-learning loop gap after
``prepare_finetune_data.py apply-corrections`` / ``export-whisper`` /
``export-translation``.

Usage::

    # Merge Whisper audiofolder export into the master train set
    python tools/merge_corrections.py whisper \\
        --corrections stark_data/corrections/whisper_export \\
        --train-dir stark_data/whisper_dataset_deepgram/train \\
        --replay-ratio 0.3

    # Merge translation JSONL pairs
    python tools/merge_corrections.py translation \\
        --corrections stark_data/corrections/pairs.jsonl \\
        --train-jsonl bible_data/sermon_pairs_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_translation(
    corrections_path: Path,
    train_jsonl: Path,
    *,
    output: Path | None = None,
    dedupe: bool = True,
) -> dict:
    """Append correction pairs into a translation JSONL train set."""
    corrections = _load_jsonl(corrections_path)
    existing = _load_jsonl(train_jsonl)
    before = len(existing)

    seen: set[tuple[str, str]] = set()
    if dedupe:
        for row in existing:
            en = (row.get("en") or row.get("source_text") or "").strip().lower()
            es = (row.get("es") or row.get("target_text") or "").strip().lower()
            if en and es:
                seen.add((en, es))

    added = 0
    for row in corrections:
        en = (row.get("en") or row.get("source_text") or row.get("corrected_english") or "").strip()
        es = (row.get("es") or row.get("target_text") or row.get("corrected_spanish") or "").strip()
        if not en or not es:
            continue
        key = (en.lower(), es.lower())
        if dedupe and key in seen:
            continue
        existing.append({"en": en, "es": es, "source": "active_learning"})
        seen.add(key)
        added += 1

    out = output or train_jsonl
    _write_jsonl(out, existing)
    return {
        "mode": "translation",
        "before": before,
        "added": added,
        "after": len(existing),
        "output": str(out),
    }


def merge_whisper(
    corrections_dir: Path,
    train_dir: Path,
    *,
    output_dir: Path | None = None,
    replay_ratio: float = 0.3,
    seed: int = 42,
) -> dict:
    """Merge a corrections audiofolder into the Whisper train directory.

    Expects HuggingFace audiofolder layout: ``*.wav`` + ``metadata.jsonl``
    (columns: ``file_name``, ``transcription``) under *corrections_dir*.
    """
    meta_path = corrections_dir / "metadata.jsonl"
    if not meta_path.exists():
        # Also accept export from prepare_finetune_data (nested)
        candidates = list(corrections_dir.glob("**/metadata.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No metadata.jsonl under {corrections_dir}")
        meta_path = candidates[0]
        corrections_dir = meta_path.parent

    corr_rows = _load_jsonl(meta_path)
    out_dir = output_dir or train_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_meta = out_dir / "metadata.jsonl"
    existing = _load_jsonl(train_meta) if train_meta.exists() else []
    before = len(existing)
    _ = seed, replay_ratio  # reserved for future stratified downsample of existing

    added = 0
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    for i, row in enumerate(corr_rows):
        src_name = row.get("file_name") or row.get("audio") or ""
        text = (row.get("transcription") or row.get("sentence") or row.get("text") or "").strip()
        if not src_name or not text:
            continue
        src = corrections_dir / src_name
        if not src.exists():
            continue
        dest_name = f"al_{stamp}_{i:05d}{src.suffix or '.wav'}"
        dest = out_dir / dest_name
        shutil.copy2(src, dest)
        existing.append({"file_name": dest_name, "transcription": text, "source": "active_learning"})
        added += 1

    _write_jsonl(train_meta, existing)
    return {
        "mode": "whisper",
        "before": before,
        "added": added,
        "after": len(existing),
        "output": str(out_dir),
        "replay_ratio": replay_ratio,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("translation", help="Merge EN→ES correction pairs into train JSONL")
    p_tr.add_argument("--corrections", type=Path, required=True)
    p_tr.add_argument("--train-jsonl", type=Path, required=True)
    p_tr.add_argument("--output", type=Path, default=None)
    p_tr.add_argument("--no-dedupe", action="store_true")

    p_wh = sub.add_parser("whisper", help="Merge corrections audiofolder into Whisper train dir")
    p_wh.add_argument("--corrections", type=Path, required=True)
    p_wh.add_argument("--train-dir", type=Path, required=True)
    p_wh.add_argument("--output-dir", type=Path, default=None)
    p_wh.add_argument("--replay-ratio", type=float, default=0.3)

    args = p.parse_args(argv)
    if args.cmd == "translation":
        summary = merge_translation(
            args.corrections,
            args.train_jsonl,
            output=args.output,
            dedupe=not args.no_dedupe,
        )
    else:
        summary = merge_whisper(
            args.corrections,
            args.train_dir,
            output_dir=args.output_dir,
            replay_ratio=args.replay_ratio,
        )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
