#!/usr/bin/env python3
"""Bridge live session diagnostics to fine-tuning data formats.

Converts diagnostics JSONL (written by dry_run_ab.py) into training-ready
formats for train_whisper.py (audiofolder) and train_gemma.py (JSONL pairs).

Subcommands:
    extract-review-queue  Export flagged segments as TSV for human review
    apply-corrections     Write corrections back to diagnostics JSONL
    export-whisper        Create Whisper audiofolder dataset from sessions
    export-translation    Create EN→ES JSONL pairs for translation training
    summary               Show training data readiness overview
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants — match prepare_whisper_dataset.py thresholds
# ---------------------------------------------------------------------------
MAX_COMPRESSION_RATIO = 2.4
MIN_AVG_LOGPROB = -1.0
MAX_NO_SPEECH_PROB = 0.6

METRICS_DIR = "metrics"

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def resolve_diagnostics_paths(
    session_ids: list[str] | None = None,
    metrics_dir: str = METRICS_DIR,
) -> list[Path]:
    """Glob diagnostics JSONL files, optionally filtered by session IDs."""
    mdir = Path(metrics_dir)
    if not mdir.is_dir():
        return []
    all_paths = sorted(mdir.glob("diagnostics_*.jsonl"))
    if not session_ids:
        return all_paths
    filtered = []
    for p in all_paths:
        # Filename: diagnostics_YYYYMMDD_HHMMSS.jsonl
        stem = p.stem  # diagnostics_20260301_113532
        sid = stem.removeprefix("diagnostics_")
        if sid in session_ids:
            filtered.append(p)
    return filtered


def load_diagnostics(paths: list[Path]) -> list[dict]:
    """Parse JSONL files, filter out event records, attach source metadata."""
    records = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"  warning: skipping invalid JSON at {p.name}:{line_num}",
                        file=sys.stderr,
                    )
                    continue
                # Skip event records (session_start, session_end, etc.)
                if "event" in rec:
                    continue
                # Must have chunk_id to be a data record
                if "chunk_id" not in rec:
                    continue
                rec["_source_file"] = str(p)
                records.append(rec)
    return records


def apply_auto_corrections_to_record(record: dict) -> tuple[str, list]:
    """Lazy-import correct_stt_output from dry_run_ab and apply to english."""
    from dry_run_ab import correct_stt_output

    english = record.get("english", "")
    if not english:
        return english, []
    corrected, corrections = correct_stt_output(english)
    return corrected, corrections


def _get_segment_metadata(record: dict) -> dict:
    """Extract aggregate segment metadata for confidence filtering."""
    seg_meta = record.get("segment_metadata") or []
    if not seg_meta:
        return {}
    avg_logprobs = [s.get("avg_logprob", 0) for s in seg_meta if s]
    compression_ratios = [s.get("compression_ratio", 0) for s in seg_meta if s]
    no_speech_probs = [s.get("no_speech_prob", 0) for s in seg_meta if s]
    return {
        "avg_logprob": sum(avg_logprobs) / len(avg_logprobs) if avg_logprobs else 0,
        "max_compression_ratio": max(compression_ratios) if compression_ratios else 0,
        "max_no_speech_prob": max(no_speech_probs) if no_speech_probs else 0,
    }


def _passes_whisper_filter(record: dict) -> bool:
    """Check if a record passes Whisper confidence thresholds."""
    meta = _get_segment_metadata(record)
    if not meta:
        return True  # No metadata = can't filter, include by default
    if meta.get("max_compression_ratio", 0) > MAX_COMPRESSION_RATIO:
        return False
    if meta.get("avg_logprob", 0) < MIN_AVG_LOGPROB:
        return False
    if meta.get("max_no_speech_prob", 0) > MAX_NO_SPEECH_PROB and meta.get("avg_logprob", 0) < -0.5:
        return False
    return True


# ---------------------------------------------------------------------------
# Subcommand: extract-review-queue
# ---------------------------------------------------------------------------


def cmd_extract_review_queue(args: argparse.Namespace) -> None:
    """Export flagged segments as TSV for human review."""
    paths = resolve_diagnostics_paths(args.session, args.metrics_dir)
    if not paths:
        print("No diagnostics files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(paths)} diagnostics file(s)...")
    records = load_diagnostics(paths)
    print(f"  {len(records)} chunk records loaded")

    # Filter: priority >= min_priority AND not already corrected
    flagged = [
        r
        for r in records
        if r.get("review_priority", 0) >= args.min_priority
        and not r.get("corrected_english")
        and not r.get("corrected_spanish")
    ]

    # Sort by priority descending, then chunk_id ascending
    flagged.sort(key=lambda r: (-r.get("review_priority", 0), r.get("chunk_id", 0)))

    if args.top_n and args.top_n > 0:
        flagged = flagged[: args.top_n]

    if not flagged:
        print("No segments match the review criteria.")
        return

    # Write TSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "chunk_id",
        "session",
        "recorded_english",
        "corrected_english",
        "spanish_gemma",
        "spanish_marian",
        "corrected_spanish",
        "stt_confidence",
        "qe_a",
        "review_priority",
        "audio_path",
        "correction_source",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in flagged:
            writer.writerow(
                {
                    "chunk_id": r.get("chunk_id", ""),
                    "session": r.get("session", ""),
                    "recorded_english": r.get("english", ""),
                    "corrected_english": "",
                    "spanish_gemma": r.get("spanish_gemma", ""),
                    "spanish_marian": r.get("spanish_marian", ""),
                    "corrected_spanish": "",
                    "stt_confidence": r.get("stt_confidence", ""),
                    "qe_a": r.get("qe_a", ""),
                    "review_priority": r.get("review_priority", ""),
                    "audio_path": r.get("audio_path", ""),
                    "correction_source": "",
                }
            )

    print(f"Wrote {len(flagged)} segments to {out_path}")


# ---------------------------------------------------------------------------
# Subcommand: apply-corrections
# ---------------------------------------------------------------------------


def cmd_apply_corrections(args: argparse.Namespace) -> None:
    """Read corrections TSV and update diagnostics JSONL files."""
    tsv_path = Path(args.corrections)
    if not tsv_path.exists():
        print(f"Corrections file not found: {tsv_path}", file=sys.stderr)
        sys.exit(1)

    # Parse TSV into lookup: (session, chunk_id) -> correction dict
    corrections = {}
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (row.get("session", ""), str(row.get("chunk_id", "")))
            corr_en = row.get("corrected_english", "").strip()
            corr_es = row.get("corrected_spanish", "").strip()
            source = row.get("correction_source", "").strip()
            if corr_en or corr_es:
                corrections[key] = {
                    "corrected_english": corr_en or None,
                    "corrected_spanish": corr_es or None,
                    "correction_source": source or "human",
                }

    print(f"Loaded {len(corrections)} human correction(s) from {tsv_path}")

    # Load all diagnostics to find which files to update
    paths = resolve_diagnostics_paths(args.session, args.metrics_dir)
    if not paths:
        print("No diagnostics files found.", file=sys.stderr)
        sys.exit(1)

    human_count = 0
    auto_count = 0

    for diag_path in paths:
        lines = diag_path.read_text(encoding="utf-8").splitlines()
        updated = False
        new_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                new_lines.append(line)
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                new_lines.append(line)
                continue

            key = (rec.get("session", ""), str(rec.get("chunk_id", "")))

            # Apply human correction if available
            if key in corrections:
                corr = corrections[key]
                if corr["corrected_english"]:
                    rec["corrected_english"] = corr["corrected_english"]
                if corr["corrected_spanish"]:
                    rec["corrected_spanish"] = corr["corrected_spanish"]
                rec["correction_source"] = corr["correction_source"]
                rec["correction_timestamp"] = datetime.now(UTC).isoformat()
                human_count += 1
                updated = True
            elif args.auto_correct and not rec.get("corrected_english"):
                # Auto-correct via correct_stt_output
                corrected, corr_list = apply_auto_corrections_to_record(rec)
                if corr_list:  # Only update if corrections were made
                    rec["corrected_english"] = corrected
                    rec["correction_source"] = "auto"
                    rec["correction_timestamp"] = datetime.now(UTC).isoformat()
                    auto_count += 1
                    updated = True

            new_lines.append(json.dumps(rec, ensure_ascii=False))

        if updated and not args.dry_run:
            # Atomic write: temp file + os.replace()
            fd, tmp = tempfile.mkstemp(dir=str(diag_path.parent), suffix=".jsonl.tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write("\n".join(new_lines) + "\n")
                os.replace(tmp, str(diag_path))
            except Exception:
                os.unlink(tmp)
                raise
            print(f"  Updated {diag_path.name}")
        elif updated and args.dry_run:
            print(f"  [dry-run] Would update {diag_path.name}")

    print(f"Applied {human_count} human + {auto_count} auto corrections")


# ---------------------------------------------------------------------------
# Subcommand: export-whisper
# ---------------------------------------------------------------------------


def cmd_export_whisper(args: argparse.Namespace) -> None:
    """Export corrected session data as Whisper audiofolder dataset."""
    paths = resolve_diagnostics_paths(args.session, args.metrics_dir)
    if not paths:
        print("No diagnostics files found.", file=sys.stderr)
        sys.exit(1)

    records = load_diagnostics(paths)
    print(f"Loaded {len(records)} chunk records")

    # Filter by confidence thresholds
    eligible = [r for r in records if _passes_whisper_filter(r)]
    print(f"  {len(eligible)} pass confidence filter (rejected {len(records) - len(eligible)})")

    # Filter out empty transcriptions
    eligible = [r for r in eligible if (r.get("corrected_english") or r.get("english", "")).strip()]
    print(f"  {len(eligible)} have non-empty transcriptions")

    if not eligible:
        print("No eligible records for Whisper export.")
        return

    # Stratified train/eval split
    random.seed(42)
    random.shuffle(eligible)
    eval_count = int(len(eligible) * args.eval_ratio)
    if args.eval_ratio > 0 and eval_count == 0 and len(eligible) > 1:
        eval_count = 1  # At least 1 eval sample when ratio > 0
    eval_set = eligible[:eval_count]
    train_set = eligible[eval_count:]

    out_dir = Path(args.output)
    accent = args.accent

    for split_name, split_data in [("train", train_set), ("eval", eval_set)]:
        split_dir = out_dir / split_name / accent
        split_dir.mkdir(parents=True, exist_ok=True)

        metadata_rows = []
        for rec in split_data:
            audio_path = rec.get("audio_path", "")
            if not audio_path:
                continue

            src = Path(audio_path)
            dst = split_dir / src.name

            # Create symlink to original audio
            if not dst.exists():
                try:
                    dst.symlink_to(src.resolve())
                except OSError:
                    # Fallback: try relative path or skip
                    try:
                        dst.symlink_to(os.path.relpath(src, dst.parent))
                    except OSError:
                        print(
                            f"  warning: could not symlink {src.name}",
                            file=sys.stderr,
                        )
                        continue

            transcription = (rec.get("corrected_english") or rec.get("english", "")).strip()
            metadata_rows.append(
                {
                    "file_name": f"{accent}/{src.name}",
                    "transcription": transcription,
                    "accent": accent,
                }
            )

        # Write metadata.csv
        meta_path = out_dir / split_name / "metadata.csv"
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "transcription", "accent"])
            writer.writeheader()
            writer.writerows(metadata_rows)

        print(f"  {split_name}: {len(metadata_rows)} samples -> {meta_path}")

    total_dur = sum(r.get("utterance_dur", 0) or 0 for r in eligible)
    print(f"Total audio: {total_dur / 3600:.2f} hours")


# ---------------------------------------------------------------------------
# Subcommand: export-translation
# ---------------------------------------------------------------------------


def cmd_export_translation(args: argparse.Namespace) -> None:
    """Export EN→ES pairs as JSONL for translation training."""
    paths = resolve_diagnostics_paths(args.session, args.metrics_dir)
    if not paths:
        print("No diagnostics files found.", file=sys.stderr)
        sys.exit(1)

    records = load_diagnostics(paths)
    print(f"Loaded {len(records)} chunk records")

    pairs = []
    for rec in records:
        # English: prefer corrected
        en = (rec.get("corrected_english") or rec.get("english", "")).strip()
        if not en:
            continue

        # Spanish cascade: corrected > gemma > marian
        es = (
            (rec.get("corrected_spanish") or "").strip()
            or (rec.get("spanish_gemma") or "").strip()
            or (rec.get("spanish_marian") or "").strip()
        )
        if not es:
            continue

        # QE filter (skip for human corrections)
        is_human = rec.get("correction_source") == "human"
        if not is_human and args.min_qe > 0:
            try:
                qe = float(rec.get("qe_a") or 0)
            except (ValueError, TypeError):
                qe = 0
            if qe < args.min_qe:
                continue

        pairs.append({"en": en, "es": es})

    if not pairs:
        print("No eligible translation pairs.")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Wrote {len(pairs)} translation pairs to {out_path}")


# ---------------------------------------------------------------------------
# Subcommand: summary
# ---------------------------------------------------------------------------


def cmd_summary(args: argparse.Namespace) -> None:
    """Print training data readiness overview."""
    paths = resolve_diagnostics_paths(args.session, args.metrics_dir)
    if not paths:
        print("No diagnostics files found.")
        return

    records = load_diagnostics(paths)
    total = len(records)
    if total == 0:
        print("No chunk records found.")
        return

    # Session breakdown
    sessions = {}
    for r in records:
        sid = r.get("session", "unknown")
        sessions.setdefault(sid, 0)
        sessions[sid] += 1

    # Correction breakdown
    human_corrected = sum(1 for r in records if r.get("correction_source") == "human")
    auto_corrected = sum(1 for r in records if r.get("correction_source") == "auto")
    uncorrected = total - human_corrected - auto_corrected

    # Whisper-eligible
    whisper_eligible = [r for r in records if _passes_whisper_filter(r)]
    whisper_eligible = [r for r in whisper_eligible if (r.get("corrected_english") or r.get("english", "")).strip()]
    total_dur = sum(r.get("utterance_dur", 0) or 0 for r in whisper_eligible)

    # Translation-eligible (QE >= 0.6 or human-corrected)
    trans_eligible = 0
    for r in records:
        en = (r.get("corrected_english") or r.get("english", "")).strip()
        es = (
            (r.get("corrected_spanish") or "").strip()
            or (r.get("spanish_gemma") or "").strip()
            or (r.get("spanish_marian") or "").strip()
        )
        if not en or not es:
            continue
        is_human = r.get("correction_source") == "human"
        try:
            qe = float(r.get("qe_a") or 0)
        except (ValueError, TypeError):
            qe = 0
        if is_human or qe >= 0.6:
            trans_eligible += 1

    # Auto-correction potential
    auto_potential = 0
    try:
        from dry_run_ab import correct_stt_output

        for r in records:
            if r.get("corrected_english") or r.get("correction_source"):
                continue
            en = r.get("english", "")
            if en:
                _, corrs = correct_stt_output(en)
                if corrs:
                    auto_potential += 1
    except ImportError:
        auto_potential = -1  # Can't compute

    # Print report
    print("=" * 60)
    print("FINE-TUNING DATA READINESS SUMMARY")
    print("=" * 60)
    print()

    print(f"Sessions: {len(sessions)}")
    for sid, count in sorted(sessions.items()):
        print(f"  {sid}: {count} chunks")
    print()

    print(f"Total chunks: {total}")
    print(f"  Human-corrected: {human_corrected}")
    print(f"  Auto-corrected:  {auto_corrected}")
    print(f"  Uncorrected:     {uncorrected}")
    print()

    print(f"Whisper-eligible: {len(whisper_eligible)} chunks")
    print(f"  Audio duration:  {total_dur / 3600:.2f} hours ({total_dur:.0f}s)")
    print()

    print(f"Translation-eligible: {trans_eligible} pairs (QE >= 0.6 or human)")
    print()

    if auto_potential >= 0:
        print(f"Auto-correction potential: {auto_potential} chunks have rule matches")
    else:
        print("Auto-correction potential: [could not import dry_run_ab]")
    print()

    # Review priority distribution
    priorities = {}
    for r in records:
        p = r.get("review_priority", 0)
        priorities.setdefault(p, 0)
        priorities[p] += 1
    print("Review priority distribution:")
    for p in sorted(priorities):
        print(f"  priority {p}: {priorities[p]} chunks")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bridge live session diagnostics to fine-tuning data formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- extract-review-queue ---
    p_extract = sub.add_parser(
        "extract-review-queue",
        help="Export flagged segments as TSV for human review",
    )
    p_extract.add_argument(
        "--session",
        nargs="*",
        help="Session IDs to include (default: all)",
    )
    p_extract.add_argument(
        "--min-priority",
        type=int,
        default=1,
        help="Minimum review_priority to include (default: 1)",
    )
    p_extract.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="Limit to top N segments by priority (default: all)",
    )
    p_extract.add_argument(
        "-o",
        "--output",
        default="stark_data/corrections/review_queue.tsv",
        help="Output TSV path",
    )
    p_extract.add_argument(
        "--metrics-dir",
        default=METRICS_DIR,
        help="Directory containing diagnostics JSONL files",
    )
    p_extract.set_defaults(func=cmd_extract_review_queue)

    # --- apply-corrections ---
    p_apply = sub.add_parser(
        "apply-corrections",
        help="Write corrections from TSV back to diagnostics JSONL",
    )
    p_apply.add_argument(
        "-c",
        "--corrections",
        required=True,
        help="Path to corrections TSV",
    )
    p_apply.add_argument(
        "--auto-correct",
        action="store_true",
        help="Also apply rule-based STT corrections to uncorrected records",
    )
    p_apply.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without writing files",
    )
    p_apply.add_argument(
        "--session",
        nargs="*",
        help="Session IDs to process (default: all)",
    )
    p_apply.add_argument(
        "--metrics-dir",
        default=METRICS_DIR,
        help="Directory containing diagnostics JSONL files",
    )
    p_apply.set_defaults(func=cmd_apply_corrections)

    # --- export-whisper ---
    p_whisper = sub.add_parser(
        "export-whisper",
        help="Create Whisper audiofolder dataset from sessions",
    )
    p_whisper.add_argument(
        "-o",
        "--output",
        default="stark_data/whisper_dataset_live",
        help="Output directory for audiofolder",
    )
    p_whisper.add_argument(
        "--accent",
        default="church_live",
        help="Accent label for metadata (default: church_live)",
    )
    p_whisper.add_argument(
        "--eval-ratio",
        type=float,
        default=0.05,
        help="Fraction of data for eval split (default: 0.05)",
    )
    p_whisper.add_argument(
        "--session",
        nargs="*",
        help="Session IDs to include (default: all)",
    )
    p_whisper.add_argument(
        "--metrics-dir",
        default=METRICS_DIR,
        help="Directory containing diagnostics JSONL files",
    )
    p_whisper.set_defaults(func=cmd_export_whisper)

    # --- export-translation ---
    p_trans = sub.add_parser(
        "export-translation",
        help="Create EN→ES JSONL pairs for translation training",
    )
    p_trans.add_argument(
        "-o",
        "--output",
        default="stark_data/live_sessions/sermon_pairs.jsonl",
        help="Output JSONL path",
    )
    p_trans.add_argument(
        "--min-qe",
        type=float,
        default=0.6,
        help="Minimum QE score to include (default: 0.6)",
    )
    p_trans.add_argument(
        "--session",
        nargs="*",
        help="Session IDs to include (default: all)",
    )
    p_trans.add_argument(
        "--metrics-dir",
        default=METRICS_DIR,
        help="Directory containing diagnostics JSONL files",
    )
    p_trans.set_defaults(func=cmd_export_translation)

    # --- summary ---
    p_summary = sub.add_parser(
        "summary",
        help="Show training data readiness overview",
    )
    p_summary.add_argument(
        "--session",
        nargs="*",
        help="Session IDs to include (default: all)",
    )
    p_summary.add_argument(
        "--metrics-dir",
        default=METRICS_DIR,
        help="Directory containing diagnostics JSONL files",
    )
    p_summary.set_defaults(func=cmd_summary)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
