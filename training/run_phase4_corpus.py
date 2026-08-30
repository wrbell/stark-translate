#!/usr/bin/env python3
"""Phase 4 corpus preprocessor — discover raw sermon WAVs, run the 10-step pipeline, write status.

Wraps ``training/preprocess_audio.py`` so a full-corpus run is one command with a
machine-readable status file for the active-learning / W17 gates.

Usage (WSL, with sermon WAVs present)::

    python training/run_phase4_corpus.py \\
        --input stark_data/raw \\
        --output stark_data/cleaned \\
        --resume

    # Dry-run: discover files and write a pending status without processing
    python training/run_phase4_corpus.py --input stark_data/raw --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def discover_wavs(input_dir: Path) -> list[str]:
    """Recursively find WAV files under *input_dir* (file or directory)."""
    if input_dir.is_file():
        return [str(input_dir)] if input_dir.suffix.lower() == ".wav" else []
    files = sorted(glob.glob(os.path.join(str(input_dir), "*.wav")))
    files += sorted(glob.glob(os.path.join(str(input_dir), "*", "*.wav")))
    files += sorted(glob.glob(os.path.join(str(input_dir), "*", "*", "*.wav")))
    seen: set[str] = set()
    unique: list[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def write_status(
    output_dir: Path,
    *,
    input_dir: Path,
    discovered: int,
    completed: int,
    rejected: int,
    errors: int,
    elapsed_s: float,
    dry_run: bool,
    extra: dict | None = None,
) -> Path:
    """Write ``phase4_status.json`` under *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "phase": 4,
        "name": "audio_preprocess",
        "written_at": datetime.now(UTC).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "dry_run": dry_run,
        "discovered_wavs": discovered,
        "completed": completed,
        "rejected": rejected,
        "errors": errors,
        "elapsed_s": round(elapsed_s, 1),
        "ready_for_training": (not dry_run) and discovered > 0 and completed > 0 and errors == 0,
    }
    if extra:
        status.update(extra)
    path = output_dir / "phase4_status.json"
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return path


def count_log_outcomes(log_path: Path) -> tuple[int, int, int]:
    """Return (completed, rejected, errors) from preprocessing_log.json."""
    if not log_path.exists():
        return 0, 0, 0
    try:
        logs = json.loads(log_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0, 0, 0
    completed = rejected = errors = 0
    for entry in logs:
        steps = entry.get("steps") or {}
        if "error" in steps:
            errors += 1
        elif steps.get("initial_gate") and steps["initial_gate"] not in ("pass", "done"):
            # initial_gate stores reason string; "pass" means accepted
            if steps.get("chunks_written", 0) == 0 and "final_gate" not in steps:
                # rejected early if convert done but no chunks — check gate reason
                reason = steps.get("initial_gate", "")
                if reason and reason not in ("pass", "done"):
                    rejected += 1
                else:
                    completed += 1
            else:
                completed += 1
        else:
            completed += 1
    return completed, rejected, errors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", "-i", type=Path, default=Path("stark_data/raw"))
    p.add_argument("--output", "-o", type=Path, default=Path("stark_data/cleaned"))
    p.add_argument("--resume", action="store_true", help="Skip already-processed stems")
    p.add_argument("--skip-demucs", action="store_true")
    p.add_argument("--diarize", action="store_true", help="Enable pyannote diarization")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover WAVs and write phase4_status.json without processing",
    )
    args = p.parse_args(argv)

    input_dir = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    output_dir = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output

    wavs = discover_wavs(input_dir)
    print(f"Discovered {len(wavs)} WAV file(s) under {input_dir}")

    if args.dry_run or not wavs:
        status_path = write_status(
            output_dir,
            input_dir=input_dir,
            discovered=len(wavs),
            completed=0,
            rejected=0,
            errors=0,
            elapsed_s=0.0,
            dry_run=True,
            extra={"message": "no WAVs found" if not wavs else "dry-run only"},
        )
        print(f"Wrote status: {status_path}")
        return 0 if args.dry_run or not wavs else 1

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "training" / "preprocess_audio.py"),
        "--input",
        str(input_dir),
        "--output",
        str(output_dir),
    ]
    if args.resume:
        cmd.append("--resume")
    if args.skip_demucs:
        cmd.append("--skip-demucs")
    if args.diarize:
        cmd.append("--diarize")

    t0 = time.perf_counter()
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.perf_counter() - t0

    log_path = output_dir / "preprocessing_log.json"
    completed, rejected, errors = count_log_outcomes(log_path)
    status_path = write_status(
        output_dir,
        input_dir=input_dir,
        discovered=len(wavs),
        completed=completed,
        rejected=rejected,
        errors=errors + (1 if result.returncode != 0 else 0),
        elapsed_s=elapsed,
        dry_run=False,
        extra={"preprocess_exit_code": result.returncode},
    )
    print(f"Wrote status: {status_path}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
