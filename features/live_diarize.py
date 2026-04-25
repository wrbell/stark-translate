#!/usr/bin/env python3
"""Live diarization daemon (Phase 9.6.1).

Watches a rolling-WAV file (the last ~30 s of audio that the live pipeline
keeps refreshing), runs pyannote, and writes one speaker label per chunk
to a JSONL file the operator UI's ``LiveDiarizationWatcher`` tails.

This file is a **scaffolding** ship for Phase 9.6.1: the daemon is
runnable but the rolling-WAV writer in ``dry_run_ab.py`` lands as 9.6.2
once it can be tested on the real church PC. Until then, the daemon will
just exit cleanly when the WAV doesn't appear.

Usage:
    python features/live_diarize.py \\
        --rolling-wav stark_data/live_sessions/<sid>/rolling.wav \\
        --output metrics/diarization_<sid>.jsonl \\
        --interval-s 10
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

logger = logging.getLogger("live_diarize")


def _load_pyannote():
    """Import pyannote lazily; return None if unavailable."""
    try:
        from features.diarize import run_diarization

        return run_diarization
    except Exception as exc:
        logger.warning("pyannote unavailable: %s — daemon will emit fake labels for testing", exc)
        return None


def _emit_fake_label(out_path: Path, chunk_id: int) -> None:
    """When pyannote isn't available, alternate Speaker A/B every chunk so the
    UI surfaces transitions during scaffolding tests."""
    speaker = "Speaker A" if chunk_id % 2 == 0 else "Speaker B"
    record = {"chunk_id": chunk_id, "speaker": speaker, "confidence": 0.5, "ts": time.time()}
    with out_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _emit_real_label(out_path: Path, chunk_id: int, run_diarization, wav_path: Path) -> None:
    """Run pyannote on the rolling WAV and emit the most-recent speaker label."""
    try:
        diarization = run_diarization(str(wav_path), min_speakers=1, max_speakers=4)
    except Exception as exc:
        logger.warning("diarization failed for chunk %s: %s", chunk_id, exc)
        return

    # pyannote returns an Annotation; pick the last labeled segment.
    last_speaker = None
    last_confidence = 0.0
    try:
        for segment, _, label in diarization.itertracks(yield_label=True):
            last_speaker = str(label)
            last_confidence = 1.0  # pyannote doesn't expose per-segment confidence directly
        del segment  # silence linter
    except Exception as exc:
        logger.warning("diarization parse failed for chunk %s: %s", chunk_id, exc)
        return

    if last_speaker is None:
        return

    record = {
        "chunk_id": chunk_id,
        "speaker": last_speaker,
        "confidence": last_confidence,
        "ts": time.time(),
    }
    with out_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rolling-wav", required=True, help="Rolling WAV file written by dry_run_ab")
    parser.add_argument("--output", required=True, help="JSONL output path (LiveDiarizationWatcher tails this)")
    parser.add_argument("--interval-s", type=float, default=10.0, help="Polling interval (default: 10)")
    parser.add_argument(
        "--max-iters",
        type=int,
        default=0,
        help="Max polling iterations before exit (0 = run forever; useful for tests)",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")

    rolling_wav = Path(args.rolling_wav)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stopped = {"flag": False}

    def _stop(*_):
        stopped["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    run_diarization = _load_pyannote()
    chunk_id = 0
    iters = 0
    logger.info("live_diarize started: rolling_wav=%s output=%s interval=%ss", rolling_wav, out_path, args.interval_s)

    while not stopped["flag"]:
        iters += 1
        if not rolling_wav.exists():
            logger.debug("rolling WAV not yet present at %s, waiting…", rolling_wav)
        elif run_diarization is not None:
            _emit_real_label(out_path, chunk_id, run_diarization, rolling_wav)
            chunk_id += 1
        else:
            _emit_fake_label(out_path, chunk_id)
            chunk_id += 1

        if args.max_iters and iters >= args.max_iters:
            break
        time.sleep(args.interval_s)

    logger.info("live_diarize stopped after %d iterations", iters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
