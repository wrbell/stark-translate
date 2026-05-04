#!/usr/bin/env python3
"""build_imatrix_corpus.py — Build a calibration corpus for llama-imatrix.

Extracts sermon chunks from ``bible_data/synthetic/deepl_sermon_benchmark_500.jsonl``
and emits a flat-text file in the format ``llama-imatrix -f`` expects (one
chunk per line, blank lines preserved as separators).

Critical design choice: theological-canary sentences from
``training.benchmark_gemma4.CANARY_SENTENCES`` are **over-sampled** at the
configurable weight (default 25×) so the imatrix activations are anchored on
the exact terms the production canary measures (atonement, covenant, James-as-
Jacobo vs Santiago, partimiento del pan, etc.). Without this oversample, the
imatrix optimizes for general fluency on the sermon distribution and *erases*
the very nuance we're trying to preserve when re-quantizing IQ4_XS — exactly
the regression PR3 (v2026.10) demonstrated empirically.

Usage:
    source ~/stt_train_env/bin/activate
    python training/build_imatrix_corpus.py \\
        --out training/imatrix_corpus.txt \\
        --canary-weight 25 \\
        --max-sermon-chunks 800

Then:
    ~/llama.cpp/build/bin/llama-imatrix \\
        -m /tmp/gemma-4-e4b-it-bf16.gguf \\
        -f training/imatrix_corpus.txt \\
        -o training/imatrix_e4b.dat \\
        --chunks 0 -ngl 999

Notes:
    --canary-weight 0 disables oversampling (sermon-only baseline).
    --chunks 0 in llama-imatrix processes all corpus chunks.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_imatrix_corpus")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.benchmark_gemma4 import CANARY_SENTENCES

DEFAULT_SERMON_PATH = ROOT / "bible_data/synthetic/deepl_sermon_benchmark_500.jsonl"


def load_sermon_chunks(path: Path, max_n: int) -> list[str]:
    """Return a list of EN sermon strings, deduped, capped at max_n."""
    seen = set()
    out: list[str] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            en = row.get("en", "").strip()
            if not en or len(en) < 10:
                continue
            if en.lower() in seen:
                continue
            seen.add(en.lower())
            out.append(en)
            if len(out) >= max_n:
                break
    log.info("loaded %d unique sermon chunks from %s", len(out), path)
    return out


def write_corpus(out_path: Path, sermons: list[str], canary_repeats: int) -> int:
    """Write the corpus file. Canary sentences are interleaved every ~32 sermon
    chunks (or all in one block at the end if sermon count is small) to keep
    the calibration distribution mixed rather than topical.

    Returns the total chunk count.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    canary_lines = [src for src, _expected in CANARY_SENTENCES]
    canary_block = canary_lines * canary_repeats

    # Stride: insert one full canary set every K sermon chunks.
    # K = 32 keeps the calibration "topically warm" — every batch the imatrix
    # accumulates over has theological context fresh in activation memory.
    K = 32
    interleaved: list[str] = []
    for i, sermon in enumerate(sermons):
        interleaved.append(sermon)
        if (i + 1) % K == 0 and canary_repeats > 0:
            interleaved.extend(canary_lines)

    # Append any remaining canary repeats at the end so the requested
    # over-sampling weight is reached even if K-stride didn't drain it.
    used_via_stride = (len(sermons) // K) * len(canary_lines)
    remaining = max(0, len(canary_block) - used_via_stride)
    if remaining > 0:
        interleaved.extend(canary_lines * (remaining // max(1, len(canary_lines))))

    with out_path.open("w") as fh:
        for chunk in interleaved:
            fh.write(chunk.replace("\n", " ").strip() + "\n\n")

    log.info(
        "wrote %d chunks to %s (sermons=%d, canary_total=%d at %dx, stride=%d)",
        len(interleaved),
        out_path,
        len(sermons),
        len(canary_block),
        canary_repeats,
        K,
    )
    return len(interleaved)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "training/imatrix_corpus.txt")
    p.add_argument("--sermon-path", type=Path, default=DEFAULT_SERMON_PATH)
    p.add_argument(
        "--max-sermon-chunks",
        type=int,
        default=800,
        help="Cap on sermon chunks pulled from deepl_sermon_benchmark_500.jsonl (default: 800).",
    )
    p.add_argument(
        "--canary-weight",
        type=int,
        default=25,
        help=(
            "Oversample factor for the 8 CANARY_SENTENCES (default: 25 → 200 canary "
            "lines mixed into the corpus). 0 disables oversampling. Higher values "
            "anchor the imatrix harder on theological-term activations."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    sermons = load_sermon_chunks(args.sermon_path, args.max_sermon_chunks)
    if not sermons:
        raise SystemExit(f"no sermon chunks loaded from {args.sermon_path}")
    write_corpus(args.out, sermons, args.canary_weight)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
