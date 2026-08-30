#!/usr/bin/env python3
"""Bench Parakeet TDT (EN-only) vs Whisper/W16 on the STT sermon holdout.

Does **not** activate Parakeet as default. Prints a comparison table; exit 0
always unless the manifest is missing. Adoption gate is manual: Parakeet must
beat W16 overall WER on EN clips at equal or better p95 latency.

Usage::

    python tools/benchmark_parakeet_en.py
    python tools/benchmark_parakeet_en.py --manifest tools/stt_bench_manifest.json --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _wer(ref: str, hyp: str) -> float:
    try:
        import jiwer

        return float(jiwer.wer(ref, hyp))
    except Exception:
        # Tiny fallback
        r = ref.lower().split()
        h = hyp.lower().split()
        if not r:
            return 0.0 if not h else 1.0
        # Hamming-ish
        miss = sum(1 for a, b in zip(r, h) if a != b) + abs(len(r) - len(h))
        return miss / max(len(r), 1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "tools" / "stt_bench_manifest.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-parakeet", action="store_true", help="Only run Whisper baseline")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    clips = json.loads(args.manifest.read_text(encoding="utf-8"))
    if isinstance(clips, dict):
        clips = clips.get("clips") or clips.get("items") or []
    if args.limit:
        clips = clips[: args.limit]

    from engines.factory import create_stt_engine

    engines = {
        "whisper": create_stt_engine(backend=args.device, stt_backend="faster-whisper"),
    }
    engines["whisper"].load()

    if not args.skip_parakeet:
        try:
            engines["parakeet"] = create_stt_engine(backend=args.device, stt_backend="parakeet")
            engines["parakeet"].load()
        except ImportError as exc:
            print(f"Parakeet unavailable ({exc}); running Whisper only.")
            args.skip_parakeet = True

    results = {name: {"wers": [], "lat_ms": []} for name in engines}

    import numpy as np

    for i, clip in enumerate(clips):
        audio_path = clip.get("audio") or clip.get("path")
        ref = clip.get("text") or clip.get("reference") or clip.get("transcript") or ""
        if not audio_path or not Path(audio_path).exists():
            print(f"  skip missing audio: {audio_path}")
            continue
        # Load audio via soundfile when available
        try:
            import soundfile as sf

            audio, sr = sf.read(audio_path)
            if sr != 16000:
                print(f"  warn: {audio_path} sr={sr} (expected 16k)")
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
        except Exception as exc:
            print(f"  skip load error {audio_path}: {exc}")
            continue

        for name, eng in engines.items():
            t0 = time.perf_counter()
            out = eng.transcribe(audio, language="en")
            lat = (time.perf_counter() - t0) * 1000
            wer = _wer(ref, out.text)
            results[name]["wers"].append(wer)
            results[name]["lat_ms"].append(out.latency_ms or lat)
            print(f"[{i}] {name}: wer={wer:.3f} lat={lat:.0f}ms  hyp={out.text[:80]!r}")

    print("\n=== Summary (EN holdout) ===")
    for name, stats in results.items():
        if not stats["wers"]:
            print(f"{name}: no clips")
            continue
        wers = sorted(stats["wers"])
        lats = sorted(stats["lat_ms"])
        p50 = lats[len(lats) // 2]
        p95 = lats[int(len(lats) * 0.95)] if len(lats) > 1 else lats[0]
        mean_wer = sum(wers) / len(wers)
        print(f"{name}: mean_wer={mean_wer:.3%}  p50={p50:.0f}ms  p95={p95:.0f}ms  n={len(wers)}")

    print(
        "\nAdoption rule: activate Parakeet for --lang en only if mean_wer ≤ Whisper/W16 "
        "and p95 ≤ Whisper p95. Default bilingual path stays Whisper."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
