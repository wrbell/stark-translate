#!/usr/bin/env python3
"""
stt_benchmark.py -- Benchmark STT backends for SRTranslate

Compares mlx-whisper against lightning-whisper-mlx (if installed) on
latency, output text, and feature support (word_timestamps, initial_prompt).

Usage:
    python stt_benchmark.py                          # Generate synthetic audio
    python stt_benchmark.py --audio path/to/file.wav # Use real audio file
    python stt_benchmark.py --runs 10                # More runs for stability
    python stt_benchmark.py --profile                # Detailed per-stage timing

Generates a summary table and writes detailed results to
metrics/stt_benchmark_<timestamp>.json.
"""

import argparse
import json
import os
import statistics
import time
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
MODEL_ID = "wbell7/distil-whisper-large-v3.5-mlx"
DEFAULT_RUNS = 5
WARMUP_RUNS = 1

# Theological prompt matching dry_run_ab.py
WHISPER_PROMPT = (
    "Sermon at Stark Road Gospel Hall. "
    "Christ Jesus, the Holy Spirit, God the Father. "
    "Atonement, propitiation, mediator, covenant, righteousness, "
    "sanctification, justification, redemption, reconciliation, "
    "repentance, reign, grace, mercy, the Gospel, epistle, apostle, "
    "Scripture, the Lord, the Word of God."
)


# ---------------------------------------------------------------------------
# Audio generation / loading
# ---------------------------------------------------------------------------


def generate_test_audio(duration_s=3.0):
    """Generate synthetic speech-like audio for benchmarking.

    Creates a 16kHz mono float32 numpy array with a mix of tones
    that exercises Whisper's encoder without requiring a real audio file.
    The content is not real speech -- latency measurements are valid but
    transcription text will be meaningless.
    """
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), dtype=np.float32)
    # Mix of frequencies to simulate broadband speech
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 500 * t)
        + 0.1 * np.sin(2 * np.pi * 1200 * t)
        + 0.05 * np.random.randn(len(t)).astype(np.float32)
    )
    # Normalize to [-0.5, 0.5] to avoid clipping
    audio = audio / (np.max(np.abs(audio)) + 1e-7) * 0.5
    return audio


def load_audio_file(path):
    """Load an audio file and return 16kHz mono float32 numpy array."""
    import subprocess
    import tempfile

    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        import scipy.io.wavfile as wav

        sr, data = wav.read(path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        if len(data.shape) > 1:
            data = data[:, 0]  # mono
        if sr != SAMPLE_RATE:
            from scipy.signal import resample

            target_len = int(len(data) * SAMPLE_RATE / sr)
            data = resample(data, target_len).astype(np.float32)
        return data
    else:
        # Use ffmpeg to convert to 16kHz mono WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    path,
                    "-ar",
                    str(SAMPLE_RATE),
                    "-ac",
                    "1",
                    "-f",
                    "wav",
                    tmp_path,
                ],
                capture_output=True,
                check=True,
            )
            return load_audio_file(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# mlx-whisper backend
# ---------------------------------------------------------------------------


def bench_mlx_whisper(
    audio, runs=DEFAULT_RUNS, warmup=WARMUP_RUNS, word_timestamps=False, initial_prompt=None, profile=False
):
    """Benchmark mlx-whisper and return structured results."""
    import mlx.core as mx
    import mlx_whisper

    mx.set_cache_limit(256 * 1024 * 1024)

    kwargs = dict(
        path_or_hf_repo=MODEL_ID,
        language="en",
        condition_on_previous_text=False,
        word_timestamps=word_timestamps,
    )
    if initial_prompt is not None:
        kwargs["initial_prompt"] = initial_prompt

    # Warmup
    for _ in range(warmup):
        mlx_whisper.transcribe(audio, **kwargs)

    latencies = []
    texts = []
    has_word_ts = False
    segment_count = 0

    for i in range(runs):
        if profile:
            # Detailed per-stage timing (approximate -- mlx-whisper doesn't
            # expose internal stages, so we measure with/without word_timestamps
            # to estimate the alignment cost)
            t0 = time.perf_counter()
            result = mlx_whisper.transcribe(audio, **kwargs)
            total_ms = (time.perf_counter() - t0) * 1000
            latencies.append(total_ms)
        else:
            t0 = time.perf_counter()
            result = mlx_whisper.transcribe(audio, **kwargs)
            latencies.append((time.perf_counter() - t0) * 1000)

        texts.append(result.get("text", "").strip())
        segments = result.get("segments", [])
        segment_count = len(segments)
        if segments and "words" in segments[0]:
            has_word_ts = True

    return {
        "backend": "mlx-whisper",
        "model": MODEL_ID,
        "word_timestamps": word_timestamps,
        "initial_prompt": initial_prompt is not None,
        "latencies_ms": [round(x, 1) for x in latencies],
        "mean_ms": round(statistics.mean(latencies), 1),
        "median_ms": round(statistics.median(latencies), 1),
        "stdev_ms": round(statistics.stdev(latencies), 1) if len(latencies) > 1 else 0.0,
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "text_sample": texts[0] if texts else "",
        "text_consistent": len(set(texts)) == 1,
        "has_word_timestamps": has_word_ts,
        "segment_count": segment_count,
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# lightning-whisper-mlx backend
# ---------------------------------------------------------------------------


def _check_lightning_available():
    """Check if lightning-whisper-mlx is installed."""
    try:
        from lightning_whisper_mlx import LightningWhisperMLX  # noqa: F401

        return True
    except ImportError:
        return False


def bench_lightning_whisper(audio, runs=DEFAULT_RUNS, warmup=WARMUP_RUNS, word_timestamps=False, initial_prompt=None):
    """Benchmark lightning-whisper-mlx and return structured results.

    NOTE: lightning-whisper-mlx's transcribe() method only accepts
    (audio_path, language). It does NOT forward word_timestamps or
    initial_prompt. We benchmark it as-is to show the actual integration gap.

    Additionally, lightning-whisper-mlx requires a FILE PATH, not a numpy
    array. We write to a temp file for each call.
    """
    import tempfile

    import scipy.io.wavfile as wav
    from lightning_whisper_mlx import LightningWhisperMLX

    # Write audio to temp file (required -- no numpy array input)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    wav.write(tmp_path, SAMPLE_RATE, audio)

    try:
        whisper = LightningWhisperMLX(
            model="distil-large-v3",
            batch_size=12,
            quant=None,
        )

        # Warmup
        for _ in range(warmup):
            whisper.transcribe(tmp_path, language="en")

        latencies = []
        texts = []

        for i in range(runs):
            t0 = time.perf_counter()
            result = whisper.transcribe(tmp_path, language="en")
            latencies.append((time.perf_counter() - t0) * 1000)
            texts.append(result.get("text", "").strip())

        return {
            "backend": "lightning-whisper-mlx",
            "model": "distil-large-v3",
            "word_timestamps": False,  # not supported via API
            "initial_prompt": False,  # not forwarded via API
            "latencies_ms": [round(x, 1) for x in latencies],
            "mean_ms": round(statistics.mean(latencies), 1),
            "median_ms": round(statistics.median(latencies), 1),
            "stdev_ms": round(statistics.stdev(latencies), 1) if len(latencies) > 1 else 0.0,
            "min_ms": round(min(latencies), 1),
            "max_ms": round(max(latencies), 1),
            "text_sample": texts[0] if texts else "",
            "text_consistent": len(set(texts)) == 1,
            "has_word_timestamps": False,
            "segment_count": len(result.get("segments", [])),
            "runs": runs,
            "notes": [
                "word_timestamps not supported (PR #20 unmerged)",
                "initial_prompt not forwarded (PR #23 unmerged)",
                "Requires file path input (no numpy array support)",
            ],
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Profile mode: measure word_timestamps overhead
# ---------------------------------------------------------------------------


def profile_word_timestamps(audio, runs=3):
    """Measure the latency cost of enabling word_timestamps in mlx-whisper."""
    print("\n--- Profiling word_timestamps overhead ---")

    result_no_wt = bench_mlx_whisper(audio, runs=runs, word_timestamps=False)
    result_wt = bench_mlx_whisper(audio, runs=runs, word_timestamps=True)

    overhead_ms = result_wt["mean_ms"] - result_no_wt["mean_ms"]
    overhead_pct = (overhead_ms / result_no_wt["mean_ms"]) * 100 if result_no_wt["mean_ms"] > 0 else 0

    print(f"  Without word_timestamps: {result_no_wt['mean_ms']:.0f}ms (median {result_no_wt['median_ms']:.0f}ms)")
    print(f"  With word_timestamps:    {result_wt['mean_ms']:.0f}ms (median {result_wt['median_ms']:.0f}ms)")
    print(f"  Overhead:                {overhead_ms:+.0f}ms ({overhead_pct:+.0f}%)")

    return {
        "without_wt_mean_ms": result_no_wt["mean_ms"],
        "with_wt_mean_ms": result_wt["mean_ms"],
        "overhead_ms": round(overhead_ms, 1),
        "overhead_pct": round(overhead_pct, 1),
    }


def profile_initial_prompt(audio, runs=3):
    """Measure the latency cost of initial_prompt in mlx-whisper."""
    print("\n--- Profiling initial_prompt overhead ---")

    result_no_prompt = bench_mlx_whisper(
        audio,
        runs=runs,
        word_timestamps=False,
        initial_prompt=None,
    )
    result_prompt = bench_mlx_whisper(
        audio,
        runs=runs,
        word_timestamps=False,
        initial_prompt=WHISPER_PROMPT,
    )

    overhead_ms = result_prompt["mean_ms"] - result_no_prompt["mean_ms"]
    overhead_pct = (overhead_ms / result_no_prompt["mean_ms"]) * 100 if result_no_prompt["mean_ms"] > 0 else 0

    print(
        f"  Without initial_prompt:  {result_no_prompt['mean_ms']:.0f}ms (median {result_no_prompt['median_ms']:.0f}ms)"
    )
    print(f"  With initial_prompt:     {result_prompt['mean_ms']:.0f}ms (median {result_prompt['median_ms']:.0f}ms)")
    print(f"  Overhead:                {overhead_ms:+.0f}ms ({overhead_pct:+.0f}%)")

    return {
        "without_prompt_mean_ms": result_no_prompt["mean_ms"],
        "with_prompt_mean_ms": result_prompt["mean_ms"],
        "overhead_ms": round(overhead_ms, 1),
        "overhead_pct": round(overhead_pct, 1),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_comparison_table(results):
    """Print a formatted comparison table of benchmark results."""
    print(f"\n{'=' * 72}")
    print("STT BENCHMARK RESULTS")
    print(f"{'=' * 72}")

    # Header
    headers = ["Metric", *[r["backend"] for r in results]]
    col_widths = [28] + [max(20, len(h) + 2) for h in headers[1:]]

    header_line = ""
    for i, h in enumerate(headers):
        header_line += h.ljust(col_widths[i] if i < len(col_widths) else 20)
    print(header_line)
    print("-" * sum(col_widths))

    # Rows
    rows = [
        ("Model", [r["model"] for r in results]),
        ("Mean latency (ms)", [f"{r['mean_ms']:.0f}" for r in results]),
        ("Median latency (ms)", [f"{r['median_ms']:.0f}" for r in results]),
        ("Stdev (ms)", [f"{r['stdev_ms']:.0f}" for r in results]),
        ("Min / Max (ms)", [f"{r['min_ms']:.0f} / {r['max_ms']:.0f}" for r in results]),
        (
            "word_timestamps",
            ["YES" if r["has_word_timestamps"] else ("N/A" if not r["word_timestamps"] else "NO") for r in results],
        ),
        ("initial_prompt", ["YES" if r["initial_prompt"] else "NO (not supported)" for r in results]),
        ("Segments", [str(r["segment_count"]) for r in results]),
        ("Text consistent", ["YES" if r["text_consistent"] else "NO" for r in results]),
        ("Runs", [str(r["runs"]) for r in results]),
    ]

    for label, values in rows:
        line = label.ljust(col_widths[0])
        for i, v in enumerate(values):
            w = col_widths[i + 1] if i + 1 < len(col_widths) else 20
            line += v.ljust(w)
        print(line)

    print(f"\n{'=' * 72}")

    # Text comparison
    print("\nTranscription output comparison:")
    for r in results:
        text = r["text_sample"][:80] + ("..." if len(r["text_sample"]) > 80 else "")
        print(f"  [{r['backend']}] {text or '(empty)'}")

    # Notes
    for r in results:
        notes = r.get("notes", [])
        if notes:
            print(f"\n  {r['backend']} notes:")
            for note in notes:
                print(f"    - {note}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark STT backends for SRTranslate",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to audio file (WAV/MP3/etc). Default: generate synthetic audio.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Duration of synthetic audio in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Number of timed runs per backend (default: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_RUNS,
        help=f"Number of warmup runs per backend (default: {WARMUP_RUNS})",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run detailed profiling (word_timestamps overhead, initial_prompt overhead)",
    )
    parser.add_argument(
        "--skip-lightning",
        action="store_true",
        help="Skip lightning-whisper-mlx benchmark even if installed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: metrics/stt_benchmark_<timestamp>.json)",
    )
    args = parser.parse_args()

    # Load or generate audio
    if args.audio:
        print(f"Loading audio from {args.audio}...")
        audio = load_audio_file(args.audio)
        audio_source = args.audio
    else:
        print(f"Generating {args.duration}s synthetic audio...")
        audio = generate_test_audio(args.duration)
        audio_source = f"synthetic_{args.duration}s"

    audio_duration = len(audio) / SAMPLE_RATE
    print(f"Audio: {audio_duration:.1f}s, {len(audio)} samples, {SAMPLE_RATE}Hz")

    all_results = []
    profile_data = {}

    # --- mlx-whisper benchmark ---
    print("\n--- mlx-whisper (word_timestamps=True, initial_prompt=True) ---")
    result_mlx = bench_mlx_whisper(
        audio,
        runs=args.runs,
        warmup=args.warmup,
        word_timestamps=True,
        initial_prompt=WHISPER_PROMPT,
    )
    all_results.append(result_mlx)
    print(f"  Mean: {result_mlx['mean_ms']:.0f}ms | Median: {result_mlx['median_ms']:.0f}ms")

    # --- lightning-whisper-mlx benchmark ---
    lightning_available = _check_lightning_available() and not args.skip_lightning
    if lightning_available:
        print("\n--- lightning-whisper-mlx (file-based, no word_timestamps/prompt) ---")
        try:
            result_lightning = bench_lightning_whisper(
                audio,
                runs=args.runs,
                warmup=args.warmup,
            )
            all_results.append(result_lightning)
            print(f"  Mean: {result_lightning['mean_ms']:.0f}ms | Median: {result_lightning['median_ms']:.0f}ms")
        except Exception as e:
            print(f"  ERROR: lightning-whisper-mlx benchmark failed: {e}")
    else:
        if args.skip_lightning:
            print("\n--- lightning-whisper-mlx: skipped (--skip-lightning) ---")
        else:
            print("\n--- lightning-whisper-mlx: not installed (pip install lightning-whisper-mlx) ---")

    # --- Profile mode ---
    if args.profile:
        profile_data["word_timestamps"] = profile_word_timestamps(audio, runs=args.runs)
        profile_data["initial_prompt"] = profile_initial_prompt(audio, runs=args.runs)

    # --- Results ---
    print_comparison_table(all_results)

    if profile_data:
        print(f"\n{'=' * 72}")
        print("PROFILING SUMMARY")
        print(f"{'=' * 72}")
        if "word_timestamps" in profile_data:
            wt = profile_data["word_timestamps"]
            print(f"  word_timestamps overhead: {wt['overhead_ms']:+.0f}ms ({wt['overhead_pct']:+.0f}%)")
        if "initial_prompt" in profile_data:
            ip = profile_data["initial_prompt"]
            print(f"  initial_prompt overhead:  {ip['overhead_ms']:+.0f}ms ({ip['overhead_pct']:+.0f}%)")
        print(f"{'=' * 72}")

    # --- Save results ---
    output_path = args.output
    if output_path is None:
        os.makedirs("metrics", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"metrics/stt_benchmark_{timestamp}.json"

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "audio_source": audio_source,
        "audio_duration_s": round(audio_duration, 2),
        "results": all_results,
        "profile": profile_data if profile_data else None,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
