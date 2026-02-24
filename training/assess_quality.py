#!/usr/bin/env python3
"""
assess_quality.py — Baseline Transcript Quality Assessment

Before fine-tuning, establish a WER baseline on 50-100 randomly sampled segments.
Samples segments for manual review and computes WER against human corrections.

Usage:
    # Step 1: Generate sample for manual review
    python assess_quality.py sample --input stark_data/transcripts --n 100

    # Step 2: (Manually add 'manual_text' field to assessment_sample.jsonl)

    # Step 3: Compute WER
    python assess_quality.py evaluate --input assessment_sample.jsonl
"""

import argparse
import json
import random
from pathlib import Path


def sample_for_assessment(transcript_dir, n=100, output_path="assessment_sample.jsonl", seed=42):
    """Randomly sample segments for manual review."""
    random.seed(seed)
    all_segments = []

    for jf in sorted(Path(transcript_dir).glob("*.json")):
        if jf.name.startswith("_"):
            continue
        with open(jf) as f:
            data = json.load(f)

        audio_path = data.get("audio_path", str(jf))
        for seg in data.get("segments", []):
            ts = seg.get("timestamp", [None, None])
            all_segments.append(
                {
                    "audio_file": audio_path,
                    "start": ts[0] if isinstance(ts, list) else seg.get("start"),
                    "end": ts[1] if isinstance(ts, list) else seg.get("end"),
                    "auto_text": seg.get("text", "").strip(),
                    "manual_text": "",  # Fill in manually
                }
            )

    if not all_segments:
        print(f"No segments found in {transcript_dir}")
        return []

    sample = random.sample(all_segments, min(n, len(all_segments)))

    with open(output_path, "w", encoding="utf-8") as f:
        for s in sample:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Sampled {len(sample)} segments from {len(all_segments)} total")
    print(f"Saved to {output_path}")
    print("\nNext steps:")
    print("  1. Listen to each segment's audio")
    print("  2. Fill in the 'manual_text' field with your transcription")
    print(f"  3. Run: python assess_quality.py evaluate --input {output_path}")

    return sample


def compute_baseline_wer(assessment_path):
    """After manual transcription, compute WER baseline."""
    import jiwer

    auto_texts, manual_texts = [], []
    skipped = 0

    with open(assessment_path, encoding="utf-8") as f:
        for line in f:
            seg = json.loads(line)
            if seg.get("manual_text", "").strip():
                auto_texts.append(seg["auto_text"])
                manual_texts.append(seg["manual_text"])
            else:
                skipped += 1

    if not manual_texts:
        print("No segments with manual_text found. Did you fill them in?")
        return None

    # Overall WER
    overall_wer = jiwer.wer(manual_texts, auto_texts)
    overall_cer = jiwer.cer(manual_texts, auto_texts)

    print(f"Baseline Assessment ({len(auto_texts)} segments, {skipped} skipped)")
    print("=" * 50)
    print(f"  WER: {overall_wer:.1%}")
    print(f"  CER: {overall_cer:.1%}")
    print()
    print("  Strategy recommendation:")
    if overall_wer < 0.10:
        print("  < 10% WER: Use directly with confidence-based filtering")
    elif overall_wer < 0.20:
        print("  10-20% WER: Filter worst segments by avg_logprob")
    elif overall_wer < 0.30:
        print("  20-30% WER: Weakly supervised pretrain + fine-tune on clean subset")
    else:
        print("  > 30% WER: Re-transcribe everything with Whisper large-v3")

    # Per-segment breakdown
    print("\nPer-segment WER distribution:")
    per_seg_wer = []
    for auto, manual in zip(auto_texts, manual_texts):
        seg_wer = jiwer.wer(manual, auto)
        per_seg_wer.append(seg_wer)

    import numpy as np

    arr = np.array(per_seg_wer)
    print(f"  Mean:   {arr.mean():.1%}")
    print(f"  Median: {np.median(arr):.1%}")
    print(f"  P90:    {np.percentile(arr, 90):.1%}")
    print(f"  Perfect (0%): {(arr == 0).sum()} segments")
    print(f"  Good (<10%):  {(arr < 0.10).sum()} segments")
    print(f"  Bad (>30%):   {(arr > 0.30).sum()} segments")

    return overall_wer


def main():
    parser = argparse.ArgumentParser(description="Baseline transcript quality assessment")
    subparsers = parser.add_subparsers(dest="command")

    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Generate sample for manual review")
    sample_parser.add_argument(
        "--input", "-i", default="stark_data/transcripts", help="Directory with JSON transcripts"
    )
    sample_parser.add_argument("--n", type=int, default=100, help="Number of segments to sample")
    sample_parser.add_argument("--output", "-o", default="assessment_sample.jsonl", help="Output JSONL path")
    sample_parser.add_argument("--seed", type=int, default=42)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Compute WER after manual transcription")
    eval_parser.add_argument(
        "--input", "-i", default="assessment_sample.jsonl", help="JSONL with manual_text filled in"
    )

    args = parser.parse_args()

    if args.command == "sample":
        sample_for_assessment(args.input, args.n, args.output, args.seed)
    elif args.command == "evaluate":
        compute_baseline_wer(args.input)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
