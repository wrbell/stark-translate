#!/usr/bin/env python3
"""
assess_quality.py — Baseline Transcript Quality Assessment

Before fine-tuning, establish a WER baseline on 50-100 randomly sampled segments.
Samples segments for manual review and computes WER against human corrections.

Usage:
    # Random sample for fully manual review
    python assess_quality.py sample --input stark_data/transcripts --n 100

    # Smart sample: stratify by confidence, auto-accept high-confidence segments
    python assess_quality.py smart-sample --input stark_data/transcripts --n 100

    # Interactive review: play audio + confirm/correct (faster than blind transcription)
    python assess_quality.py review --input assessment_smart_sample.jsonl

    # Cross-model check: automated agreement WER between two Whisper models
    python assess_quality.py cross-check --input stark_data/transcripts --n 50

    # Compute WER after manual transcription
    python assess_quality.py evaluate --input assessment_sample.jsonl
"""

import argparse
import json
import os
import random
from pathlib import Path

# Confidence thresholds aligned with engines/CLAUDE.md
CONFIDENCE_THRESHOLDS = {
    "avg_logprob": {"good": -0.3, "flag": -0.5, "reject": -1.0},
    "no_speech_prob": {"good": 0.1, "flag": 0.3, "reject": 0.6},
    "compression_ratio": {"good": 1.8, "flag": 2.0, "reject": 2.4},
}


def classify_segment(seg):
    """Classify a segment by confidence into 'high', 'medium', 'low', or 'reject'.

    Uses avg_logprob, no_speech_prob, and compression_ratio thresholds.
    Returns 'medium' if all three confidence fields are missing.
    """
    avg_logprob = seg.get("avg_logprob")
    no_speech_prob = seg.get("no_speech_prob")
    compression_ratio = seg.get("compression_ratio")

    # If all three fields missing (e.g. transformers backend), default to medium
    if avg_logprob is None and no_speech_prob is None and compression_ratio is None:
        return "medium"

    worst = "high"

    for metric_name, value in [
        ("avg_logprob", avg_logprob),
        ("no_speech_prob", no_speech_prob),
        ("compression_ratio", compression_ratio),
    ]:
        if value is None:
            continue
        thresholds = CONFIDENCE_THRESHOLDS[metric_name]

        # For avg_logprob: lower is worse (more negative = less confident)
        if metric_name == "avg_logprob":
            if value < thresholds["reject"]:
                return "reject"
            if value < thresholds["flag"]:
                worst = "low"
            elif value < thresholds["good"] and worst not in ("low",):
                worst = "medium"
        else:
            # For no_speech_prob and compression_ratio: higher is worse
            if value > thresholds["reject"]:
                return "reject"
            if value > thresholds["flag"]:
                worst = "low"
            elif value > thresholds["good"] and worst not in ("low",):
                worst = "medium"

    return worst


def _extract_segments(transcript_dir):
    """Extract all segments from transcript JSONs in a directory.

    Returns list of dicts with audio_file, start, end, auto_text, and
    any confidence fields (avg_logprob, no_speech_prob, compression_ratio).
    """
    all_segments = []

    for jf in sorted(Path(transcript_dir).glob("*.json")):
        if jf.name.startswith("_"):
            continue
        with open(jf) as f:
            data = json.load(f)

        audio_path = data.get("audio_path", str(jf))
        for seg in data.get("segments", []):
            ts = seg.get("timestamp", [None, None])
            entry = {
                "audio_file": audio_path,
                "start": ts[0] if isinstance(ts, list) else seg.get("start"),
                "end": ts[1] if isinstance(ts, list) else seg.get("end"),
                "auto_text": seg.get("text", "").strip(),
            }
            # Carry forward confidence fields if present
            for field in ("avg_logprob", "no_speech_prob", "compression_ratio"):
                if field in seg:
                    entry[field] = seg[field]
            all_segments.append(entry)

    return all_segments


def sample_for_assessment(transcript_dir, n=100, output_path="assessment_sample.jsonl", seed=42):
    """Randomly sample segments for manual review."""
    random.seed(seed)
    all_segments = _extract_segments(transcript_dir)

    if not all_segments:
        print(f"No segments found in {transcript_dir}")
        return []

    sample = random.sample(all_segments, min(n, len(all_segments)))

    # Add manual_text field for manual filling
    for s in sample:
        s["manual_text"] = ""

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


def smart_sample_for_assessment(
    transcript_dir, n=100, output_path="assessment_smart_sample.jsonl", seed=42, spot_check_ratio=0.3
):
    """Stratified sample by confidence: auto-accept high, review low/medium.

    Algorithm:
    1. Classify all segments into high/medium/low/reject buckets
    2. Discard reject bucket
    3. Include ALL low segments (mandatory review)
    4. Random-sample spot_check_ratio of medium segments
    5. Fill remaining budget from high bucket (auto-accepted with manual_text pre-filled)
    """
    random.seed(seed)
    all_segments = _extract_segments(transcript_dir)

    if not all_segments:
        print(f"No segments found in {transcript_dir}")
        return []

    # Classify into buckets
    buckets = {"high": [], "medium": [], "low": [], "reject": []}
    for seg in all_segments:
        bucket = classify_segment(seg)
        buckets[bucket].append(seg)

    # Print distribution
    print("Confidence distribution:")
    for bucket_name in ("high", "medium", "low", "reject"):
        print(f"  {bucket_name}: {len(buckets[bucket_name])}")
    print()

    # Build sample
    sample = []

    # All low segments (mandatory review)
    low_segs = list(buckets["low"])
    random.shuffle(low_segs)
    for seg in low_segs:
        if len(sample) >= n:
            break
        seg["confidence_bucket"] = "low"
        seg["reviewed"] = False
        seg["manual_text"] = ""
        sample.append(seg)

    # Spot-check portion of medium segments
    medium_segs = list(buckets["medium"])
    random.shuffle(medium_segs)
    n_medium = max(1, int(len(medium_segs) * spot_check_ratio)) if medium_segs else 0
    for seg in medium_segs[:n_medium]:
        if len(sample) >= n:
            break
        seg["confidence_bucket"] = "medium"
        seg["reviewed"] = False
        seg["manual_text"] = ""
        sample.append(seg)

    # Fill remaining from high bucket (auto-accepted)
    remaining = n - len(sample)
    if remaining > 0:
        high_segs = list(buckets["high"])
        random.shuffle(high_segs)
        for seg in high_segs[:remaining]:
            seg["confidence_bucket"] = "high"
            seg["reviewed"] = True  # Auto-accepted
            seg["manual_text"] = seg["auto_text"]  # Pre-filled
            sample.append(seg)

    with open(output_path, "w", encoding="utf-8") as f:
        for s in sample:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    n_review = sum(1 for s in sample if not s.get("reviewed", False))
    n_auto = sum(1 for s in sample if s.get("reviewed", False))
    print(f"Smart sample: {len(sample)} segments ({n_review} to review, {n_auto} auto-accepted)")
    print(f"Saved to {output_path}")
    print(f"\nNext: python assess_quality.py review --input {output_path}")

    return sample


def play_audio_clip(audio_file, start, end):
    """Play an audio clip from start to end seconds.

    Uses sounddevice + soundfile. Falls back to printing path on error.
    """
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        print(f"  [sounddevice/soundfile not installed — open manually: {audio_file}]")
        print(f"  [segment: {start:.1f}s - {end:.1f}s]")
        return

    try:
        sr = 16000
        start_frame = int(start * sr) if start else 0
        stop_frame = int(end * sr) if end else -1
        data, actual_sr = sf.read(audio_file, start=start_frame, stop=stop_frame)
        sd.play(data, actual_sr)
        sd.wait()
    except Exception as e:
        print(f"  [Playback error: {e}]")
        print(f"  [Open manually: {audio_file} @ {start:.1f}s - {end:.1f}s]")


def interactive_review(jsonl_path):
    """Interactive terminal review of unreviewed segments.

    For each unreviewed segment:
    - Shows confidence info + auto transcript
    - Plays audio
    - User: Enter=accept, type=correct, /r=replay, /s=skip, /q=quit
    Auto-saves after each entry.
    """
    # Load all entries
    entries = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    unreviewed = [(i, e) for i, e in enumerate(entries) if not e.get("reviewed", False)]

    if not unreviewed:
        print("All segments already reviewed!")
        return entries

    print(f"{len(unreviewed)} segments to review (of {len(entries)} total)")
    print("Controls: Enter=accept, type text=correct, /r=replay, /s=skip, /q=save & quit\n")

    reviewed_count = 0
    for pos, (idx, entry) in enumerate(unreviewed):
        bucket = entry.get("confidence_bucket", "?")
        logprob = entry.get("avg_logprob")
        logprob_str = f"avg_logprob={logprob:.2f}" if logprob is not None else ""
        auto_text = entry.get("auto_text", "")

        print(f"[{pos + 1}/{len(unreviewed)}] {bucket} | {logprob_str}")
        print(f'  "{auto_text}"')

        # Play audio
        audio_file = entry.get("audio_file", "")
        start = entry.get("start")
        end = entry.get("end")
        if audio_file and start is not None and end is not None:
            play_audio_clip(audio_file, start, end)

        while True:
            user_input = input("  > ").strip()

            if user_input == "/r":
                if audio_file and start is not None and end is not None:
                    play_audio_clip(audio_file, start, end)
                continue
            elif user_input == "/s":
                print("  [skipped]")
                break
            elif user_input == "/q":
                print("Saving and quitting...")
                _atomic_write_jsonl(jsonl_path, entries)
                print(f"Reviewed {reviewed_count} segments this session.")
                return entries
            elif user_input == "":
                # Accept as correct
                entries[idx]["manual_text"] = auto_text
                entries[idx]["reviewed"] = True
                reviewed_count += 1
                print("  [accepted]")
                break
            else:
                # User typed a correction
                entries[idx]["manual_text"] = user_input
                entries[idx]["reviewed"] = True
                reviewed_count += 1
                print("  [corrected]")
                break

        # Auto-save after each entry
        _atomic_write_jsonl(jsonl_path, entries)

    print(f"\nDone! Reviewed {reviewed_count} segments.")
    return entries


def _atomic_write_jsonl(path, entries):
    """Write JSONL atomically via tmp file + os.replace."""
    tmp_path = str(path) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(tmp_path, str(path))


def cross_check(
    transcript_dir, n=50, model_name="openai/whisper-large-v3", output_path="cross_check_results.jsonl", seed=42
):
    """Cross-model agreement check between original transcripts and a second Whisper model.

    Fully automated — no human needed. Computes per-segment WER between
    the original auto_text and a re-transcription from a different model.
    """
    import jiwer
    import numpy as np

    random.seed(seed)
    all_segments = _extract_segments(transcript_dir)

    if not all_segments:
        print(f"No segments found in {transcript_dir}")
        return []

    # Filter to segments with valid audio files and timestamps
    valid = [s for s in all_segments if s.get("audio_file") and s.get("start") is not None and s.get("end") is not None]
    sample = random.sample(valid, min(n, len(valid)))

    print(f"Cross-checking {len(sample)} segments with {model_name}...")

    # Load cross-check model via faster-whisper
    try:
        import torch
        from faster_whisper import WhisperModel
    except ImportError as e:
        print(f"Error: {e}. Install faster-whisper and torch.")
        return []

    fw_model_map = {
        "distil-whisper/distil-large-v3.5": "distil-large-v3.5",
        "distil-whisper/distil-large-v3": "distil-large-v3",
        "openai/whisper-large-v3": "large-v3",
        "openai/whisper-large-v3-turbo": "large-v3-turbo",
        "openai/whisper-large-v2": "large-v2",
        "openai/whisper-medium": "medium",
    }
    fw_model = fw_model_map.get(model_name, model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"

    print(f"Loading {fw_model} on {device}...")
    model = WhisperModel(fw_model, device=device, compute_type=compute_type)

    results = []
    for i, seg in enumerate(sample):
        audio_file = seg["audio_file"]
        start = seg["start"]
        end = seg["end"]

        try:
            import soundfile as sf

            sr = 16000
            start_frame = int(start * sr)
            stop_frame = int(end * sr)
            audio_data, _ = sf.read(audio_file, start=start_frame, stop=stop_frame, dtype="float32")

            segments_iter, _ = model.transcribe(audio_data, language="en")
            cross_text = " ".join(s.text.strip() for s in segments_iter).strip()
        except Exception as e:
            print(f"  [{i + 1}/{len(sample)}] Error: {e}")
            cross_text = ""

        original = seg["auto_text"]
        if original and cross_text:
            seg_wer = jiwer.wer(original, cross_text)
        elif not original and not cross_text:
            seg_wer = 0.0
        else:
            seg_wer = 1.0

        result = {
            "audio_file": audio_file,
            "start": start,
            "end": end,
            "original_text": original,
            "cross_text": cross_text,
            "agreement_wer": round(seg_wer, 4),
        }
        results.append(result)

        if (i + 1) % 10 == 0 or i == len(sample) - 1:
            print(f"  [{i + 1}/{len(sample)}] done")

    # Write results
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    wers = np.array([r["agreement_wer"] for r in results])
    print(f"\nCross-check summary ({len(results)} segments):")
    print(f"  Mean agreement WER:   {wers.mean():.1%}")
    print(f"  Median agreement WER: {np.median(wers):.1%}")
    print(f"  P90 agreement WER:    {np.percentile(wers, 90):.1%}")
    print(f"  Perfect matches:      {(wers == 0).sum()}")
    print(f"  High disagreement (>30%): {(wers > 0.30).sum()}")
    print(f"Saved to {output_path}")

    return results


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

    # Sample command (original random sampling)
    sample_parser = subparsers.add_parser("sample", help="Generate random sample for manual review")
    sample_parser.add_argument(
        "--input", "-i", default="stark_data/transcripts", help="Directory with JSON transcripts"
    )
    sample_parser.add_argument("--n", type=int, default=100, help="Number of segments to sample")
    sample_parser.add_argument("--output", "-o", default="assessment_sample.jsonl", help="Output JSONL path")
    sample_parser.add_argument("--seed", type=int, default=42)

    # Smart-sample command (confidence-stratified)
    smart_parser = subparsers.add_parser(
        "smart-sample", help="Confidence-stratified sample (auto-accept high, review low/medium)"
    )
    smart_parser.add_argument("--input", "-i", default="stark_data/transcripts", help="Directory with JSON transcripts")
    smart_parser.add_argument("--n", type=int, default=100, help="Total segments to sample")
    smart_parser.add_argument("--output", "-o", default="assessment_smart_sample.jsonl", help="Output JSONL path")
    smart_parser.add_argument("--seed", type=int, default=42)
    smart_parser.add_argument("--spot-check", type=float, default=0.3, help="Fraction of medium segments to spot-check")

    # Review command (interactive terminal)
    review_parser = subparsers.add_parser("review", help="Interactive audio review of unreviewed segments")
    review_parser.add_argument("--input", "-i", default="assessment_smart_sample.jsonl", help="JSONL from smart-sample")

    # Cross-check command (automated cross-model agreement)
    cross_parser = subparsers.add_parser("cross-check", help="Automated cross-model agreement WER (no human needed)")
    cross_parser.add_argument("--input", "-i", default="stark_data/transcripts", help="Directory with JSON transcripts")
    cross_parser.add_argument("--n", type=int, default=50, help="Number of segments to check")
    cross_parser.add_argument("--model", default="openai/whisper-large-v3", help="Cross-check Whisper model")
    cross_parser.add_argument("--output", "-o", default="cross_check_results.jsonl", help="Output JSONL path")
    cross_parser.add_argument("--seed", type=int, default=42)

    # Evaluate command (existing)
    eval_parser = subparsers.add_parser("evaluate", help="Compute WER after manual transcription")
    eval_parser.add_argument(
        "--input", "-i", default="assessment_sample.jsonl", help="JSONL with manual_text filled in"
    )

    args = parser.parse_args()

    if args.command == "sample":
        sample_for_assessment(args.input, args.n, args.output, args.seed)
    elif args.command == "smart-sample":
        smart_sample_for_assessment(args.input, args.n, args.output, args.seed, args.spot_check)
    elif args.command == "review":
        interactive_review(args.input)
    elif args.command == "cross-check":
        cross_check(args.input, args.n, args.model, args.output, args.seed)
    elif args.command == "evaluate":
        compute_baseline_wer(args.input)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
