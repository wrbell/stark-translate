#!/usr/bin/env python3
"""
transcribe_church.py — Whisper Transcription of Cleaned Church Audio

Generates clean training labels using Distil-Whisper large-v3 on CUDA.
Use this INSTEAD of YouTube auto-captions for training labels (~7% WER vs ~15% WER).

Usage:
    python transcribe_church.py
    python transcribe_church.py --input stark_data/cleaned/chunks --output stark_data/transcripts
    python transcribe_church.py --model openai/whisper-large-v3  # Max quality
"""

import argparse
import glob
import json
import os
import time

import torch
from transformers import pipeline


def transcribe_all(data_dir="stark_data/cleaned/chunks",
                   output_dir="stark_data/transcripts",
                   model_name="distil-whisper/distil-large-v3",
                   batch_size=8):
    """Transcribe all cleaned audio with Whisper on CUDA."""
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {model_name} on {device}...")
    stt_pipe = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
        torch_dtype=dtype,
    )

    audio_files = sorted(glob.glob(os.path.join(data_dir, "*.wav")))
    if not audio_files:
        print(f"No WAV files found in {data_dir}")
        return

    print(f"Transcribing {len(audio_files)} files...")
    all_transcripts = []
    start_time = time.time()

    for i, audio_file in enumerate(audio_files):
        result = stt_pipe(
            audio_file,
            return_timestamps=True,
            chunk_length_s=30,
            stride_length_s=5,
            generate_kwargs={"language": "en"},
        )

        transcript = {
            "audio_path": audio_file,
            "text": result["text"],
            "segments": result.get("chunks", []),
        }

        out_path = os.path.join(
            output_dir,
            os.path.basename(audio_file).replace(".wav", ".json"),
        )
        with open(out_path, "w") as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)

        all_transcripts.append(transcript)

        if (i + 1) % 10 == 0 or i == len(audio_files) - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60
            print(f"  [{i+1}/{len(audio_files)}] {rate:.0f} files/min | "
                  f"Last: {len(result['text'])} chars, "
                  f"{len(result.get('chunks', []))} segments")

    elapsed = time.time() - start_time
    total_chars = sum(len(t["text"]) for t in all_transcripts)
    total_segs = sum(len(t["segments"]) for t in all_transcripts)
    print(f"\nDone in {elapsed:.0f}s: {len(all_transcripts)} files, "
          f"{total_segs} segments, {total_chars} chars")

    # Save summary
    summary_path = os.path.join(output_dir, "_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "model": model_name,
            "device": device,
            "files": len(all_transcripts),
            "segments": total_segs,
            "chars": total_chars,
            "elapsed_s": round(elapsed, 1),
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe cleaned church audio with Whisper"
    )
    parser.add_argument("--input", "-i", default="stark_data/cleaned/chunks",
                        help="Directory containing cleaned WAV files")
    parser.add_argument("--output", "-o", default="stark_data/transcripts",
                        help="Output directory for JSON transcripts")
    parser.add_argument("--model", "-m", default="distil-whisper/distil-large-v3",
                        help="Whisper model to use")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    transcribe_all(args.input, args.output, args.model, args.batch_size)


if __name__ == "__main__":
    main()
