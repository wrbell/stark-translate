#!/usr/bin/env python3
"""
transcribe_church.py — Whisper Transcription of Cleaned Church Audio

Generates clean training labels using Distil-Whisper large-v3 on CUDA.
Use this INSTEAD of YouTube auto-captions for training labels (~7% WER vs ~15% WER).

Supports two backends:
  - transformers pipeline (default, simpler)
  - faster-whisper (--backend faster-whisper) for confidence scores + speed

Usage:
    python transcribe_church.py
    python transcribe_church.py --input stark_data/cleaned/chunks --output stark_data/transcripts
    python transcribe_church.py --model openai/whisper-large-v3  # Max quality
    python transcribe_church.py --backend faster-whisper          # With confidence scores
    python transcribe_church.py --resume                          # Resume interrupted run
"""

import argparse
import glob
import json
import logging
import os
import time

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def transcribe_with_transformers(data_dir, output_dir, model_name, batch_size,
                                  resume=False):
    """Transcribe all cleaned audio with HuggingFace Whisper pipeline on CUDA."""
    from transformers import pipeline

    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info(f"Loading {model_name} on {device}...")
    stt_pipe = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
        torch_dtype=dtype,
    )

    # Search top-level and accent subdirectories
    audio_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.wav"),
                                   recursive=True))
    if not audio_files:
        logger.warning(f"No WAV files found in {data_dir} (searched recursively)")
        return

    # Resume support: skip already-transcribed files
    if resume:
        existing = set()
        for f in glob.glob(os.path.join(output_dir, "*.json")):
            if not os.path.basename(f).startswith("_"):
                existing.add(os.path.basename(f).replace(".json", ".wav"))
        # Also check accent subdirectories in output
        for f in glob.glob(os.path.join(output_dir, "*", "*.json")):
            if not os.path.basename(f).startswith("_"):
                existing.add(os.path.basename(f).replace(".json", ".wav"))
        before = len(audio_files)
        audio_files = [f for f in audio_files
                       if os.path.basename(f) not in existing]
        if before != len(audio_files):
            logger.info(f"Resume: skipping {before - len(audio_files)} "
                        f"already-transcribed files")

    if not audio_files:
        logger.info("All files already transcribed.")
        return

    logger.info(f"Transcribing {len(audio_files)} files...")
    all_transcripts = []
    start_time = time.time()

    for i, audio_file in enumerate(audio_files):
        try:
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

        except Exception as e:
            logger.error(f"Failed to transcribe {os.path.basename(audio_file)}: {e}")
            continue

        if (i + 1) % 10 == 0 or i == len(audio_files) - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60
            logger.info(f"  [{i+1}/{len(audio_files)}] {rate:.0f} files/min | "
                        f"Last: {len(result['text'])} chars, "
                        f"{len(result.get('chunks', []))} segments")

    elapsed = time.time() - start_time
    total_chars = sum(len(t["text"]) for t in all_transcripts)
    total_segs = sum(len(t["segments"]) for t in all_transcripts)
    logger.info(f"Done in {elapsed:.0f}s: {len(all_transcripts)} files, "
                f"{total_segs} segments, {total_chars} chars")

    # Save summary
    _save_summary(output_dir, model_name, "cuda" if torch.cuda.is_available() else "cpu",
                  all_transcripts, elapsed, backend="transformers")


def transcribe_with_faster_whisper(data_dir, output_dir, model_name, batch_size,
                                    resume=False):
    """Transcribe using faster-whisper for confidence scores + speed.

    faster-whisper exposes avg_logprob, no_speech_prob, and compression_ratio
    per segment, plus word-level confidence when word_timestamps=True.
    These are critical for the confidence-based flagging layer (Layer 3).
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.error("faster-whisper not installed. Run: pip install faster-whisper")
        logger.error("Falling back to transformers backend.")
        return transcribe_with_transformers(data_dir, output_dir, model_name,
                                            batch_size, resume)

    os.makedirs(output_dir, exist_ok=True)

    # Map HuggingFace model names to faster-whisper equivalents
    fw_model_map = {
        "distil-whisper/distil-large-v3": "distil-large-v3",
        "openai/whisper-large-v3": "large-v3",
        "openai/whisper-large-v2": "large-v2",
        "openai/whisper-medium": "medium",
    }
    fw_model = fw_model_map.get(model_name, model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"

    logger.info(f"Loading faster-whisper model '{fw_model}' on {device}...")
    model = WhisperModel(fw_model, device=device, compute_type=compute_type)

    # Search top-level and accent subdirectories
    audio_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.wav"),
                                   recursive=True))
    if not audio_files:
        logger.warning(f"No WAV files found in {data_dir} (searched recursively)")
        return

    # Resume support
    if resume:
        existing = set()
        for f in glob.glob(os.path.join(output_dir, "*.json")):
            if not os.path.basename(f).startswith("_"):
                existing.add(os.path.basename(f).replace(".json", ".wav"))
        for f in glob.glob(os.path.join(output_dir, "*", "*.json")):
            if not os.path.basename(f).startswith("_"):
                existing.add(os.path.basename(f).replace(".json", ".wav"))
        before = len(audio_files)
        audio_files = [f for f in audio_files
                       if os.path.basename(f) not in existing]
        if before != len(audio_files):
            logger.info(f"Resume: skipping {before - len(audio_files)} "
                        f"already-transcribed files")

    if not audio_files:
        logger.info("All files already transcribed.")
        return

    logger.info(f"Transcribing {len(audio_files)} files with confidence scores...")
    all_transcripts = []
    start_time = time.time()

    for i, audio_file in enumerate(audio_files):
        try:
            segments_iter, info = model.transcribe(
                audio_file,
                language="en",
                word_timestamps=True,
                vad_filter=True,
            )

            segments = []
            full_text_parts = []
            for seg in segments_iter:
                seg_data = {
                    "text": seg.text.strip(),
                    "start": seg.start,
                    "end": seg.end,
                    "avg_logprob": seg.avg_logprob,
                    "no_speech_prob": seg.no_speech_prob,
                    "compression_ratio": seg.compression_ratio,
                }
                # Word-level confidence
                if seg.words:
                    seg_data["words"] = [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability,
                        }
                        for w in seg.words
                    ]
                segments.append(seg_data)
                full_text_parts.append(seg.text.strip())

            transcript = {
                "audio_path": audio_file,
                "text": " ".join(full_text_parts),
                "segments": segments,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
            }

            out_path = os.path.join(
                output_dir,
                os.path.basename(audio_file).replace(".wav", ".json"),
            )
            with open(out_path, "w") as f:
                json.dump(transcript, f, indent=2, ensure_ascii=False)

            all_transcripts.append(transcript)

        except Exception as e:
            logger.error(f"Failed to transcribe {os.path.basename(audio_file)}: {e}")
            continue

        if (i + 1) % 10 == 0 or i == len(audio_files) - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60
            logger.info(f"  [{i+1}/{len(audio_files)}] {rate:.0f} files/min | "
                        f"Last: {len(full_text_parts)} segments")

    elapsed = time.time() - start_time
    total_chars = sum(len(t["text"]) for t in all_transcripts)
    total_segs = sum(len(t["segments"]) for t in all_transcripts)
    logger.info(f"Done in {elapsed:.0f}s: {len(all_transcripts)} files, "
                f"{total_segs} segments, {total_chars} chars")

    _save_summary(output_dir, model_name,
                  "cuda" if torch.cuda.is_available() else "cpu",
                  all_transcripts, elapsed, backend="faster-whisper")


def _save_summary(output_dir, model_name, device, all_transcripts, elapsed,
                  backend="transformers"):
    """Save transcription run summary."""
    total_chars = sum(len(t["text"]) for t in all_transcripts)
    total_segs = sum(len(t["segments"]) for t in all_transcripts)

    summary_path = os.path.join(output_dir, "_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "backend": backend,
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
    parser.add_argument("--backend", choices=["transformers", "faster-whisper"],
                        default="transformers",
                        help="Transcription backend (faster-whisper adds confidence scores)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip files that already have transcripts in the output dir")
    args = parser.parse_args()

    if args.backend == "faster-whisper":
        transcribe_with_faster_whisper(
            args.input, args.output, args.model, args.batch_size, args.resume)
    else:
        transcribe_with_transformers(
            args.input, args.output, args.model, args.batch_size, args.resume)


if __name__ == "__main__":
    main()
