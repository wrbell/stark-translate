#!/usr/bin/env python3
"""
preprocess_audio.py — 10-Step Audio Cleaning Pipeline for Church Sermon Data

Transforms raw YouTube church audio into clean training data for Whisper fine-tuning.
Runs on Windows/WSL desktop (A2000 Ada 16GB VRAM).

Usage:
    python preprocess_audio.py --input stark_data/raw --output stark_data/cleaned
    python preprocess_audio.py --input stark_data/raw/sermon1.wav --output stark_data/cleaned
    python preprocess_audio.py --download --urls urls.txt --output stark_data/raw
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Step 1: Download (optional — normally done separately via yt-dlp)
# ---------------------------------------------------------------------------

def download_audio(urls_file, output_dir="stark_data/raw"):
    """Download audio from YouTube URLs using yt-dlp."""
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run([
        "yt-dlp", "-f", "bestaudio",
        "--extract-audio", "--audio-format", "wav",
        "--batch-file", urls_file,
        "-P", output_dir,
    ], check=True)
    # Also grab YouTube auto-captions for baseline comparison
    subprocess.run([
        "yt-dlp", "--write-auto-subs", "--sub-lang", "en",
        "--skip-download",
        "--batch-file", urls_file,
        "-P", output_dir,
    ], check=True)
    print(f"Downloaded audio to {output_dir}")


# ---------------------------------------------------------------------------
# Step 2: Format Conversion
# ---------------------------------------------------------------------------

def convert_to_whisper_format(input_path, output_path):
    """Convert to 16kHz mono WAV (Whisper's required format)."""
    subprocess.run([
        "ffmpeg", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-f", "wav",
        output_path, "-y",
    ], check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Step 3: Initial Quality Gate
# ---------------------------------------------------------------------------

def initial_quality_gate(audio_path, min_snr_db=10.0):
    """Reject files with very low SNR or clipping."""
    audio, sr = sf.read(audio_path)

    # Clipping detection
    clip_ratio = np.mean(np.abs(audio) > 0.99)
    if clip_ratio > 0.01:
        return False, f"Clipping: {clip_ratio:.2%} of samples"

    # Simple SNR estimate (energy in top 90th percentile vs bottom 10th)
    frame_size = int(sr * 0.1)  # 100ms frames
    frame_energy = np.array([
        np.mean(audio[i:i + frame_size] ** 2)
        for i in range(0, len(audio) - frame_size, frame_size)
    ])
    nonzero = frame_energy[frame_energy > 0]
    if len(nonzero) < 2:
        return False, "Insufficient audio energy"

    speech_energy = np.percentile(nonzero, 90)
    noise_energy = np.percentile(nonzero, 10)
    snr = 10 * np.log10(speech_energy / max(noise_energy, 1e-10))

    if snr < min_snr_db:
        return False, f"Low SNR: {snr:.1f} dB"

    return True, f"OK (SNR: {snr:.1f} dB, clip: {clip_ratio:.4%})"


# ---------------------------------------------------------------------------
# Step 4: Segment Classification (speech/music/noise)
# ---------------------------------------------------------------------------

def classify_segments(audio_path):
    """Classify audio into speech/music/noise regions using inaSpeechSegmenter."""
    from inaSpeechSegmenter import Segmenter
    seg = Segmenter()
    segments = seg(audio_path)
    # Returns list of (label, start_time, end_time)
    # Labels: 'speech', 'music', 'noise', 'noEnergy'
    return segments


# ---------------------------------------------------------------------------
# Step 5: Source Separation (vocals from music)
# ---------------------------------------------------------------------------

def separate_vocals(audio_path, output_dir):
    """Use demucs to isolate vocals from music segments."""
    subprocess.run([
        "python", "-m", "demucs",
        "--two-stems", "vocals",
        "-n", "htdemucs",
        "-o", output_dir,
        audio_path,
    ], check=True)
    # Output: output_dir/htdemucs/filename/vocals.wav
    stem = Path(audio_path).stem
    vocals_path = os.path.join(output_dir, "htdemucs", stem, "vocals.wav")
    return vocals_path


# ---------------------------------------------------------------------------
# Step 6: Bandpass Filter
# ---------------------------------------------------------------------------

def bandpass_filter(input_path, output_path):
    """Remove sub-bass rumble and high-freq hiss via ffmpeg."""
    subprocess.run([
        "ffmpeg", "-i", input_path,
        "-af", "highpass=f=80,lowpass=f=8000",
        output_path, "-y",
    ], check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Step 7: Denoise
# ---------------------------------------------------------------------------

def denoise_audio(audio, sr=16000):
    """Conservative spectral gating — don't over-clean.

    Whisper was trained on noisy audio. Moderate noise improves robustness.
    """
    import noisereduce as nr
    return nr.reduce_noise(
        y=audio, sr=sr,
        prop_decrease=0.7,      # Conservative: 0.6-0.8
        n_fft=512,
        stationary=False,       # Non-stationary for church reverb
    )


# ---------------------------------------------------------------------------
# Step 8: Loudness Normalization
# ---------------------------------------------------------------------------

def normalize_loudness(audio, sr=16000, target_lufs=-16.0):
    """EBU R128 loudness normalization with true peak limiting."""
    import pyloudnorm as pyln
    meter = pyln.Meter(sr)
    current_lufs = meter.integrated_loudness(audio)
    if np.isinf(current_lufs):
        return audio
    normalized = pyln.normalize.loudness(audio, current_lufs, target_lufs)
    # True peak limiting at -1 dBTP
    peak = np.max(np.abs(normalized))
    if peak > 0.891:
        normalized = normalized * (0.891 / peak)
    return normalized


# ---------------------------------------------------------------------------
# Step 9: VAD Chunking
# ---------------------------------------------------------------------------

def vad_chunk(audio_path, min_dur=1.0, max_dur=30.0, padding=0.1):
    """Silero VAD-based chunking with WhisperX-style merge."""
    import torch
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
    (get_speech_timestamps, _, read_audio, _, _) = utils

    wav = read_audio(audio_path, sampling_rate=16000)
    timestamps = get_speech_timestamps(
        wav, model, sampling_rate=16000,
        min_speech_duration_ms=500,
        min_silence_duration_ms=300,
    )

    # Merge short segments, split long ones
    chunks = []
    pending_start = None
    pending_end = None

    for ts in timestamps:
        start = max(0, ts['start'] / 16000 - padding)
        end = ts['end'] / 16000 + padding

        if pending_start is None:
            pending_start = start
            pending_end = end
            continue

        # Try merging with pending chunk
        if start - pending_end < 0.5 and (start - pending_start + (end - start)) <= max_dur:
            pending_end = end
        else:
            # Emit pending chunk
            dur = pending_end - pending_start
            if dur >= min_dur:
                if dur > max_dur:
                    # Split at midpoint
                    mid = (pending_start + pending_end) / 2
                    chunks.append((pending_start, mid))
                    chunks.append((mid, pending_end))
                else:
                    chunks.append((pending_start, pending_end))
            pending_start = start
            pending_end = end

    # Emit last pending
    if pending_start is not None:
        dur = pending_end - pending_start
        if dur >= min_dur:
            if dur > max_dur:
                mid = (pending_start + pending_end) / 2
                chunks.append((pending_start, mid))
                chunks.append((mid, pending_end))
            else:
                chunks.append((pending_start, pending_end))

    return chunks


# ---------------------------------------------------------------------------
# Step 10: Speaker Diarization (optional)
# ---------------------------------------------------------------------------

def diarize_speakers(audio_path, hf_token=None):
    """Identify primary speaker via pyannote (requires HuggingFace token)."""
    from pyannote.audio import Pipeline
    token = hf_token or os.environ.get("HF_TOKEN")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )
    diarization = pipeline(audio_path)
    return diarization


# ---------------------------------------------------------------------------
# Step 11: Final Quality Gate
# ---------------------------------------------------------------------------

def final_quality_gate(chunk_audio, sr=16000, min_snr=15.0):
    """Reject chunks that don't meet training quality standards."""
    duration = len(chunk_audio) / sr
    if duration < 1.0 or duration > 30.0:
        return False, "duration_out_of_range"

    # Silence ratio
    energy = chunk_audio ** 2
    silence_ratio = np.mean(energy < 1e-6)
    if silence_ratio > 0.5:
        return False, "too_much_silence"

    # SNR check on chunk
    frame_size = int(sr * 0.05)  # 50ms frames
    frame_energy = np.array([
        np.mean(chunk_audio[i:i + frame_size] ** 2)
        for i in range(0, len(chunk_audio) - frame_size, frame_size)
    ])
    nonzero = frame_energy[frame_energy > 0]
    if len(nonzero) < 2:
        return False, "insufficient_audio"

    speech_e = np.percentile(nonzero, 80)
    noise_e = np.percentile(nonzero, 10)
    snr = 10 * np.log10(speech_e / max(noise_e, 1e-10))
    if snr < min_snr:
        return False, f"low_snr_{snr:.1f}"

    return True, "pass"


# ---------------------------------------------------------------------------
# Full Pipeline Orchestration
# ---------------------------------------------------------------------------

def process_single_file(input_path, output_dir, skip_demucs=False, skip_diarize=True):
    """Run the full 10-step pipeline on a single audio file."""
    stem = Path(input_path).stem
    work_dir = os.path.join(output_dir, "_work", stem)
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "chunks"), exist_ok=True)

    log = {"input": input_path, "stem": stem, "steps": {}}

    # Step 2: Convert to Whisper format
    wav_16k = os.path.join(work_dir, f"{stem}_16k.wav")
    print(f"  [1/10] Converting to 16kHz mono...")
    convert_to_whisper_format(input_path, wav_16k)
    log["steps"]["convert"] = "done"

    # Step 3: Initial quality gate
    print(f"  [2/10] Initial quality gate...")
    passed, reason = initial_quality_gate(wav_16k)
    log["steps"]["initial_gate"] = reason
    if not passed:
        print(f"  REJECTED: {reason}")
        return log

    # Step 4: Classify segments
    print(f"  [3/10] Classifying segments (speech/music/noise)...")
    try:
        segments = classify_segments(wav_16k)
        music_time = sum(e - s for label, s, e in segments if label == 'music')
        speech_time = sum(e - s for label, s, e in segments if label in ('speech', 'male', 'female'))
        log["steps"]["classify"] = {
            "speech_sec": round(speech_time, 1),
            "music_sec": round(music_time, 1),
        }
        print(f"    Speech: {speech_time:.0f}s, Music: {music_time:.0f}s")
    except Exception as e:
        print(f"    Skipping classification: {e}")
        log["steps"]["classify"] = f"skipped: {e}"

    # Step 5: Source separation (if music detected)
    active_audio = wav_16k
    if not skip_demucs and music_time > 10:
        print(f"  [4/10] Source separation (demucs)...")
        try:
            vocals = separate_vocals(wav_16k, work_dir)
            if os.path.exists(vocals):
                active_audio = vocals
                log["steps"]["demucs"] = "done"
        except Exception as e:
            print(f"    Skipping demucs: {e}")
            log["steps"]["demucs"] = f"skipped: {e}"
    else:
        print(f"  [4/10] Skipping demucs (no significant music)")
        log["steps"]["demucs"] = "skipped"

    # Step 6: Bandpass filter
    print(f"  [5/10] Bandpass filter (80Hz-8kHz)...")
    filtered = os.path.join(work_dir, f"{stem}_filtered.wav")
    bandpass_filter(active_audio, filtered)
    log["steps"]["bandpass"] = "done"

    # Step 7: Denoise
    print(f"  [6/10] Spectral denoising...")
    audio, sr = sf.read(filtered)
    audio = denoise_audio(audio, sr)
    log["steps"]["denoise"] = "done"

    # Step 8: Normalize
    print(f"  [7/10] Loudness normalization (-16 LUFS)...")
    audio = normalize_loudness(audio, sr)
    normalized_path = os.path.join(work_dir, f"{stem}_normalized.wav")
    sf.write(normalized_path, audio, sr)
    log["steps"]["normalize"] = "done"

    # Step 9: VAD chunking
    print(f"  [8/10] VAD chunking...")
    chunks = vad_chunk(normalized_path)
    log["steps"]["vad_chunks"] = len(chunks)
    print(f"    Found {len(chunks)} speech chunks")

    # Step 10: Diarization (optional)
    if not skip_diarize:
        print(f"  [9/10] Speaker diarization...")
        try:
            diarization = diarize_speakers(normalized_path)
            log["steps"]["diarize"] = "done"
        except Exception as e:
            print(f"    Skipping diarization: {e}")
            log["steps"]["diarize"] = f"skipped: {e}"
    else:
        print(f"  [9/10] Skipping diarization")
        log["steps"]["diarize"] = "skipped"

    # Step 11: Export chunks with final quality gate
    print(f"  [10/10] Exporting chunks with quality gate...")
    audio_full, _ = sf.read(normalized_path)
    passed_chunks = 0
    rejected_chunks = 0

    for i, (start, end) in enumerate(chunks):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        chunk_audio = audio_full[start_sample:end_sample]

        ok, reason = final_quality_gate(chunk_audio, sr)
        if ok:
            chunk_path = os.path.join(
                output_dir, "chunks", f"{stem}_chunk{i:04d}.wav"
            )
            sf.write(chunk_path, chunk_audio, sr)
            passed_chunks += 1
        else:
            rejected_chunks += 1

    log["steps"]["final_gate"] = {
        "passed": passed_chunks,
        "rejected": rejected_chunks,
    }
    print(f"    Exported {passed_chunks} chunks, rejected {rejected_chunks}")

    return log


def main():
    parser = argparse.ArgumentParser(
        description="10-step audio preprocessing pipeline for church sermon data"
    )
    parser.add_argument("--input", "-i", default="stark_data/raw",
                        help="Input audio file or directory")
    parser.add_argument("--output", "-o", default="stark_data/cleaned",
                        help="Output directory for cleaned chunks")
    parser.add_argument("--download", action="store_true",
                        help="Download audio from YouTube first")
    parser.add_argument("--urls", default="urls.txt",
                        help="File with YouTube URLs (one per line)")
    parser.add_argument("--skip-demucs", action="store_true",
                        help="Skip source separation step")
    parser.add_argument("--skip-diarize", action="store_true", default=True,
                        help="Skip speaker diarization (default: skip)")
    parser.add_argument("--diarize", action="store_true",
                        help="Enable speaker diarization")
    args = parser.parse_args()

    if args.download:
        download_audio(args.urls, args.input)

    os.makedirs(args.output, exist_ok=True)

    # Collect input files
    input_path = Path(args.input)
    if input_path.is_file():
        files = [str(input_path)]
    else:
        files = sorted(glob.glob(os.path.join(args.input, "*.wav")))
        if not files:
            print(f"No WAV files found in {args.input}")
            sys.exit(1)

    print(f"Processing {len(files)} files...")
    all_logs = []

    for i, f in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] {os.path.basename(f)}")
        log = process_single_file(
            f, args.output,
            skip_demucs=args.skip_demucs,
            skip_diarize=not args.diarize,
        )
        all_logs.append(log)

    # Save processing log
    log_path = os.path.join(args.output, "preprocessing_log.json")
    with open(log_path, "w") as f:
        json.dump(all_logs, f, indent=2)
    print(f"\nProcessing log saved to {log_path}")


if __name__ == "__main__":
    main()
