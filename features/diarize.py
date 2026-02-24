#!/usr/bin/env python3
"""
diarize.py — Speaker Diarization for Church Sermon Audio

Uses pyannote-audio 3.1 for 2-speaker (default) or multi-speaker diarization.
Combines diarization with Whisper STT to produce speaker-labeled transcripts.

Input:
  - A single WAV file, OR
  - A live session directory from stark_data/live_sessions/{SESSION_ID}/
    (reassembles chunk_*.wav files in order)

Output:
  - JSONL file with speaker labels, timestamps, and transcribed text
  - One JSON line per speech segment

Requirements:
  - pyannote-audio >= 3.1 (pip install pyannote.audio)
  - HF_TOKEN env var with a HuggingFace token that has accepted
    pyannote/speaker-diarization-3.1 terms of use
  - mlx-whisper for transcription

Usage:
    python diarize.py audio.wav
    python diarize.py audio.wav --min-speakers 2 --max-speakers 2
    python diarize.py stark_data/live_sessions/20260208_183356/
    python diarize.py audio.wav --embeddings speaker_samples/ --max-speakers 4
    python diarize.py audio.wav --output results/diarized.jsonl
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000  # Whisper requires 16kHz
DEFAULT_MIN_SPEAKERS = 2
DEFAULT_MAX_SPEAKERS = 2
OUTPUT_DIR = "metrics/diarization"

# Whisper initial_prompt — same theological vocabulary bias as dry_run_ab.py
WHISPER_PROMPT = (
    "Sermon at Stark Road Gospel Hall. "
    "God is always capitalized. Christ Jesus, the Holy Spirit, God the Father. "
    "Topics: salvation, atonement, propitiation, mediator, covenant, "
    "righteousness, sanctification, justification, redemption, Savior, "
    "reconciliation, intercession, predestination, sovereignty, "
    "sin, repentance, forgiveness, reign, bow, glory, grace, mercy, "
    "the Gospel, epistle, apostle, tabernacle, congregation, "
    "First Timothy, Second Timothy, First Corinthians, Ephesians, "
    "Romans, Hebrews, Galatians, Colossians, Thessalonians, Philippians, "
    "Genesis, Exodus, Leviticus, Deuteronomy, Isaiah, Jeremiah, Ezekiel, "
    "Scripture, the Lord, the Word of God."
)


# ---------------------------------------------------------------------------
# Audio Loading
# ---------------------------------------------------------------------------


def load_audio_file(path):
    """Load a single WAV file and return (audio_array, sample_rate)."""
    import scipy.io.wavfile as wav

    sr, audio = wav.read(path)
    # Convert to float32 normalized [-1, 1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Convert stereo to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        from scipy.signal import resample

        target_len = int(len(audio) * SAMPLE_RATE / sr)
        audio = resample(audio, target_len).astype(np.float32)
        sr = SAMPLE_RATE

    return audio, sr


def load_session_directory(session_dir):
    """Reassemble chunk_*.wav files from a live session directory.

    Chunks are sorted by filename (chunk_0001.wav, chunk_0002.wav, etc.)
    and concatenated into a single audio array.
    """
    session_path = Path(session_dir)
    if not session_path.is_dir():
        print(f"ERROR: Session directory not found: {session_dir}", file=sys.stderr)
        sys.exit(1)

    chunk_files = sorted(session_path.glob("chunk_*.wav"))
    if not chunk_files:
        print(f"ERROR: No chunk_*.wav files found in {session_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"  Found {len(chunk_files)} chunks in {session_dir}")
    segments = []
    for cf in chunk_files:
        audio, _ = load_audio_file(str(cf))
        segments.append(audio)

    combined = np.concatenate(segments)
    duration = len(combined) / SAMPLE_RATE
    print(f"  Combined audio: {duration:.1f}s ({duration / 60:.1f} min)")
    return combined, SAMPLE_RATE


def load_input_audio(input_path):
    """Load audio from a WAV file or a session directory."""
    p = Path(input_path)
    if p.is_dir():
        return load_session_directory(input_path)
    elif p.is_file() and p.suffix.lower() in (".wav", ".flac", ".mp3", ".m4a"):
        print(f"  Loading audio: {input_path}")
        audio, sr = load_audio_file(input_path)
        duration = len(audio) / sr
        print(f"  Duration: {duration:.1f}s ({duration / 60:.1f} min)")
        return audio, sr
    else:
        print(f"ERROR: Input must be a WAV file or session directory: {input_path}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Speaker Embedding / Identification
# ---------------------------------------------------------------------------


def load_speaker_embeddings(embeddings_dir, inference_obj):
    """Load enrollment WAV samples and compute speaker embeddings.

    Each WAV file in the embeddings directory represents one known speaker.
    The filename (without extension) is used as the speaker label.

    Args:
        embeddings_dir: Directory containing WAV files, one per speaker.
            e.g., embeddings/pastor_john.wav, embeddings/elder_mark.wav
        inference_obj: pyannote Inference object for embedding extraction.

    Returns:
        dict mapping speaker_label -> embedding_array
    """
    emb_path = Path(embeddings_dir)
    if not emb_path.is_dir():
        print(f"WARNING: Embeddings directory not found: {embeddings_dir}", file=sys.stderr)
        return {}

    embeddings = {}
    for wav_file in sorted(emb_path.glob("*.wav")):
        label = wav_file.stem  # filename without extension = speaker name
        try:
            embedding = inference_obj(str(wav_file))
            embeddings[label] = embedding
            print(f"  Enrolled speaker: {label} (from {wav_file.name})")
        except Exception as e:
            print(f"  WARNING: Failed to enroll {wav_file.name}: {e}", file=sys.stderr)

    if not embeddings:
        print(f"WARNING: No valid speaker embeddings loaded from {embeddings_dir}", file=sys.stderr)

    return embeddings


def identify_speakers(diarization, embeddings, inference_obj, audio_path):
    """Map anonymous diarization labels (SPEAKER_00, etc.) to known speaker names.

    For each diarization speaker label, extract embeddings from their segments,
    average them, and find the closest match in the enrollment embeddings.

    Args:
        diarization: pyannote Annotation object from the diarization pipeline.
        embeddings: dict from load_speaker_embeddings().
        inference_obj: pyannote Inference object.
        audio_path: Path to the audio file for embedding extraction.

    Returns:
        dict mapping anonymous label -> identified name (or original label if no match).
    """
    from pyannote.core import Segment

    if not embeddings:
        return {}

    # Get unique speaker labels from diarization
    speaker_labels = set()
    for segment, _, label in diarization.itertracks(yield_label=True):
        speaker_labels.add(label)

    label_map = {}
    for label in speaker_labels:
        # Collect segments for this speaker
        speaker_segments = []
        for segment, _, seg_label in diarization.itertracks(yield_label=True):
            if seg_label == label:
                speaker_segments.append(segment)

        if not speaker_segments:
            label_map[label] = label
            continue

        # Extract embedding from longest segments (more reliable)
        speaker_segments.sort(key=lambda s: s.duration, reverse=True)
        top_segments = speaker_segments[: min(5, len(speaker_segments))]

        # Average embeddings across top segments
        seg_embeddings = []
        for seg in top_segments:
            try:
                # Crop audio to segment and extract embedding
                crop = Segment(seg.start, min(seg.end, seg.start + 30.0))
                emb = inference_obj.crop(audio_path, crop)
                seg_embeddings.append(emb)
            except Exception:
                continue

        if not seg_embeddings:
            label_map[label] = label
            continue

        avg_embedding = np.mean(seg_embeddings, axis=0)

        # Find closest enrolled speaker by cosine similarity
        best_name = label
        best_sim = -1.0
        for name, ref_emb in embeddings.items():
            # Cosine similarity
            sim = float(
                np.dot(avg_embedding, ref_emb) / (np.linalg.norm(avg_embedding) * np.linalg.norm(ref_emb) + 1e-8)
            )
            if sim > best_sim:
                best_sim = sim
                best_name = name

        # Threshold: only accept if similarity is reasonable
        if best_sim > 0.5:
            label_map[label] = best_name
            print(f"  Identified {label} -> {best_name} (similarity: {best_sim:.3f})")
        else:
            label_map[label] = label
            print(f"  {label}: best match was {best_name} ({best_sim:.3f}) -- below threshold")

    return label_map


# ---------------------------------------------------------------------------
# Diarization Pipeline
# ---------------------------------------------------------------------------


def run_diarization(audio_path, min_speakers, max_speakers):
    """Run pyannote speaker diarization pipeline.

    Args:
        audio_path: Path to WAV file (must be on disk for pyannote).
        min_speakers: Minimum expected speakers.
        max_speakers: Maximum expected speakers.

    Returns:
        pyannote Annotation object with speaker segments.
    """
    # Check HF_TOKEN
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set.", file=sys.stderr)
        print("  pyannote-audio requires a HuggingFace token.", file=sys.stderr)
        print("  1. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1", file=sys.stderr)
        print("  2. export HF_TOKEN=hf_...", file=sys.stderr)
        sys.exit(1)

    from pyannote.audio import Pipeline

    print("[1/3] Loading pyannote speaker-diarization-3.1...")
    t0 = time.time()
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    print(f"  Pipeline loaded ({time.time() - t0:.1f}s)")

    print(f"[2/3] Running diarization (speakers: {min_speakers}-{max_speakers})...")
    t0 = time.time()
    diarization = pipeline(
        audio_path,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    elapsed = time.time() - t0
    audio_dur = _get_audio_duration(audio_path)
    rtf = elapsed / audio_dur if audio_dur > 0 else 0
    print(f"  Diarization complete ({elapsed:.1f}s, RTF={rtf:.2f}x)")

    # Print speaker summary
    speaker_times = {}
    for segment, _, label in diarization.itertracks(yield_label=True):
        dur = segment.end - segment.start
        speaker_times[label] = speaker_times.get(label, 0.0) + dur

    print(f"  Speakers found: {len(speaker_times)}")
    for label, total in sorted(speaker_times.items(), key=lambda x: -x[1]):
        pct = total / audio_dur * 100 if audio_dur > 0 else 0
        print(f"    {label}: {total:.1f}s ({pct:.0f}%)")

    return diarization


def _get_audio_duration(audio_path):
    """Get duration of an audio file in seconds."""
    import scipy.io.wavfile as wav

    sr, audio = wav.read(audio_path)
    if audio.ndim > 1:
        return len(audio) / sr
    return len(audio) / sr


# ---------------------------------------------------------------------------
# Transcription (per-segment STT)
# ---------------------------------------------------------------------------


def transcribe_segments(audio, diarization, label_map=None):
    """Transcribe each diarized segment using mlx-whisper.

    Args:
        audio: Full audio as numpy float32 array (16kHz mono).
        diarization: pyannote Annotation object.
        label_map: Optional dict mapping anonymous labels to identified names.

    Returns:
        List of dicts with speaker, start, end, duration, text.
    """
    import mlx.core as mx
    import mlx_whisper

    mx.set_cache_limit(100 * 1024 * 1024)

    model_id = "mlx-community/distil-whisper-large-v3"
    print(f"[3/3] Transcribing segments with {model_id}...")

    # Warm up Whisper
    t0 = time.time()
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    mlx_whisper.transcribe(silence, path_or_hf_repo=model_id, condition_on_previous_text=False)
    print(f"  Whisper ready ({time.time() - t0:.1f}s)")

    # Collect segments sorted by time
    segments = []
    for segment, _, label in diarization.itertracks(yield_label=True):
        speaker = label_map.get(label, label) if label_map else label
        segments.append(
            {
                "speaker": speaker,
                "start": segment.start,
                "end": segment.end,
                "duration": segment.end - segment.start,
            }
        )

    # Sort by start time
    segments.sort(key=lambda s: s["start"])

    # Merge adjacent segments from the same speaker (within 0.5s gap)
    merged = []
    for seg in segments:
        if merged and merged[-1]["speaker"] == seg["speaker"] and seg["start"] - merged[-1]["end"] < 0.5:
            merged[-1]["end"] = seg["end"]
            merged[-1]["duration"] = merged[-1]["end"] - merged[-1]["start"]
        else:
            merged.append(dict(seg))

    print(f"  Segments: {len(segments)} raw -> {len(merged)} merged")

    # Transcribe each merged segment
    results = []
    prev_text = ""
    for i, seg in enumerate(merged):
        start_sample = int(seg["start"] * SAMPLE_RATE)
        end_sample = int(seg["end"] * SAMPLE_RATE)
        segment_audio = audio[start_sample:end_sample]

        # Skip very short segments
        if len(segment_audio) < SAMPLE_RATE * 0.3:
            continue

        prompt = WHISPER_PROMPT
        if prev_text:
            prompt = WHISPER_PROMPT + " " + prev_text[-200:]

        try:
            result = mlx_whisper.transcribe(
                segment_audio,
                path_or_hf_repo=model_id,
                language="en",
                condition_on_previous_text=False,
                initial_prompt=prompt,
                word_timestamps=True,
            )
            text = result["text"].strip()
        except Exception as e:
            text = f"[transcription error: {e}]"

        if not text:
            continue

        prev_text = text

        # Extract confidence
        confidence = None
        stt_segments = result.get("segments", [])
        if stt_segments:
            logprobs = [s.get("avg_logprob", 0) for s in stt_segments if s.get("avg_logprob") is not None]
            if logprobs:
                mean_lp = sum(logprobs) / len(logprobs)
                confidence = round(min(1.0, max(0.0, 1.0 + mean_lp)), 2)

        entry = {
            "speaker": seg["speaker"],
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "duration": round(seg["duration"], 3),
            "text": text,
            "confidence": confidence,
        }
        results.append(entry)

        # Progress
        ts_start = _format_timestamp(seg["start"])
        ts_end = _format_timestamp(seg["end"])
        conf_str = f" (conf={confidence:.2f})" if confidence is not None else ""
        print(f"  [{ts_start}-{ts_end}] {seg['speaker']}{conf_str}: {text[:80]}")

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _format_timestamp(seconds):
    """Format seconds as HH:MM:SS.mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def write_jsonl(results, output_path, metadata=None):
    """Write diarized transcript to JSONL file.

    Each line is a JSON object with: speaker, start, end, duration, text, confidence.
    The first line is a metadata header if provided.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        if metadata:
            f.write(json.dumps({"_metadata": metadata}) + "\n")
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    print(f"\n  Output: {output_path}")
    print(f"  Segments: {len(results)}")
    if results:
        speakers = set(r["speaker"] for r in results)
        total_dur = sum(r["duration"] for r in results)
        print(f"  Speakers: {len(speakers)} ({', '.join(sorted(speakers))})")
        print(f"  Total speech: {total_dur:.1f}s ({total_dur / 60:.1f} min)")


def write_temp_wav(audio, sample_rate):
    """Write audio array to a temporary WAV file (pyannote needs a file path)."""
    import tempfile

    import scipy.io.wavfile as wav

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(tmp.name, sample_rate, audio)
    return tmp.name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Speaker diarization for church sermon audio (pyannote 3.1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python diarize.py sermon.wav
    python diarize.py sermon.wav --min-speakers 2 --max-speakers 2
    python diarize.py stark_data/live_sessions/20260208_183356/
    python diarize.py sermon.wav --embeddings speaker_samples/ --max-speakers 4
    python diarize.py sermon.wav -o metrics/diarization/sermon.jsonl
        """,
    )
    parser.add_argument("input", help="WAV file or live session directory")
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=DEFAULT_MIN_SPEAKERS,
        help=f"Minimum speakers (default: {DEFAULT_MIN_SPEAKERS})",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=DEFAULT_MAX_SPEAKERS,
        help=f"Maximum speakers (default: {DEFAULT_MAX_SPEAKERS})",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Directory of WAV enrollment samples for speaker identification "
        "(one file per speaker, filename = speaker name)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (default: metrics/diarization/<input_name>.jsonl)",
    )
    parser.add_argument(
        "--no-transcribe", action="store_true", help="Skip STT — output only speaker segments with timestamps"
    )
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print("  Speaker Diarization (pyannote 3.1)")
    print(f"  Input: {args.input}")
    print(f"  Speakers: {args.min_speakers}-{args.max_speakers}")
    if args.embeddings:
        print(f"  Enrollment: {args.embeddings}")
    print(f"{'=' * 60}\n")

    # Load audio
    audio, sr = load_input_audio(args.input)

    # Write to temp file for pyannote (it needs a file path)
    tmp_path = write_temp_wav(audio, sr)

    try:
        # Run diarization
        diarization = run_diarization(tmp_path, args.min_speakers, args.max_speakers)

        # Optional: speaker identification from enrollment embeddings
        label_map = {}
        if args.embeddings:
            hf_token = os.environ.get("HF_TOKEN")
            from pyannote.audio import Inference

            embedding_model = Inference(
                "pyannote/embedding",
                use_auth_token=hf_token,
                window="whole",
            )
            enrolled = load_speaker_embeddings(args.embeddings, embedding_model)
            if enrolled:
                label_map = identify_speakers(diarization, enrolled, embedding_model, tmp_path)

        # Transcribe segments (or just output timestamps)
        if args.no_transcribe:
            results = []
            for segment, _, label in diarization.itertracks(yield_label=True):
                speaker = label_map.get(label, label) if label_map else label
                results.append(
                    {
                        "speaker": speaker,
                        "start": round(segment.start, 3),
                        "end": round(segment.end, 3),
                        "duration": round(segment.end - segment.start, 3),
                        "text": None,
                        "confidence": None,
                    }
                )
            results.sort(key=lambda r: r["start"])
        else:
            results = transcribe_segments(audio, diarization, label_map)

        # Determine output path
        output_path = args.output
        if output_path is None:
            input_name = Path(args.input).stem
            if Path(args.input).is_dir():
                input_name = Path(args.input).name
            output_path = os.path.join(OUTPUT_DIR, f"{input_name}.jsonl")

        # Write output
        metadata = {
            "input": args.input,
            "min_speakers": args.min_speakers,
            "max_speakers": args.max_speakers,
            "embeddings": args.embeddings,
            "timestamp": datetime.now().isoformat(),
            "audio_duration_s": round(len(audio) / sr, 1),
            "num_segments": len(results),
            "speakers": sorted(set(r["speaker"] for r in results)) if results else [],
        }
        write_jsonl(results, output_path, metadata)

        # Print summary
        print(f"\n{'=' * 60}")
        print("  DIARIZATION COMPLETE")
        print(f"{'=' * 60}")

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
