#!/usr/bin/env python3
"""
batch_translate.py — Offline MP3 Batch Processor

Takes a directory of MP3 sermon files, runs them through the full pipeline
(STT → MarianMT + Gemma 4B translation → Piper TTS), saves Spanish WAV
output, English/Spanish transcripts, and a timing summary report.

Usage:
    python tools/batch_translate.py "/path/to/mp3s" \\
        --speaker "Stephen Grant" \\
        --output-dir stark_data/arlington_2026/

    python tools/batch_translate.py "/path/to/mp3s" \\
        --max-segments 3         # Quick test (3 segments per file)

    python tools/batch_translate.py "/path/to/mp3s" \\
        --skip-tts               # Transcribe + translate only, no audio output
"""

import os

os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import logging
import re
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_translate")

SAMPLE_RATE = 16000

# Theological vocabulary bias (same as dry_run_ab.py)
WHISPER_PROMPT = (
    "Sermon at Stark Road Gospel Hall. "
    "Christ Jesus, the Holy Spirit, God the Father. "
    "Atonement, propitiation, mediator, covenant, righteousness, "
    "sanctification, justification, redemption, reconciliation, "
    "repentance, reign, grace, mercy, the Gospel, epistle, apostle, "
    "Scripture, the Lord, the Word of God."
)


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------


def load_audio_file(path: str) -> np.ndarray:
    """Load an audio file (MP3/WAV/etc) as 16kHz mono float32 via ffmpeg."""
    import subprocess
    import tempfile

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
        import scipy.io.wavfile as wav

        sr, data = wav.read(tmp_path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype != np.float32:
            data = data.astype(np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return np.clip(data, -1.0, 1.0)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def save_wav(audio: np.ndarray, sample_rate: int, path: str) -> None:
    """Save float32 audio array to WAV file (int16)."""
    audio_int16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def wav_to_mp3(wav_path: str, mp3_path: str | None = None, bitrate: str = "128k") -> str:
    """Convert a WAV file to MP3 via ffmpeg. Returns the MP3 path."""
    import subprocess

    if mp3_path is None:
        mp3_path = os.path.splitext(wav_path)[0] + ".mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-b:a", bitrate, mp3_path],
        capture_output=True,
        check=True,
    )
    return mp3_path


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio from orig_sr to target_sr."""
    if orig_sr == target_sr:
        return audio
    from scipy.signal import resample

    num_samples = int(len(audio) * target_sr / orig_sr)
    return resample(audio, num_samples).astype(np.float32)


# ---------------------------------------------------------------------------
# VAD segmentation
# ---------------------------------------------------------------------------


def segment_audio_vad(
    audio: np.ndarray,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
) -> list[tuple[np.ndarray, int, int]]:
    """Segment audio using Silero VAD into speech chunks.

    Returns list of (audio_chunk, start_sample, end_sample) tuples.
    """
    import torch

    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    get_speech_ts = utils[0]

    audio_tensor = torch.from_numpy(audio)
    timestamps = get_speech_ts(audio_tensor, model, sampling_rate=SAMPLE_RATE)

    segments: list[tuple[np.ndarray, int, int]] = []
    for ts in timestamps:
        start = ts["start"]
        end = ts["end"]
        duration = (end - start) / SAMPLE_RATE
        if duration < min_duration:
            continue
        if duration > max_duration:
            chunk_samples = int(max_duration * SAMPLE_RATE)
            for offset in range(start, end, chunk_samples):
                seg_end = min(offset + chunk_samples, end)
                seg = audio[offset:seg_end]
                if len(seg) >= int(min_duration * SAMPLE_RATE):
                    segments.append((seg, offset, seg_end))
        else:
            segments.append((audio[start:end], start, end))

    logger.info("VAD found %d speech segments", len(segments))
    return segments


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def slugify(name: str) -> str:
    """Convert a filename stem into a filesystem-friendly slug."""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s_]+", "_", name)
    return name.strip("_")


def discover_mp3s(input_dir: Path, speaker_filter: str | None = None) -> list[Path]:
    """Find MP3 files in input_dir, optionally filtered by speaker name."""
    mp3s = sorted(input_dir.glob("*.mp3"))
    if not mp3s:
        # Try case-insensitive
        mp3s = sorted(p for p in input_dir.iterdir() if p.suffix.lower() == ".mp3")

    if speaker_filter:
        filtered = [p for p in mp3s if speaker_filter.lower() in p.stem.lower()]
        logger.info(
            "Speaker filter %r: %d/%d files match",
            speaker_filter,
            len(filtered),
            len(mp3s),
        )
        return filtered

    return mp3s


# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------


def process_file(
    mp3_path: Path,
    output_dir: Path,
    *,
    stt_engine,
    marian_engine,
    gemma_engine,
    tts_engine,
    mlx_executor: ThreadPoolExecutor,
    skip_tts: bool = False,
    max_segments: int | None = None,
    lang: str = "en",
) -> dict:
    """Process a single MP3 file through the full pipeline.

    Returns a summary dict with timing and segment stats.
    """
    file_start = time.time()
    stem = slugify(mp3_path.stem)
    file_dir = output_dir / stem
    file_dir.mkdir(parents=True, exist_ok=True)

    segments_jsonl = file_dir / "segments.jsonl"

    # Determine which segments were already processed (resume support)
    done_indices: set[int] = set()
    if segments_jsonl.exists():
        with open(segments_jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        done_indices.add(json.loads(line)["segment_index"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        if done_indices:
            logger.info("Resuming %s: %d segments already done", stem, len(done_indices))

    # Load audio
    logger.info("Loading audio: %s", mp3_path.name)
    t0 = time.time()
    audio = load_audio_file(str(mp3_path))
    load_ms = (time.time() - t0) * 1000
    audio_duration = len(audio) / SAMPLE_RATE
    logger.info(
        "Audio loaded: %.1fs (%.1f min), load time: %.0fms",
        audio_duration,
        audio_duration / 60,
        load_ms,
    )

    # VAD segmentation
    logger.info("Running VAD segmentation...")
    t0 = time.time()
    vad_segments = segment_audio_vad(audio)
    vad_ms = (time.time() - t0) * 1000
    logger.info("VAD: %d segments in %.0fms", len(vad_segments), vad_ms)

    if max_segments is not None:
        vad_segments = vad_segments[:max_segments]
        logger.info("Limiting to %d segments (--max-segments)", max_segments)

    # Translation direction
    source_lang = lang
    target_lang = "es" if lang == "en" else "en"

    # Per-segment processing
    all_en: list[str] = []
    all_es_marian: list[str] = []
    all_es_gemma: list[str] = []
    tts_audio_chunks: list[np.ndarray] = []
    tts_sample_rate: int | None = None

    timing_stt: list[float] = []
    timing_marian: list[float] = []
    timing_gemma: list[float] = []
    timing_tts: list[float] = []

    total = len(vad_segments)
    latency_sum = 0.0
    segments_processed = 0

    for i, (seg_audio, start_sample, end_sample) in enumerate(vad_segments):
        start_sec = start_sample / SAMPLE_RATE
        end_sec = end_sample / SAMPLE_RATE
        seg_duration = len(seg_audio) / SAMPLE_RATE

        # Skip already-processed segments
        if i in done_indices:
            # Reload text from JSONL for transcript assembly
            # (we'll reconstruct at the end from the file)
            continue

        # ETA calculation
        if segments_processed > 0:
            avg_latency = latency_sum / segments_processed
            remaining = total - (i + 1)
            eta_sec = avg_latency * remaining
            eta_str = f"  ETA: {_fmt_duration(eta_sec)}"
        else:
            eta_str = ""

        logger.info(
            "[%s] Segment %d/%d (%.1fs-%.1fs, %.1fs audio)%s",
            stem,
            i + 1,
            total,
            start_sec,
            end_sec,
            seg_duration,
            eta_str,
        )

        seg_start = time.time()
        seg_record: dict = {
            "segment_index": i,
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
            "duration_sec": round(seg_duration, 2),
        }

        # 1. STT
        t0 = time.perf_counter()
        stt_result = mlx_executor.submit(
            stt_engine.transcribe,
            seg_audio,
            language=source_lang,
            initial_prompt=WHISPER_PROMPT,
        ).result(timeout=120)
        stt_ms = (time.perf_counter() - t0) * 1000
        source_text = stt_result.text.strip()
        timing_stt.append(stt_ms)

        seg_record["source_text"] = source_text
        seg_record["stt_ms"] = round(stt_ms, 1)
        seg_record["stt_confidence"] = stt_result.confidence

        if not source_text:
            logger.info("  [STT] Empty transcription, skipping segment")
            _append_jsonl(segments_jsonl, seg_record)
            continue

        logger.info("  [STT] %.0fms: %s", stt_ms, source_text[:100])

        # 2. MarianMT translate
        t0 = time.perf_counter()
        marian_result = marian_engine.translate(
            source_text,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        marian_ms = (time.perf_counter() - t0) * 1000
        marian_text = marian_result.text.strip()
        timing_marian.append(marian_ms)

        seg_record["marian_text"] = marian_text
        seg_record["marian_ms"] = round(marian_ms, 1)

        logger.info("  [Marian] %.0fms: %s", marian_ms, marian_text[:100])

        # 3. Gemma 4B translate
        t0 = time.perf_counter()
        gemma_result = mlx_executor.submit(
            gemma_engine.translate,
            source_text,
            source_lang=source_lang,
            target_lang=target_lang,
        ).result(timeout=120)
        gemma_ms = (time.perf_counter() - t0) * 1000
        gemma_text = gemma_result.text.strip()
        timing_gemma.append(gemma_ms)

        seg_record["gemma_text"] = gemma_text
        seg_record["gemma_ms"] = round(gemma_ms, 1)

        logger.info("  [Gemma] %.0fms: %s", gemma_ms, gemma_text[:100])

        # 4. TTS (on Gemma translation)
        if not skip_tts:
            t0 = time.perf_counter()
            tts_result = tts_engine.synthesize(gemma_text, language=target_lang)
            tts_ms = (time.perf_counter() - t0) * 1000
            timing_tts.append(tts_ms)
            tts_audio_chunks.append(tts_result.audio)
            tts_sample_rate = tts_result.sample_rate
            seg_record["tts_ms"] = round(tts_ms, 1)
            logger.info("  [TTS] %.0fms", tts_ms)

        # Accumulate text
        all_en.append(source_text)
        all_es_marian.append(marian_text)
        all_es_gemma.append(gemma_text)

        # Write incrementally
        _append_jsonl(segments_jsonl, seg_record)

        seg_elapsed = time.time() - seg_start
        latency_sum += seg_elapsed
        segments_processed += 1

    # If resuming, reload all segments from JSONL for transcript assembly
    if done_indices:
        all_en, all_es_marian, all_es_gemma = _reload_transcripts(segments_jsonl)

    # Write full transcripts
    _write_transcript(file_dir / "transcript_en.txt", all_en)
    target_label = target_lang
    _write_transcript(file_dir / f"transcript_{target_label}_marian.txt", all_es_marian)
    _write_transcript(file_dir / f"transcript_{target_label}_gemma.txt", all_es_gemma)

    # Concatenate TTS audio with silence gaps
    if tts_audio_chunks and tts_sample_rate:
        silence = np.zeros(int(0.3 * tts_sample_rate), dtype=np.float32)
        parts: list[np.ndarray] = []
        for chunk in tts_audio_chunks:
            parts.append(chunk)
            parts.append(silence)
        concat_audio = np.concatenate(parts)
        wav_path = str(file_dir / f"audio_{target_label}.wav")
        save_wav(concat_audio, tts_sample_rate, wav_path)
        tts_duration = len(concat_audio) / tts_sample_rate
        logger.info("TTS audio saved: %s (%.1fs)", wav_path, tts_duration)

        mp3_path = wav_to_mp3(wav_path)
        wav_kb = os.path.getsize(wav_path) / 1024
        mp3_kb = os.path.getsize(mp3_path) / 1024
        logger.info(
            "MP3 compressed: %s (%.0f KB -> %.0f KB, %.0f%% reduction)",
            mp3_path,
            wav_kb,
            mp3_kb,
            (1 - mp3_kb / wav_kb) * 100 if wav_kb > 0 else 0,
        )

    file_elapsed = time.time() - file_start

    return {
        "file": mp3_path.name,
        "slug": stem,
        "audio_duration_sec": round(audio_duration, 1),
        "segment_count": total,
        "segments_processed": segments_processed,
        "processing_time_sec": round(file_elapsed, 1),
        "avg_stt_ms": round(sum(timing_stt) / len(timing_stt), 1) if timing_stt else 0,
        "avg_marian_ms": round(sum(timing_marian) / len(timing_marian), 1) if timing_marian else 0,
        "avg_gemma_ms": round(sum(timing_gemma) / len(timing_gemma), 1) if timing_gemma else 0,
        "avg_tts_ms": round(sum(timing_tts) / len(timing_tts), 1) if timing_tts else 0,
        "total_stt_ms": round(sum(timing_stt), 1),
        "total_marian_ms": round(sum(timing_marian), 1),
        "total_gemma_ms": round(sum(timing_gemma), 1),
        "total_tts_ms": round(sum(timing_tts), 1),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _append_jsonl(path: Path, record: dict) -> None:
    """Append a JSON record to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _reload_transcripts(jsonl_path: Path) -> tuple[list[str], list[str], list[str]]:
    """Reload transcripts from a segments JSONL file, sorted by index."""
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    records.sort(key=lambda r: r.get("segment_index", 0))

    en = [r["source_text"] for r in records if r.get("source_text")]
    marian = [r["marian_text"] for r in records if r.get("marian_text")]
    gemma = [r["gemma_text"] for r in records if r.get("gemma_text")]
    return en, marian, gemma


def _write_transcript(path: Path, lines: list[str]) -> None:
    """Write transcript lines to a text file."""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h{mins:02d}m"


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def generate_summary(file_results: list[dict], output_dir: Path, total_elapsed: float) -> str:
    """Generate a markdown summary report."""
    lines: list[str] = []
    lines.append("# Batch Translation Summary")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total processing time:** {_fmt_duration(total_elapsed)}")
    lines.append(f"**Files processed:** {len(file_results)}")
    lines.append("")

    total_audio = sum(r["audio_duration_sec"] for r in file_results)
    total_segments = sum(r["segment_count"] for r in file_results)
    lines.append(f"**Total audio:** {_fmt_duration(total_audio)} ({total_audio / 60:.1f} min)")
    lines.append(f"**Total segments:** {total_segments}")
    if total_elapsed > 0:
        rtf = total_audio / total_elapsed
        lines.append(f"**Real-time factor:** {rtf:.2f}x (1.0 = real-time)")
    lines.append("")

    # Per-file table
    lines.append("## Per-File Results")
    lines.append("")
    lines.append("| File | Duration | Segments | Processing | RTF |")
    lines.append("|------|----------|----------|------------|-----|")
    for r in file_results:
        dur = _fmt_duration(r["audio_duration_sec"])
        proc = _fmt_duration(r["processing_time_sec"])
        rtf = r["audio_duration_sec"] / r["processing_time_sec"] if r["processing_time_sec"] > 0 else 0
        lines.append(f"| {r['file']} | {dur} | {r['segment_count']} | {proc} | {rtf:.2f}x |")
    lines.append("")

    # Timing breakdown
    lines.append("## Timing Breakdown (averages per segment)")
    lines.append("")
    lines.append("| File | STT | MarianMT | Gemma 4B | TTS |")
    lines.append("|------|-----|----------|----------|-----|")
    for r in file_results:
        lines.append(
            f"| {r['slug'][:30]} "
            f"| {r['avg_stt_ms']:.0f}ms "
            f"| {r['avg_marian_ms']:.0f}ms "
            f"| {r['avg_gemma_ms']:.0f}ms "
            f"| {r['avg_tts_ms']:.0f}ms |"
        )
    lines.append("")

    # Aggregate timing breakdown
    total_stt = sum(r["total_stt_ms"] for r in file_results)
    total_marian = sum(r["total_marian_ms"] for r in file_results)
    total_gemma = sum(r["total_gemma_ms"] for r in file_results)
    total_tts = sum(r["total_tts_ms"] for r in file_results)
    total_pipeline = total_stt + total_marian + total_gemma + total_tts

    if total_pipeline > 0:
        lines.append("## Aggregate Time Distribution")
        lines.append("")
        lines.append(f"- **STT (Whisper):** {total_stt / 1000:.1f}s ({total_stt / total_pipeline * 100:.0f}%)")
        lines.append(f"- **MarianMT:** {total_marian / 1000:.1f}s ({total_marian / total_pipeline * 100:.0f}%)")
        lines.append(f"- **Gemma 4B:** {total_gemma / 1000:.1f}s ({total_gemma / total_pipeline * 100:.0f}%)")
        lines.append(f"- **Piper TTS:** {total_tts / 1000:.1f}s ({total_tts / total_pipeline * 100:.0f}%)")
        lines.append(f"- **Other (VAD, I/O):** {total_elapsed - total_pipeline / 1000:.1f}s")
    lines.append("")

    report = "\n".join(lines)
    report_path = output_dir / "summary.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Summary written to %s", report_path)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline MP3 batch processor: STT → Translation → TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing MP3 files",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        help="Filter filenames containing this speaker name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: stark_data/batch_<timestamp>/)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        choices=["en", "es"],
        help="Source language (default: en)",
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Skip TTS synthesis (transcribe + translate only)",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Limit segments per file (for quick testing)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    # Discover MP3 files
    mp3s = discover_mp3s(input_dir, speaker_filter=args.speaker)
    if not mp3s:
        logger.error("No MP3 files found in %s", input_dir)
        sys.exit(1)

    logger.info("Found %d MP3 files:", len(mp3s))
    for mp3 in mp3s:
        logger.info("  %s", mp3.name)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _PROJECT_ROOT / "stark_data" / f"batch_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # --- Engine setup ---
    from engines.factory import create_stt_engine, create_translation_engine, create_tts_engine

    logger.info("Loading engines...")

    stt_engine = create_stt_engine(backend="mlx")
    marian_engine = create_translation_engine(backend="mlx", engine_type="marian")
    gemma_engine = create_translation_engine(backend="mlx", engine_type="gemma")

    stt_engine.load()
    marian_engine.load()
    gemma_engine.load()

    tts_engine = None
    if not args.skip_tts:
        target_lang = "es" if args.lang == "en" else "en"
        voices = {target_lang: "es_MX-claude-high" if target_lang == "es" else "en_US-lessac-high"}
        tts_engine = create_tts_engine(voices=voices)
        tts_engine.load()

    # MLX thread safety: single thread for all MLX calls (STT + Gemma)
    mlx_executor = ThreadPoolExecutor(max_workers=1)

    logger.info("All engines loaded. Starting batch processing.")

    # --- Process files ---
    t_total_start = time.time()
    file_results: list[dict] = []

    for idx, mp3 in enumerate(mp3s, start=1):
        logger.info("")
        logger.info("=" * 60)
        logger.info("FILE %d/%d: %s", idx, len(mp3s), mp3.name)
        logger.info("=" * 60)

        result = process_file(
            mp3,
            output_dir,
            stt_engine=stt_engine,
            marian_engine=marian_engine,
            gemma_engine=gemma_engine,
            tts_engine=tts_engine,
            mlx_executor=mlx_executor,
            skip_tts=args.skip_tts,
            max_segments=args.max_segments,
            lang=args.lang,
        )
        file_results.append(result)

        logger.info(
            "Completed %s: %d segments in %s",
            mp3.name,
            result["segment_count"],
            _fmt_duration(result["processing_time_sec"]),
        )

    total_elapsed = time.time() - t_total_start

    # --- Summary ---
    report = generate_summary(file_results, output_dir, total_elapsed)
    print("\n" + report)

    # Cleanup
    mlx_executor.shutdown(wait=False)
    stt_engine.unload()
    marian_engine.unload()
    gemma_engine.unload()
    if tts_engine:
        tts_engine.unload()


if __name__ == "__main__":
    main()
