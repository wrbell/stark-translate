#!/usr/bin/env python3
"""
roundtrip_test.py — End-to-End Bilingual Roundtrip Quality Test

Validates the full pipeline roundtrip:
  EN text → TTS → STT → EN→ES translation → TTS → STT → ES→EN translation → EN text

Exercises every engine (TTS, STT, Translation) in both directions and produces
measurable quality metrics (WER against originals).

Documents are loaded from text files (not embedded) to avoid content-filter issues.
Place .txt files in tools/roundtrip_texts/ or pass paths via --input-file.

Usage:
    # Place text files in tools/roundtrip_texts/ then:
    python tools/roundtrip_test.py                              # Run all documents
    python tools/roundtrip_test.py --max-chunks 3               # Quick test (3 chunks/doc)
    python tools/roundtrip_test.py --input-file my_text.txt     # Custom text file
    python tools/roundtrip_test.py --output-dir /tmp/roundtrip  # Custom output dir
"""

import os

os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import difflib
import logging
import sys
import textwrap
import time
import wave
from pathlib import Path

# Ensure project root is on sys.path so `engines` package is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from scipy.signal import resample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("roundtrip")

# ---------------------------------------------------------------------------
# Minimal built-in sample (fallback when no text files are provided)
# ---------------------------------------------------------------------------
BUILTIN_SAMPLE = (
    "We hold these truths to be self-evident, that all men are created equal, "
    "that they are endowed by their Creator with certain unalienable Rights, "
    "that among these are Life, Liberty and the pursuit of Happiness.\n\n"
    "That to secure these rights, Governments are instituted among Men, "
    "deriving their just powers from the consent of the governed. "
    "That whenever any Form of Government becomes destructive of these ends, "
    "it is the Right of the People to alter or to abolish it, and to institute "
    "new Government, laying its foundation on such principles and organizing "
    "its powers in such form, as to them shall seem most likely to effect "
    "their Safety and Happiness."
)


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, target_words: int = 200) -> list[str]:
    """Split text into chunks of ~target_words each.

    First tries paragraph boundaries (double newline). If the text has no
    paragraph breaks (e.g. Gutenberg prose), falls back to sentence-level
    splitting. Piper TTS quality degrades on very long inputs; ~200 words
    is the sweet spot.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # If we got a single huge paragraph, split on sentence boundaries instead
    if len(paragraphs) == 1 and len(paragraphs[0].split()) > target_words * 1.5:
        import re

        sentences = re.split(r"(?<=[.!?])\s+", paragraphs[0])
        paragraphs = sentences

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())
        if current_words + para_words > target_words and current:
            chunks.append(" ".join(current))
            current = [para]
            current_words = para_words
        else:
            current.append(para)
            current_words += para_words

    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------
def save_wav(audio: np.ndarray, sample_rate: int, path: str) -> None:
    """Save float32 audio array to WAV file (int16)."""
    audio_int16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio from orig_sr to target_sr using scipy."""
    if orig_sr == target_sr:
        return audio
    num_samples = int(len(audio) * target_sr / orig_sr)
    return resample(audio, num_samples).astype(np.float32)


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------
def load_documents(
    input_files: list[str] | None,
    texts_dir: Path,
) -> dict[str, str]:
    """Load documents from explicit file paths or the texts directory.

    Returns dict mapping document name (stem) -> full text.
    """
    docs: dict[str, str] = {}

    # Explicit --input-file paths
    if input_files:
        for fpath in input_files:
            p = Path(fpath)
            if not p.exists():
                logger.error("Input file not found: %s", fpath)
                continue
            docs[p.stem] = p.read_text(encoding="utf-8").strip()
        return docs

    # Auto-discover from tools/roundtrip_texts/
    if texts_dir.is_dir():
        for txt_file in sorted(texts_dir.glob("*.txt")):
            docs[txt_file.stem] = txt_file.read_text(encoding="utf-8").strip()

    if not docs:
        logger.warning(
            "No text files found in %s and no --input-file given. "
            "Using built-in sample. Place .txt files in %s for real tests.",
            texts_dir,
            texts_dir,
        )
        docs["sample"] = BUILTIN_SAMPLE

    return docs


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate using jiwer."""
    from jiwer import wer

    ref = reference.strip().lower()
    hyp = hypothesis.strip().lower()
    if not ref:
        return 0.0 if not hyp else 1.0
    return wer(ref, hyp)


def word_diff(reference: str, hypothesis: str) -> str:
    """Generate a word-level diff between reference and hypothesis."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    diff = difflib.unified_diff(
        ref_words,
        hyp_words,
        fromfile="original",
        tofile="roundtrip",
        lineterm="",
        n=3,
    )
    return "\n".join(diff)


def generate_report(
    results: dict[str, list[dict]],
    output_dir: Path,
    total_elapsed: float,
) -> str:
    """Generate roundtrip_report.txt content and write it."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  ROUNDTRIP QUALITY REPORT")
    lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Total elapsed: {total_elapsed:.1f}s")
    lines.append("=" * 72)

    all_stt_wers: list[float] = []
    all_roundtrip_wers: list[float] = []

    for doc_name, chunks in results.items():
        lines.append("")
        lines.append(f"  Document: {doc_name}")
        lines.append(f"  Chunks: {len(chunks)}")
        lines.append("-" * 72)

        # Per-chunk table header
        lines.append(
            f"  {'Chunk':>5}  {'STT WER':>8}  {'RT WER':>8}  "
            f"{'TTS(ms)':>8}  {'STT(ms)':>8}  {'Trans(ms)':>9}  "
            f"{'TTS2(ms)':>8}  {'STT2(ms)':>9}  {'Back(ms)':>8}"
        )
        lines.append("  " + "-" * 68)

        doc_stt_wers: list[float] = []
        doc_rt_wers: list[float] = []

        for c in chunks:
            stt_wer = c["stt_wer"]
            rt_wer = c["roundtrip_wer"]
            doc_stt_wers.append(stt_wer)
            doc_rt_wers.append(rt_wer)

            lines.append(
                f"  {c['chunk_idx']:>5}  {stt_wer:>7.1%}  {rt_wer:>7.1%}  "
                f"{c['tts_en_ms']:>8.0f}  {c['stt_en_ms']:>8.0f}  {c['trans_es_ms']:>9.0f}  "
                f"{c['tts_es_ms']:>8.0f}  {c['stt_es_ms']:>9.0f}  {c['trans_en_ms']:>8.0f}"
            )

        # Aggregate stats for this document
        lines.append("")
        if doc_stt_wers:
            mean_stt = sum(doc_stt_wers) / len(doc_stt_wers)
            mean_rt = sum(doc_rt_wers) / len(doc_rt_wers)
            sorted_stt = sorted(doc_stt_wers)
            sorted_rt = sorted(doc_rt_wers)
            median_stt = sorted_stt[len(sorted_stt) // 2]
            median_rt = sorted_rt[len(sorted_rt) // 2]
            p95_stt = sorted_stt[int(len(sorted_stt) * 0.95)] if len(sorted_stt) > 1 else sorted_stt[0]
            p95_rt = sorted_rt[int(len(sorted_rt) * 0.95)] if len(sorted_rt) > 1 else sorted_rt[0]

            lines.append(f"  STT WER     — mean: {mean_stt:.1%}  median: {median_stt:.1%}  P95: {p95_stt:.1%}")
            lines.append(f"  Roundtrip WER — mean: {mean_rt:.1%}  median: {median_rt:.1%}  P95: {p95_rt:.1%}")

        all_stt_wers.extend(doc_stt_wers)
        all_roundtrip_wers.extend(doc_rt_wers)

    # Combined stats
    lines.append("")
    lines.append("=" * 72)
    lines.append("  COMBINED (all documents)")
    lines.append("=" * 72)
    if all_stt_wers:
        mean_stt = sum(all_stt_wers) / len(all_stt_wers)
        mean_rt = sum(all_roundtrip_wers) / len(all_roundtrip_wers)
        sorted_stt = sorted(all_stt_wers)
        sorted_rt = sorted(all_roundtrip_wers)
        median_stt = sorted_stt[len(sorted_stt) // 2]
        median_rt = sorted_rt[len(sorted_rt) // 2]

        lines.append(f"  Total chunks: {len(all_stt_wers)}")
        lines.append(f"  STT WER     — mean: {mean_stt:.1%}  median: {median_stt:.1%}")
        lines.append(f"  Roundtrip WER — mean: {mean_rt:.1%}  median: {median_rt:.1%}")

    # Qualitative diffs
    lines.append("")
    lines.append("=" * 72)
    lines.append("  QUALITATIVE DIFFS (original vs back-translated)")
    lines.append("=" * 72)

    for doc_name, chunks in results.items():
        lines.append(f"\n  --- {doc_name} ---")
        for c in chunks:
            lines.append(f"\n  Chunk {c['chunk_idx']} (roundtrip WER: {c['roundtrip_wer']:.1%}):")
            lines.append(f"  ORIGINAL:      {textwrap.shorten(c['original'], 120)}")
            lines.append(f"  BACK-TRANSLATED: {textwrap.shorten(c['backtranslated'], 120)}")
            diff = word_diff(c["original"], c["backtranslated"])
            if diff:
                for dline in diff.split("\n")[:15]:
                    lines.append(f"    {dline}")

    report = "\n".join(lines)

    report_path = output_dir / "roundtrip_report.txt"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report written to %s", report_path)
    return report


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_roundtrip(
    documents: dict[str, str],
    output_dir: Path,
    max_chunks: int | None = None,
) -> dict[str, list[dict]]:
    """Run the full roundtrip pipeline on all documents."""
    from engines.factory import create_stt_engine, create_translation_engine, create_tts_engine

    # --- Engine setup ---
    logger.info("Setting up engines...")

    stt_engine = create_stt_engine(backend="mlx")
    gemma_engine = create_translation_engine(backend="mlx", engine_type="gemma")
    tts_engine = create_tts_engine(voices={"en": "en_US-lessac-high", "es": "es_MX-claude-high"})

    logger.info("Loading STT engine...")
    stt_engine.load()
    logger.info("Loading translation engine...")
    gemma_engine.load()
    logger.info("Loading TTS engine...")
    tts_engine.load()

    logger.info("All engines loaded. Starting roundtrip pipeline.")

    results: dict[str, list[dict]] = {}

    for doc_name, doc_text in documents.items():
        logger.info("Processing document: %s", doc_name)
        doc_dir = output_dir / doc_name
        doc_dir.mkdir(parents=True, exist_ok=True)

        chunks = chunk_text(doc_text)
        if max_chunks is not None:
            chunks = chunks[:max_chunks]

        logger.info("  %d chunks (max_chunks=%s)", len(chunks), max_chunks)
        chunk_results: list[dict] = []

        for i, chunk in enumerate(chunks, start=1):
            idx = f"{i:02d}"
            logger.info("  Chunk %s/%d (%d words)", idx, len(chunks), len(chunk.split()))

            result = process_chunk(
                chunk=chunk,
                chunk_idx=i,
                doc_dir=doc_dir,
                idx=idx,
                stt_engine=stt_engine,
                gemma_engine=gemma_engine,
                tts_engine=tts_engine,
            )
            chunk_results.append(result)

        results[doc_name] = chunk_results

        # Reassemble full back-translated text
        full_backtranslated = "\n\n".join(c["backtranslated"] for c in chunk_results)
        full_path = output_dir / f"{doc_name}_full_en.txt"
        full_path.write_text(full_backtranslated, encoding="utf-8")
        logger.info("  Full back-translation: %s", full_path)

    # Cleanup
    stt_engine.unload()
    gemma_engine.unload()
    tts_engine.unload()

    return results


def process_chunk(
    *,
    chunk: str,
    chunk_idx: int,
    doc_dir: Path,
    idx: str,
    stt_engine,
    gemma_engine,
    tts_engine,
) -> dict:
    """Process a single chunk through the full roundtrip pipeline.

    Steps:
      1. EN text → Piper TTS → WAV
      2. WAV → resample 22050→16000 → Whisper STT → transcribed EN
      3. Transcribed EN → TranslateGemma (en→es) → Spanish text
      4. Spanish text → Piper TTS → WAV
      5. WAV → resample → Whisper STT (language="es") → transcribed ES
      6. Transcribed ES → TranslateGemma (es→en) → back-translated EN
    """
    # Save original
    (doc_dir / f"{idx}_en_original.txt").write_text(chunk, encoding="utf-8")

    # Step 1: EN text → TTS → WAV
    logger.info("    [1/6] EN TTS...")
    tts_en = tts_engine.synthesize(chunk, language="en")
    en_wav_path = str(doc_dir / f"{idx}_en_audio.wav")
    save_wav(tts_en.audio, tts_en.sample_rate, en_wav_path)
    tts_en_ms = tts_en.latency_ms

    # Step 2: WAV → resample → STT
    logger.info("    [2/6] EN STT...")
    audio_16k = resample_audio(tts_en.audio, tts_en.sample_rate, 16000)
    stt_en = stt_engine.transcribe(audio_16k, language="en")
    en_transcribed = stt_en.text
    (doc_dir / f"{idx}_en_transcribed.txt").write_text(en_transcribed, encoding="utf-8")
    stt_en_ms = stt_en.latency_ms

    # Step 3: EN → ES translation
    logger.info("    [3/6] EN→ES translation...")
    trans_es = gemma_engine.translate(en_transcribed, source_lang="en", target_lang="es")
    es_text = trans_es.text
    (doc_dir / f"{idx}_es_translated.txt").write_text(es_text, encoding="utf-8")
    trans_es_ms = trans_es.latency_ms

    # Step 4: ES text → TTS → WAV
    logger.info("    [4/6] ES TTS...")
    tts_es = tts_engine.synthesize(es_text, language="es")
    es_wav_path = str(doc_dir / f"{idx}_es_audio.wav")
    save_wav(tts_es.audio, tts_es.sample_rate, es_wav_path)
    tts_es_ms = tts_es.latency_ms

    # Step 5: ES WAV → resample → STT
    logger.info("    [5/6] ES STT...")
    es_audio_16k = resample_audio(tts_es.audio, tts_es.sample_rate, 16000)
    stt_es = stt_engine.transcribe(es_audio_16k, language="es")
    es_transcribed = stt_es.text
    (doc_dir / f"{idx}_es_transcribed.txt").write_text(es_transcribed, encoding="utf-8")
    stt_es_ms = stt_es.latency_ms

    # Step 6: ES → EN back-translation
    logger.info("    [6/6] ES→EN back-translation...")
    trans_en = gemma_engine.translate(es_transcribed, source_lang="es", target_lang="en")
    en_backtranslated = trans_en.text
    (doc_dir / f"{idx}_en_backtranslated.txt").write_text(en_backtranslated, encoding="utf-8")
    trans_en_ms = trans_en.latency_ms

    # Compute WER
    stt_wer = compute_wer(chunk, en_transcribed)
    roundtrip_wer = compute_wer(chunk, en_backtranslated)

    logger.info(
        "    STT WER: %.1f%%  Roundtrip WER: %.1f%%  "
        "(TTS %.0fms + STT %.0fms + Trans %.0fms + TTS %.0fms + STT %.0fms + Back %.0fms)",
        stt_wer * 100,
        roundtrip_wer * 100,
        tts_en_ms,
        stt_en_ms,
        trans_es_ms,
        tts_es_ms,
        stt_es_ms,
        trans_en_ms,
    )

    return {
        "chunk_idx": chunk_idx,
        "original": chunk,
        "en_transcribed": en_transcribed,
        "es_translated": es_text,
        "es_transcribed": es_transcribed,
        "backtranslated": en_backtranslated,
        "stt_wer": stt_wer,
        "roundtrip_wer": roundtrip_wer,
        "tts_en_ms": tts_en_ms,
        "stt_en_ms": stt_en_ms,
        "trans_es_ms": trans_es_ms,
        "tts_es_ms": tts_es_ms,
        "stt_es_ms": stt_es_ms,
        "trans_en_ms": trans_en_ms,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end bilingual roundtrip quality test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Place .txt files in tools/roundtrip_texts/ or pass paths via --input-file.
            Each file becomes a separate test document with its own output subdirectory.
        """),
    )
    parser.add_argument(
        "--input-file",
        action="append",
        dest="input_files",
        help="Path to a .txt file to use as test input (repeatable)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Max chunks per document (for quick tests)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: tools/roundtrip_output)",
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    texts_dir = script_dir / "roundtrip_texts"
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "roundtrip_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load documents
    documents = load_documents(args.input_files, texts_dir)
    if not documents:
        logger.error("No documents to process. Exiting.")
        sys.exit(1)

    logger.info("Documents loaded: %s", list(documents.keys()))
    for name, text in documents.items():
        logger.info("  %s: %d words, %d chars", name, len(text.split()), len(text))

    # Run pipeline
    t_start = time.time()
    results = run_roundtrip(documents, output_dir, max_chunks=args.max_chunks)
    total_elapsed = time.time() - t_start

    # Generate report
    report = generate_report(results, output_dir, total_elapsed)
    print("\n" + report)


if __name__ == "__main__":
    main()
