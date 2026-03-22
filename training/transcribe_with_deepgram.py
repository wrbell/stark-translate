#!/usr/bin/env python3
"""
transcribe_with_deepgram.py — Deepgram Nova-3 Oracle Transcription

Generates high-quality ground-truth transcription labels using Deepgram Nova-3
with Tier 1 theological keyterm boosting. Output format matches transcribe_church.py
for seamless pipeline integration.

Uses async API with concurrency limiting for rate limit safety.

Usage:
    python training/transcribe_with_deepgram.py --input stark_data/raw/midwest
    python training/transcribe_with_deepgram.py --input stark_data/cleaned/chunks --resume
    python training/transcribe_with_deepgram.py --input stark_data/raw/midwest --lang es
"""

import argparse
import asyncio
import glob
import json
import logging
import os
import sys
import time

# Ensure project root is on path for tools/settings imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def transcribe_file(
    audio_path: str,
    client,
    options,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Transcribe a single audio file via Deepgram async API."""
    async with semaphore:
        try:
            with open(audio_path, "rb") as f:
                buffer = f.read()

            source = {"buffer": buffer, "mimetype": "audio/wav"}
            # Large sermon files (40-160 MB) need generous timeouts
            import httpx

            timeout = httpx.Timeout(300.0, connect=30.0)
            response = await client.listen.asyncrest.v("1").transcribe_file(source, options, timeout=timeout)

            # Extract transcript and segments
            channel = response.results.channels[0]
            alt = channel.alternatives[0]
            transcript = alt.transcript

            # Build segments from utterances or words
            segments = []
            if hasattr(response.results, "utterances") and response.results.utterances:
                for utt in response.results.utterances:
                    segments.append(
                        {
                            "text": utt.transcript,
                            "start": utt.start,
                            "end": utt.end,
                            "confidence": utt.confidence,
                        }
                    )
            elif alt.words:
                # Group words into segments by sentence boundaries
                current_seg = {"words": [], "start": 0, "end": 0}
                for w in alt.words:
                    if not current_seg["words"]:
                        current_seg["start"] = w.start
                    current_seg["words"].append(w)
                    current_seg["end"] = w.end
                    # Split on punctuated ends
                    if w.punctuated_word and w.punctuated_word.rstrip().endswith((".", "?", "!")):
                        text = " ".join(getattr(ww, "punctuated_word", ww.word) for ww in current_seg["words"])
                        avg_conf = sum(ww.confidence for ww in current_seg["words"]) / len(current_seg["words"])
                        segments.append(
                            {
                                "text": text.strip(),
                                "start": current_seg["start"],
                                "end": current_seg["end"],
                                "confidence": round(avg_conf, 4),
                            }
                        )
                        current_seg = {"words": [], "start": 0, "end": 0}
                # Flush remaining
                if current_seg["words"]:
                    text = " ".join(getattr(ww, "punctuated_word", ww.word) for ww in current_seg["words"])
                    avg_conf = sum(ww.confidence for ww in current_seg["words"]) / len(current_seg["words"])
                    segments.append(
                        {
                            "text": text.strip(),
                            "start": current_seg["start"],
                            "end": current_seg["end"],
                            "confidence": round(avg_conf, 4),
                        }
                    )

            # Word-level data
            words = []
            if alt.words:
                words = [
                    {
                        "word": getattr(w, "punctuated_word", w.word),
                        "start": w.start,
                        "end": w.end,
                        "confidence": w.confidence,
                        "speaker": getattr(w, "speaker", None),
                    }
                    for w in alt.words
                ]

            duration = response.metadata.duration if hasattr(response, "metadata") and response.metadata else 0

            return {
                "audio_path": audio_path,
                "text": transcript,
                "segments": segments,
                "words": words,
                "duration": duration,
                "metadata": {
                    "source": "deepgram-nova-3",
                    "model": options.model,
                    "language": options.language,
                    "keyterms_used": len(options.keyterm) if options.keyterm else 0,
                },
            }

        except Exception as e:
            logger.error(f"Failed to transcribe {os.path.basename(audio_path)}: {e}")
            return None


async def transcribe_deepgram_async(
    input_dir: str,
    output_dir: str,
    resume: bool = False,
    lang: str = "en",
    max_concurrent: int = 5,
    api_key: str | None = None,
    boost_path: str | None = None,
) -> None:
    """Async batch transcription with Deepgram Nova-3."""
    from deepgram import DeepgramClient, PrerecordedOptions

    from tools.glossary import load_boost_keyterms

    os.makedirs(output_dir, exist_ok=True)

    # Resolve API key
    if not api_key:
        api_key = os.environ.get("STARK_DEEPGRAM__API_KEY", "")
    if not api_key:
        try:
            from settings import settings

            api_key = settings.deepgram.api_key
        except Exception:
            pass
    if not api_key:
        logger.error("No Deepgram API key. Set STARK_DEEPGRAM__API_KEY or pass --api-key.")
        return

    client = DeepgramClient(api_key)

    # Load Tier 1 keyterms
    keyterms = load_boost_keyterms(boost_path)
    if keyterms:
        logger.info(f"Loaded {len(keyterms)} Tier 1 keyterms for boosting")
    else:
        logger.warning("No Tier 1 keyterms loaded. Transcribing without keyterm boosting.")

    options = PrerecordedOptions(
        model="nova-3",
        language=lang,
        smart_format=True,
        punctuate=True,
        utterances=True,
        keyterm=keyterms if keyterms else None,
    )

    # Find audio files
    audio_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True))
    if not audio_files:
        logger.warning(f"No WAV files found in {input_dir}")
        return

    # Resume support
    if resume:
        existing = set()
        for f in glob.glob(os.path.join(output_dir, "*.deepgram.json")):
            stem = os.path.basename(f).replace(".deepgram.json", ".wav")
            existing.add(stem)
        before = len(audio_files)
        audio_files = [f for f in audio_files if os.path.basename(f) not in existing]
        if before != len(audio_files):
            logger.info(f"Resume: skipping {before - len(audio_files)} already-transcribed files")

    if not audio_files:
        logger.info("All files already transcribed.")
        return

    logger.info(f"Transcribing {len(audio_files)} files with Deepgram Nova-3...")

    semaphore = asyncio.Semaphore(max_concurrent)
    all_transcripts = []
    start_time = time.time()

    # Process sequentially to respect rate limits and provide progress
    for i, audio_file in enumerate(audio_files):
        result = await transcribe_file(audio_file, client, options, semaphore)

        if result:
            # Save individual transcript
            out_path = os.path.join(
                output_dir,
                os.path.basename(audio_file).replace(".wav", ".deepgram.json"),
            )
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            all_transcripts.append(result)

        if (i + 1) % 5 == 0 or i == len(audio_files) - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60
            logger.info(
                f"  [{i + 1}/{len(audio_files)}] {rate:.1f} files/min | "
                f"Last: {len(result['segments']) if result else 0} segments"
            )

    elapsed = time.time() - start_time
    total_chars = sum(len(t["text"]) for t in all_transcripts)
    total_segs = sum(len(t["segments"]) for t in all_transcripts)
    logger.info(f"Done in {elapsed:.0f}s: {len(all_transcripts)} files, {total_segs} segments, {total_chars:,} chars")

    # Save summary
    summary_path = os.path.join(output_dir, "_deepgram_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "source": "deepgram-nova-3",
                "language": lang,
                "keyterms_used": len(keyterms),
                "files": len(all_transcripts),
                "segments": total_segs,
                "chars": total_chars,
                "elapsed_s": round(elapsed, 1),
            },
            f,
            indent=2,
        )


def transcribe_deepgram(
    input_dir: str,
    output_dir: str,
    resume: bool = False,
    lang: str = "en",
    api_key: str | None = None,
) -> None:
    """Synchronous wrapper for async Deepgram transcription."""
    asyncio.run(
        transcribe_deepgram_async(
            input_dir=input_dir,
            output_dir=output_dir,
            resume=resume,
            lang=lang,
            api_key=api_key,
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with Deepgram Nova-3 (oracle ground truth)")
    parser.add_argument(
        "--input",
        "-i",
        default="stark_data/cleaned/chunks",
        help="Directory containing WAV files to transcribe",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="stark_data/transcripts",
        help="Output directory for .deepgram.json transcripts",
    )
    parser.add_argument(
        "--lang",
        default="en",
        choices=["en", "es"],
        help="Transcription language (default: en)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Deepgram API key (or set STARK_DEEPGRAM__API_KEY env var)",
    )
    parser.add_argument(
        "--boost-terms",
        default=None,
        help="Path to Tier 1 boost terms JSON (default: bible_data/glossary/tier1_boost.json)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Max concurrent API requests (default: 5)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files that already have .deepgram.json transcripts",
    )
    args = parser.parse_args()

    asyncio.run(
        transcribe_deepgram_async(
            input_dir=args.input,
            output_dir=args.output,
            resume=args.resume,
            lang=args.lang,
            max_concurrent=args.max_concurrent,
            api_key=args.api_key,
            boost_path=args.boost_terms,
        )
    )


if __name__ == "__main__":
    main()
