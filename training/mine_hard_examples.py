#!/usr/bin/env python3
"""
mine_hard_examples.py — Mine hard examples from a Whisper LoRA adapter.

Runs batched fp16 inference over a pool of aligned chunks, computes per-chunk
WER against Deepgram ground truth, and writes results to JSONL for downstream
filtering by build_hard_subset.py.

Supports resume: if the output JSONL already exists, processed chunks are
skipped on restart.

Usage:
    python training/mine_hard_examples.py \
        --adapter whisper_ablation/W14_combined50k_3ep \
        --chunks-json ablation/sermon_whisper_chunks_w14_combined.json \
        --deepgram-dir /mnt/d/Data/stt-data/deepgram_transcripts \
        --audio-dir stt-data \
        --output whisper_ablation/w14_mined.jsonl

    # Resume after crash
    python training/mine_hard_examples.py --resume [same args]
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

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

# Hard memory cap (GB) — prevents OOM-kill on WSL2 (48 GB limit).
# Set high enough for PyTorch virtual memory + HF cache mmap pages.
_HARD_MEM_CAP_GB = 40


def _get_rss_gb():
    """Return current process RSS in GB, or None if psutil unavailable."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e9
    except ImportError:
        return None


def find_tier1_terms(text: str, terms: list[str]) -> list[str]:
    """Find Tier 1 theological terms present in text (case-insensitive).

    Returns the list of matching terms in their original casing from the
    glossary, not from the input text.
    """
    text_lower = text.lower()
    return [t for t in terms if t.lower() in text_lower]


def load_resume_set(output_path: Path) -> set[tuple[str, int]]:
    """Load already-processed (source, chunk_idx) pairs from existing JSONL.

    Returns an empty set if the file does not exist or is empty.
    """
    processed = set()
    if not output_path.exists():
        return processed
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                processed.add((record["source"], record["chunk_idx"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return processed


def mine(args):
    """Run batched inference over all chunks and write per-chunk WER to JSONL."""
    import soundfile as sf
    import torch

    os.environ["USE_TF"] = "0"
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from training.align_deepgram_chunks import (
        ALIGNMENT_TOLERANCE,
        find_audio_file,
        find_deepgram_words_for_chunk,
        group_chunks_by_source,
        load_deepgram_transcript,
    )

    # Note: no RLIMIT_AS cap here. PyTorch + HuggingFace cache use >40GB virtual
    # address space (mmap pages, CUDA allocator). The mining script is inference-only
    # with bounded real memory (~5GB RSS). RSS is monitored via _get_rss_gb().

    # Load model + adapter
    logger.info("Loading %s in fp16...", args.model)
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if args.adapter and os.path.exists(args.adapter):
        logger.info("Loading adapter from %s...", args.adapter)
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        logger.info("Adapter merged into base model")

    model.eval()
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids

    # Load Tier 1 glossary
    tier1_terms = []
    if args.tier1_glossary and os.path.exists(args.tier1_glossary):
        with open(args.tier1_glossary) as f:
            tier1_terms = json.load(f)
        logger.info("Loaded %d Tier 1 terms", len(tier1_terms))

    # Load chunks and group by source
    logger.info("Loading chunks from %s", args.chunks_json)
    with open(args.chunks_json) as f:
        chunks = json.load(f)
    grouped = group_chunks_by_source(chunks)
    total_chunks = sum(len(v) for v in grouped.values())
    logger.info("Loaded %d chunks across %d sources", total_chunks, len(grouped))

    # Resume support
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resume_set = load_resume_set(output_path) if args.resume else set()
    if resume_set:
        logger.info("Resume: skipping %d already-processed chunks", len(resume_set))

    # Mine
    processed = 0
    skipped = 0
    start_time = time.time()

    with open(output_path, "a") as out_f:
        for source, source_chunks in sorted(grouped.items()):
            # Load Deepgram transcript
            dg = load_deepgram_transcript(Path(args.deepgram_dir), source)
            if dg is None:
                continue
            dg_words = dg.get("words", [])
            if not dg_words:
                continue

            # Find audio file
            audio_path = find_audio_file(Path(args.audio_dir), source)
            if audio_path is None:
                continue

            info = sf.info(str(audio_path))
            sr = info.samplerate

            # Accumulate batch
            batch_mels = []
            batch_meta = []  # (source, chunk_idx, start, end, reference)

            for chunk_idx, chunk in enumerate(source_chunks):
                if (source, chunk_idx) in resume_set:
                    skipped += 1
                    continue

                chunk_start = chunk.get("start", 0.0)
                chunk_end = chunk.get("end", 0.0)

                # Get Deepgram reference text
                matched_words = find_deepgram_words_for_chunk(dg_words, chunk_start, chunk_end, ALIGNMENT_TOLERANCE)
                reference = " ".join(w.get("word", "") for w in matched_words).strip()
                if not reference:
                    continue

                # Read audio segment
                try:
                    start_frame = int(chunk_start * sr)
                    end_frame = int(chunk_end * sr)
                    audio_data, _ = sf.read(str(audio_path), start=start_frame, stop=end_frame, dtype="float32")
                    if sr != 16000:
                        import librosa

                        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

                    mel = processor(audio_data, sampling_rate=16000).input_features[0]
                    batch_mels.append(mel)
                    batch_meta.append((source, chunk_idx, chunk_start, chunk_end, reference))
                except Exception as e:
                    logger.warning("Failed to read %s chunk %d: %s", source, chunk_idx, e)
                    continue

                # Process batch when full
                if len(batch_mels) >= args.batch_size:
                    _process_batch(batch_mels, batch_meta, model, processor, tier1_terms, out_f)
                    processed += len(batch_mels)
                    batch_mels = []
                    batch_meta = []

            # Flush remaining batch for this source
            if batch_mels:
                _process_batch(batch_mels, batch_meta, model, processor, tier1_terms, out_f)
                processed += len(batch_mels)
                batch_mels = []
                batch_meta = []

            # Progress logging
            if processed > 0 and processed % 1000 < args.batch_size:
                elapsed = time.time() - start_time
                rss = _get_rss_gb()
                rss_str = f" | RSS: {rss:.1f} GB" if rss is not None else ""
                logger.info(
                    "  Processed %d chunks (%.1f/s, skipped %d resume)%s",
                    processed,
                    processed / elapsed,
                    skipped,
                    rss_str,
                )

    elapsed = time.time() - start_time
    logger.info(
        "Mining complete: %d chunks processed, %d skipped (resume), %.0fs elapsed",
        processed,
        skipped,
        elapsed,
    )


def _process_batch(batch_mels, batch_meta, model, processor, tier1_terms, out_f):
    """Run inference on a batch of mel spectrograms and write results to JSONL."""
    import jiwer
    import torch

    # Pad and stack
    input_features = [{"input_features": mel} for mel in batch_mels]
    batch = processor.feature_extractor.pad(input_features, return_tensors="pt")
    batch_tensor = batch["input_features"].to(model.device, dtype=torch.float16)

    # Generate
    with torch.no_grad():
        predicted_ids = model.generate(batch_tensor, max_length=225)

    # Decode and compute WER per item
    for i, (source, chunk_idx, start, end, reference) in enumerate(batch_meta):
        prediction = processor.tokenizer.decode(predicted_ids[i], skip_special_tokens=True).strip()

        # Normalized WER/CER
        ref_norm = reference.lower().strip()
        pred_norm = prediction.lower().strip()
        wer = jiwer.wer(ref_norm, pred_norm) if ref_norm else 1.0
        cer = jiwer.cer(ref_norm, pred_norm) if ref_norm else 1.0

        # Tier 1 term detection
        tier1_matches = find_tier1_terms(reference, tier1_terms)

        record = {
            "source": source,
            "chunk_idx": chunk_idx,
            "start": round(start, 2),
            "end": round(end, 2),
            "reference": reference,
            "prediction": prediction,
            "wer": round(wer, 4),
            "cer": round(cer, 4),
            "has_tier1": bool(tier1_matches),
            "tier1_terms": tier1_matches,
        }
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Flush every batch
    out_f.flush()

    # Free GPU memory
    del batch_tensor, predicted_ids
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Mine hard examples from a Whisper LoRA adapter by computing per-chunk WER."
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter directory (default: base model, no adapter)",
    )
    parser.add_argument(
        "--chunks-json",
        type=str,
        required=True,
        help="Path to whisper chunks JSON (e.g., ablation/sermon_whisper_chunks_w14_combined.json)",
    )
    parser.add_argument(
        "--deepgram-dir",
        type=str,
        required=True,
        help="Directory containing Deepgram transcript JSONs",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        required=True,
        help="Directory containing source WAV files (searched recursively)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path for mined results",
    )
    parser.add_argument(
        "--tier1-glossary",
        type=str,
        default="bible_data/glossary/tier1_boost.json",
        help="Path to Tier 1 theological terms JSON",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of chunks per inference batch (default: 8)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="Base Whisper model name",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip chunks already present in the output JSONL",
    )
    args = parser.parse_args()
    mine(args)


if __name__ == "__main__":
    main()
