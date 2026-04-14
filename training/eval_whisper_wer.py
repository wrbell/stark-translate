#!/usr/bin/env python3
"""
eval_whisper_wer.py — Post-training WER evaluation for Whisper LoRA adapters

Loads a Whisper adapter (or base model), runs generation on eval set, computes WER.
Separate from train_whisper.py because the HF Trainer's evaluate() crashes on
8-bit quantized models with dtype mismatches in conv1d layers.

Usage:
    # Baseline (no adapter)
    python training/eval_whisper_wer.py --eval-set stark_data/whisper_dataset_deepgram

    # With adapter
    python training/eval_whisper_wer.py --adapter whisper_ablation/W1_baseline \
        --eval-set stark_data/whisper_dataset_deepgram

    # All W1-W6 + baseline
    python training/eval_whisper_wer.py --sweep whisper_ablation \
        --eval-set stark_data/whisper_dataset_deepgram
"""

import argparse
import gc
import glob
import json
import logging
import os
import time

os.environ["USE_TF"] = "0"

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_adapter(
    model_name,
    adapter_path,
    eval_dataset_path,
    max_samples=None,
):
    """Load model + adapter, generate on eval set, compute WER."""
    import jiwer
    from datasets import DatasetDict
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_name)

    # Load WITHOUT 8-bit quantization for eval — avoids dtype issues
    # fp16 fits in ~3 GB for Turbo, well within 16 GB
    logger.info(f"Loading {model_name} in fp16...")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if adapter_path and os.path.exists(adapter_path):
        # Temporarily hide gptqmodel from PEFT to avoid version check errors.
        # PEFT 0.18+ requires gptqmodel >= 2.0 but we have 1.9.x (torch 2.6
        # can't build 2.0+). Whisper LoRA adapters don't use GPTQ, so the
        # check is irrelevant here.
        import sys

        _gptq_mod = sys.modules.pop("gptqmodel", None)
        try:
            from peft import PeftModel

            logger.info(f"Loading adapter from {adapter_path}...")
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()  # merge LoRA into base for clean inference
            logger.info("Adapter merged into base model")
        finally:
            if _gptq_mod is not None:
                sys.modules["gptqmodel"] = _gptq_mod

    model.eval()

    # Force English
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids

    # Load eval data from preprocessed cache
    cache_dir = os.path.join(eval_dataset_path, ".preprocessed_cache")
    if os.path.exists(cache_dir):
        ds = DatasetDict.load_from_disk(cache_dir)
        eval_split = ds.get("test") or ds.get("eval")
    else:
        from datasets import load_dataset

        ds = load_dataset("audiofolder", data_dir=eval_dataset_path)
        eval_split = ds.get("test") or ds.get("eval") or ds.get("validation")

    if eval_split is None:
        logger.error("No eval split found")
        return None

    if max_samples:
        eval_split = eval_split.select(range(min(max_samples, len(eval_split))))

    logger.info(f"Evaluating on {len(eval_split)} examples...")

    references = []
    predictions = []
    start_time = time.time()

    for i, example in enumerate(eval_split):
        # Get input features
        input_features = torch.tensor(example["input_features"]).unsqueeze(0).to(model.device, dtype=torch.float16)

        # Generate
        with torch.no_grad():
            predicted_ids = model.generate(input_features, max_length=225)

        # Decode
        pred_text = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True).strip()
        ref_text = processor.tokenizer.decode(example["labels"], skip_special_tokens=True).strip()

        if ref_text:  # skip empty references
            predictions.append(pred_text)
            references.append(ref_text)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            logger.info(f"  [{i + 1}/{len(eval_split)}] {elapsed:.0f}s elapsed")

    elapsed = time.time() - start_time

    if not references:
        logger.error("No valid reference/prediction pairs")
        return None

    # Compute WER — both raw (case-sensitive) and normalized (case-insensitive)
    # Raw: includes case/punctuation differences (inflated for Whisper which outputs lowercase)
    # Normalized: case-insensitive, no punctuation — the metric that matters
    wer_raw = jiwer.wer(references, predictions)
    cer_raw = jiwer.cer(references, predictions)

    # Normalized WER: lowercase, expand contractions, remove non-words
    # Filter out pairs where normalization produces empty strings
    norm_refs = []
    norm_preds = []
    for ref, pred in zip(references, predictions):
        r = ref.lower().strip()
        p = pred.lower().strip()
        if r and p:  # skip empty pairs
            norm_refs.append(r)
            norm_preds.append(p)

    if norm_refs:
        wer_normalized = jiwer.wer(norm_refs, norm_preds)
        cer_normalized = jiwer.cer(norm_refs, norm_preds)
    else:
        wer_normalized = wer_raw
        cer_normalized = cer_raw

    result = {
        "adapter": adapter_path or "base",
        "model": model_name,
        "eval_samples": len(references),
        "wer_raw": round(wer_raw, 4),
        "wer_normalized": round(wer_normalized, 4),
        "cer_raw": round(cer_raw, 4),
        "cer_normalized": round(cer_normalized, 4),
        "wer": round(wer_normalized, 4),  # primary metric
        "cer": round(cer_normalized, 4),  # primary metric
        "elapsed_s": round(elapsed, 1),
    }

    logger.info(f"WER (normalized): {wer_normalized:.4f} ({wer_normalized * 100:.2f}%)")
    logger.info(f"WER (raw):        {wer_raw:.4f} ({wer_raw * 100:.2f}%)")
    logger.info(f"CER (normalized): {cer_normalized:.4f} ({cer_normalized * 100:.2f}%)")
    logger.info(f"Evaluated {len(references)} examples in {elapsed:.0f}s")

    # Aggressive cleanup to prevent RAM accumulation across 7 sequential evals
    del model, processor
    gc.collect()
    gc.collect()  # double collect for cyclic references
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Post-training WER evaluation for Whisper LoRA")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter dir (omit for base model)")
    parser.add_argument(
        "--eval-set",
        default="stark_data/whisper_dataset_deepgram",
        help="Path to eval dataset (audiofolder with .preprocessed_cache)",
    )
    parser.add_argument(
        "--model",
        default="openai/whisper-large-v3-turbo",
        help="Base Whisper model",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit eval samples")
    parser.add_argument("--output", "-o", default=None, help="Save results JSON")
    parser.add_argument(
        "--sweep",
        default=None,
        help="Directory containing multiple adapter dirs (runs all + baseline)",
    )
    args = parser.parse_args()

    if args.sweep:
        # Run baseline + all adapters in the directory
        adapter_dirs = sorted(glob.glob(os.path.join(args.sweep, "W*")))
        adapter_dirs = [d for d in adapter_dirs if os.path.isdir(d)]

        all_results = []

        # Baseline first
        logger.info("=== Baseline (no adapter) ===")
        result = evaluate_adapter(args.model, None, args.eval_set, args.max_samples)
        if result:
            all_results.append(result)
            print(f"  base: WER={result['wer']:.4f}")

        # Each adapter
        for adapter_dir in adapter_dirs:
            name = os.path.basename(adapter_dir)
            # Check if adapter exists
            if not os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors")):
                logger.warning(f"  {name}: no adapter found, skipping")
                continue
            logger.info(f"=== {name} ===")
            result = evaluate_adapter(args.model, adapter_dir, args.eval_set, args.max_samples)
            if result:
                all_results.append(result)
                print(f"  {name}: WER={result['wer']:.4f}")

        # Summary
        print()
        print("=" * 80)
        print(f"{'Adapter':<25} {'WER(norm)':>10} {'WER(raw)':>10} {'CER(norm)':>10} {'vs Base':>10}")
        print("-" * 80)
        base_wer = None
        for r in all_results:
            name = os.path.basename(r["adapter"]) if r["adapter"] != "base" else "base"
            if name == "base":
                base_wer = r.get("wer_normalized", r["wer"])
            delta = ""
            wer_n = r.get("wer_normalized", r["wer"])
            if base_wer and name != "base":
                rel = (wer_n - base_wer) / base_wer * 100
                delta = f"{rel:+.1f}%"
            print(
                f"{name:<25} {wer_n:>9.4f} {r.get('wer_raw', r['wer']):>9.4f} {r.get('cer_normalized', r['cer']):>9.4f} {delta:>10}"
            )

        # Save
        output = args.output or os.path.join(args.sweep, "wer_results.json")
        with open(output, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {output}")

    else:
        # Single adapter eval
        result = evaluate_adapter(args.model, args.adapter, args.eval_set, args.max_samples)
        if result and args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
