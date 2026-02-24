#!/usr/bin/env python3
"""
train_whisper.py — Whisper LoRA Fine-Tuning for Church Audio

Applies LoRA to both encoder (acoustic domain shift) and decoder (vocabulary patterns)
of Whisper Large-V3-Turbo (openai/whisper-large-v3-turbo). ~809M params (32→4 decoder
layers), MIT license. Faster training than full large-v3 due to smaller decoder.
Runs on Windows/WSL with A2000 Ada 16GB VRAM.

Config: r=32, alpha=64, target q_proj+v_proj, bf16, gradient checkpointing
VRAM: ~6-8 GB (Turbo) / ~8-10 GB (Distil/large-v3)
Expected re-tune time: ~4-8 hrs on A2000 for 3 epochs on 20 hrs audio

Usage:
    python train_whisper.py
    python train_whisper.py --dataset stark_data/cleaned --output fine_tuned_whisper_mi
    python train_whisper.py --dataset stark_data/cleaned --target-modules q_proj v_proj k_proj out_proj
    python train_whisper.py --replay-ratio 0.3  # Mix 30% general-domain data
    python train_whisper.py --resume             # Resume from last checkpoint
"""

import argparse
import logging
from collections import Counter

import numpy as np
import torch
from datasets import Audio, load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import WeightedRandomSampler
from transformers import (
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def prepare_mixed_dataset(church_dataset, processor, replay_ratio=0.3):
    """Mix general-domain replay samples with church data to prevent forgetting.

    Anti-forgetting safeguard: mix 70% general-domain + 30% church-domain samples.
    LoRA inherently preserves base weights, but replay buffer provides extra safety.
    """
    if replay_ratio <= 0:
        return church_dataset

    from datasets import concatenate_datasets

    # Load general English ASR as replay buffer
    logger.info(f"Loading general-domain replay data (ratio={replay_ratio})...")
    try:
        general = load_dataset(
            "mozilla-foundation/common_voice_16_1",
            "en",
            split="train[:2000]",
            trust_remote_code=True,
        )
        general = general.cast_column("audio", Audio(sampling_rate=16000))
    except Exception as e:
        logger.warning(f"Could not load replay data: {e}")
        logger.warning("Proceeding without replay buffer.")
        return church_dataset

    # Process the general dataset with the same preprocessing
    def prepare_general(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        # Common Voice uses "sentence" column
        text = batch.get("text") or batch.get("sentence", "")
        batch["labels"] = processor.tokenizer(text).input_ids
        return batch

    general = general.map(
        prepare_general,
        remove_columns=general.column_names,
        num_proc=4,
    )

    # Calculate mix sizes
    church_size = len(church_dataset)
    replay_size = int(church_size * replay_ratio / (1 - replay_ratio))
    replay_subset = general.select(range(min(replay_size, len(general))))

    logger.info(f"  Church: {church_size}, Replay: {len(replay_subset)}")
    mixed = concatenate_datasets([church_dataset, replay_subset])
    return mixed.shuffle(seed=42)


def make_compute_metrics(processor, eval_accent_labels):
    """Create a compute_metrics function that reports per-accent WER.

    Args:
        processor: WhisperProcessor for decoding predictions.
        eval_accent_labels: list of accent strings aligned with eval dataset indices.
    """
    import jiwer

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 padding with pad token id for decoding
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Overall WER
        wer_overall = jiwer.wer(label_str, pred_str)
        metrics = {"wer_overall": wer_overall}

        # Per-accent WER
        if eval_accent_labels and len(eval_accent_labels) == len(pred_str):
            accent_preds = {}
            accent_labels = {}
            for accent, p, l in zip(eval_accent_labels, pred_str, label_str):
                accent_preds.setdefault(accent, []).append(p)
                accent_labels.setdefault(accent, []).append(l)

            accent_wers = {}
            for accent in sorted(accent_preds.keys()):
                accent_wer = jiwer.wer(accent_labels[accent], accent_preds[accent])
                metrics[f"wer_{accent}"] = accent_wer
                accent_wers[accent] = accent_wer

            # Accent WER gap (fairness metric)
            if len(accent_wers) > 1:
                metrics["wer_accent_gap"] = max(accent_wers.values()) - min(accent_wers.values())

        return metrics

    return compute_metrics


class AccentBalancedTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with WeightedRandomSampler for accent-balanced batches."""

    def __init__(self, *args, accent_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._accent_sampler = accent_sampler

    def _get_train_sampler(self):
        if self._accent_sampler is not None:
            return self._accent_sampler
        return super()._get_train_sampler()


def build_accent_sampler(accent_labels):
    """Build a WeightedRandomSampler that balances accent representation per batch.

    Each sample's weight is inversely proportional to its accent's frequency,
    so minority accents are sampled more often.
    """
    counts = Counter(accent_labels)
    total = len(accent_labels)
    # Weight = total / (num_accents * count_for_this_accent)
    num_accents = len(counts)
    weights = [total / (num_accents * counts[accent]) for accent in accent_labels]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=total,
        replacement=True,
    )


def fine_tune_whisper(
    dataset_path="stark_data/cleaned",
    output_dir="fine_tuned_whisper_mi",
    model_name="openai/whisper-large-v3-turbo",
    target_modules=None,
    lora_r=32,
    lora_alpha=64,
    batch_size=4,
    grad_accum=4,
    epochs=3,
    lr=1e-4,
    replay_ratio=0.3,
    resume_from_checkpoint=False,
    accent_balance=True,
):
    """LoRA fine-tuning for Whisper on church audio.

    Research-validated config (LoRA-Whisper, Interspeech 2024):
    - r=32, alpha=64 (most commonly validated)
    - Both encoder + decoder (encoder for acoustic adaptation, decoder for vocab)
    - bf16 with gradient checkpointing on Ada architecture
    - ~6-8 GB VRAM (Turbo) / ~8-10 GB (Distil/large-v3), fits comfortably on A2000 Ada
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    logger.info(f"Loading {model_name}...")
    processor = WhisperProcessor.from_pretrained(model_name)

    # 8-bit quantization for the base model (LoRA trains in bf16 on top)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )

    # LoRA config — encoder + decoder
    # S2-LoRA paper: v_proj and out_proj capture the most important adaptations
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Turbo has only 4 decoder layers (vs 32 in large-v3), so LoRA adapts faster
    # but the decoder has less capacity — monitor theological term accuracy closely
    if "turbo" in model_name.lower():
        logger.info("Turbo model detected — 4 decoder layers, faster training expected")
        logger.info("Monitor theological term accuracy (lighter decoder = less vocabulary capacity)")

    # Load dataset (expects audiofolder format with audio + transcription columns)
    logger.info(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset("audiofolder", data_dir=dataset_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Detect text column name (audiofolder uses "transcription", older format uses "text")
    sample_cols = dataset.column_names["train"]
    text_col = "transcription" if "transcription" in sample_cols else "text"
    has_accent = "accent" in sample_cols
    if has_accent:
        logger.info("Accent labels detected in dataset")

    # Extract accent labels before processing (needed for sampler + eval)
    train_accent_labels = None
    eval_accent_labels = None
    if has_accent:
        train_accent_labels = dataset["train"]["accent"]

    # Determine columns to remove (keep accent out of model inputs but save it first)
    remove_cols = [c for c in sample_cols if c not in ("input_features", "labels")]

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = processor.tokenizer(batch[text_col]).input_ids
        return batch

    dataset = dataset.map(
        prepare_dataset,
        remove_columns=remove_cols,
        num_proc=4,
    )

    # Split into train/eval if no eval split exists
    if "test" not in dataset and "validation" not in dataset:
        logger.info("No eval split found, creating 95/5 train/eval split...")
        split = dataset["train"].train_test_split(test_size=0.05, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        # Split accent labels correspondingly
        if train_accent_labels:
            all_labels = np.array(train_accent_labels)
            train_indices = split["train"]._indices
            test_indices = split["test"]._indices
            if train_indices is not None:
                train_accent_labels = all_labels[train_indices.column(0).to_pylist()].tolist()
                eval_accent_labels = all_labels[test_indices.column(0).to_pylist()].tolist()
            else:
                # Fallback: indices not available, skip per-accent eval
                train_accent_labels = None
                eval_accent_labels = None
    else:
        train_dataset = dataset["train"]
        eval_dataset = dataset.get("test") or dataset.get("validation")
        # For pre-split datasets with accent column, extract eval accent labels
        if has_accent and eval_dataset is not None:
            eval_split_name = "test" if "test" in dataset else "validation"
            # Re-load just to get accent labels before columns were removed
            raw_dataset = load_dataset("audiofolder", data_dir=dataset_path)
            if eval_split_name in raw_dataset:
                eval_accent_labels = raw_dataset[eval_split_name]["accent"]

    # Optionally mix with general-domain data
    if replay_ratio > 0:
        logger.info(f"Mixing with {replay_ratio:.0%} general-domain replay data...")
        pre_mix_size = len(train_dataset)
        train_dataset = prepare_mixed_dataset(train_dataset, processor, replay_ratio)
        # Extend accent labels for replay samples (tagged as "general")
        if train_accent_labels:
            extra = len(train_dataset) - pre_mix_size
            train_accent_labels = train_accent_labels + ["general"] * extra

    logger.info(f"Training: {len(train_dataset)} examples")
    if eval_dataset:
        logger.info(f"Evaluation: {len(eval_dataset)} examples")

    # Log accent distribution
    if train_accent_labels:
        accent_counts = Counter(train_accent_labels)
        logger.info("Training accent distribution:")
        for accent, count in sorted(accent_counts.items()):
            logger.info(f"  {accent}: {count}")

    # Build accent-balanced sampler
    accent_sampler = None
    if accent_balance and train_accent_labels and len(set(train_accent_labels)) > 1:
        logger.info("Building accent-balanced sampler...")
        accent_sampler = build_accent_sampler(train_accent_labels)

    # Force English language to prevent Scottish→Welsh misclassification
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids

    # Training args optimized for A2000 Ada 16GB
    # bf16 preferred over fp16 on Ada architecture (compute capability 8.9)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        num_train_epochs=epochs,
        learning_rate=lr,
        bf16=True,
        optim="adamw_bnb_8bit",
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=500 if eval_dataset else None,
        predict_with_generate=True,
        generation_max_length=225,
        warmup_steps=500,
        remove_unused_columns=False,
        report_to="none",
        load_best_model_at_end=True if eval_dataset else False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        model=model,
    )

    # Build compute_metrics with per-accent WER
    metrics_fn = None
    if eval_dataset and eval_accent_labels:
        metrics_fn = make_compute_metrics(processor, eval_accent_labels)
    elif eval_dataset:
        # Basic WER without per-accent breakdown
        metrics_fn = make_compute_metrics(processor, [])

    trainer = AccentBalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
        compute_metrics=metrics_fn,
        accent_sampler=accent_sampler,
    )

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info(f"Whisper LoRA adapters saved to {output_dir}")

    # Log final metrics
    # TODO: Use --eval-chunked flag to test chunked vs sequential Turbo inference modes
    # Chunked mode is faster but may introduce stitching artifacts at chunk boundaries
    if eval_dataset:
        metrics = trainer.evaluate()
        logger.info(f"Final eval metrics: {metrics}")
        trainer.save_metrics("eval", metrics)


def main():
    parser = argparse.ArgumentParser(description="Whisper LoRA fine-tuning for church audio")
    parser.add_argument("--dataset", "-d", default="stark_data/cleaned", help="Path to audiofolder dataset")
    parser.add_argument("--output", "-o", default="fine_tuned_whisper_mi", help="Output directory for LoRA adapters")
    parser.add_argument(
        "--model",
        default="openai/whisper-large-v3-turbo",
        help="Base Whisper model (default: large-v3-turbo, alt: distil-whisper/distil-large-v3)",
    )
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "v_proj"],
        help="LoRA target modules (extend: q_proj v_proj k_proj out_proj fc1 fc2)",
    )
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--replay-ratio", type=float, default=0.3, help="Ratio of general-domain replay data (0 to disable)"
    )
    parser.add_argument(
        "--accent-balance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable accent-balanced sampling (default: True)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    parser.add_argument(
        "--eval-chunked",
        action="store_true",
        help="Use chunked inference during evaluation (faster but may have stitching artifacts)",
    )
    args = parser.parse_args()

    # Resolve resume_from_checkpoint: True means auto-detect last checkpoint
    resume = True if args.resume else None

    fine_tune_whisper(
        dataset_path=args.dataset,
        output_dir=args.output,
        model_name=args.model,
        target_modules=args.target_modules,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        lr=args.lr,
        replay_ratio=args.replay_ratio,
        resume_from_checkpoint=resume,
        accent_balance=args.accent_balance,
    )


if __name__ == "__main__":
    main()
