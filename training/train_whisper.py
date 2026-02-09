#!/usr/bin/env python3
"""
train_whisper.py — Whisper LoRA Fine-Tuning for Church Audio

Applies LoRA to both encoder (acoustic domain shift) and decoder (vocabulary patterns)
of Distil-Whisper large-v3. Runs on Windows/WSL with A2000 Ada 16GB VRAM.

Config: r=32, alpha=64, target q_proj+v_proj, bf16, gradient checkpointing
VRAM: ~8-10 GB | Training time: ~5-8 hrs for 3 epochs on 20 hrs audio

Usage:
    python train_whisper.py
    python train_whisper.py --dataset stark_data/cleaned --output fine_tuned_whisper_mi
    python train_whisper.py --dataset stark_data/cleaned --target-modules q_proj v_proj k_proj out_proj
    python train_whisper.py --replay-ratio 0.3  # Mix 30% general-domain data
    python train_whisper.py --resume             # Resume from last checkpoint
"""

import argparse
import logging
import os

import torch
from datasets import Audio, load_dataset
from peft import LoraConfig, get_peft_model
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
            "mozilla-foundation/common_voice_16_1", "en",
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
        batch["input_features"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
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


def fine_tune_whisper(
    dataset_path="stark_data/cleaned",
    output_dir="fine_tuned_whisper_mi",
    model_name="distil-whisper/distil-large-v3",
    target_modules=None,
    lora_r=32,
    lora_alpha=64,
    batch_size=4,
    grad_accum=4,
    epochs=3,
    lr=1e-4,
    replay_ratio=0.3,
    resume_from_checkpoint=False,
):
    """LoRA fine-tuning for Whisper on church audio.

    Research-validated config (LoRA-Whisper, Interspeech 2024):
    - r=32, alpha=64 (most commonly validated)
    - Both encoder + decoder (encoder for acoustic adaptation, decoder for vocab)
    - bf16 with gradient checkpointing on Ada architecture
    - ~8-10 GB VRAM, fits comfortably on A2000 Ada
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

    # Load dataset (expects audiofolder format with audio + text columns)
    logger.info(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset("audiofolder", data_dir=dataset_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names["train"],
        num_proc=4,
    )

    # Split into train/eval if no eval split exists
    if "test" not in dataset and "validation" not in dataset:
        logger.info("No eval split found, creating 95/5 train/eval split...")
        split = dataset["train"].train_test_split(test_size=0.05, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset["train"]
        eval_dataset = dataset.get("test") or dataset.get("validation")

    # Optionally mix with general-domain data
    if replay_ratio > 0:
        logger.info(f"Mixing with {replay_ratio:.0%} general-domain replay data...")
        train_dataset = prepare_mixed_dataset(train_dataset, processor, replay_ratio)

    logger.info(f"Training: {len(train_dataset)} examples")
    if eval_dataset:
        logger.info(f"Evaluation: {len(eval_dataset)} examples")

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

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info(f"Whisper LoRA adapters saved to {output_dir}")

    # Log final metrics
    if eval_dataset:
        metrics = trainer.evaluate()
        logger.info(f"Final eval metrics: {metrics}")
        trainer.save_metrics("eval", metrics)


def main():
    parser = argparse.ArgumentParser(
        description="Whisper LoRA fine-tuning for church audio"
    )
    parser.add_argument("--dataset", "-d", default="stark_data/cleaned",
                        help="Path to audiofolder dataset")
    parser.add_argument("--output", "-o", default="fine_tuned_whisper_mi",
                        help="Output directory for LoRA adapters")
    parser.add_argument("--model", default="distil-whisper/distil-large-v3",
                        help="Base Whisper model")
    parser.add_argument("--target-modules", nargs="+",
                        default=["q_proj", "v_proj"],
                        help="LoRA target modules (extend: q_proj v_proj k_proj out_proj fc1 fc2)")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--replay-ratio", type=float, default=0.3,
                        help="Ratio of general-domain replay data (0 to disable)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint")
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
    )


if __name__ == "__main__":
    main()
