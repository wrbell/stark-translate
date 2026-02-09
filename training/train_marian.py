#!/usr/bin/env python3
"""
train_marian.py — MarianMT Full Fine-Tune Fallback

If TranslateGemma QLoRA proves too heavy or bitsandbytes causes issues,
Helsinki-NLP/opus-mt-en-es (298MB) is small enough for full fine-tuning.
Much faster iteration, lower quality ceiling.

Usage:
    python train_marian.py
    python train_marian.py --bible-data bible_data/aligned/verse_pairs_train.jsonl
    python train_marian.py --epochs 5 --lr 3e-5 --batch-size 64
    python train_marian.py --resume  # Resume from last checkpoint
"""

import argparse
import logging
import os
import sys

from datasets import load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def fine_tune_marian(
    bible_data="bible_data/aligned/verse_pairs_train.jsonl",
    test_data="bible_data/holdout/verse_pairs_test.jsonl",
    output_dir="fine_tuned_marian_mi",
    model_name="Helsinki-NLP/opus-mt-en-es",
    batch_size=32,
    grad_accum=1,
    epochs=5,
    lr=5e-5,
    resume_from_checkpoint=None,
):
    """Full fine-tune MarianMT on biblical text.

    Only 298MB -- no LoRA needed. Big batches, fast epochs.
    Lower quality ceiling than TranslateGemma but dramatically higher iteration speed.
    Trained partly on OPUS Bible corpus so it has some baseline theological coverage.
    """
    if not os.path.exists(bible_data):
        logger.error(f"Training data not found: {bible_data}")
        logger.error("Run prepare_bible_corpus.py first.")
        sys.exit(1)

    logger.info(f"Loading {model_name}...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    logger.info(f"Loading data from {bible_data}...")
    train_dataset = load_dataset("json", data_files=bible_data, split="train")
    logger.info(f"  Train: {len(train_dataset)} examples")

    # Load eval set if available
    eval_dataset = None
    if os.path.exists(test_data):
        eval_dataset = load_dataset("json", data_files=test_data, split="train")
        # Limit eval size for speed during training
        if len(eval_dataset) > 1000:
            eval_dataset = eval_dataset.select(range(1000))
        logger.info(f"  Eval: {len(eval_dataset)} examples")

    def tokenize(example):
        inputs = tokenizer(
            example["en"],
            truncation=True,
            max_length=128,
            padding=False,
        )
        targets = tokenizer(
            text_target=example["es"],
            truncation=True,
            max_length=128,
            padding=False,
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        tokenize,
        remove_columns=train_dataset.column_names,
        num_proc=4,
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            tokenize,
            remove_columns=eval_dataset.column_names,
            num_proc=4,
        )

    # Data collator handles dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        learning_rate=lr,
        bf16=True,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=3,
        warmup_steps=200,
        weight_decay=0.01,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=1000 if eval_dataset else None,
        predict_with_generate=True,
        generation_max_length=128,
        dataloader_num_workers=4,
        report_to="none",
        load_best_model_at_end=True if eval_dataset else False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"MarianMT fine-tuned model saved to {output_dir}")

    # Save final eval metrics
    if eval_dataset:
        metrics = trainer.evaluate()
        logger.info(f"Final eval metrics: {metrics}")
        trainer.save_metrics("eval", metrics)

    # Quick translation sanity check
    logger.info("\nSanity check translations:")
    test_sentences = [
        "For God so loved the world.",
        "The Lord is my shepherd.",
        "By grace you have been saved through faith.",
    ]
    for sent in test_sentences:
        inputs = tokenizer(sent, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        import torch
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"  EN: {sent}")
        logger.info(f"  ES: {translation}")


def main():
    parser = argparse.ArgumentParser(
        description="MarianMT full fine-tune fallback for EN->ES translation"
    )
    parser.add_argument("--bible-data", default="bible_data/aligned/verse_pairs_train.jsonl",
                        help="Path to Bible verse pairs JSONL")
    parser.add_argument("--test-data", default="bible_data/holdout/verse_pairs_test.jsonl",
                        help="Path to holdout test JSONL for eval during training")
    parser.add_argument("--output", "-o", default="fine_tuned_marian_mi",
                        help="Output directory")
    parser.add_argument("--model", default="Helsinki-NLP/opus-mt-en-es",
                        help="Base MarianMT model")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint")
    args = parser.parse_args()

    resume = True if args.resume else None

    fine_tune_marian(
        bible_data=args.bible_data,
        test_data=args.test_data,
        output_dir=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        lr=args.lr,
        resume_from_checkpoint=resume,
    )


if __name__ == "__main__":
    main()
