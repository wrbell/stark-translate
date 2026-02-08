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
"""

import argparse
import os

from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments


def fine_tune_marian(
    bible_data="bible_data/aligned/verse_pairs_train.jsonl",
    output_dir="fine_tuned_marian_mi",
    model_name="Helsinki-NLP/opus-mt-en-es",
    batch_size=32,
    epochs=5,
    lr=5e-5,
):
    """Full fine-tune MarianMT on biblical text.

    Only 298MB — no LoRA needed. Big batches, fast epochs.
    Lower quality ceiling than TranslateGemma but dramatically higher iteration speed.
    """
    print(f"Loading {model_name}...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    print(f"Loading data from {bible_data}...")
    dataset = load_dataset("json", data_files=bible_data, split="train")
    print(f"  {len(dataset)} training examples")

    def tokenize(example):
        inputs = tokenizer(example["en"], truncation=True, max_length=128)
        targets = tokenizer(text_target=example["es"], truncation=True, max_length=128)
        inputs["labels"] = targets["input_ids"]
        return inputs

    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        bf16=True,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=3,
        warmup_steps=200,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"MarianMT fine-tuned model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="MarianMT full fine-tune fallback for EN→ES translation"
    )
    parser.add_argument("--bible-data", default="bible_data/aligned/verse_pairs_train.jsonl",
                        help="Path to Bible verse pairs JSONL")
    parser.add_argument("--output", "-o", default="fine_tuned_marian_mi",
                        help="Output directory")
    parser.add_argument("--model", default="Helsinki-NLP/opus-mt-en-es",
                        help="Base MarianMT model")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    fine_tune_marian(
        bible_data=args.bible_data,
        output_dir=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
