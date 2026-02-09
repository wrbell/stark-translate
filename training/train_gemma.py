#!/usr/bin/env python3
"""
train_gemma.py — TranslateGemma QLoRA Fine-Tuning on Biblical Text

QLoRA fine-tuning for TranslateGemma 4B/12B on ~155K Bible verse pairs
plus theological glossary pairs. Runs on Windows/WSL with A2000 Ada 16GB VRAM.

Config: r=16, alpha=16, 4-bit NF4, target all-linear, paged AdamW
VRAM: ~10-12 GB (4B) | ~14-15 GB (12B)
Training: ~8-12 hrs (4B) | ~18-27 hrs (12B) for 3 epochs

Usage:
    python train_gemma.py A                    # Fine-tune 4B (Approach A)
    python train_gemma.py B                    # Fine-tune 12B (Approach B)
    python train_gemma.py A --epochs 5 --lr 1e-4
    python train_gemma.py A --bible-data bible_data/aligned/verse_pairs_train.jsonl
    python train_gemma.py A --resume           # Resume from checkpoint
"""

import argparse
import logging
import os
import sys

import torch
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def fine_tune_gemma(
    approach="A",
    bible_data="bible_data/aligned/verse_pairs_train.jsonl",
    glossary_data="bible_data/glossary/glossary_pairs.jsonl",
    output_dir=None,
    lora_r=16,
    lora_alpha=16,
    epochs=3,
    lr=2e-4,
    max_seq_length=512,
    resume_from_checkpoint=None,
):
    """QLoRA fine-tuning for TranslateGemma on biblical text.

    Research-validated config (arXiv:2402.15061):
    - r=16 for domain adaptation
    - target "all-linear" for best quality per QLoRA findings
    - 4-bit NF4 quantization via bitsandbytes
    - Paged AdamW 32-bit optimizer
    - Sequence packing for short Bible verses
    """
    model_name = ("google/translategemma-4b-it" if approach == "A"
                  else "google/translategemma-12b-it")
    if output_dir is None:
        output_dir = f"fine_tuned_gemma_mi_{approach}"

    is_12b = "12b" in model_name
    batch_size = 1 if is_12b else 2
    grad_accum = 8 if is_12b else 4

    logger.info(f"Fine-tuning {model_name} (Approach {approach})")
    logger.info(f"  LoRA: r={lora_r}, alpha={lora_alpha}")
    logger.info(f"  Batch: {batch_size} x {grad_accum} = {batch_size * grad_accum} effective")
    logger.info(f"  Output: {output_dir}")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # QLoRA config — target all linear layers for best domain adaptation
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    # Load Bible + glossary data
    logger.info("Loading training data...")
    datasets_to_merge = []

    if os.path.exists(bible_data):
        bible_ds = load_dataset("json", data_files=bible_data, split="train")
        datasets_to_merge.append(bible_ds)
        logger.info(f"  Bible pairs: {len(bible_ds)}")
    else:
        logger.warning(f"Bible data not found at {bible_data}")

    if glossary_data and os.path.exists(glossary_data):
        glossary_ds = load_dataset("json", data_files=glossary_data, split="train")
        datasets_to_merge.append(glossary_ds)
        logger.info(f"  Glossary pairs: {len(glossary_ds)}")

    if not datasets_to_merge:
        logger.error("No training data found!")
        sys.exit(1)

    if len(datasets_to_merge) > 1:
        full_ds = concatenate_datasets(datasets_to_merge).shuffle(seed=42)
    else:
        full_ds = datasets_to_merge[0].shuffle(seed=42)

    logger.info(f"  Total training examples: {len(full_ds)}")

    def format_for_translategemma(example):
        """Format using TranslateGemma's required chat template."""
        messages = [
            {"role": "user", "content": [
                {"type": "text",
                 "source_lang_code": "en",
                 "target_lang_code": "es",
                 "text": example["en"]}
            ]},
            {"role": "assistant", "content": example["es"]},
        ]
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            # Fallback: simple prompt format if chat template fails
            formatted = (f"<start_of_turn>user\n"
                        f"Translate from English to Spanish:\n{example['en']}"
                        f"<end_of_turn>\n"
                        f"<start_of_turn>model\n{example['es']}<end_of_turn>")
        return {"text": formatted}

    full_ds = full_ds.map(format_for_translategemma, remove_columns=full_ds.column_names)

    # Training config — optimized for A2000 Ada 16GB
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        gradient_checkpointing=True,
        bf16=True,
        max_seq_length=max_seq_length,
        packing=True,               # Pack multiple short verses per sequence
        optim="paged_adamw_32bit",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        dataloader_num_workers=4,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=full_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model + tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"TranslateGemma QLoRA adapters ({approach}) saved to {output_dir}")

    # Save training metrics
    trainer.save_metrics("train", trainer.state.log_history[-1] if trainer.state.log_history else {})


def main():
    parser = argparse.ArgumentParser(
        description="TranslateGemma QLoRA fine-tuning on biblical text"
    )
    parser.add_argument("approach", nargs="?", default="A",
                        choices=["A", "B"],
                        help="A = 4B model, B = 12B model")
    parser.add_argument("--bible-data", default="bible_data/aligned/verse_pairs_train.jsonl",
                        help="Path to Bible verse pairs JSONL")
    parser.add_argument("--glossary-data", default="bible_data/glossary/glossary_pairs.jsonl",
                        help="Path to glossary pairs JSONL")
    parser.add_argument("--output", "-o", help="Output directory for QLoRA adapters")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint")
    args = parser.parse_args()

    resume = True if args.resume else None

    fine_tune_gemma(
        approach=args.approach,
        bible_data=args.bible_data,
        glossary_data=args.glossary_data,
        output_dir=args.output,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        epochs=args.epochs,
        lr=args.lr,
        max_seq_length=args.max_seq_length,
        resume_from_checkpoint=resume,
    )


if __name__ == "__main__":
    main()
