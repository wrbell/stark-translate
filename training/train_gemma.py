#!/usr/bin/env python3
"""
train_gemma.py — TranslateGemma QLoRA Fine-Tuning on Biblical Text

QLoRA fine-tuning for TranslateGemma 4B/12B on Bible verse pairs
plus theological glossary pairs. Runs on Windows/WSL with A2000 Ada 16GB VRAM.

Config: r=16, alpha=16, 4-bit NF4, target all-linear, paged AdamW
VRAM: ~10-12 GB (4B) | ~14-15 GB (12B)

Data Scaling Strategy:
  TranslateGemma already has strong translation capability — fine-tuning is
  domain adaptation, not re-learning translation. Research (arXiv:2409.03454)
  shows diminishing returns beyond 10K-30K pairs for LLM translation FT.
  Start small (--max-pairs 20000), scale up only if improvement plateaus.

  --max-pairs 20000   Start here (default). ~3-4 GPU hrs for 4B.
  --max-pairs 50000   Scale up if BLEU improvement < +2 at 20K.
  --max-pairs 0       Use full corpus (155K+). Risk of register overfitting.

  The script prioritizes modern translations (WEB/Espanol Sencillo) over
  archaic ones (KJV/RVR1909) when subsampling, and always includes all
  glossary pairs with oversampling.

Usage:
    python train_gemma.py A                    # Fine-tune 4B (default 20K pairs)
    python train_gemma.py B                    # Fine-tune 12B
    python train_gemma.py A --max-pairs 50000  # Scale up after plateau
    python train_gemma.py A --max-pairs 0      # Full corpus (not recommended first run)
    python train_gemma.py A --sermon-data stark_data/live_sessions/sermon_pairs.jsonl
    python train_gemma.py A --epochs 5 --lr 1e-4
    python train_gemma.py A --resume           # Resume from checkpoint
"""

import argparse
import json
import logging
import os
import random
import sys

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modern-translation priority list for data scaling
# When subsampling, prefer modern register translations over archaic ones.
# This prevents the model from learning overly formal/archaic Spanish.
# ---------------------------------------------------------------------------
MODERN_TRANSLATIONS = {"web", "espanol_sencillo", "bbe"}  # modern/accessible
ARCHAIC_TRANSLATIONS = {"kjv", "rvr1909", "asv", "ylt"}   # formal/archaic

# Glossary oversampling factor — ensures theological terms are well-represented
# even in small training runs. 229 terms * 2 pairs/term * 15x = ~6,870 pairs.
GLOSSARY_OVERSAMPLE = 15


def _subsample_bible_pairs(pairs, max_pairs, seed=42):
    """Subsample Bible pairs, prioritizing modern translations.

    Strategy: include ALL modern-translation pairs first, then fill remaining
    budget with archaic pairs. This prevents archaic register overfitting
    (KJV 1611 + RVR1909 producing overly formal Spanish for spoken input).

    Args:
        pairs: list of dicts with 'en', 'es', and optionally 'source_en'/'source_es'
        max_pairs: target number of pairs (0 = no limit)
        seed: random seed for reproducibility
    """
    if max_pairs <= 0 or len(pairs) <= max_pairs:
        return pairs

    rng = random.Random(seed)

    # Separate modern vs archaic based on source translation metadata
    modern = []
    archaic = []
    unknown = []
    for p in pairs:
        src = (p.get("source_en", "") + " " + p.get("source_es", "")).lower()
        if any(m in src for m in MODERN_TRANSLATIONS):
            modern.append(p)
        elif any(a in src for a in ARCHAIC_TRANSLATIONS):
            archaic.append(p)
        else:
            unknown.append(p)

    # If we can't determine source, treat all as equal priority
    if not modern and not archaic:
        rng.shuffle(pairs)
        return pairs[:max_pairs]

    # Priority: all modern pairs, then fill with archaic
    result = list(modern)
    remaining_budget = max_pairs - len(result)

    if remaining_budget > 0:
        pool = archaic + unknown
        rng.shuffle(pool)
        result.extend(pool[:remaining_budget])
    elif len(result) > max_pairs:
        # More modern pairs than budget — subsample modern too
        rng.shuffle(result)
        result = result[:max_pairs]

    rng.shuffle(result)

    modern_count = min(len(modern), max_pairs)
    archaic_count = len(result) - modern_count
    logger.info(f"  Subsampled to {len(result)} pairs "
                f"({modern_count} modern, {archaic_count} archaic/unknown)")
    return result


def fine_tune_gemma(
    approach="A",
    bible_data="bible_data/aligned/verse_pairs_train.jsonl",
    glossary_data="bible_data/glossary/glossary_pairs.jsonl",
    sermon_data=None,
    output_dir=None,
    lora_r=16,
    lora_alpha=16,
    epochs=3,
    lr=2e-4,
    max_seq_length=512,
    max_pairs=20000,
    resume_from_checkpoint=None,
):
    """QLoRA fine-tuning for TranslateGemma on biblical text.

    Research-validated config (arXiv:2402.15061):
    - r=16 for domain adaptation
    - target "all-linear" for best quality per QLoRA findings
    - 4-bit NF4 quantization via bitsandbytes
    - Paged AdamW 32-bit optimizer
    - Sequence packing for short Bible verses

    Data scaling (arXiv:2409.03454, ALMA-R):
    - Start with 20K pairs (--max-pairs 20000). TranslateGemma already knows
      how to translate; it just needs theological vocabulary adaptation.
    - Scale up to 50K only if BLEU improvement < +2 points at 20K.
    - Using all 155K risks memorizing biblical phrasing over generalizable
      theological patterns. ALMA-R matched GPT-4 with only 22K pairs.
    - Glossary pairs are always fully included with 15x oversampling.
    - Sermon pairs (from live sessions) are always fully included at 2x weight.
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
    logger.info(f"  Max pairs: {max_pairs if max_pairs > 0 else 'unlimited'}")
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

    # Load training data with scaling strategy:
    #   1. Bible pairs: subsampled to max_pairs, modern translations prioritized
    #   2. Glossary pairs: always fully included, oversampled 15x
    #   3. Sermon pairs: always fully included (real-domain, highest value)
    logger.info("Loading training data...")
    all_pairs = []

    # --- Bible corpus (subsampled) ---
    if os.path.exists(bible_data):
        with open(bible_data) as f:
            bible_pairs = [json.loads(line) for line in f]
        original_count = len(bible_pairs)
        bible_pairs = _subsample_bible_pairs(bible_pairs, max_pairs)
        logger.info(f"  Bible pairs: {len(bible_pairs)}"
                    f" (of {original_count} available"
                    f"{', subsampled' if len(bible_pairs) < original_count else ''})")
        all_pairs.extend(bible_pairs)
    else:
        logger.warning(f"Bible data not found at {bible_data}")

    # --- Glossary pairs (always included, oversampled) ---
    if glossary_data and os.path.exists(glossary_data):
        with open(glossary_data) as f:
            glossary_pairs = [json.loads(line) for line in f]
        oversampled = glossary_pairs * GLOSSARY_OVERSAMPLE
        logger.info(f"  Glossary pairs: {len(glossary_pairs)} x {GLOSSARY_OVERSAMPLE} "
                    f"= {len(oversampled)}")
        all_pairs.extend(oversampled)
    else:
        logger.info("  No glossary data (skipping)")

    # --- Sermon pairs from live sessions (always included, highest value) ---
    if sermon_data and os.path.exists(sermon_data):
        with open(sermon_data) as f:
            sermon_pairs = [json.loads(line) for line in f]
        # Include sermon pairs twice — real-domain data is more valuable
        # than bulk Bible pairs for generalization to spoken register
        doubled = sermon_pairs * 2
        logger.info(f"  Sermon pairs: {len(sermon_pairs)} x 2 = {len(doubled)} "
                    f"(real-domain, 2x weighted)")
        all_pairs.extend(doubled)
    elif sermon_data:
        logger.info(f"  Sermon data not found at {sermon_data} (skipping)")

    if not all_pairs:
        logger.error("No training data found!")
        sys.exit(1)

    # Shuffle and convert to HuggingFace Dataset
    random.Random(42).shuffle(all_pairs)
    # Normalize to just en/es columns for the formatter
    normalized = [{"en": p["en"], "es": p["es"]} for p in all_pairs]
    full_ds = Dataset.from_list(normalized)

    logger.info(f"  Total training examples: {len(full_ds)}")
    if max_pairs > 0 and max_pairs < 50000:
        logger.info(f"  NOTE: Using {max_pairs} pairs. If BLEU improvement < +2 points,")
        logger.info(f"        re-run with --max-pairs {max_pairs * 2} to scale up.")

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
    parser.add_argument("--sermon-data", default=None,
                        help="Path to sermon pairs JSONL (from live sessions)")
    parser.add_argument("--output", "-o", help="Output directory for QLoRA adapters")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-pairs", type=int, default=20000,
                        help="Max Bible pairs to use (0=unlimited). "
                        "Start at 20K, scale to 50K if BLEU < +2. "
                        "Glossary and sermon pairs are always fully included.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint")
    args = parser.parse_args()

    resume = True if args.resume else None

    fine_tune_gemma(
        approach=args.approach,
        bible_data=args.bible_data,
        glossary_data=args.glossary_data,
        sermon_data=args.sermon_data,
        output_dir=args.output,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        epochs=args.epochs,
        lr=args.lr,
        max_seq_length=args.max_seq_length,
        max_pairs=args.max_pairs,
        resume_from_checkpoint=resume,
    )


if __name__ == "__main__":
    main()
