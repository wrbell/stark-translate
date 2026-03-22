#!/usr/bin/env python3
"""
train_gemma.py — TranslateGemma QLoRA Fine-Tuning on Biblical Text

QLoRA fine-tuning for TranslateGemma 4B/12B on Bible verse pairs
plus theological glossary pairs. Runs on Windows/WSL with A2000 Ada 16GB VRAM.

Config: r=16, alpha=32, 4-bit NF4, target all-linear, paged AdamW 8-bit
VRAM: ~10-12 GB (4B) | ~14-15 GB (12B)

Tuning knobs:
  - lora_alpha: currently set to r (conservative). Try alpha=2*r for
    stronger adaptation if underfitting theological terms.

Data Scaling Strategy:
  TranslateGemma already has strong translation capability — fine-tuning is
  domain adaptation, not re-learning translation. Research (arXiv:2409.03454)
  shows diminishing returns beyond 10K-30K pairs for LLM translation FT.
  Start small (--max-pairs 20000), scale up only if improvement plateaus.

  --max-pairs 20000   Start here (default). ~1-2 GPU hrs for 4B.
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
    python train_gemma.py A --lr 5e-6          # If 1e-5 still slightly degrades
    python train_gemma.py A --max-steps 150    # Stop after 150 gradient steps
    python train_gemma.py A --neftune 5        # NEFTune noise injection (alpha=5)
    python train_gemma.py A --replay-ratio 0.2 # Mix 20% general EN→ES pairs
    python train_gemma.py A --lora-dropout 0           # Disable LoRA dropout
    python train_gemma.py A --glossary-oversample 3    # Increase glossary oversampling
    python train_gemma.py A --resume                   # Resume from checkpoint
"""

import argparse
import json
import logging
import os
import random
import sys

os.environ["USE_TF"] = "0"  # Prevent transformers from importing TF/Keras (not needed for PyTorch training)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Reduce VRAM fragmentation
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # Default to GPU 0 if not set by user/shell

import torch

torch.set_float32_matmul_precision("high")  # Enable TF32 on Ada — ~2x matmul throughput

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
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
ARCHAIC_TRANSLATIONS = {"kjv", "rvr1909", "asv", "ylt"}  # formal/archaic

# Glossary oversampling default — ensures theological terms are well-represented
# even in small training runs. 229 terms * 2 pairs/term * 2x = ~916 pairs.
# Higher values increase theological term weight but reduce batch diversity.
GLOSSARY_OVERSAMPLE_DEFAULT = 2

# Memory caps — allow shared GPU memory but prevent runaway allocation.
# A2000 Ada has 16GB VRAM; Windows shares additional system RAM as GPU memory.
VRAM_CAP = "15GiB"  # Leave ~1GB headroom for OS/display
SHARED_MEM_CAP = "24GiB"  # Use ~80% of available 30GB shared system RAM
VRAM_WARN_PERCENT = 85  # Log warning when VRAM usage exceeds this %


class VRAMMonitorCallback(TrainerCallback):
    """Logs GPU + shared memory usage periodically during training."""

    def on_log(self, args, state, control, **kwargs):
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        pct = (reserved / total) * 100
        level = logging.WARNING if pct > VRAM_WARN_PERCENT else logging.INFO
        logger.log(
            level, f"[VRAM] {allocated:.1f}GB alloc / {reserved:.1f}GB reserved / {total:.1f}GB total ({pct:.0f}%)"
        )
        if torch.cuda.max_memory_allocated() > 0:
            peak = torch.cuda.max_memory_allocated() / 1e9
            logger.log(level, f"[VRAM] Peak allocated: {peak:.1f}GB")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 != 0 or not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        pct = (reserved / total) * 100
        if pct > VRAM_WARN_PERCENT:
            logger.warning(
                f"[VRAM] Step {state.global_step}: {allocated:.1f}GB alloc / {reserved:.1f}GB reserved ({pct:.0f}%)"
            )


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
    logger.info(f"  Subsampled to {len(result)} pairs ({modern_count} modern, {archaic_count} archaic/unknown)")
    return result


def _load_replay_pairs(ratio, domain_count, seed=42):
    """Load general-domain EN→ES pairs from Helsinki-NLP/opus-100 for replay.

    Replay data anchors general translation ability during domain adaptation,
    preventing catastrophic forgetting (mirrors Whisper's 30% LibriSpeech approach).

    Args:
        ratio: fraction of domain data to add as replay (e.g. 0.2 = 20%)
        domain_count: number of domain-specific training pairs
        seed: random seed for reproducibility

    Returns:
        list of dicts with 'en' and 'es' keys
    """
    replay_count = int(domain_count * ratio)
    if replay_count <= 0:
        return []

    logger.info(f"  Loading {replay_count} general EN→ES replay pairs from opus-100...")
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "Helsinki-NLP/opus-100",
            "en-es",
            split="train",
            streaming=True,
        )
        pairs = []
        rng = random.Random(seed)
        # Reservoir sampling to get a random subset without loading entire dataset
        for i, example in enumerate(ds):
            pair = {"en": example["translation"]["en"], "es": example["translation"]["es"]}
            if len(pairs) < replay_count:
                pairs.append(pair)
            else:
                j = rng.randint(0, i)
                if j < replay_count:
                    pairs[j] = pair
            # Stop early once we've seen enough candidates for good sampling
            if i >= replay_count * 10:
                break

        logger.info(f"  Loaded {len(pairs)} general replay pairs")
        return pairs
    except Exception as e:
        logger.warning(f"  Failed to load replay data: {e}")
        logger.warning("  Continuing without replay buffer.")
        return []


def _save_training_manifest(
    output_dir,
    approach,
    model_name,
    bible_data,
    glossary_data,
    sermon_data,
    max_pairs,
    glossary_oversample,
    lr,
    max_steps,
    lora_r,
    lora_alpha,
    neftune_alpha,
    replay_ratio,
    trainer,
):
    """Save a training manifest for reproducibility and provenance tracking."""
    import hashlib
    from datetime import UTC, datetime

    def _file_hash(path):
        if not path or not os.path.exists(path):
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _file_lines(path):
        if not path or not os.path.exists(path):
            return 0
        with open(path) as f:
            return sum(1 for _ in f)

    manifest = {
        "run_id": os.path.basename(output_dir),
        "timestamp": datetime.now(UTC).isoformat(),
        "model": model_name,
        "approach": approach,
        "config": {
            "lr": lr,
            "max_steps": max_steps,
            "max_pairs": max_pairs,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "neftune_alpha": neftune_alpha,
            "replay_ratio": replay_ratio,
            "glossary_oversample": glossary_oversample,
        },
        "data": {
            "bible_pairs": {
                "file": bible_data,
                "count": _file_lines(bible_data),
                "used": min(max_pairs, _file_lines(bible_data)) if max_pairs > 0 else _file_lines(bible_data),
                "sha256": _file_hash(bible_data),
            },
            "glossary_pairs": {
                "file": glossary_data,
                "count": _file_lines(glossary_data),
                "oversample": glossary_oversample,
                "sha256": _file_hash(glossary_data),
            },
            "sermon_pairs": {
                "file": sermon_data,
                "count": _file_lines(sermon_data) if sermon_data else 0,
                "sha256": _file_hash(sermon_data),
            },
        },
        "results": {},
    }

    # Extract final metrics from trainer
    if trainer.state.log_history:
        last = trainer.state.log_history[-1]
        manifest["results"] = {k: v for k, v in last.items() if isinstance(v, int | float) and k != "epoch"}

    # Save adapter hash
    adapter_path = os.path.join(output_dir, "adapter_model.safetensors")
    manifest["adapter_sha256"] = _file_hash(adapter_path)

    manifest_path = os.path.join(output_dir, "training_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Training manifest saved to {manifest_path}")


def fine_tune_gemma(
    approach="A",
    bible_data="bible_data/aligned/verse_pairs_train.jsonl",
    glossary_data="bible_data/glossary/glossary_pairs.jsonl",
    sermon_data=None,
    output_dir=None,
    lora_r=16,
    lora_alpha=32,
    epochs=1,
    lr=1e-5,
    max_seq_length=512,
    max_pairs=20000,
    max_steps=-1,
    neftune_alpha=None,
    lora_dropout=0.05,
    replay_ratio=0.0,
    glossary_oversample=GLOSSARY_OVERSAMPLE_DEFAULT,
    resume_from_checkpoint=None,
):
    """QLoRA fine-tuning for TranslateGemma on biblical text.

    Research-validated config (arXiv:2402.15061):
    - r=16 for domain adaptation
    - target "all-linear" for best quality per QLoRA findings
    - 4-bit NF4 quantization via bitsandbytes
    - Paged AdamW 8-bit optimizer
    - Sequence packing for short Bible verses

    Data scaling (arXiv:2409.03454, ALMA-R):
    - Start with 20K pairs (--max-pairs 20000). TranslateGemma already knows
      how to translate; it just needs theological vocabulary adaptation.
    - Scale up to 50K only if BLEU improvement < +2 points at 20K.
    - Using all 155K risks memorizing biblical phrasing over generalizable
      theological patterns. ALMA-R matched GPT-4 with only 22K pairs.
    - Glossary pairs are always fully included with oversampling (--glossary-oversample).
    - Sermon pairs (from live sessions) are always fully included at 1x (no oversampling).
    """
    model_name = "google/translategemma-4b-it" if approach == "A" else "google/translategemma-12b-it"
    if output_dir is None:
        output_dir = f"fine_tuned_gemma_mi_{approach}"

    batch_size = 2
    grad_accum = 4

    logger.info(f"Fine-tuning {model_name} (Approach {approach})")
    logger.info(f"  LoRA: r={lora_r}, alpha={lora_alpha}")
    logger.info(f"  Batch: {batch_size} x {grad_accum} = {batch_size * grad_accum} effective")
    logger.info(f"  Max pairs: {max_pairs if max_pairs > 0 else 'unlimited'}")
    if max_steps > 0:
        logger.info(f"  Max steps: {max_steps}")
    if neftune_alpha is not None:
        logger.info(f"  NEFTune noise alpha: {neftune_alpha}")
    if replay_ratio > 0:
        logger.info(f"  Replay ratio: {replay_ratio:.0%} general EN→ES")
    logger.info(f"  Output: {output_dir}")

    # 4-bit quantization config — BF16 compute for torch.compile compatibility
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
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        max_memory={0: VRAM_CAP, "cpu": SHARED_MEM_CAP},
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # QLoRA config — target all linear layers for best domain adaptation
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    # Load training data with scaling strategy:
    #   1. Bible pairs: subsampled to max_pairs, modern translations prioritized
    #   2. Glossary pairs: always fully included, oversampled by --glossary-oversample factor
    #   3. Sermon pairs: always fully included (real-domain, highest value)
    logger.info("Loading training data...")
    all_pairs = []

    # --- Bible corpus (subsampled) ---
    if os.path.exists(bible_data):
        with open(bible_data) as f:
            bible_pairs = [json.loads(line) for line in f]
        original_count = len(bible_pairs)
        bible_pairs = _subsample_bible_pairs(bible_pairs, max_pairs)
        logger.info(
            f"  Bible pairs: {len(bible_pairs)}"
            f" (of {original_count} available"
            f"{', subsampled' if len(bible_pairs) < original_count else ''})"
        )
        all_pairs.extend(bible_pairs)
    else:
        logger.warning(f"Bible data not found at {bible_data}")

    # --- Glossary pairs (always included, oversampled) ---
    if glossary_data and os.path.exists(glossary_data):
        with open(glossary_data) as f:
            glossary_pairs = [json.loads(line) for line in f]
        oversampled = glossary_pairs * glossary_oversample
        logger.info(f"  Glossary pairs: {len(glossary_pairs)} x {glossary_oversample} = {len(oversampled)}")
        all_pairs.extend(oversampled)
    else:
        logger.info("  No glossary data (skipping)")

    # --- Sermon pairs from live sessions (always included, highest value) ---
    if sermon_data and os.path.exists(sermon_data):
        with open(sermon_data) as f:
            sermon_pairs = [json.loads(line) for line in f]
        logger.info(f"  Sermon pairs: {len(sermon_pairs)} (real-domain, included 1x)")
        all_pairs.extend(sermon_pairs)
    elif sermon_data:
        logger.info(f"  Sermon data not found at {sermon_data} (skipping)")

    # --- General-domain replay buffer (prevents catastrophic forgetting) ---
    if replay_ratio > 0:
        replay_pairs = _load_replay_pairs(replay_ratio, len(all_pairs))
        if replay_pairs:
            logger.info(f"  Replay pairs: {len(replay_pairs)} ({replay_ratio:.0%} of {len(all_pairs)} domain pairs)")
            all_pairs.extend(replay_pairs)

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
            {
                "role": "user",
                "content": [
                    {"type": "text", "source_lang_code": "en", "target_lang_code": "es", "text": example["en"]}
                ],
            },
            {"role": "assistant", "content": example["es"]},
        ]
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        except (TypeError, KeyError, ValueError) as e:
            # Fallback: simple prompt format if chat template fails
            logger.warning(f"Chat template failed ({e}), using fallback format")
            formatted = (
                f"<start_of_turn>user\n"
                f"Translate from English to Spanish:\n{example['en']}"
                f"<end_of_turn>\n"
                f"<start_of_turn>model\n{example['es']}<end_of_turn>"
            )
        return {"text": formatted}

    # Verify which formatting path fires (critical — wrong template = guaranteed regression)
    _test_msg = [
        {
            "role": "user",
            "content": [{"type": "text", "source_lang_code": "en", "target_lang_code": "es", "text": "test"}],
        },
        {"role": "assistant", "content": "prueba"},
    ]
    try:
        _test_result = tokenizer.apply_chat_template(_test_msg, tokenize=False)
        logger.info(f"  Chat template OK. Sample: {_test_result[:120]}...")
    except Exception as e:
        logger.warning(f"  Chat template FAILED ({e}) — all examples will use fallback format!")
        logger.warning("  This likely means training data format won't match the model's expected input.")

    full_ds = full_ds.map(
        format_for_translategemma,
        remove_columns=full_ds.column_names,
        num_proc=min(4, os.cpu_count() or 1),
        desc="Formatting for TranslateGemma",
    )

    # Truncation length validation
    lengths = [len(tokenizer.encode(ex["text"])) for ex in full_ds]
    max_len = max(lengths)
    median_len = sorted(lengths)[len(lengths) // 2]
    over_limit = sum(1 for l in lengths if l > max_seq_length)
    logger.info(f"  Token lengths: max={max_len}, median={median_len}")
    if over_limit > 0:
        logger.warning(f"  {over_limit} examples exceed max_seq_length={max_seq_length} and will be truncated!")

    # Split 5% for eval (overfitting detection + early stopping)
    split = full_ds.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    logger.info(f"  Train/eval split: {len(train_ds)} train, {len(eval_ds)} eval")

    # Training config — optimized for A2000 Ada 16GB
    sft_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=0.5,
        optim="paged_adamw_8bit",
        bf16=True,
        max_length=max_seq_length,
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=400,
        save_total_limit=3,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        seed=42,
        report_to="none",
    )
    if max_steps > 0:
        sft_kwargs["max_steps"] = max_steps
    if neftune_alpha is not None:
        sft_kwargs["neftune_noise_alpha"] = neftune_alpha
    training_args = SFTConfig(**sft_kwargs)

    torch.cuda.empty_cache()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[VRAMMonitorCallback()],
    )

    logger.info("Starting training...")
    logger.info(f"[VRAM] Memory caps: GPU={VRAM_CAP}, shared={SHARED_MEM_CAP}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final LoRA adapters + tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"TranslateGemma QLoRA adapters ({approach}) saved to {output_dir}")

    # Save training metrics
    if trainer.state.log_history:
        trainer.save_metrics("train", trainer.state.log_history[-1])
    else:
        logger.warning("No training metrics to save.")

    # Save training manifest for reproducibility
    _save_training_manifest(
        output_dir=output_dir,
        approach=approach,
        model_name=model_name,
        bible_data=bible_data,
        glossary_data=glossary_data,
        sermon_data=sermon_data,
        max_pairs=max_pairs,
        glossary_oversample=glossary_oversample,
        lr=lr,
        max_steps=max_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        neftune_alpha=neftune_alpha,
        replay_ratio=replay_ratio,
        trainer=trainer,
    )


def main():
    parser = argparse.ArgumentParser(description="TranslateGemma QLoRA fine-tuning on biblical text")
    parser.add_argument("approach", nargs="?", default="A", choices=["A", "B"], help="A = 4B model, B = 12B model")
    parser.add_argument(
        "--bible-data", default="bible_data/aligned/verse_pairs_train.jsonl", help="Path to Bible verse pairs JSONL"
    )
    parser.add_argument(
        "--glossary-data", default="bible_data/glossary/glossary_pairs.jsonl", help="Path to glossary pairs JSONL"
    )
    parser.add_argument("--sermon-data", default=None, help="Path to sermon pairs JSONL (from live sessions)")
    parser.add_argument("--output", "-o", help="Output directory for QLoRA adapters")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=20000,
        help="Max Bible pairs to use (0=unlimited). "
        "Start at 20K, scale to 50K if BLEU < +2. "
        "Glossary and sermon pairs are always fully included.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Stop after N gradient steps (overrides --epochs). -1 = use epochs.",
    )
    parser.add_argument(
        "--neftune",
        type=float,
        default=None,
        help="NEFTune noise alpha (e.g. 5). Adds noise to embeddings during training for regularization.",
    )
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=0.0,
        help="Fraction of general EN→ES pairs to mix in (e.g. 0.2 = 20%%). "
        "Loads from Helsinki-NLP/opus-100 to prevent catastrophic forgetting.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate (default 0.05). Set to 0 to disable.",
    )
    parser.add_argument(
        "--glossary-oversample",
        type=int,
        default=GLOSSARY_OVERSAMPLE_DEFAULT,
        help="Glossary pair oversampling factor (default 2). Higher values increase "
        "theological term representation at cost of batch diversity.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    args = parser.parse_args()

    resume = True if args.resume else None
    if resume:
        out = args.output or f"fine_tuned_gemma_mi_{args.approach}"
        if not os.path.isdir(out) or not any(
            d.startswith("checkpoint-") for d in os.listdir(out) if os.path.isdir(os.path.join(out, d))
        ):
            logger.warning("--resume passed but no checkpoints found in %s. Starting fresh.", out)
            resume = None

    fine_tune_gemma(
        approach=args.approach,
        bible_data=args.bible_data,
        glossary_data=args.glossary_data,
        sermon_data=args.sermon_data,
        output_dir=args.output,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        epochs=args.epochs,
        lr=args.lr,
        max_seq_length=args.max_seq_length,
        max_pairs=args.max_pairs,
        max_steps=args.max_steps,
        neftune_alpha=args.neftune,
        replay_ratio=args.replay_ratio,
        glossary_oversample=args.glossary_oversample,
        resume_from_checkpoint=resume,
    )


if __name__ == "__main__":
    main()
