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
import hashlib
import json
import logging
import os
from collections import Counter
from datetime import UTC, datetime

os.environ["USE_TF"] = "0"  # Prevent transformers from importing TF/Keras (not needed for PyTorch training)

import torch
from datasets import Audio, load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import WeightedRandomSampler
from transformers import (
    BitsAndBytesConfig,
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

    from datasets import interleave_datasets

    # Load general English ASR as replay buffer
    logger.info(f"Loading general-domain replay data (ratio={replay_ratio})...")
    try:
        general = load_dataset(
            "openslr/librispeech_asr",
            "clean",
            split="train.100[:2000]",
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
        # LibriSpeech uses "text" column
        text = batch.get("text", "")
        batch["labels"] = processor.tokenizer(text).input_ids
        return batch

    general = general.map(
        prepare_general,
        remove_columns=general.column_names,
        num_proc=1,  # single-process: avoids deadlock on HDD/symlink I/O
    )

    # Ensure replay subset has ONLY input_features + labels (match church dataset columns)
    church_cols = set(church_dataset.column_names)
    replay_cols = set(general.column_names)
    extra_cols = replay_cols - church_cols
    if extra_cols:
        logger.info(f"  Dropping extra replay columns: {extra_cols}")
        general = general.remove_columns(list(extra_cols))

    # Use interleave_datasets instead of concatenate_datasets to avoid RAM OOM.
    # concatenate_datasets materializes both datasets into RAM (~1 MB per chunk).
    # At 100K chunks that's 100 GB — instant OOM.
    # interleave_datasets creates index mappings without copying data.
    #
    # CRITICAL: Use "all_exhausted" so the replay buffer cycles and ALL church data
    # is seen. "first_exhausted" stops when the smaller dataset (2K replay) runs out,
    # which at 198K church data means only 2.5% of training data is used.
    church_prob = 1.0 - replay_ratio
    logger.info(f"  Church: {len(church_dataset)} ({church_prob:.0%}), Replay: {len(general)} ({replay_ratio:.0%})")
    logger.info("  Using interleave_datasets (memory-safe, all_exhausted)")
    mixed = interleave_datasets(
        [church_dataset, general],
        probabilities=[church_prob, replay_ratio],
        seed=42,
        stopping_strategy="all_exhausted",
    )
    return mixed


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

    def _get_train_sampler(self, dataset=None):
        if self._accent_sampler is not None:
            return self._accent_sampler
        return super()._get_train_sampler(dataset)


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


def _save_training_manifest(
    output_dir,
    model_name,
    dataset_path,
    target_modules,
    lora_r,
    lora_alpha,
    batch_size,
    grad_accum,
    epochs,
    lr,
    replay_ratio,
    accent_balance,
    accent_distribution,
    trainer,
    eval_metrics=None,
):
    """Save a training manifest for reproducibility and provenance tracking."""

    def _file_hash(path):
        if not path or not os.path.exists(path):
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    manifest = {
        "run_id": os.path.basename(output_dir),
        "timestamp": datetime.now(UTC).isoformat(),
        "model": model_name,
        "config": {
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "target_modules": target_modules,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "effective_batch_size": batch_size * grad_accum,
            "epochs": epochs,
            "lr": lr,
            "replay_ratio": replay_ratio,
            "accent_balance": accent_balance,
        },
        "data": {
            "dataset_path": dataset_path,
            "accent_distribution": accent_distribution or {},
        },
        "results": {},
    }

    # Extract final metrics from trainer
    if trainer.state.log_history:
        last = trainer.state.log_history[-1]
        manifest["results"] = {k: v for k, v in last.items() if isinstance(v, int | float) and k != "epoch"}

    # Include eval WER if available
    if eval_metrics:
        manifest["results"]["eval"] = {k: v for k, v in eval_metrics.items() if isinstance(v, int | float)}

    # Save adapter hash
    adapter_path = os.path.join(output_dir, "adapter_model.safetensors")
    manifest["adapter_sha256"] = _file_hash(adapter_path)

    manifest_path = os.path.join(output_dir, "training_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Training manifest saved to {manifest_path}")


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
    init_from_adapter=None,
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
    # CRITICAL: Do NOT set task_type. PEFT's SEQ_2_SEQ_LM and CAUSAL_LM wrappers
    # both inject input_ids into forward(), which Whisper doesn't accept (it uses
    # input_features). Omitting task_type creates a base PeftModel that passes
    # through to WhisperForConditionalGeneration.forward() cleanly.
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # Curriculum learning: load pre-trained adapter weights with fresh optimizer state.
    # Unlike --resume (which restores optimizer momentum/velocity), this only loads the
    # LoRA A/B matrices — the optimizer starts from scratch for a new learning trajectory.
    if init_from_adapter:
        from safetensors.torch import load_file

        weights_path = os.path.join(init_from_adapter, "adapter_model.safetensors")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Adapter weights not found: {weights_path}")
        adapter_weights = load_file(weights_path)

        # Handle PEFT version differences in key naming.
        # Older PEFT: lora_A.weight / lora_B.weight
        # Newer PEFT: lora_A.default.weight / lora_B.default.weight
        # Detect which convention the current model uses and remap if needed.
        model_keys = set(model.state_dict().keys())
        model_has_default = any(".lora_A.default." in k for k in model_keys)
        file_has_default = any(".lora_A.default." in k for k in adapter_weights)

        if model_has_default and not file_has_default:
            # Remap old-style keys to new-style: lora_A.weight → lora_A.default.weight
            remapped = {}
            for k, v in adapter_weights.items():
                new_key = k.replace(".lora_A.weight", ".lora_A.default.weight").replace(
                    ".lora_B.weight", ".lora_B.default.weight"
                )
                remapped[new_key] = v
            logger.info("Remapped %d adapter keys: old PEFT (no .default) → new PEFT (.default)", len(remapped))
            adapter_weights = remapped
        elif not model_has_default and file_has_default:
            # Remap new-style keys to old-style: lora_A.default.weight → lora_A.weight
            remapped = {}
            for k, v in adapter_weights.items():
                new_key = k.replace(".lora_A.default.weight", ".lora_A.weight").replace(
                    ".lora_B.default.weight", ".lora_B.weight"
                )
                remapped[new_key] = v
            logger.info("Remapped %d adapter keys: new PEFT (.default) → old PEFT (no .default)", len(remapped))
            adapter_weights = remapped

        incompatible = model.load_state_dict(adapter_weights, strict=False)
        if incompatible.missing_keys:
            # Filter to only LoRA keys — base model keys are expected to be "missing"
            # since we only load the adapter weights, not the full model.
            lora_missing = [k for k in incompatible.missing_keys if "lora_" in k]
            if lora_missing:
                raise ValueError(
                    f"LoRA keys missing when loading adapter from {init_from_adapter}: {lora_missing}. "
                    f"Check that lora_r and target_modules match the source adapter."
                )
        if incompatible.unexpected_keys:
            logger.warning("Unexpected keys in init-from adapter: %s", incompatible.unexpected_keys)
        # Verify at least some LoRA weights were loaded
        lora_loaded = [k for k in adapter_weights if "lora_" in k]
        if not lora_loaded:
            raise ValueError(
                f"No LoRA weights found in {weights_path}. File contains keys: {list(adapter_weights.keys())[:5]}"
            )
        logger.info(
            "Initialized adapter from %s (%d LoRA tensors loaded: %s)",
            init_from_adapter,
            len(lora_loaded),
            ", ".join(lora_loaded[:4]),
        )

    model.print_trainable_parameters()

    # Enable gradient flow through quantized + LoRA layers.
    # enable_input_require_grads() hooks the decoder token embedding, but Whisper's
    # encoder input is mel spectrograms via conv1, not token embeddings. We need to
    # explicitly hook the encoder conv1 layer so gradient checkpointing can propagate
    # gradients through the frozen encoder to the LoRA-adapted attention layers.
    model.enable_input_require_grads()

    def _make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    # Hook encoder conv1 (first layer that processes mel spectrograms)
    model.base_model.model.model.encoder.conv1.register_forward_hook(_make_inputs_require_grad)

    # Turbo has only 4 decoder layers (vs 32 in large-v3), so LoRA adapts faster
    # but the decoder has less capacity — monitor theological term accuracy closely
    if "turbo" in model_name.lower():
        logger.info("Turbo model detected — 4 decoder layers, faster training expected")
        logger.info("Monitor theological term accuracy (lighter decoder = less vocabulary capacity)")

    # Load dataset — use preprocessed cache if available (skips slow audiofolder scan)
    cache_dir = os.path.join(dataset_path, ".preprocessed_cache")
    if os.path.exists(cache_dir):
        logger.info(f"Loading preprocessed dataset from cache: {cache_dir}")
        from datasets import DatasetDict

        dataset = DatasetDict.load_from_disk(cache_dir)
        logger.info(f"  Loaded: {dataset}")
    else:
        logger.info(f"Loading dataset from {dataset_path} (first run — will cache for next time)...")
        dataset = load_dataset("audiofolder", data_dir=dataset_path)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        # Detect text column name
        sample_cols = dataset.column_names["train"]
        text_col = "transcription" if "transcription" in sample_cols else "text"
        remove_cols = [c for c in sample_cols if c not in ("input_features", "labels")]

        def prepare_dataset(batch):
            audio = batch["audio"]
            batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
            batch["labels"] = processor.tokenizer(batch[text_col]).input_ids
            return batch

        logger.info("Preprocessing dataset (mapping audio → mel spectrograms)...")
        dataset = dataset.map(
            prepare_dataset,
            remove_columns=remove_cols,
            num_proc=1,
        )
        dataset.save_to_disk(cache_dir)
        logger.info(f"Saved preprocessed dataset to cache: {cache_dir}")

    # Accent labels not tracked in preprocessed cache — skip per-accent metrics
    train_accent_labels = None
    eval_accent_labels = None

    # Split into train/eval if no eval split exists
    available_splits = list(dataset.keys())
    eval_split_name = (
        "eval"
        if "eval" in available_splits
        else ("test" if "test" in available_splits else ("validation" if "validation" in available_splits else None))
    )

    if eval_split_name is None:
        logger.info("No eval split found, creating 95/5 train/eval split...")
        split = dataset["train"].train_test_split(test_size=0.05, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        # Accent labels: skip per-accent tracking for auto-split (too fragile with _indices)
        train_accent_labels = None
        eval_accent_labels = None
    else:
        train_dataset = dataset["train"]
        eval_dataset = dataset[eval_split_name]
        logger.info(f"Using pre-split eval set: '{eval_split_name}' ({len(eval_dataset)} examples)")
        # Accent labels not available after column removal — skip
        eval_accent_labels = None

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
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },  # non-reentrant: compatible with conv1 requires_grad hook
        num_train_epochs=epochs,
        learning_rate=lr,
        bf16=True,
        optim="adamw_bnb_8bit",
        dataloader_pin_memory=True,
        dataloader_num_workers=0,  # in-process: avoids fork-based memory duplication that causes OOM on 17K dataset
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        eval_strategy="no",  # eval at end only — mid-training eval crashes on 8bit+bf16 dtype mismatch
        eval_steps=None,
        predict_with_generate=False,  # disable generation during training — eval WER computed post-training
        generation_max_length=225,
        warmup_steps=500,
        remove_unused_columns=False,  # keep input_features — Whisper needs it but patched forward hides it
        report_to="none",
        load_best_model_at_end=False,  # no mid-training eval, so no "best" to pick
    )

    # Whisper needs a custom data collator that handles input_features (mel spectrograms)
    # + labels (token IDs), NOT the standard Seq2Seq collator which expects input_ids
    import dataclasses

    @dataclasses.dataclass
    class WhisperDataCollator:
        processor: WhisperProcessor

        def __call__(self, features):
            # Pad input features (mel spectrograms)
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            # Cast to bfloat16 to match 8-bit quantized model's conv layer dtypes
            batch["input_features"] = batch["input_features"].to(torch.bfloat16)

            # Pad labels (token IDs)
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # Remove BOS token if present (Whisper doesn't use it as a prefix)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
                labels = labels[:, 1:]

            batch["labels"] = labels
            # Ensure no input_ids key — Whisper uses input_features, not input_ids
            batch.pop("input_ids", None)
            return batch

    data_collator = WhisperDataCollator(processor=processor)

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
    # NOTE: Post-training eval is DISABLED inside the trainer. The trainer's evaluate()
    # causes system RAM OOM (31 GB RSS on 17K dataset) when it builds prediction tensors.
    # Use the standalone eval script instead:
    #   python training/eval_whisper_wer.py --adapter <output_dir> --eval-set <dataset>
    eval_metrics = None

    # Build accent distribution for manifest
    accent_distribution = None
    if train_accent_labels:
        accent_distribution = dict(Counter(train_accent_labels))

    # Save training manifest for reproducibility
    _save_training_manifest(
        output_dir=output_dir,
        model_name=model_name,
        dataset_path=dataset_path,
        target_modules=target_modules,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        batch_size=batch_size,
        grad_accum=grad_accum,
        epochs=epochs,
        lr=lr,
        replay_ratio=replay_ratio,
        accent_balance=accent_balance,
        accent_distribution=accent_distribution,
        trainer=trainer,
        eval_metrics=eval_metrics,
    )


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
        "--init-from",
        type=str,
        default=None,
        help="Initialize adapter weights from a pre-trained adapter (fresh optimizer). For curriculum learning.",
    )
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
        init_from_adapter=args.init_from,
    )


if __name__ == "__main__":
    main()
