#!/usr/bin/env python3
"""
train_gemma4.py — Unsloth QLoRA SFT for Gemma 4 E2B / E4B as EN<->ES translator.

UNTESTED — Unsloth install pending (Phase A1). The script is checked in for
review before its first execution. See docs/gemma4_tuning/phase_a_infrastructure.md.

Differences from the older training/train_gemma.py:
  - Unsloth FastModel (not vanilla AutoModelForCausalLM + bnb).
  - Production prompt format mirrors engines/llamacpp_engine.py:115-118 verbatim.
  - enable_thinking=False applied via chat template on EVERY example, baking
    non-thinking behavior into the adapter so we can drop the runtime workaround
    at engines/llamacpp_engine.py:138 (only after v1 is permanent — see Phase E6).
  - PLE (Per-Layer Embeddings) frozen — do NOT target embed_tokens with LoRA.
  - Vision/audio towers frozen (finetune_vision_layers=False).
  - r=8, alpha=8, dropout=0 (Unsloth's recommended Gemma 4 config).
  - Packing + max_seq_length=1024; effective batch 16 via grad_accum.

Usage:
    # Full corpus (Phase C):
    python training/train_gemma4.py \\
        --base unsloth/gemma-4-E4B-it \\
        --train-data bible_data/v1_corpus/mixed_25k.jsonl \\
        --epochs 2 \\
        --output fine_tuned_gemma4_e4b_v1

    # Spike with separate verse/sermon files (Phase B):
    python training/train_gemma4.py \\
        --base unsloth/gemma-4-E4B-it \\
        --verse-pairs hybrid_runs/data/verse_1800.jsonl \\
        --sermon-pairs hybrid_runs/data/sermon_1800.jsonl \\
        --output fine_tuned_gemma4_e4b_spike \\
        --max-steps 200

JSONL row format: {"en": "...", "es": "...", ...} or
                  {"en": "...", "es": "...", "direction": "en2es" | "es2en"}
The optional "direction" field controls language direction; default is en2es.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Reduce VRAM fragmentation for Gemma 4 PLE bf16 embeddings + bnb scratch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["USE_TF"] = "0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_gemma4")

# Mirror of engines/llamacpp_engine.py:_LANG_NAMES so prompt format matches production.
LANG_NAMES = {"en": "English", "es": "Spanish", "hi": "Hindi", "zh": "Chinese"}


def build_user_prompt(text: str, source_lang: str, target_lang: str) -> str:
    """Mirror of engines/llamacpp_engine.py lines 115-118.

    Any drift here breaks adapter quality in production. If the production
    engine prompt changes, update this function in lockstep.
    """
    src_name = LANG_NAMES.get(source_lang, source_lang)
    tgt_name = LANG_NAMES.get(target_lang, target_lang)
    return (
        f"Translate the following {src_name} text to {tgt_name}. Output only the translation, nothing else.\n\n{text}"
    )


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("skipping malformed JSON in %s line %d: %s", path, i, exc)
    return rows


def to_messages(row: dict) -> dict | None:
    """Convert a {en, es, [direction]} row into a chat-template messages dict.

    Returns None when the row is malformed (missing en or es).
    """
    en = row.get("en")
    es = row.get("es")
    if not en or not es:
        return None
    direction = row.get("direction", "en2es")
    if direction == "es2en":
        src, tgt, src_lang, tgt_lang = es, en, "es", "en"
    else:
        src, tgt, src_lang, tgt_lang = en, es, "en", "es"
    user_content = build_user_prompt(src, src_lang, tgt_lang)
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": tgt},
        ]
    }


def file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_training_rows(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    if args.train_data:
        rows.extend(load_jsonl(args.train_data))
        log.info("loaded %d rows from --train-data %s", len(rows), args.train_data)
    if args.verse_pairs:
        v = load_jsonl(args.verse_pairs)
        rows.extend(v)
        log.info("loaded %d rows from --verse-pairs %s", len(v), args.verse_pairs)
    if args.sermon_pairs:
        s = load_jsonl(args.sermon_pairs)
        rows.extend(s)
        log.info("loaded %d rows from --sermon-pairs %s", len(s), args.sermon_pairs)
    if args.glossary_pairs:
        g = load_jsonl(args.glossary_pairs)
        rows.extend(g)
        log.info("loaded %d rows from --glossary-pairs %s", len(g), args.glossary_pairs)
    if not rows:
        raise SystemExit("no training data; provide --train-data and/or --verse-pairs / --sermon-pairs")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.max_pairs and args.max_pairs > 0 and len(rows) > args.max_pairs:
        rows = rows[: args.max_pairs]
        log.info("subsampled to %d rows (--max-pairs)", len(rows))
    return rows


def save_manifest(args: argparse.Namespace, output_dir: Path, final_loss: float | None) -> None:
    manifest = {
        "run_id": output_dir.name,
        "timestamp": datetime.now(UTC).isoformat(),
        "base_model": args.base,
        "config": {
            "lr": args.lr,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "max_pairs": args.max_pairs,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "max_seq_length": args.max_seq_length,
            "per_device_train_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "packing": args.packing,
            "warmup_steps": args.warmup_steps,
            "seed": args.seed,
        },
        "data": {
            "train_data": str(args.train_data) if args.train_data else None,
            "verse_pairs": str(args.verse_pairs) if args.verse_pairs else None,
            "sermon_pairs": str(args.sermon_pairs) if args.sermon_pairs else None,
            "glossary_pairs": str(args.glossary_pairs) if args.glossary_pairs else None,
            "train_data_sha256": file_sha256(args.train_data),
            "verse_pairs_sha256": file_sha256(args.verse_pairs),
            "sermon_pairs_sha256": file_sha256(args.sermon_pairs),
        },
        "results": {"final_loss": final_loss},
        "adapter_sha256": file_sha256(output_dir / "adapter_model.safetensors"),
        "production_prompt_template": (
            "Translate the following {src} text to {tgt}. Output only the translation, nothing else.\n\n{text}"
        ),
        "enable_thinking": False,
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("wrote training_manifest.json")


def train(args: argparse.Namespace) -> None:
    # Imports deferred so --help works without Unsloth installed.
    try:
        from unsloth import FastModel
    except ImportError as exc:
        raise SystemExit(
            f"Unsloth not installed. Phase A1: pip install unsloth into stt_train_env. (import error: {exc})"
        ) from exc
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading base model %s (load_in_4bit=True, max_seq_length=%d)", args.base, args.max_seq_length)
    t0 = time.perf_counter()
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.base,
        load_in_4bit=True,
        max_seq_length=args.max_seq_length,
        full_finetuning=False,
    )
    log.info("base model loaded in %.1fs", time.perf_counter() - t0)

    log.info("attaching LoRA adapters (r=%d, alpha=%d)", args.lora_r, args.lora_alpha)
    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        # Do NOT target embed_tokens — Gemma 4 PLE must stay frozen.
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    rows = collect_training_rows(args)
    log.info("converting %d rows to chat-template messages", len(rows))
    ds_rows: list[dict] = []
    skipped = 0
    for r in rows:
        m = to_messages(r)
        if m is None:
            skipped += 1
            continue
        ds_rows.append(m)
    if skipped:
        log.warning("skipped %d malformed rows (missing en/es)", skipped)

    dataset = Dataset.from_list(ds_rows)

    # Apply chat template with enable_thinking=False on every example.
    def render(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return {"text": text}

    log.info("rendering chat template (enable_thinking=False)")
    dataset = dataset.map(render, remove_columns=["messages"])

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs if args.max_steps <= 0 else 1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        bf16=True,
        fp16=False,
        seed=args.seed,
        dataset_text_field="text",
        packing=args.packing,
        max_seq_length=args.max_seq_length,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    log.info("starting training")
    train_result = trainer.train()

    # Sanity gate from the plan: text-only loss must settle in 1-3 by step ~500.
    # 13-15 means multimodal got engaged. Higher than ~5 by mid-training means
    # the chat template / prompt format is wrong - re-check build_user_prompt
    # against engines/llamacpp_engine.py:115-138.
    final_loss = float(train_result.training_loss) if train_result.training_loss is not None else None
    log.info("final training loss: %s", final_loss)
    if final_loss is not None and final_loss > 5.0:
        log.error(
            "FINAL LOSS %.2f > 5.0 — likely chat template / prompt format mismatch. "
            "Verify build_user_prompt() matches engines/llamacpp_engine.py:115-138 exactly.",
            final_loss,
        )
    if final_loss is not None and final_loss > 10.0:
        log.error("FINAL LOSS %.2f > 10.0 — likely multimodal got engaged; abort and investigate.", final_loss)

    log.info("saving adapter to %s", output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    save_manifest(args, output_dir, final_loss)
    log.info("done")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--base",
        default="unsloth/gemma-4-E4B-it",
        help="HF model id of the base model (default: Unsloth E4B). Use unsloth/gemma-4-E2B-it for E2B.",
    )
    p.add_argument("--output", required=True, help="output directory for adapter + manifest")
    # Data: either a pre-mixed --train-data, or component files.
    p.add_argument("--train-data", type=Path, default=None, help="pre-mixed JSONL of {en, es, [direction]} rows")
    p.add_argument("--verse-pairs", type=Path, default=None, help="Bible verse pairs JSONL")
    p.add_argument("--sermon-pairs", type=Path, default=None, help="sermon parallel pairs JSONL")
    p.add_argument("--glossary-pairs", type=Path, default=None, help="glossary-tagged pairs JSONL")
    p.add_argument("--max-pairs", type=int, default=0, help="cap total training rows (0 = no cap)")
    # Training shape.
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=0, help="if > 0, override --epochs (useful for spikes)")
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument(
        "--grad-accum", type=int, default=8, help="gradient accumulation steps (effective batch = pdbs * grad_accum)"
    )
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--packing", action=argparse.BooleanOptionalAction, default=True)
    # LoRA.
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=8)
    # Optim.
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=5)
    # Misc.
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
