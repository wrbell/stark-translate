#!/usr/bin/env python3
"""
train_gemma4_cpo.py — Phase D contrastive preference optimization on top of v1.1 SFT.

UNTESTED at training time — checked in for review before first run. See
docs/gemma4_tuning/phase_d_preference_optimization.md.

Approach:
  - Load the v1.1 base model (either an HF model id pointing to a merged-bf16
    checkpoint, or `--base unsloth/gemma-4-E4B-it` + `--init-adapter
    fine_tuned_gemma4_e4b_v1.1` to merge in-memory). Loaded 4-bit via Unsloth.
  - Attach a fresh LoRA (same r=8, alpha=8 as SFT — keeps the parameter count
    light and inference latency unchanged).
  - Train CPO via TRL's CPOTrainer on (prompt, chosen, rejected) triples
    produced by `tools/build_preference_triples.py score`.

CPO hyperparameters per the plan + ALMA paper (arxiv.org/abs/2401.08417):
  - learning_rate=5e-6  (10x lower than SFT — preference signal is noisier)
  - beta=0.1            (CPO temperature; ALMA default)
  - num_train_epochs=1  (1 epoch per ALMA — multiple epochs hurt)

Sanity gate: chosen-reward should rise above rejected-reward by ~step 50;
loss should be monotonic-decreasing. If chosen and rejected stay within 0.1
of each other after 100 steps, CPO isn't learning the preference signal —
abort and check the triples (margin too narrow? data corrupted?).

Usage:
    HF_HOME=/mnt/d/Data/stt-data/cache /home/wbell/unsloth_env/bin/python \\
        training/train_gemma4_cpo.py \\
        --base unsloth/gemma-4-E4B-it \\
        --init-adapter fine_tuned_gemma4_e4b_v1.1 \\
        --triples preference/v1.1_triples.jsonl \\
        --output fine_tuned_gemma4_e4b_v2_cpo \\
        --epochs 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["USE_TF"] = "0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_gemma4_cpo")


def load_triples(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def train(args: argparse.Namespace) -> None:
    try:
        from unsloth import FastModel
    except ImportError as exc:
        raise SystemExit(f"Unsloth not installed in this venv. (import error: {exc})") from exc
    from datasets import Dataset
    from trl import CPOConfig, CPOTrainer

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading base %s in 4-bit (max_seq_length=%d)", args.base, args.max_seq_length)
    t0 = time.perf_counter()
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.base,
        load_in_4bit=True,
        max_seq_length=args.max_seq_length,
        full_finetuning=False,
    )
    log.info("base loaded in %.1fs", time.perf_counter() - t0)

    if args.init_adapter:
        # Load the v1.1 SFT adapter as the starting point. Unsloth supports this via
        # FastModel.from_pretrained(adapter_name) or via peft.PeftModel.load_adapter.
        log.info("attaching init adapter from %s", args.init_adapter)
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.init_adapter), is_trainable=True)
        log.info("init adapter attached and marked trainable")
    else:
        log.info("no init adapter — attaching fresh LoRA r=%d alpha=%d", args.lora_r, args.lora_alpha)
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
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )

    # Build dataset. CPOTrainer expects {prompt, chosen, rejected} columns.
    log.info("loading triples from %s", args.triples)
    triples = load_triples(args.triples)
    log.info(
        "loaded %d triples (margin distribution: min=%.3f median=%.3f max=%.3f)",
        len(triples),
        min(t["margin"] for t in triples),
        sorted(t["margin"] for t in triples)[len(triples) // 2],
        max(t["margin"] for t in triples),
    )

    # CPOTrainer wants the prompt as the user-formatted text; the trainer will
    # itself wrap it in the chat template. To match production exactly, render
    # the chat template here too and pass the rendered string.
    def render(t: dict) -> dict:
        # The prompt field already contains the production user prompt
        # (built by build_preference_triples.build_user_prompt). Wrap in chat template.
        wrapped = tokenizer.apply_chat_template(
            [{"role": "user", "content": t["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return {"prompt": wrapped, "chosen": t["chosen"], "rejected": t["rejected"]}

    dataset = Dataset.from_list([render(t) for t in triples])
    log.info("dataset: %d rows, columns=%s", len(dataset), dataset.column_names)

    cpo_config = CPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",
        beta=args.beta,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        bf16=True,
        fp16=False,
        seed=args.seed,
        report_to="none",
    )

    # CPOTrainer calls processing_class(prompt, ...) positionally. For Gemma 4
    # the FastModel returns Gemma4Processor (multimodal), whose __call__ takes
    # `images` as the first positional. Passing the processor causes prompts
    # to be misinterpreted as images and text=None -> NoneType subscript error.
    # Solution: pass the inner GemmaTokenizer (text-only) which takes text first.
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    log.info("CPOTrainer using text tokenizer: %s", type(text_tokenizer).__name__)
    trainer = CPOTrainer(
        model=model,
        tokenizer=text_tokenizer,
        train_dataset=dataset,
        args=cpo_config,
    )

    log.info(
        "starting CPO training (lr=%s, beta=%s, epochs=%d, %d triples)", args.lr, args.beta, args.epochs, len(triples)
    )
    train_result = trainer.train()

    final_loss = float(train_result.training_loss) if train_result.training_loss is not None else None
    log.info("final CPO loss: %s", final_loss)

    log.info("saving adapter -> %s", output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Manifest
    manifest = {
        "run_id": output_dir.name,
        "timestamp": datetime.now(UTC).isoformat(),
        "kind": "cpo",
        "base_model": args.base,
        "init_adapter": str(args.init_adapter) if args.init_adapter else None,
        "config": {
            "lr": args.lr,
            "beta": args.beta,
            "epochs": args.epochs,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "max_seq_length": args.max_seq_length,
            "max_prompt_length": args.max_prompt_length,
            "per_device_train_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "warmup_steps": args.warmup_steps,
            "seed": args.seed,
        },
        "data": {
            "triples_path": str(args.triples),
            "triples_sha256": file_sha256(args.triples),
            "n_triples": len(triples),
        },
        "results": {"final_loss": final_loss},
        "adapter_sha256": file_sha256(output_dir / "adapter_model.safetensors"),
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("wrote training_manifest.json")
    log.info("done")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="unsloth/gemma-4-E4B-it", help="base model HF id")
    p.add_argument(
        "--init-adapter",
        type=Path,
        default=None,
        help="LoRA adapter dir to start from (e.g. fine_tuned_gemma4_e4b_v1.1). "
        "If omitted, attaches a fresh LoRA on top of the base.",
    )
    p.add_argument("--triples", type=Path, required=True, help="JSONL of {prompt, chosen, rejected, ...}")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=1, help="CPO is typically 1 epoch (ALMA recipe)")
    p.add_argument("--per-device-batch-size", type=int, default=1, help="CPO uses bs=1, grad_accum for effective batch")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--max-prompt-length", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-6, help="10x lower than SFT")
    p.add_argument("--beta", type=float, default=0.1, help="CPO temperature; ALMA default")
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--lora-r", type=int, default=8, help="ignored if --init-adapter is set")
    p.add_argument("--lora-alpha", type=int, default=8, help="ignored if --init-adapter is set")
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
