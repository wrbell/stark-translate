# Patch Instructions for `training/train_gemma.py`

## Context

The adapters produced by the current script perform **worse** than the base TranslateGemma model. Root cause: catastrophic forgetting from an excessively high learning rate (2e-4), extreme glossary oversampling (15×), too many epochs (3), and no validation monitoring. TranslateGemma is already a strong translator — it needs gentle domain adaptation, not aggressive retraining.

## Hardware

- NVIDIA RTX A2000 Ada (16 GB VRAM, sm_89, Ada Lovelace)
- 30 GB shared system RAM available for GPU overflow
- Intel Xeon CPU
- Windows / WSL2

## Rules

1. Follow every step in order.
2. Do not rename functions, change model names, change the quantization config type, or restructure the file layout.
3. Do not add `flash_attention_2`, `use_dora`, or `modules_to_save` — these will OOM or degrade quality on this hardware.
4. Output the **complete modified script** when done.

---

## STEP 1 — Environment variable (after line 42)

Insert this line immediately after the existing `os.environ["USE_TF"] = "0"` line:

```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Reduce VRAM fragmentation on 16GB card
```

Do NOT add a duplicate `import os` — it is already imported on line 38.

---

## STEP 2 — Global constants (line 68)

Replace:
```python
GLOSSARY_OVERSAMPLE = 15
```

With:
```python
GLOSSARY_OVERSAMPLE = 3  # Reduced from 15 — prevents overfitting to glossary lookup patterns
```

No other constants change. Keep `VRAM_CAP`, `SHARED_MEM_CAP`, `VRAM_WARN_PERCENT` as-is.

---

## STEP 3 — Function signature defaults (line 152–165)

Replace the function signature with:

```python
def fine_tune_gemma(
    approach="A",
    bible_data="bible_data/aligned/verse_pairs_train.jsonl",
    glossary_data="bible_data/glossary/glossary_pairs.jsonl",
    sermon_data=None,
    output_dir=None,
    lora_r=16,
    lora_alpha=32,          # Changed from 16 — scaling factor = alpha/r = 2.0
    epochs=1,               # Changed from 3 — one pass to prevent catastrophic forgetting
    lr=1e-5,                # Changed from 2e-4 — 20× lower for domain adaptation on a strong model
    max_seq_length=512,
    max_pairs=20000,
    resume_from_checkpoint=None,
):
```

Also update the docstring inside the function to reflect the new defaults. Replace the lines about `r=16 for domain adaptation` and `Paged AdamW 32-bit optimizer` and the data scaling notes to match. Specifically:
- Change `r=16, alpha=16` references to `r=16, alpha=32`
- Change `Paged AdamW 32-bit` to `Paged AdamW 8-bit`
- Change `Glossary pairs are always fully included with 15x oversampling` to `3x oversampling`
- Change `Sermon pairs (from live sessions) are always fully included at 2x weight` to `1x (no oversampling)`

---

## STEP 4 — Model loading (line 211–216)

Replace the `model = AutoModelForCausalLM.from_pretrained(...)` call with:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,                        # Explicit dtype for stability
    max_memory={0: VRAM_CAP, "cpu": SHARED_MEM_CAP},
    # Do NOT add attn_implementation — let transformers default to SDPA (safest on A2000 Ada + Gemma 3)
)
```

---

## STEP 5 — Pad token setup (after tokenizer loading, around line 207–208)

Replace:
```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

With:
```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
```

---

## STEP 6 — LoRA config (line 219–225)

Replace the entire `peft_config = LoraConfig(...)` block with:

```python
peft_config = LoraConfig(
    r=lora_r,                   # 16 — full capacity for theological domain
    lora_alpha=lora_alpha,      # 32 — scaling factor = 2.0
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    # No modules_to_save — training lm_head/embed_tokens full-rank would OOM on 16GB and cause forgetting
    # No use_dora — adds ~15-20% memory overhead, not needed for 1-epoch adaptation
)
```

---

## STEP 7 — Sermon pair loading (lines 259–269)

Replace the sermon loading block with:

```python
# --- Sermon pairs from live sessions (always included, highest value) ---
if sermon_data and os.path.exists(sermon_data):
    with open(sermon_data) as f:
        sermon_pairs = [json.loads(line) for line in f]
    logger.info(f"  Sermon pairs: {len(sermon_pairs)} (real-domain, included 1x)")
    all_pairs.extend(sermon_pairs)
elif sermon_data:
    logger.info(f"  Sermon data not found at {sermon_data} (skipping)")
```

The previous `* 2` doubling is removed. Sermon data is already high-value; oversampling it biases the model toward a narrow register.

---

## STEP 8 — Chat template verification (inside `format_for_translategemma`, lines 286–307)

Replace the entire `format_for_translategemma` function with:

```python
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
        # Narrow exception — don't silently swallow real bugs
        logger.warning(f"Chat template failed ({e}), using fallback format")
        formatted = (
            f"<start_of_turn>user\n"
            f"Translate from English to Spanish:\n{example['en']}"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n{example['es']}<end_of_turn>"
        )
    return {"text": formatted}
```

Then, immediately after defining this function and BEFORE the `.map()` call, add a one-shot template verification:

```python
# Verify which formatting path fires (critical — wrong template = guaranteed regression)
_test_msg = [
    {"role": "user", "content": [
        {"type": "text", "source_lang_code": "en", "target_lang_code": "es", "text": "test"}
    ]},
    {"role": "assistant", "content": "prueba"},
]
try:
    _test_result = tokenizer.apply_chat_template(_test_msg, tokenize=False)
    logger.info(f"  Chat template OK. Sample: {_test_result[:120]}...")
except Exception as e:
    logger.warning(f"  Chat template FAILED ({e}) — all examples will use fallback format!")
    logger.warning(f"  This likely means training data format won't match the model's expected input.")
```

---

## STEP 9 — Validation split (after `full_ds` is created on line 309, before SFTConfig)

After the `.map()` call that creates the final `full_ds`, add:

```python
# Held-out validation split — mandatory to detect forgetting early
split_ds = full_ds.train_test_split(test_size=0.05, seed=42)
train_dataset = split_ds["train"]
eval_dataset = split_ds["test"]
logger.info(f"  Train: {len(train_dataset)} examples, Eval: {len(eval_dataset)} examples")
```

---

## STEP 10 — SFTConfig (lines 312–331)

Replace the entire `training_args = SFTConfig(...)` block with:

```python
training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    learning_rate=lr,
    warmup_ratio=0.1,                                       # Changed from 0.03 — longer warmup avoids early destabilization
    lr_scheduler_type="cosine",
    max_grad_norm=0.5,                                      # Changed from 0.3 — less aggressive clipping
    optim="paged_adamw_8bit",                               # Changed from 32bit — halves optimizer state memory (~1-2 GB saved)
    bf16=True,
    max_seq_length=max_seq_length,
    packing=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # New — faster + avoids subtle gradient bugs with PEFT
    logging_steps=50,
    eval_strategy="steps",                                   # New — monitor forgetting during training
    eval_steps=200,                                          # New
    save_strategy="steps",
    save_steps=400,
    save_total_limit=3,                                      # Keep — prevents checkpoint accumulation filling disk
    dataloader_num_workers=4,
    dataloader_prefetch_factor=2,                            # New — smoother data pipeline
    seed=42,                                                 # New — reproducible runs
    report_to="none",
)
```

---

## STEP 11 — SFTTrainer (lines 333–340)

Replace the entire `trainer = SFTTrainer(...)` call with:

```python
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,     # Changed from full_ds — now uses the 95% train split
    eval_dataset=eval_dataset,       # New — the 5% held-out split
    peft_config=peft_config,
    processing_class=tokenizer,
    callbacks=[VRAMMonitorCallback()],
)
```

---

## STEP 12 — Pre-training cache clear (before `trainer.train()`, around line 342)

Add before the `trainer.train()` call:

```python
torch.cuda.empty_cache()
```

---

## STEP 13 — Fix model saving (lines 347–349)

Replace:
```python
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
logger.info(f"TranslateGemma QLoRA adapters ({approach}) saved to {output_dir}")
```

With:
```python
# Save only the LoRA adapter weights (~20-50 MB), not the full quantized model (~2-5 GB)
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
logger.info(f"TranslateGemma QLoRA adapters ({approach}) saved to {output_dir}")
```

---

## STEP 14 — Guard save_metrics (line 352)

Replace:
```python
trainer.save_metrics("train", trainer.state.log_history[-1] if trainer.state.log_history else {})
```

With:
```python
if trainer.state.log_history:
    trainer.save_metrics("train", trainer.state.log_history[-1])
else:
    logger.warning("No training metrics to save (training may have been too short for a log entry).")
```

---

## STEP 15 — Argparse defaults (lines 366–369)

Update these four lines to match the new function signature defaults:

```python
parser.add_argument("--lora-r", type=int, default=16)
parser.add_argument("--lora-alpha", type=int, default=32)     # Changed from 16
parser.add_argument("--epochs", type=int, default=1)           # Changed from 3
parser.add_argument("--lr", type=float, default=1e-5)          # Changed from 2e-4
```

---

## STEP 16 — Resume checkpoint guard (line 382)

Replace:
```python
resume = True if args.resume else None
```

With:
```python
resume = None
if args.resume:
    _out = args.output or f"fine_tuned_gemma_mi_{args.approach}"
    if os.path.isdir(_out) and any(d.startswith("checkpoint-") for d in os.listdir(_out)):
        resume = True
    else:
        logger.warning(f"--resume passed but no checkpoints found in {_out}. Starting fresh.")
```

---

## STEP 17 — Docstring at top of file (lines 1–33)

Update the module docstring to reflect the new defaults. Key changes:
- `Config: r=16, alpha=32` (was alpha=16)
- `--max-pairs 20000   Start here (default). ~1-2 GPU hrs for 4B.` (was 3-4 hrs, shorter with 1 epoch)
- Remove the `python train_gemma.py A --epochs 5 --lr 1e-4` usage example (dangerously high)
- Replace with: `python train_gemma.py A --lr 5e-6          # If 1e-5 still slightly degrades`

---

## Summary of all changes

| What | Old | New | Why |
|------|-----|-----|-----|
| `GLOSSARY_OVERSAMPLE` | 15 | 3 | Prevents rigid glossary-lookup overfitting |
| `lora_alpha` | 16 | 32 | Scaling factor 2.0 gives adapter more influence per step |
| `epochs` | 3 | 1 | One pass prevents catastrophic forgetting of pretrained weights |
| `lr` | 2e-4 | 1e-5 | 20× lower — the single most important fix |
| `warmup_ratio` | 0.03 | 0.1 | Longer warmup avoids early destabilization |
| `max_grad_norm` | 0.3 | 0.5 | Less aggressive clipping preserves gradient direction |
| `optim` | paged_adamw_32bit | paged_adamw_8bit | Halves optimizer state memory (~1-2 GB saved) |
| `gradient_checkpointing_kwargs` | (none) | `use_reentrant=False` | Faster, avoids PEFT gradient bugs |
| Sermon oversampling | 2× | 1× (no duplication) | Removes narrow register bias |
| Eval split | None | 5% held-out | Monitors forgetting during training |
| `model.save_pretrained()` | Saves full quantized model | `trainer.save_model()` | Saves only LoRA adapters (~20-50 MB) |
| Chat template except | Bare `Exception` | Narrow types + verification log | Detects broken template before training |
| `PYTORCH_CUDA_ALLOC_CONF` | (none) | `expandable_segments:True` | Reduces VRAM fragmentation |
| `torch.cuda.empty_cache()` | (none) | Before trainer.train() | Reclaims fragmented memory |
| `save_metrics` | Unguarded `[-1]` | Guarded with `if` | Prevents IndexError on short runs |
| Resume guard | Unguarded `True` | Checks for checkpoint dirs | Prevents ValueError on fresh output dir |
| `seed` | (none) | 42 | Reproducible runs |
| `save_total_limit` | 3 | 3 | Preserved (was accidentally dropped in earlier feedback) |
| `dataloader_prefetch_factor` | (none) | 2 | Smoother data pipeline |
| `pad_token_id` | Only on tokenizer | Also on model.config | Ensures consistent padding behavior |

## Run commands

```bash
# Quick rescue test (~30-45 min on A2000 Ada)
python train_gemma.py A --max-pairs 8000 --lr 1e-5

# Full rescue run (~1-2 hrs)
python train_gemma.py A --max-pairs 20000 --lr 1e-5

# If 1e-5 still slightly degrades, try:
python train_gemma.py A --max-pairs 20000 --lr 5e-6
```

After training, evaluate both the base model and the new adapter using the **exact same** prompt format (with `source_lang_code`/`target_lang_code`) at `temperature=0.0`.

## What NOT to do

- Do NOT set `lr` ≥ 2e-5 on the first run
- Do NOT set `epochs` > 1 on the first run
- Do NOT add `modules_to_save=["lm_head", "embed_tokens"]` — will OOM and cause forgetting
- Do NOT add `use_dora=True` — adds ~15-20% memory overhead, not needed yet
- Do NOT add `attn_implementation="flash_attention_2"` — flaky on WSL + Gemma 3 sliding window attention
- Do NOT reduce `lora_r` below 16 — the adapter needs capacity for the theological domain
- Do NOT reduce `max_seq_length` below 512 — hurts packing density with short Bible verses
