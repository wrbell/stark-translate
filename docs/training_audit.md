# `train_gemma.py` — Full Audit

**Hardware context:** 16 GB A2000 Ada (Ada Lovelace / sm_89), 30 GB shared RAM, Intel Xeon, Windows/WSL

---

## CRITICAL — Bugs & Correctness Issues

### 1. `bf16=True` will silently fall back or crash on your A2000

The A2000 Ada (sm_89) *does* have hardware BF16, so this *technically* works — but it's subtly suboptimal. The Ada generation cards have **2× throughput for FP16 vs BF16** on the CUDA cores (only the Tensor Cores are symmetric). Since QLoRA 4-bit compute goes through the CUDA cores for dequantization math, you're leaving performance on the table.

**Fix:** Change `bnb_4bit_compute_dtype` and the training flag to FP16:

```python
# In BitsAndBytesConfig:
bnb_4bit_compute_dtype=torch.float16,  # was torch.bfloat16

# In SFTConfig:
bf16=False,
fp16=True,
```

This alone could give you a **10-20% wall-clock speedup** on Ada.

### 2. `model.save_pretrained()` saves the entire quantized base model, not just LoRA adapters

Line 347: `model.save_pretrained(output_dir)` — when the model has a PEFT adapter attached, this saves the **full 4-bit quantized model weights + adapter** which is both enormous (~2-5 GB) and useless (you can't meaningfully reload a 4-bit quantized checkpoint for inference on different hardware). What you want is just the LoRA adapter weights (~20-50 MB).

**Fix:**
```python
# Replace:
model.save_pretrained(output_dir)

# With:
trainer.model.save_pretrained(output_dir)  # SFTTrainer unwraps PEFT correctly
# Or explicitly:
from peft import PeftModel
model.save_pretrained(output_dir, safe_serialization=True)  # only if PEFT-wrapped
```

Better yet, use `trainer.save_model(output_dir)` which handles PEFT unwrapping automatically:
```python
trainer.save_model(output_dir)
```

### 3. `resume_from_checkpoint` logic is broken

Line 382: `resume = True if args.resume else None`

`SFTTrainer.train(resume_from_checkpoint=True)` tells the Trainer to look for the latest checkpoint directory inside `output_dir`. But if `output_dir` doesn't have a `checkpoint-*` subdirectory yet, this will raise a `ValueError`. You should instead pass the actual checkpoint path, or just let `True` do its thing but guard against the missing-checkpoint case.

**Fix:**
```python
resume = True if args.resume else None
# Add a guard:
if resume and not any(
    d.startswith("checkpoint-") for d in os.listdir(output_dir or f"fine_tuned_gemma_mi_{args.approach}")
    if os.path.isdir(os.path.join(output_dir or f"fine_tuned_gemma_mi_{args.approach}", d))
):
    logger.warning("--resume passed but no checkpoints found. Starting fresh.")
    resume = None
```

### 4. `pad_token = eos_token` can degrade training quality

Line 208: Setting `pad_token = eos_token` means the loss function will compute gradients on padding tokens (since the label for pad positions will be `eos_token_id` rather than `-100`). With packing enabled this is less of an issue, but for any unpacked sequences it silently degrades the signal.

**Fix:** Also set the pad token ID in the model config and ensure the data collator ignores it:
```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
```

Since you have `packing=True`, this is partially mitigated — but it's still best practice to be explicit.

### 5. `save_metrics` will crash if `log_history` is empty

Line 352: `trainer.state.log_history[-1]` will `IndexError` if training somehow produces zero log entries (e.g., very small dataset with logging_steps=50 that finishes in < 50 steps). Guard it:

```python
if trainer.state.log_history:
    trainer.save_metrics("train", trainer.state.log_history[-1])
else:
    logger.warning("No training metrics to save.")
```

---

## HIGH — Performance Bottlenecks

### 6. `gradient_checkpointing=True` without `use_reentrant=False`

PyTorch's default reentrant checkpointing is slower and has known bugs with PEFT/LoRA. The non-reentrant version is faster and more correct.

**Fix:**
```python
gradient_checkpointing=True,
gradient_checkpointing_kwargs={"use_reentrant": False},
```

This is a **free speedup** (5-10%) and avoids subtle gradient bugs.

### 7. `paged_adamw_32bit` — use the 8-bit version instead

The 32-bit paged optimizer stores full-precision optimizer states in system RAM when VRAM overflows. On your 16 GB A2000, this means optimizer states page out to your Xeon's RAM over PCIe, which is extremely slow. The 8-bit version (`paged_adamw_8bit`) halves the optimizer state memory with negligible quality loss, keeping more states on-GPU.

**Fix:**
```python
optim="paged_adamw_8bit",  # was paged_adamw_32bit
```

This will **halve optimizer memory** (~1-2 GB savings) and reduce PCIe paging.

### 8. `dataloader_num_workers=4` may hurt on Windows/WSL

On Windows (and WSL1), multiprocessing for dataloaders has high overhead due to spawn-based process creation. With `packing=True` and text data (no images), the dataloader is not the bottleneck — the GPU is. Workers will consume RAM and CPU for no benefit.

**Fix:**
```python
dataloader_num_workers=2,  # or 0 for WSL1; 2 is fine for WSL2
dataloader_pin_memory=True,  # ensure DMA transfers
```

### 9. No `torch.compile` — free 10-20% on Ada

Your A2000 Ada (sm_89) fully supports `torch.compile`. Adding it gives automatic kernel fusion.

**Fix — add to SFTConfig:**
```python
torch_compile=True,
```

Or apply it before training:
```python
model = torch.compile(model)
```

Note: first epoch will be slower due to compilation. Net savings over a full 3-epoch run are significant.

### 10. VRAM caps are leaving memory on the table

```python
VRAM_CAP = "15GiB"
SHARED_MEM_CAP = "12GiB"
```

You said you have 30 GB of shared RAM. The `"cpu"` key in `max_memory` is the overflow budget. With 12 GiB you're only using 40% of what's available for CPU offload. If the 12B model barely fits, bump this up.

**Fix:**
```python
SHARED_MEM_CAP = "24GiB"  # Use 80% of 30GB shared, leave room for OS/dataset
```

### 11. No `eval_dataset` — you're flying blind

There's no validation split. You can train for 3 epochs and have no idea if you're overfitting (especially with the 15× glossary oversampling). This is a correctness issue masquerading as a missing feature.

**Fix:**
```python
# After creating full_ds:
split = full_ds.train_test_split(test_size=0.05, seed=42)
train_ds = split["train"]
eval_ds = split["test"]

# In SFTConfig:
eval_strategy="steps",
eval_steps=500,

# In SFTTrainer:
train_dataset=train_ds,
eval_dataset=eval_ds,
```

### 12. `max_seq_length=512` — likely truncating data

Bible verses are short, but the TranslateGemma chat template adds significant overhead (special tokens, role markers, language codes). A verse pair that's 80 tokens of content can easily become 150+ tokens after formatting. With packing, multiple examples get concatenated, and the packing target is 512. This is fine for packing density, but verify that individual formatted examples don't exceed 512 before packing — if they do, they get silently truncated.

**Fix — add a validation check:**
```python
# After formatting, check for truncation:
lengths = [len(tokenizer.encode(ex["text"])) for ex in full_ds]
max_len = max(lengths)
over_limit = sum(1 for l in lengths if l > max_seq_length)
logger.info(f"  Token lengths: max={max_len}, median={sorted(lengths)[len(lengths)//2]}")
if over_limit > 0:
    logger.warning(f"  {over_limit} examples exceed max_seq_length={max_seq_length} and will be truncated!")
```

---

## MEDIUM — Efficiency & Robustness

### 13. The `.map()` call is single-threaded

Line 309: `full_ds.map(format_for_translategemma, ...)` runs in a single process. For 20K+ examples this is fine (seconds), but for 50K+ or the full 155K corpus it'll take minutes.

**Fix:**
```python
full_ds = full_ds.map(
    format_for_translategemma,
    remove_columns=full_ds.column_names,
    num_proc=4,  # parallelize formatting
    desc="Formatting for TranslateGemma",
)
```

### 14. Glossary oversampling creates duplicate tokens in the packing buffer

With `packing=True`, 15× duplicated glossary entries will be packed together into the same 512-token sequences. The model sees near-identical gradient signals many times per batch, which reduces the *effective* batch diversity. This is the worst kind of oversampling for packed training.

**Fix — use a custom sampler or reduce oversampling:**
```python
GLOSSARY_OVERSAMPLE = 5   # Reduce from 15 to 5
# Better: add glossary pairs with slightly varied prompting:
# "Translate this theological term:" vs "Translate:" vs bare pair
```

Or disable packing for glossary-heavy datasets and use a weighted random sampler instead.

### 15. No `seed` set in SFTConfig

The training itself is not seeded, making runs non-reproducible.

```python
seed=42,
```

### 16. `max_grad_norm=0.3` is very aggressive

A gradient clipping norm of 0.3 is unusually tight. Standard QLoRA fine-tuning uses 1.0. Overly aggressive clipping slows convergence — you're throwing away gradient signal each step.

**Fix:**
```python
max_grad_norm=1.0,  # standard for QLoRA
```

### 17. Learning rate may be too high for domain adaptation

`lr=2e-4` is fine for training from scratch or heavy fine-tuning, but for domain adaptation on a model that already translates well, this is aggressive. The ALMA-R paper used `2e-5` for their CPT stage and `5e-6` for the preference tuning stage.

**Suggestion:**
```python
lr=5e-5,  # less aggressive for domain adaptation; 2e-4 risks catastrophic forgetting
```

At minimum, log the learning rate schedule and watch for loss spikes.

### 18. `warmup_ratio=0.03` is very short

With 20K samples, effective batch size 8, and 3 epochs, you have ~7,500 steps. 3% warmup = ~225 steps. For a model that already has strong priors, a longer warmup (5-10%) helps avoid early destabilization.

```python
warmup_ratio=0.06,
```

### 19. VRAM monitor only fires on `on_log` (every 50 steps)

If VRAM spikes between logging steps and triggers OOM, you'll never see the warning. Consider also monitoring on `on_step_end` at a lower frequency:

```python
def on_step_end(self, args, state, control, **kwargs):
    if state.global_step % 10 == 0:  # lighter than every step
        # same monitoring logic
```

### 20. No early stopping

With no eval set and no early stopping, you're committed to all 3 epochs even if the model converged in epoch 1. Combined with aggressive oversampling, this risks overfitting.

```python
from transformers import EarlyStoppingCallback

callbacks=[VRAMMonitorCallback(), EarlyStoppingCallback(early_stopping_patience=3)],
```

(Requires adding an eval dataset per suggestion #11.)

---

## LOW — Code Quality & Defensive Fixes

### 21. CUDA memory caps should be set *before* any torch.cuda calls

Lines 72-73 set Python constants, but the actual CUDA memory limits should be configured via environment variables before PyTorch initializes CUDA:

```python
# At the very top, before `import torch`:
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

This enables PyTorch's expandable segment allocator which reduces fragmentation on cards with limited VRAM.

### 22. No `CUDA_VISIBLE_DEVICES` guard

If the machine has multiple GPUs (unlikely but possible), the script will use whatever `device_map="auto"` picks. Be explicit:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

### 23. The exception handler in `format_for_translategemma` is too broad

Line 299: `except Exception` swallows all errors including `KeyError`, `TypeError`, etc. If the chat template has a real bug, you'll silently get misformatted training data.

**Fix:**
```python
except (jinja2.exceptions.TemplateError, KeyError) as e:
    logger.warning(f"Chat template failed ({e}), using fallback format")
    formatted = ...
```

### 24. Missing `torch.cuda.empty_cache()` before training

After loading the model and before `trainer.train()`, clear the CUDA cache to reclaim fragmented memory:

```python
torch.cuda.empty_cache()
logger.info("Starting training...")
```

### 25. `lora_alpha=16` with `lora_r=16` means scaling factor = 1.0

The LoRA scaling factor is `alpha/r = 16/16 = 1.0`. This is fine, but the more common practice for domain adaptation is `alpha = 2*r` (scaling = 2.0) to give the adapter more influence. Worth experimenting with:

```python
lora_alpha=32,  # scaling factor = 2.0
```

### 26. No `neftune_noise_alpha` — free quality boost

NEFTune adds noise to embeddings during training and has been shown to improve fine-tuning quality with negligible overhead.

```python
# In SFTConfig:
neftune_noise_alpha=5,
```

---

## Summary — Quick-Win Priority List

| # | Fix | Impact | Effort |
|---|-----|--------|--------|
| 1 | FP16 instead of BF16 | ~15% faster | 2 lines |
| 7 | `paged_adamw_8bit` | ~1-2 GB VRAM saved | 1 line |
| 6 | `use_reentrant=False` | ~5-10% faster | 1 line |
| 9 | `torch.compile` | ~10-20% faster | 1 line |
| 2 | Fix `save_pretrained` | Correct checkpoint saving | 1 line |
| 16 | `max_grad_norm=1.0` | Faster convergence | 1 line |
| 11 | Add eval split | Know if you're overfitting | 6 lines |
| 17 | Lower learning rate to 5e-5 | Avoid catastrophic forgetting | 1 line |
| 21 | `expandable_segments:True` | Reduce VRAM fragmentation | 1 line |
| 26 | `neftune_noise_alpha=5` | Free quality boost | 1 line |

The top 4 items alone (FP16, 8-bit optimizer, non-reentrant checkpointing, torch.compile) should give you roughly a **30-40% wall-clock speedup** and save **1-2 GB of VRAM** — meaningful headroom on a 16 GB card.
