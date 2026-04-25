---

**You are an expert LLM fine-tuning engineer specializing in Unsloth + TranslateGemma (March 2026 best practices).**

Your task is to completely rewrite the file `training/train_gemma.py` into a new file called **`train_gemma_unsloth.py`**.

The new script must preserve 100 % of the original command-line interface and functionality while switching to Unsloth for speed, lower VRAM, and stability on a 16 GB A2000 Ada. It must also incorporate every lesson from the three failed HF runs (catastrophic forgetting, packing cross-contamination, oversampling toxicity, etc.).

**Exact requirements (follow in strict order, do not deviate or add extra features):**

1. Keep the **exact same** argparse structure and default values from the latest patched `train_gemma.py`. Support all original arguments (`approach`, `bible_data`, `glossary_data`, `sermon_data`, `output_dir`, `max_pairs`, `lr`, `epochs`, `lora_r`, `lora_alpha`, etc.).

2. **Add two new command-line arguments** (use `parser.add_argument`):
   ```python
   parser.add_argument("--run-test-matrix", action="store_true", help="Run the built-in hyperparameter test matrix instead of single training")
   parser.add_argument("--matrix-pairs", type=int, default=3000, help="Number of pairs to use per matrix experiment (keep small for speed)")
   ```

3. Add this global constant right after imports:
   ```python
   GLOSSARY_OVERSAMPLE = 3      # Critical — prevents rigid glossary mode and catastrophic forgetting
   ```

4. Preserve the **exact latest data-loading logic** from the patched version (Bible JSONL + glossary oversampled by `GLOSSARY_OVERSAMPLE` + sermon data loaded **once** only).

5. Right after `full_ds = Dataset.from_list(all_pairs)` (and any `.map()`), insert the 5 % validation split exactly as before:
   ```python
   # 5% held-out validation split (mandatory to detect forgetting early)
   split_ds = full_ds.train_test_split(test_size=0.05, seed=42)
   train_dataset = split_ds["train"]
   eval_dataset = split_ds["test"]
   ```

6. Load the model **exactly** with Unsloth (March 2026 recommended way for TranslateGemma / Gemma 3):
   ```python
   from unsloth import FastLanguageModel
   import torch

   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name=model_name,
       max_seq_length=2048,
       dtype=None,                    # auto BF16 on Ada
       load_in_4bit=True,
   )

   model = FastLanguageModel.get_peft_model(
       model,
       r=lora_r,
       target_modules="all-linear",
       lora_alpha=lora_alpha,
       lora_dropout=0.05,
       bias="none",
       use_gradient_checkpointing="unsloth",  # Unsloth's faster version
       random_state=3407,
       use_rslora=False,
       loftq_config=None,
   )
   ```

7. Use `SFTTrainer` from `trl` (Unsloth is fully compatible and fastest this way) with this `SFTConfig` (replace the whole block):
   ```python
   training_args = SFTConfig(
       output_dir=output_dir,
       per_device_train_batch_size=2,
       gradient_accumulation_steps=8,
       learning_rate=lr,
       num_train_epochs=epochs,
       max_seq_length=2048,
       packing=True,                          # Unsloth packing is stable
       optim="paged_adamw_8bit",
       bf16=True,
       gradient_checkpointing=True,
       warmup_ratio=0.1,
       max_grad_norm=0.5,
       lr_scheduler_type="cosine",
       logging_steps=50,
       eval_strategy="steps",
       eval_steps=150,
       save_strategy="steps",
       save_steps=300,
       save_total_limit=2,
       report_to="none",
   )
   ```

8. In the `SFTTrainer` call pass **both** `train_dataset` and `eval_dataset`.

9. Use the **exact same** chat-template formatting function from the original script (messages with `source_lang_code` / `target_lang_code`). Do not change it.

10. At the end of training save in standard PEFT format:
    ```python
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    ```

**NEW REQUIREMENT — TEST MATRIX (must be implemented)**

When `--run-test-matrix` is passed, **ignore** normal training and instead automatically run the following 8 quick experiments (each using `--matrix-pairs` examples and exactly 1 epoch). Save each in a subfolder like `output_dir/matrix_run_2_lr5e6_r16_os3`.

Embed this exact matrix in the code (use a list of dicts):

```python
TEST_MATRIX = [
    {"name": "base",          "lr": 5e-6, "oversample": 3, "r": 16, "alpha": 32, "packing": True},
    {"name": "lower_lr",      "lr": 3e-6, "oversample": 3, "r": 16, "alpha": 32, "packing": True},
    {"name": "higher_rank",   "lr": 5e-6, "oversample": 3, "r": 32, "alpha": 64, "packing": True},
    {"name": "less_os",       "lr": 5e-6, "oversample": 1, "r": 16, "alpha": 32, "packing": True},
    {"name": "more_os",       "lr": 5e-6, "oversample": 5, "r": 16, "alpha": 32, "packing": True},
    {"name": "conservative",  "lr": 2e-6, "oversample": 2, "r": 8,  "alpha": 16, "packing": True},
    {"name": "no_packing",    "lr": 5e-6, "oversample": 3, "r": 16, "alpha": 32, "packing": False},
    {"name": "balanced",      "lr": 4e-6, "oversample": 3, "r": 24, "alpha": 48, "packing": True},
]
```

For each run:
- Print a clear header: `=== MATRIX RUN X/8: {name} (lr={lr}, oversample={oversample}, r={r}) ===`
- Temporarily override the relevant parameters
- Train
- Print final train loss + eval loss
- Save adapter to `output_dir/matrix_{name}`

After all runs finish, print a clean markdown summary table of the 8 runs (name, lr, oversample, r, final train loss, eval loss).

**Do NOT**:
- Train `lm_head` or `embed_tokens`
- Use DoRA
- Change the data pipeline logic
- Add any other features

**Output ONLY the complete, ready-to-run `train_gemma_unsloth.py` script** (no explanations, no diffs, no markdown outside the code). Include clear comments explaining the key choices (Unsloth kernels, low LR, reduced oversampling, test matrix, etc.).

The script must support these two commands:
- Normal run: `python train_gemma_unsloth.py A --max-pairs 8000 --lr 5e-6`
- Matrix sweep: `python train_gemma_unsloth.py A --run-test-matrix --matrix-pairs 3000`

---

## Implementation Review (2026-03-20)

Review of this spec against the current state of `train_gemma.py` (post-ablation patches)
and lessons learned from the Phase 1 ablation study.

### Current Environment

- GPU: NVIDIA RTX 2000 Ada, 17.2GB VRAM, compute capability 8.9
- Stack: PyTorch 2.6.0, CUDA 12.4, trl 0.29.0, peft 0.18.1, transformers 4.57.6, bitsandbytes 0.49.2
- Unsloth: **not installed yet** — must be installed and compatibility-tested before writing the script

### What Looks Good

1. **`FastLanguageModel.from_pretrained` API** — Correct Unsloth loading pattern. Auto-detects
   BF16 on Ada, handles 4-bit quantization internally (no manual `BitsAndBytesConfig` needed).

2. **`use_gradient_checkpointing="unsloth"`** — Unsloth's custom gradient checkpointing is ~30%
   faster than HF's. Free speed win.

3. **Standard PEFT save format** — Compatible with our existing eval pipeline
   (`evaluate_translation.py` loads via `PeftModel.from_pretrained`), so no eval code changes needed.

4. **`SFTTrainer` from trl** — Unsloth hooks into the same trl trainer. Our `VRAMMonitorCallback`
   and other callbacks can carry over unchanged.

5. **Test matrix concept** — Good for quick hyperparameter sweeps. But needs alignment with our
   already-completed ablation (see conflicts section below).

### What Needs Fixing

| Issue | In spec | Should be | Why |
|-------|---------|-----------|-----|
| `max_seq_length=2048` | 2048 | **512** | Bible verses rarely exceed 200 tokens. 2048 wastes VRAM (Unsloth pre-allocates KV cache to this length) and slows training. Our production config and all ablation runs use 512. |
| `gradient_accumulation_steps=8` | 8 | **4** | Effective batch jumps from 8 to 16, changing learning dynamics. All ablation runs used batch=2 x grad_accum=4 = 8 effective. Keep consistent for apples-to-apples comparison. |
| ~~`GLOSSARY_OVERSAMPLE=3` hardcoded~~ | 3 | **FIXED: CLI flag `--glossary-oversample`, default 2** | Implemented in `train_gemma.py`. Default reduced to 2x for scale-up runs. At 20K pairs, 2x = 458/20458 = 2.2%. At 8K, 2x = 458/8458 = 5.4%. |
| `packing=True` default | True | **Needs head-to-head validation** | Spec claims "Unsloth packing is stable" but we haven't verified on TranslateGemma. Unsloth handles attention masking better than raw trl, but our entire ablation used `packing=False`. Must test packing=True vs False with Unsloth before making it the default. |
| `random_state=3407` | 3407 | **42** | Our entire pipeline uses seed=42 (data splits, training, subsampling). Consistency matters for reproducibility across HF and Unsloth runs. |
| Missing CLI flags | — | Must add | Spec doesn't include `--max-steps`, `--neftune`, `--replay-ratio`, `--lora-dropout` — all features we've already built, tested, and are actively using in the ablation. The script must be a superset of `train_gemma.py`, not a subset. |
| `eval_steps=150` | 150 | **200** | Match current HF config for direct comparison. |
| `save_steps=300` | 300 | **400** | Match current HF config. |
| `lora_dropout=0.05` hardcoded | 0.05 | **Use `--lora-dropout` CLI arg** | We just added this flag for Phase 2 B3 (dropout=0 test). Must be parameterized. |

### Data Needed Before Writing the Script

These must be answered empirically before implementation:

1. **Unsloth + TranslateGemma compatibility test** — TranslateGemma uses a custom chat template
   with `source_lang_code`/`target_lang_code` content fields. Must verify that
   `FastLanguageModel.from_pretrained("google/translategemma-4b-it")` loads successfully AND that
   the tokenizer's `apply_chat_template` still works with Unsloth's patched model. If the chat
   template breaks, training data format will be wrong and we'll get silent quality degradation
   (same failure mode as the packed run).

2. **VRAM benchmark** — Unsloth claims ~60% less VRAM. On our A2000 Ada 17.2GB, HF 4-bit uses
   ~10-12GB for 4B and ~14-15GB for 12B. If Unsloth drops 4B to ~6-8GB, it opens room for larger
   batch sizes. If it drops 12B to ~9-11GB, 12B training becomes comfortable without shared memory
   pressure. Need actual numbers on our hardware.

3. **Speed benchmark** — HF pipeline runs ~4s/step (nopack, batch=2, grad_accum=4). Unsloth claims
   2x speedup. If true, the entire ablation drops from ~9.5hrs to ~5hrs and scale-up runs become
   much more practical. Need to validate with a quick 50-step test on our hardware.

4. **Packing correctness test** — Train two 50-step models (packing=True vs False) with Unsloth,
   then translate the same 20 sentences with each. Compare outputs. If identical quality, packing
   is safe and gives ~30% training speedup from fewer padding tokens.

5. **Unsloth version pinning** — The Unsloth API has changed significantly between releases. Need
   to identify a specific version compatible with our stack (PyTorch 2.6.0 + CUDA 12.4 + trl 0.29.0
   + peft 0.18.1). Pin it in requirements and the script.

### Test Matrix Conflicts with Ablation

The embedded test matrix overlaps with our ablation but doesn't align:

| Unsloth matrix run | Overlaps with ablation | Conflict? |
|--------------------|----------------------|-----------|
| `base` (lr=5e-6, r=16) | A3 (lr=5e-6, r=16) | Same config but Unsloth uses packing=True — tests packing, not lr |
| `lower_lr` (lr=3e-6) | Between A3 and A4 | Novel — worth testing if ablation shows lr sensitivity |
| `higher_rank` (r=32) | — | Novel — opposite direction from A5 (r=4). Tests more capacity |
| `less_os` (oversample=1) | — | Novel — tests glossary underweight |
| `more_os` (oversample=5) | — | Novel but risky — known forgetting concern from oversampling |
| `conservative` (lr=2e-6, r=8) | Between A4 and A5 | Good combination test |
| `no_packing` | All current ablation runs | Redundant — our entire ablation IS no-packing |
| `balanced` (lr=4e-6, r=24) | — | Novel interpolation point |

**Recommendation:** After ablation results are in, replace the matrix with runs that are
**complementary** to what we already know. The ablation covers step count, lr extremes, rank,
and replay. The Unsloth matrix should focus on **Unsloth-specific variables**:
- Packing on vs off (does Unsloth's packing implementation actually work for TranslateGemma?)
- Larger effective batch sizes (enabled by Unsloth's lower VRAM footprint)
- `max_seq_length` impact (256 vs 512 — shorter sequences train faster with Unsloth)
- Speed/quality tradeoff at different training durations

### 4B vs 12B Strategy

The production architecture already uses both models in an A/B comparison pipeline:
- **4B** — speculative draft (~550ms translation, used for fast partials while speaking)
- **12B** — quality pass (~2.1s translation, replaces 4B output on silence detection)

#### Recommendation: Fine-tune 12B first, use it to distill into 4B

```
Phase 1: Complete 4B ablation on HF trainer (running now)
    |     Lock 4B hyperparams from ablation results.
    v
Phase 2: Install Unsloth, validate 4B compatibility
    |     Quick 50-step test: does Unsloth 4B match HF 4B BLEU?
    |     Measure VRAM and speed improvements.
    v
Phase 3: Fine-tune 12B with Unsloth
    |     Same winning hyperparams, possibly higher rank (r=32)
    |     for more capacity. Unsloth's lower VRAM makes 12B
    |     practical (~9-11GB estimated vs 14-15GB with HF).
    v
Phase 4: Evaluate 12B independently
    |     Run same 500-verse eval + theological spot-check.
    |     If 12B scores significantly higher on theo terms
    |     (e.g. 7/8 vs 4B's 5/8), proceed to distillation.
    |     If both score similarly, skip distillation.
    v
Phase 5 (conditional): Knowledge distillation
    |     Use fine-tuned 12B to generate high-quality translations
    |     for the training corpus (especially difficult theological
    |     sentences). Add these as supplementary training pairs
    |     for 4B. Re-fine-tune 4B on original + distilled data.
    v
Phase 6: Deploy both adapters to Mac for live A/B testing
```

#### Why this order works

- **12B has more capacity** for theological distinctions (expiación vs propiciación, Jacobo vs
  Santiago) — likely scores higher on the 8-term spot-check than 4B.
- **12B translations can augment 4B training data** where reference translations are weak. Archaic
  RVR1909 references may teach the model formal Spanish; 12B can generate modern alternatives.
- **Sequence-level knowledge distillation** (Kim & Rush, 2016) is well-proven for MT. The 12B
  acts as "teacher", 4B as "student". Both models improve: 12B from direct fine-tuning, 4B from
  fine-tuning + distilled data.
- **VRAM feasibility is the key unlock**: HF 4-bit 12B uses ~14-15GB (tight on 17.2GB A2000,
  requires shared memory). Unsloth 4-bit 12B should drop to ~9-11GB (comfortable, no shared
  memory pressure). This is a primary motivation for switching to Unsloth.

#### When NOT to distill

Distillation (Phase 5) adds complexity. Skip it if:
- 4B already scores 6/8+ on theological terms after fine-tuning — marginal gain from distillation
- 12B doesn't meaningfully outperform 4B on the spot-check — no useful teacher signal
- Time pressure — distillation adds another training cycle (~2-4 hours)

#### 12B-specific config adjustments

| Parameter | 4B value | 12B value | Rationale |
|-----------|----------|-----------|-----------|
| `lora_r` | from ablation | 32 (if ablation winner < 32) | 12B has more layers/parameters; higher rank captures more adaptation |
| `lora_alpha` | 2 x r | 2 x r | Standard scaling |
| `batch_size` | 2 | 1 | 12B uses more VRAM per sample |
| `grad_accum` | 4 | 8 | Compensate for smaller batch to keep effective batch ~8 |
| `max_memory` | 15GiB GPU | 16GiB GPU | Use more VRAM for 12B, less headroom needed with Unsloth |
| `save_total_limit` | 3 | 2 | 12B checkpoints are larger |

### Prerequisite Checklist

Before writing `train_gemma_unsloth.py`:

- [ ] Ablation results reviewed and winning config locked
- [ ] Unsloth installed in `/home/wbell/stt_train_env/` and version pinned
- [ ] `FastLanguageModel.from_pretrained("google/translategemma-4b-it")` loads successfully
- [ ] TranslateGemma chat template works with Unsloth-patched tokenizer
- [ ] VRAM measured: Unsloth 4B idle, Unsloth 4B training peak, Unsloth 12B idle
- [ ] Speed measured: steps/second for 50-step run (Unsloth 4B vs HF 4B)
- [ ] Packing test: 50-step train with packing=True vs False, compare 20 translations
- [ ] Test matrix revised to complement (not duplicate) ablation findings
- [ ] All existing CLI flags accounted for: `--max-steps`, `--neftune`, `--replay-ratio`, `--lora-dropout`, ~~`--glossary-oversample`~~ (DONE in train_gemma.py)
