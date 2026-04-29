# Phase A — Infrastructure

**Goal:** stand up the tooling needed to train Gemma 4 properly. No model training yet.

**Wall clock:** 5–7 days.

## A1. Install Unsloth into a dedicated venv

Unsloth's pip install cascades — it pins specific `torch`, `trl`, `transformers`, `xformers`, `triton`, and CUDA-lib versions. Installing into the existing `/home/wbell/stt_train_env/` would downgrade `trl 0.29 → 0.24` and replace the `torch +cu126` build with a CUDA-12.8 build, breaking `train_gemma.py`, `train_whisper.py`, and the existing benchmark scripts. **Use a separate venv** at `/home/wbell/unsloth_env/`:

```bash
python3.12 -m venv /home/wbell/unsloth_env
/home/wbell/unsloth_env/bin/pip install -U pip
/home/wbell/unsloth_env/bin/pip install unsloth gguf
```

Verify:

```bash
/home/wbell/unsloth_env/bin/python -c "from unsloth import FastModel; print('OK')"
```

The Gemma-4-tuning scripts (`training/train_gemma4.py`, `training/export_gguf.py`) use this venv. The non-Unsloth scripts (`training/qe_filter.py`, `training/glossary_annotate.py`, the older `train_gemma.py`, etc.) keep using `/home/wbell/stt_train_env/`.

**First-time model load:** `FastModel.from_pretrained("unsloth/gemma-4-E4B-it", load_in_4bit=True, max_seq_length=4096)` should download ~5 GB into `~/.cache/huggingface/`, then `nvidia-smi` should show ~10 GB resident. If higher, recheck `load_in_4bit=True` was honored.

## A2. Build `training/train_gemma4.py` (new file)

Alongside the existing `training/train_gemma.py` — do **not** delete or patch the old trainer. Key differences:

- **Loader:** Unsloth `FastModel`, not `AutoModelForCausalLM`.
- **Prompt format (mirror of `engines/llamacpp_engine.py:115-118`):**
  ```python
  user_content = (
      f"Translate the following {src_name} text to {tgt_name}. "
      f"Output only the translation, nothing else.\n\n{text}"
  )
  ```
- **Chat template:** `tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)` on every example.
- **LoRA config:**
  ```python
  model = FastModel.get_peft_model(
      model,
      r=8, lora_alpha=8, lora_dropout=0, bias="none",
      finetune_vision_layers=False,
      finetune_language_layers=True,
      finetune_attention_modules=True,
      finetune_mlp_modules=True,
      use_gradient_checkpointing="unsloth",
  )
  ```
  - **Do not** target `embed_tokens` (PLE must stay frozen).
- **Optimizer / schedule:** `optim="adamw_8bit"`, `lr=2e-4`, `warmup_steps=5`, `lr_scheduler_type="linear"`.
- **Sequence handling:** `packing=True`, `max_seq_length=1024`. Sermon turns avg ~120 tokens; packing 8× lifts effective throughput 6–8×.
- **Batching:** `per_device_train_batch_size=2, gradient_accumulation_steps=8` → effective batch 16.
- **Sanity asserts (loud failures, not warnings):**
  - text-only loss must settle in **1–3 by step ~500**, not 13–15 (the latter means multimodal got engaged — abort).
  - `use_cache=True` during training is **mandatory** for Gemma 4 shared KV layers (Unsloth caveat) — without it, logits go to garbage.
  - bf16 only; fp16 audio attention overflows (Unsloth caveat — even though we're text-only, the safety bar is bf16).

**Output:** `fine_tuned_gemma4_e4b/adapter_model.safetensors` + `training_manifest.json` (epochs, final loss, data size, eval scores when available).

## A3. Build `training/export_gguf.py` (new file)

The single biggest gap in current tooling: **no automated merge-and-quantize**. Steps:

1. **Merge into bf16 base** (NEVER into 4-bit base — destroys quality):
   ```python
   model.save_pretrained_merged("merged_bf16", tokenizer, save_method="merged_16bit")
   ```
2. **Convert to GGUF f16:**
   ```bash
   python ~/llama.cpp/convert_hf_to_gguf.py merged_bf16 --outtype f16 --outfile out_f16.gguf
   ```
3. **Quantize Q4_K_M:**
   ```bash
   ~/llama.cpp/build/bin/llama-quantize out_f16.gguf out_q4km.gguf Q4_K_M
   ```
4. **Verify chat template metadata** (loud failure if missing). The current llama.cpp build at `~/llama.cpp/` does not ship a `llama-gguf-tool` binary (only `llama-gguf` r/w), so we use the Python `gguf` package directly via `training/export_gguf.py`'s `verify_gguf_metadata()` helper. It checks: `tokenizer.chat_template` is present and ≥100 chars; `tokenizer.ggml.eos_token_id` is a non-negative int; `general.architecture == "gemma4"`. If `enable_thinking` semantics aren't preserved, fail loud — this is the same class of bug that cost 14× latency in production.
5. **Round-trip sanity test:** load both `merged_bf16` (in-process) and `out_q4km.gguf` (via `llama-server` on port 8092), run the 8 canary sentences through both, compute COMET delta. **Pass criterion: ≤ 0.3 COMET drop** (Gemma-4 community baseline for Q4_K_M).

Optional one-shot wrapper: `model.save_pretrained_gguf("dir", tokenizer, quantization_method="q4_k_m")` does the same end-to-end. Use whichever is more debuggable.

## A4. Build `training/qe_filter.py` (new file)

Load **CometKiwi-XL** (`Unbabel/wmt23-cometkiwi-da-xl`) once, score a JSONL of `{"en":..., "es":...}` pairs, write filtered output. CLI:

```bash
python training/qe_filter.py \
    --input bible_data/synthetic/deepl_sermon_pairs_full.jsonl \
    --output bible_data/synthetic/deepl_sermon_pairs_kiwi80.jsonl \
    --threshold 0.80
```

Used in two places:
1. Phase C — clean synthetic training data (general threshold 0.80, sermon threshold 0.85).
2. Phase D — score candidates for preference triple construction.

**Strict no-leakage rule:** train with CometKiwi-XL, **eval with xCOMET-XL + COMET-22**. Using the same scorer for training and eval inflates apparent gains.

## A5. Build `training/glossary_annotate.py` (new file)

For each (en, es) pair where an EN term from `bible_data/glossary/tier2_master.json` is present and its ES translation is present in the target, wrap the source span:

```
Input:  "He spoke about the atonement."
Output: "He spoke about the <g>atonement||expiación</g>."
```

Per WMT 2025 terminology task: training-time inline annotation at ~30% mixture rate beats both decode-time hard constraints (degrade fluency) and zero-tag training (no glossary control).

CLI:
```bash
python training/glossary_annotate.py \
    --input pairs.jsonl \
    --output pairs_annotated.jsonl \
    --glossary bible_data/glossary/tier2_master.json \
    --rate 0.30
```

**Gotcha:** Spanish inflection. *expiación* is the headword but text may have *expiaciones*, *expiar*, etc. v1 should match headword only (high precision, lower recall); v2 can add a Spanish stemmer.

## Definition of done for Phase A

- [ ] `from unsloth import FastModel` succeeds in `/home/wbell/unsloth_env/`.
- [ ] `training/train_gemma4.py` runs end-to-end on a 100-pair toy dataset (no crashes, loss decreasing, ~10 GB peak VRAM).
- [ ] `training/export_gguf.py` produces a working Q4_K_M GGUF whose canary outputs match merged-bf16 within 0.3 COMET (deferred to end of Phase B; metadata-only verification in Phase A is sufficient).
- [ ] `training/qe_filter.py` scores 1000 pairs in < 5 minutes on the A2000 Ada.
- [ ] `training/glossary_annotate.py --self-test` passes 8/8 hand-checked cases.

## Findings from initial audit (2026-04-29)

These were uncovered while standing up the tooling — flagged here so they don't surprise the next phase:

- **Holdout-eval bug confirmed.** `bible_data/aligned/verse_pairs_test.jsonl` (and the symlink at `bible_data/holdout/verse_pairs_test.jsonl`) is the 2-line stub (`{"en": "Genesis text", "es": "Texto de Génesis"}`, etc.) that the earlier S-sweep audit warned about. `bible_data/eval_registry.json` claims 500 entries with sha256 `e314def9...` but the file is 145 bytes. **Phase C cannot run a real holdout eval until this is rebuilt** via `tools/build_eval_sets.py --verse-count 500 --seed 42`. Add to Phase C1 prerequisites.
- **Platense Spanish alignment was broken — now fixed.** Root cause: SpaPlatense is Catholic canon (78 books, 37,255 verses), the other sources are Protestant canon (66 books, 31,102 verses), and the original aligner joined on row-order `verse_id` instead of `(book, chapter, verse)`. ~120,777 pairs (~50% of `verse_pairs_train.jsonl`) were silently misaligned past Genesis — Psalm 1:1 EN paired with Job 14:15 ES, etc. **Fixed via `tools/fix_platense_alignment.py` + `tools/rebuild_verse_pairs.py`** (structural SQLite join on `(book_name, chapter, verse)`); new corpus is `bible_data/aligned/verse_pairs_train_v2.jsonl` (265,271 pairs, all canonically aligned). Original `verse_pairs_train.jsonl` preserved as historical record. **Full postmortem and impact-on-S1–S9 analysis at [`docs/platense_alignment_bug.md`](../platense_alignment_bug.md).** Phase C C1 corpus mix needs revisiting — modern-register Spanish is now satisfied by realigned Platense, eliminating the need for new PD source acquisition.
- **`llama-gguf-tool` does not exist** in this `~/llama.cpp/` build (8782, e97492369). Use `training/export_gguf.py`'s `verify_gguf_metadata()` (Python `gguf.GGUFReader`) for chat-template and EOS verification.
- **transformers 5.5.3, trl 0.29.0, peft 0.18.1, torch 2.9.1+cu126** in stt_train_env. Unsloth wants to downgrade `trl → 0.24` and replace `torch +cu126 → 2.10.0` (CUDA 12.8 build), which would break the older trainers — hence the separate `unsloth_env`.
- **`unbabel-comet 2.2.7` ↔ `transformers 5.5.3` API conflict.** `comet 2.2.7` (latest) pins `transformers<5.0,>=4.17`, but `stt_train_env` already has `transformers 5.5.3` installed. Loading any CometKiwi model raises `AttributeError: XLMRobertaTokenizer has no attribute build_inputs_with_special_tokens` — the method was removed in transformers 5.x. **Resolution options for the user to choose:**
  1. Stand up a third venv (`comet_env`) pinned at `transformers==4.46.x` for QE only; `qe_filter.py` runs from there.
  2. Patch `~/stt_train_env/lib/python3.12/site-packages/comet/encoders/base.py:307` to call the new transformers tokenizer API directly.
  3. Wait for an upstream `unbabel-comet` release that supports transformers 5.x (no ETA).
  4. Switch QE to a different package (BLEURT, or load CometKiwi via plain `transformers` ourselves — most work).
  Option 1 is fastest and least invasive; pick it unless there's a reason not to.
- **CometKiwi-XL is gated on HF.** User `wbell7` is logged in but not on the access list for `Unbabel/wmt23-cometkiwi-da-xl`. Visit https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xl and click "Request access" / accept the license. The smaller `Unbabel/wmt22-cometkiwi-da` (~580M params) is already cached and accessible — usable as a fallback once the transformers conflict above is resolved.
- **Gemma 4 E4B fully cached** at `/mnt/d/Data/stt-data/cache/hub/models--unsloth--gemma-4-E4B-it/` (16 GB bf16; Unsloth quantizes to 4-bit on load via `load_in_4bit=True`). Offline-loadable verified.
- **`unbabel-comet 2.2.7` and Python `gguf 0.18.0` are installed** in stt_train_env; `unsloth + gguf` are installed in unsloth_env. `glossary_annotate.py` self-test passes 8/8; the rest of `qe_filter.py` end-to-end is blocked on the comet/transformers conflict above.
