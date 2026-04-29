# Gemma 4 Tuning — Overview

Plan for fine-tuning Gemma 4 **E2B** and **E4B** as better English↔Spanish translators for the Stark Road live translation pipeline. Replaces / supersedes the TranslateGemma S1–S9 sweep documented in `docs/archive/training/gemma_tuning_test_matrix.md`.

## Goal

Ship Gemma-4-tuned adapters (E4B and E2B) that beat stock Gemma 4 by:
- **≥ +2 COMET-22** on the 500-verse holdout AND on the 422-chunk sermon eval
- **≥ 7/8 → target 8/8** on the theological canary set
- **No regression** in latency, peak VRAM, or hallucination ratio

Two ship targets:
- **v1** (Phase C): plain SFT with the right prompt, right glossary integration, filtered synthetic data
- **v2** (Phase D): preference optimization (CPO / ARPO) layered on top of v1

## Why the prior sweep plateaued

S6 — the winner of the earlier sweep — achieved COMET delta = **−0.0002** vs. its base. Three structural reasons:

1. **Base model was already a translator.** TranslateGemma 4B is a Gemma 3 derivative purpose-built for translation. SFT on top has very limited headroom — you compete with Google's own training data. Gemma 4 E4B is a **generalist instruct model**, so the room for domain-specific gains is much larger.
2. **All sermon "ground truth" was synthetic** (DeepL or 12B distillation). Plain SFT on synthetic refs has a hard upper bound at the teacher's quality.
3. **No preference signal** — only point-estimate cross-entropy. Translation quality is rank-ordered (multiple acceptable refs); SFT alone cannot exploit that.

Smaller contributing factors documented for completeness:
- Archaic Spanish bias (RVR1909, Platense) at high mixture rate, dragging modern-register quality.
- Theological-term ceiling (6/8) capped by data sparsity for niche terms (e.g. only 2 *propiciación* training pairs).
- Eval rubric was COMET-vs-12B-baseline — designed to detect parity with a larger model, not to detect targeted gains in theological terminology.

**Discovered and fixed during Phase A staging (2026-04-29):** the Platense Spanish half of `verse_pairs_train.jsonl` (~50% of the corpus) was **misaligned from Psalms onward** because the original aligner joined per-source files by row-order `verse_id` instead of canonical `(book, chapter, verse)` — fine for Protestant-canon sources but broken for Catholic-canon Platense (which interleaves 12 apocryphal books). The S1–S9 sweep trained on roughly half-noise verse pairs without anyone noticing. This likely outranks the three structural reasons above as a contributor to the plateau. **Fixed**: `bible_data/aligned/verse_pairs_train_v2.jsonl` (265K pairs, structurally re-joined). Full postmortem at [`docs/platense_alignment_bug.md`](../platense_alignment_bug.md).

## What changed: Gemma 4 architecture matters

"Gemma 4" and "Gemma 3n" are the **same model line** for fine-tuning purposes — Google launched the E2B/E4B (MatFormer + PLE) line under "Gemma 3n" mid-2025; HF and Vertex docs from late-2025/2026 call the continuation "Gemma 4". Most authoritative how-to material is filed under **Gemma 3n** — search for that name.

| Property | Implication for fine-tuning |
|---|---|
| **MatFormer** (E2B is a sliced sub-net of E4B) | Fine-tuning E4B does **not** yield a usable E2B adapter — LoRA deltas are not slice-equivalent. **Train E2B and E4B as separate jobs.** |
| **Per-Layer Embeddings (PLE)** kept in bf16 even when 4-bit quantized | **Freeze PLE** — do not target `embed_tokens` with LoRA, do not include PLE in QLoRA quantization. (This is also why VRAM was undercounted by 2× in v2026.5 benchmarks.) |
| **Thinking mode** on by default — production uses `enable_thinking: false` (`engines/llamacpp_engine.py:138`) | **Train with `enable_thinking=False` applied to every example.** This bakes non-thinking behavior into the adapter and lets us drop the runtime workaround once v1 is permanent. |
| **Vision/audio towers** in E4B | **Freeze.** Saves several GB. `finetune_vision_layers=False` in Unsloth. |
| **Tokenizer** = Gemma 3 SentencePiece (256K), no language tags | Language is in the prompt — already matches our setup. **No new special tokens needed.** |
| **VRAM**: E4B QLoRA ≈ 10 GB (fits 16 GB), E4B LoRA bf16 ≈ 17 GB (does NOT fit) | **Must use QLoRA via Unsloth.** Vanilla HF + PEFT + bnb works but is ~1.5× slower and uses ~2× more memory at our hardware tier. |

The existing `training/train_gemma.py` was built for the **TranslateGemma** chat template (with `source_lang_code` / `target_lang_code` fields). It does **not** produce adapters that match the production Gemma 4 prompt. We will write a new trainer (`training/train_gemma4.py`) alongside the old one rather than patching it.

## Phase sequencing (overview)

| Phase | Wall clock | Output |
|---|---|---|
| **A — Infrastructure** | 5–7 days | New trainer, merge→GGUF pipeline, QE filter, glossary annotator |
| **B — Spike** | 1–2 days | E2E smoke test on ~3.6K pairs, GGUF round-trip verified |
| **C — Domain SFT (v1)** | ~1 week | E4B and E2B adapters meeting v1 ship criteria |
| **D — Preference Opt (v2)** | ~1 week | CPO / ARPO refinement |
| **E — Deploy** | 1–2 days | A/B in production for one Sunday service |

Detailed plan per phase in the other files in this directory.

## Open decisions (need user input before Phase C)

1. **Teacher distillation budget.** ~$50–150 for one Gemini-2.5-Pro pass on the toughest 5% of sermon chunks.
2. **Human-translated reference data.** Optional ~$200–800 for professional translation of ~200 representative sermon chunks. High-leverage if budget allows.
3. **CPO/ARPO in v1 or v2?** Plan defaults to v2.
4. **Unsloth dependency.** Strong recommend yes; flagged because it's a direction shift.

## What we're explicitly NOT doing

- **Span-level preference (xMaP / MAPS).** No clean open-source 2026 recipe.
- **Liger Kernel stacked on Unsloth.** Conflicts; no Gemma-4 compat statement.
- **Decode-time hard glossary constraints.** Degrade fluency.
- **Fine-tuning the vision tower.** Frozen.
- **Training E4B and slicing to E2B.** Doesn't work with LoRA.
- **Switching to ALMA-base or other MT-specialized models.** Gemma 4 is what's deployed.
- **Patching `train_gemma.py` to handle Gemma 4.** Old trainer wired for TranslateGemma chat template; clean new trainer is less work.

## Sources / further reading

- [Unsloth Gemma 4 train](https://unsloth.ai/docs/models/gemma-4/train)
- [Unsloth Gemma 3n fine-tune](https://docs.unsloth.ai/models/gemma-3-how-to-run-and-fine-tune/gemma-3n-how-to-run-and-fine-tune)
- [Unsloth Saving to GGUF](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)
- [Google Gemma 3n overview](https://ai.google.dev/gemma/docs/gemma-3n)
- [HF transformers PLE issue #45206](https://github.com/huggingface/transformers/issues/45206)
- [CPO / ALMA paper](https://arxiv.org/abs/2401.08417)
- [X-ALMA / ARPO paper](https://arxiv.org/abs/2410.03115)
- [WMT 2025 terminology task](https://www2.statmt.org/wmt25/pdf/2025.wmt-1.30.pdf)
- [Unbabel COMET](https://github.com/Unbabel/COMET)
- Internal: `docs/archive/training/gemma_tuning_test_matrix.md` (S1–S9 history)
- Internal: `docs/archive/v2026.5/BENCHMARK.md` (Phase 1A llama.cpp wins)
