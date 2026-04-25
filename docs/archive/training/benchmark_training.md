# Community Benchmarks & Findings for TranslateGemma Fine-Tuning

Collected 2026-03-20. These findings informed our patched `training/train_gemma.py` config
for the cycle-1 rescue run (8K pairs, lr=1e-5, 1 epoch, alpha=32).

---

## Sources

1. **HF Discussion: TranslateGemma-4B fine-tuning with Unsloth**
   https://huggingface.co/google/translategemma-4b-it/discussions/4

2. **Google Official: QLoRA Fine-Tuning with HF Transformers**
   https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora

3. **Google: Gemma Model Fine-Tuning Overview**
   https://ai.google.dev/gemma/docs/tune

4. **Unsloth: Gemma 3 Fine-Tuning Guide**
   https://unsloth.ai/blog/gemma3

5. **TranslateGemma Technical Report (arXiv:2601.09012)**
   https://arxiv.org/abs/2601.09012

6. **Kikuyu TranslateGemma-12B Case Study (2.29 -> 19.61 BLEU)**
   https://c-elo.com/blog/kikuyu-translategemma-12b

7. **LiteRT Community: TranslateGemma-4B Conversion Notes**
   https://huggingface.co/litert-community/TranslateGemma-4B-IT/discussions/1

---

## Google Official QLoRA Config (Source 2)

From Google's canonical Gemma QLoRA guide, targeting text-to-SQL:

| Parameter | Value |
|-----------|-------|
| r | 16 |
| lora_alpha | 16 |
| dropout | 0.05 |
| target_modules | `all-linear` |
| modules_to_save | `lm_head`, `embed_tokens` |
| lr | 2e-4 |
| epochs | 3 |
| optimizer | `adamw_torch_fused` |
| scheduler | constant |
| warmup_ratio | 0.03 |
| max_grad_norm | 0.3 |
| batch_size | 8 x 4 grad_accum |
| packing | True |
| max_seq_length | 512 |

**Note:** This config targets a *task change* (text-to-SQL), not domain adaptation
on an already-capable translation model. The aggressive lr (2e-4) and
`modules_to_save` are appropriate there but dangerous for TranslateGemma
EN->ES where the base model already scores 19.7 BLEU.

---

## Kikuyu TranslateGemma-12B Case Study (Source 6)

The most relevant community reference. Fine-tuned TranslateGemma-12B for
English->Kikuyu translation (a language with near-zero base capability).

### Their progression

| Version | BLEU | What changed |
|---------|------|-------------|
| Zero-shot | 2.29 | Repetitive nonsense ("muno muno muno...") |
| V1 | 18.16 | Baseline LoRA fine-tune |
| V3 | 15.93 | Over-regularized (dropout 0.1, weight_decay 0.02, neftune 7) |
| V2 (final) | 19.61 | Reduced regularization, removed embed_tokens, neftune 5 |

### Their final config

| Parameter | Value |
|-----------|-------|
| r | 128 |
| dropout | 0 |
| weight_decay | 0.01 |
| neftune_noise_alpha | 5 |
| lr | 2e-4 |
| training steps | ~900 (early stopped) |
| data | 30,430 pairs (95/5 split) |
| modules_to_save | None (removed embed_tokens) |
| hardware | NVIDIA H200 / L40S via Unsloth |

### Key findings

1. **`embed_tokens` in LoRA targets disrupted vocabulary.** Removing it gained
   +1.45 BLEU. The embedding layer is too sensitive for LoRA on a model that
   already has a working vocabulary.

2. **Over-regularization is worse than under-regularization.** V3 added dropout
   (0.1), higher weight decay (0.02), and higher NEFTune (7). Translations became
   grammatically correct but lost semantic precision. V2 with less regularization
   scored 3.68 BLEU higher.

3. **Dropout 0.1 caused degradation.** Their best run used dropout=0. For
   short training runs (< 1 epoch), dropout adds noise without the training
   duration needed to benefit from its regularization effect.

4. **NEFTune alpha=5 worked, alpha=7 hurt.** Moderate noise helps generalization;
   too much destabilizes the adapter. (We removed NEFTune entirely for cycle 1
   as a conservative choice; could re-add at 5 in cycle 2.)

5. **30K pairs were sufficient** for a language with zero base capability.
   For EN->ES where the base model is already strong, 8-20K pairs for domain
   adaptation should be more than enough.

---

## Unsloth Gemma 3 Findings (Source 4)

- Unsloth achieves 1.6x faster fine-tuning with 60% less VRAM vs HF + FA2.
- **float16 mixed precision causes infinity gradients** on T4, RTX 20x, V100.
  BF16 is required on Ada (our A2000 Ada supports BF16 natively).
- Their benchmark used r=32, batch_size=2, grad_accum=4 on all-linear modules.
- Gemma 3 27B fits under 22GB VRAM with Unsloth's optimizations.

---

## TranslateGemma Architecture (Source 5)

From the technical report (arXiv:2601.09012):

- **Two-stage training:** SFT on synthetic + human parallel data, then RL with
  MetricX-QE and AutoMQM reward models.
- The RL stage specifically optimizes translation quality metrics, which means
  aggressive fine-tuning can undo the RL-learned preferences (catastrophic
  forgetting of quality signals, not just translation ability).
- Uses a structured chat template with `source_lang_code` and `target_lang_code`
  fields in JSON format — must be preserved exactly during fine-tuning.

---

## Chat Template Format (Sources 1, 5)

The HF discussion (Source 1) shows the required JSON schema:

```python
json_payload = json.dumps([{
    "type": "text",
    "source_lang_code": "en",
    "target_lang_code": "es",
    "text": source_text
}], ensure_ascii=False)

full_prompt = f"user\n{json_payload}\nmodel\n{target_text}"
```

Our script uses `tokenizer.apply_chat_template()` which wraps this automatically.
The template verification block confirms the correct format fires before training.

---

## Packing Warning (Sources 4, our trl 0.29 logs)

Both Unsloth and trl 0.29 warn that **packing without flash attention causes
cross-contamination between packed examples**. Without proper attention masking,
the model can attend across sequence boundaries within a packed batch.

Options if this degrades quality:
1. Set `packing=False` (slower, ~2-3x more steps, but safe)
2. Install flash-attn and set `attn_implementation="flash_attention_2"`
3. Upgrade to trl >= 0.34 which has improved packing attention masks

We kept packing enabled for cycle 1 (speed) but should monitor eval loss for
anomalies that suggest cross-contamination.

---

## How Our Config Compares

| Parameter | Google Official | Kikuyu (best) | Our Cycle 1 | Rationale |
|-----------|----------------|---------------|-------------|-----------|
| r | 16 | 128 | 16 | VRAM-constrained (16GB vs H200) |
| alpha | 16 | N/A | 32 | 2x rank scaling for stronger adaptation |
| dropout | 0.05 | 0 | 0.05 | Conservative; could drop to 0 in cycle 2 |
| lr | 2e-4 | 2e-4 | 1e-5 | 20x lower — base model already scores 19.7 BLEU on EN->ES |
| epochs | 3 | <1 (early stop) | 1 | Match Kikuyu's effective training duration |
| neftune | N/A | 5 | removed | Stability for cycle 1; re-add at 5 if underfitting |
| modules_to_save | lm_head, embed | None | None | Kikuyu proved embed_tokens hurts translation |
| optimizer | adamw_fused | N/A | paged_adamw_8bit | Saves ~1-2GB on 16GB card |
| scheduler | constant | N/A | cosine | Better for single-epoch (smooth decay) |
| packing | True | N/A | True | Speed; monitor for cross-contamination |
| data | N/A | 30,430 | 8,000 (rescue) | Base already translates EN->ES; domain adaptation only |

### Why our lr is 20x lower than community

Every community example (Google, Kikuyu) uses lr=2e-4, but they are either:
- Teaching a **new task** (text-to-SQL) where the model has no prior capability
- Teaching a **new language** (Kikuyu) where zero-shot BLEU is 2.29

Our case is **domain adaptation on an existing capability** (EN->ES base BLEU=19.7).
The first training run at lr=5e-5 already caused catastrophic forgetting
(theological term accuracy dropped from 62% to 38-50%). At lr=2e-4 it would
be far worse. The 1e-5 rate preserves the base model's translation quality
while nudging it toward theological vocabulary.

---

## Baseline Metrics (our base model, no adapter)

Evaluated on 500 holdout verses (2026-03-20):

| Metric | Score |
|--------|-------|
| SacreBLEU | 19.7 |
| chrF++ | 44.4 |
| COMET | 0.7516 |

| Genre | BLEU | Verses |
|-------|------|--------|
| History | 24.8 | 89 |
| Poetry | 22.5 | 55 |
| Prophecy | 17.1 | 198 |
| Gospels | 18.7 | 50 |
| Epistles | 19.1 | 71 |
| Apocalyptic | 13.1 | 1 |

**Target for cycle 1:** BLEU > 21.7 (+2), COMET >= 0.75, no theological term regression.
