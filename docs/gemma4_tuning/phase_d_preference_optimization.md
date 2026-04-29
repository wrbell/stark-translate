# Phase D — Preference Optimization (v2 ship target)

**Goal:** layer CPO / ARPO on top of v1 SFT for **+0.5–1.5 COMET-22 over v1**, with no canary regression.

**Wall clock:** ~1 week.

This phase is **optional for v1 ship**. Recommended only after v1 is in production and stable.

## D1. Generate candidates

For each of ~5K source sentences (held out from C1 training data), generate **4 translations** from the v1 SFT model at **temperature 0.7**.

```bash
python training/build_preference_triples.py generate \
    --model models/gemma-4-e4b-it-q4km-v1.gguf \
    --sources bible_data/preference_pool/sources_5k.jsonl \
    --candidates 4 \
    --temperature 0.7 \
    --output preference/raw_candidates.jsonl
```

## D2. Score with CometKiwi-XL

Build `(source, chosen, rejected)` triples — chosen = top-scored candidate, rejected = bottom-scored, only keep triples with **margin > 0.05** to avoid noisy gradients.

```bash
python training/build_preference_triples.py score \
    --candidates preference/raw_candidates.jsonl \
    --margin 0.05 \
    --output preference/triples_v1.jsonl
```

Reuses `training/qe_filter.py`'s CometKiwi-XL loader.

## D3. Run CPO (1 epoch)

Contrastive Preference Optimization — ALMA recipe (`arxiv.org/abs/2401.08417`). TRL has `CPOTrainer` natively.

```python
# training/train_gemma4_cpo.py — wraps trl.CPOTrainer
trainer = CPOTrainer(
    model=base_model,
    tokenizer=tokenizer,
    train_dataset=preference_dataset,
    args=CPOConfig(
        learning_rate=5e-6,   # 10× lower than SFT
        beta=0.1,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
    ),
    peft_config=lora_config,  # same r=8 LoRA on top of v1 merged base
)
```

**Critical:** start from the **v1 merged bf16 model**, not the original Gemma 4 base. The v1 SFT is the foundation CPO refines.

## D4. If over-rejection observed → graduate to ARPO / X-ALMA

If CPO degrades on novel inputs (a known CPO pathology), switch to **ARPO / X-ALMA** — adaptive penalty version that fixes the over-rejection (`arxiv.org/abs/2410.03115`). Same pipeline; `beta` becomes a per-pair adaptive function based on chosen/rejected similarity.

Detection: run v2 against a held-out general-domain set (not used for CPO triples). If COMET drops > 0.3, that's over-rejection.

## D5. v2 eval gate

Same harness as Phase C3:

- [ ] **≥ +0.5 COMET-22 over v1** on both verse and sermon eval.
- [ ] **No canary regression** (must hold v1's 7/8 or 8/8).
- [ ] **No hallucination regression**.
- [ ] **General-domain regression check** — translate 200 OPUS-100 holdout pairs; COMET must not drop > 0.3 vs. v1.

## Strict no-leakage rule (research finding, repeated for emphasis)

Train CPO with **CometKiwi-XL** as the reward signal. Eval v2 with **xCOMET-XL + COMET-22** (different family, supervised reference-based). Using CometKiwi for both inflates apparent gains by 1–2 COMET points.

## What we considered and rejected

- **Span-level preference (xMaP / MAPS).** No clean open-source 2026 recipe. Revisit in 6 months.
- **Self-play loop (multiple CPO cycles).** ALMA shows 2 cycles capture ≥ 95% of the gain — stick with one for v2.
- **DPO with sampled negatives.** Strictly worse than CPO for MT in the ALMA results.
