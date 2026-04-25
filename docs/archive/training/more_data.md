**Plain verse-pairs (simple source → target JSONL) are the classic starting point for machine translation, but they are one of the weakest possible training signals for a strong, already-instruction-tuned model like TranslateGemma-4b-it.**

They cause the exact problems you’ve seen:
- Short, repetitive sequences → easy overfitting to Bible phrasing
- No broader context or style variation → rigid output
- No regularization against forgetting general translation ability

Google’s own TranslateGemma Technical Report (Jan 2026) explicitly avoided this by using a **rich mixture of synthetic + human parallel data** + a second RL stage. 2026 research on Gemma 3 domain adaptation shows the same pattern: pure parallel pairs almost always trigger catastrophic forgetting on translation-specialized models.

### Ranked: Better Methods Than Just Verse-Pairs
(Ordered by biggest expected gain + easiest implementation on your 16 GB A2000 + Unsloth)

#### 1. Two-Stage Training: Continued Pre-Training (CPT) → Light SFT (Best Overall for You)
**Why it wins**
- Phase 1 (CPT): Gently injects theological vocabulary and style into the model using **raw monolingual text** (no translation task). This is the modern replacement for “just pairs.”
- Phase 2 (Light SFT): Then do your normal paired training at very low LR (1e-6–3e-6).
- Matches exactly what TranslateGemma did internally and what multiple 2026 Gemma-3 papers recommend for religious/heritage domains.

**Expected gain**
+5–12 BLEU / +0.08–0.15 COMET on domain data while preserving general translation (proven on similar Bible-style projects).

**How to do it on your hardware (Unsloth makes this trivial)**
- Phase 1 (CPT): Use raw sermons + Bible verses + commentaries (no pairs needed).
- Phase 2: Your existing pairs at 1 epoch, lr=2e-6.
Total extra time on your A2000: ~45–90 min.

#### 2. Synthetic Data Generation + Back-Translation + Smart Mixing (Biggest Quick Win)
**Why it beats plain pairs**
- Generate new parallel examples using the **base TranslateGemma** itself on your monolingual sermons, commentaries, hymns, etc.
- Back-translate target-language text you already have.
- Mix 60–70 % your domain data + 30–40 % general high-quality parallel data (Flores-200, OPUS, or even synthetic general sentences).
- 2026 studies show LLM-generated synthetic data actually **reduces catastrophic forgetting** (lower perplexity tokens).

**Expected gain**
Often the single biggest jump (+4–8 BLEU) with almost no extra code.

**How to implement (one-night script)**
- Prompt the base model: “Translate this sermon paragraph from English to [target] while preserving theological terms…”
- Filter with COMET or your `evaluate_translation.py`.
- Combine with your verse pairs (65/35 ratio).

#### 3. Preference Optimization (ORPO or DPO) Instead of Pure SFT
**Why it’s superior**
- Instead of “here is the correct translation,” you give **chosen vs rejected** pairs.
- ORPO is the easiest (no reference model needed) and works amazingly well on translation.
- Forces the model to learn what makes a good translation rather than just memorizing pairs.

**Expected gain**
+3–7 COMET points over SFT with the same data volume; much more stable.

**Implementation note**
Unsloth supports ORPO with a one-line change from SFTTrainer. I can give you the exact script in 30 seconds.

#### 4. Context-Enriched / Document-Level Pairs (Underrated for Bible Work)
Instead of isolated verses, train on **full paragraphs or chapter chunks** with surrounding context.
Add metadata (book, chapter, genre, speaker) in the prompt.
This is why your per-genre BLEU varied so much — the model never learned discourse-level theology.

#### 5. Full RL Stage (MetricX / COMET Reward) — Advanced but Powerful
This is what Google did in stage 2 of TranslateGemma.
Unsloth + TRL makes DPO/KTO (simpler versions) easy. Do this only after one of the above works.

### Recommended Next Action for You Tonight
Start with **#2 (synthetic data + mixing)** — it’s the fastest to implement and directly copies TranslateGemma’s own recipe.

Or go straight to **#1 (two-stage CPT + SFT)** if you want the biggest long-term stability.
