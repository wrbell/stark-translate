# Multilingual Fine-Tuning Proposal: Hindi & Chinese for TranslateGemma

> **Status:** Proposal
> **Date:** 2025-02-14
> **Scope:** Domain adaptation of TranslateGemma for EN→HI and EN→ZH theological translation
> **Prerequisite:** Whisper STT pipeline established and stable

---

## Executive Summary

TranslateGemma natively supports both Hindi and Chinese as part of its rigorously benchmarked 55-language WMT24++ set. No new model download is needed — the existing `mlx-community/translategemma-4b-it-4bit` and `12b-it-4bit` models work out of the box by changing `target_lang_code` from `"es"` to `"hi"` or `"zh-Hans"`.

Fine-tuning is therefore **domain adaptation** (theological/biblical register), not teaching new languages. This dramatically reduces data requirements (5K–50K curated pairs vs. millions) and compute (one overnight run per language on the A2000 Ada).

Biblical parallel corpora are available and freely licensed:
- **Hindi:** Indian Revised Version (IRV 2019) — ~31K verses, CC-BY-SA 4.0
- **Chinese:** Chinese Union Version (CUV 1919) — ~31K verses in both Simplified and Traditional, **public domain**

Total timeline: **~8–10 days** from zero-shot baseline to integrated multilingual pipeline.

---

## Table of Contents

1. [TranslateGemma Language Support](#1-translategemma-language-support)
2. [Training Data](#2-training-data)
3. [Theological Glossaries](#3-theological-glossaries)
4. [QLoRA Training Strategy](#4-qlora-training-strategy)
5. [Partial Translation Models](#5-partial-translation-models)
6. [Linguistic Challenges](#6-linguistic-challenges)
7. [Evaluation Stack](#7-evaluation-stack)
8. [Display & UI](#8-display--ui)
9. [Compute Estimates](#9-compute-estimates)
10. [Timeline](#10-timeline)
11. [Risks & Mitigations](#11-risks--mitigations)
12. [References](#12-references)

---

## 1. TranslateGemma Language Support

### Native Coverage

Both Hindi and Chinese are among TranslateGemma's 55 core WMT24++ languages with production-tier quality guarantees:

- **Hindi** (`hi` / `hi-IN`): Part of the original WMT24 evaluation set with additional post-edits
- **Simplified Chinese** (`zh-Hans` / `zh`): New references and post-edits in WMT24++
- **Traditional Chinese** (`zh-Hant` / `zh-TW`): Separate variant, also benchmarked

### Language Codes

The chat template accepts ISO 639-1 codes with optional regional variants:

| Language | Recommended Code | Alternatives |
|----------|-----------------|--------------|
| Hindi | `hi` | `hi-IN`, `hi-Latn` (transliteration) |
| Simplified Chinese | `zh-Hans` | `zh`, `zh-CH` |
| Traditional Chinese | `zh-Hant` | `zh-TW` |

### Chat Template Usage

Identical to the existing Spanish pipeline — only `target_lang_code` changes:

```python
messages = [{"role": "user", "content": [
    {"type": "text",
     "source_lang_code": "en",
     "target_lang_code": "hi",      # or "zh-Hans", "zh-Hant"
     "text": "For God so loved the world..."}
]}]

prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
response = generate(model, tokenizer, prompt=prompt, max_tokens=200)
```

### Quality Benchmarks

Aggregate WMT24++ performance (all 55 languages):

| Model | MetricX (lower=better) | COMET (higher=better) |
|-------|:----------------------:|:---------------------:|
| TranslateGemma 27B | **3.09** | **84.4** |
| TranslateGemma 12B | 3.60 | 83.5 |
| TranslateGemma 4B | 5.32 | 81.6 |

The Marathi result (closely related to Hindi) shows >25% error reduction from base Gemma, suggesting comparable or better gains for Hindi as a higher-resource language.

### MLX Models (Already Downloaded)

| Model | Disk | RAM | HuggingFace ID |
|-------|------|-----|----------------|
| 4B 4-bit | ~2.2 GB | ~2.5 GB | `mlx-community/translategemma-4b-it-4bit` |
| 12B 4-bit | ~6.6 GB | ~7 GB | `mlx-community/translategemma-12b-it-4bit` |

These are the same models used for Spanish. No separate download needed.

---

## 2. Training Data

### 2.1 Biblical Parallel Corpora

#### Hindi

| Translation | ID | Verses | License | Source |
|-------------|-----|--------|---------|--------|
| Indian Revised Version (IRV 2019) | `hin2017` | ~31K (full Bible) | **CC-BY-SA 4.0** | `bible-nlp/biblenlp-corpus` |
| Hindi Contemporary Version | `hincv` | ~31K | Biblica copyright (redistributable) | `bible-nlp/biblenlp-corpus` |
| Hindi Easy Reading Version | `hin2010` | ~31K | Copyright (NOT redistributable) | — |

**Usable:** IRV (definite), HCV (license review needed). ERV cannot be used.

**Verse pair yield:** ~31K verses × 5 public-domain English translations (KJV, ASV, WEB, BBE, YLT) = **~155K pairs** — matches the existing Spanish pipeline exactly.

Also available in `Helsinki-NLP/bible_para`:
```python
dataset = load_dataset("bible_para", lang1="en", lang2="hi")
```

#### Chinese

| Translation | ID | Script | Verses | License | Source |
|-------------|-----|--------|--------|---------|--------|
| CUV with New Punctuation | `cmn-cu89s` | Simplified | ~31K | **Public Domain** | `bible-nlp/biblenlp-corpus` |
| CUV with New Punctuation | `cmn-cu89t` | Traditional | ~31K | **Public Domain** | `bible-nlp/biblenlp-corpus` |
| Free Easy-to-read Bible | `cmnfeb` | Simplified | ~8K (NT only) | CC-BY-SA 4.0 | `bible-nlp/biblenlp-corpus` |

**Verse pair yield:**
- CUV-S + CUV-T × 5 English = **~310K pairs**
- Adding cmnfeb (NT only): **+~40K pairs**
- **Total: ~350K pairs** — significantly more than Spanish

**Copyright boundary (same rules as Spanish):** Do NOT use Chinese New Version (CNV), Revised CUV (RCUV 2010), or Chinese Living Bible.

### 2.2 General-Domain Supplementary Corpora

| Corpus | en-hi Pairs | en-zh Pairs | License |
|--------|-------------|-------------|---------|
| Samanantar (AI4Bharat) | ~10.1M | — | CC0 |
| CCMatrix | ~15.1M | ~30M+ | CC0 |
| WikiMatrix | ~231K | ~787K | CC-BY-SA 3.0 |
| IITB English-Hindi | ~1.49M | — | CC-BY-NC-SA 4.0 |
| OPUS-100 | ~1M (capped) | ~1M (capped) | CC-BY-4.0 |
| FLORES-200 | ~3K (eval only) | ~3K (eval only) | CC-BY-SA 4.0 |

Since TranslateGemma already speaks both languages natively, **5K–50K curated pairs per language** is sufficient for domain adaptation. These massive corpora serve as supplementary replay data to prevent forgetting.

### 2.3 Comparison with Existing Spanish Pipeline

| Factor | Spanish (current) | Hindi (proposed) | Chinese (proposed) |
|--------|-------------------|------------------|-------------------|
| Bible translations usable | 2 | 1–2 | 2–3 |
| Maximum verse pairs | ~155K | ~155K–310K | ~350K |
| Best license | Public domain (RVR1909) | CC-BY-SA 4.0 (IRV) | **Public domain** (CUV) |
| General parallel data | Large | Very large (Samanantar) | Massive (CCMatrix) |
| Theological glossary | Built (229 terms) | **Needs building** | **Partially available** |

---

## 3. Theological Glossaries

### 3.1 Hindi Theological Terms

Hindi Bible translations use **heavily Sanskritized vocabulary** (not Persianized/Urdu-register). This was a deliberate 19th-century missionary decision. Key terms:

| English | Hindi (Devanagari) | Transliteration | Disambiguation Challenge |
|---------|-------------------|-----------------|--------------------------|
| God | परमेश्वर | Parameshwar | NOT ईश्वर (Ishwar) — carries Hindu connotations |
| Lord | प्रभु | Prabhu | Universal |
| Atonement | प्रायश्चित | Prayashchit | In Hinduism = self-imposed penance; Christianity = substitutionary |
| Covenant | वाचा | Vacha | Also means "promise/word" |
| Grace | अनुग्रह | Anugrah | "Favor/kindness" |
| Righteousness | धार्मिकता | Dharmikta | Derived from "dharma" — broader in Hindu philosophy |
| Sanctification | पवित्रीकरण | Pavitrikaran | Compound: pavitra (holy) + karan (making) |
| Salvation | उद्धार | Uddhaar | Also "rescue/deliverance" in secular Hindi |
| Sin | पाप | Paap | Same word in Hindu usage |
| Faith | विश्वास | Vishwas | Also "trust/confidence" generally |
| Baptism | बपतिस्मा | Baptisma | Transliterated from Greek |
| Cross | क्रूस | Kroos | Transliterated from Portuguese "cruz" |
| Resurrection | पुनरुत्थान | Punarutthan | "Rising again" |
| Trinity | त्रिएकता | Tri-ekta | Neologism: tri (three) + ekta (oneness) |

**Critical honorific rule:** Hindi has a three-tier pronoun system:
- **तू (tu)** — intimate: Used for God in prayer and divine address (Hindi Bible standard)
- **तुम (tum)** — familiar: Friends, peers
- **आप (aap)** — formal: Elders, strangers

Generic MT models default to आप (formal). Theological fine-tuning must teach तू for divine address.

### 3.2 Chinese Theological Terms

| English | Simplified | Traditional | Pinyin | Notes |
|---------|-----------|-------------|--------|-------|
| God | 神 or 上帝 | 神 or 上帝 | Shén / Shàngdì | **Two CUV editions exist** — must pick one |
| Lord | 主 | 主 | Zhǔ | Universal |
| Atonement | 赎罪 | 贖罪 | Shúzuì | "Redeem sin" |
| Covenant | 约 / 盟约 | 約 / 盟約 | Yuē / Méngyuē | 约 also = "appointment" |
| Grace | 恩典 | 恩典 | Ēndiǎn | Specific to religious register |
| Righteousness | 公义 / 义 | 公義 / 義 | Gōngyì / Yì | 义 also = "justice" |
| Sanctification | 成圣 | 成聖 | Chéngshèng | "Becoming holy" |
| Salvation | 救恩 | 救恩 | Jiù'ēn | Theological compound |
| Sin | 罪 | 罪 | Zuì | Also = "crime" secularly |
| Holy Spirit | 圣灵 | 聖靈 | Shènglíng | Protestant; Catholic = 圣神 |
| Baptism | 洗礼 | 洗禮 | Xǐlǐ | "Washing ceremony" (sprinkling); 浸礼 = immersion |
| Trinity | 三位一体 | 三位一體 | Sān wèi yī tǐ | "Three persons one body" |
| Justification | 称义 | 稱義 | Chēngyì | "Declared righteous" |
| YHWH | 耶和华 | 耶和華 | Yēhéhuá | Protestant transliteration |

**The "Term Question":** The choice between 上帝 (Shangdi) and 神 (Shen) for "God" has been debated since the 1840s. Two separate editions of the CUV exist. Training data must be consistent with the target audience's preference.

**Protestant vs. Catholic splits:** 圣灵/圣神 (Holy Spirit), 耶和华/雅威 (YHWH), 洗礼/圣洗 (baptism), 教会/教堂 (church). These are not interchangeable.

### 3.3 Glossary Action Items

- [ ] Build Hindi theological glossary (~100–150 terms) modeled on `build_glossary.py`
- [ ] Adapt existing Chinese theological resources (TranslationDirectory.com, Kingdom Speak) into NLP format
- [ ] Decide 上帝 vs 神 edition for Chinese (recommend 神 — more common in mainland churches)
- [ ] Include honorific mappings (Hindi तू/तुम/आप) in training examples

---

## 4. QLoRA Training Strategy

### 4.1 Separate Adapters (Recommended)

Train **separate LoRA adapters** per language direction:
- `lora-en-hi/` — English → Hindi
- `lora-en-zh/` — English → Chinese
- `lora-en-es/` (existing) — English → Spanish

**Why not joint training?**
- Hindi (SOV) and Chinese (SVO with heavy reordering) are maximally different linguistically — joint training risks interference
- Research (ACL MRL 2025) shows LoRA provides **no inherent advantage** over full FT for forgetting prevention — the data/model-size ratio is what matters
- A single adapter trained on all three risks catastrophic forgetting: performance collapse approaching 0 BLEU on unseen pairs has been observed with large fine-tuning datasets
- Separate adapters allow per-language optimization and protect existing Spanish quality

**Adapter switching at inference:** Load the appropriate adapter based on target language. On MLX, this means loading the base model once and swapping adapter weights — fast and memory-efficient.

### 4.2 QLoRA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rank (r) | **32** | Up from r=16 for Spanish — new language directions need more capacity |
| Alpha | **64** | 2× rank, consensus from Lightning AI experiments |
| Target modules | `"all-linear"` | Same as Spanish; maximum adaptation for new directions |
| Quantization | NF4, double quant, bf16 compute | Standard QLoRA recipe |
| Batch size | 1 (grad accum 4–8) | Conservative for 16GB VRAM |
| Learning rate | 2e-4 | Standard QLoRA rate |
| Max seq length | **768** (Hindi), **512** (Chinese) | Hindi needs more room due to higher token fertility |
| Sequence packing | Enabled | Bible verses are short |
| Optimizer | `paged_adamw_32bit` | Memory-efficient |
| Epochs | 3 | Standard for domain adaptation |

### 4.3 Data Mix for Anti-Forgetting

| Proportion | Content | Purpose |
|-----------|---------|---------|
| 60–70% | Target language theological data | Domain adaptation |
| 20–30% | General-domain target language (sampled from OPUS/Samanantar/CCMatrix) | Robustness |
| 10% | Spanish theological data (replay) | Prevent forgetting if using shared adapter |

With separate adapters, the Spanish replay is not strictly necessary but is good practice if you ever merge adapters.

### 4.4 Tokenizer Considerations

Gemma 3's **262K vocabulary** (same as Gemini 2.0) covers both scripts natively. No extension needed.

| Script | Tokens per Word | Implication |
|--------|----------------|-------------|
| English | ~1.3 | Baseline |
| Spanish | ~1.3–1.5 | Similar to English |
| Hindi (Devanagari) | **~2.5–3.5** | Higher fertility; increase max_seq_length |
| Chinese | **~1.5–2.0/char** | Gemma 3 tokenizer explicitly optimized for CJK |

Hindi's high fertility means:
- Context window pressure (compensate with longer max_seq_length)
- ~2× compute cost per token (budget slightly more training time)
- Use sequence packing aggressively to amortize padding

### 4.5 Key Research Findings

| Paper | Venue | Finding |
|-------|-------|---------|
| Conditions for Catastrophic Forgetting in Multilingual Translation | ACL MRL 2025 | LoRA ≈ full FT for forgetting; data/model ratio is critical |
| LoRA Learns Less and Forgets Less | ICML 2024 | LoRA better at preserving source-domain; learns less in target |
| MLAS-LoRA | ACL 2025 | Language-specific LoRA modules avoid parameter interference |
| Fine-Tuning LLMs to Translate | EMNLP 2024 | 32 samples already outperform baselines; 1024 is the efficiency sweet spot |
| Continually Adding New Languages | arXiv 2025 | LayRA: apply LoRA to first 10 + last 2 layers only, freeze middle |

---

## 5. Partial Translation Models

### 5.1 Current Architecture (Spanish)

MarianMT (`Helsinki-NLP/opus-mt-en-es`) provides fast ~80ms partials while TranslateGemma produces high-quality finals on silence.

### 5.2 Hindi Partial Model

**Recommended: IndicTrans2-Dist (AI4Bharat)**
- ~211M parameters (~1/5 of full IndicTrans2)
- Outperforms Google Translate, NLLB-54B, and GPT-3.5 on en→hi benchmarks
- Exceeds baselines by 4–8 BLEU/chrF++ on en→Indic directions
- CTranslate2 inference versions available
- Long context models (2048 tokens) released Jan 2025

Alternative: `Helsinki-NLP/opus-mt-en-hi` (smaller, lower quality)

### 5.3 Chinese Partial Model

**Recommended: `Helsinki-NLP/opus-mt-en-zh`**
- ~300M parameters, ~80ms latency
- Decent quality for partials (not as strong as IndicTrans2 for Hindi)

Alternative: GemmaX2-28-2B (Xiaomi Research, NAACL 2025) — outperforms TowerInstruct and X-ALMA, competitive with Google Translate, but would need MLX conversion.

### 5.4 The SOV Partial Display Problem

Hindi's Subject-Object-Verb word order creates a fundamental issue for real-time partials:

```
English partial:  "Jesus loved the..."
Spanish partial:  "Jesús amó el..."      (SVO - works fine)
Hindi partial:    "यीशु ने..."            (SOV - verb missing, garbled)
```

The verb — the most semantically critical element — comes **last** in Hindi. A half-finished English sentence produces a grammatically incomplete Hindi partial.

**Recommended mitigation:** Show English partial + Hindi/Chinese final only:

```
[While speaking]  English: "Jesus loved the world, that he gave..."  (live partial)
                  Hindi:   ...                                       (waiting)
                  Chinese: ...                                       (waiting)

[On silence]      English: "Jesus loved the world, that he gave his only begotten Son."
                  Hindi:   "यीशु ने दुनिया से ऐसा प्रेम किया कि उसने अपना एकलौता पुत्र दे दिया।"
                  Chinese: "神爱世人，甚至将他的独生子赐给他们。"
```

Alternative: Increase wait-k to 8–12 tokens (vs. 3–5 for Spanish) before generating Hindi partials.

---

## 6. Linguistic Challenges

### 6.1 Hindi

| Challenge | Impact | Mitigation |
|-----------|--------|------------|
| SOV word order | Garbled partials | English partials + Hindi finals only |
| Postpositions (not prepositions) | Decoder must delay relational markers | Model handles this; fine-tuning improves |
| तू/तुम/आप honorific system | Wrong pronoun for God/prayer | Include explicit honorific examples in training data |
| Gender agreement in verbs (648 inflected forms) | Theological entities must agree (परमेश्वर=masc, पवित्र आत्मा=fem) | Glossary with gender annotations |
| Sanskritized register | Training data is formal "Shuddh Hindi", not colloquial | Use IRV (formal) as primary source |
| Code-switching (Hinglish) | Diaspora churches mix Hindi/English | Not critical for EN→HI direction |
| Devanagari rendering | Conjunct forms, headline stroke, taller glyphs | Proper fonts + line-height: 1.6 |

### 6.2 Chinese

| Challenge | Impact | Mitigation |
|-----------|--------|------------|
| Simplified vs. Traditional | Two character sets | Train on Simplified, use OpenCC for Traditional conversion |
| No word boundaries | Tokenization, evaluation affected | Use chrF++ (character-level); SacreBLEU with `--tokenize zh` |
| Classical Chinese in CUV | Literary register unfamiliar to younger readers | Accept CUV register as target (standard for Chinese Protestants) |
| Measure words/classifiers | Wrong classifier sounds non-native | Fine-tuning on biblical text teaches correct classifiers |
| 上帝/神 Term Question | Two CUV editions exist | Pick one and enforce consistency via glossary |
| Protestant vs. Catholic terminology | 圣灵/圣神, 耶和华/雅威 | Constrain to Protestant terms in training |
| CJK font size | Full Noto Sans SC = ~16MB | Google Fonts auto-subsetting or system font fallback |

### 6.3 Length Ratios

| Language Pair | Word Count Change | Visual Area Change |
|--------------|-------------------|-------------------|
| EN → ES | +15–25% | Similar |
| EN → HI | **+15–35%** | **+40–60%** (wider + taller characters) |
| EN → ZH | **-30–50%** word count | Similar width (fixed-width characters) |

---

## 7. Evaluation Stack

### 7.1 Metrics

| Metric | Hindi | Chinese | Notes |
|--------|-------|---------|-------|
| **chrF++ (primary)** | Standard | Standard | Character-level; avoids Chinese segmentation issues; handles Hindi morphology |
| SacreBLEU | Standard tokenizer | **`--tokenize zh`** | Must use Chinese tokenizer |
| COMET | `wmt22-comet-da` | `wmt22-comet-da` | Reference-based neural; same model as Spanish |
| CometKiwi (QE) | `wmt22-cometkiwi-da` | `wmt22-cometkiwi-da` | Reference-free; same as Spanish QE pipeline |
| LaBSE similarity | Supported | Supported | Cross-lingual cosine similarity |
| Theological term accuracy | **Manual evaluation** | **Manual evaluation** | Most critical; not captured by automatic metrics |

All existing evaluation infrastructure (COMET, CometKiwi, LaBSE) supports Hindi and Chinese out of the box.

### 7.2 Holdout Test Sets

Same stratification strategy as Spanish — by biblical genre:

| Genre | Approximate Verses | Key Challenges |
|-------|-------------------|----------------|
| Pentateuch / Torah | ~500 | Creation vocabulary, legal terminology |
| History | ~500 | Proper names, genealogies |
| Poetry / Wisdom | ~500 | Metaphor, parallelism |
| Prophecy | ~500 | Apocalyptic imagery, archaic register |
| Gospels | ~500 | Parables, direct speech, honorifics |
| Epistles | ~500 | Doctrinal vocabulary, complex theology |
| Apocalyptic | ~100 | Symbolic language, rare terms |

### 7.3 Expected Improvements Post Fine-Tuning

| Metric | Expected Gain | Basis |
|--------|--------------|-------|
| chrF++ | +2–5 points on biblical text | Comparable to Spanish domain adaptation |
| COMET | +0.02–0.05 | Neural metric, smaller absolute changes |
| Theological term accuracy | 40–60% → **80%+** | Primary goal of fine-tuning |
| Hindi honorific accuracy | Low → **>90%** | Explicit training signal |
| 上帝/神 consistency | Mixed → **>95%** | Constrained by training data |

---

## 8. Display & UI

### 8.1 Font Stack

```css
:root {
  --font-en: 'Inter', -apple-system, sans-serif;
  --font-es: var(--font-en);  /* Latin script */
  --font-hi: 'Noto Sans Devanagari', 'Hind', 'Mangal', sans-serif;
  --font-zh: 'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei',
             'Hiragino Sans GB', sans-serif;
}

.lang-en, .lang-es { font-family: var(--font-en); line-height: 1.4; }
.lang-hi { font-family: var(--font-hi); line-height: 1.6; }
.lang-zh { font-family: var(--font-zh); line-height: 1.5; }
```

- Set `lang` attributes on HTML elements: `lang="hi"`, `lang="zh-Hans"`, `lang="zh-Hant"`
- Hindi webfont (Noto Sans Devanagari): ~200KB — load directly
- Chinese: use system font stack to avoid 16MB download; Google Fonts auto-subsets if self-hosting needed

### 8.2 Multi-Language Display Strategy

**Projector (audience_display.html):**
English + Spanish side-by-side (existing layout) — the two largest language groups.

**Mobile (mobile_display.html):**
Per-user language selection via QR code. Each audience member picks their language and sees full-screen translation on their phone. The existing WebSocket architecture supports this — add a `target_lang` filter per client connection.

```
┌──────────────────────────────┐
│  [EN] [ES] [HI] [ZH]        │  ← Language selector tabs
├──────────────────────────────┤
│                              │
│  यीशु ने दुनिया से ऐसा प्रेम    │
│  किया कि उसने अपना एकलौता     │
│  पुत्र दे दिया।                │
│                              │
│  (Full-screen, large font)   │
└──────────────────────────────┘
```

**WebSocket protocol extension:**
```json
{
  "type": "translation",
  "source": "For God so loved the world...",
  "translations": {
    "es": "Porque de tal manera amó Dios al mundo...",
    "hi": "यीशु ने दुनिया से ऐसा प्रेम किया...",
    "zh-Hans": "神爱世人，甚至将他的独生子赐给他们...",
    "zh-Hant": "神愛世人，甚至將他的獨生子賜給他們..."
  },
  "is_final": true
}
```

Each display client filters for its configured language. Pipeline sends all translations over the same WebSocket.

---

## 9. Compute Estimates

### 9.1 Training (A2000 Ada 16GB, WSL2)

| Run | Data | Steps/Epoch | Time/Epoch | Total (3 epochs) | VRAM |
|-----|------|-------------|------------|-------------------|------|
| Hindi QLoRA (50K pairs) | Biblical + curated | ~3,125 | ~2–3 hrs | **6–9 hrs** | ~10–13 GB |
| Chinese QLoRA (50K pairs) | Biblical + curated | ~3,125 | ~2–3 hrs | **6–9 hrs** | ~8–10 GB |
| Hindi domain-only (5K theological) | Glossary + verses | ~313 | ~15–25 min | **45–75 min** | ~8–10 GB |
| Chinese domain-only (5K theological) | Glossary + verses | ~313 | ~15–25 min | **45–75 min** | ~8–10 GB |
| Evaluation (chrF++/COMET per lang) | Holdout sets | — | — | **30–60 min** | ~6 GB |

Each language fits in a single overnight run. Hindi is slightly more expensive due to longer sequences (higher token fertility).

### 9.2 Inference (M3 Pro, 18GB, MLX)

Expected latency (based on Spanish baselines):

| Component | Spanish (measured) | Hindi (expected) | Chinese (expected) |
|-----------|-------------------|------------------|-------------------|
| mlx-whisper STT | ~580ms | Same | Same |
| TranslateGemma 4B translation | ~650ms | ~700–800ms | ~600–700ms |
| TranslateGemma 12B translation | ~1.4s | ~1.5–1.8s | ~1.3–1.5s |
| Partial model | ~80ms (MarianMT) | ~80–100ms (IndicTrans2-Dist) | ~80ms (opus-mt-en-zh) |
| **End-to-end final (4B)** | **~1.3s** | **~1.4–1.6s** | **~1.3–1.4s** |
| **End-to-end final (12B)** | **~1.9s** | **~2.1–2.4s** | **~1.9–2.1s** |

Hindi latency is slightly higher due to Devanagari token fertility. Chinese should be similar to Spanish.

### 9.3 Memory Budget (MLX, M3 Pro 18GB)

| Configuration | Estimated RAM | Feasible? |
|--------------|--------------|-----------|
| Whisper + 4B (1 language) | ~5–6 GB | Yes |
| Whisper + 4B + 12B (1 language) | ~9 GB | Yes |
| Whisper + 4B (3 languages, adapter switching) | ~5–6 GB | Yes |
| Whisper + 4B + 12B (3 languages, adapter switching) | ~9 GB | Yes |
| Whisper + 3× partial models + 4B + 12B | ~12–14 GB | Tight but feasible |

Adapter switching does not increase base memory — only the active adapter is loaded.

---

## 10. Timeline

### Phase-by-Phase

| Phase | Duration | Details |
|-------|----------|---------|
| **Phase 1: Zero-shot baseline** | 1 day | Change `target_lang_code`, run existing pipeline on test sentences |
| **Phase 2: Data preparation** | 3–5 days | Download corpora, build Hindi/Chinese glossaries (~100–150 terms each) |
| **Phase 3: Hindi QLoRA** | 1 night | 50K pairs, r=32, 3 epochs on A2000 Ada |
| **Phase 4: Chinese QLoRA** | 1 night | 50K pairs, r=32, 3 epochs on A2000 Ada |
| **Phase 5: Evaluation** | 1 day | chrF++, COMET, CometKiwi, theological term accuracy |
| **Phase 6: Adapter transfer** | 1 day | Transfer to Mac, add language switching to pipeline |
| **Phase 7: Display updates** | 1 day | Fonts, mobile language selector, WebSocket filtering |
| **Total** | **~8–10 days** | Much faster than Spanish (no Whisper tuning needed) |

### Dependencies

```
Phase 1 (baseline) ──→ Phase 2 (data prep) ──→ Phase 3 (Hindi) ──→ Phase 5 (eval)
                                             ├→ Phase 4 (Chinese) ──┘
                                             └→ Phase 7 (display) ──→ Phase 6 (integration)
```

Hindi and Chinese training can run on consecutive nights. Display work can proceed in parallel with training.

---

## 11. Risks & Mitigations

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|------------|
| Hindi partials garbled (SOV word order) | High | Certain | Show English partials + Hindi finals only |
| 上帝/神 inconsistency in Chinese output | Medium | Likely | Constrain to one edition in glossary + training data |
| Hindi तू/आप honorific errors | Medium | Likely | Include explicit honorific examples in fine-tuning data |
| Catastrophic forgetting of Spanish | Medium | Low (separate adapters) | Separate adapters + optional Spanish replay |
| Hindi token fertility degrades quality | Medium | Possible | Increase max_seq_length to 768, use packing |
| Chinese font download too large | Low | Avoidable | System font stack; Google Fonts auto-subsetting |
| IndicTrans2-Dist unavailable for MLX | Medium | Unknown | Fall back to `Helsinki-NLP/opus-mt-en-hi` |
| CUV register too archaic for young readers | Low | Context-dependent | Supplement with modern Chinese Bible data (cmnfeb) |

---

## 12. References

### TranslateGemma

- TranslateGemma Technical Report — [arXiv:2601.09012](https://arxiv.org/abs/2601.09012)
- WMT24++ Evaluation — [arXiv:2502.12404](https://arxiv.org/html/2502.12404v1)
- Google Blog: TranslateGemma — [blog.google](https://blog.google/innovation-and-ai/technology/developers-tools/translategemma/)
- MLX Community Models — [HuggingFace](https://huggingface.co/mlx-community)

### Fine-Tuning Research

- Conditions for Catastrophic Forgetting in Multilingual Translation — [ACL MRL 2025](https://aclanthology.org/2025.mrl-main.23/)
- LoRA Learns Less and Forgets Less — [ICML 2024](https://arxiv.org/abs/2405.09673)
- MLAS-LoRA: Language-Aware Parameters — [ACL 2025](https://aclanthology.org/2025.acl-long.762/)
- Fine-Tuning LLMs to Translate — [EMNLP 2024](https://aclanthology.org/2024.emnlp-main.24/)
- LoRA Insights from Hundreds of Experiments — [Lightning AI](https://lightning.ai/pages/community/lora-insights/)
- QLoRA: Efficient Finetuning of Quantized LLMs — [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

### Corpora & Datasets

- BibleNLP Corpus (833 languages) — [HuggingFace](https://huggingface.co/datasets/bible-nlp/biblenlp-corpus)
- Helsinki-NLP/bible_para — [HuggingFace](https://huggingface.co/datasets/Helsinki-NLP/bible_para)
- Samanantar (49.6M Indic pairs) — [AI4Bharat](https://huggingface.co/datasets/ai4bharat/samanantar)
- IIT Bombay English-Hindi — [HuggingFace](https://huggingface.co/datasets/cfilt/iitb-english-hindi)
- eBible Corpus — [GitHub](https://github.com/BibleNLP/ebible)

### Language-Specific Models

- IndicTrans2 — [GitHub](https://github.com/AI4Bharat/IndicTrans2), [TMLR 2023](https://openreview.net/forum?id=vfT4YuzAYA)
- GemmaX2-28-9B — [NAACL 2025](https://aclanthology.org/2025.naacl-long.280/)
- OpenCC (Simplified ↔ Traditional conversion) — [GitHub](https://github.com/BYVoid/OpenCC)

### Evaluation

- SacreBLEU — [GitHub](https://github.com/mjpost/sacrebleu)
- COMET / CometKiwi — [GitHub](https://github.com/Unbabel/COMET)
- COMTAIL (Indian language MT metric) — [arXiv:2509.17667](https://arxiv.org/html/2509.17667v1)

### Display & Fonts

- W3C Devanagari Gap Analysis — [w3.org](https://www.w3.org/TR/deva-gap/)
- Noto Sans Devanagari — [Google Fonts](https://fonts.google.com/noto/specimen/Noto+Sans+Devanagari)
- CJK Font Best Practices — [Tony Baloney](https://tonybaloney.github.io/posts/cjk-chinese-japanese-korean-llm-ai-best-practices.html)

### Church Translation Tools

- OneAccord (live church AI translation) — [oneaccord.ai](https://www.oneaccord.ai/)
- SIL Serval (Bible NMT API) — [ai.sil.org](https://ai.sil.org/projects/serval)
- spf.io (church captioning) — [spf.io](https://www.spf.io/solutions/religious/)
