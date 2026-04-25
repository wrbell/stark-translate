# Multilingual Todo List: Hindi & Chinese

> **Status:** Not started
> **Last updated:** 2026-02-14
> **Prerequisite:** English-Spanish pipeline stable, accent-diverse STT fine-tuning underway
> **Reference:** [`multilingual_tuning_proposal.md`](multilingual_tuning_proposal.md) for full rationale and research

---

## Phase 0: Zero-Shot Baseline (1 day)

Validate that TranslateGemma works out of the box for both languages before investing in fine-tuning.

- [ ] Add `hi` and `zh-Hans` to `dry_run_ab.py` target language options
- [ ] Run 10 test sentences through TranslateGemma 4B with `target_lang_code="hi"`
- [ ] Run 10 test sentences through TranslateGemma 4B with `target_lang_code="zh-Hans"`
- [ ] Run 10 theological sentences (John 3:16, Romans 8:28, etc.) through both
- [ ] Measure baseline latency: Hindi (expect ~700-800ms 4B), Chinese (expect ~600-700ms 4B)
- [ ] Spot-check: Does Hindi use तू (intimate) for God, or आप (formal)? (expect: wrong)
- [ ] Spot-check: Does Chinese use 神 or 上帝 consistently? (expect: mixed)
- [ ] Record baseline chrF++, COMET on 50 sample translations per language
- [ ] **GO/NO-GO:** If zero-shot quality is usable, proceed. If gibberish, investigate model/tokenizer.

---

## Phase 1: Data Preparation (3-5 days)

### 1A: Biblical Parallel Corpus

- [ ] Download Hindi IRV (Indian Revised Version 2019) from `bible-nlp/biblenlp-corpus`
  - ~31K verses, CC-BY-SA 4.0
  - ID: `hin2017`
- [ ] Download Chinese CUV (Chinese Union Version) from `bible-nlp/biblenlp-corpus`
  - Simplified: `cmn-cu89s`, ~31K verses, public domain
  - Traditional: `cmn-cu89t`, ~31K verses, public domain
- [ ] Download Chinese Free Easy-to-read Bible: `cmnfeb`, ~8K NT verses, CC-BY-SA 4.0
- [ ] Align verse pairs (JOIN on verse ID):
  - EN (KJV, ASV, WEB, BBE, YLT) x Hindi IRV = ~155K pairs
  - EN x Chinese CUV-S + CUV-T = ~310K pairs
  - EN x cmnfeb (NT only) = ~40K bonus pairs
- [ ] Export as JSONL with fields: `source_lang_code`, `target_lang_code`, `source_text`, `target_text`, `verse_id`
- [ ] Create stratified holdout test sets (~3.1K verses each, by biblical genre)

### 1B: Hindi Theological Glossary (~100-150 terms)

- [ ] Build Hindi glossary modeled on `build_glossary.py` (229 Spanish terms as template)
- [ ] Include core terms from proposal Section 3.1:
  - परमेश्वर (Parameshwar) for God, NOT ईश्वर (Ishwar)
  - प्रभु (Prabhu) for Lord
  - प्रायश्चित (Prayashchit) for Atonement
  - वाचा (Vacha) for Covenant
  - अनुग्रह (Anugrah) for Grace
  - बपतिस्मा (Baptisma) for Baptism
  - etc.
- [ ] Add gender annotations (परमेश्वर=masc, पवित्र आत्मा=fem)
- [ ] Add honorific mappings: तू for divine address, आप for human formal
- [ ] Include ~31 proper names (यीशु, मूसा, इब्राहीम, etc.)
- [ ] Generate training pairs from glossary entries (source + target per term)

### 1C: Chinese Theological Glossary (~100-150 terms)

- [ ] Build Chinese glossary from proposal Section 3.2
- [ ] **Decision: 神 vs 上帝** — recommend 神 (more common in mainland churches)
- [ ] Enforce Protestant terminology consistently:
  - 圣灵 (not 圣神) for Holy Spirit
  - 耶和华 (not 雅威) for YHWH
  - 洗礼 (not 圣洗) for Baptism
  - 教会 (not 教堂) for Church
- [ ] Include Simplified→Traditional mapping for all terms (OpenCC integration later)
- [ ] Generate training pairs from glossary entries

### 1D: Partial Translation Models

- [ ] **Hindi:** Download and test IndicTrans2-Dist (AI4Bharat, ~211M params)
  - Check if MLX conversion available
  - Fallback: `Helsinki-NLP/opus-mt-en-hi`
  - Measure latency (target: <100ms)
- [ ] **Chinese:** Download and test `Helsinki-NLP/opus-mt-en-zh` (~300M params)
  - Measure latency (target: <80ms)
- [ ] **Decision: Hindi partial display strategy**
  - Hindi is SOV — partials will be garbled ("यीशु ने..." = verb missing)
  - Recommended: show English partial + Hindi final only (no Hindi partial)
  - Alternative: increase wait-k to 8-12 tokens before generating Hindi partial

---

## Phase 2: QLoRA Training (2 overnight runs)

### 2A: Hindi QLoRA (Night 1)

- [ ] Prepare training data: ~155K EN-HI verse pairs + glossary + honorific examples
- [ ] Mix: 60-70% theological, 20-30% general-domain (Samanantar sample), 10% ES replay
- [ ] Configure QLoRA:
  - r=32, alpha=64 (higher rank than Spanish due to new direction)
  - Target: `"all-linear"`
  - NF4, double quant, bf16 compute
  - Batch 1, grad accum 4-8
  - **max_seq_length=768** (Hindi has 2.5-3.5x token fertility)
  - Packing enabled
  - 3 epochs
- [ ] Run on A2000 Ada (estimate: ~6-9 hrs, 10-13 GB VRAM)
- [ ] Save adapter to `fine_tuned_gemma_mi_HI/`

### 2B: Chinese QLoRA (Night 2)

- [ ] Prepare training data: ~310K EN-ZH verse pairs + glossary
- [ ] Mix: 60-70% theological, 20-30% general-domain (CCMatrix sample), 10% ES replay
- [ ] Configure QLoRA:
  - r=32, alpha=64
  - Target: `"all-linear"`
  - NF4, double quant, bf16 compute
  - Batch 1, grad accum 4-8
  - max_seq_length=512 (Chinese is more compact)
  - Packing enabled
  - 3 epochs
- [ ] Run on A2000 Ada (estimate: ~6-9 hrs, 8-10 GB VRAM)
- [ ] Save adapter to `fine_tuned_gemma_mi_ZH/`

### 2C: Optional — 12B Variants

- [ ] **Decision gate:** Only if 4B theological term accuracy < 70% for either language
- [ ] Hindi 12B QLoRA: ~18-27 hrs (3 nights), tight on 16GB VRAM
- [ ] Chinese 12B QLoRA: ~18-27 hrs (3 nights)

---

## Phase 3: Evaluation (1 day)

### 3A: Automatic Metrics

- [ ] Run chrF++ on holdout test set (primary metric — handles morphology and no-space scripts)
- [ ] Run SacreBLEU on holdout test set
  - Hindi: standard tokenizer
  - Chinese: **must use `--tokenize zh`**
- [ ] Run COMET (`wmt22-comet-da`) on holdout test set
- [ ] Run CometKiwi (`wmt22-cometkiwi-da`) on sample translations (reference-free QE)
- [ ] Run LaBSE cross-lingual similarity on sample translations

### 3B: Theological Term Accuracy

- [ ] Hindi: Spot-check all ~100-150 glossary terms — does the model use correct terms?
- [ ] Hindi: Check honorific accuracy — does the model use तू for God in prayer?
- [ ] Chinese: Spot-check all ~100-150 glossary terms
- [ ] Chinese: Check 神/上帝 consistency — does the model stick to the chosen term?
- [ ] Chinese: Check Protestant vs. Catholic terminology — 圣灵 not 圣神, etc.

### 3C: Quality Gates

| Metric | Hindi Target | Chinese Target |
|--------|-------------|----------------|
| chrF++ improvement | > +2 points over zero-shot | > +2 points over zero-shot |
| COMET improvement | > +0.02 over zero-shot | > +0.02 over zero-shot |
| Theological term accuracy | > 80% | > 80% |
| Hindi honorific accuracy | > 90% (तू for God) | N/A |
| 上帝/神 consistency | N/A | > 95% |

- [ ] **GO/NO-GO:** If quality gates pass, proceed to integration. If not, investigate data/training.

---

## Phase 4: Pipeline Integration (1-2 days)

### 4A: Adapter Loading

- [ ] Transfer Hindi and Chinese LoRA adapters to Mac
- [ ] Convert PyTorch adapters to MLX format (if needed)
- [ ] Implement adapter switching in `dry_run_ab.py`:
  - Load base TranslateGemma once
  - Swap adapter weights based on `target_lang_code`
  - Verify: switching adapters does NOT increase base memory (~9 GB stays constant)
- [ ] Add `--target-langs` CLI argument to `dry_run_ab.py` (default: `es`, options: `es,hi,zh-Hans`)

### 4B: Partial Translation Integration

- [ ] Wire up Hindi partial model (IndicTrans2-Dist or opus-mt-en-hi) in pipeline
- [ ] Wire up Chinese partial model (opus-mt-en-zh) in pipeline
- [ ] Implement partial display strategy:
  - Spanish: show partials normally (SVO — works fine)
  - Hindi: English partial only, show Hindi on final (SOV issue)
  - Chinese: show partials normally (SVO — similar to English)

### 4C: WebSocket Protocol

- [ ] Extend WebSocket messages to include multi-language translations:
  ```json
  {
    "type": "translation",
    "source": "For God so loved the world...",
    "translations": {
      "es": "Porque de tal manera amó Dios al mundo...",
      "hi": "यीशु ने दुनिया से ऐसा प्रेम किया...",
      "zh-Hans": "神爱世人，甚至将他的独生子赐给他们..."
    },
    "is_final": true
  }
  ```
- [ ] Each display client filters for its configured language
- [ ] Pipeline sends all active translations over the same WebSocket

---

## Phase 5: Display Updates (1 day)

### 5A: Fonts

- [ ] Add Noto Sans Devanagari webfont (~200KB) for Hindi
- [ ] Add Chinese font stack (system fonts, no 16MB download):
  `'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', 'Hiragino Sans GB', sans-serif`
- [ ] Set `lang` attributes: `lang="hi"`, `lang="zh-Hans"`
- [ ] Adjust line-height: Hindi = 1.6, Chinese = 1.5 (vs 1.4 for Latin scripts)

### 5B: Mobile Display

- [ ] Add language selector tabs to `mobile_display.html`: [EN] [ES] [HI] [ZH]
- [ ] Each tab shows full-screen translation in selected language
- [ ] Store language preference in localStorage
- [ ] Test rendering on iPhone/Android with both scripts

### 5C: Audience Display

- [ ] Default: English + Spanish side-by-side (existing, unchanged)
- [ ] Optional: configurable second language via URL param (`?lang2=hi`)
- [ ] QR code links to mobile display with language picker

### 5D: OBS Overlay

- [ ] Update `obs_overlay.html` to support language switching
- [ ] Test with Devanagari and CJK character rendering in OBS

---

## Phase 6: Quality Monitoring (ongoing)

- [ ] Extend `translation_qe.py` to run CometKiwi on Hindi and Chinese outputs
- [ ] Add per-language QE thresholds:
  - Hindi: CometKiwi > 0.83 (slightly lower due to SOV reordering)
  - Chinese: CometKiwi > 0.85 (same as Spanish)
- [ ] Log per-language metrics to `metrics/translation_qe.jsonl`
- [ ] Add Hindi back-translation via `Helsinki-NLP/opus-mt-hi-en` for Tier 2 QE
- [ ] Add Chinese back-translation via `Helsinki-NLP/opus-mt-zh-en` for Tier 2 QE

---

## Summary Timeline

| Phase | Duration | GPU Hours | Human Hours |
|-------|----------|-----------|-------------|
| 0: Zero-shot baseline | 1 day | 0 | 2 |
| 1: Data preparation | 3-5 days | 0 | 8-15 |
| 2: QLoRA training | 2 nights | 12-18 | 1 |
| 3: Evaluation | 1 day | 1-2 | 3-5 |
| 4: Pipeline integration | 1-2 days | 0 | 8-12 |
| 5: Display updates | 1 day | 0 | 4-6 |
| 6: Quality monitoring | ongoing | 0 | 2-3 |
| **Total** | **~8-10 days** | **~13-20** | **~28-44** |

---

## Dependencies

```
Phase 0 (baseline) ──→ Phase 1 (data prep) ──→ Phase 2A (Hindi QLoRA)  ──→ Phase 3 (eval)
                                             ├→ Phase 2B (Chinese QLoRA) ──┘
                                             └→ Phase 5 (display) ──→ Phase 4 (integration)
```

Hindi and Chinese training run on consecutive nights. Display work proceeds in parallel with training. Phase 6 is ongoing after deployment.

---

## Key Decisions to Make

| Decision | Options | Recommendation |
|----------|---------|----------------|
| 神 vs 上帝 for Chinese | 神 (common mainland), 上帝 (more reverent) | **神** — matches CUV majority edition |
| Hindi partial strategy | Show garbled SOV partials vs. English-only partials | **English partial + Hindi final** |
| Hindi partial model | IndicTrans2-Dist (better quality) vs opus-mt-en-hi (simpler) | **IndicTrans2-Dist** if MLX-compatible |
| LoRA rank | r=16 (conservative) vs r=32 (more capacity) | **r=32** — new language directions need capacity |
| Simplified vs Traditional Chinese | Train on Simplified, convert with OpenCC | **Simplified primary**, OpenCC for Traditional |
| Train 12B variants? | Only if 4B < 70% theological accuracy | **Defer** until Phase 3 results |
