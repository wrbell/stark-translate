# Training Data Provenance Log

Tracks exactly which data files and counts went into each training run.

> **⚠️ 2026-04-29 — Platense alignment bug discovered.** Every entry below that consumed `bible_data/aligned/verse_pairs_train.jsonl` was trained on a corpus where ~50% of verse pairs (`es_source == 'platense'`) were silently misaligned — the row-order `verse_id` did not map to the same canonical verse across Protestant- and Catholic-canon sources. The fix and full impact analysis are in [`docs/platense_alignment_bug.md`](./platense_alignment_bug.md). New training should pull from `bible_data/aligned/verse_pairs_train_v2.jsonl` instead.

## Source Files Reference

| ID | File | Description | Count |
|----|------|-------------|-------|
| **V** | `bible_data/aligned/verse_pairs_train.jsonl` | Bible verse pairs (KJV/ASV/WEB ↔ RVR1909/ES) | ~27,130 total |
| **G** | `bible_data/glossary/glossary_pairs.jsonl` | Theological glossary term + sentence pairs | 458 (229 terms × 2 pairs) |
| **H1200** | `bible_data/synthetic/hybrid_sermon_pairs.jsonl` | 1200 chunks, 70% 12B + 30% DeepL | 1,200 |
| **H1800** | `bible_data/synthetic/hybrid_sermon_pairs_1800.jsonl` | 1800 chunks, 60% 12B + 40% DeepL | 1,800 |
| **D5000** | `bible_data/synthetic/deepl_sermon_pairs_5000.jsonl` | 5000 chunks, 100% DeepL glossary-enforced | 5,000 |
| **HYMN_PD** | `bible_data/hymns/hymn_pairs_pd.jsonl` | Public-domain same-original EN–ES hymn stanza pairs (train; ~5% spice) | 52 |
| **HYMN_HOLD** | `bible_data/hymns/hymn_pairs_pd_holdout.jsonl` | Hymn pair holdout (disjoint; includes SRGH-called PD hymns with pairs) | 13 |
| **HYMN_CAND** | `bible_data/hymns/hymn_candidates_synthetic.jsonl` | PD English stanzas with no verified PD Spanish (`needs_synthetic`) | 74 |
| **G_HYMN** | `bible_data/hymns/glossary_hymn_allowlist.json` | Curated hymn-domain glossary allowlist (review/merge via `build_glossary.py --from-hymns` / `--merge-hymn-allowlist`; not auto-injected) | 36 terms |

Hymn corpus notes (see [`hymn_data.md`](./hymn_data.md)): PD texts only — not New BHB 2019 compilation dumps. Suggested mix spice: `0.80 * S6 sources + 0.15 * glossary + 0.05 * HYMN_PD` (documentation only; no hymn S-run executed in this log). Regenerate with `python training/prepare_hymn_corpus.py all --seed 42`.

### Sermon Chunk Source

All sermon pairs originate from `ablation/sermon_whisper_chunks_expanded.json`:
- 24,595 unique chunks (≥20 chars, deduplicated)
- From 35 sources: 33 gospel messages + 2 conference sermons
- Date range: June 2025 – March 2026
- Transcribed by faster-whisper large-v3 fp16

The `--max-chunks` and `--seed 42` flags in `generate_hybrid_synthetic.py` determine which subset is used.

---

## S1/S2/S3 — Training Config Sweep (2026-03-21)

All use: V=8000 + G×2 (1014) + H1200 = ~10,214 pairs (11.5% sermon)

| Run | Dir | LR | Steps | Extras | Verse | Glossary | Sermon | Total |
|-----|-----|----|-------|--------|-------|----------|--------|-------|
| S1 | `S1_lr1e5_50steps` | 1e-5 | 50 | — | 8000 | 458×2=916 | 1200 (H1200) | ~10,116 |
| S2 | `S2_lr3e6_100steps` | 3e-6 | 100 | — | 8000 | 458×2=916 | 1200 (H1200) | ~10,116 |
| S3 | `S3_lr1e6_neftune` | 1e-6 | full | neftune=5 | 8000 | 458×2=916 | 1200 (H1200) | ~10,116 |

Sermon source: H1200 = 1200 chunks from `sermon_whisper_chunks.json` (original 1880, capped to 1200)
- 840 via TranslateGemma-12B (522 cached + 318 uncached)
- 360 via DeepL Pro (glossary-enforced)

## S4/S5/S6 — Ratio Sweep (2026-03-21/22)

All use: H1800 sermon data, G×1, S1 config (lr=1e-5, 50 steps)

| Run | Dir | Verse | Glossary | Sermon | Total | Sermon % |
|-----|-----|-------|----------|--------|-------|----------|
| S4 | `S4_sermon_only` | 0 | 458×1=458 | 1800 (H1800) | ~2,258 | 80% |
| S5 | `S5_sermon_heavy` | 500 | 458×1=458 | 1800 (H1800) | ~2,758 | 65% |
| S6 | `S6_balanced` | 1800 | 458×1=458 | 1800 (H1800) | ~4,058 | 44% |

Sermon source: H1800 = 1800 chunks from `sermon_whisper_chunks_expanded.json` (24,595, capped to 1800)
- 1080 via TranslateGemma-12B (285 cached + 795 uncached)
- 720 via DeepL Pro (glossary-enforced, 40%)

## S8 — Pure DeepL Scale (2026-03-22)

| Run | Dir | Verse | Glossary | Sermon | Total | Sermon % |
|-----|-----|-------|----------|--------|-------|----------|
| S8 | `S8_deepl_only` | 5000 | 458×1=458 | 5000 (D5000) | ~10,458 | 48% |

Config: lr=1e-5, 75 steps, glossary-oversample=1
Sermon source: D5000 = 5000 chunks from `sermon_whisper_chunks_expanded.json` (capped to 5000, seed=42)
- 5000 via DeepL Pro (100% glossary-enforced, 207 terms)
- 0 via 12B
- **Key test**: does pure DeepL glossary enforcement match or beat the 12B hybrid?
- **DATA LEAKAGE NOTE**: 89 chunks from `Gospel_Message_(3_15_26)_F1-ZLv-pMMg` (post-cutoff eval sermon) were included. This was generated BEFORE `--train-only` flag was added. Future runs must use `--train-only` to prevent this. Impact is minor (89/5000 = 1.8%) but noted for provenance.

## S7 — Hybrid Scale (planned)

| Run | Dir | Verse | Glossary | Sermon | Total | Sermon % |
|-----|-----|-------|----------|--------|-------|----------|
| S7 | `S7_scaled_balanced` | 5000 | 507×1=507 | 5000 (60/40 12B/DeepL) | ~10,507 | 48% |

Config: lr=1e-5, 75 steps, glossary-oversample=1
Sermon source: 5000 chunks from expanded pool
- ~3000 via TranslateGemma-12B (~800 cached from S1-S6, ~2200 uncached)
- ~2000 via DeepL Pro (glossary-enforced)
- **Requires ~4.5 hrs GPU for 12B inference**

## S9 — DeepL Double Scale (planned)

| Run | Dir | Verse | Glossary | Sermon | Total | Sermon % |
|-----|-----|-------|----------|--------|-------|----------|
| S9 | `S9_deepl_10k` | 5000 | 507×1=507 | 10000 (100% DeepL) | ~15,507 | 64% |

Config: lr=1e-5, 100 steps, glossary-oversample=1
Sermon source: 10000 chunks from expanded pool (100% DeepL)
- Needs additional 5000 DeepL translations (~$2.50, ~15 min API)
- **Tests diminishing returns at 2× sermon scale**
