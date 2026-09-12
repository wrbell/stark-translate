# Hymn-domain data (`bible_data/hymns/`)

Copyright-safe EN↔ES hymn stanza pairs and glossary candidates for Gemma 4 translation SFT
(prepared in the TranslateGemma era; that program is historical, the current one is
[`docs/gemma4_tuning/overview.md`](../../docs/gemma4_tuning/overview.md)). This slice is a
**small spice (~5%)** of a translation mix — not a third pillar beside verse + sermon. No
training run has used it yet. **Do not** mix hymn *singing* audio into Whisper / W16 sermon LoRA
datasets. Narrative and provenance: [`docs/hymn_data.md`](../../docs/hymn_data.md); the live-pipeline
hymn items are `hymn-translation-boundary` and `hymn-capture-suppression` in `docs/backlog.json`.

## License and copyright

- **Allowed:** individual hymn *texts* whose authors died ≥70 years ago (public domain). Typical BHB-core authors: Watts, Newton, Cowper, Deck, Darby, Kelly, Bonar, Clephane, Crosby (19th c.), etc.
- **Not allowed as a training dump:** the *New Believers Hymn Book* (2019) compilation, numbering, 2019-only added hymns, Gospel Folio music edition, BHB+ app database, or a wholesale scrape of gospelriver.com treated as “the book.”
- John Ritchie Ltd / New BHB holds **compilation** copyright. We train on independently attested **public-domain hymn texts**, not on the compilation as a work.
- gospelriver.com / BHB app may be used as a **lookup** for first lines that also exist on Hymnary.org or in pre-1929 scans. Training text is attributed to the PD hymn text, not “New BHB #N.”
- See [`sources.json`](sources.json) and [`reports/license.md`](reports/license.md) for source-by-source notes.

## Regenerate (offline)

```bash
python training/prepare_hymn_corpus.py all --seed 42
```

Optional:

```bash
python training/prepare_hymn_corpus.py build-index --fetch   # refresh PD evidence URLs
python training/prepare_hymn_corpus.py deepl --deepl-key "$KEY"  # synthetic ES only; not merged into PD pairs
```

Seeds consumed (checked in; required for CI):

| File | Role |
|------|------|
| `seed_index.json` | ≥40 hymns + PD verdicts + SRGH-called items |
| `seed_stanzas_en.json` | PD English stanzas |
| `seed_es_pairs.json` | Verified same-original EN–ES pairs |
| `glossary_hymn_allowlist.json` | Curated 25–40 terms for optional glossary merge |
| `srgh_called_hymns.json` | Hymns announced in SRGH recordings |

## Outputs

| File | Role | Provenance ID |
|------|------|---------------|
| `bhb_pd_index.jsonl` | PD index + metadata | — |
| `hymn_stanzas_en.jsonl` | English stanzas | — |
| `hymn_pairs_pd.jsonl` | Train PD pairs | **HYMN_PD** |
| `hymn_pairs_pd_holdout.jsonl` | Holdout (disjoint) | **HYMN_HOLD** |
| `hymn_candidates_synthetic.jsonl` | EN with no PD ES | **HYMN_CAND** |
| `glossary_hymn_candidates.json` | Proposed glossary terms | **G_HYMN** |
| `hymn_pairs_deepl.jsonl` | Optional DeepL synthetic ES; only exists after the `deepl` subcommand runs | — |

## Suggested SFT mix (historical proposal, not an executed run)

`0.80 * (S6 sources) + 0.15 * glossary + 0.05 * hymn_pairs_pd` — "S6" is the superseded
TranslateGemma sweep winner; a Gemma 4 run would substitute the verse + sermon components of
`training/run_gemma4_e4b_domain_sft.sh`.

Warn if train hymn pairs exceed 800. Prefer quality `same_original` pairs over thematic fuzzy matches.
