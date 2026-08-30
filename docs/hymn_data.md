# Hymn-domain translation data

Operator notes for the copyright-safe EN↔ES hymn slice used as **~5% spice** in TranslateGemma / Gemma 4 SFT mixes.

## Audience

Volunteers and engineers preparing translation fine-tune mixes. **Not** for Whisper / W16 sermon audio.

## Quick regen

```bash
python training/prepare_hymn_corpus.py all --seed 42
```

Offline after seeds are checked in. Optional DeepL (never merged into PD pairs):

```bash
python training/prepare_hymn_corpus.py deepl --deepl-key "$DEEPL_API_KEY"
```

Glossary review (no silent merge of 80 terms):

```bash
python training/build_glossary.py --from-hymns bible_data/hymns/glossary_hymn_candidates.json
# Merge only curated allowlist when intentionally exporting:
python training/build_glossary.py --merge-hymn-allowlist bible_data/hymns/glossary_hymn_allowlist.json --output /tmp/glossary_with_hymns
```

## Copyright

- Train on public-domain hymn **texts** (author death ≥70 years / pre-1929 publication), not the New Believers Hymn Book (2019) / John Ritchie **compilation**.
- Lemmel “Turn Your Eyes Upon Jesus” (d.1961) and similar modern texts are indexed as `excluded_modern` with **zero lyrics** in train files.
- No hymn-singing audio in Whisper datasets. No New BHB / LEC full lyric dumps.

Details: [`bible_data/hymns/README.md`](../bible_data/hymns/README.md), [`bible_data/hymns/sources.json`](../bible_data/hymns/sources.json).

## Suggested mix (documentation only)

`0.80 * (current S6 sources) + 0.15 * glossary + 0.05 * hymn_pairs_pd`

Do **not** treat hymns as a third pillar beside verse + sermon. Warn if `hymn_pairs_pd.jsonl` exceeds ~800 rows.

## Provenance IDs

See [`data_provenance.md`](./data_provenance.md): **HYMN_PD**, **HYMN_HOLD**, **HYMN_CAND**, **G_HYMN**.
