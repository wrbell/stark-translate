# Hymn corpus coverage

- Index hymns: **42**
- Public domain: **40**
- Excluded modern: **2**
- EN stanzas: **139**
- Train PD pairs (HYMN_PD): **52**
- Holdout pairs (HYMN_HOLD): **13**
- Synthetic candidates (HYMN_CAND): **74**
- Rejected thematic: **1**
- Glossary allowlist (G_HYMN curated): **36**
- SRGH-called indexed: **7**
- SRGH-called with PD pairs or excluded: **3/7**

## Do not train (excluded_modern)

- `pd-turn-your-eyes-upon-jesus` — O soul, are you weary and troubled? (Helen Howarth Lemmel, d.1961): Helen Howarth Lemmel d.1961 — not public domain. Zero lyrics in stanza/pair files.
- `pd-great-is-thy-faithfulness` — Great is Thy faithfulness, O God my Father (Thomas O. Chisholm, d.1960): Thomas O. Chisholm d.1960; 1923 publication still under copyright in many jurisdictions. Index only.

## Suggested SFT mix (not an executed run)

`0.80 * (current S6 sources) + 0.15 * glossary + 0.05 * hymn_pairs_pd`

Hymns are a small spice (~5%), never a third pillar beside verse + sermon.
