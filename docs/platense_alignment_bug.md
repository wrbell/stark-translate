# Platense Verse-Pair Alignment Bug — Postmortem and Fix

**Discovered:** 2026-04-29 (during Phase A staging of the Gemma 4 tuning project).
**Affects:** every training run that consumed `bible_data/aligned/verse_pairs_train.jsonl` from its creation on 2026-03-22 through 2026-04-29 — i.e. **all of S1–S9, all of W12–W16's bible-replay buffer, and any other run that pulled from the aligned verse corpus.**
**Status:** root cause identified, structural fix applied (`bible_data/aligned/verse_pairs_train_v2.jsonl`), original file preserved unchanged for historical record.

---

## (a) The problem

`bible_data/aligned/verse_pairs_train.jsonl` was the canonical EN↔ES Bible verse-pair training corpus, built by `training/prepare_bible_corpus.py` from per-source JSONL files derived from the [scrollmapper/bible_databases](https://github.com/scrollmapper/bible_databases) SQLite collection. The aligner joined per-source files by a flat `verse_id` integer column.

**The flat verse_id was the row index within each source's own ordering, not a canonical reference to (book, chapter, verse).** This worked for the Protestant-canon sources (KJV, ASV, BBE, WEB, YLT, RVR1909) because those translations all use the same 66-book ordering — every source's row N happens to refer to the same canonical verse.

**Platense (Biblia Platense / Straubinger) is Catholic canon — 78 books, 37,255 verses.** It interleaves 12 deuterocanonical books (Tobit, Judith, Wisdom, Sirach, Baruch, I/II Maccabees, Prayer of Manasses, I/II Esdras, Additional Psalm, Laodiceans) into the OT and adds the Greek expansions to Daniel/Esther. So Platense's row-order verse_id N maps to a *different* canonical verse than the Protestant sources' row-order verse_id N.

Concretely: by the time we reach what canonical numbering calls Psalm 1 (vid=13941 in KJV/RVR1909/WEB), Platense has accumulated 813 extra verses' worth of apocrypha, so Platense's vid=13941 is actually Job 14:15. The drift accumulates further through the Bible — by Revelation, the Catholic-vs-Protestant book_id offset is +7.

**The aligner produced ~120,777 silently-misaligned (EN, ES) pairs** — about half of the entire 241,591-pair corpus. Genesis pairs all happened to be correct (no drift had accumulated yet). Every pair from Psalms onward where `es_source == 'platense'` had EN paired with the wrong ES verse — sometimes wildly different (Psalm 1:1 EN ↔ Job 14:15 ES, etc.).

The bug is silently consistent: every `web→platense` pair at vid=13941 has the same wrong content, because the aligner deterministically does the wrong thing. It does not look like obvious noise — it looks like internally consistent (but wrong) data, which is why it survived undetected through ten months and nine training sweeps.

### How it was found

During Phase A of the Gemma 4 tuning project, while building `hybrid_runs/data/verse_1800.jsonl` (a 1,800-verse subset for the Phase B spike), I was sampling rows from `verse_pairs_train.jsonl` and printing samples for sanity check. The first few `*→platense` samples were obvious mistranslations:

> EN: "to abstain from things offered to idols, and blood..." (Acts 21:25)
> ES: "Finalmente, fue puesto en prisión por Aretas, rey de los árabes..." (a paraphrase of 2 Cor 11:32 — completely different verse)

Investigation traced this to the verse_id misalignment. `*→rvr1909` pairs were verified correct on the same vids. The bug was Platense-specific.

---

## (b) How it was fixed

The original scrollmapper SQLite databases were still on disk at `/mnt/d/Data/stt-data/bible_data/scrollmapper/formats/sqlite/` — including `SpaPlatense.db` and `SpaRV.db`, both of which carry `(book_id, chapter, verse)` columns. Critically, **the 66 Protestant-canon book NAMES are byte-identical between SpaPlatense_books and SpaRV_books** (only the book_ids differ — Platense Psalms = book_id 21, RVR Psalms = book_id 19 — because Platense numbers the apocrypha into the same sequence).

This permits an **exact structural join** with no text matching:

1. `tools/fix_platense_alignment.py` opens `SpaPlatense.db`, reads every `(book_name, chapter, verse, text)` tuple via the `SpaPlatense_books`/`SpaPlatense_verses` join.
2. For each Platense verse, looks up the canonical `verse_id` from `SpaRV.db` keyed by `(book_name, chapter, verse)`.
3. Apocryphal verses (book name not in the RVR1909 66-book set) are dropped — they have no Protestant-canon counterpart by definition.
4. Writes `bible_data/es/platense_realigned.jsonl` with the corrected canonical verse_ids and a provenance side-file `bible_data/es/platense_realignment_report.json`.

Result: **30,275 of 35,789 Platense verses (84.6%) recovered to their correct canonical positions; 4,375 dropped as apocryphal (correct behaviour); 1,139 dropped as no-match.** Zero collisions — every output verse_id is unique.

Spot-check, before vs after for `web→platense` at canonical vid 13941:

```
OLD (verse_pairs_train.jsonl):
  en: "Happy are those who do not follow the counsel of the wicked..."  (Psalm 1:1 ✓)
  es: "Entonces respondería a tu llamado, y Tú amarías la obra de tus manos."  (Job 14:15 — WRONG)

NEW (verse_pairs_train_v2.jsonl):
  en: "Happy are those who do not follow the counsel of the wicked..."  (Psalm 1:1 ✓)
  es: "¡Dichoso el hombre que no sigue el consejo de los malvados, ni pone el pie en el camino de..."  (Psalm 1:1 ✓)
```

5. `tools/rebuild_verse_pairs.py` then re-runs the join across all (en_translation, es_translation) combinations using the realigned Platense + the SQLite EN sources, writing `bible_data/aligned/verse_pairs_train_v2.jsonl` (87 MB, 265,271 pairs vs the old 80 MB, 241,591 pairs).

The original `bible_data/aligned/verse_pairs_train.jsonl` was **left unchanged** as the historical record of what every prior training run actually consumed. New runs should pull from `verse_pairs_train_v2.jsonl`.

### Files added by this fix

| Path | Tracked? | Description |
|------|----------|-------------|
| `tools/fix_platense_alignment.py` | yes | Re-key Platense JSONL using SQLite (book, chapter, verse) join + content validator |
| `tools/rebuild_verse_pairs.py` | yes | Rebuild the joined corpus using realigned Platense |
| `bible_data/es/platense_realignment_report.json` | yes | Per-book provenance + counts |
| `bible_data/aligned/verse_pairs_train_v2.rebuild_report.json` | yes | Pair counts per (en, es) combination, old-vs-new diff |
| `bible_data/es/platense_realigned.jsonl` | **no** (regen) | Corrected Platense — 27,258 verses, canonical IDs. ~7.7 MB; exceeds the 500 KB pre-commit large-file limit. Regenerate in ~5s via `python tools/fix_platense_alignment.py` (requires the scrollmapper SQLite at `/mnt/d/Data/stt-data/bible_data/scrollmapper/formats/sqlite/`). |
| `bible_data/aligned/verse_pairs_train_v2.jsonl` | **no** (regen) | Rebuilt joined corpus, ~87 MB. Gitignored by `bible_data/aligned/*.jsonl`. Regenerate via `python tools/rebuild_verse_pairs.py` after the realigned Platense exists. |

### How to regenerate the gitignored data files

```bash
# Step 1: ensure scrollmapper SQLite is present (once; ~30s clone)
git clone https://github.com/scrollmapper/bible_databases.git \
    /mnt/d/Data/stt-data/bible_data/scrollmapper

# Step 2: regenerate Platense (writes bible_data/es/platense_realigned.jsonl, ~5s)
python tools/fix_platense_alignment.py

# Step 3: regenerate the joined corpus (writes bible_data/aligned/verse_pairs_train_v2.jsonl, ~5s)
python tools/rebuild_verse_pairs.py
```

### What was NOT changed

- `bible_data/aligned/verse_pairs_train.jsonl` — preserved as-is for historical record.
- `bible_data/es/platense.jsonl` — preserved (the row-order-vid version that fed the bug). Don't use this directly for new alignment work; use the SQLite source via `fix_platense_alignment.py`.
- `bible_data/aligned/verse_pairs_test.jsonl` — was already a 2-line stub from a separate eval-set bug (see `docs/gemma4_tuning/phase_c_domain_sft.md` C0 prerequisite). The platense fix does not touch this.

---

## (c) Impact on previous training

Every TranslateGemma fine-tune (S1–S9) and every Whisper run that used `verse_pairs_train.jsonl` as its bible-replay buffer was trained on a corpus where **roughly half of the verse pairs were silent garbage** — internally-consistent EN/ES strings drawn from completely different verses.

### Quantitative scope

- `verse_pairs_train.jsonl` has 241,591 pairs.
- Of these, 120,777 had `es_source == 'platense'`. Of those, **~110,000 were misaligned** (Genesis is fine, but the bug starts somewhere in early Chronicles/Psalms and persists through Revelation).
- The correctly-aligned half (`*→rvr1909`, ~120,814 pairs) was unaffected.
- TranslateGemma sweeps subsampled from this corpus with `--max-pairs 20000` or 50000 (training/train_gemma.py:31-33). Assuming uniform sampling, **about 50% of every training batch was junk pairs** — meaningless EN/ES gradient signal pretending to be translation supervision.

### Qualitative impact on observed results

The plateau documented in `docs/gemma4_tuning/overview.md` ("S6 winner achieved COMET delta = -0.0002 vs baseline") had several attributed causes; this bug is now a **fourth, likely-dominant** contributor that wasn't in the original diagnosis:

1. (Original) Base model was already a translator → low headroom.
2. (Original) All sermon "ground truth" was synthetic → ceiling at teacher quality.
3. (Original) No preference signal → SFT can't exploit rank ordering.
4. **(New) ~50% of bible-pair training data was random EN-vs-different-ES junk.** This is the kind of corruption that:
   - **Caps achievable BLEU/COMET** (impossible to learn from random pairs).
   - **Drives the "verse-only collapses general translation" symptom** observed in S4 (sermon-only) — the verse pairs were partly *teaching the model to hallucinate*, so removing them helped marginally.
   - **Explains the "learning-rate cliff" between steps 50–150** documented for S2/S3 — at higher LR, the model overfit to noise faster; at lower LR it just couldn't extract enough signal to move.
   - **Is consistent with the theological-term ceiling at 6/8** — for any term whose rare training pairs landed disproportionately in the broken Platense half, the model never saw a clean (EN-term, ES-term) alignment.

### What this changes about Phase C planning

`docs/gemma4_tuning/phase_c_domain_sft.md` C1 ("Drop archaic Spanish from the primary mix") is **partly obsoleted** by this fix:

- **Before fix:** archaic Spanish (RVR1909, Platense) was the primary modern-register concern, and the recommendation was to push Modern English → broken Platense was unusable so we'd rely on RVR1909 even though it's archaic.
- **After fix:** Platense is now correctly aligned and is a *modernish* Spanish source (early-20th-century Spanish, somewhat formal but not as archaic as RVR1909). It's a usable modern-register source for Phase C, eliminating the need to either re-align Platense (option #2 in `phase_a_infrastructure.md`) or source new modern PD Spanish (option #3).

### What's still owed

1. **Re-baseline.** The "S6 winner" benchmark should be re-run on `verse_pairs_train_v2.jsonl` to establish a corrected baseline before claiming v1 ship gains. The previously-reported "-0.0002 COMET vs baseline" likely understates how bad S6 actually was relative to what was achievable on clean data.
2. **Update phase_c_domain_sft.md C1 corpus mix.** Modern-register requirement is now satisfiable; mixture should be revisited.
3. **Replicate the alignment audit on every other parallel corpus the project uses.** If Platense slipped through, others might have too — particularly anything sourced from a different-canon tradition (Greek Orthodox, Ethiopian, etc.). For now, only the Protestant-canon English sources are confirmed correctly aligned.

---

## How to avoid this class of bug going forward

- **Never align by row-order verse_id alone when sources may have different canonical scopes.** Always join on `(book, chapter, verse)` or some equivalent canonical reference.
- **Audit alignment quality with a content sanity check** at corpus build time: take 20 random pairs, render side-by-side, eyeball them. This bug would have been caught in the first manual review.
- **Tag every aligned-pair file with the join key used.** A `_join_method: "row_order_vid"` field in the manifest would have made this bug visible in any provenance audit.
