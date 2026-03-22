# Sermon Data Sorting & Training Cutoff Policy

## Data Inventory (as of 2026-03-22)

35 sermon recordings in `stark_data/raw/midwest/`, each with a companion metadata JSON.

| Type | Count | Date Range |
|------|-------|------------|
| `gospel` | 32 | Jun 2025 – Mar 2026 |
| `conference` | 2 | Oct 18–19, 2025 |
| `gospel_baptism` | 1 | Dec 7, 2025 |
| `ministry` | 0 | (future) |
| **Total** | **35** | **Jun 2025 – Mar 2026** |

24,595 transcribed chunks in `ablation/sermon_whisper_chunks_expanded.json` (after dedup + ≥20 char filter).

## Classification Rules

Type is parsed from the `title` field in each metadata JSON:

| Pattern | Type |
|---------|------|
| `"Conference"` in title | `conference` |
| `"Baptism"` in title | `gospel_baptism` |
| `"Gospel"` in title (default) | `gospel` |
| anything else | `ministry` |

Date is extracted via regex from the title:
- Parenthesized: `(M/DD/YY)` or `(MM/DD/YY)` — e.g., `Gospel Message (10/12/25)`
- Conference inline: `10/18/25` at end of title
- Year mapping: `25→2025`, `26→2026`

## Training Cutoff: 2026-03-14

**Rule**: Only data from **before** March 14, 2026 may be used for Whisper fine-tuning training. Data on or after this date is evaluation/test only.

**Rationale**: We train on historical sermons and evaluate on unseen future sermons. This prevents data leakage — the model should generalize to new speakers and topics, not memorize recent services it might encounter in live deployment.

| Split | Rule | Current Count | Hours |
|-------|------|---------------|-------|
| **train** | `date < 2026-03-14` | 34 sermons | ~35 hrs |
| **eval** | `date >= 2026-03-14` | 1 sermon | ~0.75 hrs |

As new sermons are downloaded, they automatically go into the eval split until the next training cycle explicitly moves the cutoff forward.

## Workflow

### Automated classification

```bash
python tools/sort_sermons.py --input stark_data/raw/midwest --cutoff 2026-03-14
```

This produces `stark_data/sermon_manifest.json` with enriched entries:
```json
{
  "video_id": "I6dy6o_ewDk",
  "title": "Gospel Message (10/12/25)",
  "type": "gospel",
  "date": "2025-10-12",
  "split": "train",
  "speakers": "Josiah Pratt and Jim Clark Sr",
  "duration_seconds": 3230,
  "wav_path": "stark_data/raw/midwest/Gospel_Message_(10_12_25)_I6dy6o_ewDk.wav"
}
```

### Using the manifest downstream

`training/prepare_whisper_dataset.py` can filter by manifest split:
- Load `sermon_manifest.json`
- Build source→split mapping
- Only include chunks whose `source` maps to `split == "train"`

### Tagging expanded chunks

```bash
python tools/sort_sermons.py --input stark_data/raw/midwest --cutoff 2026-03-14 \
    --tag-chunks ablation/sermon_whisper_chunks_expanded.json
```

This adds a `split` field to each chunk in the expanded JSON based on its `source` → date mapping.

## Future Considerations

- **New downloads**: Any sermon downloaded after the cutoff automatically gets `split: eval`
- **Cutoff advancement**: When starting a new training cycle, update the cutoff date and re-run the sort script
- **Ministry meetings**: When ministry/teaching recordings are added, the `"ministry"` type will activate — no code changes needed
- **Multi-assembly**: If sermons from other assemblies are added, extend with an `assembly` field parsed from the `channel` metadata
