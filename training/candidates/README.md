# Jacobo/Santiago preference candidates — 2026-09-10

[`jacobo_candidates_20260910.jsonl`](jacobo_candidates_20260910.jsonl) contains 60
**unapproved** EN→ES preference triples: 30 person-name examples and 30 epistle-title
examples. These are 20 original synthetic teaching sentences in three original framing
variants, not 60 independent observations. They contain no sermon extracts, Bible-edition
quotations, evaluation examples or generated inference outputs.

Each row records its author, creation date, core/frame ids, source-text hash and the
deliberate rejected-name substitution. `approved_for_training: false` and
`review_status: unapproved` keep the file out of SFT/CPO. The CPO loader rejects those
markers before importing Unsloth. Do not remove the markers to bypass review.

The [manifest](jacobo_candidates_20260910.manifest.json) records file hashes and zero
exact normalized text overlaps against the seven available local holdout/canary files.
The required `bible_data/aligned/verse_pairs_test_v2.jsonl` was absent on this Mac.
**The full holdout gate is therefore pending**, as are semantic overlap review and
bilingual approval. Exact string checks cannot establish absence of paraphrase or topic
overlap. Holdouts are read only for exclusion, never sampled as training sources.

Reproduce to a new output path, without model calls or ML packages:

```bash
python -S training/candidates/build_jacobo_candidates.py --output /tmp/jacobo_candidates.jsonl
# On WSL, additionally supply the real mounted v2 holdout if needed:
python -S training/candidates/build_jacobo_candidates.py \
  --holdout /mounted/verse_pairs_test_v2.jsonl --output /tmp/jacobo_candidates_wsl.jsonl
```

Before any CPO experiment, rerun against all required WSL holdouts, review the Spanish,
proper-name conventions and rejected alternatives with a bilingual reviewer, and create
a separate approved/scored export compatible with the trainer's existing margin fields.
Keep this candidate artifact and original corpora unchanged. No training, adapter export,
Mac A/B, #136 acceptance or #137 real-correction evidence is claimed here.
