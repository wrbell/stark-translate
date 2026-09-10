# Offline Hindi zero-shot baseline

Both Gemma 4 models completed 43 identical English inputs three times each using
the shared translation engine and the unmodified prompt (`none`). This probes
text translation only: it does not add live Hindi STT, language switching or TTS.

| Model | Observations | Median translation | p95 translation | Peak Metal allocation |
|---|---:|---:|---:|---:|
| E4B OptiQ | 129 | 891.9 ms | 1618.5 ms | 6.70 GB |
| E2B OptiQ | 129 | 552.6 ms | 922.2 ms | 4.39 GB |

E2B's median was 38.0% lower on this sample. These are isolated translation
durations, not caption-delivery measurements. Both processes completed normally;
all 258 generations stopped before their token budgets, and all three repeats
produced the same text for each input/model. Peak Metal allocation covers the
whole model process, including loading and warmup.

There are no Hindi references or approved Hindi terminology checks, so no WER,
chrF++ or canary accuracy is claimed. The original 18 English theological inputs
remain in this probe, but their Spanish required terms are deliberately disabled.
For example, the E4B output for “Christ is the Surety of a better covenant.” is
“मसीह एक बेहतर वाचा का surety है।” The retained English word is visible evidence
that successful generation alone does not establish readiness. E2B produces
“मसीह एक बेहतर वाचा का साखकर्ता है।” A Hindi reviewer must assess the wording.

The [full comparison](comparison.md) contains actual changed examples;
[comparison.json](comparison.json) contains the derived metrics. The anonymous
[review form](blind_review.jsonl) is blank. Keep `review_key.jsonl` away from
reviewers. No review has been completed by generating these files.

The [frozen manifest](../mac_v2026_14_manifest_v2.json) identifies the input text.
[Raw E4B](raw/quality_e4b_none_hi.json) and
[raw E2B](raw/quality_e2b_none_hi.json) retain all observations, resolved models,
revisions, package versions, code hashes, memory and exit status. Runs used cached
models in offline mode, sequentially, with no simultaneous GPU benchmark.

Reproduce with:

```bash
python tools/mac_evaluation.py quality --manifest docs/evaluation/mac_v2026_14_manifest_v2.json --output metrics/mac_roadmap/hindi_new --runs 3 --policies none --target hi
python tools/mac_evaluation.py report --manifest docs/evaluation/mac_v2026_14_manifest_v2.json --input metrics/mac_roadmap/hindi_new --output docs/evaluation/hindi_new
```

Live Hindi and Chinese remain later feature decisions.
