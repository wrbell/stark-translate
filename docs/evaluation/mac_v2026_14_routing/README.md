# Synthetic EN/ES routing exercise

The conservative routing predicate behaved as specified. It kept the two
allowlisted operational phrases on Marian and sent the non-allowlisted window
sentence to Gemma in every run. The legacy policy sent all three phrases to
Marian. **Keep conservative routing opt-in:** this is a functional path exercise,
not a natural-speech quality result or a reason to change the default.

All **24 runs completed with exit code zero**, producing 72 final captions and
90 partials. Each policy ran three alternating E4B/E2B pairs for English and
Spanish. The two roughly ten-second inputs were synthesized with Piper
`en_US-lessac-high` and `es_MX-claude-high`, with explicit pauses between phrases.
English used Parakeet MLX and Spanish used MLX Whisper. Silence remained
0.5 seconds, partial cadence 0.6 seconds, and playback speed 1×.

## Actual route execution

These counts come from the latest `session_summary` in each completed diagnostic
file. Each table row contains three runs and nine final captions.

| Policy | Model | Source language | Marian finals | Gemma final requests |
|---|---|---|---:|---:|
| Legacy | E4B | EN | 9 | 0 |
| Legacy | E4B | ES | 9 | 0 |
| Legacy | E2B | EN | 9 | 0 |
| Legacy | E2B | ES | 9 | 0 |
| Conservative | E4B | EN | 6 | 3 |
| Conservative | E4B | ES | 6 | 3 |
| Conservative | E2B | EN | 6 | 3 |
| Conservative | E2B | ES | 6 | 3 |

There is no explicit per-caption route field. The
[per-phrase analysis](routing-analysis.md) therefore labels individual routes
as **derived** from zero versus positive `tps_a`, reconciles them against these
authoritative counters, and checks the fixed predicate against the actual
recognized source and confidence. It preserves every individual observation
in [JSON](routing-analysis.json).

| Actual recognized source | Confidence in every repeat | Conservative shortcut |
|---|---:|---|
| Would you please take your seats? | 1.00 | Marian |
| Please turn to the next page. | 1.00 | Marian |
| The window is open. | 1.00 | Gemma |
| Por favor tomen asiento. | 0.90 | Marian |
| Pueden sentarse. | 0.91 | Marian |
| La ventana está abierta. | 0.94 | Gemma |

All 72 recognized captions match their declared scripts after punctuation,
case and whitespace normalization. English phrase one ends with a question
mark instead of the script's period. The confidence values are a rounded
**1 + mean-logprob proxy, not a calibrated probability**. They are safely above
the unchanged 0.8 threshold; no missing or low confidence was replaced with a
passing value. This exercise does not test behavior near that threshold.

The six translated strings were stable across models, policies and repeats.
In particular, both routes produced “La ventana está abierta.” and “The window
is open.” for the negative controls. This verifies execution and observed
output consistency; it does not establish that Marian and Gemma have equivalent
meaning quality on unseen speech.

## Measured cost of keeping the negative control on Gemma

Each cell below is p50 / nearest-rank p95 in milliseconds for the same window
sentence over three repeats. With three observations, p95 is the maximum.

| Source language | Model | Legacy speech end → final | Conservative speech end → final |
|---|---|---:|---:|
| EN | E4B | 881 / 903 | 1,382 / 3,252 |
| EN | E2B | 865 / 905 | 1,144 / 1,195 |
| ES | E4B | 1,721 / 1,789 | 2,215 / 2,915 |
| ES | E2B | 1,624 / 1,638 | 1,955 / 1,978 |

The conservative policy deliberately narrows eligibility for the shortcut; it
does not speed up these non-allowlisted examples. The
[canonical comparison](comparison.md) separates language, model, policy and
runtime cohort and includes partial timing. The supplementary analysis also
shows STT, translation and queue components per phrase. Timings are recomputed
from raw observations, not from averages of run percentiles.

Only 57 of the 72 finalized utterances produced a partial caption; the other
15 are counted explicitly in the analysis JSON. Reported first-partial timing
therefore covers emitting utterances, and update gaps include the scripted
silences. This probe does not establish continuous partial-caption coverage.

All final rows are schema 2 `replay_realtime` silence endpoints. They measure
estimated last speech frame to final payload readiness, excluding browser
delivery. **Visible final acknowledgment coverage is 0/72.** The synthetic
inputs, short phrases and fixed voices cannot establish the church caption
latency target or natural English/Spanish WER.

## Post-setup runtime identity

Every run shares the same recorded core source/package signature and startup
pipeline SHA-256
`63398f50a561198881dfb9b8123bdc531bfd2adbb6656e153a4b0d1c9fbf4515`.
This is a separate cohort from the earlier 45-second screen, following the
packaged VAD/setup change. A startup source-file hash does not claim to hash
imported bytecode, and the sequential policy groups are not a randomized trial.

Each run records loading the installed **Silero VAD 6.2.1** JIT artifact,
2,272,526 bytes, with SHA-256
`e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720`.
The preserved [screening cache proof](../mac_v2026_14_screening/raw/provenance/vad-cache-proof.json)
shows this weight file is byte-identical to the previously used local Hub copy.
The new runs record `source: installed_package`; they do not require a Hub
download. Resolved model revisions, actual Marian CT2 weight digests and process
peak RSS/Metal memory remain in every original run and the analysis JSON.

## Retained evidence

The [raw index](raw_index.json) preserves 96 unchanged files: 24 run JSONs,
24 final CSVs, 24 partial JSONLs and 24 diagnostic JSONLs, with hashes and sizes.
The [frozen audio manifest](../mac_v2026_14_routing_synthetic.json) contains the
scripts, generation positions, voices and audio hashes; the
[experiment specification](../mac_v2026_14_routing_experiments.json) records the
flags. The existing [audio builder](build_audio.py) is preserved separately.

With the original metric paths present, regenerate the report without inference:

```bash
python tools/mac_evaluation.py report \
  --manifest docs/evaluation/mac_v2026_14_routing_synthetic.json \
  --input metrics/mac_roadmap/routing \
  --output docs/evaluation/mac_v2026_14_routing
```

No human reference, approval, natural-audio score or browser latency observation
was created for this report. Bilingual meaning review and natural-speech gates
remain pending.
