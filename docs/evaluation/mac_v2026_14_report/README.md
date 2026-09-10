# Mac baseline and quality results

Keep E4B as the initial default. E2B is faster on the tested inputs but misses more
of the prescribed English-to-Spanish terminology checks. Bilingual review remains
pending, so the measurements do not justify a default change.

| Identical EN→ES text inputs | E4B | E2B |
|---|---:|---:|
| Translation median, original prompt | 839.9 ms | 516.6 ms |
| Translation p95, original prompt | 1442.6 ms | 832.4 ms |
| Full-term canary checks | 13/18 | 11/18 |
| chrF++ against aligned verse references | 44.4 | 45.1 |
| Translation median, terminology examples | 955.1 ms | 553.1 ms |
| Full-term canary checks, terminology examples | 15/18 | 14/18 |

These are isolated translation durations. E2B's original-prompt median is 38.5%
lower on the 43 English inputs; it passes two fewer terminology checks. The chrF++
figures measure agreement with a particular reference wording, not a percentage
of correct translations. The 25 Spanish inputs have a separate distribution:
original-prompt medians are 989.1/618.9 ms and chrF++ 48.7/46.0 for E4B/E2B.

The [complete comparison](comparison.md) shows all 18 canaries, actual changed
translations, p50/p95, partial delay and update gaps. Each configuration ran three
times. [JSON results](comparison.json) preserve endpoint and runtime cohorts.
The anonymous [review form](blind_review.jsonl) is still blank; the answer key
must be kept separate from reviewers. Examples improve the automatic canaries,
but remain opt-in until meaning and terminology review is completed.

## Speech recognition

Both engines independently processed the same 50 English and 11 Spanish audio
candidates, three times each. On English, Parakeet measured 132.2 ms median /
179.8 ms p95; Whisper measured 568.5 / 639.2 ms. On Spanish, the corresponding
figures were 131.1 / 212.9 ms and 557.8 / 4311.6 ms. Whisper was given the language;
Parakeet uses automatic language detection. There are zero approved transcript
references, so WER and theological-term recall are unavailable. Speed alone does
not change the current English-Parakeet / Spanish-Whisper defaults.

## Historical replay interpretation

All 18 real-time replays completed: both models, three repeats, two 150-second
English sermon clips and a separately labeled synthetic Spanish clip. Collection
spanned instrumentation and lifecycle fixes. The report keeps different recorded
source/configuration cohorts separate; these repeats cannot be pooled into one
current-runtime result. The later frozen 45-second experiment screen supplies a
same-source baseline for comparing scheduling and routing options.

The historical runs have no browser acknowledgments. Their schema 2
`speech_end_to_final_ms` measures estimated last speech frame to final payload
readiness on the server, not display delivery. Their medians do not establish the
sub-second caption goal. Silence, smart cuts, hard cuts and EOF remain separate;
older `e2e_latency_ms` values retain their processing-time meaning.

The first historical clip's local PCM was verified against its parent sermon at
1290 seconds (21:30); the second at 1170 seconds (19:30). An older narrative says
20:30 for the first clip. The frozen audio hashes and PCM match establish the
current input identity; archived measurement values have not been changed.

## Reproducibility and remaining gates

[Raw run files](raw_index.json) retain source revisions/hashes, package versions,
resolved settings, model revisions, lifecycle memory and exit status. Raw CSVs
and partial observations are copied alongside them. Existing local audio is
identified by hash in the [manifest](../mac_v2026_14_manifest_v2.json); it is not
included in this report. Original text-quality observations are retained in
[`mac_v2026_14_quality/raw`](../mac_v2026_14_quality/raw/).

The structural reference repair changed only reference metadata and scores, not
model inputs, predictions or timings. One ambiguous verse reference remains
unscored. These references have not received bilingual human approval.

Natural Spanish audio, at least 50 approved natural utterances per language,
bilingual meaning/terminology review, two-speaker labels and physical second-output
testing remain pending. The [offline Hindi baseline](../mac_v2026_14_hindi/README.md)
has its own report and does not enable live Hindi.
