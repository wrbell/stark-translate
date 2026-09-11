# CPU Whisper small/base comparison, 2026-09-10

The recovered cohort completed **12 serial workers and 600 evaluation items** on
frozen source `eddb0adb4741307680079802c47f33cb21d56aec`, with no failed rows,
empty hypotheses, fallbacks or cleanup errors. **Base fails the declared quality
gate in both languages.** Its shorter isolated calls do not qualify it for an
opt-in pipeline trial or a production change. Whisper-small remains the CPU Lite
model; the separate [cadence result](lite-cadence-result.md) remains unchanged.

Each worker transcribed the same 50 original public development recordings for
its language, using small or base, over three repeats with alternating model
order. That is 100 unique recordings evaluated six times, not 600 independent
recordings. The [audited manifest](public_data/manifest.json) keeps development
separate from confirmation. Inputs were original 16 kHz mono recordings, not the
five-recording normalized concatenations used in the latency screens.

## Quality and measured calls

Word error rates were identical across all three repeats for each model/language.
Scoring applies NFKC, case folding and punctuation-to-space normalization;
accents and digits remain. WER is total word edits divided by reference words,
not mean sentence WER. No pipeline corrections enter these isolated scores.

| Language | Small edits / words | Small WER | Base edits / words | Base WER | Increase | Allowed increase |
|---|---:|---:|---:|---:|---:|---:|
| EN | 57 / 1,059 | 5.38% | 95 / 1,059 | 8.97% | 3.59 percentage points | 1.00 point |
| ES | 57 / 1,094 | 5.21% | 126 / 1,094 | 11.52% | 6.31 percentage points | 1.00 point |

The unchanged gate permits base WER no higher than small WER plus the larger of
one percentage point or 5% of small WER, with no glossary recall loss in any
repeat. Both languages fail the WER gate in every repeat. English has zero term
opportunities; Spanish has one glossary phrase type in one recording, `milenio`, and
both models retain it. This limited lexical check is not theological or meaning
approval.

Each timing row below is one worker's 50 whole-recording calls. The tail column
is its descriptive nearest-rank 95th percentile; **50 observations do not meet
the required 100 per run for a p95 claim**. No repeats or languages are pooled.

| Language | Model | Repeat | Call median ms | Observed tail ms | Peak RSS MiB |
|---|---|---:|---:|---:|---:|
| EN | small | 0 | 1,413.7 | 1,749.6 | 1,069.1 |
| EN | small | 1 | 1,435.7 | 1,913.9 | 1,296.0 |
| EN | small | 2 | 1,493.3 | 2,092.8 | 1,273.2 |
| EN | base | 0 | 479.2 | 948.1 | 747.0 |
| EN | base | 1 | 447.8 | 730.8 | 695.0 |
| EN | base | 2 | 494.3 | 832.7 | 716.2 |
| ES | small | 0 | 1,491.8 | 2,251.4 | 1,130.9 |
| ES | small | 1 | 1,604.9 | 2,253.4 | 963.8 |
| ES | small | 2 | 1,633.6 | 2,308.9 | 1,222.0 |
| ES | base | 0 | 517.2 | 901.1 | 698.7 |
| ES | base | 1 | 604.9 | 967.2 | 689.7 |
| ES | base | 2 | 603.0 | 1,058.1 | 691.1 |

Call timing excludes file decoding and model loading. Recorded load durations,
including built-in silence warmup, range from 1,302.5–1,714.0 ms for small and
470.8–579.9 ms for base. No evaluation recording was warmed or omitted. RSS is
each process's high-water mark including imports, loading and decoding; it is
not inference-only allocation or a memory-leak diagnosis. These Mac CPU numbers
do not certify x86 performance.

## Changed outputs and scope

Normalized output is stable across repeats within each model/language. Small and
base differ on 22 English recordings: base has more word errors on 16, fewer on
3 and the same count on 3. They differ on 35 Spanish recordings: 30 worse, 2
better and 3 equal. These counts describe the same 50 references once per
language, not pooled repeats.

For English item `1519-786465049904960526`, small exactly matches the reference;
base incurs five errors in 25 words, including `enrolling on a gap-year course`
becoming `I'm rolling on a cap your course`. Conversely, item
`1588-17646385371758249908` improves from one error to none: base correctly
recognizes `hit the photographer` where small produces `hid the photographer`.
Neither example overrides the complete cohort result.

For Spanish development item `1651-15070404026815095185`, the reference describes
combining travel and learning during a gap year. Small makes one word error;
base makes ten, including `aprendizaje` becoming `aprendizas de`. Base also has
isolated improvements: item `1555-1193954095328160401` preserves `camposanto` as
one word where small splits it, but changes `paloma` to `palomas`; total errors
fall from two to one. In the sole glossary item
`1631-13434181649028080555`, both retain `milenio`, while total word errors rise
from one with small to five with base. A retained glossary word cannot establish
that the surrounding sentence is correct.

The quality configuration is CT2 int8 on CPU, **beam 5, four CPU threads and one
worker**, forced language, no initial prompt and no fallback. Production CPU Lite
uses **beam 1 and three STT threads**, its pipeline prompt/corrections, VAD,
partials, queues and Marian translation. The isolated comparison therefore
establishes neither production WER nor caption-delivery latency. Its references
are public read speech without local bilingual approval, and remain ineligible
for training. No microphone, speaker or browser visibility was tested.

## Recovery and evidence

The [failed v1 index](quality/stt-cpu-development-failed-v1/index.json) remains
incomplete: all 600 evaluation rows failed because `soundfile` was missing in
the old isolated Lite environment. Model loading and built-in warmup had already
occurred, but evaluation decoding/transcription did not. Those failures are
environment evidence, not bad-recognition results; they were not overwritten or
merged into v2 quality scores.

Recovery used a fresh evaluation environment, preserving every original package
version and the working environments. It added SoundFile 0.14.0 plus colorama,
jiwer, lxml, portalocker, RapidFuzz, sacrebleu and tabulate. CT2 4.8.2,
faster-whisper 1.2.1 and NumPy 2.4.6 stayed unchanged; Torch and MLX are absent.
The retained environment validation records successful dependency and EN/ES file
decode checks, with no device or model test during environment preparation.

Actual selected artifacts are the managed Systran small model at revision
`536b0662742c02347bc0e980a01041f333bce120` and the Systran base snapshot
`ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66`. Requested and actual paths agree;
recorded model inventories remain identical across each model's six workers.
This review reused the retained inventory hashes and did not reopen weights.

Workers ran from 18:25:06.749490 to 18:36:15.769597 UTC. The
[recovery receipt](quality/stt-cpu-development-v2/recovery-state.json) recorded
completion at 18:36:18.903288 UTC. The [report](quality/stt-cpu-development-v2/report.json),
SHA-256 `0060b721479f6cdf5540dfd183b09ec3781e0b8f9d24d35e95cf5f0d04e2b0bf`,
and [raw index](quality/stt-cpu-development-v2/index.json) preserve all 12 workers.
The [evidence index](quality/stt-cpu-development-v2/evidence-index.json) binds their
original bytes, environment freezes, coordinator sources and independent review.
The [environment validation](quality/stt-cpu-development-v2/environment/validation.json)
records preserved package versions and actual file-decoding checks. No audio,
weights or environment directory is included in this portable metadata archive.

Independent retained-data review reproduced all 12 worker summaries and both
language decisions, checked source/config/model/manifest bindings, and recomputed
all 600 word-edit totals and per-worker call, engine and real-time-factor
distributions. No new inference was executed for the review. Independent Lite
deadline experiments retain their own unchanged guards and small-model baseline;
this rejected base comparison cannot qualify a combination or confirmation.
