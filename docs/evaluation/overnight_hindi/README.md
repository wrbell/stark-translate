# Offline Hindi church-audio baseline — 2026-09-10

The actual Mac run completed **41 English utterances from 195 seconds of church
recordings**, plus eight theological text probes, with three translations per
input and model. Both E4B and E2B generated all 147 expected outputs. None was
empty or exhausted its generation budget. This establishes offline availability,
not Hindi accuracy or live Hindi support.

Parakeet transcribed each recording once. Both models consumed the identical
saved, hashed English transcripts using the shared Gemma 4 prompt, thinking off,
no adapter, no draft model, and no prompt cache. Segmentation used offline Silero
with 500 ms silence, 30 ms padding and an 8-second maximum. These are not live
caption endpoint measurements.

| Model | Audio-derived translate p50 / p95 | Text-probe translate p50 / p95 | Generated / expected |
|---|---:|---:|---:|
| E4B OptiQ 4-bit | 789 / 1,264 ms | 845 / 926 ms | 147 / 147 |
| E2B OptiQ 4-bit | 476 / 689 ms | 503 / 579 ms | 147 / 147 |

These durations cover translation only. E4B ran before E2B in separate processes;
this offline baseline was not an alternating real-time latency experiment.
Forty-one of the 49 unique inputs produced different E4B/E2B wording on the first
repeat. Differences need a Hindi-speaking reviewer: generated script, punctuation
and wording agreement do not establish semantic correctness.

For example, for “Justification is by faith alone, not by works of the law.”:

- E4B: “औचित्य केवल विश्वास से है, न कि नियम के कामों से।”
- E2B: “न्याय केवल विश्वास से है, विधि के कर्मों से नहीं।”

For “The Holy Spirit convicts the world of sin and righteousness.”:

- E4B: “पवित्र आत्मा संसार को पाप और धार्मिकता का बोध कराती है।”
- E2B: “पवित्र आत्मा दुनिया को पाप और धार्मिकता के लिए दोषी ठहराता है।”

These are unapproved predictions, retained to make terminology and meaning review
concrete. There are no local Hindi references, so no reference score or theological
term-recall score is claimed. The eight probes do not replace human-verified Hindi
verse pairs. Hindi QLoRA and live-language integration remain later decisions.

[Machine report](report.json) records counts and distributions. Raw [STT rows](stt/rows.jsonl),
[E4B rows](e4b/rows.jsonl) and [E2B rows](e2b/rows.jsonl) retain every prediction,
repeat, sample boundary and audio-input hash. Each directory contains its immutable
manifest and completion/model/runtime metadata. The [raw index](raw_index.json)
binds the archived copies to local execution paths and SHA-256 hashes. Original
recordings were read only and are not copied into this report.

Reproduction commands and the required review boundary are in
[the offline baseline guide](../../offline_hindi_baseline.md). All artifacts are
marked `evaluation_only`; nothing was approved or exported into training data.
