# Aborted normalized v1 screen and v4 integrity pilot

This archive preserves two separate cohorts. Neither qualifies a latency candidate
or changes a production default. The later full Standard v2 screen must be assessed
from its own complete evidence; these runs cannot fill its missing controls or
repeats. See the [follow-up protocol](../README.md).

| Cohort | Recorded source | Retained results | Interpretation |
|---|---|---|---|
| [Standard normalized v1](standard-screen-normalized-v1/run-summary.json) | `a8511eed53fde804a688b4dd56a52c077e773eca` | 13 of 96 planned results; 12 recorded integrity passes and one failure, all repeat 0 | Entire screen aborted after a physical STT trace mismatch; incomplete and ineligible for selection |
| [Measurement pilot v4](measurement-pilot-v4/run-summary.json) | `eddb0adb4741307680079802c47f33cb21d56aec` | Three Spanish E4B results: opening control, early 2 s / 160 ms arm, closing control; one repeat | All three recorded integrity checks passed after native-worker draining; a regression pilot, not candidate qualification |

The [machine-readable interpretation](interpretation.json) retains these distinct
statuses and source identities. The [original archive index](archive-index.json)
binds 28 files and remains unchanged by this review. A separate
[review index](review-index.json) binds the new narrative and lightweight audit.

## Why v1 stopped

The persisted failure was
`followup_std_v1_early_2s_160ms_e4b_r0_fleurs_development_es_es`. Its child returned
zero and source accounting was marked complete, but the integrity check reported:

```text
Replay PCM/identity integrity failed: physical_trace: Missing keyed partial physical result; physical_trace: unmatched started/finished identities
```

The source summary could be finalized before a cancelled native inference call
had finished. The [abort receipt](abort-receipt.json) records interruption of the
cohort runner and termination of its currently active child at
`2026-09-10T15:16:07.320201+00:00`; the
[runner log](standard-screen-normalized-v1/runner.log) preserves the failure,
subsequent completed results and the final `KeyboardInterrupt`. All 13 persisted
results remain archived, including the failed one. Twelve individual integrity
passes do not restore an incomplete screen's eligibility.

The separately retained [interrupted session artifacts](input-inventory.json)
belong to
`followup_std_v1_early_4s_160ms_e2b_r0_fleurs_development_en_en`. This is a fourteenth
started child, **not a fourteenth persisted benchmark result**. Its
[raw lifecycle](interrupted-v1-run/session_lifecycle_followup_std_v1_early_4s_160ms_e2b_r0_fleurs_development_en_en.json)
says `completed`, exit code 0, because the child handled SIGTERM as a graceful
stop. That original marker is preserved; it does not establish complete replay
coverage, EOF success or a completed benchmark result. Nine raw artifacts remain
in `interrupted-v1-run/`; no replacement result JSON was synthesized.

## What v4 establishes

The [v4 protocol](measurement-pilot-v4/protocol.json) contains only Spanish,
Gemma E4B, baseline and early 2 s / 160 ms, with one repeat. Its
[runner log](measurement-pilot-v4/runner.log) records three successful finishes.
The [readable summaries](measurement-pilot-v4/run-summary.json) have no completion
errors and report passed replay integrity and complete source accounting. This
supports the recorded physical-trace completion check after the drain fix.

It is insufficient for the required two-of-three-repeat median gate, independent
confirmation, quality certification or default promotion. Summary endpoint
samples are descriptive; they are not a paired speed report. The retained runs
have 6–8 finals each, and the archive makes no p95 claim. Such a claim requires at
least 100 eligible fixed-span observations in **each individual** opening,
candidate and closing run, never pooled across repeats, endpoints or languages.
Both cohorts enabled research traces with capacity 131,072. No native display
visibility, bilingual approval, microphone or speaker functionality is certified.

## Evidence map and review limits

Each cohort folder contains its original protocol bytes, provenance, runner log,
readable run summaries, raw-evidence archive, source-code archive and evidence
index. The raw archives retain 40 indexed members for v1 and 10 for v4: original
result JSON, provenance, available session logs and CSV metrics. Their
[v1 index](standard-screen-normalized-v1/evidence-index.json) and
[v4 index](measurement-pilot-v4/evidence-index.json) record member names, sizes and
SHA-256 hashes. Full source provenance records 363 v1 and 370 v4 code-file hashes;
the creator verified those bytes against the respective source commit. These
source archives do not include model weights or audio.

The preserved [archive creator](archive_v1_v4.py.txt) and
[cohort helper](archive_cohort.py.txt) show copy/hash and archive readback checks.
Original raw payloads were not rewritten; no audio is included. Absolute local
paths in provenance describe the original execution environment and are not
portable download locations.

The independent [metadata review](metadata-review.json) rehashed 23 small files,
checked all 28 indexed file sizes, cross-checked nested/top-level indexes, frozen
protocol/source identities, runner/result counts and interrupted-session metadata.
It found no metadata inconsistency. **It did not reopen or hash the four compressed
archives or the larger interrupted diagnostics payload**, recompute summaries from
large raw results, or repeat model execution. Those five payload hashes remain
recorded producer evidence pending any later independent byte verification in an
idle window. This distinction preserves the current active benchmark's isolation.

The [review helper](review_metadata.py) uses only Python 3.11 standard-library
metadata operations and writes a new receipt exclusively. To repeat this limited
audit, pass a new output path with `--output`; it never decompresses an archive.
The review ran no models, audio devices, Git mutation, build or installation.

A subsequent idle-window [complete byte verification](complete-byte-verification.json)
read and matched all 28 originally indexed files, including the four compressed
archives. It supplements the earlier metadata-only review without changing its
recorded scope or the original measurements.
