# Normalized replay integrity pilot — September 10, 2026

Two real-time English E4B baseline runs on frozen source
`a8511eed53fde804a688b4dd56a52c077e773eca` completed with exit 0, six finals and
45 emitted translated previews each. Both retained source ledgers were complete;
actual runtime identity and saved final PCM/physical STT trace checks passed with
no completion errors. This validates the measurement path sufficiently to start
paired screening. It does **not** demonstrate an optimization, sub-second delivery,
visible-browser timing, or human translation quality.

The input is the five-recording public development replay, 46.94 seconds, SHA-256
`2169fde4d0b8b2f1d2e8df9f26f5b066337da454ccca56f844234cdf34d350c4`.
Each original recording received its declared fixed RMS/peak-limited level transform;
the live pipeline used gain 1. The failed original-level pilot remains separate in
[../input-level-pilot/](../input-level-pilot/README.md). Original references and audio
are evaluation-only; no local approval or training eligibility was assigned.

| Run | Silence endings | Forced smart cuts | Finals | Previews | Production transcript WER |
|---|---:|---:|---:|---:|---:|
| Opening baseline | 5 | 1 | 6 | 45 | 7/102 = 6.86% |
| Closing baseline | 5 | 1 | 6 | 45 | 7/102 = 6.86% |

There are no candidate arms and only one repetition. Endpoint samples remain
separate in [run-summary.json](run-summary.json); neither endpoint cohort supports
a p95 claim. The reference set contains no glossary term opportunities. Integrity
checks establish recorded shapes, hashes, counts and physical-call identity, not
waveform equality or semantic speech coverage.

[raw/](raw/) preserves the two full run JSON files, exact provenance and production
CSVs. [runner.log](runner.log) records both successful completions.
[evidence-index.json](evidence-index.json) hashes every archived evidence payload.
Raw absolute paths remain unchanged; local audio is not copied into this report.
The original/derived audio identities and normalization recipe travel in provenance,
so this is portable report evidence, not a self-contained audio replay bundle.

[archive_cohort.py.txt](archive_cohort.py.txt) reproduces the archive with the Python
standard library. [WORKFLOW.md](WORKFLOW.md) specifies the pending larger-cohort
archival and reporting procedure, including failed runs and unqualified arms.
