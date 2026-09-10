# Supplemental overnight analysis contract

`tools/overnight_analysis.py` reads the frozen `overnight_bench.py` result format
from `911f4ae`. It imports only the Python standard library, never loads models,
and does not edit the benchmark, audio, observed predictions or evaluation
references. Run it independently while collection proceeds or after completion:

```bash
python tools/overnight_analysis.py \
  --input /path/to/frozen-checkout/metrics/overnight_screen \
  --metrics /path/to/frozen-checkout/metrics \
  --output /path/to/separate-report
```

The output contains `analysis.json` and `README.md`. An optional `--client-map`
selects a connection for each session when multiple visible clients exist.
Otherwise each session requires exactly one eligible visible connection. Output
must be outside the evidence directories. Re-running replaces only these report
files. Missing sessions or unreadable result files make the matrix incomplete;
no candidate is selected while the declared matrix is incomplete.

## Evidence validation

The analyzer retains failed arms and their reasons. It requires successful
subprocess/lifecycle/harness completion, unchanged diagnostic SHA-256 and size,
raw final/partial records equal to the harness observations, matching final
summary/configuration, source/model identity, and valid schema-2 real-time source
bounds and final timing. Every used raw artifact receives a path, byte size and
SHA-256. A missing browser log remains missing evidence; it never becomes zero
browser latency.

Comparisons require the same source/model-artifact cohort, clip, model size and
repeat. Only unique exact `sample_start`, `sample_end`, `sample_rate`,
`speech_end_sample`, endpoint and capture-clock matches are paired. Missing or
ambiguous controls are **unassessable**, with no fabricated zero loss or gain.
Every candidate is compared with both its opening and closing baseline; their
own difference is reported as baseline drift. Raw signed candidate-minus-control
deltas are retained, including negative improvements. Silence, smart cuts, hard
cuts, EOF and other endpoints are separate. Medians and nearest-rank p95 are
recomputed from raw observations, never averaged across runs.

## Browser and preview interpretation

Visible ACKs must match their session and event identity. Partial ACKs additionally
match revision, utterance, clock and delivery mode; final ACKs match the unique
CSV chunk, original sample bounds and diagnostic final event. Duplicate, stale,
hidden and mismatched events are excluded. Each client retains its own coverage
and timing. Browser latency is an upper bound including the return network hop;
server timestamps and browser clocks are never compared directly.

The controlled benchmark reuses one visible audience tab, but `client_id` is the
server process's socket object ID and changes on each subprocess launch. The
analyzer therefore binds the sole eligible visible connection **per session**,
never intersects those IDs across sessions. Paired `declared_display` measurements
retain both source session IDs and connection IDs and use only one connection
from each run; all other client distributions stay separate. A socket ID is not
proof of stable physical browser identity. Two eligible visible connections are
ambiguous unless an explicit `--client-map mapping.json` supplies a JSON object
of session IDs to connection ID strings. Missing/invalid entries fail closed.
`--client-id` remains a literal selector, useful only if the specified ID exists
in every compared run. Do not use a single historical socket ID for the matrix.

First-preview timing counts finalized utterances, with explicit missing-utterance
lists. Intermediate preview ACKs can be missing because queued delivery coalesced
updates; that alone does not establish lost first-preview coverage. Server update
gaps, ACK observation gaps, the last server preview-to-final gap, and previews
emitted after final readiness are reported separately. The last gap is not an
acoustic latency measurement.

Final readiness precedes sending and rendering. A preview emitted after that
readiness timestamp does **not** by itself prove a stale browser repaint. For each
client, `preview_acks_after_final_ack` joins the preview to its own final and
compares reconstructed server ACK receipt times (capture/speech-end reference plus
the recorded upper-bound duration). Differences within 0.2 ms are rounding ties.
Missing final ACKs or clock references are explicitly unassessable. This is ACK
observation order; it does not prove which caption remained on screen. The report
keeps both signals, and the automatic selector retains its conservative readiness
guard pending manual review; it does not automatically waive the quality guard.

Literal whitespace-token common prefixes and retracted suffixes describe visible
text changes. They are **not** word-level accuracy, WER, meaning preservation or
human approval. First/last preview strings, original revision event IDs and changed
matched final outputs remain inspectable. A smart-cut preview can include audio
later retained for the next final, so its rewrite is not necessarily an error.

## Conservative follow-up selection

`worth_confirming` is a research recommendation, never default promotion. For a
candidate to qualify:

- The declared matrix and at least three candidate repetitions are recorded.
- Every repetition has valid compatible opening/closing controls and unchanged
  final segmentation/source coverage.
- One bound visible connection per session has at least 95% timed final and first
  translated-preview utterance coverage in all three runs. Candidate server
  previews are complete and have no emission after final readiness requiring
  browser-order review. This conservative server guard is not a stale-repaint claim.
- Every paired endpoint and first-preview p95 respects the larger of a 5% or
  100 ms allowed regression. Server update and last-preview-to-final gaps also
  satisfy that guard. Peak process RSS and Metal memory each stay within 1 GiB
  of both controls; missing peak measurements fail this check.
- The same target (one final endpoint or first visible preview) improves its
  median by at least 15% **or** 150 ms versus both controls in at least two thirds
  of repeats, and in the pooled matched observations versus both controls.
- Optimizations with explicit execution evidence must actually execute: memo
  hits, speculative final reuse, prefix-cache hits, relevant preview kinds or
  scheduler events. A flag without the intended event cannot claim its benefit.
  First-preview timing, allocator and VAD-worker flags have limited activation
  telemetry; those limits are stated rather than converted into proof.

The 45-second screen does not support p95 promotion: at least 100 eligible finals
across multiple recordings/runs are required by the experiment plan, and endpoint
sample sizes still matter. Quality-changing candidates require inspection of real
changed outputs and the existing canaries before longer confirmation. Natural
Spanish and bilingual human review remain separate gates. No numerical lexical
proxy substitutes for them.

## Counters, trace and work

Counters come from the latest final session summary once per session. Trace
statistics describe retained events and retain the discarded-event count.
`emitted_preview_processed_audio_seconds_proxy` covers only emitted previews;
it excludes some discarded/suppressed work, ordinary final STT and speculative
STT. `total_decoded_audio_seconds` is therefore null. Existing traces cannot
justify a total compute reduction from this proxy alone.

Generation latency/prefill/decode may belong to speculation completed before the
final endpoint. They must not be added to post-end latency. Separate diagnostic
stage wall times describe the final path. `broadcast_ms` means completion of
queue admission for queued captions and awaited sends for the original delivery
mode; compare actual ACK upper bounds when evaluating audience experience.
