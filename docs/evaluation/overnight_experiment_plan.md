# September 10 autonomous experiment plan

This extends the [earlier paired screen](mac_v2026_14_screening/README.md). It does
not repeat the earlier silence/cadence/routing sweeps without a new hypothesis.
Standard E4B remains the production default. Experimental scheduling and routing
stay opt-in until their measured behavior and quality are reviewed.

## Outcome measures

The primary target is estimated speech end to final payload readiness, schema 2,
with an aspirational median below 1 second. Silence endings, forced/smart cuts,
EOF, Pause and Stop remain separate. Fast preview responsiveness targets a median
of at most 800 ms from first captured speech sample to first translated preview.
Update gaps are measured within an utterance; a long spoken utterance is not itself
a pipeline stall.

For actual visible audience browsers, report receipt-to-render overhead and
speech-end-to-acknowledgement separately. The latter is an upper bound including
return-network time. Hidden tabs, accelerated replay and missing acknowledgements
cannot satisfy the delivery gate. Require at least 95% acknowledgement coverage.

A promising screening change improves the targeted median by at least 15% or
150 ms without worsening the other p95 by more than the larger of 5% and 100 ms.
It must preserve source coverage, avoid new failures and be confirmed on both
historical 150-second recordings. At least 100 eligible final observations across
multiple runs are needed before a p95 promotion claim. A 45-second screen selects
follow-ups; it does not certify a production default.

Report STT/translation outputs, exact matching final bounds, missing source
intervals, memory, queues, errors and resolved settings. Output agreement is not
reference quality. Human bilingual review, natural Spanish and representative
references remain separate pending requirements. Quality-changing experiments
require actual changed examples, not an invented quality percentage.

## Independent hypotheses

Each runs alone before any combined configuration is considered. Partial cadence
stays 0.6 seconds; the standard baseline uses 0.5-second silence.

| Experiment | Hypothesis / failure to watch |
|---|---|
| Latest partial queue | Replace obsolete queued recognition work while preserving FIFO finals; verify that fresh previews do not starve. |
| Queued caption delivery | Slow network clients should not hold inference; measure queue wait and real browser delivery, including dropped/stale updates. |
| Early first preview | Start the first preview at 0.35 seconds while retaining later cadence; watch additional short-input recognition errors and contention. |
| Exact Marian memo | Repeated identical partial text can reuse CPU translations; measure hit rate and terminology identity. |
| Short-pause preview | Produce a revisable preview after a 128 ms pause without finalizing the utterance; watch false boundaries and extra work. |
| Short-pause speculation | Start a private final candidate, reuse only after ordinary final STT exactly confirms the input; bound attempts and count waste. |
| Clause previews | Limit long-utterance preview delay with four-second clause opportunities; verify ordering and final coverage. |
| Rolling STT window | Reprocess less audio for previews; inspect lost context, revised wording and total audio seconds decoded. |
| Streaming STT | Carry recognition state across preview updates; measure correctness and resets on final/pause/session boundaries. |
| VAD worker, Torch | Move serialized VAD work off the hot loop; inspect queue delay and source timing. |
| VAD worker, ONNX | Test backend isolation together with the alternative runtime; compare to worker/Torch and the historical inline ONNX result. |
| Exact prompt-prefix cache | Reuse shared prompt KV state, preserving exact token prefix; numerical batch differences can still alter output. |
| MLX cache 512 / 1024 MiB | Test allocator reuse against the 256 MiB baseline; report memory growth and latency together. |

The prompt-cache development probe produced identical text in 35/36 paired cases;
one E4B case changed capitalization. Therefore this is a quality-changing
experiment until evaluated, despite using an exact token prefix. MTP remains off:
its previous bounded negative result does not justify another overnight sweep.

## Execution and evidence

1. Rehearse the integrated operator with controlled English/Spanish recordings;
   verify readiness, Pause/Resume, language restart, Stop, review, support and summary.
   Microphone and physical output devices are explicitly deferred to tomorrow.
2. Freeze the source checkout, audio hashes, resolved settings and model revisions.
   Run one inference process at a time. Use the same visible audience browser and
   stable ports; it reconnects across runs.
3. Run three paired real-time replays per configuration, alternating E4B/E2B order
   and reversing candidate order on the middle repetition. Include baseline anchors
   at each repetition's beginning and end. Preserve failed attempts.
4. Confirm worthwhile candidates on both historical recordings. Combine only changes
   with evidence that their individual behavior warrants it; report rejected ideas.
5. Run a full-hour standard replay and a separate full-hour CPU Lite replay, with
   memory/queue/persistence/completion checks and a written rehearsal note. Include a
   complete recorded hymn and spoken material when assessing the laptop-stand-in
   rehearsal issue. Controlled audio is not physical microphone certification.
6. The separately requested offline Hindi R&D baseline is already archived. No
   further Hindi work belongs to this EN↔ES latency program; it does not add live
   Hindi or establish Hindi translation accuracy.
7. Refresh reports, release-independent installation evidence, backlog and guides;
   run the full checks and review, then merge validated source through PR #192.

## Lite gates and handoff

CPU Lite uses multilingual small Whisper int8 plus Marian CPU finals. Its target
floor to certify is four physical AVX2 x86 cores, 8 GB RAM and an SSD; Mac CPU tests
are useful evidence but do not certify x86. Targets: preview p50/p95 ≤1.5/2.5 s,
silence-final p50/p95 ≤2/4 s, and peak RSS ≤3.5 GiB. Optional E2B CPU quality mode
uses the same runtime with a pinned native llama.cpp server and targets 16 GB RAM.

RTX2070 Lite targets the original 8 GB card and 16 GB system RAM, E2B Q4_K_M,
Whisper Turbo CT2 and Marian CPU. Targets: preview p95 ≤1.5 s, final p50/p95
≤1.5/3 s and GPU peak ≤6.5 GiB. These are hardware gates to execute, not measurements
made on the Mac. See [Lite setup and evidence](../lite_profiles.md).

Package publishing, release tags and PyPI are pending by user choice. Source push,
PR review, justified issue closure and merge into main are authorized. Natural
Spanish, two-speaker recordings, bilingual approvals, WSL training and actual
x86/RTX2070 execution remain explicit external dependencies.

Browser collection limitation observed at 05:46 UTC: the Mac native-app surface
reported a locked screen while the audience document still reported `visible` and
continued rendering acknowledgements. Lock onset was not observed. Reports must
distinguish this browser protocol telemetry from an attended physical display
check; server latency and source coverage remain independently measurable.

The [subsequent EN↔ES experiment proposals](../latency_next_experiments.md) target
earlier endpoint commitment, deadline-aware partial admission and measured TDT
readback overhead. They are research follow-ups, not current optimizations or
evidence that a default should change.
