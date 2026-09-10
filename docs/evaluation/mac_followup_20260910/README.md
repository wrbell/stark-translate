# Mac EN↔ES follow-up implementation and experiments

Work is in progress on `codex/mac-en-es-closeout`, tracked by
[draft PR #196](https://github.com/wrbell/stark-translate/pull/196). Defaults remain unchanged;
no release or package publication is authorized for this follow-up. The
[43-item audit](backlog-audit.md) preserves each original acceptance condition.

## Completed evidence

- [Final c13 source validation](final-c13f51f/source-validation.md): completed CPU,
  static, text-only GPU and pre-commit evidence, with exact-source CI kept separate.
- [Natural hymn control](final-c13f51f/hymn-capture.md): the 350-second silent file
  replay completed without entering music hold; it does not validate hold recovery.
- [Hymn text-boundary packet](final-c13f51f/hymn-boundary.md): 102 calls through the
  actual correction stage, using supplied raw/newline/quoted hypotheses. Natural
  boundaries and bilingual meaning still require independent review.
- [Model-source continuity](final-c13f51f/model-source-continuity.md): additive c13
  evidence preserves the earlier inventory and its optional/offline residuals;
  it does not claim a global B615 clearance.

- [Standard endpoint/deadline screen](standard-screen-result.md): 96 technically
  valid replays, zero of 24 arms qualified. Median-only improvements failed other
  quality, preview, queue or memory guards; defaults remain unchanged.
- [Spanish Parakeet screen](spanish-parakeet-result.md): 18 valid replays, zero
  of two arms qualified. Faster final delivery did not preserve all preview and
  queue guards. Both cohorts now include immutable raw and source archives.
- [CPU Lite cadence screen](lite-cadence-result.md): 24 technically valid replays,
  zero of four arms qualified. Longer cadence helped some final medians but
  reduced preview coverage; CPU Marian finals and 0.6-second cadence remain.
- [CPU Lite independent deadline screen](lite-deadline-result.md): 24 technically
  valid replays at unchanged 0.6-second cadence, zero of four arms qualified.
  Both languages have formal no-combination plans; no rejected candidate enters
  confirmation. Raw output, source and independent reviews are retained.
- [Capture loss accounting](capture-loss-accounting.md): observed worker FIFO loss,
  unknown PortAudio overflow and terminal accounting are separate. Mocked checks
  passed; no additional live reliability certification is claimed.

- [Python 3.11 Stop repair](stop-cancellation.md): replace the queue wait that
  could swallow the first cancellation. The deterministic production regression
  fails with the old wait and passes with the repair; 88 focused checks and
  20 subtests pass without native devices.
- [Capture repair validation](capture-source-validation.md): full Python 3.11/3.12
  CI on `f7b959d` passed 2,834 tests plus 20 subtests per interpreter. The installed
  Lite audit checked 60 dependencies with zero known vulnerabilities; the local
  project was skipped, `[eval]` was outside that audit and the broader audit job
  was skipped. All six earlier fixture failures remain recorded.
- [Earlier source validation](current-source-validation.md): Python 3.11/3.12
  suites and lint passed on `476e349`, including the startup fallback cases.
  Dependency-audit jobs were skipped; no new audit clearance is claimed. The
  separate receipt preserves the earlier failed suite and observation-time records.
- [Public input audit](public_data/README.md): 100 EN + 100 ES original FLEURS
  recordings, 50 development / 50 confirmation per language, pinned source,
  hashes, independent archive verification and portable reference metadata.
  Public read speech is engineering evidence, not church or local human review.
- [CPU STT comparison](cpu-stt-comparison.md): all 12 workers and 600 items
  completed. Whisper-base reduced isolated call time but raised WER beyond the
  unchanged allowance in both languages; no base pipeline trial follows.
  Failed-v1 dependency errors and the successful fresh environment are retained.
- [Development STT comparison](quality/stt-mlx-development-report.json): all
  twelve isolated runs completed, 50 recordings per engine/language/repeat.
  Three repeats alternated model order; no fallback or failed item occurred.
- [Parakeet profiling](profiling/development-r1.json): all 18 paired controls
  and profiled calls had identical outputs. Actually sampled scalar readbacks
  represented median 13.23% of profiled STT wall time; observer perturbation was
  median +1.96%, p95 +21.54%. This authorizes a separate joint-evaluation trial,
  not a speedup claim. Scalar waits include pending MLX computation.
- [Joint-evaluation trial](profiling/joint-development-r1.json): all 18 paired
  outputs remained exact. Median paired savings were 15.13 ms (9.18%), with
  positive medians in all three repeats. This isolated gain is below the
  15%/150 ms promotion gate; no production integration or caption gain is claimed.
- [Compiled decoder trial](profiling/compile-development-r1.json): all 18 triplets
  completed, but exact-output checks failed in six and first-call cost produced
  a worse tail. Compilation did not improve on joint evaluation at the paired
  median. This candidate is rejected; no runtime switch was added.
- [Fixed-reference E4B/E2B translation](translation-comparison.md): all twelve
  runs completed with unchanged outputs across repeats, no empty outputs or
  exhausted budgets. E2B reduced isolated translation time but scored 11/18
  lexical canaries versus E4B 13/18, with slightly lower reference chrF in both
  directions. The report retains every canary and changed public examples.
  A separate unreviewed blinded packet is prepared; its key stays private.
- [Attended operator checks](../attended_mic_20260910/README.md): EN and ES
  microphone readiness/stop and EN pause/resume passed. The room had no detected
  speech. A separately labeled file replay produced a visible bilingual final;
  review drafts survived save and reload without becoming approved training data.
- [Security feasibility](../security_feasibility_20260910/README.md): remaining
  live Marian/Piper fallbacks use pinned model revisions. An isolated compatible
  Torch/audio installation passed native imports, bundled CPU VAD and its
  51-package audit. The [full-application candidate](torch-full-application-candidate.md)
  also passed installation, native imports and CPU VAD; its remediated audit
  found zero known issues across 123 third-party distributions, with the
  unpublished application explicitly skipped. Installed normalized EN and ES replays
  passed model identity, persistence, saved audio and complete source/EOF checks;
  matched-source performance and final-source packaging remain pending.
- Training recipes now perform real CPU config/data checks before GPU startup,
  preserve source LoRA tensors during explicit expansion, correctly initialize
  new DoRA magnitudes, and materialize W17's JSON selection into an audio dataset.
  [Sixty original candidate triples](../../../training/candidates/README.md)
  remain unapproved. Missing WSL artifacts/holdouts are not treated as passing.

The [normalized integrity pilot](normalized-integrity-pilot/README.md) passed both
actual production runs on `a8511ee`. The subsequent 96-run Standard EN/ES screen
v1 was aborted after 13 completed process runs. Its Spanish E4B
`early_2s_160ms` repeat 0 exited successfully but retained an unmatched native
partial-STT start/finish: cancelling the asyncio wrapper had allowed the session
summary to freeze while the physical call was still running. The entire v1
cohort remains failed technical evidence, including the interrupted next run;
it supplies no optimization selection result.

Source `d8c3f05` joins native inference workers before model unload and final
session summaries, including at file EOF. It cancels queued inference that has
not started while preserving queued TTS already promised by published finals.
The separate v4 integrity pilot on `eddb0ad` completed three Spanish E4B runs
(opening control, `early_2s_160ms`, closing control) with passing integrity and
source-ledger checks. This verifies the shutdown repair in those runs; it does
not establish a speed improvement. The fresh 96-run Standard v2 cohort ran
from 15:33 to 17:19 UTC on frozen `eddb0ad`; all runs passed technical integrity.
The [selection result](standard-screen-result.md) rejects all 24 model/language
arms despite median-only gains in half of the candidate repeats. It remains
separate from v1 and the v4 pilot. Local receipts for those earlier cohorts remain under
`.cache/mac-en-es-closeout/standard-screen-normalized-v1` and
`.cache/mac-en-es-closeout/measurement-pilot-v4`, with the v1 abort receipt beside
those directories.

The [device follow-up](../tts_routing_20260910/README.md) retains successful
native TTS routing and a retest with zero parent-handoff drops but remaining
worker callback FIFO loss; its mixed-input evidence contains no incidental text.
The [accounting repair](capture-loss-accounting.md) preserves that failed
measurement and distinguishes it from unknown PortAudio driver loss.
The [input identity repair](../tts_routing_20260910/CAPTURE_ANALYSIS.md#input-device-identity-follow-up)
in `d8c3f05` binds exact microphone name and host API through preflight,
start/restart and microphone tests, resolving the index inside the native child.
Missing or ambiguous selections fail rather than selecting another input.
A [two-second native identity probe](../tts_routing_20260910/raw/identity/microphone-identity-probe-20260910.json)
opened the built-in microphone at current index 1 despite a supplied stale index
2, and rejected a missing name. It discarded its samples and ran no STT;
post-fix live-pipeline capture reliability remains unvalidated.

The [hymn source repairs](hymn-source-repairs.md) preserve existing text
delimiters and accepted speech onset after music hold; Python 3.11/3.12 CI and
lint passed on `ffa34c5`. The [350-second natural control](final-c13f51f/hymn-capture.md)
and [102-call text packet](final-c13f51f/hymn-boundary.md) have now executed.
The control never entered music hold and the supplied delimiters remain hypotheses;
#193/#194 still require natural transition labels and bilingual meaning review.

The user now prohibits microphone capture and output playback for the rest of
this session. Further physical checks, including the prepared integrated TTS
rehearsal, are deferred. Remaining inference uses file input with TTS disabled
or isolated text calls; earlier device receipts retain their original scope.

## Development STT result

| Language | Engine | Corpus WER, all three repeats | Per-repeat whole-recording STT p50 |
|---|---|---:|---:|
| EN | Parakeet v3 MLX | 5.005% | 220–224 ms |
| EN | Whisper Turbo MLX | 4.438% | 699–739 ms |
| ES | Parakeet v3 MLX | 3.931% | 224–270 ms |
| ES | Whisper Turbo MLX | 3.016% | 719–876 ms |

These are isolated whole-recording calls, excluding capture, VAD, queues,
translation and rendering. They are not caption-delivery latency. Spanish's
0.914-percentage-point WER increase is inside the predeclared maximum of one
absolute point or 5% relative, whichever is larger. Both hit the only Spanish
glossary opportunity; that denominator cannot certify church terminology.
The [Spanish pipeline screen](spanish-parakeet-result.md) subsequently completed
18 runs. Median-only gains passed in five of six comparisons, but preview and
queue guards rejected both arms. Whisper remains the Spanish default; these
results authorize no confirmation or combination.

## Frozen replay protocol

The [Standard screen](protocol/standard-screen.json) independently tests 2/4 s
early-clause buffers with 160/240 ms pauses and 100/250 ms partial-STT deadline
margins. The [CPU Lite screen](protocol/lite-screen.json) tests 0.6/0.9/1.2 s
partial cadence. Three repeats alternate E4B/E2B order and have opening/closing
controls. Public development replays apply a declared per-record RMS 0.08/peak 0.95
linear level rule, then concatenate five recordings with one second of declared
silence. The [unmodified-level pilot](input-level-pilot/README.md) produced no
finals because its English input fell below the unchanged energy filter; it
is retained separately. Original recordings and quality results are unchanged; [provenance](protocol/public-replay-provenance.json)
retains every reference and source boundary. No confirmation audio enters tuning.

The selection target is delivery of the opening control's frozen VAD-positive
source mask for each original final. Candidate segmentation cannot shrink that
mask. Every required source interval must be delivered; omitted VAD-negative
trailing silence is allowed. Original full buffered-span completion is retained
as a separate conservative diagnostic. This is machine-VAD coverage, not semantic
or acoustic ground truth. Full capture/disposition/EOF accounting is required.

Median improvement must reach 15% or 150 ms against **both** controls in at least
two of three repeats. The tail guard uses explicit nearest-rank p95, permitting
at most the larger of 5% or 100 ms regression. At least 100 eligible observations
are required for a p95 claim. Preview source coverage may lose at most two
percentage points. First-preview delay and update-gap p95 must also avoid
regression beyond the larger of 5% or 100 ms; missing evidence cannot silently
become a zero, and no-preview controls are explicitly N/A. These additional
responsiveness guards were declared before this follow-up replay cohort.
Actual selected STT and loaded model identities must match each declared arm;
only the explicit Spanish Parakeet experiment may change that STT selection.
Missing/duplicate runs, inference failures, source loss,
unbounded queues, memory regressions or quality-gate failures cannot qualify.
The declared memory guard permits at most the larger of 10% or 256 MiB above
each control's separate process-lifetime RSS/Metal peak (never summed).
Queue guard: complete bounded bookkeeping and zero terminal outstanding work;
at most one additional pending final request above either control; maximum final
queue wait and per-stage queue p95 may increase by at most 5% or 100 ms, whichever
is larger. Disjoint first/last windows use up to 16 requests each, requiring
  at least two in each window. Positive growth may exceed the control by at most
  100 ms or 5% of its maximum wait. This bounded screen does not establish a
  long-service trend.
Browser acknowledgments and physical visibility remain separate from this
server-side engineering screen. Research traces explicitly retain at most 131,072
events for these cohorts; overflow still invalidates physical-stage completeness.
The ordinary trace capacity remains 8,192 and tracing remains off by default.
Nothing here changes a production model default.

## Remaining execution

The bounded Standard, Spanish Parakeet, CPU Lite cadence/deadline and smaller
CPU STT comparisons have completed without a qualifying candidate. Untouched
confirmation is therefore unused; all negative results are retained. Silent hymn
diagnostics and c13 source checks are also complete.

The [installed-delivery report](final-c13f51f/installed-delivery.md) separates completed
artifact, operator and EN/ES smoke checks from full-service acceptance. The c13
Standard pipeline completed its required writes. A separate 128 MiB monitor
reconstruction recovered the terminal read, but the original terminal validator
then failed its nonempty-preview check on one blank translated preview. Preserve
both the original monitor failure and this application failure. The preview
repair and new-source Standard/Lite full rehearsals are tracked in
[implementation status](../../mac_implementation_status.md), alongside actual
terminal acceptance, archive verification and reviewed merge. External human/device/CUDA
gates remain separate; the sub-second caption-delivery goal is not established.

[Completed evidence index](completed-evidence-index.json) binds the retained raw
quality/profiling files. Initial STT/profiling used frozen runtime `c1a9041`; joint evaluation used
`1254616`. Subsequent
measurement hardening and experiments will identify their own exact sources.
