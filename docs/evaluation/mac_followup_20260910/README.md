# Mac EN↔ES follow-up implementation and experiments

Work is in progress on `codex/mac-en-es-closeout`, tracked by
[draft PR #196](https://github.com/wrbell/stark-translate/pull/196). Defaults remain unchanged;
no release or package publication is authorized for this follow-up. The
[43-item audit](backlog-audit.md) preserves each original acceptance condition.

## Completed evidence

- [Public input audit](public_data/README.md): 100 EN + 100 ES original FLEURS
  recordings, 50 development / 50 confirmation per language, pinned source,
  hashes, independent archive verification and portable reference metadata.
  Public read speech is engineering evidence, not church or local human review.
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
actual production runs on `a8511ee`. The independent 96-run Standard EN/ES
screen is now executing on that frozen runtime. The [device follow-up](../tts_routing_20260910/README.md)
retains successful native TTS routing and a retest with zero parent-handoff drops
but remaining upstream microphone loss; its mixed-input evidence contains no incidental text.

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
The [conditional Spanish pipeline screen](protocol/spanish-parakeet-screen.json)
is therefore eligible to run. Whisper remains the Spanish default, and untouched
confirmation plus bilingual review are still required.

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

Run the bounded live-pipeline comparisons, smaller CPU STT comparison, and untouched
confirmation for qualifying candidates. Retain negative results. Integrate
appropriate checks and documentation, rebuild/verify artifacts, run final
Standard and CPU Lite full-service rehearsals, review the final changes, and
merge only after validation. External/human/device/CUDA gates keep their actual
pending status; the sub-second caption-delivery goal is not yet established.

[Completed evidence index](completed-evidence-index.json) binds the retained raw
quality/profiling files. Initial STT/profiling used frozen runtime `c1a9041`; joint evaluation used
`1254616`. Subsequent
measurement hardening and experiments will identify their own exact sources.
