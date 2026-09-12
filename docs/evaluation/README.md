# Reproducible Mac evaluation

`tools/mac_evaluation.py` freezes inputs, runs one model process at a time, and
reports distributions from individual observations. E4B remains the default.

The September 10 continuation is tracked by the
[overnight EN↔ES experiment plan](overnight_experiment_plan.md),
[integrated operator rehearsal](overnight_operator_rehearsal.md) and
[current implementation status](../mac_implementation_status.md).
Its [complete 96-run English screen](overnight_screen_20260910/README.md) selected
0/28 arms and did not meet the sub-second final goal. It uses a separate frozen
source cohort; do not pool it with earlier runs. Browser-DOM ACK telemetry was
collected while the native Mac was locked, so physical screen visibility remains
unverified. Negative results, small endpoint counts and original hashes are retained. Lite installation and CPU evidence are in
[Lite profiles](../lite_profiles.md).

The [752 local validation](overnight_validation_20260910.json),
[mechanical artifact receipt](overnight_artifact_validation_20260910.json) and
[static check receipt](overnight_static_validation_752ab9a_20260910.json) bind to
frozen source `752ab9a`. They do not replace the `911f4ae` screen or the older
installed hour. The [terminal service report](overnight_endurance_20260910/README.md)
records completed repaired Standard and CPU Lite hours, their separate timing and
quality limits, selected waveform checks and the actual truncated long-summary UI.
Lite completed functionally but does not qualify as a fast production profile.
The validation JSON preserves its earlier dated start checkpoint; it is not the
terminal endurance record. Publication and device/human acceptance remain separate.

The [final integration report](overnight_final_validation_20260910.md) and its
[machine-readable record](overnight_final_validation_20260910.json) bind later
CPU/static/artifact validation and remote CI to source `84832fb`. They retain the
real intermediate interpreter-exit failure and its operator cleanup fix. Exact
wheel comparison found five changed operator modules and 138 unchanged code/resource
members; pipeline/engine bytes remain unchanged, while the completed hours still
belong to the actual 752 wheel. Full receipts and modest logs are retained in the
[final evidence directory](overnight_final_validation_20260910/README.md).

The later [bootstrap review corrections](bootstrap_review_20260910/README.md)
record explicit-environment creation and setup-before-service regression checks.
They change the source installer; the frozen runtime results retain their own identity.

## Frozen inputs

- `mac_v2026_14_manifest.json`: original frozen audio/transcript inputs. Retains
  both historical 150-second English sermon replays and the synthetic Piper
  Spanish clip. Includes 50 EN and 11 ES unreviewed audio candidates, 18 canaries
  and 50 text translation items (25 per direction).
- `mac_v2026_14_manifest_v2.json`: same model inputs and audio, with corrected
  translation references. The old parallel corpus joins sequential verse IDs
  across editions with different numbering. Exact source lookup plus unique
  book/chapter/verse now selects KJV/RVR1909 references; 49/50 items resolve.
  One ambiguous phrase has no score. These are structural checks, not human review.
- `mac_v2026_14_screening.json`: a 45-second prefix of the first historical clip,
  with its own hash and parent provenance. This bounds individual experiments;
  it does not replace the full historical baseline.
- `mac_v2026_14_experiments.json`: fixed cadence 0.6 s, three alternating E4B/E2B
  pairs for each opt-in experiment. Only one behavior changes per screen.
- `mac_v2026_14_routing_synthetic.json` and `mac_v2026_14_routing_experiments.json`:
  separate Piper EN/ES operational phrases for exercising routing decisions with
  unchanged confidence thresholds. These are functional probes, not natural
  speech references or substitutes for the sermon benchmark.

Predicted transcripts never count as references. Use `annotate` to create a new
manifest with human-reviewed text/provenance; it rejects changes to immutable
audio identity. Natural Spanish and two-speaker gates remain pending.

## Run

Install `.[mlx,eval]` in the selected environment. Existing `stt_env` was retained;
additional reporting packages used for this run live in `.cache/mac-eval-deps`.
All model runs are sequential to avoid GPU contention.

> **2026-09-12 note:** the launcher default is now the promoted `venv` (Torch 2.13.0) selected by
> `.stark-python`; `stt_env` is the untouched rollback environment. The commands below run under
> either; the recorded runs above used the environment named in their receipts.

```bash
python tools/mac_evaluation.py validate --manifest docs/evaluation/mac_v2026_14_manifest_v2.json
python tools/mac_evaluation.py stt --manifest docs/evaluation/mac_v2026_14_manifest_v2.json --output metrics/mac_roadmap/stt --runs 3
python tools/mac_evaluation.py quality --manifest docs/evaluation/mac_v2026_14_manifest_v2.json --output metrics/mac_roadmap/quality_new --runs 3 --policies none church
python tools/mac_evaluation.py replay --manifest docs/evaluation/mac_v2026_14_manifest_v2.json --output metrics/mac_roadmap/baseline_new --tag unique_run --runs 3
python tools/mac_evaluation.py experiments --manifest docs/evaluation/mac_v2026_14_screening.json --spec docs/evaluation/mac_v2026_14_experiments.json --output metrics/mac_roadmap/experiments --tag unique_screen --runs 3
python tools/mac_evaluation.py report --manifest docs/evaluation/mac_v2026_14_manifest_v2.json --input metrics/mac_roadmap/baseline_new --output docs/evaluation/baseline_new
```

The original quality runs used v1. `realign-references` created v2 and
`rescore-quality` rebound only reference metadata, preserving every prediction,
timing and original run hash. Reports skip quality runs with a different manifest
hash. Do not use the original v1 reference-based scores.

## Read the results

[Baseline and STT overview](mac_v2026_14_report/README.md) links the combined
Markdown/JSON report and frozen raw observations. Its input staging directory
contains only `baseline`, `quality_v2` and `stt` from `metrics/mac_roadmap`, so
the Hindi probe and 45-second experiments remain separate. To reproduce that
combined report, place those three directories under one input directory and
pass it to `report` with the v2 manifest. Timing CSV/JSONL paths are relative to
the repository's `metrics` directory; `raw_index.json` maps the archived copies
back to their original paths.

[Translation comparison](mac_v2026_14_quality/comparison.md) includes all 18
canaries, latency cost, chrF++, and actual changed outputs. Canary checks require
all specified terms. chrF++ measures agreement with a specific wording; it is
not a universal translation-quality percentage. The sample has duplicate source
wording and archaic verse language and is not representative of all sermons.

`blind_review.jsonl` provides anonymous A/B responses and blank meaning-error and
terminology ratings; keep `review_key.jsonl` away from reviewers. Automated
generation of that form does not mean bilingual review has happened.

Replay reports separate language/provenance, silence, smart cuts, hard cuts, EOF,
and timing schema. Server `speech_end_to_final_ms` stops at payload readiness.
Client receipt-to-render is local browser overhead; speech-end-to-ack is an upper
bound including return-network time. Hidden tabs and accelerated replay cannot
pass caption-delivery gates. Legacy `e2e_latency_ms` remains processing time.

Schema-2 `first_stream` stage: speech end → first streamed translation tokens
visible (upper bound, includes return network); reported, not a gate; one ACK per
client per chunk. The first batch must actually be sent and synchronously mutate
the visible display DOM. If delivery coalesces that batch away, no first-stream
observation is recorded for that client; later batches cannot substitute for it.

**Replay-ending interpretation:** the file stream appends virtual silence to let
pending captions finish. Older raw/harness tables group by the emitted
`endpoint_reason`; a `silence` row can therefore include an EOF-assisted ending.
Where `padding_samples` is positive and the timing source is replay, the corrected
supplemental analysis labels the analytical endpoint separately (for example,
`silence_replay_tail`). It preserves the original reason and every measurement.
Use those separate groups for latency decisions; a tail/EOF-final improvement
alone cannot select an optimization. First previews emitted earlier from real
recorded audio remain valid preview observations. Archived rows without sufficient
padding/source metadata cannot be retrospectively certified as natural silence.

[Hindi](mac_v2026_14_hindi/README.md) is an independent offline zero-shot probe
(`quality --target hi`), with no reference score or live-language integration.
The [operator rehearsal](mac_v2026_14_rehearsal.md) records actual browser,
session, summary and TTS behavior separately from benchmark acceptance.
Physical output-device, natural
Spanish, two-speaker and bilingual review gates are explicitly separate.

[The frozen latency screen](mac_v2026_14_screening/README.md) compares all eight
configurations on both models across 48 completed real-time runs. It includes
failed optimization hypotheses, execution counters and matched-caption analysis
where lower silence thresholds changed segmentation. It supports retaining all
experimental options as opt-in; no combined optimization was justified.

[The bilingual routing probes](mac_v2026_14_routing/README.md) record 24 completed
synthetic runs. The conservative policy used Marian for the two allowlisted
phrases and Gemma for the control phrase in both languages and on both models.
Observed confidence values and route counters are retained; these checks do not
calibrate either STT engine's confidence or establish natural-speech quality.

The [runtime snapshot](mac_v2026_14_runtime.json) supplements per-run package,
code and model identities. The [security scope](mac_v2026_14_security.md) records
what the checks covered and the remaining unpinned model-download findings.
The [validation record](mac_v2026_14_validation.json) records the final unit,
type, lint, HTML, security, replay and setup checks with local evidence hashes.
The [local installation evidence](mac_v2026_14_installation.md) records final
wheel/sdist/Mac ZIP identities, actual unpacked-ZIP launch/build/install checks,
and installed EN/ES inference. It distinguishes the GPU-exercised artifact from
the final shell-adjusted wheel using exact member hashes. These are post-build
records; publication and the human, device and visible-browser gates remain open.

## Index of dated evidence (added 2026-09-12)

Every directory under `docs/evaluation/`, newest first. Files inside are immutable once written;
corrections are dated appends.

| Directory | Date | What it records |
|---|---|---|
| [`series4_20260912/`](series4_20260912/STATUS.md) | 2026-09-12 | Series 4: runtime fixes on a paired identity screen (P1), partial-reuse attribution and arm screen (P2, rejected), Smart Turn v3 end-of-utterance feasibility (P3, no-go), Marian-band review packet (P4), first-visible ACK (P6) |
| [`series3_20260912/`](series3_20260912/STATUS.md) | 2026-09-12 | Series 3: stage attribution (L-A), tail screen with three rejected arms (L-B), first-token report (L-C), hymn second control, endurance on the promoted runtime, installed smoke and runtime audit |
| [`followup_20260911/`](followup_20260911/STATUS.md) | 2026-09-11 | Torch 2.13 runtime promotion with the `.stark-python` rollback pointer, tail screen (both arms rejected) |
| [`overnight_20260911/`](overnight_20260911/STATUS.md) | 2026-09-11 | Stage attribution, opt-in E2B draft (rejected), diarization interpreter, B615 pinning, Torch 2.13 candidate, v2026.14.0.0 publication |
| [`mac_followup_20260910/`](mac_followup_20260910/README.md) | 2026-09-10 | EN↔ES follow-up: normalized Standard / Spanish Parakeet / Lite cadence and deadline screens, CPU STT comparison, public FLEURS data, hymn diagnostics and source repairs, capture-loss accounting, live HF pinning, source validation and delivery packets (`final-c13f51f/`, `final-760e948/`) |
| [`attended_mic_20260910/`](attended_mic_20260910/README.md) | 2026-09-10 | Attended quiet-room EN/ES microphone sessions (readiness, pause/resume, restart) |
| [`tts_routing_20260910/`](tts_routing_20260910/README.md) | 2026-09-10 | Synthetic speaker-to-microphone caption checks, Spanish capture-loss failure, device identity probe, per-language TTS routing |
| [`overnight_endurance_20260910/`](overnight_endurance_20260910/README.md) | 2026-09-10 | Standard and CPU Lite service hours on the installed wheel (observational) |
| [`overnight_screen_20260910/`](overnight_screen_20260910/README.md) | 2026-09-10 | 96-run English screen, 0/28 arms selected |
| [`overnight_closeout_20260910/`](overnight_closeout_20260910/README.md) | 2026-09-10 | PR #192 merge, issue closures, bootstrap ZIP evidence |
| [`overnight_final_validation_20260910/`](overnight_final_validation_20260910/README.md) | 2026-09-10 | Final integration validation bound to `84832fb` |
| [`bootstrap_review_20260910/`](bootstrap_review_20260910/README.md) | 2026-09-10 | Bootstrap review corrections |
| [`security_feasibility_20260910/`](security_feasibility_20260910/README.md), [`overnight_security/`](overnight_security/README.md) | 2026-09-10 | Security feasibility and audit runs |
| [`mac_v2026_14_report/`](mac_v2026_14_report/README.md), [`mac_v2026_14_quality/`](mac_v2026_14_quality/comparison.md), [`mac_v2026_14_screening/`](mac_v2026_14_screening/README.md), [`mac_v2026_14_routing/`](mac_v2026_14_routing/README.md), [`mac_v2026_14_hindi/`](mac_v2026_14_hindi/README.md) | 2026-09-09/10 | v2026.14 frozen baseline and STT report, translation comparison, 48-run latency screen, routing probes, Hindi probe |
| [`overnight_hindi/`](overnight_hindi/README.md) | 2026-09-10 | Offline church-audio Hindi baseline (completed R&D, no live path) |

Contracts and plans used by the harnesses: [`audio_sources.md`](audio_sources.md),
[`overnight_analysis_contract.md`](overnight_analysis_contract.md),
[`stt_primary_benchmark_contract.md`](stt_primary_benchmark_contract.md),
[`overnight_experiment_plan.md`](overnight_experiment_plan.md),
[`overnight_operator_rehearsal.md`](overnight_operator_rehearsal.md),
[`mac_v2026_14_rehearsal.md`](mac_v2026_14_rehearsal.md),
[`mac_v2026_14_installation.md`](mac_v2026_14_installation.md),
[`mac_v2026_14_security.md`](mac_v2026_14_security.md),
[`overnight_final_validation_20260910.md`](overnight_final_validation_20260910.md).
Machine-readable receipts at this level (`*_20260910.json`, `mac_v2026_14_*.json`) are named by
the documents above; the Lite smokes are `lite_cpu_smoke_20260910.json`,
`lite_cpu_quality_preparation_20260910.json`, `lite_cpu_quality_smoke_20260910.json` and
`lite_installer_security_20260910.json`.

## Schema 2 field reference (added 2026-09-12)

Producer fields (`tools/pipeline_timing.py`, `ChunkTiming.metrics()`, written to the diagnostics
rows and `timing_stages_ms`):

| Field | Meaning |
|---|---|
| `speech_end_to_final_ms` | Estimated speech end (end of the last VAD-positive frame) → final payload ready on the server; `null` for non-real-time replay |
| `vad_wait_ms` | Speech end → the endpoint decision (silence countdown or cut) |
| `stt_queue_wait_ms`, `translation_queue_wait_ms` | Time the final waited for the STT worker / the translation lock |
| `finalization_overhead_ms`, `broadcast_ms` | Bookkeeping after translation; serialization and send |
| `endpoint_reason` | Why the utterance ended: silence, smart cut, hard cut, EOF, pause or stop (kept distinct; replay tails on appended silence are labelled separately by the analysis, e.g. `silence_replay_tail`) |
| `timing_source` | Live capture clock vs replay clock |
| `final_translation_route`, `final_stt_route` | Which engine produced the final (Gemma or Marian) and whether the final STT was a full call or reused a partial (experiment only) |

First token (derived, series 3): `first_token_ms = timing_stages_ms.translation_started +
ttft_ms_a − timing_stages_ms.speech_end`, the moment the first translated token could reach the
wire. It is an engineering measure on machine-timed replays; the browser counterpart is the
`first_stream` acknowledgement below.

Display fields (`RenderTracker.acknowledge()`, written to `metrics/display_metrics_<session>.jsonl`;
one record per acknowledged message per client):

| Field | Meaning |
|---|---|
| `stage` | `partial`, `complete`, or `first_stream` (the first streamed batch of a final) |
| `receive_to_render_ms` | Browser receipt → render opportunity (two animation frames after a DOM mutation) |
| `send_to_ack_ms` | Server send → acknowledgement received |
| `speech_end_to_ack_upper_bound_ms` | Estimated speech end → acknowledgement, including the return network hop; recorded only when the tab was visible |
| `speech_start_to_preview_ack_upper_bound_ms`, `captured_end_to_preview_ack_upper_bound_ms`, `speech_end_to_preview_ack_upper_bound_ms` | The same upper bound for previews, measured from speech start, from the captured end of the preview's audio, and from speech end |

The `first_stream` stage is reported as `first_visible_ms` by `tools/overnight_bench.py` and as
the `first_visible` block by `tools/tail_screen_report.py` and `tools/mac_evaluation.py`; it is
one acknowledgement per client per chunk, reported and not gated, and it does not exist for
headless replays. Legacy `e2e_latency_ms` (submission → processing done) and `true_e2e_ms` (first
speech observed → processing done) are processing measurements and are never compared with the
fields above.
