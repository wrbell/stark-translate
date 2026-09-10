# Reproducible Mac evaluation

`tools/mac_evaluation.py` freezes inputs, runs one model process at a time, and
reports distributions from individual observations. E4B remains the default.

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
