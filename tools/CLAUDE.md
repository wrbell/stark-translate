# tools/ — Evaluation, Screens, Monitoring, Review, Adapter Deployment

> Paired with [`AGENTS.md`](./AGENTS.md). Inventory and contracts describe v2026.14 source on
> `main`. Numbers belong in the evidence documents ([index](../docs/evaluation/README.md)) and are
> quoted only in [`README.md`](../README.md) § Measured performance.

Quality layers 4–6 from the root guide live here (YouTube caption comparison, translation QE,
active learning), together with the reproducible Mac evaluation harness, the replay screens, the
capture/health runtime helpers, setup helpers and the documentation backlog tooling. Importing any
module must not load a model; scripts that need one spawn `dry_run_ab.py` or a worker subprocess.

## Inventory

| Group | Scripts | Notes |
|-------|---------|-------|
| Reproducible Mac evaluation | `mac_evaluation.py` (`prepare`, `validate`, `annotate`, `realign-references`, `rescore-quality`, `stt`, `quality`, `replay`, `experiments`, `report`), `pipeline_timing.py` | Frozen manifests and results: [`docs/evaluation/README.md`](../docs/evaluation/README.md). One model process at a time; predicted transcripts never become references |
| Replay screens | `replay_bench.py` (`--prepare`, `--configs NAME=ARGV`, `--configs-file`, `--manifest`, `--baseline`, `--tag`), `tail_screen_report.py` (`--runs`, `--output`, `--markdown`), `replay_integrity.py`, `replay_client_barrier.py`, `benchmark_identity.py`, `fixed_span_delivery.py` | Sequential `dry_run_ab.py --audio-file` runs; argv must match current flags. `tail_screen_report.py` computes the declared gates offline (median, tails incl. cuts, previews, sequence-aligned identity, relative RSS/Metal, lifecycle; optional `first_visible` from display metrics). Integrity guard and identity check fail closed |
| Stage attribution | `silence_final_stages.py`, `stt_overlap_attribution.py`, `latency_trace.py`, `final_queue_pressure.py` | Where silence-final time goes (schema 2 stages, STT/translation overlap on the session clock, bounded queue telemetry); no inference |
| Latency research controls (opt-in) | `latency_experiments.py` (`STARK_EXPERIMENT_*`), `latency_scheduler.py`, `incremental_stt.py`, `preview_candidates.py`, `music_recovery.py`, `source_coverage.py`, `overnight_bench.py`, `overnight_analysis.py` | Validated before startup, never promoted by backend selection; `overnight_bench.py` runs serial paired experiments with a visible audience browser and refuses to fabricate missing ACKs; every screened arm is closed ([registry](../docs/latency_next_experiments.md)) |
| Endurance | `endurance_monitor.py` (`--pid`, `--session`, `--duration-seconds`, `--stop-file`) | Read-only psutil sampling; the pipeline pid comes from `/api/session/status` |
| EN↔ES follow-up evaluation | `mac_followup_quality.py`, `mac_followup_latency.py` (strips `STARK_*`), `mac_followup_selection.py`, `public_replay.py`, `public_speech.py`, `mac_bilingual_review.py`, `parakeet_profile.py`, `parakeet_joint_eval.py`, `parakeet_compile_eval.py`, `synthetic_two_voice_clip.py` | Frozen public development/confirmation split; blinded review packets (HMAC-seeded labels; private key outside `docs/`); Parakeet profiling and the output-exact joint-decode qualification; the #133 two-voice fixture. No default promotion or human approval |
| Isolated benchmarks | `benchmark_mlx_accel.py`, `benchmark_parakeet_en.py`, `benchmark_stt_engines.py`, `benchmark_translate_engines.py`, `benchmark_latency.py`, `stt_benchmark.py`, `stt_roundtrip_compare.py`, `roundtrip_test.py`, `score_comet22.py`, `mts_acceptance_probe.py`, `test_adaptive_model.py`, `*_bench_manifest.json` | Engine-level numbers; live-pipeline claims require `replay_bench.py` or `mac_evaluation.py` |
| Live session monitoring | `live_caption_monitor.py`, `kpi_report.py`, `validate_session.py`, `translation_qe.py`, `mine_hallucination_phrases.py` | Post-session KPIs ([`docs/metrics.md`](../docs/metrics.md) — targets there predate schema 2) |
| Review, corrections, active learning | `review_data.py`, `export_review.py`, `merge_corrections.py` (`translation`, `whisper`), `session_lifecycle.py`, `lock_data.py`, `build_eval_sets.py`, `prepare_finetune_data.py`, `glossary.py`, `training_preflight.py` | Shared normalizer with the operator Review UI; CPU-only preflight for the W17 and Gemma recipes; see contracts below |
| Adapter lifecycle | `manage_adapters.py` (`register`, `activate`, `rollback`, `list`, `export`), `deploy_adapters.py`, `health_check.py`, `convert_models_to_both.py` | Design: [`docs/deploy.md`](../docs/deploy.md) |
| Capture, health, delivery | `isolated_audio.py` (`IsolatedInputStream`, `probe_audio`), `capture_worker.py`, `capture_handoff.py`, `capture_protocol.py`, `input_devices.py`, `pipeline_health.py`, `persistence.py`, `operational_logging.py`, `caption_delivery.py`, `display_server.py` | Disposable PortAudio child with 5 s startup / 3 s idle no-input timeouts; exact device identity resolved in the child; low-rate health/control channel read by the operator; acknowledged background persistence; bounded per-client caption writers; the audience HTTP server serves only the public display bundle |
| Setup / runtime helpers | `marian_ct2_setup.py`, `vad_runtime.py`, `installed_smoke.py`, `release_artifacts.py`, `check_dependency_audit.py` (`--runtime mac`), `audio_bridge.py`, `audio_bridge_client.py`, `llama_runtime.py` | Used by `stark-translate setup/doctor`, packaging checks, the runtime audit, the Docker audio bridge and the Lite session-owned `llama-server` (pinned `b10883` archives, hash-verified; [`docs/lite_profiles.md`](../docs/lite_profiles.md)) |
| Offline Hindi baseline | `offline_hindi.py` (`prepare`, `transcribe`, `translate --size e4b\|e2b`, `report`), `offline_hindi_manifest.json` | [Completed offline R&D](../docs/evaluation/overnight_hindi/README.md); no live integration; #138 decision pending |
| Corpus builders | `build_v1_corpus.py`, `build_preference_triples.py`, `rebuild_verse_pairs.py`, `fix_platense_alignment.py`, `batch_translate.py`, `download_roundtrip_texts.py`, `sort_sermons.py` | Gemma 4 tuning data ([`docs/gemma4_tuning/`](../docs/gemma4_tuning/overview.md)); Platense realignment postmortem in [`docs/platense_alignment_bug.md`](../docs/platense_alignment_bug.md) |
| Documentation | `render_backlog.py` (`validate`, `render [--check]`, `check-links`) | Canonical backlog [`docs/backlog.json`](../docs/backlog.json); tests in `tests/test_documentation.py` |

## Measurement rules

- Server `speech_end_to_final_ms` (schema 2) stops at payload readiness; `speech_end_to_ack_upper_bound_ms` is measured from the **estimated speech end to the visible browser's acknowledgement** (render and return network included) and exists only for visible tabs; `send_to_ack_ms` is the narrower server-send → ACK span. First token = `timing_stages_ms.translation_started + ttft_ms_a − timing_stages_ms.speech_end`; its browser counterpart is the `first_stream` ACK stage (`first_visible_ms` in `overnight_bench.py`, the `first_visible` block in `tail_screen_report.py` and `mac_evaluation.py`). Legacy `e2e_latency_ms` / `true_e2e_ms` are processing measurements. Definitions: [`docs/evaluation/README.md`](../docs/evaluation/README.md).
- `pipeline_timing.py` speech end = end of the last VAD-positive frame; no timestamp is comparable across hosts. `ChunkTiming` fields are declared dataclass fields (`relative_stages()` subtracts floats from every undeclared attribute).
- Replay reports separate language/provenance, silence vs smart vs hard cuts, EOF handling and timing schema; never pool cohorts with different manifest hashes. Replay tails that end on appended silence are labelled separately (`silence_replay_tail`).
- Frozen screens (`mac_v2026_14_screening.json`, 45 s) bound experiments; they do not replace the full historical baseline or natural-speech quality gates.
- A screen declares its protocol before run 1, keeps one inference process on the GPU, and never re-runs a rejected arm as a confirmation or in a combination. Screens without enough eligible finals make no p95 claim (`p95_claim_eligible: false`).

## Harness rules (what has cost time before)

- `replay_bench.py` inherits the environment: settings env names such as `STARK_VAD_MAX_UTTERANCE` and `STARK_TRANSLATE_MARIAN_INTRA_THREADS` reach the runs; `mac_followup_latency.py` strips all `STARK_*`. Run it as `python -m tools.replay_bench`; it writes `metrics/` under the current checkout, so two-checkout paired screens use per-checkout `metrics/` and distinct tag prefixes.
- Engine `engines.*` INFO logs do not reach replay session logs (only the `dry_run_ab` logger has the console handler); prove load-time state by receipts.
- `tools/parakeet_profile.py` always constructs the engine with `joint_scalar_eval=False`; its instrumented arm parses the stock decode source.
- `tools/mac_bilingual_review.py` hard-codes `ENGINES=("e4b","e2b")`; other pairings need a small generator that imports `blind_labels` and mirrors `make_packet` (series 4 P4 did this).
- Waiters: wait on a pid with `kill -0`, not `pgrep -f "<pattern>"` (it matches its own command line); `pkill -f` with a substring kills your own waiters; relaunching a script that `>`-redirects a log another instance writes truncates it.

## Review and correction contracts (`review_data.py`, `session_lifecycle.py`)

- Corrections are revisioned sidecars; predictions and audio are never rewritten.
- Only sessions with recorded completion (`session_lifecycle_<id>.json`, `status: completed`) and explicit per-item approvals export training data. Transcript and translation approvals are independent; unknown language requires an explicit choice.
- Evaluation and training splits cannot cross (`tests/test_operator_review.py`, `tests/test_correction_import_safety.py`).
- Portable bundles round-trip and merge idempotently; older revisions cannot overwrite newer corrections.
- **Evidence status:** these paths are tested with fixtures. No human-approved correction from a real session exists yet (#137), so no "active learning loop closed" claim is allowed.

## Health check (adapter gate)

`python tools/health_check.py --backend mlx [--adapter DIR] [--n-canaries 8]` runs the first 8 of
the 18 canaries in `training/theological_canaries.py` (all 18 are used by the evaluation harness),
checks expected terms, latency (`--max-latency`, default 5 s) and a word-ratio hallucination band
(0.5–2.5). Run before `manage_adapters.py activate`.

## YouTube caption comparison (quality layer 4)

`live_caption_monitor.py` aligns local STT against the livestream's captions. Use
`find_global_offset_by_text()` for offsets larger than the window and check `_wer is None` before
aggregating. Cross-system WER is disagreement, not ground truth: track trends and flag windows
above 20 % rather than reporting it as accuracy.

## Translation QE (quality layer 5)

`translation_qe.py`: Tier 1 heuristics (length ratio, untranslated overlap; no model), Tier 2
back-translation via Marian ES→EN plus BERTScore, Tier 3 LaBSE similarity. Reference-based scoring
(`score_comet22.py`, chrF++ in the evaluation report) applies only to frozen manifests with
approved references; chrF++ against a specific wording is not a universal quality percentage.

## Adapter deployment

`manage_adapters.py register/activate/rollback` maintains the per-model `active` / `previous`
slots and `adapters/manifest.json` (written by `register`); `deploy_adapters.py` pushes to
endpoints (SSH keys per machine still needed). `stark-translate setup` reuses complete
`adapters/marian_ct2/*/active` directories and never modifies them (`marian_ct2_setup.py`).

## Recording evidence

Session artifacts under `metrics/` (`session_*.log`, `ab_metrics_*.csv`, `diagnostics_*.jsonl`,
`display_metrics_*.jsonl`, `session_lifecycle_*.json`, `session_metadata_*.json` with
`audio_source` mic/file) are the only acceptable basis for a status claim: cite the session id and
the file, never "tests passed". A microphone session and a file replay on the same build can reach
opposite conclusions, so `audio_source` is part of every claim. Automation never opens the
microphone or plays audio; those checks are attended. Evidence directories under
`docs/evaluation/` are immutable once written; corrections are dated appends. Microphone and
device evidence: [quiet-room receipts](../docs/evaluation/attended_mic_20260910/README.md),
[synthetic checks and identity probe](../docs/evaluation/tts_routing_20260910/README.md),
[capture-loss accounting](../docs/evaluation/mac_followup_20260910/capture-loss-accounting.md).
Lite CPU smoke evidence (synthetic inputs, hashes, commands):
[`docs/evaluation/lite_cpu_smoke_20260910.json`](../docs/evaluation/lite_cpu_smoke_20260910.json).

## Backlog pointers

`caption-delivery-goal`, `visible-browser-timing-run`, `mac-live-mic-stall`,
`issue-137-active-learning`, `issue-135-mac-ab`, `natural-spanish-refs`,
`conservative-marian-routing`, `eou-endpointing-feasibility` in
[`docs/backlog.json`](../docs/backlog.json).
