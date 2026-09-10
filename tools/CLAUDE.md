# tools/ — Evaluation, Monitoring, Review, Adapter Deployment

> Paired with [`AGENTS.md`](./AGENTS.md). Inventory and contracts reflect the local
> branch (base `5154fb9`, v2026.14 candidate); main is v2026.13. Numbers belong in the
> linked evidence documents, not here.

Quality layers 4–6 from the root guide live here (YouTube caption comparison,
translation QE, active learning), together with the reproducible Mac evaluation
harness, the replay benchmark, setup helpers and the documentation backlog tooling.
Importing any module must not load a model; scripts that need one spawn
`dry_run_ab.py` or a worker subprocess.

## Inventory

| Group | Scripts | Notes |
|-------|---------|-------|
| Reproducible Mac evaluation | `mac_evaluation.py` (`prepare`, `validate`, `annotate`, `realign-references`, `rescore-quality`, `stt`, `quality`, `replay`, `experiments`, `report`), `pipeline_timing.py` | Frozen manifests and results: [`docs/evaluation/README.md`](../docs/evaluation/README.md). One model process at a time; predicted transcripts never become references |
| Replay benchmark | `replay_bench.py` (`--prepare`, `--configs NAME=ARGV`, `--configs-file`, `--manifest`, `--baseline`, `--tag`) | Sequential `dry_run_ab.py --audio-file` runs; argv must match current flags (`--stt-backend parakeet-mlx` is valid on Mac) |
| Isolated benchmarks | `benchmark_mlx_accel.py`, `benchmark_parakeet_en.py`, `benchmark_stt_engines.py`, `benchmark_translate_engines.py`, `benchmark_latency.py`, `stt_benchmark.py`, `stt_roundtrip_compare.py`, `roundtrip_test.py`, `score_comet22.py`, `mts_acceptance_probe.py`, `test_adaptive_model.py`, `*_bench_manifest.json` | Engine-level numbers; live-pipeline claims require `replay_bench.py` or `mac_evaluation.py` |
| Live session monitoring | `live_caption_monitor.py`, `kpi_report.py`, `validate_session.py`, `translation_qe.py`, `mine_hallucination_phrases.py` | Post-session KPIs ([`docs/metrics.md`](../docs/metrics.md) — targets there predate schema 2) |
| Review, corrections, active learning | `review_data.py`, `export_review.py`, `merge_corrections.py` (`translation`, `whisper`), `session_lifecycle.py`, `lock_data.py`, `build_eval_sets.py`, `prepare_finetune_data.py`, `glossary.py` | Shared normalizer with the operator Review UI; see contracts below |
| Adapter lifecycle | `manage_adapters.py` (`register`, `activate`, `rollback`, `list`, `export`), `deploy_adapters.py`, `health_check.py`, `convert_models_to_both.py` | Design: [`docs/deploy.md`](../docs/deploy.md) |
| Setup / runtime helpers | `marian_ct2_setup.py`, `vad_runtime.py`, `installed_smoke.py`, `release_artifacts.py`, `audio_bridge.py`, `audio_bridge_client.py` | Used by `stark-translate setup/doctor`, packaging checks and the Docker audio bridge |
| Corpus builders | `build_v1_corpus.py`, `build_preference_triples.py`, `rebuild_verse_pairs.py`, `fix_platense_alignment.py`, `batch_translate.py`, `download_roundtrip_texts.py`, `sort_sermons.py` | Gemma 4 tuning data ([`docs/gemma4_tuning/`](../docs/gemma4_tuning/overview.md)); Platense realignment postmortem in [`docs/platense_alignment_bug.md`](../docs/platense_alignment_bug.md) |
| Documentation | `render_backlog.py` (`validate`, `render [--check]`, `check-links`) | Canonical backlog [`docs/backlog.json`](../docs/backlog.json); tests in `tests/test_documentation.py` |

## Measurement rules

- Server `speech_end_to_final_ms` (schema 2) stops at payload readiness; `speech_end_to_ack_upper_bound_ms` adds the browser and return-network hop and exists only for visible tabs. Legacy `e2e_latency_ms` / `true_e2e_ms` are processing measurements. Definitions: [`docs/evaluation/README.md`](../docs/evaluation/README.md), [`docs/archive/v2026.13/MAC_LATENCY.md`](../docs/archive/v2026.13/MAC_LATENCY.md).
- `pipeline_timing.py` speech end = end of the last VAD-positive frame; no timestamp is comparable across hosts.
- Replay reports separate language/provenance, silence vs smart vs hard cuts, EOF handling and timing schema; never pool cohorts with different manifest hashes.
- Frozen screens (`mac_v2026_14_screening.json`, 45 s) bound experiments; they do not replace the full historical baseline or natural-speech quality gates.

## Review and correction contracts (`review_data.py`, `session_lifecycle.py`)

- Corrections are revisioned sidecars; predictions and audio are never rewritten.
- Only sessions with recorded completion (`session_lifecycle_<id>.json`, `status: completed`) and explicit per-item approvals export training data. Transcript and translation approvals are independent; unknown language requires an explicit choice.
- Evaluation and training splits cannot cross (`tests/test_operator_review.py`, `tests/test_correction_import_safety.py`).
- Portable bundles round-trip and merge idempotently; older revisions cannot overwrite newer corrections.
- **Evidence status:** these paths are tested with fixtures. No human-approved correction from a real session exists yet (#137), so no "active learning loop closed" claim is allowed.

## Health check (adapter gate)

`python tools/health_check.py --backend mlx [--adapter DIR] [--n-canaries 8]` runs the
first 8 of the 18 canaries in `training/theological_canaries.py` (all 18 are used by
the evaluation harness), checks expected terms, latency (`--max-latency`, default 5 s)
and a word-ratio hallucination band (0.5–2.5). Run before `manage_adapters.py activate`.

## YouTube caption comparison (quality layer 4)

`live_caption_monitor.py` aligns local STT against the livestream's captions.
Use `find_global_offset_by_text()` for offsets larger than the window and check
`_wer is None` before aggregating. Cross-system WER is disagreement, not ground truth:
track trends and flag windows above 20 % rather than reporting it as accuracy.

## Translation QE (quality layer 5)

`translation_qe.py`: Tier 1 heuristics (length ratio, untranslated overlap; no model),
Tier 2 back-translation via Marian ES→EN plus BERTScore, Tier 3 LaBSE similarity.
Reference-based scoring (`score_comet22.py`, chrF++ in the evaluation report) applies
only to frozen manifests with approved references; chrF++ against a specific wording is
not a universal quality percentage.

## Adapter deployment

`manage_adapters.py register/activate/rollback` maintains the per-model `active` /
`previous` slots and manifest under `adapters/`; `deploy_adapters.py` pushes to
endpoints (SSH keys per machine still needed). `stark-translate setup` reuses complete `adapters/marian_ct2/*/active`
directories and never modifies them (`marian_ct2_setup.py`).

## Recording evidence

Session artifacts under `metrics/` (`session_*.log`, `ab_metrics_*.csv`,
`diagnostics_*.jsonl`, `display_metrics_*.jsonl`, `session_lifecycle_*.json`,
`session_metadata_*.json` with `audio_source` mic/file) are the only acceptable basis
for a status claim. Tonight's built-in-mic stall (`20260909_233204_799019_en`) versus
the passing file replays (`..._233546_027169_en`, `..._233823_034893_es`) is the
canonical example: same build, different `audio_source`, different conclusion.

## Backlog pointers

`caption-delivery-goal`, `visible-browser-timing-run`, `issue-137-active-learning`,
`issue-135-mac-ab`, `natural-spanish-refs` in [`docs/backlog.json`](../docs/backlog.json).
