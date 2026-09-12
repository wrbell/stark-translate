# Documentation map

Every document under `docs/`, grouped by what you are trying to do. Root guides:
[`README.md`](../README.md) (product overview, measured performance, quick start),
[`CLAUDE.md`](../CLAUDE.md) (developer guide), [`AGENTS.md`](../AGENTS.md) (short agent guide),
[`CLAUDE-macbook.md`](../CLAUDE-macbook.md), [`CLAUDE-windows.md`](../CLAUDE-windows.md).
Directory guides: [`engines/`](../engines/CLAUDE.md), [`tools/`](../tools/CLAUDE.md),
[`displays/`](../displays/CLAUDE.md), [`features/`](../features/CLAUDE.md),
[`training/`](../training/CLAUDE.md), each paired with an `AGENTS.md`.

## Start here

| You want to… | Read |
|---|---|
| Run captions on a Sunday | [`operator_runbook.md`](operator_runbook.md) |
| Set up or roll back the Mac | [`packaging/macos.md`](packaging/macos.md) |
| Run on a church PC or an RTX 2070 | [`lite_profiles.md`](lite_profiles.md), [`packaging/windows.md`](packaging/windows.md) |
| Know what is implemented and proven | [`mac_implementation_status.md`](mac_implementation_status.md) |
| Understand the contracts | [`current_architecture.md`](current_architecture.md) |
| See what is left | [`backlog.md`](backlog.md) (rendered from [`backlog.json`](backlog.json)) |

## Contracts and status

| Doc | Role |
|---|---|
| [`current_architecture.md`](current_architecture.md) | Two-pass pipeline, schema 2 measurement, operator control plane, model resolution, deployment targets |
| [`mac_implementation_status.md`](mac_implementation_status.md) | Implemented features, acceptance evidence, open gates (canonical status) |
| [`backlog.json`](backlog.json) / [`backlog.md`](backlog.md) | Remaining work with statuses, dependencies and acceptance; validated by `tools/render_backlog.py` |
| [`roadmap.md`](roadmap.md) | Current state, active work and archived history by phase |
| [`overnight_status.md`](overnight_status.md) | September 10 delivery status with pointers to the later boards |
| [`issue_closure_audit.md`](issue_closure_audit.md) | GitHub issue acceptance and evidence matrix |
| [`latency_next_experiments.md`](latency_next_experiments.md) | Closed-arm registry: every screened latency idea, its gate and outcome |

## Runbooks and operations

| Doc | Role |
|---|---|
| [`operator_runbook.md`](operator_runbook.md) | Day-of-event workflow for non-technical operators |
| [`operator_reliability.md`](operator_reliability.md) | Operator reliability model, support bundles, storage cleanup |
| [`mac_pipeline_refresh.md`](mac_pipeline_refresh.md) | Mac inference and validation runbook |
| [`wsl_pipeline_refresh.md`](wsl_pipeline_refresh.md) | Ordered WSL training refresh (Phase 4, E4B domain SFT, W17, transfer) |
| [`deploy.md`](deploy.md) | Adapter registry, health gate and deployment design |
| [`security_offline_model_paths.md`](security_offline_model_paths.md) | Operator-only offline model paths (2026-09-11) |
| [`metrics.md`](metrics.md) | KPI definitions and tuning guide (targets predate schema 2) |

## Packaging

| Doc | Role |
|---|---|
| [`packaging/macos.md`](packaging/macos.md) | macOS installation, models, interpreter pointer and launchd, runtime audit, package checks |
| [`packaging/windows.md`](packaging/windows.md) | Windows MSI delivery status and boundaries (assets in `packaging/windows/`) |
| [`packaging/linux-docker.md`](packaging/linux-docker.md) | Linux and Docker (GHCR image) |
| [`packaging/models.md`](packaging/models.md) | Model manifest and managed Marian CT2 artifacts |
| [`packaging/pypi.md`](packaging/pypi.md) | PyPI publication (deferred by decision; gated job) |
| [`containerization.md`](containerization.md) | Containerization and distribution design |
| [`lite_profiles.md`](lite_profiles.md) | Lite profile contract, admission floors, pinned artifacts, CPU evidence |

## Evaluation and evidence

Metric definitions, frozen inputs and the index of every dated evidence directory live in
[`evaluation/README.md`](evaluation/README.md). Dated boards, newest first:

| Board | What it records |
|---|---|
| [`evaluation/series4_20260912/STATUS.md`](evaluation/series4_20260912/STATUS.md) | Series 4 (2026-09-12): runtime fixes on an identity screen, first-visible ACK, `partial_reuse_ms` arm rejected, Smart Turn no-go, Marian review packet |
| [`evaluation/series3_20260912/STATUS.md`](evaluation/series3_20260912/STATUS.md) | Series 3 (2026-09-12): stage attribution, tail screen (three arms rejected), first-token report, endurance on the promoted runtime |
| [`evaluation/followup_20260911/STATUS.md`](evaluation/followup_20260911/STATUS.md) | 2026-09-11 daytime: Torch 2.13 promotion with rollback pointer, PyPI deferral, tail screen |
| [`evaluation/overnight_20260911/STATUS.md`](evaluation/overnight_20260911/STATUS.md) | Overnight 2026-09-11: stage attribution, E2B draft, diarization interpreter, B615 pinning, v2026.14.0.0 publication |
| [`evaluation/mac_followup_20260910/README.md`](evaluation/mac_followup_20260910/README.md) | EN↔ES follow-up: normalized screens, CPU STT comparison, hymn diagnostics, source validation |
| [`evaluation/attended_mic_20260910/README.md`](evaluation/attended_mic_20260910/README.md) | Attended quiet-room microphone sessions |
| [`evaluation/tts_routing_20260910/README.md`](evaluation/tts_routing_20260910/README.md) | Synthetic acoustic checks and device identity probe |
| [`evaluation/overnight_endurance_20260910/README.md`](evaluation/overnight_endurance_20260910/README.md) | Standard and Lite service hours |
| [`evaluation/overnight_screen_20260910/README.md`](evaluation/overnight_screen_20260910/README.md) | 96-run English screen (0/28 arms) |
| [`evaluation/overnight_closeout_20260910/README.md`](evaluation/overnight_closeout_20260910/README.md) | PR #192 merge and issue closures |
| [`evaluation/overnight_final_validation_20260910/README.md`](evaluation/overnight_final_validation_20260910/README.md) | Final integration validation |
| [`evaluation/bootstrap_review_20260910/README.md`](evaluation/bootstrap_review_20260910/README.md) | Bootstrap review corrections |
| [`evaluation/security_feasibility_20260910/README.md`](evaluation/security_feasibility_20260910/README.md), [`evaluation/overnight_security/README.md`](evaluation/overnight_security/README.md) | Security feasibility and audit runs |
| [`evaluation/mac_v2026_14_report/README.md`](evaluation/mac_v2026_14_report/README.md), [`mac_v2026_14_screening/README.md`](evaluation/mac_v2026_14_screening/README.md), [`mac_v2026_14_routing/README.md`](evaluation/mac_v2026_14_routing/README.md), [`mac_v2026_14_hindi/README.md`](evaluation/mac_v2026_14_hindi/README.md) | v2026.14 frozen baseline, latency screen, routing probes, Hindi probe |
| [`evaluation/overnight_hindi/README.md`](evaluation/overnight_hindi/README.md) | Offline Hindi baseline (completed R&D, no live path) |

Contracts used by the harnesses: [`evaluation/audio_sources.md`](evaluation/audio_sources.md),
[`evaluation/overnight_analysis_contract.md`](evaluation/overnight_analysis_contract.md),
[`evaluation/stt_primary_benchmark_contract.md`](evaluation/stt_primary_benchmark_contract.md).

## Models and research

| Doc | Role |
|---|---|
| [`mlx_cuda_parity.md`](mlx_cuda_parity.md) | MLX ↔ CUDA model parity checklist; Gemma 4 OptiQ as Mac default |
| [`mlx_mtp_notes.md`](mlx_mtp_notes.md) | Gemma 4 MTP drafter on MLX (offline experiment; live `--mts` rejected) |
| [`cuda_latency_proposal.md`](cuda_latency_proposal.md) | CUDA latency proposal for the A2000 box with ready-to-run scripts |
| [`live_diarization.md`](live_diarization.md) | Live diarization design and p95 budget (#133) |
| [`offline_hindi_baseline.md`](offline_hindi_baseline.md) | Offline Hindi audio baseline |
| [`gemma4_tuning/overview.md`](gemma4_tuning/overview.md) | Gemma 4 E2B/E4B QLoRA SFT and CPO program (phases A–E, v1 results, v3 directions) |
| [`hymn_data.md`](hymn_data.md), [`data_provenance.md`](data_provenance.md) | Hymn-domain data and the training-data provenance log |
| [`platense_alignment_bug.md`](platense_alignment_bug.md) | Verse-pair alignment postmortem (why v2 corpora exist) |

## Historical

| Doc | Role |
|---|---|
| [`archive/`](archive/) | Dated benchmarks and notes per release (v2026.5 → v2026.13), research, troubleshooting |
| [`mac_implementation_status_20260909.md`](mac_implementation_status_20260909.md) | Status snapshot before the September 10 integration |
| [`mac_pipeline_refresh_20260830.md`](mac_pipeline_refresh_20260830.md) | Mac refresh notes of 2026-08-30 (latency table invalidated by #172) |
| [`release_plan.md`](release_plan.md) | Release plan for the 2026-03-01 Sunday test (historical) |
