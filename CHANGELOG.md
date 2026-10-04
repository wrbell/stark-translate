# Changelog

Release history of stark-translate, moved from the root `CLAUDE.md` on 2026-10-04.
Detailed notes live under [`docs/archive/`](docs/archive/); this file links,
it does not quote. The release workflow builds each GitHub Release body from
`git log`.

## Work on `main` after the v2026.14 tag

One board per run:

| Board | Outcome |
| --- | --- |
| [Overnight 2026-09-11](docs/evaluation/overnight_20260911/STATUS.md) | Stage attribution, opt-in E2B draft (rejected), diarization interpreter, B615 pinning, tag and release published |
| [Follow-up 2026-09-11](docs/evaluation/followup_20260911/STATUS.md) | Torch 2.13 runtime promoted with the `.stark-python` rollback pointer, PyPI deferred, tail screen (both arms rejected) |
| [Series 3, 2026-09-12](docs/evaluation/series3_20260912/STATUS.md) | Attribution, L-B screen (three admitted arms rejected), first-token report, coverage gate 65, launcher pointer in launchd/bootstrap, Mac runtime audit |
| [Series 4, 2026-09-12](docs/evaluation/series4_20260912/STATUS.md) | Output-identical runtime fixes merged on a paired identity screen (keep-warm after the final, first stream token, Parakeet joint decode; wired limit rejected), first-visible `first_stream` ACK, `partial_reuse_ms` arm rejected (text guard), Smart Turn v3 endpointing no-go, Marian-band review packet |

## Releases

| Era | Summary | Evidence |
| --- | --- | --- |
| v2026.5–6 | llama.cpp CUDA default; operator control plane | [`v2026.5/BENCHMARK.md`](docs/archive/v2026.5/BENCHMARK.md) |
| v2026.7–8 | W16 Whisper CT2; Marian CT2 partials on CUDA | [`v2026.7/STT_BENCHMARK.md`](docs/archive/v2026.7/STT_BENCHMARK.md), [`v2026.8/MARIAN_BENCHMARK.md`](docs/archive/v2026.8/MARIAN_BENCHMARK.md) |
| v2026.9–11 | llama.cpp tuning, IQ4_XS rejected, imatrix calibration | [`v2026.9/GEMMA_OPTIM_PHASE2.md`](docs/archive/v2026.9/GEMMA_OPTIM_PHASE2.md), [`v2026.10/IQ4_XS_BENCHMARK.md`](docs/archive/v2026.10/IQ4_XS_BENCHMARK.md), [`v2026.11/IMATRIX_CALIBRATION.md`](docs/archive/v2026.11/IMATRIX_CALIBRATION.md) |
| v2026.12 | Gemma 4 OptiQ E4B Mac default; EOS bug #172 fixed | [`docs/mlx_cuda_parity.md`](docs/mlx_cuda_parity.md) |
| v2026.13 | Mac latency fixes #180–191; Parakeet EN; Marian CT2 Mac; replay harness | [`v2026.13/MAC_LATENCY.md`](docs/archive/v2026.13/MAC_LATENCY.md) |
| v2026.14.0.0 (published 2026-09-11) | Reliability (isolated capture, health, work lease), schema 2, setup, Review/export, screening, Lite profiles, latency experiments, lay operator page, offline Hindi baseline (PR #192, EN↔ES follow-up PR #196) | [`docs/mac_implementation_status.md`](docs/mac_implementation_status.md), [`docs/lite_profiles.md`](docs/lite_profiles.md) |
