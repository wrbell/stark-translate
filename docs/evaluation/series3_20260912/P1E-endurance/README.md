# P1-E — promoted-runtime service endurance replay (2026-09-11)

**Result: completed.** The operator (`venv/bin/python -m operator_app.cli operator`, `main` @ `a46649e`... launched at
`dc983c6`+#212/#213 state; see note) replayed the full 3,640 s natural English service
(`stark_data/raw/Gospel_Message_(12_14_25)_5D2rOMvkwrk.wav`, SHA256 `8bec0f10…`) as session `20260911_144155_276040_en`
through the production Start path (`POST /api/session/start`, profile standard, Parakeet EN, E4B, recording off, TTS off).
Lifecycle `completed` (exit 0), 2026-09-11T18:41:58Z → 2026-09-11T19:47:23Z; peak RSS 3.38 GiB,
peak Metal 8.66 GiB; health phase `ready` for all 769 monitor samples, no stale health,
monitor errors [].

| | value |
|---|---|
| finals / partials | 563 / 3049 |
| endpoints | 38 hard_cut, 401 silence, 124 smart_cut |
| final routes | 360 gemma, 203 marian |
| silence-final speech_end→final p50 / p95 (all routes) | 1213.8 / 2236.2 ms |
| Gemma-routed silence p50 / p95 | 1631.3 / 2563.3 ms |
| Marian-routed silence p50 / p95 | 843.7 / 1049.3 ms |
| smart+hard cuts p50 / p95 | 2572.0 / 5935.0 ms |
| silence p50, first third → last third of the service | 1241.4 → 914.2 ms |
| process-tree RSS (monitor, 5 s samples) p50 / max | 510 / 663 MiB; first 10 samples mean None → last 10 mean None MiB |
| process-tree CPU (100 % = one core) p50 / p95 / max | 42.2 / 81.4 / 117.8 % |

Monitor: `tools/endurance_monitor.py` (`--duration-seconds 3850`, `partial`, stop reason `deadline` — the
monitor's window ended before the pipeline exited because it was re-attached about a minute after the session
started, so process cleanup was verified from the lifecycle record and `pgrep` rather than by the monitor).
Raw records under [`raw/`](raw/) (lifecycle, metadata, health, hardware, monitor report, start/preflight responses);
launch script [`run_endurance.sh`](run_endurance.sh). The 116 MB source WAV, the 3.6k-line diagnostics and the
per-final CSV stay local (`.cache/series3-20260912/P1E/`).

Caveats: the first 403 s of the source are digital silence; the first ~10 minutes of the run overlapped two local
full test-suite executions and the operator was launched before PR #214 merged (the pipeline code is the checkout at
launch time, `c914e14`). This is an observational run on the promoted runtime: it is not a causal comparison with the
2026-09-10 `stt_env` endurance cohorts, and it certifies no human quality, visible display or microphone behaviour.
