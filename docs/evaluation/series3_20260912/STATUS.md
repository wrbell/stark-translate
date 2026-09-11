# Series 3 execution status — 2026-09-11 → 2026-09-12 (latency P0 + automatable backlog)

- **Run id:** series3-20260912 · **Base:** `main` @ `dc983c6` · **Machine:** MacBook M3 Pro 18 GB (MLX); launcher `venv/bin/python` (Torch 2.13.0 / TorchAudio 2.11.0) via `.stark-python`; `stt_env` rollback untouched (freeze SHA256 `a09be8422c195824…`)
- **Plan:** approved 2026-09-11 ~15:00 EDT. Lanes: P2-H harness PR, P2-L launcher PR, P3-C coverage gate, P3-S installed smoke + runtime audit, P1-H hymn second control + slices, P1-E promoted-runtime endurance, L-A attribution, L-B arms (only those L-A admits), L-C first-token report, P2-R closed-arm registry, C1 closeout.
- **Constraints acknowledged:** production defaults unchanged; no microphone capture or audio playback; `stt_env` never modified; one inference process at a time; no rejected arm re-run or combined (the allocator arm was found already applied and rejected, so it is not run); no fabricated labels; evidence immutable once written.
- **States:** PENDING → RUNNING → DONE | REJECTED (ran, gate failed; still evidence) | BLOCKED (cannot proceed; log path in Note) | STOPPED (07:30 EDT cut-off, or "no measured mechanism (L-A)" for an arm). A crashed run is retried once, then BLOCKED.

| Lane | State | Started (UTC) | Ended (UTC) | Evidence | PR | Note |
|---|---|---|---|---|---|---|
| S0 pre-flight (caffeinate, worktrees, status branch) | DONE | 2026-09-11T19:05Z | 2026-09-11T19:12Z | [STATUS.md](STATUS.md) | #211 | caffeinate pid 11163; stt_env freeze a09be842…; pointer venv |
| P2-H harness PR (Codex): cache-limit consistency, `partial_recheck_translation`, cpu_ms + first_stream_token trace, attribution tool, aligned G4, clip prep | RUNNING | 2026-09-11T19:14Z | | | | |
| P2-L launcher PR (Codex): launchd honours `.stark-python`, bootstrap protects `stt_env`, `check_dependency_audit --runtime mac`, `scripts/audit_mac_runtime.sh` | RUNNING | 2026-09-11T19:14Z | | | | |
| P3-C coverage gate 50 → 65 | RUNNING | 2026-09-11T19:10Z | | | #212 (auto-merge) | measured 68.17 % on main |
| P3-S installed smoke + Mac runtime audit | RUNNING | 2026-09-11T19:10Z | | | | smoke passed (5 routes 200, 45 runtime files, verse parser ok); pip-audit 0 findings / 124; `--runtime mac` check pending P2-L |
| P1-H hymn #193 second control + slices | RUNNING | 2026-09-11T19:16Z | | | | wrapper running under its 600 s bound on venv; slices cut |
| P1-E promoted-runtime service endurance (3,640 s replay) | PENDING | | | | | |
| L-A attribution | PENDING | | | | | after P2-H merges |
| L-B arm 2 `marian_threads_2` | PENDING | | | | | only if L-A admits |
| L-B arm 3 `max_utterance_6` | PENDING | | | | | only if L-A admits |
| L-B arm 4 `partial_recheck_translation` | PENDING | | | | | only if L-A admits |
| L-C first-token report | PENDING | | | | | |
| P2-R closed-arm registry PR | PENDING | | | | | after L-B |
| C1 closeout docs | PENDING | | | | | last |

## Log

- 19:05Z S0 started on `main` @ `dc983c6` (clean). `caffeinate -dims` running. Worktrees created for the harness and launcher Codex lanes, the coverage-gate PR and this status branch.
- 19:16Z S0 DONE. Codex C1 (harness) and C2 (launcher) launched 19:14Z. Coverage PR #212 opened with auto-merge. Installed smoke passed from outside the checkout; venv pip-audit clean (124 distributions). Hymn second control started on the GPU (promoted interpreter); the two search slices were cut and hashed.
