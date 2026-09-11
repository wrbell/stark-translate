# Series 3 execution status — 2026-09-11 → 2026-09-12 (latency P0 + automatable backlog)

- **Run id:** series3-20260912 · **Base:** `main` @ `dc983c6` · **Machine:** MacBook M3 Pro 18 GB (MLX); launcher `venv/bin/python` (Torch 2.13.0 / TorchAudio 2.11.0) via `.stark-python`; `stt_env` rollback untouched (freeze SHA256 `a09be8422c195824…`)
- **Plan:** approved 2026-09-11 ~15:00 EDT. Lanes: P2-H harness PR, P2-L launcher PR, P3-C coverage gate, P3-S installed smoke + runtime audit, P1-H hymn second control + slices, P1-E promoted-runtime endurance, L-A attribution, L-B arms (only those L-A admits), L-C first-token report, P2-R closed-arm registry, C1 closeout.
- **Constraints acknowledged:** production defaults unchanged; no microphone capture or audio playback; `stt_env` never modified; one inference process at a time; no rejected arm re-run or combined (the allocator arm was found already applied and rejected, so it is not run); no fabricated labels; evidence immutable once written.
- **States:** PENDING → RUNNING → DONE | REJECTED (ran, gate failed; still evidence) | BLOCKED (cannot proceed; log path in Note) | STOPPED (07:30 EDT cut-off, or "no measured mechanism (L-A)" for an arm). A crashed run is retried once, then BLOCKED.

| Lane | State | Started (UTC) | Ended (UTC) | Evidence | PR | Note |
|---|---|---|---|---|---|---|
| S0 pre-flight (caffeinate, worktrees, status branch) | DONE | 2026-09-11T19:05Z | 2026-09-11T19:12Z | [STATUS.md](STATUS.md) | #211 | caffeinate pid 11163; stt_env freeze a09be842…; pointer venv |
| P2-H harness PR (Codex): cache-limit consistency, `partial_recheck_translation`, cpu_ms + first_stream_token trace, attribution tool, aligned G4, clip prep | DONE | 2026-09-11T19:14Z | 2026-09-11T18:58Z | [LA-attribution/README.md](LA-attribution/README.md) | #214 (merged `a46649e`) | rebuilt on `c914e14` by cherry-pick; #213 hunks verified on main after merge |
| P2-L launcher PR (Codex): launchd honours `.stark-python`, bootstrap protects `stt_env`, `check_dependency_audit --runtime mac`, `scripts/audit_mac_runtime.sh` | DONE | 2026-09-11T19:14Z | 2026-09-11T18:52Z | [P3-runtime-smoke/README.md](P3-runtime-smoke/README.md) | #213 (merged `c914e14`) | Codex done 18:44Z; full suite 3040 passed |
| P3-C coverage gate 50 → 65 | DONE | 2026-09-11T19:10Z | 2026-09-11T18:36Z | [P3-runtime-smoke/README.md](P3-runtime-smoke/README.md) | #212 (merged `e92f831`) | measured 68.17 % on main; gate now 65 |
| P3-S installed smoke + Mac runtime audit | DONE | 2026-09-11T19:10Z | 2026-09-11T19:12Z | [P3-runtime-smoke/README.md](P3-runtime-smoke/README.md) | #211 (this PR) | smoke passed (5 routes 200, 45 runtime files, verse parser ok); pip-audit 0 findings / 124; `--runtime mac` check passed |
| P1-H hymn #193 second control + slices | DONE | 2026-09-11T18:35Z | 2026-09-11T18:41Z | [P1H-hymn-control/README.md](P1H-hymn-control/README.md) | #211 (this PR) | completed_technical_diagnostic, 0 errors; 18 finals, same endpoint mix and routing as the c13f51f control; hold never entered; two search slices cut and hashed |
| P1-E promoted-runtime service endurance (3,640 s replay) | DONE | 2026-09-11T18:42Z | 2026-09-11T19:48Z | [P1E-endurance/README.md](P1E-endurance/README.md) | #211 (this PR) | lifecycle completed exit 0; peak Metal 8.7 GiB; health ready throughout; observational |
| L-A attribution | DONE | 2026-09-11T19:48Z | 2026-09-11T20:09Z | [LA-attribution/README.md](LA-attribution/README.md) | #211 (this PR) | 34.6 % of Gemma decode overlapped by partial STT; live tok/s −13 % vs isolated with process CPU > 1 core at p95; smart cuts 40 % of finals at +1.5 s; all three arms admitted |
| L-B arm 2 `marian_threads_2` | RUNNING | 2026-09-11T20:22Z | | [LB-tail-screen/protocol.json](LB-tail-screen/protocol.json) | | admitted by L-A; 24-run screen on clips A/B |
| L-B arm 3 `max_utterance_6` | RUNNING | 2026-09-11T20:22Z | | [LB-tail-screen/protocol.json](LB-tail-screen/protocol.json) | | admitted by L-A; 24-run screen on clips A/B |
| L-B arm 4 `partial_recheck_translation` | RUNNING | 2026-09-11T20:22Z | | [LB-tail-screen/protocol.json](LB-tail-screen/protocol.json) | | admitted by L-A; 24-run screen on clips A/B |
| L-C first-token report | PENDING | | | | | |
| P2-R closed-arm registry PR | PENDING | | | | | after L-B |
| C1 closeout docs | PENDING | | | | | last |

## Log

- 19:05Z S0 started on `main` @ `dc983c6` (clean). `caffeinate -dims` running. Worktrees created for the harness and launcher Codex lanes, the coverage-gate PR and this status branch.
- 19:16Z S0 DONE. Codex C1 (harness) and C2 (launcher) launched 19:14Z. Coverage PR #212 opened with auto-merge. Installed smoke passed from outside the checkout; venv pip-audit clean (124 distributions). Hymn second control started on the GPU (promoted interpreter); the two search slices were cut and hashed.
- 19:25Z #212 merged (main `e92f831`, coverage gate 65). L-C first-token report written from the six traced control runs (Gemma first token p50 1.13–1.21 s; all-route first-visible p50 1.09–1.19 s vs payload-ready 1.6–2.0 s).
- 18:45Z P1-H DONE (second control on the promoted runtime: identical segmentation to the first control, hold never entered; slices 400–470 s and 700–780 s cut and hashed). P1-E endurance started through the operator on venv (3,640 s replay, recording off, monitor attached).
- 18:55Z Codex lanes finished 18:44Z; reviewed; launcher PR #213 and harness PR #214 opened with auto-merge (harness: one existing dispatch test updated for the new constructor argument). Endurance replay still running.
- 19:05Z #213 merged (main `c914e14`), hunks verified. #214's first commit header exceeded commitlint's 100 chars; while shortening it a `git reset --soft origin/main` after main had moved produced a squash that would have reverted #213 — caught before CI finished, branch rebuilt by cherry-pick onto `c914e14` (6e00f5e), #213 hunks verified on the branch, force-pushed; CI restarted.
- 19:12Z #214 merged (main `a46649e`, parent `c914e14`; #213 hunks verified). P3-S DONE (`--runtime mac` check passed on the venv audit). Endurance replay continues; L-A starts when it ends.
- 19:50Z P1-E DONE (3,640 s replay through the operator on the promoted runtime completed cleanly). Main pulled to `a46649e`; L-A started on the GPU.
- 20:22Z L-A DONE; all three arms admitted with measured mechanisms. L-B protocol declared (protocol.json copied before run 1) and the 24-run screen started on the GPU (≈ 2.6 h).
