# Follow-up execution status — 2026-09-11 (daytime series)

- **Run id:** followup-20260911 · **Base:** `main` @ `50ca78f` · **Machine:** MacBook M3 Pro 18 GB (MLX), `stt_env` Python 3.11.11 (freeze SHA256 `a09be8422c195824…`, rollback env)
- **Plan:** approved 2026-09-11 ~11:15 EDT. Lanes: P1–P4 Torch 2.13 promotion (source PR, promoted `venv` build, GPU revalidation, launcher pointer + rollback drill), D1 PyPI deferral, X1–X3 declared p95-tail screen (E2B draft γ=3, opt-in `serial_finals`), C1 closeout.
- **Constraints acknowledged:** production defaults unchanged (E4B, 0.5 s silence, 0.6 s cadence, Parakeet EN / Whisper ES, Marian CT2 previews); no microphone capture or audio playback; working `stt_env` never modified (it is the rollback); one inference process at a time; no rejected screen arm re-run as a confirmation; no fabricated references or labels; published tags never move.
- **States:** PENDING → RUNNING → DONE | REJECTED (ran, gate failed; still evidence) | BLOCKED (cannot proceed; log path in Note) | STOPPED (19:00 EDT cut-off). A crashed run is retried once, then BLOCKED.

| Lane | State | Started (UTC) | Ended (UTC) | Evidence | PR | Note |
|---|---|---|---|---|---|---|
| S0 pre-flight (caffeinate, worktrees, status branch) | DONE | 2026-09-11T15:11Z | 2026-09-11T15:16Z | [STATUS.md](STATUS.md) | #206 | caffeinate pid 73151; stt_env freeze a09be842…; 4 worktrees |
| P1 promotion source PR (Codex) | DONE | 2026-09-11T15:16Z | 2026-09-11T15:30Z | [P-torch213-promotion/README.md](P-torch213-promotion/README.md) | #208 (merged `337790a`) | Codex done 15:22Z; full suite 2975 passed; pyproject pins, preflight bounds, `.stark-python` pointer, constraints file, docs |
| P2 promoted `venv` build (Torch 2.13.0 / TorchAudio 2.11.0) | DONE | 2026-09-11T15:33Z | 2026-09-11T15:38Z | [P-torch213-promotion/README.md](P-torch213-promotion/README.md) | — | wheel from `337790a`; hash-pinned set identical to the audited candidate (123); pip check, native smoke, pip-audit 0/124, doctor en/es/en+diarize pass |
| P3 GPU revalidation (normalized EN/ES + 3 paired replays vs `stt_env`) | RUNNING | 2026-09-11T15:45Z | | | | serial GPU; ~40 min |
| P4 launcher switch + rollback drill | PENDING | | | | | |
| D1 PyPI deferral (workflow gate + backlog) | DONE | 2026-09-11T15:20Z | 2026-09-11T15:34Z | [backlog.md](../../backlog.md) | #207 (merged `71de7b1`) | publish job gated on PYPI_PUBLISH_ENABLED; item deferred |
| X1 tail harness + `serial_finals` flag PR (Codex) | RUNNING | 2026-09-11T15:16Z | | | #209 (auto-merge) | Codex done 15:33Z; full suite 2980 passed |
| X2 tail screen runs (2 clips × 360 s, ctl/A1/A2 × 3) | PENDING | | | | | after P4 and X1 |
| X3 tail evidence PR | PENDING | | | | | |
| C1 closeout docs (backlog, status guides) | PENDING | | | | | last |

## Log

- 15:11Z S0 started on `main` @ `50ca78f` (clean, no open PRs). `caffeinate -dims` running. Worktrees created for the promotion and tail Codex lanes, the PyPI deferral and this status branch.
- 15:34Z Codex lanes C1 (promotion) and C2 (tail harness) running since 15:16Z. PyPI deferral PR #207 opened with auto-merge. 360 s clips cut: A = 12_14_25 @ 1290 s (speech coverage 0.895), B = 2_8_26 @ 1170 s (0.760; the best 360 s window anywhere in that sermon is 0.777, so the declared coverage floor is 0.75, recorded before any run); no music-like gaps in either.
- 15:50Z D1 merged (#207, main `71de7b1`). Promotion source PR #208 opened with auto-merge after review and a full local suite (2975 passed); branch updated onto main. Tail Codex lane still running. Tail-screen protocol.json declared (sha256 fd19d940…) and 360 s clips frozen before any run.
- 15:33Z #208 merged (main `337790a`); #207 and #199–#205 hunks verified present. Promoted `venv` build chain started from `337790a` (wheel → hash-pinned resolve/install → native smoke → pip-audit → doctor).
- 15:40Z Tail harness PR #209 opened with auto-merge (serial_finals flag, STT overlap tracing, route fix, tail_screen_report). Promoted venv: wheel built from `337790a`, metadata OK, hash-pinned resolution identical to the audited candidate's third-party set (123 packages); install running.
- 15:45Z P2 DONE: promoted `venv` built from `337790a` (torch 2.13.0, torchaudio 2.11.0, mlx 0.32.2; audit 0/124; doctor passes). P3 started on the GPU: normalized FLEURS EN/ES replays from the installed wheel, then 3 alternating paired 150 s replays `stt_env` vs `venv`.
