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
| P3 GPU revalidation (normalized EN/ES + 3 paired replays vs `stt_env`) | DONE | 2026-09-11T15:39Z | 2026-09-11T15:58Z | [P-torch213-promotion/README.md](P-torch213-promotion/README.md) | — | FLEURS EN+ES `passed_functionality`; 3/3 pairs byte-identical 26/26, silence p50 deltas −85/+10/−26 ms, previews within 2 pp, Metal flat |
| P4 launcher switch + rollback drill | DONE | 2026-09-11T15:58Z | 2026-09-11T17:59Z | [P-torch213-promotion/README.md](P-torch213-promotion/README.md) | #210 (merged `77de207`) | final drill on main `77de207`: promoted pointer → operator up, preflight en/es pass; rollback pointer → stt_env, preflight passes with the rollback note; pointer restored to venv; stt_env freeze unchanged |
| D1 PyPI deferral (workflow gate + backlog) | DONE | 2026-09-11T15:20Z | 2026-09-11T15:34Z | [backlog.md](../../backlog.md) | #207 (merged `71de7b1`) | publish job gated on PYPI_PUBLISH_ENABLED; item deferred |
| X1 tail harness + `serial_finals` flag PR (Codex) | DONE | 2026-09-11T15:16Z | 2026-09-11T15:42Z | [X-tail-screen/README.md](X-tail-screen/README.md) | #209 (merged `6f160bb`) | serial_finals flag, STT overlap tracing, route fix, tail_screen_report; full suite 2980 passed |
| X2 tail screen runs (2 clips × 360 s, ctl/A1/A2 × 3) | DONE | 2026-09-11T16:00Z | 2026-09-11T17:58Z | | | promoted venv, main `6f160bb`; protocol declared before any run (amended 16:10Z: draft pageout rule made comparative, before any draft result); ctl_r0 kept, screen resumed 16:10Z |
| X3 tail evidence PR | DONE | 2026-09-11T18:00Z | 2026-09-11T18:20Z | [X-tail-screen/README.md](X-tail-screen/README.md) | #206 (this PR) | both arms REJECTED on the declared gates; L1 route-mixture correction appended |
| C1 closeout docs (backlog, status guides) | DONE | 2026-09-11T18:20Z | 2026-09-11T18:30Z | [backlog.md](../../backlog.md) | #206 (this PR) | new item mac-torch213-promotion (validated); caption-delivery-goal, torch, packaging, pypi notes; CLAUDE.md, overnight_status, latency_next_experiments; render/validate/check-links/doc tests pass |

## Log

- 15:11Z S0 started on `main` @ `50ca78f` (clean, no open PRs). `caffeinate -dims` running. Worktrees created for the promotion and tail Codex lanes, the PyPI deferral and this status branch.
- 15:34Z Codex lanes C1 (promotion) and C2 (tail harness) running since 15:16Z. PyPI deferral PR #207 opened with auto-merge. 360 s clips cut: A = 12_14_25 @ 1290 s (speech coverage 0.895), B = 2_8_26 @ 1170 s (0.760; the best 360 s window anywhere in that sermon is 0.777, so the declared coverage floor is 0.75, recorded before any run); no music-like gaps in either.
- 15:50Z D1 merged (#207, main `71de7b1`). Promotion source PR #208 opened with auto-merge after review and a full local suite (2975 passed); branch updated onto main. Tail Codex lane still running. Tail-screen protocol.json declared (sha256 fd19d940…) and 360 s clips frozen before any run.
- 15:33Z #208 merged (main `337790a`); #207 and #199–#205 hunks verified present. Promoted `venv` build chain started from `337790a` (wheel → hash-pinned resolve/install → native smoke → pip-audit → doctor).
- 15:40Z Tail harness PR #209 opened with auto-merge (serial_finals flag, STT overlap tracing, route fix, tail_screen_report). Promoted venv: wheel built from `337790a`, metadata OK, hash-pinned resolution identical to the audited candidate's third-party set (123 packages); install running.
- 15:45Z P2 DONE: promoted `venv` built from `337790a` (torch 2.13.0, torchaudio 2.11.0, mlx 0.32.2; audit 0/124; doctor passes). P3 started on the GPU: normalized FLEURS EN/ES replays from the installed wheel, then 3 alternating paired 150 s replays `stt_env` vs `venv`.
- 15:50Z #209 merged (main `6f160bb`); #208/#207 hunks verified on origin/main. P3: normalized FLEURS EN and ES replays from the installed wheel both `passed_functionality` (validator passed: true); paired replays running (r0 ctl done). The main checkout stays at `337790a` until the paired runs finish so both arms run identical pipeline code.
- 16:03Z P3 DONE (equivalence screen passed). P4: `.stark-python` → `venv/bin/python`; `./run_operator.sh` from a conda-base terminal starts the promoted interpreter (healthz in 2 s, preflight en/es pass). Rollback drill found a gap: with the pointer on `stt_env`, the operator starts but preflight reports `Runtime dependencies: fail` (torch 2.10 outside the new >=2.13 bound) — Start would be blocked on the rollback env. Fix: preflight accepts the retained rollback ranges with a note (PR pending); drill repeats after merge. Tail screen X2 started at 16:00Z on the GPU (18 runs; the first launch's tag bug was fixed and its partial run deleted).
- 16:12Z Preflight rollback fix PR #210 opened (auto-merge); the repeated drill against the fix shows the rollback runtime passing readiness. Tail screen: the machine-wide pageout counter grew by 1,877 pages during the plain control run, so the draft arm's literal 'no pageouts' abort rule would have fired on noise; amended (recorded in protocol.json before any draft result was read) to 'draft pageout growth > max(5× the clip's median control growth, 20,000 pages)'; the in-progress draft run was discarded and the screen resumed from ts0911_A_draft_g3_r0.
- 18:00Z Tail screen complete: 18/18 runs rc 0; draft arm peak Metal 13.4 GiB (budget 15.5), pageout growth within the amended limit. #210 merged (main `77de207`); final launcher drill on main: promoted and rollback pointers both pass readiness. P4 DONE. Report generated (raw rule and amended-pageout-rule variants).
- 18:30Z X3 and C1 DONE. Status PR #206 marked ready with auto-merge.

## Summary

**Everything planned ran; nothing waited on a human.** Four code PRs merged (#207 PyPI deferral gate, #208 Torch 2.13 promotion source + launcher pointer, #209 tail-screen harness + `serial_finals`, #210 preflight rollback fix) plus this status/evidence PR. Production defaults are unchanged; no microphone or speaker was used; `stt_env` is unmodified (freeze SHA `a09be8422c195824…`).

| Lane | Outcome | One line |
|---|---|---|
| P1–P4 Torch 2.13 promotion | DONE | `venv/` (Torch 2.13.0 / TorchAudio 2.11.0, audited constraints, 0/124 audit findings) is the launcher default via `.stark-python`; FLEURS EN/ES replays pass; 3/3 paired replays byte-identical vs `stt_env`; rollback drill passes both ways after #210 fixed a preflight gap that would have blocked Start on the rollback env. |
| D1 PyPI | DONE | Deferred by decision; the publish job is gated on `PYPI_PUBLISH_ENABLED` (unset), so future tags no longer fail; backlog item `deferred`. |
| X1–X3 tail screen | DONE (both arms REJECTED) | E2B draft γ=3: faster Gemma decode but the resident model set starves STT (p95 0.5 → 2.5–3 s), previews −30 %, Gemma p95 ×2–3. `serial_finals`: overlap is rare, waiting only hurts. 173–174 Gemma-routed silence finals per arm — screen without p95 claim. |
| L1 correction | DONE | The overnight "Gemma cohort" was a route mixture; Gemma final translation median is ~0.8 s, Marian-routed finals are already sub-second at the median. |

**For Willem**
1. Saturday: attended microphone session on the promoted environment (`./run_operator.sh` from any terminal now starts `venv/bin/python`). If anything misbehaves: `printf '%s\n' "$PWD/stt_env/bin/python" > .stark-python`.
2. Latency: no qualified component exists and both tail hypotheses are closed; the next hypothesis should come from the new `physical_stt` overlap trace fields, and a live draft needs a text-only memory budget this Mac does not have.
3. #133 natural two-speaker clip still needs a recording (unchanged).

**Not done / caveats:** no human-quality, visible-display, live-microphone or x86/RTX 2070 evidence was produced; all latency numbers are engineering screens on machine-timed replays; the wheel installed in `venv` is from `337790a` while launches from the checkout run current `main` code.
