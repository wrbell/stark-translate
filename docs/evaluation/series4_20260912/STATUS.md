# Series 4 execution status — 2026-09-11 → 2026-09-12 (latency proposals P1–P6)

- **Run id:** series4-20260912 · **Base:** `main` @ `5227a73` · **Machine:** MacBook M3 Pro 18 GB (MLX); launcher `venv/bin/python` (Torch 2.13.0 / TorchAudio 2.11.0) via `.stark-python`; `stt_env` rollback untouched (freeze SHA256 `a09be8422c195824…`)
- **Plan:** approved 2026-09-11 19:30 EDT. Lanes: S0 pre-flight; P1 runtime overhead fixes (wired limit at load, keep-warm after the final, first stream token, Parakeet joint scalar eval) with micro-bench and a paired identity screen; P2 partial-reuse attribution, arm PR and declared screen; P3 end-of-utterance classifier offline feasibility; P4 Marian-vs-Gemma review packet for 8–12-word finals; P6 first-visible acknowledgement instrumentation; P2-R registry; C1 closeout. P5 (E2B / QAT quant) is a decision item and is not run.
- **Constraints acknowledged:** production defaults unchanged (E4B finals, 0.5 s silence, 0.6 s cadence, Parakeet EN / Whisper ES, Marian CT2 previews); no microphone capture or audio playback; `stt_env` never modified; one inference process at a time; no rejected arm re-run or combined; no fabricated labels or references; evidence immutable once written; no Chinese-origin models (Smart Turn v3 is Pipecat/Daily, BSD-2, US).
- **Policy (approved 2026-09-11):** exact-output engineering changes (P1) merge on a paired identity screen (byte-identical finals, no p95 regression beyond max(5 %, 100 ms), previews within 2 pp, Metal within 1 GiB); behaviour-changing arms (P2) keep the 15 %/150 ms screening gate plus text guards and stay opt-in.
- **States:** PENDING → RUNNING → DONE | REJECTED (ran, gate failed; still evidence) | BLOCKED (cannot proceed; log path in Note) | STOPPED (07:30 EDT cut-off). A crashed run is retried once, then BLOCKED. A screened PR whose gates fail stays open with the numbers in its description and its lane is REJECTED; that is terminal.

| Lane | State | Started (UTC) | Ended (UTC) | Evidence | PR | Note |
|---|---|---|---|---|---|---|
| S0 pre-flight (caffeinate, worktrees, status branch) | DONE | 2026-09-11T23:35Z | 2026-09-11T23:41Z | [STATUS.md](STATUS.md) | #215 | caffeinate pid 48194; stt_env freeze a09be842…; pointer venv; five worktrees |
| P1 runtime PR (Codex): wired limit at load, keep-warm after the final, first stream token, Parakeet joint scalar eval | RUNNING | 2026-09-11T23:41Z | | | | Codex C1 finished 23:51Z; under review |
| P1 micro-bench (text bench + Parakeet profile, before/after, same interpreter) | PENDING | | | | | |
| P1 paired identity screen (2 clips × 3 pairs, control checkout vs P1 checkout) | PENDING | | | | | |
| P2 attribution (last-partial coverage, lead and text identity over the six control runs) | DONE | 2026-09-11T23:45Z | 2026-09-11T23:50Z | [P2-attribution/README.md](P2-attribution/README.md) | #215 (this PR) | 267 silence finals: last partial within 300 ms of speech end in 98.5 %, its STT done before finalize in 100 % (lead p50 318 ms), text equal 84 %; final STT p50 360 ms; design A admitted |
| P2 arm PR (Codex): opt-in `partial_reuse_ms` | RUNNING | 2026-09-11T23:41Z | | | | Codex C3 finished 23:49Z; under review |
| P2 arm screen (declared; control vs reuse-100 vs reuse-300) | PENDING | | | | | |
| P3 EOU classifier offline feasibility (Smart Turn v3, CPU) | DONE | 2026-09-11T23:42Z | 2026-09-11T23:52Z | [P3-eou-feasibility/README.md](P3-eou-feasibility/README.md) | #215 (this PR) | no-go: AUC 0.54–0.57; internal pauses judged complete 46–48 % vs true ends 56–57 %; clip A misses 76 % of ends; no arm declared |
| P4 Marian-vs-Gemma review packet (8–12-word Gemma-routed finals) | DONE | 2026-09-11T23:47Z | 2026-09-11T23:55Z | [P4-marian-band-packet/README.md](P4-marian-band-packet/README.md) | #215 (this PR) | packet 031184b1…: 30 unique finals, 22 differ; chrF p50 89; glossary 14/14 both; human review pending |
| P6 first-visible ACK PR (Codex) | RUNNING | 2026-09-11T23:41Z | | | | Codex C2 launched in `../SRTranslate-wt-p6` |
| P2-R registry rows (P1 engineering result, P2 outcome) | PENDING | | | | | |
| C1 closeout docs | PENDING | | | | | |

## Log

- 23:35Z S0 started on `main` @ `5227a73` (clean). `caffeinate -dims` running (pid 48194). Worktrees: `../SRTranslate-wt-p1` (`codex/series4-p1-runtime`), `../SRTranslate-wt-p6` (`codex/series4-p6-ack`), `../SRTranslate-wt-p2` (`codex/series4-p2-reuse`), `../SRTranslate-wt-base` (detached `5227a73`, paired-screen control checkout), `../SRTranslate-wt-status` (this branch); each links `stark_data/{raw,replay,live_sessions}`, `adapters` and copies `.stark-python`. GPU idle; 159 GB free.
- 23:41Z S0 DONE. Status PR #215 opened (draft). Codex C1 (P1 runtime fixes), C2 (P6 first-visible ACK) and C3 (P2 `partial_reuse_ms` arm) launched in their worktrees. Smart Turn v3 pinned download and the P2 attribution / P4 packet builds start off-GPU.
- 23:55Z P2 attribution, P3 feasibility (no-go) and P4 packet DONE off-GPU. Codex C3 (P2 arm) and C1 (P1) finished; C2 (P6) still running. `venv` has no pytest/ruff (runtime env): unit tests run with `stt_env/bin/python -m pytest` (read-only use of the rollback env).
