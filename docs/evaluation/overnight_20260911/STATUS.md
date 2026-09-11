# Overnight execution status — 2026-09-10 → 2026-09-11

- **Run id:** overnight-20260911 · **Base:** `main` @ `9983f68` · **Machine:** MacBook M3 Pro 18 GB (MLX), `stt_env` Python 3.11.11
- **Plan:** approved 2026-09-10 evening. Lanes: L1 silence-final stage attribution, L2 E2B-as-draft speculative decoding, L4 diarization latency half-gate, L5 B615 pinning, L6 audited Torch 2.13 candidate, L8 `v2026.14.0.0` tag, L7 closeout docs. Hymn #193 (L3) deliberately not run tonight.
- **Constraints acknowledged:** production defaults unchanged (E4B, 0.5 s silence, 0.6 s cadence, Parakeet EN / Whisper ES, Marian CT2 previews); no microphone capture or audio playback; working `stt_env` untouched (freeze SHA256 `a09be8422c195824…`); one inference process at a time; no rejected screen arm re-run as a confirmation; no fabricated references or labels; published tags never move.
- **States:** PENDING → RUNNING → DONE | REJECTED (ran, gate failed; still evidence) | BLOCKED (cannot proceed; log path in Note) | STOPPED (07:30 cut-off). A crashed run is retried once, then BLOCKED.

| Lane | State | Started (UTC) | Ended (UTC) | Evidence | PR | Note |
|---|---|---|---|---|---|---|
| L0 pre-flight (worktree cleanup, caffeinate, branches) | DONE | 2026-09-11T02:00Z | 2026-09-11T02:10Z | [L0/README.md](L0/README.md) | | |
| L1 silence-final stage attribution (analysis) | DONE | 2026-09-11T02:05Z | 2026-09-11T02:20Z | [L1-silence-stages/README.md](L1-silence-stages/README.md) | #199 (merged) | 401 silence finals per cohort; checksum OK; 0.49 s trigger + 0.40 s STT + 0.38 s Gemma at p50 |
| L2 E2B-draft speculative decoding: isolated text bench | REJECTED | 2026-09-11T03:06Z | 2026-09-11T03:15Z | [L2-e2b-draft/README.md](L2-e2b-draft/README.md) | #200 (merged `d5433f9`) | 21/21 byte-identical; medium/long −18…−23 % at γ=2–3 (gate met) but canary-length −7…−14 % (gate unmet); Metal 6.4 → 10.5 GiB |
| L2b E2B-draft: paired real-time replay (informational) | REJECTED | | | [L2-e2b-draft/README.md](L2-e2b-draft/README.md) | — | not executed: entry condition (full L2 gate) unmet; retained as a p95-tail hypothesis for a separately declared experiment |
| L4 diarization #133 latency half-gate | DONE | 2026-09-11T02:41Z | 2026-09-11T03:06Z | [L4-diarize-half-gate/README.md](L4-diarize-half-gate/README.md) | #201 (merged `50f81c6`) | 12/12 runs; church clip p95 within +50 ms in 3/3 pairs, synthetic 2/3; labels deterministic, 8/9 consistent on accented synthetic voices; captions identical; natural two-speaker labels still pending |
| L5 B615 pinning (Codex code lane) | DONE | 2026-09-11T02:03Z | 2026-09-11T02:47Z | [L5-b615-pinning/README.md](L5-b615-pinning/README.md) | #202 (merged `2cf0f7a`) | 13 optional-live sites + legacy Piper loader pin or reject; 105 offline sites documented |
| L6 audited Torch 2.13 candidate env from main | DONE | 2026-09-11T02:07Z | 2026-09-11T02:36Z | [L6-torch213-candidate/README.md](L6-torch213-candidate/README.md) | #198 | audit 0/124 findings; EN+ES normalized replay checks passed; 3/3 paired replays equivalent with byte-identical finals; launcher untouched; go recommended for an attended promotion |
| L8 tag `v2026.14.0.0` + workflow outcomes | DONE | 2026-09-11T02:53Z | 2026-09-11T03:24Z | [release page](https://github.com/wrbell/stark-translate/releases/tag/v2026.14.0.0) | #203 (merged `bbdcfe5`) | Release: success (mac/nvidia/windows ZIPs); Windows MSI: success (MSI uploaded); PyPI: build ok, publish `invalid-publisher` (trusted publisher unmapped, expected); Docker: failed on the pre-existing missing `workers.py` COPY → fixed in #203, GHCR rebuild from `bbdcfe5` (run 34558272252) failed one step later on `training/__init__.py` (force-included but not copied; `stark_translate/` was missing too) → second Dockerfile fix #204 (`7aac2fd`); GHCR rebuild run 34559651143 from `7aac2fd` **succeeded** and pushed `ghcr.io/wrbell/stark-translate:latest ghcr.io/wrbell/stark-translate:main ghcr.io/wrbell/stark-translate:v2026.14.0.0` (image built from main after the two Dockerfile fixes, not from the tag commit) |
| L7 closeout docs (backlog, status, CLAUDE.md) | DONE | 2026-09-11T03:16Z | 2026-09-11T03:30Z | [backlog.md](../../backlog.md) | #198 (this PR) | backlog.json items updated (B615 → validated/met; diarize, torch, caption-delivery, pypi, two-speaker notes); release statements in CLAUDE.md/README/roadmap/status guides; render/validate/check-links/doc tests pass |

## Log

- 02:10Z L0 started. Six `codex/overnight-*` branches were rebased before #192 merged, so they are not ancestors of `main`; their worktrees were removed (branch refs kept, `git branch -d` refused). The one tracked modification (overnight-lite `tools/endurance_monitor.py`) is saved in `.cache/overnight-20260911/L0/overnight-lite-uncommitted.patch`. Detached bench worktrees (ancestors) removed; their specs saved beside it.
- 02:12Z Status PR #198 opened (draft). Codex lanes launched in `../SRTranslate-wt-{b615,stages,e2b-draft,diarize}`. ECAPA snapshot prefetched at pinned revision. v6 endurance archive reconstructed (SHA256 `a9efcd8f…`). Torch 2.13 scratch wheel built from `9983f68` (metadata OK); resolve/install/smoke/audit chain running off-GPU.
- 02:20Z L1 DONE. Codex lanes all finished; PRs #199 (merged), #200, #201, #202 (auto-merge). L6 EN/ES normalized replay checks passed; paired stt_env vs Torch 2.13 replays started on the GPU (six alternating 150 s runs).
- 02:40Z L6 DONE (equivalence screen, go recommended). Found and fixed: torchaudio 2.11 needs TorchCodec for load(); diarization WAV read switched to soundfile on the #201 branch. Harness PRs were BEHIND main after #199 merged; branches updated sequentially (#200 first).
- 02:47Z #200 and #202 merged (main `2cf0f7a`); prior hunks verified present. #201 updated to main, CI running. L4 replays running (2/12 done). L5 DONE.
- 02:53Z #201 merged (main `50f81c6`, all four overnight code PRs in, prior hunks verified). Main CI green. Annotated tag `v2026.14.0.0` pushed; tag workflows started.
- 02:56Z Tag workflows: Release succeeded; PyPI publish failed with `invalid-publisher` (trusted publisher for owner wrbell / repo stark-translate / workflow pypi.yml / environment pypi is not configured on PyPI) — expected, not retried; Docker and MSI still running.
- 03:06Z L4 DONE (screening). L2 text bench started on the GPU. Tag workflows so far: Release success, MSI success, PyPI publish invalid-publisher (expected), Docker running.
- 03:15Z L2 text bench complete: byte-identical output, strong medium/long gain, short-sentence gain below gate → REJECTED; L2b not run. Docker tag workflow failed (see L8 note); Release and MSI succeeded, PyPI invalid-publisher.
- 03:24Z PR #203 (Dockerfile) merged; GHCR image build for `v2026.14.0.0` dispatched from main `bbdcfe5` (run 34558272252). Closeout docs committed. Codex worktrees removed; working `stt_env` freeze unchanged (`a09be8422c195824…`); `git diff 9983f68 -- settings.py models.lock.json stark_translate/profiles.py pyproject.toml` is empty.

## Morning summary

**Everything planned ran; nothing waited on a human.** Four code PRs merged (#199 stage attribution, #200 opt-in E2B draft, #201 diarization interpreter + synthetic clip + soundfile fix, #202 B615 pinning) plus the Dockerfile fixes (#203, #204). `v2026.14.0.0` is tagged and published as a GitHub Release with the MSI. Production defaults, `stt_env` and the launcher are untouched. No microphone or speaker was used.

| Lane | Outcome | One line |
|---|---|---|
| L1 stage attribution | DONE | A silence final is 0.49 s silence trigger + 0.40 s Parakeet + 0.38 s Gemma at the median (1.43–1.46 s total); everything else < 5 ms. Sub-second needs STT+translation under 0.5 s combined; the real lever is the p95 tails (Metal contention, long finals). |
| L2 E2B draft for E4B | REJECTED (gate) | Byte-identical output 21/21; medium/long sentences −18…−23 % at γ=2–3, but canary-length (14-token) sentences only −7…−14 % → below the 15 %/150 ms gate; Metal 6.4 → 10.5 GiB. Paired replay not run. Kept as a p95-tail hypothesis. |
| L4 diarization half-gate | DONE (screening) | `--diarize` within the +50 ms p95 budget on 3/3 natural-clip pairs, 2/3 synthetic; captions identical; deterministic labels, 8/9 consistent on the accented synthetic voices. Natural two-speaker labels still pending. |
| L5 B615 pinning | DONE | 13 optional-live sites + legacy Piper loader pin or raise `UnpinnedModelError`; 105 offline sites documented; backlog item now validated/met. |
| L6 Torch 2.13 candidate | DONE | Built from current main: 0/124 audit findings, EN/ES replay checks all pass, 3/3 paired replays equivalent with byte-identical finals and flat Metal peak. **Go recommended** for an attended launcher switch with rollback. |
| L8 tag | DONE | Release + MSI succeeded; PyPI `invalid-publisher`; Docker failed on a pre-existing build-context defect, fixed in #203 + #204, and the GHCR image for `v2026.14.0.0` was then built from `7aac2fd` and pushed (run 34559651143). |

**Decisions for Willem**
1. Promote the Torch 2.13 environment (rebuild final source, switch launcher with rollback) — evidence in `L6-torch213-candidate/`.
2. Map the PyPI trusted publisher (owner `wrbell`, repo `stark-translate`, workflow `pypi.yml`, environment `pypi`) and re-run the PyPI workflow for the existing tag, or leave PyPI unpublished.
3. Whether to declare a p95-tail latency experiment (E2B draft on long finals + Metal contention) — the medians cannot reach sub-second with any single 15 % gain.
4. Record a natural two-speaker clip so the #133 label half can be closed.

**Not done / caveats:** hymn #193 packet deliberately skipped; no human-quality, visible-display, live-microphone or x86/RTX 2070 evidence was produced; all latency numbers here are engineering screens on machine-timed replays with small endpoint counts (no p95 claims).
- 03:40Z GHCR rebuild 34558272252 failed: `Forced include not found: /tmp/build/training/__init__.py`. The Dockerfile build context also lacked `stark_translate/`; second fix PR opened with auto-merge, rebuild to be re-dispatched after merge.
- 04:36Z GHCR rebuild 34559651143 (main `7aac2fd`, tag input `v2026.14.0.0`, push=true) succeeded: ghcr.io/wrbell/stark-translate:latest ghcr.io/wrbell/stark-translate:main ghcr.io/wrbell/stark-translate:v2026.14.0.0. All planned work complete; status PR #198 set to auto-merge.
