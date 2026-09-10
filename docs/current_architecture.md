# Current architecture — v2026.14 candidate (local branch)

> **Scope:** Inference and operator contracts on `codex/mac-reliability-roadmap`
> (base `5154fb9`, version `2026.14.0.0`), proposed in draft
> [PR #192](https://github.com/wrbell/stark-translate/pull/192) — open, **not merged**.
> **Main** remains at v2026.13 until the authorized final merge. Do not describe
> local-only behavior as shipped on main/PyPI without integration evidence.
> Overnight worktrees (latency scheduling, lite CPU, operator UI, reliability) are
> **pending integration** and are not part of these contracts yet
> ([`overnight_status.md`](./overnight_status.md)).
>
> **Known open bug (2026-09-09):** built-in microphone session
> `20260909_233204_799019_en` stalled after model load — no audio frames, lifecycle
> stuck at `running`, operator showed RUNNING from the CSV header, audience stayed
> disconnected; a standalone `sounddevice` record probe stalled too. File-replay
> sessions on the same build passed. Live-mic and physical-device checks are deferred
> to tomorrow (`mac-live-mic-stall` in [`backlog.json`](./backlog.json)).

## Two-pass live pipeline

```
Mic 48 kHz → resample 16 kHz → Silero VAD (packaged 6.2.1, no Torch Hub)
        │
        ├─ PARTIAL (every 0.6 s of new speech, while talking)
        │     STT: Parakeet MLX (EN) or mlx-whisper turbo (ES)
        │     Translate: Marian CT2 int8 on CPU (HF fallback)
        │     UI: italic preview — fast, revisable
        │
        └─ FINAL (0.5 s silence or max utterance)
              Same STT path
              Translate: Gemma 4 E4B OptiQ (default) — careful quality
              Optional: --gemma4-size e2b for measured speed tradeoff only
              UI: replaces partial
```

**Threading (Mac):** `ThreadPoolExecutor(max_workers=2)` — STT on utterance N+1
overlaps translation on N. MLX ≥ 0.31.2 thread-local streams; first Gemma
forward and weight materialization on the load thread (`warm_mlx_model`).

**CUDA training box (separate):** faster-whisper W16 CT2 + Marian CT2 + Gemma 4
E4B via llama.cpp. See [`CLAUDE-windows.md`](../CLAUDE-windows.md) and
[`docs/cuda_latency_proposal.md`](./cuda_latency_proposal.md).

## Measurement schema (schema 2)

| Field | Meaning | Use |
|-------|---------|-----|
| `speech_end_to_final_ms` | Estimated speech end → final payload ready | Primary pipeline latency |
| `speech_end_to_ack_upper_bound_ms` | Visible browser render + return network | Upper bound on delivery |
| `e2e_latency_ms` (legacy) | Processing after submission | **Archived only** — not speech-end-to-display |

Historical v2026.13 tables in [`docs/archive/v2026.13/MAC_LATENCY.md`](./archive/v2026.13/MAC_LATENCY.md)
retain their original definitions. The sub-second median caption-delivery goal is
**not achieved** in current measurements.

## Operator control plane

- FastAPI + vanilla JS at `http://host:9000/operator/` (`operator_app/main.py`)
- Pre-flight gates Start; mid-session pause/resume/lang_flip/vad/fallback
- Live Review with independent transcript/translation approval and export guards
- Session lifecycle: explicit completion after worker/drain on SIGINT/SIGTERM
- Known gap: RUNNING is derived from the session CSV header, so a stalled capture
  (tonight's mic bug) is not detected — pipeline-health/capture-worker helpers are
  being drafted in the reliability worktree, pending integration
- Day-of workflow: [`operator_runbook.md`](./operator_runbook.md) (root-owned UI evidence)

## Model resolution and setup

- `stark-translate setup --backend mlx` — pinned manifest, managed Marian CT2 cache
- Reuses complete local CT2 adapters or converts both HF directions atomically
- Default Piper voices match EN/ES profile; VAD loads bundled Silero weights
- Do **not** recreate the operator's working `stt_env`

Details: [`mac_implementation_status.md`](./mac_implementation_status.md),
[`packaging/macos.md`](./packaging/macos.md) (lite agent owns cross-platform packaging prose).

## Opt-in experiments (not defaults)

| Experiment | Status | Notes |
|------------|--------|-------|
| Gemma 4 MTP / `--mts` | Off (#177) | Byte-identical; ≤14% win at low acceptance |
| Conservative Marian routing | Opt-in | 24/24 synthetic routing probes passed |
| Shorter silence / faster cadence | Rejected | 48-run screen — caption/content regressions |
| ONNX VAD | Opt-in | Packaged JIT default passed CPU loads |
| Terminology examples | Opt-in | Quality report in evaluation README |

## Quality layers (summary)

1. **WSL preprocessing** — 10-step audio pipeline ([`training/AGENTS.md`](../training/AGENTS.md))
2. **Data assessment** — WER sampling and strategy
3. **Confidence flagging** — STT thresholds ([`engines/AGENTS.md`](../engines/AGENTS.md))
4. **YouTube comparison** — windowed WER ([`tools/AGENTS.md`](../tools/AGENTS.md))
5. **Translation QE** — CometKiwi/LaBSE tiers
6. **Active learning** — infer → review → merge → retrain

## Deployment targets (equal priority)

Per user decision (2026-09-09 overnight plan):

- **Mac MLX** — primary inference path documented here
- **Lite CPU inference** — implementation in progress in the lite worktree (CPU
  profile, lite preflight, llama runtime helper); certification pending integration
  and hardware
- **Native Windows / RTX 2070** — MSI/Docker workflows exist; validation on the
  2070 pending hardware

## CI/CD

10 GitHub Actions workflow files in `.github/workflows/`: Lint, Test (coverage gate),
Security, Release, Windows MSI Release, PyPI Publish (pending by user choice), Docker
Image (GHCR), Label PRs, Commitlint, Stale. `tests/test_documentation.py` checks this
count against the filesystem, so update the guides when workflows change.

## Related documents

| Doc | Role |
|-----|------|
| [`backlog.json`](./backlog.json) | Machine-readable remaining work |
| [`overnight_status.md`](./overnight_status.md) | Overnight doc worktree status |
| [`evaluation/README.md`](./evaluation/README.md) | Manifests, cohort boundaries |
| [`roadmap.md`](./roadmap.md) | Long-range phases and archived metrics |
