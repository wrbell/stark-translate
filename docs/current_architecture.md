# Current architecture — v2026.14 source

> **Scope:** Inference and operator contracts for v2026.14 source (`2026.14.0.0`).
> [PR #192](https://github.com/wrbell/stark-translate/pull/192) records its integration history and current PR state.
> The last published release recorded here is v2026.13. These contracts do not imply
> package publication or service certification. Recorded evidence and remaining
> validation are in [`overnight_status.md`](./overnight_status.md).
>
> **Live microphone (2026-09-09 → 10):** built-in microphone session
> `20260909_233204_799019_en` stalled after model load — no audio frames, lifecycle
> stuck at `running`, operator showed RUNNING from the CSV header, audience stayed
> disconnected; a standalone `sounddevice` record probe stalled too. File-replay
> sessions on the same build passed. The fix is **implemented** (isolated capture with
> no-input timeouts, health-derived readiness — § Audio capture and § Operator control
> plane). [September 10 checks](evaluation/attended_mic_20260910/README.md)
> passed real EN/ES capture/readiness and stop plus EN pause/resume after permission
> was granted. The room was quiet: spoken-microphone and physical-device gates
> remain pending (`mac-live-mic-stall`, `issue-131-smoke` in [`backlog.json`](./backlog.json)).

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

**Audio capture (reliability, integrated 2026-09-10):** the live microphone is opened by a
disposable PortAudio child (`tools/capture_worker.py`) behind
`tools/isolated_audio.IsolatedInputStream`; the parent process never touches the native
device. No samples within `startup_timeout` (5 s) or an idle gap over `idle_timeout` (3 s)
raise `AudioCaptureError` and fail the session instead of hanging. Callback → asyncio handoff
is bounded (`tools/capture_handoff.py`, overflow counted as a capture failure). File replay
(`STARK_AUDIO_SOURCE=file`) and the Docker bridge (`STARK_AUDIO_SOURCE=ws`) use the same
loop. Pause closes the capture child; buffered speech ≥ 0.7 s is finalized before pausing.

Sub-minimum utterances expire at their silence endpoint. Discarding them resets
audio and sample timing together, so later speech cannot inherit an old noise
frame across an unbuffered gap. Scoped `utterance_discarded` events remove only
the abandoned preview; pending STT/translation and queued delivery cannot repaint
it. Music hold and capture-error recovery share the same invalidation contract.

Authoritative final publication also closes its explicit session/utterance against
late previews, through the producer, delivery queue, health inventory and displays.
An older final preserves the next utterance's preview. Ordinary previews remain
eligible while a final computes; experimental closure at final admission stays
opt-in. Preview text and timing join final diagnostics by capture utterance identity,
not by the independent final chunk counter.

**Health/control channel:** `tools/pipeline_health.py` writes a low-rate snapshot (`phase`
∈ `loading, listening, ready, paused, input_error, …`, input/caption ages, error counts,
last captions) and accepts pause/resume/stop control; readers mark it `stale` after 3 s.
Deployment profiles (`stark_translate/profiles.py`): `standard` (default) keeps the
selection above; `lite-cpu`, `lite-cpu-quality`, `lite-cuda-8gb` replace it with pinned
faster-whisper CT2 + ONNX Silero + Marian CT2 and Marian or Gemma 4 E2B finals through a
session-owned `llama-server` ([`lite_profiles.md`](./lite_profiles.md)).

**CUDA training box (separate):** faster-whisper W16 CT2 + Marian CT2 + Gemma 4
E4B via llama.cpp. See [`CLAUDE-windows.md`](../CLAUDE-windows.md) and
[`docs/cuda_latency_proposal.md`](./cuda_latency_proposal.md).

## Measurement schema (schema 2)

| Field | Meaning | Use |
|-------|---------|-----|
| `speech_end_to_final_ms` | Estimated speech end → final payload ready | Primary pipeline latency |
| `speech_end_to_ack_upper_bound_ms` | Estimated speech end → visible browser's acknowledgement (render + return network included); visible tabs only | Upper bound on delivery |
| `send_to_ack_ms` / `receive_to_render_ms` | Server send → ACK; browser receive → render opportunity (client-reported) | Diagnostics, not the delivery gate |
| `e2e_latency_ms` (legacy) | Processing after submission | **Archived only** — not speech-end-to-display |

Historical v2026.13 tables in [`docs/archive/v2026.13/MAC_LATENCY.md`](./archive/v2026.13/MAC_LATENCY.md)
retain their original definitions. The sub-second median caption-delivery goal is
**not achieved** in current measurements.

## Operator control plane

- FastAPI + vanilla JS at `http://127.0.0.1:9000/operator/` (`operator_app/main.py`); page
  organized for lay volunteers (start/stop, mic and voice choice, audience link + QR,
  health list, caption preview, troubleshooting, support export)
- Pre-flight gates Start (`/api/preflight`, profile-aware via `operator_app/lite_preflight.py`);
  capability gating (`/api/capabilities`); idle-only device probes
  (`/api/audio/test-input|test-output`, native calls in disposable processes)
- Mid-session pause/resume/lang_flip/vad/fallback over the shared cooperative control channel
- Readiness comes from the pipeline health channel: `/api/session/status` reports
  `phase`, `ready` (only when health says `ready` and is not stale) and `stale`; RUNNING is
  **no longer** inferred from the CSV header. Real built-in-mic capture/readiness
  passed for EN and ES; spoken-microphone caption accuracy remains pending
- One model/audio job per operator (`operator_app/work_lease.py`); cleanup touches only owned
  subprocesses, including detached children (`operator_app/processes.py`); explicit runtime
  choices (e.g. TTS off) are preserved for reproducible runs
- Live Review with independent transcript/translation approval and export guards
  (`/api/review/...`); scoped support bundles and regenerable-log cleanup
  (`/api/support/...`, `/api/storage/...`, `operator_app/support.py`)
- Operator HTTP/WebSocket access defaults to localhost, with Host and browser Origin
  validation. Explicit remote binding warns that controls and private session data are
  unauthenticated; it is not a public hosting configuration. Audience HTTP serves only
  eight required display assets, independently of the operator (`tools/display_server.py`).
- Review exports use a separate approved projection: private drafts/notes and unapproved
  target text never enter bundles. Schema 2 export IDs bind that projection; cached files
  and download members are validated, and older unsafe downloads require re-export.
- Session lifecycle: explicit completion after worker/drain on SIGINT/SIGTERM;
  generation-checked controls cannot stop a newer session after waiting on an old one.
- Day-of workflow: [`operator_runbook.md`](./operator_runbook.md) (recorded browser evidence)

## Model resolution and setup

- `stark-translate setup --backend mlx` — pinned manifest, managed Marian CT2 cache
- Reuses complete local CT2 adapters or converts both HF directions atomically
- Default Piper voices match EN/ES profile; VAD loads bundled Silero weights
- MLX Whisper resolves the selected primary/fallback settings through the shared resolver.
  Automatic Distil fallback is English-only; Spanish load failures surface visibly, and
  low-confidence Spanish output is never retried on an English-only model.
- Do **not** recreate the operator's working `stt_env`

Details: [`mac_implementation_status.md`](./mac_implementation_status.md),
[`packaging/macos.md`](./packaging/macos.md) (installation paths and artifact checks).

## Opt-in experiments (not defaults)

| Experiment | Status | Notes |
|------------|--------|-------|
| Gemma 4 MTP / `--mts` | Off (#177); live `--mts` **rejected before load** by `validate_live_mts` | Offline probe only: byte-identical; ≤14% win at low acceptance |
| Latency experiments (`tools/latency_experiments.py`) | Opt-in, validated before startup | Provisional previews, fixed-prefix cache, bounded allocator, pause speculation — evidence via `tools/overnight_bench.py` (frozen source and recorded browser telemetry) |
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

- **Mac MLX** — primary inference path documented here (`standard` profile)
- **Lite CPU inference** — **implemented**: `lite-cpu` (Whisper small CT2 int8, Marian CT2
  finals) and `lite-cpu-quality` (adds Gemma 4 E2B Q4_K_M via CPU `llama-server`), Torch-free
  `lite-cpu` extra, `stark-translate-lite` entry point, lite preflight admission floors,
  pinned artifacts in `models.lock.json`. Evidence so far: isolated Mac CPU install +
  synthetic EN/ES caption/TTS replays and an installed CPU E2B inference smoke
  ([`lite_profiles.md`](./lite_profiles.md)). x86 CPU performance, natural-speech quality
  and any latency gate: **pending**
- **Native Windows / RTX 2070** — `lite-cuda-8gb` implemented (Whisper turbo CT2
  int8_float16 + Marian + E2B via CUDA `llama-server`, pinned Windows CUDA 12.4 archives);
  MSI is a scaffold plan; **nothing has run on a 2070 or native Windows yet**

## CI/CD

10 GitHub Actions workflow files in `.github/workflows/`: Lint, Test (coverage gate),
Security, Release, Windows MSI Release, PyPI Publish (deferred by decision; publish job gated), Docker
Image (GHCR), Label PRs, Commitlint, Stale. `tests/test_documentation.py` checks this
count against the filesystem, so update the guides when workflows change.

## Related documents

| Doc | Role |
|-----|------|
| [`backlog.json`](./backlog.json) | Machine-readable remaining work |
| [`overnight_status.md`](./overnight_status.md) | Overnight documentation deliverables and unfinished areas |
| [`lite_profiles.md`](./lite_profiles.md) | Lite profile contract, pinned artifacts, CPU smoke evidence |
| [`evaluation/README.md`](./evaluation/README.md) | Manifests, cohort boundaries |
| [`roadmap.md`](./roadmap.md) | Long-range phases and archived metrics |
