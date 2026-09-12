# CLAUDE-macbook.md — Mac Inference Environment Guide

> **Machine:** MacBook Pro M3 Pro (Mac15,6), 18 GB unified memory, 12-core CPU, 18-core GPU,
> macOS 26 (`platform` recorded per session in `metrics/hardware_<session>.json`).
> **Role:** inference, operator UI, browser displays, Mac evaluation and screens. Training happens
> on WSL ([`CLAUDE-windows.md`](CLAUDE-windows.md)). Parent: [`CLAUDE.md`](CLAUDE.md) (standing
> constraints), agent guide [`AGENTS.md`](AGENTS.md). Evidence:
> [`docs/mac_implementation_status.md`](docs/mac_implementation_status.md); measured numbers with
> their sources: [`README.md`](README.md) § Measured performance.

---

## Current defaults (from `settings.py`, `engines/factory.py`, `dry_run_ab.py`)

| Stage | Default | Override |
|-------|---------|----------|
| VAD | Packaged Silero 6.2.1 (torch), threshold 0.3, silence 0.5 s, max utterance 8 s | `--vad-threshold`, `--silence-trigger`, `--vad-backend onnx` |
| Partial STT (every 0.6 s of speech) | `--lang en`: Parakeet TDT v3 MLX · `--lang es`: mlx-whisper large-v3-turbo | `--partial-interval`, `--stt-backend mlx` (force Whisper for EN) |
| Partial translation | Marian CT2 int8 on CPU (4 threads) from `adapters/marian_ct2/<dir>/active` or the managed setup cache; HF PyTorch fallback | `STARK_TRANSLATE_MARIAN_BACKEND=hf\|ct2` |
| Final STT | Same engine as partials; Whisper automatic fallback is English-only. Spanish keeps multilingual Turbo and never retries English-only Distil | `--word-timestamps` (off by default); confidence thresholds unchanged |
| Final translation | Gemma 4 E4B OptiQ (`mlx-community/gemma-4-e4b-it-OptiQ-4bit`); short high-confidence finals may take the Marian route | `--gemma4-size e2b`; `--model-family translategemma [--ab]`; `--routing-policy conservative` |
| TTS (off) | Piper `en_US-lessac-high` / `es_MX-claude-high`; `--tts-output ws\|wav\|both\|local` | `--tts` / explicit `--no-tts`, `--tts-device-en/--tts-device-es` |
| Runtime fixes on by default (series 4, #217) | Keep-warm after each final (`STARK_WARMUP_AFTER_FINAL`), first stream token at 1 (`STARK_STREAM_FIRST_TOKEN_BATCH_SIZE`), hash-pinned Parakeet joint decode (`STARK_PARAKEET_JOINT_EVAL`) | Set to `0` to disable; the Metal wired limit (`STARK_MLX_WIRED_LIMIT`) is opt-in and was rejected as a default for peak RSS |
| Experiments (all off) | `--idle-warmup-only`, `--final-aware-partials`, `--terminology-prompt church`, and the `STARK_EXPERIMENT_*` controls in `tools/latency_experiments.py` (validated before startup) | Keep off. Every screened arm is closed: [registry](docs/latency_next_experiments.md) |
| MTP drafter (`--mts`, #177) | **Rejected before any model loads** (`validate_live_mts`); `--no-mts` is the explicit off switch | Offline experiment only ([`docs/mlx_mtp_notes.md`](docs/mlx_mtp_notes.md)) |
| Profile | `standard`. `stark-translate-lite` / `--profile lite-cpu` runs the CPU Lite profile on this Mac for implementation testing only; it certifies neither an x86 CPU nor a 2070 | [`docs/lite_profiles.md`](docs/lite_profiles.md) |

Policy: fast revisable partials, careful finals. Sub-second median speech-end-to-caption is the
goal, **not achieved** (`caption-delivery-goal` in the backlog).

---

## Environment

### Verified packages (2026-09-11, promoted `venv`)

Python 3.11.11 · MLX 0.32.2 · mlx-lm 0.31.3 · mlx-whisper 0.4.3 · mlx-optiq 0.4.34 ·
parakeet-mlx 0.5.x · PyTorch 2.13.0 (Silero VAD, Marian HF fallback only) · TorchAudio 2.11.0 ·
CTranslate2 4.7.1 (Marian int8 on CPU) · silero-vad 6.2.1. Pins live under
`[project.optional-dependencies]` `mlx` in `pyproject.toml` plus
`constraints/macos-arm64-py311-runtime.txt`; upgrading the MLX/OptiQ/Parakeet or PyTorch lines
requires another replay gate. `venv` is runtime-only; run checks with `stt_env/bin/python -m
pytest|ruff|mypy`. `stt_env` (Torch 2.10) is the rollback environment and is never modified.

### Installation (checkout or Mac source ZIP)

Authoritative steps, models, launchd and rollback: [`docs/packaging/macos.md`](docs/packaging/macos.md).

```bash
python3.11 -m venv venv && venv/bin/python -m pip install --upgrade 'pip>=26.2' 'setuptools>=83.0.0'
venv/bin/python -m pip install -c constraints/macos-arm64-py311-runtime.txt '.[mlx]'   # add ,eval or ,diarization as needed
venv/bin/stark-translate setup --backend mlx [--include e2b tts translategemma]
venv/bin/stark-translate doctor --backend mlx --lang en    # and --lang es; no Metal models loaded
printf '%s\n' "$PWD/venv/bin/python" > .stark-python       # launcher pointer; rollback: stt_env/bin/python
./run_operator.sh                                          # operator UI on http://localhost:9000/operator/
```

`bootstrap.sh --skip-systemd` performs install + setup + preflight in one step and refuses to
install into `stt_env`; `stark-translate launchd render|install|uninstall` manages the optional login
service; `scripts/runtime_env.sh` resolves the interpreter (`STARK_PYTHON` > `VENV` > `VIRTUAL_ENV` >
`CONDA_PREFIX` > `.stark-python`). Hugging Face login is only needed for gated repos (pyannote).
Grant microphone access under **System Settings → Privacy & Security → Microphone** for the
terminal or launcher that runs the pipeline.

### Models (`models.lock.json`)

`setup --backend mlx` fetches only the Mac defaults for both directions: Parakeet TDT 0.6B v3,
whisper-large-v3-turbo, Gemma 4 E4B OptiQ, Marian opus-mt en-es / es-en sources plus derived int8
CT2 artifacts. Optional `--include` profiles add Gemma 4 E2B OptiQ, Piper EN/ES voices and
TranslateGemma 4B. Resolution order for setup and inference is shared (`engines/model_paths.py`):
explicit path → `--models-dir` / `STARK_MODELS_DIR` → project `models/` → Hugging Face cache; pinned
snapshots already in the HF cache are reused. Existing `adapters/marian_ct2/*/active` directories
are used first and never modified; the manual converter is `scripts/convert_marian_ct2.py
--quantization int8`.

### Memory

Peak usage is recorded per session in `metrics/session_lifecycle_<id>.json`
(`memory.peak_rss_bytes`, `memory.peak_metal_bytes`); read that file rather than a static table.
TranslateGemma A/B (`--ab`) loads two models and is the memory-heavy configuration. The Metal cache
limit is 256 MB per engine (`cache_limit_mb`), overridden process-wide by
`STARK_EXPERIMENT_MLX_CACHE_MB`. Screens compare peak RSS and Metal against the control with the
relative rule in `tools/tail_screen_report.py` (10 % + 256 MiB), which is what rejected the wired
limit as a default.

### Runtime audit

`scripts/audit_mac_runtime.sh --output "metrics/runtime-audit-$(date +%Y%m%d-%H%M%S)"` (monthly;
needs pip-audit) and `python tools/check_dependency_audit.py --runtime mac` fail closed on an
incomplete report.

---

## Running

**Operator UI (recommended):** `./run_operator.sh` → `http://localhost:9000/operator/`.
Start/Stop/Pause, language flip, VAD threshold, fallback toggle, preflight, device lists, verse
highlights, summary, Review/export. Session artifacts land in `metrics/`.

**Direct CLI (debugging / replay):**

```bash
source venv/bin/activate  # stt_env = rollback env
python dry_run_ab.py                                   # EN→ES, mic, Mac defaults (attended only)
python dry_run_ab.py --lang es                         # ES→EN
python dry_run_ab.py --audio-file clip.wav --session-id demo_en   # file replay, exits after drain
python dry_run_ab.py --dry-run-text "For God so loved the world"  # no mic
python dry_run_ab.py --gemma4-size e2b                 # opt-in model; compare latency and quality
python dry_run_ab.py --tts --tts-output local --tts-device-en "MacBook Pro Speakers" --tts-device-es "BlackHole 2ch"
python dry_run_ab.py --diarize --diarize-mode embed    # live speaker labels; natural two-speaker clip still pending
```

Displays: `http://localhost:8080/displays/audience_display.html` (projector), `ab_display.html`
(operator comparison), `mobile_display.html` (phones via QR). Ports: 8080 HTTP, 8765 captions,
8766 TTS audio, 9000 operator. Protocol and timing semantics: [`displays/CLAUDE.md`](displays/CLAUDE.md).

### Capture and readiness

PortAudio runs in a disposable child (`tools/isolated_audio.py`, `tools/capture_worker.py`); a
5 s startup with no samples or a 3 s idle gap raises `AudioCaptureError` and fails the session
instead of hanging. The operator derives readiness from `tools/pipeline_health.py` phases
(`loading → listening → ready`, `paused`, `input_error`, `stale`), never from a CSV header or a
process. Exact microphone name and host API cross preflight, restart and capture
(`tools/input_devices.py`). Quiet-room sessions reached ready; the synthetic Spanish check retained
upstream sample loss, so sustained live acceptance (#131) is open:
[quiet-room receipts](docs/evaluation/attended_mic_20260910/README.md),
[synthetic checks](docs/evaluation/tts_routing_20260910/README.md),
[capture-loss accounting](docs/evaluation/mac_followup_20260910/capture-loss-accounting.md).
Automation never opens the microphone; live checks are attended.

---

## Pipeline notes

- **Overlap:** STT(N+1) runs concurrently with translation(N) on a 2-worker pool (thread-local
  MLX streams; pinned mlx 0.32.2). Weights and the first Gemma forward are materialized on the
  load thread (`warm_mlx_model`). `--multiprocess` is an escape hatch that shares the same
  prompt/stop helpers (#176).
- **Stop tokens:** `ensure_stop_tokens()` adds the family's turn terminator and preserves loader
  EOS ids; Gemma 4 uses `<turn|>`, TranslateGemma `<end_of_turn>`. Never hand-edit
  `_eos_token_ids`. Details: [`engines/CLAUDE.md`](engines/CLAUDE.md).
- **Marian/VAD PyTorch:** share `_pytorch_lock`; VAD stays on the asyncio thread.
- **Confidence flagging:** English Whisper finals can retry with the fallback model when
  `avg_logprob < -1.2` or `compression_ratio > 2.4`; automatic fallback is disabled for Spanish.
  Words with probability `< 0.5` are listed as low-confidence; fallback events go to the
  active-learning JSONL. Parakeet confidence is a TDT proxy and does not use this chain.
- **Music hold:** `--music-threshold` / `--music-holdoff` configure an energy/VAD heuristic that can
  hold new STT and emit `music_hold`. It can miss singing: for attended live use, **Pause** before
  congregational singing and **Resume** before spoken prayer or preaching
  ([runbook](docs/operator_runbook.md)); automatic singing suppression is not validated (#193).
- **Timing:** schema 2 `speech_end_to_final_ms` (server), `speech_end_to_ack_upper_bound_ms`
  (visible browser, includes return network), first token
  (`translation_started + ttft − speech_end`) with the `first_stream` browser ACK; legacy
  `e2e_latency_ms` is processing time. Definitions:
  [`docs/evaluation/README.md`](docs/evaluation/README.md).

---

## Adapters on the Mac

- **Gemma (MLX):** `--adapter-dir DIR` (primary) and `--adapter-dir-b DIR` (12B in `--ab`) pass
  `adapter_path=` to `mlx_lm.load`. Gate before use: `python tools/health_check.py --backend mlx
  --adapter DIR` (8 canaries by default).
- **Whisper LoRA (W16/W17):** mlx-whisper and Parakeet do not load LoRA. The exported CT2 model runs
  through `FasterWhisperEngine` on CPU (`--backend cpu --stt-backend faster-whisper`) for A/B
  (#135); PyTorch/PEFT loading is for offline evaluation only.
- **Registry:** `tools/manage_adapters.py register/activate/rollback` (`active` / `previous`
  slots), [`docs/deploy.md`](docs/deploy.md).
- **Transfer:** copy from WSL by scp/USB into `adapters/`; provenance in
  [`training/CLAUDE.md`](training/CLAUDE.md).

---

## Evaluation and screens on the Mac (no WSL required)

| Task | Tool | Notes |
|------|------|-------|
| Frozen quality/latency comparison | `tools/mac_evaluation.py` | Manifests and reports: [`docs/evaluation/README.md`](docs/evaluation/README.md); one model process at a time |
| Replay screen on real audio | `python -m tools.replay_bench --manifest <runs.json> --tag <tag> --configs "arm=<argv>"` | Sequential `dry_run_ab.py --audio-file` runs; inherits `STARK_*`; writes `metrics/` under the current checkout (two-checkout screens use per-checkout `metrics/` and distinct tags) |
| Screen gates from recorded artifacts | `tools/tail_screen_report.py --runs <runs.jsonl> --output <report.json> [--markdown …]` | G1 median, G2 tails incl. cuts, G3, G4 sequence-aligned identity, G5 previews, G6 memory (relative RSS/Metal), G7 lifecycle; optional `display_metrics_jsonl` → `first_visible` |
| Stage attribution | `tools/silence_final_stages.py`, `tools/stt_overlap_attribution.py` | Where the time goes on silence finals; no inference |
| Endurance | `tools/endurance_monitor.py --pid <pipeline pid> --session <id> …` | Read-only; the pid comes from `/api/session/status` |
| Synthetic STT gate | `tools/stt_roundtrip_compare.py` | Piper → STT → WER + term recall; not natural audio |
| Adapter gate | `tools/health_check.py --backend mlx` | 8 of 18 canaries |
| YouTube caption comparison | `tools/live_caption_monitor.py` | Cross-system WER = disagreement |
| Translation QE | `tools/translation_qe.py` | Tier 1 heuristics, Tier 2 back-translation, Tier 3 LaBSE |
| Blinded bilingual packet | `tools/mac_bilingual_review.py` | HMAC-seeded labels; private key stays out of `docs/` |

Declare the protocol before run 1, keep one inference process on the GPU, never re-run a rejected
arm ([registry](docs/latency_next_experiments.md)), and write the evidence README with source commit,
commands, hashes, result and a "does not certify" sentence. Open Mac gates (church Spanish
references, blinded bilingual review, physical display timing, two-speaker diarization, second
physical output) are in [`docs/backlog.json`](docs/backlog.json).

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Session fails with "Microphone delivered no samples" or health shows `input_error` | Isolated capture timed out. Check microphone permission for the launching app, the operator's idle device probe, and `metrics/session_<id>.log`; run `--audio-file` to confirm the rest of the pipeline |
| Operator shows RUNNING with no partials | RUNNING means frames are arriving; silence or filtered input can produce no captions. Inspect the input level, frame/heartbeat health and session log. Stale input or an `input_error` indicates capture failure; no-caption silence alone does not |
| Preflight fails on models | `stark-translate setup --backend mlx` (add `--include ...`); set `STARK_MODELS_DIR` at launch if setup used `--models-dir` |
| Marian preflight fails | Setup needs a complete CT2 artifact for the selected direction; rerun setup or `scripts/convert_marian_ct2.py` |
| Gemma output truncated or runs to `max_tokens` | Stop-token regression (#172) — verify `ensure_stop_tokens` logs "added=" for the family; never hand-edit `_eos_token_ids` |
| Uniform 4-bit Gemma 4 quant produces garbage | Only OptiQ repos are supported (PLE layers) |
| First Gemma forward crashes off the load thread | `warm_mlx_model` must run on the load thread; see `tests_gpu/test_mlx_worker_first_forward.py` |
| Peak RSS jumps by gigabytes with unchanged speed | `STARK_MLX_WIRED_LIMIT` is set; leave it unset (mlx-lm already sets the limit per call) |
| Parakeet loads with "stock decode" in the log | The pinned joint-decode source hash drifted after an upgrade; re-qualify with `tools/parakeet_joint_eval.py` before re-pinning |
| Metal cache growth | `mx.set_cache_limit(256 MB)` is set per engine; avoid `--word-timestamps` in live sessions |
| libomp / duplicate OpenMP crash | [`docs/archive/troubleshooting/macos_libomp_fix.md`](docs/archive/troubleshooting/macos_libomp_fix.md) |
| PyTorch fp16 on MPS inf/nan, bitsandbytes on Mac | Expected — MLX quantized models are the Mac path |
| `faster-whisper` on Mac | CTranslate2 runs on CPU only here; fine for W16 A/B and confidence checks |
| Phone can't connect | Same LAN; ports 8080/8765 open; scan the QR on the audience display |
| Battery throttling | Plug in for measurements |
| `pytest`/`ruff` missing in `venv` | Expected; use `stt_env/bin/python -m …` |
