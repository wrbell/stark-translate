# CLAUDE-macbook.md — Mac Inference Environment Guide

> **Machine:** MacBook Pro M3 Pro (Mac15,6), 18 GB unified memory, 12-core CPU, 18-core GPU,
> macOS 26 (`platform` recorded per session in `metrics/hardware_<session>.json`).
> **Role:** inference, operator UI, browser displays, Mac evaluation. Training happens on
> WSL ([`CLAUDE-windows.md`](./CLAUDE-windows.md)). Parent: [`CLAUDE.md`](./CLAUDE.md).
>
> **Source (2026-09-10):** v2026.14 (`2026.14.0.0`), tracked by [PR #192](https://github.com/wrbell/stark-translate/pull/192).
> The last published release recorded here is v2026.13. Source integration and
> published artifacts are separate from the acceptance evidence below.
> Evidence: [`docs/mac_implementation_status.md`](docs/mac_implementation_status.md);
> contracts: [`docs/current_architecture.md`](docs/current_architecture.md); remaining work:
> [`docs/backlog.json`](docs/backlog.json). **Do not recreate `stt_env`.** Validation counts
> and latency numbers live only in the linked evidence documents.

---

## Current defaults (from `settings.py`, `engines/factory.py`, `dry_run_ab.py`)

| Stage | Default | Override |
|-------|---------|----------|
| VAD | Packaged Silero 6.2.1 (torch), threshold 0.3, silence 0.5 s, max utterance 8 s | `--vad-threshold`, `--silence-trigger`, `--vad-backend onnx` |
| Partial STT (every 0.6 s of speech) | `--lang en`: Parakeet TDT v3 MLX · `--lang es`: mlx-whisper large-v3-turbo | `--partial-interval`, `--stt-backend mlx` (force Whisper for EN) |
| Partial translation | Marian CT2 int8 on CPU (4 threads) from `adapters/marian_ct2/<dir>/active` or the managed setup cache; HF PyTorch fallback | `STARK_TRANSLATE_MARIAN_BACKEND=hf\|ct2` |
| Final STT | Same engine as partials; Whisper automatic fallback is English-only. Spanish keeps multilingual Turbo and never retries English-only Distil | `--word-timestamps` (off by default); confidence thresholds unchanged |
| Final translation | Gemma 4 E4B OptiQ (`mlx-community/gemma-4-e4b-it-OptiQ-4bit`) | `--gemma4-size e2b`; `--model-family translategemma [--ab]` |
| TTS (off) | Piper `en_US-lessac-high` / `es_MX-claude-high`; `--tts-output ws\|wav\|both\|local` | `--tts` / explicit `--no-tts` (`c5fb689`), `--tts-device-en/--tts-device-es` |
| Experiments (all off) | `--idle-warmup-only`, `--final-aware-partials`, `--routing-policy conservative`, `--terminology-prompt church`, latency experiments in `tools/latency_experiments.py` (opt-in, validated before startup) | Keep off; the [completed 96-run screen](docs/evaluation/overnight_screen_20260910/README.md) selected no experimental arms |
| MTP drafter (`--mts`, #177) | **Rejected before any model loads** (`validate_live_mts`: "Live --mts is unavailable"); `--no-mts` is the explicit off switch | Offline experiment only ([`docs/mlx_mtp_notes.md`](docs/mlx_mtp_notes.md)) |
| Profile | `standard` (default). `stark-translate-lite` / `--profile lite-cpu` runs the CPU Lite profile on this Mac for implementation testing only — it certifies neither an x86 CPU nor a 2070 | [`docs/lite_profiles.md`](docs/lite_profiles.md) |

Policy: fast revisable partials, careful finals. Sub-second median speech-end-to-caption
is the goal, **not achieved**, and is active engineering (`caption-delivery-goal`).

---

## Environment

### Verified packages (2026-09-09, `stt_env`)

Python 3.11.11 · MLX 0.32.2 · mlx-lm 0.31.3 · mlx-whisper 0.4.3 · mlx-optiq 0.4.34 ·
parakeet-mlx 0.5.x · PyTorch 2.10.0 (Silero VAD, Marian HF fallback only) ·
CTranslate2 4.7.1 (Marian int8 on CPU) · silero-vad 6.2.1. Pins live in
`pyproject.toml` `[mlx]`; upgrading the MLX/OptiQ/Parakeet or PyTorch lines requires
another replay gate.

### Installation (checkout or Mac source ZIP)

Authoritative steps: [`docs/packaging/macos.md`](docs/packaging/macos.md).

```bash
brew install ffmpeg portaudio
python3.11 -m venv venv                      # or keep using the existing stt_env
venv/bin/python -m pip install '.[mlx]'      # add ,eval or ,diarization as needed
venv/bin/stark-translate setup --backend mlx                # Mac defaults, both directions
venv/bin/stark-translate setup --backend mlx --include e2b tts translategemma   # optional profiles
venv/bin/stark-translate doctor --backend mlx --lang en     # preflight without loading Metal models
venv/bin/stark-translate doctor --backend mlx --lang es
VENV="$PWD/venv" ./run_operator.sh           # operator UI on http://localhost:9000/operator/
```

`bootstrap.sh --skip-systemd` performs install + setup + preflight in one step;
`stark-translate launchd render|install|uninstall` manages the optional login service.
`requirements-mac.txt` is **deprecated** (v2026.7) and kept only as a pip-cache key.
Hugging Face login is only needed for gated repos (pyannote); the default Mac models are
public. Grant microphone access under **System Settings → Privacy & Security → Microphone**
for the terminal or launcher that runs the pipeline.

### Models (`models.lock.json`)

`setup --backend mlx` fetches only the Mac defaults for both directions: Parakeet TDT
0.6B v3, whisper-large-v3-turbo, Gemma 4 E4B OptiQ, Marian opus-mt en-es / es-en sources
plus derived int8 CT2 artifacts. Optional `--include` profiles add Gemma 4 E2B OptiQ,
Piper EN/ES voices, and TranslateGemma 4B. Resolution order for setup and inference is
shared (`engines/model_paths.py`): explicit path → `--models-dir` / `STARK_MODELS_DIR` →
project `models/` → Hugging Face cache; pinned snapshots already in the HF cache are
reused. Existing `adapters/marian_ct2/*/active` directories are used first and never
modified; the manual converter remains `scripts/convert_marian_ct2.py --quantization int8`.

### Memory

Peak usage is recorded per session in `metrics/session_lifecycle_<id>.json`
(`memory.peak_rss_bytes`, `memory.peak_metal_bytes`); the 2026-09-09 ES file-replay session
`20260909_233823_034893_es` (E4B, Whisper turbo, Marian CT2) recorded roughly 4.2 GB RSS
and 9.1 GB peak Metal. TranslateGemma A/B (`--ab`) loads two models and is the
memory-heavy configuration; check the lifecycle file rather than a static table. Metal
cache limit is 256 MB per engine (`cache_limit_mb`).

---

## Running

**Operator UI (recommended):** `./run_operator.sh` → `http://localhost:9000/operator/`.
Start/Stop/Pause, language flip, VAD threshold, fallback toggle, preflight, device lists,
verse highlights, summary, Review/export. Session artifacts land in `metrics/`.

**Direct CLI (debugging / replay):**

```bash
source stt_env/bin/activate            # or venv
python dry_run_ab.py                                   # EN→ES, mic, Mac defaults
python dry_run_ab.py --lang es                         # ES→EN
python dry_run_ab.py --audio-file clip.wav --session-id demo_en   # file replay, exits after drain
python dry_run_ab.py --dry-run-text "For God so loved the world"  # no mic
python dry_run_ab.py --gemma4-size e2b                 # separately evaluated fast finals
python dry_run_ab.py --tts --tts-output local --tts-device-en "MacBook Pro Speakers" --tts-device-es "BlackHole 2ch"
python dry_run_ab.py --diarize --diarize-mode embed    # live speaker labels (gate not run, #133)
```

Displays: `http://localhost:8080/displays/audience_display.html` (projector),
`ab_display.html` (operator comparison), `mobile_display.html` (phones via QR).
Ports: 8080 HTTP, 8765 captions, 9000 operator. Protocol and timing semantics:
[`displays/CLAUDE.md`](displays/CLAUDE.md).

### Built-in microphone stall (2026-09-09) — fix implemented, live retest deferred

Session `20260909_233204_799019_en` (`audio_source: mic`) loaded all models, printed
"Listening...", served the audience page, then received no audio frames; its lifecycle
file stayed `status: running`, the operator showed RUNNING because the CSV header existed,
and the audience display stayed disconnected. A standalone `sounddevice` record probe
stalled too. File-replay sessions on the same build passed.

What changed (integrated on the candidate branch, `c5fb689`):

- **Isolated capture:** `tools/isolated_audio.py` `IsolatedInputStream` runs PortAudio in a
  disposable child (`tools/capture_worker.py`); the parent never opens the native device.
  A **5 s startup timeout** with no samples, or a **3 s idle gap**, raises
  `AudioCaptureError("Microphone delivered no samples…")` and fails the session instead of
  hanging. `tools/capture_handoff.py` bounds the callback → asyncio handoff.
- **Readiness/health:** `tools/pipeline_health.py` publishes `loading → listening → ready`
  (first input frame), `paused`, `input_error`, with staleness; the operator
  (`operator_app/pipeline_manager.py`) derives `ready` from this channel, not from the CSV
  header. `operator_app/audio_tests.py` runs idle-only device probes in disposable processes.
- **Ownership:** `operator_app/processes.py` cleans only owned subprocesses;
  `operator_app/work_lease.py` allows one model/audio job per operator.

**Not yet proven:** a real built-in-microphone session on this Mac (deferred to the next attended session),
physical second output, hotplug. Until then #131 stays `in_progress`.

---

## Pipeline notes

- **Overlap:** STT(N+1) runs concurrently with translation(N) on a 2-worker pool (mlx ≥ 0.31.2 thread-local streams). Weights and the first Gemma forward are materialized on the load thread (`warm_mlx_model`). `--multiprocess` is an escape hatch that now shares the same prompt/stop helpers (#176).
- **Stop tokens:** `ensure_stop_tokens()` adds the family's turn terminator and preserves loader EOS ids; Gemma 4 uses `<turn|>`, TranslateGemma `<end_of_turn>`. The old "add id 106 by hand" fix must not be applied to Gemma 4. Details: [`engines/CLAUDE.md`](engines/CLAUDE.md).
- **Marian/VAD PyTorch:** share `_pytorch_lock`; VAD stays on the asyncio thread.
- **Confidence flagging:** English Whisper finals can retry with the fallback model when `avg_logprob < -1.2` or `compression_ratio > 2.4`; automatic fallback is disabled for Spanish so English-only Distil cannot produce Spanish results. Words with probability `< 0.5` are listed as low-confidence; fallback events go to the active-learning JSONL. Parakeet confidence is a TDT proxy and does not use this fallback chain.
- **Music hold:** `--music-threshold` / `--music-holdoff` configure an energy/VAD heuristic: sustained high-energy audio classified as non-speech can hold new STT and emit `music_hold`. It can miss singing; the fresh Standard hour retained hymn-region fragments without recorded hold events. For attended live use, **Pause** before or during congregational singing and **Resume** before spoken prayer/preaching. See the [runbook](docs/operator_runbook.md); automatic singing suppression is not validated.
- **Timing:** schema 2 `speech_end_to_final_ms` (server) and `speech_end_to_ack_upper_bound_ms` (visible browser, includes return network) — legacy `e2e_latency_ms` is processing time. Definitions: [`docs/evaluation/README.md`](docs/evaluation/README.md); measured history: [`docs/archive/v2026.13/MAC_LATENCY.md`](docs/archive/v2026.13/MAC_LATENCY.md).

---

## Adapters on the Mac

- **Gemma (MLX):** `--adapter-dir DIR` (primary) and `--adapter-dir-b DIR` (12B in `--ab`) pass `adapter_path=` to `mlx_lm.load`. Gate before use: `python tools/health_check.py --backend mlx --adapter DIR` (8 canaries by default).
- **Whisper LoRA (W16/W17):** mlx-whisper and Parakeet do not load LoRA. The exported CT2 model runs through `FasterWhisperEngine` on CPU (`--backend cpu --stt-backend faster-whisper`) for A/B (#135); PyTorch/PEFT loading is for offline evaluation only.
- **Registry:** `tools/manage_adapters.py register/activate/rollback` (`active` / `previous` slots), `docs/deploy.md`.
- **Transfer:** copy from WSL by scp/USB into `adapters/`; sizes and training provenance in [`training/CLAUDE.md`](training/CLAUDE.md).

---

## Evaluation on the Mac (no WSL required)

| Task | Tool | Notes |
|------|------|-------|
| Frozen quality/latency comparison | `tools/mac_evaluation.py` | Manifests and reports in [`docs/evaluation/README.md`](docs/evaluation/README.md); one model process at a time |
| Replay matrix on real audio | `tools/replay_bench.py` | Sequential `dry_run_ab.py --audio-file` runs |
| Synthetic STT gate | `tools/stt_roundtrip_compare.py` | Piper → STT → WER + term recall; not natural audio |
| Adapter gate | `tools/health_check.py --backend mlx` | 8 of 18 canaries |
| YouTube caption comparison | `tools/live_caption_monitor.py` | Cross-system WER = disagreement |
| Translation QE | `tools/translation_qe.py` | Tier 1 heuristics, Tier 2 back-translation, Tier 3 LaBSE |

Open Mac gates: natural Spanish references, blinded bilingual review, visible-browser
timing run, two-speaker diarization clip, second physical output, Sunday dry run —
[`docs/backlog.json`](docs/backlog.json).

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Session fails with "Microphone delivered no samples" or health shows `input_error` | Isolated capture timed out (above). Check microphone permission for the launching app, the operator's idle device probe, and `metrics/session_<id>.log`; run `--audio-file` to confirm the rest of the pipeline |
| Operator shows RUNNING with no partials | Should no longer happen (readiness comes from `pipeline_health`); if it does, capture the session id and health snapshot — it is evidence for `mac-live-mic-stall` |
| Preflight fails on models | `stark-translate setup --backend mlx` (add `--include ...`); set `STARK_MODELS_DIR` at launch if setup used `--models-dir` |
| Marian preflight fails | Setup needs a complete CT2 artifact for the selected direction; rerun setup or `scripts/convert_marian_ct2.py` |
| Gemma output truncated or runs to `max_tokens` | Stop-token regression (#172) — verify `ensure_stop_tokens` logs "added=" for the family; never hand-edit `_eos_token_ids` |
| Uniform 4-bit Gemma 4 quant produces garbage | Only OptiQ repos are supported (PLE layers) |
| First Gemma forward crashes off the load thread | `warm_mlx_model` must run on the load thread; see `tests_gpu/test_mlx_worker_first_forward.py` |
| Metal cache growth | `mx.set_cache_limit(256 MB)` is set per engine; avoid `--word-timestamps` in live sessions |
| libomp / duplicate OpenMP crash | [`docs/archive/troubleshooting/macos_libomp_fix.md`](docs/archive/troubleshooting/macos_libomp_fix.md) |
| PyTorch fp16 on MPS inf/nan, bitsandbytes on Mac | Expected — MLX quantized models are the Mac path |
| `faster-whisper` on Mac | CTranslate2 runs on CPU only here; fine for W16 A/B and confidence checks |
| Phone can't connect | Same LAN; ports 8080/8765 open; scan the QR on the audience display |
| Battery throttling | Plug in for measurements |

Earlier versions of this guide (Distil-Whisper model table, TranslateGemma-era latency
budget, hand-applied EOS fix) are preserved in git history; their measurements are
archived under [`docs/archive/`](docs/archive/).
