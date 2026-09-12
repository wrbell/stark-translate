# CLAUDE-windows.md — Windows Machines: WSL Training and Native Lite Inference

> **Two different jobs, two different environments — never share a venv between them.**
>
> | Role | Where it runs | Environment | Guide sections |
> |------|---------------|-------------|----------------|
> | **A. Training** (preprocess, Deepgram labels, Whisper LoRA, Gemma 4 QLoRA/CPO, exports, CUDA benchmarks) | WSL2 Ubuntu on the desktop with the **NVIDIA A2000 Ada 16 GB**, 64 GB RAM | `~/stt_train_env` + `requirements-windows.txt` | Part A |
> | **B. Lite inference** (live captions for a church PC) | **Native Windows** (or Linux x86) — RTX 2070 8 GB (`lite-cuda-8gb`) or CPU-only (`lite-cpu`, `lite-cpu-quality`) | `.venv-lite` + `pip install '.[lite-cuda,tts]'` or `'.[lite-cpu,tts]'`; entry point `stark-translate-lite` | Part B |
>
> Parent: [`CLAUDE.md`](./CLAUDE.md) · backlog: [`docs/backlog.json`](docs/backlog.json) ·
> training detail: [`training/CLAUDE.md`](training/CLAUDE.md) · Lite contract:
> [`docs/lite_profiles.md`](docs/lite_profiles.md).
>
> **State:** no WSL job has run since 2026-04-30; Phase 4, the E4B domain SFT recipe, W17 and
> the CUDA latency proposal are scripted and wait for hardware time (the box is not reachable from
> the Mac, so CUDA work is delivered as scripts). Lite profiles are implemented with Mac CPU
> evidence only; nothing has run on an RTX 2070 or native Windows and no Lite latency gate has
> passed ([`docs/lite_profiles.md`](docs/lite_profiles.md)). Flags below were re-read from the
> scripts' argument parsers at `c00e697` (2026-09-12); Part A install steps are as last executed
> in 2026-03/04. Paths marked (WSL) exist on the training box, not in a Mac checkout. Long-form
> history: [`docs/archive/training/claude_windows_design_notes.md`](docs/archive/training/claude_windows_design_notes.md).
> Standing constraints: [`AGENTS.md`](AGENTS.md).

---

# Part A — WSL2 training environment (A2000 Ada)

## A1. WSL2 + CUDA

```bash
# PowerShell (Admin)
wsl --install -d Ubuntu-24.04

# Inside WSL — the Windows NVIDIA driver passes through; do NOT apt-install a Linux driver
nvidia-smi                        # A2000 Ada, ~16 GB visible
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get -y install cuda-toolkit-12-6   # toolkit only matters for building flash-attn; match it to the torch wheel (cu124 → 12-4) when you build it
nvcc --version
```

## A2. Python environment

```bash
sudo apt-get install python3.12 python3.12-venv ffmpeg pkg-config \
  libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
  libswscale-dev libswresample-dev libavfilter-dev
python3.12 -m venv ~/stt_train_env && source ~/stt_train_env/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124   # first
pip install -r requirements-windows.txt                                                        # training deps
pip install ruff mypy pytest pytest-cov pytest-timeout pre-commit bandit vulture                # dev tools
pip install ninja && pip install flash-attn --no-build-isolation                                # optional FA-2
```

`requirements-windows.txt` is the training environment; `requirements-mac.txt` /
`requirements-nvidia.txt` are deprecated for inference (use the `pyproject.toml` extras). Gemma 4
QLoRA additionally needs Unsloth (`docs/gemma4_tuning/phase_a_infrastructure.md`); CometKiwi
scoring uses a separate `comet_env`. If pip fails on `pkg_resources`, constrain
`setuptools<81` via `PIP_CONSTRAINT`.

## A3. Storage and shell

Training data and the HF cache live on `D:` (`/mnt/d/Data/stt-data`); `stark_data/`,
`stt-data/` and `bible_data/` in the checkout are symlinks there. Keep hot data on the WSL/D:
side, not `/mnt/c/`.

```bash
# ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512     # keep nvidia-smi "shared" near zero
export HF_HOME=/mnt/d/Data/stt-data/cache
export CUDA_VISIBLE_DEVICES=0
export STARK_DEEPGRAM__API_KEY=...                        # nested STARK_ settings key
export STARK_GEMMA4_VERSE=bible_data/aligned/verse_pairs_train_v2.jsonl   # v2 corpus for the SFT recipe (WSL)
alias stt='source ~/stt_train_env/bin/activate && cd /mnt/e/Code/stark-translate'
```

## A4. Pipeline refresh runbook (ordered)

Execute [`docs/wsl_pipeline_refresh.md`](docs/wsl_pipeline_refresh.md); this is the index.

| # | Stage | Command | Gate |
|---|-------|---------|------|
| 1 | Phase 4 full preprocess | `training/run_phase4_preprocess.sh` (env `STARK_RAW_DIR`, `STARK_CLEANED_DIR`; wraps `run_phase4_corpus.py --resume`) | `stark_data/cleaned/phase4_status.json`: `ready_for_training`, `errors == 0` |
| 2 | Gemma 4 E4B domain SFT → GGUF | `training/run_gemma4_e4b_domain_sft.sh` (set `STARK_GEMMA4_VERSE` to the v2 corpus) | `export_gguf.py --sanity-test` canaries ≥ 7/8, target 8/8 |
| 3 | W17 Whisper DoRA + hard-mix → CT2 | `training/run_w17_curriculum.sh` then `python tools/benchmark_stt_engines.py --manifest tools/stt_bench_manifest.json` | Match or beat W16 on overall and Tier 1 WER, no p95 regression ([v2026.7 bench](docs/archive/v2026.7/STT_BENCHMARK.md)) |
| 4 | Optional Parakeet EN bench (CUDA) | `python tools/benchmark_parakeet_en.py [--manifest] [--limit] [--device]` | Informational; the CUDA bilingual default stays Whisper. The Mac already uses Parakeet MLX for EN |
| 5 | Mac transfer / Phase 7 A/B (#135) | § A7 | `tools/health_check.py --backend mlx` 8-canary pass; A/B note |
| 6 | Phase 8 active learning | `tools/prepare_finetune_data.py` → human review → `tools/merge_corrections.py translation|whisper` → retrain | Real approved corrections, not fixtures (#137) |
| 7 | CUDA latency proposal | `scripts/cuda/build_llamacpp.sh` → `convert_gemma4_assistant_gguf.sh` → `bench_mtp.sh` / `retest_flash_attn.sh` | Gates in [`docs/cuda_latency_proposal.md`](docs/cuda_latency_proposal.md); nothing executed yet |

Before starting: `nvidia-smi` OK, venv active, sermon WAVs under `stark_data/raw/` (WSL), W16
located at `adapters/whisper_turbo_ct2/active` (CT2, WSL; the Mac checkout's `adapters/` holds only Marian CT2).
`run_w17_curriculum.sh` now uses `out_proj`. Its mandatory CPU preflight validates
the actual model configuration, source adapter tensors and intended corpus before
training. Mac source/fixture checks passed; run the preflight on the real WSL
artifacts before the first W17 run.

## A5. Training programs (summary — flags in `training/CLAUDE.md`)

| Program | Entry | State |
|---------|-------|-------|
| Whisper LoRA (W16 deployed on CUDA; W17 scripted) | `train_whisper.py` → `export_ct2.py` (default `--quantization int8_float16`, 5-clip sanity gate, `--sanity-wer-max 0.30`) | W17 untrained |
| Gemma 4 E4B/E2B QLoRA + CPO | `train_gemma4.py`, `train_gemma4_cpo.py` → `export_gguf.py --qtype Q4_K_M --sanity-test` | v1/v1.1/v2-cpo trained (parity, Jacobo canary failing #136); production recipe not run |
| TranslateGemma QLoRA | `train_gemma.py` + `run_*.sh` sweeps | Historical (S1–S9), superseded |
| Marian full fine-tune | `train_marian.py` | Fallback; production Marian is stock CT2 int8 |
| Piper voices | `prepare_piper_dataset.py`, `train_piper.py`, `export_piper_onnx.py` | Scripted; stock voices in production |

Evaluation entry points: `training/evaluate_translation.py` (BLEU/chrF++/COMET),
`training/evaluate_sermon.py`, `training/eval_whisper_wer.py`, `tools/benchmark_stt_engines.py`,
`tools/benchmark_translate_engines.py`, `training/benchmark_gemma4.py` (historical HF NF4
comparison). Numbers belong in `docs/archive/` or `docs/evaluation/`, never in guides.

## A6. CUDA inference on the training box

The A2000 also serves as the CUDA inference reference (v2026.8+): W16 Whisper CT2 +
Marian CT2 `int8_float16` partials + Gemma 4 E4B Q4_K_M finals through `llama-server`.

```bash
./start_server.sh                # NO_DRAFT=true default: target-only E4B
./start_server.sh --no-draft     # same, explicit
./start_server.sh --mtp          # opt-in official Gemma 4 MTP assistant (SPEC_N=3, f16 KV) — needs scripts/cuda/convert_gemma4_assistant_gguf.sh
./start_server.sh --flash-attn   # retest only; regressed on b8782
./start_server.sh --e2b-draft    # historical T4 draft (measured loss) — do not use
python dry_run_ab.py --backend cuda --lang en
```

llama.cpp pin `b10883` in `start_server.sh`, `Dockerfile` (`ARG LLAMA_CPP_REF`) and
`tools/llama_runtime.py`; Docker enables MTP with `STARK_LLAMA_MTP=1`
(`docker/entrypoint.sh`). The Mac live path is different: `dry_run_ab.py --mts` is rejected
before any model loads (`validate_live_mts`) and MTP stays an offline MLX experiment
([`docs/mlx_mtp_notes.md`](docs/mlx_mtp_notes.md), #177). Measured CUDA numbers:
[`docs/archive/v2026.5/BENCHMARK.md`](docs/archive/v2026.5/BENCHMARK.md),
[`v2026.7/STT_BENCHMARK.md`](docs/archive/v2026.7/STT_BENCHMARK.md),
[`v2026.8/MARIAN_BENCHMARK.md`](docs/archive/v2026.8/MARIAN_BENCHMARK.md),
[`v2026.9/GEMMA_OPTIM_PHASE2.md`](docs/archive/v2026.9/GEMMA_OPTIM_PHASE2.md). This box is
**not** the RTX 2070 target — Ada figures do not transfer to Turing (Part B).

## A7. Model transfer to Mac

Export first, then copy the **exported** artifacts — the Mac inference engines do not load
raw PEFT adapters for STT:

| Artifact | Export | Mac consumer |
|----------|--------|--------------|
| Whisper LoRA (W16/W17) | `training/export_ct2.py` → CTranslate2 (sanity gate built in) | `FasterWhisperEngine` on CPU for A/B only; Mac EN default is Parakeet MLX and ES is mlx-whisper — neither loads LoRA |
| Gemma 4 QLoRA | `training/export_gguf.py` → Q4_K_M GGUF for CUDA; MLX loads the safetensors adapter directory via `--adapter-dir` | `MLXGemmaEngine`, gated by `tools/health_check.py --backend mlx` (8 of 18 canaries) |
| Marian CT2 | `scripts/convert_marian_ct2.py --quantization int8` (setup does this automatically) | `adapters/marian_ct2/<dir>/active` or the managed setup cache |

```bash
# WSL — verify contents before copying
ls -la fine_tuned_gemma4_e4b_v1/            # adapter_config.json + adapter_model.safetensors
ls -la adapters/whisper_turbo_ct2/active/   # model.bin + config.json (CT2)
python tools/manage_adapters.py export --model whisper_turbo_ct2 --target user@mac:~/stark-translate/adapters/   # rsync
# or scp/USB/AirDrop, or a dry-run of the deploy pipeline:
python tools/deploy_adapters.py --cycle N --models whisper_turbo_ct2 --endpoints mac-dev --dry-run

# Mac — register, gate, activate
python tools/manage_adapters.py register --adapter <dir> --model <name> --eval-file <metrics.json>
python tools/health_check.py --backend mlx --adapter <dir>
python tools/manage_adapters.py activate --model <name> --version <version> --base-model <hf_repo>
```

`activate` runs the health check itself; its built-in base-model map only knows the
TranslateGemma names (`gemma_4b`, `gemma_12b`), so pass `--base-model` for Gemma 4 adapters.
Versions default to the adapter directory name (`--version` overrides); `register` writes
`adapters/manifest.json` with `versions`, `active`, `previous` and the safetensors SHA-256.
Remote endpoints in `tools/deploy_adapters.py` still need SSH keys
([`docs/deploy.md`](docs/deploy.md)).

## A8. WSL-specific notes

- **Driver:** install only the CUDA toolkit inside WSL; the Windows driver provides the GPU.
- **Filesystem:** `/mnt/c/` is slow; keep datasets, checkpoints and caches on D:/WSL storage.
- **Shared memory spill:** watch `nvidia-smi`; `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`.
- **Memory caps:** `align_deepgram_chunks.py` hard-codes a 12 GB `RLIMIT_AS`; `recover_shards.py`
  takes `--mem-cap-gb` (default 12); `mine_hard_examples.py` deliberately sets none (PyTorch
  virtual reservations exceed 40 GB).
- **FlashAttention-2:** optional for training; the llama.cpp `-fa` retest is a separate CUDA item.

---

# Part B — Native Windows / RTX 2070 Lite inference

Lite is the same pipeline, operator UI, displays and Review format with bounded product
profiles; the source of truth is [`docs/lite_profiles.md`](docs/lite_profiles.md) (implementation and
acceptance evidence). Packaging: [`docs/packaging/windows.md`](docs/packaging/windows.md) and
[`packaging/windows/README.md`](packaging/windows/README.md) (the unsigned MSI ships with the
v2026.14.0.0 GitHub Release; native Windows installation and first launch are unverified),
[`docs/packaging/models.md`](docs/packaging/models.md).

## B1. Profiles (`stark_translate/profiles.py`)

| Profile | STT (EN and ES) | Partial / final translation | Admission floor |
|---------|-----------------|-----------------------------|-----------------|
| `standard` (default of `stark-translate`) | Platform selection (Parakeet/Whisper on Mac, W16 CT2 on CUDA) | Marian CT2 / Gemma 4 E4B | Existing checks |
| `lite-cpu` (default of `stark-translate-lite`) | Whisper **small** CT2 int8, CPU | Marian CT2 / **Marian CT2** (no Gemma) | 4 physical cores, 8 GiB RAM |
| `lite-cpu-quality` | same | Marian CT2 / Gemma 4 **E2B** Q4_K_M via CPU llama.cpp | 4 cores, 16 GiB RAM |
| `lite-cuda-8gb` | Whisper large-v3-turbo CT2 int8_float16, CUDA | Marian CT2 / Gemma 4 E2B Q4_K_M via CUDA llama.cpp | 4 cores, 16 GiB RAM, 8 GB VRAM, sm_75+ |

Profiles are selected explicitly (`--profile` or `STARK_PROFILE`); hardware discovery never
upgrades a CPU launch to a GPU or an E2B launch to a larger model. They disable A/B,
speculative drafting, extra STT fallback models, multiprocess and live diarization; ONNX
Silero VAD replaces the Torch wrapper; E2B/native-runtime failures fail the session instead
of silently substituting a model. Floors are admission checks, **not** certified latency or
memory guarantees.

## B2. Install (native Windows, PowerShell; Linux uses `bin/` instead of `Scripts\`)

```powershell
py -3.11 -m venv .venv-lite
.venv-lite\Scripts\python -m pip install ".[lite-cuda,tts]"      # RTX 2070; use ".[lite-cpu,tts]" for CPU-only
py -3.11 -m venv .venv-lite-build
.venv-lite-build\Scripts\python -m pip install ".[lite-build]"    # one-time Marian CT2 conversion interpreter

.venv-lite\Scripts\stark-translate-lite setup --profile lite-cuda-8gb --models-dir D:\lite-models `
    --converter-python D:\path\.venv-lite-build\Scripts\python.exe --include tts
$env:STARK_MODELS_DIR = "D:\lite-models"
.venv-lite\Scripts\stark-translate-lite doctor --profile lite-cuda-8gb --json
.venv-lite\Scripts\stark-translate-lite operator --profile lite-cuda-8gb
```

- The runtime extra is Torch-free (CT2, faster-whisper, ONNX Runtime, tokenizers). CUDA 12
  cuBLAS and cuDNN 9 are an OS prerequisite for faster-whisper/CT2 on the 2070; the standard
  Torch wheel is not their installer.
- Windows setup downloads the pinned llama.cpp `b10883` CUDA 12.4 archive plus the matching
  `cudart` DLL archive (`tools/llama_runtime.py`); `--build-native` is for Linux CUDA only.
  Setup selects E2B only — E4B is never downloaded for Lite.
- `setup --offline` uses only the prepared cache and fails with the missing path instead of
  downloading; build wheelhouses for the target OS/arch/Python and install with
  `pip --no-index --find-links`.
- Each session starts its own `llama-server` on a free port with a random alias, verifies the
  native files, and never adopts or kills another listener.
- Lite never inherits the standard path's W16 CT2 adapter preference; it always uses the
  pinned artifact in `models.lock.json`.

## B3. Validation state

Passed on a Mac (2026-09-10): the isolated `lite-cpu,tts` install, offline setup reuse, synthetic
EN/ES CPU replays with TTS WAVs, the `lite-cpu-quality` E2B GGUF and native llama.cpp download and
verification, and one installed CPU E2B synthetic smoke with owned-server cleanup. **Not run:**
anything on an RTX 2070 or native Windows (sustained speech, VRAM/OOM, thermal, process cleanup,
MSI first launch). Receipts, hashes and limits: [`docs/lite_profiles.md`](docs/lite_profiles.md);
backlog `lite-cpu-inference`, `rtx2070-native-validation`, `windows-msi-bootstrap`. Do not reuse
A2000 (Ada) figures for the 2070 (Turing, no BF16); physical microphone, second audio output and
bilingual approval are separate gates (#131, `physical-second-output`, `bilingual-blinded-review`).

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| WSL doesn't see the GPU | Update the Windows NVIDIA driver; confirm WSL2 (`wsl -l -v`); never install a Linux driver inside WSL |
| CUDA OOM during training | Batch size 1 with more `--grad-accum`; gradient checkpointing is on in `train_whisper.py`; Gemma 4 via Unsloth QLoRA only (bf16 LoRA does not fit 16 GB) |
| `bitsandbytes` CUDA errors | Match the CUDA build: `pip install bitsandbytes --prefer-binary` |
| `demucs` OOM in Phase 4 | Use `--skip-demucs` for a first pass or process shorter files |
| `pyannote` auth error | Hugging Face token with the accepted pyannote user agreement (`--diarize` is optional) |
| Deepgram "No API key" | `STARK_DEEPGRAM__API_KEY` (double underscore) or `--api-key` |
| Domain SFT preflight rejects the corpus | The script defaults to `bible_data/aligned/verse_pairs_train_v2.jsonl` and rejects v1/missing configured inputs. Set explicit paths to the intended prepared corpora; do not bypass the guard |
| W17 PEFT "target modules not found" | The recipe now uses `out_proj` and preflight rejects `o_proj`. Inspect the selected model config and source adapter rank/tensors with the real WSL preflight before training |
| `export_ct2.py` sanity gate fails | Try `--quantization int8_bfloat16` or `float16` to isolate; canary clips must exist under `stark_data/whisper_dataset_deepgram/eval/` (WSL) |
| Adapter won't load on Mac | Both `adapter_config.json` and `adapter_model.safetensors` present; Whisper LoRA is CPU-CT2 only on Mac |
| Lite `doctor` fails admission | Check cores/RAM/VRAM floors in B1; pick the matching profile explicitly, it will not auto-downgrade |
| Lite setup: "corrupt native installation" | Move the invalid native directory aside and rerun setup; it fails closed rather than replacing evidence |
| Lite E2B session fails to start | Expected behavior on a missing/invalid GGUF or server: fix the artifact; there is no silent HF fallback |
