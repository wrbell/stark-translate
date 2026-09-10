# Stark Translate Lite

Lite is the same pipeline, operator, displays and Review data format. The
`stark-translate-lite` entry point defaults to `lite-cpu`, including on a Mac.
Choose another profile explicitly; hardware discovery never silently upgrades a
CPU launch to Metal or an E2B launch to a larger HF model.

| Profile | STT (EN and ES) | Provisional / final translation | Initial admission floor |
| --- | --- | --- | --- |
| `lite-cpu` | multilingual Whisper small, CT2 INT8, CPU | CPU Marian INT8 / CPU Marian INT8 | 4 physical cores, 8 GiB RAM |
| `lite-cpu-quality` | same small INT8 | CPU Marian / Gemma 4 E2B Q4_K_M via CPU llama.cpp | 4 physical cores, 16 GiB RAM |
| `lite-cuda-8gb` | Whisper Turbo, CT2 INT8_FP16, CUDA | CPU Marian / Gemma 4 E2B Q4_K_M via CUDA llama.cpp | 4 physical cores, 16 GiB RAM, 8 GB NVIDIA VRAM, sm_75+ |
| `standard` | existing platform selection | existing platform selection | existing standard checks |

These are product admission floors and evaluation targets, **not certified latency
or memory guarantees**. Original RTX 2070 (8 GB, compute capability 7.5), native
Windows and Linux x86 remain external certification targets. Mac CPU replay can
validate the CPU implementation and offline package path; it does not certify an
x86 CPU or a 2070. Use 16 GiB RAM for CPU deployments with other applications.
The initial budget leaves three CT2 STT threads, one STT worker and one Marian
thread. Final output quality and sustained backlog must be measured independently
for each tier. No latency gate has passed for Lite yet.

The profiles disable A/B, speculative drafting, extra STT fallback models,
multprocess and live diarization. TTS is an optional CPU extra. E2B failures fail
the selected session rather than silently loading HF NF4 or publishing a model
error as a translation. CPU normal finals are actual Marian translations; no
missing-Gemma sentinel reaches the audience.

## Prepare an isolated CPU runtime

Use Python 3.11 or 3.12. Keep the working standard environment separate. Commands
below work from a source checkout or replace `.` with a local release wheel.
Windows venv interpreters live in `Scripts/python.exe` and executables in
`Scripts`; Unix examples use `bin`.

```sh
python3 -m venv .venv-lite
.venv-lite/bin/python -m pip install '.[lite-cpu,tts]'
python3 -m venv .venv-lite-build
.venv-lite-build/bin/python -m pip install '.[lite-build]'
.venv-lite/bin/stark-translate-lite setup --models-dir /absolute/lite-models \
  --converter-python /absolute/.venv-lite-build/bin/python --include tts
STARK_MODELS_DIR=/absolute/lite-models .venv-lite/bin/stark-translate-lite doctor --json
STARK_MODELS_DIR=/absolute/lite-models .venv-lite/bin/stark-translate-lite operator
```

The runtime extra contains CT2, faster-whisper, ONNX Runtime, the Marian tokenizer,
SentencePiece and audio resampling dependencies. It does not require Torch,
Silero's Torch wrapper, MLX, bitsandbytes or training dependencies. The separate
build interpreter converts the pinned Marian HF sources once and runs the
existing CPU nonempty translation smoke before atomic publication. Existing
**verified managed** Marian artifacts are reused; Lite does not select working
adapter directories with unknown source revisions. The build interpreter can be
removed after setup. Conversion is intentionally not a runtime dependency.

`models.lock.json` pins Whisper small to
`536b0662742c02347bc0e980a01041f333bce120`, Turbo to
`0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`, both Marian sources, E2B and optional
Piper voices. Silero ONNX is v6.2.1 commit
`7e30209a3e901f9842f81b225f3e93d8199902b1` (MIT), SHA-256
`1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`.
Its NumPy wrapper preserves recurrent state and stream resets at 16 kHz without
importing Torch. E2B and ONNX files are checked against pinned hashes at load.
Session metadata includes the resolved profile, model paths/revisions, STT
settings, and owned native server/model hashes and command line.

## Offline handoff and quality mode

Prepare the entire cache online, then copy it with its `.installed`, export
manifests, `active.json` and native files intact. Build a wheelhouse for the
**target OS, CPU architecture and Python version**, not on a different Mac
architecture. Install with `pip --no-index --find-links /wheelhouse` and validate:

```sh
STARK_MODELS_DIR=/absolute/lite-models .venv-lite/bin/stark-translate-lite setup --offline
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 STARK_MODELS_DIR=/absolute/lite-models \
  .venv-lite/bin/stark-translate-lite operator
```

`setup --offline` never downloads missing assets. A missing source/conversion or
native artifact fails with the path that must be prepared. E2B CPU mode is opt-in:

```sh
.venv-lite/bin/stark-translate-lite setup --profile lite-cpu-quality \
  --models-dir /absolute/lite-models --converter-python /absolute/.venv-lite-build/bin/python
STARK_MODELS_DIR=/absolute/lite-models .venv-lite/bin/stark-translate-lite operator --profile lite-cpu-quality
```

CPU llama.cpp receives `-ngl 0`, no KV/operation offload, three threads, one slot,
512-token context and quantized KV. It therefore remains a CPU test on a Mac with
Metal installed. Windows CPU, macOS CPU and Linux CPU setup selects a pinned
upstream native archive. The source revision is
`91f6a6cf361385700bbe15981f0f39909df77498` / `b10883`; archive SHA-256 values are in
`tools/llama_runtime.py`. Files are inventoried and verified before each owned
server starts. Corrupt existing native installations fail closed without replacing
the retained evidence; move the invalid directory aside and rerun setup.

## RTX 2070 target

Install `.[lite-cuda,tts]` in a separate runtime and provide CUDA 12 cuBLAS and
cuDNN 9 as required by faster-whisper/CT2. These native GPU libraries remain an
OS deployment prerequisite; the standard Torch wheel is not their installer.
Windows setup downloads the pinned llama.cpp CUDA 12.4 archive and matching
runtime DLL archive. Linux setup builds the exact source commit with
`CMAKE_CUDA_ARCHITECTURES=75` when `--build-native` is explicitly supplied:

```sh
.venv-lite/bin/stark-translate-lite setup --profile lite-cuda-8gb \
  --models-dir /absolute/lite-models --build-native \
  --converter-python /absolute/.venv-lite-build/bin/python
STARK_MODELS_DIR=/absolute/lite-models .venv-lite/bin/stark-translate-lite doctor --profile lite-cuda-8gb --json
STARK_MODELS_DIR=/absolute/lite-models .venv-lite/bin/stark-translate-lite operator --profile lite-cuda-8gb
```

Linux build needs Git, CMake and the CUDA toolkit. `--build-native` is unnecessary
for the Windows archive. Setup chooses E2B only; it does not also download E4B.
The 512-token context is deliberately bounded and should be checked with the
longest supported utterance before deployment. Do not claim BF16 support or reuse
Ada latency figures for Turing. Upstream references:
[NVIDIA capability table](https://developer.nvidia.com/cuda/gpus),
[CT2 quantization support](https://opennmt.net/CTranslate2/quantization.html),
[faster-whisper GPU requirements](https://github.com/SYSTRAN/faster-whisper#gpu),
[llama.cpp build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

## Validation and ownership

A session starts a fresh local llama-server on an unused port with a random model
alias; readiness checks health and that exact alias. It never adopts or kills an
existing listener. The child inherits the pipeline process group, and normal
shutdown waits for or kills only the owned child. The operator's shared
cooperative control/lifecycle channel handles Windows pause/resume/stop; profile
selection does not create a competing control protocol.

A controlled CPU replay (run from an isolated data directory) uses:

```sh
STARK_MODELS_DIR=/absolute/lite-models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /absolute/.venv-lite/bin/python /absolute/source/dry_run_ab.py \
  --profile lite-cpu --lang en --audio-file /absolute/approved-input.wav \
  --session-id lite_cpu_en_01 --gain 1 --no-ab --tts --tts-output wav
```

Repeat in Spanish with a different explicit session ID and exact input/reference
manifest. Compare observed first partial, partial gap, final readiness, STT
quality, Marian adequacy, CPU/RAM, backlog and completed lifecycle. Keep synthetic
smokes separate from natural speech quality gates. E2B quality-mode runs need their
own latency/memory cohort. GPU runs require real 2070 sustained speech, VRAM/OOM,
thermal soak and OS process-cleanup checks. Physical microphone, second audio
output/hotplug and human bilingual approval remain separate acceptance gates.
