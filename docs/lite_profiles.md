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

The [installed natural-English service rehearsal](evaluation/overnight_endurance_20260910/README.md)
completed on this Mac's CPU with consistent retained spans, required writes and
process cleanup. First translated previews were sparse and observed final latency
tails were large. Lite is functionally exercised, but cannot be recommended as a
fast production profile today; the observational Standard/Lite results are not a
causal paired comparison or a human-quality certificate.

**2026-09-10 cadence update:** the
[normalized CPU Lite screen](evaluation/mac_followup_20260910/lite-cadence-result.md)
completed 24 file replays with 0/4 language/cadence arms qualified. Slower partial
intervals lost preview coverage and responsiveness, including cases with faster
final medians. The 0.6-second cadence remains unchanged. These runs used
Whisper-small and Marian finals; the harness's `e2b` label does not indicate
Gemma inference or a test of `lite-cpu-quality`.

CPU Whisper small/base quality recovery and independent CPU Lite deadline screens
remain pending. Deadline screens retain the default cadence; rejected cadence
arms cannot enter a combination. The separate
[Standard endpoint/deadline](evaluation/mac_followup_20260910/standard-screen-result.md)
and [Spanish Parakeet](evaluation/mac_followup_20260910/spanish-parakeet-result.md)
results leave the Mac Gemma E4B and Spanish Whisper defaults unchanged. None of
these file cohorts certifies microphone reliability, physical output or bilingual
service quality.

The profiles disable A/B, speculative drafting, extra STT fallback models,
multiprocess and live diarization. TTS is an optional CPU extra. E2B failures fail
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
.venv-lite/bin/python -m pip install --upgrade 'pip>=26.2' 'setuptools>=83.0.0'
.venv-lite/bin/python -m pip install '.[lite-cpu,tts]'
python3 -m venv .venv-lite-build
.venv-lite-build/bin/python -m pip install --upgrade 'pip>=26.2' 'setuptools>=83.0.0'
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

## Installed CPU smoke observed on 2026-09-10

The isolated `lite-cpu,tts` installation from commit `0d5a875` passed an actual
outside-checkout import check for 14 runtime/operator modules. Its distributions
and loaded modules contain no Torch, MLX, Silero Python wrapper or bitsandbytes;
`pip check` passes. CT2 4.8.2, faster-whisper 1.2.1, ONNX Runtime 1.29.0,
transformers 5.17.0 and Piper 1.8.0 were resolved. Both Marian tokenizers also
loaded and encoded offline without Torch. Setup installed four selected assets,
reused four, and failed none; its offline repeat reused all eight. Verified
existing Marian CT2 builds were copied to the new cache without conversion or
modification of the working environment or adapters.

The wheel SHA-256 is
`a1578b40b6a18e9710e18d7bd1ba4fdb00d91fd2cf8595f687c98b079ed9f5e3`.
The executed installed pipeline SHA-256 is
`227661cdd1a865eff9150dd0b726b1377b6282568d8688c5d1eec931b3c8afae`.
The [machine-readable evidence](evaluation/lite_cpu_smoke_20260910.json) records
model revisions, model file hashes, dependency versions, exact commands and raw
artifact checksums. The original preparation and logs remain under
`.cache/overnight-20260910/`; the environment is `lite-runtime`, the prepared
cache is `lite-models`, and replay artifacts are `lite-cpu-smoke-attempt01`.

| Synthetic input | Observed STT | Observed CPU Marian final | Completion |
| --- | --- | --- | --- |
| EN, 3.283 s | The grace of God brings salvation. | La gracia de Dios trae salvación. | exit 0, completed lifecycle, one TTS WAV |
| ES, 3.248 s | La gracia de Dios trae salvación. | God's grace brings salvation. | exit 0, completed lifecycle, one TTS WAV |

Both replays used the installed package from a separate data directory with
network model access disabled, Whisper small INT8 and CPU Marian. No Gemma,
GPU, microphone or physical speaker output was used. The short smoke inputs and
outputs are synthetic and do not establish translation quality or performance
on natural speech. Timing fields remain in the raw CSV but are not promoted as a
latency gate. Lifecycle peak process RSS was approximately 1.28 GiB (EN) and
1.39 GiB (ES) on this Mac, not a claim about total deployment memory.

One provenance limitation was found: the historical lifecycle Git lookup walked
from the installed venv into its ancestor checkout and recorded that checkout's
HEAD. The wheel and executed pipeline hashes above identify the tested code;
the observed Git values in the raw records are retained rather than rewritten.
Managed model revisions are also independently bound by profile metadata and
artifact receipts even where a legacy lifecycle field labels an explicit local
path's revision as unknown. Follow-up integration should reject unrelated ancestor
Git identities and read managed model receipts directly.

For the standard CPU/CUDA path, an untouched STT default now delegates to the
factory's existing active CT2 adapter preference. Explicit model configuration
(including explicitly selecting the stock `large-v3-turbo` alias) takes priority.
Lite profiles always select their pinned artifact and never inherit that adapter
preference. This correction has targeted loader contract tests. The completed Lite replays
above identify the earlier installed pipeline hash; this follow-up did not rerun
the models.


## Optional CPU quality artifacts prepared on 2026-09-10

The same isolated environment completed explicit `lite-cpu-quality` setup without
changing the default CPU profile. It installed the pinned E2B GGUF and native
llama.cpp archive, reused the six other selected model entries, and failed none.
An offline repeat reused all seven model entries and the verified native runtime.
No conversion or working environment changes were needed. E4B was not downloaded.

The E2B Q4_K_M file is 3,427,877,920 bytes with SHA-256
`62adb571af12205e1e6ce0f2a4bd2835441f3e9a47d782af1539c0bc8831107f`.
The macOS arm64 archive is pinned to llama.cpp `b10883`, commit
`91f6a6cf361385700bbe15981f0f39909df77498`, with SHA-256
`a83a885bf2fa4ffb7c11b3c8c6ed7e7ff8f7bd61733b1d4fa2ce8ec7cb587588`.
The installed `llama-server` executable SHA-256 is
`d707b6db4c1397a7383176fba12d339e5b33c7513669d74c8fbc2a76f6979a72`.
Its version command returned `0.4.0-dev` (build 10883, commit `91f6a6cf3`),
built with AppleClang 21.0.0.21000101 for Darwin arm64.

The [preparation evidence](evaluation/lite_cpu_quality_preparation_20260910.json)
contains exact setup commands, environment, native file inventory and raw log
hashes. This is download, offline reuse, integrity and executable-version
verification only. No native server or E2B model inference was started; CPU-quality
memory, output fidelity, latency and real RTX 2070 behavior remain unmeasured.
The environment still identifies source `0d5a875`; install the integrated source
before collecting new performance evidence.

## Native server log limits and privacy

Managed llama-server stdout/stderr is drained by a dedicated thread in reads of
at most 4 KiB, with bounded line fragments, into a nonblocking 2,048-record queue.
A separate writer stores structured records in `metrics/llama_SESSION.log`.
Each record's message is capped at 4,000 characters. Rotation keeps a current
file of at most 20 MiB and five backups, at most 120 MiB per native session log
set. Rotation can discard the oldest messages even during an active session.
Queue overload drops operational log records and counts them; a failed log sink
also increments an operational counter while the native output is still drained.
These conditions do not invalidate required audio or caption recording.

Native messages can contain private text. They stay in this local log, are never
forwarded to the shared application logger, and are excluded from support exports
by the support API's file allowlist, including its text/audio options. Newly
created and rotated native logs have mode `0600` on macOS/Linux; Windows access
uses the data directory's inherited ACL. Do not manually share them without review.

Age-based cleanup deletes only `metrics/llama_SESSION.log` and numbered backups
`.1` through `.5` older than 30 days when session lifecycle and diagnostic integrity
prove that the session completed. Active, interrupted and unknown sessions are
not deleted by age. Original audio, diagnostics, corrections and exports are
never part of this log cleanup. The owned-server stop sequence terminates the
child (10-second grace, then kill with a 5-second wait), waits at most one second
for its output reader and at most three seconds for the asynchronous writer.
`log_snapshot()` reports queued, dropped, write failures, read failures and an
incomplete-drain indicator separately; final values are retained in native
provenance. The native process remains the only process this owner terminates.


## Installed CPU quality smoke observed on 2026-09-10

The integrated installed wheel built from `a1d7cdf` completed one English-to-Spanish
synthetic replay as `overnight_lite_quality_smoke02_en`. Whisper small INT8 produced
“The grace of God brings salvation.” The owned E2B server produced “La gracia de
Dios trae la salvación.” The lifecycle completed with exit 0, all four required
writes completed, and no pending or failed recording writes. This check did not
use TTS, a microphone, a browser client or physical output.

The wheel SHA-256 is
`c5c12d8658c01a79c5ad93252d15af9f68a6c61a07469a0e80fa6772934e4da3`;
the installed pipeline SHA-256 is
`ba5905fe8bcf4f2e2c8e90c1dc97aaefb4ac5c2255e73ac97c13efe2a89a1ef8`.
Session metadata binds the exact prepared GGUF and native executable hashes,
CPU backend, `-ngl 0`, `--no-kv-offload`, `--no-op-offload`, three threads,
512-token context and q8 KV cache. The generic lifecycle model label is still
`llama.cpp`; actual model identity is retained in `session_metadata.managed_llama`.

The single silence-final sample recorded 3,452.8 ms from estimated speech end to
server payload readiness, 1,251.3 ms STT, and 428.7 ms translation. These are smoke
observations, not latency targets or certification. The retained 1,106,247,680-byte
peak RSS covers the Python pipeline only; the owned llama-server is excluded.
Combined process-tree peak memory was not sampled and cannot be reconstructed
from the system RAM snapshot. No RAM floor gate is claimed.

The native log contains 22 valid structured records, is 7,260 bytes with mode
`0600`, and ends with the native cleanup message. Both successful-run processes
and the earlier failed-run process were absent at the read-only audit. Rotation
and retention were covered by unit tests, not exercised by this short run. This
wheel did not persist final native log counters; the follow-up completion-metadata
wiring is tested separately and does not rewrite the observed result.

The first attempt, `overnight_lite_quality_smoke_en`, supplied a nonexistent audio
filename and failed before model loading. Its original traceback and lifecycle
marker remain preserved. That early failure left a stored `running` marker even
though its process exited; dead-PID status handling prevents export. The input
validation correction is separate from this historical evidence.

The [machine-readable smoke evidence](evaluation/lite_cpu_quality_smoke_20260910.json)
indexes hashes for both attempts and the executed wheel. Spanish-source E2B,
natural speech, sustained latency, total memory, native Windows/Linux and real
RTX 2070 acceptance remain separate validation work.


## Installer security check on 2026-09-10

The installed Lite dependency audit found vulnerabilities in the environment's
seeded installer tools, pip 24.0 and setuptools 65.5.0. Only those two packages
were upgraded in `.cache/overnight-20260910/lite-runtime`: pip is now 26.2.1 and
setuptools 84.0.0. No other existing package version changed, and the working
`stt_env` was not modified. A repeat installed-path audit reports zero known
vulnerabilities across 60 audited dependencies. The locally built, unpublished
`stark-translate` package is the sole expected PyPI audit skip; project source
security remains covered by source scans and review.

`pip check` passes. Fourteen actual installed runtime/operator imports from
outside the checkout still have no Torch, MLX, Silero wrapper or bitsandbytes
distributions or loaded modules. No model inference ran during this remediation;
the earlier smoke artifacts are unchanged. The
[evidence report](evaluation/lite_installer_security_20260910.json) includes the
exact commands, version changes and hashes for both audit reports and import checks.

New installation instructions upgrade pip to at least 26.2 and setuptools to at
least 83.0.0 before installing Lite or converter dependencies. Bootstrap applies
the same minimums and stops if this tool upgrade fails. For an offline package
installation, prepare those installer wheels in the wheelhouse too and use
`--no-index --find-links /absolute/wheelhouse` for the upgrade and installation;
model-cache `setup --offline` is separate from Python package installation.

The security workflow now audits an actual isolated Linux `lite-cpu,tts`
installation on packaging changes (including `pyproject.toml`) and weekly. It
runs `pip check`, scans the complete installed dependency inventory, rejects
unexpected unaudited or non-Lite packages, and retains the JSON report. The final
security gate fails on audit failure, cancellation, failed change detection or an
unexpectedly skipped required audit. Sixteen local tests exercise these report
and shell-gate cases; the new GitHub job itself was not executed locally. This
job downloads Python packages only, never model weights, and does not certify
Windows, macOS or RTX 2070 runtime behavior.
