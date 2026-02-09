# Fast STT Options for SRTranslate

Feasibility assessment of faster speech-to-text alternatives to the current
mlx-whisper implementation. Research conducted 2025-02-08.

## Current Baseline

- **Library:** mlx-whisper 0.4.2 (0.4.3 released 2025-08-29)
- **Model:** `mlx-community/distil-whisper-large-v3`
- **Typical latency:** ~400-500ms per chunk (with word_timestamps=True)
- **Target:** 100-200ms (50-75% reduction)

---

## Option 1: lightning-whisper-mlx

**Repository:** https://github.com/mustafaaljadery/lightning-whisper-mlx
**PyPI:** `pip install lightning-whisper-mlx` (v0.0.10, all versions released April 2, 2024)
**License:** Not specified (no LICENSE file in repo)

### What It Does Differently

Three claimed optimizations over vanilla mlx-whisper:

1. **Batched decoding** -- processes multiple audio segments simultaneously for
   higher throughput. This benefits longer audio files but provides minimal gain
   for short (<5s) live chunks.
2. **Quantized models** -- offers 4-bit and 8-bit quantization options for
   non-distilled models. Distilled models (including distil-large-v3) always
   use the base (unquantized) MLX community weights.
3. **Distilled model support** -- first-class support for distil-whisper variants.

### Supported Models

tiny, small, base, medium, large, large-v2, large-v3, distil-small.en,
distil-medium.en, distil-large-v2, distil-large-v3. Also has a merged but
unreleased PR for large-v3-turbo.

### API

```python
from lightning_whisper_mlx import LightningWhisperMLX

whisper = LightningWhisperMLX(model="distil-large-v3", batch_size=12, quant=None)
result = whisper.transcribe(audio_path="/path/to/audio.mp3", language="en")
# Returns: {"text": "...", "segments": [...], "language": "en"}
```

**Critical limitation:** The `transcribe()` method signature is:

```python
def transcribe(self, audio_path, language=None):
    result = transcribe_audio(
        audio_path,
        path_or_hf_repo=f'./mlx_models/{self.name}',
        language=language,
        batch_size=self.batch_size,
    )
    return result
```

It does NOT forward `word_timestamps`, `initial_prompt`, `condition_on_previous_text`,
or any other kwargs to the underlying `transcribe_audio()` function.

There are two open (unmerged) PRs that would fix this:
- PR #20: word_timestamps + turbo model support (open since ~mid 2024)
- PR #23: generic **kwargs forwarding (open since ~mid 2024)

Neither has been merged. The repo appears **unmaintained** -- last PyPI release
was April 2, 2024 (10 versions all released the same day), and the open PRs
have had no maintainer response.

### Underlying Implementation

The `transcribe_audio()` function underneath actually DOES support the full
parameter set (word_timestamps, initial_prompt, temperature, etc.) because it
is essentially a fork/copy of the mlx-whisper transcription code. The problem
is solely in the `LightningWhisperMLX` wrapper class not passing those through.

### Benchmark Reality Check

The README claims "10x faster than Whisper CPP, 4x faster than current MLX
Whisper." However, independent benchmarks tell a different story:

**mac-whisper-speedtest results (M4 Pro, 24GB, large model):**

| Implementation             | Avg Time (s) | vs mlx-whisper |
|----------------------------|-------------- |----------------|
| fluidaudio-coreml          | 0.19          | 5.3x faster    |
| parakeet-mlx               | 0.50          | 2.0x faster    |
| **mlx-whisper**            | **1.02**      | **baseline**   |
| insanely-fast-whisper      | 1.13          | 1.1x slower    |
| whisper.cpp                | 1.23          | 1.2x slower    |
| **lightning-whisper-mlx**  | **1.82**      | **1.8x slower** |
| whisperkit                 | 2.22          | 2.2x slower    |

**lightning-whisper-mlx is actually 1.8x SLOWER than mlx-whisper.** The PR #20
reviewer confirms: "mlx-whisper is now faster than this implementation."

This likely happened because mlx-whisper received significant optimizations
(batched decoding was upstreamed) since lightning-whisper-mlx forked, while
lightning-whisper-mlx has been stagnant.

### Dependencies

Requires: huggingface_hub, mlx, numba, numpy, **torch**, tqdm, more-itertools,
tiktoken==0.3.3, scipy.

The `torch` dependency is problematic -- our project already loads torch for
Silero VAD, but pinning tiktoken to 0.3.3 may conflict with other packages.

### Known Issues (from GitHub Issues)

- #24: ValueError with quantized models (matrix dimension mismatch)
- #15: Hallucinations ("Thanks for watching!")
- #21: Missing transcription toward end of audio files
- #11: Quantization failure
- #5: No memory cleanup API

### Verdict: SKIP

- **Slower** than mlx-whisper in independent benchmarks
- Missing critical features (word_timestamps, initial_prompt not exposed)
- Unmaintained (no releases or PR merges since April 2024)
- Would require forking and maintaining ourselves
- Adds torch + numba as dependencies

---

## Option 2: WhisperKit (Argmax, CoreML)

**Repository:** https://github.com/argmaxinc/WhisperKit
**License:** MIT
**Installation:** `brew install whisperkit-cli` or Swift Package Manager

### What It Does

Converts Whisper models to CoreML format for optimized inference on Apple's
Neural Engine (ANE). Written in Swift with a local server that exposes an
OpenAI-compatible REST API.

### Python Integration

WhisperKit itself is Swift-only. Python access is through the local server:

1. Start WhisperKit server (runs on localhost:50060)
2. Connect from Python via OpenAI SDK or HTTP client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:50060/v1", api_key="not-needed")
result = client.audio.transcriptions.create(
    model="openai_whisper-large-v3",
    file=open("audio.wav", "rb"),
    response_format="verbose_json",
    timestamp_granularities=["word", "segment"],
    prompt="Sermon at Stark Road Gospel Hall...",
)
```

### Features

- Word timestamps: Yes (via `timestamp_granularities`)
- Initial prompt: Yes (via `prompt` parameter in the API)
- Streaming: Yes (via Server-Sent Events)
- Language detection: Yes
- Temperature control: Yes

### Benchmark Performance

In the mac-whisper-speedtest benchmark (M4 Pro), WhisperKit scored 2.22s --
actually **slower** than both mlx-whisper (1.02s) and lightning-whisper-mlx
(1.82s). However, this may not reflect the latest optimizations, and the
benchmark tested the large model (not distil-large-v3).

### Integration Complexity

- Requires running a separate Swift server process
- Audio must be passed via HTTP file upload (not in-memory numpy arrays)
- Extra latency from HTTP round-trip + file serialization
- Model selection is fixed at server startup
- No direct access to segment-level metadata (avg_logprob,
  compression_ratio, no_speech_prob) that we use for confidence scoring

### Verdict: NOT RECOMMENDED for our use case

- Server architecture adds latency for short chunks (HTTP overhead)
- Cannot pass numpy arrays directly -- must serialize to file
- No access to Whisper's internal confidence metrics
- Currently slower than mlx-whisper in benchmarks
- Adds operational complexity (managing a separate server process)
- Good for batch/app integration, poor for our real-time pipeline

---

## Option 3: Upgraded mlx-whisper (Current Library, Optimized)

**Repository:** https://github.com/ml-explore/mlx-examples/tree/main/whisper
**PyPI:** `pip install mlx-whisper` (v0.4.3, Aug 2025)

### Recent Optimizations

mlx-whisper has been actively maintained with performance improvements:

- v0.4.3 (Aug 2025): Latest release, stability improvements
- Feb 2025 PR #1259: Improved alignment computation (reduced upcasting,
  eliminated unnecessary array conversions, precise softmax)
- Batched decoding upstreamed from lightning-whisper-mlx's approach
- Native word_timestamps support
- Full feature set (initial_prompt, condition_on_previous_text, etc.)

### Why It Wins

In the mac-whisper-speedtest benchmark, mlx-whisper is the **fastest
Whisper-based implementation** -- only the non-Whisper alternatives (FluidAudio
CoreML using Parakeet TDT, parakeet-mlx) are faster.

### Optimization Strategies (Within mlx-whisper)

Since switching libraries will not help, focus on reducing time within
mlx-whisper:

1. **Skip word_timestamps for partials** (already implemented in dry_run_ab.py).
   word_timestamps adds ~100-200ms for alignment computation.

2. **Shorter audio chunks.** Whisper processes in 30-second windows. Shorter
   chunks = less decoding time. Our 2-second chunks are already good.

3. **Upgrade to v0.4.3.** The alignment speedup PR was merged in Feb 2025.

4. **Increase MLX cache limit.** Already at 256MB in dry_run_ab.py, which is
   reasonable. Could try 512MB if memory permits.

5. **Use distil-large-v3-turbo** if/when available on mlx-community. The turbo
   variant has fewer decoder layers and is significantly faster.

6. **Reduce initial_prompt length.** Current prompt is ~40 words. Each token
   in the prompt adds prefill time. Test with shorter prompts.

7. **Profile the pipeline.** The 400-500ms may not all be Whisper -- measure
   audio preprocessing, VAD, and numpy-to-MLX conversion separately.

### Verdict: BEST OPTION -- optimize what we have

---

## Option 4: FluidAudio / Parakeet (Non-Whisper, Fastest)

**Repository:** https://github.com/FluidInference/FluidAudio
**Performance:** 0.19s (5.3x faster than mlx-whisper in benchmarks)

### What It Is

NVIDIA's Parakeet TDT (Token-and-Duration Transducer) model converted to
CoreML by FluidInference. NOT a Whisper model -- different architecture with
potentially different accuracy characteristics.

### Why It Is Fast

- Uses Apple's ANE (Neural Engine) via CoreML, not the GPU
- TDT architecture is inherently faster than encoder-decoder (Whisper)
- 0.6B parameters, purpose-built for on-device STT

### Limitations for Our Use Case

- **Swift-only** (no Python API). Same server-based integration issues as
  WhisperKit.
- **No initial_prompt equivalent.** Cannot bias toward theological vocabulary.
- **No word_timestamps** (TDT gives different timestamp semantics).
- **No confidence metrics** (no avg_logprob, compression_ratio).
- **Unknown accuracy** on church audio / theological terms. Whisper is
  extensively validated; Parakeet less so.
- **25 European languages** -- supports English and Spanish, but translation
  quality estimation depends on Whisper's specific output format.

### Verdict: INTERESTING BUT INCOMPATIBLE

Worth monitoring for future consideration, especially if a Python binding
or MLX-native port emerges. Not viable as a drop-in replacement today.

---

## Feature Comparison Table

| Feature                    | mlx-whisper (current) | lightning-whisper-mlx | WhisperKit      | FluidAudio/Parakeet |
|----------------------------|-----------------------|-----------------------|-----------------|---------------------|
| **Latency (benchmark)**    | 1.02s                 | 1.82s (SLOWER)        | 2.22s (SLOWER)  | 0.19s (FASTER)      |
| **Python API**             | Native                | Native                | REST only       | Swift only          |
| **numpy array input**      | Yes                   | No (file path only)   | No (file only)  | No                  |
| **word_timestamps**        | Yes                   | No (PR open)          | Yes (via API)   | No                  |
| **initial_prompt**         | Yes                   | No (not forwarded)    | Yes (via API)   | No                  |
| **avg_logprob/conf**       | Yes                   | Yes (underlying)      | Partial         | No                  |
| **compression_ratio**      | Yes                   | Yes (underlying)      | No              | No                  |
| **condition_on_prev_text** | Yes                   | No (not forwarded)    | Unknown         | No                  |
| **distil-large-v3**        | Yes                   | Yes                   | Yes             | N/A (not Whisper)   |
| **Maintained**             | Actively (Apple)      | Stale (Apr 2024)      | Actively        | Actively            |
| **Dependencies**           | mlx, numpy            | mlx, torch, numba     | Swift runtime   | Swift runtime       |
| **LoRA adapter support**   | Via model path        | Via model path        | Unknown         | No                  |
| **License**                | MIT                   | Unspecified           | MIT             | Proprietary         |

---

## Migration Effort Estimates

| Option                  | Effort    | Risk  | Expected Gain |
|-------------------------|-----------|-------|---------------|
| Upgrade mlx-whisper     | ~1 hour   | None  | 5-15%         |
| Lightning-whisper-mlx   | ~4 hours  | High  | Negative (slower) |
| WhisperKit server       | ~8 hours  | Med   | Negative (slower + HTTP overhead) |
| FluidAudio/Parakeet     | ~16 hours | High  | 3-5x (but loses critical features) |

---

## Recommendation

### Do First (immediate, low risk):

1. **Upgrade mlx-whisper to 0.4.3.** The Feb 2025 alignment speedup and any
   other accumulated improvements will help, especially for word_timestamps
   passes.

2. **Profile the actual pipeline.** Measure where the 400-500ms is spent:
   - numpy-to-MLX array conversion
   - Whisper encoder forward pass
   - Whisper decoder forward pass
   - Word timestamp alignment
   - Result extraction / postprocessing

3. **Benchmark with and without word_timestamps.** If the gap is >150ms, the
   existing optimization (word_timestamps=False for partials) is the biggest
   win available.

### Do NOT Do:

- **Do not switch to lightning-whisper-mlx.** It is slower, unmaintained,
  and missing critical features. The benchmark data is unambiguous.

- **Do not integrate WhisperKit.** The server architecture adds latency and
  complexity for our use case (short, frequent audio chunks from a live mic).

### Monitor for Future:

- **distil-large-v3-turbo on MLX.** When available, this should provide a
  significant speed boost with minimal accuracy loss. The turbo models use
  fewer decoder layers.

- **Parakeet/FluidAudio Python bindings.** If a native Python API emerges,
  the 5x speed advantage would be worth a serious evaluation, even with the
  feature gaps.

- **mlx-whisper speculative decoding.** If mlx-whisper adds support for using
  a smaller model (like distil-small.en) to draft tokens for distil-large-v3,
  this could provide significant speedups similar to what we do for
  TranslateGemma translation.
