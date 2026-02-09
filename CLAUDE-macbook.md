# CLAUDE-macbook.md — Mac Inference Environment Guide

> **Machine:** M3 Pro MacBook (Mac15,6), 18GB unified memory, 12-core CPU (6P+6E), 18-core GPU, Metal 4, MLX acceleration
> **Role:** Inference, live demos, quality monitoring, browser displays, A/B testing
> **Parent doc:** [`CLAUDE.md`](./CLAUDE.md)

---

## Hardware Profile (Verified)

| Spec | Value |
|------|-------|
| Model | MacBook Pro (Mac15,6, MRX43LL/A) |
| Chip | Apple M3 Pro |
| CPU | 12 cores (6 performance + 6 efficiency) |
| GPU | 18 cores |
| Memory | 18 GB unified (17.2 GB usable) |
| Metal | Metal 4 |
| L2 Cache | 4 MB |
| Display | 3024x1964 Retina XDR (14") |
| iogpu wired limit | 0 (unlimited GPU memory) |

### Inference Memory Budget

| Component | Framework | Memory |
|-----------|-----------|--------|
| Distil-Whisper large-v3 | mlx-whisper | ~1.5 GB |
| TranslateGemma 4B 4-bit | mlx-lm | ~2.5 GB |
| TranslateGemma 12B 4-bit | mlx-lm | ~7 GB |
| MarianMT CT2 int8 | ctranslate2 | ~76 MB |
| MarianMT PyTorch | transformers | ~300 MB |
| Silero VAD | PyTorch | ~2 MB |
| **Total (4B only)** | | **~4.3 GB** |
| **Total (A/B mode)** | | **~11.3 GB** |
| **Headroom (18 GB, A/B)** | | **~6.7 GB** |

With 18 GB unified memory, A/B mode (both Gemma models in MLX 4-bit) fits comfortably at ~11.3 GB peak, leaving ~6.7 GB for macOS and buffers.

**Important:** PyTorch/MPS with bitsandbytes 4-bit is CUDA-only (~18s/weight on MPS). PyTorch fp16 on MPS causes inf/nan with TranslateGemma. MLX 4-bit is the correct approach for Apple Silicon inference.

---

## Environment Setup

### Prerequisites

- macOS 14+ (Sonoma or later for Metal/MLX support)
- Grant mic access: **System Settings → Privacy & Security → Microphone**
- Homebrew installed

### Verified Environment

- Python 3.11.11
- MLX 0.30.6, mlx-lm 0.30.6, mlx-whisper 0.4.3
- PyTorch 2.10.0 (used for Silero VAD and MarianMT PyTorch backend only)
- CTranslate2 (MarianMT int8 backend)

### Installation

```bash
# Update Homebrew
brew update
brew install ffmpeg portaudio

# Create virtualenv
python3.11 -m venv stt_env
source stt_env/bin/activate

# Install all dependencies from requirements file
pip install -r requirements-mac.txt

# HuggingFace login (required for TranslateGemma model access)
huggingface-cli login

# Download all models
python setup_models.py
```

The `requirements-mac.txt` file includes all dependencies. Key packages:

```
# Core inference (MLX-based, NOT PyTorch for main inference)
mlx                                     # Apple ML framework
mlx-lm                                  # TranslateGemma 4B/12B 4-bit inference
mlx-whisper                             # Distil-Whisper STT inference
ctranslate2                             # MarianMT int8 for fast partial translations

# Audio & VAD (still uses PyTorch for Silero VAD)
torch torchvision torchaudio
sounddevice silero-vad

# Translation (PyTorch MarianMT runs in parallel with CT2 for comparison)
transformers sentencepiece protobuf
websockets                              # WebSocket server for browser displays

# Evaluation & monitoring
jiwer sacrebleu
faster-whisper                          # CTranslate2-based confidence scores
peft                                    # For loading LoRA/QLoRA adapters

# Translation QE
unbabel-comet                           # CometKiwi
sentence-transformers                   # LaBSE

# YouTube caption monitoring
youtube-transcript-api streamlink innertube
```

### First-Run Model Downloads

Run `python setup_models.py` to download and verify all models. Uses Wi-Fi — total ~12 GB for A/B mode:

| Model | Framework | Size | Purpose |
|-------|-----------|------|---------|
| `mlx-community/distil-whisper-large-v3` | mlx-whisper | ~1.5 GB | STT |
| `mlx-community/translategemma-4b-it-4bit` | mlx-lm | ~2.2 GB disk, ~2.5 GB RAM | Translation A |
| `mlx-community/translategemma-12b-it-4bit` | mlx-lm | ~6.6 GB disk, ~7 GB RAM | Translation B |
| `Helsinki-NLP/opus-mt-en-es` (CT2 int8) | ctranslate2 | ~76 MB | Fast partial translation |
| `Helsinki-NLP/opus-mt-en-es` (PyTorch) | transformers | ~300 MB | Comparison logging |
| `Unbabel/wmt22-cometkiwi-da` | comet | ~580 MB | Translation QE |
| `sentence-transformers/LaBSE` | sentence-transformers | ~470 MB | Cross-lingual similarity |
| `Helsinki-NLP/opus-mt-es-en` | transformers | ~75 MB | Back-translation |

**Memory note:** Both Gemma models fit simultaneously in MLX 4-bit (~2.5 + ~7 + ~1.5 GB Whisper + ~0.3 GB MarianMT = ~11.3 GB), leaving ~6.7 GB for macOS. Monitor with `mx.metal.get_active_memory()`.

**Critical EOS fix:** TranslateGemma requires adding `<end_of_turn>` (id=106) to `tokenizer._eos_token_ids`. The default EOS token `<eos>` (id=1) is never generated, causing 256 pad tokens (~5s wasted). This fix is applied in `dry_run_ab.py`.

---

## Core Pipeline (`dry_run_ab.py`)

### Two-Pass Architecture

```
Mic (48kHz) → Resample 16kHz → Silero VAD ─┐
                                             │
         ┌───────────────────────────────────┘
         │
         ├─ PARTIAL (on 1s of new speech, while speaking)
         │    mlx-whisper STT (~500ms)
         │    MarianMT CT2 int8 EN→ES (~50ms)      ← italic in display
         │    Total: ~580ms
         │
         └─ FINAL (on silence gap or 8s max utterance)
              mlx-whisper STT (~500ms, word timestamps)
              TranslateGemma 4B EN→ES (~650ms)       ← replaces partial
              TranslateGemma 12B EN→ES (~1.4s, --ab) ← side-by-side
              Total: ~1.3s (4B) / ~1.9s (A/B)
                                   │
                                   ▼
                        WebSocket (0.0.0.0:8765)
                                   │
            ┌──────────┬───────────┼───────────┐
            ▼          ▼           ▼           ▼
        Audience    A/B/C       Mobile      CSV +
        Display    Compare     Display     Diagnostics
       (projector) (operator)  (QR code)    (JSONL)
```

### Key Details

- **STT:** `mlx-whisper` with `mlx-community/distil-whisper-large-v3`, word timestamps on finals only (disabled for partials to save ~100-200ms)
- **Fast translation:** MarianMT CT2 int8 (`ct2_opus_mt_en_es/`, 76MB). PyTorch variant runs in parallel for comparison logging
- **Quality translation:** TranslateGemma via `mlx-lm` — 4B always, 12B with `--ab` flag
- **Speculative decoding:** 4B as draft model for 12B via `mlx_lm.generate(draft_model=)`, configurable with `--num-draft-tokens`
- **Serving:** HTTP on `0.0.0.0:8080` (`--http-port`) serves display HTML to phones. WebSocket on `0.0.0.0:8765` pushes transcriptions
- **Metal cache:** `mx.set_cache_limit(100 * 1024 * 1024)` prevents cache growth with word_timestamps
- **Model pre-warming:** 1-token forward pass during silence gaps to avoid cold-start latency
- **Background I/O:** WAV/JSONL/CSV writes run on background threads to avoid blocking inference

### Metrics

Logs approach, latency (STT/translate/E2E), tokens/sec, confidence scores, WER, BLEU, text pairs → CSV and JSONL export. Both pipelines process each audio chunk in parallel via `asyncio.gather` + `run_in_executor`.

### Understanding the Fine-Tuned Adapters

The Mac loads LoRA/QLoRA adapters trained on the Windows desktop. These are small (~60–120 MB) parameter patches that customize the base models for church content without modifying the original weights.

**Whisper LoRA adapter** (`fine_tuned_whisper_mi/`): Trained on 20–50 hours of church sermon audio via pseudo-labeling. Adapts both encoder (reverberant church acoustics, PA system artifacts) and decoder (theological vocabulary: "sanctification," "propitiation," biblical proper names). Expected WER reduction: 10–30% relative on church audio.

**TranslateGemma QLoRA adapter** (`fine_tuned_gemma_mi_A/` or `_B/`): Trained on ~155K Bible verse pairs from public-domain translations (KJV/ASV/WEB/BBE/YLT ↔ RVR1909/Español Sencillo) plus a 229-term theological glossary. Improves translation of biblical terminology, proper name conventions (James→Santiago for the epistle, James→Jacobo for the apostle), and religious register. Expected improvement: +3–8 SacreBLEU points on biblical text.

Both adapters are produced by the training pipeline documented in [`CLAUDE-windows.md`](./CLAUDE-windows.md). The full fine-tuning strategy, data sources, and research basis are in [`CLAUDE.md`](./CLAUDE.md) under "Fine-Tuning Strategy."

**Toggling adapters:** You can instantly compare fine-tuned vs. base model performance:

```python
# Fine-tuned inference
output_ft = model.generate(**inputs)

# Disable adapter → base model inference
model.disable_adapter_layers()
output_base = model.generate(**inputs)

# Re-enable adapter
model.enable_adapter_layers()
```

### Loading Fine-Tuned Models

After receiving LoRA adapters from the WSL training machine, place folders in project root. MLX supports loading adapters via `adapter_path=`:

```python
import mlx_lm

# --- TranslateGemma with LoRA adapter ---
model, tokenizer = mlx_lm.load(
    "mlx-community/translategemma-4b-it-4bit",
    adapter_path="./fine_tuned_gemma_mi_A"  # LoRA adapter folder
)
```

For Whisper LoRA adapters, the PyTorch/PEFT path is still needed for evaluation (mlx-whisper does not yet support LoRA directly):

```python
from transformers import WhisperForConditionalGeneration
from peft import PeftModel

base_whisper = WhisperForConditionalGeneration.from_pretrained(
    "distil-whisper/distil-large-v3", device_map="auto")
model_whisper = PeftModel.from_pretrained(base_whisper, "./fine_tuned_whisper_mi")
```

**Adapter sizes:** ~60 MB each (Whisper LoRA), ~80-120 MB (Gemma QLoRA). Transfer via USB, AirDrop, or scp.

### Running

```bash
# Activate environment
cd project_dir
source stt_env/bin/activate

# Run — 4B only (default, ~4.3 GB RAM)
python dry_run_ab.py

# Run — A/B mode with both 4B and 12B (~11.3 GB RAM)
python dry_run_ab.py --ab

# Key flags
#   --http-port 8080    HTTP server for LAN/phone access
#   --ws-port 8765      WebSocket server port
#   --vad-threshold 0.3 VAD sensitivity (0-1)
#   --gain auto         Mic gain (auto-calibrates)
#   --num-draft-tokens  Enable speculative decoding (4B drafts for 12B)

# Open displays
open http://localhost:8080/displays/audience_display.html
open http://localhost:8080/displays/ab_display.html
# Phones: scan QR on audience display or go to http://<LAN-IP>:8080/displays/mobile_display.html
```

---

## Confidence Scoring (integrated in `dry_run_ab.py`)

### Using `faster-whisper` for Segment + Word Confidence

```python
from faster_whisper import WhisperModel

model = WhisperModel("distil-whisper/distil-large-v3", device="cpu",
                     compute_type="float16")  # MPS not yet supported; use CPU

segments, info = model.transcribe(audio_path, word_timestamps=True)

for segment in segments:
    # Segment-level signals
    print(f"avg_logprob: {segment.avg_logprob:.3f}")
    print(f"no_speech_prob: {segment.no_speech_prob:.3f}")
    print(f"compression_ratio: {segment.compression_ratio:.2f}")

    # Word-level confidence
    for word in segment.words:
        print(f"  '{word.word}' prob={word.probability:.3f} "
              f"[{word.start:.2f}-{word.end:.2f}]")
```

**Note:** `faster-whisper` uses CTranslate2 which currently requires CPU on macOS. For MPS inference with confidence, use `whisper-timestamped` instead, which wraps the standard Whisper model.

### Flagging Thresholds

```python
def should_flag(segment):
    """Returns (flag_for_review: bool, auto_reject: bool, reasons: list)"""
    reasons = []
    reject = False
    flag = False

    if segment.avg_logprob < -1.0 and segment.no_speech_prob > 0.6:
        reject = True
        reasons.append("silent_hallucination")
    if segment.compression_ratio > 2.4:
        reject = True
        reasons.append("repetition_hallucination")
    if segment.avg_logprob < -0.5:
        flag = True
        reasons.append("low_confidence")
    if segment.compression_ratio > 2.0:
        flag = True
        reasons.append("high_compression")
    if any(w.probability < 0.3 for w in segment.words):
        flag = True
        reasons.append("low_word_confidence")

    return flag, reject, reasons
```

### Building the Review Queue

Score every segment, sort by composite quality, route the bottom 5–15% to human review. Log flagged segments to `metrics/confidence_flags.jsonl`:

```json
{"timestamp": "2026-02-08T10:23:45", "segment_start": 45.2, "segment_end": 48.7,
 "text": "and he said unto them", "avg_logprob": -0.62, "compression_ratio": 1.9,
 "low_conf_words": [{"word": "unto", "prob": 0.24}], "flag_reasons": ["low_confidence", "low_word_confidence"]}
```

---

## Live YouTube Caption Comparison (`tools/live_caption_monitor.py`)

### Architecture

Two parallel streams with time-aligned comparison:

```
┌─────────────────┐     ┌──────────────────┐
│  Local Mic/Audio │     │ YouTube Livestream│
│  via sounddevice │     │  via streamlink   │
└────────┬────────┘     └────────┬─────────┘
         │                       │
    ┌────▼────┐            ┌─────▼──────┐
    │ Whisper  │            │ InnerTube  │
    │  (MLX)   │            │  Captions  │
    └────┬────┘            └─────┬──────┘
         │                       │
         └───────┬───────────────┘
                 │
         ┌───────▼───────┐
         │ Time-Aligned   │
         │ WER Comparison │
         └───────┬───────┘
                 │
         ┌───────▼───────┐
         │  JSONL Logs +  │
         │  Streamlit Viz │
         └───────────────┘
```

### Fetching YouTube Captions

**During livestream** — poll InnerTube timed-text endpoint:

```python
import innertube

client = innertube.InnerTube("WEB")
# Fetch player data for live video
player = client.player(video_id="LIVE_VIDEO_ID")
# Extract caption track URLs from player response
# Poll every 5-10 seconds for new segments
```

**After stream ends** — retrieve full regenerated caption track:

```python
from youtube_transcript_api import YouTubeTranscriptApi

transcript = YouTubeTranscriptApi.get_transcript("VIDEO_ID", languages=['en'])
# Returns list of {'text': str, 'start': float, 'duration': float}
```

Or via yt-dlp:

```bash
yt-dlp --write-auto-subs --sub-lang en --skip-download "VIDEO_URL"
```

### Windowed Comparison

```python
import jiwer

def compare_window(local_text, youtube_text):
    """Compare two transcript strings for a 30-second window."""
    transforms = jiwer.Compose([
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    result = jiwer.process_words(youtube_text, local_text, 
                                  reference_transform=transforms,
                                  hypothesis_transform=transforms)
    return {
        'wer': result.wer,
        'cer': jiwer.cer(youtube_text, local_text),
        'insertions': result.insertions,
        'deletions': result.deletions,
        'substitutions': result.substitutions,
    }
```

**Interpreting results:** Cross-system WER reflects disagreement, not your true error rate. Track trends over time; flag windows where WER spikes above 20%.

---

## Translation Quality Estimation (`tools/translation_qe.py`)

### Tier 1: Always-On Scoring (~150–300ms per segment)

```python
from comet import download_model, load_from_checkpoint
from sentence_transformers import SentenceTransformer
import numpy as np

# Load models (once at startup)
cometkiwi_path = download_model("Unbabel/wmt22-cometkiwi-da")
cometkiwi = load_from_checkpoint(cometkiwi_path)
labse = SentenceTransformer("sentence-transformers/LaBSE")

def score_translation(source_en: str, translated_es: str) -> dict:
    # CometKiwi (reference-free)
    comet_input = [{"src": source_en, "mt": translated_es}]
    comet_score = cometkiwi.predict(comet_input, batch_size=1).scores[0]

    # LaBSE cross-lingual similarity
    embeddings = labse.encode([source_en, translated_es])
    labse_sim = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))

    # Length ratio (Spanish typically 15-25% longer)
    len_ratio = len(translated_es.split()) / max(len(source_en.split()), 1)
    len_ratio_ok = 0.9 <= len_ratio <= 1.6

    # Untranslated detection
    source_words = set(source_en.lower().split())
    trans_words = set(translated_es.lower().split())
    overlap = len(source_words & trans_words) / max(len(source_words), 1)
    likely_untranslated = overlap > 0.7  # High overlap = probably not translated

    return {
        'comet_qe': comet_score,
        'labse_similarity': float(labse_sim),
        'length_ratio': len_ratio,
        'length_ratio_ok': len_ratio_ok,
        'likely_untranslated': likely_untranslated,
        'tier1_pass': comet_score > 0.70 and labse_sim > 0.80 and not likely_untranslated
    }
```

### Tier 2: Back-Translation (Triggered by Tier 1 Flags)

```python
from transformers import MarianMTModel, MarianTokenizer

# Load MarianMT es→en (~75MB, fast)
marian_name = "Helsinki-NLP/opus-mt-es-en"
marian_tokenizer = MarianTokenizer.from_pretrained(marian_name)
marian_model = MarianMTModel.from_pretrained(marian_name)

def back_translate_check(source_en: str, translated_es: str) -> dict:
    # Translate Spanish back to English
    inputs = marian_tokenizer(translated_es, return_tensors="pt", padding=True)
    outputs = marian_model.generate(**inputs)
    back_en = marian_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Compare original English to back-translated English
    # (Use BERTScore or simple word overlap as a proxy)
    from jiwer import wer
    round_trip_wer = wer(source_en.lower(), back_en.lower())

    return {
        'back_translated_en': back_en,
        'round_trip_wer': round_trip_wer,
        'round_trip_pass': round_trip_wer < 0.30  # < 30% divergence
    }
```

### Latency Budget on M3 Pro (18-core GPU, 18GB, MLX)

| Component | Time | Notes |
|-----------|------|-------|
| mlx-whisper STT | ~500ms | Word timestamps add ~100-200ms (finals only) |
| MarianMT CT2 int8 (partials) | ~50ms | 3.3x faster than PyTorch |
| TranslateGemma-4B translation (finals) | ~650ms | Via mlx-lm |
| TranslateGemma-12B translation (finals) | ~1.4s | Via mlx-lm, --ab mode |
| CometKiwi scoring | ~100-200ms | |
| LaBSE similarity | ~50-100ms | |
| Simple checks | < 5ms | |
| **Partial path total** | **~580ms** | STT + MarianMT CT2 |
| **Final path total (4B)** | **~1.3s** | STT + TranslateGemma 4B |
| **Final path total (12B)** | **~1.9s** | STT + TranslateGemma 12B |

Tier 2 back-translation adds ~50-100ms (MarianMT is small and fast).

---

## Browser Displays

The primary UI uses static HTML pages served over HTTP (`--http-port 8080`) with live data pushed via WebSocket (`--ws-port 8765`). All displays auto-detect the WebSocket host for LAN connectivity.

### Display Modes

| Display | File | Purpose |
|---------|------|---------|
| **Audience** | `displays/audience_display.html` | Projector-friendly EN/ES side-by-side, fading context, fullscreen toggle, QR code overlay |
| **A/B/C Compare** | `displays/ab_display.html` | Operator view: Gemma 4B / MarianMT / 12B side-by-side with latency stats |
| **Mobile** | `displays/mobile_display.html` | Responsive phone/tablet view with model toggle + Spanish-only mode |
| **Church** | `displays/church_display.html` | Simplified church-oriented layout |

### Accessing Displays

```bash
# Local (operator)
open http://localhost:8080/displays/audience_display.html
open http://localhost:8080/displays/ab_display.html

# LAN (phones, projector)
# Scan QR code on audience display, or navigate to:
http://<LAN-IP>:8080/displays/mobile_display.html
```

### Design Notes

- Fullscreen buttons on audience and A/B/C displays (bottom-right, auto-hide)
- Scroll history in audience display (all sentences kept, scrollable, auto-scroll with pause)
- QR code overlay on audience display (click header to toggle)
- 16px+ fonts for demo readability at projection distance
- All displays work on phones via LAN without any app installation

---

## Post-Transfer Evaluation (`training/evaluate_translation.py`)

After transferring fine-tuned adapters from WSL, verify quality on Mac before live use.

### Quick Smoke Test

```python
import mlx_lm

# TranslateGemma via MLX (primary inference path)
model, tokenizer = mlx_lm.load(
    "mlx-community/translategemma-4b-it-4bit",
    adapter_path="./fine_tuned_gemma_mi_A"  # optional, if adapter exists
)

# Add EOS fix
tokenizer._eos_token_ids.add(106)  # <end_of_turn>

messages = [{"role": "user", "content": [
    {"type": "text", "source_lang_code": "en", "target_lang_code": "es",
     "text": "For God so loved the world that he gave his one and only Son."}
]}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
output = mlx_lm.generate(model, tokenizer, prompt=input_text, max_tokens=128)
print("Translation:", output)

# Whisper LoRA smoke test (PyTorch/PEFT, for evaluation only)
from transformers import WhisperForConditionalGeneration
from peft import PeftModel
base_w = WhisperForConditionalGeneration.from_pretrained("distil-whisper/distil-large-v3")
model_w = PeftModel.from_pretrained(base_w, "./fine_tuned_whisper_mi")
print("Whisper adapter loaded:", model_w.peft_config)
```

### Theological Term Spot-Check

Run a targeted check on the terms that generic MT systems get wrong:

```python
SPOT_CHECK = [
    ("Christ's atonement covers all sins.", "expiación"),
    ("The new covenant was sealed in blood.", "pacto"),
    ("Justified by grace through faith.", "gracia"),
    ("The righteousness of God is revealed.", "justicia"),
    ("The epistle of James teaches about works.", "Santiago"),
    ("James and John left their nets.", "Jacobo"),
    ("He preached about sanctification.", "santificación"),
    ("The propitiation for our sins.", "propiciación"),
]

for en_text, expected_term in SPOT_CHECK:
    # ... translate with fine-tuned model ...
    found = expected_term.lower() in translation.lower()
    print(f"{'✅' if found else '❌'} {en_text[:50]}... → expected '{expected_term}'")
```

### Batch Evaluation with SacreBLEU/chrF++/COMET

The full `training/evaluate_translation.py` script (same as WSL version) can run on Mac for independent verification. Copy `bible_data/aligned/verse_pairs_test.jsonl` (~3.1K verses) alongside the adapters.

```bash
# Install evaluation libs (if not already present)
pip install sacrebleu unbabel-comet

# Run evaluation
python training/evaluate_translation.py
```

**Expected improvement targets after fine-tuning:**
- SacreBLEU: +3–8 points over base TranslateGemma on biblical text
- chrF++: +2–5 points (character-level, captures Spanish morphology better)
- COMET: +0.02–0.05 (neural metric, most reliable)
- Theological term accuracy: 80%+ on spot-check (vs. ~40–60% for base model)

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Cold boot slowness | Pipeline pre-warms models during silence gaps; first utterance may be slower |
| Battery throttling | Plug in for A/B tests (~80% speed on battery) |
| Metal cache growing | `mx.set_cache_limit(100 * 1024 * 1024)` prevents cache growth with word_timestamps |
| TranslateGemma generates pad tokens | Add `<end_of_turn>` (id=106) to `tokenizer._eos_token_ids` — default EOS (id=1) never generates |
| High memory in A/B mode | Both models ~11.3 GB; run 4B-only (no `--ab`) if memory-constrained |
| PyTorch fp16 on MPS fails | Expected — causes inf/nan with TranslateGemma. Use MLX 4-bit (the default) |
| bitsandbytes on Mac | CUDA-only; do not use on Mac. MLX 4-bit is the correct approach |
| `faster-whisper` MPS error | CTranslate2 runs on CPU only on macOS; this is fine for confidence scoring |
| CometKiwi slow | Run on CPU (`device="cpu"` in COMET config) |
| Phone can't connect | Ensure `--http-port 8080` is accessible; check firewall allows port 8080 and 8765 |
| WebSocket drops on phone | Displays have auto-reconnect; check that both devices are on same LAN |
| Fine-tuned model not loading | Verify adapter folder has `adapter_config.json` + weights; use `adapter_path=` with mlx-lm |
| Translation quality regressed | Compare against base model; check training data quality |
| Theological terms still wrong | Run spot-check script; may need more glossary training pairs or constrained decoding |
| YouTube caption fetch fails | Livestream captions may not be available for all videos; fall back to post-stream extraction |
