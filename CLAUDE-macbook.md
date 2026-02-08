# CLAUDE-macbook.md — Mac Inference Environment Guide

> **Machine:** M3 Pro MacBook (Mac15,6), 18GB unified memory, 12-core CPU (6P+6E), 18-core GPU, Metal 4, MPS acceleration
> **Role:** Inference, live demos, quality monitoring, Streamlit UI, A/B testing
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

| Component | VRAM (4-bit) | VRAM (fp16) |
|-----------|-------------|-------------|
| Distil-Whisper large-v3 | ~1.5 GB | ~1.5 GB |
| TranslateGemma 4B | ~2.6 GB | ~8 GB |
| TranslateGemma 12B | ~6 GB | ~24 GB |
| Silero VAD | ~2 MB | ~2 MB |
| **Total (parallel 4-bit)** | **~10.1 GB** | — |
| **Headroom (18 GB)** | **~7.9 GB** | — |

With 18 GB unified memory and no iogpu wired limit, parallel mode (both Gemma models in 4-bit) fits comfortably. fp16 is only viable for the 4B model alone.

---

## Environment Setup

### Prerequisites

- macOS 14+ (Sonoma or later for full MPS support)
- Grant mic access: **System Settings → Privacy & Security → Microphone**
- Homebrew installed

### Verified Environment

- Python 3.11.11
- PyTorch 2.10.0 (MPS available, fp16 confirmed)
- bitsandbytes 0.49.1 (4-bit quantization on Apple Silicon)

### Installation

```bash
# Update Homebrew
brew update
brew install ffmpeg portaudio

# Create virtualenv
python3 -m venv stt_env
source stt_env/bin/activate

# Core inference libs (standard PyTorch includes MPS on Apple Silicon)
pip install torch torchvision torchaudio
pip install transformers accelerate
pip install sounddevice silero-vad
pip install sentencepiece protobuf
pip install bitsandbytes                # 4-bit quantization (both Gemma models in RAM)
pip install websockets                  # WebSocket server for browser display

# UI & metrics
pip install streamlit pandas streamlit-webrtc

# Evaluation & monitoring
pip install jiwer sacrebleu
pip install faster-whisper              # For confidence scores
pip install whisper-timestamped         # For word-level confidence
pip install peft                        # For loading LoRA/QLoRA adapters

# Translation QE
pip install unbabel-comet               # CometKiwi
pip install sentence-transformers       # LaBSE
pip install language_tool_python        # Spanish grammar checking

# YouTube caption monitoring
pip install youtube-transcript-api
pip install streamlink                  # Live stream audio capture
pip install innertube                   # InnerTube API access

# Back-translation
pip install sentencepiece               # MarianMT dependency
```

### First-Run Model Downloads

First run auto-downloads models from Hugging Face — use Wi-Fi:

| Model | Size | Purpose |
|-------|------|---------|
| `distil-whisper/distil-large-v3` | ~1.5 GB | STT |
| `google/translategemma-4b` | ~5 GB (4-bit) | Translation (Approach A) |
| `google/translategemma-12b` | ~12 GB (4-bit) | Translation (Approach B) |
| `Unbabel/wmt22-cometkiwi-da` | ~580 MB | Translation QE |
| `sentence-transformers/LaBSE` | ~470 MB | Cross-lingual similarity |
| `Helsinki-NLP/opus-mt-es-en` | ~75 MB | Back-translation |

**Memory note:** With 18GB unified memory, both Gemma models fit simultaneously in 4-bit (4B ~2.6 GB + 12B ~6 GB + Whisper ~1.5 GB = ~10 GB), leaving ~8 GB for macOS and buffers. Always use `load_in_4bit=True` on Gemma models. Monitor with `torch.mps.driver_allocated_memory()`.

---

## Core Pipeline (`stt_pipeline.py`)

### Key Constants

```python
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.2    # 200ms for partials (set to 3.0 for full-sentence mode)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
```

### Pipeline Flow

1. **Audio Input** → `sounddevice.InputStream` at 16kHz mono
2. **VAD** → Silero VAD (small model, threshold 0.5) filters silence
3. **STT** → Distil-Whisper `distil-large-v3` with float16 on MPS
4. **Translation** → TranslateGemma (4B or 12B) with 4-bit quantization
5. **Metrics** → Logs approach, latency, WER, BLEU, text pairs → CSV export

Both pipelines process each audio chunk in parallel via `asyncio.gather`.

### Understanding the Fine-Tuned Adapters

The Mac loads LoRA/QLoRA adapters trained on the Windows desktop. These are small (~60–120 MB) parameter patches that customize the base models for church content without modifying the original weights.

**Whisper LoRA adapter** (`fine_tuned_whisper_mi/`): Trained on 20–50 hours of church sermon audio via pseudo-labeling. Adapts both encoder (reverberant church acoustics, PA system artifacts) and decoder (theological vocabulary: "sanctification," "propitiation," biblical proper names). Expected WER reduction: 10–30% relative on church audio.

**TranslateGemma QLoRA adapter** (`fine_tuned_gemma_mi_A/` or `_B/`): Trained on ~155K Bible verse pairs from public-domain translations (KJV/ASV/WEB/BBE/YLT ↔ RVR1909/Español Sencillo) plus a ~200-term theological glossary. Improves translation of biblical terminology, proper name conventions (James→Santiago for the epistle, James→Jacobo for the apostle), and religious register. Expected improvement: +3–8 SacreBLEU points on biblical text.

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

After receiving LoRA/QLoRA adapters from the WSL training machine, place folders in project root:

```python
import os
from transformers import WhisperForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Whisper LoRA adapter ---
if os.path.exists("./fine_tuned_whisper_mi"):
    base_whisper = WhisperForConditionalGeneration.from_pretrained(
        "distil-whisper/distil-large-v3", device_map="auto")
    self.stt_model = PeftModel.from_pretrained(base_whisper, "./fine_tuned_whisper_mi")
    print("✅ Loaded fine-tuned Whisper (church audio LoRA)")

# --- TranslateGemma QLoRA adapter ---
gemma_path = "./fine_tuned_gemma_mi_A" if approach == 'A' else "./fine_tuned_gemma_mi_B"
if os.path.exists(gemma_path):
    base_gemma = AutoModelForCausalLM.from_pretrained(
        "google/translategemma-4b-it",   # or 12b-it for approach B
        device_map="auto",
        load_in_4bit=True               # Must match training quantization
    )
    self.translate_model = PeftModel.from_pretrained(base_gemma, gemma_path)
    self.translate_tokenizer = AutoTokenizer.from_pretrained(gemma_path)
    print(f"✅ Loaded fine-tuned TranslateGemma ({approach}, biblical QLoRA)")
```

**Adapter sizes:** ~60 MB each (Whisper LoRA), ~80–120 MB (Gemma QLoRA). Transfer via USB, AirDrop, or scp.

### Running

```bash
# Activate environment
cd project_dir
source stt_env/bin/activate

# CLI A/B test
python stt_pipeline.py

# Streamlit UI
streamlit run stt_ui.py
# Opens at localhost:8501
```

---

## Confidence Scoring (`confidence_scorer.py`)

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

## Live YouTube Caption Comparison (`live_caption_monitor.py`)

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
    │  (MPS)   │            │  Captions  │
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

## Translation Quality Estimation (`translation_qe.py`)

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

### Latency Budget on M3 Pro (18-core GPU, 18GB)

| Component | Time |
|-----------|------|
| TranslateGemma-4B translation | ~500–2000ms |
| CometKiwi scoring | ~100–200ms |
| LaBSE similarity | ~50–100ms |
| Simple checks | < 5ms |
| **Total (Tier 1)** | **~700–2300ms** |

Tier 2 back-translation adds ~50–100ms (MarianMT is small and fast).

---

## Streamlit Dashboard (`stt_ui.py`)

### Features

- **Tabs:** Live Output A, Live Output B, Metrics Dashboard, YouTube Comparison
- **Sidebar:** Approach selector, optional ground truth input, monitoring toggles
- **Metrics tab:** Real-time latency charts, WER trends, translation QE scores, export button
- **YouTube tab:** Side-by-side transcript diff, rolling cross-system WER chart

### Running

```bash
streamlit run stt_ui.py
# Opens at localhost:8501
```

For true live mic streaming (instead of simulated chunks), add `streamlit-webrtc`:

```bash
pip install streamlit-webrtc
```

### Design Notes

- Green badges for Approach A (fast), blue for Approach B (accurate)
- 16px+ fonts for demo readability at coffee shop distance
- Responsive layout for MacBook screen
- Export buttons for all metric tables → CSV for Numbers/Excel analysis

---

## Post-Transfer Evaluation (`evaluate_translation.py`)

After transferring fine-tuned adapters from WSL, verify quality on Mac before live use.

### Quick Smoke Test

```python
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer

# Whisper LoRA smoke test
base_w = WhisperForConditionalGeneration.from_pretrained("distil-whisper/distil-large-v3")
model_w = PeftModel.from_pretrained(base_w, "./fine_tuned_whisper_mi")
print("✅ Whisper adapter loaded:", model_w.peft_config)

# TranslateGemma QLoRA smoke test
base_g = AutoModelForCausalLM.from_pretrained(
    "google/translategemma-4b-it", device_map="auto", load_in_4bit=True)
model_g = PeftModel.from_pretrained(base_g, "./fine_tuned_gemma_mi_A")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_gemma_mi_A")
print("✅ TranslateGemma adapter loaded:", model_g.peft_config)

# Quick translation test
messages = [{"role": "user", "content": [
    {"type": "text", "source_lang_code": "en", "target_lang_code": "es",
     "text": "For God so loved the world that he gave his one and only Son."}
]}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model_g.device)
import torch
with torch.no_grad():
    output = model_g.generate(**inputs, max_new_tokens=128)
print("Translation:", tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
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

The full `evaluate_translation.py` script (same as WSL version) can run on Mac for independent verification. Copy `bible_data/aligned/verse_pairs_test.jsonl` (~3.1K verses) alongside the adapters.

```bash
# Install evaluation libs (if not already present)
pip install sacrebleu unbabel-comet

# Run evaluation
python evaluate_translation.py
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
| Cold boot slowness | Warm models: `torch.mps.empty_cache()` at startup |
| Battery throttling | Plug in for A/B tests (~80% speed on battery) |
| MPS not available | Check `torch.backends.mps.is_available()` — CPU fallback adds ~1.5x latency |
| High memory on Approach B | Ensure `load_in_4bit=True` on Gemma 12B; don't run A+B simultaneously |
| `faster-whisper` MPS error | CTranslate2 doesn't support MPS yet; use CPU or switch to `whisper-timestamped` |
| CometKiwi slow on MPS | Run on CPU (`device="cpu"` in COMET config); MPS support is experimental |
| Streamlit audio lag | Use `streamlit-webrtc` for direct mic access instead of `st.audio` widget |
| Fine-tuned model not loading | Verify adapter folder has `adapter_config.json` + `adapter_model.safetensors`; check PEFT version matches WSL |
| QLoRA adapter requires 4-bit | Must load base model with `load_in_4bit=True` to match training quantization |
| Translation quality regressed | Compare against base model (disable adapter: `model.disable_adapter_layers()`); check training data quality |
| Theological terms still wrong | Run spot-check script; may need more glossary training pairs or constrained decoding |
| YouTube caption fetch fails | Livestream captions may not be available for all videos; fall back to post-stream extraction |
