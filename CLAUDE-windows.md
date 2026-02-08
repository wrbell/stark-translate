# CLAUDE-windows.md — Windows/WSL Training Environment Guide

> **Machine:** Windows Desktop, WSL2/Ubuntu, NVIDIA A2000 Ada (16GB VRAM), 64GB RAM
> **Role:** Audio preprocessing, data quality assessment, fine-tuning (Whisper + Gemma LoRA), feedback loop retraining
> **Parent doc:** [`CLAUDE.md`](./CLAUDE.md)

---

## Environment Setup

### WSL2 + CUDA Installation

```bash
# Enable WSL2 (PowerShell as Admin)
wsl --install -d Ubuntu-24.04

# Inside WSL — NVIDIA drivers should pass through from Windows
nvidia-smi  # Verify GPU is visible (A2000 Ada, 16GB)

# Install CUDA toolkit 12.x
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4

# Verify
nvcc --version
```

### Python Environment

```bash
sudo apt-get install python3.12 python3.12-venv ffmpeg

python3.12 -m venv stt_train_env
source stt_train_env/bin/activate

# PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Training libs
pip install transformers accelerate
pip install peft                        # LoRA
pip install trl                         # SFTTrainer for QLoRA fine-tuning
pip install datasets                    # HuggingFace datasets
pip install bitsandbytes                # 4-bit quantization (QLoRA) + 8-bit optimizers
pip install sentencepiece protobuf

# Audio preprocessing
pip install noisereduce pyloudnorm pydub
pip install demucs                      # Source separation (CUDA-accelerated)
pip install silero-vad
pip install pyannote-audio              # Speaker diarization (CUDA)
pip install inaSpeechSegmenter          # Speech/music classification

# Evaluation
pip install jiwer sacrebleu
pip install unbabel-comet               # COMET translation quality metric

# Data collection
sudo apt install yt-dlp
```

### Shared Folder for Model Transfer

WSL mounts Windows drives at `/mnt/c/`. Set up a shared project folder:

```bash
# Create shared directory (accessible from both Windows and WSL)
mkdir -p /mnt/c/Users/YourName/Projects/stt_bilingual

# Symlink for convenience inside WSL
ln -s /mnt/c/Users/YourName/Projects/stt_bilingual ~/stt_project
```

All fine-tuned model outputs go to this shared folder. Copy LoRA adapter folders to the Mac via USB, AirDrop, or network share.

---

## Phase 1: Data Collection

### Download Church Audio

```bash
cd ~/stt_project

# List all available videos with dates
yt-dlp --flat-playlist --dateafter 20210201 \
  --print "%(title)s - %(duration_string)s - %(url)s" \
  "https://www.youtube.com/c/StarkRoadGospelHall/videos" > video_list.txt

# Review video_list.txt, pick 10-20 representative URLs
# Include variety: different speakers, singing segments, Q&A sessions
# Save selected URLs to urls.txt (one per line)

# Download audio only (highest quality)
mkdir -p stark_data/raw
yt-dlp -f bestaudio --extract-audio --audio-format wav \
  --batch-file urls.txt -P stark_data/raw/

# Also grab YouTube's auto-captions for baseline comparison
yt-dlp --write-auto-subs --sub-lang en --skip-download \
  --batch-file urls.txt -P stark_data/raw/
```

**Sampling strategy:** Aim for 10–20 hours total. Include recent sermons (last 2 years) for current speaker habits, a few older ones for variety, and at least 2–3 with noticeable background noise to test robustness.

---

## Phase 2: Audio Preprocessing (`preprocess_audio.py`)

The 10-step pipeline. Run each file through sequentially.

### Step 1–2: Format Conversion & Initial Quality Gate

```python
import subprocess
import numpy as np
import soundfile as sf
import os

def convert_to_whisper_format(input_path, output_path):
    """Convert to 16kHz mono WAV."""
    subprocess.run([
        "ffmpeg", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-f", "wav",
        output_path, "-y"
    ], check=True)

def initial_quality_gate(audio_path, min_snr_db=10.0):
    """Reject files with very low SNR or clipping."""
    audio, sr = sf.read(audio_path)

    # Clipping detection
    clip_ratio = np.mean(np.abs(audio) > 0.99)
    if clip_ratio > 0.01:
        return False, f"Clipping: {clip_ratio:.2%} of samples"

    # Simple SNR estimate (energy in top 90th percentile vs bottom 10th)
    frame_energy = np.array([
        np.mean(audio[i:i+1600]**2)
        for i in range(0, len(audio)-1600, 1600)
    ])
    speech_energy = np.percentile(frame_energy[frame_energy > 0], 90)
    noise_energy = np.percentile(frame_energy[frame_energy > 0], 10)
    snr = 10 * np.log10(speech_energy / max(noise_energy, 1e-10))

    if snr < min_snr_db:
        return False, f"Low SNR: {snr:.1f} dB"

    return True, f"OK (SNR: {snr:.1f} dB, clip: {clip_ratio:.4%})"
```

### Step 3–4: Segment Classification & Source Separation

```python
from inaSpeechSegmenter import Segmenter

def classify_segments(audio_path):
    """Classify audio into speech/music/noise regions."""
    seg = Segmenter()
    segments = seg(audio_path)
    # Returns list of (label, start_time, end_time)
    # Labels: 'speech', 'music', 'noise', 'noEnergy'
    return segments

def separate_vocals(audio_path, output_dir):
    """Use demucs to isolate vocals from music segments."""
    subprocess.run([
        "python", "-m", "demucs",
        "--two-stems", "vocals",
        "-n", "htdemucs",
        "-o", output_dir,
        audio_path
    ], check=True)
    # Output: output_dir/htdemucs/filename/vocals.wav
```

### Step 5–6: Denoise & Normalize

```python
import noisereduce as nr
import pyloudnorm as pyln

def denoise_audio(audio, sr=16000):
    """Conservative spectral gating — don't over-clean."""
    return nr.reduce_noise(
        y=audio, sr=sr,
        prop_decrease=0.7,     # Conservative: 0.6-0.8
        n_fft=512,
        stationary=False       # Non-stationary for church reverb
    )

def bandpass_filter(input_path, output_path):
    """Remove sub-bass rumble and high-freq hiss."""
    subprocess.run([
        "ffmpeg", "-i", input_path,
        "-af", "highpass=f=80,lowpass=f=8000",
        output_path, "-y"
    ], check=True)

def normalize_loudness(audio, sr=16000, target_lufs=-16.0):
    """EBU R128 loudness normalization."""
    meter = pyln.Meter(sr)
    current_lufs = meter.integrated_loudness(audio)
    normalized = pyln.normalize.loudness(audio, current_lufs, target_lufs)
    # True peak limiting
    peak = np.max(np.abs(normalized))
    if peak > 0.891:  # -1 dBTP
        normalized = normalized * (0.891 / peak)
    return normalized
```

### Step 7–8: VAD Chunking & Diarization

```python
import torch

def vad_chunk(audio_path, min_dur=1.0, max_dur=30.0, padding=0.1):
    """Silero VAD-based chunking with WhisperX-style merge."""
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
    (get_speech_timestamps, _, read_audio, _, _) = utils

    wav = read_audio(audio_path, sampling_rate=16000)
    timestamps = get_speech_timestamps(wav, model, sampling_rate=16000,
                                        min_speech_duration_ms=500,
                                        min_silence_duration_ms=300)

    chunks = []
    for ts in timestamps:
        start = max(0, ts['start'] / 16000 - padding)
        end = ts['end'] / 16000 + padding
        duration = end - start

        if duration < min_dur:
            # Try merging with next chunk
            continue
        if duration > max_dur:
            # Split at minimum energy point
            # (simplified: split at midpoint)
            mid = (start + end) / 2
            chunks.append((start, mid))
            chunks.append((mid, end))
        else:
            chunks.append((start, end))

    return chunks

def diarize_speakers(audio_path):
    """Optional: identify primary speaker via pyannote."""
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                         use_auth_token="YOUR_HF_TOKEN")
    diarization = pipeline(audio_path)
    # Returns timeline of speaker labels
    return diarization
```

### Step 9–10: Final Quality Gate & Export

```python
def final_quality_gate(chunk_audio, sr=16000, min_snr=15.0):
    """Reject chunks that don't meet training quality standards."""
    duration = len(chunk_audio) / sr
    if duration < 1.0 or duration > 30.0:
        return False, "duration_out_of_range"

    # Silence ratio
    energy = chunk_audio ** 2
    silence_ratio = np.mean(energy < 1e-6)
    if silence_ratio > 0.5:
        return False, "too_much_silence"

    # SNR check on chunk
    frame_energy = np.array([
        np.mean(chunk_audio[i:i+800]**2)
        for i in range(0, len(chunk_audio)-800, 800)
    ])
    if len(frame_energy[frame_energy > 0]) < 2:
        return False, "insufficient_audio"

    speech_e = np.percentile(frame_energy[frame_energy > 0], 80)
    noise_e = np.percentile(frame_energy[frame_energy > 0], 10)
    snr = 10 * np.log10(speech_e / max(noise_e, 1e-10))
    if snr < min_snr:
        return False, f"low_snr_{snr:.1f}"

    return True, "pass"
```

---

## Phase 3: Data Quality Assessment (`assess_quality.py`)

Before fine-tuning, establish a WER baseline on 50–100 randomly sampled segments.

```python
import random
import json
import jiwer
from pathlib import Path

def sample_for_assessment(transcript_dir, n=100):
    """Randomly sample segments for manual review."""
    all_segments = []
    for jf in Path(transcript_dir).glob("*.json"):
        data = json.load(open(jf))
        for seg in data.get("segments", []):
            all_segments.append({
                'audio_file': data['audio_path'],
                'start': seg['timestamp'][0],
                'end': seg['timestamp'][1],
                'auto_text': seg['text'],
            })
    sample = random.sample(all_segments, min(n, len(all_segments)))

    # Export for manual transcription
    with open("assessment_sample.jsonl", "w") as f:
        for s in sample:
            f.write(json.dumps(s) + "\n")

    print(f"Sampled {len(sample)} segments. Manually transcribe 'manual_text' field.")
    return sample

def compute_baseline_wer(assessment_path):
    """After manual transcription, compute WER."""
    auto_texts, manual_texts = [], []
    for line in open(assessment_path):
        seg = json.loads(line)
        if 'manual_text' in seg and seg['manual_text']:
            auto_texts.append(seg['auto_text'])
            manual_texts.append(seg['manual_text'])

    overall_wer = jiwer.wer(manual_texts, auto_texts)
    print(f"Baseline WER: {overall_wer:.1%} on {len(auto_texts)} segments")
    print(f"  < 10%: Use directly with confidence filtering")
    print(f"  10-20%: Filter worst segments by avg_logprob")
    print(f"  20-30%: Weakly supervised pretrain + fine-tune on clean subset")
    print(f"  > 30%: Re-transcribe everything with Whisper large-v3")
    return overall_wer
```

**Faster alternative:** Run the same audio through 2–3 ASR systems (Whisper, YouTube, one other). Segments where all produce average CER < 5% are reliably high-quality — use these directly.

---

## Phase 4: Transcription (`transcribe_church.py`)

Generate clean labels using Whisper large-v3 (not YouTube auto-captions).

```python
import glob, json, os
from transformers import pipeline
import torch

def transcribe_all(data_dir="stark_data/cleaned", output_dir="stark_data/transcripts"):
    """Transcribe all cleaned audio with Whisper large-v3 on CUDA."""
    os.makedirs(output_dir, exist_ok=True)

    stt_pipe = pipeline(
        "automatic-speech-recognition",
        model="distil-whisper/distil-large-v3",  # Or "openai/whisper-large-v3" for max quality
        device="cuda",
        torch_dtype=torch.float16
    )

    for audio_file in sorted(glob.glob(os.path.join(data_dir, "*.wav"))):
        result = stt_pipe(audio_file, return_timestamps=True,
                          chunk_length_s=30, stride_length_s=5)

        transcript = {
            "audio_path": audio_file,
            "text": result["text"],
            "segments": result["chunks"]
        }

        out_path = os.path.join(output_dir,
                                os.path.basename(audio_file).replace(".wav", ".json"))
        json.dump(transcript, open(out_path, "w"), indent=2)
        print(f"Transcribed {os.path.basename(audio_file)}: "
              f"{len(result['chunks'])} segments, {len(result['text'])} chars")

if __name__ == "__main__":
    transcribe_all()
```

---

## Phase 4b: Biblical Parallel Corpus Preparation (`prepare_bible_corpus.py`)

Build the ~155K verse-pair training set for TranslateGemma fine-tuning. All sources are public domain or CC-licensed.

### Download & Align

```python
from datasets import load_dataset
import json, sqlite3, os

def download_biblenlp():
    """Primary source: BibleNLP corpus on HuggingFace (833 languages, CC-BY-4.0)."""
    ds = load_dataset("bible-nlp/biblenlp-corpus", languages=["eng", "spa"])
    return ds

def download_helsinki():
    """Alternative: Helsinki-NLP bible_para (Christodoulopoulos/Steedman, CC0-1.0)."""
    ds = load_dataset("Helsinki-NLP/bible_para", lang1="en", lang2="es")
    return ds

def align_scrollmapper(db_dir="bible_data/scrollmapper"):
    """
    scrollmapper/bible_databases: SQL/JSON/CSV with numeric verse IDs.
    Verse ID format: BBCCCVVV (e.g., 01001001 = Genesis 1:1).
    JOIN on verse ID gives exact alignment across translations.
    """
    pairs = []
    # English translations: t_kjv, t_asv, t_web, t_bbe, t_ylt
    # Spanish translations: t_rvr (RVR1909)
    en_tables = ["t_kjv", "t_asv", "t_web", "t_bbe", "t_ylt"]
    es_tables = ["t_rvr"]

    conn = sqlite3.connect(os.path.join(db_dir, "bible-sqlite.db"))

    for en_table in en_tables:
        for es_table in es_tables:
            query = f"""
                SELECT e.t AS en_text, s.t AS es_text, e.id AS verse_id
                FROM {en_table} e
                INNER JOIN {es_table} s ON e.id = s.id
                WHERE e.t IS NOT NULL AND s.t IS NOT NULL
                  AND LENGTH(e.t) > 5 AND LENGTH(s.t) > 5
            """
            cursor = conn.execute(query)
            for row in cursor:
                pairs.append({
                    "en": row[0].strip(),
                    "es": row[1].strip(),
                    "verse_id": row[2],
                    "en_source": en_table,
                    "es_source": es_table,
                })

    conn.close()
    return pairs  # ~155K pairs from 5 EN × 1 ES (add Español Sencillo separately)


def export_training_jsonl(pairs, output_path="bible_data/aligned/verse_pairs.jsonl"):
    """Export as JSONL for HuggingFace Trainer."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Stratified split: 90% train, 10% test (stratify by OT/NT and book genre)
    # OT books 1-39, NT books 40-66
    train, test = [], []
    for p in pairs:
        book = int(str(p["verse_id"])[:2])
        if hash(p["verse_id"]) % 10 == 0:  # ~10% holdout
            test.append(p)
        else:
            train.append(p)

    for split, data in [("train", train), ("test", test)]:
        path = output_path.replace(".jsonl", f"_{split}.jsonl")
        with open(path, "w") as f:
            for p in data:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Exported {len(data)} pairs to {path}")


if __name__ == "__main__":
    pairs = align_scrollmapper()
    print(f"Total verse pairs: {len(pairs)}")
    export_training_jsonl(pairs)
```

### Theological Glossary (`build_glossary.py`)

```python
"""
Build a ~200-500 term EN→ES theological glossary for constrained decoding.
Terms are injected as training examples AND used for soft constraint augmentation.
"""

THEOLOGICAL_GLOSSARY = {
    # Soteriology (salvation)
    "atonement": "expiación",
    "propitiation": "propiciación",
    "redemption": "redención",
    "justification": "justificación",
    "sanctification": "santificación",
    "salvation": "salvación",
    "grace": "gracia",
    "faith": "fe",
    "repentance": "arrepentimiento",
    "forgiveness": "perdón",
    "reconciliation": "reconciliación",

    # Theology proper
    "righteousness": "justicia",
    "holiness": "santidad",
    "sovereignty": "soberanía",
    "omnipotence": "omnipotencia",
    "trinity": "trinidad",
    "incarnation": "encarnación",
    "resurrection": "resurrección",
    "ascension": "ascensión",

    # Ecclesiology & worship
    "covenant": "pacto",           # Protestant convention (vs. "alianza" Catholic)
    "congregation": "congregación",
    "fellowship": "comunión",
    "baptism": "bautismo",
    "communion": "comunión",       # Context-dependent: also "Lord's Supper" = "Cena del Señor"
    "tithe": "diezmo",
    "offering": "ofrenda",
    "worship": "adoración",
    "praise": "alabanza",
    "prayer": "oración",
    "sermon": "sermón",
    "gospel": "evangelio",
    "scripture": "escritura",
    "prophecy": "profecía",
    "parable": "parábola",
    "epistle": "epístola",
    "psalm": "salmo",
    "hymn": "himno",

    # Eschatology
    "rapture": "arrebatamiento",
    "tribulation": "tribulación",
    "millennium": "milenio",
    "judgment": "juicio",
    "eternal life": "vida eterna",

    # Key proper names (EN → ES Bible convention)
    "James (apostle)": "Jacobo",
    "James (epistle)": "Santiago",
    "John": "Juan",
    "Peter": "Pedro",
    "Paul": "Pablo",
    "Matthew": "Mateo",
    "Luke": "Lucas",
    "Mark": "Marcos",
    "Moses": "Moisés",
    "Abraham": "Abraham",
    "Isaiah": "Isaías",
    "Jeremiah": "Jeremías",
    "Ezekiel": "Ezequiel",
    "Daniel": "Daniel",
}

def create_glossary_training_pairs(glossary=THEOLOGICAL_GLOSSARY):
    """Convert glossary to short training examples for TranslateGemma."""
    pairs = []
    for en_term, es_term in glossary.items():
        # Direct term pair
        pairs.append({"en": en_term, "es": es_term})
        # In-sentence examples
        pairs.append({
            "en": f"The pastor spoke about {en_term} in today's sermon.",
            "es": f"El pastor habló sobre {es_term} en el sermón de hoy."
        })
    return pairs

def augment_with_soft_constraints(verse_pairs, glossary=THEOLOGICAL_GLOSSARY):
    """
    Soft constraint training (Dinu et al., 2019):
    Append target terminology to source sentences during training.
    Model learns to use glossary terms without hard constraints at inference.
    """
    augmented = []
    for pair in verse_pairs:
        en_lower = pair["en"].lower()
        constraints = []
        for en_term, es_term in glossary.items():
            if en_term.lower() in en_lower:
                constraints.append(f"{en_term}={es_term}")
        if constraints:
            # Append constraints to source
            augmented.append({
                "en": f"{pair['en']} [GLOSSARY: {', '.join(constraints)}]",
                "es": pair["es"],
            })
        else:
            augmented.append(pair)
    return augmented
```

---

## Phase 4c: Data Augmentation for Biblical Text

Three techniques expand the effective training set beyond the raw ~155K verse pairs:

### Multi-Reference Pairing

Using the same source verse with different target translations teaches the model that multiple valid translations exist:

```python
def create_multi_reference_pairs(scrollmapper_pairs):
    """Group by verse_id, create (en_variant, es) pairs for all combinations."""
    from collections import defaultdict
    by_verse = defaultdict(lambda: {"en": set(), "es": set()})
    
    for p in scrollmapper_pairs:
        by_verse[p["verse_id"]]["en"].add(p["en"])
        by_verse[p["verse_id"]]["es"].add(p["es"])
    
    expanded = []
    for vid, texts in by_verse.items():
        for en in texts["en"]:
            for es in texts["es"]:
                expanded.append({"en": en, "es": es, "verse_id": vid})
    return expanded
```

### Back-Translation Augmentation

Use base TranslateGemma to translate Spanish verses back to English, creating synthetic (back-translated English, original Spanish) pairs:

```python
def back_translate_augment(es_verses, model, tokenizer, batch_size=16):
    """Generate synthetic EN from real ES to expand training diversity."""
    synthetic_pairs = []
    for es_verse in es_verses:
        messages = [{"role": "user", "content": [
            {"type": "text", "source_lang_code": "es",
             "target_lang_code": "en", "text": es_verse}
        ]}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=256)
        back_en = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        synthetic_pairs.append({"en": back_en, "es": es_verse})
    return synthetic_pairs
```

---

## WSL2-Specific Training Notes

Critical optimizations for training on WSL2 with the A2000 Ada:

### GPU Driver Configuration

Do **not** install the NVIDIA Linux driver inside WSL2. The Windows host driver handles GPU passthrough. Install only the CUDA toolkit (without driver component):

```bash
# Correct: install only toolkit
sudo apt-get install cuda-toolkit-12-4

# WRONG — do NOT do this:
# sudo apt-get install nvidia-driver-xxx
```

### Filesystem Performance

Store all training data on the Linux native filesystem, not the Windows mount:

```bash
# FAST — use for all training data, checkpoints, cached models
~/stt_project/stark_data/
~/stt_project/bible_data/
~/.cache/huggingface/

# SLOW — only use for final adapter output (to share with Mac)
/mnt/c/Users/YourName/Projects/fine_tuned_*/
```

The `/mnt/c/` path uses 9P protocol with significant overhead. I/O-heavy training on `/mnt/c/` can be **5–10× slower** than native Linux paths.

### GPU Shared Memory Monitoring

WSL2 can silently spill GPU memory to system RAM via shared memory, crippling throughput:

```bash
# Monitor during training — "shared" column should stay near zero
watch -n 1 nvidia-smi

# Set allocation config to mitigate
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

If shared memory usage climbs during training, reduce batch size or model size.

### FlashAttention-2

The Ada architecture (compute capability 8.9) fully supports FlashAttention-2. Enable it for faster training with lower memory:

```python
model = WhisperForConditionalGeneration.from_pretrained(
    "distil-whisper/distil-large-v3",
    attn_implementation="flash_attention_2",  # Requires flash-attn package
    torch_dtype=torch.bfloat16,
)
```

Install: `pip install flash-attn --no-build-isolation`

---

## Phase 5: Fine-Tuning

### Whisper LoRA (`train_whisper.py`) — Shared for A/B

Apply LoRA to **both encoder and decoder**. Encoder adapts to acoustic domain (reverb, PA coloring, background noise); decoder learns domain vocabulary patterns. Research from LoRA-Whisper (Interspeech 2024) confirms encoder LoRA is critical for acoustic adaptation.

```python
from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq)
from datasets import load_dataset, Audio
from peft import LoraConfig, get_peft_model
import torch

def fine_tune_whisper(dataset_path="stark_data/cleaned",
                      output_dir="/mnt/c/Users/YourName/Projects/fine_tuned_whisper_mi"):
    """LoRA fine-tuning on A2000 Ada 16GB VRAM.
    
    Research-validated config:
    - r=32, α=64 (most commonly validated; ATC paper found α=256 worth exploring)
    - Both encoder + decoder (encoder for acoustic adaptation, decoder for vocab)
    - bf16 (Ada arch supports natively) with gradient checkpointing
    - ~8-10 GB VRAM, fits comfortably on A2000 Ada
    """
    processor = WhisperProcessor.from_pretrained("distil-whisper/distil-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained(
        "distil-whisper/distil-large-v3",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_8bit=True              # Quantize base model for LoRA
    )

    # LoRA config — encoder + decoder, ~1-2% trainable parameters
    # S2-LoRA paper: v_proj and out_proj capture more important adaptation
    # than q_proj and k_proj; prioritize value/output projections
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],  # Minimum; extend to k_proj, out_proj, fc1, fc2 for max
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Should show ~1-2% trainable

    # Load dataset
    dataset = load_dataset("audiofolder", data_dir=dataset_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    dataset = dataset.map(prepare_dataset,
                          remove_columns=dataset.column_names["train"],
                          num_proc=4)

    # Training args optimized for A2000 Ada 16GB
    # bf16 preferred over fp16 on Ada architecture (compute capability 8.9)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,     # Effective batch = 16
        gradient_checkpointing=True,       # Saves ~30% VRAM
        num_train_epochs=3,
        learning_rate=1e-4,                # Standard LoRA rate (higher than full FT)
        bf16=True,                         # Ada natively supports BF16
        optim="adamw_bnb_8bit",            # 8-bit optimizer states
        dataloader_pin_memory=True,
        logging_steps=10,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        predict_with_generate=True,
        generation_max_length=225,
        warmup_steps=500,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=DataCollatorForSeq2Seq(processor.tokenizer, model=model),
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    print(f"Whisper LoRA adapters saved to {output_dir}")

if __name__ == "__main__":
    fine_tune_whisper()
```

### TranslateGemma QLoRA (`train_gemma.py`) — Biblical Domain Adaptation

TranslateGemma (Jan 2026, built on Gemma 3) supports 55 languages / ~500 pairs including EN→ES. The 4B variant loads at ~2.6 GB in 4-bit via bitsandbytes. Must follow its exact chat template with `source_lang_code` / `target_lang_code` fields.

**Data:** ~155K Bible verse pairs from `prepare_bible_corpus.py` + ~1K theological glossary pairs from `build_glossary.py`. With sequence packing, Bible verses (rarely >200 tokens) pack efficiently.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
import torch, json

def fine_tune_gemma(approach='A',
                    bible_data="bible_data/aligned/verse_pairs_train.jsonl",
                    glossary_data="bible_data/glossary/glossary_pairs.jsonl",
                    output_base="/mnt/c/Users/YourName/Projects"):
    """QLoRA fine-tuning for TranslateGemma on biblical text.
    
    Research-validated config (arXiv:2402.15061 — Domain-specific MT with LLMs):
    - r=16 for domain adaptation (validated in paper)
    - target "all-linear" for best quality per QLoRA findings
    - 4-bit NF4 quantization via bitsandbytes
    - Paged AdamW 32-bit optimizer
    - Sequence packing for short Bible verses
    
    VRAM: ~10-12 GB for 4B, ~14-15 GB for 12B (tight on A2000 Ada)
    """
    model_name = "google/translategemma-4b-it" if approach == 'A' else "google/translategemma-12b-it"
    output_dir = f"{output_base}/fine_tuned_gemma_mi_{approach}"

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # QLoRA config — target all linear layers for best domain adaptation
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
    )

    # Load Bible + glossary data
    bible_ds = load_dataset("json", data_files=bible_data, split="train")
    if glossary_data:
        glossary_ds = load_dataset("json", data_files=glossary_data, split="train")
        from datasets import concatenate_datasets
        full_ds = concatenate_datasets([bible_ds, glossary_ds]).shuffle(seed=42)
    else:
        full_ds = bible_ds.shuffle(seed=42)

    def format_for_translategemma(example):
        """Format using TranslateGemma's required chat template."""
        messages = [
            {"role": "user", "content": [
                {"type": "text",
                 "source_lang_code": "en",
                 "target_lang_code": "es",
                 "text": example["en"]}
            ]},
            {"role": "assistant", "content": example["es"]}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    full_ds = full_ds.map(format_for_translategemma,
                          remove_columns=full_ds.column_names)

    # Training config — optimized for A2000 Ada 16GB
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1 if '12b' in model_name else 2,
        gradient_accumulation_steps=8 if '12b' in model_name else 4,
        learning_rate=2e-4,
        num_train_epochs=3,
        gradient_checkpointing=True,
        bf16=True,
        max_seq_length=512,             # Bible verses rarely exceed 200 tokens
        packing=True,                   # Pack multiple short verses per sequence
        optim="paged_adamw_32bit",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=full_ds,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"TranslateGemma QLoRA adapters ({approach}) saved to {output_dir}")


if __name__ == "__main__":
    import sys
    approach = sys.argv[1] if len(sys.argv) > 1 else 'A'
    fine_tune_gemma(approach)
    # Run: python train_gemma.py A   (4B, ~8-12 hrs)
    # Run: python train_gemma.py B   (12B, ~18-27 hrs, tight on VRAM)
```

### MarianMT Fallback — Full Fine-Tune (`train_marian.py`)

If TranslateGemma QLoRA proves too heavy or bitsandbytes causes issues on WSL, `Helsinki-NLP/opus-mt-en-es` (298 MB encoder-decoder, trained partly on OPUS Bible corpus) is small enough for full fine-tuning with no LoRA needed. Much faster iteration, lower quality ceiling.

```python
from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def fine_tune_marian(bible_data="bible_data/aligned/verse_pairs_train.jsonl",
                     output_dir="/mnt/c/Users/YourName/Projects/fine_tuned_marian_mi"):
    model_name = "Helsinki-NLP/opus-mt-en-es"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)  # Only 298MB, full FT is fine

    dataset = load_dataset("json", data_files=bible_data, split="train")

    def tokenize(example):
        inputs = tokenizer(example["en"], truncation=True, max_length=128)
        targets = tokenizer(text_target=example["es"], truncation=True, max_length=128)
        inputs["labels"] = targets["input_ids"]
        return inputs

    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,    # Small model = big batches
        num_train_epochs=5,
        learning_rate=5e-5,
        bf16=True,
        logging_steps=100,
        save_steps=1000,
    )

    Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer).train()
    model.save_pretrained(output_dir)
    print(f"MarianMT fine-tuned model saved to {output_dir}")
```

---

## Phase 6: Active Learning Feedback Loop

### Cycle Structure

```
┌──────────────────────────────────────────────────────────┐
│  Cycle N (repeat 3-5 times until WER gains < 1-2%)      │
│                                                          │
│  1. INFER (Mac)                                          │
│     └─ Run inference with confidence scoring             │
│     └─ Flag bottom 5-15% segments                        │
│                                                          │
│  2. CORRECT (Mac — Label Studio / Prodigy)               │
│     └─ Human reviews flagged segments only               │
│     └─ Edit Whisper output (5-10x faster than from       │
│        scratch)                                          │
│     └─ Save corrections to stark_data/corrections/       │
│                                                          │
│  3. RETRAIN (WSL)                                        │
│     └─ Merge corrections into training dataset           │
│     └─ Mix 70% general-domain replay + 30% church data   │
│     └─ LoRA retrain with curriculum learning             │
│     └─ Evaluate on held-out church set + general set     │
│                                                          │
│  4. DEPLOY (Mac)                                         │
│     └─ Copy new LoRA adapters to Mac                     │
│     └─ Reload in pipeline, run next A/B evaluation       │
└──────────────────────────────────────────────────────────┘
```

### Anti-Forgetting Safeguards

```python
# In training script: mix replay buffer with domain data
from datasets import concatenate_datasets

def prepare_mixed_dataset(church_dataset, replay_ratio=0.7):
    """Mix general-domain replay samples with church data."""
    # Load a general English ASR dataset as replay buffer
    general = load_dataset("mozilla-foundation/common_voice_16_1",
                           "en", split="train[:1000]")
    general = general.cast_column("audio", Audio(sampling_rate=16000))

    # Calculate mix sizes
    church_size = len(church_dataset)
    replay_size = int(church_size * replay_ratio / (1 - replay_ratio))
    replay_subset = general.select(range(min(replay_size, len(general))))

    mixed = concatenate_datasets([church_dataset, replay_subset])
    return mixed.shuffle(seed=42)
```

### Curriculum Learning Strategy

```python
def sort_by_difficulty(dataset, model):
    """Sort training data: easy first, hard last."""
    difficulties = []
    for example in dataset:
        # Use model's own confidence as difficulty proxy
        with torch.no_grad():
            output = model.generate(
                example["input_features"].unsqueeze(0).to("cuda"),
                output_scores=True, return_dict_in_generate=True
            )
            scores = model.compute_transition_scores(
                output.sequences, output.scores, normalize_logits=True
            )
            avg_confidence = scores.mean().item()
        difficulties.append(-avg_confidence)  # Lower confidence = harder

    # Sort indices by difficulty (easy first)
    sorted_indices = sorted(range(len(difficulties)), key=lambda i: difficulties[i])
    return dataset.select(sorted_indices)
```

In training args, use a learning rate scheduler that warms up slowly to allow the model to learn easy patterns first before tackling hard cases.

### Tracking Convergence

After each cycle, evaluate on:
- **Domain test set:** 100 held-out church segments (should improve each cycle)
- **General test set:** LibriSpeech test-clean subset (should stay roughly constant — no forgetting)

```python
def evaluate_cycle(model, processor, church_test, general_test):
    """Post-cycle evaluation."""
    from jiwer import wer

    church_wer = compute_wer_on_dataset(model, processor, church_test)
    general_wer = compute_wer_on_dataset(model, processor, general_test)

    print(f"Church WER: {church_wer:.1%}")
    print(f"General WER: {general_wer:.1%}")

    if church_wer_improvement < 0.01:  # < 1% relative improvement
        print("Diminishing returns — consider stopping after this cycle.")

    return church_wer, general_wer
```

---

## Phase 6b: Translation Evaluation (`evaluate_translation.py`)

Evaluate fine-tuned TranslateGemma on the holdout Bible test set using three complementary metrics. Run on WSL after training; also runnable on Mac for post-transfer verification.

```python
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sacrebleu
from comet import download_model, load_from_checkpoint
import torch

def evaluate_biblical_translation(
    adapter_dir="/mnt/c/Users/YourName/Projects/fine_tuned_gemma_mi_A",
    test_data="bible_data/aligned/verse_pairs_test.jsonl",
    base_model="google/translategemma-4b-it",
):
    """Evaluate fine-tuned TranslateGemma on holdout Bible verses.
    
    Metrics:
    - SacreBLEU: n-gram precision (standard MT metric)
    - chrF++: character-level n-grams (better for Spanish morphology)
    - COMET: neural metric with highest correlation to human judgments
    """
    # Load fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", torch_dtype=torch.bfloat16, load_in_4bit=True)
    model = PeftModel.from_pretrained(base, adapter_dir)

    # Load test set
    test = [json.loads(line) for line in open(test_data)]
    
    sources, references, hypotheses = [], [], []
    for example in test:
        messages = [{"role": "user", "content": [
            {"type": "text", "source_lang_code": "en",
             "target_lang_code": "es", "text": example["en"]}
        ]}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=256)
        translation = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        sources.append(example["en"])
        references.append(example["es"])
        hypotheses.append(translation.strip())

    # SacreBLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"SacreBLEU: {bleu.score:.1f}")

    # chrF++
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    print(f"chrF++: {chrf.score:.1f}")

    # COMET (neural, highest human correlation)
    comet_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_path)
    comet_input = [{"src": s, "mt": h, "ref": r} 
                   for s, h, r in zip(sources, hypotheses, references)]
    comet_score = comet_model.predict(comet_input, batch_size=8)
    print(f"COMET: {comet_score.system_score:.4f}")

    # Per-genre breakdown
    genres = {
        "pentateuch": range(1, 6),     # Genesis-Deuteronomy
        "history": range(6, 18),       # Joshua-Esther
        "poetry": range(18, 23),       # Job-Song of Solomon
        "prophecy": range(23, 40),     # Isaiah-Malachi
        "gospels": range(40, 44),      # Matthew-John
        "epistles": range(44, 66),     # Acts-Jude
        "apocalyptic": range(66, 67),  # Revelation
    }
    for genre, book_range in genres.items():
        genre_hyps = [h for h, ex in zip(hypotheses, test) 
                      if int(str(ex.get("verse_id", "01001001"))[:2]) in book_range]
        genre_refs = [r for r, ex in zip(references, test)
                      if int(str(ex.get("verse_id", "01001001"))[:2]) in book_range]
        if genre_hyps:
            genre_bleu = sacrebleu.corpus_bleu(genre_hyps, [genre_refs])
            print(f"  {genre}: BLEU {genre_bleu.score:.1f} ({len(genre_hyps)} verses)")

    return {"bleu": bleu.score, "chrf": chrf.score, "comet": comet_score.system_score}


def evaluate_theological_terms(
    adapter_dir="/mnt/c/Users/YourName/Projects/fine_tuned_gemma_mi_A",
    base_model="google/translategemma-4b-it",
):
    """Spot-check critical theological term translations."""
    from build_glossary import THEOLOGICAL_GLOSSARY

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", torch_dtype=torch.bfloat16, load_in_4bit=True)
    model = PeftModel.from_pretrained(base, adapter_dir)

    test_sentences = [
        ("Christ's atonement covers all sins.", "expiación"),
        ("The covenant between God and Abraham.", "pacto"),
        ("We are saved by grace through faith.", "gracia"),
        ("The righteousness of God is revealed.", "justicia"),
        ("James wrote about faith and works.", "Santiago"),  # Epistle context
        ("James and John were fishermen.", "Jacobo"),        # Apostle context
    ]

    correct = 0
    for en_sentence, expected_es_term in test_sentences:
        messages = [{"role": "user", "content": [
            {"type": "text", "source_lang_code": "en",
             "target_lang_code": "es", "text": en_sentence}
        ]}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=128)
        translation = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        found = expected_es_term.lower() in translation.lower()
        correct += found
        status = "✅" if found else "❌"
        print(f"{status} '{en_sentence}' → '{translation.strip()}' (expected: {expected_es_term})")

    print(f"\nTheological term accuracy: {correct}/{len(test_sentences)}")


if __name__ == "__main__":
    print("=== Corpus-level metrics ===")
    scores = evaluate_biblical_translation()
    print(f"\n=== Theological term spot-check ===")
    evaluate_theological_terms()
```

---

## Model Transfer to Mac

After training, copy LoRA adapter folders to the Mac:

```bash
# From WSL — adapters are small (~60MB each)
ls -la /mnt/c/Users/YourName/Projects/fine_tuned_whisper_mi/
# Should contain: adapter_config.json, adapter_model.bin (or .safetensors)

# Transfer options:
# 1. USB drive
# 2. Network share (scp, rsync)
# 3. AirDrop (copy to Windows first, then AirDrop from iPhone/iPad)

# On Mac — place in project root
cp -r /path/to/fine_tuned_whisper_mi ./project_dir/
cp -r /path/to/fine_tuned_gemma_mi_A ./project_dir/
cp -r /path/to/fine_tuned_gemma_mi_B ./project_dir/
```

Verify on Mac:

```python
# Quick smoke test
from peft import PeftModel
from transformers import WhisperForConditionalGeneration

base = WhisperForConditionalGeneration.from_pretrained("distil-whisper/distil-large-v3")
model = PeftModel.from_pretrained(base, "./fine_tuned_whisper_mi")
print("Adapter loaded successfully:", model.peft_config)
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA out of memory during training | Reduce `per_device_train_batch_size` to 1; ensure `gradient_checkpointing=True`; use 4-bit quant for Gemma |
| WSL doesn't see GPU | Update Windows NVIDIA driver; check `nvidia-smi` in WSL; ensure WSL2 (not WSL1) |
| `bitsandbytes` CUDA errors | Install matching CUDA version: `pip install bitsandbytes --prefer-binary` |
| Slow data loading | Increase `num_proc` in `dataset.map()`; use `dataloader_num_workers=4` in training args |
| `demucs` CUDA OOM | Process long files in chunks; or use CPU with `--device cpu` flag |
| `pyannote-audio` auth error | Need HuggingFace token with accepted user agreement for pyannote models |
| LoRA adapter won't load on Mac | Verify `adapter_config.json` + `adapter_model.safetensors` both present; check PEFT version matches |
| Fine-tuned model worse than base | Check for data quality issues; reduce learning rate; increase replay buffer ratio |
| Training loss not decreasing | Verify dataset preprocessing; check that labels are correctly tokenized; try higher learning rate |
| WSL filesystem slow on /mnt/c/ | Store training data in WSL native filesystem (`~/`), only output final adapters to `/mnt/c/` |
| WSL GPU shared memory spill | Monitor `nvidia-smi` — shared memory should stay near zero. Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` |
| TranslateGemma 12B QLoRA OOM | Reduce batch to 1, increase grad accum to 8. If still OOM, train only 4B variant |
| TranslateGemma chat template errors | Must use exact template with `source_lang_code` / `target_lang_code` fields; verify `tokenizer.apply_chat_template()` works |
| Bible corpus verse alignment off | Verify verse ID format matches (BBCCCVVV); check for missing verses in deuterocanonical sections |
