# TODO: Quantize TranslateGemma for NVIDIA Deployment

## Goal

Export TranslateGemma 4B and 12B as persistent GPTQ 4-bit models for deployment on NVIDIA GPUs (RTX 3060 12GB, A2000 Ada 16GB). Persistent quantization loads in seconds without re-quantizing on every startup.

## Target Hardware

| GPU | VRAM | TranslateGemma 4B | TranslateGemma 12B | 4B + Whisper Turbo | 12B + Whisper Turbo |
|-----|------|-------------------|--------------------|--------------------|---------------------|
| RTX 3060 | 12 GB | 2.5 GB (9.5 free) | 7 GB (5 free) | 4 GB (8 free) | 8.5 GB (3.5 free) |
| A2000 Ada | 16 GB | 2.5 GB (13.5 free) | 7 GB (9 free) | 4 GB (12 free) | 8.5 GB (7.5 free) |

Whisper large-v3-turbo INT8 via faster-whisper: ~1.5 GB VRAM.

## Quantization Formats

### GPTQ 4-bit (recommended for NVIDIA deployment)

- **Persistent**: saves to disk as safetensors, loads instantly — no quantize-on-startup
- **Fast inference**: `exllamav2` kernels optimized for NVIDIA (Ampere/Ada)
- **Calibration**: requires 128 translation samples (we use sermon pairs)
- **Quality**: minimal degradation at 4-bit with calibration
- **Compatibility**: works with `transformers`, `auto-gptq`, `text-generation-inference`
- **LoRA**: adapters from S6/S8 training are compatible (merge before or after quantization)

```bash
pip install auto-gptq optimum
```

### AWQ 4-bit (alternative)

- Slightly better quality than GPTQ at same bit-width in some benchmarks
- `autoawq` library, also saves persistently
- Less mature kernel support than GPTQ's exllamav2

```bash
pip install autoawq
```

### bitsandbytes NF4 (current, for development only)

- Load-time quantization — quantizes on every cold start (~2-6 min)
- Already used in `train_gemma.py` for QLoRA training
- Good for development/training but too slow for production deployment
- No persistent export

## GPTQ Export Plan

### Prerequisites

```bash
source /home/wbell/stt_train_env/bin/activate
pip install auto-gptq optimum
```

### Step 1: Export 4B GPTQ (~10-15 min)

```bash
python training/benchmark_quantize.py
# Or manually:
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
tokenizer = AutoTokenizer.from_pretrained('google/translategemma-4b-it')
gptq_config = GPTQConfig(bits=4, dataset=calib_data, tokenizer=tokenizer)
model = AutoModelForCausalLM.from_pretrained(
    'google/translategemma-4b-it', quantization_config=gptq_config, device_map='auto'
)
model.save_pretrained('models/translategemma-4b-gptq')
tokenizer.save_pretrained('models/translategemma-4b-gptq')
"
```

Output: `models/translategemma-4b-gptq/` (~2.5 GB on disk)

### Step 2: Export 12B GPTQ (~30-45 min)

```bash
# Same process, more time (5 shards + calibration)
# Output: models/translategemma-12b-gptq/ (~7 GB on disk)
```

### Step 3: Verify on A2000

```bash
python tools/health_check.py --adapter none --base-model models/translategemma-4b-gptq
python tools/health_check.py --adapter none --base-model models/translategemma-12b-gptq
```

5 canary sentences (atonement, James/Santiago, propitiation, breaking of bread, resurrection).

### Step 4: Transfer to 3060

```bash
rsync -av models/translategemma-{4b,12b}-gptq/ user@3060-machine:~/models/
```

### Step 5: Load on 3060 (instant — no quantization needed)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("models/translategemma-12b-gptq", device_map="auto")
# Loads in ~5 seconds, 7 GB VRAM
```

## Calibration Data

128 sermon translation samples from our training data. The script at `training/benchmark_quantize.py` uses 8 theological sentences repeated to 128. For best quality, use actual sermon pairs from `bible_data/synthetic/deepl_sermon_pairs_5000.jsonl`:

```python
import json
calib_data = []
with open("bible_data/synthetic/deepl_sermon_pairs_5000.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 128: break
        pair = json.loads(line)
        calib_data.append(f"Translate from English to Spanish: {pair['en']}")
```

## Integration with Inference Pipeline

### `engines/cuda_engine.py` changes

```python
# Current (bitsandbytes load-time):
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

# New (GPTQ persistent):
model = AutoModelForCausalLM.from_pretrained("models/translategemma-12b-gptq", device_map="auto")
# No quantization_config needed — GPTQ metadata is in the safetensors
```

### `settings.py` addition

```python
class TranslationSettings(BaseSettings):
    # Add:
    quantized_4b_path: str = "models/translategemma-4b-gptq"
    quantized_12b_path: str = "models/translategemma-12b-gptq"
    use_quantized: bool = True  # Use GPTQ models if available
```

### LoRA adapter compatibility

LoRA adapters trained on the fp16/bitsandbytes base model can be applied to GPTQ models:

```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("models/translategemma-4b-gptq", device_map="auto")
model = PeftModel.from_pretrained(base, "hybrid_runs/S6_balanced")
```

Or merge the adapter into the model BEFORE GPTQ export for maximum inference speed (no adapter overhead at runtime).

## Estimated Timeline

| Step | Time | GPU |
|------|------|-----|
| Export 4B GPTQ | ~15 min | ~8 GB peak |
| Export 12B GPTQ | ~45 min | ~14 GB peak |
| Health check (both) | ~5 min | ~7 GB peak |
| Transfer to 3060 | ~5 min | Network |
| **Total** | **~70 min** | |

## Files

| File | Status | What |
|------|--------|------|
| `training/benchmark_quantize.py` | **EXISTS** | Benchmark + GPTQ export (chained after Whisper ablation) |
| `models/translategemma-4b-gptq/` | **TODO** | Exported 4B GPTQ model |
| `models/translategemma-12b-gptq/` | **TODO** | Exported 12B GPTQ model |
| `engines/cuda_engine.py` | **TODO** | Update to load GPTQ models |
| `settings.py` | **TODO** | Add quantized model paths |
