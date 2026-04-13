#!/usr/bin/env python3
"""Benchmark quantized TranslateGemma models (4B + 12B) at INT4 and INT8."""

import gc
import json
import os
import sys
import time

os.environ["USE_TF"] = "0"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CANARY = [
    "The atonement of Christ reconciles us to God.",
    "James wrote about faith and works.",
    "The propitiation for our sins was the blood of Christ.",
    "The breaking of bread is a solemn remembrance.",
    "Paul wrote to the Corinthians about the resurrection.",
]

CONFIGS = [
    (
        "google/translategemma-4b-it",
        "4bit",
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    ),
    ("google/translategemma-4b-it", "8bit", BitsAndBytesConfig(load_in_8bit=True)),
    (
        "google/translategemma-12b-it",
        "4bit",
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    ),
    ("google/translategemma-12b-it", "8bit", BitsAndBytesConfig(load_in_8bit=True)),
]

results = []

for model_id, quant, bnb_config in CONFIGS:
    name = f"{model_id.split('/')[-1]}_{quant}"
    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")

    load_start = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print(f"  FAILED to load: {e}")
        results.append({"name": name, "status": "LOAD_FAILED", "error": str(e)})
        continue
    load_time = time.time() - load_start

    vram_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  Loaded in {load_time:.0f}s, VRAM: {vram_gb:.1f} GB")

    # Translate canary sentences using proper chat template
    translations = []
    latencies = []
    for sentence in CANARY:
        msg = [
            {
                "role": "user",
                "content": [{"type": "text", "source_lang_code": "en", "target_lang_code": "es", "text": sentence}],
            },
            {"role": "assistant", "content": ""},
        ]
        inputs = tokenizer.apply_chat_template(msg, tokenize=True, return_tensors="pt").to(model.device)
        attention_mask = torch.ones_like(inputs)
        start = time.time()
        with torch.no_grad():
            out = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=128, do_sample=False)
        latency = time.time() - start
        es = tokenizer.decode(out[0][inputs.shape[1] :], skip_special_tokens=True)
        translations.append(es)
        latencies.append(latency)
        print(f"  [{latency:.1f}s] {sentence[:50]}... → {es[:50]}...")

    avg_latency = sum(latencies) / len(latencies)
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3

    results.append(
        {
            "name": name,
            "status": "OK",
            "load_time_s": round(load_time, 1),
            "vram_gb": round(vram_gb, 1),
            "peak_vram_gb": round(peak_vram, 1),
            "avg_latency_s": round(avg_latency, 2),
            "translations": translations,
        }
    )

    print(f"  Avg latency: {avg_latency:.2f}s, Peak VRAM: {peak_vram:.1f} GB")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# Summary
print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"{'Model':<35} {'Status':<8} {'VRAM':>6} {'Peak':>6} {'Latency':>8}")
print("-" * 70)
for r in results:
    if r["status"] == "OK":
        print(f"{r['name']:<35} {'OK':<8} {r['vram_gb']:>5.1f}G {r['peak_vram_gb']:>5.1f}G {r['avg_latency_s']:>7.2f}s")
    else:
        print(f"{r['name']:<35} {'FAIL':<8}")

with open("hybrid_runs/quantize_benchmark.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\nSaved to hybrid_runs/quantize_benchmark.json")

# === GPTQ Export ===
print(f"\n{'=' * 60}")
print("GPTQ 4-bit Export (persistent quantization for 3060)")
print(f"{'=' * 60}")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

    GPTQ_MODELS = [
        ("google/translategemma-4b-it", "models/translategemma-4b-gptq"),
        ("google/translategemma-12b-it", "models/translategemma-12b-gptq"),
    ]

    # Calibration data from our sermon translation pairs
    calib_data = [
        "Translate from English to Spanish: The atonement of Christ reconciles us to God.",
        "Translate from English to Spanish: James wrote about faith and works.",
        "Translate from English to Spanish: The propitiation for our sins was the blood of Christ.",
        "Translate from English to Spanish: The breaking of bread is a solemn remembrance.",
        "Translate from English to Spanish: Paul wrote to the Corinthians about the resurrection.",
        "Translate from English to Spanish: For God so loved the world that He gave His only begotten Son.",
        "Translate from English to Spanish: The wages of sin is death, but the gift of God is eternal life.",
        "Translate from English to Spanish: Blessed are the poor in spirit, for theirs is the kingdom of heaven.",
    ]
    # Pad to 128 samples by repeating
    while len(calib_data) < 128:
        calib_data.extend(calib_data[: 128 - len(calib_data)])
    calib_data = calib_data[:128]

    for model_id, output_dir in GPTQ_MODELS:
        name = model_id.split("/")[-1]
        print(f"\n--- {name} GPTQ 4-bit ---")
        start = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        gptq_config = GPTQConfig(bits=4, dataset=calib_data, tokenizer=tokenizer)

        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=gptq_config, device_map="auto")

        elapsed = time.time() - start
        vram = torch.cuda.memory_allocated() / 1024**3
        print(f"  Quantized in {elapsed / 60:.1f} min, VRAM: {vram:.1f} GB")

        # Save
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        size_gb = (
            sum(
                os.path.getsize(os.path.join(output_dir, f))
                for f in os.listdir(output_dir)
                if f.endswith(".safetensors")
            )
            / 1024**3
        )
        print(f"  Saved to {output_dir} ({size_gb:.1f} GB on disk)")

        # Quick translate test with proper chat template
        test_msg = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": "en",
                        "target_lang_code": "es",
                        "text": "The grace of God is sufficient.",
                    }
                ],
            },
            {"role": "assistant", "content": ""},
        ]
        test_ids = tokenizer.apply_chat_template(test_msg, tokenize=True, return_tensors="pt")
        if hasattr(test_ids, "input_ids"):
            test_ids = test_ids.input_ids
        test_ids = test_ids.to(model.device)
        test_mask = torch.ones_like(test_ids)
        with torch.no_grad():
            out = model.generate(test_ids, attention_mask=test_mask, max_new_tokens=64, do_sample=False)
        print(f"  Test: {tokenizer.decode(out[0][test_ids.shape[1] :], skip_special_tokens=True)[:100]}")

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

except ImportError:
    print("auto-gptq not installed. Run: pip install auto-gptq optimum")
except Exception as e:
    print(f"GPTQ export failed: {e}")
