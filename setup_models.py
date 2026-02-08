#!/usr/bin/env python3
"""
setup_models.py — Download & Verify All Models for Current Machine

Detects hardware (MPS = Mac inference, CUDA = Windows training) and downloads
only the models needed. Reports memory usage after each load.

Usage:
    python setup_models.py              # Auto-detect machine, download all
    python setup_models.py --mac        # Force Mac inference models
    python setup_models.py --windows    # Force Windows training models
    python setup_models.py --skip-12b   # Skip TranslateGemma 12B (saves ~12GB)
    python setup_models.py --dry-run    # Show what would be downloaded
"""

import argparse
import gc
import sys
import time


def get_device_info():
    """Detect hardware and return device config."""
    import torch

    info = {
        "pytorch": torch.__version__,
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
    }

    if info["cuda"]:
        info["device"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = torch.cuda.get_device_properties(0).total_mem / 1024**3
        info["dtype"] = "bfloat16"
    elif info["mps"]:
        info["device"] = "mps"
        info["gpu_name"] = "Apple Silicon (MPS)"
        import subprocess
        result = subprocess.run(["sysctl", "hw.memsize"], capture_output=True, text=True)
        mem_bytes = int(result.stdout.strip().split(": ")[1])
        info["gpu_mem_gb"] = mem_bytes / 1024**3  # Unified memory
        info["dtype"] = "float16"
    else:
        info["device"] = "cpu"
        info["gpu_name"] = "CPU only"
        info["gpu_mem_gb"] = 0
        info["dtype"] = "float32"

    return info


def mem_usage_mb():
    """Current process RSS in MB."""
    import psutil
    return psutil.Process().memory_info().rss / 1024**2


def mps_allocated_mb():
    """MPS GPU memory allocated (Apple Silicon only)."""
    import torch
    if torch.backends.mps.is_available():
        try:
            return torch.mps.driver_allocated_memory() / 1024**2
        except Exception:
            return 0
    return 0


def cuda_allocated_mb():
    """CUDA GPU memory allocated."""
    import torch
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def gpu_mem_mb():
    """GPU memory allocated on current device."""
    import torch
    if torch.cuda.is_available():
        return cuda_allocated_mb()
    elif torch.backends.mps.is_available():
        return mps_allocated_mb()
    return 0


# ---------------------------------------------------------------------------
# Model Loaders
# ---------------------------------------------------------------------------

def load_vad():
    """Silero VAD (~2MB)."""
    import torch
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad',
                                   trust_repo=True)
    return model, utils


def load_whisper(device, dtype_str):
    """Distil-Whisper distil-large-v3 (~1.5GB)."""
    import torch
    from transformers import pipeline

    dtype = getattr(torch, dtype_str)
    pipe = pipeline(
        "automatic-speech-recognition",
        model="distil-whisper/distil-large-v3",
        device=device,
        torch_dtype=dtype,
    )
    return pipe


def test_whisper(pipe):
    """Quick inference test with 1s of silence."""
    import numpy as np
    silence = np.zeros(16000, dtype=np.float32)
    result = pipe({"raw": silence, "sampling_rate": 16000},
                  generate_kwargs={"language": "en"})
    return result["text"]


def load_gemma(model_name, device, use_4bit=False):
    """Load TranslateGemma model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    load_kwargs = {"device_map": "auto"}

    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            quant_mode = "4-bit"
        except Exception as e:
            print(f"    4-bit failed ({e}), falling back to fp16")
            load_kwargs["torch_dtype"] = torch.float16
            quant_mode = "fp16 (4-bit failed)"
    else:
        load_kwargs["torch_dtype"] = torch.float16
        quant_mode = "fp16"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    return model, tokenizer, quant_mode


def test_gemma(model, tokenizer, text="Hello, how are you?"):
    """Quick translation test."""
    import torch

    messages = [{"role": "user", "content": [
        {"type": "text", "source_lang_code": "en",
         "target_lang_code": "es", "text": text}
    ]}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=64)

    translation = tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    return translation


def unload(model, label):
    """Free model memory."""
    import torch
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download & verify models")
    parser.add_argument("--mac", action="store_true", help="Force Mac inference mode")
    parser.add_argument("--windows", action="store_true", help="Force Windows training mode")
    parser.add_argument("--skip-12b", action="store_true", help="Skip TranslateGemma 12B")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without downloading")
    parser.add_argument("--no-test", action="store_true", help="Download only, skip inference tests")
    args = parser.parse_args()

    # Detect hardware
    info = get_device_info()

    if args.mac:
        role = "mac"
    elif args.windows:
        role = "windows"
    elif info["mps"]:
        role = "mac"
    elif info["cuda"]:
        role = "windows"
    else:
        role = "mac"  # CPU fallback uses mac model set

    print(f"{'='*60}")
    print(f"  Model Setup — {role.upper()} mode")
    print(f"{'='*60}")
    print(f"  Device:    {info['device']} ({info['gpu_name']})")
    print(f"  Memory:    {info['gpu_mem_gb']:.1f} GB")
    print(f"  PyTorch:   {info['pytorch']}")
    print(f"  Dtype:     {info['dtype']}")
    print()

    # Define model plan
    models = [
        ("Silero VAD", "~2 MB", True),
        ("Distil-Whisper distil-large-v3", "~1.5 GB", True),
        ("TranslateGemma 4B", "~2.6 GB (4-bit) / ~8 GB (fp16)", True),
    ]
    if not args.skip_12b:
        models.append(("TranslateGemma 12B", "~6 GB (4-bit) / ~24 GB (fp16)",
                        role == "mac"))

    print("  Models to download:")
    for name, size, needed in models:
        status = "DOWNLOAD" if needed else "SKIP"
        print(f"    [{status}] {name} ({size})")
    print()

    if args.dry_run:
        print("Dry run — no downloads performed.")
        return

    results = {}

    # 1. Silero VAD
    print(f"[1/{len(models)}] Silero VAD...")
    t0 = time.time()
    try:
        vad_model, vad_utils = load_vad()
        elapsed = time.time() - t0
        results["vad"] = "OK"
        print(f"  OK ({elapsed:.1f}s)")
    except Exception as e:
        results["vad"] = f"FAIL: {e}"
        print(f"  FAIL: {e}")

    # 2. Distil-Whisper
    print(f"\n[2/{len(models)}] Distil-Whisper distil-large-v3...")
    t0 = time.time()
    try:
        stt_pipe = load_whisper(info["device"], info["dtype"])
        elapsed = time.time() - t0
        gpu_mb = gpu_mem_mb()
        results["whisper"] = "OK"
        print(f"  Downloaded + loaded ({elapsed:.1f}s, GPU: {gpu_mb:.0f} MB)")

        if not args.no_test:
            text = test_whisper(stt_pipe)
            print(f"  Inference test: '{text.strip()[:80]}'")
    except Exception as e:
        results["whisper"] = f"FAIL: {e}"
        print(f"  FAIL: {e}")

    # 3. TranslateGemma 4B
    print(f"\n[3/{len(models)}] TranslateGemma 4B (google/translategemma-4b-it)...")
    t0 = time.time()
    use_4bit = (role == "mac")  # 4-bit on Mac to save memory
    try:
        gemma4b, tok4b, quant = load_gemma(
            "google/translategemma-4b-it", info["device"], use_4bit=use_4bit
        )
        elapsed = time.time() - t0
        gpu_mb = gpu_mem_mb()
        results["gemma_4b"] = f"OK ({quant})"
        print(f"  Downloaded + loaded ({elapsed:.1f}s, {quant}, GPU: {gpu_mb:.0f} MB)")

        if not args.no_test:
            translation = test_gemma(gemma4b, tok4b,
                                      "Good morning, welcome to our service.")
            print(f"  Translation test: '{translation[:80]}'")

        # Keep 4B loaded if we also need 12B (to test parallel mode)
        if args.skip_12b or len(models) <= 3:
            unload(gemma4b, "4B")
    except Exception as e:
        results["gemma_4b"] = f"FAIL: {e}"
        print(f"  FAIL: {e}")
        gemma4b = None

    # 4. TranslateGemma 12B (optional)
    if not args.skip_12b and len(models) > 3:
        print(f"\n[4/{len(models)}] TranslateGemma 12B (google/translategemma-12b-it)...")
        t0 = time.time()
        try:
            gemma12b, tok12b, quant = load_gemma(
                "google/translategemma-12b-it", info["device"], use_4bit=use_4bit
            )
            elapsed = time.time() - t0
            gpu_mb = gpu_mem_mb()
            results["gemma_12b"] = f"OK ({quant})"
            print(f"  Downloaded + loaded ({elapsed:.1f}s, {quant}, GPU: {gpu_mb:.0f} MB)")

            if not args.no_test:
                translation = test_gemma(gemma12b, tok12b,
                                          "Justification by faith is the cornerstone of the Gospel.")
                print(f"  Translation test: '{translation[:80]}'")

            # If both loaded, report parallel memory
            if gemma4b is not None:
                print(f"\n  PARALLEL MODE: Both models loaded, GPU: {gpu_mem_mb():.0f} MB")

            unload(gemma12b, "12B")
        except Exception as e:
            results["gemma_12b"] = f"FAIL: {e}"
            print(f"  FAIL: {e}")

        if gemma4b is not None:
            unload(gemma4b, "4B")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for name, status in results.items():
        icon = "OK" if status.startswith("OK") else "FAIL"
        if icon == "FAIL":
            all_ok = False
        print(f"  [{icon:4s}] {name}: {status}")

    if all_ok:
        print(f"\nAll models ready. Run: python dry_run_ab.py")
    else:
        print(f"\nSome models failed. Check errors above.")
        print(f"If 4-bit failed, try: python dry_run_ab.py --swap")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
