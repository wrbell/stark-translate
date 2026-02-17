#!/usr/bin/env python3
"""
setup_models.py — Download & Verify All Models for Current Machine

Downloads MLX models for Mac inference, PyTorch models for Windows training.
Reports memory usage and runs quick inference tests.

Usage:
    python setup_models.py              # Auto-detect machine, download all
    python setup_models.py --mac        # Force Mac inference models
    python setup_models.py --windows    # Force Windows training models
    python setup_models.py --nvidia-inference  # NVIDIA/CUDA inference models
    python setup_models.py --skip-12b   # Skip TranslateGemma 12B
    python setup_models.py --dry-run    # Show what would be downloaded
"""

import argparse
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
    elif info["mps"]:
        info["device"] = "mps"
        info["gpu_name"] = "Apple Silicon (MPS)"
        import subprocess
        result = subprocess.run(["sysctl", "hw.memsize"], capture_output=True, text=True)
        mem_bytes = int(result.stdout.strip().split(": ")[1])
        info["gpu_mem_gb"] = mem_bytes / 1024**3
    else:
        info["device"] = "cpu"
        info["gpu_name"] = "CPU only"
        info["gpu_mem_gb"] = 0

    # Check MLX availability
    try:
        import mlx.core as mx
        info["mlx"] = True
        info["mlx_version"] = mx.__version__
    except ImportError:
        info["mlx"] = False
        info["mlx_version"] = None

    return info


# ---------------------------------------------------------------------------
# Mac Models (MLX)
# ---------------------------------------------------------------------------

def download_mac_models(skip_12b=False, dry_run=False, no_test=False):
    """Download and verify MLX models for Mac inference."""
    from huggingface_hub import snapshot_download

    models = [
        ("Silero VAD", "snakers4/silero-vad", "~2 MB", "torch_hub"),
        ("Distil-Whisper (MLX)", "mlx-community/distil-whisper-large-v3", "~1.5 GB", "mlx_whisper"),
        ("Whisper Large-V3-Turbo (MLX)", "mlx-community/whisper-large-v3-turbo", "~1.1 GB", "mlx_whisper"),
        ("TranslateGemma 4B (MLX 4-bit)", "mlx-community/translategemma-4b-it-4bit", "~2.2 GB", "mlx_lm"),
    ]
    if not skip_12b:
        models.append(
            ("TranslateGemma 12B (MLX 4-bit)", "mlx-community/translategemma-12b-it-4bit", "~6.6 GB", "mlx_lm")
        )

    print("  Models to download:")
    for name, model_id, size, _ in models:
        print(f"    [DOWNLOAD] {name} ({size}) — {model_id}")
    print()

    if dry_run:
        print("Dry run — no downloads performed.")
        return True

    results = {}

    # 1. Silero VAD
    print(f"[1/{len(models)}] Silero VAD...")
    t0 = time.time()
    try:
        import torch
        model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
        print(f"  OK ({time.time()-t0:.1f}s)")
        results["vad"] = "OK"
    except Exception as e:
        print(f"  FAIL: {e}")
        results["vad"] = f"FAIL: {e}"

    # 2. Distil-Whisper (MLX)
    print(f"\n[2/{len(models)}] Distil-Whisper (MLX)...")
    t0 = time.time()
    try:
        snapshot_download("mlx-community/distil-whisper-large-v3")
        print(f"  Downloaded ({time.time()-t0:.1f}s)")

        if not no_test:
            import mlx_whisper
            import numpy as np
            silence = np.zeros(16000, dtype=np.float32)
            result = mlx_whisper.transcribe(
                silence,
                path_or_hf_repo="mlx-community/distil-whisper-large-v3",
                condition_on_previous_text=False,
            )
            print(f"  Inference test: '{result['text'].strip()[:80]}'")
        results["whisper"] = "OK"
    except Exception as e:
        print(f"  FAIL: {e}")
        results["whisper"] = f"FAIL: {e}"

    # 3. Whisper Large-V3-Turbo (MLX)
    print(f"\n[3/{len(models)}] Whisper Large-V3-Turbo (MLX)...")
    t0 = time.time()
    try:
        snapshot_download("mlx-community/whisper-large-v3-turbo")
        print(f"  Downloaded ({time.time()-t0:.1f}s)")

        if not no_test:
            import mlx_whisper
            import numpy as np
            silence = np.zeros(16000, dtype=np.float32)
            result = mlx_whisper.transcribe(
                silence,
                path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                condition_on_previous_text=False,
            )
            print(f"  Inference test: '{result['text'].strip()[:80]}'")
        results["whisper_turbo"] = "OK"
    except Exception as e:
        print(f"  FAIL: {e}")
        results["whisper_turbo"] = f"FAIL: {e}"

    # 4. TranslateGemma 4B (MLX)
    print(f"\n[4/{len(models)}] TranslateGemma 4B (MLX 4-bit)...")
    t0 = time.time()
    try:
        snapshot_download("mlx-community/translategemma-4b-it-4bit")
        print(f"  Downloaded ({time.time()-t0:.1f}s)")

        if not no_test:
            from mlx_lm import load, generate
            model_4b, tok_4b = load("mlx-community/translategemma-4b-it-4bit")
            eot = tok_4b.convert_tokens_to_ids("<end_of_turn>")
            tok_4b._eos_token_ids = {tok_4b.eos_token_id, eot}

            msgs = [{"role": "user", "content": [
                {"type": "text", "source_lang_code": "en",
                 "target_lang_code": "es", "text": "Good morning, welcome to our service."}
            ]}]
            prompt = tok_4b.apply_chat_template(msgs, add_generation_prompt=True)
            result = generate(model_4b, tok_4b, prompt=prompt, max_tokens=64, verbose=False)
            clean = result.split("<end_of_turn>")[0].strip()
            print(f"  Translation test: '{clean[:80]}'")
            del model_4b

        results["gemma_4b"] = "OK"
    except Exception as e:
        print(f"  FAIL: {e}")
        results["gemma_4b"] = f"FAIL: {e}"

    # 5. TranslateGemma 12B (MLX)
    if not skip_12b:
        print(f"\n[5/{len(models)}] TranslateGemma 12B (MLX 4-bit)...")
        t0 = time.time()
        try:
            snapshot_download("mlx-community/translategemma-12b-it-4bit")
            print(f"  Downloaded ({time.time()-t0:.1f}s)")

            if not no_test:
                from mlx_lm import load, generate
                model_12b, tok_12b = load("mlx-community/translategemma-12b-it-4bit")
                eot = tok_12b.convert_tokens_to_ids("<end_of_turn>")
                tok_12b._eos_token_ids = {tok_12b.eos_token_id, eot}

                msgs = [{"role": "user", "content": [
                    {"type": "text", "source_lang_code": "en",
                     "target_lang_code": "es",
                     "text": "Justification by faith is the cornerstone of the Gospel."}
                ]}]
                prompt = tok_12b.apply_chat_template(msgs, add_generation_prompt=True)
                result = generate(model_12b, tok_12b, prompt=prompt, max_tokens=64, verbose=False)
                clean = result.split("<end_of_turn>")[0].strip()
                print(f"  Translation test: '{clean[:80]}'")
                del model_12b

            results["gemma_12b"] = "OK"
        except Exception as e:
            print(f"  FAIL: {e}")
            results["gemma_12b"] = f"FAIL: {e}"

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

    return all_ok


# ---------------------------------------------------------------------------
# Windows Models (PyTorch/CUDA)
# ---------------------------------------------------------------------------

def download_windows_models(skip_12b=False, dry_run=False, no_test=False):
    """Download PyTorch models for Windows/CUDA training."""
    from huggingface_hub import snapshot_download

    models = [
        ("Distil-Whisper", "distil-whisper/distil-large-v3", "~1.5 GB"),
        ("Whisper Large-V3-Turbo", "openai/whisper-large-v3-turbo", "~1.5 GB"),
        ("TranslateGemma 4B", "google/translategemma-4b-it", "~8 GB"),
    ]
    if not skip_12b:
        models.append(("TranslateGemma 12B", "google/translategemma-12b-it", "~24 GB"))

    print("  Models to download:")
    for name, model_id, size in models:
        print(f"    [DOWNLOAD] {name} ({size}) — {model_id}")
    print()

    if dry_run:
        print("Dry run — no downloads performed.")
        return True

    all_ok = True
    for name, model_id, size in models:
        print(f"Downloading {name}...")
        t0 = time.time()
        try:
            snapshot_download(model_id)
            print(f"  OK ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  FAIL: {e}")
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# NVIDIA Inference Models (PyTorch/CUDA — not training)
# ---------------------------------------------------------------------------

def download_nvidia_inference_models(skip_12b=False, dry_run=False, no_test=False):
    """Download models for NVIDIA/CUDA inference (not training).

    This is distinct from --windows which downloads full-precision models for
    fine-tuning. NVIDIA inference uses CTranslate2 Whisper Turbo, MarianMT,
    and quantized TranslateGemma.
    """
    # TODO: Implement NVIDIA inference model downloads:
    #   - Whisper Large-V3-Turbo (CTranslate2 format for faster-whisper)
    #   - MarianMT EN->ES (CTranslate2 int8)
    #   - TranslateGemma 4B (bitsandbytes 4-bit)
    #   - Silero VAD
    print("NVIDIA inference model download not yet implemented.")
    print("Use --windows for training models, or --mac for MLX inference models.")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download & verify models")
    parser.add_argument("--mac", action="store_true", help="Force Mac inference mode (MLX)")
    parser.add_argument("--windows", action="store_true", help="Force Windows training mode (PyTorch)")
    parser.add_argument("--nvidia-inference", action="store_true", help="NVIDIA/CUDA inference models (not training)")
    parser.add_argument("--skip-12b", action="store_true", help="Skip TranslateGemma 12B")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without downloading")
    parser.add_argument("--no-test", action="store_true", help="Download only, skip inference tests")
    args = parser.parse_args()

    info = get_device_info()

    if args.mac:
        role = "mac"
    elif args.windows:
        role = "windows"
    elif args.nvidia_inference:
        role = "nvidia-inference"
    elif info["mps"] and info.get("mlx"):
        role = "mac"
    elif info["cuda"]:
        role = "windows"
    else:
        role = "mac"

    print(f"{'='*60}")
    print(f"  Model Setup — {role.upper()} mode")
    print(f"{'='*60}")
    print(f"  Device:    {info['device']} ({info['gpu_name']})")
    print(f"  Memory:    {info['gpu_mem_gb']:.1f} GB")
    print(f"  PyTorch:   {info['pytorch']}")
    if info.get("mlx"):
        print(f"  MLX:       {info['mlx_version']}")
    print()

    if role == "mac":
        ok = download_mac_models(args.skip_12b, args.dry_run, args.no_test)
    elif role == "nvidia-inference":
        ok = download_nvidia_inference_models(args.skip_12b, args.dry_run, args.no_test)
    else:
        ok = download_windows_models(args.skip_12b, args.dry_run, args.no_test)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
