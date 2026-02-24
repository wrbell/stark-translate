#!/usr/bin/env python3
"""
convert_models_to_both.py — Dual-Endpoint Model Export for Mac (MLX) + NVIDIA (CTranslate2/faster-whisper)

Converts and quantizes models for both deployment targets in the bilingual church
transcription pipeline. Handles Whisper Large-V3-Turbo (STT), TranslateGemma 4B
(translation), and MarianMT (fast partial translations).

Export targets:
    Mac (MLX):   Pre-quantized 4-bit models from mlx-community — downloaded as-is
    NVIDIA:      CTranslate2 int8 for Whisper/MarianMT, bitsandbytes 4-bit for Gemma

Output structure:
    {output_dir}/
    +-- whisper-turbo-ct2/          # CTranslate2 int8 Whisper for faster-whisper
    |   +-- model.bin
    |   +-- tokenizer.json
    |   +-- ...
    +-- marian-ct2/                 # CTranslate2 int8 MarianMT (optional)
    |   +-- model.bin
    |   +-- ...
    +-- README.txt                  # Export manifest

Usage:
    python tools/convert_models_to_both.py --all
    python tools/convert_models_to_both.py --whisper --skip-mlx
    python tools/convert_models_to_both.py --gemma --lora-path fine_tuned_gemma_mi_A/
    python tools/convert_models_to_both.py --marian --output-dir ./exported_models/
    python tools/convert_models_to_both.py --all --dry-run
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model IDs — kept in sync with settings.py and engines/factory.py
# ---------------------------------------------------------------------------

# Mac / MLX (pre-quantized from mlx-community)
MLX_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
MLX_GEMMA_4B_MODEL = "mlx-community/translategemma-4b-it-4bit"
MLX_GEMMA_12B_MODEL = "mlx-community/translategemma-12b-it-4bit"

# NVIDIA / CUDA (full-precision source models for conversion)
CUDA_WHISPER_MODEL = "openai/whisper-large-v3-turbo"
CUDA_GEMMA_4B_MODEL = "google/translategemma-4b-it"
CUDA_MARIAN_MODEL = "Helsinki-NLP/opus-mt-en-es"

# Project root for resolving relative paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Optional imports — guarded for cross-platform compatibility
# ---------------------------------------------------------------------------


def _check_ctranslate2():
    """Check if CTranslate2 is available."""
    try:
        import ctranslate2  # noqa: F401

        return True
    except ImportError:
        return False


def _check_mlx():
    """Check if MLX is available (Apple Silicon)."""
    try:
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False


def _check_peft():
    """Check if PEFT (LoRA) is available."""
    try:
        import peft  # noqa: F401

        return True
    except ImportError:
        return False


def _check_transformers():
    """Check if transformers is available."""
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def get_dir_size_mb(path):
    """Calculate total size of a directory in MB."""
    total = 0
    p = Path(path)
    if not p.exists():
        return 0.0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


def format_size(size_mb):
    """Format size in MB to a human-readable string."""
    if size_mb >= 1024:
        return f"{size_mb / 1024:.1f} GB"
    return f"{size_mb:.1f} MB"


# ---------------------------------------------------------------------------
# Whisper Export
# ---------------------------------------------------------------------------


def export_whisper_mlx(output_dir, dry_run=False):
    """Download/reference the pre-quantized MLX Whisper model.

    MLX Whisper models from mlx-community are already quantized and optimized
    for Apple Silicon. No conversion is needed -- huggingface_hub caches them
    locally on first use (via mlx_whisper.transcribe() or snapshot_download).

    This function ensures the model is cached locally and documents its location.
    """
    logger.info("Whisper (MLX): Pre-quantized model from mlx-community")
    logger.info(f"  Model ID: {MLX_WHISPER_MODEL}")
    logger.info("  Format:   MLX weights (pre-quantized, no conversion needed)")
    logger.info("  Note:     Cached by huggingface_hub on first use")

    if dry_run:
        logger.info("  [DRY RUN] Would download/verify model via snapshot_download")
        return {
            "model": "Whisper",
            "target": "MLX",
            "format": "MLX fp16",
            "size": "~1.1 GB",
            "status": "SKIP (dry run)",
        }

    try:
        from huggingface_hub import snapshot_download

        t0 = time.time()
        cache_path = snapshot_download(MLX_WHISPER_MODEL)
        elapsed = time.time() - t0
        size_mb = get_dir_size_mb(cache_path)
        logger.info(f"  Cached at: {cache_path}")
        logger.info(f"  Size:      {format_size(size_mb)}")
        logger.info(f"  Time:      {elapsed:.1f}s")
        return {"model": "Whisper", "target": "MLX", "format": "MLX fp16", "size": format_size(size_mb), "status": "OK"}
    except Exception as e:
        logger.error(f"  FAILED: {e}")
        return {"model": "Whisper", "target": "MLX", "format": "MLX fp16", "size": "—", "status": f"FAIL: {e}"}


def export_whisper_ct2(output_dir, dry_run=False):
    """Convert Whisper Large-V3-Turbo to CTranslate2 int8 for faster-whisper.

    Uses ctranslate2.converters.TransformersConverter to convert the HuggingFace
    Whisper model to CTranslate2 format with int8 quantization. The output is
    compatible with faster-whisper for NVIDIA GPU inference.
    """
    ct2_dir = Path(output_dir) / "whisper-turbo-ct2"

    logger.info("Whisper (CUDA/CT2): Converting to CTranslate2 int8")
    logger.info(f"  Source:  {CUDA_WHISPER_MODEL}")
    logger.info(f"  Output:  {ct2_dir}")
    logger.info("  Format:  CTranslate2 int8 (for faster-whisper)")

    if dry_run:
        logger.info("  [DRY RUN] Would convert model via TransformersConverter")
        return {
            "model": "Whisper",
            "target": "CUDA/CT2",
            "format": "CT2 int8",
            "size": "~800 MB",
            "status": "SKIP (dry run)",
        }

    if not _check_ctranslate2():
        logger.error("  ctranslate2 not installed. Install with: pip install ctranslate2")
        return {
            "model": "Whisper",
            "target": "CUDA/CT2",
            "format": "CT2 int8",
            "size": "—",
            "status": "FAIL: ctranslate2 not installed",
        }

    if not _check_transformers():
        logger.error("  transformers not installed. Install with: pip install transformers")
        return {
            "model": "Whisper",
            "target": "CUDA/CT2",
            "format": "CT2 int8",
            "size": "—",
            "status": "FAIL: transformers not installed",
        }

    try:
        import ctranslate2

        t0 = time.time()
        ct2_dir.mkdir(parents=True, exist_ok=True)

        # Convert via CTranslate2's built-in Whisper/Transformers converter
        converter = ctranslate2.converters.TransformersConverter(
            CUDA_WHISPER_MODEL,
            copy_files=[
                "tokenizer.json",
                "preprocessor_config.json",
                "special_tokens_map.json",
                "normalizer.json",
                "added_tokens.json",
                "vocab.json",
                "merges.txt",
            ],
        )
        converter.convert(
            output_dir=str(ct2_dir),
            quantization="int8",
            force=True,
        )

        elapsed = time.time() - t0
        size_mb = get_dir_size_mb(ct2_dir)
        logger.info(f"  Converted in {elapsed:.1f}s")
        logger.info(f"  Size: {format_size(size_mb)}")
        return {
            "model": "Whisper",
            "target": "CUDA/CT2",
            "format": "CT2 int8",
            "size": format_size(size_mb),
            "status": "OK",
        }

    except Exception as e:
        logger.error(f"  FAILED: {e}")
        return {"model": "Whisper", "target": "CUDA/CT2", "format": "CT2 int8", "size": "—", "status": f"FAIL: {e}"}


def validate_whisper_mlx():
    """Validate MLX Whisper by transcribing a short silence array."""
    logger.info("  Validating MLX Whisper...")
    try:
        import mlx_whisper
        import numpy as np

        silence = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        result = mlx_whisper.transcribe(
            silence,
            path_or_hf_repo=MLX_WHISPER_MODEL,
            condition_on_previous_text=False,
        )
        text = result.get("text", "").strip()
        logger.info(f"  MLX Whisper validation OK (output: '{text[:60]}')")
        return True
    except ImportError:
        logger.warning("  mlx_whisper not available — skipping MLX validation")
        return False
    except Exception as e:
        logger.warning(f"  MLX Whisper validation failed: {e}")
        return False


def validate_whisper_ct2(ct2_dir):
    """Validate CTranslate2 Whisper by transcribing a short silence array."""
    logger.info("  Validating CT2 Whisper (faster-whisper)...")
    try:
        import numpy as np
        from faster_whisper import WhisperModel

        model = WhisperModel(
            str(ct2_dir),
            device="cpu",  # Use CPU for validation even if CUDA available
            compute_type="int8",
        )
        silence = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        segments, info = model.transcribe(silence, language="en")
        text = " ".join(s.text for s in segments).strip()
        logger.info(f"  CT2 Whisper validation OK (output: '{text[:60]}')")
        del model
        return True
    except ImportError:
        logger.warning("  faster-whisper not available — skipping CT2 validation")
        return False
    except Exception as e:
        logger.warning(f"  CT2 Whisper validation failed: {e}")
        return False


# ---------------------------------------------------------------------------
# TranslateGemma Export
# ---------------------------------------------------------------------------


def export_gemma_mlx(output_dir, lora_path=None, dry_run=False):
    """Download/reference the pre-quantized MLX TranslateGemma 4B model.

    MLX TranslateGemma models from mlx-community are already 4-bit quantized.
    No conversion is needed — huggingface_hub caches them on first use.

    If a LoRA adapter path is provided, this documents that the adapter should
    be applied at inference time via mlx_lm's adapter loading (no merge needed
    for MLX — adapters are loaded dynamically).
    """
    logger.info("TranslateGemma 4B (MLX): Pre-quantized 4-bit from mlx-community")
    logger.info(f"  Model ID: {MLX_GEMMA_4B_MODEL}")
    logger.info("  Format:   MLX 4-bit quantized (no conversion needed)")

    if lora_path:
        lora_p = Path(lora_path)
        if not lora_p.is_absolute():
            lora_p = PROJECT_ROOT / lora_p
        logger.info(f"  LoRA:     {lora_p}")
        logger.info("  Note:     MLX loads LoRA adapters dynamically at inference time.")
        logger.info("            No merge step is needed for MLX deployment.")

    if dry_run:
        logger.info("  [DRY RUN] Would download/verify model via snapshot_download")
        return {
            "model": "TranslateGemma 4B",
            "target": "MLX",
            "format": "MLX 4-bit",
            "size": "~2.2 GB",
            "status": "SKIP (dry run)",
        }

    try:
        from huggingface_hub import snapshot_download

        t0 = time.time()
        cache_path = snapshot_download(MLX_GEMMA_4B_MODEL)
        elapsed = time.time() - t0
        size_mb = get_dir_size_mb(cache_path)
        logger.info(f"  Cached at: {cache_path}")
        logger.info(f"  Size:      {format_size(size_mb)}")
        logger.info(f"  Time:      {elapsed:.1f}s")
        return {
            "model": "TranslateGemma 4B",
            "target": "MLX",
            "format": "MLX 4-bit",
            "size": format_size(size_mb),
            "status": "OK",
        }
    except Exception as e:
        logger.error(f"  FAILED: {e}")
        return {
            "model": "TranslateGemma 4B",
            "target": "MLX",
            "format": "MLX 4-bit",
            "size": "—",
            "status": f"FAIL: {e}",
        }


def export_gemma_cuda(output_dir, lora_path=None, dry_run=False):
    """Prepare TranslateGemma 4B for CUDA inference with bitsandbytes 4-bit.

    For NVIDIA deployment, TranslateGemma is quantized on-the-fly at load time
    using bitsandbytes NF4 quantization. No separate export/conversion step is
    needed — the model loads directly from HuggingFace with:

        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "google/translategemma-4b-it",
            quantization_config=bnb_config,
            device_map="auto",
        )

    If a LoRA adapter path is provided, this function merges the adapter into
    the base model and saves the merged weights for direct loading.
    """
    logger.info("TranslateGemma 4B (CUDA): bitsandbytes 4-bit (on-the-fly quantization)")
    logger.info(f"  Model ID: {CUDA_GEMMA_4B_MODEL}")
    logger.info("  Format:   bitsandbytes NF4 (quantized at load time, no export needed)")
    logger.info("  Note:     Model is downloaded from HuggingFace and quantized on-the-fly")
    logger.info("            when loaded with BitsAndBytesConfig(load_in_4bit=True).")

    merged_dir = Path(output_dir) / "translategemma-4b-merged" if lora_path else None

    if lora_path:
        lora_p = Path(lora_path)
        if not lora_p.is_absolute():
            lora_p = PROJECT_ROOT / lora_p
        logger.info(f"  LoRA:     {lora_p}")
        logger.info(f"  Merge to: {merged_dir}")

    if dry_run:
        if lora_path:
            logger.info("  [DRY RUN] Would merge LoRA adapter into base model and save")
        else:
            logger.info("  [DRY RUN] Would download/verify base model via snapshot_download")
            logger.info("            (No conversion needed — quantized at load time)")
        return {
            "model": "TranslateGemma 4B",
            "target": "CUDA/bnb",
            "format": "bnb NF4",
            "size": "~8 GB (fp16)" if not lora_path else "~8 GB (merged)",
            "status": "SKIP (dry run)",
        }

    # If LoRA path provided, merge the adapter into the base model
    if lora_path:
        lora_p = Path(lora_path)
        if not lora_p.is_absolute():
            lora_p = PROJECT_ROOT / lora_p

        if not lora_p.exists():
            logger.error(f"  LoRA adapter path not found: {lora_p}")
            return {
                "model": "TranslateGemma 4B",
                "target": "CUDA/bnb",
                "format": "bnb NF4",
                "size": "—",
                "status": f"FAIL: LoRA path not found: {lora_p}",
            }

        if not _check_peft():
            logger.error("  peft not installed. Install with: pip install peft")
            return {
                "model": "TranslateGemma 4B",
                "target": "CUDA/bnb",
                "format": "bnb NF4",
                "size": "—",
                "status": "FAIL: peft not installed",
            }

        if not _check_transformers():
            logger.error("  transformers not installed. Install with: pip install transformers")
            return {
                "model": "TranslateGemma 4B",
                "target": "CUDA/bnb",
                "format": "bnb NF4",
                "size": "—",
                "status": "FAIL: transformers not installed",
            }

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            t0 = time.time()
            logger.info("  Loading base model in fp16 for LoRA merge...")
            base_model = AutoModelForCausalLM.from_pretrained(
                CUDA_GEMMA_4B_MODEL,
                torch_dtype=torch.float16,
                device_map="cpu",  # Merge on CPU to avoid VRAM pressure
            )

            logger.info(f"  Loading LoRA adapter from {lora_p}...")
            model = PeftModel.from_pretrained(base_model, str(lora_p))

            logger.info("  Merging LoRA weights into base model...")
            model = model.merge_and_unload()

            merged_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Saving merged model to {merged_dir}...")
            model.save_pretrained(str(merged_dir))

            # Copy tokenizer
            tokenizer = AutoTokenizer.from_pretrained(CUDA_GEMMA_4B_MODEL)
            tokenizer.save_pretrained(str(merged_dir))

            elapsed = time.time() - t0
            size_mb = get_dir_size_mb(merged_dir)
            logger.info(f"  Merged in {elapsed:.1f}s")
            logger.info(f"  Size: {format_size(size_mb)}")

            del model, base_model
            return {
                "model": "TranslateGemma 4B",
                "target": "CUDA/bnb",
                "format": "fp16 (merged)",
                "size": format_size(size_mb),
                "status": "OK",
            }

        except Exception as e:
            logger.error(f"  FAILED: {e}")
            return {
                "model": "TranslateGemma 4B",
                "target": "CUDA/bnb",
                "format": "bnb NF4",
                "size": "—",
                "status": f"FAIL: {e}",
            }

    # No LoRA — just ensure the base model is cached
    try:
        from huggingface_hub import snapshot_download

        t0 = time.time()
        cache_path = snapshot_download(CUDA_GEMMA_4B_MODEL)
        elapsed = time.time() - t0
        size_mb = get_dir_size_mb(cache_path)
        logger.info(f"  Cached at: {cache_path}")
        logger.info(f"  Size:      {format_size(size_mb)}")
        logger.info(f"  Time:      {elapsed:.1f}s")
        return {
            "model": "TranslateGemma 4B",
            "target": "CUDA/bnb",
            "format": "bnb NF4",
            "size": format_size(size_mb),
            "status": "OK",
        }
    except Exception as e:
        logger.error(f"  FAILED: {e}")
        return {
            "model": "TranslateGemma 4B",
            "target": "CUDA/bnb",
            "format": "bnb NF4",
            "size": "—",
            "status": f"FAIL: {e}",
        }


# ---------------------------------------------------------------------------
# MarianMT Export
# ---------------------------------------------------------------------------


def export_marian_pytorch(output_dir, dry_run=False):
    """Ensure MarianMT PyTorch weights are cached locally.

    MarianMT (Helsinki-NLP/opus-mt-en-es) is small enough (~298 MB) that no
    quantization is needed for Mac deployment. The PyTorch weights are used
    directly for both Mac and CUDA inference.
    """
    logger.info("MarianMT (PyTorch): Direct PyTorch weights — no conversion needed")
    logger.info(f"  Model ID: {CUDA_MARIAN_MODEL}")
    logger.info("  Format:   PyTorch fp32 (~298 MB)")
    logger.info("  Note:     Small enough to run on CPU without quantization")

    if dry_run:
        logger.info("  [DRY RUN] Would download/verify model via snapshot_download")
        return {
            "model": "MarianMT",
            "target": "PyTorch",
            "format": "fp32",
            "size": "~298 MB",
            "status": "SKIP (dry run)",
        }

    try:
        from huggingface_hub import snapshot_download

        t0 = time.time()
        cache_path = snapshot_download(CUDA_MARIAN_MODEL)
        elapsed = time.time() - t0
        size_mb = get_dir_size_mb(cache_path)
        logger.info(f"  Cached at: {cache_path}")
        logger.info(f"  Size:      {format_size(size_mb)}")
        logger.info(f"  Time:      {elapsed:.1f}s")
        return {
            "model": "MarianMT",
            "target": "PyTorch",
            "format": "fp32",
            "size": format_size(size_mb),
            "status": "OK",
        }
    except Exception as e:
        logger.error(f"  FAILED: {e}")
        return {"model": "MarianMT", "target": "PyTorch", "format": "fp32", "size": "—", "status": f"FAIL: {e}"}


def export_marian_ct2(output_dir, dry_run=False):
    """Convert MarianMT to CTranslate2 int8 for NVIDIA deployment.

    Uses ctranslate2.converters.OpusMTConverter to produce an int8 quantized
    model. The CT2 model is ~76 MB and runs inference in ~50ms — significantly
    faster than the PyTorch fp32 variant (~80ms).
    """
    ct2_dir = Path(output_dir) / "marian-ct2"

    logger.info("MarianMT (CUDA/CT2): Converting to CTranslate2 int8")
    logger.info(f"  Source:  {CUDA_MARIAN_MODEL}")
    logger.info(f"  Output:  {ct2_dir}")
    logger.info("  Format:  CTranslate2 int8 (~76 MB, ~50ms inference)")

    if dry_run:
        logger.info("  [DRY RUN] Would convert model via OpusMTConverter")
        return {
            "model": "MarianMT",
            "target": "CUDA/CT2",
            "format": "CT2 int8",
            "size": "~76 MB",
            "status": "SKIP (dry run)",
        }

    if not _check_ctranslate2():
        logger.error("  ctranslate2 not installed. Install with: pip install ctranslate2")
        return {
            "model": "MarianMT",
            "target": "CUDA/CT2",
            "format": "CT2 int8",
            "size": "—",
            "status": "FAIL: ctranslate2 not installed",
        }

    try:
        import ctranslate2

        t0 = time.time()
        ct2_dir.mkdir(parents=True, exist_ok=True)

        converter = ctranslate2.converters.OpusMTConverter(CUDA_MARIAN_MODEL)
        converter.convert(
            output_dir=str(ct2_dir),
            quantization="int8",
            force=True,
        )

        elapsed = time.time() - t0
        size_mb = get_dir_size_mb(ct2_dir)
        logger.info(f"  Converted in {elapsed:.1f}s")
        logger.info(f"  Size: {format_size(size_mb)}")
        return {
            "model": "MarianMT",
            "target": "CUDA/CT2",
            "format": "CT2 int8",
            "size": format_size(size_mb),
            "status": "OK",
        }

    except Exception as e:
        logger.error(f"  FAILED: {e}")
        return {"model": "MarianMT", "target": "CUDA/CT2", "format": "CT2 int8", "size": "—", "status": f"FAIL: {e}"}


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary(results):
    """Print a formatted summary table of all export results."""
    if not results:
        logger.info("No exports performed.")
        return

    # Column widths
    col_model = max(len(r["model"]) for r in results)
    col_target = max(len(r["target"]) for r in results)
    col_format = max(len(r["format"]) for r in results)
    col_size = max(len(r["size"]) for r in results)
    col_status = max(len(r["status"]) for r in results)

    # Minimum widths for header readability
    col_model = max(col_model, 10)
    col_target = max(col_target, 8)
    col_format = max(col_format, 8)
    col_size = max(col_size, 8)
    col_status = max(col_status, 8)

    header = (
        f"  {'Model':<{col_model}}  "
        f"{'Target':<{col_target}}  "
        f"{'Format':<{col_format}}  "
        f"{'Size':<{col_size}}  "
        f"{'Status':<{col_status}}"
    )
    separator = "  " + "-" * (col_model + col_target + col_format + col_size + col_status + 8)

    print()
    print(f"{'=' * (len(separator) + 2)}")
    print("  EXPORT SUMMARY")
    print(f"{'=' * (len(separator) + 2)}")
    print(header)
    print(separator)

    for r in results:
        status_prefix = "OK  " if r["status"] == "OK" else "!!! " if r["status"].startswith("FAIL") else "--- "
        print(
            f"  {r['model']:<{col_model}}  "
            f"{r['target']:<{col_target}}  "
            f"{r['format']:<{col_format}}  "
            f"{r['size']:<{col_size}}  "
            f"{status_prefix}{r['status']}"
        )

    print(f"{'=' * (len(separator) + 2)}")

    # Count outcomes
    ok_count = sum(1 for r in results if r["status"] == "OK")
    fail_count = sum(1 for r in results if r["status"].startswith("FAIL"))
    skip_count = sum(1 for r in results if "dry run" in r["status"].lower() or "SKIP" in r["status"])

    print(f"  {ok_count} OK, {fail_count} failed, {skip_count} skipped")
    print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_exports(args):
    """Execute the requested model exports and return results."""
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    do_mlx = not args.skip_mlx
    do_cuda = not args.skip_cuda
    lora_path = getattr(args, "lora_path", None)
    dry_run = args.dry_run
    validate = not args.no_validate

    # -----------------------------------------------------------------------
    # Whisper
    # -----------------------------------------------------------------------
    if args.whisper or args.all:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  WHISPER LARGE-V3-TURBO")
        logger.info("=" * 60)

        if do_mlx:
            r = export_whisper_mlx(output_dir, dry_run=dry_run)
            results.append(r)
            if validate and r["status"] == "OK" and _check_mlx():
                validate_whisper_mlx()

        if do_cuda:
            r = export_whisper_ct2(output_dir, dry_run=dry_run)
            results.append(r)
            if validate and r["status"] == "OK":
                ct2_dir = output_dir / "whisper-turbo-ct2"
                validate_whisper_ct2(ct2_dir)

    # -----------------------------------------------------------------------
    # TranslateGemma
    # -----------------------------------------------------------------------
    if args.gemma or args.all:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  TRANSLATEGEMMA 4B")
        logger.info("=" * 60)

        if do_mlx:
            results.append(export_gemma_mlx(output_dir, lora_path=lora_path, dry_run=dry_run))

        if do_cuda:
            results.append(export_gemma_cuda(output_dir, lora_path=lora_path, dry_run=dry_run))

    # -----------------------------------------------------------------------
    # MarianMT
    # -----------------------------------------------------------------------
    if args.marian or args.all:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  MARIANMT (Helsinki-NLP/opus-mt-en-es)")
        logger.info("=" * 60)

        # PyTorch weights are used on both targets (Mac CPU + CUDA)
        results.append(export_marian_pytorch(output_dir, dry_run=dry_run))

        if do_cuda:
            results.append(export_marian_ct2(output_dir, dry_run=dry_run))

    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Dual-endpoint model export for Mac (MLX) + NVIDIA (CTranslate2/faster-whisper). "
            "Converts and quantizes Whisper, TranslateGemma, and MarianMT for deployment."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tools/convert_models_to_both.py --all\n"
            "  python tools/convert_models_to_both.py --whisper --skip-mlx\n"
            "  python tools/convert_models_to_both.py --gemma --lora-path fine_tuned_gemma_mi_A/\n"
            "  python tools/convert_models_to_both.py --all --dry-run\n"
        ),
    )

    # Model selection (at least one required unless --all)
    model_group = parser.add_argument_group("model selection")
    model_group.add_argument(
        "--whisper",
        action="store_true",
        help="Export Whisper Large-V3-Turbo for both targets",
    )
    model_group.add_argument(
        "--gemma",
        action="store_true",
        help="Export TranslateGemma 4B for both targets",
    )
    model_group.add_argument(
        "--marian",
        action="store_true",
        help="Export MarianMT (opus-mt-en-es) for both targets",
    )
    model_group.add_argument(
        "--all",
        action="store_true",
        help="Export all models (Whisper + TranslateGemma + MarianMT)",
    )

    # Target selection
    target_group = parser.add_argument_group("target selection")
    target_group.add_argument(
        "--skip-mlx",
        action="store_true",
        help="Skip MLX (Mac/Apple Silicon) exports",
    )
    target_group.add_argument(
        "--skip-cuda",
        action="store_true",
        help="Skip CUDA (NVIDIA/CTranslate2) exports",
    )

    # Options
    options_group = parser.add_argument_group("options")
    options_group.add_argument(
        "--output-dir",
        default="./exported_models/",
        help="Base output directory for converted models (default: ./exported_models/)",
    )
    options_group.add_argument(
        "--lora-path",
        default=None,
        help="Path to LoRA adapter directory to merge before export (TranslateGemma only)",
    )
    options_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing any conversions or downloads",
    )
    options_group.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip post-export validation inference tests",
    )

    args = parser.parse_args()

    # Require at least one model selection
    if not (args.whisper or args.gemma or args.marian or args.all):
        parser.error("Specify at least one of: --whisper, --gemma, --marian, --all")

    # Warn if both targets skipped
    if args.skip_mlx and args.skip_cuda:
        parser.error("Cannot skip both --skip-mlx and --skip-cuda — nothing to export")

    # Header
    logger.info("=" * 60)
    logger.info("  Dual-Endpoint Model Export")
    logger.info("=" * 60)
    logger.info(f"  Output dir:  {args.output_dir}")
    logger.info(f"  MLX target:  {'SKIP' if args.skip_mlx else 'enabled'}")
    logger.info(f"  CUDA target: {'SKIP' if args.skip_cuda else 'enabled'}")
    logger.info(f"  Dry run:     {args.dry_run}")
    if args.lora_path:
        logger.info(f"  LoRA path:   {args.lora_path}")
    logger.info("")

    results = run_exports(args)
    print_summary(results)

    # Exit code: 0 if all OK or skipped, 1 if any failures
    has_failures = any(r["status"].startswith("FAIL") for r in results)
    return 1 if has_failures else 0


if __name__ == "__main__":
    sys.exit(main() or 0)
