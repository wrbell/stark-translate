#!/usr/bin/env python3
"""
export_piper_onnx.py — Export Trained Piper Checkpoint to ONNX for Deployment

Converts a trained Piper .ckpt file (PyTorch Lightning VITS checkpoint) to
ONNX format with a companion JSON config file. The exported model can be
loaded by PiperVoice for inference on any platform without PyTorch.

Output structure:
    piper_voices/{lang}/
    +-- voice.onnx          # ONNX model (~50-100 MB depending on quality)
    +-- voice.onnx.json     # Voice config (sample_rate, phoneme_type, etc.)

Usage:
    python export_piper_onnx.py --checkpoint logs/piper_train_en_.../epoch=3000.ckpt --lang en
    python export_piper_onnx.py --checkpoint model.ckpt --lang es --output piper_voices/es
    python export_piper_onnx.py --checkpoint model.ckpt --lang en --optimize
    python export_piper_onnx.py --checkpoint model.ckpt --lang en --no-validate
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default sample rate for Piper VITS models
PIPER_SAMPLE_RATE = 22050

# Test sentences for validation synthesis (same as train_piper.py)
TEST_SENTENCES = {
    "en": "The grace of our Lord Jesus Christ be with you all.",
    "es": "La gracia de nuestro Señor Jesucristo sea con todos vosotros.",
    "hi": "\u0939\u092e\u093e\u0930\u0947 \u092a\u094d\u0930\u092d\u0941 \u092f\u0940\u0936\u0941 \u092e\u0938\u0940\u0939 \u0915\u0940 \u0915\u0943\u092a\u093e \u0924\u0941\u092e \u0938\u092c \u0915\u0947 \u0938\u093e\u0925 \u0939\u094b\u0964",
    "zh": "\u613f\u6211\u4eec\u4e3b\u8036\u7a23\u57fa\u7763\u7684\u6069\u60e0\u4e0e\u4f60\u4eec\u4f17\u4eba\u540c\u5728\u3002",
}

# Language metadata for the voice config JSON
# Maps our language codes to Piper's expected format
LANGUAGE_METADATA = {
    "en": {
        "language_code": "en-us",
        "language_family": "en",
        "language_region": "US",
        "phoneme_type": "espeak",
        "espeak_voice": "en-us",
    },
    "es": {
        "language_code": "es-es",
        "language_family": "es",
        "language_region": "ES",
        "phoneme_type": "espeak",
        "espeak_voice": "es",
    },
    "hi": {
        "language_code": "hi-in",
        "language_family": "hi",
        "language_region": "IN",
        "phoneme_type": "espeak",
        "espeak_voice": "hi",
    },
    "zh": {
        "language_code": "zh-cn",
        "language_family": "zh",
        "language_region": "CN",
        "phoneme_type": "espeak",
        "espeak_voice": "cmn",
    },
}


def export_checkpoint_to_onnx(checkpoint_path, output_onnx_path):
    """Export a Piper .ckpt checkpoint to ONNX format.

    TODO: Confirm the exact export command/API for Piper. Piper's training
    repository provides an export utility, but the exact interface may vary:

    Option A (CLI):
        python -m piper_train.export_onnx checkpoint.ckpt output.onnx

    Option B (Python API):
        from piper_train.export_onnx import export_onnx
        export_onnx(checkpoint_path, output_path)

    Option C (Standalone script):
        piper_train/export_onnx.py checkpoint.ckpt output.onnx

    The command below uses Option A as the most likely pattern.
    """
    logger.info("Exporting checkpoint to ONNX...")
    logger.info(f"  Input:  {checkpoint_path}")
    logger.info(f"  Output: {output_onnx_path}")

    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)

    # TODO: Confirm this is the correct export command for the installed
    # version of piper-tts / piper-train.
    export_cmd = [
        "python",
        "-m",
        "piper_train.export_onnx",
        str(checkpoint_path),
        str(output_onnx_path),
    ]

    logger.info(f"  Command: {' '.join(export_cmd)}")

    try:
        result = subprocess.run(
            export_cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
        )
        if result.returncode != 0:
            logger.error(f"ONNX export failed (exit code {result.returncode})")
            if result.stderr:
                logger.error(f"  stderr: {result.stderr[:500]}")
            return False
        if result.stdout:
            logger.info(f"  {result.stdout.strip()}")
    except FileNotFoundError:
        logger.error("piper_train not found. Install with: pip install piper-tts")
        return False
    except subprocess.TimeoutExpired:
        logger.error("Export timed out after 5 minutes")
        return False

    if not Path(output_onnx_path).exists():
        logger.error(f"Export completed but ONNX file not found at {output_onnx_path}")
        return False

    size_mb = Path(output_onnx_path).stat().st_size / (1024 * 1024)
    logger.info(f"  ONNX model exported: {size_mb:.1f} MB")
    return True


def generate_voice_config(output_json_path, lang, checkpoint_path=None):
    """Generate the companion .onnx.json config file for a Piper voice.

    This JSON file tells PiperVoice how to load and use the ONNX model:
    sample rate, phoneme system, language, speaker info, etc.

    TODO: Confirm the exact schema expected by PiperVoice. The schema below
    is based on existing Piper voice config files from the model repository.
    Fields like "num_speakers", "speaker_id_map", and "inference" may need
    adjustment based on the training configuration.
    """
    lang_meta = LANGUAGE_METADATA.get(lang, LANGUAGE_METADATA["en"])

    config = {
        "audio": {
            "sample_rate": PIPER_SAMPLE_RATE,
            "quality": "high",
        },
        "espeak": {
            "voice": lang_meta["espeak_voice"],
        },
        "language": {
            "code": lang_meta["language_code"],
            "family": lang_meta["language_family"],
            "region": lang_meta["language_region"],
            "name_native": "",
            "name_english": "",
            "country_english": "",
        },
        "model": {
            "architecture": "vits",
            "key": f"church-{lang}",
            "description": f"Church-domain Piper voice ({lang}), fine-tuned on Stark Road Gospel Hall audio",
        },
        "inference": {
            "noise_scale": 0.667,
            "length_scale": 1.0,
            "noise_w": 0.8,
            "phoneme_silence": {
                ",": 0.15,
                ".": 0.30,
                "?": 0.30,
                "!": 0.30,
                ";": 0.20,
                ":": 0.20,
            },
        },
        "num_speakers": 1,
        "speaker_id_map": {},
        "phoneme_type": lang_meta["phoneme_type"],
        "phoneme_map": {},
        "phoneme_id_map": {},
        # Provenance metadata
        "_training": {
            "source_checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "project": "SRTranslate",
            "domain": "church",
        },
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"  Voice config written: {output_json_path}")
    return True


def optimize_onnx(onnx_path):
    """Run onnx-simplifier to reduce model size and improve inference speed.

    onnxsim performs constant folding, dead code elimination, and graph
    optimization. Typically reduces model size by 5-15%.
    """
    logger.info("Optimizing ONNX model with onnx-simplifier...")

    original_size = Path(onnx_path).stat().st_size
    optimized_path = str(onnx_path) + ".optimized"

    try:
        result = subprocess.run(
            ["python", "-m", "onnxsim", str(onnx_path), optimized_path],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.warning(f"onnx-simplifier failed: {result.stderr[:200] if result.stderr else 'unknown error'}")
            logger.warning("Continuing with unoptimized model")
            return False

        # Replace original with optimized version
        optimized_size = Path(optimized_path).stat().st_size
        os.replace(optimized_path, onnx_path)

        reduction = (1 - optimized_size / original_size) * 100
        logger.info(f"  Original:  {original_size / (1024 * 1024):.1f} MB")
        logger.info(f"  Optimized: {optimized_size / (1024 * 1024):.1f} MB")
        logger.info(f"  Reduction: {reduction:.1f}%")
        return True

    except FileNotFoundError:
        logger.warning("onnx-simplifier not installed. Install with: pip install onnxsim")
        logger.warning("Continuing with unoptimized model")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("onnx-simplifier timed out after 5 minutes")
        # Clean up partial output
        if Path(optimized_path).exists():
            os.remove(optimized_path)
        return False


def validate_exported_model(onnx_path, config_path, lang):
    """Validate the exported model by synthesizing a test sentence.

    Loads the ONNX model via PiperVoice and synthesizes one sentence to verify
    the export was successful and the model produces valid audio.

    TODO: Confirm the PiperVoice API for loading and synthesizing from an
    ONNX model + config. The API below is based on Piper's documented usage.
    """
    logger.info("Validating exported model...")

    test_text = TEST_SENTENCES.get(lang, TEST_SENTENCES["en"])
    test_wav = Path(onnx_path).parent / "test_validation.wav"

    try:
        # TODO: Confirm this import path. It may be `piper` or `piper_tts`
        # depending on the installed package.
        from piper import PiperVoice

        voice = PiperVoice.load(str(onnx_path), config_path=str(config_path))

        # Synthesize test sentence
        import wave

        with wave.open(str(test_wav), "wb") as wf:
            voice.synthesize(test_text, wf)

        # Check the output
        if test_wav.exists() and test_wav.stat().st_size > 0:
            with wave.open(str(test_wav), "rb") as wf:
                duration = wf.getnframes() / wf.getframerate()
            logger.info("  Validation successful!")
            logger.info(f"  Test audio: {test_wav}")
            logger.info(f"  Duration:   {duration:.2f}s")
            logger.info(f'  Text:       "{test_text}"')
            return True
        else:
            logger.warning("  Validation produced empty audio")
            return False

    except ImportError:
        logger.warning("  piper-tts not installed for validation. Install with: pip install piper-tts")
        logger.warning("  Skipping validation. Model may still be valid.")
        return False
    except Exception as e:
        logger.warning(f"  Validation failed: {e}")
        logger.warning("  The exported model may still work. Check manually.")
        return False


def export_piper(checkpoint, output_dir, lang, optimize=False, validate=True):
    """Full export pipeline: .ckpt -> .onnx + .json, with optional optimization.

    Steps:
    1. Export .ckpt to .onnx via piper_train.export_onnx
    2. Generate companion .onnx.json config with voice metadata
    3. Optionally run onnx-simplifier for size reduction
    4. Optionally validate by loading and synthesizing a test sentence
    """
    project_root = Path(__file__).resolve().parent.parent

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    if output_dir is None:
        output_dir = project_root / "piper_voices" / lang
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / "voice.onnx"
    config_path = output_dir / "voice.onnx.json"

    logger.info("Piper ONNX Export")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Output:     {output_dir}")
    logger.info(f"  Language:   {lang}")
    logger.info(f"  Optimize:   {optimize}")
    logger.info(f"  Validate:   {validate}")

    # Step 1: Export to ONNX
    if not export_checkpoint_to_onnx(checkpoint_path, onnx_path):
        logger.error("ONNX export failed")
        sys.exit(1)

    # Step 2: Generate config JSON
    generate_voice_config(config_path, lang, checkpoint_path)

    # Step 3: Optimize (optional)
    if optimize:
        optimize_onnx(onnx_path)

    # Step 4: Validate (optional)
    if validate:
        validate_exported_model(onnx_path, config_path, lang)

    # Final summary
    onnx_size = onnx_path.stat().st_size / (1024 * 1024)
    config_size = config_path.stat().st_size / 1024

    logger.info(f"\n{'=' * 50}")
    logger.info("Export Summary")
    logger.info(f"{'=' * 50}")
    logger.info(f"  ONNX model:  {onnx_path}  ({onnx_size:.1f} MB)")
    logger.info(f"  Voice config: {config_path}  ({config_size:.1f} KB)")
    logger.info(f"  Language:     {lang}")
    logger.info(f"  Sample rate:  {PIPER_SAMPLE_RATE} Hz")
    logger.info("")
    logger.info("  To use this voice:")
    logger.info("    from piper import PiperVoice")
    logger.info(f'    voice = PiperVoice.load("{onnx_path}")')


def main():
    parser = argparse.ArgumentParser(description="Export trained Piper checkpoint to ONNX for deployment")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to the .ckpt file from train_piper.py")
    parser.add_argument("--output", "-o", default=None, help="Output directory (default: piper_voices/{lang})")
    parser.add_argument(
        "--lang", default="en", choices=["en", "es", "hi", "zh"], help="Language for voice metadata (default: en)"
    )
    parser.add_argument("--optimize", action="store_true", help="Run onnx-simplifier for size reduction")
    parser.add_argument("--no-validate", action="store_true", help="Skip validation synthesis after export")
    args = parser.parse_args()

    export_piper(
        checkpoint=args.checkpoint,
        output_dir=args.output,
        lang=args.lang,
        optimize=args.optimize,
        validate=not args.no_validate,
    )


if __name__ == "__main__":
    main()
