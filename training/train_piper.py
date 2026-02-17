#!/usr/bin/env python3
"""
train_piper.py — Piper TTS Voice Fine-Tuning Wrapper for Church Audio

Wraps Piper's VITS-based training workflow for church-domain voice fine-tuning.
Runs on Windows/WSL desktop (A2000 Ada 16GB VRAM).

Piper uses a VITS (Variational Inference Text-to-Speech) architecture with
phoneme-level input. Training fine-tunes a pretrained voice checkpoint on
church-specific audio to capture the speaker's voice characteristics, cadence,
and pronunciation of theological terms.

VRAM: ~4-8 GB depending on batch size (VITS is much lighter than LLMs)
Training time: ~2-6 hrs for 3000 epochs on 1-2 hrs of audio data

Usage:
    python train_piper.py --lang en
    python train_piper.py --lang es --dataset stark_data/piper_dataset/es
    python train_piper.py --lang en --epochs 5000 --batch-size 32
    python train_piper.py --lang en --checkpoint logs/piper_train_en_.../epoch=2000.ckpt
    python train_piper.py --lang en --sample-every 100
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base checkpoints — pretrained Piper voices to fine-tune from
# ---------------------------------------------------------------------------
# These are the recommended starting points for each language. Piper provides
# pretrained VITS checkpoints that already have good phoneme coverage for the
# target language. Fine-tuning from these converges much faster than training
# from scratch.
#
# TODO: Confirm exact checkpoint names from Piper's model repository at
#       https://github.com/rhasspy/piper/blob/master/VOICES.md
#       The names below follow Piper's {lang}_{region}-{speaker}-{quality} convention.
PIPER_BASE_CHECKPOINTS = {
    "en": "en_US-lessac-high",
    "es": "es_ES-carlfm-high",
    "hi": "hi_IN-kusal-medium",
    "zh": "zh_CN-huayan-medium",
}

# Test sentences for periodic synthesis during training — one per language.
# These are biblical benedictions, matching the church domain.
TEST_SENTENCES = {
    "en": "The grace of our Lord Jesus Christ be with you all.",
    "es": "La gracia de nuestro Señor Jesucristo sea con todos vosotros.",
    "hi": "\u0939\u092e\u093e\u0930\u0947 \u092a\u094d\u0930\u092d\u0941 \u092f\u0940\u0936\u0941 \u092e\u0938\u0940\u0939 \u0915\u0940 \u0915\u0943\u092a\u093e \u0924\u0941\u092e \u0938\u092c \u0915\u0947 \u0938\u093e\u0925 \u0939\u094b\u0964",
    "zh": "\u613f\u6211\u4eec\u4e3b\u8036\u7a23\u57fa\u7763\u7684\u6069\u60e0\u4e0e\u4f60\u4eec\u4f17\u4eba\u540c\u5728\u3002",
}

# Piper's default sample rate for VITS models
PIPER_SAMPLE_RATE = 22050


def validate_dataset(dataset_dir):
    """Validate that the dataset directory has the required LJSpeech structure.

    Expected:
        dataset_dir/
        +-- wav/           # Audio files
        +-- metadata.csv   # filename|transcription|normalized_transcription
    """
    dataset_path = Path(dataset_dir)
    metadata_path = dataset_path / "metadata.csv"
    wav_dir = dataset_path / "wav"

    if not dataset_path.exists():
        logger.error(f"Dataset directory does not exist: {dataset_path}")
        return False

    if not metadata_path.exists():
        logger.error(f"metadata.csv not found in {dataset_path}")
        logger.error("Run prepare_piper_dataset.py first to create the dataset.")
        return False

    if not wav_dir.exists():
        logger.error(f"wav/ directory not found in {dataset_path}")
        return False

    # Count entries in metadata.csv
    n_entries = 0
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n_entries += 1

    # Count WAV files
    wav_files = list(wav_dir.glob("*.wav"))
    n_wavs = len(wav_files)

    if n_entries == 0:
        logger.error("metadata.csv is empty")
        return False

    if n_wavs == 0:
        logger.error("No WAV files found in wav/ directory")
        return False

    if n_entries != n_wavs:
        logger.warning(f"metadata.csv has {n_entries} entries but wav/ has {n_wavs} files")

    logger.info(f"Dataset validated: {n_entries} entries, {n_wavs} WAV files")
    return True


def run_piper_preprocessing(dataset_dir, output_dir, lang):
    """Run Piper's preprocessing step to convert text to phonemes.

    Piper requires a preprocessing step that:
    1. Reads metadata.csv
    2. Converts text to phoneme sequences using espeak-ng
    3. Creates a training-ready dataset with phoneme IDs

    TODO: Confirm the exact piper_train preprocessing command. Piper's training
    code may use a different entry point or expect the data in a specific format.
    The command below follows the pattern from Piper's training documentation.
    """
    logger.info("Running Piper preprocessing (text -> phonemes)...")

    # TODO: Confirm this is the correct preprocessing command for piper-tts.
    # Piper's training pipeline may use `python -m piper_train.preprocess`
    # or a standalone script. The --language flag maps to espeak-ng language codes.
    espeak_lang_map = {
        "en": "en-us",
        "es": "es",
        "hi": "hi",
        "zh": "cmn",  # Mandarin Chinese in espeak-ng
    }
    espeak_lang = espeak_lang_map.get(lang, lang)

    preprocess_cmd = [
        "python", "-m", "piper_train.preprocess",
        "--language", espeak_lang,
        "--input-dir", str(dataset_dir),
        "--output-dir", str(output_dir),
        "--dataset-format", "ljspeech",
        "--sample-rate", str(PIPER_SAMPLE_RATE),
    ]

    logger.info(f"  Command: {' '.join(preprocess_cmd)}")

    try:
        result = subprocess.run(
            preprocess_cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout for preprocessing
        )
        if result.returncode != 0:
            logger.error(f"Preprocessing failed (exit code {result.returncode})")
            if result.stderr:
                logger.error(f"  stderr: {result.stderr[:500]}")
            return False
        if result.stdout:
            logger.info(f"  stdout: {result.stdout[:500]}")
    except FileNotFoundError:
        logger.error("piper_train not found. Install with: pip install piper-tts")
        logger.error("See https://github.com/rhasspy/piper for installation.")
        return False
    except subprocess.TimeoutExpired:
        logger.error("Preprocessing timed out after 30 minutes")
        return False

    logger.info("Preprocessing complete")
    return True


def synthesize_test_sample(checkpoint_path, lang, output_wav, text=None):
    """Synthesize a test sentence using a training checkpoint.

    Called periodically during training to monitor voice quality.

    TODO: Confirm the correct inference command for Piper checkpoints (.ckpt).
    During training, Piper may expose a synthesis function or require exporting
    to ONNX first. This uses the piper_train inference mode as a best guess.
    """
    if text is None:
        text = TEST_SENTENCES.get(lang, TEST_SENTENCES["en"])

    # TODO: Confirm this command works with .ckpt files during training.
    # Piper's normal inference uses ONNX models, but during training we may
    # need to use the PyTorch checkpoint directly via piper_train.
    synth_cmd = [
        "python", "-m", "piper_train.infer",
        "--checkpoint", str(checkpoint_path),
        "--output", str(output_wav),
        "--text", text,
    ]

    try:
        result = subprocess.run(
            synth_cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            logger.info(f"  Test sample saved: {output_wav}")
            return True
        else:
            logger.warning(f"  Test synthesis failed: {result.stderr[:200] if result.stderr else 'unknown error'}")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning(f"  Test synthesis skipped: {e}")
        return False


def find_latest_checkpoint(log_dir):
    """Find the most recent checkpoint in a training log directory.

    Piper saves checkpoints as epoch=NNNN-step=MMMM.ckpt files.

    Returns path to the latest checkpoint, or None.
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return None

    # Search for checkpoint files in common locations
    ckpt_patterns = [
        "*.ckpt",
        "checkpoints/*.ckpt",
        "lightning_logs/*/checkpoints/*.ckpt",
    ]

    all_ckpts = []
    for pattern in ckpt_patterns:
        all_ckpts.extend(log_path.glob(pattern))

    if not all_ckpts:
        return None

    # Sort by modification time (most recent last)
    all_ckpts.sort(key=lambda p: p.stat().st_mtime)
    return all_ckpts[-1]


def list_checkpoints(log_dir):
    """List all training checkpoints sorted by epoch."""
    log_path = Path(log_dir)
    if not log_path.exists():
        return []

    ckpt_patterns = [
        "*.ckpt",
        "checkpoints/*.ckpt",
        "lightning_logs/*/checkpoints/*.ckpt",
    ]

    all_ckpts = []
    for pattern in ckpt_patterns:
        all_ckpts.extend(log_path.glob(pattern))

    # Sort by name (which includes epoch number)
    all_ckpts.sort(key=lambda p: p.name)
    return all_ckpts


def train_piper(
    lang="en",
    dataset_dir=None,
    checkpoint=None,
    batch_size=16,
    epochs=3000,
    lr=1e-4,
    sample_every=50,
):
    """Run Piper TTS training via piper_train.

    Fine-tunes a pretrained Piper voice checkpoint on church audio to capture
    the speaker's voice, cadence, and pronunciation patterns.

    TODO: Confirm piper_train CLI arguments. The training command below follows
    Piper's documented training workflow but may need adjustments based on the
    installed version of piper-tts.
    """
    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent

    if dataset_dir is None:
        dataset_dir = project_root / "stark_data" / "piper_dataset" / lang
    else:
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.is_absolute():
            dataset_dir = project_root / dataset_dir

    # Validate dataset
    if not validate_dataset(dataset_dir):
        sys.exit(1)

    # Determine base checkpoint
    base_checkpoint = PIPER_BASE_CHECKPOINTS.get(lang)
    if base_checkpoint is None:
        logger.error(f"No base checkpoint defined for language: {lang}")
        logger.error(f"Available: {list(PIPER_BASE_CHECKPOINTS.keys())}")
        sys.exit(1)

    # Setup logging directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = project_root / "logs" / f"piper_train_{lang}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Preprocessing: convert text to phonemes
    preprocessed_dir = dataset_dir / "_preprocessed"
    if not run_piper_preprocessing(dataset_dir, preprocessed_dir, lang):
        logger.error("Preprocessing failed. Cannot continue.")
        sys.exit(1)

    # Build training command
    # TODO: Confirm the exact piper_train training CLI. The arguments below
    # are based on Piper's training documentation and common PyTorch Lightning
    # patterns. Key parameters that may differ:
    #   --dataset-dir vs --data-dir
    #   --quality high/medium/low
    #   --max-epochs vs --max_epochs
    train_cmd = [
        "python", "-m", "piper_train",
        "--dataset-dir", str(preprocessed_dir),
        "--accelerator", "gpu",
        "--devices", "1",
        "--batch-size", str(batch_size),
        "--validation-split", "0.05",
        "--max-epochs", str(epochs),
        "--learning-rate", str(lr),
        "--checkpoint-epochs", str(sample_every),
        "--default-root-dir", str(log_dir),
        "--quality", "high",
    ]

    # Resume from checkpoint if specified
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = project_root / ckpt_path
        if not ckpt_path.exists():
            logger.error(f"Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        train_cmd.extend(["--resume-from-checkpoint", str(ckpt_path)])
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
    else:
        # TODO: Confirm how to specify the pretrained base model to fine-tune from.
        # Piper may download the model automatically by name, or may require a
        # local path to a .ckpt file. The flag might be --pretrained-model or
        # --resume-from-single-speaker-checkpoint.
        train_cmd.extend(["--pretrained", base_checkpoint])
        logger.info(f"Fine-tuning from base checkpoint: {base_checkpoint}")

    # Save training config for reproducibility
    config = {
        "lang": lang,
        "dataset_dir": str(dataset_dir),
        "base_checkpoint": base_checkpoint,
        "resume_checkpoint": str(checkpoint) if checkpoint else None,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "sample_every": sample_every,
        "log_dir": str(log_dir),
        "timestamp": timestamp,
        "command": train_cmd,
    }
    config_path = log_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Training config saved to {config_path}")

    # Launch training
    logger.info(f"\nStarting Piper training...")
    logger.info(f"  Language:    {lang}")
    logger.info(f"  Base model:  {base_checkpoint}")
    logger.info(f"  Dataset:     {dataset_dir}")
    logger.info(f"  Batch size:  {batch_size}")
    logger.info(f"  Epochs:      {epochs}")
    logger.info(f"  LR:          {lr}")
    logger.info(f"  Log dir:     {log_dir}")
    logger.info(f"  Command:     {' '.join(train_cmd)}")

    start_time = time.time()

    try:
        process = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output and log to file
        train_log_path = log_dir / "train_output.log"
        with open(train_log_path, "w") as log_file:
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()

        process.wait()
        elapsed = time.time() - start_time

        if process.returncode != 0:
            logger.error(f"Training exited with code {process.returncode}")
            logger.error(f"Check logs at {train_log_path}")
            sys.exit(1)

    except FileNotFoundError:
        logger.error("piper_train not found. Install with: pip install piper-tts")
        logger.error("See https://github.com/rhasspy/piper for installation.")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
        process.terminate()
        process.wait()
        elapsed = time.time() - start_time

    # Post-training: list checkpoints and synthesize final test sample
    logger.info(f"\nTraining complete in {elapsed/3600:.1f} hours")
    logger.info(f"Logs: {log_dir}")
    logger.info(f"Train output: {train_log_path}")

    checkpoints = list_checkpoints(log_dir)
    if checkpoints:
        logger.info(f"\nCheckpoints ({len(checkpoints)} total):")
        for ckpt in checkpoints:
            size_mb = ckpt.stat().st_size / (1024 * 1024)
            logger.info(f"  {ckpt.name}  ({size_mb:.1f} MB)")

        # Synthesize test sample from the latest checkpoint
        latest = checkpoints[-1]
        test_wav = log_dir / "test_final.wav"
        logger.info(f"\nSynthesizing test sample from {latest.name}...")
        synthesize_test_sample(latest, lang, test_wav)
    else:
        logger.warning("No checkpoints found after training")


def main():
    parser = argparse.ArgumentParser(
        description="Piper TTS voice fine-tuning for church audio"
    )
    parser.add_argument("--lang", default="en", choices=["en", "es", "hi", "zh"],
                        help="Target language (default: en)")
    parser.add_argument("--dataset", "-d", default=None,
                        help="Path to Piper dataset "
                        "(default: stark_data/piper_dataset/{lang})")
    parser.add_argument("--checkpoint", "-c", default=None,
                        help="Path to .ckpt file to resume training from")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size (default: 16)")
    parser.add_argument("--epochs", type=int, default=3000,
                        help="Maximum training epochs (default: 3000)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--sample-every", type=int, default=50,
                        help="Synthesize test sample every N epochs (default: 50)")
    args = parser.parse_args()

    train_piper(
        lang=args.lang,
        dataset_dir=args.dataset,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        sample_every=args.sample_every,
    )


if __name__ == "__main__":
    main()
