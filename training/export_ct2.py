#!/usr/bin/env python3
"""export_ct2.py — Merge LoRA adapter -> bf16 Whisper -> CTranslate2, with verification.

Mirrors training/export_gguf.py's structure but for Whisper STT models. Output
loads via faster_whisper.WhisperModel and slots into the project's existing
FasterWhisperEngine code path with no engine changes (the engine accepts a
local CT2 directory as model_id).

Steps (any failure aborts the pipeline):
  1. Load base Whisper in bf16, attach LoRA adapter via peft, merge_and_unload().
     Saves the merged model to a temp dir.
     CRITICAL: never merge into a quantized base — destroys quality.
  2. Save the AutoProcessor (preprocessor_config + tokenizer) alongside the
     merged checkpoint — required by ct2-transformers-converter.
  3. Strip residual PEFT metadata (adapter_config.json) from the temp dir to
     stop the converter detecting it as still-PEFT.
  4. Invoke ct2-transformers-converter with --quantization (default
     int8_float16, the Ampere/Ada sweet spot).
  5. Sanity-check: load the resulting CT2 model with faster_whisper, transcribe
     N canary clips, abort if WER drift exceeds the configured threshold.
  6. Emit export_manifest.json with provenance (source adapter SHA, base ID,
     quantization, sanity WER, total disk size).

Usage:
    source ~/stt_train_env/bin/activate
    python training/export_ct2.py \\
        --adapter whisper_ablation/W16_mixed_w7 \\
        --base openai/whisper-large-v3-turbo \\
        --output whisper_ct2/W16_mixed_w7

    # Switch quantization (e.g. int8 for VRAM-constrained deploys)
    python training/export_ct2.py ... --quantization int8

    # Skip the sanity test (faster, no WER gate)
    python training/export_ct2.py ... --no-sanity

    # Keep the merged HF intermediate around for debugging
    python training/export_ct2.py ... --keep-intermediate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("export_ct2")

# Five short canary clips with reference text — used by --sanity (default on).
# Picked from the Stark Road eval set, durations 2-5 s, all contain at least one
# Tier 1 theological term so any quantization regression on theological vocab
# surfaces immediately.
DEFAULT_CANARY_CLIPS: list[tuple[str, str]] = [
    (
        "stark_data/whisper_dataset_deepgram/eval/Gospel_Message_(3_15_26)_F1-ZLv-pMMg_00001.wav",
        "the perfect righteousness of god is witnessed in the savior's",
    ),
    (
        "stark_data/whisper_dataset_deepgram/eval/Gospel_Message_(3_15_26)_F1-ZLv-pMMg_00002.wav",
        "it is in the cross of christ we trace his righteousness his wondrous",
    ),
    (
        "stark_data/whisper_dataset_deepgram/eval/Gospel_Message_(3_15_26)_F1-ZLv-pMMg_00003.wav",
        "god could not pass the sinner by his sins demand that he must",
    ),
    (
        "stark_data/whisper_dataset_deepgram/eval/Gospel_Message_(3_15_26)_F1-ZLv-pMMg_00004.wav",
        "but in the cross of christ we see how god can save yet righteous",
    ),
    (
        "stark_data/whisper_dataset_deepgram/eval/Gospel_Message_(3_15_26)_F1-ZLv-pMMg_00000.wav",
        "let's start with hymn numbers",
    ),
]

# Files ct2-transformers-converter needs alongside the merged HF checkpoint.
# `tokenizer.json` is BPE; `vocab.json` + `merges.txt` are the older format —
# Whisper ships tokenizer.json. `preprocessor_config.json` carries log-Mel
# extractor params. `generation_config.json` carries forced_decoder_ids defaults.
WHISPER_COPY_FILES = [
    "tokenizer.json",
    "preprocessor_config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "normalizer.json",
]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def merge_adapter_to_bf16(base_model_id: str, adapter_dir: Path, output_dir: Path) -> None:
    """Load base Whisper in bf16, attach adapter, merge_and_unload, save."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    log.info("loading base %s in bf16", base_model_id)
    t0 = time.perf_counter()
    base = AutoModelForSpeechSeq2Seq.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    log.info("base loaded in %.1fs", time.perf_counter() - t0)

    log.info("attaching adapter from %s", adapter_dir)
    model = PeftModel.from_pretrained(base, str(adapter_dir))

    log.info("merge_and_unload (this can take a few minutes)")
    t0 = time.perf_counter()
    merged = model.merge_and_unload()
    log.info("merged in %.1fs", time.perf_counter() - t0)

    log.info("saving merged bf16 model to %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir), safe_serialization=True)

    # AutoProcessor.save_pretrained() omits preprocessor_config.json — it only
    # writes processor_config.json + tokenizer files. faster-whisper requires
    # preprocessor_config.json (otherwise WhisperFeatureExtractor falls back to
    # the v1/v2 default of 80 mel bins, which mismatches large-v3-turbo's 128).
    # Save the FeatureExtractor explicitly so the file lands in the merged dir.
    log.info("saving processor (tokenizer + feature_extractor)")
    from transformers import WhisperFeatureExtractor

    AutoProcessor.from_pretrained(base_model_id).save_pretrained(str(output_dir))
    WhisperFeatureExtractor.from_pretrained(base_model_id).save_pretrained(str(output_dir))

    # Strip any residual PEFT metadata from the temp dir so the converter
    # doesn't try to re-attach the adapter.
    for stray in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
        p = output_dir / stray
        if p.exists():
            log.info("removing residual peft file: %s", stray)
            p.unlink()

    log.info("merged bf16 model saved")


def run_ct2_converter(
    merged_dir: Path,
    output_dir: Path,
    quantization: str,
    copy_files: list[str],
    force: bool = True,
) -> None:
    binary = shutil.which("ct2-transformers-converter")
    if not binary:
        raise SystemExit("ct2-transformers-converter not on PATH. Install ctranslate2: pip install '.[cuda]'")
    # Only forward copy_files that actually exist in the merged dir — the
    # converter is strict and aborts on missing ones.
    present = [f for f in copy_files if (merged_dir / f).exists()]
    missing = [f for f in copy_files if not (merged_dir / f).exists()]
    if missing:
        log.info("copy_files skipped (not in merged dir): %s", missing)

    cmd = [
        binary,
        "--model",
        str(merged_dir),
        "--output_dir",
        str(output_dir),
        "--quantization",
        quantization,
    ]
    if present:
        cmd.extend(["--copy_files", *present])
    if force:
        cmd.append("--force")

    log.info("running: %s", " ".join(cmd))
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    log.info("ct2-transformers-converter done in %.1fs", time.perf_counter() - t0)


def directory_size_bytes(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def sanity_test_ct2(
    ct2_dir: Path,
    canary_clips: list[tuple[str, str]],
    project_root: Path,
    quantization: str,
    wer_drift_max: float,
) -> dict:
    """Load the CT2 model with faster_whisper, transcribe canaries, compute WER."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit("faster-whisper is not installed; cannot sanity-test the CT2 export") from exc
    try:
        import jiwer
    except ImportError as exc:
        raise SystemExit("jiwer is not installed; required for sanity WER computation") from exc

    log.info("loading CT2 model for sanity test (compute_type=%s)", quantization)
    model = WhisperModel(str(ct2_dir), device="cuda", compute_type=quantization)

    refs: list[str] = []
    preds: list[str] = []
    per_clip: list[dict] = []
    for rel, ref in canary_clips:
        audio = project_root / rel
        if not audio.exists():
            log.warning("sanity canary missing: %s", audio)
            continue
        t0 = time.perf_counter()
        segments, _info = model.transcribe(str(audio), language="en")
        text = " ".join(s.text.strip() for s in segments).strip().lower()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        refs.append(ref)
        preds.append(text)
        per_clip.append({"clip": Path(rel).name, "ref": ref, "pred": text, "latency_ms": round(latency_ms, 1)})

    if not refs:
        raise SystemExit("no sanity canaries available — refusing to certify the export")

    wer = float(jiwer.wer(refs, preds))
    log.info("sanity WER: %.4f over %d clips", wer, len(refs))
    for rec in per_clip:
        log.info("  %s [%5.1f ms]  pred: %s", rec["clip"], rec["latency_ms"], rec["pred"][:80])

    if wer > wer_drift_max:
        raise SystemExit(
            f"FAIL: sanity WER {wer:.3f} exceeds threshold {wer_drift_max:.3f}. "
            "Quantization may have damaged the merged model. Try --quantization int8_bfloat16 "
            "or --quantization float16 to isolate."
        )
    return {"wer": round(wer, 4), "n_clips": len(refs), "per_clip": per_clip}


def write_manifest(
    output_dir: Path,
    base_model_id: str,
    adapter_dir: Path,
    quantization: str,
    sanity: dict | None,
) -> Path:
    adapter_st = adapter_dir / "adapter_model.safetensors"
    manifest = {
        "version": "1.0",
        "exported_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_model_id": base_model_id,
        "adapter_dir_relative": str(adapter_dir),
        "adapter_sha256": sha256_file(adapter_st) if adapter_st.exists() else None,
        "ct2_quantization": quantization,
        "ct2_dir_relative": str(output_dir),
        "ct2_total_bytes": directory_size_bytes(output_dir),
        "ct2_files": sorted(p.name for p in output_dir.iterdir() if p.is_file()),
        "sanity": sanity,
    }
    manifest_path = output_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    log.info("wrote %s", manifest_path)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", type=Path, required=True, help="LoRA adapter dir (e.g. whisper_ablation/W16_mixed_w7)")
    p.add_argument("--base", type=str, default="openai/whisper-large-v3-turbo", help="Base Whisper model id")
    p.add_argument("--output", type=Path, required=True, help="Output CT2 dir (e.g. whisper_ct2/W16_mixed_w7)")
    p.add_argument(
        "--quantization",
        type=str,
        default="int8_float16",
        choices=["int8", "int8_float32", "int8_float16", "int8_bfloat16", "int16", "float16", "bfloat16", "float32"],
        help="CTranslate2 compute type. int8_float16 is the Ampere/Ada sweet spot (default).",
    )
    p.add_argument("--no-sanity", action="store_true", help="Skip the post-conversion sanity test")
    p.add_argument(
        "--sanity-wer-max",
        type=float,
        default=0.30,
        help="Abort if sanity WER exceeds this threshold. Permissive default since the canary set is small.",
    )
    p.add_argument("--keep-intermediate", action="store_true", help="Keep the merged HF intermediate dir")
    p.add_argument(
        "--intermediate",
        type=Path,
        default=None,
        help="Where to put the merged HF intermediate (defaults to a temp dir)",
    )
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if not args.adapter.exists():
        raise SystemExit(f"adapter dir not found: {args.adapter}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[1]

    intermediate_owned = args.intermediate is None
    if intermediate_owned:
        intermediate_root = Path(tempfile.mkdtemp(prefix="export_ct2_merged_"))
    else:
        intermediate_root = args.intermediate
        intermediate_root.mkdir(parents=True, exist_ok=True)

    try:
        log.info("=== STEP 1/3: merge LoRA into bf16 base ===")
        merge_adapter_to_bf16(args.base, args.adapter, intermediate_root)

        log.info("=== STEP 2/3: convert merged HF -> CTranslate2 (%s) ===", args.quantization)
        run_ct2_converter(intermediate_root, args.output, args.quantization, WHISPER_COPY_FILES)

        sanity_result: dict | None = None
        if not args.no_sanity:
            log.info("=== STEP 3/3: sanity test (faster_whisper + jiwer) ===")
            sanity_result = sanity_test_ct2(
                args.output,
                DEFAULT_CANARY_CLIPS,
                project_root,
                args.quantization,
                args.sanity_wer_max,
            )
        else:
            log.warning("STEP 3/3 skipped (--no-sanity). Quality not verified.")

        write_manifest(args.output, args.base, args.adapter, args.quantization, sanity_result)
    finally:
        if intermediate_owned and not args.keep_intermediate:
            log.info("removing intermediate dir %s", intermediate_root)
            shutil.rmtree(intermediate_root, ignore_errors=True)
        elif args.keep_intermediate:
            log.info("kept intermediate dir at %s", intermediate_root)

    log.info("=== DONE ===")
    log.info("CT2 model ready at %s (size: %.1f MB)", args.output, directory_size_bytes(args.output) / 1024 / 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
