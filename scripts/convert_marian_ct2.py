#!/usr/bin/env python3
"""convert_marian_ct2.py — Convert HF MarianMT to CTranslate2.

Vendor-model converter that mirrors training/export_ct2.py minus the LoRA merge.
MarianMT (Helsinki-NLP/opus-mt-*) is an off-the-shelf model; we don't fine-tune
it, so this is a one-shot HF → CT2 conversion plus sanity gate.

Output loads via ``ctranslate2.Translator(output_dir, ...)`` and slots into
``engines.cuda_engine.MarianCT2Engine`` (the engine accepts a local CT2
directory plus a HF tokenizer dir for the SentencePiece pieces).

Steps (any failure aborts):
  1. Snapshot the HF model + tokenizer files into a temp dir (download_only).
  2. Invoke ct2-transformers-converter with --quantization (default
     int8_float16 — Ampere/Ada sweet spot, mirrors v2026.7's Whisper default).
  3. Sanity-check: load the converted model with ctranslate2.Translator,
     translate the 8 theological canary sentences, abort if fewer than
     ``--sanity-min-canary`` outputs contain the expected target-language term.
  4. Emit export_manifest.json with provenance (model_id, direction,
     quantization, sanity score, ct2 + tokenizer file list, total disk size).

Usage (with the WSL training venv active):

    python scripts/convert_marian_ct2.py \\
        --model-id Helsinki-NLP/opus-mt-en-es \\
        --output adapters/marian_ct2/en-es/active

    # Other direction
    python scripts/convert_marian_ct2.py \\
        --model-id Helsinki-NLP/opus-mt-es-en \\
        --output adapters/marian_ct2/es-en/active

    # Try a different quantization
    python scripts/convert_marian_ct2.py ... --quantization int8

    # Re-run, overwriting an existing model.bin
    python scripts/convert_marian_ct2.py ... --force

    # Skip the sanity gate (faster, no canary verification)
    python scripts/convert_marian_ct2.py ... --no-sanity
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
log = logging.getLogger("convert_marian_ct2")

# Files MarianTokenizer expects alongside model.bin. The CT2 converter writes
# its own config.json (the model spec) into the output dir, so we must NOT
# pass HF's config.json via --copy_files (collides). source.spm + target.spm
# are the SentencePiece pieces; vocab.json is the shared vocab; the rest are
# tokenizer metadata that MarianTokenizer.from_pretrained() reads.
MARIAN_COPY_FILES = [
    "source.spm",
    "target.spm",
    "vocab.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

# Theological-term presence canaries, paired per direction. Mirrors
# training.benchmark_gemma4.CANARY_SENTENCES — the source EN sentences are
# the same set; we add a parallel ES→EN set for the reverse direction.
CANARY_EN_TO_ES: list[tuple[str, str]] = [
    ("And you know, when we think about the atonement — what Christ did for us on that cross...", "expiación"),
    ("God made a covenant with Abraham, and brothers, He keeps His promises.", "pacto"),
    ("It's only by grace, friends. We can't earn it — it's grace through faith.", "gracia"),
    ("And Paul writes about the righteousness of God here in Romans, and what does he mean by that?", "justicia"),
    ("Now, if you turn to the book of James — James has a lot to say about faith and works.", "Santiago"),
    ("You think about James and John, just regular fishermen on the Sea of Galilee.", "Jacobo"),
    ("The speaker tonight was talking about sanctification — what it means to be set apart.", "santificación"),
    ("Let us remember the Lord at the breaking of bread this morning.", "partimiento"),
]

# ES → EN canaries. Sources are simple ES sentences containing the same
# theological concepts; expected EN substrings are common renderings.
CANARY_ES_TO_EN: list[tuple[str, str]] = [
    ("La expiación de Cristo en la cruz nos reconcilia con Dios.", "atonement"),
    ("Dios hizo un pacto con Abraham y guarda sus promesas.", "covenant"),
    ("Es solo por gracia, hermanos. No la podemos ganar.", "grace"),
    ("Pablo escribe acerca de la justicia de Dios en Romanos.", "righteousness"),
    ("Si abrimos el libro de Santiago, vemos mucho sobre la fe y las obras.", "James"),
    ("Pensemos en Jacobo y Juan, pescadores del mar de Galilea.", "James"),
    ("El predicador habló sobre la santificación, ser apartados para Dios.", "sanctification"),
    ("Recordemos al Señor en el partimiento del pan esta mañana.", "breaking of bread"),
]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def directory_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def detect_direction(model_id: str) -> str:
    """Infer 'en-es' or 'es-en' from the HF model id. Raises on ambiguity."""
    lower = model_id.lower()
    if "opus-mt-en-es" in lower:
        return "en-es"
    if "opus-mt-es-en" in lower:
        return "es-en"
    raise SystemExit(
        f"Could not infer translation direction from model_id {model_id!r}. "
        "Pass --direction {en-es,es-en} explicitly."
    )


def snapshot_hf_model(model_id: str, output_dir: Path) -> None:
    """Download the HF Marian model + tokenizer files into output_dir.

    Uses transformers' ``snapshot_download`` semantics via from_pretrained +
    save_pretrained, so any HF cache hits are reused automatically.
    """
    from transformers import MarianMTModel, MarianTokenizer

    log.info("snapshotting %s into %s", model_id, output_dir)
    t0 = time.perf_counter()
    tokenizer = MarianTokenizer.from_pretrained(model_id)
    model = MarianMTModel.from_pretrained(model_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    log.info("snapshot done in %.1fs", time.perf_counter() - t0)


def run_ct2_converter(
    src_dir: Path,
    output_dir: Path,
    quantization: str,
    copy_files: list[str],
    force: bool = True,
) -> None:
    binary = shutil.which("ct2-transformers-converter")
    if not binary:
        raise SystemExit("ct2-transformers-converter not on PATH. Install ctranslate2: pip install ctranslate2>=4.5")
    present = [f for f in copy_files if (src_dir / f).exists()]
    missing = [f for f in copy_files if not (src_dir / f).exists()]
    if missing:
        log.info("copy_files skipped (not in snapshot dir): %s", missing)

    cmd = [
        binary,
        "--model",
        str(src_dir),
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


def sanity_test_ct2(
    ct2_dir: Path,
    direction: str,
    quantization: str,
    min_canary: int,
) -> dict:
    """Translate the canary set with the converted model; require ``min_canary`` term hits."""
    try:
        import ctranslate2
    except ImportError as exc:
        raise SystemExit("ctranslate2 not installed; cannot sanity-test the export") from exc

    from transformers import MarianTokenizer

    canaries = CANARY_EN_TO_ES if direction == "en-es" else CANARY_ES_TO_EN

    log.info("loading CT2 model for sanity test (compute_type=%s)", quantization)
    # Try cuda first so the gate matches the production deployment surface;
    # fall back to cpu if CUDA isn't available (e.g. CI runner).
    try:
        translator = ctranslate2.Translator(str(ct2_dir), device="cuda", compute_type=quantization)
        device = "cuda"
    except Exception as exc:
        log.info("CUDA unavailable (%s); falling back to CPU for sanity test", exc)
        translator = ctranslate2.Translator(str(ct2_dir), device="cpu", compute_type=quantization)
        device = "cpu"

    tokenizer = MarianTokenizer.from_pretrained(str(ct2_dir))

    per_clip: list[dict] = []
    hits = 0
    for src_text, expected_term in canaries:
        t0 = time.perf_counter()
        ids = tokenizer(src_text, return_tensors=None)["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(ids)
        result = translator.translate_batch([tokens], max_decoding_length=128)
        out_tokens = result[0].hypotheses[0]
        out_ids = tokenizer.convert_tokens_to_ids(out_tokens)
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        hit = expected_term.lower() in out_text.lower()
        if hit:
            hits += 1
        per_clip.append(
            {
                "src": src_text,
                "expected_term": expected_term,
                "translation": out_text,
                "term_present": hit,
                "latency_ms": round(latency_ms, 1),
            }
        )
        log.info(
            "  [%s] %5.1fms  expected=%s  hit=%s  out=%s",
            "PASS" if hit else "MISS",
            latency_ms,
            expected_term,
            hit,
            out_text[:80],
        )

    # Release before logging — large CT2 models can hold non-trivial CUDA memory
    del translator

    score = f"{hits}/{len(canaries)}"
    log.info("sanity score: %s (threshold: %d)", score, min_canary)
    if hits < min_canary:
        raise SystemExit(
            f"FAIL: sanity score {score} below threshold ({min_canary}). "
            "Quantization may have damaged the model. Try --quantization int8_float16 "
            "or --quantization float16 to isolate."
        )
    return {
        "score": score,
        "hits": hits,
        "total": len(canaries),
        "device_tested_on": device,
        "per_clip": per_clip,
    }


def write_manifest(
    output_dir: Path,
    model_id: str,
    direction: str,
    quantization: str,
    sanity: dict | None,
) -> Path:
    model_bin = output_dir / "model.bin"
    manifest = {
        "version": "1.0",
        "exported_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_id": model_id,
        "direction": direction,
        "ct2_quantization": quantization,
        "ct2_dir_relative": str(output_dir),
        "ct2_total_bytes": directory_size_bytes(output_dir),
        "model_bin_sha256": sha256_file(model_bin) if model_bin.exists() else None,
        "files": sorted(p.name for p in output_dir.iterdir() if p.is_file()),
        "sanity": sanity,
    }
    manifest_path = output_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    log.info("wrote %s", manifest_path)
    return manifest_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="HF Marian model id (e.g. Helsinki-NLP/opus-mt-en-es).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CT2 directory (e.g. adapters/marian_ct2/en-es/active).",
    )
    p.add_argument(
        "--direction",
        type=str,
        default=None,
        choices=["en-es", "es-en"],
        help="Translation direction. Inferred from --model-id if omitted.",
    )
    p.add_argument(
        "--quantization",
        type=str,
        default="int8_float16",
        choices=["int8", "int8_float32", "int8_float16", "int8_bfloat16", "int16", "float16", "bfloat16", "float32"],
        help="CTranslate2 compute type. int8_float16 is the Ampere/Ada default (mirrors v2026.7 Whisper).",
    )
    p.add_argument(
        "--no-sanity",
        action="store_true",
        help="Skip the post-conversion canary check.",
    )
    p.add_argument(
        "--sanity-min-canary",
        type=int,
        default=6,
        help="Minimum canary term hits required to pass (default: 6 of 8). Mirrors the Gemma 4 canary gate.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing model.bin in --output.",
    )
    p.add_argument(
        "--keep-snapshot",
        action="store_true",
        help="Keep the HF snapshot intermediate dir after conversion.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    direction = args.direction or detect_direction(args.model_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if (args.output / "model.bin").exists() and not args.force:
        log.info(
            "model.bin already at %s — skipping conversion (use --force to overwrite). "
            "Re-running sanity gate against existing model.",
            args.output,
        )
        sanity_result: dict | None = None
        if not args.no_sanity:
            sanity_result = sanity_test_ct2(
                args.output,
                direction,
                args.quantization,
                args.sanity_min_canary,
            )
        write_manifest(args.output, args.model_id, direction, args.quantization, sanity_result)
        return 0

    snapshot_root = Path(tempfile.mkdtemp(prefix="convert_marian_ct2_snapshot_"))

    try:
        log.info("=== STEP 1/3: snapshot HF model ===")
        snapshot_hf_model(args.model_id, snapshot_root)

        log.info("=== STEP 2/3: convert HF -> CTranslate2 (%s) ===", args.quantization)
        run_ct2_converter(snapshot_root, args.output, args.quantization, MARIAN_COPY_FILES)

        sanity_result = None
        if not args.no_sanity:
            log.info("=== STEP 3/3: sanity test (canary terms) ===")
            sanity_result = sanity_test_ct2(
                args.output,
                direction,
                args.quantization,
                args.sanity_min_canary,
            )
        else:
            log.warning("STEP 3/3 skipped (--no-sanity). Quality not verified.")

        write_manifest(args.output, args.model_id, direction, args.quantization, sanity_result)
    finally:
        if not args.keep_snapshot:
            log.info("removing snapshot dir %s", snapshot_root)
            shutil.rmtree(snapshot_root, ignore_errors=True)
        else:
            log.info("kept snapshot dir at %s", snapshot_root)

    log.info("=== DONE ===")
    log.info(
        "CT2 model ready at %s (%.1f MB)",
        args.output,
        directory_size_bytes(args.output) / 1024 / 1024,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
