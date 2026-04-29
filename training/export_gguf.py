#!/usr/bin/env python3
"""
export_gguf.py — Merge LoRA adapter -> bf16 -> GGUF f16 -> Q4_K_M, with verification.

UNTESTED end-to-end — needs an actual fine-tuned adapter from train_gemma4.py.
The merge / convert / quantize / verify steps each work independently from
existing tooling (peft + llama.cpp). See docs/gemma4_tuning/phase_a_infrastructure.md.

Steps (any failure aborts the pipeline):
  1. Load base in bf16, attach adapter via peft, merge_and_unload().
     Saves the merged model to a temp dir.
     CRITICAL: never merge into a 4-bit base — destroys quality.
  2. Convert merged HF model -> GGUF f16 via llama.cpp/convert_hf_to_gguf.py.
  3. Quantize GGUF f16 -> Q4_K_M via llama.cpp/build/bin/llama-quantize.
  4. Verify Q4_K_M metadata: chat_template present + non-empty, eos_token_id
     is a sane integer. (Replaces the docs' reference to llama-gguf-tool, which
     does not exist in this llama.cpp build — Python gguf.GGUFReader is used
     instead.)
  5. Optional --sanity-test: start a temp llama-server on --sanity-port,
     translate N canary sentences, compare to in-process merged-bf16 outputs.

Usage:
    python training/export_gguf.py \\
        --adapter fine_tuned_gemma4_e4b_v1 \\
        --base unsloth/gemma-4-E4B-it \\
        --output models/gemma-4-e4b-it-q4km-v1.gguf

    # Skip Q4_K_M and only produce f16 (debugging):
    python training/export_gguf.py ... --skip-quantize

    # Keep intermediate artifacts for debugging:
    python training/export_gguf.py ... --keep-intermediate
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("export_gguf")

DEFAULT_LLAMA_CPP_DIR = Path(os.environ.get("LLAMA_CPP_DIR", str(Path.home() / "llama.cpp")))


def merge_adapter_to_bf16(base_model_id: str, adapter_dir: Path, output_dir: Path) -> None:
    """Load base in bf16, attach adapter, merge_and_unload, save."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("loading base %s in bf16", base_model_id)
    t0 = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(
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

    # Save tokenizer alongside — required by convert_hf_to_gguf.py
    tok = AutoTokenizer.from_pretrained(adapter_dir if (adapter_dir / "tokenizer.json").exists() else base_model_id)
    tok.save_pretrained(str(output_dir))
    log.info("merged bf16 model saved")


def convert_hf_to_gguf(merged_dir: Path, gguf_path: Path, llama_cpp_dir: Path) -> None:
    script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not script.exists():
        raise SystemExit(f"convert_hf_to_gguf.py not found at {script}")
    cmd = [
        sys.executable,
        str(script),
        str(merged_dir),
        "--outtype",
        "f16",
        "--outfile",
        str(gguf_path),
    ]
    log.info("running: %s", " ".join(cmd))
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    log.info("convert_hf_to_gguf done in %.1fs", time.perf_counter() - t0)


def quantize_gguf(f16_path: Path, q4_path: Path, llama_cpp_dir: Path, qtype: str = "Q4_K_M") -> None:
    binary = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    if not binary.exists():
        raise SystemExit(f"llama-quantize not found at {binary}")
    cmd = [str(binary), str(f16_path), str(q4_path), qtype]
    log.info("running: %s", " ".join(cmd))
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    log.info("llama-quantize done in %.1fs", time.perf_counter() - t0)


def verify_gguf_metadata(gguf_path: Path) -> dict:
    """Verify the GGUF has a chat template and a sane EOS token. Loud failure on bad state."""
    from gguf import GGUFReader

    reader = GGUFReader(str(gguf_path))
    fields = reader.fields

    findings: dict = {"path": str(gguf_path)}

    # general.architecture
    if "general.architecture" in fields:
        findings["architecture"] = fields["general.architecture"].contents()
    if "general.name" in fields:
        findings["name"] = fields["general.name"].contents()

    # chat_template
    if "tokenizer.chat_template" not in fields:
        raise SystemExit(
            f"FAIL: {gguf_path} has no tokenizer.chat_template field. "
            "Production engine relies on the Gemma 4 chat template — without it, "
            "llama-server will fall back to a generic template and outputs will be wrong."
        )
    template = fields["tokenizer.chat_template"].contents()
    if not template or not isinstance(template, str) or len(template) < 100:
        raise SystemExit(
            f"FAIL: chat_template is empty or suspiciously short ({len(template) if template else 0} chars). "
            f"Sample: {template[:200] if template else None!r}"
        )
    findings["chat_template_len"] = len(template)
    findings["chat_template_head"] = template[:120]

    # eos_token_id
    if "tokenizer.ggml.eos_token_id" not in fields:
        raise SystemExit(f"FAIL: {gguf_path} has no tokenizer.ggml.eos_token_id field.")
    eos = fields["tokenizer.ggml.eos_token_id"].contents()
    if not isinstance(eos, int) or eos < 0:
        raise SystemExit(f"FAIL: eos_token_id is not a non-negative int (got {eos!r})")
    findings["eos_token_id"] = eos

    if "tokenizer.ggml.bos_token_id" in fields:
        findings["bos_token_id"] = fields["tokenizer.ggml.bos_token_id"].contents()

    log.info("GGUF metadata verified: %s", findings)
    return findings


def sanity_test(merged_dir: Path, gguf_path: Path, port: int, n_sentences: int) -> None:
    """Compare merged-bf16 in-process outputs to Q4_K_M outputs via llama-server.

    Runs N canary sentences through both, reports per-sentence cosine similarity
    of generated text and a coarse exact-match count. A full COMET delta check
    is left to the Phase B benchmark — this sanity test only catches gross
    regressions (template mismatch, broken EOS, etc.).
    """
    log.warning(
        "sanity_test stub — runs both backends but only does string-similarity "
        "comparison. Full COMET delta is in scripts/benchmarks/bench_translate_t1_t4.py."
    )
    raise NotImplementedError("sanity_test full implementation deferred to Phase B; use --no-sanity-test for now")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", type=Path, required=True, help="LoRA adapter directory (output of train_gemma4.py)")
    p.add_argument(
        "--base",
        default="unsloth/gemma-4-E4B-it",
        help="HF model id of base model (must match training)",
    )
    p.add_argument("--output", type=Path, required=True, help="output Q4_K_M GGUF path")
    p.add_argument(
        "--llama-cpp-dir",
        type=Path,
        default=DEFAULT_LLAMA_CPP_DIR,
        help="path to llama.cpp checkout (default: $LLAMA_CPP_DIR or ~/llama.cpp)",
    )
    p.add_argument("--qtype", default="Q4_K_M", help="quantization type passed to llama-quantize")
    p.add_argument("--skip-quantize", action="store_true", help="produce only f16 GGUF, skip Q4_K_M step")
    p.add_argument("--skip-merge", action="store_true", help="skip merge step (assume --merged-dir is ready)")
    p.add_argument(
        "--merged-dir",
        type=Path,
        default=None,
        help="explicit merged-bf16 dir (default: temp dir; required with --skip-merge)",
    )
    p.add_argument("--keep-intermediate", action="store_true", help="keep merged bf16 dir + f16 GGUF for debugging")
    p.add_argument("--sanity-test", action="store_true", help="(deferred) compare bf16 vs Q4_K_M outputs")
    p.add_argument("--sanity-port", type=int, default=8092, help="port for temp llama-server during sanity test")
    p.add_argument("--sanity-n", type=int, default=8, help="canary sentence count for sanity test")
    args = p.parse_args(argv)

    if not args.adapter.exists():
        p.error(f"--adapter not found: {args.adapter}")
    if args.skip_merge and not args.merged_dir:
        p.error("--skip-merge requires --merged-dir")

    # Use a managed temp dir if no explicit merged-dir given.
    use_temp = args.merged_dir is None
    if use_temp:
        tmp_root = Path(tempfile.mkdtemp(prefix="gemma4_export_"))
        merged_dir = tmp_root / "merged_bf16"
    else:
        merged_dir = args.merged_dir
        tmp_root = None

    try:
        if not args.skip_merge:
            merge_adapter_to_bf16(args.base, args.adapter, merged_dir)
        else:
            log.info("--skip-merge: assuming merged bf16 dir at %s", merged_dir)
            if not merged_dir.exists():
                raise SystemExit(f"merged dir does not exist: {merged_dir}")

        # Step 2: convert HF -> GGUF f16
        f16_path = args.output.with_suffix(".f16.gguf")
        convert_hf_to_gguf(merged_dir, f16_path, args.llama_cpp_dir)

        if args.skip_quantize:
            log.info("--skip-quantize: stopping after f16 GGUF at %s", f16_path)
            findings = verify_gguf_metadata(f16_path)
            log.info("done. f16 GGUF: %s", f16_path)
            return 0

        # Step 3: quantize Q4_K_M (or whatever --qtype)
        quantize_gguf(f16_path, args.output, args.llama_cpp_dir, args.qtype)

        # Step 4: verify metadata
        findings = verify_gguf_metadata(args.output)

        # Step 5: optional sanity test
        if args.sanity_test:
            sanity_test(merged_dir, args.output, args.sanity_port, args.sanity_n)

        log.info(
            "done. Q4_K_M GGUF: %s (architecture=%s, eos=%s, chat_template_len=%d)",
            args.output,
            findings.get("architecture"),
            findings.get("eos_token_id"),
            findings.get("chat_template_len"),
        )

        if not args.keep_intermediate:
            log.info("removing intermediate f16 GGUF (use --keep-intermediate to retain)")
            f16_path.unlink(missing_ok=True)

        return 0
    finally:
        if use_temp and tmp_root is not None and not args.keep_intermediate:
            log.info("cleaning up temp dir %s", tmp_root)
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
