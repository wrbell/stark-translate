#!/usr/bin/env python3
"""Offline, standard-library-only checks for the W17 and Gemma training recipes.

Reads the trainers' actual argparse declarations without importing the trainers.
No tensors, model downloads, GPU packages, subprocesses, or training are loaded.
Passing this check does not certify CUDA execution, quality, or human approval.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import struct
import sys
import unicodedata
import wave
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = date(2026, 3, 14)
FRESH_EVAL_IDS = ("4Es8SrciqV0", "vRT5RswIHu8", "FOVTvZednUQ", "yOzWGOTvTaA")
WHISPER_TARGETS = {"q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"}
DEFAULT_HOLDOUT = ROOT / "bible_data/aligned/verse_pairs_test_v2.jsonl"
OTHER_HOLDOUTS = (
    "bible_data/aligned/verse_pairs_test.jsonl",
    "bible_data/holdout/verse_pairs_test.jsonl",
    "bible_data/hymns/hymn_pairs_pd_holdout.jsonl",
    "bible_data/synthetic/deepl_sermon_benchmark_500.jsonl",
    "tools/stt_bench_manifest.json",
    "docs/evaluation/mac_v2026_14_manifest_v2.json",
    "training/theological_canaries.py",
)
TEXT_FIELDS = {
    "en",
    "es",
    "source_text",
    "target_text",
    "text",
    "transcription",
    "ground_truth",
    "reference_text",
    "reference_translation",
    "draft_transcript",
}
ID_FIELDS = {"verse_id", "video_id", "chunk_source", "source_id"}


class PreflightError(ValueError):
    """An input is missing or incompatible; do not start the GPU recipe."""


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_rows(path: Path) -> list[dict]:
    """Fail closed on malformed/empty files; do not silently discard training data."""
    if not path.is_file():
        raise PreflightError(f"Missing input file: {path}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PreflightError(f"Malformed JSON: {path}:{line_no}") from exc
                if not isinstance(row, dict):
                    raise PreflightError(f"Expected object: {path}:{line_no}")
                rows.append(row)
    if not rows:
        raise PreflightError(f"Empty corpus: {path}")
    return rows


def parse_trainer_args(script: str, argv: list[str]) -> argparse.Namespace:
    """Evaluate only argparse setup statements from main, never module imports/body.

    AST extraction deliberately fails if a future parser needs other setup: a parser
    refactor requires reviewing this reader, rather than importing a CUDA trainer.
    """
    path = ROOT / "training" / script
    tree = ast.parse(path.read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    setup = []
    parser_name = None
    for node in main.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and func.attr == "ArgumentParser":
                parser_name = node.targets[0].id
                setup.append(node)
                continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == parser_name
                and func.attr == "add_argument"
            ):
                setup.append(node)
                continue
        if parser_name:
            break
    if not parser_name or len(setup) < 2:
        raise PreflightError(f"Cannot read argparse setup safely: {path}")
    scope = {"argparse": argparse, "Path": Path, "__doc__": ast.get_docstring(tree)}
    exec(compile(ast.Module(body=setup, type_ignores=[]), str(path), "exec"), scope)  # nosec B102 - local argparse only
    return scope[parser_name].parse_args(argv)


def normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def walk_rows(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_rows(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_rows(child)


def holdout_index(paths: list[Path] | None = None, *, allow_missing=False) -> dict:
    required = paths if paths is not None else [DEFAULT_HOLDOUT]
    selected = list(dict.fromkeys([*required, *(ROOT / p for p in OTHER_HOLDOUTS if (ROOT / p).is_file())]))
    result = {"texts": set(), "ids": set(), "files": [], "missing": []}
    for path in selected:
        if not path.is_file():
            if not allow_missing:
                raise PreflightError(f"Missing required holdout: {path}")
            result["missing"].append(str(path))
            continue
        if path.suffix == ".py":
            tree = ast.parse(path.read_text())
            value = next(
                ast.literal_eval(n.value)
                for n in tree.body
                if isinstance(n, ast.AnnAssign)
                and isinstance(n.target, ast.Name)
                and n.target.id == "THEOLOGICAL_CANARIES"
            )
        elif path.suffix == ".jsonl":
            value = json_rows(path)
        else:
            value = json.loads(path.read_text())
        for row in walk_rows(value):
            for key, text in row.items():
                if key in TEXT_FIELDS and isinstance(text, str) and text.strip():
                    result["texts"].add(normalized(text))
                if key in ID_FIELDS and text is not None:
                    result["ids"].add((key, str(text)))
                if key == "audio_path_relative" and isinstance(text, str):
                    result["ids"].add(("chunk_source", re.sub(r"_\d+$", "", Path(text).stem)))
        result["files"].append({"path": str(path), "sha256": sha256(path)})
    return result


def check_training_row(row: dict, index: dict, label: str) -> None:
    if row.get("split", "train") != "train" or row.get("usage") == "evaluation_only":
        raise PreflightError(f"Evaluation row in training input: {label}")
    if row.get("approved_for_training") is False or row.get("review_status") == "unapproved":
        raise PreflightError(f"Unapproved candidate in training input: {label}")
    for key, value in row.items():
        if key in TEXT_FIELDS and isinstance(value, str) and normalized(value) in index["texts"]:
            raise PreflightError(f"Holdout text overlap ({key}): {label}")
        if key in ID_FIELDS and value is not None and (key, str(value)) in index["ids"]:
            raise PreflightError(f"Holdout identifier overlap ({key}): {label}")
    serialized = json.dumps(row, ensure_ascii=False)
    if row.get("source") and ("chunk_source", str(row["source"])) in index["ids"]:
        raise PreflightError(f"Holdout sermon source overlap: {label}")
    if any(video in serialized for video in FRESH_EVAL_IDS):
        raise PreflightError(f"Fresh-eval video in training input: {label}")
    for key in ("date", "source_date", "upload_date", "sermon_date"):
        if row.get(key):
            raw = str(row[key])
            try:
                parsed = date.fromisoformat(raw[:10])
            except ValueError as exc:
                raise PreflightError(f"Unparseable source date ({key}): {label}") from exc
            if parsed > CUTOFF:
                raise PreflightError(f"Post-cutoff source date: {label}")
    # Existing sermon filenames encode month_day_two-digit-year.
    for match in re.finditer(r"\((\d{1,2})_(\d{1,2})_(\d{2})\)", serialized):
        month, day, year = map(int, match.groups())
        if date(2000 + year, month, day) > CUTOFF:
            raise PreflightError(f"Post-cutoff sermon filename: {label}")


def resolve_model_config(model: str, explicit: str | Path | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise PreflightError(f"Missing model config: {path}")
        return path
    local = Path(model) / "config.json"
    if local.is_file():
        return local
    hf_home = Path(
        os.environ.get("HF_HOME", Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface")
    )
    hub = Path(os.environ.get("HF_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE", hf_home / "hub")))
    cached = hub / ("models--" + model.replace("/", "--"))
    ref = cached / "refs/main"
    if ref.is_file():
        path = cached / "snapshots" / ref.read_text().strip() / "config.json"
        if path.is_file():
            return path
    raise PreflightError("No local Whisper config.json; supply --model-config (no download performed)")


def adapter_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    """Read and verify safetensors header/length without importing tensor libraries."""
    if not path.is_file():
        raise PreflightError(f"Missing adapter weights: {path}")
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise PreflightError(f"Truncated safetensors: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_size <= min(size - 8, 100_000_000):
            raise PreflightError(f"Invalid safetensors header length: {path}")
        header = json.loads(stream.read(header_size))
    shapes, intervals = {}, []
    dtype_bytes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        shape = info["shape"]
        start, end = info["data_offsets"]
        if (
            not shape
            or any(type(n) is not int or n <= 0 for n in shape)
            or info["dtype"] not in dtype_bytes
            or end - start != math.prod(shape) * dtype_bytes[info["dtype"]]
            or not 0 <= start < end <= size - 8 - header_size
        ):
            raise PreflightError(f"Invalid safetensors tensor metadata: {name}")
        shapes[name] = tuple(shape)
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise PreflightError("Adapter tensor offsets overlap or contain gaps")
        cursor = end
    if not shapes or cursor != size - 8 - header_size:
        raise PreflightError("Empty adapter or unaccounted tensor bytes")
    return shapes


def canonical_lora_key(key: str) -> str:
    return key.replace(".default.", ".")


def expected_whisper_shapes(config: dict, targets: set[str], rank: int, dora: bool) -> dict:
    expected = {}
    d_model = config["d_model"]
    for tower in ("encoder", "decoder"):
        for layer in range(config[f"{tower}_layers"]):
            prefix = f"base_model.model.model.{tower}.layers.{layer}."
            for target in sorted(targets):
                if target in ("fc1", "fc2"):
                    stems = [prefix + target]
                    ffn = config[f"{tower}_ffn_dim"]
                    n_in, n_out = (d_model, ffn) if target == "fc1" else (ffn, d_model)
                else:
                    attentions = ("self_attn", "encoder_attn") if tower == "decoder" else ("self_attn",)
                    stems = [prefix + attention + "." + target for attention in attentions]
                    n_in = n_out = d_model
                for stem in stems:
                    expected[stem + ".lora_A.weight"] = (rank, n_in)
                    expected[stem + ".lora_B.weight"] = (n_out, rank)
                    if dora:
                        expected[stem + ".lora_magnitude_vector.weight"] = (n_out,)
    return expected


def validate_whisper_config(args: argparse.Namespace) -> dict:
    targets = set(args.target_modules)
    if not targets or not targets <= WHISPER_TARGETS or len(targets) != len(args.target_modules):
        raise PreflightError("Invalid Whisper targets; use q_proj v_proj k_proj out_proj fc1 fc2 (never o_proj)")
    if args.lora_r <= 0 or args.lora_alpha <= 0 or not 0 <= args.replay_ratio < 1:
        raise PreflightError("Invalid LoRA rank/alpha or replay ratio")
    if args.require_replay and args.replay_ratio <= 0:
        raise PreflightError("Required replay must have a positive ratio; W17 must not train hard-only")
    for key in ("batch_size", "grad_accum", "epochs", "lr"):
        value = getattr(args, key, 1)
        if value <= 0 or not math.isfinite(value):
            raise PreflightError(f"Invalid Whisper training parameter: {key}")
    if args.allow_target_expansion and not args.init_from:
        raise PreflightError("--allow-target-expansion requires --init-from")
    config_path = resolve_model_config(args.model, args.model_config)
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "whisper":
        raise PreflightError("Model config is not a Whisper architecture")
    for key in ("d_model", "encoder_layers", "decoder_layers", "encoder_ffn_dim", "decoder_ffn_dim"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise PreflightError(f"Missing/invalid Whisper geometry: {key}")
    report = {
        "model_config": str(config_path),
        "model_config_sha256": sha256(config_path),
        "targets": sorted(targets),
        "rank": args.lora_r,
        "use_dora": args.use_dora,
    }
    if not args.init_from:
        return report
    adapter = Path(args.init_from)
    config_file = adapter / "adapter_config.json"
    if not config_file.is_file():
        raise PreflightError(f"Missing adapter config: {config_file}")
    source = json.loads(config_file.read_text())
    if source.get("base_model_name_or_path") != args.model:
        raise PreflightError("Source adapter base_model_name_or_path does not match requested model")
    if source.get("peft_type") != "LORA" or source.get("r") != args.lora_r:
        raise PreflightError("Source adapter LoRA type/rank mismatch")
    if source.get("lora_alpha") != args.lora_alpha:
        raise PreflightError("Source adapter alpha mismatch would change its effective scale")
    if (
        any(
            source.get(k)
            for k in (
                "rank_pattern",
                "alpha_pattern",
                "modules_to_save",
                "layers_to_transform",
                "layers_pattern",
                "use_rslora",
                "fan_in_fan_out",
            )
        )
        or source.get("bias", "none") != "none"
    ):
        raise PreflightError("Unsupported source adapter pattern/bias; requires explicit compatibility review")
    source_targets = source.get("target_modules")
    if not isinstance(source_targets, list) or not source_targets or not set(source_targets) <= targets:
        raise PreflightError("Source adapter targets must be an explicit subset of requested Whisper targets")
    source_targets = set(source_targets)
    source_dora = bool(source.get("use_dora", False))
    if source_dora and not args.use_dora:
        raise PreflightError("Cannot discard source DoRA magnitude parameters")
    if not args.allow_target_expansion and (source_targets != targets or source_dora != args.use_dora):
        raise PreflightError("Adapter targets/DoRA differ; explicit --allow-target-expansion is required")
    weights = adapter / "adapter_model.safetensors"
    raw_shapes = adapter_shapes(weights)
    actual = {canonical_lora_key(k): v for k, v in raw_shapes.items()}
    if len(actual) != len(raw_shapes):
        raise PreflightError("Duplicate source adapter keys after PEFT name normalization")
    expected = expected_whisper_shapes(config, source_targets, args.lora_r, source_dora)
    if actual != expected:
        bad = sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))
        raise PreflightError(f"Source adapter tensor keys/shapes mismatch: {bad[:5]}")
    report.update(
        {
            "adapter_config_sha256": sha256(config_file),
            "adapter_sha256": sha256(weights),
            "source_tensor_count": len(actual),
            "fresh_targets": sorted(targets - source_targets),
            "fresh_dora_magnitudes": args.use_dora and not source_dora,
        }
    )
    return report


def check_adapter_load(
    missing: list[str], unexpected: list[str], loaded: list[str], *, fresh_targets: set[str], fresh_dora: bool
) -> None:
    """Only explicitly new modules/magnitudes may remain fresh after init-from."""
    if unexpected or not loaded:
        raise PreflightError(f"Source adapter weights were not all consumed: {unexpected[:5]}")
    bad = []
    for key in missing:
        if ".lora_" not in key:
            continue
        stem, parameter = key.split(".lora_", 1)
        target = stem.rsplit(".", 1)[-1]
        if target in fresh_targets or (fresh_dora and parameter.startswith("magnitude_vector.")):
            continue
        bad.append(key)
    if bad:
        raise PreflightError(f"Required source LoRA tensors missing after initialization: {bad[:5]}")


def initialize_new_dora_magnitudes(model, loaded_keys: list[str]) -> int:
    """Rebase new magnitudes on loaded W16 directions before optimizer creation.

    PEFT's DoraLinearLayer.update_layer computes norm(W + scaling * B @ A).
    Calling it after load_state_dict preserves the source LoRA effective weight;
    leaving the original norm(W) would silently rescale that weight. This helper
    imports nothing; it is invoked only inside an explicitly requested GPU run.
    Source DoRA adapters never take this path, preserving their learned magnitudes.
    """
    stems = {key.split(".lora_A.", 1)[0] for key in loaded_keys if ".lora_A." in key}
    modules = dict(model.named_modules())
    for stem in sorted(stems):
        module = modules.get(stem)
        magnitudes = getattr(module, "lora_magnitude_vector", {})
        if "default" not in magnitudes or not callable(getattr(magnitudes["default"], "update_layer", None)):
            raise PreflightError(f"PEFT cannot reinitialize the new DoRA magnitude safely: {stem}")
        magnitudes["default"].update_layer(
            base_layer=module.get_base_layer(),
            lora_A=module.lora_A["default"].weight,
            lora_B=module.lora_B["default"].weight,
            scaling=module.scaling["default"],
        )
    if not stems:
        raise PreflightError("No loaded source LoRA directions for DoRA reinitialization")
    return len(stems)


def validate_gemma(args: argparse.Namespace, index: dict) -> dict:
    if args.base not in {"unsloth/gemma-4-E4B-it", "unsloth/gemma-4-E2B-it"}:
        raise PreflightError("Expected a separately trained Gemma 4 E4B or E2B base")
    for field in ("lora_r", "lora_alpha", "per_device_batch_size", "grad_accum", "max_seq_length", "epochs", "lr"):
        if getattr(args, field) <= 0 or not math.isfinite(getattr(args, field)):
            raise PreflightError(f"Invalid Gemma training parameter: {field}")
    for field in ("max_pairs", "max_steps", "warmup_steps"):
        if getattr(args, field) < 0:
            raise PreflightError(f"Invalid Gemma training parameter: {field}")
    paths = [p for p in (args.train_data, args.verse_pairs, args.sermon_pairs, args.glossary_pairs) if p]
    if not paths:
        raise PreflightError("No Gemma training corpus selected")
    files, directions = [], Counter()
    for path in paths:
        if path.resolve() in {
            (ROOT / p).resolve()
            for p in ("bible_data/aligned/verse_pairs_train.jsonl", "bible_data/verse_pairs_train.jsonl")
        }:
            raise PreflightError("Misaligned v1 Bible corpus selected; rebuild/use verse_pairs_train_v2.jsonl")
        rows = json_rows(path)
        for number, row in enumerate(rows, 1):
            label = f"{path}:{number}"
            if any(not isinstance(row.get(key), str) or not row[key].strip() for key in ("en", "es")):
                raise PreflightError(f"Gemma requires nonempty en/es strings: {label}")
            direction = row.get("direction", "en2es")
            if direction not in {"en2es", "es2en"}:
                raise PreflightError(f"Invalid translation direction: {label}")
            forbidden = {"esv", "nasb", "niv", "nlt", "nvi", "lbla", "rvr1960", "dhh"}
            if any(str(row.get(key, "")).lower() in forbidden for key in ("en_source", "es_source")):
                raise PreflightError(f"Disallowed Bible edition: {label}")
            check_training_row(row, index, label)
            directions[direction] += 1
        files.append({"path": str(path), "sha256": sha256(path), "rows": len(rows)})
    return {
        "corpora": files,
        "directions_before_sampling": dict(directions),
        "rows_before_sampling": sum(directions.values()),
        "max_pairs": args.max_pairs,
        "base": args.base,
        "rank": args.lora_r,
        "packing": args.packing,
        "tokenization_and_cuda_execution": "pending",
    }


def validate_w17_data(chunks_path: Path, deepgram_dir: Path, audio_dir: Path, index: dict) -> tuple[dict, list[dict]]:
    if not chunks_path.is_file():
        raise PreflightError(f"Missing Whisper chunks: {chunks_path}")
    chunks = json.loads(chunks_path.read_text())
    if not isinstance(chunks, list) or not chunks:
        raise PreflightError("Whisper chunks must be a nonempty JSON list")
    eval_sources = {
        r.get("source") for r in chunks if isinstance(r, dict) and r.get("split") in {"eval", "test", "validation"}
    }
    audio_files = {}
    for path in audio_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".wav":
            audio_files.setdefault(path.stem, []).append(path)
    selected, excluded, checked_sources = [], 0, {}
    for number, row in enumerate(chunks, 1):
        if not isinstance(row, dict):
            raise PreflightError(f"Invalid Whisper chunk #{number}")
        if row.get("source") in eval_sources:
            excluded += 1
            continue
        label = f"{chunks_path}:{number}"
        check_training_row(row, index, label)
        source = row.get("source")
        start, end = row.get("start"), row.get("end")
        if (
            not isinstance(source, str)
            or not source
            or Path(source).name != source
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not 0 <= start < end
            or not math.isfinite(end)
        ):
            raise PreflightError(f"Invalid Whisper source/timing: {label}")
        if source not in checked_sources:
            transcripts = [
                p for p in (deepgram_dir / f"{source}.deepgram.json", deepgram_dir / f"{source}.json") if p.is_file()
            ]
            audio = audio_files.get(source, [])
            if len(transcripts) != 1 or len(audio) != 1:
                raise PreflightError(f"Require one exact Deepgram transcript and source WAV for {source}")
            words = json.loads(transcripts[0].read_text()).get("words")
            if not isinstance(words, list) or not words:
                raise PreflightError(f"Missing Deepgram words: {transcripts[0]}")
            for word in words:
                if (
                    not isinstance(word, dict)
                    or not isinstance(word.get("word"), str)
                    or not isinstance(word.get("start"), (int, float))
                    or not isinstance(word.get("end"), (int, float))
                    or not 0 <= word["start"] <= word["end"]
                    or not math.isfinite(word["end"])
                ):
                    raise PreflightError(f"Malformed Deepgram word: {transcripts[0]}")
            with wave.open(str(audio[0]), "rb") as wav:
                if wav.getframerate() != 16000 or wav.getnchannels() != 1:
                    raise PreflightError(f"Expected 16 kHz mono source WAV: {audio[0]}")
                duration = wav.getnframes() / wav.getframerate()
            checked_sources[source] = {
                "audio": str(audio[0]),
                "duration": duration,
                "deepgram_sha256": sha256(transcripts[0]),
                "words": words,
            }
        record = checked_sources[source]
        if end > record["duration"] + 0.001:
            raise PreflightError(f"Chunk exceeds source WAV: {label}")
        words = [w["word"] for w in record["words"] if w["start"] >= start - 0.1 and w["end"] <= end + 0.1]
        text = " ".join(words).strip()
        if len(text) < 20:
            raise PreflightError(f"Chunk has insufficient Deepgram-aligned text (min-chars 20): {label}")
        check_training_row({"transcription": text}, index, label)
        selected.append(row)
    if not selected:
        raise PreflightError("No eligible Whisper training chunks remain")
    return {
        "chunks_sha256": sha256(chunks_path),
        "training_chunks": len(selected),
        "excluded_evaluation_chunks": excluded,
        "sources": len(checked_sources),
        "source_inputs": {
            source: {k: v for k, v in record.items() if k != "words"} for source, record in checked_sources.items()
        },
        "mining_alignment_replay_and_cuda_execution": "pending",
    }, selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", choices=("w17", "gemma"))
    parser.add_argument(
        "--holdout", type=Path, action="append", help="Required holdout (repeatable); default v2 holdout"
    )
    parser.add_argument("--chunks-json", type=Path)
    parser.add_argument("--deepgram-dir", type=Path)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument(
        "--training-chunks-output", type=Path, help="Write checked train-only chunks; omit for read-only dry-run"
    )
    parser.add_argument("--report", type=Path)
    # Trainer flags are parsed by the actual trainer declarations after --.
    raw = list(sys.argv[1:] if argv is None else argv)
    divider = raw.index("--") if "--" in raw else len(raw)
    args = parser.parse_args(raw[:divider])
    if args.report and (
        args.report.exists()
        or (args.training_chunks_output and args.report.resolve() == args.training_chunks_output.resolve())
    ):
        parser.error("Report output must be a new file distinct from the training-chunks output")
    trainer_argv = raw[divider + 1 :]
    report = {
        "schema_version": 1,
        "recipe": args.recipe,
        "cpu_only": True,
        "training_started": False,
        "cuda_gate": "pending",
        "status": "failed",
    }
    try:
        trainer = parse_trainer_args("train_whisper.py" if args.recipe == "w17" else "train_gemma4.py", trainer_argv)
        if args.recipe == "w17":
            report["configuration"] = validate_whisper_config(trainer)
        index = holdout_index(args.holdout)
        report["holdouts"] = index["files"]
        if args.recipe == "gemma":
            report["data"] = validate_gemma(trainer, index)
        else:
            if not all((args.chunks_json, args.deepgram_dir, args.audio_dir)):
                raise PreflightError("W17 requires --chunks-json, --deepgram-dir and --audio-dir")
            report["data"], selected = validate_w17_data(args.chunks_json, args.deepgram_dir, args.audio_dir, index)
            if args.training_chunks_output:
                if (
                    args.training_chunks_output.resolve() == args.chunks_json.resolve()
                    or args.training_chunks_output.exists()
                ):
                    raise PreflightError("Training chunks output must be a new file; never overwrite the source")
                args.training_chunks_output.parent.mkdir(parents=True, exist_ok=True)
                args.training_chunks_output.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n")
        report["status"] = "passed"
    except (PreflightError, OSError, ValueError, KeyError, TypeError) as exc:
        report["error"] = str(exc)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
