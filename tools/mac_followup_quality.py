"""Audited public EN/ES evaluation; inference occurs only in serial child processes.

Run ``audit`` first. Public annotations are references, never local human approval
or training permission. Development and confirmation are separate run cohorts.
Importing this module does not import ML, audio or inference libraries.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import statistics
import struct
import sys
import tarfile
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
LANGUAGES = {"en": "en_us", "es": "es_419"}
PARTITIONS = {"development": "dev", "confirmation": "test"}
STT_MODELS = {
    "parakeet-mlx": "mlx-community/parakeet-tdt-0.6b-v3",
    "whisper-mlx": "mlx-community/whisper-large-v3-turbo",
    "ct2-small": "Systran/faster-whisper-small",
    "ct2-base": "Systran/faster-whisper-base",
}
TRANSLATION_MODELS = {
    "e4b": "mlx-community/gemma-4-e4b-it-OptiQ-4bit",
    "e2b": "mlx-community/gemma-4-e2b-it-OptiQ-4bit",
}
NORMALIZATION = {
    "version": 1,
    "description": "NFKC + casefold; Unicode punctuation becomes spaces; whitespace collapsed; accents and digits retained; no number expansion, STT corrections or reference-dependent rewriting",
}
CHRF_CONTRACT = {
    "implementation": "mac_followup_quality.chrf-v1",
    "character_order": 6,
    "beta": 2,
    "case_sensitive": True,
    "whitespace": "removed with str.split",
    "aggregation": "sum clipped matching/reference/hypothesis n-gram counts across corpus, then average precision/recall over orders present in both and compute F-beta; no word n-grams",
    "scale": "0..100; reference overlap, not semantic or human quality",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value, *, exclusive=False) -> None:
    from tools.mac_evaluation import write_json

    write_json(path, value, exclusive=exclusive)


def relative_file(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected safe relative file: {relative}")
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise ValueError(f"Missing or escaping file: {relative}")
    return target


def wav_header(data: bytes) -> dict:
    """Read PCM/IEEE-float RIFF shape without decoding or importing audio code."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Expected RIFF WAVE")
    offset, fmt, data_bytes = 12, None, None
    while offset + 8 <= len(data):
        kind, size = struct.unpack_from("<4sI", data, offset)
        body = offset + 8
        if body + size > len(data):
            raise ValueError("Truncated WAV chunk")
        if kind == b"fmt ":
            if fmt is not None or size < 16:
                raise ValueError("Invalid WAV fmt")
            fmt = struct.unpack_from("<HHIIHH", data, body)
        elif kind == b"data":
            if data_bytes is not None:
                raise ValueError("Multiple WAV data chunks")
            data_bytes = size
        offset = body + size + (size % 2)
    if fmt is None or data_bytes is None:
        raise ValueError("Missing WAV format/audio")
    encoding, channels, rate, byte_rate, block, bits = fmt
    if encoding not in (1, 3) or not block or block != channels * bits // 8 or data_bytes % block:
        raise ValueError("Unsupported WAV encoding/alignment")
    return {"encoding": encoding, "channels": channels, "sample_rate": rate, "num_samples": data_bytes // block}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(" " if unicodedata.category(c).startswith("P") else c for c in text).split())


def word_errors(reference: str, hypothesis: str) -> dict:
    """Levenshtein word counts with stable substitution/deletion/insertion ties."""
    a, b = normalize(reference).split(), normalize(hypothesis).split()
    previous = [(j, 0, 0, j) for j in range(len(b) + 1)]
    for i, left in enumerate(a, 1):
        current = [(i, 0, i, 0)]
        for j, right in enumerate(b, 1):
            if left == right:
                current.append(previous[j - 1])
                continue
            cost, sub, delete, insert = previous[j - 1]
            candidates = [(cost + 1, sub + 1, delete, insert)]
            cost, sub, delete, insert = previous[j]
            candidates.append((cost + 1, sub, delete + 1, insert))
            cost, sub, delete, insert = current[j - 1]
            candidates.append((cost + 1, sub, delete, insert + 1))
            current.append(min(candidates, key=lambda value: value[0]))
        previous = current
    errors, sub, delete, insert = previous[-1]
    return {
        "errors": errors,
        "reference_words": len(a),
        "substitutions": sub,
        "deletions": delete,
        "insertions": insert,
        "wer": errors / len(a) if a else None,
    }


def contains_term(text: str, term: str) -> bool:
    return f" {normalize(term)} " in f" {normalize(text)} "


def terms_in(reference: str, lang: str, glossary: dict) -> list[str]:
    candidates = glossary.keys() if lang == "en" else glossary.values()
    return sorted({term for term in candidates if term and contains_term(reference, term)})


def chrf_counts(reference: str, hypothesis: str) -> list[list[int]]:
    a, b = "".join(reference.split()), "".join(hypothesis.split())
    result = []
    for order in range(1, 7):
        ref = Counter(a[i : i + order] for i in range(len(a) - order + 1))
        hyp = Counter(b[i : i + order] for i in range(len(b) - order + 1))
        result.append([sum(ref.values()), sum(hyp.values()), sum((ref & hyp).values())])
    return result


def chrf_score(counts: list[list[int]]) -> float:
    valid = [(match / hyp, match / ref) for ref, hyp, match in counts if ref and hyp]
    if not valid:
        return 0.0
    precision = sum(p for p, _ in valid) / len(valid)
    recall = sum(r for _, r in valid) / len(valid)
    return 500 * precision * recall / (4 * precision + recall) if precision + recall else 0.0


def distribution(values) -> dict:
    valid = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    return {
        "n": len(valid),
        "p50": statistics.median(valid) if valid else None,
        "p95": valid[math.ceil(0.95 * len(valid)) - 1] if valid else None,
    }


def audit_manifest(manifest_path: Path, downloads: Path, audio_root: Path) -> tuple[dict, dict]:
    """Recheck selected rows against complete TSVs and exact selected tar members."""
    from tools.public_speech import parse_tsv, select

    manifest = read_json(manifest_path)
    if manifest.get("dataset") != "google/fleurs" or manifest.get("revision") != REVISION:
        raise ValueError("Expected pinned google/fleurs revision")
    records = manifest["records"]
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Empty or duplicate record IDs")
    sources = {s["file"]: s["sha256"] for s in manifest["sources"]}
    if len(sources) != len(manifest["sources"]):
        raise ValueError("Duplicate source identities")
    tables, evidence = {}, []
    for lang, config in LANGUAGES.items():
        for partition, split in PARTITIONS.items():
            name = f"data/{config}/{split}.tsv"
            path = relative_file(downloads / REVISION, name)
            actual = digest(path)
            if sources.get(name) != actual:
                raise ValueError(f"TSV hash mismatch: {name}")
            tables[lang, partition] = parse_tsv(path.read_text())
            evidence.append(
                {
                    "file": name,
                    "sha256": actual,
                    "size_bytes": path.stat().st_size,
                    "rows": len(tables[lang, partition]),
                }
            )
    ids = {
        p: {r["sentence_id"] for (lang, part), rows in tables.items() if part == p for r in rows} for p in PARTITIONS
    }
    if ids["development"] & ids["confirmation"]:
        raise ValueError("Upstream sentence IDs cross development/confirmation")
    texts = {
        p: {normalize(r["reference"]) for (lang, part), rows in tables.items() if part == p for r in rows}
        for p in PARTITIONS
    }
    copied = []
    glossary_path = ROOT / "bible_data/glossary/tier2_master.json"
    glossary = read_json(glossary_path)
    for lang, config in LANGUAGES.items():
        for partition, split in PARTITIONS.items():
            chosen = [r for r in records if (r["source_lang"], r["partition"]) == (lang, partition)]
            if not chosen:
                raise ValueError(f"Missing language/partition: {lang}/{partition}")
            expected = select(tables[lang, partition], len(chosen))
            if [r["filename"] for r in chosen] != [r["filename"] for r in expected]:
                raise ValueError("Selection differs from declared deterministic rule")
            other = "es" if lang == "en" else "en"
            parallel = defaultdict(set)
            parallel_files = defaultdict(list)
            for r in tables[other, partition]:
                parallel[r["sentence_id"]].add(r["reference"])
                parallel_files[r["sentence_id"]].append(r["filename"])
            for row, upstream in zip(chosen, expected, strict=True):
                if any(row.get(k) != value for k, value in upstream.items()):
                    raise ValueError(f"Selected annotation changed: {row['id']}")
                refset = parallel[row["sentence_id"]]
                if (
                    row.get("target_lang") != other
                    or len(refset) != 1
                    or row.get("translation_reference") not in refset
                ):
                    raise ValueError(f"Unverified parallel reference: {row['id']}")
                if row.get("human_approved_locally") is not False or row.get("training_eligible") is not False:
                    raise ValueError("Public evaluation cannot imply approval/training eligibility")
                raw = relative_file(audio_root, row["path"]).read_bytes()
                if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                    raise ValueError(f"Selected WAV hash mismatch: {row['id']}")
                header = wav_header(raw)
                if (header["channels"], header["sample_rate"], header["num_samples"]) != (1, 16000, row["num_samples"]):
                    raise ValueError(f"Selected WAV shape mismatch: {row['id']}")
                copied.append(
                    {
                        **row,
                        "terms": terms_in(row["reference"], lang, glossary),
                        "parallel_reference_evidence": {
                            "tsv": f"data/{LANGUAGES[other]}/{split}.tsv",
                            "tsv_sha256": sources[f"data/{LANGUAGES[other]}/{split}.tsv"],
                            "sentence_id": row["sentence_id"],
                            "matching_audio_filenames": sorted(parallel_files[row["sentence_id"]]),
                            "unique_reference_texts": 1,
                        },
                    }
                )
            archive_name = f"data/{config}/audio/{split}.tar.gz"
            archive = relative_file(downloads / REVISION, archive_name)
            actual = digest(archive)
            if sources.get(archive_name) != actual:
                raise ValueError(f"Archive hash mismatch: {archive_name}")
            wanted = {r["filename"]: r for r in chosen}
            seen = set()
            with tarfile.open(archive, "r|gz") as stream:
                for member in stream:
                    filename = Path(member.name).name
                    if filename not in wanted:
                        continue
                    if filename in seen or not member.isfile() or member.size > 16_000_000:
                        raise ValueError("Invalid/duplicate selected tar member")
                    audio = stream.extractfile(member)
                    if audio is None or hashlib.sha256(audio.read()).hexdigest() != wanted[filename]["sha256"]:
                        raise ValueError("Archive/local audio identity mismatch")
                    seen.add(filename)
            if seen != wanted.keys():
                raise ValueError("Selected audio absent from original archive")
            evidence.append(
                {
                    "file": archive_name,
                    "sha256": actual,
                    "size_bytes": archive.stat().st_size,
                    "selected_original_members_verified": len(seen),
                }
            )
    if len(copied) != len(records):
        raise ValueError("Records outside allowed language/partition")
    if len({r["sha256"] for r in records}) != len(records):
        raise ValueError("Duplicate audio content in selected evaluation")
    portable = {
        **manifest,
        "source_manifest_sha256": digest(manifest_path),
        "records": copied,
        "scoring_normalization": NORMALIZATION,
        "terms_source": {
            "file": "bible_data/glossary/tier2_master.json",
            "sha256": digest(glossary_path),
            "matching": "normalized whole word/phrase; unique term types per reference; broad glossary match, not a theological accuracy judgment",
        },
        "audio_resolution": "Pass --audio-root pointing to the original local fleurs-v1 directory; no audio is included here.",
    }
    from training.theological_canaries import THEOLOGICAL_CANARIES

    portable["canaries"] = [
        {
            "id": f"canary_{i:02}",
            "source": c["en"],
            "source_lang": "en",
            "target_lang": "es",
            "required_substrings": c["expected_substrings"],
            "reference": None,
            "reference_provenance": "repository lexical canary; no full reference translation",
            "human_approved_locally": False,
            "training_eligible": False,
        }
        for i, c in enumerate(THEOLOGICAL_CANARIES)
    ]
    portable["canary_source"] = {
        "file": "training/theological_canaries.py",
        "sha256": digest(ROOT / "training/theological_canaries.py"),
    }
    audit = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "completed": True,
        "source_manifest_sha256": digest(manifest_path),
        "portable_manifest_content_sha256": json_hash(portable),
        "records": len(records),
        "counts": dict(Counter(f"{r['source_lang']}/{r['partition']}" for r in records)),
        "full_upstream_sentence_counts": {p: len(v) for p, v in ids.items()},
        "cross_partition_sentence_id_overlap": [],
        "cross_partition_normalized_text_overlap": sorted(texts["development"] & texts["confirmation"]),
        "all_selected_audio_hashes_unique": True,
        "all_selected_audio_matches_original_archive": True,
        "sources": evidence,
        "human_approved_locally": False,
        "training_eligible": False,
        "limitations": [
            "Speaker identity is unavailable; speaker-disjointness is not established.",
            "Read public speech and parallel annotations do not represent live church conditions.",
            "Upstream/model-pretraining contamination is unknown; no independent model holdout claim.",
        ],
    }
    return portable, audit


def validate_portable(manifest: dict) -> None:
    if manifest.get("revision") != REVISION or not manifest.get("source_manifest_sha256"):
        raise ValueError("Run audit first and use its portable manifest")
    if manifest.get("scoring_normalization") != NORMALIZATION:
        raise ValueError("Scoring normalization changed")
    rows = manifest["records"]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Empty/duplicate evaluation identities")
    partition_ids = defaultdict(set)
    for row in rows:
        if row["source_lang"] not in LANGUAGES or row["partition"] not in PARTITIONS:
            raise ValueError("Invalid language/partition")
        if row.get("human_approved_locally") is not False or row.get("training_eligible") is not False:
            raise ValueError("Evaluation-only provenance required")
        if (
            not row.get("reference")
            or not row.get("translation_reference")
            or not row.get("parallel_reference_evidence")
        ):
            raise ValueError("Missing upstream references/evidence")
        partition_ids[row["partition"]].add(row["sentence_id"])
    if partition_ids["development"] & partition_ids["confirmation"]:
        raise ValueError("Sentence partition overlap")


def selected_items(manifest: dict, task: str, partition: str, lang: str, include_canaries: bool) -> list[dict]:
    validate_portable(manifest)
    rows = [r for r in manifest["records"] if r["partition"] == partition and r["source_lang"] == lang]
    if not rows:
        raise ValueError("No records for selected language/partition")
    if task == "stt":
        return rows
    items = [
        {
            "id": r["id"],
            "source": r["reference"],
            "reference": r["translation_reference"],
            "source_lang": lang,
            "target_lang": r["target_lang"],
            "partition": partition,
            "sentence_id": r["sentence_id"],
            "domain": "fleurs_parallel",
            "required_substrings": [],
            "reference_provenance": "upstream_fleurs_annotation",
            "human_approved_locally": False,
            "training_eligible": False,
        }
        for r in rows
    ]
    if include_canaries and lang == "en":
        items.extend({**r, "partition": "reused_canary", "domain": "theological_canary"} for r in manifest["canaries"])
    return items


def engine_config(name: str, task: str, model_override: str | None = None) -> dict:
    models = STT_MODELS if task == "stt" else TRANSLATION_MODELS
    if name not in models:
        raise ValueError(f"Unsupported {task} engine: {name}")
    return {
        "name": name,
        "task": task,
        "requested_model_id": model_override or models[name],
        "fallback_on_low_conf": False,
        "initial_prompt": None,
        "word_timestamps": False,
        "beam_size": None if name == "parakeet-mlx" else 5,
        "language_control": "automatic (API cannot force)" if name == "parakeet-mlx" else "forced",
        "device": "cpu" if name.startswith("ct2-") else "mlx",
        "compute_type": "int8" if name.startswith("ct2-") else None,
        "cpu_threads": 4,
        "num_workers": 1,
        "terminology_prompt": "none",
        "prompt_cache": False,
        "drafting": False,
        "thinking": False,
        "cache_limit_mb": 256,
    }


def model_inventory(config: dict) -> dict:
    """Resolve locally and hash the exact files before measuring (not timed inference)."""
    from engines.model_paths import resolve_model_path

    resolved = resolve_model_path(config["requested_model_id"], local_only=True)
    if not resolved or not Path(resolved).is_dir():
        raise FileNotFoundError(
            f"Model is not installed locally: {config['requested_model_id']}; install/pin it separately"
        )
    path = Path(resolved).resolve()
    files = [
        {"path": str(p.relative_to(path)), "sha256": digest(p), "size_bytes": p.stat().st_size}
        for p in sorted(path.rglob("*"))
        if p.is_file() and not p.name.startswith(".")
    ]
    if not files:
        raise ValueError("Empty model directory")
    return {
        "resolved_path": str(path),
        "snapshot_revision": path.name if path.parent.name == "snapshots" else None,
        "files": files,
        "files_sha256": json_hash(files),
    }


def make_engine(config: dict, resolved: str, lang: str):
    """Only called by the isolated worker; exact resolved paths defeat env model swaps."""
    name = config["name"]
    if config["task"] == "translate":
        from engines.mlx_engine import MLXGemmaEngine

        return MLXGemmaEngine(
            model_id=resolved,
            model_family="gemma4",
            use_prompt_cache=False,
            terminology_prompt="none",
            cache_limit_mb=256,
        )
    if name == "parakeet-mlx":
        from engines.parakeet_mlx_engine import ParakeetMLXEngine

        return ParakeetMLXEngine(model_id=resolved)
    if name == "whisper-mlx":
        from engines.mlx_engine import MLXWhisperEngine

        return MLXWhisperEngine(model_id=resolved, fallback_on_low_conf=False, session_language=lang)
    from engines.cuda_engine import FasterWhisperEngine

    return FasterWhisperEngine(
        model_id=resolved,
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=1,
        fallback_on_low_conf=False,
        local_files_only=True,
    )


def memory_snapshot() -> dict:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS ru_maxrss is bytes; Linux/other Unix reports KiB.
    result = {
        "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        "rss_scope": "process high-water RSS including model load, imports and input decoding; not additive with Metal",
        "peak_metal_bytes": None,
    }
    mx = sys.modules.get("mlx.core")
    if mx is not None:
        result["peak_metal_bytes"] = int(mx.get_peak_memory())
    return result


def read_audio(path: Path, item: dict):
    import numpy as np
    import soundfile as sf

    if digest(path) != item["sha256"]:
        raise ValueError("Audio changed after audit")
    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if rate != 16000 or audio.shape != (item["num_samples"],) or not np.isfinite(audio).all():
        raise ValueError("Expected finite original 16k mono audio")
    return audio


def measure_item(engine, item: dict, config: dict, audio_root: Path) -> dict:
    if config["task"] == "stt":
        audio = read_audio(relative_file(audio_root, item["path"]), item)
        start = time.perf_counter()
        result = engine.transcribe(
            audio,
            language=item["source_lang"],
            initial_prompt=None,
            word_timestamps=False,
            beam_size=config["beam_size"],
        )
        wall = (time.perf_counter() - start) * 1000
        if result.used_fallback:
            raise RuntimeError("Per-call model fallback makes this arm ineligible")
        terms = item.get("terms", [])
        return {
            "id": item["id"],
            "sentence_id": item["sentence_id"],
            "partition": item["partition"],
            "source_lang": item["source_lang"],
            "audio_sha256": item["sha256"],
            "audio_samples": item["num_samples"],
            "reference": item["reference"],
            "hypothesis": result.text,
            "wer_counts": word_errors(item["reference"], result.text),
            "terms": terms,
            "term_hits": [t for t in terms if contains_term(result.text, t)],
            "engine_latency_ms": result.latency_ms,
            "call_wall_ms": wall,
            "real_time_factor": wall / (item["num_samples"] / 16),
            "confidence": result.confidence,
            "avg_logprob": result.avg_logprob,
            "no_speech_prob": result.no_speech_prob,
            "used_fallback": result.used_fallback,
            "confidence_caveat": "engine-specific proxy, not calibrated across engines",
        }
    from engines.translation_prompts import dynamic_max_tokens

    start = time.perf_counter()
    result = engine.translate(item["source"], source_lang=item["source_lang"], target_lang=item["target_lang"])
    wall = (time.perf_counter() - start) * 1000
    terms = item.get("required_substrings", [])
    return {
        **item,
        "hypothesis": result.text,
        "engine_latency_ms": result.latency_ms,
        "call_wall_ms": wall,
        "generated_tokens": result.generated_tokens,
        "requested_max_tokens": dynamic_max_tokens(item["source"]),
        "budget_exhausted": result.generated_tokens >= dynamic_max_tokens(item["source"])
        if result.generated_tokens is not None
        else None,
        "finish_reason": result.finish_reason,
        "ttft_ms": result.ttft_ms,
        "prefill_ms": result.prefill_ms,
        "decode_ms": result.decode_ms,
        "chrf_counts": chrf_counts(item["reference"], result.text) if item.get("reference") else None,
        "canary_pass": all(t.casefold() in result.text.casefold() for t in terms) if terms else None,
        "canary_matching": "existing case-insensitive substring contract; a lexical check, not semantic approval",
    }


def source_identity() -> dict:
    from tools.mac_evaluation import environment

    data = environment()
    for package in ("numpy", "soundfile", "faster-whisper", "pydantic-settings"):
        try:
            data["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            data["versions"][package] = None
    # Include all engine/helper sources, even when untracked, for exact tool lineage.
    paths = [Path(__file__), ROOT / "models.lock.json", ROOT / "settings.py", *sorted((ROOT / "engines").glob("*.py"))]
    data["quality_source_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    data["normalization"] = NORMALIZATION
    return data


def run_worker(args) -> bool:
    from tools.benchmark_identity import load_primary_model

    manifest = read_json(args.manifest)
    items = selected_items(manifest, args.task, args.partition, args.lang, args.include_canaries)
    data = {
        "schema_version": 1,
        "completed": False,
        "status": "starting",
        "rows": [],
        "task": args.task,
        "engine": args.engine,
        "partition": args.partition,
        "source_lang": args.lang,
        "repeat": args.repeat,
        "manifest_sha256": digest(args.manifest),
        "items_sha256": json_hash(items),
        "expected_ids": [r["id"] for r in items],
        "human_approved_locally": False,
        "training_eligible": False,
        "environment": source_identity(),
        "started_at": datetime.now(UTC).isoformat(),
        "normalization": NORMALIZATION,
        "chrf_contract": CHRF_CONTRACT,
    }
    config = engine_config(args.engine, args.task, args.model_override)
    data["config"] = config
    engine = None
    save(args.output, data)
    try:
        inventory = model_inventory(config)
        data["model"] = inventory
        engine = make_engine(config, inventory["resolved_path"], args.lang)
        started = time.perf_counter()
        data["model_identity"] = load_primary_model(engine, inventory["resolved_path"])
        data["load_and_builtin_warmup_ms"] = (time.perf_counter() - started) * 1000
        data["warmup"] = "engine built-in silence/first forward only; no evaluation items warmed or omitted"
        data["status"] = "running"
        save(args.output, data)
        for item in items:
            try:
                row = measure_item(engine, item, config, args.audio_root)
                row["status"] = "ok"
            except Exception as exc:
                row = {"id": item["id"], "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            data["rows"].append(row)
            data.update(memory_snapshot())
            save(args.output, data)
        data["completed"] = len(data["rows"]) == len(items) and all(r["status"] == "ok" for r in data["rows"])
        data["status"] = "completed" if data["completed"] else "failed_items"
    except Exception as exc:
        data.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if engine is not None:
            try:
                engine.unload()
            except Exception as exc:
                data.update(completed=False, status="cleanup_failed", cleanup_error=f"{type(exc).__name__}: {exc}")
        data.update(memory_snapshot())
        data["ended_at"] = datetime.now(UTC).isoformat()
        save(args.output, data)
    return data["completed"]


def clean_environment() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("STARK_") or k == "STARK_MODELS_DIR"}
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONHASHSEED="0",
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        VECLIB_MAXIMUM_THREADS="4",
    )
    return env


def schedule(engines: list[str], languages: list[str], repeats: int) -> list[tuple]:
    if repeats < 1:
        raise ValueError("Repeats must be positive")
    return [
        (repeat, lang, name)
        for repeat in range(repeats)
        for lang in languages
        for name in (engines if repeat % 2 == 0 else list(reversed(engines)))
    ]


def run_serial(args) -> dict:
    """Use process-group cleanup from the existing harness; never parallelize workers."""
    from tools.mac_evaluation import _run_evaluation_worker

    manifest = read_json(args.manifest)
    validate_portable(manifest)
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ValueError("Timeout must be finite and positive")
    models = STT_MODELS if args.command == "stt" else TRANSLATION_MODELS
    if any(name not in models for name in args.engines) or len(set(args.engines)) != len(args.engines):
        raise ValueError("Invalid/duplicate engine selection")
    if len(set(args.languages)) != len(args.languages):
        raise ValueError("Duplicate languages")
    overrides = {}
    for value in args.model_override:
        key, separator, model = value.partition("=")
        if not separator or not model or key not in args.engines or key in overrides:
            raise ValueError("Override must be unique selected ENGINE=MODEL_ID_OR_LOCAL_PATH")
        overrides[key] = model
    planned = schedule(args.engines, args.languages, args.repeats)
    args.output.mkdir(parents=True, exist_ok=False)
    index = {
        "schema_version": 1,
        "completed": False,
        "manifest_sha256": digest(args.manifest),
        "task": args.command,
        "partition": args.partition,
        "serial": True,
        "runs": [],
        "planned_runs": len(planned),
        "environment": source_identity(),
        "human_approved_locally": False,
    }
    save(args.output / "index.json", index)
    for repeat, lang, name in planned:
        filename = f"{args.command}_{name}_{lang}_{args.partition}_r{repeat}.json"
        destination = args.output / filename
        command = [
            str(args.python),
            str(Path(__file__).resolve()),
            "_worker",
            "--task",
            args.command,
            "--manifest",
            str(args.manifest.resolve()),
            "--audio-root",
            str(args.audio_root.resolve()),
            "--engine",
            name,
            "--lang",
            lang,
            "--partition",
            args.partition,
            "--repeat",
            str(repeat),
            "--output",
            str(destination.resolve()),
        ]
        if args.include_canaries:
            command.append("--include-canaries")
        if name in overrides:
            command += ["--model-override", overrides[name]]
        result = _run_evaluation_worker(
            command, destination, args.manifest, env=clean_environment(), timeout=args.timeout_seconds
        )
        run = read_json(destination)
        index["runs"].append(
            {
                "file": filename,
                "sha256": digest(destination),
                "engine": name,
                "source_lang": lang,
                "repeat": repeat,
                "returncode": result.returncode,
                "completed": run.get("completed") is True and result.returncode == 0,
            }
        )
        save(args.output / "index.json", index)
    index["completed"] = len(index["runs"]) == len(planned) and all(r["completed"] for r in index["runs"])
    save(args.output / "index.json", index)
    return index


def row_matches_item(row: dict, item: dict, task: str) -> bool:
    """Validate lineage and recompute scores so wrong/stale row data cannot pass."""
    if row.get("status") != "ok" or not isinstance(row.get("hypothesis"), str):
        return False
    shared = ("id", "source_lang", "partition")
    if any(row.get(k) != item.get(k) for k in shared):
        return False
    hypothesis = row["hypothesis"]
    if task == "stt":
        terms = item.get("terms", [])
        return (
            row.get("audio_sha256") == item["sha256"]
            and row.get("audio_samples") == item["num_samples"]
            and row.get("reference") == item["reference"]
            and row.get("wer_counts") == word_errors(item["reference"], hypothesis)
            and row.get("terms") == terms
            and row.get("term_hits") == [t for t in terms if contains_term(hypothesis, t)]
            and row.get("used_fallback") is False
        )
    terms = item.get("required_substrings", [])
    expected_canary = all(t.casefold() in hypothesis.casefold() for t in terms) if terms else None
    counts = chrf_counts(item["reference"], hypothesis) if item.get("reference") else None
    return (
        all(row.get(k) == item.get(k) for k in ("source", "reference", "target_lang", "domain", "required_substrings"))
        and row.get("chrf_counts") == counts
        and row.get("canary_pass") == expected_canary
    )


def summarize_run(run: dict, manifest: dict) -> dict:
    expected = selected_items(
        manifest,
        run["task"],
        run["partition"],
        run["source_lang"],
        any(i.startswith("canary_") for i in run["expected_ids"]),
    )
    rows = run.get("rows", [])
    valid = (
        run.get("completed") is True
        and run.get("returncode") == 0
        and run.get("items_sha256") == json_hash(expected)
        and [r["id"] for r in rows] == [r["id"] for r in expected]
        and all(row_matches_item(row, item, run["task"]) for row, item in zip(rows, expected, strict=True))
        and run.get("normalization") == NORMALIZATION
        and run.get("chrf_contract") == CHRF_CONTRACT
        and run.get("model_identity", {}).get("primary_identity_verified") is True
        and run.get("model_identity", {}).get("actual_model_id") == run.get("model", {}).get("resolved_path")
    )
    result = {k: run.get(k) for k in ("task", "engine", "partition", "source_lang", "repeat", "status", "error")}
    result.update(
        eligible_for_comparison=valid,
        rows=len(rows),
        expected_rows=len(expected),
        failed_rows=sum(r.get("status") != "ok" for r in rows),
        peak_rss_bytes=run.get("peak_rss_bytes"),
        peak_metal_bytes=run.get("peak_metal_bytes"),
        model_files_sha256=run.get("model", {}).get("files_sha256"),
        source_sha256=run.get("environment", {}).get("quality_source_sha256"),
        load_and_builtin_warmup_ms=run.get("load_and_builtin_warmup_ms"),
        call_wall_ms=distribution(r.get("call_wall_ms") for r in rows),
        engine_latency_ms=distribution(r.get("engine_latency_ms") for r in rows),
    )
    if run["task"] == "stt":
        counts = [r["wer_counts"] for r in rows if r.get("wer_counts")]
        errors = sum(r["errors"] for r in counts)
        words = sum(r["reference_words"] for r in counts)
        opportunities = sum(len(r.get("terms", [])) for r in rows)
        result.update(
            reference_words=words,
            word_errors=errors,
            wer=errors / words if words else None,
            term_opportunities=opportunities,
            term_recall=sum(len(r.get("term_hits", [])) for r in rows) / opportunities if opportunities else None,
            real_time_factor=distribution(r.get("real_time_factor") for r in rows),
        )
    else:
        counts = [r["chrf_counts"] for r in rows if r.get("chrf_counts")]
        summed = [[sum(c[i][j] for c in counts) for j in range(3)] for i in range(6)]
        canaries = [r for r in rows if r.get("canary_pass") is not None]
        result.update(
            reference_items=len(counts),
            chrf=chrf_score(summed) if counts else None,
            canaries_passed=sum(r["canary_pass"] for r in canaries),
            canaries_total=len(canaries),
            budget_exhausted=sum(r.get("budget_exhausted") is True for r in rows),
            empty_outputs=sum(not r.get("hypothesis", "").strip() for r in rows),
            domain_call_wall_ms={
                domain: distribution(r.get("call_wall_ms") for r in rows if r.get("domain") == domain)
                for domain in ("fleurs_parallel", "theological_canary")
            },
        )
    return result


def report(directory: Path, manifest_path: Path, output: Path) -> dict:
    if output.exists() or output.with_suffix(".md").exists():
        raise FileExistsError("Use new report paths; preserve existing evidence")
    manifest = read_json(manifest_path)
    validate_portable(manifest)
    index = read_json(directory / "index.json")
    if index["manifest_sha256"] != digest(manifest_path):
        raise ValueError("Manifest differs from run index")
    summaries = []
    for entry in index["runs"]:
        path = relative_file(directory, entry["file"])
        if digest(path) != entry["sha256"]:
            raise ValueError("Run content differs from index")
        run = read_json(path)
        if run.get("manifest_sha256") != digest(manifest_path):
            raise ValueError("Worker manifest differs")
        try:
            summary = summarize_run(run, manifest)
            if any(run.get(k) != entry.get(k) for k in ("engine", "source_lang", "repeat", "returncode")):
                summary.update(eligible_for_comparison=False, error="Worker identity differs from index")
        except (KeyError, ValueError) as exc:
            summary = {
                "eligible_for_comparison": False,
                "error": f"Incomplete worker: {exc}",
                "engine": entry["engine"],
            }
        summaries.append({"file": entry["file"], "sha256": entry["sha256"], **summary})
    data = {
        "schema_version": 1,
        "completed": index["completed"]
        and len(summaries) == index["planned_runs"]
        and all(r["eligible_for_comparison"] for r in summaries),
        "manifest_sha256": digest(manifest_path),
        "runs": summaries,
        "normalization": NORMALIZATION,
        "chrf_contract": CHRF_CONTRACT,
        "pooling": "No pooling across engines, languages, partitions or repeats; failed/incomplete arms remain visible.",
        "human_review": "pending; no model promotion or training eligibility",
        "live_latency_gate": "not measured",
    }
    save(output, data, exclusive=True)
    lines = [
        "# Public speech quality evaluation",
        "",
        f"Manifest: `{data['manifest_sha256']}`.",
        "",
        "These are isolated whole-recording STT or fixed-reference-text translation measurements. They exclude VAD, queueing, audio capture and browser delivery. No model promotion or human quality approval is implied.",
        "",
        "| Engine | Direction | Partition / repeat | Complete + identity valid | n | Call p50 / p95 ms | WER | chrF | Canary |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]

    def fmt(value):
        return "—" if value is None else f"{value:.3f}"

    for row in summaries:
        latency = row.get("call_wall_ms", {})
        lines.append(
            f"| {row.get('engine')} | {row.get('source_lang', 'unknown')} | {row.get('partition', 'unknown')} / {row.get('repeat', '—')} | {row['eligible_for_comparison']} | {row.get('rows', 0)} | {fmt(latency.get('p50'))} / {fmt(latency.get('p95'))} | {fmt(row.get('wer'))} | {fmt(row.get('chrf'))} | {row.get('canaries_passed', 0)}/{row.get('canaries_total', 0)} |"
        )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return data


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--downloads", type=Path, required=True)
    audit.add_argument("--audio-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    for task, models in (("stt", STT_MODELS), ("translate", TRANSLATION_MODELS)):
        run = sub.add_parser(task)
        run.add_argument("--manifest", type=Path, required=True)
        run.add_argument("--audio-root", type=Path, required=True)
        run.add_argument("--output", type=Path, required=True)
        run.add_argument("--partition", choices=list(PARTITIONS), default="development")
        run.add_argument("--languages", choices=list(LANGUAGES), nargs="+", default=list(LANGUAGES))
        run.add_argument("--engines", choices=list(models), nargs="+", default=list(models))
        run.add_argument("--repeats", type=int, default=1)
        run.add_argument("--timeout-seconds", type=float, default=3600)
        run.add_argument("--python", type=Path, default=Path(sys.executable))
        run.add_argument("--model-override", action="append", default=[])
        run.add_argument("--include-canaries", action="store_true")
    worker = sub.add_parser("_worker", help=argparse.SUPPRESS)
    for name in ("manifest", "audio-root", "output"):
        worker.add_argument(f"--{name}", type=Path, required=True)
    worker.add_argument("--task", choices=["stt", "translate"], required=True)
    worker.add_argument("--engine", required=True)
    worker.add_argument("--lang", choices=list(LANGUAGES), required=True)
    worker.add_argument("--partition", choices=list(PARTITIONS), required=True)
    worker.add_argument("--repeat", type=int, required=True)
    worker.add_argument("--include-canaries", action="store_true")
    worker.add_argument("--model-override")
    rep = sub.add_parser("report")
    rep.add_argument("--manifest", type=Path, required=True)
    rep.add_argument("--input", type=Path, required=True)
    rep.add_argument("--output", type=Path, required=True)
    return p


def main() -> int:
    args = parser().parse_args()
    if args.command == "audit":
        if args.output.exists():
            raise FileExistsError("Use a new audit output directory")
        portable, audit = audit_manifest(args.manifest, args.downloads, args.audio_root)
        args.output.mkdir(parents=True)
        save(args.output / "manifest.json", portable, exclusive=True)
        audit["portable_manifest_sha256"] = digest(args.output / "manifest.json")
        save(args.output / "audit.json", audit, exclusive=True)
        print(json.dumps({"records": audit["records"], "counts": audit["counts"]}))
    elif args.command in ("stt", "translate"):
        return 0 if run_serial(args)["completed"] else 1
    elif args.command == "_worker":
        return 0 if run_worker(args) else 1
    else:
        report(args.input, args.manifest, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
