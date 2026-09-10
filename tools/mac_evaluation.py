"""Reproducible Mac quality and latency comparisons; no model imports at startup.

Prepare an immutable manifest, run isolated model workers / alternating 1x
replays, and produce reports with explicit missing human-validation gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data: object, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def environment() -> dict:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=ROOT, capture_output=True, check=False)
    versions = {}
    for name in ("mlx", "mlx-lm", "mlx-optiq", "parakeet-mlx", "mlx-whisper", "ctranslate2", "sacrebleu"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "code_sha": sha.stdout.strip(),
        "tracked_diff_sha256": hashlib.sha256(diff.stdout).hexdigest(),
        "source_sha256": {
            name: digest(ROOT / name)
            for name in (
                "dry_run_ab.py",
                "settings.py",
                "engines/mlx_engine.py",
                "engines/translation_prompts.py",
                "engines/model_paths.py",
                "tools/pipeline_timing.py",
                "tools/mac_evaluation.py",
            )
            if (ROOT / name).exists()
        },
        "platform": platform.platform(),
        "python": sys.version,
        "versions": versions,
        "created_at": datetime.now(UTC).isoformat(),
    }


def stats(values: list[float]) -> dict:
    values = sorted(float(x) for x in values if math.isfinite(float(x)))
    return {
        "n": len(values),
        "p50": statistics.median(values) if values else None,
        "p95": values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)] if values else None,
    }


def _jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            except ValueError:
                continue
    return rows


def prepare_manifest(root: Path, output: Path) -> dict:
    """Freeze existing inputs; machine predictions are NEVER human references."""
    from training.theological_canaries import THEOLOGICAL_CANARIES

    replay_path = root / "stark_data/replay/manifest.json"
    replay = []
    if replay_path.exists():
        for i, clip in enumerate(json.loads(replay_path.read_text())["clips"]):
            wav = (replay_path.parent / clip["path"]).resolve()
            if digest(wav) != clip["sha256"]:
                raise ValueError(f"Replay checksum mismatch: {wav}")
            replay.append(
                {
                    **clip,
                    "id": f"replay_{i}",
                    "path": str(wav.relative_to(root.resolve())),
                    "provenance": "synthetic_piper" if clip["lang"] == "es" else "church_replay",
                    "reference_status": "unreviewed",
                    "conditions": ["speech"],
                }
            )
    candidates: dict[str, list] = {"en": [], "es": []}
    seen = set()
    for path in sorted((root / "metrics").glob("diagnostics_*.jsonl"), reverse=True):
        sid = path.stem.removeprefix("diagnostics_")
        lang = sid.rsplit("_", 1)[-1]
        if lang not in candidates:
            continue
        for row in _jsonl(path):
            if "event" in row or "chunk_id" not in row or not row.get("audio_path"):
                continue
            wav = Path(row["audio_path"])
            wav = wav if wav.is_absolute() else root / wav
            if not wav.is_file() or not wav.resolve().is_relative_to(root.resolve()):
                continue
            audio_hash = digest(wav)
            if audio_hash in seen:
                continue
            seen.add(audio_hash)
            candidates[lang].append(
                {
                    "id": f"{sid}_{row['chunk_id']}",
                    "session_id": sid,
                    "chunk_id": row["chunk_id"],
                    "path": str(wav.resolve().relative_to(root.resolve())),
                    "sha256": audio_hash,
                    "lang": lang,
                    "provenance": "unconfirmed_live_session",
                    "reference_status": "pending",
                    "reference_text": None,
                    "reference_translation": None,
                    "draft_transcript": row.get("english", ""),
                    "conditions": [],
                    "terms": [],
                    "speaker": None,
                }
            )
    # Spread selections across the available sessions instead of taking one run.
    selected = []
    for rows in candidates.values():
        n = min(50, len(rows))
        selected.extend(rows[(i * len(rows)) // n] for i in range(n))
    translations = [
        {
            "id": f"canary_{i:02}",
            "source_lang": "en",
            "target_lang": "es",
            "source": c["en"],
            "reference": None,
            "required_terms": c["expected_substrings"],
            "domain": "theological_canary",
        }
        for i, c in enumerate(THEOLOGICAL_CANARIES)
    ]
    reference_path = root / "bible_data/aligned/verse_pairs_test.jsonl"
    refs = [r for r in _jsonl(reference_path) if 6 <= len(r.get("en", "").split()) <= 35 and r.get("es")]
    # Stable reference sample from the existing held-out parallel corpus.
    refs.sort(key=lambda r: hashlib.sha256((r["en"] + r["es"]).encode()).hexdigest())
    for i, row in enumerate(refs[:25]):
        for src, target in (("en", "es"), ("es", "en")):
            translations.append(
                {
                    "id": f"verse_{i:02}_{src}",
                    "source_lang": src,
                    "target_lang": target,
                    "source": row[src],
                    "reference": row[target],
                    "required_terms": [],
                    "domain": "public_domain_verse",
                    "verse_id": row.get("verse_id"),
                }
            )
    data = {
        "schema_version": 1,
        "id": output.stem,
        "created_at": datetime.now(UTC).isoformat(),
        "usage": "evaluation_only",
        "reference_corpus_sha256": digest(reference_path),
        "replays": replay,
        "utterances": selected,
        "translations": translations,
        "gates": {
            "human_reference_target_per_language": 50,
            "natural_spanish": "pending",
            "two_speaker": "pending",
            "bilingual_blind_review": "pending",
        },
    }
    data = realign_references(data, root / "bible_data/scrollmapper/formats/sqlite")
    write_json(output, data, exclusive=True)
    return data


def validate_manifest(manifest: dict, root: Path = ROOT) -> dict:
    """Verify all audio and count only explicitly approved natural references."""
    if manifest.get("usage") != "evaluation_only":
        raise ValueError("Expected an evaluation-only manifest")
    counts = {"en": 0, "es": 0}
    for row in manifest.get("replays", []) + manifest.get("utterances", []):
        if digest(root / row["path"]) != row["sha256"]:
            raise ValueError(f"Audio changed: {row['id']}")
    for row in manifest.get("utterances", []):
        if (
            row.get("reference_status") == "approved"
            and row.get("reference_text")
            and row.get("provenance") == "natural_speech"
        ):
            counts[row["lang"]] += 1
    return {"approved_natural_utterances": counts, "natural_audio_gate": all(n >= 50 for n in counts.values())}


def apply_annotations(manifest: dict, annotations: list[dict]) -> dict:
    """Return a new frozen manifest; preserve audio identity and immutable inputs."""
    result = json.loads(json.dumps(manifest))
    by_id = {r["id"]: r for r in result["utterances"]}
    allowed = {
        "reference_text",
        "reference_translation",
        "reference_status",
        "provenance",
        "conditions",
        "terms",
        "speaker",
    }
    for change in annotations:
        row = by_id[change["id"]]
        if set(change) - allowed - {"id"}:
            raise ValueError("Annotations cannot change audio or language identity")
        if change.get("reference_status") == "approved" and not change.get("reference_text", row.get("reference_text")):
            raise ValueError("Approved references need a transcript")
        row.update({k: v for k, v in change.items() if k in allowed})
    return result


def realign_references(manifest: dict, database_dir: Path) -> dict:
    """Repair text references by book/chapter/verse, never sequential row IDs.

    Preserve every model input. Ambiguous text or unsupported verse numbering
    loses its reference instead of generating a misleading score.
    """
    import sqlite3

    result = json.loads(json.dumps(manifest))
    indexes = {"en": defaultdict(set), "es": defaultdict(set)}
    canonical = {"en": {}, "es": {}}
    sources = {"en": ("KJV", "ASV", "BBE", "OEB", "YLT"), "es": ("SpaRV", "SpaPlatense")}
    checksums = {}
    for lang, names in sources.items():
        for name in names:
            path = database_dir / f"{name}.db"
            if not path.exists():
                continue
            checksums[name] = digest(path)
            with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
                rows = connection.execute(
                    f"SELECT b.name,v.chapter,v.verse,v.text FROM {name}_verses v JOIN {name}_books b ON v.book_id=b.id"  # nosec B608: fixed internal translation names
                )
                for book, chapter, verse, text in rows:
                    if not text:
                        continue
                    key = (book.casefold(), chapter, verse)
                    # Platense Psalms have versification differences even after
                    # a structural join; require an independently checked source.
                    if name == "SpaPlatense" and key[0] == "psalms":
                        continue
                    indexes[lang][text.strip()].add(key)
                    if name in ("KJV", "SpaRV"):
                        canonical[lang][key] = text.strip()
    for row in result["translations"]:
        if row["domain"] != "public_domain_verse":
            continue
        keys = indexes[row["source_lang"]].get(row["source"].strip(), set())
        key = next(iter(keys)) if len(keys) == 1 else None
        row["reference"] = canonical[row["target_lang"]].get(key)
        row["reference_status"] = "structurally_aligned" if row["reference"] else "unavailable"
        row["canonical_reference"] = list(key) if key else None
        row["reference_translation"] = "SpaRV" if row["target_lang"] == "es" else "KJV"
    result["reference_database_sha256"] = checksums
    result["reference_alignment"] = (
        "exact source text lookup; unique book name/chapter/verse; KJV and RVR1909 reference"
    )
    return result


def rescore_quality(source: Path, destination: Path, manifest_path: Path) -> None:
    """Rebind references without rerunning or changing any measured prediction."""
    manifest = json.loads(manifest_path.read_text())
    by_id = {r["id"]: r for r in manifest["translations"]}
    run = json.loads(source.read_text())
    for row in run["rows"]:
        fixed = by_id[row["id"]]
        if any(row[k] != fixed[k] for k in ("source", "source_lang", "target_lang")):
            raise ValueError("Rescoring cannot change model inputs or language direction")
        for key in ("reference", "reference_status", "canonical_reference", "reference_translation"):
            if key in fixed:
                row[key] = fixed[key]
    run["original_manifest_sha256"] = run["manifest_sha256"]
    run["source_run_sha256"] = digest(source)
    run["manifest_sha256"] = digest(manifest_path)
    write_json(destination, run, exclusive=True)


def _model_id(size: str) -> str:
    from settings import settings

    return getattr(settings.translation, f"mlx_model_gemma4_{size}")


def quality_worker(args) -> None:
    """One model per process to avoid retaining another model's Metal memory."""
    from engines.mlx_engine import MLXGemmaEngine

    manifest = json.loads(args.manifest.read_text())
    model_id = _model_id(args.size)
    engine = MLXGemmaEngine(
        model_id=model_id, model_family="gemma4", use_prompt_cache=False, terminology_prompt=args.policy
    )
    engine.load()
    engine.translate("The grace of God is sufficient.", source_lang="en", target_lang="es")
    data = {
        "environment": environment(),
        "manifest_sha256": digest(args.manifest),
        "size": args.size,
        "model_id": model_id,
        "policy": args.policy,
        "target_override": args.target,
        "rows": [],
        "human_review": "pending",
        "completed": False,
    }
    try:
        from engines.model_paths import resolve_model_path

        data["resolved_model"] = str(resolve_model_path(model_id))
    except (ImportError, TypeError):
        data["resolved_model"] = model_id
    try:
        for item in manifest["translations"]:
            if args.target and item["source_lang"] != "en":
                continue
            target = args.target or item["target_lang"]
            row = {
                **item,
                "target_lang": target,
                "reference": None if args.target else item["reference"],
                "required_terms": [] if args.target else item["required_terms"],
                "runs": [],
            }
            for _ in range(args.runs):
                res = engine.translate(item["source"], source_lang=item["source_lang"], target_lang=target)
                row["runs"].append(
                    {
                        "text": res.text,
                        "latency_ms": res.latency_ms,
                        "generated_tokens": res.generated_tokens,
                        "finish_reason": res.finish_reason,
                    }
                )
            terms = row["required_terms"]
            row["canary_pass"] = (
                all(all(t.casefold() in x["text"].casefold() for t in terms) for x in row["runs"]) if terms else None
            )
            data["rows"].append(row)
            write_json(args.output, data)
            print(
                f"{args.size}/{args.policy} {item['id']}: {stats([r['latency_ms'] for r in row['runs']])}", flush=True
            )
        import mlx.core as mx

        data["peak_metal_bytes"] = mx.get_peak_memory()
        import resource

        data["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        data["completed"] = True
    finally:
        engine.unload()
        write_json(args.output, data)


def run_quality(args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    for policy in args.policies:
        for size in args.sizes:
            output = args.output / f"quality_{size}_{policy}{'_' + args.target if args.target else ''}.json"
            if output.exists():
                raise FileExistsError(output)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "_quality-worker",
                "--manifest",
                str(args.manifest.resolve()),
                "--size",
                size,
                "--policy",
                policy,
                "--runs",
                str(args.runs),
                "--output",
                str(output.resolve()),
            ]
            if args.target:
                command += ["--target", args.target]
            env = dict(os.environ, STARK_TRANSLATE_TERMINOLOGY_PROMPT=policy)
            with output.with_suffix(".log").open("w") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            print(f"{size}/{policy}: exit {result.returncode}; {output}", flush=True)
            if result.returncode:
                raise RuntimeError(f"Quality worker failed; see {output.with_suffix('.log')}")


def replay_schedule(clips: list[dict], sizes: list[str], runs: int) -> list[tuple[int, dict, str]]:
    if runs < 1:
        raise ValueError("runs must be positive")
    return [
        (repeat, clip, size)
        for repeat in range(runs)
        for clip in clips
        for size in (sizes if repeat % 2 == 0 else list(reversed(sizes)))
    ]


def stt_worker(args) -> None:
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    from tools.stt_roundtrip_compare import make_engine, normalize

    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    engine = make_engine(args.engine)
    engine.load()
    result = {
        "engine": args.engine,
        "model_id": engine.model_id,
        "environment": environment(),
        "manifest_sha256": digest(args.manifest),
        "rows": [],
        "completed": False,
    }
    try:
        for row in manifest["utterances"]:
            rate, audio = wavfile.read(ROOT / row["path"])
            if np.issubdtype(audio.dtype, np.integer):
                audio = audio.astype(np.float32) / max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
            audio = audio.astype(np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if rate != 16000:
                common = math.gcd(rate, 16000)
                audio = resample_poly(audio, 16000 // common, rate // common)
            hypotheses = [engine.transcribe(audio, language=row["lang"]) for _ in range(args.runs)]
            approved = row.get("reference_status") == "approved" and row.get("reference_text")
            reference = row["reference_text"] if approved else None
            hypothesis = hypotheses[0].text
            wer_counts = None
            if reference:
                from jiwer import process_words

                measures = process_words(normalize(reference), normalize(hypothesis))
                wer_counts = {
                    "errors": measures.substitutions + measures.deletions + measures.insertions,
                    "reference_words": measures.hits + measures.substitutions + measures.deletions,
                }
            result["rows"].append(
                {
                    "id": row["id"],
                    "lang": row["lang"],
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "wer_counts": wer_counts,
                    "reference_status": row.get("reference_status"),
                    "provenance": row["provenance"],
                    "terms": row.get("terms", []),
                    "term_hits": [term for term in row.get("terms", []) if normalize(term) in normalize(hypothesis)],
                    "latency_ms": [r.latency_ms for r in hypotheses],
                }
            )
            write_json(args.output, result)
        import resource

        import mlx.core as mx

        result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["peak_metal_bytes"] = mx.get_peak_memory()
        result["completed"] = True
    finally:
        engine.unload()
        write_json(args.output, result)


def run_stt(args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("mlx", "parakeet-mlx"):
        destination = args.output / f"stt_{name}.json"
        if destination.exists():
            raise FileExistsError(destination)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_stt-worker",
            "--manifest",
            str(args.manifest.resolve()),
            "--output",
            str(destination.resolve()),
            "--engine",
            name,
            "--runs",
            str(args.runs),
        ]
        with destination.with_suffix(".log").open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise RuntimeError(f"STT worker failed; see {destination.with_suffix('.log')}")
        print(f"{name}: {destination}", flush=True)


def run_replays(args) -> None:
    from tools.replay_bench import run_replay

    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    clips = manifest["replays"]
    if args.language:
        clips = [c for c in clips if c["lang"] == args.language]
    for index, (repeat, clip, size) in enumerate(replay_schedule(clips, args.sizes, args.runs)):
        tag = f"{args.tag}_{args.experiment}_{size}_r{repeat}_{clip['id']}_{clip['lang']}"
        destination = args.output / f"{tag}.json"
        if destination.exists():
            print(f"Already recorded: {tag}", flush=True)
            continue
        extra = [
            "--model-family",
            "gemma4",
            "--gemma4-size",
            size,
            "--no-mts",
            "--stt-backend",
            "parakeet-mlx" if clip["lang"] == "en" else "mlx",
            "--partial-interval",
            "0.6",
        ]
        if args.pipeline_args:
            extra.extend(args.pipeline_args)
        run_environment = environment()
        try:
            report = run_replay(clip, ROOT / clip["path"], tag, extra, index, ROOT / "metrics")
        except RuntimeError as exc:
            report = {"error": str(exc), "session_id": tag}
        report.update(
            {
                "size": size,
                "experiment": args.experiment,
                "repeat": repeat,
                "clip_id": clip["id"],
                "language": clip["lang"],
                "provenance": clip["provenance"],
                "replay_speed": float(os.environ.get("STARK_REPLAY_SPEED", "1")),
                "manifest_sha256": digest(args.manifest),
                "environment": run_environment,
                "experiment_settings": {
                    k: v for k, v in os.environ.items() if k.startswith(("STARK_VAD_", "STARK_TRANSLATE_"))
                },
            }
        )
        write_json(destination, report)
        print(f"{tag}: {report.get('chunk_count', 0)} finals; {report.get('error', 'OK')}", flush=True)


def run_experiments(args) -> None:
    """Run bounded experiments separately, each with three alternating pairs."""
    spec = json.loads(args.spec.read_text())
    for config in spec["experiments"]:
        options = argparse.Namespace(**vars(args))
        options.output = args.output / config["name"]
        options.experiment = config["name"]
        options.pipeline_args = spec["baseline_arguments"] + config["arguments"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", options.experiment):
            raise ValueError("Invalid experiment name")
        run_replays(options)


def _chrf(rows: list[dict]) -> float | None:
    pairs = [(r["runs"][0]["text"], r["reference"]) for r in rows if r.get("reference") and r.get("runs")]
    if not pairs:
        return None
    try:
        from sacrebleu.metrics import CHRF
    except ImportError:
        return None
    return CHRF(word_order=2).corpus_score([p[0] for p in pairs], [[p[1] for p in pairs]]).score


def report_results(directory: Path, output: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    report = {
        "evaluation": validate_manifest(manifest),
        "quality": [],
        "replays": [],
        "human_review": "pending",
        "default_model": "e4b",
        "latency_gate": "pending_browser_ack",
    }
    quality = []
    failures = []
    for path in sorted(directory.rglob("quality_*.json")):
        run = json.loads(path.read_text())
        if run.get("manifest_sha256") and run["manifest_sha256"] != digest(manifest_path):
            continue
        if not run.get("rows") or not run.get("completed"):
            failures.append({"file": str(path), "error": "Quality run incomplete; see adjacent log"})
            continue
        quality.append(run)
        for lang in sorted({r["source_lang"] for r in run["rows"]}):
            rows = [r for r in run["rows"] if r["source_lang"] == lang]
            canaries = [r for r in rows if r.get("canary_pass") is not None]
            report["quality"].append(
                {
                    "size": run["size"],
                    "policy": run["policy"],
                    "source_lang": lang,
                    "target_lang": rows[0]["target_lang"],
                    "items": len(rows),
                    "reference_items": sum(bool(r.get("reference")) for r in rows),
                    "latency_ms": stats([x["latency_ms"] for r in rows for x in r["runs"]]),
                    "canaries_passed": sum(r["canary_pass"] for r in canaries),
                    "canaries_total": len(canaries),
                    "chrf_plus_plus": _chrf(rows),
                    "peak_metal_bytes": run.get("peak_metal_bytes"),
                }
            )
    report["stt"] = []
    for path in sorted(directory.rglob("stt_*.json")):
        run = json.loads(path.read_text())
        if not run.get("completed"):
            failures.append({"file": str(path), "error": "STT run incomplete; see adjacent log"})
            continue
        for lang in ("en", "es"):
            rows = [r for r in run["rows"] if r["lang"] == lang]
            counts = [r["wer_counts"] for r in rows if r.get("wer_counts")]
            word_count = sum(c["reference_words"] for c in counts)
            terms = sum(len(r["terms"]) for r in rows if r.get("reference"))
            report["stt"].append(
                {
                    "engine": run["engine"],
                    "lang": lang,
                    "items": len(rows),
                    "latency_ms": stats([x for r in rows for x in r["latency_ms"]]),
                    "reference_items": len(counts),
                    "reference_words": word_count,
                    "wer": sum(c["errors"] for c in counts) / word_count if word_count else None,
                    "term_recall": sum(len(r["term_hits"]) for r in rows if r.get("reference")) / terms
                    if terms
                    else None,
                }
            )
    report["quality_deltas"] = []
    for baseline in report["quality"]:
        if baseline["size"] != "e4b":
            continue
        fast = next(
            (
                r
                for r in report["quality"]
                if r["size"] == "e2b" and all(r[k] == baseline[k] for k in ("source_lang", "target_lang", "policy"))
            ),
            None,
        )
        if fast:
            before, after = baseline["latency_ms"]["p50"], fast["latency_ms"]["p50"]
            report["quality_deltas"].append(
                {
                    "policy": baseline["policy"],
                    "source_lang": baseline["source_lang"],
                    "target_lang": baseline["target_lang"],
                    "e2b_p50_reduction_percent": 100 * (before - after) / before if before else None,
                    "e2b_canary_pass_delta": fast["canaries_passed"] - baseline["canaries_passed"],
                    "e2b_chrf_delta": fast["chrf_plus_plus"] - baseline["chrf_plus_plus"]
                    if fast["chrf_plus_plus"] is not None and baseline["chrf_plus_plus"] is not None
                    else None,
                }
            )
    groups: dict[tuple, list] = defaultdict(list)
    event_groups: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(directory.rglob("*.json")):
        run = json.loads(path.read_text())
        if "experiment" not in run:
            continue
        if run.get("error"):
            failures.append({"session_id": run.get("session_id"), "error": run["error"]})
            continue
        if float(run.get("replay_speed", 1)) != 1:
            failures.append(
                {"session_id": run.get("session_id"), "error": "Accelerated replay excluded from latency gate"}
            )
            continue
        group_key = (run["experiment"], run["size"], run["language"], run["provenance"])
        events = event_groups[group_key]
        first_utterances = set()
        prior_emission = None
        for partial in _jsonl(ROOT / "metrics" / f"partials_{run['session_id']}.jsonl"):
            if partial.get("timing_schema_version") != 2:
                continue
            if partial.get("captured_end_to_partial_ms") is not None:
                events["captured_end_to_partial_ms"].append(partial["captured_end_to_partial_ms"])
            uid = partial.get("utterance_id")
            if uid not in first_utterances and partial.get("speech_start_to_partial_ms") is not None:
                events["first_partial_ms"].append(partial["speech_start_to_partial_ms"])
                first_utterances.add(uid)
            emitted = partial.get("emitted_at_ms")
            if emitted is not None:
                if prior_emission is not None:
                    events["partial_update_gap_ms"].append(emitted - prior_emission)
                prior_emission = emitted
        for ack in _jsonl(ROOT / "metrics" / f"display_metrics_{run['session_id']}.jsonl"):
            if ack.get("event") != "caption_rendered" or not ack.get("visible") or ack.get("stage") != "complete":
                continue
            for field in ("receive_to_render_ms", "speech_end_to_ack_upper_bound_ms"):
                if ack.get(field) is not None:
                    events[field].append(ack[field])
        csv_path = ROOT / "metrics" / f"ab_metrics_{run['session_id']}.csv"
        if csv_path.exists():
            with csv_path.open() as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                # Separate source, endpoint reason and timing definition, never average percentiles.
                schema = row.get("timing_schema_version") or "legacy"
                key = (
                    run["experiment"],
                    run["size"],
                    run["language"],
                    run["provenance"],
                    schema,
                    row.get("endpoint_reason") or row.get("finalization_reason") or "unknown",
                )
                groups[key].append(row)
    for key, rows in groups.items():
        fields = {}
        for field in (
            "speech_end_to_final_ms",
            "vad_wait_ms",
            "stt_queue_wait_ms",
            "translation_queue_wait_ms",
            "e2e_latency_ms",
        ):
            values = []
            for row in rows:
                try:
                    values.append(float(row[field]))
                except (KeyError, ValueError, TypeError):
                    pass
            fields[field] = stats(values)
        report["replays"].append(
            dict(zip(("experiment", "size", "language", "provenance", "schema", "endpoint"), key, strict=True))
            | {"metrics": fields}
        )
    report["failures"] = failures
    report["caption_events"] = [
        dict(zip(("experiment", "size", "language", "provenance"), key, strict=True))
        | {"metrics": {name: stats(values) for name, values in metrics.items()}}
        for key, metrics in event_groups.items()
    ]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", report)
    lines = [
        "# Mac E4B / E2B comparison",
        "",
        "E4B remains the default. No model-selection UI has been added.",
        "",
        "Human-reference audio coverage: " + json.dumps(report["evaluation"]),
        "",
        "Natural Spanish / two-speaker / bilingual review gates remain pending until explicitly validated.",
        "",
        "## Translation on identical input",
        "",
        "chrF++ measures agreement with public-domain verse wording, not overall translation accuracy. References use unique book/chapter/verse alignment to KJV and RVR1909; ambiguous source text has no score. Existing model outputs are preserved when references are repaired. Canary checks require every listed term; older benchmarks checked only the first term.",
        "",
        "| Model | Prompt | Direction | Items | References | p50 ms | p95 ms | Canaries | chrF++ |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(v):
        return "—" if v is None else f"{v:.1f}"

    for r in report["quality"]:
        lines.append(
            f"| {r['size']} | {r['policy']} | {r['source_lang']}→{r['target_lang']} | {r['items']} | {r['reference_items']} | {fmt(r['latency_ms']['p50'])} | {fmt(r['latency_ms']['p95'])} | {r['canaries_passed']}/{r['canaries_total']} | {fmt(r['chrf_plus_plus'])} |"
        )
    lines += [
        "",
        "### All theological canaries",
        "",
        "| Input | Model / prompt | Result | Required terms | Output |",
        "|---|---|---|---|---|",
    ]
    for run in quality:
        for row in run["rows"]:
            if row.get("canary_pass") is None:
                continue
            actual = row["runs"][0]["text"].replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {row['id']} | {run['size']} / {run['policy']} | {'pass' if row['canary_pass'] else 'miss'} | {', '.join(row['required_terms'])} | {actual} |"
            )
    lines += [
        "",
        "## E2B tradeoff relative to E4B",
        "",
        "| Prompt | Direction | Median latency reduction | Canary pass difference | chrF++ difference |",
        "|---|---|---:|---:|---:|",
    ]
    for delta in report["quality_deltas"]:
        lines.append(
            f"| {delta['policy']} | {delta['source_lang']}→{delta['target_lang']} | {fmt(delta['e2b_p50_reduction_percent'])}% | {delta['e2b_canary_pass_delta']:+d} | {fmt(delta['e2b_chrf_delta'])} |"
        )
    lines += [
        "",
        "## STT on saved session audio",
        "",
        "Unreviewed machine transcripts are not WER references. Unconfirmed session recordings cannot pass the natural-speech gate.",
        "",
        "| Engine | Language | Audio items | Approved references | WER | p50 ms | p95 ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in report["stt"]:
        lines.append(
            f"| {r['engine']} | {r['lang']} | {r['items']} | {r['reference_items']} | {fmt(r['wer'])} | {fmt(r['latency_ms']['p50'])} | {fmt(r['latency_ms']['p95'])} |"
        )
    lines += [
        "",
        "## Real-time server latency",
        "",
        "Speech end is estimated from captured VAD-positive frames. Server-final latency ends at payload readiness; it is not browser display latency. Synthetic and natural inputs and endpoint types are separated.",
        "",
        "| Experiment | Model | Language/source | Endpoint | n | p50 ms | p95 ms |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for r in report["replays"]:
        s = r["metrics"]["speech_end_to_final_ms"]
        lines.append(
            f"| {r['experiment']} | {r['size']} | {r['language']}/{r['provenance']} | {r['endpoint']} | {s['n']} | {fmt(s['p50'])} | {fmt(s['p95'])} |"
        )
    lines += [
        "",
        "## Partial delivery and visible browser timing",
        "",
        "First-partial delay starts at the first captured speech frame. Update gaps are between emitted partials, including speaking pauses. Browser receipt-to-render is measured on the client; speech-end-to-ack is an upper bound including return-network time.",
        "",
        "| Experiment | Model | Source | Metric | n | p50 ms | p95 ms |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in report["caption_events"]:
        for name, values in row["metrics"].items():
            lines.append(
                f"| {row['experiment']} | {row['size']} | {row['language']}/{row['provenance']} | {name} | {values['n']} | {fmt(values['p50'])} | {fmt(values['p95'])} |"
            )
    # Per-item output differences + deterministic, blinded review rows.
    by_model = {(r["size"], r["policy"], r.get("target_override")): r for r in quality}
    blind, key_rows = [], []
    lines += ["", "## Changed translation examples", ""]
    for identity, left in by_model.items():
        size, policy, target = identity
        right = by_model.get(("e2b", policy, target))
        if size != "e4b" or not right:
            continue
        right_rows = {r["id"]: r for r in right["rows"]}
        for a in left["rows"]:
            b = right_rows.get(a["id"])
            if not b:
                continue
            first, second = a["runs"][0]["text"], b["runs"][0]["text"]
            pair_id = f"{policy}_{target or 'bilingual'}_{a['id']}"
            swap = int(hashlib.sha256(pair_id.encode()).hexdigest(), 16) % 2
            blind.append(
                {
                    "id": pair_id,
                    "source": a["source"],
                    "reference": a["reference"],
                    "A": second if swap else first,
                    "B": first if swap else second,
                    "meaning_error_A": None,
                    "meaning_error_B": None,
                    "terminology_preference": None,
                    "reviewed": False,
                }
            )
            key_rows.append({"id": pair_id, "A": "e2b" if swap else "e4b", "B": "e4b" if swap else "e2b"})
            if first != second:
                lines += [f"**{pair_id}** — {a['source']}", "", f"- E4B: {first}", f"- E2B: {second}", ""]
    for name, records in (("blind_review.jsonl", blind), ("review_key.jsonl", key_rows)):
        (output / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    lines += [
        "## Pending gates",
        "",
        "- ≥50 approved natural-speech reference utterances per language.",
        "- Bilingual review of meaning errors and terminology preferences (blind_review.jsonl).",
        "- Visible-browser render acknowledgments for the sub-second caption-delivery gate.",
        "- Physical second-output / hotplug and real two-speaker validation.",
        "",
        f"Recorded failed runs: {len(failures)}. Failures are retained in comparison.json.",
    ]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    validate = subs.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    annotate = subs.add_parser("annotate")
    annotate.add_argument("--manifest", type=Path, required=True)
    annotate.add_argument("--annotations", type=Path, required=True)
    annotate.add_argument("--output", type=Path, required=True)
    repair = subs.add_parser("realign-references")
    repair.add_argument("--manifest", type=Path, required=True)
    repair.add_argument("--database-dir", type=Path, default=ROOT / "bible_data/scrollmapper/formats/sqlite")
    repair.add_argument("--output", type=Path, required=True)
    rescore = subs.add_parser("rescore-quality")
    rescore.add_argument("--manifest", type=Path, required=True)
    rescore.add_argument("--input", type=Path, required=True)
    rescore.add_argument("--output", type=Path, required=True)
    experiments = subs.add_parser("experiments")
    experiments.add_argument("--manifest", type=Path, required=True)
    experiments.add_argument("--spec", type=Path, required=True)
    experiments.add_argument("--output", type=Path, required=True)
    experiments.add_argument("--tag", required=True)
    experiments.add_argument("--runs", type=int, default=3)
    experiments.add_argument("--sizes", nargs="+", choices=["e4b", "e2b"], default=["e4b", "e2b"])
    experiments.add_argument("--language", choices=["en", "es"])
    for name in ("quality", "_quality-worker", "replay", "report", "stt", "_stt-worker"):
        sub = subs.add_parser(name)
        sub.add_argument("--manifest", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        if name == "report":
            sub.add_argument("--input", type=Path, required=True)
        else:
            sub.add_argument("--runs", type=int, default=3)
            if name == "_stt-worker":
                sub.add_argument("--engine", choices=["mlx", "parakeet-mlx"], required=True)
            elif name == "_quality-worker":
                sub.add_argument("--size", choices=["e4b", "e2b"], required=True)
                sub.add_argument("--policy", choices=["none", "church"], default="none")
            else:
                sub.add_argument("--sizes", nargs="+", choices=["e4b", "e2b"], default=["e4b", "e2b"])
            if name in ("quality", "_quality-worker"):
                sub.add_argument("--target", choices=["hi"])
                if name == "quality":
                    sub.add_argument("--policies", nargs="+", choices=["none", "church"], default=["none"])
            elif name == "replay":
                sub.add_argument("--tag", default=datetime.now().strftime("mac_%Y%m%d_%H%M%S"))
                sub.add_argument("--experiment", default="baseline")
                sub.add_argument("--language", choices=["en", "es"])
                sub.add_argument("--pipeline-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    if hasattr(args, "runs") and args.runs < 1:
        parser.error("--runs must be positive")
    if args.command == "prepare":
        prepare_manifest(ROOT, args.output)
    elif args.command == "validate":
        print(json.dumps(validate_manifest(json.loads(args.manifest.read_text())), indent=2))
    elif args.command == "annotate":
        data = apply_annotations(json.loads(args.manifest.read_text()), _jsonl(args.annotations))
        data["id"] = args.output.stem
        data["parent_manifest_sha256"] = digest(args.manifest)
        write_json(args.output, data, exclusive=True)
    elif args.command == "realign-references":
        data = realign_references(json.loads(args.manifest.read_text()), args.database_dir)
        data["id"] = args.output.stem
        data["parent_manifest_sha256"] = digest(args.manifest)
        write_json(args.output, data, exclusive=True)
    elif args.command == "rescore-quality":
        rescore_quality(args.input, args.output, args.manifest)
    elif args.command == "experiments":
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
            parser.error("tag must use letters, digits, underscores or hyphens")
        run_experiments(args)
    elif args.command == "quality":
        run_quality(args)
    elif args.command == "_quality-worker":
        quality_worker(args)
    elif args.command == "stt":
        run_stt(args)
    elif args.command == "_stt-worker":
        stt_worker(args)
    elif args.command == "replay":
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag + args.experiment):
            parser.error("tag and experiment must use letters, digits, underscores or hyphens")
        run_replays(args)
    else:
        report_results(args.input, args.output, args.manifest)


if __name__ == "__main__":
    main()
