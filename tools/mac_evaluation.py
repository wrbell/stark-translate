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
import tempfile
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
    encoded = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    _write_text(path, encoded, exclusive=exclusive)


def _write_text(path: Path, encoded: str, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as f:
            f.write(encoded)
        return
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(encoded)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _numeric(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except (ValueError, TypeError):
        return None


def _pipeline_arguments(run: dict) -> list[str] | None:
    if "experiment" not in run:
        return None
    command = run.get("command", [])
    arguments = command[command.index("--http-port") + 2 :] if "--http-port" in command else []
    # These dimensions already have separate report groups or a fixed 1x gate.
    for name in ("--gemma4-size", "--replay-speed"):
        arguments = _without_argument(arguments, name)
    return arguments


def _runtime_cohort(run: dict) -> str:
    """Separate inference code/config changes without treating doc commits as model changes."""
    env = run.get("environment", {})
    metadata = run.get("session_metadata", {})
    payload = {
        "sources": {
            k: v
            for k, v in env.get("source_sha256", {}).items()
            if k.startswith("engines/") or k in {"dry_run_ab.py", "settings.py", "tools/pipeline_timing.py"}
        },
        "versions": env.get("versions"),
        "platform": env.get("platform"),
        "python": env.get("python"),
        "settings": run.get("experiment_settings"),
        "vad": metadata.get("vad"),
        "translation": metadata.get("translation"),
        "stt_backend": metadata.get("stt_backend"),
        "pipeline_arguments": _pipeline_arguments(run),
    }
    startup_hash = _startup_pipeline_sha256(run)
    if startup_hash:
        payload["sources"]["dry_run_ab.py"] = startup_hash
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _startup_pipeline_sha256(run: dict) -> str | None:
    lifecycle = run.get("session_lifecycle", {})
    value = lifecycle.get("pipeline_sha256")
    if (
        lifecycle.get("session_id") == run.get("session_id")
        and lifecycle.get("schema_version") == 1
        and isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value)
    ):
        return value
    return None


def _clip_matches(run: dict, manifest: dict) -> bool:
    from tools.replay_bench import argument_value

    clip = run.get("clip", {})
    expected = next((c for c in manifest.get("replays", []) if c["id"] == run.get("clip_id", clip.get("id"))), None)
    if expected is None or any(clip.get(k) != expected.get(k) for k in ("id", "sha256", "path", "lang", "provenance")):
        return False
    metadata = run.get("session_metadata", {})
    command = run.get("command", [])
    command_audio = argument_value(command, "--audio-file")
    return (
        run.get("language") == expected["lang"]
        and run.get("provenance") == expected["provenance"]
        and metadata.get("input_audio_sha256", expected["sha256"]) == expected["sha256"]
        and argument_value(command, "--session-id") == run.get("session_id")
        and argument_value(command, "--lang") == expected["lang"]
        and argument_value(command, "--gemma4-size") == run.get("size")
        and command_audio is not None
        and (ROOT / command_audio).resolve() == (ROOT / expected["path"]).resolve()
        and _numeric(argument_value(command, "--replay-speed", run.get("replay_speed")))
        == _numeric(run.get("replay_speed"))
    )


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
    values = sorted(n for x in values if (n := _numeric(x)) is not None)
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


def validate_manifest(manifest: dict, root: Path | None = None) -> dict:
    """Verify all audio and count only explicitly approved natural references."""
    root = root or ROOT
    if manifest.get("usage") != "evaluation_only":
        raise ValueError("Expected an evaluation-only manifest")
    counts = {"en": 0, "es": 0}
    for section in ("replays", "utterances", "translations"):
        ids = [row.get("id") for row in manifest.get(section, [])]
        if any(not isinstance(identity, str) or not identity for identity in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"Manifest {section} needs unique, nonempty IDs")
    for row in manifest.get("replays", []) + manifest.get("utterances", []):
        if digest(root / row["path"]) != row["sha256"]:
            raise ValueError(f"Audio changed: {row['id']}")
    approved_audio = set()
    for row in manifest.get("utterances", []):
        if row.get("lang") not in counts:
            raise ValueError("Utterance language must be en or es")
        if (
            row.get("reference_status") == "approved"
            and row.get("reference_text")
            and row.get("provenance") == "natural_speech"
        ):
            identity = row["lang"], row["sha256"]
            if identity not in approved_audio:
                counts[row["lang"]] += 1
                approved_audio.add(identity)
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
    if len(by_id) != len(manifest["translations"]):
        raise ValueError("Rescoring manifest contains duplicate input IDs")
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


def _without_argument(arguments: list[str], name: str) -> list[str]:
    result = []
    skip = False
    for arg in arguments:
        if skip:
            skip = False
        elif arg == name:
            skip = True
        elif not arg.startswith(name + "="):
            result.append(arg)
    return result


def _resolved_model(model_id: str) -> dict:
    """Record the actual offline snapshot and revision without loading a model."""
    from engines.model_paths import resolve_model_path

    resolved = resolve_model_path(model_id, local_only=True)
    path = Path(resolved) if resolved else None
    return {
        "resolved_model": resolved or model_id,
        "model_revision": path.name if path and path.parent.name == "snapshots" else None,
    }


def _model_id(size: str) -> str:
    from settings import settings

    return getattr(settings.translation, f"mlx_model_gemma4_{size}")


def quality_worker(args) -> None:
    """One model per process to avoid retaining another model's Metal memory."""
    from engines.mlx_engine import MLXGemmaEngine
    from engines.translation_prompts import dynamic_max_tokens

    manifest = json.loads(args.manifest.read_text())
    model_id = _model_id(args.size)
    engine = MLXGemmaEngine(
        model_id=model_id, model_family="gemma4", use_prompt_cache=False, terminology_prompt=args.policy
    )
    engine.load()
    engine.translate("The grace of God is sufficient.", source_lang="en", target_lang=args.target or "es")
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
        data.update(_resolved_model(model_id))
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
                        "requested_max_tokens": dynamic_max_tokens(item["source"]),
                        "budget_exhausted": res.generated_tokens >= dynamic_max_tokens(item["source"])
                        if res.generated_tokens is not None
                        else None,
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
            result = _run_evaluation_worker(
                command, output, args.manifest, env=env, timeout=getattr(args, "timeout_seconds", 1800)
            )
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
        "supported_languages": ["en", "es"],
        "language_control": "auto" if args.engine == "parakeet-mlx" else "forced",
        **_resolved_model(engine.model_id),
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
                    "audio_sha256": row["sha256"],
                    "language_control": result["language_control"],
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
        completed = _run_evaluation_worker(
            command, destination, args.manifest, timeout=getattr(args, "timeout_seconds", 1800)
        )
        if completed.returncode:
            raise RuntimeError(f"STT worker failed; see {destination.with_suffix('.log')}")
        print(f"{name}: {destination}", flush=True)


def _run_evaluation_worker(command, destination, manifest, *, env=None, timeout=1800):
    from tools.replay_bench import run_child

    if timeout is None or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Worker timeout must be finite and positive")
    data = {
        "command": command,
        "manifest_sha256": digest(manifest),
        "environment": environment(),
        "completed": False,
        "rows": [],
        "timeout_s": timeout,
    }
    write_json(destination, data, exclusive=True)
    try:
        with destination.with_suffix(".log").open("w") as log:
            result = run_child(command, cwd=ROOT, env=env, stdout=log, timeout=timeout)
        failure = f"Worker exited with status {result.returncode}" if result.returncode else None
    except subprocess.TimeoutExpired:
        failure = f"Worker exceeded {timeout:g}s timeout and was terminated"
        result = subprocess.CompletedProcess(command, -1)
    except OSError as exc:
        failure = f"Worker failed to launch: {exc}"
        result = subprocess.CompletedProcess(command, -1)
    if destination.exists():
        data.update(json.loads(destination.read_text()))
    data.update(command=command, returncode=result.returncode, timeout_s=timeout)
    if failure:
        data.update(completed=False, error=failure)
    write_json(destination, data)
    return result


def run_replays(args) -> None:
    from tools.replay_bench import REPLAY_MANAGED_ARGS, argument_value, run_replay

    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    clips = manifest["replays"]
    if args.language:
        clips = [c for c in clips if c["lang"] == args.language]
    for index, (repeat, clip, size) in enumerate(replay_schedule(clips, args.sizes, args.runs)):
        tag = f"{args.tag}_{args.experiment}_{size}_r{repeat}_{clip['id']}_{clip['lang']}"
        destination = args.output / f"{tag}.json"
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
            if any(
                arg.split("=", 1)[0] in REPLAY_MANAGED_ARGS | {"--gemma4-size", "--replay-speed"}
                for arg in args.pipeline_args
            ):
                raise ValueError("Pipeline args override comparison-managed audio, model, or replay speed")
            extra.extend(args.pipeline_args)
        extra += ["--replay-speed", "1"]
        run_environment = environment()
        experiment_settings = {k: v for k, v in os.environ.items() if k.startswith(("STARK_VAD_", "STARK_TRANSLATE_"))}
        if destination.exists():
            previous = json.loads(destination.read_text())
            if previous.get("error") or previous.get("returncode") != 0:
                raise ValueError(f"Recorded replay failed/incomplete: {destination}; inspect it and use a fresh tag")
            command = previous.get("command", [])
            old_extra = command[command.index("--http-port") + 2 :] if "--http-port" in command else []
            if (
                not _clip_matches(previous, manifest)
                or previous.get("size") != size
                or _without_argument(old_extra, "--replay-speed") != _without_argument(extra, "--replay-speed")
                or _numeric(argument_value(old_extra, "--replay-speed", previous.get("replay_speed"))) != 1
                or not (ROOT / "metrics" / f"ab_metrics_{tag}.csv").exists()
                or previous.get("experiment_settings", {}) != experiment_settings
                or _runtime_cohort(previous)
                != _runtime_cohort({**previous, "environment": run_environment, "session_lifecycle": {}})
            ):
                raise ValueError(f"Replay configuration/source changed for {destination}; use a fresh tag")
            print(f"Already recorded compatible successful replay: {tag}", flush=True)
            continue
        try:
            report = run_replay(
                clip,
                ROOT / clip["path"],
                tag,
                extra,
                index,
                ROOT / "metrics",
                timeout_s=getattr(args, "timeout_seconds", None),
            )
        except RuntimeError as exc:
            failure_path = ROOT / "metrics" / f"replay_{tag}.json"
            report = json.loads(failure_path.read_text()) if failure_path.exists() else {}
            report.update(error=report.get("error", str(exc)), session_id=tag)
        report.update(
            {
                "size": size,
                "experiment": args.experiment,
                "repeat": repeat,
                "clip_id": clip["id"],
                "language": clip["lang"],
                "provenance": clip["provenance"],
                "replay_speed": report.get("replay_speed", 1.0),
                "manifest_sha256": digest(args.manifest),
                "environment": run_environment,
                "experiment_settings": experiment_settings,
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


def _quality_inputs_match(run: dict, manifest: dict) -> bool:
    target = run.get("target_override")
    expected = {r["id"]: r for r in manifest.get("translations", []) if not target or r["source_lang"] == "en"}
    rows = run.get("rows", [])
    if len(rows) != len(expected) or {r.get("id") for r in rows} != set(expected):
        return False
    return all(
        row.get("source") == expected[row["id"]]["source"]
        and row.get("source_lang") == expected[row["id"]]["source_lang"]
        and row.get("target_lang") == (target or expected[row["id"]]["target_lang"])
        and row.get("reference") == (None if target else expected[row["id"]].get("reference"))
        and row.get("required_terms", []) == ([] if target else expected[row["id"]].get("required_terms", []))
        and bool(row.get("runs"))
        for row in rows
    )


def _stt_inputs_match(run: dict, manifest: dict) -> bool:
    expected = {r["id"]: r for r in manifest.get("utterances", [])}
    rows = run.get("rows", [])
    if len(rows) != len(expected) or {r.get("id") for r in rows} != set(expected):
        return False
    for row in rows:
        item = expected[row["id"]]
        reference = item.get("reference_text") if item.get("reference_status") == "approved" else None
        if (
            row.get("audio_sha256") != item["sha256"]
            or row.get("lang") != item["lang"]
            or row.get("provenance") != item["provenance"]
            or row.get("reference") != (reference or None)
            or row.get("terms", []) != item.get("terms", [])
            or not row.get("latency_ms")
        ):
            return False
    return True


def _preserve_blind_reviews(path: Path, proposed: list[dict]) -> list[dict]:
    """Keep annotations only when the same labeled comparison is unchanged.

    Even unfinished ratings and reviewer notes count as work. Refuse to replace
    their source/reference/outputs or remove their pair; use a fresh report
    directory for a different cohort. Validation precedes every artifact write.
    """
    if not path.exists():
        return proposed
    pairs = {row["id"]: dict(row) for row in proposed}
    immutable = {"id", "source", "reference", "A", "B"}
    seen = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            prior = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"Invalid existing blind review at {path}:{line_number}; no report files changed") from exc
        if not isinstance(prior, dict) or not isinstance(prior.get("id"), str) or prior["id"] in seen:
            raise ValueError(
                f"Invalid or duplicate existing blind-review ID at {path}:{line_number}; no report files changed"
            )
        seen.add(prior["id"])
        annotations = {key: value for key, value in prior.items() if key not in immutable}
        has_work = bool(prior.get("reviewed")) or any(
            value is not None and value != "" for key, value in annotations.items() if key != "reviewed"
        )
        incoming = pairs.get(prior["id"])
        same_pair = incoming is not None and all(prior.get(key) == incoming.get(key) for key in immutable)
        if not same_pair:
            if has_work:
                raise ValueError(
                    f"Reviewed pair {prior['id']} changed or disappeared; use a fresh report directory to preserve human ratings"
                )
            continue
        incoming.update(annotations)
    return [pairs[row["id"]] for row in proposed]


def _report_browser_acknowledgments(rows: list[dict], acknowledgments: list[dict], session_id: str) -> dict:
    """Join visible final ACKs to unique CSV chunks; never pool clients or endpoints."""
    chunks: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("chunk_id") not in (None, ""):
            chunks[str(row["chunk_id"])].append(row)

    def endpoint_key(row):
        return (
            str(row.get("timing_schema_version") or "legacy"),
            row.get("endpoint_reason") or row.get("finalization_reason") or "unknown",
            row.get("timing_source") or "unknown",
        )

    endpoints: dict[tuple, set] = defaultdict(set)
    for cid, matching in chunks.items():
        if len(matching) == 1 and endpoint_key(matching[0])[0] == "2":
            endpoints[endpoint_key(matching[0])].add(cid)
    samples: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    received: dict[tuple, set] = defaultdict(set)
    clients = set()
    seen_events, seen_chunks = set(), set()
    unmatched = 0
    for ack in acknowledgments:
        identity = ack.get("client_id"), ack.get("event_id")
        if (
            ack.get("event") != "caption_rendered"
            or ack.get("visible") is not True
            or ack.get("timing_schema_version") != 2
            or any(not isinstance(value, str) or not value for value in identity)
            or identity in seen_events
        ):
            continue
        if ack.get("session_id", session_id) != session_id:
            if ack.get("stage") == "complete":
                unmatched += 1
            continue
        seen_events.add(identity)
        client = identity[0]
        # Partial-only browsers are real observed clients with zero final
        # coverage, not evidence that no browser was present.
        if ack.get("stage") in {"partial", "final", "translation_a"}:
            clients.add(client)
            continue
        if ack.get("stage") != "complete":
            continue
        cid = str(ack["chunk_id"]) if ack.get("chunk_id") is not None else ""
        matching = chunks.get(cid, [])
        if len(matching) != 1 or endpoint_key(matching[0])[0] != "2":
            unmatched += 1
            continue
        if (client, cid) in seen_chunks:
            continue
        seen_chunks.add((client, cid))
        clients.add(client)
        key = (client, *endpoint_key(matching[0]))
        received[key].add(cid)
        for field in ("receive_to_render_ms", "speech_end_to_ack_upper_bound_ms"):
            if (delay := _numeric(ack.get(field))) is not None:
                samples[key][field].append(delay)

    cohorts = []
    for client in sorted(clients):
        for endpoint, finalized in sorted(endpoints.items()):
            key = (client, *endpoint)
            acknowledged = received[key]
            cohorts.append(
                {
                    "session_id": session_id,
                    "client_id": client,
                    **dict(zip(("schema", "endpoint", "timing_source"), endpoint, strict=True)),
                    "metrics": {
                        field: stats(samples[key][field])
                        for field in ("receive_to_render_ms", "speech_end_to_ack_upper_bound_ms")
                    },
                    "final_ack_coverage": {
                        "received_final_chunks": len(acknowledged),
                        "finalized_chunks": len(finalized),
                        "missing_final_chunks": sorted(finalized - acknowledged),
                    },
                }
            )
    acknowledged = {cid for _, cid in seen_chunks}
    return {
        "cohorts": cohorts,
        "client_count": len(clients),
        "unmatched_final_acks": unmatched,
        "coverage": {
            "session_id": session_id,
            "received_final_chunks": len(acknowledged),
            "finalized_chunks": len(chunks),
            "missing_final_chunks": sorted(set(chunks) - acknowledged),
            "available": bool(chunks),
        },
    }


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
    exclusions = []
    manifest_hash = digest(manifest_path)
    for path in sorted(directory.rglob("quality_*.json")):
        run = json.loads(path.read_text())
        if run.get("manifest_sha256") != manifest_hash:
            exclusions.append(
                {"file": str(path), "reason": "Quality manifest differs or is missing; rescore references explicitly"}
            )
            continue
        if not run.get("rows") or not run.get("completed"):
            failures.append({"file": str(path), "error": "Quality run incomplete; see adjacent log"})
            continue
        if not _quality_inputs_match(run, manifest):
            exclusions.append({"file": str(path), "reason": "Quality inputs/references do not match the full manifest"})
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
                    "runtime_cohort": _runtime_cohort(run),
                }
            )
    report["stt"] = []
    for path in sorted(directory.rglob("stt_*.json")):
        run = json.loads(path.read_text())
        if run.get("manifest_sha256") != manifest_hash:
            exclusions.append({"file": str(path), "reason": "STT manifest differs or is missing"})
            continue
        if not run.get("completed"):
            failures.append({"file": str(path), "error": "STT run incomplete; see adjacent log"})
            continue
        if not _stt_inputs_match(run, manifest):
            exclusions.append({"file": str(path), "reason": "STT rows do not match all frozen audio inputs/references"})
            continue
        for lang in ("en", "es"):
            rows = [r for r in run["rows"] if r["lang"] == lang]
            counts = [r["wer_counts"] for r in rows if r.get("wer_counts")]
            word_count = sum(c["reference_words"] for c in counts)
            terms = sum(len(r["terms"]) for r in rows if r.get("reference"))
            report["stt"].append(
                {
                    "engine": run["engine"],
                    "language_control": run.get("language_control"),
                    "model_id": run.get("model_id"),
                    "model_revision": run.get("model_revision"),
                    "runtime_cohort": _runtime_cohort(run),
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
                if r["size"] == "e2b"
                and all(r[k] == baseline[k] for k in ("source_lang", "target_lang", "policy", "runtime_cohort"))
            ),
            None,
        )
        if fast:
            before, after = baseline["latency_ms"]["p50"], fast["latency_ms"]["p50"]
            report["quality_deltas"].append(
                {
                    "policy": baseline["policy"],
                    "runtime_cohort": baseline["runtime_cohort"],
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
    source_cohorts = {}
    browser_groups: dict[tuple, list] = defaultdict(list)
    browser_client_counts: dict[tuple, int] = defaultdict(int)
    unmatched_acks: dict[tuple, int] = defaultdict(int)
    ack_coverage: dict[tuple, list] = defaultdict(list)
    counter_sessions: dict[tuple, dict] = defaultdict(dict)
    seen_sessions = set()
    for path in sorted(directory.rglob("*.json")):
        run = json.loads(path.read_text())
        if "experiment" not in run:
            continue
        if run.get("error") or run.get("returncode") != 0:
            failures.append(
                {"session_id": run.get("session_id"), "error": run.get("error", "Replay did not complete successfully")}
            )
            continue
        if not _clip_matches(run, manifest):
            exclusions.append(
                {"file": str(path), "reason": "Replay audio identity/language/provenance does not match manifest"}
            )
            continue
        if run.get("session_id") in seen_sessions:
            exclusions.append({"file": str(path), "reason": "Duplicate replay session; samples counted once"})
            continue
        seen_sessions.add(run["session_id"])
        actual_speed = run.get("session_metadata", {}).get("replay_speed", run.get("replay_speed"))
        if _numeric(actual_speed) != 1 or _numeric(run.get("replay_speed")) != 1:
            failures.append(
                {"session_id": run.get("session_id"), "error": "Accelerated replay excluded from latency gate"}
            )
            continue
        if not run.get("session_lifecycle"):
            from tools.replay_bench import read_lifecycle

            run["session_lifecycle"] = read_lifecycle(ROOT / "metrics", run["session_id"])
        cohort = _runtime_cohort(run)
        startup_hash = _startup_pipeline_sha256(run)
        source_cohorts[cohort] = {
            "pipeline_hash_basis": "startup_source_file" if startup_hash else "recorded_environment_snapshot",
            "startup_pipeline_sha256": startup_hash,
            "note": "Source file observed before model loading; imported engine hashes still come from the environment snapshot."
            if startup_hash
            else "Historical snapshot timing may be after inference; identical loaded code is not proved.",
        }
        group_key = (
            run["experiment"],
            run["size"],
            run["language"],
            run["provenance"],
            run["clip"]["id"],
            _runtime_cohort(run),
        )
        events = event_groups[group_key]
        for record in reversed(_jsonl(ROOT / "metrics" / f"diagnostics_{run['session_id']}.jsonl")):
            if record.get("event") == "session_summary" and isinstance(record.get("latency_experiment_counters"), dict):
                counter_sessions[group_key][run["session_id"]] = {
                    k: v
                    for k, v in record["latency_experiment_counters"].items()
                    if isinstance(v, int) and not isinstance(v, bool) and v >= 0
                }
                break
        first_utterances = {}
        prior_emission = None
        partials = _jsonl(ROOT / "metrics" / f"partials_{run['session_id']}.jsonl")
        partials.sort(
            key=lambda p: _numeric(p.get("emitted_at_ms")) if _numeric(p.get("emitted_at_ms")) is not None else math.inf
        )
        for partial in partials:
            if partial.get("timing_schema_version") != 2:
                continue
            if (delay := _numeric(partial.get("captured_end_to_partial_ms"))) is not None:
                events["captured_end_to_partial_ms"].append(delay)
            uid = partial.get("utterance_id")
            delay = _numeric(partial.get("speech_start_to_partial_ms"))
            if uid is not None and delay is not None:
                first_utterances[uid] = min(first_utterances.get(uid, math.inf), delay)
            emitted = _numeric(partial.get("emitted_at_ms"))
            if emitted is not None:
                if prior_emission is not None:
                    events["partial_update_gap_ms"].append(emitted - prior_emission)
                prior_emission = emitted
        events["first_partial_ms"].extend(first_utterances.values())
        rows = []
        csv_path = ROOT / "metrics" / f"ab_metrics_{run['session_id']}.csv"
        if csv_path.exists():
            with csv_path.open() as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                # Separate source, endpoint reason and timing definition, never average percentiles.
                schema = row.get("timing_schema_version") or "legacy"
                key = (
                    *group_key,
                    schema,
                    row.get("endpoint_reason") or row.get("finalization_reason") or "unknown",
                )
                groups[key].append(row)
        browser = _report_browser_acknowledgments(
            rows,
            _jsonl(ROOT / "metrics" / f"display_metrics_{run['session_id']}.jsonl"),
            run["session_id"],
        )
        browser_groups[group_key].extend(browser["cohorts"])
        browser_client_counts[group_key] += browser["client_count"]
        unmatched_acks[group_key] += browser["unmatched_final_acks"]
        ack_coverage[group_key].append(browser["coverage"])
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
            dict(
                zip(
                    ("experiment", "size", "language", "provenance", "clip_id", "runtime_cohort", "schema", "endpoint"),
                    key,
                    strict=True,
                )
            )
            | {"metrics": fields}
        )
    report["source_cohorts"] = source_cohorts
    report["failures"] = failures
    report["excluded_runs"] = exclusions
    report["caption_events"] = [
        dict(zip(("experiment", "size", "language", "provenance", "clip_id", "runtime_cohort"), key, strict=True))
        | {
            "metrics": {name: stats(values) for name, values in metrics.items()},
            "browser_client_sessions": browser_client_counts[key],
            "browser_render_cohorts": browser_groups[key],
            "unmatched_final_acks": unmatched_acks[key],
            "latency_experiment_counters": {
                name: sum(row.get(name, 0) for row in counter_sessions[key].values())
                for name in sorted({name for row in counter_sessions[key].values() for name in row})
            },
            "counter_sessions": counter_sessions[key],
            "final_ack_coverage": {
                "received_final_chunks": sum(s["received_final_chunks"] for s in ack_coverage[key]),
                "finalized_chunks": sum(s["finalized_chunks"] for s in ack_coverage[key]),
                "sessions": ack_coverage[key],
            },
        }
        for key, metrics in event_groups.items()
    ]
    output.mkdir(parents=True, exist_ok=True)
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
        "Speech end is estimated from captured VAD-positive frames. Server-final latency ends at payload readiness; it is not browser display latency. Clips, inference code/config cohorts, source types and endpoints are kept separate. Cohort IDs bind recorded source hashes, package versions and effective settings. A matching lifecycle source hash observed at startup takes precedence over later pipeline-file snapshots; this does not prove imported engine bytecode, and older snapshot timing may be ambiguous.",
        "",
        "| Experiment | Model | Clip / cohort | Language/source | Endpoint | n | p50 ms | p95 ms |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for r in report["replays"]:
        s = r["metrics"]["speech_end_to_final_ms"]
        lines.append(
            f"| {r['experiment']} | {r['size']} | {r['clip_id']} / {r['runtime_cohort']} | {r['language']}/{r['provenance']} | {r['endpoint']} | {s['n']} | {fmt(s['p50'])} | {fmt(s['p95'])} |"
        )
    lines += [
        "",
        "## Partial delivery and visible browser timing",
        "",
        "First-partial delay starts at the first captured speech frame; one earliest delay is counted per known utterance. Update gaps use chronological emissions, including speaking pauses. These partial timings end at server readiness, before browser rendering.",
        "",
        "| Experiment | Model | Clip / cohort | Source | Metric | n | p50 ms | p95 ms |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in report["caption_events"]:
        for name, values in row["metrics"].items():
            lines.append(
                f"| {row['experiment']} | {row['size']} | {row['clip_id']} / {row['runtime_cohort']} | {row['language']}/{row['provenance']} | {name} | {values['n']} | {fmt(values['p50'])} | {fmt(values['p95'])} |"
            )
    lines += [
        "",
        "### Visible final latency by browser session and endpoint",
        "",
        "Only visible schema-2 final ACKs matched to a unique CSV chunk contribute. Each browser session, capture timing source and endpoint has its own distribution; clients and silence/forced-cut/EOF endpoints are never pooled. Duplicate chunk acknowledgments and unmatched/stale events are excluded. Speech-end-to-ack includes return-network time. Coverage alone does not make a run acceptance eligible; the latency gate remains pending.",
        "",
        "| Experiment | Model | Clip / cohort | Session / client | Endpoint / timing source | Metric | n | p50 ms | p95 ms | ACKs / finals |",
        "|---|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in report["caption_events"]:
        for browser in row["browser_render_cohorts"]:
            coverage = browser["final_ack_coverage"]
            for name, values in browser["metrics"].items():
                lines.append(
                    f"| {row['experiment']} | {row['size']} | {row['clip_id']} / {row['runtime_cohort']} | {browser['session_id']} / {browser['client_id']} | {browser['endpoint']} / {browser['timing_source']} | {name} | {values['n']} | {fmt(values['p50'])} | {fmt(values['p95'])} | {coverage['received_final_chunks']} / {coverage['finalized_chunks']} |"
                )
    lines += [
        "",
        "### Visible final acknowledgment coverage",
        "",
        "Coverage counts each finalized chunk once when any visible client acknowledged it. Missing acknowledgments are missing evidence, not proof of display failure; replay shutdown can race the final browser acknowledgment.",
        "",
        "| Experiment | Model | Clip / cohort | Received final chunks | Finalized chunks |",
        "|---|---|---|---:|---:|",
    ]
    for row in report["caption_events"]:
        coverage = row["final_ack_coverage"]
        lines.append(
            f"| {row['experiment']} | {row['size']} | {row['clip_id']} / {row['runtime_cohort']} | {coverage['received_final_chunks']} | {coverage['finalized_chunks']} |"
        )
    # Per-item output differences + deterministic, blinded review rows.
    by_model = {(r["size"], r["policy"], r.get("target_override"), _runtime_cohort(r)): r for r in quality}
    blind, key_rows = [], []
    lines += ["", "## Changed translation examples", ""]
    for identity, left in by_model.items():
        size, policy, target, cohort = identity
        right = by_model.get(("e2b", policy, target, cohort))
        if size != "e4b" or not right:
            continue
        right_rows = {r["id"]: r for r in right["rows"]}
        for a in left["rows"]:
            b = right_rows.get(a["id"])
            if not b:
                continue
            first, second = a["runs"][0]["text"], b["runs"][0]["text"]
            pair_id = f"{policy}_{target or 'bilingual'}_{cohort}_{a['id']}"
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
    blind = _preserve_blind_reviews(output / "blind_review.jsonl", blind)
    lines += [
        "## Pending gates",
        "",
        "- ≥50 approved natural-speech reference utterances per language.",
        "- Bilingual review of meaning errors and terminology preferences (blind_review.jsonl).",
        "- Visible-browser render acknowledgments for the sub-second caption-delivery gate.",
        "- Physical second-output / hotplug and real two-speaker validation.",
        "",
        f"Recorded failed runs: {len(failures)}. Excluded incompatible/duplicate runs: {len(exclusions)}. Details are retained in comparison.json.",
    ]
    # Human-review validation above must succeed before changing any artifact.
    encoded_rows = {
        name: "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records)
        for name, records in (("blind_review.jsonl", blind), ("review_key.jsonl", key_rows))
    }
    write_json(output / "comparison.json", report)
    for name, encoded in encoded_rows.items():
        _write_text(output / name, encoded)
    _write_text(output / "comparison.md", "\n".join(lines) + "\n")
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
    experiments.add_argument("--timeout-seconds", type=float, default=None)
    for name in ("quality", "_quality-worker", "replay", "report", "stt", "_stt-worker"):
        sub = subs.add_parser(name)
        sub.add_argument("--manifest", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        if name == "report":
            sub.add_argument("--input", type=Path, required=True)
        else:
            sub.add_argument("--runs", type=int, default=3)
            if name in ("quality", "stt", "replay"):
                sub.add_argument("--timeout-seconds", type=float, default=None if name == "replay" else 1800)
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
