"""Offline church audio → Parakeet English → shared Gemma Hindi baseline.

Separate transcription and translation commands keep both models' inputs
identical. Outputs are evaluation-only, with new directories required for every
attempt. This command never starts the live pipeline, TTS, or a training export.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import re
import sys
import wave
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mac_evaluation import digest, environment, stats, write_json

SEGMENTATION = {
    "sampling_rate": 16000,
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "max_speech_duration_s": 8.0,
    "min_silence_duration_ms": 500,
    "speech_pad_ms": 30,
    "return_seconds": False,
}


def validate_manifest(manifest: dict, data_root: Path | None = None) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("usage") != "evaluation_only":
        raise ValueError("A schema-1 evaluation-only Hindi manifest is required")
    if manifest.get("source_lang") != "en" or manifest.get("target_lang") != "hi":
        raise ValueError("This offline baseline only supports English source and Hindi target")
    ids = set()
    for item in [*manifest.get("audio_sources", []), *manifest.get("text_probes", [])]:
        if not isinstance(item.get("id"), str) or not item["id"] or item["id"] in ids:
            raise ValueError("Every audio source and text probe needs a unique ID")
        ids.add(item["id"])
        reference = item.get("reference_hi")
        if reference is not None and (
            not isinstance(reference, str) or not reference.strip() or not item.get("reference_provenance")
        ):
            raise ValueError("Hindi references require nonempty text and explicit reference provenance")
    if not manifest.get("audio_sources"):
        raise ValueError("At least one actual church audio source is required")
    for source in manifest["audio_sources"]:
        if source.get("provenance") != "church_replay":
            raise ValueError("Audio sources must declare church_replay provenance")
        path = Path(source["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Audio paths must be relative to the explicit data root")
        if not re.fullmatch(r"[0-9a-f]{64}", source.get("sha256", "")):
            raise ValueError("Audio sources require a SHA-256")
        duration = source.get("duration")
        if isinstance(duration, bool) or not isinstance(duration, (float, int)) or not 0 < duration <= 300:
            raise ValueError("Audio source duration must be between zero and 300 seconds")
        if data_root is not None:
            path = (data_root / path).resolve()
            if not path.is_relative_to(data_root.resolve()) or digest(path) != source["sha256"]:
                raise ValueError(f"Audio source is missing, changed, or outside data root: {source['id']}")
    for item in manifest.get("text_probes", []):
        if not isinstance(item.get("source"), str) or not item["source"].strip():
            raise ValueError("Text probes require an English source transcript")


def prepare_manifest(data_root: Path, base_manifest: Path, output: Path) -> dict:
    """Freeze existing source hashes and eight canaries; never invent references."""
    from training.theological_canaries import canary_sentences

    source = json.loads(base_manifest.read_text())
    december, february = [row for row in source["replays"] if row["lang"] == "en"][:2]
    short = data_root / "stark_data/replay/mac_screening_45s.wav"
    if not short.is_file():
        raise FileNotFoundError("Existing December 45s screening clip is required; no audio is generated here")
    manifest = {
        "schema_version": 1,
        "id": "offline_hindi_church_audio_v1",
        "usage": "evaluation_only",
        "source_lang": "en",
        "target_lang": "hi",
        "base_manifest_sha256": digest(base_manifest),
        "reference_status": "Hindi references unavailable in local corpus; bilingual review pending",
        "audio_sources": [
            {
                **december,
                "id": "december_45s",
                "path": str(short.relative_to(data_root)),
                "sha256": digest(short),
                "duration": 45.0,
                "parent_audio_sha256": december["sha256"],
                "reference_hi": None,
                "reference_provenance": None,
            },
            {**february, "id": "february_150s", "reference_hi": None, "reference_provenance": None},
        ],
        "text_probes": [
            {
                "id": f"canary_{index + 1}",
                "kind": "theological_text_probe",
                "source": row["en"],
                "source_provenance": "training/theological_canaries.py",
                "reference_hi": None,
                "reference_provenance": None,
                "required_hindi_terms": None,
            }
            for index, row in enumerate(canary_sentences(8))
        ],
    }
    validate_manifest(manifest, data_root)
    write_json(output, manifest, exclusive=True)
    return manifest


def load_audio(path: Path, expected_hash: str):
    """Decode immutable PCM16 bytes and resample once, before segmentation."""
    import numpy as np

    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != expected_hash:
        raise ValueError(f"Audio changed before decoding: {path}")
    with wave.open(io.BytesIO(encoded)) as wav:
        if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError("The baseline requires uncompressed PCM16 WAV sources")
        rate, channels, frames = wav.getframerate(), wav.getnchannels(), wav.getnframes()
        pcm = wav.readframes(frames)
    if len(pcm) != frames * channels * 2:
        raise ValueError("Truncated source WAV")
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32).reshape(-1, channels).mean(axis=1) / 32768
    if rate != 16000:
        from scipy.signal import resample_poly

        common = math.gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // common, rate // common).astype(np.float32)
    return audio, {"sample_rate": rate, "channels": channels, "frames": frames}


def segment_audio(audio, model, timestamp_function):
    import torch

    model.reset_states()
    boundaries = timestamp_function(torch.from_numpy(audio), model, **SEGMENTATION)
    previous_end = 0
    for boundary in boundaries:
        start, end = int(boundary["start"]), int(boundary["end"])
        if not previous_end <= start < end <= len(audio):
            raise ValueError("VAD emitted overlapping or out-of-range sample bounds")
        previous_end = end
        yield start, end


def baseline_environment() -> dict:
    result = environment()
    for name in ("tools/offline_hindi.py", "tools/vad_runtime.py", "engines/parakeet_mlx_engine.py"):
        result["source_sha256"][name] = digest(ROOT / name)
    for name in ("silero-vad", "torch", "numpy", "scipy"):
        try:
            result["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["versions"][name] = None
    return result


def configure_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if os.environ.get("STARK_EXPERIMENT_GEMMA_PREFIX_CACHE", "false").lower() not in {"0", "false"}:
        raise ValueError("Unset STARK_EXPERIMENT_GEMMA_PREFIX_CACHE for the zero-shot baseline")
    if os.environ.get("STARK_EXPERIMENT_MLX_CACHE_MB", "256") != "256":
        raise ValueError("Unset STARK_EXPERIMENT_MLX_CACHE_MB for the zero-shot baseline")


def reserve_run(output: Path, manifest_path: Path, stage: str) -> tuple[dict, dict]:
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", manifest, exclusive=True)
    metadata = {
        "schema_version": 1,
        "usage": "evaluation_only",
        "stage": stage,
        "status": "running",
        "manifest_sha256": digest(manifest_path),
        "environment": baseline_environment(),
        "human_review": "pending",
        "quality_gate": "pending_references_and_bilingual_review",
        "live_readiness": "not_assessed",
    }
    write_json(output / "metadata.json", metadata)
    return manifest, metadata


def append_row(output: Path, row: dict) -> None:
    with (output / "rows.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def finish_run(output: Path, metadata: dict, engine, model_ids: dict) -> None:
    """An unload failure cannot leave successful completion evidence behind."""
    try:
        if engine is not None:
            engine.unload()
    except BaseException as exc:
        metadata.update(status="failed", cleanup_error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        from tools.session_lifecycle import completion_metadata

        metadata["completion"] = completion_metadata(model_ids, ROOT)
        rows = output / "rows.jsonl"
        metadata["rows_sha256"] = digest(rows) if rows.is_file() else None
        write_json(output / "metadata.json", metadata)


def transcribe(args) -> None:
    configure_offline()
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, args.data_root)
    manifest, metadata = reserve_run(args.output, args.manifest, "english_transcription")
    engine = None
    model_id = "mlx-community/parakeet-tdt-0.6b-v3"
    try:
        from engines.model_paths import resolve_model_path
        from engines.parakeet_mlx_engine import ParakeetMLXEngine
        from tools.vad_runtime import load_packaged_vad

        resolved = resolve_model_path(model_id, local_only=True)
        if resolved is None:
            raise FileNotFoundError("Cached Parakeet model is required; run the Mac setup first")
        vad, utilities, artifact = load_packaged_vad("torch")
        metadata.update(
            stt_model_id=model_id,
            resolved_model=resolved,
            vad=artifact,
            segmentation=SEGMENTATION,
            segmentation_semantics="offline Silero segmentation, not live endpoint timing",
            stt_settings={
                "language": "en",
                "language_forcing_supported": False,
                "initial_prompt": None,
                "word_timestamps": True,
                "corrections": "off",
                "dtype": "bfloat16",
                "cache_limit_mb": 256,
                "warmup_seconds": 1.0,
            },
        )
        write_json(args.output / "metadata.json", metadata)
        engine = ParakeetMLXEngine(model_id=resolved)
        engine.load()  # load, inference and unload all stay on this thread
        counts = {}
        for source in manifest["audio_sources"]:
            audio, original = load_audio(args.data_root / source["path"], source["sha256"])
            if abs(len(audio) / 16000 - source["duration"]) > 0.01:
                raise ValueError(f"Source duration differs from manifest: {source['id']}")
            counts[source["id"]] = 0
            for index, (start, end) in enumerate(segment_audio(audio, vad, utilities[0])):
                segment = audio[start:end]
                result = engine.transcribe(segment, language="en", word_timestamps=True)
                row = {
                    "id": f"{source['id']}:utterance:{index + 1}",
                    "kind": "church_audio_utterance",
                    "usage": "evaluation_only",
                    "audio_source_id": source["id"],
                    "audio_path": source["path"],
                    "audio_sha256": source["sha256"],
                    "original_audio": original,
                    "sample_start": start,
                    "sample_end": end,
                    "sample_rate": 16000,
                    "sample_coordinates": "source file after mono float32 resampling; half-open",
                    "sermon_offset_s": source.get("offset_s"),
                    "segment_float32_sha256": hashlib.sha256(segment.astype("<f4").tobytes()).hexdigest(),
                    "source": result.text,
                    "stt": asdict(result),
                    "reference_hi": None,
                    "reference_provenance": None,
                    "human_review": "pending",
                }
                append_row(args.output, row)
                counts[source["id"]] += 1
                print(f"STT {row['id']}: {result.text}", flush=True)
        if any(count == 0 for count in counts.values()):
            raise ValueError(f"VAD found no utterances in at least one source: {counts}")
        metadata.update(status="completed", utterances_by_source=counts)
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        finish_run(args.output, metadata, engine, {"stt": model_id})


def load_transcripts(directory: Path, manifest_path: Path) -> list[dict]:
    metadata = json.loads((directory / "metadata.json").read_text())
    rows_path = directory / "rows.jsonl"
    if (
        metadata.get("status") != "completed"
        or metadata.get("stage") != "english_transcription"
        or metadata.get("usage") != "evaluation_only"
        or metadata.get("manifest_sha256") != digest(manifest_path)
        or not rows_path.is_file()
        or metadata.get("rows_sha256") != digest(rows_path)
    ):
        raise ValueError("Transcription must be completed and unchanged for this exact manifest")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate transcription IDs")
    manifest = json.loads(manifest_path.read_text())
    sources = {source["id"]: source for source in manifest["audio_sources"]}
    counts = {identity: 0 for identity in sources}
    for row in rows:
        source = sources.get(row.get("audio_source_id"))
        if (
            source is None
            or row.get("audio_sha256") != source["sha256"]
            or row.get("audio_path") != source["path"]
            or row.get("sample_rate") != 16000
            or not 0 <= row["sample_start"] < row["sample_end"] <= round(source["duration"] * 16000)
            or row.get("usage") != "evaluation_only"
        ):
            raise ValueError("Transcription source provenance or sample bounds differ from the manifest")
        counts[source["id"]] += 1
    if counts != metadata.get("utterances_by_source") or any(count == 0 for count in counts.values()):
        raise ValueError("Transcription coverage is incomplete")
    return rows


def translation_observation(result, source: str) -> dict:
    from engines.translation_prompts import dynamic_max_tokens

    budget = dynamic_max_tokens(source)
    return {
        **asdict(result),
        "requested_max_tokens": budget,
        "budget_exhausted": result.generated_tokens >= budget if result.generated_tokens is not None else None,
        "output_nonempty": bool(result.text.strip()),
        "has_devanagari": bool(re.search(r"[\u0900-\u097f]", result.text)),
        "latin_fragments": re.findall(r"[A-Za-z]+", result.text),
        "script_checks_are_quality_scores": False,
    }


def translate(args) -> None:
    configure_offline()
    inputs = load_transcripts(args.transcripts, args.manifest)
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    inputs += manifest["text_probes"]
    if len({row["id"] for row in inputs}) != len(inputs):
        raise ValueError("Transcription and text probe IDs collide")
    manifest, metadata = reserve_run(args.output, args.manifest, "hindi_translation")
    engine = None
    model_id = None
    try:
        from engines.mlx_engine import MLXGemmaEngine
        from engines.model_paths import resolve_model_path
        from settings import settings

        model_id = getattr(settings.translation, f"mlx_model_gemma4_{args.size}")
        resolved = resolve_model_path(model_id, local_only=True)
        if resolved is None:
            raise FileNotFoundError(f"Cached {args.size} model is required; run the Mac setup first")
        metadata.update(
            size=args.size,
            model_id=model_id,
            resolved_model=resolved,
            transcripts_sha256=digest(args.transcripts / "rows.jsonl"),
            transcription_metadata_sha256=digest(args.transcripts / "metadata.json"),
            runs=args.runs,
            settings={
                "source_lang": "en",
                "target_lang": "hi",
                "model_family": "gemma4",
                "terminology_prompt": "none",
                "enable_thinking": False,
                "adapter": None,
                "draft_model": None,
                "prefix_cache": False,
                "allocation_cache_mb": 256,
                "max_tokens": "shared dynamic_max_tokens(source)",
                "execution": "serial offline translation; no real-time caption latency",
            },
        )
        write_json(args.output / "metadata.json", metadata)
        engine = MLXGemmaEngine(model_id=resolved, model_family="gemma4", use_prompt_cache=False)
        engine.load()
        engine.translate("The grace of God is sufficient.", source_lang="en", target_lang="hi")
        for item in inputs:
            for run in range(args.runs):
                result = (
                    engine.translate(item["source"], source_lang="en", target_lang="hi")
                    if item["source"].strip()
                    else None
                )
                append_row(
                    args.output,
                    {
                        **item,
                        "usage": "evaluation_only",
                        "run": run,
                        "source_lang": "en",
                        "target_lang": "hi",
                        "observation": translation_observation(result, item["source"]) if result is not None else None,
                        "skip_reason": "empty_stt" if result is None else None,
                        "human_review": "pending",
                    },
                )
                print(f"{args.size} {item['id']} repeat {run}: {result.text if result else '[empty STT]'}", flush=True)
        metadata.update(status="completed", inputs=len(inputs))
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        finish_run(args.output, metadata, engine, {"translation": model_id})


def report(args) -> dict:
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    transcripts = load_transcripts(args.transcripts, args.manifest)
    expected = {row["id"]: row for row in [*transcripts, *manifest["text_probes"]]}
    if len(expected) != len(transcripts) + len(manifest["text_probes"]):
        raise ValueError("Transcription and text probe IDs collide")
    summary: dict = {
        "usage": "evaluation_only",
        "manifest_sha256": digest(args.manifest),
        "transcripts_sha256": digest(args.transcripts / "rows.jsonl"),
        "quality_gate": "pending_references_and_bilingual_review",
        "live_readiness": "not_assessed",
        "qlora_decision": "not_assessed",
        "audio_sources": len(manifest["audio_sources"]),
        "audio_utterances": len(transcripts),
        "text_probes": len(manifest["text_probes"]),
        "models": [],
    }
    for directory in args.translations:
        metadata = json.loads((directory / "metadata.json").read_text())
        if (
            metadata.get("stage") != "hindi_translation"
            or metadata.get("status") != "completed"
            or metadata.get("usage") != "evaluation_only"
            or metadata.get("manifest_sha256") != summary["manifest_sha256"]
            or metadata.get("transcripts_sha256") != summary["transcripts_sha256"]
            or metadata.get("rows_sha256") != digest(directory / "rows.jsonl")
        ):
            raise ValueError("Report inputs must be completed, unchanged, and use identical STT inputs")
        rows = [json.loads(line) for line in (directory / "rows.jsonl").read_text().splitlines() if line.strip()]
        wanted = {(identity, run) for identity in expected for run in range(metadata["runs"])}
        if len(rows) != len(wanted) or {(row["id"], row["run"]) for row in rows} != wanted:
            raise ValueError("Translation input coverage is incomplete or duplicated")
        if any(row["source"] != expected[row["id"]]["source"] for row in rows):
            raise ValueError("Translation source texts differ from the frozen STT/text inputs")
        observations = [row["observation"] for row in rows if row["observation"] is not None]
        summary["models"].append(
            {
                "size": metadata["size"],
                "model_id": metadata["model_id"],
                "directory": str(directory.resolve()),
                "metadata_sha256": digest(directory / "metadata.json"),
                "rows_sha256": metadata["rows_sha256"],
                "expected_observations": len(wanted),
                "generated_observations": len(observations),
                "empty_stt_skips": len(rows) - len(observations),
                "empty_translations": sum(not row["output_nonempty"] for row in observations),
                "budget_exhausted": sum(row["budget_exhausted"] is True for row in observations),
                "budget_status_unknown": sum(row["budget_exhausted"] is None for row in observations),
                "with_devanagari": sum(row["has_devanagari"] for row in observations),
                "with_latin_fragments": sum(bool(row["latin_fragments"]) for row in observations),
                "translation_latency_ms": stats([row["latency_ms"] for row in observations]),
                "translation_latency_by_input_kind_ms": {
                    kind: stats(
                        [
                            row["observation"]["latency_ms"]
                            for row in rows
                            if row["kind"] == kind and row["observation"] is not None
                        ]
                    )
                    for kind in sorted({row["kind"] for row in rows})
                },
                "references_available": sum(bool(row.get("reference_hi")) for row in expected.values()),
                "human_review": "pending",
            }
        )
    write_json(args.output, summary, exclusive=True)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Freeze existing local inputs without loading models")
    prepare.add_argument("--data-root", type=Path, default=Path.cwd())
    prepare.add_argument("--base-manifest", type=Path, default=ROOT / "docs/evaluation/mac_v2026_14_manifest_v2.json")
    prepare.add_argument("--output", type=Path, required=True)
    for name in ("transcribe", "translate", "report"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, default=ROOT / "tools/offline_hindi_manifest.json")
        command.add_argument("--output", type=Path, required=True)
        if name == "transcribe":
            command.add_argument("--data-root", type=Path, default=Path.cwd())
        else:
            command.add_argument("--transcripts", type=Path, required=True)
        if name == "translate":
            command.add_argument("--size", choices=("e4b", "e2b"), required=True)
            command.add_argument("--runs", type=int, choices=range(1, 4), default=1)
        if name == "report":
            command.add_argument("--translations", type=Path, nargs="+", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_manifest(args.data_root, args.base_manifest, args.output)
    else:
        globals()[args.command](args)


if __name__ == "__main__":
    main()
