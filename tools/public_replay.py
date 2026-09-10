"""Assemble declared read-speech replays, retaining source hashes and explicit level changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.mac_evaluation import write_json
from tools.mac_followup_quality import digest, relative_file, validate_portable


def level_audio(audio, normalization):
    """Apply a declared fixed level rule; never clip or modify the source file."""
    import numpy as np

    if normalization not in {"none", "rms-0.08"}:
        raise ValueError("Unknown normalization")
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Empty/nonfinite audio")
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(audio)))
    gain = 1.0
    if normalization != "none":
        if rms == 0 or peak == 0:
            raise ValueError("Cannot level a silent recording")
        gain = min(0.08 / rms, 0.95 / peak)
    result = (audio * gain).astype(np.float32)
    return result, {
        "rule": normalization,
        "gain": gain,
        "original_rms": rms,
        "original_peak": peak,
        "output_rms": float(np.sqrt(np.mean(result.astype(np.float64) ** 2))),
        "output_peak": float(np.max(np.abs(result))),
        "clipping": False,
    }


def prepare(
    manifest_path: Path,
    audio_root: Path,
    output: Path,
    *,
    count: int = 5,
    partition: str = "development",
    normalization: str = "none",
) -> dict:
    import numpy as np
    import soundfile as sf

    manifest = json.loads(manifest_path.read_text())
    validate_portable(manifest)
    if not 1 <= count <= 50 or partition not in {"development", "confirmation"}:
        raise ValueError("Expected 1..50 recordings in a declared partition")
    if normalization not in {"none", "rms-0.08"}:
        raise ValueError("Unknown normalization")
    output.mkdir(parents=True, exist_ok=False)
    clips = []
    for lang in ("en", "es"):
        selected = [r for r in manifest["records"] if r["partition"] == partition and r["source_lang"] == lang][:count]
        if len(selected) != count:
            raise ValueError("Insufficient declared recordings")
        arrays, spans, position = [], [], 0
        for item in selected:
            path = relative_file(audio_root, item["path"])
            if digest(path) != item["sha256"]:
                raise ValueError("Original audio hash mismatch")
            audio, rate = sf.read(path, dtype="float32")
            if rate != 16000 or audio.shape != (item["num_samples"],) or not np.isfinite(audio).all():
                raise ValueError("Invalid source recording")
            audio, level = level_audio(audio, normalization)
            arrays.extend((audio, np.zeros(16000, dtype=np.float32)))
            spans.append(
                {
                    "id": item["id"],
                    "sample_start": position,
                    "sample_end": position + len(audio),
                    "original_audio_sha256": item["sha256"],
                    "level_transform": level,
                    "reference": item["reference"],
                    "translation_reference": item["translation_reference"],
                    "padding_after_samples": 16000,
                }
            )
            position += len(audio) + 16000
        wav = output / f"fleurs_{partition}_{lang}.wav"
        sf.write(wav, np.concatenate(arrays), 16000, subtype="FLOAT")
        clips.append(
            {
                "id": f"fleurs_{partition}_{lang}",
                "lang": lang,
                "path": str(wav.resolve()),
                "sha256": digest(wav),
                "duration": position / 16000,
                "sample_rate": 16000,
                "provenance": f"fleurs_{partition}_read_speech_concatenation",
                "level_normalization": normalization,
                "reference_status": "upstream_public_annotations_not_local_approval",
                "spans": spans,
            }
        )
    report = {
        "schema_version": 1,
        "manifest_sha256": digest(manifest_path),
        "clips": clips,
        "selection": f"first declared {partition} records in deterministic stratified manifest order",
        "level_normalization": normalization,
        "changes": "float32 samples, explicitly declared per-record level transform, exactly 1s digital silence after each recording",
        "limitations": "Edited natural read speech, not a continuous service or speaker-transition gate",
        "attribution": "Google FLEURS and contributors, CC BY 4.0; source attribution travels in original manifest",
        "training_eligible": False,
        "confirmation_used": partition == "confirmation",
    }
    write_json(output / "manifest.json", report, exclusive=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--partition", choices=["development", "confirmation"], default="development")
    parser.add_argument("--normalization", choices=["none", "rms-0.08"], default="none")
    args = parser.parse_args()
    prepare(
        args.manifest,
        args.audio_root,
        args.output,
        count=args.count,
        partition=args.partition,
        normalization=args.normalization,
    )


if __name__ == "__main__":
    main()
