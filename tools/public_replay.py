"""Assemble a declared development-only read-speech replay without changing references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.mac_evaluation import write_json
from tools.mac_followup_quality import digest, relative_file, validate_portable


def prepare(manifest_path: Path, audio_root: Path, output: Path, *, count: int = 5) -> dict:
    import numpy as np
    import soundfile as sf

    manifest = json.loads(manifest_path.read_text())
    validate_portable(manifest)
    if not 1 <= count <= 50:
        raise ValueError("Expected 1..50 development recordings per language")
    output.mkdir(parents=True, exist_ok=False)
    clips = []
    for lang in ("en", "es"):
        selected = [r for r in manifest["records"] if r["partition"] == "development" and r["source_lang"] == lang][
            :count
        ]
        arrays, spans, position = [], [], 0
        for item in selected:
            path = relative_file(audio_root, item["path"])
            if digest(path) != item["sha256"]:
                raise ValueError("Original audio hash mismatch")
            audio, rate = sf.read(path, dtype="float32")
            if rate != 16000 or audio.shape != (item["num_samples"],) or not np.isfinite(audio).all():
                raise ValueError("Invalid source recording")
            arrays.extend((audio, np.zeros(16000, dtype=np.float32)))
            spans.append(
                {
                    "id": item["id"],
                    "sample_start": position,
                    "sample_end": position + len(audio),
                    "original_audio_sha256": item["sha256"],
                    "reference": item["reference"],
                    "translation_reference": item["translation_reference"],
                    "padding_after_samples": 16000,
                }
            )
            position += len(audio) + 16000
        wav = output / f"fleurs_development_{lang}.wav"
        sf.write(wav, np.concatenate(arrays), 16000, subtype="FLOAT")
        clips.append(
            {
                "id": f"fleurs_development_{lang}",
                "lang": lang,
                "path": str(wav.resolve()),
                "sha256": digest(wav),
                "duration": position / 16000,
                "sample_rate": 16000,
                "provenance": "fleurs_development_read_speech_concatenation",
                "reference_status": "upstream_public_annotations_not_local_approval",
                "spans": spans,
            }
        )
    report = {
        "schema_version": 1,
        "manifest_sha256": digest(manifest_path),
        "clips": clips,
        "selection": "first declared development records in deterministic stratified manifest order",
        "changes": "float32 original samples concatenated with exactly 1s digital silence after each recording",
        "limitations": "Edited natural read speech, not a continuous service or speaker-transition gate",
        "attribution": "Google FLEURS and contributors, CC BY 4.0; source attribution travels in original manifest",
        "training_eligible": False,
        "confirmation_used": False,
    }
    write_json(output / "manifest.json", report, exclusive=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    prepare(args.manifest, args.audio_root, args.output, count=args.count)


if __name__ == "__main__":
    main()
