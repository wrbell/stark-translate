"""Generate a silent-to-device Piper fixture for the #133 latency half-gate.

Synthetic turns are not natural two-speaker label acceptance evidence. Importing
this module does not load voices; only explicitly invoking main does so.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import wave
from pathlib import Path

import numpy as np

from engines.tts_engine import PiperTTSEngine

SAMPLE_RATE = 16000
SOURCE_MANIFEST = Path(__file__).resolve().parents[1] / "stark_data/replay/manifest.json"
CHURCH_CLIP = "Gospel_Message_(12_14_25)_5D2rOMvkwrk.wav"
SENTENCES = (
    "Good morning, and welcome to our gospel meeting this Lord's day.",
    "Thank you, brother. Let us open our Bibles to the gospel of John, chapter three.",
    "Jesus said to Nicodemus that a man must be born again to see the kingdom of God.",
    "That is a hard saying, and Nicodemus asked how such a thing could be.",
    "The answer is not religion or good works, but the grace of God through faith.",
    "For God so loved the world that he gave his only begotten Son.",
    "Whoever believes in him shall not perish but have everlasting life.",
    "That promise is for every person in this room this morning.",
    "We will sing hymn forty-two before the closing prayer.",
    "Please remain seated while the young people come forward.",
    "Let us give thanks for the message and for the fellowship we share.",
    "Go in peace, and may the Lord bless you and keep you this week.",
)


def resample_mono(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Downmix frames/channels and resample, without any playback dependency."""
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
        raise ValueError("TTS must return nonempty finite mono or frames/channels audio")
    if sample_rate <= 0:
        raise ValueError("TTS sample rate must be positive")
    if sample_rate == SAMPLE_RATE:
        return samples
    try:
        from scipy.signal import resample_poly
    except ImportError:
        count = math.ceil(len(samples) * SAMPLE_RATE / sample_rate)
        return np.interp(np.arange(count) * sample_rate / SAMPLE_RATE, np.arange(len(samples)), samples).astype(
            np.float32
        )
    divisor = math.gcd(sample_rate, SAMPLE_RATE)
    return resample_poly(samples, SAMPLE_RATE // divisor, sample_rate // divisor).astype(np.float32)


def nonnegative_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("seconds must be finite and nonnegative")
    return seconds


def church_manifest_entry(source_manifest: Path, destination: Path) -> tuple[dict, Path, Path]:
    """Read the existing entry unchanged and validate its portable WAV location."""
    if destination.resolve() == source_manifest.resolve():
        raise ValueError("--emit-manifest must not overwrite the source manifest")
    clips = json.loads(source_manifest.read_text())["clips"]
    matches = [clip for clip in clips if clip["path"] == CHURCH_CLIP]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {CHURCH_CLIP} entry in {source_manifest}")
    entry = matches[0]
    source = source_manifest.parent / entry["path"]
    target = destination.parent / entry["path"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError(f"Church clip checksum mismatch: {source}")
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError(f"Refusing to overwrite a different church clip: {target}")
    return entry, source, target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--voice-a", default="en_US-lessac-high")
    parser.add_argument("--voice-b", default="es_MX-claude-high")
    parser.add_argument("--gap-s", type=nonnegative_seconds, default=1.2)
    parser.add_argument("--lead-s", type=nonnegative_seconds, default=0.5)
    parser.add_argument(
        "--seed", type=int, default=0, help="Python/NumPy seed; Piper/ONNX determinism is not guaranteed"
    )
    parser.add_argument("--emit-manifest", type=Path)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST, help="Existing church replay manifest")
    args = parser.parse_args(argv)
    paths = [args.output, args.recipe]
    church = None
    if args.emit_manifest:
        paths.append(args.emit_manifest)
        try:
            church = church_manifest_entry(args.source_manifest, args.emit_manifest)
        except (OSError, ValueError, KeyError) as exc:
            parser.error(str(exc))
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        parser.error("output, recipe and manifest must be different files")
    protected = {args.source_manifest.resolve()}
    if church:
        protected.update(path.resolve() for path in church[1:])
    if protected.intersection(resolved):
        parser.error("outputs must not overwrite the source manifest or church clip")

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    voices = {"a": args.voice_a, "b": args.voice_b}
    engine = PiperTTSEngine(voices=voices)
    lead_samples = round(args.lead_s * SAMPLE_RATE)
    gap_samples = round(args.gap_s * SAMPLE_RATE)
    parts = [np.zeros(lead_samples, dtype=np.float32)]
    position = lead_samples
    segments = []
    try:
        engine.load()
        for index, sentence in enumerate(SENTENCES, start=1):
            voice = "a" if index % 2 else "b"
            result = engine.synthesize(sentence, language=voice)
            samples = resample_mono(result.audio, result.sample_rate)
            segments.append(
                {
                    "index": index,
                    "voice": voice,
                    "text": sentence,
                    "start_s": position / SAMPLE_RATE,
                    "end_s": (position + len(samples)) / SAMPLE_RATE,
                }
            )
            parts.append(samples)
            position += len(samples)
            if index < len(SENTENCES):
                parts.append(np.zeros(gap_samples, dtype=np.float32))
                position += gap_samples
    finally:
        engine.unload()
    audio = np.concatenate(parts)
    peak = float(np.max(np.abs(audio)))
    if peak == 0:
        raise ValueError("TTS returned only silence")
    audio *= 0.9 / peak
    pcm = np.rint(audio * 32767).astype("<i2")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    recipe = {
        "output": str(args.output.resolve()),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "duration_s": len(pcm) / SAMPLE_RATE,
        "sample_rate": SAMPLE_RATE,
        "voices": voices,
        "seed": args.seed,
        "seed_caveat": "Python/NumPy seeded; Piper/ONNX synthesis may not be bitwise reproducible",
        "accent_caveat": f"voice b is {args.voice_b} reading English; natural two-speaker labels remain pending",
        "segments": segments,
    }
    args.recipe.parent.mkdir(parents=True, exist_ok=True)
    args.recipe.write_text(json.dumps(recipe, indent=2) + "\n")
    if church:
        entry, source, target = church
        args.emit_manifest.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        synthetic = {
            "path": os.path.relpath(args.output.resolve(), args.emit_manifest.parent.resolve()),
            "sha256": recipe["sha256"],
            "duration": recipe["duration_s"],
            "lang": "en",
            "offset_s": 0,
            "rms": float(np.sqrt(np.mean((pcm.astype(np.float64) / 32768) ** 2))),
        }
        args.emit_manifest.write_text(json.dumps({"clips": [synthetic, entry]}, indent=2) + "\n")
    print(
        f"Wrote {args.output}: 12 synthetic turns, {recipe['duration_s']:.3f}s, 16000 Hz mono PCM16; recipe {args.recipe}"
    )


if __name__ == "__main__":
    main()
