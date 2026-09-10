"""Reserved-window device probe; this file has not itself certified any route.

Calls production Piper synthesis/playback and OutputDeviceResolver. The extra
sounddevice wait observes native stream completion, not acoustic onset/hearing.
Generated speech is synthetic test material and is never training eligible.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
import wave
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

TEXTS = {
    "en": "Hello friends. This is an English microphone routing test. We are checking live captions on this computer.",
    "es": "Hola amigos. Esta es una prueba del micrófono en español. Estamos comprobando los subtítulos en esta computadora.",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def playback(engine, resolver, audio, rate, language, spec):
    import sounddevice as sd

    expected = resolver.resolve(spec)
    attempts = []

    def observed_play(samples, sample_rate, *, device=None):
        attempt = {"requested_device": device, "started_at": datetime.now(UTC).isoformat()}
        attempts.append(attempt)
        try:
            previous = sd.get_stream()
        except RuntimeError:
            previous = None
        began = time.perf_counter()
        try:
            engine.play(samples, sample_rate, device=device)
            opened = time.perf_counter()
            stream = sd.get_stream()
            if stream is previous:
                raise RuntimeError("Playback did not create a new native stream")
            attempt.update(
                actual_device=stream.device,
                samplerate=stream.samplerate,
                channels=stream.channels,
                reported_latency_s=stream.latency,
                active_after_open=stream.active,
                host_open_call_ms=(opened - began) * 1000,
            )
            status = sd.wait(ignore_errors=False)
            attempt.update(
                wait_completed=True,
                host_completion_ms=(time.perf_counter() - began) * 1000,
                active_after_wait=stream.active,
                stopped_after_wait=stream.stopped,
                closed_after_wait=stream.closed,
                callback_status=str(status),
                callback_error=bool(status),
            )
        except Exception as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            raise

    resolver.play(observed_play, audio, rate, language=language, spec=spec)
    passed = bool(attempts) and all(
        row.get("actual_device") == expected
        and row.get("requested_device") == expected
        and row.get("wait_completed")
        and not row.get("active_after_wait")
        and not row.get("callback_error")
        and not row.get("error")
        for row in attempts
    )
    return {"spec": spec, "resolved_device": expected, "attempts": attempts, "passed": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-playback", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-en", default="MacBook Pro Speakers")
    parser.add_argument("--device-es", default="Microsoft Teams Audio")
    parser.add_argument("--play-existing", type=Path, help="Existing probe receipt: replay its synthetic WAV only")
    parser.add_argument("--language", choices=("en", "es"), help="Required with --play-existing")
    args = parser.parse_args()
    if args.play_existing and not args.language:
        parser.error("--play-existing requires --language")
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / "receipt.json"
    report = {
        "schema_version": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "completed": False,
        "rows": [],
        "synthetic": True,
        "human_reviewed": False,
        "training_eligible": False,
        "scope": "production Piper/route host calls and native completion; no human-heard/physical second-output claim",
        "python": sys.executable,
        "probe_source_sha256": digest(Path(__file__)),
        "versions": {},
        "source_sha256": {
            name: digest(ROOT / name)
            for name in (
                "engines/tts_engine.py",
                "engines/audio_devices.py",
                "engines/model_paths.py",
                "models.lock.json",
            )
        },
    }

    def save():
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    save()
    engine = None
    try:
        for name in ("piper-tts", "onnxruntime", "numpy", "sounddevice"):
            report["versions"][name] = importlib.metadata.version(name)
        import numpy as np
        import sounddevice as sd

        from engines.audio_devices import OutputDeviceResolver, list_output_devices
        from engines.model_paths import resolve_piper_voice
        from engines.tts_engine import PiperTTSEngine

        report["devices"] = list_output_devices()
        resolver = OutputDeviceResolver()
        routes = {"en": args.device_en, "es": args.device_es}
        languages = [args.language] if args.play_existing else ["en", "es"]
        for language in languages:
            resolver.resolve(routes[language])  # missing route fails before synthesis/playback
        if args.play_existing:
            prior = json.loads(args.play_existing.read_text())
            if prior.get("synthetic") is not True or prior.get("training_eligible") is not False:
                raise ValueError("Expected explicitly synthetic, ineligible source receipt")
            source = next(row for row in prior["rows"] if row["language"] == args.language)
            path = Path(source["audio_path"]).resolve()
            if not path.is_relative_to(args.play_existing.resolve().parent) or digest(path) != source["audio_sha256"]:
                raise ValueError("Synthetic WAV escaped its receipt or changed")
            engine = PiperTTSEngine()  # play() needs no loaded voice/model
            report["synthetic_source_receipt_sha256"] = digest(args.play_existing)
        else:
            voices = {"en": "en_US-lessac-high", "es": "es_MX-claude-high"}
            resolved = {language: resolve_piper_voice(name) for language, name in voices.items()}
            if not all(resolved.values()):
                raise FileNotFoundError("Both existing local Piper voices are required; no download in this probe")
            report["voices"] = {
                lang: {
                    "name": voices[lang],
                    "path": path,
                    "sha256": digest(Path(path)),
                    "config_sha256": digest(Path(path + ".json")),
                }
                for lang, path in resolved.items()
            }
            engine = PiperTTSEngine(voices=resolved)
            engine.load()
        for language in languages:
            if args.play_existing:
                with wave.open(str(path), "rb") as wav:
                    if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                        raise ValueError("Expected generated mono PCM16 WAV")
                    rate = wav.getframerate()
                    audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32) / 32768
                row = {**source, "synthesis_this_process": False}
            else:
                result = engine.synthesize(TEXTS[language], language=language)
                audio, rate = result.audio, result.sample_rate
                row = {
                    "language": language,
                    "text": TEXTS[language],
                    "synthesis_ms": result.latency_ms,
                    "synthesis_this_process": True,
                }
            duration = len(audio) / rate
            if not 0 < duration <= 15 or not np.isfinite(audio).all():
                raise ValueError("Synthetic clip must be finite and no longer than 15 seconds")
            if not args.play_existing:
                path = (args.output / f"synthetic_{language}.wav").resolve()
                with wave.open(str(path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(rate)
                    wav.writeframes(np.clip(audio * 32768, -32768, 32767).astype(np.int16).tobytes())
                row.update(audio_path=str(path), audio_sha256=digest(path))
            row.update(duration_s=duration, frames=len(audio), sample_rate=rate, status="playback_pending")
            report["rows"].append(row)
            save()
            row["playback"] = playback(engine, resolver, audio, rate, language, routes[language])
            row["status"] = "passed" if row["playback"]["passed"] else "failed"
            save()
        report["completed"] = all(row["status"] == "passed" for row in report["rows"])
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if engine is not None:
            import sounddevice as sd

            sd.stop()
            engine.unload()
        report["ended_at"] = datetime.now(UTC).isoformat()
        save()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
