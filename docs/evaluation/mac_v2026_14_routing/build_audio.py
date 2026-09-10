"""Build explicitly synthetic audio for exercising existing routing decisions."""

import hashlib
import json
import math
import wave
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from engines.factory import create_tts_engine
from settings import settings

phrases = {
    "en": ["Would you please take your seats.", "Please turn to the next page.", "The window is open."],
    "es": ["Por favor tomen asiento.", "Pueden sentarse.", "La ventana está abierta."],
}
rate = 16000
voices = {lang: settings.tts.voices[lang] for lang in phrases}
engine = create_tts_engine(voices=voices)
engine.load()
replays = []
try:
    for lang, texts in phrases.items():
        pieces = [np.zeros(rate, dtype=np.int16)]
        recipe = []
        for index, text in enumerate(texts):
            result = engine.synthesize(text, language=lang)
            common = math.gcd(rate, result.sample_rate)
            audio = resample_poly(result.audio, rate // common, result.sample_rate // common)
            start = sum(len(part) for part in pieces) / rate
            pieces.extend([(np.clip(audio, -1, 1) * 32767).astype(np.int16), np.zeros(int(rate * 1.5), dtype=np.int16)])
            recipe.append(
                {
                    "text": text,
                    "start_s": start,
                    "duration_s": len(audio) / rate,
                    "conservative_allowlist_member": index < 2,
                    "route_requires_observed_confidence_at_least": 0.8,
                }
            )
        pcm = np.concatenate(pieces)
        path = Path(f"stark_data/replay/mac_routing_synthetic_{lang}.wav")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(pcm.tobytes())
        replays.append(
            {
                "id": f"routing_synthetic_{lang}",
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "duration": len(pcm) / rate,
                "lang": lang,
                "provenance": "synthetic_piper",
                "reference_status": "synthetic_script_not_human_reference",
                "voice": voices[lang],
                "conditions": ["operational_phrases", "explicit_silence", "synthetic"],
                "recipe": recipe,
            }
        )
finally:
    engine.unload()

manifest = {
    "schema_version": 1,
    "id": "mac_v2026_14_routing_synthetic",
    "usage": "evaluation_only",
    "created_at": datetime.now(UTC).isoformat(),
    "purpose": "Bounded routing path exercise, separate from sermon latency and natural-speech quality gates.",
    "replays": replays,
    "utterances": [],
    "translations": [],
    "gates": {"natural_spanish": "pending", "two_speaker": "pending", "bilingual_blind_review": "pending"},
}
destination = Path("docs/evaluation/mac_v2026_14_routing_synthetic.json")
with destination.open("x") as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
