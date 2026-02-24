#!/usr/bin/env python3
"""
evaluate_piper.py -- TTS Quality Evaluation for Piper Church Voices

Round-trip evaluation: synthesize text with Piper TTS, transcribe back with
Whisper STT, compare original vs transcribed text using jiwer (WER, CER).
Also performs prosody analysis (speaking rate, pause detection) and
theological term spot-checking.

Designed for the SRTranslate bilingual church transcription pipeline.

Usage:
    python evaluate_piper.py --voice piper_voices/en/voice.onnx --lang en
    python evaluate_piper.py --voice es_ES-carlfm-high --lang es
    python evaluate_piper.py --voice piper_voices/en/voice.onnx --lang en --skip-roundtrip
    python evaluate_piper.py --compare piper_voices/en/voice.onnx en_US-lessac-high --lang en
    python evaluate_piper.py --voice piper_voices/en/voice.onnx --lang en --output-dir results/
"""

import argparse
import csv
import io
import json
import logging
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports -- guarded so the script can at least parse args / show
# help even if heavy dependencies are missing.
# ---------------------------------------------------------------------------

try:
    from piper import PiperVoice

    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False

try:
    import mlx.core as mx
    import mlx_whisper

    MLX_WHISPER_AVAILABLE = True
except ImportError:
    MLX_WHISPER_AVAILABLE = False

try:
    import jiwer

    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIPER_SAMPLE_RATE = 22050
WHISPER_SAMPLE_RATE = 16000
DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"

# Normal speaking rate ranges (words per minute)
SPEAKING_RATE_RANGES = {
    "en": (120, 180),
    "es": (130, 190),
    "hi": (100, 160),
    "zh": (120, 180),
}

# Silence detection: energy threshold relative to RMS of the full signal.
# A frame is "silent" if its RMS energy is below this fraction of the
# overall RMS.
SILENCE_ENERGY_RATIO = 0.02
SILENCE_FRAME_MS = 20  # analysis frame length in milliseconds
LONG_PAUSE_THRESHOLD_S = 1.0  # pauses longer than this are flagged

# ---------------------------------------------------------------------------
# Theological test corpus
# ---------------------------------------------------------------------------

TEST_SENTENCES = {
    "en": [
        "The atonement of Christ provides propitiation for our sins.",
        "Sanctification is the work of the Holy Spirit in the believer.",
        "The covenant of grace reveals God's righteousness to all nations.",
        "Justification by faith alone is the cornerstone of the Gospel.",
        "The mediator of the new covenant is Christ Jesus our Lord.",
        "Reconciliation with God comes through the blood of the cross.",
        "The epistle to the Romans teaches the doctrine of redemption.",
        "Repentance and faith are gifts of God's sovereign grace.",
        "The apostle Paul proclaimed the mystery of the Gospel to the Gentiles.",
        "Scripture alone is the final authority for faith and practice.",
        "The Lord Jesus Christ will return in glory to judge the living and the dead.",
        "By grace you have been saved through faith, and this is not of yourselves.",
        "The righteousness of God is revealed from faith to faith as it is written.",
        "Blessed are those who hunger and thirst for righteousness, for they shall be filled.",
        "For God so loved the world that He gave His only begotten Son.",
    ],
    "es": [
        "La expiación de Cristo provee propiciación por nuestros pecados.",
        "La santificación es la obra del Espíritu Santo en el creyente.",
        "El pacto de gracia revela la justicia de Dios a todas las naciones.",
        "La justificación por la fe sola es la piedra angular del Evangelio.",
        "El mediador del nuevo pacto es Cristo Jesús nuestro Señor.",
        "La reconciliación con Dios viene por la sangre de la cruz.",
        "La epístola a los Romanos enseña la doctrina de la redención.",
        "El arrepentimiento y la fe son dones de la gracia soberana de Dios.",
        "El apóstol Pablo proclamó el misterio del Evangelio a los gentiles.",
        "Solo la Escritura es la autoridad final para la fe y la práctica.",
        "El Señor Jesucristo volverá en gloria para juzgar a los vivos y a los muertos.",
        "Porque por gracia sois salvos por medio de la fe, y esto no de vosotros.",
        "La justicia de Dios se revela de fe en fe, como está escrito.",
        "Bienaventurados los que tienen hambre y sed de justicia, porque ellos serán saciados.",
        "Porque de tal manera amó Dios al mundo que ha dado a su Hijo unigénito.",
    ],
    "hi": [
        "मसीह का प्रायश्चित हमारे पापों के लिए प्रायश्चित प्रदान करता है।",
        "पवित्रीकरण विश्वासी में पवित्र आत्मा का कार्य है।",
        "अनुग्रह की वाचा सभी जातियों के लिए परमेश्वर की धार्मिकता प्रकट करती है।",
        "केवल विश्वास द्वारा धर्मीकरण सुसमाचार की आधारशिला है।",
        "नई वाचा का मध्यस्थ हमारा प्रभु मसीह यीशु है।",
    ],
    "zh": [
        "基督的赎罪为我们的罪提供了挽回祭。",
        "成圣是圣灵在信徒中的工作。",
        "恩典之约向万国显明了上帝的公义。",
        "唯独因信称义是福音的基石。",
        "新约的中保是我们的主基督耶稣。",
    ],
}

# Theological terms for spot-checking pronunciation clarity
THEOLOGICAL_TERMS = {
    "en": [
        "atonement",
        "propitiation",
        "sanctification",
        "justification",
        "righteousness",
        "reconciliation",
        "mediator",
        "covenant",
        "epistle",
        "apostle",
    ],
    "es": [
        "expiación",
        "propiciación",
        "santificación",
        "justificación",
        "justicia",
        "reconciliación",
        "mediador",
        "pacto",
        "epístola",
        "apóstol",
    ],
    "hi": [
        "प्रायश्चित",
        "पवित्रीकरण",
        "धर्मीकरण",
        "धार्मिकता",
        "मध्यस्थ",
    ],
    "zh": [
        "赎罪",
        "成圣",
        "称义",
        "公义",
        "中保",
    ],
}


# ---------------------------------------------------------------------------
# Piper synthesis helpers
# ---------------------------------------------------------------------------


def load_piper_voice(voice_path_or_name, lang="en"):
    """Load a Piper voice from an ONNX path or a named model.

    If *voice_path_or_name* points to an existing file, load it directly.
    Otherwise, treat it as a Piper model name (e.g. ``en_US-lessac-high``)
    and let PiperVoice resolve it from the default model directory.
    """
    if not PIPER_AVAILABLE:
        logger.error("piper-tts not installed. Install with: pip install piper-tts")
        return None

    voice_path = Path(voice_path_or_name)

    if voice_path.exists() and voice_path.suffix == ".onnx":
        config_path = voice_path.with_suffix(".onnx.json")
        if not config_path.exists():
            logger.error(f"Voice config not found: {config_path}")
            return None
        logger.info(f"Loading Piper voice from ONNX: {voice_path}")
        return PiperVoice.load(str(voice_path), config_path=str(config_path))
    else:
        # Treat as a named model -- PiperVoice.load may support this
        # depending on the installed version and local model cache.
        logger.info(f"Loading Piper voice by name: {voice_path_or_name}")
        try:
            return PiperVoice.load(str(voice_path_or_name))
        except Exception as e:
            logger.error(f"Could not load Piper voice '{voice_path_or_name}': {e}")
            return None


def synthesize_to_wav_bytes(voice, text):
    """Synthesize *text* with *voice* and return raw WAV bytes.

    Returns:
        bytes: Complete WAV file content (header + PCM data).
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize(text, wf)
    return buf.getvalue()


def wav_bytes_to_float32(wav_bytes, target_sr=None):
    """Convert WAV bytes to a float32 numpy array.

    Optionally resample to *target_sr* (e.g. 16 kHz for Whisper).

    Returns:
        tuple: (audio_float32, sample_rate)
    """
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        logger.warning(f"Unexpected sample width {sampwidth}, attempting int16 decode")
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # Convert stereo to mono
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    # Resample if needed
    if target_sr is not None and target_sr != sr:
        from scipy.signal import resample

        target_len = int(len(samples) * target_sr / sr)
        samples = resample(samples, target_len).astype(np.float32)
        sr = target_sr

    return samples, sr


def get_audio_duration(wav_bytes):
    """Return the duration in seconds of a WAV byte buffer."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


# ---------------------------------------------------------------------------
# Whisper STT helper
# ---------------------------------------------------------------------------


def transcribe_audio(audio_float32, whisper_model=DEFAULT_WHISPER_MODEL, lang="en"):
    """Transcribe a float32 audio array using mlx-whisper.

    Returns:
        str: Transcribed text (stripped).
    """
    if not MLX_WHISPER_AVAILABLE:
        logger.error("mlx-whisper not installed. Install with: pip install mlx-whisper")
        return ""

    mx.set_cache_limit(100 * 1024 * 1024)

    lang_map = {"en": "en", "es": "es", "hi": "hi", "zh": "zh"}
    whisper_lang = lang_map.get(lang, "en")

    result = mlx_whisper.transcribe(
        audio_float32,
        path_or_hf_repo=whisper_model,
        language=whisper_lang,
        condition_on_previous_text=False,
    )
    return result.get("text", "").strip()


# ---------------------------------------------------------------------------
# Prosody analysis
# ---------------------------------------------------------------------------


def analyze_prosody(wav_bytes, text, lang="en"):
    """Analyze prosody of synthesized audio.

    Returns a dict with:
        - duration_s: total audio length
        - word_count: number of whitespace-delimited tokens
        - speaking_rate_wpm: words per minute
        - rate_in_range: whether the rate falls within the normal range
        - long_pauses: list of (start_s, duration_s) for pauses > threshold
        - num_long_pauses: count of long pauses
    """
    audio, sr = wav_bytes_to_float32(wav_bytes)
    duration_s = len(audio) / sr

    # Word count -- simple whitespace split
    words = text.split()
    word_count = len(words)
    speaking_rate_wpm = (word_count / duration_s) * 60.0 if duration_s > 0 else 0.0

    lo, hi = SPEAKING_RATE_RANGES.get(lang, (120, 180))
    rate_in_range = lo <= speaking_rate_wpm <= hi

    # Pause detection via frame-level energy thresholding
    frame_len = int(sr * SILENCE_FRAME_MS / 1000)
    overall_rms = np.sqrt(np.mean(audio**2)) + 1e-10
    threshold = overall_rms * SILENCE_ENERGY_RATIO

    n_frames = len(audio) // frame_len
    is_silent = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        frame = audio[i * frame_len : (i + 1) * frame_len]
        frame_rms = np.sqrt(np.mean(frame**2))
        is_silent[i] = frame_rms < threshold

    # Identify contiguous silent regions
    long_pauses = []
    in_pause = False
    pause_start = 0
    for i in range(n_frames):
        if is_silent[i] and not in_pause:
            in_pause = True
            pause_start = i
        elif not is_silent[i] and in_pause:
            pause_dur_s = (i - pause_start) * SILENCE_FRAME_MS / 1000
            if pause_dur_s >= LONG_PAUSE_THRESHOLD_S:
                long_pauses.append(
                    (
                        round(pause_start * SILENCE_FRAME_MS / 1000, 3),
                        round(pause_dur_s, 3),
                    )
                )
            in_pause = False
    # Handle trailing silence
    if in_pause:
        pause_dur_s = (n_frames - pause_start) * SILENCE_FRAME_MS / 1000
        if pause_dur_s >= LONG_PAUSE_THRESHOLD_S:
            long_pauses.append(
                (
                    round(pause_start * SILENCE_FRAME_MS / 1000, 3),
                    round(pause_dur_s, 3),
                )
            )

    return {
        "duration_s": round(duration_s, 3),
        "word_count": word_count,
        "speaking_rate_wpm": round(speaking_rate_wpm, 1),
        "rate_in_range": rate_in_range,
        "rate_range": (lo, hi),
        "long_pauses": long_pauses,
        "num_long_pauses": len(long_pauses),
    }


# ---------------------------------------------------------------------------
# Round-trip evaluation (TTS -> STT -> compare)
# ---------------------------------------------------------------------------


def evaluate_round_trip(voice, text, lang="en", whisper_model=DEFAULT_WHISPER_MODEL):
    """Synthesize text, transcribe back, compute WER/CER.

    Returns a dict with synthesis time, transcription, WER, CER, and prosody.
    """
    # Synthesize
    t0 = time.perf_counter()
    wav_bytes = synthesize_to_wav_bytes(voice, text)
    synth_ms = (time.perf_counter() - t0) * 1000

    duration_s = get_audio_duration(wav_bytes)

    # Prosody analysis (does not need Whisper)
    prosody = analyze_prosody(wav_bytes, text, lang=lang)

    # STT round-trip
    audio_16k, _ = wav_bytes_to_float32(wav_bytes, target_sr=WHISPER_SAMPLE_RATE)
    t0 = time.perf_counter()
    transcription = transcribe_audio(audio_16k, whisper_model=whisper_model, lang=lang)
    stt_ms = (time.perf_counter() - t0) * 1000

    # WER / CER
    wer = 0.0
    cer = 0.0
    if JIWER_AVAILABLE and transcription:
        try:
            wer = jiwer.wer(text.lower(), transcription.lower())
            cer = jiwer.cer(text.lower(), transcription.lower())
        except Exception as e:
            logger.warning(f"jiwer error: {e}")
    elif not JIWER_AVAILABLE:
        logger.warning("jiwer not installed -- skipping WER/CER")

    return {
        "original": text,
        "transcription": transcription,
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "synth_ms": round(synth_ms, 1),
        "stt_ms": round(stt_ms, 1),
        "duration_s": round(duration_s, 3),
        "prosody": prosody,
    }


def evaluate_theological_terms(voice, lang="en", whisper_model=DEFAULT_WHISPER_MODEL):
    """Spot-check individual theological terms.

    Synthesizes each term in isolation, transcribes it back, and checks
    whether the transcription matches (case-insensitive exact match).

    Returns a list of per-term result dicts.
    """
    terms = THEOLOGICAL_TERMS.get(lang, [])
    if not terms:
        logger.warning(f"No theological terms defined for language '{lang}'")
        return []

    results = []
    for term in terms:
        wav_bytes = synthesize_to_wav_bytes(voice, term)
        audio_16k, _ = wav_bytes_to_float32(wav_bytes, target_sr=WHISPER_SAMPLE_RATE)
        transcription = transcribe_audio(audio_16k, whisper_model=whisper_model, lang=lang)

        # Normalize for comparison -- strip punctuation and lowercase
        import re

        norm_original = re.sub(r"[^\w\s]", "", term.lower()).strip()
        norm_transcription = re.sub(r"[^\w\s]", "", transcription.lower()).strip()
        match = norm_original == norm_transcription

        results.append(
            {
                "term": term,
                "transcription": transcription,
                "match": match,
            }
        )

        status = "OK" if match else "MISMATCH"
        logger.info(f"  [{status}] {term!r} -> {transcription!r}")

    return results


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------


def evaluate_voice(
    voice_path_or_name,
    lang="en",
    whisper_model=DEFAULT_WHISPER_MODEL,
    output_dir=None,
    skip_roundtrip=False,
):
    """Run the full evaluation suite for a single Piper voice.

    Returns a summary dict.
    """
    project_root = Path(__file__).resolve().parent.parent

    if output_dir is None:
        output_dir = project_root / "metrics"
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    voice = load_piper_voice(voice_path_or_name, lang=lang)
    if voice is None:
        logger.error("Failed to load voice -- aborting evaluation")
        return None

    voice_label = Path(voice_path_or_name).stem if Path(voice_path_or_name).exists() else str(voice_path_or_name)
    timestamp = datetime.now().isoformat()
    sentences = TEST_SENTENCES.get(lang, TEST_SENTENCES["en"])

    logger.info(f"Evaluating voice: {voice_label}")
    logger.info(f"  Language:       {lang}")
    logger.info(f"  Whisper model:  {whisper_model}")
    logger.info(f"  Test sentences: {len(sentences)}")
    logger.info(f"  Skip roundtrip: {skip_roundtrip}")

    # ------------------------------------------------------------------
    # Round-trip evaluation on test sentences
    # ------------------------------------------------------------------
    sentence_results = []
    if not skip_roundtrip:
        if not MLX_WHISPER_AVAILABLE:
            logger.error("mlx-whisper required for round-trip evaluation but not installed")
            skip_roundtrip = True
        if not JIWER_AVAILABLE:
            logger.warning("jiwer not installed -- WER/CER will be 0.0")

    if not skip_roundtrip:
        logger.info("\n--- Round-trip evaluation (TTS -> STT -> compare) ---")
        for i, sentence in enumerate(sentences, 1):
            logger.info(f"  [{i}/{len(sentences)}] {sentence[:60]}{'...' if len(sentence) > 60 else ''}")
            result = evaluate_round_trip(
                voice,
                sentence,
                lang=lang,
                whisper_model=whisper_model,
            )
            result["voice"] = voice_label
            result["language"] = lang
            result["timestamp"] = timestamp
            sentence_results.append(result)
            logger.info(
                f"    WER={result['wer']:.2%}  CER={result['cer']:.2%}  "
                f"rate={result['prosody']['speaking_rate_wpm']:.0f} WPM  "
                f"dur={result['duration_s']:.2f}s"
            )
    else:
        # Prosody-only mode: synthesize but do not run STT
        logger.info("\n--- Prosody analysis only (--skip-roundtrip) ---")
        for i, sentence in enumerate(sentences, 1):
            logger.info(f"  [{i}/{len(sentences)}] {sentence[:60]}{'...' if len(sentence) > 60 else ''}")
            wav_bytes = synthesize_to_wav_bytes(voice, sentence)
            prosody = analyze_prosody(wav_bytes, sentence, lang=lang)
            result = {
                "voice": voice_label,
                "language": lang,
                "original": sentence,
                "transcription": "",
                "wer": None,
                "cer": None,
                "synth_ms": None,
                "stt_ms": None,
                "duration_s": prosody["duration_s"],
                "prosody": prosody,
                "timestamp": timestamp,
            }
            sentence_results.append(result)
            logger.info(
                f"    rate={prosody['speaking_rate_wpm']:.0f} WPM  "
                f"dur={prosody['duration_s']:.2f}s  "
                f"pauses={prosody['num_long_pauses']}"
            )

    # ------------------------------------------------------------------
    # Theological term spot-check
    # ------------------------------------------------------------------
    term_results = []
    if not skip_roundtrip:
        logger.info("\n--- Theological term spot-check ---")
        term_results = evaluate_theological_terms(
            voice,
            lang=lang,
            whisper_model=whisper_model,
        )

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------
    summary = _build_summary(
        voice_label,
        lang,
        sentence_results,
        term_results,
        timestamp,
        skip_roundtrip=skip_roundtrip,
    )

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    _write_csv(output_dir / "tts_evaluation.csv", sentence_results, voice_label, lang)
    _write_json_summary(output_dir / "tts_evaluation_summary.json", summary)
    _print_summary_table(summary)

    return summary


def _build_summary(voice_label, lang, sentence_results, term_results, timestamp, skip_roundtrip=False):
    """Compute aggregate metrics from per-sentence results."""
    summary = {
        "voice": voice_label,
        "language": lang,
        "timestamp": timestamp,
        "num_sentences": len(sentence_results),
        "skip_roundtrip": skip_roundtrip,
    }

    if not skip_roundtrip and sentence_results:
        wers = [r["wer"] for r in sentence_results if r["wer"] is not None]
        cers = [r["cer"] for r in sentence_results if r["cer"] is not None]
        if wers:
            summary["mean_wer"] = round(sum(wers) / len(wers), 4)
            summary["max_wer"] = round(max(wers), 4)
            summary["min_wer"] = round(min(wers), 4)
        if cers:
            summary["mean_cer"] = round(sum(cers) / len(cers), 4)
            summary["max_cer"] = round(max(cers), 4)
            summary["min_cer"] = round(min(cers), 4)

    # Prosody aggregates
    rates = [r["prosody"]["speaking_rate_wpm"] for r in sentence_results]
    durations = [r["duration_s"] for r in sentence_results if r["duration_s"] is not None]
    if rates:
        summary["mean_speaking_rate_wpm"] = round(sum(rates) / len(rates), 1)
    if durations:
        summary["total_audio_duration_s"] = round(sum(durations), 2)

    rate_flags = [r for r in sentence_results if not r["prosody"]["rate_in_range"]]
    summary["num_rate_out_of_range"] = len(rate_flags)

    total_long_pauses = sum(r["prosody"]["num_long_pauses"] for r in sentence_results)
    summary["total_long_pauses"] = total_long_pauses

    # Theological term accuracy
    if term_results:
        n_match = sum(1 for t in term_results if t["match"])
        summary["term_accuracy"] = round(n_match / len(term_results), 4)
        summary["term_correct"] = n_match
        summary["term_total"] = len(term_results)
        summary["term_mismatches"] = [
            {"term": t["term"], "got": t["transcription"]} for t in term_results if not t["match"]
        ]

    return summary


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _write_csv(csv_path, sentence_results, voice_label, lang):
    """Append sentence-level results to the CSV log."""
    file_exists = csv_path.exists()

    fieldnames = [
        "voice",
        "language",
        "test_sentence",
        "round_trip_wer",
        "round_trip_cer",
        "speaking_rate_wpm",
        "duration_s",
        "timestamp",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in sentence_results:
            writer.writerow(
                {
                    "voice": r["voice"],
                    "language": r["language"],
                    "test_sentence": r["original"],
                    "round_trip_wer": r["wer"] if r["wer"] is not None else "",
                    "round_trip_cer": r["cer"] if r["cer"] is not None else "",
                    "speaking_rate_wpm": r["prosody"]["speaking_rate_wpm"],
                    "duration_s": r["duration_s"],
                    "timestamp": r["timestamp"],
                }
            )

    logger.info(f"CSV results appended to {csv_path}")


def _write_json_summary(json_path, summary):
    """Write (or update) the JSON summary file.

    If the file already exists, its contents are loaded and the new summary
    is appended under the voice name key so that multiple evaluations
    accumulate in a single file.
    """
    all_summaries = {}
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                all_summaries = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    key = f"{summary['voice']}_{summary['language']}_{summary['timestamp']}"
    all_summaries[key] = summary

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    logger.info(f"JSON summary written to {json_path}")


def _print_summary_table(summary):
    """Print a human-readable summary table to stdout."""
    print(f"\n{'=' * 60}")
    print("TTS EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Voice:              {summary['voice']}")
    print(f"  Language:           {summary['language']}")
    print(f"  Sentences tested:   {summary['num_sentences']}")

    if not summary.get("skip_roundtrip"):
        mean_wer = summary.get("mean_wer")
        mean_cer = summary.get("mean_cer")
        if mean_wer is not None:
            print(f"  Mean WER:           {mean_wer:.2%}")
            print(f"  Min/Max WER:        {summary.get('min_wer', 0):.2%} / {summary.get('max_wer', 0):.2%}")
        if mean_cer is not None:
            print(f"  Mean CER:           {mean_cer:.2%}")
            print(f"  Min/Max CER:        {summary.get('min_cer', 0):.2%} / {summary.get('max_cer', 0):.2%}")

    mean_rate = summary.get("mean_speaking_rate_wpm")
    if mean_rate is not None:
        print(f"  Mean speaking rate: {mean_rate:.0f} WPM")
    print(f"  Rate out of range:  {summary.get('num_rate_out_of_range', 0)} sentences")
    print(f"  Long pauses (>1s):  {summary.get('total_long_pauses', 0)}")

    total_dur = summary.get("total_audio_duration_s")
    if total_dur is not None:
        print(f"  Total audio:        {total_dur:.1f}s")

    if "term_accuracy" in summary:
        print("\n  Theological terms:")
        print(
            f"    Accuracy:         {summary['term_accuracy']:.0%} ({summary['term_correct']}/{summary['term_total']})"
        )
        if summary.get("term_mismatches"):
            print("    Mismatches:")
            for m in summary["term_mismatches"]:
                print(f"      {m['term']!r} -> {m['got']!r}")

    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Comparison mode
# ---------------------------------------------------------------------------


def compare_voices(voice_a, voice_b, lang="en", whisper_model=DEFAULT_WHISPER_MODEL, output_dir=None):
    """Evaluate two voices side by side and print a comparison table.

    Useful for comparing a stock Piper voice against a fine-tuned church voice.
    """
    logger.info(f"Comparing two voices ({lang}):")
    logger.info(f"  Voice A: {voice_a}")
    logger.info(f"  Voice B: {voice_b}")

    summary_a = evaluate_voice(
        voice_a,
        lang=lang,
        whisper_model=whisper_model,
        output_dir=output_dir,
    )
    summary_b = evaluate_voice(
        voice_b,
        lang=lang,
        whisper_model=whisper_model,
        output_dir=output_dir,
    )

    if summary_a is None or summary_b is None:
        logger.error("One or both voices failed to evaluate -- cannot compare")
        return

    _print_comparison(summary_a, summary_b)


def _print_comparison(a, b):
    """Print a side-by-side comparison of two voice summaries."""
    print(f"\n{'=' * 72}")
    print("VOICE COMPARISON")
    print(f"{'=' * 72}")

    col_w = 24
    label_w = 24

    header = "Metric".ljust(label_w) + a["voice"][:col_w].ljust(col_w) + b["voice"][:col_w].ljust(col_w)
    print(header)
    print("-" * (label_w + col_w * 2))

    rows = [
        ("Language", a["language"], b["language"]),
        ("Sentences", str(a["num_sentences"]), str(b["num_sentences"])),
    ]

    if not a.get("skip_roundtrip") and not b.get("skip_roundtrip"):

        def fmt_pct(v):
            return f"{v:.2%}" if v is not None else "N/A"

        rows.extend(
            [
                ("Mean WER", fmt_pct(a.get("mean_wer")), fmt_pct(b.get("mean_wer"))),
                ("Mean CER", fmt_pct(a.get("mean_cer")), fmt_pct(b.get("mean_cer"))),
            ]
        )

        # Delta
        if a.get("mean_wer") is not None and b.get("mean_wer") is not None:
            delta_wer = b["mean_wer"] - a["mean_wer"]
            sign = "+" if delta_wer >= 0 else ""
            rows.append(("WER delta (B-A)", "", f"{sign}{delta_wer:.2%}"))

    def fmt_rate(s):
        r = s.get("mean_speaking_rate_wpm")
        return f"{r:.0f} WPM" if r is not None else "N/A"

    rows.extend(
        [
            ("Speaking rate", fmt_rate(a), fmt_rate(b)),
            ("Rate out of range", str(a.get("num_rate_out_of_range", 0)), str(b.get("num_rate_out_of_range", 0))),
            ("Long pauses", str(a.get("total_long_pauses", 0)), str(b.get("total_long_pauses", 0))),
        ]
    )

    if "term_accuracy" in a and "term_accuracy" in b:
        rows.append(
            (
                "Term accuracy",
                f"{a['term_accuracy']:.0%} ({a['term_correct']}/{a['term_total']})",
                f"{b['term_accuracy']:.0%} ({b['term_correct']}/{b['term_total']})",
            )
        )

    for label, va, vb in rows:
        print(f"{label.ljust(label_w)}{str(va).ljust(col_w)}{str(vb).ljust(col_w)}")

    print(f"{'=' * 72}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Piper TTS voice quality via round-trip STT comparison",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Piper voice name or path to custom ONNX model",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        choices=["en", "es", "hi", "zh"],
        help="Language code (default: en)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Metrics output directory (default: metrics/)",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default=DEFAULT_WHISPER_MODEL,
        help=f"Whisper model for STT verification (default: {DEFAULT_WHISPER_MODEL})",
    )
    parser.add_argument(
        "--skip-roundtrip",
        action="store_true",
        help="Skip round-trip WER evaluation (prosody analysis only)",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("VOICE_A", "VOICE_B"),
        help="Compare two voices side by side (e.g. stock vs fine-tuned)",
    )
    args = parser.parse_args()

    # Dependency checks
    if not PIPER_AVAILABLE:
        logger.error("piper-tts is required. Install with: pip install piper-tts")
        sys.exit(1)

    if args.compare:
        compare_voices(
            args.compare[0],
            args.compare[1],
            lang=args.lang,
            whisper_model=args.whisper_model,
            output_dir=args.output_dir,
        )
    elif args.voice:
        evaluate_voice(
            args.voice,
            lang=args.lang,
            whisper_model=args.whisper_model,
            output_dir=args.output_dir,
            skip_roundtrip=args.skip_roundtrip,
        )
    else:
        parser.error("Either --voice or --compare must be specified")


if __name__ == "__main__":
    main()
