#!/usr/bin/env python3
"""
dry_run_ab.py — Live A/B Bilingual Speech-to-Text Dry Run

Mic → Silero VAD → Distil-Whisper STT → TranslateGemma (4B + 12B) → WebSocket → Browser

Architecture:
  - STT: Distil-Whisper via mlx-whisper (Apple Silicon native)
  - Translation: TranslateGemma via mlx-lm (4-bit quantized, Apple Silicon native)
  - Both 4B and 12B loaded simultaneously (~9GB total)

Usage:
    python dry_run_ab.py                     # Both models (A/B parallel)
    python dry_run_ab.py --4b-only           # 4B only (lighter)
    python dry_run_ab.py --chunk-duration 5  # Longer chunks
    python dry_run_ab.py --ws-port 9000      # Different WebSocket port
"""

import argparse
import asyncio
import csv
import json
import os
import signal
import sys
import time
from datetime import datetime

import numpy as np
import sounddevice as sd
import torch
import websockets
import mlx_whisper

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000          # Whisper/VAD require 16kHz
MIC_SAMPLE_RATE = 48000      # Most mics are 48kHz native; will resample to 16kHz
CHUNK_DURATION = 2.0         # seconds of speech — more context = better word accuracy
VAD_THRESHOLD = 0.3          # Lower threshold for better sensitivity
WS_PORT = 8765
MIC_DEVICE = None            # None = auto-detect best input device
CSV_PATH = f"metrics/ab_metrics_{datetime.now():%Y%m%d_%H%M%S}.csv"

# MLX model IDs (4-bit quantized, community-converted)
MLX_MODEL_A = "mlx-community/translategemma-4b-it-4bit"   # ~2.2GB
MLX_MODEL_B = "mlx-community/translategemma-12b-it-4bit"  # ~6.6GB

# Whisper initial_prompt — biases decoder toward theological vocabulary
# that Whisper otherwise misrecognizes (e.g. "media" instead of "mediator")
WHISPER_PROMPT = (
    "Sermon at Stark Road Gospel Hall. "
    "God is always capitalized. Christ Jesus, the Holy Spirit, God the Father. "
    "Topics: salvation, atonement, propitiation, mediator, covenant, "
    "righteousness, sanctification, justification, redemption, Savior, "
    "reconciliation, intercession, predestination, sovereignty, "
    "sin, repentance, forgiveness, reign, bow, glory, grace, mercy, "
    "the Gospel, epistle, apostle, tabernacle, congregation, "
    "First Timothy, Second Timothy, First Corinthians, Ephesians, "
    "Romans, Hebrews, Galatians, Colossians, Thessalonians, Philippians, "
    "Genesis, Exodus, Leviticus, Deuteronomy, Isaiah, Jeremiah, Ezekiel, "
    "Scripture, the Lord, the Word of God."
)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

audio_queue = asyncio.Queue()
ws_clients = set()
chunk_id = 0
all_results = []
prev_text = ""  # last chunk's transcription — fed to Whisper as context

# Models (set during init)
vad_model = None
vad_utils = None
stt_pipe = None
mlx_a_model = None
mlx_a_tokenizer = None
mlx_b_model = None
mlx_b_tokenizer = None
marian_model = None
marian_tokenizer = None


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_vad():
    """Load Silero VAD (~2MB)."""
    print("[1/6] Loading Silero VAD...")
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
    print("  VAD ready")
    return model, utils


def load_whisper():
    """Load Distil-Whisper via mlx-whisper (Apple Silicon native)."""
    import mlx_whisper
    import mlx.core as mx
    mx.set_cache_limit(100 * 1024 * 1024)  # prevent memory growth

    model_id = "mlx-community/distil-whisper-large-v3"
    print(f"[2/6] Loading {model_id} (MLX)...")
    t0 = time.time()
    # Warm up — first call downloads and compiles the model
    silence = np.zeros(16000, dtype=np.float32)
    mlx_whisper.transcribe(silence, path_or_hf_repo=model_id,
                           condition_on_previous_text=False)
    print(f"  Whisper ready ({time.time()-t0:.1f}s)")
    return model_id  # mlx_whisper uses model_id per call, no persistent object


def load_mlx_gemma(model_id, label):
    """Load a TranslateGemma model via MLX (4-bit, Apple Silicon native)."""
    from mlx_lm import load
    import mlx.core as mx
    mx.set_cache_limit(100 * 1024 * 1024)  # prevent memory growth in repeated inference

    print(f"  Loading {model_id}...")
    t0 = time.time()
    model, tokenizer = load(model_id)

    # Fix: add <end_of_turn> as EOS so generation stops after the translation
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    tokenizer._eos_token_ids = {tokenizer.eos_token_id, eot_id}

    elapsed = time.time() - t0
    print(f"  {label} ready ({elapsed:.1f}s)")
    return model, tokenizer


def load_translation_models(load_b=True):
    """Load TranslateGemma model(s) via MLX."""
    print("[3/5] Loading TranslateGemma models (MLX 4-bit)...")
    a_model, a_tok = load_mlx_gemma(MLX_MODEL_A, "Approach A (4B)")

    b_model, b_tok = None, None
    if load_b:
        try:
            b_model, b_tok = load_mlx_gemma(MLX_MODEL_B, "Approach B (12B)")
        except Exception as e:
            print(f"  12B load failed: {e}")
            print("  Running 4B only.")

    return a_model, a_tok, b_model, b_tok


def load_marian():
    """Load MarianMT (~298MB) for fast partial translations."""
    from transformers import MarianMTModel, MarianTokenizer
    model_id = "Helsinki-NLP/opus-mt-en-es"
    print(f"[4/6] Loading {model_id} (MarianMT)...")
    t0 = time.time()
    tokenizer = MarianTokenizer.from_pretrained(model_id)
    model = MarianMTModel.from_pretrained(model_id)
    model.eval()
    # Warm up
    inputs = tokenizer("Hello", return_tensors="pt", padding=True)
    model.generate(**inputs, max_new_tokens=16)
    print(f"  MarianMT ready ({time.time()-t0:.1f}s)")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Translation Quality Estimation (lightweight, real-time)
# ---------------------------------------------------------------------------

import re

# Common English words unlikely in valid Spanish translations
_EN_WORDS = re.compile(
    r'\b(the|and|of|that|have|for|not|with|you|this|but|his|from|they|'
    r'been|said|each|which|their|will|other|about|many|then|them|these|'
    r'would|could|should|because|into|after|before|between|under|through)\b',
    re.IGNORECASE,
)

def qe_length_ratio(source, translation):
    """Check if Spanish translation length is plausible vs English source.
    Spanish is typically 15-25% longer than English.
    Returns a score 0-1 (1 = perfect ratio, lower = suspicious).
    """
    if not source or not translation:
        return 0.0
    ratio = len(translation) / len(source)
    # Expected range: 0.8 to 1.8 (generous bounds)
    if 0.9 <= ratio <= 1.6:
        return 1.0
    elif 0.7 <= ratio <= 2.0:
        return 0.7
    elif 0.5 <= ratio <= 2.5:
        return 0.4
    return 0.1


def qe_untranslated(source, translation):
    """Detect if translation contains significant untranslated English.
    Returns a score 0-1 (1 = fully translated, lower = untranslated content).
    """
    if not translation:
        return 0.0
    # Count English stopwords in translation
    en_matches = _EN_WORDS.findall(translation)
    words = translation.split()
    if not words:
        return 0.0
    en_ratio = len(en_matches) / len(words)
    if en_ratio < 0.05:
        return 1.0
    elif en_ratio < 0.15:
        return 0.7
    elif en_ratio < 0.30:
        return 0.4
    return 0.1


def qe_score(source, translation):
    """Combined lightweight QE score (0-1). Higher = better."""
    lr = qe_length_ratio(source, translation)
    ut = qe_untranslated(source, translation)
    return round((lr + ut) / 2, 2)


# ---------------------------------------------------------------------------
# Translation (MLX)
# ---------------------------------------------------------------------------

def translate_mlx(model, tokenizer, text):
    """Translate English to Spanish using TranslateGemma via MLX.

    Returns (translation, latency_ms, generation_tps).
    """
    from mlx_lm import generate

    if model is None or tokenizer is None:
        return "(model not loaded)", 0.0, 0.0

    messages = [{"role": "user", "content": [
        {"type": "text",
         "source_lang_code": "en",
         "target_lang_code": "es",
         "text": text}
    ]}]

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

    # Cap max_tokens proportional to input (Spanish ~1.3x English tokens + margin)
    input_words = len(text.split())
    max_tok = max(32, int(input_words * 2.5))

    t0 = time.perf_counter()
    result = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_tok,
        verbose=False,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    # generate returns the full text including prompt — extract just the generation
    clean = result.split("<end_of_turn>")[0].strip()
    # Estimate tps from output tokens and time
    out_tokens = len(tokenizer.encode(clean))
    gen_tps = out_tokens / (latency_ms / 1000) if latency_ms > 0 else 0.0
    return clean, latency_ms, gen_tps


def translate_marian(text):
    """Fast English→Spanish via MarianMT (~50-100ms). For partials."""
    if marian_model is None or marian_tokenizer is None:
        return "(MarianMT not loaded)", 0.0

    t0 = time.perf_counter()
    inputs = marian_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated = marian_model.generate(**inputs, max_new_tokens=128)
    result = marian_tokenizer.decode(translated[0], skip_special_tokens=True)
    latency_ms = (time.perf_counter() - t0) * 1000
    return result, latency_ms


# ---------------------------------------------------------------------------
# Audio Processing
# ---------------------------------------------------------------------------

def detect_macbook_mic():
    """Find the MacBook Pro built-in microphone by name.
    Returns (device_index, measured_rms) so gain can be auto-calibrated.
    """
    devices = sd.query_devices()
    # Find MacBook Pro mic by name
    for idx, d in enumerate(devices):
        if d['max_input_channels'] > 0 and 'MacBook Pro' in d['name']:
            # Quick RMS measurement for gain calibration
            test = sd.rec(int(1.0 * d['default_samplerate']),
                          samplerate=d['default_samplerate'],
                          channels=1, dtype='float32', device=idx)
            sd.wait()
            rms = float(np.sqrt(np.mean(test[:, 0]**2)))
            print(f"  Using: [{idx}] {d['name']} (RMS={rms:.4f})")
            return idx, rms

    # Fallback: use system default input if MacBook Pro mic not found
    print("  WARNING: MacBook Pro mic not found, using system default", file=sys.stderr)
    default_idx = sd.default.device[0]
    d = sd.query_devices(default_idx)
    test = sd.rec(int(1.0 * d['default_samplerate']),
                  samplerate=d['default_samplerate'],
                  channels=1, dtype='float32', device=default_idx)
    sd.wait()
    rms = float(np.sqrt(np.mean(test[:, 0]**2)))
    print(f"  Using default: [{default_idx}] {d['name']} (RMS={rms:.4f})")
    return default_idx, rms


MIC_GAIN = 1.0  # Set during mic detection based on measured signal level
TARGET_RMS = 0.08  # Target RMS for speech audio fed to VAD/Whisper

def audio_callback(indata, frames, time_info, status):
    """sounddevice callback — resample from mic rate to 16kHz and push to queue."""
    if status:
        print(f"  Audio status: {status}", file=sys.stderr)
    raw = indata[:, 0].copy()
    # Clamp to [-1, 1] — some mics deliver out-of-range samples that break VAD
    raw = np.clip(raw, -1.0, 1.0)
    # Amplify quiet mic signals
    if MIC_GAIN != 1.0:
        raw = np.clip(raw * MIC_GAIN, -1.0, 1.0)
    if MIC_SAMPLE_RATE != SAMPLE_RATE:
        from scipy.signal import resample
        target_len = int(len(raw) * SAMPLE_RATE / MIC_SAMPLE_RATE)
        raw = resample(raw, target_len).astype(np.float32)
    audio_queue.put_nowait(raw)


def is_speech(audio_chunk, model, utils):
    """Check if audio chunk contains speech using Silero VAD (streaming mode).

    TODO: Singing/hymns cause VAD and STT to break — likely high RMS + tonal
    content triggers clipping or VAD confusion. Investigate: lower gain during
    music segments, or use inaSpeechSegmenter to detect music and skip/handle
    differently.
    """
    tensor = torch.from_numpy(audio_chunk).float()
    speech_prob = model(tensor, SAMPLE_RATE).item()
    return speech_prob > VAD_THRESHOLD


def _whisper_prompt():
    """Build Whisper prompt: theological vocab + previous transcription."""
    if prev_text:
        return WHISPER_PROMPT + " " + prev_text
    return WHISPER_PROMPT


async def process_partial(audio_data, utterance_id):
    """Fast partial: STT (~500ms) + MarianMT (~80ms). Italic in UI."""
    try:
        t0 = time.perf_counter()
        result = mlx_whisper.transcribe(
            audio_data,
            path_or_hf_repo=stt_pipe,
            language="en",
            condition_on_previous_text=False,
            initial_prompt=_whisper_prompt(),
        )
        stt_latency = (time.perf_counter() - t0) * 1000
        english = result["text"].strip()

        if not english:
            return

        # Fast MarianMT translation for partial display
        spanish, marian_latency = translate_marian(english)
        total = stt_latency + marian_latency

        print(f"  partial ({total:.0f}ms): {english} | {spanish}          ", end="\r")
        await broadcast({
            "type": "translation",
            "stage": "partial",
            "chunk_id": utterance_id,
            "english": english,
            "spanish_a": spanish,
            "spanish_b": None,
            "stt_latency_ms": round(stt_latency, 1),
            "latency_a_ms": round(marian_latency, 1),
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        print(f"\n  ERROR in partial: {e}", file=sys.stderr)


async def process_final(audio_data):
    """Final STT on full utterance + translation. High quality."""
    global chunk_id, prev_text

    chunk_id += 1
    cid = chunk_id

    try:
        # --- STT on full utterance audio ---
        e2e_start = time.perf_counter()
        t0 = e2e_start
        result = mlx_whisper.transcribe(
            audio_data,
            path_or_hf_repo=stt_pipe,
            language="en",
            condition_on_previous_text=False,
            initial_prompt=_whisper_prompt(),
        )
        stt_latency = (time.perf_counter() - t0) * 1000
        english = result["text"].strip()

        # Extract confidence
        stt_confidence = None
        segments = result.get("segments", [])
        if segments:
            avg_logprobs = [s.get("avg_logprob", 0) for s in segments if "avg_logprob" in s]
            if avg_logprobs:
                mean_logprob = sum(avg_logprobs) / len(avg_logprobs)
                stt_confidence = round(min(1.0, max(0.0, 1.0 + mean_logprob)), 2)

        if not english:
            return

        prev_text = english[-200:]

        # Broadcast refined English (replaces partial)
        print(f"\n{'='*60}")
        print(f"Chunk #{cid} | STT: {stt_latency:.0f}ms | EN: {english}")
        await broadcast({
            "type": "translation",
            "stage": "stt",
            "chunk_id": cid,
            "english": english,
            "spanish_a": None,
            "spanish_b": None,
            "stt_latency_ms": round(stt_latency, 1),
            "stt_confidence": stt_confidence,
            "timestamp": datetime.now().isoformat(),
        })

        # --- Translate ---
        spanish_b, lat_b, tps_b, qe_b = None, 0.0, 0.0, None

        if mlx_b_model is not None:
            loop = asyncio.get_event_loop()
            task_a = loop.run_in_executor(
                None, lambda: translate_mlx(mlx_a_model, mlx_a_tokenizer, english))
            task_b = loop.run_in_executor(
                None, lambda: translate_mlx(mlx_b_model, mlx_b_tokenizer, english))

            spanish_a, lat_a, tps_a = await task_a
            qe_a = qe_score(english, spanish_a)
            await broadcast({
                "type": "translation",
                "stage": "translation_a",
                "chunk_id": cid,
                "english": english,
                "spanish_a": spanish_a,
                "spanish_b": None,
                "stt_latency_ms": round(stt_latency, 1),
                "latency_a_ms": round(lat_a, 1),
                "stt_confidence": stt_confidence,
                "tps_a": round(tps_a, 1),
                "qe_a": qe_a,
                "timestamp": datetime.now().isoformat(),
            })

            spanish_b, lat_b, tps_b = await task_b
            qe_b = qe_score(english, spanish_b) if spanish_b and spanish_b != "(model not loaded)" else None
        else:
            spanish_a, lat_a, tps_a = translate_mlx(mlx_a_model, mlx_a_tokenizer, english)
            qe_a = qe_score(english, spanish_a)

        e2e_latency = (time.perf_counter() - e2e_start) * 1000

        conf_str = f" | conf: {stt_confidence:.2f}" if stt_confidence is not None else ""
        qe_str = f" | QE: A={qe_a}"
        if qe_b is not None:
            qe_str = f" | QE: A={qe_a} B={qe_b}"
            print(f"  +{lat_b:.0f}ms B ({tps_b:.0f} t/s): {spanish_b}")
        print(f"  +{lat_a:.0f}ms A ({tps_a:.0f} t/s): {spanish_a}")
        print(f"  E2E: {e2e_latency:.0f}ms{conf_str}{qe_str}")

        # Final broadcast
        result_data = {
            "type": "translation",
            "stage": "complete",
            "chunk_id": cid,
            "english": english,
            "spanish_a": spanish_a,
            "spanish_b": spanish_b,
            "stt_latency_ms": round(stt_latency, 1),
            "latency_a_ms": round(lat_a, 1),
            "latency_b_ms": round(lat_b, 1),
            "e2e_latency_ms": round(e2e_latency, 1),
            "stt_confidence": stt_confidence,
            "tps_a": round(tps_a, 1),
            "tps_b": round(tps_b, 1),
            "qe_a": qe_a,
            "qe_b": qe_b,
            "timestamp": datetime.now().isoformat(),
        }
        all_results.append(result_data)
        await broadcast(result_data)
        write_csv_row(result_data)

    except Exception as e:
        print(f"\n  ERROR in chunk #{cid}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# WebSocket Server
# ---------------------------------------------------------------------------

async def ws_handler(websocket, path=None):
    """Handle new WebSocket connections."""
    ws_clients.add(websocket)
    print(f"  Browser connected ({len(ws_clients)} client(s))")
    try:
        async for _ in websocket:
            pass  # We only send, not receive
    except websockets.ConnectionClosed:
        pass
    finally:
        ws_clients.discard(websocket)
        print(f"  Browser disconnected ({len(ws_clients)} client(s))")


async def broadcast(data):
    """Send data to all connected WebSocket clients."""
    if not ws_clients:
        print("  [ws] No clients connected, skipping broadcast")
        return
    msg = json.dumps(data)
    dead = set()
    clients = list(ws_clients)
    results = await asyncio.gather(
        *[client.send(msg) for client in clients],
        return_exceptions=True,
    )
    for client, result in zip(clients, results):
        if isinstance(result, Exception):
            print(f"  [ws] Send failed: {result}")
            dead.add(client)
    ws_clients.difference_update(dead)
    ok = len(clients) - len(dead)
    if ok > 0:
        print(f"  [ws] Sent to {ok} client(s)")


# ---------------------------------------------------------------------------
# CSV Logging
# ---------------------------------------------------------------------------

def init_csv():
    """Initialize CSV file with headers."""
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "chunk_id", "timestamp", "english",
            "spanish_a", "spanish_b",
            "stt_latency_ms", "latency_a_ms", "latency_b_ms",
            "e2e_latency_ms", "stt_confidence", "tps_a", "tps_b",
            "qe_a", "qe_b",
        ])
    print(f"  CSV: {CSV_PATH}")


def write_csv_row(data):
    """Append a row to the CSV log."""
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            data["chunk_id"], data["timestamp"], data["english"],
            data["spanish_a"], data["spanish_b"],
            data["stt_latency_ms"], data["latency_a_ms"], data["latency_b_ms"],
            data["e2e_latency_ms"], data.get("stt_confidence", ""),
            data.get("tps_a", ""), data.get("tps_b", ""),
            data.get("qe_a", ""), data.get("qe_b", ""),
        ])


# ---------------------------------------------------------------------------
# Summary Stats
# ---------------------------------------------------------------------------

def print_summary():
    """Print summary statistics on exit."""
    if not all_results:
        print("\nNo results to summarize.")
        return

    n = len(all_results)
    stt_lats = [r["stt_latency_ms"] for r in all_results]
    a_lats = [r["latency_a_ms"] for r in all_results]
    b_lats = [r["latency_b_ms"] for r in all_results if r["latency_b_ms"] > 0]
    same = sum(1 for r in all_results if r["spanish_a"] == r["spanish_b"])
    a_faster = sum(1 for r in all_results
                   if r["latency_b_ms"] > 0 and r["latency_a_ms"] < r["latency_b_ms"])

    print(f"\n{'='*60}")
    print(f"SESSION SUMMARY — {n} chunks")
    print(f"{'='*60}")
    print(f"  STT avg:     {np.mean(stt_lats):.0f}ms (median {np.median(stt_lats):.0f}ms)")
    print(f"  A (4B) avg:  {np.mean(a_lats):.0f}ms (median {np.median(a_lats):.0f}ms)")
    if b_lats:
        print(f"  B (12B) avg: {np.mean(b_lats):.0f}ms (median {np.median(b_lats):.0f}ms)")
        print(f"  Same output: {same}/{n} ({same/n:.0%})")
        print(f"  A faster:    {a_faster}/{n} ({a_faster/n:.0%})")
    else:
        print(f"  B (12B):     not loaded")
    print(f"  CSV saved:   {CSV_PATH}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------

async def audio_loop():
    """Main audio capture and processing loop with error recovery."""
    print("\nListening... (Ctrl+C to stop)\n")

    PARTIAL_INTERVAL = 1.0  # seconds between partial STT updates
    SILENCE_TRIGGER = 0.8   # seconds of silence to trigger final processing
    MAX_UTTERANCE = 8.0     # force-process if speaker doesn't pause

    speech_buffer = np.array([], dtype=np.float32)
    silence_frames = 0
    max_silence_frames = int(SILENCE_TRIGGER * SAMPLE_RATE / 512)
    frame_count = 0
    speech_frame_count = 0
    last_status_time = time.time()
    last_partial_len = 0    # audio length (samples) at last partial
    utterance_id = 0        # tracks current utterance for partial updates

    while True:
        try:
            stream = sd.InputStream(
                samplerate=MIC_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=int(MIC_SAMPLE_RATE * 0.032),  # ~32ms frames
                callback=audio_callback,
                device=MIC_DEVICE,
            )

            with stream:
                while True:
                    try:
                        audio_frame = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue

                    frame_count += 1
                    has_speech = is_speech(audio_frame, vad_model, vad_utils)

                    if has_speech:
                        if len(speech_buffer) == 0:
                            utterance_id += 1  # new utterance starting
                            last_partial_len = 0
                        speech_buffer = np.concatenate([speech_buffer, audio_frame])
                        silence_frames = 0
                        speech_frame_count += 1
                    else:
                        silence_frames += 1
                        # Keep buffering during brief pauses so words aren't dropped
                        if len(speech_buffer) > 0 and silence_frames < max_silence_frames:
                            speech_buffer = np.concatenate([speech_buffer, audio_frame])

                    # Periodic status line (~every 3s)
                    now = time.time()
                    if now - last_status_time >= 3.0:
                        buf_dur = len(speech_buffer) / SAMPLE_RATE
                        rms = float(np.sqrt(np.mean(audio_frame**2)))
                        print(f"  [status] frames={frame_count} speech={speech_frame_count} buf={buf_dur:.1f}s silence={silence_frames} rms={rms:.4f}", end="\r")
                        last_status_time = now

                    buffer_duration = len(speech_buffer) / SAMPLE_RATE
                    new_audio = (len(speech_buffer) - last_partial_len) / SAMPLE_RATE

                    # --- Partial: fire every PARTIAL_INTERVAL of new audio ---
                    if (new_audio >= PARTIAL_INTERVAL
                            and silence_frames < max_silence_frames
                            and buffer_duration < MAX_UTTERANCE):
                        await process_partial(speech_buffer.copy(), utterance_id)
                        last_partial_len = len(speech_buffer)

                    # --- Final: on silence gap or max duration ---
                    should_finalize = (
                        (buffer_duration >= 0.5 and silence_frames >= max_silence_frames)
                        or buffer_duration >= MAX_UTTERANCE
                    )

                    if should_finalize and buffer_duration >= 0.5:
                        print()  # newline after partial line
                        await process_final(speech_buffer.copy())
                        speech_buffer = np.array([], dtype=np.float32)
                        silence_frames = 0
                        speech_frame_count = 0
                        frame_count = 0
                        last_partial_len = 0
                        vad_model.reset_states()

        except sd.PortAudioError as e:
            print(f"\n  Mic error: {e} — retrying in 2s...", file=sys.stderr)
            speech_buffer = np.array([], dtype=np.float32)
            # Drain stale audio from queue
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            await asyncio.sleep(2)


async def main_async(args):
    """Start WebSocket server and audio loop."""
    global vad_model, vad_utils, stt_pipe
    global mlx_a_model, mlx_a_tokenizer, mlx_b_model, mlx_b_tokenizer
    global marian_model, marian_tokenizer

    print(f"{'='*60}")
    print(f"  Bilingual A/B Dry Run (MLX)")
    print(f"  Mode: {'A/B parallel' if args.run_ab else '4B only'}")
    print(f"  WebSocket: ws://localhost:{args.ws_port}")
    print(f"{'='*60}\n")

    # Load models
    vad_model, vad_utils = load_vad()
    stt_pipe = load_whisper()
    mlx_a_model, mlx_a_tokenizer, mlx_b_model, mlx_b_tokenizer = \
        load_translation_models(load_b=args.run_ab)
    marian_model, marian_tokenizer = load_marian()

    # Detect best microphone and auto-calibrate gain
    global MIC_DEVICE, MIC_GAIN
    if MIC_DEVICE is None:
        print("[5/6] Detecting microphone...")
        MIC_DEVICE, mic_rms = detect_macbook_mic()
        if mic_rms > 0 and MIC_GAIN == 1.0:
            # Auto-calibrate: scale signal to TARGET_RMS
            MIC_GAIN = max(1.0, min(20.0, TARGET_RMS / mic_rms))
            print(f"  Auto-gain: {MIC_GAIN:.1f}x (mic RMS={mic_rms:.4f}, target={TARGET_RMS})")

    print(f"[6/6] Starting WebSocket server on port {args.ws_port}...")
    init_csv()

    # Start WebSocket server
    ws_server = await websockets.serve(
        ws_handler, "localhost", args.ws_port,
        ping_interval=None,  # disable pings — inference blocks event loop
    )
    print(f"  WebSocket ready — open ab_display.html in browser")

    # Run audio loop
    try:
        await audio_loop()
    except KeyboardInterrupt:
        pass
    finally:
        print_summary()
        ws_server.close()
        await ws_server.wait_closed()


def main():
    parser = argparse.ArgumentParser(
        description="Live A/B bilingual speech-to-text dry run (MLX)"
    )
    parser.add_argument("--ab", action="store_true", dest="run_ab",
                        help="Load both 4B and 12B for A/B comparison (default: 4B only)")
    parser.add_argument("--chunk-duration", type=float, default=2.0,
                        help="Seconds of speech to accumulate before processing")
    parser.add_argument("--ws-port", type=int, default=8765,
                        help="WebSocket server port")
    parser.add_argument("--vad-threshold", type=float, default=0.3,
                        help="VAD speech threshold (0-1)")
    parser.add_argument("--device", type=int, default=None,
                        help="Audio input device index (default: auto-detect)")
    parser.add_argument("--gain", type=float, default=None,
                        help="Mic gain multiplier (default: auto-calibrate)")
    args = parser.parse_args()

    global CHUNK_DURATION, WS_PORT, VAD_THRESHOLD, MIC_DEVICE, MIC_GAIN
    CHUNK_DURATION = args.chunk_duration
    WS_PORT = args.ws_port
    VAD_THRESHOLD = args.vad_threshold
    MIC_DEVICE = args.device
    if args.gain is not None:
        MIC_GAIN = args.gain  # Explicit gain skips auto-calibration

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\nStopping...")
        print_summary()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print_summary()


if __name__ == "__main__":
    main()
