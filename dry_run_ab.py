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

SAMPLE_RATE = 16000
CHUNK_DURATION = 3.0        # seconds of speech to accumulate before processing
VAD_THRESHOLD = 0.5
WS_PORT = 8765
CSV_PATH = f"metrics/ab_metrics_{datetime.now():%Y%m%d_%H%M%S}.csv"

# MLX model IDs (4-bit quantized, community-converted)
MLX_MODEL_A = "mlx-community/translategemma-4b-it-4bit"   # ~2.2GB
MLX_MODEL_B = "mlx-community/translategemma-12b-it-4bit"  # ~6.6GB

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

audio_queue = asyncio.Queue()
ws_clients = set()
chunk_id = 0
all_results = []

# Models (set during init)
vad_model = None
vad_utils = None
stt_pipe = None
mlx_a_model = None
mlx_a_tokenizer = None
mlx_b_model = None
mlx_b_tokenizer = None


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_vad():
    """Load Silero VAD (~2MB)."""
    print("[1/4] Loading Silero VAD...")
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
    print("  VAD ready")
    return model, utils


def load_whisper():
    """Load Distil-Whisper via mlx-whisper (Apple Silicon native)."""
    import mlx_whisper
    import mlx.core as mx
    mx.set_cache_limit(100 * 1024 * 1024)  # prevent memory growth

    model_id = "mlx-community/distil-whisper-large-v3"
    print(f"[2/4] Loading {model_id} (MLX)...")
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
    print("[3/4] Loading TranslateGemma models (MLX 4-bit)...")
    a_model, a_tok = load_mlx_gemma(MLX_MODEL_A, "Approach A (4B)")

    b_model, b_tok = None, None
    if load_b:
        try:
            b_model, b_tok = load_mlx_gemma(MLX_MODEL_B, "Approach B (12B)")
        except Exception as e:
            print(f"  12B load failed: {e}")
            print("  Running 4B only.")

    return a_model, a_tok, b_model, b_tok


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
    from mlx_lm import stream_generate

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

    t0 = time.perf_counter()
    chunks = []
    gen_tps = 0.0
    for response in stream_generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=128,
    ):
        chunks.append(response.text)
        gen_tps = response.generation_tps
        if response.finish_reason:
            break
    latency_ms = (time.perf_counter() - t0) * 1000

    translation = "".join(chunks)
    clean = translation.split("<end_of_turn>")[0].strip()
    return clean, latency_ms, gen_tps


# ---------------------------------------------------------------------------
# Audio Processing
# ---------------------------------------------------------------------------

def audio_callback(indata, frames, time_info, status):
    """sounddevice callback — push raw audio frames to async queue."""
    if status:
        print(f"  Audio status: {status}", file=sys.stderr)
    audio_queue.put_nowait(indata[:, 0].copy())


def is_speech(audio_chunk, model, utils):
    """Check if audio chunk contains speech using Silero VAD."""
    (get_speech_timestamps, _, _, _, _) = utils
    tensor = torch.from_numpy(audio_chunk).float()
    timestamps = get_speech_timestamps(tensor, model, sampling_rate=SAMPLE_RATE,
                                        threshold=VAD_THRESHOLD)
    return len(timestamps) > 0


async def process_chunk(audio_data):
    """Process a speech chunk: STT → Translate A → Translate B → broadcast."""
    global chunk_id

    chunk_id += 1
    cid = chunk_id

    try:
        # STT (MLX Whisper)
        e2e_start = time.perf_counter()
        t0 = e2e_start
        result = mlx_whisper.transcribe(
            audio_data,
            path_or_hf_repo=stt_pipe,  # model ID string
            language="en",
            condition_on_previous_text=False,
        )
        stt_latency = (time.perf_counter() - t0) * 1000
        english = result["text"].strip()

        # Extract confidence from Whisper segments
        stt_confidence = None
        segments = result.get("segments", [])
        if segments:
            avg_logprobs = [s.get("avg_logprob", 0) for s in segments if "avg_logprob" in s]
            if avg_logprobs:
                mean_logprob = sum(avg_logprobs) / len(avg_logprobs)
                stt_confidence = round(min(1.0, max(0.0, 1.0 + mean_logprob)), 2)

        if not english:
            return

        # Translate (MLX)
        spanish_a, lat_a, tps_a = translate_mlx(mlx_a_model, mlx_a_tokenizer, english)
        spanish_b, lat_b, tps_b = translate_mlx(mlx_b_model, mlx_b_tokenizer, english)

        e2e_latency = (time.perf_counter() - e2e_start) * 1000

        # Lightweight translation QE
        qe_a = qe_score(english, spanish_a)
        qe_b = qe_score(english, spanish_b) if spanish_b and spanish_b != "(model not loaded)" else None

        # Build result
        result_data = {
            "type": "translation",
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

        # Terminal output
        conf_str = f" | conf: {stt_confidence:.2f}" if stt_confidence is not None else ""
        qe_str = f" | QE: A={qe_a} B={qe_b}" if qe_b is not None else f" | QE: A={qe_a}"
        print(f"\n{'='*60}")
        print(f"Chunk #{cid} | STT: {stt_latency:.0f}ms | A: {lat_a:.0f}ms ({tps_a:.0f} t/s) | B: {lat_b:.0f}ms ({tps_b:.0f} t/s) | E2E: {e2e_latency:.0f}ms{conf_str}{qe_str}")
        print(f"  EN: {english}")
        print(f"  A (4B):  {spanish_a}")
        print(f"  B (12B): {spanish_b}")

        # WebSocket broadcast
        await broadcast(result_data)

        # CSV log
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
        return
    msg = json.dumps(data)
    dead = set()
    results = await asyncio.gather(
        *[client.send(msg) for client in ws_clients],
        return_exceptions=True,
    )
    for client, result in zip(list(ws_clients), results):
        if isinstance(result, Exception):
            dead.add(client)
    ws_clients -= dead


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

    speech_buffer = np.array([], dtype=np.float32)
    silence_frames = 0
    max_silence_frames = int(0.8 * SAMPLE_RATE / 512)  # ~0.8s of silence triggers processing

    while True:
        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=512,
                callback=audio_callback,
            )

            with stream:
                while True:
                    try:
                        audio_frame = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue

                    has_speech = is_speech(audio_frame, vad_model, vad_utils)

                    if has_speech:
                        speech_buffer = np.concatenate([speech_buffer, audio_frame])
                        silence_frames = 0
                    else:
                        silence_frames += 1

                    # Process when we have enough speech and hit a silence gap,
                    # or when buffer exceeds max duration
                    buffer_duration = len(speech_buffer) / SAMPLE_RATE
                    should_process = (
                        (buffer_duration >= CHUNK_DURATION and silence_frames >= max_silence_frames)
                        or buffer_duration >= CHUNK_DURATION * 3  # Force process at 3x
                    )

                    if should_process and buffer_duration >= 0.5:
                        await process_chunk(speech_buffer.copy())
                        speech_buffer = np.array([], dtype=np.float32)
                        silence_frames = 0

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

    print(f"{'='*60}")
    print(f"  Bilingual A/B Dry Run (MLX)")
    print(f"  Mode: {'4B only' if args.only_4b else 'A/B parallel'}")
    print(f"  WebSocket: ws://localhost:{args.ws_port}")
    print(f"{'='*60}\n")

    # Load models
    vad_model, vad_utils = load_vad()
    stt_pipe = load_whisper()
    mlx_a_model, mlx_a_tokenizer, mlx_b_model, mlx_b_tokenizer = \
        load_translation_models(load_b=not args.only_4b)

    print(f"[4/4] Starting WebSocket server on port {args.ws_port}...")
    init_csv()

    # Start WebSocket server
    ws_server = await websockets.serve(ws_handler, "localhost", args.ws_port)
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
    parser.add_argument("--4b-only", action="store_true", dest="only_4b",
                        help="Load only the 4B model (skip 12B)")
    parser.add_argument("--chunk-duration", type=float, default=3.0,
                        help="Seconds of speech to accumulate before processing")
    parser.add_argument("--ws-port", type=int, default=8765,
                        help="WebSocket server port")
    parser.add_argument("--vad-threshold", type=float, default=0.5,
                        help="VAD speech threshold (0-1)")
    args = parser.parse_args()

    global CHUNK_DURATION, WS_PORT, VAD_THRESHOLD
    CHUNK_DURATION = args.chunk_duration
    WS_PORT = args.ws_port
    VAD_THRESHOLD = args.vad_threshold

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
