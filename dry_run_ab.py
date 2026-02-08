#!/usr/bin/env python3
"""
dry_run_ab.py — Live A/B Bilingual Speech-to-Text Dry Run

Mic → Silero VAD → Distil-Whisper STT → TranslateGemma (4B + 12B) → WebSocket → Browser

Two modes:
  - Parallel (default): Both Gemma models loaded in 4-bit via bitsandbytes
  - Swap (fallback):    One model at a time in fp16 if 4-bit fails on macOS

Usage:
    python dry_run_ab.py                     # Parallel mode (both models in 4-bit)
    python dry_run_ab.py --swap              # Swap mode (one model at a time)
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
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
CHUNK_DURATION = 3.0        # seconds of speech to accumulate before processing
VAD_THRESHOLD = 0.5
WS_PORT = 8765
SWAP_MODE = False
CSV_PATH = f"metrics/ab_metrics_{datetime.now():%Y%m%d_%H%M%S}.csv"

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
gemma_a_model = None
gemma_a_tokenizer = None
gemma_b_model = None
gemma_b_tokenizer = None


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
    """Load Distil-Whisper on MPS with float16."""
    print("[2/4] Loading Distil-Whisper distil-large-v3...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = pipeline(
        "automatic-speech-recognition",
        model="distil-whisper/distil-large-v3",
        device=device,
        torch_dtype=torch.float16,
    )
    print(f"  Whisper ready on {device}")
    return pipe


def load_gemma_4bit(model_name, label):
    """Load a TranslateGemma model in 4-bit quantization."""
    print(f"  Loading {model_name} (4-bit)...")
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        print(f"  {label} ready (4-bit)")
        return model, tokenizer
    except Exception as e:
        print(f"  4-bit load failed for {label}: {e}")
        return None, None


def load_gemma_fp16(model_name, label):
    """Load a TranslateGemma model in fp16 (swap mode fallback)."""
    print(f"  Loading {model_name} (fp16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"  {label} ready (fp16)")
    return model, tokenizer


def unload_model(model, label):
    """Unload a model to free memory."""
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    import gc
    gc.collect()
    print(f"  Unloaded {label}")


def load_gemma_parallel():
    """Load both Gemma models in 4-bit for parallel mode."""
    print("[3/4] Loading TranslateGemma models (parallel mode)...")
    a_model, a_tok = load_gemma_4bit("google/translategemma-4b-it", "Approach A (4B)")
    b_model, b_tok = load_gemma_4bit("google/translategemma-12b-it", "Approach B (12B)")

    if a_model is None or b_model is None:
        print("  WARNING: 4-bit loading failed. Try --swap mode.")
        if a_model is None:
            print("  Falling back to fp16 for 4B...")
            a_model, a_tok = load_gemma_fp16("google/translategemma-4b-it", "Approach A (4B)")
        if b_model is None:
            print("  Skipping 12B in parallel mode (too large for fp16)")

    return a_model, a_tok, b_model, b_tok


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate(model, tokenizer, text):
    """Translate English to Spanish using TranslateGemma chat template."""
    if model is None or tokenizer is None:
        return "(model not loaded)", 0.0

    messages = [{"role": "user", "content": [
        {"type": "text",
         "source_lang_code": "en",
         "target_lang_code": "es",
         "text": text}
    ]}]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)
    latency_ms = (time.perf_counter() - t0) * 1000

    translation = tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return translation, latency_ms


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


async def process_chunk(audio_data, swap_mode=False):
    """Process a speech chunk: STT → Translate A → Translate B → broadcast."""
    global chunk_id, gemma_a_model, gemma_a_tokenizer, gemma_b_model, gemma_b_tokenizer

    chunk_id += 1
    cid = chunk_id

    # STT
    t0 = time.perf_counter()
    result = stt_pipe(
        {"raw": audio_data, "sampling_rate": SAMPLE_RATE},
        generate_kwargs={"language": "en"},
    )
    stt_latency = (time.perf_counter() - t0) * 1000
    english = result["text"].strip()

    if not english:
        return

    # Translate
    if swap_mode:
        # Swap mode: load/translate/unload one at a time
        print(f"  [#{cid}] Swap: loading 4B...")
        gemma_a_model, gemma_a_tokenizer = load_gemma_fp16(
            "google/translategemma-4b-it", "4B")
        spanish_a, lat_a = translate(gemma_a_model, gemma_a_tokenizer, english)
        unload_model(gemma_a_model, "4B")
        gemma_a_model = None

        print(f"  [#{cid}] Swap: loading 12B...")
        gemma_b_model, gemma_b_tokenizer = load_gemma_fp16(
            "google/translategemma-12b-it", "12B")
        spanish_b, lat_b = translate(gemma_b_model, gemma_b_tokenizer, english)
        unload_model(gemma_b_model, "12B")
        gemma_b_model = None
    else:
        # Parallel mode: both models already loaded
        spanish_a, lat_a = translate(gemma_a_model, gemma_a_tokenizer, english)
        spanish_b, lat_b = translate(gemma_b_model, gemma_b_tokenizer, english)

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
        "timestamp": datetime.now().isoformat(),
    }
    all_results.append(result_data)

    # Terminal output
    print(f"\n{'='*60}")
    print(f"Chunk #{cid} | STT: {stt_latency:.0f}ms | A: {lat_a:.0f}ms | B: {lat_b:.0f}ms")
    print(f"  EN: {english}")
    print(f"  A (4B):  {spanish_a}")
    print(f"  B (12B): {spanish_b}")

    # WebSocket broadcast
    await broadcast(result_data)

    # CSV log
    write_csv_row(result_data)


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
    if ws_clients:
        msg = json.dumps(data)
        await asyncio.gather(
            *[client.send(msg) for client in ws_clients],
            return_exceptions=True,
        )


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
    b_lats = [r["latency_b_ms"] for r in all_results]
    same = sum(1 for r in all_results if r["spanish_a"] == r["spanish_b"])
    a_faster = sum(1 for r in all_results if r["latency_a_ms"] < r["latency_b_ms"])

    print(f"\n{'='*60}")
    print(f"SESSION SUMMARY — {n} chunks")
    print(f"{'='*60}")
    print(f"  STT avg:     {np.mean(stt_lats):.0f}ms (median {np.median(stt_lats):.0f}ms)")
    print(f"  A (4B) avg:  {np.mean(a_lats):.0f}ms (median {np.median(a_lats):.0f}ms)")
    print(f"  B (12B) avg: {np.mean(b_lats):.0f}ms (median {np.median(b_lats):.0f}ms)")
    print(f"  Same output: {same}/{n} ({same/n:.0%})")
    print(f"  A faster:    {a_faster}/{n} ({a_faster/n:.0%})")
    print(f"  CSV saved:   {CSV_PATH}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------

async def audio_loop(swap_mode=False):
    """Main audio capture and processing loop."""
    print("\nListening... (Ctrl+C to stop)\n")

    speech_buffer = np.array([], dtype=np.float32)
    silence_frames = 0
    max_silence_frames = int(0.8 * SAMPLE_RATE / 512)  # ~0.8s of silence triggers processing

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
                await process_chunk(speech_buffer.copy(), swap_mode=swap_mode)
                speech_buffer = np.array([], dtype=np.float32)
                silence_frames = 0


async def main_async(args):
    """Start WebSocket server and audio loop."""
    global vad_model, vad_utils, stt_pipe
    global gemma_a_model, gemma_a_tokenizer, gemma_b_model, gemma_b_tokenizer

    print(f"{'='*60}")
    print(f"  Bilingual A/B Dry Run")
    print(f"  Mode: {'Swap' if args.swap else 'Parallel'}")
    print(f"  WebSocket: ws://localhost:{args.ws_port}")
    print(f"{'='*60}\n")

    # Load models
    vad_model, vad_utils = load_vad()
    stt_pipe = load_whisper()

    if args.swap:
        print("[3/4] Swap mode — models loaded on demand")
    else:
        gemma_a_model, gemma_a_tokenizer, gemma_b_model, gemma_b_tokenizer = \
            load_gemma_parallel()

    print(f"[4/4] Starting WebSocket server on port {args.ws_port}...")
    init_csv()

    # Start WebSocket server
    ws_server = await websockets.serve(ws_handler, "localhost", args.ws_port)
    print(f"  WebSocket ready — open ab_display.html in browser")

    # Run audio loop
    try:
        await audio_loop(swap_mode=args.swap)
    except KeyboardInterrupt:
        pass
    finally:
        print_summary()
        ws_server.close()
        await ws_server.wait_closed()


def main():
    parser = argparse.ArgumentParser(
        description="Live A/B bilingual speech-to-text dry run"
    )
    parser.add_argument("--swap", action="store_true",
                        help="Use swap mode (one model at a time)")
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
