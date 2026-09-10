#!/usr/bin/env python3
"""
dry_run_ab.py — Live A/B Bilingual Speech-to-Text Dry Run

Mic → Silero VAD → Whisper STT → TranslateGemma (4B + 12B) → WebSocket → Browser

Architecture:
  - STT: Whisper via mlx-whisper (Apple Silicon) or faster-whisper (CUDA)
  - Translation: TranslateGemma via mlx-lm (Apple Silicon) or transformers (CUDA)
  - Both 4B and 12B loaded simultaneously (~9GB total on MLX)

Usage:
    python dry_run_ab.py                         # Auto-detect backend, 4B only
    python dry_run_ab.py --ab                    # Both 4B and 12B (A/B parallel)
    python dry_run_ab.py --backend cuda          # Force CUDA backend
    python dry_run_ab.py --no-ab                 # Explicitly disable 12B
    python dry_run_ab.py --low-vram              # MarianMT-only (no Gemma)
    python dry_run_ab.py --dry-run-text "Hello"  # Test pipeline without mic
    python dry_run_ab.py --tts                   # Enable TTS audio output
    python dry_run_ab.py --tts --tts-output wav  # TTS to WAV files
    python dry_run_ab.py --tts --tts-output both # TTS to WebSocket + WAV
    python dry_run_ab.py --chunk-duration 5      # Longer chunks
    python dry_run_ab.py --ws-port 9000          # Different WebSocket port
"""

import os
import sys

# Resolve only the product name before optional acceleration imports. A CPU
# lite launch must not import Torch/MLX just because the host has them installed.
for _index, _argument in enumerate(sys.argv[1:], 1):
    if _argument == "--profile" and _index + 1 < len(sys.argv):
        os.environ["STARK_PROFILE"] = sys.argv[_index + 1]
    elif _argument.startswith("--profile="):
        os.environ["STARK_PROFILE"] = _argument.split("=", 1)[1]
_LITE_IMPORTS = os.environ.get("STARK_PROFILE", "standard").startswith("lite-")

os.environ["NUMBA_THREADING_LAYER"] = "workqueue"  # Prevent numba from loading its own libomp (conflicts with torch's)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Safety net for any remaining libomp duplicates

# Backend detection — MLX for Apple Silicon, CUDA for NVIDIA
# Must come before other imports so we know which inference paths are available.
try:
    if _LITE_IMPORTS:
        raise ImportError("Lite uses CPU/CUDA runtime")
    import mlx.core as _mx  # noqa: F401
    import mlx_lm as _mlx_lm  # noqa: F401

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

import argparse
import asyncio
import atexit
import csv
import http.server
import json
import logging
import multiprocessing
import platform
import queue as queue_module
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import psutil
import sounddevice as sd

try:
    if _LITE_IMPORTS:
        raise ImportError("Lite does not require PyTorch")
    import torch
except ImportError:
    torch = None
import websockets

from settings import settings
from stark_translate.profiles import apply_profile, resolve_profile
from tools.capture_handoff import CaptureHandoff
from tools.isolated_audio import AudioCaptureError
from tools.latency_experiments import LatencyExperiments
from tools.latency_scheduler import ExactTextMemo, LatestSTTWorker
from tools.latency_trace import LatencyTrace
from tools.persistence import PersistenceExecutor
from tools.pipeline_health import PipelineHealth
from tools.pipeline_timing import (
    SAMPLE_COLUMNS,
    TIMING_COLUMNS,
    AudioFrame,
    AudioTimeline,
    CaptureSampleClock,
    CaptureStamp,
    ChunkTiming,
    RenderTracker,
    capture_stamp,
    milliseconds,
)
from tools.preview_candidates import RollingPreview, TranslationCandidate, common_prefix_words

RUNTIME_PROFILE = resolve_profile(settings.profile)
_managed_llama_server = None

# ---------------------------------------------------------------------------
# Structured logging — logger used for VAD diagnostics, pipeline events,
# and WS tracking.  User-facing translation output still uses print().
# File handler is added in main() after SESSION_ID/log-level are resolved.
# ---------------------------------------------------------------------------
logger = logging.getLogger("stark")
logger.setLevel(logging.DEBUG)  # handler levels control actual output
# Console handler (default INFO, overridden by --log-level)
_console_handler = next((h for h in logger.handlers if getattr(h, "_stark_console", False)), None)
if _console_handler is None:
    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler._stark_console = True
    _console_handler.setLevel(logging.INFO)
    _console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_console_handler)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000  # Whisper/VAD require 16kHz
MIC_SAMPLE_RATE = 48000  # Most mics are 48kHz native; will resample to 16kHz
CHUNK_DURATION = 2.0  # seconds of speech — more context = better word accuracy
VAD_THRESHOLD = 0.3  # Lower threshold for better sensitivity
WS_PORT = 8765
MIC_DEVICE = None  # None = auto-detect best input device
# Session paths — set in main() after SOURCE_LANG is resolved so the language
# tag is included, keeping EN and ES data separate.
SESSION_ID = f"{datetime.now():%Y%m%d_%H%M%S}"
CSV_PATH = f"metrics/ab_metrics_{SESSION_ID}.csv"
AUDIO_DIR = f"stark_data/live_sessions/{SESSION_ID}"  # per-chunk WAVs for fine-tuning
DIAG_PATH = f"metrics/diagnostics_{SESSION_ID}.jsonl"  # structured review queue
PARTIALS_PATH = f"metrics/partials_{SESSION_ID}.jsonl"
_session_stop_requested = False
_clean_session_shutdown = False
_session_model_ids = {}
_session_main_task = None
_health = None
_RECORD_AUDIO = True
_capture_handoff = None
# Live diarization (Phase 9.6.1) — off unless --diarize. Daemon is a subprocess.
DIARIZE_ENABLED = False
DIARIZE_MODE = "embed"
DIARIZE_JSONL = ""
DIARIZE_INTERVAL_S = 2.0
_diarize_proc = None
_rolling_window = None
EXIT_AFTER_REPLAY = False
_SESSION_CLOCK_ORIGIN = time.perf_counter()
_utterance_timings: dict[int, ChunkTiming] = {}
_chunk_timings: dict[int, ChunkTiming] = {}
_render_tracker = RenderTracker()
_broadcast_sequence = 0
_speaker_pending: dict[int, tuple[float, float, str | None]] = {}
_warmup_future = None
_translation_active = threading.Event()
_experiment_lock = threading.Lock()
_active_stt_workers = {"partial": 0, "final": 0}
_experiment_counters = {}
_INPUT_AUDIO_HASH = None
_latency = LatencyExperiments()
_latency_trace = LatencyTrace()
_stt_scheduler = None
_caption_delivery = None
_vad_pool = None
_marian_memo = ExactTextMemo(0)
_closed_utterances = set()
_partial_sequence = 0
_partial_emitted_sequence = {}
_partial_source_text = {}
_pause_epochs = {}
_speculative_candidates = {}
_speculation_tasks = set()
_speculation_attempts = {}
_rolling_previews = {}
_incremental_stt = None
NUM_DRAFT_TOKENS = 3  # speculative decoding: 4B drafts tokens for 12B to verify
WORD_TIMESTAMPS = False  # per-word timestamps/confidence (adds ~200-400ms DTW pass)
BEAM_SIZE = 1  # Whisper beam search width: 1=greedy (fastest), 5=default
MULTIPROCESS = False  # separate OS processes for STT and translation
MUSIC_THRESHOLD = 0.15  # RMS threshold for music detection (no speech + high energy)
MUSIC_HOLDOFF = 5.0  # seconds of music-like audio before entering music-hold mode

# Inference backend — resolved in main() from --backend flag
# Values: "mlx" (Apple Silicon), "cuda" (NVIDIA), "cpu" (fallback)
BACKEND = "mlx"

# Language direction — resolved in main() from --lang flag.
# Default: English→Spanish.  --lang es flips to Spanish→English.
SOURCE_LANG = "en"
TARGET_LANG = "es"

# Whisper model IDs
WHISPER_MODEL_TURBO = "mlx-community/whisper-large-v3-turbo"  # Default: faster, 4 decoder layers
WHISPER_MODEL_DISTIL = "wbell7/distil-whisper-large-v3.5-mlx"  # Fallback: if Turbo regresses

# Pipeline thread pool for GPU inference. Initialized in main_async() after
# backend detection:
#   - MLX:  max_workers=2 (MLX >= 0.31.2 thread-local streams — STT∥translation)
#   - CUDA: max_workers=2 (thread-safe — enables true STT/Translation overlap)
_pipeline_pool = None  # type: ThreadPoolExecutor | None


def pipeline_pool_max_workers(backend: str) -> int:
    """Return ThreadPoolExecutor size for STT/translation overlap.

    CUDA has long been thread-safe (2 workers). As of MLX 0.31.2, independent
    models may also run concurrently on separate threads (thread-local streams),
    so Mac matches CUDA. CPU stays at 1.
    """
    if backend in ("cuda", "mlx"):
        return 2
    return 1


# Separate pool for PyTorch-only work (MarianMT). Runs concurrently with the
# MLX pipeline pool — safe because MarianMT uses PyTorch/CPU, not Metal GPU.
_pytorch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pytorch")

# Shared PyTorch lock. Silero VAD and the HF MarianMT partial path both touch
# PyTorch from different threads; concurrent calls cause Metal heap corruption
# on macOS. Imported from engines._locks so MarianHFEngine and the VAD path
# actually share serialization (separate locks before v2026.8).
# CT2 path (MarianCT2Engine) is internally thread-safe and does NOT use this.
from engines._locks import _pytorch_lock

# --- Multiprocess worker state (only used when MULTIPROCESS is True) ---
# When enabled, STT and Translation run in separate OS processes with their
# own Metal contexts, enabling true GPU parallelism between the two.
_stt_worker_proc = None
_stt_worker_conn = None  # multiprocessing.Connection for STT worker IPC
_trans_worker_proc = None
_trans_worker_conn = None  # multiprocessing.Connection for Translation worker IPC
_stt_comm_pool = None  # ThreadPoolExecutor(1) for async STT pipe I/O
_trans_comm_pool = None  # ThreadPoolExecutor(1) for async Translation pipe I/O
_RUN_AB = False  # Whether A/B mode is enabled (needed when models aren't in main process)

# --- TTS state (set up in main_async when --tts is enabled) ---
tts_engine = None  # PiperTTSEngine instance
_tts_pool = None  # ThreadPoolExecutor(1) for ONNX TTS (thread-safe, separate from MLX)
tts_ws_clients: set = set()  # WebSocket clients for TTS audio stream
_tts_chunk_counter = 0  # monotonic counter for TTS audio chunks

# --- Partial/final coordination ---
# When a final is pending, partials are skipped to avoid starving the pipeline.
_final_pending = threading.Event()
_final_pending_utterance_id = (
    None  # which utterance is being finalized (new partials for OTHER utterances pass through)
)
# Track the active partial future so we can cancel it when a final arrives.
_active_partial_future = None
_partial_future_lock = threading.Lock()
# prevent fire-and-forget partial tasks from being garbage-collected (RUF006)
_partial_tasks = set()

# MLX model IDs (4-bit quantized, community-converted) — overridden from settings in main()
MLX_MODEL_A = "mlx-community/translategemma-4b-it-4bit"  # ~2.2GB
MLX_MODEL_B = "mlx-community/translategemma-12b-it-4bit"  # ~6.6GB
# Optional LoRA adapters + TurboQuant (set from CLI / settings in main())
ADAPTER_DIR_A = None  # 4B adapter path
ADAPTER_DIR_B = None  # 12B adapter path (optional)
USE_TURBOQUANT = False
TURBOQUANT_KEY_BITS = 3
TURBOQUANT_VAL_BITS = 4
# Gemma 4 OptiQ + MTS (assistant drafter) — opt-in via --model-family gemma4
MODEL_FAMILY = "gemma4"
MLX_DRAFT_MODEL_ID = None
MLX_DRAFT_MODEL = None
USE_MTS = False
LIVE_MTS_UNAVAILABLE = (
    "Live --mts is unavailable: the supported mlx-lm loader cannot load the Gemma 4 assistant drafter. "
    "MTP remains an offline experiment (docs/mlx_mtp_notes.md); run with --no-mts. "
    "The live pipeline will not silently continue after a drafter failure."
)


def validate_live_mts(requested, disabled=False, configured=False):
    if not disabled and (requested is True or (requested is None and configured)):
        raise ValueError(LIVE_MTS_UNAVAILABLE)


# [P7-1E] Whisper initial_prompt — capped at ~40 words to reduce prefill time.
# Biases decoder toward theological vocabulary that Whisper otherwise
# misrecognizes (e.g. "media" instead of "mediator").
WHISPER_PROMPT_EN = (
    "Sermon at Stark Road Gospel Hall. "
    "Christ Jesus, the Holy Spirit, God the Father. "
    "Atonement, propitiation, mediator, covenant, righteousness, "
    "sanctification, justification, redemption, reconciliation, "
    "repentance, reign, grace, mercy, the Gospel, epistle, apostle, "
    "Scripture, the Lord, the Word of God."
)

WHISPER_PROMPT_ES = (
    "Predicación en la Iglesia Cristiana. "
    "Cristo Jesús, el Espíritu Santo, Dios Padre. "
    "La expiación, la propiciación, el mediador, el pacto, la justicia, "
    "la santificación, la justificación, la redención, la reconciliación, "
    "el arrepentimiento, el reino, la gracia, la misericordia, "
    "el Evangelio, la epístola, el apóstol, la Escritura, el Señor, "
    "la Palabra de Dios."
)

# Default to English; overridden in main() when --lang es is used.
WHISPER_PROMPT = WHISPER_PROMPT_EN

# ---------------------------------------------------------------------------
# Live Testing Diagnostics
# ---------------------------------------------------------------------------

# Misrecognized word → likely correct theological word
# Flagged when the common word appears in STT output during a sermon
HOMOPHONE_FLAGS = {
    "rain": "reign",
    "rein": "reign",
    "media": "mediator",
    "profit": "prophet",
    "alter": "altar",
    "prey": "pray",
    "angles": "angels",
    "piece": "peace",
    "patients": "patience",
    "presents": "presence",
    "council": "counsel",
    "palms": "psalms",
    "boosting": "boasting",
}

# Near-miss corrections: edit-distance-close domain words that aren't true
# homophones but appear as STT artifacts for theological vocabulary.
# "prugot" appeared as a Whisper hallucination suffix on "Daniel" (March 1).
NEAR_MISS_CORRECTIONS = {
    "exhitation": "exaltation",
    "self-exhitation": "self-exaltation",
    "exhaltation": "exaltation",
    "self-exhaltation": "self-exaltation",
    "prugot": "proved God",
    "danic": "Daniel",
}

# Multi-word phrase corrections for STT artifacts that span multiple words.
# "damn you" → "Daniel" is a critical phonetic near-miss: Whisper mishears
# the name Daniel as offensive text (March 1 2026, chunk 229, conf=0.64).
PHRASE_CORRECTIONS = {
    "waver in the beach": "wavering in speech",
    "waver in the speech": "wavering in speech",
    "damn you": "Daniel",
    "damn you,": "Daniel,",
}

# [P7-6B] Theological terms that need TranslateGemma quality — MarianMT alone
# may mistranslate these (e.g. "grace" → "gracia" in humor sense, "covenant" →
# "alianza" instead of "pacto").  Includes HOMOPHONE_FLAGS target words.
THEOLOGICAL_TERMS = frozenset(
    {
        "atonement",
        "propitiation",
        "mediator",
        "covenant",
        "righteousness",
        "sanctification",
        "justification",
        "redemption",
        "reconciliation",
        "repentance",
        "grace",
        "mercy",
        "gospel",
        "epistle",
        "apostle",
        "scripture",
        "reign",
        "salvation",
        "resurrection",
        "crucifixion",
        "baptism",
        "communion",
        "trinity",
        "prophecy",
        "parable",
        # Core names — serve as theological context markers for STT correction
        "christ",
        "jesus",
        "god",
        "lord",
    }
    | frozenset(HOMOPHONE_FLAGS.values())
)


def should_use_marian_only(english: str, stt_confidence: float | None) -> bool:
    """[P7-6B] Decide if an utterance is simple enough for MarianMT-only.

    Returns True when ALL of these hold:
      1. Word count < 8
      2. No theological terms present
      3. STT confidence >= 0.8

    This saves 200-400ms by skipping TranslateGemma on trivial phrases
    like "good morning" or "let's turn to page five".
    """
    policy = settings.translation.routing_policy
    if policy == "off":
        return False
    if policy == "conservative":
        from engines.translation_prompts import conservative_marian_route

        return conservative_marian_route(english, SOURCE_LANG, stt_confidence)
    words = english.lower().split()
    if len(words) >= 8:
        return False
    if stt_confidence is None or stt_confidence < 0.8:
        return False
    if any(w.strip(".,;:!?\"'()") in THEOLOGICAL_TERMS for w in words):
        return False
    return True


# Utterances ending with these words likely got cut mid-phrase
_SPLIT_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "in",
        "to",
        "for",
        "and",
        "but",
        "or",
        "by",
        "with",
        "his",
        "her",
        "their",
        "our",
        "my",
        "your",
        "its",
        "this",
        "that",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "has",
        "have",
        "had",
        "he",
        "she",
        "we",
        "they",
        "it",
    }
)

# Known-good short phrases that get low STT confidence but are correct.
# Used to dampen false alarms in review_priority scoring.
_SHORT_PHRASE_WHITELIST = frozenset(
    {
        "amen",
        "thank you",
        "hallelujah",
        "praise god",
        "praise the lord",
        "yes",
        "no",
        "okay",
        "good morning",
        "good evening",
        "amén",
        "gracias",
        "aleluya",
        "alabado sea dios",
        "buenos días",
        "buenas noches",
    }
)

# ---------------------------------------------------------------------------
# Hallucination suppression — catches phantom text Whisper generates from
# silence, breaths, mic pops, and ambient noise.  Data-driven from analysis
# of March 1 2026 live sessions (49 hallucinations across 611 chunks),
# extended 2026-08 for ES phantoms + whisper-guard-style metric tiers.
# ---------------------------------------------------------------------------

# Phrases Whisper hallucinates from near-silence (its most common training
# data phrases).  Suppressed when confidence is below the paired threshold.
# Multi-word phrases also match as substrings of the full utterance.
_HALLUCINATION_PHRASES: dict[str, float] = {
    "thank you": 0.70,
    "thanks for watching": 0.80,
    "please subscribe": 0.80,
    "bye": 0.50,
    "you": 0.40,
    # Spanish YouTube / silence phantoms
    "gracias": 0.70,
    "gracias por ver": 0.80,
    "gracias por ver este video": 0.80,
    "suscríbete": 0.80,
    "suscribete": 0.80,
    "nos vemos": 0.70,
    "hasta luego": 0.70,
    "subtitulos": 0.80,
    "subtítulos": 0.80,
}

# Minimum utterance duration (seconds).  Chunks at or below this are
# nearly always VAD floor artifacts containing only breaths/clicks.
_MIN_UTTERANCE_DUR = 0.80  # VAD floor is 0.77s; anything <= 0.80 is suspect

# Metric-based hard-drop thresholds (aligned with engines + whisper-guard)
_NO_SPEECH_SUPPRESS = 0.6
_NO_SPEECH_SHORT_DUR = 1.6  # seconds — stricter when utterance is short
_COMPRESSION_RATIO_HARD_DROP = 2.4
_UNIQUE_TOKEN_RATIO_MIN = 0.35  # below → repetitive garbage (longer text only)
_UNIQUE_TOKEN_MIN_WORDS = 8


def _normalize_suppress_text(text: str) -> str:
    """Lowercase + strip trailing punctuation for phrase matching."""
    return text.strip().lower().rstrip(".,!?;:'\"")


def _phrase_matches(normalized: str, phrase: str) -> bool:
    """Exact match always; multi-word phrases also match as substrings."""
    if normalized == phrase:
        return True
    if " " in phrase and phrase in normalized:
        return True
    return False


def _max_segment_metric(segment_meta: list | None, key: str) -> float | None:
    """Return the max numeric value of *key* across STT segment metadata."""
    if not segment_meta:
        return None
    values = []
    for seg in segment_meta:
        if not isinstance(seg, dict):
            continue
        val = seg.get(key)
        if val is not None:
            try:
                values.append(float(val))
            except (TypeError, ValueError):
                continue
    return max(values) if values else None


def _should_suppress(
    text: str,
    confidence: float | None,
    utterance_dur: float | None,
    *,
    no_speech_prob: float | None = None,
    compression_ratio: float | None = None,
) -> str | None:
    """Check if STT output should be suppressed as a hallucination.

    Returns a reason string if suppressed, or None if the text should be kept.

    Tiers:
      0. Hard-drop: compression_ratio > 2.4, or no_speech_prob ≥ 0.6
      1. Known hallucination phrases at low confidence (exact / substring)
      2. Ultra-short VAD-floor chunks at low confidence
      3. Very short text (<=2 words) at very low confidence (<0.30)
      4. Short fragments (<=4 words) at low confidence (<0.50), not whitelisted
      5. Low unique-token ratio on longer repetitive text
    """
    t = text.strip()
    if not t:
        return None

    t_lower = _normalize_suppress_text(t)
    word_count = len(t.split())
    conf = confidence if confidence is not None else 1.0

    # Tier 0a: hard-drop after temperature fallback still left high CR
    if compression_ratio is not None and compression_ratio > _COMPRESSION_RATIO_HARD_DROP:
        return f"compression_ratio={compression_ratio:.2f} > {_COMPRESSION_RATIO_HARD_DROP}"

    # Tier 0b: no_speech_prob — silence phantoms with healthy-looking text
    if no_speech_prob is not None and no_speech_prob >= _NO_SPEECH_SUPPRESS:
        short = utterance_dur is not None and utterance_dur < _NO_SPEECH_SHORT_DUR
        if short or conf < 0.85:
            return f"no_speech_prob={no_speech_prob:.2f} ≥ {_NO_SPEECH_SUPPRESS}" + (
                f" (short dur={utterance_dur:.2f}s)" if short else f" (conf={conf:.2f})"
            )

    # Tier 1: Known hallucination phrases below their confidence threshold
    for phrase, threshold in _HALLUCINATION_PHRASES.items():
        if _phrase_matches(t_lower, phrase) and conf < threshold:
            return f"hallucination phrase {phrase!r} (conf={conf:.2f} < {threshold})"

    # Tier 2: VAD-floor duration chunks (<=0.80s) with low confidence
    # Every 0.77s chunk in March 1 data was a hallucination
    if utterance_dur is not None and utterance_dur <= _MIN_UTTERANCE_DUR and conf < 0.70:
        return f"VAD-floor chunk (dur={utterance_dur:.2f}s, conf={conf:.2f})"

    # Tier 3: Ultra-short text at very low confidence
    if word_count <= 2 and conf < 0.30:
        return f"ultra-short low-conf (words={word_count}, conf={conf:.2f})"

    # Tier 4: Short fragments at low confidence, not in whitelist
    # In March 1 data, every chunk with <=4 words and conf < 0.50 that
    # wasn't a known phrase ("Amen", "Good morning") was garbage text.
    if word_count <= 4 and conf < 0.50 and t_lower not in _SHORT_PHRASE_WHITELIST:
        return f"short fragment not whitelisted (words={word_count}, conf={conf:.2f})"

    # Tier 5: unique-token ratio (whisper-guard L3) — repetitive fluent garbage
    if word_count >= _UNIQUE_TOKEN_MIN_WORDS:
        tokens = t_lower.split()
        unique_ratio = len(set(tokens)) / len(tokens)
        if unique_ratio < _UNIQUE_TOKEN_RATIO_MIN:
            return f"low unique-token ratio ({unique_ratio:.2f} < {_UNIQUE_TOKEN_RATIO_MIN})"

    return None


# Session-level diagnostic accumulators
diag_homophones = []  # [(chunk_id, flagged_word, likely_word, text)]
diag_bad_splits = []  # [(chunk_id, last_word, text)]
diag_marian_diverge = []  # [(chunk_id, marian_text, gemma_text, similarity)]
diag_durations = []  # [(chunk_id, duration_s)]
diag_low_confidence = []  # [(chunk_id, confidence, text)]
diag_empty_stt = []  # [(stage, id, buffer_duration_s)]
diag_force_cuts = []  # [(chunk_id, cut_type, buffer_duration_s, cut_position_s)]
diag_near_misses = []  # [(chunk_id, original_word, correction, match_type, text)]
diag_music_holds = []  # [(start_frame, end_frame, duration_s)]
diag_stt_corrections = []  # [(chunk_id, original, corrected, correction_type)]
partial_translations = {}  # utterance_id → last MarianMT translation
partial_latencies = {}  # utterance_id → {"pt_ms": float}


def check_homophones(cid, text):
    """Flag potential homophone misrecognitions in STT output."""
    words = text.lower().split()
    for w in words:
        clean = w.strip(".,!?;:'\"")
        if clean in HOMOPHONE_FLAGS:
            diag_homophones.append((cid, clean, HOMOPHONE_FLAGS[clean], text))
            print(f"  >> HOMOPHONE: '{clean}' -> maybe '{HOMOPHONE_FLAGS[clean]}'?")


def check_bad_split(cid, text):
    """Flag utterances that end with function words (likely mid-phrase cut)."""
    words = text.strip().split()
    if words:
        last = words[-1].lower().strip(".,!?;:'\"")
        if last in _SPLIT_WORDS:
            diag_bad_splits.append((cid, last, text))
            print(f"  >> BAD SPLIT: ends with '{last}'")


def check_near_miss(cid, text, threshold=0.80):
    """Flag words that are edit-distance-close to theological terms.

    Three checks in order:
      1. Exact match in NEAR_MISS_CORRECTIONS dict.
      2. Multi-word phrase match in PHRASE_CORRECTIONS dict.
      3. Fuzzy match against THEOLOGICAL_TERMS via SequenceMatcher.
    """
    from difflib import SequenceMatcher

    text_lower = text.lower()

    # 1. Multi-word phrase corrections
    for phrase, correction in PHRASE_CORRECTIONS.items():
        if phrase in text_lower:
            diag_near_misses.append((cid, phrase, correction, "phrase", text))
            print(f"  >> NEAR-MISS PHRASE: '{phrase}' -> '{correction}'")

    # 2. Per-word checks
    words = text_lower.split()
    for w in words:
        clean = w.strip(".,!?;:'\"")
        if not clean or clean in HOMOPHONE_FLAGS:
            continue  # already caught by check_homophones

        # Exact near-miss correction
        if clean in NEAR_MISS_CORRECTIONS:
            diag_near_misses.append((cid, clean, NEAR_MISS_CORRECTIONS[clean], "exact", text))
            print(f"  >> NEAR-MISS: '{clean}' -> '{NEAR_MISS_CORRECTIONS[clean]}'")
            continue

        # Fuzzy match against theological terms
        if len(clean) >= 5:  # skip short words to avoid noise
            for term in THEOLOGICAL_TERMS:
                ratio = SequenceMatcher(None, clean, term).ratio()
                if ratio >= threshold and clean != term:
                    diag_near_misses.append((cid, clean, term, "fuzzy", text))
                    print(f"  >> NEAR-MISS (fuzzy {ratio:.0%}): '{clean}' -> '{term}'?")
                    break  # one match per word is enough


def correct_stt_output(text):
    """Context-aware STT correction for theological domain.

    Three correction layers:
      1. PHRASE_CORRECTIONS — always applied (multi-word, unambiguous)
      2. NEAR_MISS_CORRECTIONS — always applied (unambiguous misspellings)
      3. HOMOPHONE_FLAGS — only when theological context is present

    Homophones are context-gated: "It started to rain" must NOT become
    "It started to reign", but "the rain of Christ" (theological context)
    should become "the reign of Christ".

    Returns (corrected_text, corrections_list) where each correction is
    (original, replacement, correction_type).
    """
    corrections = []
    result = text

    # 1. Always apply phrase corrections (multi-word, unambiguous)
    for phrase, replacement in PHRASE_CORRECTIONS.items():
        if phrase in result.lower():
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            result = pattern.sub(replacement, result)
            corrections.append((phrase, replacement, "phrase"))

    # 2. Always apply near-miss corrections (unambiguous misspellings)
    words = result.split()
    for i, w in enumerate(words):
        clean = w.lower().strip(".,!?;:'\"()")
        if clean in NEAR_MISS_CORRECTIONS:
            replacement = NEAR_MISS_CORRECTIONS[clean]
            # Preserve leading case
            if w and w[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            # Preserve trailing punctuation
            stripped = w.rstrip(".,!?;:'\"()")
            trailing = w[len(stripped) :]
            words[i] = replacement + trailing
            corrections.append((clean, NEAR_MISS_CORRECTIONS[clean], "near_miss"))
    result = " ".join(words)

    # 3. Context-gated homophone corrections
    # Only apply if ANY theological term is present in the text
    words_lower = {w.lower().strip(".,!?;:'\"()") for w in result.split()}
    has_theological_context = bool(words_lower & THEOLOGICAL_TERMS)

    if has_theological_context:
        words = result.split()
        for i, w in enumerate(words):
            clean = w.lower().strip(".,!?;:'\"()")
            if clean in HOMOPHONE_FLAGS:
                replacement = HOMOPHONE_FLAGS[clean]
                if w and w[0].isupper():
                    replacement = replacement[0].upper() + replacement[1:]
                stripped = w.rstrip(".,!?;:'\"()")
                trailing = w[len(stripped) :]
                words[i] = replacement + trailing
                corrections.append((clean, HOMOPHONE_FLAGS[clean], "homophone"))
        result = " ".join(words)

    return result, corrections


def check_marian_divergence(cid, marian_text, gemma_text):
    """Compare MarianMT partial vs TranslateGemma final translation."""
    if not marian_text or not gemma_text:
        return
    m_words = set(marian_text.lower().split())
    g_words = set(gemma_text.lower().split())
    if not m_words or not g_words:
        return
    intersection = m_words & g_words
    union = m_words | g_words
    similarity = len(intersection) / len(union) if union else 1.0
    diag_marian_diverge.append((cid, marian_text, gemma_text, round(similarity, 2)))
    if similarity < 0.3:
        print(f"  >> MARIAN/GEMMA divergence: {similarity:.0%} overlap")


def compute_word_stability(partial_text: str, final_text: str) -> float | None:
    """LCS-based word stability: % of partial words preserved in final (sequential).

    Measures how many words from the partial (MarianMT) translation appear
    in the same order in the final (TranslateGemma) translation. Higher
    values mean less jarring visual replacement on screen.
    """
    if not partial_text or not final_text:
        return None
    p_words = partial_text.lower().split()
    f_words = final_text.lower().split()
    if not p_words:
        return None
    # Standard DP LCS — O(n*m), partials are <30 words so trivial
    m, n = len(p_words), len(f_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p_words[i - 1] == f_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return round(dp[m][n] / m, 3)


def _log_stt_drop(stage, uid_or_cid, buffer_duration):
    """Log when Whisper returns empty text for an audio chunk.

    Called from process_partial() and _pipeline_coordinator() to give
    operators visibility into silently dropped audio.
    """
    diag_empty_stt.append((stage, uid_or_cid, round(buffer_duration, 2)))
    print(f"  [DROP] {stage} #{uid_or_cid}: empty STT ({buffer_duration:.1f}s audio)")

    record = {
        "event": "empty_stt_drop",
        "stage": stage,
        "id": uid_or_cid,
        "buffer_duration_s": round(buffer_duration, 2),
        "session": SESSION_ID,
        "timestamp": datetime.now().isoformat(),
    }
    _io_pool.submit(_write_jsonl_record, record)


def _write_jsonl_record(record):
    """Append a JSONL record to the diagnostics file (runs on _io_pool)."""
    os.makedirs(os.path.dirname(DIAG_PATH), exist_ok=True)
    with open(DIAG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def _write_partial_record(record):
    """Persist every emitted partial, independently of last-partial caches."""
    os.makedirs(os.path.dirname(PARTIALS_PATH), exist_ok=True)
    with open(PARTIALS_PATH, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _compute_force_cut(buffer_len, sample_rate, last_silence_boundary, min_cut_s=0.5):
    """Decide where to cut a maxed-out speech buffer.

    Returns (cut_type, split_position) where cut_type is "smart" or "hard".
    A smart cut splits at the last detected silence boundary (if far enough
    into the buffer). A hard cut sends the entire buffer.
    """
    min_samples = int(min_cut_s * sample_rate)
    if last_silence_boundary > min_samples:
        return ("smart", last_silence_boundary)
    return ("hard", buffer_len)


def print_diagnostics():
    """Print live testing diagnostics summary."""
    print(f"\n{'~' * 60}")
    print("DIAGNOSTICS SUMMARY")
    print(f"{'~' * 60}")

    if diag_homophones:
        print(f"\n  Homophone flags: {len(diag_homophones)}")
        for cid, flagged, likely, text in diag_homophones:
            print(f"    #{cid}: '{flagged}' -> '{likely}' in: {text[:60]}")
    else:
        print("\n  Homophone flags: 0 (clean)")

    if diag_near_misses:
        print(f"\n  Near-miss flags: {len(diag_near_misses)}")
        for cid, orig, correction, match_type, text in diag_near_misses[:10]:
            print(f"    #{cid}: '{orig}' -> '{correction}' ({match_type}) in: {text[:60]}")
    else:
        print("\n  Near-miss flags: 0 (clean)")

    if diag_bad_splits:
        print(f"\n  Bad sentence splits: {len(diag_bad_splits)}")
        for cid, last, text in diag_bad_splits[:10]:
            print(f"    #{cid}: ends with '{last}': ...{text[-40:]}")
    else:
        print("\n  Bad sentence splits: 0 (clean)")

    if diag_marian_diverge:
        sims = [s for _, _, _, s in diag_marian_diverge]
        avg_sim = sum(sims) / len(sims)
        low = sum(1 for s in sims if s < 0.3)
        print(f"\n  Marian/Gemma avg similarity: {avg_sim:.0%} ({low} high-divergence)")
        for cid, mt, gt, sim in sorted(diag_marian_diverge, key=lambda x: x[3])[:5]:
            print(f"    #{cid} ({sim:.0%}): MT='{mt[:40]}' | TG='{gt[:40]}'")

    if diag_durations:
        durs = [d for _, d in diag_durations]
        short = sum(1 for d in durs if d < 1.0)
        long_count = sum(1 for d in durs if d > 6.0)
        print(
            f"\n  Utterance durations: avg={sum(durs) / len(durs):.1f}s, "
            f"<1s={short}, >6s={long_count}, "
            f"min={min(durs):.1f}s, max={max(durs):.1f}s"
        )

    if diag_low_confidence:
        print(f"\n  Low confidence chunks: {len(diag_low_confidence)}")
        for cid, conf, text in diag_low_confidence:
            print(f"    #{cid} (conf={conf:.2f}): {text[:60]}")
    else:
        print("\n  Low confidence chunks: 0 (clean)")

    if diag_empty_stt:
        partials = sum(1 for s, _, _ in diag_empty_stt if s == "partial")
        finals = sum(1 for s, _, _ in diag_empty_stt if s == "final")
        total_dur = sum(d for _, _, d in diag_empty_stt)
        print(
            f"\n  Empty STT drops: {len(diag_empty_stt)} ({partials} partial, {finals} final, {total_dur:.1f}s audio)"
        )
        for stage, uid, dur in diag_empty_stt[:10]:
            print(f"    {stage} #{uid}: {dur:.1f}s")
    else:
        print("\n  Empty STT drops: 0 (clean)")

    if diag_force_cuts:
        smart = sum(1 for _, ct, _, _ in diag_force_cuts if ct == "smart")
        hard = sum(1 for _, ct, _, _ in diag_force_cuts if ct == "hard")
        print(f"\n  Force-cuts: {len(diag_force_cuts)} ({smart} smart, {hard} hard)")
        for cid_fc, ct, dur, pos in diag_force_cuts[:10]:
            print(f"    #{cid_fc}: {ct} at {pos:.1f}s of {dur:.1f}s buffer")
    else:
        print("\n  Force-cuts: 0 (clean)")

    if diag_music_holds:
        total_hold = sum(d for _, _, d in diag_music_holds)
        print(f"\n  Music holds: {len(diag_music_holds)} ({total_hold:.1f}s total)")
        for start, end, dur in diag_music_holds[:10]:
            print(f"    frame {start}-{end}: {dur:.1f}s")
    else:
        print("\n  Music holds: 0")

    if diag_stt_corrections:
        print(f"\n  STT corrections: {len(diag_stt_corrections)}")
        for cid_c, orig, fixed, ctype in diag_stt_corrections[:15]:
            print(f"    #{cid_c}: '{orig}' → '{fixed}' ({ctype})")
    else:
        print("\n  STT corrections: 0")

    # Resource usage summary
    snap = get_resource_snapshot()
    print("\n  Resource usage (current):")
    print(f"    Process RAM: {snap['process_ram_gb']:.1f}GB")
    print(
        f"    System RAM:  {snap['ram_used_gb']:.1f}GB / {psutil.virtual_memory().total / (1024**3):.0f}GB ({snap['ram_percent']:.0f}%)"
    )
    print(f"    CPU:         {snap['cpu_percent']:.0f}%")

    # [P7-6C] Pipeline overlap statistics
    if _pipeline_total > 0:
        pct = _pipeline_overlaps / _pipeline_total if _pipeline_total else 0
        print(
            f"\n  [P7-6C] Pipeline overlap: {_pipeline_overlaps}/{_pipeline_total} "
            f"chunks ({pct:.0%}) had STT(N) overlapping Translation(N-1)"
        )
    else:
        print("\n  [P7-6C] Pipeline overlap: no chunks processed")

    print(f"\n  Fine-tuning data: {AUDIO_DIR}/")
    print(f"  Review queue:     {DIAG_PATH}")
    print(f"  Mic gain used:    {MIC_GAIN:.1f}x")
    print(f"{'~' * 60}")


# ---------------------------------------------------------------------------
# Hardware Profile & Resource Monitoring (for portability planning)
# ---------------------------------------------------------------------------


def get_hardware_profile():
    """Capture hardware info for portability analysis."""
    import subprocess

    profile = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "python_version": platform.python_version(),
    }
    # Apple Silicon GPU info
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        import json as _json

        gpu_data = _json.loads(result.stdout)
        gpus = gpu_data.get("SPDisplaysDataType", [])
        if gpus:
            gpu = gpus[0]
            profile["gpu_model"] = gpu.get("sppci_model", "unknown")
            profile["gpu_cores"] = gpu.get("sppci_cores", "unknown")
            profile["metal_support"] = gpu.get("spmetal_supported", "unknown")
    except Exception:
        profile["gpu_model"] = "unknown"
    return profile


def get_resource_snapshot():
    """Capture current CPU/memory usage for a single chunk."""
    mem = psutil.virtual_memory()
    proc = psutil.Process()
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_gb": round(mem.used / (1024**3), 2),
        "ram_percent": mem.percent,
        "process_ram_gb": round(proc.memory_info().rss / (1024**3), 2),
    }


# ---------------------------------------------------------------------------
# HTTP Static Server (serves display pages to phones on LAN)
# ---------------------------------------------------------------------------


def get_local_ip():
    """Get the local network IP for LAN access."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def start_http_server(port, directory):
    """Start a simple HTTP server in a background thread."""
    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(*args, directory=directory, **kwargs)
    server = http.server.HTTPServer(("0.0.0.0", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

audio_queue = asyncio.Queue(maxsize=64)
ws_clients = set()
chunk_id = 0
all_results = []
prev_text = ""  # last chunk's transcription — fed to Whisper as context
_last_final_text = ""  # consecutive duplicate suppression

# Models (set during init)
vad_model = None
vad_utils = None
_vad_provenance = {}
stt_pipe = None
mlx_a_model = None
mlx_a_tokenizer = None
mlx_b_model = None
mlx_b_tokenizer = None
# MarianMT partial translator (TranslationEngine instance). Either
# MarianCT2Engine (CUDA, ~30-50ms) or MarianHFEngine (PyTorch fallback,
# ~50-250ms). Selected by the factory at startup based on
# settings.translation.marian_backend + adapters/marian_ct2/<dir>/active presence.
_marian_engine = None

# [P7-2B] Prompt caches: pre-computed KV caches for the fixed TranslateGemma
# chat template prefix. Reused (via deep copy) on every translation call to
# skip re-prefilling the ~30-40 token template overhead (saves 50-80ms).
mlx_a_prompt_cache = None  # KV cache for 4B model template prefix
mlx_b_prompt_cache = None  # KV cache for 12B model template prefix
# [P7-3B] Pre-tokenized suffix tokens: the closing part of the prompt after
# where the English text is inserted. Tokenized once at startup.
mlx_a_suffix_tokens = None
mlx_b_suffix_tokens = None


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------


def load_vad():
    """Load the installed Silero package's bundled weights without network access."""
    from tools.vad_runtime import load_managed_onnx_vad, load_packaged_vad

    global _vad_provenance
    print("[1/6] Loading packaged Silero VAD...")
    model, utils, _vad_provenance = (
        load_managed_onnx_vad() if RUNTIME_PROFILE.lite else load_packaged_vad(settings.vad.backend)
    )
    print(f"  VAD ready (silero-vad {_vad_provenance['package_version']}, {settings.vad.backend})")
    return model, utils


def _resolve_mlx_stt_backend() -> str:
    """Pick the Mac STT engine: explicit setting wins; ``auto`` → Parakeet for English.

    v2026.13 gate (tools/stt_roundtrip_compare.py, Piper → STT): EN WER 0.6 % vs
    whisper-turbo 0.2 % with equal theological-term recall at 4.5× lower latency, so
    English sessions default to ``parakeet-mlx``. Spanish stayed on whisper-turbo
    (synthetic ES WER 9.3 % vs 6.8 %). ``--stt-backend mlx`` forces Whisper.
    """
    requested = getattr(settings.stt, "backend", "auto")
    if requested != "auto":
        return requested
    if SOURCE_LANG != "en":
        return "mlx"
    try:
        import parakeet_mlx  # noqa: F401  -- optional extra; fall back to Whisper when absent
    except ImportError:
        return "mlx"
    return "parakeet-mlx"


def load_whisper(backend="mlx"):
    """Load Whisper STT model for the given backend.

    MLX backend: uses mlx-whisper with whisper-large-v3-turbo (default)
                 or distil-whisper-large-v3.5 (fallback).
    CUDA backend: uses faster-whisper with large-v3-turbo on GPU.
    CPU backend: uses faster-whisper with large-v3-turbo on CPU.

    Returns:
        MLX: model_id string (mlx_whisper uses it per call)
        CUDA/CPU: faster_whisper.WhisperModel instance
    """
    if backend == "mlx" and _resolve_mlx_stt_backend() == "parakeet-mlx":
        # NVIDIA Parakeet TDT v3 on MLX (multilingual, ~5x cheaper GPU time than
        # whisper-large-v3-turbo). Returned as an STTEngine; _run_stt_mlx and
        # process_partial dispatch on isinstance(stt_pipe, STTEngine).
        from engines.parakeet_mlx_engine import ParakeetMLXEngine

        engine = ParakeetMLXEngine(model_id=settings.stt.parakeet_mlx_model)
        print(f"[2/6] Loading {engine.model_id} (Parakeet MLX)...")
        t0 = time.time()
        engine.load()  # warm forward on the load thread (thread-local streams)
        print(f"  Parakeet ready ({time.time() - t0:.1f}s)")
        return engine

    if backend == "mlx":
        global mlx_whisper  # make available to process_partial / _run_stt
        import mlx.core as mx
        import mlx_whisper

        # [P7-4B] Increased from 100MB to 256MB — allows MLX to keep more intermediate
        # computation results cached in Metal memory, reducing recomputation.
        # With 18GB unified memory and ~11.3GB used by models, plenty of headroom.
        mx.set_cache_limit(256 * 1024 * 1024)

        from engines.model_paths import resolve_model_path

        model_id = resolve_model_path(WHISPER_MODEL_TURBO)
        print(f"[2/6] Loading {model_id} (MLX)...")
        t0 = time.time()
        try:
            # Warm up — first call downloads and compiles the model
            silence = np.zeros(16000, dtype=np.float32)
            mlx_whisper.transcribe(silence, path_or_hf_repo=model_id, condition_on_previous_text=False)
            print(f"  Whisper Turbo ready ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"  Turbo load failed ({e}), falling back to distil...")
            model_id = resolve_model_path(WHISPER_MODEL_DISTIL)
            t0 = time.time()
            silence = np.zeros(16000, dtype=np.float32)
            mlx_whisper.transcribe(silence, path_or_hf_repo=model_id, condition_on_previous_text=False)
            print(f"  Whisper Distil ready ({time.time() - t0:.1f}s)")
        # Materialize cached Whisper weights for pool-thread use (MLX >= 0.31.2).
        if hasattr(mx, "synchronize"):
            mx.synchronize()
        return model_id  # mlx_whisper uses model_id per call, no persistent object

    elif backend in ("cuda", "cpu"):
        from engines.factory import create_stt_engine
        from engines.model_paths import resolve_model_path

        if settings.stt.backend == "hf":
            # Explicit standard HF selection has a distinct artifact/config API.
            model = create_stt_engine(
                backend=backend,
                stt_backend="hf",
                compile_mode=settings.stt.compile_mode,
                warmup_seconds=settings.stt.warmup_seconds,
            )
            model.load()
            return model
        if settings.stt.backend not in ("auto", "faster-whisper"):
            raise ValueError(
                f"STT {settings.stt.backend!r} is unsupported by this CPU/CUDA pipeline; choose faster-whisper or hf"
            )
        requested = settings.stt.whisper_cuda_model
        from stark_translate.profiles import resolve_profile_model

        auto_model = (
            settings.profile == "standard"
            and requested == "large-v3-turbo"
            and "whisper_cuda_model" not in settings.stt.model_fields_set
        )
        # An untouched standard default delegates to the factory's existing
        # active-adapter preference. Explicit overrides, even the stock alias,
        # bypass that preference; Lite always resolves its exact pinned model.
        model_path = (
            None
            if auto_model
            else (
                resolve_profile_model(requested)
                if settings.profile != "standard"
                else resolve_model_path(requested, local_only=settings.stt.local_files_only)
            )
        )
        if model_path is None and not auto_model:
            raise FileNotFoundError(f"Pinned STT model {requested} missing; run setup for {settings.profile}")
        compute = settings.stt.whisper_cuda_compute_type
        if backend == "cpu" and compute == "int8_float16":
            compute = "int8"
        model = create_stt_engine(
            backend=backend,
            model_id=model_path,
            stt_backend=settings.stt.backend,
            compute_type=compute,
            cpu_threads=settings.stt.cpu_threads,
            num_workers=settings.stt.num_workers,
            local_files_only=settings.stt.local_files_only,
            fallback_on_low_conf=settings.stt.fallback_on_low_conf,
            fallback_threshold=settings.stt.fallback_threshold,
            hallucination_threshold=settings.stt.hallucination_threshold,
        )
        print(f"[2/6] Loading {model.model_id} ({backend}, {compute})...")
        model.load()
        return model


def profile_artifact_provenance():
    if not RUNTIME_PROFILE.lite:
        return None
    from engines.model_paths import load_model_manifest
    from stark_translate.profiles import resolve_profile_model

    manifest = load_model_manifest()["models"]
    return {
        key: {
            "repo_id": manifest[key].get("repo_id"),
            "revision": manifest[key].get("revision"),
            "sha256": manifest[key].get("sha256"),
            "path": resolve_profile_model(key),
        }
        for key in sorted(RUNTIME_PROFILE.model_keys())
    }


def load_mlx_gemma(model_id, label, adapter_path=None, model_family: str | None = None):
    """Legacy tuple facade over the shared MLX engine lifecycle."""
    from engines.mlx_engine import MLXGemmaEngine

    engine = MLXGemmaEngine(
        model_id=model_id,
        model_family=MODEL_FAMILY if model_family is None else model_family,
        adapter_path=adapter_path,
        use_prompt_cache=False,
        use_turboquant=USE_TURBOQUANT,
        turboquant_key_bits=TURBOQUANT_KEY_BITS,
        turboquant_val_bits=TURBOQUANT_VAL_BITS,
        terminology_prompt=settings.translation.terminology_prompt,
    )
    print(f"  Loading {label}...")
    engine.load()
    return engine._model, engine._tokenizer


def _build_prompt_cache(model, tokenizer, label):
    """[P7-2B] Pre-compute KV cache for the fixed TranslateGemma chat template prefix.

    The TranslateGemma prompt has a fixed prefix (the chat template + language
    codes) that is identical for every translation. By pre-filling the KV cache
    once at startup, we skip re-computing those ~30-40 tokens on each call.

    Returns (prompt_cache, prefix_token_count, suffix_tokens).
    """
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache

    # Build a prompt with a known marker so we can split prefix/suffix
    marker = "SPLIT_HERE"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "source_lang_code": SOURCE_LANG, "target_lang_code": TARGET_LANG, "text": marker}
            ],
        }
    ]
    full_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    # full_prompt is a list of token IDs
    if isinstance(full_prompt, str):
        full_tokens = tokenizer.encode(full_prompt, add_special_tokens=False)
    else:
        full_tokens = list(full_prompt)

    # Tokenize just the marker to find where it appears
    marker_tokens = tokenizer.encode(marker, add_special_tokens=False)

    # Find marker position in the full token sequence
    marker_len = len(marker_tokens)
    prefix_end = None
    for i in range(len(full_tokens) - marker_len + 1):
        if full_tokens[i : i + marker_len] == marker_tokens:
            prefix_end = i
            break

    if prefix_end is None:
        print(f"  WARNING: Could not locate marker in prompt for {label}, skipping cache")
        return None, 0, []

    prefix_tokens = full_tokens[:prefix_end]
    suffix_tokens = full_tokens[prefix_end + marker_len :]

    if len(prefix_tokens) < 3:
        print(f"  WARNING: Prefix too short ({len(prefix_tokens)} tokens) for {label}, skipping cache")
        return None, 0, suffix_tokens

    # Create the KV cache and pre-fill with the prefix
    prompt_cache = make_prompt_cache(model)
    prompt_array = mx.array(prefix_tokens)

    # Run generate_step with max_tokens=0 to just fill the cache
    for _ in generate_step(
        prompt_array,
        model,
        max_tokens=0,
        prompt_cache=prompt_cache,
    ):
        pass  # max_tokens=0 means no tokens generated, just prefill

    mx.eval([c.state for c in prompt_cache])
    print(f"  {label} prompt cache: {len(prefix_tokens)} prefix tokens cached, {len(suffix_tokens)} suffix tokens")

    return prompt_cache, len(prefix_tokens), suffix_tokens


def load_translation_models(load_b=True):
    """Load TranslateGemma or Gemma 4 model(s) via MLX.

    When ``MODEL_FAMILY == "gemma4"``, loads OptiQ E4B (or CLI override) and
    optionally the assistant drafter for MTS. TranslateGemma prompt cache is
    skipped for Gemma 4 (different chat template).
    """
    global mlx_a_prompt_cache, mlx_b_prompt_cache
    global mlx_a_suffix_tokens, mlx_b_suffix_tokens
    global MLX_DRAFT_MODEL

    if USE_MTS:
        raise RuntimeError(LIVE_MTS_UNAVAILABLE)

    family_label = "Gemma 4" if MODEL_FAMILY == "gemma4" else "TranslateGemma"
    print(f"[3/6] Loading {family_label} models (MLX)...")
    a_model, a_tok = load_mlx_gemma(MLX_MODEL_A, f"Approach A ({MLX_MODEL_A})", adapter_path=ADAPTER_DIR_A)

    # Prompt cache is TranslateGemma-specific (structured lang codes)
    if MODEL_FAMILY == "translategemma":
        mlx_a_prompt_cache, _, mlx_a_suffix_tokens = _build_prompt_cache(a_model, a_tok, "4B")
    else:
        mlx_a_prompt_cache, mlx_a_suffix_tokens = None, None
        print("  Prompt cache skipped (gemma4 instruct path)")

    # Live MTP is explicitly unavailable; offline research uses mlx_spec.
    MLX_DRAFT_MODEL = None

    b_model, b_tok = None, None
    if load_b and MODEL_FAMILY == "translategemma":
        try:
            b_model, b_tok = load_mlx_gemma(MLX_MODEL_B, "Approach B (12B)", adapter_path=ADAPTER_DIR_B)
            mlx_b_prompt_cache, _, mlx_b_suffix_tokens = _build_prompt_cache(b_model, b_tok, "12B")
        except Exception as e:
            print(f"  12B load failed: {e}")
            print("  Running 4B only.")
    elif load_b and MODEL_FAMILY == "gemma4":
        # Gemma 4 A/B: optional E2B as secondary (not TG 12B)
        try:
            from engines.factory import resolve_mlx_translation_model_id

            e2b_id = resolve_mlx_translation_model_id(model_family="gemma4", size="e2b")
            b_model, b_tok = load_mlx_gemma(e2b_id, f"Approach B ({e2b_id})")
            mlx_b_prompt_cache, mlx_b_suffix_tokens = None, None
        except Exception as e:
            print(f"  Gemma 4 E2B load failed: {e}")
            print("  Running E4B only.")

    return a_model, a_tok, b_model, b_tok


def load_marian():
    """Load the MarianMT partial-translation engine (CT2 or HF fallback).

    Picks ``MarianCT2Engine`` when ``adapters/marian_ct2/<dir>/active/model.bin``
    exists and ctranslate2 imports — typically ~3-5× faster than the HF path.
    Otherwise falls back to ``MarianHFEngine`` (PyTorch).

    Backend choice and compute_type honor settings.translation.marian_backend
    and STARK_TRANSLATE__MARIAN_COMPUTE_TYPE.
    """
    from engines.factory import create_translation_engine

    # Direction is fixed per-session by SOURCE_LANG. Future flip support could
    # cache both engines if settings.translation.marian_eager_both is True.
    target_lang = "en" if SOURCE_LANG == "es" else "es"
    model_id = "Helsinki-NLP/opus-mt-es-en" if SOURCE_LANG == "es" else "Helsinki-NLP/opus-mt-en-es"
    print(f"[4/6] Loading {model_id} (Marian partial translator)...")
    t0 = time.time()
    ct2_path = None
    if RUNTIME_PROFILE.lite:
        from engines.model_paths import resolve_marian_ct2

        ct2_path = resolve_marian_ct2(f"{SOURCE_LANG}-{target_lang}", managed_only=True)
        if ct2_path is None:
            raise FileNotFoundError("Pinned managed Marian CT2 missing; run setup for the selected profile")
    engine = create_translation_engine(
        ct2_path=ct2_path,
        backend=BACKEND,
        engine_type="marian",
        model_id=model_id,
        marian_backend=settings.translation.marian_backend,
        device=None if settings.translation.marian_device == "auto" else settings.translation.marian_device,
        intra_threads=settings.translation.marian_intra_threads,
        compute_type=settings.translation.marian_compute_type,
        max_new_tokens=settings.translation.marian_max_new_tokens,
        warmup_passes=settings.translation.marian_warmup_passes,
        source_lang=SOURCE_LANG,
        target_lang=target_lang,
    )
    engine.load()
    print(f"  Marian ready ({engine.backend}, {time.time() - t0:.1f}s)")
    return engine


def _probe_llamacpp_server(url):
    """Return True if a llama-server is reachable at ``url`` and reports healthy."""
    from engines.llamacpp_engine import LlamaCppEngine

    try:
        probe = LlamaCppEngine(server_url=url)
        probe.load()  # raises RuntimeError if unreachable / unhealthy
        probe.unload()
        return True
    except Exception:
        return False


def _resolve_cuda_engine_choice():
    """Decide whether to use ``llamacpp`` or ``hf`` on CUDA.

    Honors ``settings.cuda.engine`` if set to a forced value. When ``"auto"``
    (default), probes the configured llama-server URL and prefers llamacpp
    when it answers, falls back to HF otherwise. Returns ``"llamacpp"`` or
    ``"hf"``.
    """
    forced = settings.cuda.engine
    if forced == "llamacpp":
        return "llamacpp"
    if forced == "hf":
        return "hf"
    # auto: probe primary URL
    primary = settings.cuda.llamacpp_url
    if _probe_llamacpp_server(primary):
        return "llamacpp"
    return "hf"


def load_cuda_translation_models(load_gemma=True, load_b=False):
    """Load translation models for CUDA backend with VRAM-aware tier detection.

    On CUDA, the engine selection follows ``settings.cuda.engine``:
      - ``llamacpp`` (or ``auto`` + reachable server): instantiate
        ``LlamaCppEngine`` against ``settings.cuda.llamacpp_url`` (and
        ``llamacpp_url_b`` for engine B in A/B mode). Production default
        post-v2026.5; ~5–9× faster than HF NF4 with 4× less VRAM.
      - ``hf`` (or ``auto`` with no server): fall back to
        ``CUDAGemmaStreamingEngine`` (HF + bnb NF4). Requires ~14 GB VRAM
        for Gemma 4 E2B; only fits on 16 GB+ cards.

    HF VRAM tier detection still applies on the HF path:
      - ``full_ab`` (≥15 GB): 4B + 12B (A/B comparison)
      - ``4b_only`` (≥5.5 GB): 4B only
      - ``marian``  (<5.5 GB): MarianMT only

    Returns (model_a, tokenizer_a, model_b, tokenizer_b). For both engine
    paths, model_a/model_b are engine instances (not raw HF models). On the
    llamacpp path, the tokenizers are ``None`` — llama-server tokenizes
    server-side and ``LlamaCppEngine.translate_streaming`` does not require
    a client tokenizer.
    """
    gemma_a, tok_a, gemma_b, tok_b = None, None, None, None

    if not load_gemma:
        return gemma_a, tok_a, gemma_b, tok_b

    engine_choice = _resolve_cuda_engine_choice()
    print(f"  CUDA translation engine: {engine_choice}")

    if engine_choice == "llamacpp":
        from engines.llamacpp_engine import LlamaCppEngine

        primary_url = settings.cuda.llamacpp_url
        print(f"[3/6] Loading llama.cpp primary engine ({primary_url})...")
        try:
            engine_a = LlamaCppEngine(server_url=primary_url, model_family="gemma4")
            engine_a.load()
            gemma_a = engine_a
            print(f"  llama.cpp primary ready (model={engine_a.model_id})")
        except Exception as e:
            print(f"  llama.cpp primary load failed: {e}")
            print("  Falling back to MarianMT-only mode")
            return None, None, None, None

        if load_b:
            secondary_url = settings.cuda.llamacpp_url_b
            print(f"[3b/6] Loading llama.cpp secondary engine ({secondary_url})...")
            try:
                engine_b = LlamaCppEngine(server_url=secondary_url, model_family="gemma4")
                engine_b.load()
                gemma_b = engine_b
                print(f"  llama.cpp secondary ready (model={engine_b.model_id})")
            except Exception as e:
                print(f"  llama.cpp secondary load failed: {e}")
                print("  Continuing with primary-only mode (no A/B)")

        return gemma_a, tok_a, gemma_b, tok_b

    # HF fallback path
    from engines.cuda_engine import CUDAGemmaStreamingEngine, detect_vram_tier

    tier, vram = detect_vram_tier()
    print(f"  CUDA VRAM tier: {tier} ({vram} MB)")

    if tier == "marian":
        print("  Low VRAM — MarianMT-only mode (no TranslateGemma)")
        return gemma_a, tok_a, gemma_b, tok_b

    model_4b_id = settings.translation.cuda_model_4b

    print(f"[3/6] Loading TranslateGemma 4B (CUDA streaming, cache={settings.cuda.use_prompt_cache})...")
    try:
        engine_a = CUDAGemmaStreamingEngine(
            model_id=model_4b_id,
            use_prompt_cache=settings.cuda.use_prompt_cache,
            streaming_batch_size=settings.cuda.streaming_batch_size,
        )
        engine_a.load()
        gemma_a = engine_a
        tok_a = engine_a._tokenizer
        print("  TranslateGemma 4B (CUDA streaming) ready")
    except Exception as e:
        print(f"  Gemma 4B load failed: {e}")
        print("  Falling back to MarianMT-only mode")
        return None, None, None, None

    if load_b and tier == "full_ab":
        model_12b_id = settings.translation.cuda_model_12b
        use_spec = settings.cuda.use_speculative
        print(f"[3b/6] Loading TranslateGemma 12B (CUDA, speculative={use_spec})...")
        try:
            engine_b = CUDAGemmaStreamingEngine(
                model_id=model_12b_id,
                use_prompt_cache=settings.cuda.use_prompt_cache,
                assistant_model_id=model_4b_id if use_spec else None,
                streaming_batch_size=settings.cuda.streaming_batch_size,
            )
            engine_b.load()
            gemma_b = engine_b
            tok_b = engine_b._tokenizer
            print("  TranslateGemma 12B (CUDA) ready")
        except Exception as e:
            print(f"  Gemma 12B load failed: {e}")
            print("  Continuing with 4B-only mode")

    return gemma_a, tok_a, gemma_b, tok_b


# ---------------------------------------------------------------------------
# Multiprocess Workers (--multiprocess flag)
# ---------------------------------------------------------------------------


def _start_workers(run_ab=False):
    """Start STT and Translation worker processes.

    Each worker runs in its own OS process with a dedicated Metal context,
    enabling true GPU parallelism between STT and Translation.
    """
    global _stt_worker_proc, _stt_worker_conn
    global _trans_worker_proc, _trans_worker_conn
    global _stt_comm_pool, _trans_comm_pool

    from workers import stt_worker_main, translation_worker_main

    print("[3/6] Starting multiprocess workers...")

    # --- STT Worker ---
    parent_conn, child_conn = multiprocessing.Pipe()
    _stt_worker_conn = parent_conn
    _stt_worker_proc = multiprocessing.Process(
        target=stt_worker_main,
        args=(child_conn, WHISPER_MODEL_TURBO),
        kwargs={"source_lang": SOURCE_LANG},
        daemon=True,
    )
    _stt_worker_proc.start()
    msg = _stt_worker_conn.recv()
    if msg != "ready":
        raise RuntimeError(f"STT worker failed to start: {msg}")
    print("  STT worker ready (separate process)")

    # --- Translation Worker ---
    model_12b_id = MLX_MODEL_B if run_ab else None
    parent_conn2, child_conn2 = multiprocessing.Pipe()
    _trans_worker_conn = parent_conn2
    _trans_worker_proc = multiprocessing.Process(
        target=translation_worker_main,
        args=(child_conn2, MLX_MODEL_A, model_12b_id, NUM_DRAFT_TOKENS),
        kwargs={
            "source_lang": SOURCE_LANG,
            "target_lang": TARGET_LANG,
            "model_family": MODEL_FAMILY,
            "adapter_path": ADAPTER_DIR_A,
            "adapter_b_path": ADAPTER_DIR_B,
            "terminology_prompt": settings.translation.terminology_prompt,
        },
        daemon=True,
    )
    _trans_worker_proc.start()
    msg = _trans_worker_conn.recv()
    if msg != "ready":
        raise RuntimeError(f"Translation worker failed to start: {msg}")
    print("  Translation worker ready (separate process)")

    # Thread pools for async pipe I/O (run_in_executor bridges async ↔ blocking pipe)
    _stt_comm_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt-comm")
    _trans_comm_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trans-comm")


def _stop_workers():
    """Shut down worker processes gracefully."""
    global _stt_worker_proc, _stt_worker_conn
    global _trans_worker_proc, _trans_worker_conn
    global _stt_comm_pool, _trans_comm_pool

    for label, conn, proc in [
        ("STT", _stt_worker_conn, _stt_worker_proc),
        ("Translation", _trans_worker_conn, _trans_worker_proc),
    ]:
        if conn is not None:
            try:
                conn.send(None)  # shutdown sentinel
            except (BrokenPipeError, OSError):
                pass
        if proc is not None:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                print(f"  {label} worker terminated (did not exit cleanly)")

    if _stt_comm_pool is not None:
        _stt_comm_pool.shutdown(wait=False)
    if _trans_comm_pool is not None:
        _trans_comm_pool.shutdown(wait=False)

    _stt_worker_proc = None
    _stt_worker_conn = None
    _trans_worker_proc = None
    _trans_worker_conn = None
    _stt_comm_pool = None
    _trans_comm_pool = None


def _run_stt_via_worker(audio_data, whisper_prompt):
    """Send STT request to worker process and return result (blocking).

    Matches the return signature of _run_stt():
    (english, stt_latency_ms, stt_confidence, segment_meta, low_conf_words)
    """
    _stt_worker_conn.send(("transcribe", audio_data, whisper_prompt, WORD_TIMESTAMPS, BEAM_SIZE))
    return _stt_worker_conn.recv()


def _run_partial_stt_via_worker(audio_data):
    """Send partial STT request to worker (blocking). Always greedy, no word timestamps.

    Returns (english, stt_lat, confidence, no_speech_prob, compression_ratio)
    or None when STT produced empty text.
    """
    _stt_worker_conn.send(("transcribe", audio_data, _whisper_prompt(), False, 1))
    result = _stt_worker_conn.recv()
    english, stt_lat, conf, segment_meta, _ = result
    if not english:
        return None
    no_speech = _max_segment_metric(segment_meta, "no_speech_prob")
    cr = _max_segment_metric(segment_meta, "compression_ratio")
    return english, stt_lat, conf, no_speech, cr


def _translate_via_worker(english, run_ab=False):
    """Send translation request to worker process and return result (blocking).

    Returns (spanish_a, lat_a, tps_a, spanish_b, lat_b, tps_b).
    """
    _trans_worker_conn.send(("translate", english, run_ab))
    return _trans_worker_conn.recv()


# ---------------------------------------------------------------------------
# Translation Quality Estimation (lightweight, real-time)
# ---------------------------------------------------------------------------

import re

# Common English words unlikely in valid Spanish translations
_EN_WORDS = re.compile(
    r"\b(the|and|of|that|have|for|not|with|you|this|but|his|from|they|"
    r"been|said|each|which|their|will|other|about|many|then|them|these|"
    r"would|could|should|because|into|after|before|between|under|through)\b",
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


# Pop telemetry when a chunk completes; A/B model B and warmups have no chunk id.
_last_gen_stats: dict[int, dict] = {}
_GEN_STAT_FIELDS = {
    "gen_tokens_a": "generated_tokens",
    "prompt_tokens_a": "prompt_tokens",
    "prefill_ms_a": "prefill_ms",
    "ttft_ms_a": "ttft_ms",
    "decode_ms_a": "decode_ms",
    "finish_reason_a": "finish_reason",
    "draft_tokens_a": "draft_tokens",
    "draft_accept_a": "draft_accept_rate",
}
_EXPERIMENT_GEN_FIELDS = {
    "cached_prompt_tokens_a": "cached_prompt_tokens",
    "prompt_cache_prepare_ms_a": "prompt_cache_prepare_ms",
    "generation_lock_wait_ms_a": "generation_lock_wait_ms",
    "prompt_cache_hit_a": "prompt_cache_hit",
}


def _generation_stats(result):
    return {
        column: getattr(result, attr, None) for column, attr in {**_GEN_STAT_FIELDS, **_EXPERIMENT_GEN_FIELDS}.items()
    }


def translate_mlx(
    model, tokenizer, text, draft_model=None, prompt_cache_template=None, suffix_tokens=None, chunk_id=None
):
    """Compatibility tuple facade; generation is shared with MLXGemmaEngine."""
    from engines.mlx_engine import translate_loaded_model

    result = translate_loaded_model(
        model,
        tokenizer,
        text,
        model_family=globals().get("MODEL_FAMILY", "gemma4"),
        source_lang=SOURCE_LANG,
        target_lang=TARGET_LANG,
        draft_model=draft_model,
        num_draft_tokens=NUM_DRAFT_TOKENS,
        prompt_cache_template=prompt_cache_template,
        suffix_tokens=suffix_tokens,
        terminology_prompt=settings.translation.terminology_prompt,
    )
    if chunk_id is not None:
        _last_gen_stats[chunk_id] = _generation_stats(result)
    return result.text, result.latency_ms, result.tokens_per_second


# ---------------------------------------------------------------------------
# Translation (CUDA)
# ---------------------------------------------------------------------------


def translate_cuda_gemma(model, tokenizer, text):
    """Translate using TranslateGemma on CUDA (direction set by SOURCE_LANG/TARGET_LANG).

    Returns (translation, latency_ms, generation_tps).
    """
    if model is None or tokenizer is None:
        return "(model not loaded)", 0.0, 0.0

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "source_lang_code": SOURCE_LANG, "target_lang_code": TARGET_LANG, "text": text}
            ],
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    prompt = prompt.to("cuda")

    input_words = len(text.split())
    max_tok = max(32, int(input_words * 1.8))

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(prompt, max_new_tokens=max_tok, do_sample=False)
    generated = output[0][prompt.shape[1] :]
    result = tokenizer.decode(generated, skip_special_tokens=False)
    latency_ms = (time.perf_counter() - t0) * 1000

    clean = result.split("<end_of_turn>")[0].strip()
    out_tokens = len(generated)
    gen_tps = out_tokens / (latency_ms / 1000) if latency_ms > 0 else 0.0
    return clean, latency_ms, gen_tps


def _translate_cuda_b(engine, text):
    """Translate using a CUDAGemmaStreamingEngine (12B), returning a 3-tuple.

    Used for the B column in A/B mode. Uses non-streaming ``translate()``
    which may include speculative decoding if an assistant model was configured.
    """
    result = engine.translate(text, source_lang=SOURCE_LANG, target_lang=TARGET_LANG)
    return result.text, result.latency_ms, result.tokens_per_second


def translate_cuda_gemma_streaming(engine, text, chunk_id):
    """Translate using CUDAGemmaStreamingEngine with token streaming.

    Pushes partial translations to ``_stream_token_queue`` via the existing
    ``_enqueue_stream_token()`` mechanism (shared with MLX streaming path).

    Args:
        engine:    A ``CUDAGemmaStreamingEngine`` instance (loaded).
        text:      Source text to translate.
        chunk_id:  Chunk ID for WebSocket broadcast correlation.

    Returns:
        ``(translation, latency_ms, generation_tps)``.
    """

    def _token_callback(partial_text, tokens_so_far):
        _enqueue_stream_token(("token", chunk_id, partial_text, tokens_so_far))

    result = engine.translate_streaming(
        text,
        source_lang=SOURCE_LANG,
        target_lang=TARGET_LANG,
        token_callback=_token_callback,
    )
    return result.text, result.latency_ms, result.tokens_per_second


# ---------------------------------------------------------------------------
# [P7-P3-6A] Streaming Translation (MLX)
# ---------------------------------------------------------------------------

# Queue for passing streaming tokens from the synchronous MLX generator
# to the async WebSocket broadcaster. Each item is either:
#   ("token", chunk_id, partial_text, tokens_so_far)  — intermediate token batch
#   ("done", chunk_id, final_text, gen_tps, latency_ms) — generation complete
_stream_token_queue = None  # set to asyncio.Queue in async context
_stream_loop = None  # event loop reference for thread-safe queue access

STREAM_TOKEN_BATCH_SIZE = 3  # Send every N tokens to reduce WebSocket overhead


def _enqueue_stream_token(item):
    """[P7-P3-6A] Thread-safe enqueue: schedule put on the event loop.
    asyncio.Queue is not thread-safe, so we use call_soon_threadsafe
    from the executor thread to schedule the put on the main loop.
    """
    if _stream_token_queue is None or _stream_loop is None:
        return
    try:
        _stream_loop.call_soon_threadsafe(_stream_token_queue.put_nowait, item)
    except Exception:
        pass  # queue full or loop closed — skip this update


def translate_mlx_streaming(model, tokenizer, text, chunk_id, prompt_cache_template=None, suffix_tokens=None):
    """Live streaming facade over the same prepared request as engine calls."""
    from engines.mlx_engine import translate_loaded_model

    def token_callback(partial, tokens):
        if partial:
            _enqueue_stream_token(("token", chunk_id, partial, tokens))

    result = translate_loaded_model(
        model,
        tokenizer,
        text,
        model_family=globals().get("MODEL_FAMILY", "gemma4"),
        source_lang=SOURCE_LANG,
        target_lang=TARGET_LANG,
        prompt_cache_template=prompt_cache_template,
        suffix_tokens=suffix_tokens,
        token_callback=token_callback,
        batch_size=STREAM_TOKEN_BATCH_SIZE,
        terminology_prompt=settings.translation.terminology_prompt,
    )
    _last_gen_stats[chunk_id] = _generation_stats(result)
    return result.text, result.latency_ms, result.tokens_per_second


async def stream_token_broadcaster():
    """[P7-P3-6A] Async task that reads streaming tokens from the queue
    and broadcasts them to WebSocket clients.

    Runs as a background task in the event loop. Exits when it receives None.
    """
    while True:
        item = await _stream_token_queue.get()
        if item is None:
            break

        msg_type = item[0]
        if msg_type == "token":
            _, cid, partial_spanish, tokens_so_far = item
            await broadcast(
                {
                    "type": "translation_stream",
                    "chunk_id": cid,
                    "partial_spanish_a": partial_spanish,
                    "tokens_so_far": tokens_so_far,
                }
            )


# [P7-4A] Pre-warm translation models during silence to keep Metal GPU hot.
# Called once when transitioning from speech to silence — prevents cold-start
# latency when speech resumes after a pause.
_warmup_pending = False  # flag to avoid repeated warmups during sustained silence
_last_warmup_time = 0.0  # perf_counter of last warmup — for periodic re-warming
_WARMUP_INTERVAL = 4.0  # seconds between periodic warmups during sustained silence


def _inference_idle():
    with _experiment_lock:
        stt_running = any(_active_stt_workers.values())
    return not (
        stt_running
        or _translation_active.is_set()
        or _final_pending.is_set()
        or (_pipeline_chunk_queue is not None and not _pipeline_chunk_queue.empty())
        or (_active_partial_future is not None and not _active_partial_future.done())
        or (_stt_scheduler is not None and _stt_scheduler.busy)
    )


def _count_experiment(name):
    with _experiment_lock:
        _experiment_counters[name] = _experiment_counters.get(name, 0) + 1


def _experiment_snapshot():
    with _experiment_lock:
        counters = dict.fromkeys(
            (
                "warmup_requested",
                "warmup_executed",
                "warmup_failed",
                "warmup_suppressed_busy",
                "warmup_suppressed_coalesced",
                "warmup_suppressed_before_run",
                "warmup_suppressed_no_pending",
                "warmup_suppressed_model_lock",
                "partial_suppressed_final_decode",
                "partial_suppressed_final_pending",
                "partial_suppressed_backlog",
                "partial_suppressed_in_flight",
                "partial_suppressed_after_stt",
                "partial_stt_started",
                "partial_stt_finished",
                "final_stt_started",
                "final_stt_finished",
                "partial_emitted",
                "final_marian_routes",
                "final_gemma_requests",
            ),
            0,
        )
        return {**counters, **_experiment_counters}


def _run_tracked_stt(kind, function, *args):
    """Track actual worker lifetime; cancelling its asyncio wrapper is not completion."""
    with _experiment_lock:
        _active_stt_workers[kind] += 1
    _count_experiment(f"{kind}_stt_started")
    try:
        return function(*args)
    finally:
        with _experiment_lock:
            _active_stt_workers[kind] -= 1
        _count_experiment(f"{kind}_stt_finished")


def _schedule_warmup(loop):
    global _warmup_future
    _count_experiment("warmup_requested")
    if settings.translation.idle_warmup_only:
        if not _inference_idle():
            _count_experiment("warmup_suppressed_busy")
            return
        if _warmup_future is not None and not _warmup_future.done():
            _count_experiment("warmup_suppressed_coalesced")
            return
    _warmup_future = loop.run_in_executor(_pipeline_pool, warmup_translation_models)


def warmup_translation_models():
    """Dummy 1-token forward pass on translation models to keep Metal GPU warm.

    Only needed for MLX backend — Metal GPU enters idle power state after
    a few seconds of inactivity, causing cold-start latency on next call.
    CUDA does not have this issue.
    """
    global _warmup_pending, _last_warmup_time
    if settings.translation.idle_warmup_only and not _inference_idle():
        _count_experiment("warmup_suppressed_before_run")
        return
    if not _warmup_pending:
        _count_experiment("warmup_suppressed_no_pending")
        return
    _warmup_pending = False
    _last_warmup_time = time.perf_counter()
    if BACKEND != "mlx":
        return  # warmup only needed for Metal GPU
    try:
        from mlx_lm import generate

        from engines.translation_prompts import build_chat_messages, chat_template_extra_kwargs

        if mlx_a_model is not None and mlx_a_tokenizer is not None:
            from engines.mlx_generation_lock import generation_guard

            with generation_guard(mlx_a_model):
                # A final can become ready after the warmup task was queued.
                if settings.translation.idle_warmup_only and not _inference_idle():
                    _warmup_pending = True
                    _count_experiment("warmup_suppressed_model_lock")
                    return
                family = globals().get("MODEL_FAMILY", "gemma4")
                messages = build_chat_messages(
                    "hello", source_lang=SOURCE_LANG, target_lang=TARGET_LANG, model_family=family
                )
                prompt = mlx_a_tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, **chat_template_extra_kwargs(model_family=family)
                )
                generate(mlx_a_model, mlx_a_tokenizer, prompt=prompt, max_tokens=1, verbose=False)
                _count_experiment("warmup_executed")
    except Exception:
        _count_experiment("warmup_failed")
        pass  # warmup is best-effort, never block the pipeline


def _latency_event(event, **fields):
    _count_experiment(event)
    _latency_trace.record(event, **fields)


def translate_marian(text):
    """Fast partial translation via the configured Marian engine.

    Returns ``(text, latency_ms)`` to preserve the historical tuple contract
    that ``process_partial`` and the dry-run path consume. Engine internally
    handles CT2 vs HF, threading, lock acquisition.
    """
    if _marian_engine is None:
        return "(MarianMT not loaded)", 0.0
    target_lang = "en" if SOURCE_LANG == "es" else "es"
    memo_started = time.perf_counter()
    key = (id(_marian_engine), getattr(_marian_engine, "model_id", None), SOURCE_LANG, target_lang, text)
    if _latency.marian_memo:
        cached = _marian_memo.get(key)
        if cached is not None:
            _latency_event("marian_memo_hit")
            return cached, (time.perf_counter() - memo_started) * 1000
        _latency_event("marian_memo_miss")
    result = _marian_engine.translate(text, source_lang=SOURCE_LANG, target_lang=target_lang)
    if _latency.marian_memo:
        _marian_memo.put(key, result.text)
    return result.text, result.latency_ms


# ---------------------------------------------------------------------------
# Audio Processing
# ---------------------------------------------------------------------------


def detect_macbook_mic():
    """Select the system default; explicit idle audio probes measure gain safely.

    Never open a native recording device here: permission prompts can block
    indefinitely. The capture child confirms sample flow after displays start.
    """
    return None, 0.0


MIC_GAIN = 1.0  # Set during mic detection based on measured signal level
TARGET_RMS = 0.08  # Target RMS for speech audio fed to VAD/Whisper

# [P7-5D] Background thread pool for non-blocking I/O (WAV saves, CSV/JSONL writes)
_io_pool = PersistenceExecutor(max_workers=2, max_pending=256)


def audio_callback(indata, frames, time_info, status):
    """Resample from the captured source rate to 16 kHz and push to queue."""
    if status:
        if _health is not None:
            _health.error(
                "audio", "capture_overflow" if str(status).startswith("capture_overflow:") else "capture_status"
            )
        if str(status).startswith("capture_overflow:"):
            _io_pool.record_failure("audio_capture", "samples_dropped")
        print(f"  Audio status: {status}", file=sys.stderr)
    # File and isolated mic stamps carry their actual callback rate. The
    # WebSocket protocol already delivers 16 kHz PCM, regardless of mic settings.
    input_rate = (
        time_info.sample_rate
        if isinstance(time_info, CaptureStamp) and time_info.sample_rate
        else SAMPLE_RATE
        if os.environ.get("STARK_AUDIO_SOURCE") == "ws"
        else MIC_SAMPLE_RATE
    )
    stamp = capture_stamp(frames, input_rate, time_info)
    raw = indata[:, 0].copy()
    # Clamp to [-1, 1] — some mics deliver out-of-range samples that break VAD
    raw = np.clip(raw, -1.0, 1.0)
    # Amplify quiet mic signals
    if MIC_GAIN != 1.0:
        raw = np.clip(raw * MIC_GAIN, -1.0, 1.0)
    # [P7-5A] Use decimate instead of resample — faster for integer factor (48k/16k = 3x)
    if input_rate != SAMPLE_RATE:
        from scipy.signal import decimate

        factor = input_rate // SAMPLE_RATE
        if input_rate % SAMPLE_RATE == 0 and factor > 1:
            raw = decimate(raw, factor, zero_phase=False).astype(np.float32)
        else:
            from scipy.signal import resample

            target_len = int(len(raw) * SAMPLE_RATE / input_rate)
            raw = resample(raw, target_len).astype(np.float32)
    if _health is not None:
        _health.input(float(np.sqrt(np.mean(raw * raw))))
        if _health.paused:
            return
    try:
        audio_queue.put_nowait(AudioFrame(raw, stamp))
    except asyncio.QueueFull:
        _io_pool.record_failure("audio_capture", "input_queue_full")
        if _health is not None:
            _health.error("capture", "input_queue_full")


def is_speech(audio_chunk, model, utils):
    """Check if audio chunk contains speech using Silero VAD (streaming mode).

    TODO: Singing/hymns cause VAD and STT to break — likely high RMS + tonal
    content triggers clipping or VAD confusion. Investigate: lower gain during
    music segments, or use inaSpeechSegmenter to detect music and skip/handle
    differently.
    """
    tensor = audio_chunk if getattr(model, "numpy_input", False) else torch.from_numpy(audio_chunk).float()
    with _pytorch_lock:
        speech_prob = model(tensor, SAMPLE_RATE).item()
    return speech_prob > VAD_THRESHOLD


# [P7-5B] Dedicated VAD thread — moves Silero inference off the main async
# event loop to reduce jitter in audio processing. Audio frames are pushed
# into _vad_in_q by the main loop; results (frame, has_speech) are read
# from _vad_out_q. This decouples VAD latency spikes from audio capture.
_vad_in_q = queue_module.Queue(maxsize=64)
_vad_out_q = queue_module.Queue(maxsize=64)
_vad_thread = None
_vad_thread_stop = threading.Event()


def _vad_worker():
    """[P7-5B] Background thread running VAD inference in a tight loop."""
    while not _vad_thread_stop.is_set():
        try:
            audio_chunk = _vad_in_q.get(timeout=0.1)
        except queue_module.Empty:
            continue
        has_speech = is_speech(audio_chunk, vad_model, vad_utils)
        _vad_out_q.put((audio_chunk, has_speech))


def start_vad_thread():
    """[P7-5B] Start the dedicated VAD worker thread."""
    global _vad_thread
    _vad_thread_stop.clear()
    _vad_thread = threading.Thread(target=_vad_worker, daemon=True, name="vad-worker")
    _vad_thread.start()
    print("  VAD thread started")


def stop_vad_thread():
    """[P7-5B] Stop the dedicated VAD worker thread."""
    _vad_thread_stop.set()
    if _vad_thread is not None:
        _vad_thread.join(timeout=2.0)


def _whisper_prompt():
    """Build Whisper prompt: theological vocab + previous transcription.
    [P7-1E] prev_text capped at 100 chars to reduce Whisper prefill time.
    """
    if prev_text:
        return WHISPER_PROMPT + " " + prev_text[-100:]
    return WHISPER_PROMPT


# Regex: same 2-4 character sequence repeating 4+ times in a row (no spaces)
_REPEATING_NGRAM_RE = re.compile(r"(.{2,4})\1{3,}")


def _is_garbage_text(text: str) -> bool:
    """Detect hallucinated/garbage STT output.

    Catches repetitive patterns like 'gagagagagaga' or 'aaaaaaa' that Whisper
    sometimes emits, plus low unique-token-ratio loops on longer utterances.
    Returns True if text looks like garbage.
    """
    t = text.strip()
    if len(t) < 6:
        return False

    t_lower = t.lower()

    # Repeating single character runs (5+ of the same char in a row)
    for ch in set(t_lower):
        if ch != " " and ch * 5 in t_lower:
            return True

    # Repeating 2-4 char n-gram patterns within individual words
    # (e.g. "gagagagaga").  Per-word check avoids false positives from
    # legitimate repeated words like "no no no no".
    words = t_lower.split()
    for word in words:
        if _REPEATING_NGRAM_RE.search(word):
            return True

    # Unique-token ratio on longer text (whisper-guard L3)
    if len(words) >= _UNIQUE_TOKEN_MIN_WORDS:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < _UNIQUE_TOKEN_RATIO_MIN:
            return True

    return False


def _translation_identity(text):
    return (
        SESSION_ID,
        SOURCE_LANG,
        TARGET_LANG,
        id(mlx_a_model),
        id(mlx_a_tokenizer),
        MODEL_FAMILY,
        ADAPTER_DIR_A,
        settings.translation.terminology_prompt,
        text,
    )


async def _speculate_pause(audio_data, utterance_id, pause_epoch, sample_bounds):
    """Start optional work early; ordinary final STT must confirm exact source.

    No generated token is broadcast, spoken or saved as a final here. A resumed
    utterance invalidates the result; running model work still drains normally.
    """
    if (
        BACKEND != "mlx"
        or MULTIPROCESS
        or mlx_a_model is None
        or mlx_b_model is not None
        or MLX_DRAFT_MODEL is not None
        or MODEL_FAMILY != "gemma4"
        or _stt_scheduler is None
    ):
        _latency_event("speculation_unsupported")
        return
    if (
        _translation_active.is_set()
        or _final_pending.is_set()
        or _stt_scheduler.busy
        or utterance_id not in _partial_source_text
        or _speculation_attempts.get(utterance_id, 0) >= 2
    ):
        _latency_event("speculation_suppressed_busy")
        return
    _speculation_attempts[utterance_id] = _speculation_attempts.get(utterance_id, 0) + 1
    _latency_event("speculation_started", utterance_id=utterance_id)
    try:
        result = await asyncio.wrap_future(
            _stt_scheduler.submit(
                "partial", _run_tracked_stt, "partial", _run_stt, audio_data, _whisper_prompt(), key=utterance_id
            )
        )
        source, _, confidence, segments, _ = result
        if (
            _pause_epochs.get(utterance_id, 0) != pause_epoch
            or utterance_id in _closed_utterances
            or _final_pending.is_set()
            or _pipeline_translation_lock.locked()
            or not source
            or _is_garbage_text(source)
            or _should_suppress(
                source,
                confidence,
                len(audio_data) / SAMPLE_RATE,
                no_speech_prob=_max_segment_metric(segments, "no_speech_prob"),
                compression_ratio=_max_segment_metric(segments, "compression_ratio"),
            )
        ):
            _latency_event("speculation_discarded_before_translate")
            return
        source, _ = correct_stt_output(source)
        if should_use_marian_only(source, confidence):
            _latency_event("speculation_skipped_marian_route")
            return
        identity = _translation_identity(source)
        from engines.mlx_engine import translate_loaded_model

        def generate():
            _latency_trace.record("speculative_generation_started", utterance_id=utterance_id)
            return translate_loaded_model(
                mlx_a_model,
                mlx_a_tokenizer,
                source,
                model_family=MODEL_FAMILY,
                source_lang=SOURCE_LANG,
                target_lang=TARGET_LANG,
                terminology_prompt=settings.translation.terminology_prompt,
            )

        async with _pipeline_translation_lock:
            _translation_active.set()
            try:
                translated = await asyncio.get_running_loop().run_in_executor(_pipeline_pool, generate)
            finally:
                _translation_active.clear()
            if _pause_epochs.get(utterance_id, 0) != pause_epoch:
                _latency_event("speculation_discarded_resumed_speech")
                return
            _speculative_candidates[utterance_id] = TranslationCandidate(
                identity, translated, pause_epoch, dict(sample_bounds)
            )
            _latency_event("speculation_completed", utterance_id=utterance_id, model_ms=translated.latency_ms)
    except asyncio.CancelledError:
        _latency_event("speculation_cancelled_pending")
    except Exception as exc:
        _latency_event("speculation_failed")
        logger.warning("Optional pause speculation failed: %s", exc)


def _confirmed_speculation(utterance_id, text, confidence):
    candidate = _speculative_candidates.pop(utterance_id, None)
    if candidate is None:
        return None
    if should_use_marian_only(text, confidence) or not candidate.confirmed(
        _translation_identity(text), _pause_epochs.get(utterance_id, 0)
    ):
        _latency_event("speculation_confirmation_mismatch")
        return None
    _latency_event("speculation_reused", utterance_id=utterance_id)
    return candidate.result


def _preview_ordering_enabled():
    """Use the same closure policy at final admission and preview publication."""
    return bool(
        _stt_scheduler is not None
        or _latency.latest_partial
        or _latency.incremental_stt != "off"
        or _latency.first_preview_s
        or _latency.speculate_pause_ms
        or _latency.pause_preview_ms
        or _latency.clause_preview_s
    )


async def process_partial(
    audio_data,
    utterance_id,
    captured_end=None,
    captured_start=None,
    sample_bounds=None,
    preview_kind="periodic",
    speech_end=None,
    timing_source="unknown",
):
    """Fast partial: STT (~300ms) + MarianMT (~80ms). Italic in UI.

    [FIX] Partials are skipped when a final is pending to avoid starving
    the MLX pipeline thread. MarianMT runs on a separate PyTorch pool
    to free ~80ms of MLX pool time per partial.
    """
    global _active_partial_future, _partial_sequence
    _partial_sequence += 1
    request_sequence = _partial_sequence
    original_audio_duration = len(audio_data) / SAMPLE_RATE
    rolling_start = 0
    processed_audio_samples = len(audio_data)
    if _latency.incremental_stt == "rolling":
        rolling_start = max(0, len(audio_data) - int(_latency.rolling_window_s * SAMPLE_RATE))
        audio_data = audio_data[rolling_start:]
        processed_audio_samples = len(audio_data)
    sample_bounds = sample_bounds or {name: None for name in SAMPLE_COLUMNS}
    partial_submitted = time.perf_counter()
    if settings.translation.final_aware_partials and _translation_active.is_set():
        _count_experiment("partial_suppressed_final_decode")
        return

    # [FIX] Skip partial if a final is queued for the SAME utterance — finals take priority.
    # Partials for a NEW utterance (different utterance_id) are allowed through.
    if _final_pending.is_set() and utterance_id == _final_pending_utterance_id:
        _count_experiment("partial_suppressed_final_pending")
        print("  [FIX] partial skipped (final pending)", end="\r")
        return

    # [FIX] Skip partial when the pipeline is falling behind.
    # Partials and finals share the MLX/CUDA pipeline pool (max_workers=2).
    # Every partial that runs while finals are queued still adds latency;
    # shed when the chunk queue is already backed up.
    # Threshold of >1: queue=0 (idle) and queue=1 (next chunk just arrived) are
    # normal — partials still run for live UX feedback.  queue>=2 means the
    # pipeline is behind — shed partials so finals can drain the backlog.
    if _pipeline_chunk_queue is not None and _pipeline_chunk_queue.qsize() > 1:
        _count_experiment("partial_suppressed_backlog")
        return

    # [FIX] At most one partial in flight on the pool — a second partial
    # queued behind the first just adds latency with no UX benefit (the
    # first partial's text is already on screen).
    if _stt_scheduler is None and _active_partial_future is not None and not _active_partial_future.done():
        _count_experiment("partial_suppressed_in_flight")
        return

    # [FILTER] Pre-STT RMS energy gate — skip breath sounds and low-energy noise
    speech_rms = float(np.sqrt(np.mean(audio_data**2)))
    if speech_rms < 0.008:
        print(f"  [FILTER] low-energy partial skipped (RMS={speech_rms:.4f})", end="\r")
        return

    try:
        loop = asyncio.get_event_loop()

        # --- Step 1: Whisper STT on the MLX pipeline pool ---
        def _partial_stt():
            nonlocal processed_audio_samples
            t0 = time.perf_counter()
            conf = None
            no_speech = None
            cr = None
            from engines.base import STTEngine

            if _incremental_stt is not None:
                result, processed = _incremental_stt.transcribe(audio_data, utterance_id)
                processed_audio_samples = processed
                _latency_trace.record(
                    "stream_preview_stt",
                    utterance_id=utterance_id,
                    processed_samples=processed,
                    sample_rate=SAMPLE_RATE,
                )
                return result
            if isinstance(stt_pipe, STTEngine):
                res = stt_pipe.transcribe(
                    audio_data,
                    language=SOURCE_LANG,
                    initial_prompt=_whisper_prompt(),
                    word_timestamps=False,
                )
                stt_lat = res.latency_ms
                english = res.text.strip()
                conf = round(min(1.0, max(0.0, res.confidence)), 2) if res.confidence is not None else None
                no_speech = getattr(res, "no_speech_prob", None)
                cr = res.compression_ratio
            elif BACKEND == "mlx":
                result = mlx_whisper.transcribe(
                    audio_data,
                    path_or_hf_repo=stt_pipe,
                    language=SOURCE_LANG,
                    condition_on_previous_text=False,
                    initial_prompt=_whisper_prompt(),
                    word_timestamps=False,
                )
                stt_lat = (time.perf_counter() - t0) * 1000
                english = result["text"].strip()
                segs = result.get("segments") or []
                logprobs = [s["avg_logprob"] for s in segs if "avg_logprob" in s]
                if logprobs:
                    conf = round(min(1.0, max(0.0, 1.0 + (sum(logprobs) / len(logprobs)))), 2)
                nsp = [s["no_speech_prob"] for s in segs if "no_speech_prob" in s]
                if nsp:
                    no_speech = max(nsp)
                crs = [s["compression_ratio"] for s in segs if "compression_ratio" in s]
                if crs:
                    cr = max(crs)
            else:
                # CUDA/CPU: faster-whisper
                segments_gen, _ = stt_pipe.transcribe(
                    audio_data,
                    language=SOURCE_LANG,
                    condition_on_previous_text=False,
                    initial_prompt=_whisper_prompt(),
                    word_timestamps=False,
                )
                segments_list = list(segments_gen)
                stt_lat = (time.perf_counter() - t0) * 1000
                english = " ".join(seg.text.strip() for seg in segments_list).strip()
                logprobs = [getattr(seg, "avg_logprob", None) for seg in segments_list]
                logprobs = [lp for lp in logprobs if lp is not None]
                if logprobs:
                    conf = round(min(1.0, max(0.0, 1.0 + (sum(logprobs) / len(logprobs)))), 2)
                nsp = [getattr(seg, "no_speech_prob", None) for seg in segments_list]
                nsp = [v for v in nsp if v is not None]
                if nsp:
                    no_speech = max(nsp)
                crs = [getattr(seg, "compression_ratio", None) for seg in segments_list]
                crs = [v for v in crs if v is not None]
                if crs:
                    cr = max(crs)
            if not english:
                return None
            return english, stt_lat, conf, no_speech, cr

        # Submit STT and track the future so process_final can cancel it
        if _stt_scheduler is not None:
            function = (lambda: _run_partial_stt_via_worker(audio_data)) if MULTIPROCESS else _partial_stt
            stt_future = asyncio.wrap_future(
                _stt_scheduler.submit("partial", _run_tracked_stt, "partial", function, key=utterance_id)
            )
        elif MULTIPROCESS:
            stt_future = loop.run_in_executor(
                _stt_comm_pool, _run_tracked_stt, "partial", _run_partial_stt_via_worker, audio_data
            )
        else:
            stt_future = loop.run_in_executor(_pipeline_pool, _run_tracked_stt, "partial", _partial_stt)
        with _partial_future_lock:
            _active_partial_future = stt_future

        stt_result = await stt_future
        _latency_trace.record("partial_stt_resumed", utterance_id=utterance_id)

        with _partial_future_lock:
            _active_partial_future = None

        if stt_result is None:
            buf_dur = len(audio_data) / SAMPLE_RATE
            _log_stt_drop("partial", utterance_id, buf_dur)
            return

        # Worker path may still return (english, stt_lat) without metrics
        if len(stt_result) == 2:
            english, stt_latency = stt_result
            stt_confidence = no_speech_prob = compression_ratio = None
        else:
            english, stt_latency, stt_confidence, no_speech_prob, compression_ratio = stt_result

        if _latency.incremental_stt == "rolling":
            rolling = _rolling_previews.setdefault(utterance_id, RollingPreview())
            english, joined = rolling.update(english, rolling_start)
            _latency_event("rolling_preview_joined" if joined else "rolling_preview_unaligned")

        # [FILTER] Suppress garbage/hallucinated text
        if _is_garbage_text(english):
            print(f"  [FILTER] garbage partial suppressed: {english!r}")
            return

        # [FILTER] Suppress phantom hallucinations (confidence plumbed from STT)
        utt_dur = len(audio_data) / SAMPLE_RATE
        suppress_reason = _should_suppress(
            english,
            stt_confidence,
            utt_dur,
            no_speech_prob=no_speech_prob,
            compression_ratio=compression_ratio,
        )
        if suppress_reason:
            print(f"  [FILTER] hallucination partial suppressed: {english!r} — {suppress_reason}")
            return

        # [FIX] Re-check after STT — a final may have arrived while we were running
        if _final_pending.is_set() and utterance_id == _final_pending_utterance_id:
            _count_experiment("partial_suppressed_after_stt")
            print("  [FIX] partial skipped after STT (final pending)", end="\r")
            return

        # --- Step 2: MarianMT on the separate PyTorch pool (frees MLX thread) ---
        spanish, marian_latency = await loop.run_in_executor(_pytorch_pool, translate_marian, english)
        if _preview_ordering_enabled() and (
            utterance_id in _closed_utterances or request_sequence <= _partial_emitted_sequence.get(utterance_id, -1)
        ):
            _latency_event("partial_suppressed_stale_result")
            return
        _partial_emitted_sequence[utterance_id] = request_sequence
        previous_source = _partial_source_text.get(utterance_id, "")
        stable_prefix = common_prefix_words(previous_source, english)
        _partial_source_text[utterance_id] = english
        _latency_event("preview_" + preview_kind)
        total = stt_latency + marian_latency
        _count_experiment("partial_emitted")

        emitted = time.perf_counter()
        preview_metadata = {
            "event_id": f"{SESSION_ID}:partial:{request_sequence}",
            "timing_schema_version": 2,
            "timing_source": timing_source,
            "clock": "session_monotonic",
            **sample_bounds,
            "utterance_id": utterance_id,
            "preview_kind": preview_kind,
            "preview_revision": request_sequence,
            "stable_source_prefix": stable_prefix,
            "caption_delivery_mode": "queued" if _latency.async_captions else "awaited",
            "captured_start_at_ms": (
                round((captured_start - _SESSION_CLOCK_ORIGIN) * 1000, 3) if captured_start is not None else None
            ),
            "captured_end_at_ms": (
                round((captured_end - _SESSION_CLOCK_ORIGIN) * 1000, 3) if captured_end is not None else None
            ),
            "speech_end_at_ms": (
                round((speech_end - _SESSION_CLOCK_ORIGIN) * 1000, 3) if speech_end is not None else None
            ),
        }

        _io_pool.submit(
            _write_partial_record,
            {
                **_session_provenance(),
                **preview_metadata,
                "emitted_at_ms": round((emitted - _SESSION_CLOCK_ORIGIN) * 1000, 3),
                "captured_end_to_partial_ms": milliseconds(emitted, captured_end),
                "speech_start_to_partial_ms": milliseconds(emitted, captured_start),
                "speech_end_to_partial_ms": (
                    milliseconds(emitted, speech_end) if timing_source != "replay_nonrealtime" else None
                ),
                "partial_processing_ms": milliseconds(emitted, partial_submitted),
                "ts": datetime.now().isoformat(),
                "buffer_s": original_audio_duration,
                "processed_audio_s": processed_audio_samples / SAMPLE_RATE,
                "processed_window_offset_samples": rolling_start,
                "processed_window_sample_rate": SAMPLE_RATE,
                "stt_ms": stt_latency,
                "marian_ms": marian_latency,
                "text_en": english if SOURCE_LANG == "en" else spanish,
                "text_es": spanish if SOURCE_LANG == "en" else english,
            },
        )

        # Store for Marian/Gemma divergence comparison and latency logging
        partial_translations[utterance_id] = spanish
        partial_latencies[utterance_id] = {
            "pt_ms": round(marian_latency, 1),
            "stt_ms": round(stt_latency, 1),
        }

        print(f"  partial ({total:.0f}ms, Marian:{marian_latency:.0f}ms): {english} | {spanish}          ", end="\r")
        await broadcast(
            {
                "type": "translation",
                "stage": "partial",
                **preview_metadata,
                "chunk_id": utterance_id,
                "english": english,
                "spanish_a": spanish,
                "spanish_b": None,
                "stt_latency_ms": round(stt_latency, 1),
                "latency_a_ms": round(marian_latency, 1),
                "marian_pt_ms": round(marian_latency, 1),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except asyncio.CancelledError:
        print("  [FIX] partial cancelled (final arrived)", end="\r")
    except Exception as e:
        print(f"\n  ERROR in partial: {e}", file=sys.stderr)
    finally:
        with _partial_future_lock:
            _active_partial_future = None


# ---------------------------------------------------------------------------
# [P7-6C] Pipeline N/N-1 Overlap
# ---------------------------------------------------------------------------
#
# Architecture:
#   Audio loop → _pipeline_submit(audio) → returns immediately
#                    ↓
#   _pipeline_coordinator (async task) picks up chunks in order:
#     1. Submit STT(N) to _pipeline_pool
#     2. While STT(N) runs, Translation(N-1) may still be running — that's fine,
#        they use different models (Whisper vs TranslateGemma)
#     3. When STT(N) completes → broadcast English → submit Translation(N)
#     4. When Translation(N) completes → broadcast Spanish → log results
#
# NOTE: MLX >= 0.31.2 supports independent concurrent GPU ops via thread-local
# streams. The pipeline pool uses max_workers=2 on both MLX and CUDA so
# STT(N) can overlap Translation(N-1). Weights are materialised on the load
# thread (mx.eval / synchronize) before pool handoff. --multiprocess remains
# an optional escape hatch (separate Metal contexts / processes).
#
# The audio loop never blocks on STT or translation, so VAD and audio
# capture continue uninterrupted.

_pipeline_chunk_queue = None  # asyncio.Queue — audio chunks awaiting processing
_pipeline_translation_lock = None  # asyncio.Lock — serialize translations to avoid
# Metal GPU contention between concurrent
# TranslateGemma calls; initialized in main_async

# [P7-6C] Counters for overlap statistics
_pipeline_overlaps = 0  # how many times STT(N) overlapped with Translation(N-1)
_pipeline_total = 0  # total chunks processed through pipeline

# KPI counters: processing success rate (attempted vs completed with breakdown)
_chunks_attempted = 0
_chunks_completed = 0
_chunks_empty_stt = 0
_chunks_hallucination = 0
_chunks_dedup = 0

# KPI: true E2E latency — maps utterance_id → perf_counter at first audio frame
_utterance_start_times: dict[int, float] = {}


def _run_stt(audio_data, whisper_prompt):
    """[P7-6C] Run Whisper STT in the pipeline pool (synchronous).

    Separated from process_final so it can be submitted as a future
    that runs concurrently with translation of the previous chunk.

    Dispatches to mlx-whisper (MLX backend) or faster-whisper (CUDA/CPU backend).

    Returns (english, stt_latency_ms, stt_confidence, segment_meta, low_conf_words).
    """
    from engines.base import STTEngine

    if _incremental_stt is not None:
        _incremental_stt.close()  # owning worker restores attention before full final STT
    if isinstance(stt_pipe, STTEngine):
        return _run_stt_engine(audio_data, whisper_prompt)
    if BACKEND == "mlx":
        return _run_stt_mlx(audio_data, whisper_prompt)
    else:
        return _run_stt_faster_whisper(audio_data, whisper_prompt)


def _run_stt_engine(audio_data, whisper_prompt):
    """Run STT through an ``engines.base.STTEngine`` (e.g. Parakeet-MLX).

    Normalises ``STTResult`` to the 5-tuple the pipeline expects. Engines
    without prompt biasing ignore ``whisper_prompt``.
    """
    res = stt_pipe.transcribe(
        audio_data,
        language=SOURCE_LANG,
        initial_prompt=whisper_prompt,
        word_timestamps=WORD_TIMESTAMPS,
        beam_size=BEAM_SIZE,
    )
    conf = round(min(1.0, max(0.0, res.confidence)), 2) if res.confidence is not None else None
    return res.text.strip(), res.latency_ms, conf, list(res.segments or []), list(res.low_confidence_words or [])


def _run_stt_mlx(audio_data, whisper_prompt):
    """Run STT via mlx-whisper (Apple Silicon / MLX backend).

    Note: mlx-whisper is always greedy (beam search not implemented).
    beam_size is only used by faster-whisper (CUDA backend).
    """
    from engines.base import STTEngine

    if isinstance(stt_pipe, STTEngine):
        return _run_stt_engine(audio_data, whisper_prompt)
    t0 = time.perf_counter()
    result = mlx_whisper.transcribe(
        audio_data,
        path_or_hf_repo=stt_pipe,
        language=SOURCE_LANG,
        condition_on_previous_text=False,
        initial_prompt=whisper_prompt,
        word_timestamps=WORD_TIMESTAMPS,
    )
    stt_latency = (time.perf_counter() - t0) * 1000
    english = result["text"].strip()

    # Extract segment-level metadata for fine-tuning
    stt_confidence = None
    segment_meta = []
    low_conf_words = []
    segments = result.get("segments", [])
    if segments:
        avg_logprobs = []
        for seg in segments:
            meta = {
                "avg_logprob": seg.get("avg_logprob"),
                "no_speech_prob": seg.get("no_speech_prob"),
                "compression_ratio": seg.get("compression_ratio"),
            }
            segment_meta.append(meta)
            if "avg_logprob" in seg:
                avg_logprobs.append(seg["avg_logprob"])
            # Extract per-word confidence
            for w in seg.get("words", []):
                if w.get("probability", 1.0) < 0.5:
                    low_conf_words.append(
                        {
                            "word": w.get("word", ""),
                            "probability": round(w["probability"], 3),
                            "start": w.get("start"),
                            "end": w.get("end"),
                        }
                    )
        if avg_logprobs:
            mean_logprob = sum(avg_logprobs) / len(avg_logprobs)
            stt_confidence = round(min(1.0, max(0.0, 1.0 + mean_logprob)), 2)

    return english, stt_latency, stt_confidence, segment_meta, low_conf_words


def _run_stt_faster_whisper(audio_data, whisper_prompt):
    """Run STT via faster-whisper (CUDA/CPU backend).

    faster-whisper returns a generator of segments with a different format
    than mlx-whisper. This function normalizes the output to match.
    """
    t0 = time.perf_counter()
    # faster-whisper expects numpy float32 array at 16kHz (same as mlx-whisper)
    segments_gen, info = stt_pipe.transcribe(
        audio_data,
        language=SOURCE_LANG,
        condition_on_previous_text=False,
        initial_prompt=whisper_prompt,
        word_timestamps=WORD_TIMESTAMPS,
        beam_size=BEAM_SIZE,
    )

    # Consume the generator to get all segments
    segments_list = list(segments_gen)
    stt_latency = (time.perf_counter() - t0) * 1000

    # Build English text from segments
    english = " ".join(seg.text.strip() for seg in segments_list).strip()

    # Extract segment-level metadata (faster-whisper uses attributes, not dicts)
    stt_confidence = None
    segment_meta = []
    low_conf_words = []
    if segments_list:
        avg_logprobs = []
        for seg in segments_list:
            meta = {
                "avg_logprob": getattr(seg, "avg_logprob", None),
                "no_speech_prob": getattr(seg, "no_speech_prob", None),
                "compression_ratio": getattr(seg, "compression_ratio", None),
            }
            segment_meta.append(meta)
            if meta["avg_logprob"] is not None:
                avg_logprobs.append(meta["avg_logprob"])
            # Extract per-word confidence from faster-whisper Word objects
            for w in seg.words or []:
                prob = getattr(w, "probability", 1.0)
                if prob < 0.5:
                    low_conf_words.append(
                        {
                            "word": getattr(w, "word", ""),
                            "probability": round(prob, 3),
                            "start": getattr(w, "start", None),
                            "end": getattr(w, "end", None),
                        }
                    )
        if avg_logprobs:
            mean_logprob = sum(avg_logprobs) / len(avg_logprobs)
            stt_confidence = round(min(1.0, max(0.0, 1.0 + mean_logprob)), 2)

    return english, stt_latency, stt_confidence, segment_meta, low_conf_words


async def _pipeline_translate_and_finalize(
    cid,
    english,
    stt_latency,
    stt_confidence,
    segment_meta,
    low_conf_words,
    audio_data,
    e2e_start,
    utterance_start=None,
    queue_wait_ms=None,
    timing=None,
):
    """[P7-6C] Run translation and finalization for a chunk.

    This is the second half of what was process_final(). It runs as a
    separate coroutine so the pipeline coordinator can start STT on the
    next chunk while this is still translating.

    The _pipeline_translation_lock ensures only one translation runs at a
    time — multiple concurrent TranslateGemma calls would thrash the Metal
    GPU and actually be slower than sequential.
    """
    global _chunks_completed
    # Every branch below assigns the A-model results, but only the A/B branches
    # assign the B-model ones. The single-model MLX path (the Mac default) then
    # hit UnboundLocalError on `qe_b` at the summary print, so every final failed.
    spanish_a = spanish_b = None
    lat_a = lat_b = tps_a = tps_b = 0.0
    qe_a = qe_b = None
    timing = timing or ChunkTiming(submitted=e2e_start)
    timing.translation_requested = time.perf_counter()
    try:
        async with _pipeline_translation_lock:
            timing.translation_lock_acquired = time.perf_counter()
            _translation_active.set()
            try:
                loop = asyncio.get_event_loop()

                def timed_translate(fn, *args):
                    timing.translation_started = timing.translation_started or time.perf_counter()
                    return fn(*args)

                def submit_translate(pool, fn, *args):
                    _count_experiment("final_marian_routes" if fn is translate_marian else "final_gemma_requests")
                    return loop.run_in_executor(pool, timed_translate, fn, *args)

                async def run_translate(pool, fn, *args):
                    return await submit_translate(pool, fn, *args)

                confirmed = _confirmed_speculation(timing.utterance_id, english, stt_confidence)
                # Only a matching final STT result can commit earlier generation.
                if confirmed is not None:
                    timing.translation_started = time.perf_counter()
                    spanish_a, lat_a, tps_a = confirmed.text, confirmed.latency_ms, confirmed.tokens_per_second
                    _last_gen_stats[cid] = _generation_stats(confirmed)
                    qe_a = qe_score(english, spanish_a)
                # --- Multiprocess path: dispatch to translation worker process ---
                elif not MULTIPROCESS and mlx_a_model is None:
                    # A missing/disabled Gemma always means real Marian finals,
                    # including CPU and --low-vram; never audience placeholder text.
                    spanish_a, lat_a = await run_translate(_pytorch_pool, translate_marian, english)
                    qe_a = qe_score(english, spanish_a)
                elif MULTIPROCESS:
                    # Adaptive routing still works — MarianMT is in the main process
                    if not _RUN_AB and should_use_marian_only(english, stt_confidence):
                        spanish_a, lat_a = await run_translate(_pytorch_pool, translate_marian, english)
                        tps_a = 0.0
                        qe_a = qe_score(english, spanish_a)
                        print("  [P7-6B] ADAPTIVE: MarianMT-only (simple utterance)")
                    else:
                        result = await run_translate(_trans_comm_pool, _translate_via_worker, english, _RUN_AB)
                        spanish_a, lat_a, tps_a, spanish_b, lat_b, tps_b = result
                        qe_a = qe_score(english, spanish_a)
                        if spanish_b:
                            qe_b = qe_score(english, spanish_b)

                # [P7-6B] Adaptive routing: skip TranslateGemma for simple
                # utterances when NOT in A/B mode and NOT on CUDA with Gemma loaded.
                elif (
                    BACKEND != "cuda"
                    and mlx_b_model is None
                    and mlx_a_model is not None
                    and should_use_marian_only(english, stt_confidence)
                ):
                    spanish_a, lat_a = await run_translate(_pytorch_pool, translate_marian, english)
                    tps_a = 0.0
                    qe_a = qe_score(english, spanish_a)
                    print("  [P7-6B] ADAPTIVE: MarianMT-only (simple utterance)")
                elif BACKEND == "cuda" or RUNTIME_PROFILE.final_engine == "llamacpp":
                    # CUDA backend: full feature parity — streaming, A/B, adaptive routing.
                    # CUDA is thread-safe, so task_a and task_b run truly concurrently
                    # on the 2-worker pipeline pool (unlike MLX which serializes).
                    if (
                        mlx_b_model is None
                        and mlx_a_model is not None
                        and should_use_marian_only(english, stt_confidence)
                    ):
                        # Adaptive routing: simple utterance, skip Gemma
                        spanish_a, lat_a = await run_translate(_pytorch_pool, translate_marian, english)
                        tps_a = 0.0
                        qe_a = qe_score(english, spanish_a)
                        print("  [P7-6B] ADAPTIVE: MarianMT-only (simple utterance)")
                    elif mlx_b_model is not None and mlx_a_model is not None:
                        # A/B mode: stream 4B + speculative-decode 12B (truly concurrent)
                        task_a = submit_translate(
                            _pipeline_pool,
                            lambda: timed_translate(translate_cuda_gemma_streaming, mlx_a_model, english, cid),
                        )
                        task_b = submit_translate(
                            _pipeline_pool,
                            lambda: timed_translate(_translate_cuda_b, mlx_b_model, english),
                        )
                        spanish_a, lat_a, tps_a = await task_a
                        qe_a = qe_score(english, spanish_a)
                        await broadcast(
                            {
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
                            }
                        )
                        spanish_b, lat_b, tps_b = await task_b
                        qe_b = qe_score(english, spanish_b) if spanish_b else None
                    elif mlx_a_model is not None:
                        # 4B-only: streaming translation
                        spanish_a, lat_a, tps_a = await run_translate(
                            _pipeline_pool,
                            lambda: timed_translate(translate_cuda_gemma_streaming, mlx_a_model, english, cid),
                        )
                        qe_a = qe_score(english, spanish_a)
                    else:
                        # Low-VRAM: MarianMT only
                        spanish_a, lat_a = await run_translate(_pytorch_pool, translate_marian, english)
                        tps_a = 0.0
                        qe_a = qe_score(english, spanish_a)

                elif mlx_b_model is not None:
                    # [P7-P3-6A] In A/B mode, stream 4B translation while 12B runs
                    # without partial broadcasts for model B.
                    task_a = submit_translate(
                        _pipeline_pool,
                        lambda: translate_mlx_streaming(
                            mlx_a_model,
                            mlx_a_tokenizer,
                            english,
                            cid,
                            prompt_cache_template=mlx_a_prompt_cache,
                            suffix_tokens=mlx_a_suffix_tokens,
                        ),
                    )
                    # Speculative decoding: 4B model drafts tokens for 12B to verify
                    # Note: prompt cache not used with speculative decoding (incompatible)
                    task_b = submit_translate(
                        _pipeline_pool,
                        lambda: translate_mlx(mlx_b_model, mlx_b_tokenizer, english, draft_model=mlx_a_model),
                    )

                    spanish_a, lat_a, tps_a = await task_a
                    qe_a = qe_score(english, spanish_a)
                    await broadcast(
                        {
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
                        }
                    )

                    spanish_b, lat_b, tps_b = await task_b
                    qe_b = qe_score(english, spanish_b) if spanish_b and spanish_b != "(model not loaded)" else None
                else:
                    # 4B/E4B-only: broadcast partials unless Gemma-4 MTS draft is active.
                    if MLX_DRAFT_MODEL is not None:
                        spanish_a, lat_a, tps_a = await run_translate(
                            _pipeline_pool,
                            lambda: translate_mlx(
                                mlx_a_model,
                                mlx_a_tokenizer,
                                english,
                                draft_model=MLX_DRAFT_MODEL,
                                chunk_id=cid,
                            ),
                        )
                    else:
                        spanish_a, lat_a, tps_a = await run_translate(
                            _pipeline_pool,
                            lambda: translate_mlx_streaming(
                                mlx_a_model,
                                mlx_a_tokenizer,
                                english,
                                cid,
                                prompt_cache_template=mlx_a_prompt_cache,
                                suffix_tokens=mlx_a_suffix_tokens,
                            ),
                        )
                    qe_a = qe_score(english, spanish_a)

            finally:
                _translation_active.clear()
        timing.translation_started = timing.translation_started or timing.translation_lock_acquired
        timing.translation_finished = time.perf_counter()
        now = time.perf_counter()
        e2e_latency = (now - e2e_start) * 1000
        true_e2e_ms = round((now - utterance_start) * 1000, 1) if utterance_start is not None else None
        silence_delay_ms = round((e2e_start - utterance_start) * 1000, 1) if utterance_start is not None else None

        # --- Diagnostics ---
        utterance_dur = len(audio_data) / SAMPLE_RATE
        diag_durations.append((cid, utterance_dur))
        check_homophones(cid, english)
        check_bad_split(cid, english)
        check_near_miss(cid, english)
        if stt_confidence is not None and stt_confidence < 0.5:
            diag_low_confidence.append((cid, stt_confidence, english))
        # Hallucination check
        for seg in segment_meta:
            cr = seg.get("compression_ratio")
            if cr is not None and cr > 2.4:
                print(f"  >> HALLUCINATION: compression_ratio={cr:.1f} (>2.4)")
                break
        # Low-confidence words
        if low_conf_words:
            words_str = ", ".join(f"'{w['word']}'({w['probability']:.0%})" for w in low_conf_words[:5])
            print(f"  >> LOW CONF WORDS: {words_str}")
        # Compare last MarianMT partial against Gemma final + word stability
        last_marian = partial_translations.pop(cid, None)
        if last_marian is None:
            # utterance_id doesn't match chunk_id — try recent entries
            for uid in list(partial_translations.keys()):
                last_marian = partial_translations.pop(uid, None)
        word_stability_pct = None
        if last_marian and spanish_a:
            check_marian_divergence(cid, last_marian, spanish_a)
            word_stability_pct = compute_word_stability(last_marian, spanish_a)

        gen_stats = _last_gen_stats.pop(cid, {})
        gen_str = ""
        for key, label, suffix in (
            ("gen_tokens_a", "tok", ""),
            ("prefill_ms_a", "pre", "ms"),
            ("ttft_ms_a", "ttft", "ms"),
            ("finish_reason_a", "fin", ""),
        ):
            value = gen_stats.get(key)
            if value is not None:
                formatted = f"{value:.0f}" if isinstance(value, (int, float)) else value
                gen_str += f" {label}={formatted}{suffix}"

        conf_str = f" | conf: {stt_confidence:.2f}" if stt_confidence is not None else ""
        qe_str = f" | QE: A={qe_a}"
        if qe_b is not None:
            qe_str = f" | QE: A={qe_a} B={qe_b}"
            print(f"  +{lat_b:.0f}ms B ({tps_b:.0f} t/s): {spanish_b}")
        print(f"  +{lat_a:.0f}ms A ({tps_a:.0f} t/s){gen_str}: {spanish_a}")
        true_e2e_str = f" | utterance-start→processing: {true_e2e_ms:.0f}ms" if true_e2e_ms is not None else ""
        ws_str = f" | ws: {word_stability_pct:.0%}" if word_stability_pct is not None else ""
        print(f"  Processing: {e2e_latency:.0f}ms{true_e2e_str}{conf_str}{qe_str}{ws_str}")

        _chunks_completed += 1

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
            "true_e2e_ms": true_e2e_ms,
            "silence_delay_ms": silence_delay_ms,
            "queue_wait_ms": queue_wait_ms,
            "stt_confidence": stt_confidence,
            "tps_a": round(tps_a, 1),
            "tps_b": round(tps_b, 1),
            "qe_a": qe_a,
            "qe_b": qe_b,
            "word_stability_pct": word_stability_pct,
            "timestamp": datetime.now().isoformat(),
        }
        result_data.update(gen_stats)
        result_data.update(_session_provenance())
        # Phase 9.6.1: speaker lookup is a JSONL read (no models). Rolling-WAV
        # export happens on _io_pool below so this stays off the GPU path.
        utt_start_ts = utt_end_ts = None
        if DIARIZE_ENABLED:
            try:
                utt_start_ts, utt_end_ts = _utterance_wallclock(timing.captured_start or utterance_start, audio_data)
                result_data["speaker"] = _lookup_speaker(utt_start_ts, utt_end_ts)
            except Exception as exc:
                logger.warning("speaker lookup failed: %s", exc)
                result_data["speaker"] = None
        if DIARIZE_ENABLED and utt_start_ts is not None:
            _speaker_pending[cid] = (utt_start_ts, utt_end_ts, result_data.get("speaker"))
        timing.final_ready = time.perf_counter()
        result_data.update(timing.metrics())
        all_results.append(result_data)
        await broadcast(result_data)
        timing.broadcast_finished = time.perf_counter()
        result_data.update(timing.metrics())
        result_data["timing_stages_ms"] = timing.relative_stages(_SESSION_CLOCK_ORIGIN)

        # --- TTS: fire-and-forget synthesis of translated text ---
        if tts_engine and settings.tts.enabled and _tts_pool is not None:
            # Dynamic language: TTS speaks TARGET_LANG (the translated output)
            tts_text = spanish_a  # Use 4B translation (always available)
            tts_lang = TARGET_LANG
            loop = asyncio.get_event_loop()
            _tts_pool.submit(
                _run_tts,
                tts_engine,
                tts_text,
                tts_lang,
                cid,
                settings.tts.output_mode,
                loop,
                timing.speech_end,
                time.perf_counter(),
            )

        # [P7-5D] Move I/O to background threads — prevents disk writes from
        # blocking the main processing loop (saves 10-30ms on the critical path).
        #
        # Extract MarianMT latency BEFORE submitting to pool — both CSV and
        # JSONL writers need it, and .pop() from one would race with the other.
        marian_lat = partial_latencies.pop(cid, None)
        if marian_lat is None:
            for uid in list(partial_latencies.keys()):
                marian_lat = partial_latencies.pop(uid, None)

        _io_pool.submit(write_csv_row, result_data, marian_lat)

        # Save audio + structured diagnostics for fine-tuning pipeline
        resources = get_resource_snapshot()

        def _save_io():
            audio_path = save_chunk_audio(audio_data, cid)
            write_diag_jsonl(
                result_data,
                audio_path,
                segment_meta=segment_meta,
                low_conf_words=low_conf_words,
                resources=resources,
                marian_lat=marian_lat,
            )
            if DIARIZE_ENABLED:
                try:
                    _export_diarize_chunk(audio_data, cid, utt_start_ts, utt_end_ts)
                except Exception as exc:
                    logger.warning("diarize export failed: %s", exc)

        _io_pool.submit(_save_io)

    except Exception as e:
        if _health is not None:
            _health.error("translation", type(e).__name__)
        logger.error("Translation error chunk #%d: %s", cid, e, exc_info=True)
        print(f"\n  ERROR in chunk #{cid} translation: {e}", file=sys.stderr)


async def _pipeline_coordinator():
    """[P7-6C] Async task that coordinates the STT/translation pipeline.

    Reads audio chunks from _pipeline_chunk_queue and processes them with
    overlap: STT for chunk N runs concurrently with translation for chunk N-1.

    Flow per chunk:
      1. Submit STT to _pipeline_pool → get a Future
      2. Immediately start a translation task for the previous chunk (if any)
         as a fire-and-forget asyncio.Task
      3. Await STT completion
      4. Broadcast English text
      5. Start translation (which will run during next chunk's STT)

    The translation lock ensures translations don't overlap with each other
    (they share the same MLX models), but STT can overlap with translation
    freely since they use different models (Whisper vs TranslateGemma).
    """
    global chunk_id, prev_text, _last_final_text, _pipeline_overlaps, _pipeline_total, _final_pending_utterance_id
    global _chunks_attempted, _chunks_completed, _chunks_empty_stt, _chunks_hallucination, _chunks_dedup

    # Track the currently-running translation task so we can measure overlap
    active_translation_task = None
    translation_tasks = []

    while True:
        item = await _pipeline_chunk_queue.get()
        if item is None:
            # Earlier tasks may still be broadcasting/logging after releasing
            # the model lock. Completion evidence must include every finalizer.
            if translation_tasks:
                await asyncio.gather(*translation_tasks)
            break

        audio_data, e2e_start, utterance_start, timing = item
        timing.dequeued = time.perf_counter()
        dequeue_time = time.perf_counter()
        queue_wait_ms = round((dequeue_time - e2e_start) * 1000, 1)
        chunk_id += 1
        cid = chunk_id
        _chunk_timings[cid] = timing
        while len(_chunk_timings) > 2048:
            _chunk_timings.pop(next(iter(_chunk_timings)))
        _pipeline_total += 1
        _chunks_attempted += 1

        # [FILTER] Pre-STT RMS energy gate — skip breath sounds and low-energy noise
        speech_rms = float(np.sqrt(np.mean(audio_data**2)))
        if speech_rms < 0.008:
            print(f"  [FILTER] low-energy final #{cid} skipped (RMS={speech_rms:.4f})")
            _final_pending.clear()
            _final_pending_utterance_id = None
            continue

        try:
            # [P7-6C] Check if translation from previous chunk is still running.
            # If so, STT(N) will overlap with Translation(N-1) — this is the
            # core latency win.
            overlap_detected = active_translation_task is not None and not active_translation_task.done()
            if overlap_detected:
                _pipeline_overlaps += 1
                print(f"  [P7-6C] OVERLAP: STT #{cid} starting while Translation #{cid - 1} still running")

            # --- STT: submit to pipeline pool ---
            loop = asyncio.get_event_loop()
            whisper_prompt = _whisper_prompt()
            timing.stt_requested = time.perf_counter()

            def timed_stt(audio=audio_data, prompt=whisper_prompt, clock=timing):
                clock.stt_started = time.perf_counter()
                try:
                    return _run_tracked_stt("final", _run_stt_via_worker if MULTIPROCESS else _run_stt, audio, prompt)
                finally:
                    clock.stt_finished = time.perf_counter()

            if _stt_scheduler is not None:
                stt_future = asyncio.wrap_future(_stt_scheduler.submit("final", timed_stt, key=timing.utterance_id))
            else:
                stt_future = loop.run_in_executor(_stt_comm_pool if MULTIPROCESS else _pipeline_pool, timed_stt)

            # Await STT completion (translation of N-1 may still be running
            # concurrently in another thread — that's the overlap)
            english, stt_latency, stt_confidence, segment_meta, low_conf_words = await stt_future
            _latency_trace.record(
                "final_stt_resumed",
                chunk_id=cid,
                worker_finish_to_resume_ms=milliseconds(time.perf_counter(), timing.stt_finished),
            )

            # [FIX] Final STT done — allow partials again for the next utterance
            _final_pending.clear()
            _final_pending_utterance_id = None

            if not english:
                buf_dur = len(audio_data) / SAMPLE_RATE
                _log_stt_drop("final", cid, buf_dur)
                _chunks_empty_stt += 1
                continue

            # [FILTER] Suppress garbage/hallucinated text
            if _is_garbage_text(english):
                print(f"  [FILTER] garbage final suppressed: {english!r}")
                _chunks_hallucination += 1
                continue

            # [FILTER] Suppress phantom hallucinations (thank you, VAD-floor, etc.)
            utt_dur = len(audio_data) / SAMPLE_RATE
            suppress_reason = _should_suppress(
                english,
                stt_confidence,
                utt_dur,
                no_speech_prob=_max_segment_metric(segment_meta, "no_speech_prob"),
                compression_ratio=_max_segment_metric(segment_meta, "compression_ratio"),
            )
            if suppress_reason:
                print(f"  [FILTER] hallucination suppressed: {english!r} — {suppress_reason}")
                _chunks_hallucination += 1
                continue

            # [DEDUP] Suppress consecutive identical finals (e.g., repeated "Amen")
            if english.strip().lower() == _last_final_text:
                print(f"  [DEDUP] suppressed consecutive duplicate: {english!r}")
                _chunks_dedup += 1
                continue
            _last_final_text = english.strip().lower()

            # [CORRECT] Apply context-aware STT corrections before translation
            english, stt_corrections = correct_stt_output(english)
            if stt_corrections:
                diag_stt_corrections.extend([(cid, orig, fixed, ctype) for orig, fixed, ctype in stt_corrections])
                for orig, fixed, ctype in stt_corrections:
                    print(f"  >> STT CORRECTED ({ctype}): '{orig}' → '{fixed}'")

            prev_text = english[-100:]  # [P7-1E] capped at 100 chars

            # [P7-P3-6A] Broadcast English immediately + translation_start signal
            print(f"\n{'=' * 60}")
            print(
                f"Chunk #{cid} | STT: {stt_latency:.0f}ms | EN: {english}"
                + (" [overlapped]" if overlap_detected else "")
            )
            await broadcast(
                {
                    "type": "translation_start",
                    "chunk_id": cid,
                    "english": english,
                    "stage": "final",
                    "stt_latency_ms": round(stt_latency, 1),
                    "stt_confidence": stt_confidence,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # --- Start translation as a fire-and-forget task ---
            # STT(N+1) may overlap this translation on the 2-worker pool
            # (CUDA and MLX >= 0.31.2). The translation lock still serializes
            # translate-vs-translate on the same models.
            active_translation_task = asyncio.create_task(
                _pipeline_translate_and_finalize(
                    cid,
                    english,
                    stt_latency,
                    stt_confidence,
                    segment_meta,
                    low_conf_words,
                    audio_data,
                    e2e_start,
                    utterance_start=utterance_start,
                    queue_wait_ms=queue_wait_ms,
                    timing=timing,
                )
            )
            translation_tasks.append(active_translation_task)

        except Exception as e:
            _final_pending.clear()  # [FIX] Don't leave flag stuck on error
            _final_pending_utterance_id = None
            if _health is not None:
                _health.error("stt", type(e).__name__)
            print(f"\n  ERROR in pipeline chunk #{cid}: {e}", file=sys.stderr)

    # Print overlap statistics
    if _pipeline_total > 0:
        print(
            f"\n  [P7-6C] Pipeline stats: {_pipeline_overlaps}/{_pipeline_total} "
            f"chunks overlapped ({_pipeline_overlaps / _pipeline_total:.0%})"
        )


async def pipeline_submit(audio_data, utterance_start=None, timing=None):
    """[P7-6C] Submit audio to the pipeline without blocking the audio loop.

    Called from audio_loop when an utterance is finalized. Returns immediately;
    the _pipeline_coordinator processes the chunk asynchronously.

    Args:
        audio_data: Audio samples for the utterance.
        utterance_start: perf_counter timestamp of first audio frame (for true E2E).
    """
    if _pipeline_chunk_queue is not None:
        submitted = time.perf_counter()
        timing = timing or ChunkTiming()
        timing.submitted = submitted
        await _pipeline_chunk_queue.put((audio_data, submitted, utterance_start, timing))


async def process_final(audio_data, finalized_utterance_id=None):
    """Final STT on full utterance + translation. High quality.

    [P7-6C] Now delegates to the pipeline coordinator for overlapped execution.
    This function returns immediately after submitting the audio chunk,
    allowing the audio loop to continue capturing the next utterance.

    [FIX] Sets _final_pending to suppress partials for the SAME utterance
    and cancels any queued partial future so finals get immediate access
    to the MLX thread. Partials for NEW utterances are allowed through.
    """
    global _final_pending_utterance_id
    # Signal partials for this utterance to stop — finals take priority on the MLX thread
    _final_pending_utterance_id = finalized_utterance_id
    _final_pending.set()
    if finalized_utterance_id is not None:
        for state in (
            _partial_emitted_sequence,
            _partial_source_text,
            _pause_epochs,
            _speculative_candidates,
            _speculation_attempts,
            _rolling_previews,
        ):
            for old_id in list(state):
                if old_id < finalized_utterance_id - 128:
                    state.pop(old_id, None)
        _closed_utterances.difference_update(
            uid for uid in tuple(_closed_utterances) if uid < finalized_utterance_id - 128
        )

    if finalized_utterance_id is not None and _preview_ordering_enabled():
        _closed_utterances.add(finalized_utterance_id)
    if _stt_scheduler is not None:
        _stt_scheduler.cancel_partial(finalized_utterance_id)

    # Cancel any queued (not-yet-started) partial STT future
    with _partial_future_lock:
        if _stt_scheduler is None and _active_partial_future is not None:
            cancelled = _active_partial_future.cancel()
            if cancelled:
                print("  [FIX] cancelled queued partial (final arriving)")

    # Extract utterance start time for true E2E latency measurement
    utterance_start = _utterance_start_times.pop(finalized_utterance_id, None)
    timing = _utterance_timings.pop(finalized_utterance_id, None)
    await pipeline_submit(audio_data, utterance_start=utterance_start, timing=timing)


# ---------------------------------------------------------------------------
# WebSocket Server
# ---------------------------------------------------------------------------


_ws_total_connections = 0
_ws_total_disconnections = 0
_ws_send_failures = 0


def _session_provenance():
    source = os.environ.get("STARK_AUDIO_SOURCE", "mic")
    kind = (
        "synthetic"
        if os.environ.get("STARK_SESSION_KIND") == "synthetic"
        else ("replay" if source == "file" else "live")
    )
    return {
        "session_id": SESSION_ID,
        "source_lang": SOURCE_LANG,
        "target_lang": TARGET_LANG,
        "session_kind": kind,
        "input_audio_path": os.environ.get("STARK_AUDIO_FILE") if source == "file" else None,
        "input_audio_sha256": _INPUT_AUDIO_HASH,
        "audio_source": source,
    }


_display_log_lock = threading.Lock()


def _write_display_record(record):
    path = os.path.join(os.path.dirname(DIAG_PATH), f"display_metrics_{SESSION_ID}.jsonl")
    with _display_log_lock:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as output:
            output.write(json.dumps({"session_id": SESSION_ID, **record}) + "\n")


async def ws_handler(websocket, path=None):
    """Handle new WebSocket connections."""
    global _ws_total_connections
    if not _latency.async_captions:
        ws_clients.add(websocket)
    _ws_total_connections += 1
    logger.info("ws_connect clients=%d total_connections=%d", len(ws_clients), _ws_total_connections)
    print(f"  Browser connected ({len(ws_clients)} client(s))")
    # Send language config so displays can swap their labels
    try:
        await websocket.send(
            json.dumps(
                {
                    "type": "lang_config",
                    "session_id": SESSION_ID,
                    "source_lang": SOURCE_LANG,
                    "target_lang": TARGET_LANG,
                    "source_label": "Español" if SOURCE_LANG == "es" else "English",
                    "target_label": "English" if SOURCE_LANG == "es" else "Español",
                }
            )
        )
    except websockets.ConnectionClosed:
        ws_clients.discard(websocket)
        return
    if _latency.async_captions:
        ws_clients.add(websocket)
    try:
        async for raw in websocket:
            if not isinstance(raw, str) or len(raw) > 2048:
                continue
            try:
                message = json.loads(raw)
            except (ValueError, TypeError):
                continue
            if not isinstance(message, dict) or message.get("type") != "caption_rendered":
                continue
            record = _render_tracker.acknowledge(websocket, message, time.perf_counter())
            if record is not None:
                _io_pool.submit(_write_display_record, record)
    except websockets.ConnectionClosed:
        pass
    finally:
        global _ws_total_disconnections
        ws_clients.discard(websocket)
        if _caption_delivery is not None:
            await _caption_delivery.remove(websocket)
        _ws_total_disconnections += 1
        _render_tracker.disconnect(websocket)
        logger.info("ws_disconnect clients=%d total_disconnections=%d", len(ws_clients), _ws_total_disconnections)
        print(f"  Browser disconnected ({len(ws_clients)} client(s))")


def _caption_before_send(client, data, started, queue_ms):
    _latency_trace.record("caption_send_started", event_id=data.get("event_id"), queue_ms=queue_ms)
    if data.get("type") != "translation":
        return
    timing = _chunk_timings.get(data.get("chunk_id")) if data.get("stage") != "partial" else None

    def preview_reference(name):
        if data.get("stage") != "partial" or data.get("timing_source") == "replay_nonrealtime":
            return None
        value = data.get(name)
        return _SESSION_CLOCK_ORIGIN + value / 1000 if value is not None else None

    _render_tracker.sent(
        client,
        data["event_id"],
        started,
        timing.speech_end if timing and timing.timing_source != "replay_nonrealtime" else None,
        data.get("stage", "complete"),
        {
            "chunk_id": data.get("chunk_id"),
            "utterance_id": data.get("utterance_id", data.get("chunk_id")),
            "preview_kind": data.get("preview_kind"),
            "preview_revision": data.get("preview_revision"),
            "timing_source": data.get("timing_source", timing.timing_source if timing else "unknown"),
            "caption_delivery_mode": data.get("caption_delivery_mode"),
            "caption_queue_wait_ms": round(queue_ms, 3),
            **{name: data.get(name) for name in ("captured_start_at_ms", "captured_end_at_ms", "speech_end_at_ms")},
            **{name: data.get(name) for name in SAMPLE_COLUMNS},
        },
        preview_start=preview_reference("captured_start_at_ms"),
        preview_end=preview_reference("captured_end_at_ms"),
        preview_speech_end=preview_reference("speech_end_at_ms"),
    )


def _caption_failed(client, error):
    global _ws_send_failures
    _ws_send_failures += 1
    ws_clients.discard(client)
    _render_tracker.disconnect(client)
    logger.warning("caption_client_failed: %s", error)


async def broadcast(data):
    """Send data to all connected WebSocket clients."""
    data.setdefault("caption_delivery_mode", "queued" if _latency.async_captions else "awaited")
    if _health is not None and data.get("type") == "translation":
        _health.caption(data)
    if not ws_clients:
        print("  [ws] No clients connected, skipping broadcast")
        return
    global _broadcast_sequence, _caption_delivery
    # Add the correlation id to the producer record too, so diagnostics can
    # be joined to acknowledgments. Never add post-send durations to payload.
    data.setdefault("session_id", SESSION_ID)
    if "event_id" not in data:
        _broadcast_sequence += 1
        data["event_id"] = f"{SESSION_ID}:{_broadcast_sequence}"
    cid = data.get("chunk_id")
    timing = _chunk_timings.get(cid) if data.get("stage") != "partial" else None
    if timing is not None:
        data.setdefault("utterance_id", timing.utterance_id)
        for name, value in timing.sample_metadata().items():
            data.setdefault(name, value)
    if _latency.async_captions:
        from tools.caption_delivery import CaptionDelivery

        if _caption_delivery is None:
            _caption_delivery = CaptionDelivery(
                before_send=_caption_before_send, on_failure=_caption_failed, on_event=_latency_event
            )
        for client in list(ws_clients):
            _caption_delivery.publish(client, data)
        return
    msg = json.dumps(data)
    dead = set()
    clients = list(ws_clients)
    if data.get("type") == "translation":
        for client in clients:
            _caption_before_send(client, data, time.perf_counter(), 0.0)
    results = await asyncio.gather(
        *[client.send(msg) for client in clients],
        return_exceptions=True,
    )
    for client, result in zip(clients, results):
        if isinstance(result, Exception):
            global _ws_send_failures
            _ws_send_failures += 1
            logger.warning("ws_send_failure: %s", result)
            print(f"  [ws] Send failed: {result}")
            dead.add(client)
    ws_clients.difference_update(dead)
    ok = len(clients) - len(dead)
    if ok > 0:
        print(f"  [ws] Sent to {ok} client(s)")


# ---------------------------------------------------------------------------
# TTS Audio WebSocket (binary PCM on separate port from text JSON)
# ---------------------------------------------------------------------------


async def tts_ws_handler(websocket, path=None):
    """Handle TTS audio WebSocket connections (binary PCM stream)."""
    tts_ws_clients.add(websocket)
    print(f"  [tts-ws] Audio client connected ({len(tts_ws_clients)} client(s))")
    try:
        async for _ in websocket:
            pass  # clients only receive, never send
    except websockets.ConnectionClosed:
        pass
    finally:
        tts_ws_clients.discard(websocket)
        print(f"  [tts-ws] Audio client disconnected ({len(tts_ws_clients)} client(s))")


async def broadcast_tts_audio(chunk_id: int, audio: np.ndarray, sample_rate: int):
    """Broadcast TTS audio as binary WebSocket frames.

    Frame format: 8-byte header (chunk_id uint32 LE + sample_rate uint32 LE)
    followed by int16 PCM audio samples.
    """
    if not tts_ws_clients:
        return
    import struct

    header = struct.pack("<II", chunk_id, sample_rate)
    audio_int16 = (audio * 32767).astype(np.int16).tobytes()
    payload = header + audio_int16
    dead = set()
    for client in list(tts_ws_clients):
        try:
            await client.send(payload)
        except Exception:
            dead.add(client)
    tts_ws_clients.difference_update(dead)
    ok = len(tts_ws_clients) - len(dead)
    if ok > 0:
        print(f"  [tts-ws] Sent {len(audio_int16) // 2} samples to {ok} client(s)")


from engines.audio_devices import OutputDeviceResolver

_tts_device_resolver = OutputDeviceResolver()


def _run_tts(engine, text, language, cid, output_mode, loop, speech_end=None, submitted=None):
    """Synthesize TTS and dispatch output (runs on _tts_pool thread).

    This is fire-and-forget — doesn't block the next STT/translation cycle.
    """
    global _tts_chunk_counter

    if not text or not text.strip():
        return

    try:
        started = time.perf_counter()
        tts_result = engine.synthesize(text, language=language)
        synthesized = time.perf_counter()
        playback_started = playback_finished = None
        _tts_chunk_counter += 1

        # WAV file output
        if output_mode in ("wav", "both"):
            import scipy.io.wavfile as wav

            tts_dir = os.path.join(AUDIO_DIR, "tts")
            os.makedirs(tts_dir, exist_ok=True)
            wav_path = os.path.join(tts_dir, f"tts_{cid:04d}.wav")
            audio_int16 = (tts_result.audio * 32767).astype(np.int16)
            wav.write(wav_path, tts_result.sample_rate, audio_int16)
            tts_result.wav_path = wav_path
            print(f"  [tts] WAV saved: {wav_path} ({len(tts_result.audio) / tts_result.sample_rate:.1f}s)")

        # WebSocket stream output
        if output_mode in ("ws", "both"):
            asyncio.run_coroutine_threadsafe(
                broadcast_tts_audio(_tts_chunk_counter, tts_result.audio, tts_result.sample_rate),
                loop,
            )

        # Local sounddevice playback (Phase 9.4.1)
        if output_mode == "local":
            playback_started = time.perf_counter()
            _tts_device_resolver.play(
                engine.play,
                tts_result.audio,
                tts_result.sample_rate,
                language=language,
                spec=settings.tts.output_devices.get(language, settings.tts.output_device),
            )
            playback_finished = time.perf_counter()

        _io_pool.submit(
            _write_jsonl_record,
            {
                "event": "tts_timing",
                "session_id": SESSION_ID,
                "chunk_id": cid,
                "timing_schema_version": 2,
                "output_mode": output_mode,
                "tts_queue_wait_ms": milliseconds(started, submitted),
                "tts_synthesis_ms": milliseconds(synthesized, started),
                "speech_end_to_synthesis_ms": milliseconds(synthesized, speech_end),
                "speech_end_to_playback_request_ms": milliseconds(playback_started, speech_end),
                "playback_call_ms": milliseconds(playback_finished, playback_started),
                "playback_boundary": "host_call_not_acoustic_onset" if playback_started is not None else None,
            },
        )

        # Log TTS latency
        tts_e2e_ms = tts_result.latency_ms
        print(
            f"  [tts] Synthesized chunk #{cid}: {tts_e2e_ms:.0f}ms, {len(tts_result.audio) / tts_result.sample_rate:.1f}s audio"
        )

    except Exception as e:
        if _health is not None:
            _health.error("tts", type(e).__name__)
        print(f"  [tts] ERROR: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Data Persistence (audio + CSV + diagnostics JSONL for fine-tuning)
# ---------------------------------------------------------------------------


def save_chunk_audio(audio_data, cid):
    """Save chunk audio as 16kHz WAV for later Whisper fine-tuning."""
    if not _RECORD_AUDIO:
        return None
    import scipy.io.wavfile as wav

    os.makedirs(AUDIO_DIR, exist_ok=True)
    path = os.path.join(AUDIO_DIR, f"chunk_{cid:04d}.wav")
    wav.write(path, SAMPLE_RATE, audio_data)
    return path


# ---------------------------------------------------------------------------
# Live diarization hook (Phase 9.6.1) — isolated; no-op unless --diarize
# ---------------------------------------------------------------------------


def _utterance_wallclock(utterance_start, audio_data):
    """Wall-clock [start, end] for a final. Delegates to features.speaker_labels."""
    from features.speaker_labels import utterance_wallclock

    duration = len(audio_data) / SAMPLE_RATE if audio_data is not None else 0.0
    return utterance_wallclock(utterance_start, duration)


def _lookup_speaker(start_ts, end_ts):
    """Best-effort speaker for a final from the daemon JSONL (overlap + carry-forward)."""
    from features.speaker_labels import assign_speaker_from_jsonl

    return assign_speaker_from_jsonl(DIARIZE_JSONL, start_ts, end_ts)


def _export_diarize_chunk(audio_data, cid, start_ts, end_ts):
    """Refresh rolling.wav + chunks.jsonl on _io_pool. Never called on the GPU pool."""
    global _rolling_window
    from features.rolling_buffer import RollingSpeechWindow

    if start_ts is None or end_ts is None:
        start_ts, end_ts = _utterance_wallclock(None, audio_data)
    if _rolling_window is None or str(_rolling_window.session_dir) != AUDIO_DIR:
        _rolling_window = RollingSpeechWindow(AUDIO_DIR)
    _rolling_window.append(audio_data, cid, start_ts, end_ts)


async def _speaker_update_loop():
    """Revisit finalized chunks after rolling diarization produces labels."""
    while True:
        await asyncio.sleep(max(0.25, DIARIZE_INTERVAL_S))
        for cid, (start, end, previous) in list(_speaker_pending.items()):
            speaker = await asyncio.get_running_loop().run_in_executor(_io_pool, _lookup_speaker, start, end)
            if speaker and speaker != previous:
                _speaker_pending[cid] = (start, end, speaker)
                record = {
                    "type": "speaker_update",
                    "chunk_id": cid,
                    "speaker": speaker,
                    "session_id": SESSION_ID,
                    "timestamp": datetime.now().isoformat(),
                }
                await broadcast(record)
                _io_pool.submit(_write_jsonl_record, {"event": "speaker_update", **record})
            # The rolling daemon cannot improve an expired chunk indefinitely.
            if time.time() - end > max(120, DIARIZE_INTERVAL_S * 5):
                _speaker_pending.pop(cid, None)


def start_diarize_daemon():
    """Spawn ``features/live_diarize.py``. Failures are logged; the pipeline continues."""
    global _diarize_proc
    if not DIARIZE_ENABLED:
        return
    from features.live_diarize import build_daemon_command, hf_token

    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DIARIZE_JSONL) or ".", exist_ok=True)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features", "live_diarize.py")
    if not hf_token() and DIARIZE_MODE == "pyannote":
        logger.warning("HF_TOKEN not set — pyannote diarization daemon not started")
        return
    cmd = build_daemon_command(
        python=sys.executable,
        script=script,
        rolling_wav=os.path.join(AUDIO_DIR, "rolling.wav"),
        output=DIARIZE_JSONL,
        mode=DIARIZE_MODE,
        interval_s=DIARIZE_INTERVAL_S,
        session_dir=AUDIO_DIR,
        chunks_jsonl=os.path.join(AUDIO_DIR, "chunks.jsonl"),
    )
    try:
        _diarize_proc = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            start_new_session=True,
        )
        logger.info("live diarization daemon pid=%s mode=%s jsonl=%s", _diarize_proc.pid, DIARIZE_MODE, DIARIZE_JSONL)
        print(f"  Diarization daemon started ({DIARIZE_MODE}) → {DIARIZE_JSONL}")
    except OSError as exc:
        logger.warning("failed to start diarization daemon: %s — pipeline continues without labels", exc)
        _diarize_proc = None


def stop_diarize_daemon():
    """SIGTERM the daemon process group. Never raises into the pipeline."""
    global _diarize_proc
    proc = _diarize_proc
    _diarize_proc = None
    if proc is None or proc.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except (ProcessLookupError, OSError, PermissionError, AttributeError):
        try:
            proc.terminate()
        except (ProcessLookupError, OSError):
            return
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, OSError, PermissionError, AttributeError):
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                pass


atexit.register(stop_diarize_daemon)


def write_diag_jsonl(data, audio_path, segment_meta=None, low_conf_words=None, resources=None, marian_lat=None):
    """Append a structured diagnostics record for the active learning loop.

    This JSONL feeds into Label Studio / human review queue. Each line has:
      - audio file path (for Whisper fine-tuning)
      - transcription + translations (for correction)
      - full Whisper segment metadata (for filtering/analysis)
      - per-word confidence (for pinpointing misrecognitions)
      - all diagnostic flags (for prioritizing review)
    """
    cid = data["chunk_id"]
    homo = [{"flagged": f, "likely": l} for c, f, l, _ in diag_homophones if c == cid]
    near_misses = [
        {"original": orig, "correction": corr, "type": mt} for c, orig, corr, mt, _ in diag_near_misses if c == cid
    ]
    bad_sp = any(c == cid for c, _, _ in diag_bad_splits)
    marian_sim = next((s for c, _, _, s in diag_marian_diverge if c == cid), None)
    marian_text = next((mt for c, mt, _, _ in diag_marian_diverge if c == cid), None)
    utt_dur = next((d for c, d in diag_durations if c == cid), None)

    # Hallucination detection: compression_ratio > 2.4 = likely garbage
    is_hallucination = False
    if segment_meta:
        for seg in segment_meta:
            cr = seg.get("compression_ratio")
            if cr is not None and cr > 2.4:
                is_hallucination = True
                break

    # Short-phrase false-alarm dampening: correct short utterances like
    # "Amen" or "Thank you" get low confidence but aren't actually errors.
    english_lower = data.get("english", "").strip().lower().rstrip(".,!?;:'\"")
    is_short_whitelist = english_lower in _SHORT_PHRASE_WHITELIST

    # Compute a review priority score (higher = more likely needs correction)
    priority = 0
    conf = data.get("stt_confidence")
    if conf is not None and conf < 0.7:
        priority += 3 if conf < 0.5 else 1
    if homo:
        priority += 2 * len(homo)
    if near_misses:
        priority += 2 * len(near_misses)
    if bad_sp:
        priority += 1
    if data.get("qe_a") is not None and data["qe_a"] < 0.7:
        priority += 2
    if marian_sim is not None and marian_sim < 0.3:
        priority += 1
    if is_hallucination:
        priority += 5
    if low_conf_words:
        priority += min(3, len(low_conf_words))  # cap contribution
    if is_short_whitelist:
        priority = max(0, priority - 2)  # dampen false alarm

    record = {
        "chunk_id": cid,
        "session": SESSION_ID,
        "timestamp": data["timestamp"],
        "audio_path": audio_path,
        "mic_gain": MIC_GAIN,
        "english": data["english"],
        "spanish_gemma": data.get("spanish_a"),
        "spanish_marian": marian_text,
        "stt_confidence": conf,
        "qe_a": data.get("qe_a"),
        "utterance_dur": round(utt_dur, 2) if utt_dur else None,
        "segment_metadata": segment_meta,
        "low_confidence_words": low_conf_words,
        "is_hallucination": is_hallucination,
        "homophone_flags": homo,
        "near_miss_flags": near_misses,
        "bad_split": bad_sp,
        "marian_similarity": marian_sim,
        "marian_backend": _marian_engine.backend if _marian_engine is not None else None,
        "review_priority": priority,
        "marian_backend_latency": marian_lat,
        "stt_latency_ms": data.get("stt_latency_ms"),
        "latency_a_ms": data.get("latency_a_ms"),
        "latency_b_ms": data.get("latency_b_ms"),
        "e2e_latency_ms": data.get("e2e_latency_ms"),
        "true_e2e_ms": data.get("true_e2e_ms"),
        "word_stability_pct": data.get("word_stability_pct"),
        "tps_a": data.get("tps_a"),
        "tps_b": data.get("tps_b"),
        "resources": resources,
        "stt_corrections": [
            {"original": orig, "correction": corr, "type": ct} for c, orig, corr, ct in diag_stt_corrections if c == cid
        ],
        "corrected_english": None,  # filled in during human review
        "corrected_spanish": None,  # filled in during human review
        "speaker": data.get("speaker"),
    }

    record.update(_session_provenance())
    record.update(
        {
            field: data.get(field)
            for field in (*_GEN_STAT_FIELDS, *TIMING_COLUMNS, *_EXPERIMENT_GEN_FIELDS, "caption_delivery_mode")
            if field in data
        }
    )
    record["timing_stages_ms"] = data.get("timing_stages_ms")
    record["event_id"] = data.get("event_id")
    import hashlib
    from pathlib import Path

    record["audio_sha256"] = (
        hashlib.sha256(Path(audio_path).read_bytes()).hexdigest() if audio_path and Path(audio_path).is_file() else None
    )

    os.makedirs(os.path.dirname(DIAG_PATH), exist_ok=True)
    with open(DIAG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def init_csv():
    """Initialize CSV file with headers."""
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "chunk_id",
                "timestamp",
                "english",
                "spanish_a",
                "spanish_b",
                "stt_latency_ms",
                "latency_a_ms",
                "latency_b_ms",
                "e2e_latency_ms",
                "true_e2e_ms",
                "silence_delay_ms",
                "queue_wait_ms",
                "stt_confidence",
                "tps_a",
                "tps_b",
                "qe_a",
                "qe_b",
                "utterance_dur",
                "homophone_flags",
                "near_miss_flags",
                "bad_split",
                "marian_similarity",
                "marian_pt_ms",
                "partial_stt_ms",
                "marian_backend",
                "word_stability_pct",
                *_GEN_STAT_FIELDS,
                "speaker",
                *TIMING_COLUMNS,
                *_EXPERIMENT_GEN_FIELDS,
                "caption_delivery_mode",
            ]
        )
    print(f"  CSV: {CSV_PATH}")


def write_csv_row(data, marian_lat=None):
    """Append a row to the CSV log.

    Args:
        data: Result dict from the pipeline.
        marian_lat: Pre-extracted MarianMT latency dict (avoids race with
                    write_diag_jsonl which used to race on partial_latencies).
    """
    cid = data["chunk_id"]
    # Gather diagnostic flags for this chunk
    homo = [f"{f}->{l}" for c, f, l, _ in diag_homophones if c == cid]
    near_miss = [f"{o}->{c}" for ci, o, c, _, _ in diag_near_misses if ci == cid]
    bad_sp = any(c == cid for c, _, _ in diag_bad_splits)
    marian_sim = next((s for c, _, _, s in diag_marian_diverge if c == cid), "")
    utt_dur = next((d for c, d in diag_durations if c == cid), "")

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                data["chunk_id"],
                data["timestamp"],
                data["english"],
                data["spanish_a"],
                data["spanish_b"],
                data["stt_latency_ms"],
                data["latency_a_ms"],
                data["latency_b_ms"],
                data["e2e_latency_ms"],
                data.get("true_e2e_ms", ""),
                data.get("silence_delay_ms", ""),
                data.get("queue_wait_ms", ""),
                data.get("stt_confidence", ""),
                data.get("tps_a", ""),
                data.get("tps_b", ""),
                data.get("qe_a", ""),
                data.get("qe_b", ""),
                round(utt_dur, 2) if utt_dur else "",
                "|".join(homo) if homo else "",
                "|".join(near_miss) if near_miss else "",
                "Y" if bad_sp else "",
                marian_sim,
                marian_lat["pt_ms"] if marian_lat else "",
                marian_lat["stt_ms"] if marian_lat and "stt_ms" in marian_lat else "",
                _marian_engine.backend if _marian_engine is not None else "",
                data.get("word_stability_pct", ""),
                *(data.get(field) for field in _GEN_STAT_FIELDS),
                data.get("speaker", "") if data.get("speaker") is not None else "",
                *(data.get(field) for field in TIMING_COLUMNS),
                *(data.get(field) for field in _EXPERIMENT_GEN_FIELDS),
                data.get("caption_delivery_mode"),
            ]
        )


# ---------------------------------------------------------------------------
# Summary Stats
# ---------------------------------------------------------------------------


def print_summary():
    """Print summary statistics on exit."""
    # Also write zero-output runs: suppression counters explain a failed screen.
    _io_pool.submit(
        _write_jsonl_record,
        {
            "event": "session_summary",
            "session": SESSION_ID,
            "timestamp": datetime.now().isoformat(),
            "chunks_attempted": _chunks_attempted,
            "chunks_completed": _chunks_completed,
            "chunks_empty_stt": _chunks_empty_stt,
            "chunks_hallucination": _chunks_hallucination,
            "chunks_dedup": _chunks_dedup,
            "ws_total_connections": _ws_total_connections,
            "ws_total_disconnections": _ws_total_disconnections,
            "ws_send_failures": _ws_send_failures,
            "latency_experiment_counters": _experiment_snapshot(),
            "latency_experiment_configuration": _latency.as_dict(),
            "latency_trace": _latency_trace.snapshot(),
        },
    )
    if not all_results:
        print("\nNo results to summarize.")
        return

    n = len(all_results)
    stt_lats = [r["stt_latency_ms"] for r in all_results]
    a_lats = [r["latency_a_ms"] for r in all_results]
    b_lats = [r["latency_b_ms"] for r in all_results if r["latency_b_ms"] > 0]
    same = sum(1 for r in all_results if r["spanish_a"] == r["spanish_b"])
    a_faster = sum(1 for r in all_results if r["latency_b_ms"] > 0 and r["latency_a_ms"] < r["latency_b_ms"])

    print(f"\n{'=' * 60}")
    print(f"SESSION SUMMARY — {n} chunks")
    print(f"{'=' * 60}")
    print(f"  STT avg:     {np.mean(stt_lats):.0f}ms (median {np.median(stt_lats):.0f}ms)")
    print(f"  A (4B) avg:  {np.mean(a_lats):.0f}ms (median {np.median(a_lats):.0f}ms)")
    if b_lats:
        print(f"  B (12B) avg: {np.mean(b_lats):.0f}ms (median {np.median(b_lats):.0f}ms)")
        print(f"  Same output: {same}/{n} ({same / n:.0%})")
        print(f"  A faster:    {a_faster}/{n} ({a_faster / n:.0%})")
    else:
        print("  B (12B):     not loaded")

    # KPI: Processing rate
    if _chunks_attempted > 0:
        rate = _chunks_completed / _chunks_attempted * 100
        print(
            f"  Processing:  {_chunks_completed}/{_chunks_attempted} ({rate:.1f}%)"
            f" — empty_stt={_chunks_empty_stt}, hallucination={_chunks_hallucination}, dedup={_chunks_dedup}"
        )

    # KPI: True E2E latency summary
    true_e2es = [r["true_e2e_ms"] for r in all_results if r.get("true_e2e_ms") is not None]
    if true_e2es:
        print(f"  Utterance-start→processing: avg={np.mean(true_e2es):.0f}ms (median {np.median(true_e2es):.0f}ms)")

    # KPI: Word stability summary
    stabilities = [r["word_stability_pct"] for r in all_results if r.get("word_stability_pct") is not None]
    if stabilities:
        print(f"  Stability:   avg={np.mean(stabilities):.0%} ({len(stabilities)} chunks with partials)")

    print(f"  CSV saved:   {CSV_PATH}")
    print(f"{'=' * 60}")
    print_diagnostics()

    # WebSocket stats
    print(
        f"  WS stats:    connections={_ws_total_connections} disconnections={_ws_total_disconnections} send_failures={_ws_send_failures}"
    )


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------


async def audio_loop():
    """Main audio capture and processing loop with error recovery."""
    global _warmup_pending, _capture_handoff  # [P7-4A]
    print("\nListening... (Ctrl+C to stop)\n")

    PARTIAL_INTERVAL = settings.vad.partial_interval
    pause_preview_fired = False
    pause_speculation_fired = False
    SILENCE_TRIGGER = settings.vad.silence_trigger
    MAX_UTTERANCE = settings.vad.max_utterance

    speech_buffer = np.array([], dtype=np.float32)
    silence_frames = 0
    max_silence_frames = int(SILENCE_TRIGGER * SAMPLE_RATE / 512)
    frame_count = 0
    speech_frame_count = 0
    last_status_time = time.time()
    last_partial_len = 0  # audio length (samples) at last partial
    utterance_id = 0  # tracks current utterance for partial updates
    timeline = AudioTimeline()
    sample_clock = CaptureSampleClock()
    replay_stream = None
    replay_consumed_samples = 0
    last_silence_boundary = 0  # sample index of last silence gap start

    # Music/hymn auto-muting state
    music_hold_active = False
    music_nonspeech_frames = 0  # consecutive non-speech high-RMS frames
    music_speech_frames = 0  # consecutive speech frames (for exiting music hold)
    music_holdoff_frames = int(MUSIC_HOLDOFF * SAMPLE_RATE / 512)  # ~2s
    music_resume_frames = int(0.5 * SAMPLE_RATE / 512)  # ~0.5s speech to exit
    music_hold_start_frame = 0

    while True:
        if _health is not None and _health.paused:
            await asyncio.sleep(0.05)
            continue
        try:
            # Phase 9.4.2: when STARK_AUDIO_SOURCE=ws, read frames from the
            # operator's /ws/audio/subscribe endpoint instead of the local
            # mic. The factory matches sd.InputStream's context-manager
            # interface so the loop below is unchanged.
            from contextlib import ExitStack

            from tools.audio_bridge_client import FileAudioStream, open_audio_stream

            capture_loop = asyncio.get_running_loop()
            input_rate = SAMPLE_RATE if os.environ.get("STARK_AUDIO_SOURCE") == "ws" else MIC_SAMPLE_RATE

            def capture_dropped():
                _io_pool.record_failure("audio_capture", "handoff_overflow")
                if _health is not None:
                    _health.error("capture", "handoff_overflow")

            _capture_handoff = CaptureHandoff(
                capture_loop,
                audio_callback,
                lambda: not audio_queue.full(),
                capture_dropped,
                wait_for_space=os.environ.get("STARK_AUDIO_SOURCE") == "file",
            )

            def stream_callback(indata, frames, time_info, status, _handoff=_capture_handoff, _input_rate=input_rate):
                # Capture before loop handoff; device callbacks must never touch
                # asyncio.Queue from their producer thread. Copy PortAudio's buffer.
                stamp = sample_clock.capture(frames, _input_rate, time_info)
                if os.environ.get("STARK_AUDIO_SOURCE") == "ws":
                    stamp = CaptureStamp(
                        stamp.start,
                        stamp.end,
                        "bridge_receipt_estimate",
                        stamp.sample_start,
                        stamp.sample_end,
                        stamp.sample_rate,
                        stamp.padding_samples,
                    )
                _handoff.put(indata.copy(), frames, stamp, status)

            stream = replay_stream or open_audio_stream(
                callback=stream_callback,
                samplerate=input_rate,
                channels=1,
                dtype="float32",
                blocksize=int(input_rate * 0.032),  # ~32ms frames
                device=MIC_DEVICE,
            )

            is_replay = isinstance(stream, FileAudioStream)
            if is_replay:
                replay_stream = stream
                stream.resume_from(replay_consumed_samples, callback=stream_callback)
            if hasattr(stream, "sample_offset"):
                stream.sample_offset = sample_clock.next_sample
            with ExitStack() as capture_context:
                capture_context.enter_context(stream)
                capture_context.enter_context(_capture_handoff)
                if _health is not None:
                    _health.phase("listening")
                while True:
                    if _health is not None and _health.paused:
                        # Stop production before final admission. File prefetch is
                        # replayed from the last consumed sample after Resume.
                        if is_replay:
                            _capture_handoff.close(record_discard=False)
                        capture_context.close()
                        if len(speech_buffer) / SAMPLE_RATE >= 0.7:
                            _utterance_timings[utterance_id] = ChunkTiming.from_timeline(
                                timeline, utterance_id, "pause"
                            )
                            await process_final(speech_buffer.copy(), utterance_id)
                        if not is_replay and not audio_queue.empty():
                            _io_pool.record_failure("audio_capture", "pause_queued_audio_discarded")
                            _health.error("capture", "pause_queued_audio_discarded")
                        speech_buffer = np.array([], dtype=np.float32)
                        timeline = AudioTimeline()
                        silence_frames = speech_frame_count = last_partial_len = 0
                        last_silence_boundary = frame_count = 0
                        pause_preview_fired = pause_speculation_fired = False
                        if vad_model is not None:
                            vad_model.reset_states()
                        while not audio_queue.empty():
                            audio_queue.get_nowait()
                        break  # close the native capture child while paused
                    # Get audio frame from sounddevice callback, run VAD inline.
                    # VAD is <1ms so running it on the asyncio thread is fine.
                    try:
                        audio_frame = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                    except TimeoutError:
                        if getattr(stream, "error", None) is not None:
                            if isinstance(stream.error, AudioCaptureError):
                                raise stream.error from None
                            raise RuntimeError("Audio source failed") from stream.error
                        # EOF is checked only once all callback blocks have drained.
                        finished = getattr(stream, "finished", None)
                        if (
                            EXIT_AFTER_REPLAY
                            and finished is not None
                            and finished.is_set()
                            and audio_queue.empty()
                            and _capture_handoff.qsize() == 0
                        ):
                            if getattr(stream, "error", None) is not None:
                                raise RuntimeError("Audio replay failed") from stream.error
                            if len(speech_buffer):
                                # Tail silence normally finalizes speech. Flush any
                                # remainder, retaining the live minimum-length gate.
                                if len(speech_buffer) / SAMPLE_RATE >= 0.7:
                                    _utterance_timings[utterance_id] = ChunkTiming.from_timeline(
                                        timeline, utterance_id, "eof"
                                    )
                                    await process_final(speech_buffer.copy(), utterance_id)
                                timeline = AudioTimeline()
                                speech_buffer = np.array([], dtype=np.float32)
                            if (
                                _pipeline_chunk_queue.empty()
                                and not any(not task.done() for task in _partial_tasks)
                                and (_active_partial_future is None or _active_partial_future.done())
                                and (_stt_scheduler is None or not _stt_scheduler.busy)
                            ):
                                return
                        continue
                    if isinstance(audio_frame, AudioFrame):
                        frame_stamp, audio_frame = audio_frame.stamp, audio_frame.samples
                    else:  # historical callers/tests with raw PCM have receipt estimates
                        frame_stamp = sample_clock.capture(len(audio_frame), SAMPLE_RATE, None)
                    if is_replay:
                        replay_consumed_samples += (
                            frame_stamp.sample_end - frame_stamp.sample_start + frame_stamp.padding_samples
                        )
                    _latency_trace.record(
                        "audio_dequeued",
                        capture_age_ms=milliseconds(time.perf_counter(), frame_stamp.end),
                        queue_depth=audio_queue.qsize(),
                    )
                    vad_started = time.perf_counter()
                    if _vad_pool is not None:
                        has_speech = await asyncio.get_running_loop().run_in_executor(
                            _vad_pool, is_speech, audio_frame, vad_model, vad_utils
                        )
                    else:
                        has_speech = is_speech(audio_frame, vad_model, vad_utils)
                    _latency_trace.record("vad_complete", elapsed_ms=(time.perf_counter() - vad_started) * 1000)

                    frame_count += 1

                    # --- Music/hymn auto-muting ---
                    frame_rms = float(np.sqrt(np.mean(audio_frame**2)))
                    if not has_speech and frame_rms > MUSIC_THRESHOLD:
                        music_nonspeech_frames += 1
                        music_speech_frames = 0
                    elif has_speech:
                        music_speech_frames += 1
                        music_nonspeech_frames = 0
                    else:
                        # Silence (no speech, low RMS)
                        music_nonspeech_frames = 0
                        music_speech_frames = 0

                    # Enter music hold
                    if not music_hold_active and music_nonspeech_frames >= music_holdoff_frames:
                        music_hold_active = True
                        music_hold_start_frame = frame_count
                        print(
                            f"\n  [MUSIC] Music detected — muting STT (RMS={frame_rms:.4f}, threshold={MUSIC_THRESHOLD})"
                        )
                        # Discard any accumulated speech buffer (it's likely music garbage)
                        if len(speech_buffer) > 0:
                            timeline = AudioTimeline()
                            _utterance_start_times.pop(utterance_id, None)
                            _utterance_timings.pop(utterance_id, None)
                            speech_buffer = np.array([], dtype=np.float32)
                            silence_frames = 0
                            last_partial_len = 0
                            last_silence_boundary = 0
                        # Broadcast music_hold to displays
                        task = asyncio.create_task(broadcast({"type": "music_hold", "active": True}))
                        _partial_tasks.add(task)
                        task.add_done_callback(_partial_tasks.discard)

                    # Exit music hold when speech resumes for ~0.5s
                    if music_hold_active and music_speech_frames >= music_resume_frames:
                        hold_dur = (frame_count - music_hold_start_frame) * 512 / SAMPLE_RATE
                        diag_music_holds.append((music_hold_start_frame, frame_count, round(hold_dur, 1)))
                        music_hold_active = False
                        music_nonspeech_frames = 0
                        print(f"  [MUSIC] Speech resumed after {hold_dur:.1f}s hold")
                        vad_model.reset_states()
                        task = asyncio.create_task(broadcast({"type": "music_hold", "active": False}))
                        _partial_tasks.add(task)
                        task.add_done_callback(_partial_tasks.discard)

                    # Skip all speech buffering when in music hold
                    if music_hold_active:
                        continue

                    if has_speech:
                        if silence_frames:
                            _pause_epochs[utterance_id] = _pause_epochs.get(utterance_id, 0) + 1
                        pause_preview_fired = False
                        pause_speculation_fired = False
                        if len(speech_buffer) == 0:
                            utterance_id += 1  # new utterance starting
                            _utterance_start_times[utterance_id] = time.perf_counter()
                            last_partial_len = 0
                            last_silence_boundary = 0
                            logger.debug("vad_speech_start utterance_id=%d frame=%d", utterance_id, frame_count)
                        speech_buffer = np.concatenate([speech_buffer, audio_frame])
                        timeline.append(len(audio_frame), frame_stamp, True)
                        silence_frames = 0
                        speech_frame_count += 1
                    else:
                        # Record silence boundary on speech→silence transition
                        if len(speech_buffer) > 0 and silence_frames == 0:
                            last_silence_boundary = len(speech_buffer)
                        silence_frames += 1
                        # Keep buffering during brief pauses so words aren't dropped
                        if len(speech_buffer) > 0 and silence_frames < max_silence_frames:
                            speech_buffer = np.concatenate([speech_buffer, audio_frame])
                            timeline.append(len(audio_frame), frame_stamp, False)

                    # Periodic status line (~every 3s)
                    now = time.time()
                    if now - last_status_time >= 3.0:
                        buf_dur = len(speech_buffer) / SAMPLE_RATE
                        rms = float(np.sqrt(np.mean(audio_frame**2)))
                        print(
                            f"  [status] frames={frame_count} speech={speech_frame_count} buf={buf_dur:.1f}s silence={silence_frames} rms={rms:.4f}",
                            end="\r",
                        )
                        last_status_time = now

                    buffer_duration = len(speech_buffer) / SAMPLE_RATE
                    new_audio = (len(speech_buffer) - last_partial_len) / SAMPLE_RATE

                    pause_ms = silence_frames * 512 / SAMPLE_RATE * 1000
                    if not has_speech and buffer_duration >= 0.7 and silence_frames < max_silence_frames:
                        clause = bool(_latency.clause_preview_s and buffer_duration >= _latency.clause_preview_s)
                        preview_delay = _latency.pause_preview_ms or (128 if clause else 0)
                        if preview_delay and pause_ms >= preview_delay and not pause_preview_fired:
                            pause_preview_fired = True
                            kind = "clause" if clause else "pause"
                            _latency_trace.record(
                                "prospective_preview_boundary",
                                utterance_id=utterance_id,
                                kind=kind,
                                **timeline.sample_metadata(),
                            )
                            task = asyncio.create_task(
                                process_partial(
                                    speech_buffer.copy(),
                                    utterance_id,
                                    timeline.last,
                                    timeline.first,
                                    timeline.sample_metadata(),
                                    preview_kind=kind,
                                    speech_end=timeline.speech_end,
                                    timing_source=timeline.source,
                                )
                            )
                            _partial_tasks.add(task)
                            task.add_done_callback(_partial_tasks.discard)
                            last_partial_len = len(speech_buffer)
                        if (
                            _latency.speculate_pause_ms
                            and pause_ms >= _latency.speculate_pause_ms
                            and not pause_speculation_fired
                        ):
                            pause_speculation_fired = True
                            task = asyncio.create_task(
                                _speculate_pause(
                                    speech_buffer.copy(),
                                    utterance_id,
                                    _pause_epochs.get(utterance_id, 0),
                                    timeline.sample_metadata(),
                                )
                            )
                            _speculation_tasks.add(task)
                            task.add_done_callback(_speculation_tasks.discard)

                    new_audio = (len(speech_buffer) - last_partial_len) / SAMPLE_RATE
                    # One earlier first preview; subsequent cadence is unchanged.
                    partial_interval = (
                        (_latency.first_preview_s or PARTIAL_INTERVAL) if last_partial_len == 0 else PARTIAL_INTERVAL
                    )
                    # --- Partial: fire every PARTIAL_INTERVAL of new audio ---
                    if (
                        new_audio >= partial_interval
                        and silence_frames < max_silence_frames
                        and buffer_duration < MAX_UTTERANCE
                    ):
                        # [FIX] Fire-and-forget: don't block audio loop on partials
                        task = asyncio.create_task(
                            process_partial(
                                speech_buffer.copy(),
                                utterance_id,
                                timeline.last,
                                timeline.first,
                                timeline.sample_metadata(),
                                speech_end=timeline.speech_end,
                                timing_source=timeline.source,
                            )
                        )
                        _partial_tasks.add(task)
                        task.add_done_callback(_partial_tasks.discard)
                        last_partial_len = len(speech_buffer)

                    # --- Final: on silence gap or max duration ---
                    # Min 0.7s buffer — sub-0.7s breath pops are almost never real speech
                    silence_triggered = buffer_duration >= 0.7 and silence_frames >= max_silence_frames
                    force_cut_triggered = buffer_duration >= MAX_UTTERANCE

                    if (silence_triggered or force_cut_triggered) and buffer_duration >= 0.7:
                        print()  # newline after partial line
                        if silence_triggered:
                            logger.debug(
                                "vad_silence_trigger utterance_id=%d buf=%.2fs silence_frames=%d",
                                utterance_id,
                                buffer_duration,
                                silence_frames,
                            )
                        if force_cut_triggered and not silence_triggered:
                            # Speaker hasn't paused — try to cut at last silence boundary
                            cut_type, split_pos = _compute_force_cut(
                                len(speech_buffer), SAMPLE_RATE, last_silence_boundary
                            )
                            cut_dur = len(speech_buffer) / SAMPLE_RATE
                            cut_pos_s = split_pos / SAMPLE_RATE
                            diag_force_cuts.append((utterance_id, cut_type, round(cut_dur, 2), round(cut_pos_s, 2)))
                            logger.debug(
                                "vad_force_cut utterance_id=%d type=%s dur=%.2fs pos=%.2fs",
                                utterance_id,
                                cut_type,
                                cut_dur,
                                cut_pos_s,
                            )
                            print(
                                f"  [{cut_type.upper()}-CUT] #{utterance_id}: {cut_pos_s:.1f}s of {cut_dur:.1f}s buffer"
                            )

                            if cut_type == "smart":
                                # Send first part as final, carry over remainder
                                taken = timeline.split(split_pos)
                                _utterance_timings[utterance_id] = ChunkTiming.from_timeline(
                                    taken, utterance_id, "smart_cut"
                                )
                                await process_final(speech_buffer[:split_pos].copy(), utterance_id)
                                speech_buffer = speech_buffer[split_pos:].copy()
                                utterance_id += 1
                                _utterance_start_times[utterance_id] = timeline.first or time.perf_counter()
                                last_partial_len = 0
                                last_silence_boundary = 0
                            else:
                                # No silence found — send entire buffer (original behavior)
                                _utterance_timings[utterance_id] = ChunkTiming.from_timeline(
                                    timeline, utterance_id, "hard_cut"
                                )
                                await process_final(speech_buffer.copy(), utterance_id)
                                timeline = AudioTimeline()
                                speech_buffer = np.array([], dtype=np.float32)
                                silence_frames = 0
                                speech_frame_count = 0
                                frame_count = 0
                                last_partial_len = 0
                                last_silence_boundary = 0
                                vad_model.reset_states()
                                _warmup_pending = True
                        else:
                            # Normal silence-triggered finalization
                            _utterance_timings[utterance_id] = ChunkTiming.from_timeline(
                                timeline, utterance_id, "silence"
                            )
                            timeline = AudioTimeline()
                            await process_final(speech_buffer.copy(), utterance_id)
                            speech_buffer = np.array([], dtype=np.float32)
                            silence_frames = 0
                            speech_frame_count = 0
                            frame_count = 0
                            last_partial_len = 0
                            last_silence_boundary = 0
                            vad_model.reset_states()
                            # [P7-4A] Schedule a GPU warmup now that speech ended
                            _warmup_pending = True

                    # [P7-4A] Pre-warm during silence: run dummy forward pass
                    # after speech→silence transition to keep Metal GPU hot.
                    # Re-trigger every _WARMUP_INTERVAL seconds of sustained silence
                    # to prevent Metal GPU from re-idling during long pauses.
                    if (
                        len(speech_buffer) == 0
                        and silence_frames > 0
                        and (_warmup_pending or (time.perf_counter() - _last_warmup_time >= _WARMUP_INTERVAL))
                    ):
                        _warmup_pending = True
                        loop = asyncio.get_event_loop()
                        _schedule_warmup(loop)

        except asyncio.CancelledError:
            # Admit captured speech before main_async sends the coordinator's
            # sentinel. Pending translation then follows the normal final drain.
            if _session_stop_requested and len(speech_buffer) / SAMPLE_RATE >= 0.7:
                _utterance_timings[utterance_id] = ChunkTiming.from_timeline(timeline, utterance_id, "stop")
                await process_final(speech_buffer.copy(), utterance_id)
            if not audio_queue.empty():
                _io_pool.record_failure("audio_capture", "stop_queued_audio_discarded")
                if _health is not None:
                    _health.error("capture", "stop_queued_audio_discarded")
            raise
        except (sd.PortAudioError, AudioCaptureError) as e:
            if len(speech_buffer):
                _io_pool.record_failure("audio_capture", "interrupted_utterance")
                if _health is not None:
                    _health.error("capture", "interrupted_utterance")
            if _health is not None:
                _health.phase("input_error")
                _health.error("audio", type(e).__name__)
            print(f"\n  Mic error: {e} — retrying in 2s...", file=sys.stderr)
            speech_buffer = np.array([], dtype=np.float32)
            timeline = AudioTimeline()
            _utterance_start_times.pop(utterance_id, None)
            # Drain stale audio from queue
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            await asyncio.sleep(2)


_ROLLING_STATS_INTERVAL = 300  # 5 minutes


async def _rolling_stats_task():
    """Print and broadcast rolling session stats every 5 minutes."""
    while True:
        await asyncio.sleep(_ROLLING_STATS_INTERVAL)
        n = len(all_results)
        if n == 0:
            continue
        stt_lats = [r["stt_latency_ms"] for r in all_results]
        a_lats = [r["latency_a_ms"] for r in all_results]
        true_e2es = [r["true_e2e_ms"] for r in all_results if r.get("true_e2e_ms") is not None]
        msg = f"[ROLLING {n} chunks] STT avg={np.mean(stt_lats):.0f}ms | A avg={np.mean(a_lats):.0f}ms"
        if true_e2es:
            msg += f" | True E2E avg={np.mean(true_e2es):.0f}ms"
        print(f"\n  {msg}")
        logger.info(msg)
        # Broadcast to operator displays
        stats_data = {
            "type": "rolling_stats",
            "chunks": n,
            "stt_avg_ms": round(float(np.mean(stt_lats)), 1),
            "a_avg_ms": round(float(np.mean(a_lats)), 1),
            "true_e2e_avg_ms": round(float(np.mean(true_e2es)), 1) if true_e2es else None,
            "timestamp": datetime.now().isoformat(),
        }
        await broadcast(stats_data)


async def main_async(args):
    """Start WebSocket server and audio loop."""
    global vad_model, vad_utils, stt_pipe
    global mlx_a_model, mlx_a_tokenizer, mlx_b_model, mlx_b_tokenizer
    global _marian_engine
    global _stream_token_queue, _stream_loop
    global _pipeline_chunk_queue, _pipeline_translation_lock
    global _RUN_AB, _clean_session_shutdown, _session_model_ids, _session_main_task, _incremental_stt

    _RUN_AB = args.run_ab
    _clean_session_shutdown = False
    _session_main_task = asyncio.current_task()

    spec_info = ""
    if args.run_ab and BACKEND == "mlx":
        spec_info = f"  Speculative decoding: 4B drafts {NUM_DRAFT_TOKENS} tokens for 12B\n"

    mp_str = " [multiprocess]" if MULTIPROCESS else ""
    mode_str = "MarianMT-only (low-VRAM)" if args.low_vram else ("A/B parallel" if args.run_ab else "4B only")

    lang_str = "Spanish→English" if SOURCE_LANG == "es" else "English→Spanish"

    print(f"{'=' * 60}")
    print("  Bilingual A/B Dry Run")
    print(f"  Backend: {BACKEND.upper()}")
    print(f"  Language: {lang_str}")
    print(f"  Mode: {mode_str}{mp_str}")
    if spec_info:
        print(spec_info, end="")
    print(f"  WebSocket: ws://localhost:{args.ws_port}")
    print(f"{'=' * 60}\n")

    # Load models — dispatch based on backend
    vad_model, vad_utils = load_vad()

    if MULTIPROCESS and BACKEND == "mlx" and not args.low_vram:
        # Multiprocess mode: STT and Translation models load in worker processes.
        # Main process only loads VAD (inline) and MarianMT (CPU, for partials).
        stt_pipe = None  # loaded in STT worker
        mlx_a_model, mlx_a_tokenizer = None, None
        mlx_b_model, mlx_b_tokenizer = None, None
        _start_workers(run_ab=args.run_ab)
    else:
        if _stt_scheduler is not None:
            stt_pipe = await asyncio.wrap_future(_stt_scheduler.submit("final", load_whisper, BACKEND))
        else:
            stt_pipe = load_whisper(BACKEND)
        if _latency.incremental_stt == "stream":
            from tools.incremental_stt import StreamingPreview

            _incremental_stt = StreamingPreview(stt_pipe)
        if args.low_vram:
            # Low-VRAM mode: skip Gemma entirely, MarianMT handles all translation
            mlx_a_model, mlx_a_tokenizer = None, None
            mlx_b_model, mlx_b_tokenizer = None, None
        elif RUNTIME_PROFILE.final_engine == "llamacpp":
            global _managed_llama_server
            from pathlib import Path

            from engines.llamacpp_engine import LlamaCppEngine
            from tools.llama_runtime import ManagedLlamaServer

            _managed_llama_server = ManagedLlamaServer(
                RUNTIME_PROFILE, log_path=Path(f"metrics/llama_{SESSION_ID}.log")
            )
            url = _managed_llama_server.start()
            mlx_a_model = LlamaCppEngine(
                server_url=url,
                model_family="gemma4",
                timeout_s=120 if BACKEND == "cpu" else 30,
                max_tokens=128,
                strict_errors=True,
            )
            mlx_a_model.load()
            mlx_a_tokenizer = mlx_b_model = mlx_b_tokenizer = None
        elif BACKEND == "mlx":
            mlx_a_model, mlx_a_tokenizer, mlx_b_model, mlx_b_tokenizer = load_translation_models(load_b=args.run_ab)
        elif BACKEND == "cuda":
            mlx_a_model, mlx_a_tokenizer, mlx_b_model, mlx_b_tokenizer = load_cuda_translation_models(
                load_gemma=not args.low_vram, load_b=args.run_ab
            )
        else:
            # CPU fallback: no Gemma, MarianMT only
            mlx_a_model, mlx_a_tokenizer = None, None
            mlx_b_model, mlx_b_tokenizer = None, None

    _marian_engine = load_marian()

    # --- TTS engine (optional, ONNX Runtime — thread-safe, separate pool) ---
    global tts_engine, _tts_pool
    if args.tts:
        from engines.tts_engine import PiperTTSEngine

        # Dynamic language: TTS speaks in TARGET_LANG (the translated language)
        tts_voice = settings.tts.voices.get(TARGET_LANG)
        if tts_voice:
            tts_engine = PiperTTSEngine(voices={TARGET_LANG: tts_voice})
            tts_engine.load()
            _tts_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
            settings.tts.enabled = True
            settings.tts.output_mode = args.tts_output
            if args.tts_device is not None:
                settings.tts.output_device = args.tts_device
            for language in ("en", "es"):
                device_spec = getattr(args, f"tts_device_{language}", None)
                if device_spec is not None:
                    settings.tts.output_devices[language] = device_spec
            device_suffix = f", device={settings.tts.output_device}" if args.tts_output == "local" else ""
            print(f"  TTS enabled: {TARGET_LANG} voice ({tts_voice}), output={args.tts_output}{device_suffix}")
        else:
            print(f"  WARNING: No TTS voice configured for {TARGET_LANG}, TTS disabled", file=sys.stderr)

    # --- Dry-run text mode: translate a single string and exit ---
    if args.dry_run_text:
        print("\n--- DRY RUN TEXT MODE ---")
        print(f"Input: {args.dry_run_text}")
        if BACKEND == "mlx" and mlx_a_model is not None:
            spanish, lat, tps = translate_mlx(
                mlx_a_model,
                mlx_a_tokenizer,
                args.dry_run_text,
                prompt_cache_template=mlx_a_prompt_cache,
                suffix_tokens=mlx_a_suffix_tokens,
            )
        elif (BACKEND == "cuda" or RUNTIME_PROFILE.final_engine == "llamacpp") and mlx_a_model is not None:
            # mlx_a_model is a CUDAGemmaStreamingEngine when using streaming path
            if hasattr(mlx_a_model, "translate_streaming"):
                result = mlx_a_model.translate(args.dry_run_text, source_lang=SOURCE_LANG, target_lang=TARGET_LANG)
                spanish, lat, tps = result.text, result.latency_ms, result.tokens_per_second
            else:
                spanish, lat, tps = translate_cuda_gemma(mlx_a_model, mlx_a_tokenizer, args.dry_run_text)
        else:
            spanish, lat = translate_marian(args.dry_run_text)
            tps = 0.0
        print(f"Translation ({TARGET_LANG}): {spanish}")
        print(f"Latency: {lat:.0f}ms, TPS: {tps:.1f}")

        # TTS in dry-run mode: synthesize translated text
        if tts_engine and settings.tts.enabled:
            tts_result = tts_engine.synthesize(spanish, language=TARGET_LANG)
            print(f"TTS: {tts_result.latency_ms:.0f}ms, {len(tts_result.audio) / tts_result.sample_rate:.1f}s audio")
            if settings.tts.output_mode in ("wav", "both"):
                import scipy.io.wavfile as wav

                os.makedirs(AUDIO_DIR, exist_ok=True)
                wav_path = os.path.join(AUDIO_DIR, "tts_dry_run.wav")
                audio_int16 = (tts_result.audio * 32767).astype(np.int16)
                wav.write(wav_path, tts_result.sample_rate, audio_int16)
                print(f"TTS WAV: {wav_path}")

        return

    # Detect best microphone and auto-calibrate gain
    global MIC_DEVICE, MIC_GAIN
    if MIC_DEVICE is None and os.environ.get("STARK_AUDIO_SOURCE") != "file":
        print("[5/6] Detecting microphone...")
        MIC_DEVICE, mic_rms = detect_macbook_mic()
        if mic_rms > 0 and MIC_GAIN == 1.0:
            # Auto-calibrate: scale signal to TARGET_RMS
            MIC_GAIN = max(1.0, min(20.0, TARGET_RMS / mic_rms))
            print(f"  Auto-gain: {MIC_GAIN:.1f}x (mic RMS={mic_rms:.4f}, target={TARGET_RMS})")

    # Save hardware profile for portability planning
    hw_profile = get_hardware_profile()
    hw_path = os.path.join(os.path.dirname(DIAG_PATH), f"hardware_{SESSION_ID}.json")
    os.makedirs(os.path.dirname(hw_path), exist_ok=True)
    with open(hw_path, "w") as f:
        json.dump(hw_profile, f, indent=2)
    print(f"  Hardware profile: {hw_path}")
    print(
        f"    {hw_profile.get('processor', '?')} | "
        f"{hw_profile['cpu_count_physical']}P+{hw_profile['cpu_count_logical'] - hw_profile['cpu_count_physical']}E cores | "
        f"{hw_profile['ram_total_gb']}GB RAM | "
        f"{hw_profile.get('gpu_model', '?')}"
    )

    print("[6/6] Starting servers...")
    _session_model_ids = {
        "stt": stt_pipe if isinstance(stt_pipe, str) else getattr(stt_pipe, "model_id", None),
        "translation_a": (MLX_MODEL_A if BACKEND == "mlx" else getattr(mlx_a_model, "model_id", None))
        if not args.low_vram
        else None,
        "translation_b": MLX_MODEL_B if BACKEND == "mlx" and _RUN_AB else None,
        "marian": getattr(_marian_engine, "model_id", None),
        "draft": MLX_DRAFT_MODEL_ID if USE_MTS else None,
    }
    metadata = {
        **_session_provenance(),
        "timing_schema_version": 2,
        "backend": BACKEND,
        "model_family": MODEL_FAMILY,
        "model_a": _session_model_ids["translation_a"],
        "profile": RUNTIME_PROFILE.to_dict(),
        "profile_artifacts": profile_artifact_provenance(),
        "stt_settings": settings.stt.model_dump(),
        "managed_llama": _managed_llama_server.provenance if _managed_llama_server else None,
        "model_b": MLX_MODEL_B if _RUN_AB else None,
        "stt_backend": settings.stt.backend,
        "vad": {**settings.vad.model_dump(), "artifact": _vad_provenance},
        "translation": settings.translation.model_dump(),
        "replay_speed": float(os.environ.get("STARK_REPLAY_SPEED", "1")),
    }
    from pathlib import Path

    from tools.session_lifecycle import source_provenance

    metadata.update(source_provenance(Path(__file__)))
    metadata["latency_experiment_configuration"] = _latency.as_dict()
    Path(os.path.dirname(DIAG_PATH), f"session_metadata_{SESSION_ID}.json").write_text(json.dumps(metadata, indent=2))
    init_csv()
    _io_pool.submit(_write_jsonl_record, {"event": "session_started", **_session_provenance()})

    # Start HTTP server for mobile access
    project_dir = os.path.dirname(os.path.abspath(__file__))
    start_http_server(args.http_port, project_dir)
    local_ip = get_local_ip()

    # Start WebSocket server (bind 0.0.0.0 so phones can connect)
    ws_server = await websockets.serve(
        ws_handler,
        "0.0.0.0",
        args.ws_port,
        ping_interval=None,  # disable pings — inference blocks event loop
    )

    # Start TTS audio WebSocket server (separate port for binary PCM)
    tts_ws_server = None
    if tts_engine and settings.tts.enabled and settings.tts.output_mode in ("ws", "both"):
        tts_ws_server = await websockets.serve(
            tts_ws_handler,
            "0.0.0.0",
            settings.tts.audio_ws_port,
            ping_interval=None,
        )
        print(f"  TTS audio WebSocket ready on port {settings.tts.audio_ws_port}")

    print(f"  WebSocket ready on port {args.ws_port}")
    print(f"  HTTP server ready on port {args.http_port}")
    print("\n  Local displays:")
    print(f"    A/B display:       file://{project_dir}/displays/ab_display.html")
    print(f"    Audience display:  file://{project_dir}/displays/audience_display.html")
    print("\n  Mobile / LAN access:")
    print(f"    http://{local_ip}:{args.http_port}/displays/mobile_display.html")
    print(f"    http://{local_ip}:{args.http_port}/displays/audience_display.html")
    if tts_ws_server:
        print("\n  TTS audio stream:")
        print(f"    ws://{local_ip}:{settings.tts.audio_ws_port}  (binary PCM, int16, mono)")

    # [P7-P3-6A] Initialize streaming translation queue and broadcaster
    _stream_token_queue = asyncio.Queue(maxsize=128)
    _stream_loop = asyncio.get_event_loop()
    stream_task = asyncio.create_task(stream_token_broadcaster())
    print("  [P7-P3-6A] Streaming translation broadcaster started")

    # [P7-6C] Initialize pipeline coordinator for STT/translation overlap
    _pipeline_chunk_queue = asyncio.Queue(maxsize=8)
    _pipeline_translation_lock = asyncio.Lock()
    pipeline_task = asyncio.create_task(_pipeline_coordinator())
    if MULTIPROCESS:
        print("  [P7-6C] Pipeline coordinator started (multiprocess: true GPU overlap)")
    else:
        print("  [P7-6C] Pipeline coordinator started (STT/translation overlap enabled)")

    speaker_task = None
    if DIARIZE_ENABLED:
        start_diarize_daemon()
        speaker_task = asyncio.create_task(_speaker_update_loop())

    # Rolling stats task — prints averages every 5 minutes
    rolling_task = asyncio.create_task(_rolling_stats_task())

    if _health is not None:

        def health_provider():
            return {
                "queues": {
                    "audio": audio_queue.qsize(),
                    "capture_handoff": _capture_handoff.qsize() if _capture_handoff is not None else 0,
                    "finals": _pipeline_chunk_queue.qsize() if _pipeline_chunk_queue else 0,
                    "stream_tokens": _stream_token_queue.qsize() if _stream_token_queue else 0,
                },
                "clients": len(ws_clients),
            }

        def apply_control(operation):
            global _session_stop_requested
            if operation == "stop":
                _session_stop_requested = True
                request_graceful_stop(_session_main_task)
            else:
                if operation == "resume":
                    _health.phase("listening")
                _health.paused = operation == "pause"

        from tools.session_lifecycle import request_graceful_stop

        _health._provider = health_provider
        _health._control = apply_control

    # Run audio loop
    try:
        await audio_loop()
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_exception = sys.exc_info()[1]
        rolling_task.cancel()
        if speaker_task is not None:
            speaker_task.cancel()
            await asyncio.gather(speaker_task, return_exceptions=True)
        # [P7-6C] Stop pipeline coordinator — send poison pill and wait
        if _pipeline_chunk_queue is not None:
            await _pipeline_chunk_queue.put(None)
        await asyncio.wait_for(pipeline_task, timeout=10.0)
        # [P7-P3-6A] Stop streaming broadcaster
        if _stream_token_queue is not None:
            await _stream_token_queue.put(None)
        stream_task.cancel()
        if _speculation_tasks:
            await asyncio.gather(*list(_speculation_tasks), return_exceptions=True)
        if _stt_scheduler is not None:
            if _incremental_stt is not None:
                await asyncio.wrap_future(_stt_scheduler.submit("final", _incremental_stt.close))
            await asyncio.get_running_loop().run_in_executor(None, _stt_scheduler.shutdown)
        if _vad_pool is not None:
            _vad_pool.shutdown(wait=True)
        if _caption_delivery is not None:
            await _caption_delivery.close()
        # Release the Marian engine first, then shut down the PyTorch pool.
        if _marian_engine is not None:
            _marian_engine.unload()
        _pytorch_pool.shutdown(wait=False)
        # Shut down TTS pool if running
        if _tts_pool is not None:
            _tts_pool.shutdown(wait=True)
        if tts_engine is not None:
            tts_engine.unload()
        # Stop multiprocess workers if running
        if MULTIPROCESS:
            _stop_workers()
        stop_diarize_daemon()
        print_summary()
        ws_server.close()
        await ws_server.wait_closed()
        if tts_ws_server is not None:
            tts_ws_server.close()
            await tts_ws_server.wait_closed()
        _clean_session_shutdown = shutdown_exception is None or (
            _session_stop_requested
            and isinstance(shutdown_exception, (asyncio.CancelledError, KeyboardInterrupt, SystemExit))
        )


def main():
    import hashlib
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Live A/B bilingual speech-to-text dry run")
    parser.add_argument(
        "--no-record-audio",
        action="store_true",
        help="Keep captions/diagnostics without retaining microphone WAV files",
    )
    parser.add_argument(
        "--ab", action="store_true", dest="run_ab", help="Load both 4B and 12B for A/B comparison (default: 4B only)"
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "mlx", "cuda", "cpu"],
        default="auto",
        help="Inference backend: auto (detect), mlx (Apple Silicon), cuda (NVIDIA)",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "llamacpp", "hf"],
        default=None,
        help=(
            "CUDA translation engine. 'auto' (default) probes the configured "
            "llama-server URL and prefers llamacpp when reachable, else HF NF4. "
            "'llamacpp' / 'hf' force one path. Has no effect on MLX/CPU backends."
        ),
    )
    parser.add_argument(
        "--llamacpp-url",
        type=str,
        default=None,
        help="Primary llama-server URL (default: http://127.0.0.1:8090). See start_server.sh.",
    )
    parser.add_argument(
        "--llamacpp-url-b",
        type=str,
        default=None,
        help="Secondary llama-server URL for --ab mode (default: http://127.0.0.1:8091).",
    )
    parser.add_argument("--no-ab", action="store_true", help="Skip 12B model, use 4B only (for low-VRAM devices)")
    parser.add_argument(
        "--low-vram", action="store_true", help="Minimal VRAM mode: MarianMT-only translation, no Gemma"
    )
    parser.add_argument(
        "--dry-run-text",
        type=str,
        default=None,
        help="Run a single text through the pipeline without mic (for testing)",
    )
    parser.add_argument(
        "--chunk-duration", type=float, default=2.0, help="Seconds of speech to accumulate before processing"
    )
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument(
        "--http-port", type=int, default=8080, help="HTTP server port for serving display pages to phones"
    )
    parser.add_argument("--vad-threshold", type=float, default=0.3, help="VAD speech threshold (0-1)")
    parser.add_argument("--vad-backend", choices=["torch", "onnx"], default=None)
    parser.add_argument("--silence-trigger", type=float, default=None)
    parser.add_argument("--partial-interval", type=float, default=None)
    parser.add_argument("--idle-warmup-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--final-aware-partials", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--routing-policy", choices=["legacy", "conservative", "off"], default=None)
    parser.add_argument("--terminology-prompt", choices=["none", "church"], default=None)
    parser.add_argument("--device", type=int, default=None, help="Audio input device index (default: auto-detect)")
    parser.add_argument("--gain", type=float, default=None, help="Mic gain multiplier (default: auto-calibrate)")
    parser.add_argument("--audio-file", help="Replay a WAV through the live audio pipeline")
    parser.add_argument("--replay-speed", type=float, default=1.0, help="Replay speed; <=0 runs unpaced")
    parser.add_argument("--session-id", help="Deterministic session ID for metrics and recordings")
    parser.add_argument(
        "--exit-after-replay",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Exit after replay drains (default: enabled with --audio-file)",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=3,
        help="Speculative decoding: tokens drafted by 4B for 12B to verify (default: 3)",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        default=False,
        help="Enable per-word timestamps/confidence in final STT (adds ~200-400ms, useful for active learning)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="Whisper beam search width: 1=greedy (fastest), 5=default beam search (default: 1)",
    )
    parser.add_argument(
        "--stt-backend",
        type=str,
        default="auto",
        choices=["auto", "faster-whisper", "hf", "mlx", "parakeet", "parakeet-mlx"],
        help=(
            "Whisper implementation within the chosen --backend hardware tier. "
            "'auto' (default): faster-whisper on cuda/cpu, mlx on Apple. "
            "'hf' enables torch.compile + spec decode. "
            "'parakeet' is EN-only (NeMo); keep Whisper for ES. "
            "Also settable via STARK_STT__BACKEND env var."
        ),
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default=None,
        choices=[None, "default", "reduce-overhead", "max-autotune"],
        help=(
            "torch.compile mode for HF Whisper (no effect on faster-whisper). "
            "'reduce-overhead' enables CUDA graphs for best steady-state speed but "
            "recompiles on each new audio shape (~5-10s first call after a shape change)."
        ),
    )
    parser.add_argument(
        "--warmup-seconds",
        type=int,
        default=1,
        help="Seconds of silence to push through the STT engine after load to drive JIT/CUDA-graph capture (0 disables; default: 1)",
    )
    parser.add_argument(
        "--multiprocess",
        action="store_true",
        default=False,
        help=(
            "Optional: run STT and translation in separate OS processes (separate Metal "
            "contexts). Default Mac path already overlaps in-process (MLX >= 0.31.2). "
            "Escape hatch for debugging / older mlx."
        ),
    )
    parser.add_argument(
        "--music-threshold",
        type=float,
        default=0.15,
        help="RMS threshold for music detection (no speech + high energy). Default: 0.15",
    )
    parser.add_argument(
        "--music-holdoff",
        type=float,
        default=5.0,
        help="Seconds of music-like audio before muting STT. Default: 5.0",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "es"],
        default="en",
        help="Input language: en (English→Spanish, default) or es (Spanish→English)",
    )
    parser.add_argument(
        "--tts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable TTS audio output of translated text (Piper ONNX)",
    )
    parser.add_argument(
        "--tts-output",
        choices=["ws", "wav", "both", "local"],
        default="ws",
        help=(
            "TTS output mode: ws (WebSocket stream), wav (file), both, "
            "local (per-language --tts-device-en/es, falling back to --tts-device). default: ws"
        ),
    )
    parser.add_argument(
        "--tts-device",
        type=int,
        default=None,
        help=(
            "sounddevice output device index for --tts-output local. "
            "Get the index from /api/devices or `python -m sounddevice`. "
            "Defaults to system default output."
        ),
    )
    for language in ("en", "es"):
        parser.add_argument(
            f"--tts-device-{language}",
            type=lambda value: int(value) if value.lstrip("+-").isdigit() else value,
            default=None,
            help=f"Local {language.upper()} TTS output index or device-name substring; overrides the language map.",
        )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level for console and file output (default: INFO)",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default=None,
        help=(
            "LoRA adapter directory for the primary (4B) MLX TranslateGemma model. "
            "Passed to mlx_lm.load(..., adapter_path=). Ignored on CUDA/llamacpp."
        ),
    )
    parser.add_argument(
        "--adapter-dir-b",
        type=str,
        default=None,
        help="Optional LoRA adapter directory for the 12B model in --ab mode (MLX only).",
    )
    parser.add_argument(
        "--turboquant",
        action="store_true",
        default=None,
        help=(
            "Request TurboQuant KV cache on MLX (soft-disabled on mlx-optiq 0.4.x "
            "with no drop-in TurboQuantKVCache for mlx_lm.generate). "
            "Overrides STARK_TRANSLATION__TURBOQUANT when set."
        ),
    )
    parser.add_argument(
        "--no-turboquant",
        action="store_true",
        default=False,
        help="Disable TurboQuant even if enabled in settings.",
    )
    parser.add_argument(
        "--model-family",
        choices=["translategemma", "gemma4"],
        default=None,
        help=(
            "Translation model family. Default: gemma4 (Mac OptiQ E4B). "
            "Use translategemma to opt out. Env: STARK_TRANSLATE_MODEL_FAMILY."
        ),
    )
    parser.add_argument(
        "--mlx-model",
        type=str,
        default=None,
        help="Override primary MLX model id (e.g. mlx-community/gemma-4-e4b-it-OptiQ-4bit).",
    )
    parser.add_argument(
        "--gemma4-size",
        choices=["e4b", "e2b"],
        default="e4b",
        help="When --model-family gemma4 and --mlx-model unset, pick E4B or E2B OptiQ (default: e4b).",
    )
    parser.add_argument(
        "--mts",
        action="store_true",
        default=None,
        help=(
            "Enable Gemma-4 assistant-drafter MTS (speculative decode) on MLX. "
            "Requires --model-family gemma4. Env: STARK_TRANSLATE_MLX_MTS."
        ),
    )
    parser.add_argument(
        "--no-mts",
        action="store_true",
        default=False,
        help="Disable MTS even if enabled in settings.",
    )
    parser.add_argument(
        "--mlx-drafter",
        type=str,
        default=None,
        help=("Gemma-4 assistant drafter model id for --mts (default: mlx-community/gemma-4-e4b-it-assistant-bf16)."),
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        default=False,
        help="Enable live speaker diarization (off by default; separate CPU daemon, no GPU)",
    )
    parser.add_argument(
        "--diarize-mode",
        choices=["embed", "pyannote"],
        default="embed",
        help="Diarization backend: embed (default, ~100-300ms/chunk) or pyannote (rolling window)",
    )
    parser.add_argument(
        "--diarize-interval-s",
        type=float,
        default=2.0,
        help="Diarization daemon poll interval in seconds (default: 2)",
    )
    from stark_translate.profiles import PROFILE_NAMES

    parser.add_argument("--profile", choices=PROFILE_NAMES, default=settings.profile)
    args = parser.parse_args()
    try:
        validate_live_mts(args.mts, args.no_mts, settings.translation.mlx_mts)
    except ValueError as exc:
        parser.error(str(exc))
    global RUNTIME_PROFILE
    try:
        RUNTIME_PROFILE = apply_profile(settings, args.profile, args.backend)
    except ValueError as exc:
        parser.error(str(exc))
    if RUNTIME_PROFILE.lite:
        if args.run_ab or args.multiprocess or args.diarize:
            parser.error("Lite profiles do not enable A/B, multiprocess or diarization")
        args.backend = RUNTIME_PROFILE.backend
        args.low_vram = settings.low_vram
        args.stt_backend = "faster-whisper"
        args.vad_backend = "onnx"
        args.gemma4_size = "e2b"
        args.model_family = "gemma4"
        args.engine = settings.cuda.engine
        args.no_mts = True

    global EXIT_AFTER_REPLAY
    EXIT_AFTER_REPLAY = bool(args.audio_file) if args.exit_after_replay is None else args.exit_after_replay
    # Validate fallible replay input before creating a running lifecycle marker,
    # health thread or inference pool. Invalid CLI input is not a started session.
    global _INPUT_AUDIO_HASH
    _INPUT_AUDIO_HASH = None
    if args.audio_file or os.environ.get("STARK_AUDIO_SOURCE") == "file":
        effective_audio_file = args.audio_file or os.environ.get("STARK_AUDIO_FILE")
        if not effective_audio_file:
            parser.error("STARK_AUDIO_FILE is required when STARK_AUDIO_SOURCE=file")
        try:
            _INPUT_AUDIO_HASH = hashlib.sha256(Path(effective_audio_file).read_bytes()).hexdigest()
        except (OSError, ValueError) as exc:
            parser.error(f"Cannot read audio input: {exc}")
    for name in ("silence_trigger", "partial_interval"):
        value = getattr(args, name)
        if value is not None and not 0.032 <= value <= 5:
            parser.error(f"--{name.replace('_', '-')} must be between 0.032 and 5 seconds")

    if args.audio_file:
        os.environ["STARK_AUDIO_SOURCE"] = "file"
        os.environ["STARK_AUDIO_FILE"] = args.audio_file
        os.environ["STARK_REPLAY_SPEED"] = str(args.replay_speed)

    # --- Resolve backend ---
    # CUDA engine selection: CLI --engine wins over env vars/defaults in settings.
    if args.engine is not None:
        settings.cuda.engine = args.engine
    if args.llamacpp_url is not None:
        settings.cuda.llamacpp_url = args.llamacpp_url
    if args.llamacpp_url_b is not None:
        settings.cuda.llamacpp_url_b = args.llamacpp_url_b

    global BACKEND
    if args.backend == "auto":
        if MLX_AVAILABLE:
            BACKEND = "mlx"
        elif torch is not None and torch.cuda.is_available():
            BACKEND = "cuda"
        else:
            BACKEND = "cpu"
    elif args.backend == "mlx":
        if not MLX_AVAILABLE:
            print("ERROR: MLX not available. Install mlx, mlx-lm, mlx-whisper.", file=sys.stderr)
            sys.exit(1)
        BACKEND = "mlx"
    elif args.backend == "cuda":
        import ctranslate2

        if ctranslate2.get_cuda_device_count() < 1:
            parser.error("CUDA profile selected but CTranslate2 cannot find a CUDA device")
        BACKEND = "cuda"
    elif args.backend == "cpu":
        BACKEND = "cpu"

    # --low-vram implies --no-ab
    if args.low_vram:
        args.run_ab = False
    # --no-ab overrides --ab
    if args.no_ab:
        args.run_ab = False
    # CUDA backend: A/B VRAM gate only applies to the HF NF4 path. The llamacpp
    # path runs E4B (~5 GB) + E2B (~3.5 GB) = ~9 GB total, fits on 12 GB cards.
    if BACKEND == "cuda" and args.run_ab and settings.cuda.engine == "hf":
        from engines.cuda_engine import detect_vram_tier

        _tier, _vram = detect_vram_tier()
        if _tier != "full_ab":
            print(
                f"WARNING: A/B mode (HF) requires >=15 GB VRAM ({_vram} MB detected). Disabling 12B.",
                file=sys.stderr,
            )
            args.run_ab = False

    global _latency, _latency_trace, _stt_scheduler, _marian_memo, _vad_pool
    _latency = LatencyExperiments.from_env()
    _latency_trace = LatencyTrace(_latency.trace)
    _marian_memo = ExactTextMemo(_latency.marian_memo)
    if _latency.latest_partial or _latency.incremental_stt != "off" or _latency.speculate_pause_ms:
        _stt_scheduler = LatestSTTWorker(on_event=_latency_event)
    if _latency.vad_worker:
        _vad_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vad-owner")

    # --- Create pipeline thread pool (backend-dependent) ---
    global _pipeline_pool
    # Both CUDA and MLX (>=0.31.2) support independent concurrent inference.
    _n_workers = pipeline_pool_max_workers(BACKEND)
    _pipeline_pool = ThreadPoolExecutor(
        max_workers=_n_workers,
        thread_name_prefix=f"{BACKEND}-pipeline",
    )

    # --- Resolve language direction ---
    global SOURCE_LANG, TARGET_LANG, WHISPER_PROMPT
    global SESSION_ID, CSV_PATH, AUDIO_DIR, DIAG_PATH, PARTIALS_PATH
    global DIARIZE_ENABLED, DIARIZE_MODE, DIARIZE_JSONL, DIARIZE_INTERVAL_S
    SOURCE_LANG = args.lang
    TARGET_LANG = "es" if args.lang == "en" else "en"
    WHISPER_PROMPT = WHISPER_PROMPT_ES if SOURCE_LANG == "es" else WHISPER_PROMPT_EN

    # Re-derive session paths with language tag so EN/ES data stays separate
    SESSION_ID = args.session_id or f"{datetime.now():%Y%m%d_%H%M%S}_{SOURCE_LANG}"
    from tools.session_lifecycle import finish_session, request_graceful_stop, source_provenance, start_session

    source = source_provenance(Path(__file__))
    lifecycle_root = Path.cwd()
    lifecycle = start_session(
        lifecycle_root,
        SESSION_ID,
        git_sha=source["git_sha"],
        source=source,
        pipeline_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    global _health, _RECORD_AUDIO
    _RECORD_AUDIO = not args.no_record_audio
    _health = PipelineHealth(lifecycle_root, SESSION_ID, persistence=_io_pool, record_audio=_RECORD_AUDIO)
    _health.start()
    CSV_PATH = f"metrics/ab_metrics_{SESSION_ID}.csv"
    AUDIO_DIR = f"stark_data/live_sessions/{SESSION_ID}"
    DIAG_PATH = f"metrics/diagnostics_{SESSION_ID}.jsonl"
    PARTIALS_PATH = f"metrics/partials_{SESSION_ID}.jsonl"
    DIARIZE_ENABLED = bool(args.diarize)
    DIARIZE_MODE = args.diarize_mode
    DIARIZE_INTERVAL_S = float(args.diarize_interval_s)
    DIARIZE_JSONL = f"metrics/diarization_{SESSION_ID}.jsonl"
    os.makedirs(os.path.dirname(PARTIALS_PATH), exist_ok=True)
    with open(PARTIALS_PATH, "w"):
        pass

    # --- Configure structured logging ---
    log_level = getattr(logging, args.log_level, logging.INFO)
    _console_handler.setLevel(log_level)
    LOG_PATH = f"metrics/session_{SESSION_ID}.log"
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.environ.get("STARK_OPERATOR_CAPTURE"):
        from tools.operational_logging import configure_log

        configure_log(logger, LOG_PATH)
    logger.info("Session %s started — log file: %s", SESSION_ID, LOG_PATH)

    global CHUNK_DURATION, WS_PORT, VAD_THRESHOLD, MIC_DEVICE, MIC_GAIN, NUM_DRAFT_TOKENS
    global WORD_TIMESTAMPS, BEAM_SIZE, MULTIPROCESS, MUSIC_THRESHOLD, MUSIC_HOLDOFF
    CHUNK_DURATION = args.chunk_duration
    WS_PORT = args.ws_port
    VAD_THRESHOLD = args.vad_threshold
    for name in ("silence_trigger", "partial_interval"):
        value = getattr(args, name)
        if value is not None:
            setattr(settings.vad, name, value)
    if args.vad_backend is not None:
        settings.vad.backend = args.vad_backend
    for name in ("idle_warmup_only", "final_aware_partials", "routing_policy", "terminology_prompt"):
        if getattr(args, name) is not None:
            setattr(settings.translation, name, getattr(args, name))
    MIC_DEVICE = args.device
    if args.audio_file:
        MIC_GAIN = 1.0
    if args.gain is not None:
        MIC_GAIN = args.gain  # Explicit gain skips auto-calibration
    NUM_DRAFT_TOKENS = args.num_draft_tokens
    WORD_TIMESTAMPS = args.word_timestamps
    BEAM_SIZE = args.beam_size
    MULTIPROCESS = args.multiprocess

    # Propagate STT engine knobs into settings so any code path constructing an
    # HFWhisperEngine via engines.factory.create_stt_engine() picks them up.
    if args.compile_mode is not None:
        settings.stt.compile_mode = args.compile_mode
    settings.stt.warmup_seconds = args.warmup_seconds
    settings.stt.backend = args.stt_backend
    MUSIC_THRESHOLD = args.music_threshold
    MUSIC_HOLDOFF = args.music_holdoff

    # MLX translation: model IDs from settings, adapters + TurboQuant + Gemma4/MTS
    global MLX_MODEL_A, MLX_MODEL_B, ADAPTER_DIR_A, ADAPTER_DIR_B
    global USE_TURBOQUANT, TURBOQUANT_KEY_BITS, TURBOQUANT_VAL_BITS
    global MODEL_FAMILY, MLX_DRAFT_MODEL_ID, USE_MTS
    MODEL_FAMILY = args.model_family or settings.translation.model_family
    settings.translation.model_family = MODEL_FAMILY

    if args.no_mts:
        USE_MTS = False
    elif args.mts is True:
        USE_MTS = True
    else:
        USE_MTS = bool(settings.translation.mlx_mts)
    settings.translation.mlx_mts = USE_MTS

    if args.mlx_drafter:
        MLX_DRAFT_MODEL_ID = args.mlx_drafter
    elif MODEL_FAMILY == "gemma4" and USE_MTS:
        MLX_DRAFT_MODEL_ID = settings.translation.mlx_drafter_gemma4
    else:
        MLX_DRAFT_MODEL_ID = None

    if args.mlx_model:
        MLX_MODEL_A = args.mlx_model
    elif MODEL_FAMILY == "gemma4":
        from engines.factory import resolve_mlx_translation_model_id

        MLX_MODEL_A = resolve_mlx_translation_model_id(model_family="gemma4", size=args.gemma4_size)
    else:
        MLX_MODEL_A = settings.translation.mlx_model_4b
    MLX_MODEL_B = settings.translation.mlx_model_12b

    # Gemma-4 MTS: prefer gamma=1 on Metal unless user explicitly set --num-draft-tokens
    if MODEL_FAMILY == "gemma4" and USE_MTS and args.num_draft_tokens == 3:
        NUM_DRAFT_TOKENS = 1
    ADAPTER_DIR_A = args.adapter_dir
    ADAPTER_DIR_B = args.adapter_dir_b
    if args.no_turboquant:
        USE_TURBOQUANT = False
    elif args.turboquant is True:
        USE_TURBOQUANT = True
    else:
        USE_TURBOQUANT = bool(settings.translation.turboquant)
    TURBOQUANT_KEY_BITS = settings.translation.turboquant_key_bits
    TURBOQUANT_VAL_BITS = settings.translation.turboquant_val_bits
    # Keep settings in sync so factory-created engines match live path
    settings.translation.turboquant = USE_TURBOQUANT

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        global _session_stop_requested
        _session_stop_requested = True
        print("\n\nStopping...")
        if request_graceful_stop(_session_main_task):
            return
        print_summary()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    completed = False
    try:
        asyncio.run(main_async(args))
        completed = True
    except KeyboardInterrupt:
        print_summary()
    except asyncio.CancelledError:
        if not _session_stop_requested:
            raise
    finally:
        if _managed_llama_server is not None:
            _managed_llama_server.stop()
        # A completed marker is export evidence: all queued diagnostics must be
        # on disk, and the pipeline must have drained without an abnormal exit.
        failure = sys.exc_info()[1]
        _io_pool.shutdown(wait=True)
        completed = completed or _clean_session_shutdown
        persistence = _io_pool.snapshot()
        completed = completed and persistence["ok"] and persistence["pending"] == 0
        health_snapshot = _health.snapshot() if _health is not None else {}
        if _health is not None and (
            not health_snapshot["input_seen"] or health_snapshot["recording"]["required_failures"]
        ):
            completed = False
        outcome = "completed" if completed else "interrupted" if _session_stop_requested else "failed"
        if not persistence["ok"] or health_snapshot.get("recording", {}).get("required_failures"):
            outcome = "failed"
        if _health is not None:
            _health.close(outcome)
        exit_code = getattr(failure, "code", 1) if failure is not None else 1
        finish_session(
            lifecycle_root,
            SESSION_ID,
            run_id=lifecycle["run_id"],
            status=outcome,
            exit_code=0 if completed else exit_code if isinstance(exit_code, int) and exit_code else 1,
            model_ids=_session_model_ids,
            persistence=persistence,
            native_server=getattr(globals().get("_managed_llama_server"), "provenance", None),
        )


if __name__ == "__main__":
    main()
