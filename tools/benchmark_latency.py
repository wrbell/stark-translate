#!/usr/bin/env python3
"""
benchmark_latency.py -- Comprehensive Pipeline Latency Benchmark for SRTranslate

Validates P7 Phase 1 + Phase 2 latency optimizations on MacBook (M3 Pro, 18GB, MLX).
Tests each pipeline stage independently, measures optimization deltas, and generates
an impact report comparing before/after for every optimization.

Usage:
    python benchmark_latency.py                        # Run all benchmarks
    python benchmark_latency.py --only stt             # STT benchmarks only
    python benchmark_latency.py --only translate        # Translation benchmarks only
    python benchmark_latency.py --only e2e             # End-to-end pipeline only
    python benchmark_latency.py --only vad             # VAD benchmarks only
    python benchmark_latency.py --only io              # I/O benchmarks only
    python benchmark_latency.py --only resample        # Resampling benchmarks only
    python benchmark_latency.py --quick                # Fewer runs (3 measured, 1 warmup)
    python benchmark_latency.py --audio sermon.wav     # Use real audio file
    python benchmark_latency.py --runs 10 --warmup 3   # Custom run count
    python benchmark_latency.py --output metrics/latency_benchmark.json
"""

import argparse
import copy
import json
import os
import queue as queue_module
import statistics
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
MIC_SAMPLE_RATE = 48000
MODEL_ID = "wbell7/distil-whisper-large-v3.5-mlx"
MLX_MODEL_A = "mlx-community/translategemma-4b-it-4bit"
MLX_MODEL_B = "mlx-community/translategemma-12b-it-4bit"

DEFAULT_RUNS = 5
DEFAULT_WARMUP = 2
QUICK_RUNS = 3
QUICK_WARMUP = 1

# Theological prompt from dry_run_ab.py
WHISPER_PROMPT = (
    "Sermon at Stark Road Gospel Hall. "
    "Christ Jesus, the Holy Spirit, God the Father. "
    "Atonement, propitiation, mediator, covenant, righteousness, "
    "sanctification, justification, redemption, reconciliation, "
    "repentance, reign, grace, mercy, the Gospel, epistle, apostle, "
    "Scripture, the Lord, the Word of God."
)

# Test sentences -- theological/church content at various lengths
TEST_SENTENCES = {
    "short": "The grace of God is sufficient.",
    "medium": (
        "For God so loved the world that he gave his only begotten son, "
        "that whosoever believeth in him should not perish."
    ),
    "long": (
        "Brothers and sisters, let us turn to the book of Romans chapter eight, "
        "where the apostle Paul reminds us that there is therefore now no "
        "condemnation to them which are in Christ Jesus, who walk not after "
        "the flesh but after the Spirit."
    ),
}

# Speculative decoding sweep values
SPEC_DRAFT_TOKENS = [3, 4, 6, 8, 12, 16]


# ---------------------------------------------------------------------------
# Utility: stats helper
# ---------------------------------------------------------------------------


def compute_stats(latencies):
    """Compute summary stats for a list of latency measurements."""
    if not latencies:
        return {"mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0, "stdev": 0, "n": 0}
    sorted_lats = sorted(latencies)
    p95_idx = max(0, int(len(sorted_lats) * 0.95) - 1)
    return {
        "mean": round(statistics.mean(latencies), 2),
        "median": round(statistics.median(latencies), 2),
        "p95": round(sorted_lats[p95_idx], 2),
        "min": round(min(latencies), 2),
        "max": round(max(latencies), 2),
        "stdev": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0.0,
        "n": len(latencies),
    }


def fmt_stats(stats):
    """Format stats dict as a compact string."""
    return (
        f"mean={stats['mean']:.1f}ms  median={stats['median']:.1f}ms  "
        f"p95={stats['p95']:.1f}ms  min={stats['min']:.1f}ms  max={stats['max']:.1f}ms  "
        f"stdev={stats['stdev']:.1f}ms  (n={stats['n']})"
    )


def delta_str(before_mean, after_mean):
    """Format a before/after delta as -XXms (XX% reduction)."""
    diff = after_mean - before_mean
    if before_mean > 0:
        pct = (diff / before_mean) * 100
    else:
        pct = 0
    return f"{diff:+.1f}ms ({pct:+.1f}%)"


# ---------------------------------------------------------------------------
# Audio generation / loading (from stt_benchmark.py)
# ---------------------------------------------------------------------------


def generate_test_audio(duration_s=3.0):
    """Generate synthetic speech-like audio for benchmarking.
    Mix of tones that exercises Whisper's encoder. Not real speech --
    latency measurements are valid but transcription text will be meaningless.
    """
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), dtype=np.float32)
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 500 * t)
        + 0.1 * np.sin(2 * np.pi * 1200 * t)
        + 0.05 * np.random.randn(len(t)).astype(np.float32)
    )
    audio = audio / (np.max(np.abs(audio)) + 1e-7) * 0.5
    return audio


def generate_48k_audio(duration_s=1.0):
    """Generate 48kHz audio to test resampling (simulates mic input)."""
    t = np.linspace(0, duration_s, int(MIC_SAMPLE_RATE * duration_s), dtype=np.float32)
    audio = (
        0.3 * np.sin(2 * np.pi * 300 * t)
        + 0.15 * np.sin(2 * np.pi * 1000 * t)
        + 0.05 * np.random.randn(len(t)).astype(np.float32)
    )
    return audio / (np.max(np.abs(audio)) + 1e-7) * 0.5


def load_audio_file(path):
    """Load an audio file and return 16kHz mono float32 numpy array."""
    import subprocess

    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        import scipy.io.wavfile as wav

        sr, data = wav.read(path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        if len(data.shape) > 1:
            data = data[:, 0]
        if sr != SAMPLE_RATE:
            from scipy.signal import resample

            target_len = int(len(data) * SAMPLE_RATE / sr)
            data = resample(data, target_len).astype(np.float32)
        return data
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", tmp_path],
                capture_output=True,
                check=True,
            )
            return load_audio_file(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

_models_loaded = {}  # cache to avoid reloading


def _load_mlx_whisper():
    """Load/warm mlx-whisper and return the model_id string."""
    if "whisper" in _models_loaded:
        return _models_loaded["whisper"]
    import mlx.core as mx
    import mlx_whisper

    mx.set_cache_limit(256 * 1024 * 1024)
    print("  Loading mlx-whisper (distil-large-v3.5)...")
    t0 = time.perf_counter()
    silence = np.zeros(16000, dtype=np.float32)
    mlx_whisper.transcribe(silence, path_or_hf_repo=MODEL_ID, condition_on_previous_text=False)
    elapsed = time.perf_counter() - t0
    print(f"  Whisper ready ({elapsed:.1f}s)")
    _models_loaded["whisper"] = MODEL_ID
    return MODEL_ID


def _load_mlx_gemma(model_id, label):
    """Load a TranslateGemma model via MLX with EOS fix."""
    key = model_id
    if key in _models_loaded:
        return _models_loaded[key]
    import mlx.core as mx
    from mlx_lm import load

    mx.set_cache_limit(256 * 1024 * 1024)
    print(f"  Loading {label} ({model_id})...")
    t0 = time.perf_counter()
    model, tokenizer = load(model_id)
    # EOS fix: add <end_of_turn> (id=106) to EOS set
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    default_eos = tokenizer.eos_token_id
    if not hasattr(tokenizer, "_eos_token_ids") or eot_id not in tokenizer._eos_token_ids:
        tokenizer._eos_token_ids = {default_eos, eot_id}
    elapsed = time.perf_counter() - t0
    print(f"  {label} ready ({elapsed:.1f}s)")
    _models_loaded[key] = (model, tokenizer)
    return model, tokenizer


def _build_prompt_cache(model, tokenizer, label):
    """Build prompt cache for TranslateGemma (replicates dry_run_ab._build_prompt_cache)."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache

    marker = "SPLIT_HERE"
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "source_lang_code": "en", "target_lang_code": "es", "text": marker}],
        }
    ]
    full_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    if isinstance(full_prompt, str):
        full_tokens = tokenizer.encode(full_prompt, add_special_tokens=False)
    else:
        full_tokens = list(full_prompt)

    marker_tokens = tokenizer.encode(marker, add_special_tokens=False)
    marker_len = len(marker_tokens)
    prefix_end = None
    for i in range(len(full_tokens) - marker_len + 1):
        if full_tokens[i : i + marker_len] == marker_tokens:
            prefix_end = i
            break

    if prefix_end is None:
        print(f"  WARNING: Could not locate marker for {label}, skipping cache")
        return None, 0, []

    prefix_tokens = full_tokens[:prefix_end]
    suffix_tokens = full_tokens[prefix_end + marker_len :]

    if len(prefix_tokens) < 3:
        print(f"  WARNING: Prefix too short for {label}, skipping cache")
        return None, 0, suffix_tokens

    prompt_cache = make_prompt_cache(model)
    prompt_array = mx.array(prefix_tokens)
    for _ in generate_step(prompt_array, model, max_tokens=0, prompt_cache=prompt_cache):
        pass
    mx.eval([c.state for c in prompt_cache])
    print(f"  {label} prompt cache: {len(prefix_tokens)} prefix tokens, {len(suffix_tokens)} suffix tokens")
    return prompt_cache, len(prefix_tokens), suffix_tokens


def _load_vad():
    """Load Silero VAD."""
    if "vad" in _models_loaded:
        return _models_loaded["vad"]
    import torch

    print("  Loading Silero VAD...")
    t0 = time.perf_counter()
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad")
    elapsed = time.perf_counter() - t0
    print(f"  VAD ready ({elapsed:.1f}s)")
    _models_loaded["vad"] = (model, utils)
    return model, utils


def _load_marian():
    """Load MarianMT PyTorch model."""
    if "marian" in _models_loaded:
        return _models_loaded["marian"]
    from transformers import MarianMTModel, MarianTokenizer

    model_id = "Helsinki-NLP/opus-mt-en-es"
    print(f"  Loading MarianMT PyTorch ({model_id})...")
    t0 = time.perf_counter()
    tokenizer = MarianTokenizer.from_pretrained(model_id)
    model = MarianMTModel.from_pretrained(model_id)
    model.eval()
    import torch

    inputs = tokenizer("Hello", return_tensors="pt", padding=True)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=16)
    elapsed = time.perf_counter() - t0
    print(f"  MarianMT PyTorch ready ({elapsed:.1f}s)")
    _models_loaded["marian"] = (model, tokenizer)
    return model, tokenizer


def _load_ct2_marian():
    """Load CTranslate2 int8 MarianMT."""
    if "ct2" in _models_loaded:
        return _models_loaded["ct2"]
    ct2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ct2_opus_mt_en_es")
    if not os.path.isdir(ct2_path):
        print("  CT2 model not found, skipping")
        _models_loaded["ct2"] = None
        return None
    import ctranslate2
    from transformers import MarianTokenizer

    print("  Loading CTranslate2 MarianMT (int8)...")
    t0 = time.perf_counter()
    translator = ctranslate2.Translator(ct2_path, device="cpu", compute_type="int8")
    tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    warm_tokens = tok.convert_ids_to_tokens(tok.encode("Hello world."))
    translator.translate_batch([warm_tokens], max_decoding_length=32)
    elapsed = time.perf_counter() - t0
    print(f"  CT2 MarianMT ready ({elapsed:.1f}s)")
    _models_loaded["ct2"] = translator
    return translator


# ---------------------------------------------------------------------------
# Translation helper (replicates dry_run_ab.translate_mlx logic)
# ---------------------------------------------------------------------------


def translate_mlx(
    model,
    tokenizer,
    text,
    draft_model=None,
    prompt_cache_template=None,
    suffix_tokens=None,
    max_tok_ratio=1.8,
    num_draft_tokens=3,
):
    """Translate English->Spanish using TranslateGemma via MLX.
    Returns (translation, latency_ms, tokens_per_sec).
    """
    from mlx_lm import generate

    if model is None or tokenizer is None:
        return "(model not loaded)", 0.0, 0.0

    input_words = len(text.split())
    max_tok = max(32, int(input_words * max_tok_ratio))

    use_cache = prompt_cache_template is not None and suffix_tokens is not None and draft_model is None

    if use_cache:
        cached = copy.deepcopy(prompt_cache_template)
        text_tokens = tokenizer.encode(text, add_special_tokens=False)
        dynamic_tokens = text_tokens + suffix_tokens
        gen_kwargs = dict(
            prompt=dynamic_tokens,
            max_tokens=max_tok,
            verbose=False,
            prompt_cache=cached,
        )
    else:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "source_lang_code": "en", "target_lang_code": "es", "text": text}],
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        gen_kwargs = dict(prompt=prompt, max_tokens=max_tok, verbose=False)
        if draft_model is not None:
            gen_kwargs["draft_model"] = draft_model
            gen_kwargs["num_draft_tokens"] = num_draft_tokens

    t0 = time.perf_counter()
    result = generate(model, tokenizer, **gen_kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000

    clean = result.split("<end_of_turn>")[0].strip()
    out_tokens = len(tokenizer.encode(clean))
    gen_tps = out_tokens / (latency_ms / 1000) if latency_ms > 0 else 0.0
    return clean, latency_ms, gen_tps


# ===================================================================
#  1. STT BENCHMARKS
# ===================================================================


def bench_stt_suite(audio, runs, warmup):
    """Run all STT benchmark variants and return results dict."""
    import mlx_whisper

    _load_mlx_whisper()
    results = {}

    def _run_stt(label, kwargs, n_runs, n_warmup):
        for _ in range(n_warmup):
            mlx_whisper.transcribe(audio, **kwargs)
        latencies = []
        texts = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            r = mlx_whisper.transcribe(audio, **kwargs)
            latencies.append((time.perf_counter() - t0) * 1000)
            texts.append(r.get("text", "").strip())
        stats = compute_stats(latencies)
        stats["text_sample"] = texts[0] if texts else ""
        stats["text_consistent"] = len(set(texts)) == 1
        return stats

    base_kwargs = dict(path_or_hf_repo=MODEL_ID, language="en", condition_on_previous_text=False)

    # 1. word_timestamps=True vs False
    print("\n  [1/6] STT with word_timestamps=True...")
    results["stt_word_ts_on"] = _run_stt(
        "word_timestamps=True", {**base_kwargs, "word_timestamps": True, "initial_prompt": WHISPER_PROMPT}, runs, warmup
    )
    print(f"        {fmt_stats(results['stt_word_ts_on'])}")

    print("  [2/6] STT with word_timestamps=False...")
    results["stt_word_ts_off"] = _run_stt(
        "word_timestamps=False",
        {**base_kwargs, "word_timestamps": False, "initial_prompt": WHISPER_PROMPT},
        runs,
        warmup,
    )
    print(f"        {fmt_stats(results['stt_word_ts_off'])}")

    # 2. with/without initial_prompt
    print("  [3/6] STT with initial_prompt...")
    results["stt_with_prompt"] = _run_stt(
        "with prompt", {**base_kwargs, "word_timestamps": False, "initial_prompt": WHISPER_PROMPT}, runs, warmup
    )
    print(f"        {fmt_stats(results['stt_with_prompt'])}")

    print("  [4/6] STT without initial_prompt...")
    results["stt_no_prompt"] = _run_stt("no prompt", {**base_kwargs, "word_timestamps": False}, runs, warmup)
    print(f"        {fmt_stats(results['stt_no_prompt'])}")

    # 3. long vs short prev_text
    long_prev = (
        "And so we see that the righteousness of God is revealed from faith to faith, as it is written, the just shall live by faith. This is the message of Romans chapter one."
        + " Filler text." * 5
    )
    short_prev = long_prev[-100:]
    long_prompt = WHISPER_PROMPT + " " + long_prev[-200:]
    short_prompt = WHISPER_PROMPT + " " + short_prev

    print("  [5/6] STT with long prev_text (200 chars)...")
    results["stt_long_prev"] = _run_stt(
        "long prev_text", {**base_kwargs, "word_timestamps": False, "initial_prompt": long_prompt}, runs, warmup
    )
    print(f"        {fmt_stats(results['stt_long_prev'])}")

    print("  [6/6] STT with short prev_text (100 chars)...")
    results["stt_short_prev"] = _run_stt(
        "short prev_text", {**base_kwargs, "word_timestamps": False, "initial_prompt": short_prompt}, runs, warmup
    )
    print(f"        {fmt_stats(results['stt_short_prev'])}")

    return results


# ===================================================================
#  2. TRANSLATION BENCHMARKS
# ===================================================================


def bench_translate_suite(runs, warmup):
    """Run all translation benchmark variants and return results dict."""
    results = {}

    # Load models
    print("\n  Loading translation models...")
    model_a, tok_a = _load_mlx_gemma(MLX_MODEL_A, "4B")
    cache_a, _, suffix_a = _build_prompt_cache(model_a, tok_a, "4B")

    # Helper: bench a single translation config across sentence lengths
    def _run_translate(label, translate_fn, n_runs, n_warmup, sentences=None):
        if sentences is None:
            sentences = TEST_SENTENCES
        per_length = {}
        for length_key, text in sentences.items():
            # Warmup
            for _ in range(n_warmup):
                translate_fn(text)
            latencies = []
            tps_list = []
            output_text = ""
            for _ in range(n_runs):
                result, lat, tps = translate_fn(text)
                latencies.append(lat)
                tps_list.append(tps)
                output_text = result
            stats = compute_stats(latencies)
            stats["tps_mean"] = round(statistics.mean(tps_list), 1) if tps_list else 0
            stats["output"] = output_text
            stats["input_words"] = len(text.split())
            per_length[length_key] = stats
        return per_length

    # --- 4B without cache ---
    print("\n  [1/5] TranslateGemma 4B without prompt cache...")
    results["4b_no_cache"] = _run_translate(
        "4B no cache", lambda text: translate_mlx(model_a, tok_a, text), runs, warmup
    )
    for k, v in results["4b_no_cache"].items():
        print(f"        {k}: {fmt_stats(v)}  tps={v['tps_mean']}")

    # --- 4B with cache ---
    print("  [2/5] TranslateGemma 4B with prompt cache...")
    results["4b_with_cache"] = _run_translate(
        "4B with cache",
        lambda text: translate_mlx(model_a, tok_a, text, prompt_cache_template=cache_a, suffix_tokens=suffix_a),
        runs,
        warmup,
    )
    for k, v in results["4b_with_cache"].items():
        print(f"        {k}: {fmt_stats(v)}  tps={v['tps_mean']}")

    # --- max_tokens ratio comparison (2.5x vs 1.8x) ---
    print("  [3/5] TranslateGemma 4B max_tokens ratio 2.5x vs 1.8x...")
    results["4b_ratio_2.5"] = _run_translate(
        "4B ratio 2.5x",
        lambda text: translate_mlx(
            model_a, tok_a, text, prompt_cache_template=cache_a, suffix_tokens=suffix_a, max_tok_ratio=2.5
        ),
        runs,
        warmup,
    )
    results["4b_ratio_1.8"] = _run_translate(
        "4B ratio 1.8x",
        lambda text: translate_mlx(
            model_a, tok_a, text, prompt_cache_template=cache_a, suffix_tokens=suffix_a, max_tok_ratio=1.8
        ),
        runs,
        warmup,
    )
    for k in TEST_SENTENCES:
        print(
            f"        {k}: 2.5x={results['4b_ratio_2.5'][k]['mean']:.1f}ms  "
            f"1.8x={results['4b_ratio_1.8'][k]['mean']:.1f}ms  "
            f"delta={delta_str(results['4b_ratio_2.5'][k]['mean'], results['4b_ratio_1.8'][k]['mean'])}"
        )

    # --- MarianMT CT2 vs PyTorch ---
    print("  [4/5] MarianMT: CT2 int8 vs PyTorch fp32...")
    marian_model, marian_tok = _load_marian()
    ct2_translator = _load_ct2_marian()

    import torch

    def _translate_marian_pt(text):
        t0 = time.perf_counter()
        inputs = marian_tok(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            translated = marian_model.generate(**inputs, max_new_tokens=128)
        result = marian_tok.decode(translated[0], skip_special_tokens=True)
        lat = (time.perf_counter() - t0) * 1000
        return result, lat, 0.0

    results["marian_pytorch"] = _run_translate("MarianMT PyTorch", _translate_marian_pt, runs, warmup)
    for k, v in results["marian_pytorch"].items():
        print(f"        PyTorch {k}: {fmt_stats(v)}")

    if ct2_translator is not None:

        def _translate_marian_ct2(text):
            t0 = time.perf_counter()
            src_tokens = marian_tok.convert_ids_to_tokens(marian_tok.encode(text))
            ct2_out = ct2_translator.translate_batch([src_tokens], max_decoding_length=128, beam_size=4)
            result = marian_tok.convert_tokens_to_string(ct2_out[0].hypotheses[0])
            lat = (time.perf_counter() - t0) * 1000
            return result, lat, 0.0

        results["marian_ct2"] = _run_translate("MarianMT CT2", _translate_marian_ct2, runs, warmup)
        for k, v in results["marian_ct2"].items():
            print(f"        CT2    {k}: {fmt_stats(v)}")
    else:
        results["marian_ct2"] = None
        print("        CT2: not available (skipped)")

    # --- Speculative decoding sweep (12B with 4B draft) ---
    print("  [5/5] Speculative decoding sweep (12B with 4B draft)...")
    try:
        model_b, tok_b = _load_mlx_gemma(MLX_MODEL_B, "12B")
        cache_b, _, suffix_b = _build_prompt_cache(model_b, tok_b, "12B")

        # Baseline: 12B without speculative decoding (with cache)
        print("        12B baseline (no spec, with cache)...")
        results["12b_no_spec"] = _run_translate(
            "12B no spec",
            lambda text: translate_mlx(model_b, tok_b, text, prompt_cache_template=cache_b, suffix_tokens=suffix_b),
            runs,
            warmup,
        )
        for k, v in results["12b_no_spec"].items():
            print(f"          {k}: {fmt_stats(v)}  tps={v['tps_mean']}")

        # Sweep num_draft_tokens
        results["spec_sweep"] = {}
        for ndt in SPEC_DRAFT_TOKENS:
            tag = f"spec_{ndt}"
            print(f"        12B speculative num_draft_tokens={ndt}...")
            results["spec_sweep"][tag] = _run_translate(
                f"12B spec ndt={ndt}",
                lambda text, _ndt=ndt: translate_mlx(model_b, tok_b, text, draft_model=model_a, num_draft_tokens=_ndt),
                runs,
                warmup,
                sentences={"medium": TEST_SENTENCES["medium"]},
            )
            v = results["spec_sweep"][tag]["medium"]
            baseline = results["12b_no_spec"]["medium"]["mean"]
            print(f"          medium: {fmt_stats(v)}  vs baseline: {delta_str(baseline, v['mean'])}")

    except Exception as e:
        print(f"        12B loading failed: {e}")
        print("        Speculative decoding benchmarks skipped.")
        results["12b_no_spec"] = None
        results["spec_sweep"] = {}

    return results


# ===================================================================
#  3. VAD BENCHMARKS
# ===================================================================


def bench_vad_suite(runs, warmup):
    """Benchmark VAD inline vs threaded."""
    import torch

    results = {}
    vad_model, vad_utils = _load_vad()

    # Generate ~32ms frames at 16kHz (512 samples per frame)
    frame_dur_samples = 512
    n_frames = 50  # process 50 frames per measurement
    frames = [
        np.clip(
            0.3
            * np.sin(
                2 * np.pi * 300 * np.linspace(0, frame_dur_samples / SAMPLE_RATE, frame_dur_samples, dtype=np.float32)
            )
            + 0.05 * np.random.randn(frame_dur_samples).astype(np.float32),
            -1,
            1,
        )
        for _ in range(n_frames * (runs + warmup))
    ]

    # --- Inline VAD ---
    print("\n  [1/2] VAD inline (synchronous)...")
    frame_idx = 0
    for _ in range(warmup):
        for _ in range(n_frames):
            tensor = torch.from_numpy(frames[frame_idx]).float()
            vad_model(tensor, SAMPLE_RATE)
            frame_idx += 1
    vad_model.reset_states()

    latencies_inline = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for _ in range(n_frames):
            tensor = torch.from_numpy(frames[frame_idx]).float()
            vad_model(tensor, SAMPLE_RATE)
            frame_idx += 1
        latencies_inline.append((time.perf_counter() - t0) * 1000)
        vad_model.reset_states()

    results["vad_inline"] = compute_stats(latencies_inline)
    results["vad_inline"]["per_frame_ms"] = round(results["vad_inline"]["mean"] / n_frames, 3)
    print(f"        {fmt_stats(results['vad_inline'])}  per_frame={results['vad_inline']['per_frame_ms']:.3f}ms")

    # --- Threaded VAD ---
    print("  [2/2] VAD threaded (via queue)...")
    vad_in_q = queue_module.Queue(maxsize=64)
    vad_out_q = queue_module.Queue(maxsize=64)
    stop_event = threading.Event()

    def vad_worker():
        while not stop_event.is_set():
            try:
                chunk = vad_in_q.get(timeout=0.1)
            except queue_module.Empty:
                continue
            tensor = torch.from_numpy(chunk).float()
            has_speech = vad_model(tensor, SAMPLE_RATE).item() > 0.3
            vad_out_q.put((chunk, has_speech))

    thread = threading.Thread(target=vad_worker, daemon=True)
    thread.start()

    # Regenerate frames for threaded benchmark
    frames_t = [
        np.clip(
            0.3
            * np.sin(
                2 * np.pi * 300 * np.linspace(0, frame_dur_samples / SAMPLE_RATE, frame_dur_samples, dtype=np.float32)
            )
            + 0.05 * np.random.randn(frame_dur_samples).astype(np.float32),
            -1,
            1,
        )
        for _ in range(n_frames * (runs + warmup))
    ]

    t_idx = 0
    # Warmup
    for _ in range(warmup):
        for _ in range(n_frames):
            vad_in_q.put(frames_t[t_idx])
            t_idx += 1
        # Drain results
        drained = 0
        while drained < n_frames:
            try:
                vad_out_q.get(timeout=1.0)
                drained += 1
            except queue_module.Empty:
                break
        vad_model.reset_states()

    latencies_threaded = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for _ in range(n_frames):
            vad_in_q.put(frames_t[t_idx])
            t_idx += 1
        # Wait for all results
        drained = 0
        while drained < n_frames:
            try:
                vad_out_q.get(timeout=2.0)
                drained += 1
            except queue_module.Empty:
                break
        latencies_threaded.append((time.perf_counter() - t0) * 1000)
        vad_model.reset_states()

    stop_event.set()
    thread.join(timeout=2.0)

    results["vad_threaded"] = compute_stats(latencies_threaded)
    results["vad_threaded"]["per_frame_ms"] = round(results["vad_threaded"]["mean"] / n_frames, 3)
    print(f"        {fmt_stats(results['vad_threaded'])}  per_frame={results['vad_threaded']['per_frame_ms']:.3f}ms")
    # The key metric for threaded VAD is jitter reduction (stdev), not raw throughput
    print(
        f"        Jitter (stdev): inline={results['vad_inline']['stdev']:.2f}ms  "
        f"threaded={results['vad_threaded']['stdev']:.2f}ms  "
        f"delta={delta_str(results['vad_inline']['stdev'], results['vad_threaded']['stdev'])}"
    )

    return results


# ===================================================================
#  4. I/O BENCHMARKS
# ===================================================================


def bench_io_suite(runs, warmup):
    """Benchmark synchronous vs async I/O writes."""
    import scipy.io.wavfile as wav

    results = {}

    # Generate test data
    audio_chunk = generate_test_audio(2.0)
    csv_row = {
        "chunk_id": 1,
        "timestamp": datetime.now().isoformat(),
        "english": "Test transcription for benchmark",
        "spanish_a": "Transcripcion de prueba para benchmark",
        "stt_latency_ms": 350.0,
        "latency_a_ms": 500.0,
    }
    jsonl_record = {
        "chunk_id": 1,
        "english": "Test",
        "spanish_gemma": "Prueba",
        "stt_confidence": 0.85,
        "review_priority": 0,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "test.wav")
        csv_path = os.path.join(tmpdir, "test.csv")
        jsonl_path = os.path.join(tmpdir, "test.jsonl")

        def _sync_write():
            wav.write(wav_path, SAMPLE_RATE, audio_chunk)
            with open(csv_path, "a") as f:
                f.write(",".join(str(v) for v in csv_row.values()) + "\n")
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(jsonl_record) + "\n")

        # --- Synchronous I/O ---
        print("\n  [1/2] Synchronous I/O (WAV + CSV + JSONL)...")
        for _ in range(warmup):
            _sync_write()

        latencies_sync = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _sync_write()
            latencies_sync.append((time.perf_counter() - t0) * 1000)

        results["io_sync"] = compute_stats(latencies_sync)
        print(f"        {fmt_stats(results['io_sync'])}")

        # --- Async I/O (ThreadPoolExecutor) ---
        print("  [2/2] Async I/O (ThreadPoolExecutor)...")
        pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="io-bench")

        for _ in range(warmup):
            f = pool.submit(_sync_write)
            f.result()

        latencies_async = []
        for _ in range(runs):
            t0 = time.perf_counter()
            future = pool.submit(_sync_write)
            # Measure blocking time on main thread (submit returns immediately)
            submit_time = (time.perf_counter() - t0) * 1000
            latencies_async.append(submit_time)
            future.result()  # ensure completion before next iteration

        pool.shutdown(wait=True)
        results["io_async"] = compute_stats(latencies_async)
        print(f"        {fmt_stats(results['io_async'])} (main thread blocking time)")
        print(f"        Savings: {delta_str(results['io_sync']['mean'], results['io_async']['mean'])}")

    return results


# ===================================================================
#  5. RESAMPLING BENCHMARKS
# ===================================================================


def bench_resample_suite(runs, warmup):
    """Benchmark scipy.signal.resample vs decimate for 48kHz->16kHz.

    Tests two scenarios:
    - Frame-level: ~32ms mic callback frames (1536 samples at 48kHz) -- the
      actual size processed per audio_callback invocation in dry_run_ab.py.
    - Buffer-level: ~2s accumulated speech buffer (96000 samples at 48kHz).
    """
    from scipy.signal import decimate, resample

    results = {}
    factor = MIC_SAMPLE_RATE // SAMPLE_RATE  # 3

    for scenario, duration_s in [("frame_32ms", 0.032), ("buffer_2s", 2.0)]:
        n_samples = int(MIC_SAMPLE_RATE * duration_s)
        audio_48k = generate_48k_audio(duration_s)[:n_samples]
        target_len = int(len(audio_48k) * SAMPLE_RATE / MIC_SAMPLE_RATE)

        # --- scipy.signal.resample ---
        label_r = f"resample_scipy_{scenario}"
        print(f"\n  scipy.signal.resample ({scenario}, {n_samples} samples)...")
        for _ in range(warmup):
            resample(audio_48k, target_len)

        latencies_resample = []
        for _ in range(runs):
            t0 = time.perf_counter()
            out_r = resample(audio_48k, target_len).astype(np.float32)
            latencies_resample.append((time.perf_counter() - t0) * 1000)

        results[label_r] = compute_stats(latencies_resample)
        print(f"        {fmt_stats(results[label_r])}  output_len={len(out_r)}")

        # --- scipy.signal.decimate ---
        label_d = f"resample_decimate_{scenario}"
        print(f"  scipy.signal.decimate ({scenario}, factor=3, zero_phase=False)...")
        for _ in range(warmup):
            decimate(audio_48k, factor, zero_phase=False)

        latencies_decimate = []
        for _ in range(runs):
            t0 = time.perf_counter()
            out_d = decimate(audio_48k, factor, zero_phase=False).astype(np.float32)
            latencies_decimate.append((time.perf_counter() - t0) * 1000)

        results[label_d] = compute_stats(latencies_decimate)
        print(f"        {fmt_stats(results[label_d])}  output_len={len(out_d)}")

        # Quality comparison
        min_len = min(len(out_r), len(out_d))
        if min_len > 1:
            correlation = float(np.corrcoef(out_r[:min_len], out_d[:min_len])[0, 1])
        else:
            correlation = 1.0
        results[f"quality_correlation_{scenario}"] = round(correlation, 6)
        print(f"        Correlation: {correlation:.6f}")
        print(f"        Delta: {delta_str(results[label_r]['mean'], results[label_d]['mean'])}")

    # Store summary keys for report (use frame-level as the representative comparison)
    results["resample_scipy"] = results.get("resample_scipy_frame_32ms", {})
    results["resample_decimate"] = results.get("resample_decimate_frame_32ms", {})
    results["quality_correlation"] = results.get("quality_correlation_frame_32ms", 0)

    return results


# ===================================================================
#  6. METAL CACHE BENCHMARK
# ===================================================================


def bench_metal_cache(audio, runs, warmup):
    """Benchmark mlx-whisper with different Metal cache limits."""
    import mlx.core as mx
    import mlx_whisper

    _load_mlx_whisper()
    results = {}

    base_kwargs = dict(
        path_or_hf_repo=MODEL_ID,
        language="en",
        condition_on_previous_text=False,
        word_timestamps=False,
        initial_prompt=WHISPER_PROMPT,
    )

    for cache_mb in [100, 256]:
        label = f"cache_{cache_mb}mb"
        print(f"\n  Metal cache limit = {cache_mb}MB...")
        mx.set_cache_limit(cache_mb * 1024 * 1024)

        for _ in range(warmup):
            mlx_whisper.transcribe(audio, **base_kwargs)

        latencies = []
        for _ in range(runs):
            t0 = time.perf_counter()
            mlx_whisper.transcribe(audio, **base_kwargs)
            latencies.append((time.perf_counter() - t0) * 1000)

        results[label] = compute_stats(latencies)
        print(f"        {fmt_stats(results[label])}")

    # Reset to 256MB (our optimization)
    mx.set_cache_limit(256 * 1024 * 1024)

    if "cache_100mb" in results and "cache_256mb" in results:
        print(f"        Delta: {delta_str(results['cache_100mb']['mean'], results['cache_256mb']['mean'])}")

    return results


# ===================================================================
#  7. END-TO-END PIPELINE BENCHMARK
# ===================================================================


def bench_e2e_suite(audio, runs, warmup):
    """Benchmark end-to-end pipeline paths."""
    import mlx_whisper
    import torch

    results = {}

    # Load all models
    print("\n  Loading all models for E2E benchmark...")
    _load_mlx_whisper()
    model_a, tok_a = _load_mlx_gemma(MLX_MODEL_A, "4B")
    cache_a, _, suffix_a = _build_prompt_cache(model_a, tok_a, "4B")
    marian_model, marian_tok = _load_marian()
    ct2_translator = _load_ct2_marian()

    stt_kwargs_partial = dict(
        path_or_hf_repo=MODEL_ID,
        language="en",
        condition_on_previous_text=False,
        word_timestamps=False,
        initial_prompt=WHISPER_PROMPT,
    )
    stt_kwargs_final = dict(
        path_or_hf_repo=MODEL_ID,
        language="en",
        condition_on_previous_text=False,
        word_timestamps=True,
        initial_prompt=WHISPER_PROMPT,
    )

    # --- Partial path: STT (no word_ts) + MarianMT CT2 ---
    print("\n  [1/4] E2E Partial path: STT(no wt) + MarianMT CT2...")
    if ct2_translator is not None:
        for _ in range(warmup):
            r = mlx_whisper.transcribe(audio, **stt_kwargs_partial)
            text = r.get("text", "").strip() or "test"
            src_tokens = marian_tok.convert_ids_to_tokens(marian_tok.encode(text))
            ct2_translator.translate_batch([src_tokens], max_decoding_length=128, beam_size=4)

        latencies = []
        for _ in range(runs):
            t0 = time.perf_counter()
            r = mlx_whisper.transcribe(audio, **stt_kwargs_partial)
            text = r.get("text", "").strip() or "test"
            src_tokens = marian_tok.convert_ids_to_tokens(marian_tok.encode(text))
            ct2_translator.translate_batch([src_tokens], max_decoding_length=128, beam_size=4)
            latencies.append((time.perf_counter() - t0) * 1000)

        results["e2e_partial_ct2"] = compute_stats(latencies)
        print(f"        {fmt_stats(results['e2e_partial_ct2'])}")
    else:
        # Fallback to PyTorch MarianMT
        for _ in range(warmup):
            r = mlx_whisper.transcribe(audio, **stt_kwargs_partial)
            text = r.get("text", "").strip() or "test"
            inputs = marian_tok(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                marian_model.generate(**inputs, max_new_tokens=128)

        latencies = []
        for _ in range(runs):
            t0 = time.perf_counter()
            r = mlx_whisper.transcribe(audio, **stt_kwargs_partial)
            text = r.get("text", "").strip() or "test"
            inputs = marian_tok(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                marian_model.generate(**inputs, max_new_tokens=128)
            latencies.append((time.perf_counter() - t0) * 1000)

        results["e2e_partial_pt"] = compute_stats(latencies)
        print(f"        {fmt_stats(results['e2e_partial_pt'])} (PyTorch fallback)")

    # --- Final 4B path: STT (word_ts) + TranslateGemma 4B ---
    print("  [2/4] E2E Final path: STT(wt) + TranslateGemma 4B (cached)...")
    for _ in range(warmup):
        r = mlx_whisper.transcribe(audio, **stt_kwargs_final)
        text = r.get("text", "").strip() or "test"
        translate_mlx(model_a, tok_a, text, prompt_cache_template=cache_a, suffix_tokens=suffix_a)

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        r = mlx_whisper.transcribe(audio, **stt_kwargs_final)
        text = r.get("text", "").strip() or "test"
        translate_mlx(model_a, tok_a, text, prompt_cache_template=cache_a, suffix_tokens=suffix_a)
        latencies.append((time.perf_counter() - t0) * 1000)

    results["e2e_final_4b"] = compute_stats(latencies)
    print(f"        {fmt_stats(results['e2e_final_4b'])}")

    # --- Final A/B parallel: STT + 4B + 12B ---
    print("  [3/4] E2E A/B path: STT(wt) + 4B(cached) || 12B(spec)...")
    try:
        model_b, tok_b = _load_mlx_gemma(MLX_MODEL_B, "12B")

        for _ in range(warmup):
            r = mlx_whisper.transcribe(audio, **stt_kwargs_final)
            text = r.get("text", "").strip() or "test"
            translate_mlx(model_a, tok_a, text, prompt_cache_template=cache_a, suffix_tokens=suffix_a)
            translate_mlx(model_b, tok_b, text, draft_model=model_a, num_draft_tokens=3)

        latencies = []
        lat_stt = []
        lat_4b = []
        lat_12b = []
        for _ in range(runs):
            t0 = time.perf_counter()
            r = mlx_whisper.transcribe(audio, **stt_kwargs_final)
            t_stt = (time.perf_counter() - t0) * 1000
            text = r.get("text", "").strip() or "test"

            t1 = time.perf_counter()
            translate_mlx(model_a, tok_a, text, prompt_cache_template=cache_a, suffix_tokens=suffix_a)
            t_4b = (time.perf_counter() - t1) * 1000

            t2 = time.perf_counter()
            translate_mlx(model_b, tok_b, text, draft_model=model_a, num_draft_tokens=3)
            t_12b = (time.perf_counter() - t2) * 1000

            total = (time.perf_counter() - t0) * 1000
            latencies.append(total)
            lat_stt.append(t_stt)
            lat_4b.append(t_4b)
            lat_12b.append(t_12b)

        results["e2e_ab"] = compute_stats(latencies)
        results["e2e_ab_breakdown"] = {
            "stt": compute_stats(lat_stt),
            "translate_4b": compute_stats(lat_4b),
            "translate_12b": compute_stats(lat_12b),
        }
        print(f"        Total: {fmt_stats(results['e2e_ab'])}")
        print(f"          STT: {fmt_stats(results['e2e_ab_breakdown']['stt'])}")
        print(f"          4B:  {fmt_stats(results['e2e_ab_breakdown']['translate_4b'])}")
        print(f"          12B: {fmt_stats(results['e2e_ab_breakdown']['translate_12b'])}")

    except Exception as e:
        print(f"        12B loading failed: {e} (skipping A/B E2E)")
        results["e2e_ab"] = None
        results["e2e_ab_breakdown"] = None

    # --- Cold start: model loading time ---
    print("  [4/4] Cold start: model load times (from cached state)...")
    results["cold_start"] = {}
    # These are already measured during loading; report from _models_loaded timing
    # We measure a fresh tokenizer.apply_chat_template + first generate as proxy
    messages_test = [
        {
            "role": "user",
            "content": [{"type": "text", "source_lang_code": "en", "target_lang_code": "es", "text": "Hello world."}],
        }
    ]
    prompt = tok_a.apply_chat_template(messages_test, add_generation_prompt=True)
    t0 = time.perf_counter()
    from mlx_lm import generate

    generate(model_a, tok_a, prompt=prompt, max_tokens=5, verbose=False)
    results["cold_start"]["first_generate_4b_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    print(f"        First generate (4B): {results['cold_start']['first_generate_4b_ms']:.1f}ms")

    return results


# ===================================================================
#  REPORT GENERATION
# ===================================================================


def generate_report(all_results):
    """Generate the P7 Optimization Impact Report."""
    print(f"\n{'=' * 70}")
    print("P7 OPTIMIZATION IMPACT REPORT")
    print(f"{'=' * 70}")

    stt = all_results.get("stt", {})
    translate = all_results.get("translate", {})
    vad = all_results.get("vad", {})
    io_results = all_results.get("io", {})
    resample = all_results.get("resample", {})
    metal = all_results.get("metal_cache", {})
    e2e = all_results.get("e2e", {})

    # --- STT Optimizations ---
    print("\nSTT Optimizations:")
    if "stt_word_ts_on" in stt and "stt_word_ts_off" in stt:
        print(
            f"  [P7-1A] word_timestamps disabled:      "
            f"{delta_str(stt['stt_word_ts_on']['mean'], stt['stt_word_ts_off']['mean'])}"
        )
    if "stt_with_prompt" in stt and "stt_no_prompt" in stt:
        print(
            f"  [P7-1E] initial_prompt overhead:        "
            f"{delta_str(stt['stt_no_prompt']['mean'], stt['stt_with_prompt']['mean'])} "
            f"(cost of prompt, accepted for quality)"
        )
    if "stt_long_prev" in stt and "stt_short_prev" in stt:
        print(
            f"  [P7-1E] Shorter prev_text (200->100):   "
            f"{delta_str(stt['stt_long_prev']['mean'], stt['stt_short_prev']['mean'])}"
        )

    # --- Translation Optimizations ---
    print("\nTranslation Optimizations:")
    if "4b_no_cache" in translate and "4b_with_cache" in translate:
        for length_key in TEST_SENTENCES:
            if length_key in translate["4b_no_cache"] and length_key in translate["4b_with_cache"]:
                nc = translate["4b_no_cache"][length_key]["mean"]
                wc = translate["4b_with_cache"][length_key]["mean"]
                print(f"  [P7-2B] Prompt cache 4B ({length_key}):       {delta_str(nc, wc)}")

    if "4b_ratio_2.5" in translate and "4b_ratio_1.8" in translate:
        for length_key in TEST_SENTENCES:
            if length_key in translate["4b_ratio_2.5"] and length_key in translate["4b_ratio_1.8"]:
                r25 = translate["4b_ratio_2.5"][length_key]["mean"]
                r18 = translate["4b_ratio_1.8"][length_key]["mean"]
                print(f"  [P7-2C] max_tokens 2.5x->1.8x ({length_key}): {delta_str(r25, r18)}")

    if "marian_pytorch" in translate and "marian_ct2" in translate and translate["marian_ct2"] is not None:
        for length_key in TEST_SENTENCES:
            if length_key in translate["marian_pytorch"] and length_key in translate["marian_ct2"]:
                pt = translate["marian_pytorch"][length_key]["mean"]
                ct = translate["marian_ct2"][length_key]["mean"]
                print(f"  MarianMT CT2 vs PyTorch ({length_key}):  {delta_str(pt, ct)}")

    # --- Infrastructure Optimizations ---
    print("\nInfrastructure Optimizations:")
    if "cache_100mb" in metal and "cache_256mb" in metal:
        print(
            f"  [P7-4B] Metal cache 100->256MB:         "
            f"{delta_str(metal['cache_100mb']['mean'], metal['cache_256mb']['mean'])}"
        )

    if "resample_scipy" in resample and "resample_decimate" in resample:
        print(
            f"  [P7-5A] decimate vs resample:           "
            f"{delta_str(resample['resample_scipy']['mean'], resample['resample_decimate']['mean'])}"
        )
        if "quality_correlation" in resample:
            print(f"          Quality correlation:             {resample['quality_correlation']:.6f}")

    if "io_sync" in io_results and "io_async" in io_results:
        print(
            f"  [P7-5D] Background I/O:                 "
            f"{delta_str(io_results['io_sync']['mean'], io_results['io_async']['mean'])} "
            f"(main thread blocking)"
        )

    if "vad_inline" in vad and "vad_threaded" in vad:
        print(
            f"  [P7-5B] VAD threading (jitter):         "
            f"{delta_str(vad['vad_inline']['stdev'], vad['vad_threaded']['stdev'])} stdev"
        )

    # --- Speculative Decoding ---
    spec_sweep = translate.get("spec_sweep", {})
    baseline_12b = translate.get("12b_no_spec", {})
    if spec_sweep and baseline_12b:
        print("\nSpeculative Decoding (12B with 4B draft):")
        baseline_mean = baseline_12b.get("medium", {}).get("mean", 0)
        best_ndt = None
        best_mean = float("inf")
        for ndt in SPEC_DRAFT_TOKENS:
            tag = f"spec_{ndt}"
            if tag in spec_sweep and "medium" in spec_sweep[tag]:
                mean = spec_sweep[tag]["medium"]["mean"]
                speedup = delta_str(baseline_mean, mean)
                print(f"  num_draft_tokens={ndt:2d}:                 {mean:.1f}ms  {speedup}")
                if mean < best_mean:
                    best_mean = mean
                    best_ndt = ndt
        if best_ndt is not None:
            print(
                f"  Optimal: num_draft_tokens={best_ndt}           "
                f"{best_mean:.1f}ms  {delta_str(baseline_mean, best_mean)}"
            )

    # --- End-to-End Pipeline ---
    print("\nEnd-to-End Pipeline:")
    if "e2e_partial_ct2" in e2e:
        print(f"  Partial path (STT+MarianMT CT2):        {e2e['e2e_partial_ct2']['mean']:.1f}ms")
    elif "e2e_partial_pt" in e2e:
        print(f"  Partial path (STT+MarianMT PT):         {e2e['e2e_partial_pt']['mean']:.1f}ms")
    if "e2e_final_4b" in e2e:
        print(f"  Final 4B path (STT+TG 4B cached):       {e2e['e2e_final_4b']['mean']:.1f}ms")
    if "e2e_ab" in e2e and e2e["e2e_ab"] is not None:
        print(f"  A/B path (STT+4B+12B sequential):       {e2e['e2e_ab']['mean']:.1f}ms")
        if e2e.get("e2e_ab_breakdown"):
            bd = e2e["e2e_ab_breakdown"]
            print(
                f"    Breakdown: STT={bd['stt']['mean']:.0f}ms  "
                f"4B={bd['translate_4b']['mean']:.0f}ms  "
                f"12B={bd['translate_12b']['mean']:.0f}ms"
            )

    print(f"\n{'=' * 70}")


# ===================================================================
#  MAIN
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive pipeline latency benchmark for SRTranslate P7 optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_latency.py                        # Run all benchmarks
  python benchmark_latency.py --only stt             # STT benchmarks only
  python benchmark_latency.py --only translate        # Translation benchmarks only
  python benchmark_latency.py --only e2e             # End-to-end pipeline only
  python benchmark_latency.py --quick                # Quick mode (fewer runs)
  python benchmark_latency.py --audio sermon.wav     # Use real audio file
  python benchmark_latency.py --runs 10 --warmup 3   # Custom run count
        """,
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=["stt", "translate", "vad", "io", "resample", "metal", "e2e"],
        help="Run only a specific benchmark category",
    )
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer runs for faster results")
    parser.add_argument(
        "--audio", type=str, default=None, help="Path to real audio file (WAV/MP3). Default: synthetic audio"
    )
    parser.add_argument(
        "--duration", type=float, default=3.0, help="Duration of synthetic audio in seconds (default: 3.0)"
    )
    parser.add_argument(
        "--runs", type=int, default=None, help=f"Number of measured runs (default: {DEFAULT_RUNS}, quick: {QUICK_RUNS})"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help=f"Number of warmup runs (default: {DEFAULT_WARMUP}, quick: {QUICK_WARMUP})",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON path (default: metrics/latency_benchmark_TIMESTAMP.json)"
    )
    args = parser.parse_args()

    # Resolve run counts
    if args.quick:
        n_runs = args.runs or QUICK_RUNS
        n_warmup = args.warmup or QUICK_WARMUP
    else:
        n_runs = args.runs or DEFAULT_RUNS
        n_warmup = args.warmup or DEFAULT_WARMUP

    # Load or generate audio
    if args.audio:
        print(f"Loading audio from {args.audio}...")
        audio = load_audio_file(args.audio)
        audio_source = args.audio
    else:
        print(f"Generating {args.duration}s synthetic audio...")
        audio = generate_test_audio(args.duration)
        audio_source = f"synthetic_{args.duration}s"

    audio_duration = len(audio) / SAMPLE_RATE
    print(f"Audio: {audio_duration:.1f}s, {len(audio)} samples, {SAMPLE_RATE}Hz")
    print(f"Runs: {n_runs} measured + {n_warmup} warmup")
    categories = args.only or "all"
    print(f"Categories: {categories}")

    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "audio_source": audio_source,
            "audio_duration_s": round(audio_duration, 2),
            "runs": n_runs,
            "warmup": n_warmup,
            "quick_mode": args.quick,
        }
    }
    run_all = args.only is None

    # --- STT ---
    if run_all or args.only == "stt":
        print(f"\n{'=' * 70}")
        print("STT BENCHMARKS")
        print(f"{'=' * 70}")
        all_results["stt"] = bench_stt_suite(audio, n_runs, n_warmup)

    # --- Metal Cache ---
    if run_all or args.only == "metal":
        print(f"\n{'=' * 70}")
        print("METAL CACHE BENCHMARKS")
        print(f"{'=' * 70}")
        all_results["metal_cache"] = bench_metal_cache(audio, n_runs, n_warmup)

    # --- Translation ---
    if run_all or args.only == "translate":
        print(f"\n{'=' * 70}")
        print("TRANSLATION BENCHMARKS")
        print(f"{'=' * 70}")
        all_results["translate"] = bench_translate_suite(n_runs, n_warmup)

    # --- VAD ---
    if run_all or args.only == "vad":
        print(f"\n{'=' * 70}")
        print("VAD BENCHMARKS")
        print(f"{'=' * 70}")
        all_results["vad"] = bench_vad_suite(n_runs, n_warmup)

    # --- I/O ---
    if run_all or args.only == "io":
        print(f"\n{'=' * 70}")
        print("I/O BENCHMARKS")
        print(f"{'=' * 70}")
        all_results["io"] = bench_io_suite(n_runs, n_warmup)

    # --- Resampling ---
    if run_all or args.only == "resample":
        print(f"\n{'=' * 70}")
        print("RESAMPLING BENCHMARKS")
        print(f"{'=' * 70}")
        all_results["resample"] = bench_resample_suite(n_runs, n_warmup)

    # --- End-to-End ---
    if run_all or args.only == "e2e":
        print(f"\n{'=' * 70}")
        print("END-TO-END PIPELINE BENCHMARKS")
        print(f"{'=' * 70}")
        all_results["e2e"] = bench_e2e_suite(audio, n_runs, n_warmup)

    # --- Report ---
    generate_report(all_results)

    # --- Save JSON ---
    output_path = args.output
    if output_path is None:
        os.makedirs("metrics", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"metrics/latency_benchmark_{timestamp}.json"

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
