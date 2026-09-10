"""Worker processes for multiprocess STT/Translation pipeline.

When --multiprocess is enabled, STT and Translation run in separate OS
processes, each with its own Metal context. This enables true GPU parallelism:
STT(N+1) can run while Translation(N) is still in progress.

Architecture:
    Main Process          STT Worker Process       Translation Worker Process
    ┌─────────────┐       ┌────────────────┐       ┌─────────────────────────┐
    │ VAD (inline) │       │ mlx-whisper    │       │ TranslateGemma 4B/12B   │
    │ MarianMT     │  Pipe │ (own Metal ctx)│  Pipe │ (own Metal ctx)         │
    │ WebSocket/HTTP│ ←──→ │                │ ←──→ │ + prompt cache           │
    │ Coordination │       │                │       │ + speculative decoding   │
    └─────────────┘       └────────────────┘       └─────────────────────────┘
"""

import os
import time

import numpy as np


def stt_worker_main(conn, model_id, cache_limit_mb=256, source_lang="en"):
    """STT worker process entry point.

    Loads mlx-whisper, warms up, then processes transcription requests
    until a shutdown sentinel (None) is received.

    Protocol:
        Request:  ("transcribe", audio_ndarray, whisper_prompt, word_timestamps, beam_size)
        Response: (english, latency_ms, confidence, segment_meta, low_conf_words)
        Shutdown: None
    """
    os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    import mlx.core as mx
    import mlx_whisper

    from engines.model_paths import resolve_model_path

    model_id = resolve_model_path(model_id)
    mx.set_cache_limit(cache_limit_mb * 1024 * 1024)

    # Load and warm up model
    silence = np.zeros(16000, dtype=np.float32)
    mlx_whisper.transcribe(
        silence,
        path_or_hf_repo=model_id,
        condition_on_previous_text=False,
    )

    conn.send("ready")

    while True:
        request = conn.recv()
        if request is None:
            break

        _, audio, prompt, word_ts, _beam_sz = request
        t0 = time.perf_counter()

        # mlx-whisper is always greedy (beam search not implemented);
        # beam_sz is accepted in the protocol but ignored here.
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=model_id,
            language=source_lang,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            word_timestamps=word_ts,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        english = result["text"].strip()

        # Extract segment metadata (mirrors _run_stt_mlx in dry_run_ab.py)
        confidence = None
        segment_meta = []
        low_conf_words = []
        segments = result.get("segments", [])
        if segments:
            logprobs = []
            for seg in segments:
                meta = {
                    "avg_logprob": seg.get("avg_logprob"),
                    "no_speech_prob": seg.get("no_speech_prob"),
                    "compression_ratio": seg.get("compression_ratio"),
                }
                segment_meta.append(meta)
                if meta["avg_logprob"] is not None:
                    logprobs.append(meta["avg_logprob"])
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
            if logprobs:
                mean_lp = sum(logprobs) / len(logprobs)
                confidence = round(min(1.0, max(0.0, 1.0 + mean_lp)), 2)

        conn.send((english, latency_ms, confidence, segment_meta, low_conf_words))


def translation_worker_main(
    conn,
    model_4b_id,
    model_12b_id=None,
    num_draft_tokens=3,
    cache_limit_mb=256,
    source_lang="en",
    target_lang="es",
    model_family="translategemma",
    adapter_path=None,
    adapter_b_path=None,
    terminology_prompt="none",
):
    """Worker facade over the canonical loader, prompts, stop rules and telemetry.

    The historical six-value pipe response remains unchanged. The parent passes
    the selected model family explicitly; Gemma 4 never receives TG prompts.
    """
    os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    from engines.mlx_engine import MLXGemmaEngine

    def make_engine(model_id, adapter):
        engine = MLXGemmaEngine(
            model_id=model_id,
            cache_limit_mb=cache_limit_mb,
            model_family=model_family,
            adapter_path=adapter,
            terminology_prompt=terminology_prompt,
        )
        engine.load()
        return engine

    try:
        a = make_engine(model_4b_id, adapter_path)
        b = make_engine(model_12b_id, adapter_b_path) if model_12b_id else None
        if b is not None and model_family == "translategemma":
            b._draft_model = a._model
            b._num_draft_tokens = num_draft_tokens
    except Exception as exc:
        conn.send({"error": str(exc)})
        return
    conn.send("ready")
    try:
        while True:
            request = conn.recv()
            if request is None:
                break
            _, text, run_ab = request
            result_a = a.translate(text, source_lang=source_lang, target_lang=target_lang)
            result_b = b.translate(text, source_lang=source_lang, target_lang=target_lang) if run_ab and b else None
            conn.send(
                (
                    result_a.text,
                    result_a.latency_ms,
                    result_a.tokens_per_second,
                    result_b.text if result_b else None,
                    result_b.latency_ms if result_b else 0.0,
                    result_b.tokens_per_second if result_b else 0.0,
                )
            )
    finally:
        a.unload()
        if b:
            b.unload()
