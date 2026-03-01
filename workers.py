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

import copy
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
):
    """Translation worker process entry point.

    Loads TranslateGemma 4B (and optionally 12B), builds prompt cache,
    and processes translation requests until shutdown.

    Protocol:
        Request:  ("translate", text, run_ab)
        Response: (translation_a, lat_a, tps_a, translation_b, lat_b, tps_b)
        Shutdown: None
    """
    os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    import mlx.core as mx
    from mlx_lm import load as mlx_load

    mx.set_cache_limit(cache_limit_mb * 1024 * 1024)

    # --- Load 4B model ---
    model_a, tok_a = mlx_load(model_4b_id)  # type: ignore[misc]
    _fix_eos(tok_a)
    prompt_cache, suffix_tokens = _build_prompt_cache(model_a, tok_a, source_lang, target_lang)

    # --- Optional 12B model ---
    model_b, tok_b = None, None
    if model_12b_id:
        model_b, tok_b = mlx_load(model_12b_id)  # type: ignore[misc]
        _fix_eos(tok_b)

    # Warm up with a test translation
    _translate_4b(model_a, tok_a, "Hello", prompt_cache, suffix_tokens, source_lang, target_lang)

    conn.send("ready")

    while True:
        request = conn.recv()
        if request is None:
            break

        _, text, run_ab = request

        # 4B translation
        trans_a, lat_a, tps_a = _translate_4b(
            model_a, tok_a, text, prompt_cache, suffix_tokens, source_lang, target_lang
        )

        # Optional 12B with speculative decoding
        trans_b, lat_b, tps_b = None, 0.0, 0.0
        if run_ab and model_b is not None:
            trans_b, lat_b, tps_b = _translate_12b(
                model_b, tok_b, text, model_a, num_draft_tokens, source_lang, target_lang
            )

        conn.send((trans_a, lat_a, tps_a, trans_b, lat_b, tps_b))


# ---------------------------------------------------------------------------
# Helper functions for translation worker
# ---------------------------------------------------------------------------


def _fix_eos(tokenizer):
    """Apply the TranslateGemma EOS fix.

    TranslateGemma uses <end_of_turn> (id=106) as its actual EOS, but
    the tokenizer default is <eos> (id=1) which the model never generates.
    """
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    default_eos = tokenizer.eos_token_id
    if not hasattr(tokenizer, "_eos_token_ids") or eot_id not in tokenizer._eos_token_ids:
        tokenizer._eos_token_ids = {default_eos, eot_id}


def _build_prompt_cache(model, tokenizer, source_lang="en", target_lang="es"):
    """Build KV prompt cache for the TranslateGemma chat template prefix.

    Same logic as MLXGemmaEngine._build_prompt_cache and
    dry_run_ab._build_prompt_cache.
    """
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache

    marker = "SPLIT_HERE"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source_lang,
                    "target_lang_code": target_lang,
                    "text": marker,
                }
            ],
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
        return None, None

    prefix_tokens = full_tokens[:prefix_end]
    suffix_tokens = full_tokens[prefix_end + marker_len :]

    if len(prefix_tokens) < 3:
        return None, suffix_tokens

    cache = make_prompt_cache(model)
    for _ in generate_step(mx.array(prefix_tokens), model, max_tokens=0, prompt_cache=cache):
        pass
    mx.eval([c.state for c in cache])

    return cache, suffix_tokens


def _translate_4b(model, tokenizer, text, prompt_cache, suffix_tokens, source_lang="en", target_lang="es"):
    """Translate using 4B model with optional prompt cache."""
    from mlx_lm import generate

    input_words = len(text.split())
    max_tok = max(32, int(input_words * 1.8))

    if prompt_cache is not None and suffix_tokens is not None:
        cached = copy.deepcopy(prompt_cache)
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
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": source_lang,
                        "target_lang_code": target_lang,
                        "text": text,
                    }
                ],
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        gen_kwargs = dict(prompt=prompt, max_tokens=max_tok, verbose=False)

    t0 = time.perf_counter()
    result = generate(model, tokenizer, **gen_kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000

    clean = result.split("<end_of_turn>")[0].strip()
    out_tokens = len(tokenizer.encode(clean))
    tps = out_tokens / (latency_ms / 1000) if latency_ms > 0 else 0.0

    return clean, latency_ms, tps


def _translate_12b(model, tokenizer, text, draft_model, num_draft_tokens, source_lang="en", target_lang="es"):
    """Translate using 12B model with speculative decoding (4B as draft)."""
    from mlx_lm import generate

    input_words = len(text.split())
    max_tok = max(32, int(input_words * 1.8))

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source_lang,
                    "target_lang_code": target_lang,
                    "text": text,
                }
            ],
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    gen_kwargs = dict(
        prompt=prompt,
        max_tokens=max_tok,
        verbose=False,
    )
    if draft_model is not None and num_draft_tokens > 0:
        gen_kwargs["draft_model"] = draft_model

    t0 = time.perf_counter()
    result = generate(model, tokenizer, **gen_kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000

    clean = result.split("<end_of_turn>")[0].strip()
    out_tokens = len(tokenizer.encode(clean))
    tps = out_tokens / (latency_ms / 1000) if latency_ms > 0 else 0.0

    return clean, latency_ms, tps
