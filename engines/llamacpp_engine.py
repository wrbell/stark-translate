"""llama.cpp translation engine via HTTP API.

Connects to a running ``llama-server`` instance for translation inference
using GGUF-quantized models. Supports speculative decoding when the server
is started with ``-md`` (draft model) flag.

This engine provides 3-5x faster CUDA inference compared to bitsandbytes NF4
by leveraging llama.cpp's compiled GGUF runtime with native KV cache
quantization (``-ctk q8_0``).

Usage:
    # Start llama-server first:
    # ./llama.cpp/build/bin/llama-server \\
    #   -m models/gemma-4-e4b-it-q4km.gguf \\
    #   -md models/gemma-4-e2b-it-q4km.gguf \\
    #   --draft 16 --draft-min 5 \\
    #   --host 127.0.0.1 --port 8090 -ngl 999

    engine = LlamaCppEngine(server_url="http://127.0.0.1:8090")
    engine.load()  # health check
    result = engine.translate("Hello world", source_lang="en", target_lang="es")
"""

import json
import logging
import time
import urllib.error
import urllib.request

from engines.base import TranslationEngine, TranslationResult
from engines.translation_prompts import clean_translation, llamacpp_user_content

logger = logging.getLogger(__name__)


class LlamaCppEngine(TranslationEngine):
    """Translation engine connecting to llama-server via HTTP.

    Constructor args:
        server_url:      llama-server base URL (default: http://127.0.0.1:8090).
        model_family:    Prompt format: 'gemma4' (instruct) or 'translategemma'.
        timeout_s:       HTTP request timeout in seconds (default: 30).
        max_tokens:      Max tokens to generate (default: 256).
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8090",
        model_family: str = "gemma4",
        timeout_s: float = 30.0,
        max_tokens: int = 256,
    ):
        self._server_url = server_url.rstrip("/")
        self._model_family = model_family
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens
        self._model_id_str: str = ""
        self._loaded = False

    def load(self) -> None:
        """Verify llama-server is running and healthy."""
        try:
            url = f"{self._server_url}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310
                data = json.loads(resp.read())
                status = data.get("status", "unknown")
            if status != "ok":
                raise RuntimeError(f"llama-server health check returned: {status}")
            logger.info("llama-server healthy at %s", self._server_url)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot connect to llama-server at {self._server_url}. "
                f"Start it with: ./llama.cpp/build/bin/llama-server -m <model.gguf> "
                f"--host 127.0.0.1 --port 8090"
            ) from exc

        # Get model info
        try:
            url = f"{self._server_url}/props"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310
                props = json.loads(resp.read())
            self._model_id_str = props.get("default_generation_settings", {}).get("model", "llama.cpp")
            logger.info("llama-server model: %s", self._model_id_str)
        except Exception:
            self._model_id_str = "llama.cpp"

        self._loaded = True

    def translate(
        self,
        text: str,
        *,
        source_lang: str = "en",
        target_lang: str = "es",
    ) -> TranslationResult:
        """Translate text via llama-server /v1/chat/completions endpoint."""
        if not self._loaded:
            raise RuntimeError("Engine not loaded -- call load() first")

        t0 = time.perf_counter()

        # Shared prompt contract (engines/translation_prompts.py) — same string
        # as MLX/CUDA Gemma 4 paths. TranslateGemma falls back to plain text
        # because llama-server does not support structured content blocks.
        user_content = llamacpp_user_content(
            text,
            source_lang=source_lang,
            target_lang=target_lang,
            model_family=self._model_family,
        )

        payload: dict = {
            "messages": [
                {"role": "user", "content": user_content},
            ],
            "max_tokens": self._max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        if self._model_family == "gemma4":
            # Gemma 4's default chat template enables chain-of-thought "reasoning"
            # mode, which puts the answer in `reasoning_content` and leaves
            # `content` empty until the model finishes thinking. For translation
            # we want a direct answer — disable thinking via the kwarg llama.cpp
            # forwards into the chat template.
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        body = json.dumps(payload).encode("utf-8")
        url = f"{self._server_url}/v1/chat/completions"
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:  # nosec B310
                result = json.loads(resp.read())
        except urllib.error.URLError as exc:
            logger.error("llama-server request failed: %s", exc)
            return TranslationResult(text=f"(llama-server error: {exc})", latency_ms=0.0)

        latency_ms = (time.perf_counter() - t0) * 1000

        # Extract translation from response
        choices = result.get("choices", [])
        if not choices:
            return TranslationResult(text="(no response from llama-server)", latency_ms=latency_ms)

        raw_text = choices[0].get("message", {}).get("content", "").strip()
        clean = clean_translation(raw_text, model_family=self._model_family)

        # Calculate tokens per second
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        gen_tps = completion_tokens / (latency_ms / 1000) if latency_ms > 0 and completion_tokens else 0.0

        return TranslationResult(
            text=clean,
            latency_ms=latency_ms,
            tokens_per_second=gen_tps,
        )

    def translate_streaming(
        self,
        text: str,
        *,
        source_lang: str = "en",
        target_lang: str = "es",
        token_callback=None,
    ) -> TranslationResult:
        """Streaming translation via llama-server SSE.

        Pushes accumulated translation text to ``token_callback`` as the server
        emits chunks. Falls back to non-streaming ``translate()`` and a single
        terminal callback if SSE parsing fails.

        Signature mirrors ``CUDAGemmaStreamingEngine.translate_streaming`` so
        ``dry_run_ab.translate_cuda_gemma_streaming`` can drive either engine.
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded -- call load() first")

        user_content = llamacpp_user_content(
            text,
            source_lang=source_lang,
            target_lang=target_lang,
            model_family=self._model_family,
        )

        payload: dict = {
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": self._max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
        if self._model_family == "gemma4":
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        body = json.dumps(payload).encode("utf-8")
        url = f"{self._server_url}/v1/chat/completions"
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )

        t0 = time.perf_counter()
        accumulated = ""
        completion_tokens = 0
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:  # nosec B310
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    piece = delta.get("content", "")
                    if not piece:
                        continue
                    accumulated += piece
                    completion_tokens += 1  # approximate — server tokens != client tokens
                    if token_callback is not None:
                        token_callback(accumulated, completion_tokens)
        except urllib.error.URLError as exc:
            logger.error("llama-server stream failed: %s — falling back to non-streaming", exc)
            return self.translate(text, source_lang=source_lang, target_lang=target_lang)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        clean = clean_translation(accumulated.strip(), model_family=self._model_family)

        gen_tps = completion_tokens / (latency_ms / 1000) if latency_ms > 0 and completion_tokens else 0.0
        return TranslationResult(text=clean, latency_ms=latency_ms, tokens_per_second=gen_tps)

    def unload(self) -> None:
        """Mark as unloaded (server keeps running independently)."""
        self._loaded = False
        logger.info("LlamaCppEngine disconnected from %s", self._server_url)

    @property
    def model_id(self) -> str:
        return self._model_id_str or "llama.cpp"

    @property
    def backend(self) -> str:
        return "llamacpp"
