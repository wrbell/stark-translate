"""Regression for the thread-local-stream failure: after load + warm_mlx_model on the
main thread, generation from pool workers must work (mlx 0.32.x / mlx-lm 0.31.3)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from engines.mlx_engine import materialize_mlx_model, warm_mlx_model
from engines.translation_prompts import build_chat_messages, chat_template_extra_kwargs

MODEL_ID = "mlx-community/gemma-4-e4b-it-OptiQ-4bit"


def test_generation_from_worker_threads_after_main_thread_warm_forward():
    from mlx_lm import load, stream_generate

    model, tok = load(MODEL_ID)
    materialize_mlx_model(model)
    warm_mlx_model(model, tok, model_family="gemma4", label="gpu-test")

    prompt = tok.apply_chat_template(
        build_chat_messages("The grace of God is sufficient.", model_family="gemma4"),
        add_generation_prompt=True,
        **chat_template_extra_kwargs(model_family="gemma4"),
    )

    def gen() -> str:
        out = ""
        for r in stream_generate(model, tok, prompt=prompt, max_tokens=32):
            out += r.text
        return out

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="gpu-test") as pool:
        outs = [f.result() for f in [pool.submit(gen), pool.submit(gen)]]
    assert all("gracia" in o.lower() for o in outs), outs
