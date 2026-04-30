#!/usr/bin/env python3
"""
build_preference_triples.py — Generate translation candidates and rank them with
CometKiwi-XL to produce CPO/DPO-style (chosen, rejected) preference triples.

Two subcommands. They are split so the heavy CometKiwi load doesn't have to
re-run if you only need to regenerate candidates (or vice-versa). The `score`
subcommand MUST run from /home/wbell/comet_env/ (transformers 4.57.6) — the
unsloth_env's transformers 5.5 breaks the comet 2.2.7 tokenizer API.

Subcommand `generate`:
  - Reads source JSONL with an `en` field
  - For each source, queries llama-server `--candidates` times at temperature
    `--temperature` (sampling diversity)
  - Always sends `chat_template_kwargs: {"enable_thinking": false}` to mirror
    production behavior
  - Output JSONL row: {"en", "candidates": [{"text", "tokens", "latency_ms"}, ...]}

Subcommand `score`:
  - Reads candidates JSONL from the previous step
  - Loads CometKiwi-XL (default Unbabel/wmt23-cometkiwi-da-xl), scores every
    (en, candidate.text) pair (reference-free QE)
  - For each source: pick the highest-scored candidate as `chosen`, the lowest
    as `rejected`. Drop the triple if (chosen_score - rejected_score) < margin
    — low-margin triples produce noisy gradients in CPO.
  - Output JSONL row: {"prompt": <user-content>, "chosen": <text>, "rejected":
    <text>, "chosen_score", "rejected_score", "margin"}

The output schema matches what TRL's CPOTrainer expects (`prompt`, `chosen`,
`rejected`).

Usage:
    # Step 1 — generate candidates (any env, just HTTP)
    python tools/build_preference_triples.py generate \\
        --sources bible_data/preference_pool/sources_2k.jsonl \\
        --server-url http://localhost:8092 \\
        --candidates 4 --temperature 0.7 \\
        --output preference/v1.1_candidates.jsonl

    # Step 2 — score with CometKiwi-XL (must run in comet_env)
    HF_HOME=/mnt/d/Data/stt-data/cache /home/wbell/comet_env/bin/python \\
        tools/build_preference_triples.py score \\
        --candidates preference/v1.1_candidates.jsonl \\
        --margin 0.05 \\
        --output preference/v1.1_triples.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_preference_triples")

# Mirror of engines/llamacpp_engine.py — keep in sync if production prompt changes.
LANG_NAMES = {"en": "English", "es": "Spanish"}


def build_user_prompt(text: str, source_lang: str = "en", target_lang: str = "es") -> str:
    src = LANG_NAMES.get(source_lang, source_lang)
    tgt = LANG_NAMES.get(target_lang, target_lang)
    return f"Translate the following {src} text to {tgt}. Output only the translation, nothing else.\n\n{text}"


# ── generate subcommand ─────────────────────────────────────────────────────


def call_llama_server(
    server_url: str,
    user_content: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
) -> tuple[str, int, float]:
    """Send one chat completion request, return (text, completion_tokens, latency_ms)."""
    payload = {
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # nosec B310
        result = json.loads(resp.read())
    latency_ms = (time.perf_counter() - t0) * 1000
    choice = result["choices"][0]
    text = choice["message"].get("content", "") or ""
    completion_tokens = result.get("usage", {}).get("completion_tokens", -1)
    return text.strip(), completion_tokens, latency_ms


def cmd_generate(args: argparse.Namespace) -> int:
    log.info("loading sources from %s", args.sources)
    sources: list[dict] = []
    with args.sources.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "en" not in row:
                log.warning("skipping row without en field: %s", row)
                continue
            sources.append(row)
    log.info("loaded %d sources", len(sources))
    if args.limit:
        sources = sources[: args.limit]
        log.info("limited to first %d sources", len(sources))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_failed = 0
    t0 = time.perf_counter()
    with args.output.open("w", encoding="utf-8") as out_f:
        for src in sources:
            user_content = build_user_prompt(src["en"])
            candidates: list[dict] = []
            for _ in range(args.candidates):
                try:
                    text, ntok, lat = call_llama_server(
                        args.server_url, user_content, args.temperature, args.max_tokens, args.timeout_s
                    )
                except (urllib.error.URLError, json.JSONDecodeError, KeyError) as exc:
                    n_failed += 1
                    log.warning("request failed for src %d: %s", n_done, exc)
                    continue
                candidates.append({"text": text, "tokens": ntok, "latency_ms": round(lat, 1)})
            if len(candidates) < 2:
                log.warning("fewer than 2 successful candidates for src %d, skipping", n_done)
                n_failed += 1
                continue
            row = {"en": src["en"], "candidates": candidates}
            # Preserve any extra source fields (es ref, verse_id, _provenance) for diagnostics
            for k, v in src.items():
                if k not in row:
                    row[k] = v
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_done += 1
            if n_done % 50 == 0:
                elapsed = time.perf_counter() - t0
                rate = n_done / max(elapsed, 1e-9)
                eta_s = (len(sources) - n_done) / max(rate, 1e-9)
                log.info(
                    "  progress %d/%d (%.1f src/s, ETA %.0fs, %d failed)",
                    n_done,
                    len(sources),
                    rate,
                    eta_s,
                    n_failed,
                )

    elapsed = time.perf_counter() - t0
    log.info(
        "wrote %d candidate-rows -> %s (wall %.1fs, %d failed)",
        n_done,
        args.output,
        elapsed,
        n_failed,
    )
    return 0


# ── score subcommand ────────────────────────────────────────────────────────


def cmd_score(args: argparse.Namespace) -> int:
    """Score every candidate via CometKiwi-XL, build (chosen, rejected) triples."""
    log.info("loading candidates from %s", args.candidates)
    rows: list[dict] = []
    with args.candidates.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    log.info("loaded %d rows with candidates", len(rows))

    # Flatten into (src, mt) samples for batch scoring
    samples: list[dict] = []
    for r in rows:
        for cand in r["candidates"]:
            samples.append({"src": r["en"], "mt": cand["text"]})
    log.info("scoring %d total candidates", len(samples))

    from comet import download_model, load_from_checkpoint

    log.info("downloading / loading %s", args.model)
    t0 = time.perf_counter()
    ckpt = download_model(args.model)
    model = load_from_checkpoint(ckpt)
    log.info("model loaded in %.1fs", time.perf_counter() - t0)

    log.info("scoring (batch_size=%d)", args.batch_size)
    t0 = time.perf_counter()
    pred = model.predict(samples=samples, batch_size=args.batch_size, accelerator="auto", progress_bar=True)
    log.info("scored %d in %.1fs", len(samples), time.perf_counter() - t0)
    flat_scores = [float(s) for s in pred.scores]

    # Build (chosen, rejected) triples
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_kept = 0
    n_dropped_low_margin = 0
    n_dropped_identical = 0
    margins: list[float] = []
    chosen_scores: list[float] = []
    rejected_scores: list[float] = []

    flat_idx = 0
    with args.output.open("w", encoding="utf-8") as out_f:
        for r in rows:
            n = len(r["candidates"])
            cand_scores = flat_scores[flat_idx : flat_idx + n]
            flat_idx += n
            # Sort: best first
            ranked = sorted(zip(r["candidates"], cand_scores), key=lambda cs: -cs[1])
            chosen_cand, chosen_score = ranked[0]
            rejected_cand, rejected_score = ranked[-1]
            margin = chosen_score - rejected_score
            if chosen_cand["text"].strip() == rejected_cand["text"].strip():
                n_dropped_identical += 1
                continue
            if margin < args.margin:
                n_dropped_low_margin += 1
                continue
            triple = {
                "prompt": build_user_prompt(r["en"]),
                "chosen": chosen_cand["text"],
                "rejected": rejected_cand["text"],
                "chosen_score": round(chosen_score, 4),
                "rejected_score": round(rejected_score, 4),
                "margin": round(margin, 4),
                "_en": r["en"],
            }
            out_f.write(json.dumps(triple, ensure_ascii=False) + "\n")
            n_kept += 1
            margins.append(margin)
            chosen_scores.append(chosen_score)
            rejected_scores.append(rejected_score)

    log.info(
        "wrote %d triples -> %s (dropped %d low-margin, %d identical-text)",
        n_kept,
        args.output,
        n_dropped_low_margin,
        n_dropped_identical,
    )
    if margins:
        log.info(
            "margin distribution: median=%.3f p25=%.3f p75=%.3f max=%.3f mean=%.3f",
            statistics.median(margins),
            statistics.quantiles(margins, n=4)[0] if len(margins) >= 4 else min(margins),
            statistics.quantiles(margins, n=4)[2] if len(margins) >= 4 else max(margins),
            max(margins),
            statistics.mean(margins),
        )
        log.info(
            "chosen score: mean=%.3f, rejected score: mean=%.3f",
            statistics.mean(chosen_scores),
            statistics.mean(rejected_scores),
        )
    return 0


# ── main ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate N translation candidates per source via llama-server")
    g.add_argument("--sources", type=Path, required=True, help="JSONL of {en, ...} source rows")
    g.add_argument("--server-url", default="http://localhost:8092", help="llama-server base URL")
    g.add_argument("--candidates", type=int, default=4, help="number of samples per source")
    g.add_argument("--temperature", type=float, default=0.7, help="sampling temperature")
    g.add_argument("--max-tokens", type=int, default=200)
    g.add_argument("--timeout-s", type=float, default=60.0)
    g.add_argument("--limit", type=int, default=0, help="cap source rows (0 = no cap)")
    g.add_argument("--output", type=Path, required=True)
    g.set_defaults(func=cmd_generate)

    s = sub.add_parser("score", help="Score candidates with CometKiwi-XL, build (chosen, rejected) triples")
    s.add_argument("--candidates", type=Path, required=True, help="output JSONL from `generate`")
    s.add_argument("--model", default="Unbabel/wmt23-cometkiwi-da-xl")
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--margin", type=float, default=0.05, help="drop triples where (chosen-rejected) < margin")
    s.add_argument("--output", type=Path, required=True)
    s.set_defaults(func=cmd_score)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
