#!/usr/bin/env python3
"""
build_v1_corpus.py — Assemble the Phase C v1 training corpus for Gemma 4 SFT.

Mixes four sources per the plan in docs/gemma4_tuning/phase_c_domain_sft.md:

  ~30% Modern-register Bible verses (from verse_pairs_train_v2.jsonl, post-holdout)
  ~30% Sermon parallel pairs (CometKiwi-XL filtered, threshold >= 0.85)
  ~30% Glossary-tagged pairs (a subset of bible+sermon run through
       glossary_annotate.py at rate=1.0 to force tags where applicable)
  ~10% OPUS-100 general-domain replay (anti-forgetting anchor)

The sermon pool is the natural bottleneck (3,587 pairs at threshold 0.85). The
script computes total corpus size as 3.33x the sermon count by default
(yielding sermon at 30%), or honors --total-target to scale up/down.

Output: a single shuffled JSONL with rows
    {"en": <text>, "es": <text>, "_provenance": {"source": ..., ...}}
ready for `training/train_gemma4.py --train-data ...`.

Usage:
    python tools/build_v1_corpus.py
    python tools/build_v1_corpus.py --total-target 25000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_v1_corpus")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("skipping malformed JSON in %s: %s", path, exc)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sample_bible(
    train_path: Path,
    n_target: int,
    rng: random.Random,
) -> list[dict]:
    """Sample modern-register bible verses from verse_pairs_train_v2.jsonl.

    Bias toward modern English (BBE/WEB) and away from ASV/KJV/YLT (which sound
    archaic in 2026 ears). Both Spanish sides (RVR1909 + Platense) are kept.
    """
    log.info("loading bible corpus from %s", train_path)
    all_pairs = load_jsonl(train_path)
    log.info("  %d total bible pairs", len(all_pairs))

    # Modern-EN preferred; archaic-EN allowed but sampled less.
    MODERN_EN = {"bbe", "web"}
    modern = [p for p in all_pairs if p.get("en_source") in MODERN_EN]
    archaic = [p for p in all_pairs if p.get("en_source") not in MODERN_EN]
    log.info("  modern-EN (BBE/WEB): %d, archaic-EN: %d", len(modern), len(archaic))

    # Allocate 70% from modern, 30% from archaic for diversity.
    n_modern = min(int(n_target * 0.70), len(modern))
    n_archaic = min(n_target - n_modern, len(archaic))
    rng.shuffle(modern)
    rng.shuffle(archaic)
    chosen = modern[:n_modern] + archaic[:n_archaic]
    rng.shuffle(chosen)
    log.info("  sampled bible: %d modern + %d archaic = %d total", n_modern, n_archaic, len(chosen))
    return chosen


def to_training_row(pair: dict, source_label: str, label_suffix: str = "") -> dict:
    return {
        "en": pair["en"],
        "es": pair["es"],
        "_provenance": {
            "source": f"{source_label}{label_suffix}",
            "verse_id": pair.get("verse_id"),
            "en_source": pair.get("en_source"),
            "es_source": pair.get("es_source"),
            "_kiwi_score": pair.get("_kiwi_score"),
            "_glossary_tags": pair.get("_glossary_tags"),
        },
    }


def sample_opus(n_target: int, rng: random.Random) -> list[dict]:
    """Reservoir-sample N pairs from OPUS-100 EN-ES streaming."""
    log.info("loading OPUS-100 EN-ES (streaming, %d-pair reservoir)...", n_target)
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("datasets not installed; skipping OPUS replay")
        return []
    try:
        ds = load_dataset("Helsinki-NLP/opus-100", "en-es", split="train", streaming=True)
    except Exception as exc:
        log.warning("OPUS-100 streaming failed: %s; skipping", exc)
        return []

    reservoir: list[dict] = []
    for i, ex in enumerate(ds):
        pair = {"en": ex["translation"]["en"], "es": ex["translation"]["es"]}
        if len(reservoir) < n_target:
            reservoir.append(pair)
        else:
            j = rng.randint(0, i)
            if j < n_target:
                reservoir[j] = pair
        if i >= n_target * 10:
            break
    log.info("  loaded %d OPUS pairs (sampled from %d streamed)", len(reservoir), i + 1)
    return reservoir


def annotate_with_glossary(
    pairs: list[dict],
    glossary_path: Path,
    rng: random.Random,
    n_target: int,
) -> list[dict]:
    """Run glossary_annotate.annotate_pair on a subset to force <g> tags.

    Picks the first `n_target` pairs from `pairs` (which should be pre-shuffled),
    runs them through annotation at rate=1.0, and returns only those where at
    least one tag was added (others are dropped — they wouldn't change after
    tagging). The caller can mix the result back into the corpus.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
    from glossary_annotate import annotate_pair, load_glossary

    glossary = load_glossary(glossary_path)
    sorted_terms = sorted(glossary.keys(), key=lambda t: -len(t))
    log.info("annotating up to %d pairs with %d glossary terms (rate=1.0)", n_target, len(glossary))

    # Walk through `pairs` in order; keep tagged ones until we hit n_target.
    tagged: list[dict] = []
    examined = 0
    for p in pairs:
        examined += 1
        annotated_en, n_tags = annotate_pair(p["en"], p["es"], glossary, sorted_terms)
        if n_tags > 0:
            new_pair = dict(p)
            new_pair["en"] = annotated_en
            new_pair["_glossary_tags"] = n_tags
            tagged.append(new_pair)
            if len(tagged) >= n_target:
                break
    log.info("  found %d taggable pairs out of %d examined", len(tagged), examined)
    return tagged


def build(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)

    # Validate ratios sum to 1.0 (within float epsilon).
    ratio_sum = args.ratio_bible + args.ratio_sermon + args.ratio_glossary + args.ratio_opus
    if abs(ratio_sum - 1.0) > 1e-6:
        raise SystemExit(f"ratios must sum to 1.0, got {ratio_sum:.6f}")

    # Load sermon pool first (it's the bottleneck).
    log.info("loading sermon pool from %s", args.sermon_path)
    sermon = load_jsonl(args.sermon_path)
    log.info("  %d filtered sermon pairs available", len(sermon))

    # Decide target sizes. If --total-target is given, scale to it; else compute
    # corpus around the sermon pool at its requested ratio.
    if args.total_target > 0:
        total = args.total_target
    else:
        total = int(len(sermon) / max(args.ratio_sermon, 1e-6))
    n_sermon = min(int(total * args.ratio_sermon), len(sermon))
    n_bible = int(total * args.ratio_bible)
    n_glossary = int(total * args.ratio_glossary)
    n_opus = int(total * args.ratio_opus)
    log.info(
        "target corpus composition (~%d total, ratios bible=%.2f sermon=%.2f glossary=%.2f opus=%.2f): "
        "sermon=%d, bible=%d, glossary-tagged=%d, opus=%d",
        total,
        args.ratio_bible,
        args.ratio_sermon,
        args.ratio_glossary,
        args.ratio_opus,
        n_sermon,
        n_bible,
        n_glossary,
        n_opus,
    )

    # Sample components.
    rng.shuffle(sermon)
    sermon_chosen = sermon[:n_sermon]
    bible_chosen = sample_bible(args.train_path, n_bible, rng)

    # Glossary tags: source from sermon + bible mix to maximize term coverage.
    # If n_glossary is large relative to the chosen bible pool, expand by loading
    # additional random bible pairs (yield is ~12% of examined, so we want the
    # pool to be at least ~10x n_glossary).
    glossary_pool = sermon[n_sermon:] + bible_chosen.copy()
    desired_pool_size = max(n_glossary * 12, len(glossary_pool))
    if desired_pool_size > len(glossary_pool):
        extra_needed = desired_pool_size - len(glossary_pool)
        log.info(
            "expanding glossary pool by %d extra bible pairs (current pool %d, target ~12x n_glossary=%d)",
            extra_needed,
            len(glossary_pool),
            n_glossary,
        )
        # Re-load the bible corpus and pull a separate random subset (not overlapping bible_chosen).
        all_bible = load_jsonl(args.train_path)
        chosen_ens = {p["en"] for p in bible_chosen}
        extra_pool = [p for p in all_bible if p["en"] not in chosen_ens]
        rng.shuffle(extra_pool)
        glossary_pool.extend(extra_pool[:extra_needed])
    rng.shuffle(glossary_pool)
    glossary_tagged = annotate_with_glossary(glossary_pool, args.glossary_path, rng, n_glossary)

    opus_chosen = sample_opus(n_opus, rng)

    # Wrap with provenance.
    suf = args.label_suffix
    rows: list[dict] = []
    rows.extend(to_training_row(p, "sermon_kiwi85", suf) for p in sermon_chosen)
    rows.extend(to_training_row(p, "bible_v2", suf) for p in bible_chosen)
    rows.extend(to_training_row(p, "glossary_tagged", suf) for p in glossary_tagged)
    rows.extend(to_training_row(p, "opus100_replay", suf) for p in opus_chosen)

    rng.shuffle(rows)

    log.info("writing %d total rows -> %s", len(rows), args.output)
    write_jsonl(args.output, rows)

    # Provenance summary.
    counts = Counter(r["_provenance"]["source"] for r in rows)
    log.info("final composition: %s", dict(counts))
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest = {
        "output": str(args.output),
        "total_rows": len(rows),
        "composition": dict(counts),
        "seed": args.seed,
        "inputs": {
            "sermon_path": str(args.sermon_path),
            "train_path": str(args.train_path),
            "glossary_path": str(args.glossary_path),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log.info("wrote manifest -> %s", manifest_path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sermon-path", type=Path, default=Path("bible_data/synthetic/deepl_sermon_pairs_kiwi85.jsonl"))
    p.add_argument("--train-path", type=Path, default=Path("bible_data/aligned/verse_pairs_train_v2.jsonl"))
    p.add_argument("--glossary-path", type=Path, default=Path("bible_data/glossary/tier2_master.json"))
    p.add_argument("--output", type=Path, default=Path("bible_data/v1_corpus/mixed.jsonl"))
    p.add_argument(
        "--total-target",
        type=int,
        default=0,
        help="Target total row count. 0 = compute as sermon_pool / ratio_sermon (sermon at its ratio).",
    )
    p.add_argument("--ratio-bible", type=float, default=0.30, help="Bible verses share (default 0.30 = v1)")
    p.add_argument("--ratio-sermon", type=float, default=0.30, help="Sermon share (default 0.30 = v1)")
    p.add_argument("--ratio-glossary", type=float, default=0.30, help="Glossary-tagged share (default 0.30 = v1)")
    p.add_argument("--ratio-opus", type=float, default=0.10, help="OPUS-100 replay share (default 0.10 = v1)")
    p.add_argument(
        "--label-suffix",
        default="",
        help="Suffix appended to _provenance.source labels (e.g. '_v1.1') so iterations are distinguishable",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
