#!/usr/bin/env python3
"""
glossary_annotate.py — Inline glossary tagging for Gemma 4 SFT training data.

For each (en, es) pair where an EN term from tier2_master.json is present in the
source AND its ES translation is present in the target, wrap the EN span:

    "He spoke about the atonement."
    -> "He spoke about the <g>atonement||expiación</g>."

Per WMT 2025 terminology task, training-time inline annotation at ~30% mixture
rate beats both decode-time hard constraints (degrade fluency) and zero-tag
training (no glossary control). See docs/gemma4_tuning/phase_a_infrastructure.md.

Matching rules:
  - Case-insensitive find, original casing preserved in output.
  - Word boundaries enforced (regex \\b) — "sin" won't match inside "single".
  - Longer terms matched first ("substitutionary atonement" before "atonement").
  - Both EN and ES sides verified before tagging — if ES translation missing
    from target, skip (avoids tagging when the human translator chose a
    synonym).
  - Multi-match terms (e.g. "James" -> Jacobo OR Santiago): tag only if at
    least one acceptable ES form appears in the target.
  - At most one tag per term per sentence (avoids tag spam in repetitive text).

Usage:
    python training/glossary_annotate.py \\
        --input pairs.jsonl \\
        --output pairs_annotated.jsonl \\
        --glossary bible_data/glossary/tier2_master.json \\
        --rate 0.30

    python training/glossary_annotate.py --self-test
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections.abc import Iterable
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("glossary_annotate")


def load_glossary(path: Path) -> dict[str, list[str]]:
    """Load tier2_master.json. Return {en_lower: [es_form, ...]}.

    Supports two formats:
      {"atonement": "expiación"}                 (single ES form)
      {"James": ["Jacobo", "Santiago"]}          (multiple ES forms — disambiguated by context)
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for en, es in raw.items():
        forms = [es] if isinstance(es, str) else list(es)
        out[en.lower()] = forms
    return out


def annotate_pair(
    en: str,
    es: str,
    glossary: dict[str, list[str]],
    sorted_terms: list[str],
) -> tuple[str, int]:
    """Return (annotated_en, n_tags_added).

    Single-pass regex match across all terms (sorted longest-first via
    alternation order) — guarantees non-overlapping matches and avoids the
    nested-tag bug that arises from sequential per-term substitution.
    """
    if not sorted_terms:
        return en, 0
    es_lower = es.lower()
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(t) for t in sorted_terms) + r")\b",
        re.IGNORECASE,
    )

    out_parts: list[str] = []
    last = 0
    n_tags = 0
    used_terms: set[str] = set()
    for m in pattern.finditer(en):
        term_lower = m.group(0).lower()
        if term_lower in used_terms:
            continue
        es_forms = glossary[term_lower]
        es_form_present = next((f for f in es_forms if f.lower() in es_lower), None)
        if es_form_present is None:
            continue
        out_parts.append(en[last : m.start()])
        out_parts.append(f"<g>{m.group(0)}||{es_form_present}</g>")
        last = m.end()
        n_tags += 1
        used_terms.add(term_lower)
    out_parts.append(en[last:])
    return "".join(out_parts), n_tags


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("skipping malformed JSON at line %d: %s", i, exc)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def annotate_file(
    input_path: Path,
    output_path: Path,
    glossary_path: Path,
    rate: float,
    seed: int,
) -> dict:
    glossary = load_glossary(glossary_path)
    # Sort longest-first so multi-word terms win over their constituents.
    sorted_terms = sorted(glossary.keys(), key=lambda t: -len(t))
    rng = random.Random(seed)

    stats = {"total": 0, "tagged_pairs": 0, "tags_added": 0, "candidates_skipped_es_missing": 0}

    def process() -> Iterable[dict]:
        for row in iter_jsonl(input_path):
            stats["total"] += 1
            if "en" not in row or "es" not in row:
                yield row
                continue
            if rng.random() >= rate:
                yield row
                continue
            annotated, n_tags = annotate_pair(row["en"], row["es"], glossary, sorted_terms)
            if n_tags > 0:
                stats["tagged_pairs"] += 1
                stats["tags_added"] += n_tags
                new_row = dict(row)
                new_row["en"] = annotated
                new_row["_glossary_tags"] = n_tags
                yield new_row
            else:
                yield row

    written = write_jsonl(output_path, process())
    stats["written"] = written
    return stats


def self_test() -> int:
    """Hand-checked cases. Returns 0 on success, 1 on failure."""
    glossary: dict[str, list[str]] = {
        "atonement": ["expiación"],
        "substitutionary atonement": ["expiación sustitutoria"],
        "james": ["Jacobo", "Santiago"],
        "sin": ["pecado"],
        "grace": ["gracia"],
    }
    sorted_terms = sorted(glossary.keys(), key=lambda t: -len(t))

    cases = [
        # (en, es, expected_annotated_en, expected_n_tags)
        (
            "He spoke about the atonement.",
            "Habló sobre la expiación.",
            "He spoke about the <g>atonement||expiación</g>.",
            1,
        ),
        # Longer term wins over shorter.
        (
            "The substitutionary atonement is central.",
            "La expiación sustitutoria es central.",
            "The <g>substitutionary atonement||expiación sustitutoria</g> is central.",
            1,
        ),
        # James -> Santiago (epistle context).
        (
            "James wrote to the twelve tribes.",
            "Santiago escribió a las doce tribus.",
            "<g>James||Santiago</g> wrote to the twelve tribes.",
            1,
        ),
        # James -> Jacobo (apostle context).
        (
            "James and John, the sons of Zebedee.",
            "Jacobo y Juan, los hijos de Zebedeo.",
            "<g>James||Jacobo</g> and John, the sons of Zebedee.",
            1,
        ),
        # ES form missing -> no tag.
        (
            "He spoke about the atonement.",
            "Habló sobre el sacrificio.",
            "He spoke about the atonement.",
            0,
        ),
        # Word boundary: "sin" should NOT match "single" or "sincere".
        (
            "He was sincere and single-minded.",
            "Fue sincero y resuelto.",
            "He was sincere and single-minded.",
            0,
        ),
        # Word boundary: "sin" SHOULD match standalone.
        (
            "We have all sinned, but grace abounds.",
            "Todos hemos pecado, pero la gracia abunda.",
            # sin-stem doesn't match because "sinned" is not "sin"; grace does match.
            "We have all sinned, but <g>grace||gracia</g> abounds.",
            1,
        ),
        # Multiple terms in one sentence — both tagged.
        (
            "Grace and atonement go together.",
            "La gracia y la expiación van juntas.",
            "<g>Grace||gracia</g> and <g>atonement||expiación</g> go together.",
            2,
        ),
    ]

    failed = 0
    for i, (en, es, want_en, want_tags) in enumerate(cases, 1):
        got_en, got_tags = annotate_pair(en, es, glossary, sorted_terms)
        ok = got_en == want_en and got_tags == want_tags
        marker = "ok" if ok else "FAIL"
        print(f"[{marker}] case {i}: tags={got_tags} (want {want_tags})")
        if not ok:
            failed += 1
            print(f"       got:  {got_en!r}")
            print(f"       want: {want_en!r}")
    print(f"\n{len(cases) - failed}/{len(cases)} self-tests passed")
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, help="JSONL of {en, es, ...} pairs")
    p.add_argument("--output", type=Path, help="annotated JSONL output")
    p.add_argument(
        "--glossary",
        type=Path,
        default=Path("bible_data/glossary/tier2_master.json"),
        help="tier2_master.json path",
    )
    p.add_argument("--rate", type=float, default=0.30, help="fraction of pairs eligible for tagging (0.0-1.0)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")
    p.add_argument("--self-test", action="store_true", help="run hand-checked cases and exit")
    args = p.parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.input or not args.output:
        p.error("--input and --output are required (or use --self-test)")
    if not 0.0 <= args.rate <= 1.0:
        p.error("--rate must be in [0.0, 1.0]")

    stats = annotate_file(args.input, args.output, args.glossary, args.rate, args.seed)
    log.info(
        "annotated %d/%d pairs (%d tags total) -> %s",
        stats["tagged_pairs"],
        stats["total"],
        stats["tags_added"],
        args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
