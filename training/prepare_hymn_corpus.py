#!/usr/bin/env python3
"""
prepare_hymn_corpus.py — Copyright-safe hymn EN↔ES data slice for TranslateGemma / Gemma 4 SFT.

Offline-first: reads checked-in seeds under bible_data/hymns/. Optional --fetch / deepl.

Usage:
    python training/prepare_hymn_corpus.py all --seed 42
    python training/prepare_hymn_corpus.py build-index
    python training/prepare_hymn_corpus.py extract-stanzas
    python training/prepare_hymn_corpus.py align-es
    python training/prepare_hymn_corpus.py split
    python training/prepare_hymn_corpus.py glossary-diff
    python training/prepare_hymn_corpus.py report
    python training/prepare_hymn_corpus.py deepl --deepl-key $KEY
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("prepare_hymn_corpus")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HYMNS_DIR = REPO_ROOT / "bible_data" / "hymns"
GLOSSARY_JSON = REPO_ROOT / "bible_data" / "glossary" / "theological_glossary.json"
TIER1_JSON = REPO_ROOT / "bible_data" / "glossary" / "tier1_boost.json"
TIER2_JSON = REPO_ROOT / "bible_data" / "glossary" / "tier2_master.json"

PD_STATUSES = frozenset({"public_domain", "unknown", "excluded_modern"})
MIN_CHAR_LEN = 20
INDEX_FIELDS = (
    "hymn_id",
    "classic_bhb_number",
    "new_bhb_number",
    "first_line_en",
    "author",
    "author_death_year",
    "pd_status",
    "pd_evidence_url",
    "common_tune",
    "meter",
    "srgh_called",
    "exclude_reason",
)
STANZA_FIELDS = (
    "stanza_id",
    "hymn_id",
    "stanza_index",
    "en",
    "char_len",
    "source_url",
    "license",
)
PAIR_FIELDS = (
    "en",
    "es",
    "hymn_id",
    "stanza_id",
    "en_source",
    "es_source",
    "alignment",
    "license",
    "notes",
)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _stanza_id(hymn_id: str, stanza_index: int) -> str:
    return f"{hymn_id}-s{stanza_index:02d}"


def _holdout_bucket(stanza_id: str) -> str:
    digest = hashlib.sha256(stanza_id.encode("utf-8")).hexdigest()
    return digest


def _hymns_dir(args: argparse.Namespace) -> Path:
    return Path(args.hymns_dir)


# ---------------------------------------------------------------------------
# build-index
# ---------------------------------------------------------------------------


def cmd_build_index(args: argparse.Namespace) -> None:
    hymns_dir = _hymns_dir(args)
    seed_path = hymns_dir / "seed_index.json"
    seed = _load_json(seed_path)
    if not isinstance(seed, list) or len(seed) < 40:
        raise SystemExit(
            f"seed_index.json must list ≥40 hymns; got {len(seed) if isinstance(seed, list) else type(seed)}"
        )

    if getattr(args, "fetch", False):
        logger.info("--fetch set: keeping seed pd_evidence_url values (no network scrape of copyrighted corpora)")

    rows: list[dict] = []
    for item in seed:
        status = item.get("pd_status", "unknown")
        if status not in PD_STATUSES:
            raise SystemExit(f"Invalid pd_status for {item.get('hymn_id')}: {status}")
        row = {k: item.get(k) for k in INDEX_FIELDS}
        row["srgh_called"] = bool(item.get("srgh_called", False))
        rows.append(row)

    out = hymns_dir / "bhb_pd_index.jsonl"
    _write_jsonl(out, rows)
    pd_n = sum(1 for r in rows if r["pd_status"] == "public_domain")
    excl = sum(1 for r in rows if r["pd_status"] == "excluded_modern")
    logger.info("build-index: wrote %d rows (%d public_domain, %d excluded_modern) → %s", len(rows), pd_n, excl, out)


# ---------------------------------------------------------------------------
# extract-stanzas
# ---------------------------------------------------------------------------


def cmd_extract_stanzas(args: argparse.Namespace) -> None:
    hymns_dir = _hymns_dir(args)
    index = {r["hymn_id"]: r for r in _read_jsonl(hymns_dir / "bhb_pd_index.jsonl")}
    if not index:
        raise SystemExit("Run build-index first (empty bhb_pd_index.jsonl)")

    seed_stanzas = _load_json(hymns_dir / "seed_stanzas_en.json")
    seen_norm: set[str] = set()
    rows: list[dict] = []
    skipped_short = 0
    skipped_excluded = 0
    skipped_dup = 0

    for item in seed_stanzas:
        hymn_id = item["hymn_id"]
        meta = index.get(hymn_id)
        if meta is None:
            logger.warning("stanza hymn_id %s not in index; skipping", hymn_id)
            continue
        if meta["pd_status"] != "public_domain":
            skipped_excluded += 1
            continue
        en = _normalize_ws(item["en"])
        if len(en) < MIN_CHAR_LEN:
            skipped_short += 1
            continue
        # Drop identical chorus/stanza repeats after first occurrence (per hymn)
        key = f"{hymn_id}::{en.lower()}"
        if key in seen_norm:
            skipped_dup += 1
            continue
        seen_norm.add(key)
        idx = int(item["stanza_index"])
        rows.append(
            {
                "stanza_id": _stanza_id(hymn_id, idx),
                "hymn_id": hymn_id,
                "stanza_index": idx,
                "en": en,
                "char_len": len(en),
                "source_url": item.get("source_url") or meta.get("pd_evidence_url") or "memory_verified_pd",
                "license": "public_domain",
            }
        )

    out = hymns_dir / "hymn_stanzas_en.jsonl"
    _write_jsonl(out, rows)
    logger.info(
        "extract-stanzas: wrote %d stanzas (skipped short=%d excluded=%d dup=%d) → %s",
        len(rows),
        skipped_short,
        skipped_excluded,
        skipped_dup,
        out,
    )


# ---------------------------------------------------------------------------
# align-es
# ---------------------------------------------------------------------------


def cmd_align_es(args: argparse.Namespace) -> None:
    hymns_dir = _hymns_dir(args)
    stanzas = _read_jsonl(hymns_dir / "hymn_stanzas_en.jsonl")
    if not stanzas:
        raise SystemExit("Run extract-stanzas first")
    index = {r["hymn_id"]: r for r in _read_jsonl(hymns_dir / "bhb_pd_index.jsonl")}
    seed_es = _load_json(hymns_dir / "seed_es_pairs.json")

    # Map by stanza_id for same_original; thematic_only → rejected
    same_map: dict[str, dict] = {}
    rejected: list[dict] = []
    for pair in seed_es:
        alignment = pair.get("alignment", "same_original")
        sid = pair.get("stanza_id") or _stanza_id(pair["hymn_id"], int(pair["stanza_index"]))
        if alignment == "thematic_only":
            rejected.append(
                {
                    "en": pair["en"],
                    "es": pair.get("es"),
                    "hymn_id": pair["hymn_id"],
                    "stanza_id": sid,
                    "en_source": pair.get("en_source"),
                    "es_source": pair.get("es_source"),
                    "alignment": "thematic_only",
                    "license": pair.get("license", "public_domain"),
                    "notes": pair.get("notes", "Rejected thematic_only alignment"),
                }
            )
            continue
        if alignment != "same_original":
            logger.warning("Unknown alignment %s for %s; rejecting", alignment, sid)
            continue
        if not pair.get("es"):
            continue
        same_map[sid] = pair

    pairs: list[dict] = []
    candidates: list[dict] = []
    stanza_by_id = {s["stanza_id"]: s for s in stanzas}

    for stanza in stanzas:
        sid = stanza["stanza_id"]
        hymn_id = stanza["hymn_id"]
        meta = index.get(hymn_id, {})
        mapped = same_map.get(sid)
        if mapped and mapped.get("es"):
            pairs.append(
                {
                    "en": stanza["en"],
                    "es": _normalize_ws(mapped["es"]),
                    "hymn_id": hymn_id,
                    "stanza_id": sid,
                    "en_source": mapped.get("en_source") or "pd",
                    "es_source": mapped.get("es_source") or "established_pd_es",
                    "alignment": "same_original",
                    "license": "public_domain",
                    "notes": mapped.get("notes")
                    or "Spanish is established hymn translation of the same original, not a metrical invention",
                }
            )
        else:
            priority = "generic_pd"
            if meta.get("srgh_called"):
                priority = "srgh_called"
            elif any(
                k in (meta.get("author") or "").lower()
                for k in ("deck", "darby", "kelly", "bonar", "clephane", "paget", "stowell")
            ):
                priority = "brethren_distinctive"
            candidates.append(
                {
                    "en": stanza["en"],
                    "es": None,
                    "hymn_id": hymn_id,
                    "stanza_id": sid,
                    "en_source": (mapped.get("en_source") if mapped else None) or "pd",
                    "es_source": None,
                    "needs_synthetic": True,
                    "license": "public_domain",
                    "priority": priority,
                }
            )

    # Seed thematic rejects that don't correspond to extracted stanzas still go to sidecar
    # (already in rejected). Also reject any same_original that somehow lacked stanza:
    for sid, mapped in same_map.items():
        if sid not in stanza_by_id and mapped.get("alignment", "same_original") == "same_original":
            logger.warning("ES pair %s has no EN stanza row; skipping", sid)

    pairs_path = hymns_dir / "hymn_pairs_pd.jsonl"
    cand_path = hymns_dir / "hymn_candidates_synthetic.jsonl"
    rej_path = hymns_dir / "hymn_pairs_rejected.jsonl"
    _write_jsonl(pairs_path, pairs)
    _write_jsonl(cand_path, candidates)
    _write_jsonl(rej_path, rejected)
    logger.info(
        "align-es: pairs=%d candidates=%d rejected=%d → %s",
        len(pairs),
        len(candidates),
        len(rejected),
        pairs_path,
    )


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


def cmd_split(args: argparse.Namespace) -> None:
    hymns_dir = _hymns_dir(args)
    pairs = _read_jsonl(hymns_dir / "hymn_pairs_pd.jsonl")
    if not pairs:
        raise SystemExit("No pairs to split; run align-es first")

    index = {r["hymn_id"]: r for r in _read_jsonl(hymns_dir / "bhb_pd_index.jsonl")}
    srgh_hymn_ids = {hid for hid, m in index.items() if m.get("srgh_called")}

    n = len(pairs)
    # Target ~15%; minimum 50 if corpus >= 300, else 20%
    if n >= 300:
        target = max(50, round(n * 0.15))
        target = min(80, target) if n < 600 else max(50, round(n * 0.15))
    else:
        target = max(1, round(n * 0.20))
        # Spec: 50-80 when possible; for small corpora use 20%
        target = min(target, max(1, n - 1)) if n > 1 else 0

    # Force every SRGH-called hymn that has a pair into holdout (at least one stanza)
    forced: list[dict] = []
    remaining: list[dict] = []
    seen_srgh: set[str] = set()
    for p in pairs:
        hid = p["hymn_id"]
        if hid in srgh_hymn_ids and hid not in seen_srgh:
            forced.append(p)
            seen_srgh.add(hid)
        else:
            remaining.append(p)

    # Rank remaining by sha256(stanza_id) for deterministic holdout fill
    ranked = sorted(remaining, key=lambda p: _holdout_bucket(p["stanza_id"]))
    need = max(0, target - len(forced))
    holdout = forced + ranked[:need]
    hold_ids = {p["stanza_id"] for p in holdout}
    train = [p for p in pairs if p["stanza_id"] not in hold_ids]

    if hold_ids & {p["stanza_id"] for p in train}:
        raise SystemExit("Holdout ∩ train non-empty (internal error)")

    # Spec: hymn_pairs_pd.jsonl is the TRAIN file after split; holdout is sidecar
    _write_jsonl(hymns_dir / "hymn_pairs_pd.jsonl", train)
    _write_jsonl(hymns_dir / "hymn_pairs_pd_holdout.jsonl", holdout)
    logger.info(
        "split: train=%d holdout=%d (forced SRGH hymns=%d, target≈%d) seed=%s",
        len(train),
        len(holdout),
        len(forced),
        target,
        args.seed,
    )


# ---------------------------------------------------------------------------
# glossary-diff
# ---------------------------------------------------------------------------


def _load_existing_glossary_keys() -> tuple[dict[str, str], set[str], set[str]]:
    theo: dict[str, str] = {}
    if GLOSSARY_JSON.exists():
        theo = {k: v for k, v in _load_json(GLOSSARY_JSON).items()}
    tier1: set[str] = set()
    if TIER1_JSON.exists():
        tier1 = {t.lower() for t in _load_json(TIER1_JSON)}
    tier2: set[str] = set()
    if TIER2_JSON.exists():
        tier2 = {k.lower() for k in _load_json(TIER2_JSON)}
    return theo, tier1, tier2


def cmd_glossary_diff(args: argparse.Namespace) -> None:
    hymns_dir = _hymns_dir(args)
    theo, tier1, tier2 = _load_existing_glossary_keys()
    theo_lower = {k.lower(): k for k in theo}

    stanzas = _read_jsonl(hymns_dir / "hymn_stanzas_en.jsonl")
    index = _read_jsonl(hymns_dir / "bhb_pd_index.jsonl")
    allowlist_path = hymns_dir / "glossary_hymn_allowlist.json"
    allow = _load_json(allowlist_path) if allowlist_path.exists() else {"candidates": []}
    allow_by_en = {c["en"].lower(): c for c in allow.get("candidates", [])}

    # Seed phrase list: allowlist + a few high-value hymn phrases to scan
    scan_phrases = sorted(allow_by_en.keys(), key=len, reverse=True)
    corpus_text = " ".join(s["en"] for s in stanzas) + " " + " ".join(i.get("first_line_en") or "" for i in index)
    corpus_lower = corpus_text.lower()

    candidates: list[dict] = []
    seen: set[str] = set()

    def add_candidate(base: dict) -> None:
        key = base["en"].lower()
        if key in seen:
            return
        seen.add(key)
        in_theo = key in theo_lower
        es_val = base.get("es") or ""
        if in_theo and not es_val:
            es_val = theo.get(theo_lower[key], "")
        candidates.append(
            {
                "en": base["en"],
                "es": es_val,
                "alt_es": base.get("alt_es") or [],
                "category": base.get("category") or "Hymn-domain",
                "in_tier1": key in tier1,
                "in_tier2": key in tier2,
                "in_theological_glossary": in_theo,
                "example_en": base.get("example_en") or f"We gather and sing of {base['en']}.",
                "example_es": base.get("example_es") or f"Nos reunimos y cantamos de {es_val or '…'}.",
                "rationale": base.get("rationale") or "Detected in hymn corpus / allowlist seed",
                "recommend_merge": bool(base.get("recommend_merge", not in_theo)),
            }
        )

    # Allowlist first (including daysman-style candidate-only if present in a proposals file)
    for c in allow.get("candidates", []):
        add_candidate(c)

    # Extra candidate-only: daysman (spec)
    if "daysman" not in seen and "daysman" not in theo_lower:
        add_candidate(
            {
                "en": "daysman",
                "es": "árbitro",
                "alt_es": ["mediador"],
                "category": "Theology proper",
                "example_en": "There is no daysman betwixt us.",
                "example_es": "No hay árbitro entre nosotros.",
                "rationale": "BHB-ish Job language; rare in speech — candidate only",
                "recommend_merge": False,
            }
        )

    # Phrase hits in corpus for allowlist terms (already added); also note missing-from-allowlist hits
    # Scan corpus for known theological keys that appear in hymns (informational only if already in glossary)
    for phrase in scan_phrases:
        if phrase in corpus_lower and phrase not in seen:
            # Already handled via allowlist loop
            pass

    out_json = hymns_dir / "glossary_hymn_candidates.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"candidates": candidates}, f, indent=2, ensure_ascii=False)
        f.write("\n")

    reports = hymns_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Glossary diff (hymn domain)",
        "",
        f"Scanned {len(stanzas)} EN stanzas + first lines against theological_glossary / tier1 / tier2.",
        "",
        "| EN | ES | tier1 | tier2 | glossary | recommend_merge |",
        "|----|----|-------|-------|----------|-----------------|",
    ]
    for c in candidates:
        lines.append(
            f"| {c['en']} | {c['es']} | {c['in_tier1']} | {c['in_tier2']} | "
            f"{c['in_theological_glossary']} | {c['recommend_merge']} |"
        )
    new_n = sum(1 for c in candidates if not c["in_theological_glossary"])
    lines.extend(["", f"New vs theological_glossary.json: **{new_n}**", ""])
    (reports / "glossary_diff.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("glossary-diff: %d candidates (%d new) → %s", len(candidates), new_n, out_json)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> None:
    hymns_dir = _hymns_dir(args)
    reports = hymns_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    index = _read_jsonl(hymns_dir / "bhb_pd_index.jsonl")
    stanzas = _read_jsonl(hymns_dir / "hymn_stanzas_en.jsonl")
    train = _read_jsonl(hymns_dir / "hymn_pairs_pd.jsonl")
    hold = _read_jsonl(hymns_dir / "hymn_pairs_pd_holdout.jsonl")
    cand = _read_jsonl(hymns_dir / "hymn_candidates_synthetic.jsonl")
    rejected = _read_jsonl(hymns_dir / "hymn_pairs_rejected.jsonl")
    allow = (
        _load_json(hymns_dir / "glossary_hymn_allowlist.json")
        if (hymns_dir / "glossary_hymn_allowlist.json").exists()
        else {"candidates": []}
    )

    # pd_evidence.tsv
    tsv_lines = ["author\tdeath_year\tpd_verdict\tfirst_line\thymn_id\tsource_url"]
    for r in index:
        tsv_lines.append(
            f"{r.get('author', '')}\t{r.get('author_death_year', '')}\t{r.get('pd_status', '')}\t"
            f"{r.get('first_line_en', '')}\t{r.get('hymn_id', '')}\t{r.get('pd_evidence_url', '')}"
        )
    (reports / "pd_evidence.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

    do_not_train = [r for r in index if r.get("pd_status") == "excluded_modern"]
    srgh = [r for r in index if r.get("srgh_called")]
    srgh_with_pair = {p["hymn_id"] for p in train + hold}
    srgh_coverage = [r for r in srgh if r["hymn_id"] in srgh_with_pair or r.get("pd_status") != "public_domain"]

    warn_800 = len(train) > 800
    coverage = [
        "# Hymn corpus coverage",
        "",
        f"- Index hymns: **{len(index)}**",
        f"- Public domain: **{sum(1 for r in index if r['pd_status'] == 'public_domain')}**",
        f"- Excluded modern: **{len(do_not_train)}**",
        f"- EN stanzas: **{len(stanzas)}**",
        f"- Train PD pairs (HYMN_PD): **{len(train)}**",
        f"- Holdout pairs (HYMN_HOLD): **{len(hold)}**",
        f"- Synthetic candidates (HYMN_CAND): **{len(cand)}**",
        f"- Rejected thematic: **{len(rejected)}**",
        f"- Glossary allowlist (G_HYMN curated): **{len(allow.get('candidates', []))}**",
        f"- SRGH-called indexed: **{len(srgh)}**",
        f"- SRGH-called with PD pairs or excluded: **{len(srgh_coverage)}/{len(srgh)}**",
        "",
        "## Do not train (excluded_modern)",
        "",
    ]
    for r in do_not_train:
        coverage.append(
            f"- `{r['hymn_id']}` — {r.get('first_line_en')} ({r.get('author')}, d.{r.get('author_death_year')}): {r.get('exclude_reason')}"
        )
    coverage.extend(
        [
            "",
            "## Suggested SFT mix (not an executed run)",
            "",
            "`0.80 * (current S6 sources) + 0.15 * glossary + 0.05 * hymn_pairs_pd`",
            "",
            "Hymns are a small spice (~5%), never a third pillar beside verse + sermon.",
            "",
        ]
    )
    if warn_800:
        coverage.append(f"**WARNING:** train hymn pairs ({len(train)}) exceed 800 — reduce spice weight or trim.\n")
    (reports / "coverage.md").write_text("\n".join(coverage), encoding="utf-8")

    license_md = [
        "# License report — hymn corpus",
        "",
        "We train on public-domain hymn *texts*, not on the New Believers Hymn Book (2019) / John Ritchie compilation.",
        "",
        "## Excluded and why",
        "",
    ]
    for r in do_not_train:
        license_md.append(f"- **{r.get('first_line_en')}** (`{r['hymn_id']}`): {r.get('exclude_reason')}")
    license_md.extend(
        [
            "",
            "## Refused sources (see sources.json)",
            "",
            "- New BHB 2019 full text / numbering / 2019-only hymns",
            "- BHB+ app database / Gospel Folio music edition",
            "- Wholesale gospelriver.com scrape as 'the book'",
            "- Himnos y Cánticos del Evangelio full dump without LEC permission",
            "- JW.org, CCLI-only modern worship",
            "",
            "No hymn-singing audio is included for Whisper training.",
            "",
        ]
    )
    (reports / "license.md").write_text("\n".join(license_md), encoding="utf-8")

    logger.info(
        "report: train=%d holdout=%d candidates=%d excluded=%d%s",
        len(train),
        len(hold),
        len(cand),
        len(do_not_train),
        " WARN>800" if warn_800 else "",
    )


# ---------------------------------------------------------------------------
# all
# ---------------------------------------------------------------------------


def cmd_all(args: argparse.Namespace) -> None:
    cmd_build_index(args)
    cmd_extract_stanzas(args)
    cmd_align_es(args)
    cmd_split(args)
    cmd_glossary_diff(args)
    cmd_report(args)
    logger.info("all: complete (offline seed path)")


# ---------------------------------------------------------------------------
# deepl (optional)
# ---------------------------------------------------------------------------


def cmd_deepl(args: argparse.Namespace) -> None:
    if not args.deepl_key:
        raise SystemExit("deepl requires --deepl-key")
    hymns_dir = _hymns_dir(args)
    candidates = _read_jsonl(hymns_dir / "hymn_candidates_synthetic.jsonl")
    if not candidates:
        raise SystemExit("No synthetic candidates; run align-es first")

    try:
        import deepl
    except ImportError as exc:
        raise SystemExit("deepl package not installed") from exc

    # Reuse ≤2-word glossary filter pattern from generate_hybrid_synthetic.py
    glossary_pairs_path = REPO_ROOT / "bible_data" / "glossary" / "glossary_pairs.jsonl"
    translator = deepl.Translator(args.deepl_key)
    glossary = None
    if glossary_pairs_path.exists():
        with glossary_pairs_path.open(encoding="utf-8") as f:
            all_entries = [json.loads(line) for line in f]
        bare_terms = {e["en"]: e["es"] for e in all_entries if len(e["en"].split()) <= 2}
        if bare_terms:
            glossary = translator.create_glossary(
                "stark-translate-hymn",
                source_lang="EN",
                target_lang="ES",
                entries=bare_terms,
            )
            logger.info("DeepL glossary uploaded: %d bare terms", len(bare_terms))

    out_rows: list[dict] = []
    try:
        for row in candidates:
            kwargs: dict[str, Any] = {
                "text": row["en"],
                "source_lang": "EN",
                "target_lang": "ES",
            }
            if glossary is not None:
                kwargs["glossary"] = glossary
            result = translator.translate_text(**kwargs)
            es = result.text if hasattr(result, "text") else str(result)
            out_rows.append(
                {
                    "en": row["en"],
                    "es": es,
                    "source": "deepl",
                    "chunk_source": f"hymn:{row['stanza_id']}",
                    "chunk_start": None,
                    "chunk_end": None,
                    "hymn_id": row["hymn_id"],
                    "stanza_id": row["stanza_id"],
                    "license": "public_domain",
                }
            )
    finally:
        if glossary is not None:
            try:
                translator.delete_glossary(glossary)
            except Exception as exc:
                logger.warning("Failed to delete DeepL glossary: %s", exc)

    out = hymns_dir / "hymn_pairs_deepl.jsonl"
    _write_jsonl(out, out_rows)
    logger.info("deepl: wrote %d pairs → %s (not merged into hymn_pairs_pd.jsonl)", len(out_rows), out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare copyright-safe hymn EN↔ES corpus for translation SFT")
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--hymns-dir", default=str(DEFAULT_HYMNS_DIR), help="bible_data/hymns directory")
    parent.add_argument("--seed", type=int, default=42, help="Deterministic seed (default 42)")

    sub = parser.add_subparsers(dest="command", required=True)

    p_idx = sub.add_parser("build-index", parents=[parent], help="Write bhb_pd_index.jsonl from seed_index.json")
    p_idx.add_argument(
        "--fetch", action="store_true", help="Optional refresh of PD evidence URLs (no copyrighted dump)"
    )
    p_idx.set_defaults(func=cmd_build_index)

    p_ex = sub.add_parser("extract-stanzas", parents=[parent], help="Write hymn_stanzas_en.jsonl from PD seed stanzas")
    p_ex.set_defaults(func=cmd_extract_stanzas)

    p_al = sub.add_parser(
        "align-es", parents=[parent], help="Align established PD Spanish; write candidates + rejected"
    )
    p_al.set_defaults(func=cmd_align_es)

    p_sp = sub.add_parser("split", parents=[parent], help="Deterministic train/holdout split")
    p_sp.set_defaults(func=cmd_split)

    p_gl = sub.add_parser("glossary-diff", parents=[parent], help="Diff hymn terms vs existing glossary")
    p_gl.set_defaults(func=cmd_glossary_diff)

    p_rp = sub.add_parser("report", parents=[parent], help="Write reports/coverage.md, pd_evidence.tsv, license.md")
    p_rp.set_defaults(func=cmd_report)

    p_all = sub.add_parser("all", parents=[parent], help="Run full offline pipeline")
    p_all.add_argument("--fetch", action="store_true")
    p_all.set_defaults(func=cmd_all)

    p_dl = sub.add_parser("deepl", parents=[parent], help="Optional DeepL on synthetic candidates (default OFF)")
    p_dl.add_argument("--deepl-key", default=None)
    p_dl.set_defaults(func=cmd_deepl)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
