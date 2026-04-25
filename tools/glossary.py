#!/usr/bin/env python3
"""
glossary.py — Tiered Glossary Helper for Deepgram Keyterms + Training

Two tiers:
  Tier 1 (Boost): 50 terms for Deepgram Nova-3 `keyterm` parameter.
    Hard limit: 500 tokens / ~80-100 terms. We cap at 50 with <420 token budget.
    Focus: multi-word theological phrases and acoustically ambiguous terms.
  Tier 2 (Master): Up to 1000 terms for normalization + active learning.
    Full theological glossary including Bible books and proper names.

Usage:
    from tools.glossary import load_boost_keyterms, load_tier, validate_boost
    keyterms = load_boost_keyterms()  # list[str] ready for Deepgram
    master = load_tier("master")      # dict[str, str] EN→ES
"""

import json
import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# Default paths (relative to project root)
DEFAULT_BOOST_PATH = "bible_data/glossary/tier1_boost.json"
DEFAULT_MASTER_PATH = "bible_data/glossary/tier2_master.json"

# Deepgram Nova-3 limits
MAX_BOOST_TERMS = 60  # soft cap (hard API limit is ~80-100)
MAX_BOOST_TOKENS = 420  # safety margin below 500 hard limit (20% buffer)

# ---------------------------------------------------------------------------
# Bible book names (66) — excluded from Tier 1 (not acoustically ambiguous)
# ---------------------------------------------------------------------------
BIBLE_BOOKS = {
    "Genesis",
    "Exodus",
    "Leviticus",
    "Numbers",
    "Deuteronomy",
    "Joshua",
    "Judges",
    "Ruth",
    "1 Samuel",
    "2 Samuel",
    "1 Kings",
    "2 Kings",
    "1 Chronicles",
    "2 Chronicles",
    "Ezra",
    "Nehemiah",
    "Esther",
    "Job",
    "Psalms",
    "Proverbs",
    "Ecclesiastes",
    "Song of Solomon",
    "Isaiah",
    "Jeremiah",
    "Lamentations",
    "Ezekiel",
    "Daniel",
    "Hosea",
    "Joel",
    "Amos",
    "Obadiah",
    "Jonah",
    "Micah",
    "Nahum",
    "Habakkuk",
    "Zephaniah",
    "Haggai",
    "Zechariah",
    "Malachi",
    "Matthew",
    "Mark",
    "Luke",
    "John",
    "Acts",
    "Romans",
    "1 Corinthians",
    "2 Corinthians",
    "Galatians",
    "Ephesians",
    "Philippians",
    "Colossians",
    "1 Thessalonians",
    "2 Thessalonians",
    "1 Timothy",
    "2 Timothy",
    "Titus",
    "Philemon",
    "Hebrews",
    "James",
    "1 Peter",
    "2 Peter",
    "1 John",
    "2 John",
    "3 John",
    "Jude",
    "Revelation",
}

# Proper names — excluded from Tier 1 (Whisper handles names well)
PROPER_NAMES = {
    "James (apostle)",
    "James (epistle)",
    "Peter",
    "Paul",
    "Moses",
    "Abraham",
    "David",
    "Solomon",
    "Elijah",
    "Adam",
    "Eve",
    "Noah",
    "Isaac",
    "Jacob",
    "Joseph",
    "Aaron",
    "Samuel",
    "Elisha",
    "Gideon",
    "Samson",
    "Saul",
    "Timothy",
    "Stephen",
    "Barnabas",
    "Nicodemus",
    "Lazarus",
    "Mary",
    "Martha",
    "Pontius Pilate",
    "Herod",
    "Pharaoh",
    "Satan",
    "Savior",
    "Redeemer",
    "Almighty",
    "Antichrist",
}

# High-priority terms for Tier 1 — multi-word phrases and acoustically
# ambiguous theological terms that benefit most from Deepgram keyterm boosting
BOOST_PRIORITY = [
    # Multi-word phrases (highest keyterm value)
    "substitutionary atonement",
    "original sin",
    "born again",
    "new creation",
    "the Lamb of God",
    "the Word of God",
    "kingdom of God",
    "kingdom of heaven",
    "eternal life",
    "blood of Christ",
    "Lord's Supper",
    "breaking of bread",
    "Holy Spirit",
    "the Son of God",
    "Heavenly Father",
    "children of God",
    "body of Christ",
    "Word of God",
    "good news",
    "great commission",
    "the Lord's prayer",
    "beloved brethren",
    "brothers and sisters",
    "high priest",
    "day of the Lord",
    "second coming",
    "new heaven",
    "new earth",
    "throne of God",
    "lake of fire",
    "abomination of desolation",
    "laying on of hands",
    "let us pray",
    "the Word of the Lord",
    "praise the Lord",
    "in Jesus' name",
    "the cross",
    # Acoustically ambiguous single terms (easily misheard)
    "propitiation",
    "sanctification",
    "justification",
    "predestination",
    "imputation",
    "reconciliation",
    "intercession",
    "consecration",
    "doxology",
    "presbyter",
    "Pharisee",
    "Sadducee",
    "rapture",
    "tribulation",
    "millennium",
]


def count_tokens(terms: list[str]) -> int:
    """Approximate token count for Deepgram keyterm budget.

    Deepgram counts tokens as whitespace-separated words.
    """
    return sum(len(term.split()) for term in terms)


def validate_boost(terms: list[str], max_terms: int = MAX_BOOST_TERMS, max_tokens: int = MAX_BOOST_TOKENS) -> None:
    """Validate Tier 1 boost terms against Deepgram limits.

    Raises ValueError if terms exceed safe limits.
    """
    n_tokens = count_tokens(terms)
    if len(terms) > max_terms:
        raise ValueError(f"Tier 1 has {len(terms)} terms (max {max_terms}). Remove {len(terms) - max_terms} terms.")
    if n_tokens > max_tokens:
        raise ValueError(
            f"Tier 1 uses {n_tokens} tokens (max {max_tokens}). Remove multi-word terms to reduce token count."
        )


def load_tier(tier: Literal["boost", "master"], path: str | None = None) -> dict[str, str] | list[str]:
    """Load a glossary tier from disk.

    Args:
        tier: "boost" returns list[str] (EN terms only), "master" returns dict[str, str] (EN→ES)
        path: Override default path

    Returns:
        Boost: list of EN keyterm strings
        Master: dict of EN→ES term mappings
    """
    if tier == "boost":
        p = Path(path or DEFAULT_BOOST_PATH)
    elif tier == "master":
        p = Path(path or DEFAULT_MASTER_PATH)
    else:
        raise ValueError(f"Unknown tier: {tier!r}. Use 'boost' or 'master'.")

    if not p.exists():
        logger.warning(f"Tier {tier!r} file not found at {p}. Run: python training/build_glossary.py --build-tiers")
        return [] if tier == "boost" else {}

    data = json.loads(p.read_text())
    return data


def load_boost_keyterms(path: str | None = None) -> list[str]:
    """Convenience function: load and validate Tier 1 terms for Deepgram.

    Returns:
        list[str] ready to pass as Deepgram PrerecordedOptions.keyterm
    """
    terms = load_tier("boost", path)
    if terms:
        validate_boost(terms)
    return terms


def build_tiers(
    glossary: dict[str, str],
    boost_size: int = 50,
    master_size: int = 1000,
) -> dict[str, list[str] | dict[str, str]]:
    """Build Tier 1 (boost) and Tier 2 (master) from a full glossary.

    Tier 1 selection:
      1. Start with BOOST_PRIORITY terms (pre-curated)
      2. Fill remaining slots from theological terms (not books/names)
      3. Validate against Deepgram token budget

    Tier 2: All terms up to master_size.
    """
    # Build Tier 1 from priority list
    boost_terms = []
    for term in BOOST_PRIORITY:
        if term in glossary and len(boost_terms) < boost_size:
            boost_terms.append(term)

    # Fill remaining slots from non-book, non-name terms
    if len(boost_terms) < boost_size:
        for term in glossary:
            if term in BIBLE_BOOKS or term in PROPER_NAMES:
                continue
            if term not in boost_terms and len(boost_terms) < boost_size:
                boost_terms.append(term)

    # Validate
    validate_boost(boost_terms)

    # Build Tier 2 (full glossary)
    master = {}
    for en, es in list(glossary.items())[:master_size]:
        master[en] = es

    n_tokens = count_tokens(boost_terms)
    logger.info(f"Tier 1 (boost): {len(boost_terms)} terms, {n_tokens} tokens")
    logger.info(f"Tier 2 (master): {len(master)} terms")

    return {"boost": boost_terms, "master": master}


def save_tiers(
    tiers: dict,
    output_dir: str = "bible_data/glossary",
) -> None:
    """Save tier files to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    boost_path = out / "tier1_boost.json"
    boost_path.write_text(json.dumps(tiers["boost"], indent=2, ensure_ascii=False))

    master_path = out / "tier2_master.json"
    master_path.write_text(json.dumps(tiers["master"], indent=2, ensure_ascii=False))

    n_tokens = count_tokens(tiers["boost"])
    print(f"Saved tiers to {output_dir}/")
    print(f"  tier1_boost.json: {len(tiers['boost'])} terms, {n_tokens} tokens (limit: {MAX_BOOST_TOKENS})")
    print(f"  tier2_master.json: {len(tiers['master'])} EN→ES term pairs")
