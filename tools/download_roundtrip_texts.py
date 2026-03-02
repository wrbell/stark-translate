#!/usr/bin/env python3
"""Download public-domain test documents for the roundtrip test.

Fetches from Project Gutenberg and saves cleaned excerpts to
tools/roundtrip_texts/. Run once before roundtrip_test.py.

Usage:
    python tools/download_roundtrip_texts.py
"""

import re
from pathlib import Path
from urllib.request import urlopen

TEXTS_DIR = Path(__file__).resolve().parent / "roundtrip_texts"


def fetch_gutenberg(url: str) -> str:
    """Fetch plain-text from Project Gutenberg, strip header/footer."""
    raw = urlopen(url).read().decode("utf-8", errors="replace")
    # Strip PG header (ends with "***") and footer (starts with "***")
    parts = raw.split("***")
    if len(parts) >= 3:
        body = "***".join(parts[2:-1])  # between second and last ***
    else:
        body = raw
    return body.strip()


def clean_declaration(text: str) -> str:
    """Extract the Declaration, remove the offensive grievance sentence."""
    # Find the start: "When in the Course..."
    start = text.find("When in the Course")
    if start < 0:
        start = 0
    excerpt = text[start:]

    # Remove the sentence-level grievance about indigenous peoples.
    # The PG text is continuous prose, so we use regex on the sentence
    # that starts with "He has excited domestic insurrections" and ends
    # at the next grievance or paragraph marker.
    excerpt = re.sub(
        r"He has excited domestic insurrections[^.]*\.\s*",
        "",
        excerpt,
    )

    # Truncate at ~1300 words
    words = excerpt.split()
    if len(words) > 1300:
        result = " ".join(words[:1300])
    else:
        result = " ".join(words)
    return result.strip()


def clean_sermon(text: str) -> str:
    """Extract a ~1500-word excerpt from Edwards' sermon."""
    # Find a recognizable starting phrase
    markers = [
        "There is no want of power in God",
        "The God that holds you over the pit",
        "Their foot shall slide in due time",
    ]
    start = 0
    for marker in markers:
        idx = text.find(marker)
        if idx >= 0:
            start = idx
            break

    excerpt = text[start:]
    words = excerpt.split()
    if len(words) > 1500:
        # End at a sentence boundary near 1500 words
        truncated = " ".join(words[:1500])
        last_period = truncated.rfind(".")
        if last_period > len(truncated) // 2:
            truncated = truncated[: last_period + 1]
        excerpt = truncated

    return excerpt.strip()


def main() -> None:
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Declaration of Independence (PG #1)
    print("Downloading Declaration of Independence...")
    url1 = "https://www.gutenberg.org/cache/epub/1/pg1.txt"
    raw1 = fetch_gutenberg(url1)
    declaration = clean_declaration(raw1)
    out1 = TEXTS_DIR / "declaration.txt"
    out1.write_text(declaration, encoding="utf-8")
    print(f"  Saved: {out1} ({len(declaration.split())} words)")

    # 2. Sinners in the Hands of an Angry God (PG #34632)
    print("Downloading Edwards sermon...")
    url2 = "https://www.gutenberg.org/cache/epub/34632/pg34632.txt"
    raw2 = fetch_gutenberg(url2)
    sermon = clean_sermon(raw2)
    out2 = TEXTS_DIR / "sermon.txt"
    out2.write_text(sermon, encoding="utf-8")
    print(f"  Saved: {out2} ({len(sermon.split())} words)")

    print(f"\nDone. Text files ready in {TEXTS_DIR}/")
    print("Now run: python tools/roundtrip_test.py")


if __name__ == "__main__":
    main()
