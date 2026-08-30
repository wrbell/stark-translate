"""Shared theological canary sentences for MT health checks and export smoke gates.

Used by tools/health_check.py, training/export_gguf.py sanity_test, and
training recipe scripts. Keep this as the single source of truth for the
8-sentence canary set (includes partimiento del pan).
"""

from __future__ import annotations

# Eight EN→ES canaries covering critical Stark Road vocabulary.
# expected_substrings are lowercased; matching is case-insensitive.
THEOLOGICAL_CANARIES: list[dict] = [
    {
        "en": "The atonement of Christ reconciles us to God.",
        "expected_substrings": ["expiación", "cristo", "dios"],
    },
    {
        "en": "James wrote about faith and works.",
        "expected_substrings": ["santiago", "fe", "obras"],
    },
    {
        "en": "The propitiation for our sins was the blood of Christ.",
        "expected_substrings": ["propiciación", "pecados", "sangre", "cristo"],
    },
    {
        "en": "The breaking of bread is a solemn remembrance.",
        "expected_substrings": ["partimiento", "pan"],
    },
    {
        "en": "Paul wrote to the Corinthians about the resurrection.",
        "expected_substrings": ["pablo", "corintios", "resurrección"],
    },
    {
        "en": "Justification is by faith alone, not by works of the law.",
        "expected_substrings": ["justificación", "fe"],
    },
    {
        "en": "The Holy Spirit convicts the world of sin and righteousness.",
        "expected_substrings": ["espíritu", "pecado"],
    },
    {
        "en": "Grace and mercy meet at the cross of our Lord Jesus Christ.",
        "expected_substrings": ["gracia", "misericordia", "cristo"],
    },
]


def canary_sentences(n: int | None = None) -> list[dict]:
    """Return the first *n* canaries (default: all eight)."""
    if n is None or n >= len(THEOLOGICAL_CANARIES):
        return list(THEOLOGICAL_CANARIES)
    return THEOLOGICAL_CANARIES[: max(1, n)]
