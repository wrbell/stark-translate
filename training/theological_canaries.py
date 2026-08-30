"""Shared theological canary sentences for MT health checks and export smoke gates.

Used by tools/health_check.py, training/export_gguf.py sanity_test, and
training recipe scripts. Keep this as the single source of truth for the
canary set (core theological + hymn-domain spice).
"""

from __future__ import annotations

# EN→ES canaries covering critical Stark Road vocabulary.
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
    # Hymn-domain spice (remembrance / assembly register)
    {
        "en": "We gather before the mercy seat in prayer.",
        "expected_substrings": ["propiciatorio"],
    },
    {
        "en": "The table of the Lord is spread for His people.",
        "expected_substrings": ["mesa", "señor"],
    },
    {
        "en": "We are gathered to Thy Name on the first day of the week.",
        "expected_substrings": ["nombre"],
    },
    {
        "en": "A little while, and we shall see the Lord.",
        "expected_substrings": ["poco", "señor"],
    },
    {
        "en": "Nothing but the blood of Jesus can wash away my sin.",
        "expected_substrings": ["sangre", "jesús"],
    },
    {
        "en": "Christ is the Surety of a better covenant.",
        "expected_substrings": ["fiador", "pacto"],
    },
    {
        "en": "This do in remembrance of Me at His table.",
        "expected_substrings": ["memoria", "mesa"],
    },
    {
        "en": "At the mercy seat we remember His propitiation for our sins.",
        "expected_substrings": ["propiciatorio", "propiciación"],
    },
    {
        "en": "Yet there is room at the gospel feast for the weary soul.",
        "expected_substrings": ["lugar", "evangelio"],
    },
    {
        "en": "We come as pilgrims to Calvary and the cross.",
        "expected_substrings": ["peregrin", "calvario"],
    },
]


def canary_sentences(n: int | None = None) -> list[dict]:
    """Return the first *n* canaries (default: all)."""
    if n is None or n >= len(THEOLOGICAL_CANARIES):
        return list(THEOLOGICAL_CANARIES)
    return THEOLOGICAL_CANARIES[: max(1, n)]
