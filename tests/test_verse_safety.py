"""Safety regressions from retained sermon fragments, without model inference."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from features.bible_reference_bounds import CHAPTER_VERSE_COUNTS, SOURCE_SHA256
from features.extract_verses import BOOK_ALIASES, VerseExtractor, normalize_number


def refs(*texts):
    extractor = VerseExtractor()
    for text in texts:
        extractor.extract_from_text(text)
    return [row["reference"] for row in extractor.references]


@pytest.mark.parametrize(
    "spelling,number",
    [
        ("twenty three", "23"),
        ("twenty-three", "23"),
        ("fifty three", "53"),
        ("fifty-three", "53"),
        ("one hundred and nineteen", "119"),
        ("one hundred seventy-six", "176"),
    ],
)
def test_complete_compound_numbers(spelling, number):
    assert normalize_number(spelling) == number


def test_recorded_luke_compounds_do_not_backtrack_into_two_numbers():
    assert refs("Luke twenty three and verse uh thirty two.") == ["Luke 23:32"]
    assert refs("Luke twenty three.") == ["Luke 23"]
    assert refs("John three sixteen.") == ["John 3:16"]
    assert refs("Psalms chapter one hundred and nineteen verse one hundred seventy-six.") == ["Psalms 119:176"]


def test_real_fragment_chain_does_not_inherit_philemon_chapter():
    extractor = VerseExtractor()
    extractor.extract_from_text("Hebrews chapter seven tells us more clearly that he was holy, harmless, undefiled.")
    extractor.extract_from_text("And Paul wrote a letter back to Philemon, his master, and said, I'm sending him home.")
    assert extractor.current_book == "Philemon"
    assert extractor.current_chapter is None
    extractor.extract_from_text("Chapter fifty three and verse eleven.")
    assert extractor.current_chapter is None
    assert [r["reference"] for r in extractor.references] == ["Hebrews 7"]
    # The recording explicitly supplies Isaiah; no book is inferred from theology.
    extractor.extract_from_text("And Isaiah.")
    extractor.extract_from_text("When he's talking about the Savior and prophesying about him,")
    extractor.extract_from_text("Chapter fifty three and verse eleven says He made many to be accounted")
    assert [r["reference"] for r in extractor.references] == ["Hebrews 7", "Isaiah 53:11"]


def test_unknown_numbered_book_clears_context_without_guessing():
    assert refs("Isaiah 53:11.", "Peter chapter two tells us that", "Verse five.") == ["Isaiah 53:11"]


def test_hymn_context_cannot_become_bible_verse_or_range():
    assert refs(
        "Isaiah 53:11.",
        "Hymn eleven.",
        "Hymn eleven, behold the Lamb of God who bore a vile world sin.",
        "Verse five, Oh, 'twas because our sins on him by God were laid, he who himself had never sinned.",
        "For sinners sin was made. We'll sing verse one and five of Hymn eleven, and thank you again for coming tonight.",
    ) == ["Isaiah 53:11"]
    assert refs("Luke 23:32.", "We will sing verse one and five of Hymn eleven.") == ["Luke 23:32"]
    assert refs("Luke 23:32.", "Stanza two.", "Verse three.") == ["Luke 23:32"]
    assert refs("Hymn eleven.", "Read Romans 8:28.", "Verse 30.") == ["Romans 8:28", "Romans 8:30"]


def test_ambiguous_chunk_end_stays_unresolved():
    assert refs(
        "Luke 23:32.",
        "Uh one on his right and one on his left. And then we'll continue at verse uh through uh we'll continue to read at verse thirty",
        "uh thirty six.",
    ) == ["Luke 23:32"]
    assert refs("Luke 23:32.", "Verses one and five.") == ["Luke 23:32"]


@pytest.mark.parametrize(
    "invalid",
    [
        "Philemon 50:11",
        "Philemon chapter fifty three verse eleven",
        "Philemon 2",
        "John 0:0",
        "Romans 8:176",
        "Romans 8:10-5",
        "Philemon 1:25-26",
    ],
)
def test_invalid_bounds_never_emit_or_commit_chapter(invalid):
    extractor = VerseExtractor()
    extractor.extract_from_text("Hebrews 7:1.")
    extractor.extract_from_text(invalid + ".")
    extractor.extract_from_text("Verse five.")
    assert [r["reference"] for r in extractor.references] == ["Hebrews 7:1"]
    assert extractor.current_chapter is None


def test_context_follows_text_order_and_book_alias_boundaries():
    assert refs("Romans 8. Verse 28.") == ["Romans 8", "Romans 8:28"]
    assert refs("Luke 23:32. Verse 34. John 3:16.") == ["Luke 23:32", "Luke 23:34", "John 3:16"]
    assert refs("remark 2:3 and microjob 4:5.") == []


def test_bundled_metadata_is_complete_and_bound_to_existing_corpus():
    assert set(CHAPTER_VERSE_COUNTS) == set(BOOK_ALIASES)
    assert sum(map(len, CHAPTER_VERSE_COUNTS.values())) == 1189
    assert sum(sum(chapters) for chapters in CHAPTER_VERSE_COUNTS.values()) == 31102
    assert CHAPTER_VERSE_COUNTS["Philemon"] == (25,)
    assert SOURCE_SHA256 == "c208b439188880c442dd77bef936926487d98ef101e11d65759b283368077234"


def test_package_relative_import_and_legacy_cli_outside_checkout(tmp_path):
    package = tmp_path / "installed" / "features"
    package.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    for name in ["extract_verses.py", "bible_reference_bounds.py"]:
        shutil.copyfile(root / "features" / name, package / name)
    env = dict(os.environ, PYTHONPATH=str(package.parent))
    script = "from features.extract_verses import VerseExtractor; e=VerseExtractor(); e.extract_from_text('Philemon 50:11.'); assert not e.references"
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env, check=True, capture_output=True, timeout=10)
    subprocess.run(
        [sys.executable, str(package / "extract_verses.py"), "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=10,
    )


def test_incomplete_verse_and_range_are_not_published_as_complete():
    assert refs("Luke twenty three and verse") == []
    assert refs("Luke 23:32 through") == []
    assert refs("Luke 23:32-") == []


def test_operator_watcher_uses_real_extractor_and_keeps_source_unchanged(tmp_path):
    import csv

    from operator_app.features import VerseHighlightWatcher

    path = tmp_path / "metrics.csv"
    rows = [
        [30, "Much. So let's turn to uh Luke chapter twenty-three and we'll start at verse thirty-two."],
        [33, "Luke twenty three and verse uh thirty two."],
        [281, "Hebrews chapter seven tells us more clearly that he was holy, harmless, undefiled."],
        [385, "And Paul wrote a letter back to Philemon, his master, and said, I'm sending him home."],
        [525, "And Isaiah."],
        [526, "When he's talking about the Savior and prophesying about him,"],
        [527, "Chapter fifty three and verse eleven says He made many to be accounted"],
        [579, "Hymn eleven."],
        [582, "Verse five, Oh, 'twas because our sins on him by God were laid,"],
        [583, "We'll sing verse one and five of Hymn eleven."],
    ]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["chunk_id", "english"])
        writer.writerows(rows)
    original = path.read_bytes()
    watcher = VerseHighlightWatcher(path)
    highlights = watcher.force_scan()
    assert [item["reference"] for item in highlights] == ["Luke 23", "Luke 23:32", "Hebrews 7", "Isaiah 53:11"]
    assert watcher.force_scan() == highlights
    assert path.read_bytes() == original


def test_reference_followed_by_normal_prose_and_another_book_is_preserved():
    assert refs("John 3:16 and Romans 5:8.") == ["John 3:16", "Romans 5:8"]
    assert refs("John 3:16 to remember his love.") == ["John 3:16"]
    assert refs("John three, sixteen.") == ["John 3:16"]
    assert refs("Romans three and uh") == []
