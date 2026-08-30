"""Tests for training/prepare_hymn_corpus.py — hymn-domain PD EN↔ES slice."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HYMNS = REPO / "bible_data" / "hymns"
sys.path.insert(0, str(REPO / "training"))

from prepare_hymn_corpus import (
    INDEX_FIELDS,
    MIN_CHAR_LEN,
    PAIR_FIELDS,
    STANZA_FIELDS,
    _normalize_ws,
    build_parser,
    cmd_all,
    main,
)

INDEX_REQUIRED = set(INDEX_FIELDS)
STANZA_REQUIRED = set(STANZA_FIELDS)
PAIR_REQUIRED = set(PAIR_FIELDS)
CAND_REQUIRED = {
    "en",
    "es",
    "hymn_id",
    "stanza_id",
    "en_source",
    "es_source",
    "needs_synthetic",
    "license",
    "priority",
}
LEMMEL_ID = "pd-turn-your-eyes-upon-jesus"


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@pytest.fixture(scope="module")
def hymn_artifacts():
    """Ensure offline pipeline outputs exist (idempotent regenerate)."""
    parser = build_parser()
    args = parser.parse_args(["all", "--seed", "42", "--hymns-dir", str(HYMNS)])
    cmd_all(args)
    return {
        "index": _read_jsonl(HYMNS / "bhb_pd_index.jsonl"),
        "stanzas": _read_jsonl(HYMNS / "hymn_stanzas_en.jsonl"),
        "train": _read_jsonl(HYMNS / "hymn_pairs_pd.jsonl"),
        "holdout": _read_jsonl(HYMNS / "hymn_pairs_pd_holdout.jsonl"),
        "candidates": _read_jsonl(HYMNS / "hymn_candidates_synthetic.jsonl"),
        "rejected": _read_jsonl(HYMNS / "hymn_pairs_rejected.jsonl"),
    }


class TestSchema:
    def test_index_schema(self, hymn_artifacts):
        for row in hymn_artifacts["index"]:
            assert set(row.keys()) >= INDEX_REQUIRED
            assert row["pd_status"] in {"public_domain", "unknown", "excluded_modern"}

    def test_stanza_schema(self, hymn_artifacts):
        for row in hymn_artifacts["stanzas"]:
            assert set(row.keys()) >= STANZA_REQUIRED
            assert row["license"] == "public_domain"
            assert row["char_len"] >= MIN_CHAR_LEN

    def test_pair_schema(self, hymn_artifacts):
        for row in hymn_artifacts["train"] + hymn_artifacts["holdout"]:
            assert set(row.keys()) >= PAIR_REQUIRED
            assert row["en"].strip()
            assert row["es"].strip()
            assert row["license"] == "public_domain"
            assert row["alignment"] == "same_original"

    def test_candidate_schema(self, hymn_artifacts):
        for row in hymn_artifacts["candidates"]:
            assert set(row.keys()) >= CAND_REQUIRED
            assert row["es"] is None
            assert row["needs_synthetic"] is True


class TestCopyrightGuards:
    def test_lemmel_excluded_modern_no_lyrics(self, hymn_artifacts):
        lemmel = [r for r in hymn_artifacts["index"] if r["hymn_id"] == LEMMEL_ID]
        assert len(lemmel) == 1
        assert lemmel[0]["pd_status"] == "excluded_modern"
        assert lemmel[0]["author_death_year"] == 1961
        for collection in ("stanzas", "train", "holdout", "candidates"):
            assert not any(r["hymn_id"] == LEMMEL_ID for r in hymn_artifacts[collection])

    def test_no_excluded_modern_lyrics_anywhere(self, hymn_artifacts):
        excluded = {r["hymn_id"] for r in hymn_artifacts["index"] if r["pd_status"] == "excluded_modern"}
        for collection in ("stanzas", "train", "holdout", "candidates"):
            for row in hymn_artifacts[collection]:
                assert row["hymn_id"] not in excluded


class TestSplit:
    def test_holdout_disjoint(self, hymn_artifacts):
        train_ids = {r["stanza_id"] for r in hymn_artifacts["train"]}
        hold_ids = {r["stanza_id"] for r in hymn_artifacts["holdout"]}
        assert train_ids.isdisjoint(hold_ids)
        assert hold_ids

    def test_srgh_called_pd_pairs_in_holdout(self, hymn_artifacts):
        srgh_pd = {
            r["hymn_id"] for r in hymn_artifacts["index"] if r.get("srgh_called") and r["pd_status"] == "public_domain"
        }
        paired = {r["hymn_id"] for r in hymn_artifacts["train"] + hymn_artifacts["holdout"]}
        srgh_with_pairs = srgh_pd & paired
        hold_hymns = {r["hymn_id"] for r in hymn_artifacts["holdout"]}
        assert srgh_with_pairs <= hold_hymns


class TestGlossaryAllowlist:
    def test_allowlist_no_exact_key_duplicates(self):
        theo = json.loads((REPO / "bible_data" / "glossary" / "theological_glossary.json").read_text(encoding="utf-8"))
        allow = json.loads((HYMNS / "glossary_hymn_allowlist.json").read_text(encoding="utf-8"))
        theo_lower = {k.lower() for k in theo}
        for c in allow["candidates"]:
            assert c["en"].lower() not in theo_lower
        assert 25 <= len(allow["candidates"]) <= 40


class TestOfflinePipeline:
    def test_main_all_offline(self, tmp_path):
        # Copy seeds into temp dir and run all
        import shutil

        dest = tmp_path / "hymns"
        shutil.copytree(
            HYMNS,
            dest,
            ignore=shutil.ignore_patterns(
                "*.jsonl",
                "reports",
                "glossary_hymn_candidates.json",
                "hymn_pairs_deepl.jsonl",
            ),
        )
        # Ensure seed files present
        assert (dest / "seed_index.json").exists()
        assert (dest / "seed_stanzas_en.json").exists()
        assert (dest / "seed_es_pairs.json").exists()
        main(["all", "--seed", "42", "--hymns-dir", str(dest)])
        assert (dest / "hymn_pairs_pd.jsonl").exists()
        assert (dest / "hymn_pairs_pd_holdout.jsonl").exists()
        assert (dest / "reports" / "coverage.md").exists()

    def test_char_min_filter(self, tmp_path):
        import shutil

        from prepare_hymn_corpus import cmd_build_index, cmd_extract_stanzas

        dest = tmp_path / "hymns"
        shutil.copytree(
            HYMNS,
            dest,
            ignore=shutil.ignore_patterns("*.jsonl", "reports", "glossary_hymn_candidates.json"),
        )
        seed_stanzas = json.loads((dest / "seed_stanzas_en.json").read_text(encoding="utf-8"))
        seed_stanzas.append(
            {
                "hymn_id": seed_stanzas[0]["hymn_id"],
                "stanza_index": 99,
                "en": "Too short",
                "source_url": "memory_verified_pd",
                "license": "public_domain",
                "en_source": "pd",
            }
        )
        (dest / "seed_stanzas_en.json").write_text(json.dumps(seed_stanzas, ensure_ascii=False), encoding="utf-8")
        parser = build_parser()
        args = parser.parse_args(["extract-stanzas", "--hymns-dir", str(dest)])
        cmd_build_index(parser.parse_args(["build-index", "--hymns-dir", str(dest)]))
        cmd_extract_stanzas(args)
        stanzas = _read_jsonl(dest / "hymn_stanzas_en.jsonl")
        assert all(s["char_len"] >= MIN_CHAR_LEN for s in stanzas)
        assert not any(s["en"] == "Too short" for s in stanzas)


class TestProvenanceDocs:
    def test_provenance_ids_map_to_files(self):
        text = (REPO / "docs" / "data_provenance.md").read_text(encoding="utf-8")
        mapping = {
            "HYMN_PD": HYMNS / "hymn_pairs_pd.jsonl",
            "HYMN_HOLD": HYMNS / "hymn_pairs_pd_holdout.jsonl",
            "HYMN_CAND": HYMNS / "hymn_candidates_synthetic.jsonl",
            "G_HYMN": HYMNS / "glossary_hymn_allowlist.json",
        }
        for pid, path in mapping.items():
            assert pid in text
            assert path.exists(), path

    def test_normalize_ws(self):
        assert _normalize_ws("  a   b\n") == "a b"


class TestRejectedThematic:
    def test_thematic_only_not_in_train(self, hymn_artifacts):
        assert any(r.get("alignment") == "thematic_only" for r in hymn_artifacts["rejected"])
        assert all(r["alignment"] == "same_original" for r in hymn_artifacts["train"])


class TestBuildGlossaryHymnHook:
    def test_load_and_review_no_merge(self, capsys):
        from build_glossary import THEOLOGICAL_GLOSSARY, load_hymn_glossary_candidates, print_hymn_glossary_review

        cands = load_hymn_glossary_candidates(HYMNS / "glossary_hymn_candidates.json")
        assert cands
        before = len(THEOLOGICAL_GLOSSARY)
        print_hymn_glossary_review(cands)
        assert len(THEOLOGICAL_GLOSSARY) == before
        out = capsys.readouterr().out
        assert "mercy seat" in out.lower() or "Mercy seat" in out or "mercy seat" in out
