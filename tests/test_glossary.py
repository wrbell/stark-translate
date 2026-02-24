"""Tests for build_glossary.py — theological glossary and training pairs."""

import json

from build_glossary import (
    THEOLOGICAL_GLOSSARY,
    augment_with_soft_constraints,
    create_glossary_training_pairs,
    export_glossary,
)


class TestTheologicalGlossary:
    def test_glossary_has_entries(self):
        assert len(THEOLOGICAL_GLOSSARY) > 200

    def test_key_theological_terms(self):
        assert THEOLOGICAL_GLOSSARY["atonement"] == "expiación"
        assert THEOLOGICAL_GLOSSARY["propitiation"] == "propiciación"
        assert THEOLOGICAL_GLOSSARY["covenant"] == "pacto"
        assert THEOLOGICAL_GLOSSARY["righteousness"] == "justicia"
        assert THEOLOGICAL_GLOSSARY["sanctification"] == "santificación"

    def test_bible_books_present(self):
        assert "Genesis" in THEOLOGICAL_GLOSSARY
        assert "Revelation" in THEOLOGICAL_GLOSSARY
        assert "Psalms" in THEOLOGICAL_GLOSSARY
        assert "Romans" in THEOLOGICAL_GLOSSARY
        assert THEOLOGICAL_GLOSSARY["Genesis"] == "Génesis"
        assert THEOLOGICAL_GLOSSARY["Revelation"] == "Apocalipsis"

    def test_proper_names_present(self):
        assert THEOLOGICAL_GLOSSARY["Moses"] == "Moisés"
        assert THEOLOGICAL_GLOSSARY["Paul"] == "Pablo"
        assert THEOLOGICAL_GLOSSARY["Peter"] == "Pedro"

    def test_james_disambiguation(self):
        assert THEOLOGICAL_GLOSSARY["James (apostle)"] == "Jacobo"
        assert THEOLOGICAL_GLOSSARY["James (epistle)"] == "Santiago"

    def test_no_duplicate_keys(self):
        """All keys should be unique (Python dicts enforce this,
        but F601 caught a duplicate 'intercession' that we fixed)."""
        keys = list(THEOLOGICAL_GLOSSARY.keys())
        assert len(keys) == len(set(keys))


class TestCreateGlossaryTrainingPairs:
    def test_returns_pairs(self):
        pairs = create_glossary_training_pairs()
        assert len(pairs) > 0

    def test_two_pairs_per_term(self):
        small_glossary = {"atonement": "expiación", "grace": "gracia"}
        pairs = create_glossary_training_pairs(small_glossary)
        assert len(pairs) == 4  # 2 terms * 2 pairs each

    def test_direct_pair_format(self):
        pairs = create_glossary_training_pairs({"faith": "fe"})
        assert pairs[0] == {"en": "faith", "es": "fe"}

    def test_sentence_pair_format(self):
        pairs = create_glossary_training_pairs({"faith": "fe"})
        assert "faith" in pairs[1]["en"]
        assert "fe" in pairs[1]["es"]
        assert "sermon" in pairs[1]["en"].lower()


class TestAugmentWithSoftConstraints:
    def test_augmentation(self, tmp_path):
        verse_file = tmp_path / "verses.jsonl"
        verse_file.write_text(
            json.dumps({"en": "By grace you have been saved through faith.", "es": "Por gracia sois salvos."})
            + "\n"
            + json.dumps({"en": "The sky is blue.", "es": "El cielo es azul."})
            + "\n"
        )

        glossary = {"grace": "gracia", "faith": "fe"}
        result = augment_with_soft_constraints(str(verse_file), glossary=glossary)

        assert len(result) == 2
        # First verse should have glossary constraint
        assert "[GLOSSARY:" in result[0]["en"]
        assert "grace=gracia" in result[0]["en"]
        assert "faith=fe" in result[0]["en"]
        # Second verse has no glossary terms
        assert "[GLOSSARY:" not in result[1]["en"]

    def test_augmentation_with_output(self, tmp_path):
        verse_file = tmp_path / "verses.jsonl"
        verse_file.write_text(json.dumps({"en": "God is love.", "es": "Dios es amor."}) + "\n")
        output_file = tmp_path / "augmented.jsonl"

        augment_with_soft_constraints(str(verse_file), output_path=str(output_file))

        assert output_file.exists()
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 1


class TestExportGlossary:
    def test_export_creates_files(self, tmp_path):
        export_glossary(output_dir=str(tmp_path))

        json_file = tmp_path / "theological_glossary.json"
        jsonl_file = tmp_path / "glossary_pairs.jsonl"
        assert json_file.exists()
        assert jsonl_file.exists()

        glossary = json.loads(json_file.read_text())
        assert len(glossary) == len(THEOLOGICAL_GLOSSARY)

    def test_export_returns_pairs_path(self, tmp_path):
        result = export_glossary(output_dir=str(tmp_path))
        assert result.endswith("glossary_pairs.jsonl")
