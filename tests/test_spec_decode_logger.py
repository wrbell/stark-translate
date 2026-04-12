"""Tests for speculative decoding rejection logger."""

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from engines.spec_decode_logger import SpecDecodeLogger


@pytest.fixture
def tmp_output_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def glossary_file(tmp_output_dir):
    path = os.path.join(tmp_output_dir, "tier1.json")
    terms = ["atonement", "covenant", "grace", "propitiation", "sanctification"]
    with open(path, "w") as f:
        json.dump(terms, f)
    return path


class TestSpecDecodeLogger:
    def test_log_rejection_creates_jsonl(self, tmp_output_dir, glossary_file):
        logger = SpecDecodeLogger(output_dir=tmp_output_dir, glossary_path=glossary_file)
        logger.log_rejection(
            source_text="The atonement of Christ",
            draft_tokens=[10, 20, 30, 40],
            accepted_tokens=[10, 20, 35, 45],
            domain="sermon",
        )
        # Find the output file
        files = [f for f in os.listdir(tmp_output_dir) if f.startswith("spec_decode_rejections_")]
        assert len(files) == 1
        with open(os.path.join(tmp_output_dir, files[0])) as f:
            record = json.loads(f.readline())
        assert record["source_text"] == "The atonement of Christ"
        assert record["acceptance_rate"] == 0.5  # 2/4 rejected
        assert record["rejection_count"] == 2
        assert record["domain"] == "sermon"
        assert "atonement" in record["theological_terms"]

    def test_acceptance_rate_all_accepted(self, tmp_output_dir):
        logger = SpecDecodeLogger(output_dir=tmp_output_dir, glossary_path="nonexistent.json")
        logger.log_rejection(
            source_text="test",
            draft_tokens=[1, 2, 3],
            accepted_tokens=[1, 2, 3],
        )
        files = [f for f in os.listdir(tmp_output_dir) if f.startswith("spec_decode_rejections_")]
        with open(os.path.join(tmp_output_dir, files[0])) as f:
            record = json.loads(f.readline())
        assert record["acceptance_rate"] == 1.0
        assert record["rejection_count"] == 0

    def test_acceptance_rate_all_rejected(self, tmp_output_dir):
        logger = SpecDecodeLogger(output_dir=tmp_output_dir, glossary_path="nonexistent.json")
        logger.log_rejection(
            source_text="test",
            draft_tokens=[1, 2, 3],
            accepted_tokens=[4, 5, 6],
        )
        files = [f for f in os.listdir(tmp_output_dir) if f.startswith("spec_decode_rejections_")]
        with open(os.path.join(tmp_output_dir, files[0])) as f:
            record = json.loads(f.readline())
        assert record["acceptance_rate"] == 0.0
        assert record["rejection_count"] == 3

    def test_length_mismatch_counts_as_rejection(self, tmp_output_dir):
        logger = SpecDecodeLogger(output_dir=tmp_output_dir, glossary_path="nonexistent.json")
        logger.log_rejection(
            source_text="test",
            draft_tokens=[1, 2, 3, 4, 5],
            accepted_tokens=[1, 2, 3],
        )
        files = [f for f in os.listdir(tmp_output_dir) if f.startswith("spec_decode_rejections_")]
        with open(os.path.join(tmp_output_dir, files[0])) as f:
            record = json.loads(f.readline())
        assert record["rejection_count"] == 2  # positions 3 and 4

    def test_tokenizer_decodes_rejected_words(self, tmp_output_dir):
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.side_effect = lambda ids: f"word_{ids[0]}"

        logger = SpecDecodeLogger(output_dir=tmp_output_dir, glossary_path="nonexistent.json")
        logger.log_rejection(
            source_text="test",
            draft_tokens=[10, 20],
            accepted_tokens=[10, 99],
            tokenizer=mock_tokenizer,
        )
        files = [f for f in os.listdir(tmp_output_dir) if f.startswith("spec_decode_rejections_")]
        with open(os.path.join(tmp_output_dir, files[0])) as f:
            record = json.loads(f.readline())
        assert "word_20" in record["rejected_words"]

    def test_summary_aggregates_by_domain(self, tmp_output_dir):
        logger = SpecDecodeLogger(output_dir=tmp_output_dir, glossary_path="nonexistent.json")
        # Log multiple rejections across domains
        logger.log_rejection("a", [1, 2], [1, 2], domain="verse")  # 100% accept
        logger.log_rejection("b", [1, 2], [1, 3], domain="verse")  # 50% accept
        logger.log_rejection("c", [1, 2], [3, 4], domain="sermon")  # 0% accept

        summary = logger.summary()
        assert summary["total_sequences"] == 3
        assert summary["by_domain"]["verse"]["count"] == 2
        assert summary["by_domain"]["verse"]["mean_acceptance"] == 0.75
        assert summary["by_domain"]["sermon"]["count"] == 1
        assert summary["by_domain"]["sermon"]["mean_acceptance"] == 0.0

    def test_theological_term_tracking(self, tmp_output_dir, glossary_file):
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.side_effect = lambda ids: "atonement" if ids[0] == 99 else "other"

        logger = SpecDecodeLogger(output_dir=tmp_output_dir, glossary_path=glossary_file)
        logger.log_rejection(
            source_text="The atonement covers sins",
            draft_tokens=[10, 99],
            accepted_tokens=[10, 50],
            tokenizer=mock_tokenizer,
        )
        summary = logger.summary()
        assert ("atonement", 1) in summary["top_rejected_terms"]

    def test_empty_draft_tokens(self, tmp_output_dir):
        logger = SpecDecodeLogger(output_dir=tmp_output_dir, glossary_path="nonexistent.json")
        logger.log_rejection("test", [], [], domain="live")
        files = [f for f in os.listdir(tmp_output_dir) if f.startswith("spec_decode_rejections_")]
        with open(os.path.join(tmp_output_dir, files[0])) as f:
            record = json.loads(f.readline())
        assert record["acceptance_rate"] == 1.0
        assert record["rejection_count"] == 0

    def test_glossary_dict_format(self, tmp_output_dir):
        """Glossary can be list of dicts with 'term' key."""
        path = os.path.join(tmp_output_dir, "glossary.json")
        with open(path, "w") as f:
            json.dump([{"term": "grace", "es": "gracia"}], f)
        logger = SpecDecodeLogger(output_dir=tmp_output_dir, glossary_path=path)
        logger.log_rejection("By grace through faith", [1], [2], domain="verse")
        files = [f for f in os.listdir(tmp_output_dir) if f.startswith("spec_decode_rejections_")]
        with open(os.path.join(tmp_output_dir, files[0])) as f:
            record = json.loads(f.readline())
        assert "grace" in record["theological_terms"]
