"""Speculative decoding rejection logger for training feedback.

Logs every draft token rejection during speculative decoding as a JSONL
record.  These records feed the active learning loop -- rejected tokens
reveal where the draft model (4B or EAGLE head) diverges from the
verifier (12B), providing organic hard-example discovery for the next
fine-tuning cycle.

Usage:
    from engines.spec_decode_logger import SpecDecodeLogger

    logger = SpecDecodeLogger()
    logger.log_rejection(
        source_text="Christ's atonement covers all sins.",
        draft_tokens=[1, 2, 3, 4],
        accepted_tokens=[1, 2, 5, 6],
        domain="sermon",
    )
    summary = logger.summary()
"""

import json
import logging
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metrics",
)


class SpecDecodeLogger:
    """Log speculative decoding rejections for training feedback.

    Args:
        output_dir:     Directory for JSONL output files.
        glossary_path:  Path to Tier 1 theological terms JSON for
                        cross-referencing rejected tokens with domain terms.
    """

    def __init__(
        self,
        output_dir: str = _DEFAULT_OUTPUT_DIR,
        glossary_path: str = "bible_data/glossary/tier1_boost.json",
    ):
        self._output_dir = output_dir
        self._tier1_terms: list[str] = []
        self._session_stats: dict[str, list[float]] = defaultdict(list)
        self._term_rejections: Counter = Counter()

        # Load Tier 1 glossary for theological term detection
        if os.path.exists(glossary_path):
            try:
                with open(glossary_path, encoding="utf-8") as f:
                    data = json.load(f)
                # Handle both list-of-strings and list-of-dicts formats
                if data and isinstance(data[0], str):
                    self._tier1_terms = [t.lower() for t in data]
                elif data and isinstance(data[0], dict):
                    self._tier1_terms = [t.get("term", "").lower() for t in data]
                logger.debug("Loaded %d Tier 1 terms for rejection tracking", len(self._tier1_terms))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load glossary from %s: %s", glossary_path, exc)

    @property
    def output_path(self) -> str:
        date_str = datetime.now(UTC).strftime("%Y%m%d")
        return os.path.join(self._output_dir, f"spec_decode_rejections_{date_str}.jsonl")

    def log_rejection(
        self,
        source_text: str,
        draft_tokens: list[int],
        accepted_tokens: list[int],
        *,
        domain: str = "live",
        draft_model: str = "",
        verifier_model: str = "",
        tokenizer=None,
    ) -> None:
        """Append a JSONL record documenting speculative decode rejections.

        Args:
            source_text:      Input text that was translated.
            draft_tokens:     Token IDs proposed by the draft model.
            accepted_tokens:  Token IDs accepted by the verifier.
            domain:           Context: "verse", "sermon", or "live".
            draft_model:      Model ID of the draft engine.
            verifier_model:   Model ID of the verifier engine.
            tokenizer:        Optional tokenizer for decoding rejected tokens
                              to human-readable words.
        """
        # Compute rejection positions
        min_len = min(len(draft_tokens), len(accepted_tokens))
        rejection_positions = [i for i in range(min_len) if draft_tokens[i] != accepted_tokens[i]]
        # Also count tokens beyond the shorter sequence as rejections
        if len(draft_tokens) > len(accepted_tokens):
            rejection_positions.extend(range(min_len, len(draft_tokens)))

        total = max(len(draft_tokens), 1)
        acceptance_rate = 1.0 - (len(rejection_positions) / total)

        # Decode rejected tokens if tokenizer available
        rejected_words: list[str] = []
        if tokenizer is not None and rejection_positions:
            rejected_ids = [draft_tokens[i] for i in rejection_positions if i < len(draft_tokens)]
            try:
                rejected_words = [tokenizer.decode([tid]).strip() for tid in rejected_ids]
            except Exception:
                pass

        # Cross-reference with theological glossary
        source_lower = source_text.lower()
        theological_terms = [t for t in self._tier1_terms if t in source_lower]

        # Track session stats
        self._session_stats[domain].append(acceptance_rate)
        for word in rejected_words:
            if word.lower() in self._tier1_terms:
                self._term_rejections[word.lower()] += 1

        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "source_text": source_text,
            "acceptance_rate": round(acceptance_rate, 4),
            "rejection_count": len(rejection_positions),
            "total_draft_tokens": len(draft_tokens),
            "rejection_positions": rejection_positions[:20],  # Cap for readability
            "rejected_words": rejected_words[:20],
            "domain": domain,
            "theological_terms": theological_terms,
            "draft_model": draft_model,
            "verifier_model": verifier_model,
        }

        try:
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Failed to write spec decode rejection log: %s", exc)

    def summary(self) -> dict:
        """Return session-level aggregated statistics.

        Returns dict with acceptance rates by domain, most-rejected
        theological terms, and total rejection count.
        """
        domain_stats = {}
        for domain, rates in self._session_stats.items():
            if rates:
                sorted_rates = sorted(rates)
                domain_stats[domain] = {
                    "count": len(rates),
                    "mean_acceptance": round(sum(rates) / len(rates), 4),
                    "min_acceptance": round(sorted_rates[0], 4),
                    "median_acceptance": round(sorted_rates[len(sorted_rates) // 2], 4),
                }

        return {
            "total_sequences": sum(len(r) for r in self._session_stats.values()),
            "by_domain": domain_stats,
            "top_rejected_terms": self._term_rejections.most_common(10),
        }
