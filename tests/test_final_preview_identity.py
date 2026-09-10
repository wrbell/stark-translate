"""Final diagnostics must join previews by capture identity, never chunk order."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from tools.pipeline_timing import ChunkTiming


def test_final_admission_prunes_only_preview_history_older_than_existing_retention_window(monkeypatch):
    import dry_run_ab as d

    translations = {1: "expired", 71: "expired", 72: "boundary", 199: "recent", 200: "current", 201: "future"}
    latencies = {uid: {"pt_ms": uid} for uid in translations}
    monkeypatch.setattr(d, "partial_translations", translations)
    monkeypatch.setattr(d, "partial_latencies", latencies)
    monkeypatch.setattr(d, "_closed_utterances", set())
    monkeypatch.setattr(d, "_stt_scheduler", None)
    monkeypatch.setattr(d, "_active_partial_future", None)
    monkeypatch.setattr(d, "pipeline_submit", AsyncMock())
    # Isolate the existing final-admission control state from other tests.
    for name in (
        "_partial_emitted_sequence",
        "_partial_source_text",
        "_pause_epochs",
        "_speculative_candidates",
        "_speculation_attempts",
        "_rolling_previews",
        "_utterance_start_times",
        "_utterance_timings",
    ):
        monkeypatch.setattr(d, name, {})
    monkeypatch.setattr(d, "_final_pending", threading.Event())
    monkeypatch.setattr(d, "_final_pending_utterance_id", None)
    asyncio.run(d.process_final(np.ones(16000), 200))
    assert translations == {72: "boundary", 199: "recent", 200: "current", 201: "future"}
    assert latencies == {uid: {"pt_ms": uid} for uid in (72, 199, 200, 201)}
    # An unidentified legacy final supplies no safe retention cutoff.
    asyncio.run(d.process_final(np.ones(16000), None))
    assert set(translations) == set(latencies) == {72, 199, 200, 201}


@pytest.mark.parametrize("utterance_id", [8, None, 10])
def test_actual_finalizer_consumes_only_its_exact_utterance_preview(monkeypatch, utterance_id):
    import dry_run_ab as d

    translations = {None: "unknown identity", 3: "other capture", 8: "correct preview", 9: "newer preview"}
    latencies = {key: {"pt_ms": index + 10, "stt_ms": index + 100} for index, key in enumerate(translations)}
    initial_translations, initial_latencies = dict(translations), dict(latencies)
    monkeypatch.setattr(d, "partial_translations", translations)
    monkeypatch.setattr(d, "partial_latencies", latencies)
    monkeypatch.setattr(d, "MULTIPROCESS", False)
    monkeypatch.setattr(d, "mlx_a_model", None)
    monkeypatch.setattr(d, "DIARIZE_ENABLED", False)
    monkeypatch.setattr(d, "tts_engine", None)
    monkeypatch.setattr(d, "_health", None)
    monkeypatch.setattr(d, "all_results", [])
    monkeypatch.setattr(d, "diag_durations", [])
    monkeypatch.setattr(d, "_last_gen_stats", {})
    monkeypatch.setattr(d, "_confirmed_speculation", lambda *args: None)
    monkeypatch.setattr(d, "translate_marian", lambda text: ("La traducción final.", 20))
    monkeypatch.setattr(d, "qe_score", lambda *args: 0.9)
    monkeypatch.setattr(d, "get_resource_snapshot", lambda: {})
    for name in ("check_homophones", "check_bad_split", "check_near_miss", "check_marian_divergence"):
        monkeypatch.setattr(d, name, Mock())
    stability = Mock(return_value=0.75)
    monkeypatch.setattr(d, "compute_word_stability", stability)
    monkeypatch.setattr(d, "write_csv_row", Mock())
    monkeypatch.setattr(d, "write_diag_jsonl", Mock())
    monkeypatch.setattr(d, "save_chunk_audio", Mock(return_value="unused-mocked-audio.wav"))
    monkeypatch.setattr(d, "_io_pool", Mock(submit=Mock(side_effect=lambda fn, *args: fn(*args))))
    messages = []

    async def broadcast(data):
        messages.append(dict(data))
        # In production another utterance can publish while final delivery
        # yields. Its latest preview must survive both CSV and JSONL submission.
        await asyncio.sleep(0)
        translations[11] = "preview published during final broadcast"
        latencies[11] = {"pt_ms": 90, "stt_ms": 900}

    monkeypatch.setattr(d, "broadcast", broadcast)

    async def run():
        monkeypatch.setattr(d, "_pipeline_translation_lock", asyncio.Lock())
        timing = ChunkTiming(utterance_id=utterance_id) if utterance_id is not None else None
        await d._pipeline_translate_and_finalize(3, "Final source", 30, 0.9, [], [], np.ones(16000), 1, timing=timing)

    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(d, "_pytorch_pool", pool)
        asyncio.run(run())
    assert len(messages) == 1 and messages[0]["chunk_id"] == 3
    assert messages[0]["utterance_id"] == utterance_id
    expected_latency = initial_latencies[8] if utterance_id == 8 else None
    if utterance_id == 8:
        stability.assert_called_once_with("correct preview", "La traducción final.")
        d.check_marian_divergence.assert_called_once_with(3, "correct preview", "La traducción final.")
        assert messages[0]["word_stability_pct"] == 0.75
        initial_translations.pop(8)
        initial_latencies.pop(8)
    else:
        stability.assert_not_called()
        d.check_marian_divergence.assert_not_called()
        assert messages[0]["word_stability_pct"] is None
    assert translations == {**initial_translations, 11: "preview published during final broadcast"}
    assert latencies == {**initial_latencies, 11: {"pt_ms": 90, "stt_ms": 900}}
    assert d.write_csv_row.call_args.args[1] == expected_latency
    assert d.write_diag_jsonl.call_args.kwargs["marian_lat"] == expected_latency
