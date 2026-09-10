from tools.source_coverage import SourceCoverage


def span(start, end):
    return {"sample_start": start, "sample_end": end, "sample_rate": 16000}


def test_eof_requires_all_source_observed_and_classified():
    ledger = SourceCoverage()
    ledger.observe(span(0, 512), "vad_non_speech")
    ledger.observe(span(512, 1024), "buffered")
    ledger.observe(span(1024, 1536), "buffered")
    ledger.outcome(span(512, 1536), "submitted", 1)
    ledger.eof(1536, 16000)
    assert ledger.snapshot()["unclassified_intervals"] == [[512, 1536]]
    assert not ledger.snapshot()["complete"]
    ledger.outcome(span(512, 1536), "final_ready", 1)
    assert ledger.snapshot()["complete"]
    assert len(ledger.snapshot()["observed"]) == 2


def test_missing_capture_duplicate_and_truncation_never_claim_complete():
    ledger = SourceCoverage(capacity=1)
    ledger.observe(span(512, 1024), "vad_non_speech")
    ledger.observe(span(900, 1400), "buffered")
    ledger.eof(2000, 16000)
    report = ledger.snapshot()
    assert report["capture_gaps"] == [[0, 512], [1024, 2000]]
    assert report["duplicate_observed_samples"] == 124
    assert report["dropped_records"] == 1
    assert not report["complete"]


def test_smart_cut_carryover_and_rejected_tail_account_separately():
    ledger = SourceCoverage()
    ledger.observe(span(0, 3000), "buffered")
    ledger.outcome(span(0, 2000), "final_ready", 1)
    ledger.outcome(span(2000, 3000), "rejected_short_silence", 2)
    assert not ledger.snapshot()["complete"]  # no actual EOF
    ledger.eof(3000, 16000)
    assert ledger.snapshot()["complete"]
    assert len(ledger.snapshot()["outcomes"]) == 2


def test_virtual_tail_is_not_source_and_unknown_bounds_fail_closed():
    ledger = SourceCoverage()
    ledger.observe(span(0, 512), "vad_non_speech")
    ledger.observe({**span(512, 512), "padding_samples": 512}, "vad_non_speech")
    ledger.eof(512, 16000)
    assert ledger.snapshot()["complete"]
    ledger.observe({"sample_rate": 16000}, "buffered")
    assert not ledger.snapshot()["complete"]


def test_observed_source_cannot_extend_beyond_declared_eof():
    ledger = SourceCoverage()
    ledger.observe(span(0, 1024), "vad_non_speech")
    ledger.eof(512, 16000)
    assert ledger.snapshot()["observed_samples_beyond_eof"] == 512
    assert not ledger.snapshot()["complete"]


def test_frozen_vad_mask_retains_transitions_inside_buffered_audio():
    ledger = SourceCoverage()
    ledger.observe(span(0, 512), "buffered", speech=True)
    ledger.observe(span(512, 1024), "buffered", speech=False)
    ledger.observe(span(1024, 1536), "buffered", speech=True)
    assert [r["vad_positive"] for r in ledger.snapshot()["observed"]] == [True, False, True]
