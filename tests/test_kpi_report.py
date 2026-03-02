"""Tests for tools/kpi_report.py — KPI scorecard generator."""

import json

import pytest


class TestLoadCsv:
    """Tests for load_csv()."""

    def test_loads_csv_rows(self, tmp_path):
        from tools.kpi_report import load_csv

        csv_file = tmp_path / "test.csv"
        csv_file.write_text("chunk_id,stt_latency_ms,e2e_latency_ms\n1,500,1200\n2,550,1300\n")
        rows = load_csv(str(csv_file))
        assert len(rows) == 2
        assert rows[0]["chunk_id"] == "1"
        assert rows[1]["e2e_latency_ms"] == "1300"

    def test_empty_csv_returns_empty(self, tmp_path):
        from tools.kpi_report import load_csv

        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("chunk_id,stt_latency_ms\n")
        rows = load_csv(str(csv_file))
        assert rows == []


class TestFindDiagnosticsForCsv:
    """Tests for find_diagnostics_for_csv()."""

    def test_finds_matching_jsonl(self, tmp_path):
        from tools.kpi_report import find_diagnostics_for_csv

        csv_file = tmp_path / "ab_metrics_20260301_180046_en.csv"
        diag_file = tmp_path / "diagnostics_20260301_180046_en.jsonl"
        csv_file.write_text("")
        diag_file.write_text("")
        result = find_diagnostics_for_csv(str(csv_file))
        assert result == str(diag_file)

    def test_returns_none_when_missing(self, tmp_path):
        from tools.kpi_report import find_diagnostics_for_csv

        csv_file = tmp_path / "ab_metrics_20260301_180046_en.csv"
        csv_file.write_text("")
        result = find_diagnostics_for_csv(str(csv_file))
        assert result is None

    def test_returns_none_for_non_matching_name(self, tmp_path):
        from tools.kpi_report import find_diagnostics_for_csv

        csv_file = tmp_path / "random_file.csv"
        csv_file.write_text("")
        result = find_diagnostics_for_csv(str(csv_file))
        assert result is None


class TestLoadDiagnostics:
    """Tests for load_diagnostics()."""

    def test_loads_jsonl_records(self, tmp_path):
        from tools.kpi_report import load_diagnostics

        jsonl_file = tmp_path / "diag.jsonl"
        jsonl_file.write_text(
            json.dumps({"chunk_id": 1, "event": "empty_stt_drop"})
            + "\n"
            + json.dumps({"chunk_id": 2, "stt_confidence": 0.8})
            + "\n"
        )
        records = load_diagnostics(str(jsonl_file))
        assert len(records) == 2
        assert records[0]["event"] == "empty_stt_drop"


class TestComputeLatencyKpis:
    """Tests for compute_latency_kpis()."""

    def _make_rows(self, e2e_values, stt_values=None):
        rows = []
        for i, e2e in enumerate(e2e_values):
            row = {"e2e_latency_ms": str(e2e), "chunk_id": str(i + 1)}
            if stt_values and i < len(stt_values):
                row["stt_latency_ms"] = str(stt_values[i])
            rows.append(row)
        return rows

    def test_basic_percentiles(self):
        from tools.kpi_report import compute_latency_kpis

        rows = self._make_rows([1000, 1200, 1500, 2000, 2500])
        result = compute_latency_kpis(rows)
        assert result["e2e_p50"] == 1500
        assert result["e2e_count"] == 5
        assert result["stall_count"] == 0

    def test_stall_filtering(self):
        from tools.kpi_report import compute_latency_kpis

        rows = self._make_rows([1000, 1200, 15000, 20000])
        result = compute_latency_kpis(rows)
        assert result["e2e_count"] == 2  # only non-stall values
        assert result["stall_count"] == 2

    def test_prefers_true_e2e_ms(self):
        from tools.kpi_report import compute_latency_kpis

        rows = [
            {"e2e_latency_ms": "1000", "true_e2e_ms": "5000", "chunk_id": "1"},
        ]
        result = compute_latency_kpis(rows)
        assert result["e2e_p50"] == 5000  # true_e2e_ms takes priority

    def test_falls_back_to_e2e_latency(self):
        from tools.kpi_report import compute_latency_kpis

        rows = [
            {"e2e_latency_ms": "1000", "chunk_id": "1"},
        ]
        result = compute_latency_kpis(rows)
        assert result["e2e_p50"] == 1000

    def test_empty_rows(self):
        from tools.kpi_report import compute_latency_kpis

        result = compute_latency_kpis([])
        assert result["e2e_count"] == 0
        assert result["e2e_p50"] == 0.0

    def test_translation_b_latency(self):
        from tools.kpi_report import compute_latency_kpis

        rows = [
            {"e2e_latency_ms": "1000", "latency_a_ms": "400", "latency_b_ms": "800", "chunk_id": "1"},
            {"e2e_latency_ms": "1200", "latency_a_ms": "450", "latency_b_ms": "0", "chunk_id": "2"},
        ]
        result = compute_latency_kpis(rows)
        assert result["trans_b_p50"] == 800  # only one >0 value


class TestComputeWerKpis:
    """Tests for compute_wer_kpis()."""

    def test_confidence_stats(self):
        from tools.kpi_report import compute_wer_kpis

        rows = [
            {"stt_confidence": "0.8"},
            {"stt_confidence": "0.6"},
            {"stt_confidence": "0.4"},
        ]
        result = compute_wer_kpis(rows, [])
        assert result["conf_mean"] == pytest.approx(0.6)
        assert result["low_conf_count"] == 1  # 0.4 < 0.5
        assert result["total_chunks"] == 3

    def test_empty_stt_count_from_diag(self):
        from tools.kpi_report import compute_wer_kpis

        diag = [
            {"event": "empty_stt_drop"},
            {"event": "empty_stt_drop"},
            {"chunk_id": 1},  # regular record
        ]
        result = compute_wer_kpis([], diag)
        assert result["empty_stt_count"] == 2


class TestComputeStabilityKpis:
    """Tests for compute_stability_kpis()."""

    def test_word_stability_column(self):
        from tools.kpi_report import compute_stability_kpis

        rows = [
            {"word_stability_pct": "0.85"},
            {"word_stability_pct": "0.95"},
        ]
        result = compute_stability_kpis(rows, [])
        assert result["source"] == "word_stability_pct"
        assert result["mean"] == pytest.approx(0.9)

    def test_falls_back_to_marian_similarity(self):
        from tools.kpi_report import compute_stability_kpis

        rows = [
            {"marian_similarity": "0.5"},
            {"marian_similarity": "0.6"},
        ]
        result = compute_stability_kpis(rows, [])
        assert result["source"] == "marian_similarity"
        assert result["mean"] == pytest.approx(0.55)

    def test_empty_data(self):
        from tools.kpi_report import compute_stability_kpis

        result = compute_stability_kpis([], [])
        assert result["mean"] is None
        assert result["count"] == 0


class TestComputeReliabilityKpis:
    """Tests for compute_reliability_kpis()."""

    def test_from_session_summary(self):
        from tools.kpi_report import compute_reliability_kpis

        rows = [{"chunk_id": str(i)} for i in range(1, 11)]  # 10 rows
        diag = [
            {
                "event": "session_summary",
                "chunks_attempted": 12,
                "chunks_completed": 10,
                "chunks_empty_stt": 1,
                "chunks_hallucination": 1,
                "chunks_dedup": 0,
            },
        ]
        result = compute_reliability_kpis(rows, diag)
        assert result["attempted"] == 12
        assert result["completed"] == 10
        assert result["rate"] == pytest.approx(83.3, abs=0.1)

    def test_estimated_from_chunk_ids(self):
        from tools.kpi_report import compute_reliability_kpis

        # chunk_ids 2,4,6 — max=6, min=2, range=5
        rows = [
            {"chunk_id": "2"},
            {"chunk_id": "4"},
            {"chunk_id": "6"},
        ]
        result = compute_reliability_kpis(rows, [])
        assert result["attempted"] == 5  # 6 - 2 + 1
        assert result["completed"] == 3
        assert result["rate"] == 60.0


class TestGrade:
    """Tests for grade()."""

    def test_higher_is_better_excellent(self):
        from tools.kpi_report import grade

        assert grade(0.99, 0.98, 0.95) == "EXCELLENT"

    def test_higher_is_better_pass(self):
        from tools.kpi_report import grade

        assert grade(0.96, 0.98, 0.95) == "PASS"

    def test_higher_is_better_needs_work(self):
        from tools.kpi_report import grade

        assert grade(0.90, 0.98, 0.95) == "NEEDS WORK"

    def test_lower_is_better_excellent(self):
        from tools.kpi_report import grade

        assert grade(1500, 2000, 3000, higher_is_better=False) == "EXCELLENT"

    def test_lower_is_better_pass(self):
        from tools.kpi_report import grade

        assert grade(2500, 2000, 3000, higher_is_better=False) == "PASS"

    def test_lower_is_better_needs_work(self):
        from tools.kpi_report import grade

        assert grade(3500, 2000, 3000, higher_is_better=False) == "NEEDS WORK"

    def test_none_value(self):
        from tools.kpi_report import grade

        assert grade(None, 0.98, 0.95) == "N/A"

    def test_exact_boundary_higher_is_better(self):
        from tools.kpi_report import grade

        assert grade(0.95, 0.98, 0.95) == "PASS"
        assert grade(0.98, 0.98, 0.95) == "EXCELLENT"


class TestGenerateScorecard:
    """Tests for generate_scorecard()."""

    def test_produces_markdown_table(self):
        from tools.kpi_report import generate_scorecard

        latency = {"e2e_p95": 2500}
        wer = {"conf_mean": 0.75}
        translation = {"qe_a_mean": 0.90}
        stability = {"mean": 0.97, "source": "word_stability_pct"}
        reliability = {"rate": 99.0}

        result = generate_scorecard(latency, wer, translation, stability, reliability)
        assert "| # | KPI" in result
        assert "PASS" in result or "EXCELLENT" in result
        assert "2500ms" in result


class TestGenerateReport:
    """Tests for generate_report()."""

    def test_includes_all_sections(self):
        from tools.kpi_report import generate_report

        latency = {
            "e2e_p50": 1200,
            "e2e_p95": 2500,
            "e2e_p99": 3000,
            "e2e_count": 50,
            "stt_p50": 500,
            "stt_p95": 700,
            "trans_a_p50": 400,
            "trans_a_p95": 600,
            "trans_b_p50": None,
            "trans_b_p95": None,
            "stall_count": 0,
        }
        wer = {
            "conf_mean": 0.75,
            "conf_p10": 0.55,
            "conf_p25": 0.65,
            "conf_count": 50,
            "empty_stt_count": 2,
            "hallucination_count": 1,
            "low_conf_count": 5,
            "low_conf_pct": 10.0,
            "total_chunks": 50,
        }
        translation = {
            "qe_a_mean": 0.90,
            "qe_a_median": 0.92,
            "qe_a_p10": 0.80,
            "qe_a_count": 50,
            "qe_b_mean": None,
            "qe_b_count": 0,
            "comet_mean": None,
            "comet_median": None,
            "comet_p10": None,
        }
        stability = {"mean": 0.97, "median": 0.98, "p10": 0.85, "count": 40, "source": "word_stability_pct"}
        reliability = {"attempted": 55, "completed": 50, "empty_stt": 3, "hallucination": 1, "dedup": 1, "rate": 90.9}

        scorecard = "| # | KPI | Value | Target | Grade |\n|---|-----|-------|--------|-------|\n"
        report = generate_report(
            ["metrics/test.csv"],
            latency,
            wer,
            translation,
            stability,
            reliability,
            scorecard,
        )
        assert "# KPI Report" in report
        assert "## 1. Latency Detail" in report
        assert "## 2. WER Proxy Detail" in report
        assert "## 3. Translation Quality Detail" in report
        assert "## 4. Word Stability Detail" in report
        assert "## 5. Processing Rate Detail" in report


class TestPercentile:
    """Tests for _percentile() helper."""

    def test_p50_odd_count(self):
        from tools.kpi_report import _percentile

        assert _percentile([1, 2, 3, 4, 5], 50) == 3

    def test_p50_even_count(self):
        from tools.kpi_report import _percentile

        assert _percentile([1, 2, 3, 4], 50) == 2.5

    def test_p95(self):
        from tools.kpi_report import _percentile

        values = list(range(1, 101))
        result = _percentile(values, 95)
        assert 95 <= result <= 96

    def test_empty(self):
        from tools.kpi_report import _percentile

        assert _percentile([], 50) == 0.0

    def test_single_value(self):
        from tools.kpi_report import _percentile

        assert _percentile([42], 50) == 42
        assert _percentile([42], 95) == 42
