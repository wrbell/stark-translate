"""Tests for pure functions in features/summarize_sermon.py."""

import csv
import json
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest


@pytest.mark.parametrize("language", ["en", "es"])
def test_short_session_actual_csv_and_cli_preserve_recorded_bilingual_text(tmp_path, monkeypatch, language):
    import dry_run_ab as pipeline

    csv_path = tmp_path / f"ab_metrics_rehearsal_{language}.csv"
    output = tmp_path / "summary.json"
    source, target = "La gracia de Dios es suficiente.", "God's grace is enough."
    if language == "en":
        source, target = target, source
    monkeypatch.setattr(pipeline, "CSV_PATH", str(csv_path))
    pipeline.init_csv()
    pipeline.write_csv_row(
        dict(
            chunk_id=1,
            timestamp="2026-09-09T23:38:41",
            english=source,
            spanish_a=target,
            spanish_b="",
            stt_latency_ms=2081.6,
            latency_a_ms=41.0,
            latency_b_ms=0.0,
            e2e_latency_ms=2127.2,
            source_lang=language,
            target_lang="es" if language == "en" else "en",
        )
    )
    # An unusable model identity proves this path needs no model or network.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "features.summarize_sermon",
            str(csv_path),
            "--model",
            "missing/model",
            "-o",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(output.read_text())
    assert summary["english"] == "God's grace is enough."
    assert summary["spanish"] == "La gracia de Dios es suficiente."
    assert summary["format"] == "short-session excerpt"
    assert summary["metadata"]["human_reviewed"] is False
    assert summary["metadata"]["model"] is None


def test_short_session_missing_translation_uses_known_reverse_direction(summary_runtime):
    summary, _, runtime, calls, _ = summary_runtime
    runtime.generate = lambda *args, **kwargs: "God is love."
    result = summary.summarize_short_session([{"text": "Dios es amor.", "source_lang": "es"}])
    assert result["english"] == "God is love."
    assert result["spanish"] == "Dios es amor."
    assert result["translation_method"] == "generated translation"
    message = [call for call in calls if call[0] == "template"][-1][1][0]["content"]
    assert "Spanish text to English" in message


def test_legacy_short_session_requires_language_resolution(tmp_path):
    from features import summarize_sermon as summary

    path = tmp_path / "legacy.csv"
    path.write_text("english,spanish_a\nDios es amor.,God is love.\n")
    entries = summary.load_csv_transcript(path)
    with pytest.raises(ValueError, match="ambiguous"):
        summary.summarize_short_session(entries)
    renamed = tmp_path / "legacy_es.csv"
    path.rename(renamed)
    assert summary.summarize_short_session(summary.load_csv_transcript(renamed))["english"] == "God is love."


def test_empty_translation_does_not_produce_success(summary_runtime):
    summary, _, runtime, _, _ = summary_runtime
    runtime.generate = lambda *args, **kwargs: ""
    with pytest.raises(ValueError, match="empty"):
        summary.summarize_short_session([{"text": "Love.", "source_lang": "en"}])


@pytest.fixture
def summary_runtime(monkeypatch, tmp_path):
    from features import summarize_sermon as summary

    calls = []

    class Tokenizer:
        eos_token_id = 1
        unk_token_id = 3

        def __init__(self):
            self._eos_token_ids = {1, 50}

        def convert_tokens_to_ids(self, token):
            return 106 if token == "<turn|>" else 3

        def apply_chat_template(self, messages, **kwargs):
            calls.append(("template", messages, kwargs))
            return "rendered-prompt"

    tokenizer = Tokenizer()
    core = types.ModuleType("mlx.core")
    core.set_cache_limit = lambda limit: None
    core.synchronize = lambda: calls.append(("synchronize", threading.get_ident()))
    mlx = types.ModuleType("mlx")
    mlx.core = core
    runtime = types.ModuleType("mlx_lm")
    runtime.load = lambda path: (calls.append(("load", path)) or "model", tokenizer)
    runtime.generate = lambda *a, **kw: calls.append(("generate", threading.get_ident(), kw)) or "Summary.<turn|>junk"
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx_lm", runtime)
    cached = tmp_path / "cached-model"
    cached.mkdir()
    (cached / "config.json").write_text('{"model_type":"gemma4"}')
    monkeypatch.setattr(
        summary, "resolve_model_for_loading", lambda identity: calls.append(("resolve", identity)) or str(cached)
    )
    return summary, tokenizer, runtime, calls, cached


def test_summary_default_uses_cache_family_stops_and_load_thread_warmup(summary_runtime):
    summary, tokenizer, _, calls, cached = summary_runtime
    model, loaded = summary.load_summarization_model()
    assert calls[0] == ("resolve", summary.settings.translation.mlx_model_gemma4_e4b)
    assert calls[1] == ("load", str(cached))
    assert loaded is tokenizer and tokenizer._eos_token_ids == {1, 50, 106}
    warmup = next(call for call in calls if call[0] == "generate")
    assert warmup[1] == threading.get_ident() and warmup[2]["max_tokens"] == 1
    assert summary.generate_text(model, tokenizer, "Summary prompt") == "Summary."
    assert all(call[2]["enable_thinking"] is False for call in calls if call[0] == "template")


def test_summary_explicit_non_gemma_override_preserves_template_and_eos(summary_runtime):
    summary, tokenizer, _, calls, cached = summary_runtime
    (cached / "config.json").write_text('{"model_type":"llama"}')
    summary.load_summarization_model("explicit/llama-model")
    assert calls[0] == ("resolve", "explicit/llama-model")
    assert tokenizer._eos_token_ids == {1, 50}
    assert all("enable_thinking" not in call[2] for call in calls if call[0] == "template")


def test_summary_failed_first_forward_is_not_reported_ready(summary_runtime):
    summary, _, runtime, _, _ = summary_runtime

    def failed(*args, **kwargs):
        raise RuntimeError("first forward failed")

    runtime.generate = failed
    with pytest.raises(RuntimeError, match="first forward failed"):
        summary.load_summarization_model()


@pytest.mark.parametrize("diarized", [False, True])
def test_spanish_summary_uses_live_translation_prompt_without_loading_another_model(
    summary_runtime, monkeypatch, diarized
):
    from engines.translation_prompts import build_chat_messages

    summary, tokenizer, runtime, calls, _ = summary_runtime
    monkeypatch.setattr(summary.settings.translation, "terminology_prompt", "none")
    model, tokenizer = summary.load_summarization_model()
    english = "God loves the world. Christ died for our sins. John 3:16 offers hope."
    spanish = "Dios ama al mundo. Cristo murió por nuestros pecados. Juan 3:16 ofrece esperanza."
    replies = iter([english, f"Here is the translation:\n{spanish}<turn|><|channel>thought"])

    def generate(loaded_model, loaded_tokenizer, **kwargs):
        assert loaded_model is model and loaded_tokenizer is tokenizer
        calls.append(("summary_generate", kwargs))
        return next(replies)

    runtime.generate = generate
    if diarized:
        result = summary.summarize_with_diarization(model, tokenizer, "Transcript", {"A": "One", "B": "Two"})
    else:
        result = summary.summarize_without_diarization(model, tokenizer, "Transcript")

    assert result["english"] == english
    assert result["spanish"] == spanish
    template = [call for call in calls if call[0] == "template"][-1]
    assert template[1] == build_chat_messages(english, source_lang="en", target_lang="es", model_family="gemma4")
    assert "Output only the translation, nothing else." in template[1][0]["content"]
    assert template[2]["enable_thinking"] is False
    assert len([call for call in calls if call[0] == "load"]) == 1
    assert tokenizer._eos_token_ids == {1, 50, 106}


@pytest.mark.parametrize("model_type,family", [("llama", None), ("translategemma", "translategemma")])
def test_spanish_translation_preserves_overridden_model_template(summary_runtime, model_type, family):
    summary, tokenizer, runtime, calls, cached = summary_runtime
    (cached / "config.json").write_text(json.dumps({"model_type": model_type}))
    model, _ = summary.load_summarization_model(f"custom/{model_type}")
    runtime.generate = lambda *args, **kwargs: "Dios es amor.<end_of_turn>"
    assert summary.translate_summary(model, tokenizer, "God is love.") == "Dios es amor."
    template = [call for call in calls if call[0] == "template"][-1]
    content = template[1][0]["content"]
    if family == "translategemma":
        assert content == [{"type": "text", "source_lang_code": "en", "target_lang_code": "es", "text": "God is love."}]
    else:
        assert isinstance(content, str) and "Output only the translation" in content
    assert "enable_thinking" not in template[2]
    assert len([call for call in calls if call[0] == "load"]) == 1


# ===================================================================
# _format_timestamp
# ===================================================================


class TestFormatTimestamp:
    def test_zero(self):
        from features.summarize_sermon import _format_timestamp

        assert _format_timestamp(0) == "00:00:00"

    def test_90_seconds(self):
        from features.summarize_sermon import _format_timestamp

        assert _format_timestamp(90) == "00:01:30"

    def test_over_one_hour(self):
        from features.summarize_sermon import _format_timestamp

        assert _format_timestamp(3661) == "01:01:01"

    def test_fractional_truncated(self):
        from features.summarize_sermon import _format_timestamp

        # int(seconds % 60) truncates fractional part
        assert _format_timestamp(90.9) == "00:01:30"


# ===================================================================
# has_diarization
# ===================================================================


class TestHasDiarization:
    def test_two_speakers(self):
        from features.summarize_sermon import has_diarization

        entries = [
            {"text": "Hello", "speaker": "Speaker A"},
            {"text": "World", "speaker": "Speaker B"},
        ]
        assert has_diarization(entries) is True

    def test_one_speaker(self):
        from features.summarize_sermon import has_diarization

        entries = [
            {"text": "Hello", "speaker": "Speaker A"},
            {"text": "World", "speaker": "Speaker A"},
        ]
        assert has_diarization(entries) is False

    def test_no_speaker_key(self):
        from features.summarize_sermon import has_diarization

        entries = [{"text": "Hello"}, {"text": "World"}]
        assert has_diarization(entries) is False

    def test_empty_list(self):
        from features.summarize_sermon import has_diarization

        assert has_diarization([]) is False


# ===================================================================
# build_transcript_text
# ===================================================================


class TestBuildTranscriptText:
    def test_short_passthrough(self):
        from features.summarize_sermon import build_transcript_text

        entries = [
            {"text": "Hello world", "speaker": None},
            {"text": "How are you", "speaker": None},
        ]
        result = build_transcript_text(entries, max_chars=1000)
        assert result == "Hello world\nHow are you"

    def test_with_speaker_labels(self):
        from features.summarize_sermon import build_transcript_text

        entries = [
            {"text": "Hello", "speaker": "Speaker A"},
            {"text": "Hi", "speaker": "Speaker B"},
        ]
        result = build_transcript_text(entries, max_chars=1000)
        assert "Speaker A: Hello" in result
        assert "Speaker B: Hi" in result

    def test_truncation(self):
        from features.summarize_sermon import build_transcript_text

        entries = [{"text": "X" * 200, "speaker": None} for _ in range(10)]
        result = build_transcript_text(entries, max_chars=500)
        assert "[... middle portion omitted for brevity ...]" in result

    def test_exact_boundary(self):
        from features.summarize_sermon import build_transcript_text

        entries = [{"text": "ABC", "speaker": None}]
        result = build_transcript_text(entries, max_chars=3)
        assert result == "ABC"


# ===================================================================
# get_speaker_texts
# ===================================================================


class TestGetSpeakerTexts:
    def test_two_speakers(self):
        from features.summarize_sermon import get_speaker_texts

        entries = [
            {"text": "Hello", "speaker": "A"},
            {"text": "World", "speaker": "B"},
            {"text": "Again", "speaker": "A"},
        ]
        result = get_speaker_texts(entries)
        assert result["A"] == "Hello Again"
        assert result["B"] == "World"

    def test_missing_speaker_defaults_to_unknown(self):
        from features.summarize_sermon import get_speaker_texts

        entries = [{"text": "Hello"}, {"text": "World"}]
        result = get_speaker_texts(entries)
        assert "Unknown" in result

    def test_single_speaker(self):
        from features.summarize_sermon import get_speaker_texts

        entries = [
            {"text": "One", "speaker": "X"},
            {"text": "Two", "speaker": "X"},
        ]
        result = get_speaker_texts(entries)
        assert len(result) == 1
        assert result["X"] == "One Two"


# ===================================================================
# load_csv_transcript
# ===================================================================


class TestLoadCsvTranscript:
    def test_valid_csv(self, tmp_path):
        from features.summarize_sermon import load_csv_transcript

        csv_path = str(tmp_path / "test.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "english"])
            writer.writeheader()
            writer.writerow({"timestamp": "00:00:01", "english": "Hello world"})
            writer.writerow({"timestamp": "00:00:05", "english": "God is love"})
        result = load_csv_transcript(csv_path)
        assert len(result) == 2
        assert result[0]["text"] == "Hello world"

    def test_empty_csv(self, tmp_path):
        from features.summarize_sermon import load_csv_transcript

        csv_path = str(tmp_path / "empty.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "english"])
            writer.writeheader()
        result = load_csv_transcript(csv_path)
        assert result == []


# ===================================================================
# load_jsonl_transcript
# ===================================================================


class TestLoadJsonlTranscript:
    def test_valid_jsonl(self, tmp_path):
        from features.summarize_sermon import load_jsonl_transcript

        path = str(tmp_path / "test.jsonl")
        with open(path, "w") as f:
            # Metadata header — should be skipped
            f.write(json.dumps({"_metadata": {"source": "test"}}) + "\n")
            f.write(json.dumps({"speaker": "A", "start": 10, "text": "Hello"}) + "\n")
            f.write(json.dumps({"speaker": "B", "start": 20, "text": "World"}) + "\n")
        result = load_jsonl_transcript(path)
        assert len(result) == 2
        assert result[0]["text"] == "Hello"
        assert result[0]["speaker"] == "A"

    def test_skips_transcription_errors(self, tmp_path):
        from features.summarize_sermon import load_jsonl_transcript

        path = str(tmp_path / "err.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"speaker": "A", "start": 0, "text": "[transcription error: timeout]"}) + "\n")
            f.write(json.dumps({"speaker": "A", "start": 5, "text": "Valid text"}) + "\n")
        result = load_jsonl_transcript(path)
        assert len(result) == 1
        assert result[0]["text"] == "Valid text"


# ===================================================================
# write_summary
# ===================================================================


class TestWriteSummary:
    def test_creates_json_file(self, tmp_path):
        from features.summarize_sermon import write_summary

        output = str(tmp_path / "summaries" / "test.json")
        data = {"english": "Summary", "spanish": "Resumen"}
        write_summary(data, output)
        with open(output) as f:
            loaded = json.load(f)
        assert loaded["english"] == "Summary"
        assert loaded["spanish"] == "Resumen"
