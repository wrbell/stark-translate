"""Offline Hindi provenance/coverage failures cannot become quality claims."""

import hashlib
import json
import wave
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from engines.base import STTResult, TranslationResult
from tools import offline_hindi as hindi


@pytest.fixture
def manifest_fixture(tmp_path, monkeypatch):
    audio = tmp_path / "sermon.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes((np.arange(16000) % 1000).astype("<i2").tobytes())
    manifest = {
        "schema_version": 1,
        "usage": "evaluation_only",
        "source_lang": "en",
        "target_lang": "hi",
        "audio_sources": [
            {
                "id": "church",
                "path": "sermon.wav",
                "sha256": hindi.digest(audio),
                "duration": 1,
                "provenance": "church_replay",
                "offset_s": 123,
            }
        ],
        "text_probes": [
            {
                "id": "canary_1",
                "kind": "theological_text_probe",
                "source": "The grace of God is sufficient.",
                "reference_hi": None,
                "reference_provenance": None,
            }
        ],
    }
    path = tmp_path / "manifest.json"
    hindi.write_json(path, manifest)
    monkeypatch.setattr(hindi, "baseline_environment", lambda: {"test": "no_models"})
    monkeypatch.setattr("engines.model_paths.resolve_model_path", lambda *a, **k: "/cached/model")
    monkeypatch.setattr("tools.session_lifecycle.completion_metadata", lambda *a, **k: {"fixture": True})
    monkeypatch.delenv("STARK_EXPERIMENT_GEMMA_PREFIX_CACHE", raising=False)
    monkeypatch.delenv("STARK_EXPERIMENT_MLX_CACHE_MB", raising=False)
    return path, manifest


def transcribe_fixture(tmp_path, monkeypatch, manifest_path):
    vad = Mock()
    monkeypatch.setattr(
        "tools.vad_runtime.load_packaged_vad",
        lambda backend: (
            vad,
            (lambda *a, **k: [{"start": 0, "end": 8000}, {"start": 8000, "end": 16000}],),
            {"sha256": "fixture_vad", "source": "installed_package"},
        ),
    )
    stt = Mock()
    stt.transcribe.side_effect = [STTResult("God loves us.", 12), STTResult("He is not guilty.", 13)]
    factory = Mock(return_value=stt)
    monkeypatch.setattr("engines.parakeet_mlx_engine.ParakeetMLXEngine", factory)
    output = tmp_path / "transcripts"
    hindi.transcribe(SimpleNamespace(manifest=manifest_path, data_root=tmp_path, output=output))
    stt.load.assert_called_once()
    stt.unload.assert_called_once()
    vad.reset_states.assert_called_once()
    return output


def test_actual_offline_stages_preserve_source_bounds_and_identical_hindi_inputs(
    tmp_path, monkeypatch, manifest_fixture
):
    path, _ = manifest_fixture
    transcripts = transcribe_fixture(tmp_path, monkeypatch, path)
    rows = hindi.load_transcripts(transcripts, path)
    assert [(row["sample_start"], row["sample_end"]) for row in rows] == [(0, 8000), (8000, 16000)]
    assert all(row["sample_rate"] == 16000 and row["reference_hi"] is None for row in rows)
    expected_pcm = (np.arange(8000) % 1000).astype(np.float32) / 32768
    assert rows[0]["segment_float32_sha256"] == hashlib.sha256(expected_pcm.astype("<f4").tobytes()).hexdigest()

    engine = Mock()
    engine.translate.return_value = TranslationResult("परमेश्वर की grace", 15, generated_tokens=8, finish_reason="stop")
    factory = Mock(return_value=engine)
    monkeypatch.setattr("engines.mlx_engine.MLXGemmaEngine", factory)
    outputs = []
    for size in ("e4b", "e2b"):
        output = tmp_path / size
        hindi.translate(SimpleNamespace(manifest=path, transcripts=transcripts, output=output, size=size, runs=2))
        outputs.append(output)
    assert all(call.kwargs == {"source_lang": "en", "target_lang": "hi"} for call in engine.translate.call_args_list)
    assert engine.translate.call_count == 14  # one warmup + six actual calls per model
    assert all(call.kwargs["use_prompt_cache"] is False for call in factory.call_args_list)
    assert [call.args[0] for call in engine.translate.call_args_list[:7]] == [
        call.args[0] for call in engine.translate.call_args_list[7:]
    ]
    summary = hindi.report(
        SimpleNamespace(manifest=path, transcripts=transcripts, translations=outputs, output=tmp_path / "report.json")
    )
    assert summary["quality_gate"] == "pending_references_and_bilingual_review"
    assert summary["live_readiness"] == summary["qlora_decision"] == "not_assessed"
    for model in summary["models"]:
        assert model["generated_observations"] == 6
        assert model["references_available"] == 0
        assert model["with_latin_fragments"] == 6  # availability diagnostic, never an accuracy score
        assert model["translation_latency_by_input_kind_ms"]["church_audio_utterance"]["n"] == 4
        assert model["translation_latency_by_input_kind_ms"]["theological_text_probe"]["n"] == 2

    # Even a newly hashed result cannot omit an input and claim full coverage.
    rows_path = outputs[0] / "rows.jsonl"
    rows_path.write_text("\n".join(rows_path.read_text().splitlines()[:-1]) + "\n")
    metadata_path = outputs[0] / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["rows_sha256"] = hindi.digest(rows_path)
    hindi.write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="coverage"):
        hindi.report(
            SimpleNamespace(
                manifest=path, transcripts=transcripts, translations=outputs, output=tmp_path / "incomplete.json"
            )
        )


def test_changed_audio_and_missing_reference_provenance_fail_before_model_load(tmp_path, manifest_fixture):
    path, manifest = manifest_fixture
    manifest["text_probes"][0]["reference_hi"] = "परमेश्वर"
    with pytest.raises(ValueError, match="reference provenance"):
        hindi.validate_manifest(manifest)
    manifest["text_probes"][0]["reference_hi"] = None
    (tmp_path / "sermon.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        hindi.validate_manifest(manifest, tmp_path)
    assert not (tmp_path / "output").exists()


def test_changed_transcripts_rejected_and_existing_run_never_overwritten(tmp_path, monkeypatch, manifest_fixture):
    path, _ = manifest_fixture
    output = transcribe_fixture(tmp_path, monkeypatch, path)
    original = (output / "metadata.json").read_bytes()
    with pytest.raises(FileExistsError):
        hindi.reserve_run(output, path, "english_transcription")
    assert (output / "metadata.json").read_bytes() == original
    with (output / "rows.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="unchanged"):
        hindi.load_transcripts(output, path)


def test_failed_generation_persists_partial_evidence_but_never_completes(tmp_path, monkeypatch, manifest_fixture):
    path, _ = manifest_fixture
    transcripts = transcribe_fixture(tmp_path, monkeypatch, path)
    engine = Mock()
    engine.translate.side_effect = [
        TranslationResult("गर्म", 1),
        TranslationResult("परमेश्वर", 3),
        RuntimeError("decode failed"),
    ]
    monkeypatch.setattr("engines.mlx_engine.MLXGemmaEngine", Mock(return_value=engine))
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="decode failed"):
        hindi.translate(SimpleNamespace(manifest=path, transcripts=transcripts, output=output, size="e4b", runs=1))
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["status"] == "failed" and metadata["rows_sha256"]
    assert len((output / "rows.jsonl").read_text().splitlines()) == 1
    engine.unload.assert_called_once()
    with pytest.raises(ValueError, match="completed"):
        hindi.report(
            SimpleNamespace(
                manifest=path, transcripts=transcripts, translations=[output], output=tmp_path / "report.json"
            )
        )


def test_unload_failure_cannot_leave_completed_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.session_lifecycle.completion_metadata", lambda *a: {})
    engine = Mock()
    engine.unload.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(RuntimeError, match="cleanup failed"):
        hindi.finish_run(tmp_path, {"status": "completed"}, engine, {})
    assert json.loads((tmp_path / "metadata.json").read_text())["status"] == "failed"


def test_nonbaseline_prefix_configuration_is_explicitly_rejected(monkeypatch):
    monkeypatch.setenv("STARK_EXPERIMENT_GEMMA_PREFIX_CACHE", "true")
    with pytest.raises(ValueError, match="PREFIX_CACHE"):
        hindi.configure_offline()


def test_segmentation_rejects_overlap_instead_of_duplicating_source_audio():
    with pytest.raises(ValueError, match="overlapping"):
        list(
            hindi.segment_audio(
                np.ones(20), Mock(), lambda *a, **k: [{"start": 0, "end": 15}, {"start": 14, "end": 20}]
            )
        )
