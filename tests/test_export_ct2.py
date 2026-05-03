"""Tests for training/export_ct2.py — Whisper LoRA -> CT2 merger.

All heavy deps (peft, transformers, ctranslate2 binary, faster_whisper) are
mocked. These tests assert the export script's contract: argument parsing,
correct sequencing of merge / convert / sanity / manifest steps, and graceful
failure when a step aborts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Inject `peft` mock before training/export_ct2 is imported indirectly. The
# project's conftest doesn't mock peft (it's only used by training scripts).
sys.modules.setdefault("peft", MagicMock())

# Now safe to import — module-level imports in export_ct2.py are stdlib-only.
from training import export_ct2


def _make_adapter_dir(tmp_path: Path) -> Path:
    adapter = tmp_path / "W16_mixed_w7"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00" * 1024)
    (adapter / "adapter_config.json").write_text("{}")
    return adapter


def test_sha256_file_deterministic(tmp_path: Path):
    p = tmp_path / "blob"
    p.write_bytes(b"hello world")
    sha = export_ct2.sha256_file(p)
    assert sha == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_directory_size_bytes(tmp_path: Path):
    (tmp_path / "a.bin").write_bytes(b"x" * 1000)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * 500)
    assert export_ct2.directory_size_bytes(tmp_path) == 1500


def test_run_ct2_converter_filters_missing_copy_files(tmp_path: Path):
    """Only files that actually exist should be passed to --copy_files."""
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "tokenizer.json").write_text("{}")
    (merged / "preprocessor_config.json").write_text("{}")
    # Intentionally missing: tokenizer_config.json, generation_config.json, etc.

    out = tmp_path / "ct2"
    with (
        patch("shutil.which", return_value="/fake/ct2-transformers-converter"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        export_ct2.run_ct2_converter(merged, out, "int8_float16", export_ct2.WHISPER_COPY_FILES)

    assert mock_run.called
    cmd = mock_run.call_args[0][0]
    assert "--quantization" in cmd
    assert "int8_float16" in cmd
    if "--copy_files" in cmd:
        idx = cmd.index("--copy_files")
        passed = set(cmd[idx + 1 :]) - {"--force"}
        assert passed == {"tokenizer.json", "preprocessor_config.json"}


def test_run_ct2_converter_missing_binary(tmp_path: Path):
    merged = tmp_path / "merged"
    merged.mkdir()
    out = tmp_path / "ct2"
    with patch("shutil.which", return_value=None), pytest.raises(SystemExit, match="not on PATH"):
        export_ct2.run_ct2_converter(merged, out, "int8", [])


def test_write_manifest_records_provenance(tmp_path: Path):
    adapter = _make_adapter_dir(tmp_path)
    out = tmp_path / "ct2_out"
    out.mkdir()
    (out / "model.bin").write_bytes(b"\x00" * 2048)

    sanity = {"wer": 0.05, "n_clips": 5, "per_clip": []}
    manifest_path = export_ct2.write_manifest(out, "openai/whisper-large-v3-turbo", adapter, "int8_float16", sanity)
    data = json.loads(manifest_path.read_text())
    assert data["base_model_id"] == "openai/whisper-large-v3-turbo"
    assert data["ct2_quantization"] == "int8_float16"
    assert data["sanity"] == sanity
    assert data["adapter_sha256"] is not None
    assert data["ct2_total_bytes"] >= 2048


def test_main_invokes_pipeline_in_order(tmp_path: Path, monkeypatch):
    adapter = _make_adapter_dir(tmp_path)
    out = tmp_path / "ct2_out"
    intermediate = tmp_path / "merged_intermediate"

    call_order: list[str] = []

    def fake_merge(base_id, adapter_dir, output_dir):
        call_order.append("merge")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "tokenizer.json").write_text("{}")

    def fake_convert(merged_dir, output_dir, quantization, copy_files, force=True):
        call_order.append("convert")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "model.bin").write_bytes(b"\x00" * 1024)
        (output_dir / "config.json").write_text("{}")

    def fake_sanity(ct2_dir, canary_clips, project_root, quantization, wer_drift_max):
        call_order.append("sanity")
        return {"wer": 0.05, "n_clips": 5, "per_clip": []}

    monkeypatch.setattr(export_ct2, "merge_adapter_to_bf16", fake_merge)
    monkeypatch.setattr(export_ct2, "run_ct2_converter", fake_convert)
    monkeypatch.setattr(export_ct2, "sanity_test_ct2", fake_sanity)

    rc = export_ct2.main(
        [
            "--adapter",
            str(adapter),
            "--base",
            "openai/whisper-large-v3-turbo",
            "--output",
            str(out),
            "--quantization",
            "int8_float16",
            "--intermediate",
            str(intermediate),
            "--keep-intermediate",
        ]
    )
    assert rc == 0
    assert call_order == ["merge", "convert", "sanity"]
    assert (out / "export_manifest.json").exists()


def test_main_no_sanity_skips_sanity_step(tmp_path: Path, monkeypatch):
    adapter = _make_adapter_dir(tmp_path)
    out = tmp_path / "ct2_out"
    intermediate = tmp_path / "merged_intermediate"

    sanity_called = []

    def fake_sanity(*a, **k):
        sanity_called.append(True)
        return {}

    monkeypatch.setattr(export_ct2, "merge_adapter_to_bf16", lambda *a, **k: a[2].mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(
        export_ct2, "run_ct2_converter", lambda merged, output, *a, **k: output.mkdir(parents=True, exist_ok=True)
    )
    monkeypatch.setattr(export_ct2, "sanity_test_ct2", fake_sanity)

    rc = export_ct2.main(
        [
            "--adapter",
            str(adapter),
            "--output",
            str(out),
            "--no-sanity",
            "--intermediate",
            str(intermediate),
            "--keep-intermediate",
        ]
    )
    assert rc == 0
    assert sanity_called == []  # sanity skipped


def test_main_aborts_on_missing_adapter(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="adapter dir not found"):
        export_ct2.main(["--adapter", str(missing), "--output", str(out)])


def test_main_aborts_on_sanity_failure(tmp_path: Path, monkeypatch):
    adapter = _make_adapter_dir(tmp_path)
    out = tmp_path / "ct2_out"
    intermediate = tmp_path / "merged_intermediate"

    def failing_sanity(*a, **k):
        raise SystemExit("FAIL: sanity WER 0.85 exceeds threshold 0.30")

    monkeypatch.setattr(export_ct2, "merge_adapter_to_bf16", lambda *a, **k: a[2].mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(
        export_ct2, "run_ct2_converter", lambda merged, output, *a, **k: output.mkdir(parents=True, exist_ok=True)
    )
    monkeypatch.setattr(export_ct2, "sanity_test_ct2", failing_sanity)

    with pytest.raises(SystemExit, match="exceeds threshold"):
        export_ct2.main(
            [
                "--adapter",
                str(adapter),
                "--output",
                str(out),
                "--intermediate",
                str(intermediate),
                "--keep-intermediate",
            ]
        )
