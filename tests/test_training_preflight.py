"""CPU-only parser, corpus, safetensors and recipe gates; no ML stack is imported."""

from __future__ import annotations

import ast
import json
import math
import os
import struct
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import training_preflight as preflight
from training.candidates.build_jacobo_candidates import audit_rows, build_rows
from training.train_gemma4_cpo import load_triples, train

ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def write_safetensors(path, shapes):
    offset, header = 0, {}
    for key, shape in shapes.items():
        length = 4
        for dim in shape:
            length *= dim
        header[key] = {"dtype": "F32", "shape": list(shape), "data_offsets": [offset, offset + length]}
        offset += length
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


@pytest.fixture
def holdout(tmp_path):
    return write_jsonl(tmp_path / "holdout.jsonl", [{"en": "A withheld fixture", "es": "Una reserva"}])


@pytest.fixture
def whisper(tmp_path):
    model = {
        "model_type": "whisper",
        "d_model": 8,
        "encoder_layers": 2,
        "decoder_layers": 1,
        "encoder_ffn_dim": 12,
        "decoder_ffn_dim": 16,
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(model))
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "openai/whisper-large-v3-turbo",
                "r": 32,
                "lora_alpha": 64,
                "target_modules": ["q_proj", "v_proj"],
                "bias": "none",
            }
        )
    )
    shapes = preflight.expected_whisper_shapes(model, {"q_proj", "v_proj"}, 32, False)
    write_safetensors(adapter / "adapter_model.safetensors", shapes)
    argv = [
        "--dataset",
        str(tmp_path / "w17"),
        "--init-from",
        str(adapter),
        "--model-config",
        str(config),
        "--target-modules",
        "q_proj",
        "v_proj",
        "k_proj",
        "out_proj",
        "fc1",
        "fc2",
        "--use-dora",
        "--allow-target-expansion",
        "--require-replay",
    ]
    return argv, model, adapter


@pytest.fixture
def w17_data(tmp_path):
    dataset, audio, deepgram = (tmp_path / name for name in ("dataset", "audio", "deepgram"))
    for directory in (dataset, audio, deepgram):
        directory.mkdir()
    source = "Fixture_Service_(3_1_26)_trainonly"
    with wave.open(str(audio / f"{source}.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(bytes(32000))
    (deepgram / f"{source}.deepgram.json").write_text(
        json.dumps(
            {
                "words": [{"word": "A realistic independent training fixture sentence", "start": 0.1, "end": 0.9}],
            }
        )
    )
    chunks = [
        {"source": source, "start": 0, "end": 1, "split": "train"},
        {"source": "heldout", "start": 0, "end": 1, "split": "eval"},
    ]
    (dataset / "chunks.json").write_text(json.dumps(chunks))
    return dataset, audio, deepgram


def test_real_parser_contract_and_unknown_flag():
    args = preflight.parse_trainer_args("train_gemma4.py", ["--output", "fixture", "--no-packing"])
    assert args.base == "unsloth/gemma-4-E4B-it"
    assert args.packing is False
    with pytest.raises(SystemExit):
        preflight.parse_trainer_args("train_gemma4.py", ["--output", "fixture", "--invented-flag"])


def test_source_whisper_adapter_coverage_and_new_modules(whisper):
    argv, _, _ = whisper
    result = preflight.validate_whisper_config(preflight.parse_trainer_args("train_whisper.py", argv))
    assert result["fresh_targets"] == ["fc1", "fc2", "k_proj", "out_proj"]
    assert result["source_tensor_count"] == 16
    assert result["fresh_dora_magnitudes"] is True


def test_real_turbo_geometry_with_sparse_local_adapter_fixture(whisper):
    argv, _, adapter = whisper
    model = {
        "model_type": "whisper",
        "d_model": 1280,
        "encoder_layers": 32,
        "decoder_layers": 4,
        "encoder_ffn_dim": 5120,
        "decoder_ffn_dim": 5120,
    }
    Path(argv[argv.index("--model-config") + 1]).write_text(json.dumps(model))
    shapes = preflight.expected_whisper_shapes(model, {"q_proj", "v_proj"}, 32, False)
    offset, header = 0, {}
    for key, shape in shapes.items():
        length = 4 * shape[0] * shape[1]
        header[key] = {"dtype": "F32", "shape": list(shape), "data_offsets": [offset, offset + length]}
        offset += length
    encoded = json.dumps(header).encode()
    with (adapter / "adapter_model.safetensors").open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)) + encoded)
        stream.truncate(8 + len(encoded) + offset)
    result = preflight.validate_whisper_config(preflight.parse_trainer_args("train_whisper.py", argv))
    assert result["source_tensor_count"] == 160


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"target_modules": ["q_proj", "o_proj"]}, "Invalid Whisper targets"),
        ({"lora_r": 16}, "rank mismatch"),
        ({"lora_alpha": 32}, "alpha mismatch"),
        ({"allow_target_expansion": False}, "explicit --allow-target-expansion"),
        ({"replay_ratio": 0}, "positive ratio"),
        ({"model_config": "missing-config.json"}, "Missing model config"),
    ],
)
def test_whisper_config_rejects_incompatible_inputs(whisper, change, message):
    args = preflight.parse_trainer_args("train_whisper.py", whisper[0])
    for key, value in change.items():
        setattr(args, key, value)
    with pytest.raises(preflight.PreflightError, match=message):
        preflight.validate_whisper_config(args)


@pytest.mark.parametrize("corruption", ["shape", "missing", "extra", "truncated"])
def test_source_adapter_tensor_metadata_fails_closed(whisper, corruption):
    argv, model, adapter = whisper
    shapes = preflight.expected_whisper_shapes(model, {"q_proj", "v_proj"}, 32, False)
    first = next(iter(shapes))
    if corruption == "shape":
        shapes[first] = (16, 8)
    elif corruption == "missing":
        del shapes[first]
    elif corruption == "extra":
        shapes["unknown.lora_A.weight"] = (32, 8)
    weights = adapter / "adapter_model.safetensors"
    write_safetensors(weights, shapes)
    if corruption == "truncated":
        weights.write_bytes(weights.read_bytes()[:-4])
    with pytest.raises(preflight.PreflightError):
        preflight.validate_whisper_config(preflight.parse_trainer_args("train_whisper.py", argv))


def test_expansion_only_allows_explicit_new_parameters():
    prefix = "base_model.model.model.encoder.layers.0."
    preflight.check_adapter_load(
        [
            prefix + "self_attn.k_proj.lora_A.default.weight",
            prefix + "self_attn.q_proj.lora_magnitude_vector.default.weight",
        ],
        [],
        [prefix + "self_attn.q_proj.lora_A.default.weight"],
        fresh_targets={"k_proj"},
        fresh_dora=True,
    )
    for missing, unexpected in (([prefix + "self_attn.q_proj.lora_A.weight"], []), ([], ["unconsumed.weight"])):
        with pytest.raises(preflight.PreflightError):
            preflight.check_adapter_load(
                missing, unexpected, ["some.weight"], fresh_targets={"k_proj"}, fresh_dora=True
            )


def test_new_dora_magnitudes_use_loaded_directions_and_preserve_effective_weight():
    # Base=(3,4), loaded delta=(2,0). A stale norm(base)=5 would rescale (5,4).
    class Magnitude:
        weight = 5.0

        def update_layer(self, *, base_layer, lora_A, lora_B, scaling):
            self.weight = math.sqrt(sum((w + scaling * lora_B[0] * a) ** 2 for w, a in zip(base_layer, lora_A)))

    magnitude = Magnitude()
    stem = "base_model.model.model.encoder.layers.0.self_attn.q_proj"
    layer = SimpleNamespace(
        lora_magnitude_vector={"default": magnitude},
        get_base_layer=lambda: [3.0, 4.0],
        lora_A={"default": SimpleNamespace(weight=[1.0, 0.0])},
        lora_B={"default": SimpleNamespace(weight=[2.0])},
        scaling={"default": 1.0},
    )
    model = SimpleNamespace(named_modules=lambda: [(stem, layer)])
    count = preflight.initialize_new_dora_magnitudes(
        model, [stem + ".lora_A.default.weight", stem + ".lora_B.default.weight"]
    )
    assert count == 1
    assert magnitude.weight == pytest.approx(math.sqrt(41))
    assert [magnitude.weight / math.sqrt(41) * w for w in [5.0, 4.0]] == pytest.approx([5.0, 4.0])
    layer.lora_magnitude_vector = {"default": object()}
    with pytest.raises(preflight.PreflightError, match="cannot reinitialize"):
        preflight.initialize_new_dora_magnitudes(model, [stem + ".lora_A.weight"])


@pytest.mark.parametrize("environment", ["HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME"])
def test_model_config_resolves_custom_huggingface_cache(tmp_path, monkeypatch, environment):
    for name in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(environment, str(tmp_path))
    hub = tmp_path / "hub" if environment == "HF_HOME" else tmp_path
    if environment == "XDG_CACHE_HOME":
        hub = tmp_path / "huggingface/hub"
    cached = hub / "models--openai--whisper-large-v3-turbo"
    (cached / "refs").mkdir(parents=True)
    (cached / "refs/main").write_text("fixture-revision")
    config = cached / "snapshots/fixture-revision/config.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    assert preflight.resolve_model_config("openai/whisper-large-v3-turbo", None) == config


def test_invalid_direct_whisper_cli_fails_before_torch_import():
    result = subprocess.run(
        [sys.executable, "-S", "training/train_whisper.py", "--target-modules", "o_proj"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Invalid Whisper targets" in result.stderr
    assert "No module named 'torch'" not in result.stderr


def test_required_replay_cannot_fall_back_to_hard_only():
    # Extract the CPU-independent helper without importing the GPU trainer.
    tree = ast.parse((ROOT / "training/train_whisper.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_mixed_dataset")
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<replay helper>", "exec"), scope)
    with pytest.raises(ValueError, match="hard-only"):
        scope["prepare_mixed_dataset"](None, None, 0, require_replay=True)


def test_w17_data_checks_real_wave_and_deepgram_and_excludes_eval(w17_data, holdout):
    dataset, audio, deepgram = w17_data
    report, selected = preflight.validate_w17_data(
        dataset / "chunks.json", deepgram, audio, preflight.holdout_index([holdout])
    )
    assert report["training_chunks"] == 1
    assert report["excluded_evaluation_chunks"] == 1
    assert selected[0]["split"] == "train"


@pytest.mark.parametrize("corruption", ["missing_audio", "empty_deepgram", "timing", "post_cutoff", "fresh_eval"])
def test_w17_rejects_unsafe_or_missing_corpus(w17_data, holdout, corruption):
    dataset, audio, deepgram = w17_data
    chunks_path = dataset / "chunks.json"
    chunks = json.loads(chunks_path.read_text())
    if corruption == "missing_audio":
        next(audio.iterdir()).unlink()
    elif corruption == "empty_deepgram":
        next(deepgram.iterdir()).write_text('{"words": []}')
    elif corruption == "timing":
        chunks[0]["end"] = 100
    elif corruption == "post_cutoff":
        chunks[0]["source"] = "Fixture_(3_15_26)_later"
    else:
        chunks[0]["source"] = "4Es8SrciqV0"
    chunks_path.write_text(json.dumps(chunks))
    with pytest.raises(preflight.PreflightError):
        preflight.validate_w17_data(chunks_path, deepgram, audio, preflight.holdout_index([holdout]))


@pytest.mark.parametrize(
    "row",
    [
        {"en": "A withheld fixture", "es": "Other"},
        {"en": "Unique", "es": "Other", "direction": "en2fr"},
        {"en": "Unique", "es": "Other", "split": "eval"},
        {"en": "Unique", "es": "Other", "approved_for_training": False},
        {"en": "Unique", "es": "Other", "chunk_source": "Gospel_(3_15_26)_late"},
        {"en": "Unique", "es": "Other", "en_source": "niv"},
        {"en": "Unique", "es": ""},
    ],
)
def test_gemma_rejects_invalid_or_heldout_rows(tmp_path, holdout, row):
    data = write_jsonl(tmp_path / "data.jsonl", [row])
    args = preflight.parse_trainer_args("train_gemma4.py", ["--output", "fixture", "--train-data", str(data)])
    with pytest.raises(preflight.PreflightError):
        preflight.validate_gemma(args, preflight.holdout_index([holdout]))


def test_gemma_checks_complete_corpus_before_sampling(tmp_path, holdout):
    data = write_jsonl(
        tmp_path / "data.jsonl",
        [
            {"en": "An independent fixture sentence", "es": "Una oración independiente", "direction": "es2en"},
            {"en": "A withheld fixture", "es": "Otra"},
        ],
    )
    args = preflight.parse_trainer_args(
        "train_gemma4.py", ["--output", "fixture", "--train-data", str(data), "--max-pairs", "1"]
    )
    with pytest.raises(preflight.PreflightError, match="Holdout text"):
        preflight.validate_gemma(args, preflight.holdout_index([holdout]))


def test_explicit_missing_corpus_is_never_silently_dropped(tmp_path, holdout):
    args = preflight.parse_trainer_args(
        "train_gemma4.py", ["--output", "fixture", "--train-data", str(tmp_path / "missing")]
    )
    with pytest.raises(preflight.PreflightError, match="Missing input file"):
        preflight.validate_gemma(args, preflight.holdout_index([holdout]))


def cpu_interpreter(tmp_path):
    path = tmp_path / "python-cpu-only"
    path.write_text(f'#!/bin/sh\nexec "{sys.executable}" -S "$@"\n')
    path.chmod(0o755)
    return str(path)


def test_gemma_recipe_dry_run_uses_actual_data_without_ml(tmp_path, holdout):
    data = write_jsonl(
        tmp_path / "training_v2.jsonl",
        [
            {"en": "An independent fixture sentence", "es": "Una oración independiente", "direction": "es2en"},
        ],
    )
    env = {
        **os.environ,
        "STARK_TRAINING_PYTHON": cpu_interpreter(tmp_path),
        "STARK_TRAINING_HOLDOUT": str(holdout),
        "STARK_GEMMA4_TRAIN": str(data),
    }
    result = subprocess.run(
        ["bash", "training/run_gemma4_e4b_domain_sft.sh", "--dry-run"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(result.stdout)
    assert report["status"] == "passed" and report["training_started"] is False
    assert report["data"]["directions_before_sampling"] == {"es2en": 1}
    assert report["data"]["corpora"][0]["sha256"] == preflight.sha256(data)
    env["STARK_GEMMA4_TRAIN"] = str(tmp_path / "explicit_missing.jsonl")
    failed = subprocess.run(
        ["bash", "training/run_gemma4_e4b_domain_sft.sh", "--dry-run"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed.returncode == 2 and "Missing input file" in failed.stdout


def test_w17_recipe_dry_run_never_runs_mining_or_gpu(whisper, w17_data, holdout, tmp_path):
    argv, _, adapter = whisper
    dataset, audio, deepgram = w17_data
    env = {
        **os.environ,
        "STARK_TRAINING_PYTHON": cpu_interpreter(tmp_path),
        "STARK_TRAINING_HOLDOUT": str(holdout),
        "STARK_WHISPER_DATASET": str(dataset),
        "STARK_W16_ADAPTER": str(adapter),
        "STARK_WHISPER_MODEL_CONFIG": argv[argv.index("--model-config") + 1],
        "STARK_AUDIO_DIR": str(audio),
        "STARK_DEEPGRAM_DIR": str(deepgram),
    }
    before = set(tmp_path.rglob("*"))
    result = subprocess.run(
        ["bash", "training/run_w17_curriculum.sh", "--dry-run"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert set(tmp_path.rglob("*")) == before
    report = json.loads(result.stdout)
    assert report["status"] == "passed" and report["training_started"] is False
    assert report["data"]["training_chunks"] == 1
    assert report["cuda_gate"] == "pending"


def test_candidate_pool_and_overlap_audit(holdout):
    rows = build_rows()
    report = audit_rows(rows, preflight.holdout_index([holdout]))
    assert len(rows) == 60
    assert report["unique_core_sentences"] == 20
    assert report["sense_counts"] == {"person_son_of_zebedee": 30, "epistle_title": 30}
    assert all(row["approved_for_training"] is False for row in rows)
    index = {"texts": {preflight.normalized(rows[0]["source_text"])}, "ids": set()}
    with pytest.raises(preflight.PreflightError, match="Holdout text"):
        audit_rows(rows, index)


def test_missing_holdout_remains_explicit_pending_for_candidates(tmp_path):
    missing = tmp_path / "missing_holdout.jsonl"
    with pytest.raises(preflight.PreflightError, match="Missing required holdout"):
        preflight.holdout_index([missing])
    report = audit_rows(build_rows(), preflight.holdout_index([missing], allow_missing=True))
    assert report["holdout_gate"] == "pending_missing_inputs"
    assert report["required_holdouts_missing"] == [str(missing)]


def test_checked_in_candidates_match_generator_and_manifest():
    candidate_path = ROOT / "training/candidates/jacobo_candidates_20260910.jsonl"
    manifest = json.loads(candidate_path.with_suffix(".manifest.json").read_text())
    assert preflight.json_rows(candidate_path) == build_rows()
    assert manifest["candidate_file_sha256"] == preflight.sha256(candidate_path)
    assert manifest["generator_sha256"] == preflight.sha256(ROOT / "training/candidates/build_jacobo_candidates.py")


def test_cpo_rejects_unapproved_before_gpu_import_and_preserves_legacy(tmp_path):
    path = write_jsonl(tmp_path / "candidates.jsonl", build_rows())
    with pytest.raises(ValueError, match="Unapproved preference"):
        train(type("Args", (), {"triples": path})())
    legacy = {"prompt": "p", "chosen": "c", "rejected": "r", "margin": 0.1}
    write_jsonl(path, [legacy])
    assert load_triples(path) == [legacy]
