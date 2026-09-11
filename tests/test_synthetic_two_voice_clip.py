"""Synthetic fixture generation uses fake TTS only, with no playback or models."""

import hashlib
import json
import sys
import wave
from unittest.mock import MagicMock

import numpy as np
import pytest

from engines.base import TTSResult
from tools import synthetic_two_voice_clip as clip


@pytest.fixture
def fake_engine(monkeypatch):
    engine = MagicMock()

    def synthesize(text, *, language):
        frequency = 220 if language == "a" else 440
        audio = (0.2 * np.sin(2 * np.pi * frequency * np.arange(2205) / 22050)).astype(np.float32)
        return TTSResult(audio=audio, sample_rate=22050, latency_ms=0, text=text)

    engine.synthesize.side_effect = synthesize
    engine.play.side_effect = AssertionError("Playback forbidden")
    constructor = MagicMock(return_value=engine)
    monkeypatch.setattr(clip, "PiperTTSEngine", constructor)
    # conftest mocks scipy; these tests exercise the installed real resampler.
    # SciPy's array dispatch inspects torch.Tensor without importing torch.
    monkeypatch.setattr(sys.modules["torch"], "Tensor", type("FakeTensor", (), {}))
    for name in list(sys.modules):
        if name == "scipy" or name.startswith("scipy."):
            monkeypatch.delitem(sys.modules, name)
    return constructor, engine


@pytest.mark.parametrize("fallback", [False, True])
def test_clip_audio_recipe_and_no_playback(tmp_path, monkeypatch, fake_engine, capsys, fallback):
    constructor, engine = fake_engine
    if fallback:
        monkeypatch.setitem(sys.modules, "scipy.signal", None)
    output = tmp_path / "audio" / "two.wav"
    recipe_path = tmp_path / "recipe.json"
    clip.main(["--output", str(output), "--recipe", str(recipe_path), "--gap-s", "0.25", "--lead-s", "0.5"])
    constructor.assert_called_once_with(voices={"a": "en_US-lessac-high", "b": "es_MX-claude-high"})
    engine.load.assert_called_once_with()
    engine.unload.assert_called_once_with()
    engine.play.assert_not_called()
    recipe = json.loads(recipe_path.read_text())
    assert recipe["output"] == str(output.resolve())
    assert recipe["sample_rate"] == 16000
    assert recipe["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert recipe["accent_caveat"] == (
        "voice b is es_MX-claude-high reading English; natural two-speaker labels remain pending"
    )
    segments = recipe["segments"]
    assert [segment["index"] for segment in segments] == list(range(1, 13))
    assert [segment["voice"] for segment in segments] == ["a", "b"] * 6
    assert [segment["text"] for segment in segments] == list(clip.SENTENCES)
    assert segments[0]["start_s"] == 0.5
    with wave.open(str(output), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) == (16000, 1, 2, "NONE")
        assert recipe["duration_s"] == wav.getnframes() / 16000 == segments[-1]["end_s"]
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    assert np.max(np.abs(pcm)) / 32767 == pytest.approx(0.9, abs=1 / 32767)
    assert not pcm[:8000].any()
    for index, segment in enumerate(segments):
        assert segment["end_s"] - segment["start_s"] == pytest.approx(0.1)
        call = engine.synthesize.call_args_list[index]
        assert call.args == (segment["text"],)
        assert call.kwargs == {"language": segment["voice"]}
        if index:
            previous = segments[index - 1]
            assert segment["start_s"] - previous["end_s"] == pytest.approx(0.25)
            assert not pcm[round(previous["end_s"] * 16000) : round(segment["start_s"] * 16000)].any()
    assert len(capsys.readouterr().out.splitlines()) == 1


def test_emit_manifest_preserves_church_entry_and_resolves_wavs(tmp_path, fake_engine):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    church = source_dir / clip.CHURCH_CLIP
    church.write_bytes(b"fixture church WAV bytes")
    entry = {
        "path": church.name,
        "sha256": hashlib.sha256(church.read_bytes()).hexdigest(),
        "duration": 150.0,
        "lang": "en",
        "offset_s": 1290,
        "rms": 0.1376,
        "extra": "preserve future fields too",
    }
    source_manifest = source_dir / "manifest.json"
    source_manifest.write_text(json.dumps({"clips": [{"path": "unrelated.wav"}, entry]}))
    original = source_manifest.read_bytes()
    output = tmp_path / "audio" / "synthetic.wav"
    manifest = tmp_path / "elsewhere" / "manifest.json"
    clip.main(
        [
            "--output",
            str(output),
            "--recipe",
            str(tmp_path / "recipe.json"),
            "--emit-manifest",
            str(manifest),
            "--source-manifest",
            str(source_manifest),
        ]
    )
    clips = json.loads(manifest.read_text())["clips"]
    assert len(clips) == 2
    assert clips[1] == entry
    assert source_manifest.read_bytes() == original
    for item in clips:
        wav = (manifest.parent / item["path"]).resolve()
        assert hashlib.sha256(wav.read_bytes()).hexdigest() == item["sha256"]
    assert clips[0]["lang"] == "en"
    assert clips[0]["offset_s"] == 0
    assert 0 < clips[0]["rms"] < 0.9
    with wave.open(str(output)) as wav:
        assert clips[0]["duration"] == wav.getnframes() / wav.getframerate()


@pytest.mark.parametrize("option,value", [("--gap-s", "-1"), ("--lead-s", "nan"), ("--gap-s", "inf")])
def test_bad_timing_rejected_before_loading(tmp_path, fake_engine, option, value):
    with pytest.raises(SystemExit) as exc:
        clip.main(["--output", str(tmp_path / "clip.wav"), "--recipe", str(tmp_path / "r.json"), option, value])
    assert exc.value.code == 2
    fake_engine[0].assert_not_called()


def test_cannot_overwrite_source_manifest(tmp_path, fake_engine):
    source = tmp_path / "manifest.json"
    source.write_text('{"clips": []}')
    with pytest.raises(SystemExit) as exc:
        clip.main(
            [
                "--output",
                str(tmp_path / "clip.wav"),
                "--recipe",
                str(tmp_path / "r.json"),
                "--emit-manifest",
                str(source),
                "--source-manifest",
                str(source),
            ]
        )
    assert exc.value.code == 2
    assert source.read_text() == '{"clips": []}'
    fake_engine[0].assert_not_called()


def test_stereo_downmix():
    samples = np.array([[0.2, 0.4], [-0.4, -0.2]], dtype=np.float32)
    np.testing.assert_allclose(clip.resample_mono(samples, 16000), [0.3, -0.3])
