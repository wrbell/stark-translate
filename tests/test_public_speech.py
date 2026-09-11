import hashlib
import io
import sys
import tarfile
import wave

import pytest

from tools.public_speech import copy_selected_audio, download, parse_tsv, select


def row(identity, samples=16000, gender="FEMALE"):
    return {
        "sentence_id": identity,
        "filename": f"{identity}.wav",
        "num_samples": samples,
        "gender": gender,
    }


def test_parser_preserves_reference_and_rejects_bad_rows():
    parsed = parse_tsv('1\t2.wav\tSí, "Señor".\tsí señor\tx\t16000\tFEMALE\n')
    assert parsed[0]["reference"] == 'Sí, "Señor".'
    assert parsed[0]["upstream_normalized_reference"] == "sí señor"
    with pytest.raises(ValueError, match="identity"):
        parse_tsv("1\t../2.wav\tx\tx\tx\t16000\tFEMALE")
    with pytest.raises(ValueError, match="Duplicate"):
        parse_tsv("1\t2.wav\tx\tx\tx\t16000\tFEMALE\n" * 2)


def test_selection_stable_balanced_unique_sentences():
    rows = [row(i, samples=(1 if i % 2 else 12) * 16000) for i in range(1, 31)]
    rows.append({**rows[0], "filename": "999.wav"})
    selected = select(rows, 10)
    assert selected == select(list(reversed(rows)), 10)
    assert len({r["sentence_id"] for r in selected}) == 10
    assert sum(r["num_samples"] == 16000 for r in selected) == 5
    with pytest.raises(ValueError, match="distinct"):
        select(rows, 31)


def test_existing_download_checks_integrity_without_network(tmp_path):
    revision = "a" * 40
    target = tmp_path / revision / "data/en_us/dev.tsv"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"original")
    target.with_name(target.name + ".sha256").write_text(hashlib.sha256(b"original").hexdigest())
    assert download(revision, "data/en_us/dev.tsv", tmp_path) == target
    target.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        download(revision, "data/en_us/dev.tsv", tmp_path)


def test_archive_copies_original_bytes_and_validates_samples(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "soundfile", raising=False)
    pytest.importorskip("soundfile")
    audio = io.BytesIO()
    with wave.open(audio, "wb") as stream:
        stream.setparams((1, 2, 16000, 16000, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 16000)
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        info = tarfile.TarInfo("dev/1.wav")
        info.size = len(audio.getvalue())
        stream.addfile(info, io.BytesIO(audio.getvalue()))
    result = copy_selected_audio(archive, [row(1)], tmp_path / "out")
    assert result["1.wav"]["sha256"] == hashlib.sha256(audio.getvalue()).hexdigest()
    assert (tmp_path / "out/1.wav").read_bytes() == audio.getvalue()
    with pytest.raises(ValueError, match="shape/rate"):
        copy_selected_audio(archive, [row(1, samples=123)], tmp_path / "bad")


def test_download_uses_pinned_dataset_client_and_rejects_paths(tmp_path, monkeypatch):
    from types import SimpleNamespace

    source = tmp_path / "downloaded"
    source.write_bytes(b"exact dataset bytes")
    calls = []

    def download_file(**kwargs):
        calls.append(kwargs)
        return str(source)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download_file))
    cache = tmp_path / "cache"
    revision = "b" * 40
    result = download(revision, "data/es_419/audio/test.tar.gz", cache)
    assert result.read_bytes() == source.read_bytes()
    assert calls == [
        {
            "repo_id": "google/fleurs",
            "repo_type": "dataset",
            "revision": revision,
            "filename": "data/es_419/audio/test.tar.gz",
            "cache_dir": cache / ".hf-cache",
        }
    ]
    assert (
        result.with_name(result.name + ".sha256").read_text().strip() == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    for relative in ("../../outside", "/tmp/file", "https://example.com", "data/en_us/train.tsv"):
        with pytest.raises(ValueError, match="source paths"):
            download(revision, relative, cache)
    with pytest.raises(ValueError, match="immutable"):
        download("main", "data/en_us/dev.tsv", cache)
    assert len(calls) == 1
