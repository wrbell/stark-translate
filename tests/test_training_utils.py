"""Tests for pure data processing functions in training scripts."""

import os

import numpy as np

# ===================================================================
# create_multi_reference_pairs (training/prepare_bible_corpus.py)
# ===================================================================


class TestCreateMultiReferencePairs:
    def test_single_pair_passthrough(self):
        from training.prepare_bible_corpus import create_multi_reference_pairs

        pairs = [{"en": "In the beginning", "es": "En el principio", "verse_id": 1}]
        result = create_multi_reference_pairs(pairs)
        assert len(result) == 1
        assert result[0]["en"] == "In the beginning"

    def test_two_en_two_es(self):
        from training.prepare_bible_corpus import create_multi_reference_pairs

        pairs = [
            {"en": "In the beginning", "es": "En el principio", "verse_id": 1, "en_source": "kjv", "es_source": "rvr"},
            {
                "en": "At the start",
                "es": "Al comienzo",
                "verse_id": 1,
                "en_source": "web",
                "es_source": "sencillo",
            },
        ]
        result = create_multi_reference_pairs(pairs)
        # 2 EN x 2 ES = 4 combinations
        assert len(result) == 4

    def test_preserves_verse_id(self):
        from training.prepare_bible_corpus import create_multi_reference_pairs

        pairs = [{"en": "Hello", "es": "Hola", "verse_id": 42}]
        result = create_multi_reference_pairs(pairs)
        assert all(r["verse_id"] == 42 for r in result)

    def test_dedup_by_set(self):
        from training.prepare_bible_corpus import create_multi_reference_pairs

        # Same EN and ES text should only produce 1 pair
        pairs = [
            {"en": "Hello", "es": "Hola", "verse_id": 1, "en_source": "kjv", "es_source": "rvr"},
            {"en": "Hello", "es": "Hola", "verse_id": 1, "en_source": "asv", "es_source": "rvr"},
        ]
        result = create_multi_reference_pairs(pairs)
        assert len(result) == 1

    def test_empty_input(self):
        from training.prepare_bible_corpus import create_multi_reference_pairs

        result = create_multi_reference_pairs([])
        assert result == []


# ===================================================================
# export_training_jsonl (training/prepare_bible_corpus.py)
# ===================================================================


class TestExportTrainingJsonl:
    def test_creates_train_test_files(self, tmp_path):
        from training.prepare_bible_corpus import export_training_jsonl

        output = str(tmp_path / "aligned" / "pairs.jsonl")
        pairs = [{"en": f"Verse {i}", "es": f"Versículo {i}", "verse_id": i} for i in range(100)]
        train, test = export_training_jsonl(pairs, output, holdout_ratio=0.1)
        train_path = output.replace(".jsonl", "_train.jsonl")
        test_path = output.replace(".jsonl", "_test.jsonl")
        assert os.path.exists(train_path)
        assert os.path.exists(test_path)
        assert len(train) + len(test) == 100

    def test_deterministic_split(self, tmp_path):
        from training.prepare_bible_corpus import export_training_jsonl

        output = str(tmp_path / "aligned" / "pairs.jsonl")
        pairs = [{"en": f"V{i}", "es": f"V{i}", "verse_id": i} for i in range(50)]
        train1, test1 = export_training_jsonl(pairs, output, holdout_ratio=0.1)
        train2, test2 = export_training_jsonl(pairs, output, holdout_ratio=0.1)
        assert len(train1) == len(train2)
        assert len(test1) == len(test2)

    def test_genre_counts_printed(self, tmp_path, capsys):
        from training.prepare_bible_corpus import export_training_jsonl

        output = str(tmp_path / "aligned" / "pairs.jsonl")
        # verse_id 01001001 → book 1 (pentateuch), 40001001 → book 40 (gospels)
        pairs = [
            {"en": "Genesis text", "es": "Texto de Génesis", "verse_id": "01001001"},
            {"en": "Matthew text", "es": "Texto de Mateo", "verse_id": "40001001"},
        ]
        export_training_jsonl(pairs, output, holdout_ratio=1.0)
        captured = capsys.readouterr()
        assert "Holdout genre distribution" in captured.out


# ===================================================================
# final_quality_gate (training/preprocess_audio.py)
# ===================================================================


class TestFinalQualityGate:
    def test_too_short(self):
        from training.preprocess_audio import final_quality_gate

        # 0.5 seconds at 16kHz
        chunk = np.sin(np.linspace(0, 100, 8000)).astype(np.float32)
        ok, reason = final_quality_gate(chunk, sr=16000)
        assert ok is False
        assert reason == "duration_out_of_range"

    def test_too_long(self):
        from training.preprocess_audio import final_quality_gate

        # 31 seconds at 16kHz
        chunk = np.sin(np.linspace(0, 100, 16000 * 31)).astype(np.float32)
        ok, reason = final_quality_gate(chunk, sr=16000)
        assert ok is False
        assert reason == "duration_out_of_range"

    def test_too_much_silence(self):
        from training.preprocess_audio import final_quality_gate

        # 2 seconds, mostly silence with a tiny bit of signal
        sr = 16000
        chunk = np.zeros(sr * 2, dtype=np.float32)
        # Add just a tiny signal in a small portion
        chunk[:100] = 0.1
        ok, reason = final_quality_gate(chunk, sr=sr)
        assert ok is False
        assert reason == "too_much_silence"

    def test_low_snr(self):
        from training.preprocess_audio import final_quality_gate

        sr = 16000
        # 2 seconds of low-level noise (uniform energy, no speech/noise contrast)
        rng = np.random.RandomState(42)
        chunk = rng.randn(sr * 2).astype(np.float32) * 0.01
        ok, reason = final_quality_gate(chunk, sr=sr, min_snr=15.0)
        assert ok is False
        assert "low_snr" in reason or reason == "insufficient_audio"

    def test_good_chunk_passes(self):
        from training.preprocess_audio import final_quality_gate

        sr = 16000
        n_samples = sr * 3  # 3 seconds
        chunk = np.zeros(n_samples, dtype=np.float32)
        rng = np.random.RandomState(42)
        # Simulate speech-like signal: loud bursts + quiet noise floor
        frame_size = sr // 10  # 100ms frames
        for i in range(0, n_samples, frame_size):
            end = min(i + frame_size, n_samples)
            if rng.rand() > 0.2:
                # "Speech" frame — loud
                chunk[i:end] = rng.randn(end - i).astype(np.float32) * 0.3
            else:
                # "Noise" frame — quiet
                chunk[i:end] = rng.randn(end - i).astype(np.float32) * 0.001
        ok, reason = final_quality_gate(chunk, sr=sr)
        assert ok is True
        assert reason == "pass"

    def test_all_zero_fails(self):
        from training.preprocess_audio import final_quality_gate

        chunk = np.zeros(16000 * 2, dtype=np.float32)
        ok, reason = final_quality_gate(chunk, sr=16000)
        assert ok is False

    def test_boundary_1s_exactly(self):
        from training.preprocess_audio import final_quality_gate

        # Exactly 1.0 seconds — should NOT pass (< 1.0 check, but len/sr == 1.0)
        # The code says `duration < 1.0` so 1.0 should pass the duration check
        sr = 16000
        t = np.linspace(0, 1.0, sr)
        chunk = (np.sin(2 * np.pi * 200 * t) * 0.5).astype(np.float32)
        rng = np.random.RandomState(42)
        chunk += rng.randn(len(chunk)).astype(np.float32) * 0.001
        ok, reason = final_quality_gate(chunk, sr=sr)
        # Duration is exactly 1.0, which is not < 1.0, so passes duration check
        # May still fail on SNR with only 1s of data
        # The key assertion: duration_out_of_range should NOT be the reason
        assert reason != "duration_out_of_range"
