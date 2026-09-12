"""In-memory instrumentation tests, without importing or running Parakeet/MLX."""

import ast
import copy
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import parakeet_profile as p


class Scalar:
    def __init__(self, value):
        self.value = value
        self.ints = 0
        self.floats = 0

    def __int__(self):
        self.ints += 1
        return int(self.value)

    def __float__(self):
        self.floats += 1
        return float(self.value)


mx = SimpleNamespace(argmax=lambda v: v, eval=lambda *a: None)


class Model:
    def decoder(self, value):
        return value

    def joint(self, value):
        return value

    def encoder(self, value):
        return value, None

    def decode(self, features, lengths):
        return self.decode_greedy(features)

    def decode_greedy(self, features):
        decoder_out = self.decoder(features[0])
        joint_out = self.joint(decoder_out)
        pred_token = int(mx.argmax(joint_out))
        confidence = float(features[1])
        decision = int(mx.argmax(features[2]))
        return [(pred_token, confidence, decision)]

    def generate(self, mel):
        features, lengths = self.encoder(mel)
        mx.eval(features, lengths)
        return self.decode(features, lengths)


def test_scalar_probe_calls_conversion_exactly_once_and_samples():
    ticks = iter([0, 0.001, 0.1, 0.103])
    probe = p.Probe(sample_every=2, clock=lambda: next(ticks))
    x = Scalar(3)
    assert [probe.call("token_readback", int, x) for _ in range(3)] == [3, 3, 3]
    assert x.ints == 3
    report = probe.summary()["token_readback"]
    assert report["calls"] == 3
    assert report["sampled_wall_ms"]["n"] == 2
    assert report["sampled_wall_sum_ms"] == pytest.approx(4)


def test_exception_and_capacity_do_not_change_call_behavior():
    ticks = iter([0, 0.001])
    probe = p.Probe(sample_every=1, max_samples=1, clock=lambda: next(ticks))

    def broken():
        raise RuntimeError("original")

    with pytest.raises(RuntimeError, match="original"):
        probe.call("bad", broken)
    assert probe.call("bad", lambda: 7) == 7
    summary = probe.summary()["bad"]
    assert summary["samples_omitted_by_capacity"] == 1
    assert summary["sampled_wall_ms"]["n"] == 1


def test_temporary_actual_method_clone_preserves_output_and_restores():
    original_generate, original_greedy = Model.generate, Model.decode_greedy
    model = Model()
    values = [Scalar(2), Scalar(0.8), Scalar(1)]
    baseline = model.generate(values)
    probe = p.Probe(sample_every=1)
    with p.observe(model, probe=probe) as (outputs, identities):
        profiled = model.generate(values)
        assert outputs == profiled == baseline
        assert {i["function"] for i in identities} == {"Model.generate", "Model.decode_greedy"}
        assert Model.decode_greedy is not original_greedy
    assert Model.generate is original_generate and Model.decode_greedy is original_greedy
    assert values[0].ints == values[2].ints == values[1].floats == 2
    assert set(probe.summary()) == {
        "token_readback",
        "confidence_readback",
        "duration_readback",
        "decoder_graph",
        "joint_graph",
        "encoder_graph",
        "encoder_materialization",
        "decode_total",
    }


def test_control_has_no_stage_or_scalar_instrumentation():
    original_greedy = Model.decode_greedy
    model = Model()
    with p.observe(model) as (raw, identities):
        assert Model.decode_greedy is original_greedy
        output = model.generate([Scalar(2), Scalar(0.8), Scalar(1)])
        assert raw == output and identities == []


def test_inherited_methods_and_audio_helper_restored_on_exception():
    class Child(Model):
        pass

    model = Child()
    original = lambda value: value
    audio = SimpleNamespace(get_logmel=original)
    with pytest.raises(RuntimeError, match="stop"), p.observe(model, probe=p.Probe(), audio_module=audio):
        assert audio.get_logmel is not original
        raise RuntimeError("stop")
    assert "generate" not in Child.__dict__ and "decode_greedy" not in Child.__dict__
    assert audio.get_logmel is original


def test_upstream_shape_change_rejected_before_any_patch():
    class Changed(Model):
        def decode_greedy(self, features):
            pred_token = int(features[0])
            confidence = float(features[1])
            decision = int(mx.argmax(features[2]))
            return pred_token, confidence, decision

    old = Changed.generate
    with pytest.raises(ValueError, match="argmax assignment changed"), p.observe(Changed(), probe=p.Probe()):
        pass
    assert Changed.generate is old


def pair(share=0.12, same=True, overflow=0):
    return {
        "completed": True,
        "exact_output_equivalence": same,
        "calls": {
            "control": {"call_wall_ms": 100},
            "profiled": {
                "call_wall_ms": 110,
                "stages": {
                    "token_readback": {"sampled_wall_sum_ms": share * 110, "samples_omitted_by_capacity": overflow},
                    "decoder_graph": {"sampled_wall_sum_ms": 200, "samples_omitted_by_capacity": 0},
                },
            },
        },
    }


def test_conditional_candidate_gate_uses_measured_share_and_exact_pairs():
    result = p.assess_pairs([pair(), pair()], 2)
    assert result["joint_evaluation_candidate_allowed"]
    assert result["sampled_readback_share_of_profiled_stt_wall"]["p50"] == pytest.approx(0.12)
    assert result["profile_vs_control_wall_percent"]["p50"] == pytest.approx(10)
    assert "not implemented" in result["joint_evaluation_candidate"]
    for rows, expected in [([pair(0.09)], 1), ([pair(same=False)], 1), ([pair(overflow=1)], 1), ([pair()], 2), ([], 1)]:
        assert not p.assess_pairs(rows, expected)["joint_evaluation_candidate_allowed"]


def test_low_overhead_is_signed_not_clipped_to_zero():
    x = pair()
    x["calls"]["profiled"]["call_wall_ms"] = 90
    assert p.assess_pairs([x], 1)["profile_vs_control_wall_percent"]["p50"] == pytest.approx(-10)


def test_raw_token_equivalence_includes_confidence_timestamp_id_and_text():
    token = SimpleNamespace(id=3, text="faith", start=1.0, end=1.1, duration=0.1, confidence=0.7)
    sentence = SimpleNamespace(text="faith", tokens=[token], start=1.0, end=1.1, duration=0.1, confidence=0.7)
    aligned = SimpleNamespace(text="faith", sentences=[sentence])
    before = p.raw_outputs([aligned])
    for field, changed in [
        ("id", 4),
        ("text", "Faith"),
        ("confidence", 0.71),
        ("start", 1.01),
        ("end", 1.11),
        ("duration", 0.11),
    ]:
        alternate = copy.deepcopy(aligned)
        setattr(alternate.sentences[0].tokens[0], field, changed)
        assert p.raw_outputs([alternate]) != before


def test_installed_tdt_source_structure_without_importing_model():
    candidates = sorted((p.ROOT / "stt_env/lib").glob("python*/site-packages/parakeet_mlx/parakeet.py"))
    if not candidates:
        pytest.skip("Optional installed package absent; synthetic method contracts cover CI")
    source = ast.parse(candidates[0].read_text())
    tdt = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "ParakeetTDT")
    for name, kind in [("decode_greedy", "greedy"), ("generate", "generate")]:
        function = next(n for n in tdt.body if isinstance(n, ast.FunctionDef) and n.name == name)
        transform = p.Instrument(kind)
        transform.visit(function)
        assert all(count == 1 for count in transform.counts.values())
        assert len(transform.counts) == (5 if kind == "greedy" else 3)


def test_import_has_no_model_or_audio_loads():
    code = "import sys; import tools.parakeet_profile; assert not any(n in sys.modules for n in ('mlx','parakeet_mlx','torch','numpy','soundfile'))"
    result = subprocess.run([sys.executable, "-c", code], cwd=p.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("baseline", [True, False])
def test_profile_worker_always_selects_stock_decode_before_loading(monkeypatch, tmp_path, baseline):
    from unittest.mock import Mock

    from engines import parakeet_mlx_engine
    from tools import benchmark_identity

    args = SimpleNamespace(
        manifest=tmp_path / "manifest.json",
        output=tmp_path / "output.json",
        languages=["en"],
        limit=1,
        repeats=1,
        sample_every=1,
        model_override=None,
        baseline_decode=baseline,
    )
    monkeypatch.setattr(p.quality, "read_json", lambda _: {})
    monkeypatch.setattr(p, "selected", lambda *args: [])
    monkeypatch.setattr(p.quality, "digest", lambda _: "hash")
    monkeypatch.setattr(p.quality, "source_identity", lambda: {})
    monkeypatch.setattr(p.quality, "save", Mock())
    monkeypatch.setattr(p.quality, "engine_config", lambda *args: {"name": "parakeet-mlx"})
    monkeypatch.setattr(p.quality, "model_inventory", lambda _: {"resolved_path": "/fake/parakeet"})
    stock_factory = Mock()
    default_factory = Mock()
    monkeypatch.setattr(parakeet_mlx_engine, "ParakeetMLXEngine", stock_factory)
    monkeypatch.setattr(p.quality, "make_engine", default_factory)
    load = Mock(side_effect=RuntimeError("unit test stops before model load"))
    monkeypatch.setattr(benchmark_identity, "load_primary_model", load)
    assert not p.profile_worker(args)
    # Both arms parse the stock greedy source, so the flag never changes the engine.
    stock_factory.assert_called_once_with(model_id="/fake/parakeet", joint_scalar_eval=False)
    default_factory.assert_not_called()
    assert load.call_args.args[0] is stock_factory.return_value
