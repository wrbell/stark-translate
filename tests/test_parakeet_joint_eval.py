"""Verify hash-bound in-memory transformation and fail-closed candidate gates."""

import copy
import hashlib
import inspect
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from tools import mac_followup_quality as quality
from tools import parakeet_joint_eval as joint

EVENTS = []


class Scalar:
    def __init__(self, value, name):
        self.value, self.name = value, name

    def __int__(self):
        EVENTS.append(("int", self.name))
        return int(self.value)

    def __float__(self):
        EVENTS.append(("float", self.name))
        return float(self.value)


def materialize(*values):
    EVENTS.append(("eval", tuple(v.name for v in values)))


mx = SimpleNamespace(argmax=lambda value: value, eval=materialize)


class Model:
    def decode_greedy(self, features, *, maximum=2):
        step = 0
        results = []
        while step < maximum:
            pred_token = int(mx.argmax(features[0]))
            entropy = features[1]
            confidence = float(entropy)
            decision = int(mx.argmax(features[2]))
            results.append((pred_token, confidence, decision, step))
            step += decision
        return results


def source_hash(function):
    return hashlib.sha256(textwrap.dedent(inspect.getsource(function)).encode()).hexdigest()


def test_exact_expressions_values_and_readback_order_with_one_eval_per_step():
    original = Model.decode_greedy
    candidate, evidence = joint.joint_method(original, source_hash(original))
    values = [Scalar(4, "token"), Scalar(0.938, "confidence"), Scalar(1, "duration")]
    EVENTS.clear()
    expected = Model().decode_greedy(values)
    assert len(EVENTS) == 6 and all(e[0] != "eval" for e in EVENTS)
    EVENTS.clear()
    with joint.install_joint_method(Model(), candidate):
        assert Model().decode_greedy(values) == expected
    assert Model.decode_greedy is original
    assert (
        EVENTS
        == [
            ("eval", ("token", "confidence", "duration")),
            ("int", "token"),
            ("float", "confidence"),
            ("int", "duration"),
        ]
        * 2
    )
    assert evidence["original_source_sha256"] == source_hash(original)
    assert len(evidence["transformed_ast_sha256"]) == 64
    assert candidate.__defaults__ == original.__defaults__
    assert candidate.__kwdefaults__ is original.__kwdefaults__


def test_inherited_method_is_restored_on_failure():
    class Child(Model):
        pass

    candidate, _ = joint.joint_method(Model.decode_greedy, source_hash(Model.decode_greedy))
    with pytest.raises(RuntimeError), joint.install_joint_method(Child(), candidate):
        assert "decode_greedy" in Child.__dict__
        raise RuntimeError("failed inference")
    assert "decode_greedy" not in Child.__dict__


def test_hash_drift_fails_before_install():
    original = Model.decode_greedy
    with pytest.raises(ValueError, match="differs"):
        joint.joint_method(original, "0" * 64)
    assert Model.decode_greedy is original


def test_intervening_scalar_use_rejected():
    class Changed:
        def decode_greedy(self, features):
            while True:
                pred_token = int(mx.argmax(features[0]))
                entropy = features[pred_token]
                confidence = float(entropy)
                decision = int(mx.argmax(features[2]))
                return pred_token, confidence, decision

    with pytest.raises(ValueError, match="dependency"):
        joint.joint_method(Changed.decode_greedy, source_hash(Changed.decode_greedy))


def test_non_loop_assignments_or_converter_changes_rejected():
    class Changed:
        def decode_greedy(self, features):
            pred_token = int(mx.argmax(features[0]))
            confidence = float(features[1])
            decision = int(mx.argmax(features[2]))
            return pred_token, confidence, decision

    with pytest.raises(ValueError, match="while body"):
        joint.joint_method(Changed.decode_greedy, source_hash(Changed.decode_greedy))


def receipt():
    items = [dict(id="one", source_lang="en", sha256="a" * 64, num_samples=16000)]
    output = {"stt": {"text": "Hello"}, "aligned_results": [{"text": "Hello", "sentences": []}]}
    control = dict(
        mode="control", call_wall_ms=100, output=output, output_sha256=quality.json_hash(output), stages=None
    )
    observed = dict(
        control,
        mode="profiled",
        call_wall_ms=110,
        stages={"token_readback": dict(sampled_wall_sum_ms=13.2, samples_omitted_by_capacity=0)},
        instrumented_functions=[dict(function="ParakeetTDT.decode_greedy", source_sha256="b" * 64)],
    )
    pair = dict(
        id="one",
        source_lang="en",
        repeat=0,
        audio_sha256="a" * 64,
        audio_samples=16000,
        partition="development",
        completed=True,
        calls={"control": control, "profiled": observed},
    )
    report = dict(
        status="completed",
        completed=True,
        manifest_sha256="c" * 64,
        items_sha256=quality.json_hash(items),
        expected_pairs=1,
        pairs=[pair],
        model={"files_sha256": "d" * 64},
        upstream_implementation={"sha256": "e" * 64},
        config={},
    )
    return report, items


def test_gate_recomputed_from_raw_outputs_ignores_false_stored_boolean():
    report, items = receipt()
    report["assessment"] = {"joint_evaluation_candidate_allowed": False}
    report["pairs"][0]["exact_output_equivalence"] = False
    assert joint.qualifying_profile(report, manifest_sha256="c" * 64, items=items)["assessment"][
        "joint_evaluation_candidate_allowed"
    ]
    report["pairs"][0]["calls"]["profiled"]["output"] = {"stt": {"text": "Different"}}
    report["pairs"][0]["calls"]["profiled"]["output_sha256"] = quality.json_hash(
        report["pairs"][0]["calls"]["profiled"]["output"]
    )
    with pytest.raises(ValueError, match="does not qualify"):
        joint.qualifying_profile(report, manifest_sha256="c" * 64, items=items)


@pytest.mark.parametrize("mutation", ["low_share", "duplicate", "missing", "hash", "source", "confirmation"])
def test_invalid_profile_receipt_cannot_authorize_candidate(mutation):
    report, items = receipt()
    pair = report["pairs"][0]
    if mutation == "low_share":
        pair["calls"]["profiled"]["stages"]["token_readback"]["sampled_wall_sum_ms"] = 1
    elif mutation == "duplicate":
        report["pairs"].append(copy.deepcopy(pair))
    elif mutation == "missing":
        report["expected_pairs"] = 2
    elif mutation == "hash":
        pair["calls"]["control"]["output_sha256"] = "broken"
    elif mutation == "source":
        pair["audio_sha256"] = "wrong"
    elif mutation == "confirmation":
        pair["partition"] = "confirmation"
    with pytest.raises(ValueError):
        joint.qualifying_profile(report, manifest_sha256="c" * 64, items=items)


def measured(before, after, exact=True, repeat=0):
    return dict(
        id="one",
        repeat=repeat,
        completed=True,
        calls={
            "control": {"call_wall_ms": before, "output": {"tokens": [1], "confidence": 0.9}},
            "joint_eval": {"call_wall_ms": after, "output": {"tokens": [1 if exact else 2], "confidence": 0.9}},
        },
    )


def test_gain_requires_exact_completed_outputs_and_non_regressed_tail():
    result = joint.assess([measured(100, 80), measured(110, 90, repeat=1)], 2)
    assert result["positive_paired_median_without_pooled_tail_regression"]
    assert result["paired_saved_ms"]["p50"] == 20
    assert result["joint_eval_wall_ms"]["p95"] == 90
    for rows, expected in [
        ([measured(100, 80, False)], 1),
        ([measured(100, 80)], 2),
        ([measured(100, 80), measured(110, 120, repeat=1)], 2),
    ]:
        assert not joint.assess(rows, expected)["positive_paired_median_without_pooled_tail_regression"]
    assert not joint.assess([], 1)["complete"]


def test_actual_control_call_is_unprofiled_and_candidate_restores(monkeypatch):
    original = Model.decode_greedy
    model = Model()
    engine = SimpleNamespace(_model=model)
    candidate, _ = joint.joint_method(original, source_hash(original))
    modes = []

    def fake_call(engine, item, audio, **kwargs):
        modes.append((kwargs["profiled"], type(engine._model).decode_greedy))
        return {"mode": "control", "output": {}}

    monkeypatch.setattr(joint.profile, "one_call", fake_call)
    assert joint.call(engine, {}, [], False, candidate, None)["mode"] == "control"
    assert joint.call(engine, {}, [], True, candidate, None)["mode"] == "joint_eval"
    assert modes == [(False, original), (False, candidate)]
    assert Model.decode_greedy is original


def test_import_and_cli_help_do_not_import_models():
    command = "import sys; import tools.parakeet_joint_eval; assert not any(k in sys.modules for k in ['mlx','torch','parakeet_mlx','numpy','soundfile'])"
    result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
