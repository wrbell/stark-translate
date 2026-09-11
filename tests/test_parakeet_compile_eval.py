"""Tensor-only compile boundary, explicit state and raw evidence gates; no ML imports."""

import copy
import hashlib
import inspect
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from tools import mac_followup_quality as quality
from tools import parakeet_compile_eval as compile_eval
from tools import parakeet_joint_eval as joint
from tools import parakeet_profile as profile


class Tensor:
    def __init__(self, value, shape=(1, 1, 3), dtype="bfloat16"):
        self.value, self.shape, self.dtype = value, shape, dtype

    def astype(self, dtype):
        return Tensor(self.value, self.shape, dtype)

    def __getitem__(self, index):
        return Tensor(self.value, (1, 1, self.shape[-1]), self.dtype)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)


class FakeMX:
    def __init__(self):
        self.compile_requests = []
        self.compiled_inputs = []
        self.failure = False

    def array(self, value):
        return Tensor(value[0][0], (1, 1), "int32")

    def argmax(self, value):
        return value

    def eval(self, *values):
        pass

    def compile(self, function, *, shapeless):
        self.compile_requests.append(shapeless)

        def compiled(*args):
            self.compiled_inputs.append(args)
            if self.failure:
                raise RuntimeError("compile rejected")
            return function(*args)

        return compiled


mx = FakeMX()


class Model:
    def __init__(self):
        self.decoder_states = []

    def decoder(self, token, state):
        self.decoder_states.append((token, state))
        value = 1 if state is None else state[0].value + 1
        return Tensor(2), (Tensor(value), Tensor(value + 10))

    def joint(self, frame, decoded):
        return Tensor(decoded.value)

    def decode_greedy(self, features):
        hidden_state = [None]
        last_token = [None]
        result = []
        batch = 0
        feature = features
        step = 0
        while step < 3:
            decoder_out, (hidden, cell) = self.decoder(
                mx.array([[last_token[batch]]]) if last_token[batch] is not None else None,
                hidden_state[batch],
            )
            decoder_out = decoder_out.astype(feature.dtype)
            decoder_hidden = (hidden.astype(feature.dtype), cell.astype(feature.dtype))
            joint_out = self.joint(feature[:, step : step + 1], decoder_out)
            pred_token = int(mx.argmax(joint_out))
            confidence = float(feature)
            decision = int(mx.argmax(feature))
            result.append((pred_token, confidence, decision, step))
            last_token[batch] = pred_token
            hidden_state[batch] = decoder_hidden
            step += decision
        return result, hidden_state


def source_hash(function):
    return hashlib.sha256(textwrap.dedent(inspect.getsource(function)).encode()).hexdigest()


def test_actual_method_transform_preserves_bootstrap_state_and_decisions():
    model = Model()
    runtime = FakeMX()
    steps = compile_eval.StepCompiler(model, runtime)
    plain, candidate, identity = compile_eval.compiled_method(
        Model.decode_greedy, source_hash(Model.decode_greedy), steps
    )
    control, state = plain(model, Tensor(1, (1, 3, 3)))
    outputs, new_state = candidate(model, Tensor(1, (1, 3, 3)))
    assert outputs == control
    assert new_state[0][0].value == state[0][0].value == 3
    assert new_state[0][1].value == state[0][1].value == 13
    assert runtime.compile_requests == [False]
    assert len(runtime.compiled_inputs) == 2
    assert all(isinstance(value, Tensor) for call in runtime.compiled_inputs for value in call)
    assert runtime.compiled_inputs[0][2].value == 1
    assert runtime.compiled_inputs[1][2].value == 2
    stats = steps.snapshot()
    assert stats["eager_bootstrap_calls"] == 1 and stats["compiled_calls"] == 2
    assert stats["signature_count"] == 1 and stats["signatures"][0]["first_dispatch_failed"] is False
    assert (
        identity["transformed_ast_sha256"]
        == joint.joint_method(Model.decode_greedy, source_hash(Model.decode_greedy))[1]["transformed_ast_sha256"]
    )
    assert len(identity["compiled_step_ast_sha256"]) == 64


def test_none_token_or_state_never_enters_compiled_graph():
    model = Model()
    runtime = FakeMX()
    steps = compile_eval.StepCompiler(model, runtime)
    frame = Tensor(1)
    steps(frame, None, (Tensor(3), Tensor(4)))
    steps(frame, Tensor(2, (1, 1), "int32"), None)
    assert not runtime.compiled_inputs and steps.snapshot()["eager_bootstrap_calls"] == 2


def test_compilation_failure_propagates_without_eager_fallback():
    model = Model()
    runtime = FakeMX()
    runtime.failure = True
    steps = compile_eval.StepCompiler(model, runtime)
    with pytest.raises(RuntimeError, match="compile rejected"):
        steps(Tensor(1), Tensor(2, (1, 1), "int32"), (Tensor(3), Tensor(4)))
    assert not model.decoder_states
    assert steps.snapshot()["failed_calls"] == 1
    assert steps.snapshot()["signatures"][0]["first_dispatch_failed"]


def test_shape_drift_capacity_is_bounded_and_no_shapeless_specialization():
    runtime = FakeMX()
    steps = compile_eval.StepCompiler(Model(), runtime, max_signatures=1)
    token = Tensor(2, (1, 1), "int32")
    steps(Tensor(1), token, (Tensor(3), Tensor(4)))
    with pytest.raises(ValueError, match="capacity"):
        steps(Tensor(1, (1, 1, 4)), token, (Tensor(3), Tensor(4)))
    with pytest.raises(ValueError, match="batch1"):
        steps(Tensor(1, (1, 2, 3)), token, (Tensor(3), Tensor(4)))
    assert len(runtime.compiled_inputs) == 1 and runtime.compile_requests == [False]


def test_source_hash_change_rejected_before_transformation():
    with pytest.raises(ValueError, match="differs"):
        compile_eval.compiled_method(Model.decode_greedy, "0" * 64, lambda *a: None)


def receipt():
    item = dict(id="one", source_lang="en", sha256="a" * 64, num_samples=16000)
    output = {"stt": {"text": "Hello", "confidence": 0.9}, "aligned_results": [{"tokens": [1, 2]}]}
    calls = {
        mode: dict(
            mode=mode,
            output=copy.deepcopy(output),
            output_sha256=quality.json_hash(output),
            call_wall_ms=100 if mode == "control" else 90,
            stages=None,
        )
        for mode in ("control", "joint_eval")
    }
    pair = dict(
        id="one",
        source_lang="en",
        repeat=0,
        partition="development",
        audio_sha256="a" * 64,
        audio_samples=16000,
        completed=True,
        calls=calls,
    )
    report = dict(
        status="completed",
        completed=True,
        returncode=0,
        manifest_sha256="b" * 64,
        items_sha256=quality.json_hash([item]),
        pairs=[pair],
        expected_pairs=1,
        candidate_source_sha256=quality.digest(__import__("pathlib").Path(joint.__file__)),
        profiler_helper_sha256=quality.digest(__import__("pathlib").Path(profile.__file__)),
        candidate_method=dict(original_source_sha256="c" * 64, transformed_ast_sha256="d" * 64),
        model={"files_sha256": "e" * 64},
        config={},
        environment={"versions": {"mlx": "0.32.2", "parakeet-mlx": "0.5.2", "numpy": "2.3.5"}},
    )
    return report, [item]


def test_gate_uses_joint_unprofiled_exact_outputs_and_measured_effect():
    report, items = receipt()
    gate = compile_eval.qualifying_joint(report, manifest_sha256="b" * 64, items=items)
    assert gate["assessment"]["all_outputs_exact"]
    report["pairs"][0]["calls"]["joint_eval"]["call_wall_ms"] = 101
    with pytest.raises(ValueError, match="measured-gain"):
        compile_eval.qualifying_joint(report, manifest_sha256="b" * 64, items=items)


@pytest.mark.parametrize("change", ["hash", "confidence", "missing", "duplicate", "profiled", "helper", "failure"])
def test_invalid_joint_evidence_does_not_authorize_compile(change):
    report, items = receipt()
    pair = report["pairs"][0]
    if change == "hash":
        pair["calls"]["joint_eval"]["output_sha256"] = "wrong"
    elif change == "confidence":
        pair["calls"]["joint_eval"]["output"]["stt"]["confidence"] = 0.899999
        pair["calls"]["joint_eval"]["output_sha256"] = quality.json_hash(pair["calls"]["joint_eval"]["output"])
    elif change == "missing":
        report["expected_pairs"] = 2
    elif change == "duplicate":
        report["pairs"].append(copy.deepcopy(pair))
    elif change == "profiled":
        pair["calls"]["control"]["stages"] = {}
    elif change == "helper":
        report["profiler_helper_sha256"] = "wrong"
    else:
        report["returncode"] = 1
    with pytest.raises(ValueError):
        compile_eval.qualifying_joint(report, manifest_sha256="b" * 64, items=items)


def test_three_way_assessment_retains_confidence_changes_and_failed_arms():
    report, _ = receipt()
    t = report["pairs"][0]
    t["calls"]["compiled_joint"] = dict(
        t["calls"]["joint_eval"], call_wall_ms=70, compile_stats_delta={"compiled_calls": 20}
    )
    result = compile_eval.triplet_assessment([t], 1)
    assert result["all_three_outputs_exact"] and result["compile_exercised_calls"] == 20
    assert result["comparisons_to_compiled_joint"]["control"]["paired_saved_ms"]["p50"] == 30
    assert result["comparisons_to_compiled_joint"]["joint_eval"]["paired_saved_ms"]["p50"] == 20
    t["calls"]["compiled_joint"] = copy.deepcopy(t["calls"]["compiled_joint"])
    t["calls"]["compiled_joint"]["output"]["stt"]["confidence"] = 0.9001
    assert not compile_eval.triplet_assessment([t], 1)["all_three_outputs_exact"]
    t["completed"] = False
    assert not compile_eval.triplet_assessment([t], 1)["completed"]


def test_control_does_not_install_candidate_and_first_compile_cost_is_retained(monkeypatch):
    original = Model.decode_greedy
    model = Model()
    steps = compile_eval.StepCompiler(model, FakeMX())
    plain, compiled, _ = compile_eval.compiled_method(original, source_hash(original), steps)
    observed = []

    def fake_call(engine, item, audio, **kwargs):
        observed.append((Model.decode_greedy, kwargs["profiled"]))
        Model.decode_greedy(model, Tensor(1, (1, 3, 3)))
        return {"mode": "control"}

    monkeypatch.setattr(profile, "one_call", fake_call)
    engine = SimpleNamespace(_model=model)
    first = compile_eval.call(engine, {}, [], "compiled_joint", plain, compiled, steps, None)
    next_call = compile_eval.call(engine, {}, [], "compiled_joint", plain, compiled, steps, None)
    control = compile_eval.call(engine, {}, [], "control", plain, compiled, steps, None)
    assert first["contains_first_compiled_signature"]
    assert not next_call["contains_first_compiled_signature"] and not control["contains_first_compiled_signature"]
    assert control["compile_stats_delta"]["compiled_calls"] == 0
    assert observed == [(compiled, False), (compiled, False), (original, False)]
    assert Model.decode_greedy is original


def test_import_is_model_free():
    command = 'import sys; import tools.parakeet_compile_eval; assert not any(k in sys.modules for k in ["mlx","torch","numpy","parakeet_mlx","soundfile"])'
    result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
