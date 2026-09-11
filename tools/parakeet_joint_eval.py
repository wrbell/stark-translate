"""Isolated, gated Parakeet joint scalar materialization experiment.

No production or upstream source files change. A hash-matched greedy method is
cloned in memory. Its three existing scalar expressions are evaluated together,
then converted with the same int/float operations. Actual inference is confined
to the child process. This is not a speed claim or automatic production switch.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import math
import sys
import textwrap
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import mac_followup_quality as quality
from tools import parakeet_profile as profile

SCALARS = ("pred_token", "confidence", "decision")


def joint_method(original, expected_source_sha256):
    """Clone a trusted installed method after verifying the profiled source hash."""
    if original.__closure__:
        raise ValueError("Expected non-closure installed greedy method")
    source = textwrap.dedent(inspect.getsource(original))
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    if source_hash != expected_source_sha256:
        raise ValueError("Installed greedy source differs from qualifying profile")
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].decorator_list:
        raise ValueError("Expected one undecorated installed function")
    if any(isinstance(node, ast.Name) and node.id.startswith("_stark_joint_") for node in ast.walk(tree)):
        raise ValueError("Reserved temporary name already present")
    assignments: dict[str, list[ast.Assign]] = {name: [] for name in SCALARS}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in assignments
        ):
            assignments[node.targets[0].id].append(node)
    if any(len(nodes) != 1 for nodes in assignments.values()):
        raise ValueError("Scalar assignment count changed")
    scalar_nodes = [assignments[name][0] for name in SCALARS]
    blocks = [
        node.body
        for node in ast.walk(tree)
        if isinstance(node, ast.While) and all(scalar in node.body for scalar in scalar_nodes)
    ]
    if len(blocks) != 1:
        raise ValueError("Scalar assignments must share one greedy while body")
    block = blocks[0]
    indices = [block.index(node) for node in scalar_nodes]
    if indices != sorted(indices):
        raise ValueError("Scalar assignment order changed")
    # Delay only scalar materialization: intermediary graph expressions cannot
    # depend on a converted scalar or have branch/side-effect statements.
    for node in block[indices[0] + 1 : indices[-1]]:
        if node in scalar_nodes:
            continue
        if not isinstance(node, ast.Assign) or any(isinstance(n, ast.Name) and n.id in SCALARS for n in ast.walk(node)):
            raise ValueError("Scalar dependency/order changed inside sampling block")
    for name, node in zip(SCALARS, scalar_nodes):
        converter = "float" if name == "confidence" else "int"
        value = node.value
        if (
            not isinstance(value, ast.Call)
            or ast.unparse(value.func) != converter
            or len(value.args) != 1
            or value.keywords
        ):
            raise ValueError("Scalar converter changed: " + name)
        if name != "confidence" and (
            not isinstance(value.args[0], ast.Call) or ast.unparse(value.args[0].func) != "mx.argmax"
        ):
            raise ValueError("Argmax expression changed: " + name)
        node.targets = [ast.Name(id="_stark_joint_" + name, ctx=ast.Store())]
        node.value = value.args[0]  # exact original expression, no dtype/math changes
    materialize = ast.Expr(
        ast.Call(
            ast.Attribute(ast.Name("mx", ast.Load()), "eval", ast.Load()),
            [ast.Name("_stark_joint_" + name, ast.Load()) for name in SCALARS],
            [],
        )
    )
    conversions = [
        ast.Assign(
            [ast.Name(name, ast.Store())],
            ast.Call(
                ast.Name("float" if name == "confidence" else "int", ast.Load()),
                [ast.Name("_stark_joint_" + name, ast.Load())],
                [],
            ),
        )
        for name in SCALARS
    ]
    block[indices[-1] + 1 : indices[-1] + 1] = [materialize, *conversions]
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    # Source is inspect.getsource of the imported method, hash-bound to a
    # qualifying profiler receipt and AST-checked above; never supplied code.
    exec(compile(tree, original.__code__.co_filename + ":stark-joint-eval", "exec"), namespace)  # nosec B102
    cloned = namespace[original.__name__]
    cloned.__defaults__, cloned.__kwdefaults__ = original.__defaults__, original.__kwdefaults__
    return cloned, {
        "function": original.__qualname__,
        "original_source_sha256": source_hash,
        "transformed_ast_sha256": hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
        "original_source_file": original.__code__.co_filename,
        "materialization": "mx.eval(token_argmax, unchanged_entropy_confidence, duration_argmax) once per greedy step before same int/float reads",
    }


@contextmanager
def install_joint_method(model, method):
    cls = type(model)
    previous = cls.__dict__.get("decode_greedy")
    try:
        cls.decode_greedy = method
        yield
    finally:
        if previous is None:
            delattr(cls, "decode_greedy")
        else:
            cls.decode_greedy = previous


def qualifying_profile(receipt, *, manifest_sha256, items):
    """Recompute the gate from hash-verified raw calls, not a stored pass boolean."""
    if receipt.get("status") != "completed" or receipt.get("completed") is not True:
        raise ValueError("Profiler run incomplete")
    if receipt.get("manifest_sha256") != manifest_sha256 or receipt.get("items_sha256") != quality.json_hash(items):
        raise ValueError("Profile manifest/item cohort differs")
    ids = {r["id"]: r for r in items}
    if len(ids) != len(items):
        raise ValueError("Duplicate selected items")
    pairs = copy.deepcopy(receipt.get("pairs", []))
    seen = set()
    function_hashes = set()
    for pair in pairs:
        item = ids.get(pair.get("id"))
        repeat = pair.get("repeat")
        if not item or type(repeat) is not int or repeat < 0 or (pair["id"], repeat) in seen:
            raise ValueError("Invalid/duplicate profiler pair")
        seen.add((pair["id"], repeat))
        if (
            pair.get("partition") != "development"
            or pair.get("source_lang") != item["source_lang"]
            or pair.get("audio_sha256") != item["sha256"]
            or pair.get("audio_samples") != item["num_samples"]
        ):
            raise ValueError("Profiler input identity mismatch")
        calls = pair.get("calls", {})
        if set(calls) != {"control", "profiled"} or pair.get("completed") is not True:
            raise ValueError("Profiler pair incomplete")
        for mode, call in calls.items():
            if call.get("mode") != mode or quality.json_hash(call.get("output")) != call.get("output_sha256"):
                raise ValueError("Profiler output hash/mode mismatch")
            if not math.isfinite(call.get("call_wall_ms", float("nan"))) or call["call_wall_ms"] <= 0:
                raise ValueError("Profiler wall time invalid")
        pair["exact_output_equivalence"] = calls["control"]["output"] == calls["profiled"]["output"]
        for ident in calls["profiled"].get("instrumented_functions", []):
            if ident.get("function", "").endswith(".decode_greedy"):
                function_hashes.add(ident["source_sha256"])
    repeats = {repeat for _, repeat in seen}
    if (
        not repeats
        or repeats != set(range(max(repeats) + 1))
        or seen != {(item_id, repeat) for item_id in ids for repeat in repeats}
    ):
        raise ValueError("Profiler pair inventory incomplete")
    assessment = profile.assess_pairs(pairs, receipt.get("expected_pairs", 0))
    if not assessment["joint_evaluation_candidate_allowed"] or len(function_hashes) != 1:
        raise ValueError("Profile does not qualify for joint-evaluation candidate")
    return {
        "assessment": assessment,
        "greedy_source_sha256": next(iter(function_hashes)),
        "model_files_sha256": receipt["model"]["files_sha256"],
        "upstream_sha256": receipt["upstream_implementation"]["sha256"],
        "runtime_versions": {
            k: receipt.get("environment", {}).get("versions", {}).get(k) for k in ("mlx", "parakeet-mlx", "numpy")
        },
        "config": receipt["config"],
    }


def assess(pairs, expected):
    complete = len(pairs) == expected and expected > 0 and all(p.get("completed") for p in pairs)
    valid = [p for p in pairs if p.get("completed")]
    exact = complete and all(p["calls"]["control"]["output"] == p["calls"]["joint_eval"]["output"] for p in valid)
    controls = [p["calls"]["control"]["call_wall_ms"] for p in valid]
    candidates = [p["calls"]["joint_eval"]["call_wall_ms"] for p in valid]
    deltas = [b - c for b, c in zip(controls, candidates)]
    percent = [(b - c) / b * 100 for b, c in zip(controls, candidates)]
    before, after = quality.distribution(controls), quality.distribution(candidates)
    change = quality.distribution(deltas)
    gain = exact and change["p50"] > 0 and after["p95"] <= before["p95"]
    return {
        "complete": complete,
        "pairs": len(pairs),
        "expected_pairs": expected,
        "all_outputs_exact": exact,
        "control_wall_ms": before,
        "joint_eval_wall_ms": after,
        "paired_saved_ms": change,
        "paired_saved_percent": quality.distribution(percent),
        "positive_paired_median_without_pooled_tail_regression": bool(gain),
        "changed_outputs": [
            {"id": p["id"], "repeat": p["repeat"]}
            for p in valid
            if p["calls"]["control"]["output"] != p["calls"]["joint_eval"]["output"]
        ],
        "quantile_method": "median; nearest-rank p95",
        "production_integration": "not authorized by this report; requires parent review of per-language/per-repeat equivalence, quality, timing and memory",
        "compile_experiment": "not implemented; conditional follow-up only",
        "interpretation": "Unprofiled production controls versus joint-evaluation candidate with the same minimal result retention hook. Positive differences mean time saved. Cold/first calls remain included; process RSS/Metal peaks are cumulative, not per-arm allocations.",
    }


def call(engine, item, audio, candidate, method, audio_module):
    if candidate:
        with install_joint_method(engine._model, method):
            row = profile.one_call(engine, item, audio, profiled=False, sample_every=1, audio_module=audio_module)
        row["mode"] = "joint_eval"
        return row
    return profile.one_call(engine, item, audio, profiled=False, sample_every=1, audio_module=audio_module)


def worker(args):
    from tools.benchmark_identity import load_primary_model

    manifest = quality.read_json(args.manifest)
    items = profile.selected(manifest, args.languages, args.limit)
    receipt = quality.read_json(args.profile_receipt)
    if quality.digest(args.profile_receipt) != args.profile_sha256:
        raise ValueError("Profiler receipt changed")
    gate = qualifying_profile(receipt, manifest_sha256=quality.digest(args.manifest), items=items)
    expected = len(items) * args.repeats
    data = {
        "schema_version": 1,
        "status": "starting",
        "completed": False,
        "pairs": [],
        "started_at": datetime.now(UTC).isoformat(),
        "manifest_sha256": quality.digest(args.manifest),
        "items_sha256": quality.json_hash(items),
        "profile_receipt_sha256": args.profile_sha256,
        "qualifying_profile": gate,
        "expected_pairs": expected,
        "environment": quality.source_identity(),
        "candidate_source_sha256": quality.digest(Path(__file__)),
        "profiler_helper_sha256": quality.digest(Path(profile.__file__)),
        "pairing": "Control/candidate order alternates by repeat + item index; same loaded model and source. No scalar/stage timing probes or extra synchronization on control. No calls discarded, including first call per mode.",
        "human_approved_locally": False,
        "training_eligible": False,
        "defaults_changed": False,
    }
    quality.save(args.output, data)
    engine = None
    try:
        config = gate["config"]
        current_versions = data["environment"]["versions"]
        if any(value is None or current_versions.get(key) != value for key, value in gate["runtime_versions"].items()):
            raise ValueError("MLX/Parakeet/NumPy version differs from qualifying profile")
        if config != quality.engine_config("parakeet-mlx", "stt", config["requested_model_id"]):
            raise ValueError("Profile engine configuration is not the current explicit Parakeet contract")
        inventory = quality.model_inventory(config)
        if inventory["files_sha256"] != gate["model_files_sha256"]:
            raise ValueError("Model files differ from qualifying profile")
        data.update(config=config, model=inventory)
        engine = quality.make_engine(config, inventory["resolved_path"], "en")
        data["model_identity"] = load_primary_model(engine, inventory["resolved_path"])
        import parakeet_mlx.audio as audio_module
        import parakeet_mlx.parakeet as upstream

        if quality.digest(Path(upstream.__file__)) != gate["upstream_sha256"]:
            raise ValueError("Installed upstream file differs from qualifying profile")
        method, identity = joint_method(type(engine._model).decode_greedy, gate["greedy_source_sha256"])
        data["candidate_method"] = identity
        data["upstream_implementation"] = receipt["upstream_implementation"]
        data["status"] = "running"
        for repeat in range(args.repeats):
            for index, item in enumerate(items):
                pair = {
                    "id": item["id"],
                    "source_lang": item["source_lang"],
                    "partition": "development",
                    "repeat": repeat,
                    "audio_sha256": item["sha256"],
                    "audio_samples": item["num_samples"],
                    "completed": False,
                    "calls": {},
                }
                order = [False, True] if (repeat + index) % 2 == 0 else [True, False]
                pair["order"] = ["joint_eval" if mode else "control" for mode in order]
                try:
                    audio = quality.read_audio(quality.relative_file(args.audio_root, item["path"]), item)
                    for candidate in order:
                        row = call(engine, item, audio, candidate, method, audio_module)
                        pair["calls"][row["mode"]] = row
                    pair["exact_output_equivalence"] = (
                        pair["calls"]["control"]["output"] == pair["calls"]["joint_eval"]["output"]
                    )
                    pair["completed"] = True
                except Exception as exc:
                    pair["error"] = f"{type(exc).__name__}: {exc}"
                data["pairs"].append(pair)
                data["assessment"] = assess(data["pairs"], expected)
                data["language_assessments"] = {
                    lang: assess(
                        [p for p in data["pairs"] if p["source_lang"] == lang],
                        sum(i["source_lang"] == lang for i in items) * args.repeats,
                    )
                    for lang in args.languages
                }
                data["repeat_assessments"] = {
                    str(r): assess([p for p in data["pairs"] if p["repeat"] == r], len(items))
                    for r in range(args.repeats)
                }
                quality.save(args.output, data)
        data["completed"] = data["assessment"]["complete"]
        data["status"] = "completed" if data["completed"] else "failed_pairs"
    except Exception as exc:
        data.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if engine is not None:
            try:
                engine.unload()
            except Exception as exc:
                data.update(completed=False, status="cleanup_failed", cleanup_error=f"{type(exc).__name__}: {exc}")
        data["ended_at"] = datetime.now(UTC).isoformat()
        quality.save(args.output, data)
    return data["completed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--profile-receipt", type=Path, required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", choices=["en", "es"], default=["en", "es"])
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        args.repeats < 1
        or not math.isfinite(args.timeout_seconds)
        or args.timeout_seconds <= 0
        or len(set(args.languages)) != len(args.languages)
    ):
        parser.error("Positive repeat/timeout and unique language selection required")
    if quality.digest(args.profile_receipt) != args.profile_sha256:
        parser.error("Profiler receipt SHA256 mismatch")
    items = profile.selected(quality.read_json(args.manifest), args.languages, args.limit)
    qualifying_profile(
        quality.read_json(args.profile_receipt), manifest_sha256=quality.digest(args.manifest), items=items
    )
    if args._worker:
        return 0 if worker(args) else 1
    from tools.mac_evaluation import _run_evaluation_worker

    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--_worker"]
    result = _run_evaluation_worker(
        command, args.output, args.manifest, env=quality.clean_environment(), timeout=args.timeout_seconds
    )
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
