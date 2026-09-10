"""Isolated TDT steady-state decoder/joint compile experiment; no production switch.

The Python greedy loop, scalar materialization, text decoding and state commits
stay outside mx.compile. Bootstrap token/state=None stays eager. The compiled
step receives frame/token/hidden/cell arrays and returns joint output/new state.
Only the inspected immutable-weight model/source version is supported.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import math
import os
import sys
import textwrap
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import mac_followup_quality as quality
from tools import parakeet_joint_eval as joint
from tools import parakeet_profile as profile

MODES = ("control", "joint_eval", "compiled_joint")
SOURCE_LOCK = {
    "parakeet_mlx/parakeet.py": "2a4208b051ed20356908848c76080ebee7df4a554be91fc338fbe6747b19fccb",
    "parakeet_mlx/rnnt.py": "474d4896be99fa5b93c47198f2bd6da5160eaa691a50c8e2827baf7ccc1132d9",
    "mlx/nn/layers/recurrent.py": "553738db5ffede77d4d97a6b431ac82475b99c34902a9a32e05b940a95f34ae7",
}
COMPILE_DOCS = [
    "https://ml-explore.github.io/mlx/build/html/usage/compile.html",
    "https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.compile.html",
]


def _tree_hash(tree):
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def compiled_method(original, expected_hash, step):
    """Keep the canonical joint-eval AST, replacing only four inspected step statements."""
    plain, canonical = joint.joint_method(original, expected_hash)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in joint.SCALARS
    ]
    indexed = {cast(ast.Name, node.targets[0]).id: node for node in nodes}
    scalar_nodes = [indexed[name] for name in joint.SCALARS]
    body = next(
        node.body
        for node in ast.walk(tree)
        if isinstance(node, ast.While) and all(scalar in node.body for scalar in scalar_nodes)
    )
    last = body.index(scalar_nodes[-1])
    for name, node in zip(joint.SCALARS, scalar_nodes, strict=True):
        node.targets = [ast.Name("_stark_joint_" + name, ast.Store())]
        node.value = cast(ast.Call, node.value).args[0]
    body[last + 1 : last + 1] = [
        ast.Expr(
            ast.Call(
                ast.Attribute(ast.Name("mx", ast.Load()), "eval", ast.Load()),
                [ast.Name("_stark_joint_" + name, ast.Load()) for name in joint.SCALARS],
                [],
            )
        ),
        *[
            ast.Assign(
                [ast.Name(name, ast.Store())],
                ast.Call(
                    ast.Name("float" if name == "confidence" else "int", ast.Load()),
                    [ast.Name("_stark_joint_" + name, ast.Load())],
                    [],
                ),
            )
            for name in joint.SCALARS
        ],
    ]
    ast.fix_missing_locations(tree)
    if _tree_hash(tree) != canonical["transformed_ast_sha256"]:
        raise ValueError("Joint-evaluation transform differs from qualifying canonical helper")
    decoder_indices = [
        i
        for i, node in enumerate(body)
        if any(isinstance(n, ast.Call) and ast.unparse(n.func) == "self.decoder" for n in ast.walk(node))
    ]
    joint_indices = [
        i
        for i, node in enumerate(body)
        if any(isinstance(n, ast.Call) and ast.unparse(n.func) == "self.joint" for n in ast.walk(node))
    ]
    if len(decoder_indices) != 1 or len(joint_indices) != 1 or joint_indices[0] - decoder_indices[0] != 3:
        raise ValueError("Inspected decoder/cast/joint step structure changed")
    first, last = decoder_indices[0], joint_indices[0]
    prefix = body[first : last + 1]
    if any(not isinstance(n, ast.Assign) for n in prefix):
        raise ValueError("Non-assignment inside decoder/joint step")
    if (
        ast.unparse(prefix[1]) != "decoder_out = decoder_out.astype(feature.dtype)"
        or ast.unparse(prefix[2]) != "decoder_hidden = (hidden.astype(feature.dtype), cell.astype(feature.dtype))"
    ):
        raise ValueError("Decoder dtype/state casts changed")
    dec_call = cast(ast.Assign, prefix[0]).value
    joint_call = cast(ast.Assign, prefix[-1]).value
    if (
        not isinstance(dec_call, ast.Call)
        or len(dec_call.args) != 2
        or dec_call.keywords
        or not isinstance(joint_call, ast.Call)
        or len(joint_call.args) != 2
        or joint_call.keywords
    ):
        raise ValueError("Decoder/joint call signature changed")
    if ast.unparse(joint_call.args[1]) != "decoder_out":
        raise ValueError("Joint predictor argument changed")
    for statement in body[:first] + body[last + 1 :]:
        if any(
            isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in {"decoder_out", "hidden", "cell"}
            for n in ast.walk(statement)
        ):
            raise ValueError("Intermediate step tensors escape the inspected prefix")
    body[first : last + 1] = [
        ast.Assign(
            [ast.Tuple([ast.Name("joint_out", ast.Store()), ast.Name("decoder_hidden", ast.Store())], ast.Store())],
            ast.Call(
                ast.Name("_stark_compiled_step", ast.Load()),
                [copy.deepcopy(joint_call.args[0]), *copy.deepcopy(dec_call.args)],
                [],
            ),
        )
    ]
    ast.fix_missing_locations(tree)
    namespace = {**original.__globals__, "_stark_compiled_step": step}
    # Trusted inspected function, bound to qualifying source and canonical AST
    # hashes; no caller-supplied code text or upstream source mutation.
    exec(compile(tree, original.__code__.co_filename + ":stark-compiled-step", "exec"), namespace)  # nosec B102
    candidate = namespace[original.__name__]
    candidate.__defaults__, candidate.__kwdefaults__ = original.__defaults__, original.__kwdefaults__
    return plain, candidate, {**canonical, "compiled_step_ast_sha256": _tree_hash(tree)}


class StepCompiler:
    """Only tensor steady state enters compile; counters/shape checks remain Python."""

    def __init__(self, model, mx, *, max_signatures=4, clock=time.perf_counter):
        decoder, join = model.decoder, model.joint
        self.clock = clock
        self.max_signatures = max_signatures
        self.calls = 0
        self.eager_bootstrap_calls = 0
        self.compiled_calls = 0
        self.failed_calls = 0
        self.signatures: dict[tuple, dict] = {}

        def eager(frame, token, state):
            decoded, (hidden, cell) = decoder(token, state)
            decoded = decoded.astype(frame.dtype)
            new_state = (hidden.astype(frame.dtype), cell.astype(frame.dtype))
            return join(frame, decoded), new_state

        def steady(frame, token, hidden, cell):
            # No scalar reads, mutation or timing in the compiled graph.
            return eager(frame, token, (hidden, cell))

        self.eager = eager
        started = clock()
        self.compiled = mx.compile(steady, shapeless=False)
        self.wrapper_creation_ms = (clock() - started) * 1000

    def __call__(self, frame, token, state):
        self.calls += 1
        if token is None or state is None:
            self.eager_bootstrap_calls += 1
            return self.eager(frame, token, state)
        if len(state) != 2:
            raise ValueError("Expected explicit hidden/cell pair")
        hidden, cell = state
        arrays = (frame, token, hidden, cell)
        shapes = tuple(tuple(value.shape) for value in arrays)
        if (
            shapes[0][:2] != (1, 1)
            or len(shapes[0]) != 3
            or shapes[1] != (1, 1)
            or len(shapes[2]) != 3
            or shapes[2] != shapes[3]
            or shapes[2][1] != 1
        ):
            raise ValueError("Only batch1, single-frame/token TDT steady state is supported")
        signature = tuple((shape, str(value.dtype)) for shape, value in zip(shapes, arrays, strict=True))
        first = signature not in self.signatures
        if first:
            if len(self.signatures) >= self.max_signatures:
                raise ValueError("Bounded compilation signature capacity exceeded")
            self.signatures[signature] = {
                "shapes": [list(s) for s in shapes],
                "dtypes": [str(v.dtype) for v in arrays],
                "first_dispatch_ms": None,
                "first_dispatch_failed": False,
            }
        self.compiled_calls += 1
        start = self.clock() if first else None
        try:
            return self.compiled(frame, token, hidden, cell)
        except Exception:
            self.failed_calls += 1
            if first:
                self.signatures[signature]["first_dispatch_failed"] = True
            raise  # Never silently substitute eager inference for a failed compile.
        finally:
            if first:
                self.signatures[signature]["first_dispatch_ms"] = (self.clock() - start) * 1000

    def snapshot(self):
        return {
            "calls": self.calls,
            "eager_bootstrap_calls": self.eager_bootstrap_calls,
            "compiled_calls": self.compiled_calls,
            "failed_calls": self.failed_calls,
            "signature_count": len(self.signatures),
            "signatures": list(copy.deepcopy(self.signatures).values()),
            "wrapper_creation_ms": self.wrapper_creation_ms,
            "cold_cost_scope": "First signature dispatch may include tracing/compilation but not deferred device execution; whole first compiled STT call below includes its completion. Neither is compiler-exclusive time.",
        }


def qualifying_joint(receipt, *, manifest_sha256, items):
    if receipt.get("completed") is not True or receipt.get("status") != "completed" or receipt.get("returncode") != 0:
        raise ValueError("Joint-evaluation receipt is incomplete")
    if receipt.get("manifest_sha256") != manifest_sha256 or receipt.get("items_sha256") != quality.json_hash(items):
        raise ValueError("Joint receipt input cohort differs")
    expected_items = {item["id"]: item for item in items}
    pairs = receipt.get("pairs", [])
    seen = set()
    for pair in pairs:
        item = expected_items.get(pair.get("id"))
        repeat = pair.get("repeat")
        if not item or type(repeat) is not int or repeat < 0 or (item["id"], repeat) in seen:
            raise ValueError("Invalid or duplicate joint pair identity")
        seen.add((item["id"], repeat))
        if (
            pair.get("source_lang") != item["source_lang"]
            or pair.get("partition") != "development"
            or pair.get("audio_sha256") != item["sha256"]
            or pair.get("audio_samples") != item["num_samples"]
        ):
            raise ValueError("Joint pair source identity differs")
        calls = pair.get("calls", {})
        if pair.get("completed") is not True or set(calls) != {"control", "joint_eval"}:
            raise ValueError("Incomplete joint pair")
        for mode, row in calls.items():
            if (
                row.get("mode") != mode
                or quality.json_hash(row.get("output")) != row.get("output_sha256")
                or row.get("stages") is not None
            ):
                raise ValueError("Joint output hash/mode or unprofiled control mismatch")
            if not math.isfinite(row.get("call_wall_ms", float("nan"))) or row["call_wall_ms"] <= 0:
                raise ValueError("Invalid joint wall time")
    repeats = {repeat for _, repeat in seen}
    if (
        not repeats
        or repeats != set(range(max(repeats) + 1))
        or seen != {(item_id, r) for item_id in expected_items for r in repeats}
    ):
        raise ValueError("Missing joint repeat/item coverage")
    assessment = joint.assess(pairs, receipt.get("expected_pairs", 0))
    if not assessment["positive_paired_median_without_pooled_tail_regression"]:
        raise ValueError("Joint experiment lacks complete exact-output and measured-gain evidence")
    if receipt.get("candidate_source_sha256") != quality.digest(Path(joint.__file__)) or receipt.get(
        "profiler_helper_sha256"
    ) != quality.digest(Path(profile.__file__)):
        raise ValueError("Qualified joint/helper implementation changed")
    return {
        "assessment": assessment,
        "greedy_source_sha256": receipt["candidate_method"]["original_source_sha256"],
        "joint_ast_sha256": receipt["candidate_method"]["transformed_ast_sha256"],
        "model_files_sha256": receipt["model"]["files_sha256"],
        "config": receipt["config"],
        "versions": {k: receipt["environment"]["versions"].get(k) for k in ("mlx", "parakeet-mlx", "numpy")},
    }


def triplet_assessment(triplets, expected):
    complete = len(triplets) == expected and expected > 0 and all(t.get("completed") for t in triplets)
    valid = [t for t in triplets if t.get("completed")]
    exact = complete and all(len({quality.json_hash(row["output"]) for row in t["calls"].values()}) == 1 for t in valid)
    comparison = {}
    for control in ("control", "joint_eval"):
        pairs = [
            {
                "id": t["id"],
                "repeat": t["repeat"],
                "completed": True,
                "calls": {"control": t["calls"][control], "joint_eval": t["calls"]["compiled_joint"]},
            }
            for t in valid
        ]
        comparison[control] = joint.assess(pairs, expected)
    return {
        "completed": complete,
        "expected_triplets": expected,
        "triplets": len(triplets),
        "all_three_outputs_exact": exact,
        "comparisons_to_compiled_joint": comparison,
        "changed_outputs": [
            {"id": t["id"], "repeat": t["repeat"]}
            for t in valid
            if len({quality.json_hash(row["output"]) for row in t["calls"].values()}) != 1
        ],
        "compile_exercised_calls": sum(
            t["calls"]["compiled_joint"].get("compile_stats_delta", {}).get("compiled_calls", 0) for t in valid
        ),
        "default_promotion": False,
        "interpretation": "Original and joint-evaluation controls are both unprofiled. Exact tokens/confidence/timestamps are required; fused floating-point operations may differ. Cold-containing triplets remain in primary metrics. Warm-only metrics are a separate conditional description, never a replacement.",
    }


def call(engine, item, audio, mode, plain, compiled, steps, audio_module):
    before = steps.snapshot()
    if mode == "control":
        result = profile.one_call(engine, item, audio, profiled=False, sample_every=1, audio_module=audio_module)
    else:
        with joint.install_joint_method(engine._model, plain if mode == "joint_eval" else compiled):
            result = profile.one_call(engine, item, audio, profiled=False, sample_every=1, audio_module=audio_module)
    result["mode"] = mode
    after = steps.snapshot()
    result["compile_stats_delta"] = {
        key: after[key] - before[key]
        for key in ("calls", "compiled_calls", "eager_bootstrap_calls", "failed_calls", "signature_count")
    }
    result["contains_first_compiled_signature"] = after["signature_count"] > before["signature_count"]
    return result


def worker(args):
    from tools.benchmark_identity import load_primary_model

    items = profile.selected(quality.read_json(args.manifest), args.languages, args.limit)
    receipt = quality.read_json(args.joint_receipt)
    if quality.digest(args.joint_receipt) != args.joint_sha256:
        raise ValueError("Joint receipt changed")
    gate = qualifying_joint(receipt, manifest_sha256=quality.digest(args.manifest), items=items)
    expected = len(items) * args.repeats
    data = {
        "schema_version": 1,
        "completed": False,
        "status": "starting",
        "triplets": [],
        "started_at": datetime.now(UTC).isoformat(),
        "manifest_sha256": quality.digest(args.manifest),
        "items_sha256": quality.json_hash(items),
        "joint_receipt_sha256": args.joint_sha256,
        "qualifying_joint": gate,
        "candidate_source_sha256": quality.digest(Path(__file__)),
        "source_lock": SOURCE_LOCK,
        "compile_docs": COMPILE_DOCS,
        "environment": quality.source_identity(),
        "expected_triplets": expected,
        "human_approved_locally": False,
        "training_eligible": False,
        "defaults_changed": False,
        "compilation_contract": {
            "scope": "tensor-only decoder/dtype casts/joint steady-state step",
            "shapeless": False,
            "bootstrap": "token=None or state=None stays eager",
            "weights": "immutable within one isolated model lifetime",
            "graph_state": "hidden/cell are explicit tensor arguments and returned outputs; Python hypothesis/state commits stay outside compile",
            "controls": "original; canonical joint scalar eval; compiled decoder/joint plus same scalar eval",
            "max_signatures": 4,
        },
        "pairing": "Three-mode Latin rotation by repeat+item index. All calls including first compile are retained; no warmup evaluation items omitted.",
    }
    quality.save(args.output, data)
    engine = steps = None
    try:
        if os.environ.get("MLX_DISABLE_COMPILE", "").strip().lower() not in {"", "0", "false"}:
            raise ValueError("MLX_DISABLE_COMPILE would invalidate this arm")
        if any(
            value is None or data["environment"]["versions"].get(key) != value
            for key, value in gate["versions"].items()
        ):
            raise ValueError("Qualified MLX/Parakeet/NumPy versions changed")
        config = gate["config"]
        if config != quality.engine_config("parakeet-mlx", "stt", config["requested_model_id"]):
            raise ValueError("Qualified engine contract changed")
        inventory = quality.model_inventory(config)
        if inventory["files_sha256"] != gate["model_files_sha256"]:
            raise ValueError("Qualified model files changed")
        data.update(config=config, model=inventory)
        engine = quality.make_engine(config, inventory["resolved_path"], "en")
        data["model_identity"] = load_primary_model(engine, inventory["resolved_path"])
        import mlx.core as mx
        import mlx.nn.layers.recurrent as recurrent
        import parakeet_mlx.audio as audio_module
        import parakeet_mlx.parakeet as upstream
        import parakeet_mlx.rnnt as rnnt

        paths = {
            "parakeet_mlx/parakeet.py": upstream.__file__,
            "parakeet_mlx/rnnt.py": rnnt.__file__,
            "mlx/nn/layers/recurrent.py": recurrent.__file__,
        }
        if type(engine._model).__name__ != "ParakeetTDT" or any(
            quality.digest(Path(path)) != SOURCE_LOCK[name] for name, path in paths.items()
        ):
            raise ValueError("Inspected upstream decoder/joint/LSTM source or model class changed")
        data["upstream_files"] = {
            name: {"path": str(Path(path).resolve()), "sha256": SOURCE_LOCK[name]} for name, path in paths.items()
        }
        steps = StepCompiler(engine._model, mx)
        plain, compiled, identity = compiled_method(
            type(engine._model).decode_greedy, gate["greedy_source_sha256"], steps
        )
        if identity["transformed_ast_sha256"] != gate["joint_ast_sha256"]:
            raise ValueError("Qualified joint AST changed")
        data["candidate_method"] = identity
        data["status"] = "running"
        for repeat in range(args.repeats):
            for index, item in enumerate(items):
                rotation = (repeat + index) % len(MODES)
                order = MODES[rotation:] + MODES[:rotation]
                triplet = {
                    "id": item["id"],
                    "source_lang": item["source_lang"],
                    "partition": "development",
                    "repeat": repeat,
                    "audio_sha256": item["sha256"],
                    "audio_samples": item["num_samples"],
                    "order": list(order),
                    "calls": {},
                    "completed": False,
                }
                try:
                    audio = quality.read_audio(quality.relative_file(args.audio_root, item["path"]), item)
                    for mode in order:
                        attempt_started = time.perf_counter()
                        attempt_before = steps.snapshot()
                        try:
                            triplet["calls"][mode] = call(
                                engine, item, audio, mode, plain, compiled, steps, audio_module
                            )
                        except Exception as exc:
                            after_failure = steps.snapshot()
                            triplet.setdefault("mode_errors", {})[mode] = {
                                "error": f"{type(exc).__name__}: {exc}",
                                "failed_attempt_wall_ms": (time.perf_counter() - attempt_started) * 1000,
                                "compile_stats_delta": {
                                    k: after_failure[k] - attempt_before[k]
                                    for k in ("compiled_calls", "failed_calls", "signature_count")
                                },
                                "contains_first_compiled_signature": after_failure["signature_count"]
                                > attempt_before["signature_count"],
                            }
                    triplet["completed"] = set(triplet["calls"]) == set(MODES)
                    if triplet["completed"]:
                        triplet["all_three_outputs_exact"] = (
                            len({row["output_sha256"] for row in triplet["calls"].values()}) == 1
                        )
                except Exception as exc:
                    triplet["error"] = f"{type(exc).__name__}: {exc}"
                data["triplets"].append(triplet)
                data["compile_stats"] = steps.snapshot()
                data["assessment"] = triplet_assessment(data["triplets"], expected)
                data["language_assessments"] = {
                    lang: triplet_assessment(
                        [t for t in data["triplets"] if t["source_lang"] == lang],
                        sum(i["source_lang"] == lang for i in items) * args.repeats,
                    )
                    for lang in args.languages
                }
                data["repeat_assessments"] = {
                    str(r): triplet_assessment([t for t in data["triplets"] if t["repeat"] == r], len(items))
                    for r in range(args.repeats)
                }
                cold = [
                    t
                    for t in data["triplets"]
                    if (
                        t.get("calls", {}).get("compiled_joint") or t.get("mode_errors", {}).get("compiled_joint", {})
                    ).get("contains_first_compiled_signature")
                ]
                warm = [
                    t
                    for t in data["triplets"]
                    if t.get("completed") and not t["calls"]["compiled_joint"]["contains_first_compiled_signature"]
                ]
                data["cold_containing_triplets"] = [
                    {"id": t["id"], "repeat": t["repeat"], "calls": t["calls"], "mode_errors": t.get("mode_errors", {})}
                    for t in cold
                ]
                data["warm_only_assessment"] = {
                    **triplet_assessment(warm, len(warm)),
                    "conditional_subset": True,
                    "parent_inventory_complete": data["assessment"]["completed"],
                    "comparison_eligible": data["assessment"]["all_three_outputs_exact"],
                }
                quality.save(args.output, data)
        data["completed"] = data["assessment"]["completed"]
        data["status"] = "completed" if data["completed"] else "failed_triplets"
    except Exception as exc:
        data.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if steps is not None:
            data["compile_stats"] = steps.snapshot()
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
    for name in ("manifest", "audio-root", "joint-receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--joint-sha256", required=True)
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
        parser.error("Positive repeat/timeout and unique languages required")
    if quality.digest(args.joint_receipt) != args.joint_sha256:
        parser.error("Joint receipt SHA256 mismatch")
    items = profile.selected(quality.read_json(args.manifest), args.languages, args.limit)
    qualifying_joint(quality.read_json(args.joint_receipt), manifest_sha256=quality.digest(args.manifest), items=items)
    if args._worker:
        return 0 if worker(args) else 1
    from tools.mac_evaluation import _run_evaluation_worker

    result = _run_evaluation_worker(
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--_worker"],
        args.output,
        args.manifest,
        env=quality.clean_environment(),
        timeout=args.timeout_seconds,
    )
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
