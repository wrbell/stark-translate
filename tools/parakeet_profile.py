"""Offline, isolated Parakeet TDT scalar-readback profiler; never imports models at startup.

Instrumentation exists only in the child process and is restored after each call.
No library files or production engines are changed. Readback wall time includes
lazy MLX dependencies, not merely Python conversion or synchronization overhead.
Both profiling arms always use stock greedy decoding (the instrumented arm parses the stock
source); --baseline-decode is accepted for compatibility and changes nothing.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import math
import sys
import textwrap
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import mac_followup_quality as quality

READBACK_LABELS = ("token_readback", "confidence_readback", "duration_readback")


class Probe:
    """Measure selected calls without adding eval/synchronize or changing arguments."""

    def __init__(self, sample_every=4, max_samples=10000, clock=time.perf_counter):
        if sample_every < 1 or max_samples < 1:
            raise ValueError("Sampling cadence and capacity must be positive")
        self.sample_every = sample_every
        self.max_samples = max_samples
        self.clock = clock
        self.counts = defaultdict(int)
        self.samples = defaultdict(list)
        self.truncated = defaultdict(int)

    def call(self, label, function, *args, **kwargs):
        self.counts[label] += 1
        selected = (self.counts[label] - 1) % self.sample_every == 0
        if not selected:
            return function(*args, **kwargs)
        if len(self.samples[label]) >= self.max_samples:
            self.truncated[label] += 1
            return function(*args, **kwargs)
        start = self.clock()
        try:
            return function(*args, **kwargs)
        finally:
            self.samples[label].append((self.clock() - start) * 1000)

    def summary(self):
        return {
            label: {
                "calls": count,
                "sample_every": self.sample_every,
                "sampled_wall_ms": quality.distribution(self.samples[label]),
                "sampled_wall_sum_ms": sum(self.samples[label]),
                "timing_samples_ms": self.samples[label],
                "samples_omitted_by_capacity": self.truncated[label],
            }
            for label, count in sorted(self.counts.items())
        }


class Instrument(ast.NodeTransformer):
    """Instrument only recognized installed assignments; reject upstream drift."""

    def __init__(self, kind):
        self.kind = kind
        self.counts = defaultdict(int)

    def wrap(self, node, label):
        self.counts[label] += 1
        return ast.copy_location(
            ast.Call(
                func=ast.Attribute(value=ast.Name(id="_stark_probe", ctx=ast.Load()), attr="call", ctx=ast.Load()),
                args=[ast.Constant(label), node.func, *node.args],
                keywords=node.keywords,
            ),
            node,
        )

    def visit_Call(self, node):
        node = self.generic_visit(node)
        name = ast.unparse(node.func)
        labels = (
            {"self.decoder": "decoder_graph", "self.joint": "joint_graph"}
            if self.kind == "greedy"
            else {"self.encoder": "encoder_graph", "mx.eval": "encoder_materialization", "self.decode": "decode_total"}
        )
        return self.wrap(node, labels[name]) if name in labels else node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        labels = {
            "pred_token": ("int", "token_readback"),
            "confidence": ("float", "confidence_readback"),
            "decision": ("int", "duration_readback"),
        }
        if self.kind == "greedy" and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in labels:
                converter, label = labels[name]
                if (
                    not isinstance(node.value, ast.Call)
                    or ast.unparse(node.value.func) != converter
                    or len(node.value.args) != 1
                ):
                    raise ValueError(f"Installed scalar assignment changed: {name}")
                if name in {"pred_token", "decision"} and not (
                    isinstance(node.value.args[0], ast.Call) and ast.unparse(node.value.args[0].func) == "mx.argmax"
                ):
                    raise ValueError(f"Installed argmax assignment changed: {name}")
                node.value = self.wrap(node.value, label)
        return node


def instrument_function(original, probe: Probe, kind: str):
    """Clone a trusted, already-installed function in memory; never execute supplied text."""
    if kind not in {"greedy", "generate"} or original.__closure__:
        raise ValueError("Expected supported non-closure installed method")
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].decorator_list:
        raise ValueError("Expected one undecorated installed function")
    transform = Instrument(kind)
    tree = transform.visit(tree)
    expected = {"token_readback", "confidence_readback", "duration_readback", "decoder_graph", "joint_graph"}
    if kind == "generate":
        expected = {"encoder_graph", "encoder_materialization", "decode_total"}
    if set(transform.counts) != expected or any(n != 1 for n in transform.counts.values()):
        raise ValueError(f"Installed {kind} structure changed: {dict(transform.counts)}")
    ast.fix_missing_locations(tree)
    namespace = {**original.__globals__, "_stark_probe": probe}
    # The AST comes exclusively from inspect.getsource(already imported method),
    # shape-checked above. No user-supplied source, string or download is executed.
    exec(compile(tree, original.__code__.co_filename + ":stark-profile", "exec"), namespace)  # nosec B102
    cloned = namespace[original.__name__]
    cloned.__defaults__, cloned.__kwdefaults__ = original.__defaults__, original.__kwdefaults__
    return cloned, {
        "function": original.__qualname__,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "source_hash_encoding": "SHA256 of dedented inspect.getsource UTF-8 bytes",
        "source_file": original.__code__.co_filename,
        "instrumented_sites": dict(transform.counts),
    }


@contextmanager
def observe(model, *, probe=None, audio_module=None):
    """Use the same minimal result-capture hook in both control/profile calls."""
    cls = type(model)
    original_generate = cls.generate
    original_greedy = cls.decode_greedy
    old_generate = cls.__dict__.get("generate")
    old_greedy = cls.__dict__.get("decode_greedy")
    raw: list = []
    identities = []
    old_logmel = audio_module.get_logmel if audio_module else None
    generated = original_generate
    greedy = original_greedy
    if probe is not None:
        greedy, ident = instrument_function(original_greedy, probe, "greedy")
        identities.append(ident)
        generated, ident = instrument_function(original_generate, probe, "generate")
        identities.append(ident)

    def capture(self, *args, **kwargs):
        value = generated(self, *args, **kwargs)
        raw[:] = value  # retain objects only; serialization is outside timed STT.
        return value

    try:
        cls.generate = capture
        if probe is not None:
            cls.decode_greedy = greedy
            if audio_module:

                def logmel(*args, **kwargs):
                    return probe.call("logmel_graph", old_logmel, *args, **kwargs)

                audio_module.get_logmel = logmel
        yield raw, identities
    finally:
        if old_generate is None:
            delattr(cls, "generate")
        else:
            cls.generate = old_generate
        if probe is not None:
            if old_greedy is None:
                delattr(cls, "decode_greedy")
            else:
                cls.decode_greedy = old_greedy
        if audio_module:
            audio_module.get_logmel = old_logmel


def raw_outputs(results):
    return [
        {
            "text": result.text,
            "sentences": [
                {
                    "text": sentence.text,
                    "start": sentence.start,
                    "end": sentence.end,
                    "duration": sentence.duration,
                    "confidence": sentence.confidence,
                    "tokens": [
                        {k: getattr(token, k) for k in ("id", "text", "start", "end", "duration", "confidence")}
                        for token in sentence.tokens
                    ],
                }
                for sentence in result.sentences
            ],
        }
        for result in results
    ]


def one_call(engine, item, audio, *, profiled: bool, sample_every: int, audio_module) -> dict:
    probe = Probe(sample_every) if profiled else None
    with observe(engine._model, probe=probe, audio_module=audio_module) as (raw, identities):
        start = time.perf_counter()
        result = engine.transcribe(audio, language=item["source_lang"], initial_prompt=None, word_timestamps=False)
        wall = (time.perf_counter() - start) * 1000
    converted = asdict(result)
    converted.pop("latency_ms")
    output = {"stt": converted, "aligned_results": raw_outputs(raw)}
    return {
        "mode": "profiled" if profiled else "control",
        "call_wall_ms": wall,
        "engine_latency_ms": result.latency_ms,
        "output": output,
        "output_sha256": quality.json_hash(output),
        "stages": probe.summary() if probe else None,
        "instrumented_functions": identities,
        "memory": quality.memory_snapshot(),
    }


def assess_pairs(pairs: list[dict], expected: int) -> dict:
    valid = [p for p in pairs if p.get("completed")]
    complete = len(valid) == len(pairs) == expected and expected > 0
    equivalence = complete and all(p["exact_output_equivalence"] for p in valid)
    shares, overhead, readback = [], [], []
    capacity_complete = True
    for pair in valid:
        control, profile = pair["calls"]["control"], pair["calls"]["profiled"]
        scalar_ms = sum(profile["stages"].get(label, {}).get("sampled_wall_sum_ms", 0) for label in READBACK_LABELS)
        readback.append(scalar_ms)
        wall = profile["call_wall_ms"]
        shares.append(scalar_ms / wall if wall else 0)
        baseline = control["call_wall_ms"]
        overhead.append((wall / baseline - 1) * 100 if baseline else None)
        capacity_complete &= all(not stage["samples_omitted_by_capacity"] for stage in profile["stages"].values())
    # Only actually timed readbacks count, never a projected mean*all-calls estimate.
    qualifies = equivalence and capacity_complete and bool(shares) and statistics_median(shares) >= 0.10
    return {
        "completed": complete,
        "all_outputs_exact": equivalence,
        "pairs": len(pairs),
        "expected_pairs": expected,
        "sampled_readback_share_of_profiled_stt_wall": quality.distribution(shares),
        "sampled_readback_wall_ms": quality.distribution(readback),
        "profile_vs_control_wall_percent": quality.distribution(overhead),
        "sample_capacity_complete": capacity_complete,
        "joint_evaluation_candidate_allowed": qualifies,
        "joint_evaluation_candidate": "not implemented; measured conditional follow-up only",
        "compile_experiment": "deferred until a joint-evaluation candidate has separate equivalent-output evidence",
        "rule": "All paired outputs exact, all calls complete, no sample overflow, and median actually sampled scalar wall / profiled STT wall >= 10%. No extrapolated samples or speedup claim.",
        "interpretation": "Scalar readbacks materialize pending MLX work. This includes decoder/joint/entropy dependencies, not just host conversion. Stage intervals are nested; never sum all stages.",
    }


def statistics_median(values):
    import statistics

    return statistics.median(values)


def selected(manifest, languages, limit):
    quality.validate_portable(manifest)
    if limit < 1 or limit > 50:
        raise ValueError("Choose 1..50 development records per language")
    return [
        r for lang in languages for r in quality.selected_items(manifest, "stt", "development", lang, False)[:limit]
    ]


def profile_worker(args):
    from tools.benchmark_identity import load_primary_model

    manifest = quality.read_json(args.manifest)
    items = selected(manifest, args.languages, args.limit)
    expected = len(items) * args.repeats
    data = {
        "schema_version": 1,
        "completed": False,
        "pairs": [],
        "manifest_sha256": quality.digest(args.manifest),
        "items_sha256": quality.json_hash(items),
        "expected_pairs": expected,
        "sample_every": args.sample_every,
        "started_at": datetime.now(UTC).isoformat(),
        "environment": quality.source_identity(),
        "profiler_source_sha256": quality.digest(Path(__file__)),
        "status": "starting",
        "control_contract": "Production engine transcribe path, scalar/stage instrumentation disabled. A minimal generate result-retention hook is shared with profiled calls so raw token/timestamp/confidence equivalence can be checked outside timing.",
        "pairing": "Same loaded model and exact input; control/profile order alternates by pair; each call retains production synchronization; no added eval/synchronize.",
        "human_approved_locally": False,
        "training_eligible": False,
    }
    quality.save(args.output, data)
    engine = None
    try:
        config = quality.engine_config("parakeet-mlx", "stt", args.model_override)
        inventory = quality.model_inventory(config)
        data.update(config=config, model=inventory)
        # The instrumented arm parses and re-installs the stock greedy source, so the
        # profiler always loads the engine with the qualified joint decode disabled.
        from engines.parakeet_mlx_engine import ParakeetMLXEngine

        engine = ParakeetMLXEngine(model_id=inventory["resolved_path"], joint_scalar_eval=False)
        data["baseline_decode"] = True
        data["model_identity"] = load_primary_model(engine, inventory["resolved_path"])
        import parakeet_mlx.audio as audio_module
        import parakeet_mlx.parakeet as upstream

        data["upstream_implementation"] = {
            "file": str(Path(upstream.__file__).resolve()),
            "sha256": quality.digest(Path(upstream.__file__)),
            "class": type(engine._model).__qualname__,
        }
        data["status"] = "running"
        for repeat in range(args.repeats):
            for item in items:
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
                order = [False, True] if len(data["pairs"]) % 2 == 0 else [True, False]
                pair["order"] = ["profiled" if mode else "control" for mode in order]
                try:
                    audio = quality.read_audio(quality.relative_file(args.audio_root, item["path"]), item)
                    for mode in order:
                        call = one_call(
                            engine,
                            item,
                            audio,
                            profiled=mode,
                            sample_every=args.sample_every,
                            audio_module=audio_module,
                        )
                        pair["calls"][call["mode"]] = call
                    pair["exact_output_equivalence"] = (
                        pair["calls"]["control"]["output"] == pair["calls"]["profiled"]["output"]
                    )
                    pair["completed"] = True
                except Exception as exc:
                    pair["error"] = f"{type(exc).__name__}: {exc}"
                data["pairs"].append(pair)
                data["assessment"] = assess_pairs(data["pairs"], expected)
                data["language_assessments"] = {
                    lang: assess_pairs(
                        [p for p in data["pairs"] if p["source_lang"] == lang],
                        sum(i["source_lang"] == lang for i in items) * args.repeats,
                    )
                    for lang in args.languages
                }
                quality.save(args.output, data)
        data["assessment"] = assess_pairs(data["pairs"], expected)
        data["completed"] = data["assessment"]["completed"]
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--audio-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--languages", nargs="+", choices=["en", "es"], default=["en", "es"])
    p.add_argument("--limit", type=int, default=3, help="First N frozen development records per language")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--sample-every", type=int, default=4)
    p.add_argument("--timeout-seconds", type=float, default=1800)
    p.add_argument("--model-override")
    p.add_argument(
        "--baseline-decode", action="store_true", help="Accepted for compatibility; stock decode is always used"
    )
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if (
        args.repeats < 1
        or args.sample_every < 1
        or not math.isfinite(args.timeout_seconds)
        or args.timeout_seconds <= 0
    ):
        p.error("Positive repeat/sampling counts and finite positive timeout required")
    if len(set(args.languages)) != len(args.languages):
        p.error("Languages must be unique")
    selected(quality.read_json(args.manifest), args.languages, args.limit)
    if args._worker:
        return 0 if profile_worker(args) else 1
    from tools.mac_evaluation import _run_evaluation_worker

    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--_worker"]
    result = _run_evaluation_worker(
        command, args.output, args.manifest, env=quality.clean_environment(), timeout=args.timeout_seconds
    )
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
