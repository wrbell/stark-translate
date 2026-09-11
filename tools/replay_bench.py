#!/usr/bin/env python3
"""Prepare replay clips and benchmark the live pipeline, sequentially.

Importing this module only loads standard-library code; models are loaded only
by explicitly launched dry_run_ab subprocesses. Config JSON maps names to argv
strings or lists, e.g. {"baseline": [], "mts": ["--mts"]}.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRIC_COLUMNS = (
    "speech_end_to_final_ms",
    "vad_wait_ms",
    "stt_queue_wait_ms",
    "translation_queue_wait_ms",
    "finalization_overhead_ms",
    "broadcast_ms",
    "true_e2e_ms",
    "e2e_latency_ms",
    "stt_latency_ms",
    "latency_a_ms",
    "silence_delay_ms",
    "queue_wait_ms",
    "marian_pt_ms",
    "gen_tokens_a",
    "prefill_ms_a",
    "ttft_ms_a",
    "decode_ms_a",
)
SPECIAL_TOKENS = ("<turn|>", "<|channel>")
REPLAY_MANAGED_ARGS = {
    "--audio-file",
    "--session-id",
    "--ws-port",
    "--http-port",
    "--lang",
    "--dry-run-text",
    "--no-exit-after-replay",
    "--backend",
}


def argument_value(arguments: list[str], name: str, default=None):
    """Match argparse's final occurrence, including --option=value syntax."""
    result = default
    for index, argument in enumerate(arguments):
        if argument.startswith(name + "="):
            result = argument.split("=", 1)[1]
        elif argument == name and index + 1 < len(arguments):
            result = arguments[index + 1]
    return result


def run_child(command: list[str], *, cwd: Path, stdout, timeout: float, env: dict | None = None):
    """Bound a worker's lifetime and terminate its process group on timeout."""
    process = subprocess.Popen(
        command, cwd=cwd, stdout=stdout, stderr=subprocess.STDOUT, env=env, start_new_session=os.name != "nt"
    )
    try:
        return subprocess.CompletedProcess(command, process.wait(timeout=timeout))
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait(timeout=5)
        finally:
            # A descendant can ignore TERM even when the main process exits.
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        raise


def read_lifecycle(metrics_dir: Path, session_id: str) -> dict:
    """Read matching source/peak metadata, without claiming completion from it."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
        return {}
    try:
        record = json.loads((metrics_dir / f"session_lifecycle_{session_id}.json").read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(record, dict) or record.get("session_id") != session_id or record.get("schema_version") != 1:
        return {}
    return record


def _pct(xs: list[float], p: float) -> float:
    """Nearest-rank percentile; the reported p50 separately uses the median."""
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, math.ceil((p / 100.0) * len(s)) - 1))
    return s[idx]


def _number(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def summarize(values: list[float]) -> dict:
    """Missing measurements remain null, rather than implying zero latency."""
    return {
        "n": len(values),
        "p50": statistics.median(values) if values else None,
        "p95": _pct(values, 95) if values else None,
        "mean": statistics.mean(values) if values else None,
    }


def parse_csv(path: Path | str) -> tuple[list[dict], dict]:
    """Read final chunks and summarize finite numeric values in known columns."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)
    metrics = {
        column: summarize([n for row in rows if (n := _number(row.get(column))) is not None])
        for column in METRIC_COLUMNS
        if column in columns
    }
    return rows, metrics


def parse_partials(path: Path | str) -> tuple[list[dict], dict]:
    """Read every emitted partial, including repeated updates to an utterance."""
    path = Path(path)
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    totals = []
    for record in records:
        stt, marian = _number(record.get("stt_ms")), _number(record.get("marian_ms"))
        if stt is not None and marian is not None:
            totals.append(stt + marian)
    return records, summarize(totals)


def parse_overlap(log: str) -> float | None:
    matches = re.findall(r"\[P7-6C\] Pipeline stats:[^\n]*?\(([\d.]+)%\)", log)
    return float(matches[-1]) if matches else None


def analyze_run(csv_path: Path | str, partials_path: Path | str, log_path: Path | str | None = None) -> dict:
    """Return latency distributions, routing proxy and generated-text counts.

    A zero tps_a marks Marian-only finals in the current pipeline. Report the
    denominator explicitly: missing throughput is unknown, not Marian-only.
    Each STT/translation text field counts as one output for token leakage.
    """
    rows, metrics = parse_csv(csv_path)
    partials, partial_stats = parse_partials(partials_path)
    metrics["partial_total_ms"] = partial_stats
    tps = [n for row in rows if (n := _number(row.get("tps_a"))) is not None]
    texts = [row.get(key) or "" for row in rows for key in ("english", "spanish_a", "spanish_b")]
    texts += [row.get(key) or "" for row in partials for key in ("text_en", "text_es")]
    log = Path(log_path).read_text(errors="replace") if log_path is not None and Path(log_path).exists() else ""
    return {
        "chunk_count": len(rows),
        "partial_count": len(partials),
        "percentile_method": "median_p50_nearest_rank_p95",
        "metrics": metrics,
        "timing_schema_versions": sorted({row.get("timing_schema_version") or "legacy" for row in rows}),
        "timing_sources": sorted({row.get("timing_source") or "unknown" for row in rows}),
        "metric_definitions": {
            "speech_end_to_final_ms": "last VAD-positive capture frame end to final payload ready; not browser rendering",
            "e2e_latency_ms": "legacy queue submission to translation/QE completion",
            "true_e2e_ms": "legacy first speech observation to translation/QE completion",
            "silence_delay_ms": "legacy first speech observation to queue submission (includes speaking)",
            "partial_total_ms": "STT plus Marian compute only; excludes cadence, queue and delivery",
        },
        "speech_end_by_endpoint": {
            reason: summarize(
                [
                    n
                    for row in rows
                    if row.get("endpoint_reason") == reason
                    and (n := _number(row.get("speech_end_to_final_ms"))) is not None
                ]
            )
            for reason in sorted({row.get("endpoint_reason") for row in rows if row.get("endpoint_reason")})
        },
        "partial_capture_to_ready_ms": summarize(
            [n for row in partials if (n := _number(row.get("captured_end_to_partial_ms"))) is not None]
        ),
        "marian_only_share": sum(n == 0 for n in tps) / len(tps) if tps else None,
        "marian_only_observations": len(tps),
        "marian_only_source": "tps_a == 0 proxy" if tps else None,
        "special_token_outputs": sum(any(token in text for token in SPECIAL_TOKENS) for text in texts),
        "special_token_counts": {token: sum(token in text for text in texts) for token in SPECIAL_TOKENS},
        "overlap_pct": parse_overlap(log),
    }


def delta_table(current: dict, baseline: dict) -> str:
    """Compare compatible replay observations, keeping speech endpoints separate."""
    for label, report in (("baseline", baseline), ("current", current)):
        if report.get("returncode") != 0 or report.get("error"):
            raise ValueError(f"Cannot compare {label}: replay is failed or lacks successful completion metadata")
        if _number(report.get("replay_speed")) != 1 or report.get("realtime_latency_eligible") is False:
            raise ValueError(f"Cannot compare {label}: latency comparison requires real-time (1x) replay")
        if report.get("percentile_method") != "median_p50_nearest_rank_p95":
            raise ValueError(f"Cannot compare {label}: rebuild percentile summaries from saved CSVs with this analyzer")
        clip = report.get("clip", {})
        audio_hash = clip.get("sha256")
        if (
            not isinstance(audio_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", audio_hash)
            or clip.get("lang") not in {"en", "es"}
        ):
            raise ValueError(f"Cannot compare {label}: frozen audio hash and language are required")
        metadata = report.get("session_metadata", {})
        if (
            metadata.get("source_lang", clip["lang"]) != clip["lang"]
            or metadata.get("input_audio_sha256", clip["sha256"]) != clip["sha256"]
        ):
            raise ValueError(f"Cannot compare {label}: actual input metadata contradicts the frozen clip")
        if not report.get("timing_schema_versions") or not report.get("timing_sources"):
            raise ValueError(f"Cannot compare {label}: timing schema and capture-source metadata are required")
    for key in ("sha256", "lang"):
        if current["clip"][key] != baseline["clip"][key]:
            raise ValueError(f"Cannot compare different replay audio {key}")
    for key in ("timing_schema_versions", "timing_sources", "metric_definitions"):
        if current.get(key) != baseline.get(key):
            raise ValueError(f"Cannot compare incompatible {key}")

    # Model/experiment settings may intentionally differ. Keep them explicit,
    # while rejecting changed or unknown pipeline implementation identities.
    hashes = [r.get("session_lifecycle", {}).get("pipeline_sha256") for r in (baseline, current)]
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes):
        raise ValueError("Cannot compare unknown pipeline source hashes; use recorded runtime cohorts")
    if hashes[0] != hashes[1]:
        raise ValueError("Cannot compare different pipeline source hashes; use separate runtime cohorts")
    config_changes = []
    for key in ("backend", "model_family", "model_a", "stt_backend", "vad", "translation"):
        before = baseline.get("session_metadata", {}).get(key)
        after = current.get("session_metadata", {}).get(key)
        if before != after:
            config_changes.append(key)

    def flatten(report):
        result = {}
        for metric, stats in report.get("metrics", {}).items():
            if metric == "speech_end_to_final_ms":
                continue  # Silence and forced cuts have different meanings.
            for stat in ("p50", "p95", "mean"):
                result[f"{metric}.{stat}"] = stats.get(stat)
        for endpoint, stats in report.get("speech_end_by_endpoint", {}).items():
            for stat in ("p50", "p95", "mean"):
                result[f"speech_end_to_final_ms[{endpoint}].{stat}"] = stats.get(stat)
        for metric in ("chunk_count", "partial_count", "marian_only_share", "special_token_outputs", "overlap_pct"):
            result[metric] = report.get(metric)
        return result

    old, new = flatten(baseline), flatten(current)
    lines = [
        "Configuration fields changed: " + (", ".join(config_changes) if config_changes else "none recorded") + ".",
        "",
        "| Metric | Baseline | Replay | Delta | Delta % |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in sorted(old.keys() | new.keys()):
        a, b = _number(old.get(key)), _number(new.get(key))
        delta = b - a if a is not None and b is not None else None
        pct = 100 * delta / a if delta is not None and a != 0 else None
        cells = ["—" if n is None else f"{n:.2f}" for n in (a, b, delta, pct)]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_configs(specs: list[str], json_path: Path | None = None) -> dict[str, list[str]]:
    configs = {}
    if json_path is not None:
        configs.update(json.loads(json_path.read_text()))
    for spec in specs:
        if "=" not in spec:
            configs.update(json.loads(Path(spec).read_text()))
        else:
            name, argv = spec.split("=", 1)
            configs[name] = argv
    if not configs:
        configs["baseline"] = []
    for name, argv in configs.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Invalid config name: {name!r}")
        args = shlex.split(argv) if isinstance(argv, str) else argv
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise ValueError(f"Config {name} must be an argv string or list")
        if any(arg.split("=", 1)[0] in REPLAY_MANAGED_ARGS for arg in args):
            raise ValueError(f"Config {name} overrides replay-managed arguments")
        configs[name] = args
    return configs


def prepare_clips(raw_dir: Path, replay_dir: Path, seconds: float = 300, offset: float = 0.0) -> Path:
    """Cut local WAVs (from ``offset`` s) without changing sample rate, channels or PCM dtype."""
    from scipy.io import wavfile

    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    if not math.isfinite(offset) or offset < 0:
        raise ValueError("offset must be finite and nonnegative")
    sources = sorted(raw_dir.glob("Gospel_Message_*.wav"))
    spanish = raw_dir / "spanish_test_2cor1.wav"
    if spanish.exists():
        sources.append(spanish)
    if not sources:
        raise FileNotFoundError(f"No replay source WAVs in {raw_dir}")
    replay_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for source in sources:
        rate, samples = wavfile.read(source)
        clip_offset = 0.0 if source == spanish else offset
        start = min(int(clip_offset * rate), len(samples))
        cut = samples[start : start + int(seconds * rate)]
        target = replay_dir / source.name
        wavfile.write(target, rate, cut)
        clips.append(
            {
                "path": target.name,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "duration": len(cut) / rate,
                "lang": "es" if source == spanish else "en",
                "offset_s": offset if source != spanish else 0.0,
            }
        )
    manifest = replay_dir / "manifest.json"
    manifest.write_text(json.dumps({"clips": clips}, indent=2) + "\n")
    return manifest


def _free_port(start: int) -> int:
    """First TCP port >= start that binds on loopback (other apps may squat on 87xx)."""
    import socket

    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free port in {start}..{start + 200}")


def run_replay(
    clip: dict,
    wav: Path,
    tag: str,
    extra: list[str],
    index: int,
    metrics_dir: Path,
    *,
    timeout_s: float | None = None,
    env: dict | None = None,
    ports: tuple[int, int] | None = None,
    backend: str = "mlx",
) -> dict:
    """Launch one child and wait before analyzing its flushed metrics."""
    if backend not in {"mlx", "cpu"}:
        raise ValueError("Replay backend must be mlx or cpu")
    if any(arg.split("=", 1)[0] in REPLAY_MANAGED_ARGS for arg in extra):
        raise ValueError("Extra args override replay-managed arguments")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("Invalid replay session tag")
    speed = float(argument_value(extra, "--replay-speed", os.environ.get("STARK_REPLAY_SPEED", "1")))
    if not math.isfinite(speed):
        raise ValueError("Replay speed must be finite")
    timeout_s = (
        timeout_s
        if timeout_s is not None
        else max(300, float(clip.get("duration", 150)) / (speed if speed > 0 else 1) * 3 + 180)
    )
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Replay timeout must be positive and finite")
    if ports is not None:
        if len(set(ports)) != 2 or any(not isinstance(port, int) or not 1024 <= port <= 65535 for port in ports):
            raise ValueError("Replay ports must be distinct unprivileged TCP ports")
        ws_port, http_port = ports
    else:
        ws_port = _free_port(8865 + index * 2)
        http_port = _free_port(ws_port + 1)
    command = [
        sys.executable,
        str(ROOT / "dry_run_ab.py"),
        "--backend",
        backend,
        "--no-ab",
        "--lang",
        clip["lang"],
        "--audio-file",
        str(wav),
        "--session-id",
        tag,
        "--ws-port",
        str(ws_port),
        "--http-port",
        str(http_port),
        *extra,
    ]
    log_path = metrics_dir / f"replay_{tag}.log"
    csv_path = metrics_dir / f"ab_metrics_{tag}.csv"
    partials_path = metrics_dir / f"partials_{tag}.jsonl"
    output_path = metrics_dir / f"replay_{tag}.json"
    for path in (log_path, csv_path, partials_path, output_path):
        if path.exists():
            raise FileExistsError(f"Replay session already exists: {path}")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    timed_out = False
    launch_error = None
    with log_path.open("x") as log:
        try:
            result = run_child(command, cwd=ROOT, stdout=log, timeout=timeout_s, env=env)
        except subprocess.TimeoutExpired:
            timed_out = True
            result = subprocess.CompletedProcess(command, -1)
        except OSError as exc:
            launch_error = str(exc)
            result = subprocess.CompletedProcess(command, -1)
    report = {
        "session_id": tag,
        "clip": clip,
        "command": command,
        "returncode": result.returncode,
        "replay_speed": speed,
        "realtime_latency_eligible": speed == 1,
        "timeout_s": timeout_s,
        "timed_out": timed_out,
    }
    lifecycle = read_lifecycle(metrics_dir, tag)
    if lifecycle:
        report["session_lifecycle"] = lifecycle
    metadata = metrics_dir / f"session_metadata_{tag}.json"
    if metadata.exists():
        report["session_metadata"] = json.loads(metadata.read_text())
    if csv_path.exists():
        report.update(analyze_run(csv_path, partials_path, log_path))
    else:
        report["error"] = "Pipeline did not write metrics CSV"
    if launch_error:
        report["error"] = f"Pipeline failed to launch: {launch_error}"
    elif timed_out:
        report["error"] = f"Pipeline exceeded {timeout_s:g}s timeout and was terminated"
    elif result.returncode:
        report["error"] = f"Pipeline exited with status {result.returncode}"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if result.returncode or "error" in report:
        raise RuntimeError(f"Replay failed; see {log_path} and {output_path}")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Prepare clips and exit; does not load models")
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--offset", type=float, default=0.0, help="Start offset in seconds for sermon clips")
    parser.add_argument("--configs", nargs="+", default=[], metavar="NAME=ARGV", help="Named configs or a JSON file")
    parser.add_argument("--configs-file", type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "stark_data/replay/manifest.json")
    parser.add_argument("--baseline", type=Path, help="Prior replay JSON to compare each result against")
    parser.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="Session prefix")
    args = parser.parse_args(argv)
    if args.prepare:
        print(prepare_clips(ROOT / "stark_data/raw", args.manifest.parent, args.seconds, args.offset))
        return
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("--tag must contain only letters, digits, underscores or hyphens")
    configs = load_configs(args.configs, args.configs_file)
    clips = json.loads(args.manifest.read_text())["clips"]
    if not clips:
        parser.error("Manifest has no clips; run --prepare first")
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    index = 0
    for name, extra in configs.items():
        for clip in clips:
            wav = (args.manifest.parent / clip["path"]).resolve()
            if hashlib.sha256(wav.read_bytes()).hexdigest() != clip["sha256"]:
                raise ValueError(f"Replay clip checksum mismatch: {wav}")
            stem = re.sub(r"[^A-Za-z0-9_-]", "_", wav.stem)
            tag = f"{args.tag}_{name}_{stem}"
            report = run_replay(clip, wav, tag, extra, index, ROOT / "metrics")
            print(f"Replay {tag}: {report['chunk_count']} chunks")
            if baseline is not None:
                print(delta_table(report, baseline))
            index += 1


if __name__ == "__main__":
    main()
