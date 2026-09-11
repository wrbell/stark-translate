"""Frozen serial server-latency screens; browser/device certification stays separate.

Opening-control source spans are frozen before candidate scoring. Every candidate
must beat both opening and closing controls, without hiding lost source audio.
This engineering screen never promotes a model or substitutes for bilingual review.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

from tools.fixed_span_delivery import fixed_span_delivery
from tools.latency_experiments import LatencyExperiments
from tools.mac_evaluation import stats, write_json
from tools.overnight_bench import (
    ROOT,
    coverage_missing,
    fingerprint,
    inspect_run,
    jsonl,
    ranges,
    schedule,
    source_identity,
    source_snapshot,
    stable_hash,
    translated_preview,
)
from tools.replay_bench import run_replay
from tools.replay_integrity import audit_replay_integrity
from tools.source_coverage import merged


def reference_contract(clip):
    """Freeze exact public references and glossary opportunities before inference."""
    from tools.mac_followup_quality import CHRF_CONTRACT, NORMALIZATION, normalize, terms_in

    public = str(clip.get("provenance", "")).startswith("fleurs_")
    base = {
        "schema_version": 1,
        "clip_sha256": clip.get("sha256"),
        "source_lang": clip.get("lang"),
        "human_approved_locally": False,
        "training_eligible": False,
    }
    if not clip.get("spans"):
        return {
            **base,
            "status": "invalid" if public else "unavailable",
            "reason": "No exact ordered source/translation references",
        }
    try:
        if clip.get("lang") not in {"en", "es"} or type(clip.get("sample_rate")) is not int or clip["sample_rate"] <= 0:
            raise ValueError("Missing language/source rate")
        spans = clip["spans"]
        if not isinstance(spans, list):
            raise ValueError("References must be a list of original source spans")
        previous, ids = 0, set()
        for span in spans:
            if not isinstance(span, dict) or not isinstance(span.get("id"), str) or not span["id"] or span["id"] in ids:
                raise ValueError("Missing/duplicate reference identity")
            ids.add(span["id"])
            start, end = span.get("sample_start"), span.get("sample_end")
            if type(start) is not int or type(end) is not int or not previous <= start < end:
                raise ValueError("Reference spans must be ordered and nonoverlapping")
            previous = end
            if any(
                not isinstance(span.get(k), str) or not normalize(span[k])
                for k in ("reference", "translation_reference")
            ):
                raise ValueError("Missing/empty exact source or translation reference")
        glossary_path = ROOT / "bible_data/glossary/tier2_master.json"
        glossary = json.loads(glossary_path.read_text())
        if not isinstance(glossary, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in glossary.items()
        ):
            raise ValueError("Invalid glossary")
        terms = sorted({term for span in spans for term in terms_in(span["reference"], clip["lang"], glossary)})
        return {
            **base,
            "status": "available",
            "reference_ids": [s["id"] for s in spans],
            "ordered_reference_spans": [
                {k: s[k] for k in ("id", "sample_start", "sample_end", "reference", "translation_reference")}
                for s in spans
            ],
            "source_reference": " ".join(s["reference"] for s in spans),
            "translation_reference": " ".join(s["translation_reference"] for s in spans),
            "terms": terms,
            "glossary_sha256": fingerprint(glossary_path),
            "term_scope": "Unique glossary phrase types occurring within original references, per clip; not repeated-token counts or semantic accuracy",
            "normalization": NORMALIZATION,
            "chrf_contract": CHRF_CONTRACT,
            "reference_scope": "Upstream public read-speech annotations; no local human approval",
        }
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return {**base, "status": "invalid", "reason": str(exc)}


def caption_quality(run, contract):
    """Post-production caption text, not isolated engine WER or per-word coverage."""
    from tools.mac_followup_quality import chrf_counts, chrf_score, contains_term, word_errors

    base = {
        "scope": "Concatenated production finals after pipeline corrections, ordered by source sample; not pure engine WER",
        "human_approved_locally": False,
        "training_eligible": False,
    }
    if not contract or contract.get("status") != "available":
        return {
            **base,
            "status": (contract or {}).get("status", "unavailable"),
            "reason": (contract or {}).get("reason", "Missing frozen references"),
            "wer": None,
            "term_recall": None,
            "chrf": None,
        }
    try:
        if run.get("clip", {}).get("sha256") != contract["clip_sha256"]:
            raise ValueError("Reference/source audio identity mismatch")
        rows = run["diagnostic_finals"]
        if not rows:
            raise ValueError("No final captions")
        lang = contract["source_lang"]
        for row in rows:
            if row.get("source_lang") != lang or row.get("target_lang") != ("es" if lang == "en" else "en"):
                raise ValueError("Final caption direction mismatch")
            if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("english", "spanish_gemma")):
                raise ValueError("Missing final source or translation text")
            if type(row.get("sample_start")) is not int or type(row.get("sample_end")) is not int:
                raise ValueError("Missing caption source order")
        rows = sorted(rows, key=lambda row: (row["sample_start"], row["sample_end"]))
        if any(left["sample_end"] > right["sample_start"] for left, right in pairwise(rows)):
            raise ValueError("Overlapping caption source spans")
        source = " ".join(row["english"] for row in rows)
        translated = " ".join(row["spanish_gemma"] for row in rows)
        counts = word_errors(contract["source_reference"], source)
        terms = contract["terms"]
        hits = [term for term in terms if contains_term(source, term)]
        chrf = chrf_counts(contract["translation_reference"], translated)
        return {
            **base,
            "status": "available",
            "reference_contract_sha256": stable_hash(contract),
            "reference_ids": contract["reference_ids"],
            "source_reference": contract["source_reference"],
            "translation_reference": contract["translation_reference"],
            "source_hypothesis": source,
            "translation_hypothesis": translated,
            "final_count": len(rows),
            "wer_counts": counts,
            "wer": counts["wer"],
            "terms": terms,
            "term_hits": hits,
            "term_opportunities": len(terms),
            "term_recall": len(hits) / len(terms) if terms else None,
            "term_scope": contract["term_scope"],
            "chrf_counts": chrf,
            "chrf": chrf_score(chrf),
            "chrf_scope": "Descriptive character overlap with concatenated reference; meaning review pending",
        }
    except (ValueError, TypeError, KeyError) as exc:
        return {**base, "status": "invalid", "reason": str(exc), "wer": None, "term_recall": None, "chrf": None}


def _model_binding(model_id, *, profile=False):
    """Freeze local resolution/config only; never import a model or hash its weights."""
    from engines.model_paths import resolve_model_path
    from stark_translate.profiles import resolve_profile_model

    resolved = resolve_profile_model(model_id) if profile else resolve_model_path(model_id, local_only=True)
    if not resolved:
        raise ValueError(f"Requested benchmark model is unavailable locally: {model_id}")
    path = Path(resolved).resolve()
    config = path / "config.json"
    if not config.is_file():
        raise ValueError(f"Requested benchmark model has no config: {model_id}")
    revision = path.name if path.parent.name == "snapshots" else None
    marker = path / ".installed"
    if marker.is_file():
        revision = json.loads(marker.read_text()).get("revision", revision)
    return {
        "requested_id": model_id,
        "resolved_path": str(path),
        "config_sha256": fingerprint(config),
        "resolved_revision": revision,
    }


def runtime_contract(spec, config, clip, size):
    """Requested intent is frozen before a run, not inferred from its actual loader."""
    from settings import STTSettings, TranslationSettings, VADSettings
    from stark_translate.profiles import resolve_profile

    profile = resolve_profile(
        spec.get("profile", "standard"), "mlx" if spec.get("profile", "standard") == "standard" else "cpu"
    )
    stt = config.get("stt_backend", "parakeet-mlx" if clip["lang"] == "en" else "mlx")
    if profile.lite:
        stt, model_id = "faster-whisper", profile.stt_model
    else:
        model_id = STTSettings.model_fields["parakeet_mlx_model" if stt == "parakeet-mlx" else "whisper_model"].default
    gemma = None if profile.lite else TranslationSettings.model_fields["mlx_model_gemma4_" + size].default
    return {
        "schema_version": 1,
        "profile": profile.to_dict(),
        "stt_backend": stt,
        "stt": _model_binding(model_id, profile=profile.lite),
        "translation_a": _model_binding(gemma) if gemma else None,
        "source_lang": clip["lang"],
        "pipeline_gain": 1.0,
        "vad": {
            "backend": profile.vad_backend or VADSettings.model_fields["backend"].default,
            "threshold": VADSettings.model_fields["threshold"].default,
            "max_utterance": VADSettings.model_fields["max_utterance"].default,
            "silence_trigger": 0.5,
            "partial_interval": config.get("partial_interval", spec.get("partial_interval", 0.6)),
        },
        "translation": {
            "model_family": "gemma4",
            "routing_policy": "legacy",
            "terminology_prompt": "none",
            "mlx_mts": False,
            "idle_warmup_only": False,
            "final_aware_partials": False,
        },
        "artifact_scope": "Resolved path/revision and config hash; not a full MLX weight checksum",
    }


def runtime_errors(run, contract):
    """Check actual loader identity, including an EN startup fallback string."""
    errors = []
    metadata = run.get("session_metadata", {})
    models = run.get("session_lifecycle", {}).get("models", {})
    if not contract or contract.get("schema_version") != 1:
        return ["Missing frozen runtime intent"]
    if any(row.get("mic_gain") != contract.get("pipeline_gain") for row in run.get("diagnostic_finals", [])):
        errors.append("Actual pipeline gain differs from fixed derived-source protocol")
    for field in ("profile", "stt_backend", "source_lang"):
        if metadata.get(field) != contract[field]:
            errors.append(f"Resolved runtime mismatch: {field}")
    for field in ("vad", "translation"):
        if any(metadata.get(field, {}).get(k) != v for k, v in contract[field].items()):
            errors.append(f"Resolved runtime mismatch: {field}")
    for role in ("stt", "translation_a"):
        expected, actual = contract[role], models.get(role)
        if expected is None:
            if actual or (role == "translation_a" and metadata.get("model_a") is not None):
                errors.append(f"Unexpected loaded model: {role}")
            continue
        if not actual or actual.get("resolution_error"):
            errors.append(f"Missing actual loaded model: {role}")
            continue
        if actual.get("requested_id") not in {expected["requested_id"], expected["resolved_path"]} or any(
            actual.get(k) != expected[k] for k in ("resolved_path", "config_sha256", "resolved_revision")
        ):
            errors.append(f"Actual model differs from requested primary: {role}")
        if role == "translation_a" and metadata.get("model_a") != expected["requested_id"]:
            errors.append("Resolved Gemma model label mismatch")
    vad = metadata.get("vad", {}).get("artifact", {})
    marian = models.get("marian", {})
    if not re.fullmatch(r"[0-9a-f]{64}", str(vad.get("sha256", ""))):
        errors.append("Missing actual VAD artifact hash")
    if not marian.get("resolved_path") or not re.fullmatch(r"[0-9a-f]{64}", str(marian.get("config_sha256", ""))):
        errors.append("Missing actual Marian artifact identity")
    if "model_bin_sha256" in marian and (
        not re.fullmatch(r"[0-9a-f]{64}", str(marian["model_bin_sha256"]))
        or marian.get("export_manifest", {}).get("model_bin_hash_matches") is not True
    ):
        errors.append("Invalid actual Marian CT2 weight hash")
    return errors


def preview_evidence(run):
    """Server emission responsiveness; actual translated previews only."""
    groups = defaultdict(list)
    for row in run["observed"]["partials"]:
        if not translated_preview(row):
            continue
        for field in ("emitted_at_ms", "captured_start_at_ms", "captured_end_at_ms", "speech_start_to_partial_ms"):
            value = row.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Missing/nonfinite preview responsiveness: " + field)
        delay = row["emitted_at_ms"] - row["captured_start_at_ms"]
        # Duration is rounded to 0.1 ms; individual stage stamps to 0.001 ms.
        if (
            abs(delay - row["speech_start_to_partial_ms"]) > 0.051001
            or row["emitted_at_ms"] + 0.01 < row["captured_end_at_ms"]
        ):
            raise ValueError("Inconsistent preview source/emission timing")
        if (
            any(
                type(row.get(k)) is not int or row[k] < 0
                for k in ("sample_start", "sample_end", "sample_rate", "utterance_id")
            )
            or not row["sample_start"] < row["sample_end"]
            or not row["sample_rate"]
            or not row["utterance_id"]
        ):
            raise ValueError("Missing preview source identity")
        groups[row["utterance_id"]].append(row)
    first, gaps = [], []
    for rows in groups.values():
        rows.sort(key=lambda r: r["emitted_at_ms"])
        row = rows[0]
        first.append(
            {
                "key": (row["sample_start"], row["sample_end"], row["sample_rate"]),
                "ms": row["speech_start_to_partial_ms"],
            }
        )
        for left, right in pairwise(rows):
            gaps.append(
                {
                    "key": (
                        left["sample_start"],
                        left["sample_end"],
                        right["sample_start"],
                        right["sample_end"],
                        right["sample_rate"],
                    ),
                    "ms": right["emitted_at_ms"] - left["emitted_at_ms"],
                }
            )
    return {"first_preview": first, "update_gap": gaps}


def preview_comparisons(control, candidate):
    before, after = preview_evidence(control), preview_evidence(candidate)
    comparisons, reasons = [], []
    for field in before:
        old, new = stats([r["ms"] for r in before[field]]), stats([r["ms"] for r in after[field]])
        status = (
            "compared"
            if old["n"] and new["n"]
            else "not_applicable_control_has_no_samples"
            if not old["n"]
            else "missing_candidate_samples"
        )
        if old["n"] and not new["n"]:
            reasons.append("missing_preview_responsiveness")
        if old["n"] and new["n"] and new["p95"] > old["p95"] + max(100, old["p95"] * 0.05):
            reasons.append(field + "_tail_regression")
        keyed = []
        for rows in (before[field], after[field]):
            index = defaultdict(list)
            for row in rows:
                index[row["key"]].append(row["ms"])
            keyed.append({key: values[0] for key, values in index.items() if len(values) == 1})
        common = sorted(keyed[0].keys() & keyed[1].keys())
        paired = [{"source_bounds": list(k), "control_ms": keyed[0][k], "candidate_ms": keyed[1][k]} for k in common]
        matched_old, matched_new = stats([r["control_ms"] for r in paired]), stats([r["candidate_ms"] for r in paired])
        if paired and matched_new["p95"] > matched_old["p95"] + max(100, matched_old["p95"] * 0.05):
            reasons.append("matched_" + field + "_tail_regression")
        comparisons.append(
            {
                "field": field,
                "status": status,
                "control": old,
                "candidate": new,
                "matched_source": paired,
                "matched_control": matched_old,
                "matched_candidate": matched_new,
                "p95_claim_eligible": min(old["n"], new["n"]) >= 100,
                "scope": "Server emission; pooled utterance cohorts may change with segmentation; exact source bounds matched when available",
                "tolerance": "candidate p95 <= control p95 + max(100 ms, 5%)",
            }
        )
    return comparisons, reasons


def configuration(spec, config, clip, size):
    profile = spec.get("profile", "standard")
    if profile not in {"standard", "lite-cpu"}:
        raise ValueError("Screen supports standard or lite-cpu only")
    env = {k: v for k, v in os.environ.items() if not k.startswith("STARK_") or k == "STARK_MODELS_DIR"}
    declared = {**spec.get("baseline_env", {}), **config.get("env", {})}
    known = {"STARK_EXPERIMENT_" + name.upper() for name in LatencyExperiments().as_dict()}
    if any(k not in known or not isinstance(v, str) for k, v in declared.items()):
        raise ValueError("Only explicit experiment environment fields are supported")
    env.update(declared)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", STARK_SESSION_KIND="replay", STARK_REPLAY_SPEED="1")
    cadence = config.get("partial_interval", spec.get("partial_interval", 0.6))
    if cadence not in {0.6, 0.9, 1.2}:
        raise ValueError("Unsupported frozen partial cadence")
    stt = config.get("stt_backend", "parakeet-mlx" if clip["lang"] == "en" else "mlx")
    if stt not in {"parakeet-mlx", "mlx"}:
        raise ValueError("Unsupported Standard STT")
    arguments = [
        "--profile",
        profile,
        "--no-tts",
        "--model-family",
        "gemma4",
        "--gemma4-size",
        size,
        "--no-mts",
        "--stt-backend",
        stt if profile == "standard" else "faster-whisper",
        "--replay-speed",
        "1",
        "--silence-trigger",
        "0.5",
        "--partial-interval",
        str(cadence),
        "--gain",
        "1",
        "--routing-policy",
        "legacy",
        "--terminology-prompt",
        "none",
        "--no-idle-warmup-only",
        "--no-final-aware-partials",
    ]
    return arguments, env, LatencyExperiments.from_env(env).as_dict()


def validate_completion(result, observed, diagnostics, expected, clip, profile, source):
    errors = []
    lifecycle, metadata = result.get("session_lifecycle", {}), result.get("session_metadata", {})
    if result.get("returncode") != 0 or result.get("timed_out") or lifecycle.get("status") != "completed":
        errors.append("Replay did not finish successfully")
    if not diagnostics.is_file() or lifecycle.get("diagnostics") != {
        "sha256": fingerprint(diagnostics),
        "size_bytes": diagnostics.stat().st_size,
    }:
        errors.append("Persisted diagnostics identity mismatch")
    if metadata.get("input_audio_sha256") != clip["sha256"] or metadata.get("replay_speed") != 1:
        errors.append("Source or replay speed mismatch")
    if metadata.get("source_lang") != clip["lang"] or metadata.get("profile", {}).get("name") != profile:
        errors.append("Language or profile mismatch")
    summary = observed["session_summary"]
    if any(r.get("latency_experiment_configuration") != expected for r in (metadata, summary)):
        errors.append("Resolved experiment settings mismatch")
    if not summary.get("source_coverage", {}).get("complete"):
        errors.append("Source accounting incomplete")
    if any(r.get("state", "").endswith("_error") for r in summary.get("source_coverage", {}).get("outcomes", [])):
        errors.append("Inference error in source ledger")
    if any(r.get("failed") for r in summary.get("latency_trace", {}).get("events", [])):
        errors.append("Physical worker failed")
    if metadata.get("backend") != ("mlx" if profile == "standard" else "cpu"):
        errors.append("Resolved backend mismatch")
    if not observed["final_count"] or summary.get("chunks_completed") != observed["final_count"]:
        errors.append("Completed final count mismatch")
    if lifecycle.get("pipeline_sha256") != source["all_code_sha256"].get("dry_run_ab.py"):
        errors.append("Pipeline source mismatch")
    return errors


def run(args):
    spec = json.loads(args.spec.read_text())
    if not re.fullmatch(r"[a-zA-Z0-9_]+", args.tag):
        raise ValueError("Invalid tag")
    planned = list(schedule(spec, args.repeats))
    if len({c["id"] for c in spec["clips"]}) != len(spec["clips"]):
        raise ValueError("Duplicate clip identifiers")
    sizes = spec.get("sizes", ["e4b", "e2b"])
    if not sizes or len(set(sizes)) != len(sizes) or set(sizes) - {"e4b", "e2b"}:
        raise ValueError("Invalid or duplicate model sizes")
    for _, config, clip, size in planned:
        configuration(spec, config, clip, size)
        if clip.get("lang") not in {"en", "es"} or not re.fullmatch(r"[a-z0-9_]+", clip["id"]):
            raise ValueError("Invalid clip")
    source = source_snapshot()
    identity = stable_hash(source_identity(source))
    contracts = {
        f"{config['name']}|{size}|{clip['id']}": runtime_contract(spec, config, clip, size)
        for _, config, clip, size in planned
    }
    references = {clip["id"]: reference_contract(clip) for clip in spec["clips"]}
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "spec": spec,
        "spec_sha256": fingerprint(args.spec),
        "source": source,
        "repeats": args.repeats,
        "tag": args.tag,
        "source_identity": identity,
        "runtime_contracts": contracts,
        "reference_contracts": references,
    }
    lock = args.output / "provenance.json"
    if lock.exists():
        if json.loads(lock.read_text()) != provenance:
            raise ValueError("Frozen cohort changed; choose a new output")
    else:
        write_json(lock, provenance, exclusive=True)
    for index, (repeat, config, clip, size) in enumerate(planned):
        session = f"{args.tag}_{config['name']}_{size}_r{repeat}_{clip['id']}_{clip['lang']}"
        destination = args.output / f"{session}.json"
        if destination.exists():
            continue  # failed runs remain failed; never overwrite favorable replacements
        wav = (ROOT / clip["path"]).resolve()
        if fingerprint(wav) != clip["sha256"] or stable_hash(source_identity(source_snapshot())) != identity:
            raise ValueError("Frozen source/audio changed")
        arguments, env, expected = configuration(spec, config, clip, size)
        print(f"Starting {session}", flush=True)
        try:
            result = run_replay(
                clip,
                wav,
                session,
                arguments,
                index,
                ROOT / "metrics",
                env=env,
                backend="mlx" if spec.get("profile", "standard") == "standard" else "cpu",
                ports=(args.ws_port, args.http_port),
                timeout_s=args.timeout,
            )
        except RuntimeError as exc:
            path = ROOT / "metrics" / f"replay_{session}.json"
            result = json.loads(path.read_text()) if path.exists() else {"session_id": session}
            result["error"] = str(exc)
        observed = inspect_run(result, ROOT / "metrics")
        diagnostics = ROOT / "metrics" / f"diagnostics_{session}.jsonl"
        errors = validate_completion(
            result, observed, diagnostics, expected, clip, spec.get("profile", "standard"), source
        )
        contract = contracts[f"{config['name']}|{size}|{clip['id']}"]
        if stable_hash(source_identity(source_snapshot())) != identity:
            errors.append("Source changed during replay")
        records = jsonl(diagnostics)
        finals = [r for r in records if r.get("timing_schema_version") == 2]
        errors.extend(runtime_errors({**result, "diagnostic_finals": finals}, contract))
        integrity = audit_replay_integrity(
            data_root=ROOT,
            session_id=session,
            metadata=result.get("session_metadata", {}),
            diagnostic_finals=finals,
            partials=observed["partials"],
            trace=observed["session_summary"].get("latency_trace", {}),
        )
        if integrity["status"] != "passed":
            errors.append("Replay PCM/identity integrity failed: " + "; ".join(integrity["errors"]))
        result.update(
            experiment=config["name"],
            repeat=repeat,
            size=size,
            clip_id=clip["id"],
            source_identity=identity,
            observed=observed,
            diagnostic_finals=finals,
            replay_integrity=integrity,
            completion_errors=errors,
            requested_settings=expected,
            requested_runtime=contract,
            reference_contract=references[clip["id"]],
            caption_quality=caption_quality({**result, "diagnostic_finals": finals}, references[clip["id"]]),
            measurement_scope="server real-time replay; no physical visibility certification",
        )
        write_json(destination, result, exclusive=True)
        print(f"Finished {session}: {errors or result.get('error') or 'OK'}", flush=True)


def speech_intervals(run, *, finalized_only=False):
    observed = run["observed"]["session_summary"]["source_coverage"]["observed"]
    if not observed or any(type(row.get("vad_positive")) is not bool for row in observed):
        raise ValueError("Missing frozen per-frame VAD classification")
    positive = [(row["start"], row["end"]) for row in observed if row["vad_positive"]]
    if finalized_only:
        positive = [
            (max(a, row["sample_start"]), min(b, row["speech_end_sample"]))
            for a, b in positive
            for row in run["diagnostic_finals"]
            if max(a, row["sample_start"]) < min(b, row["speech_end_sample"])
        ]
    return merged(positive)


def score_pair(opening, candidate, closing):
    runs = (opening, candidate, closing)
    if len({r["source_identity"] for r in runs}) != 1 or len({r["clip"]["sha256"] for r in runs}) != 1:
        raise ValueError("Cannot pool different runtime/audio identities")
    anchors = [
        {
            "id": str(i),
            **{
                k: row[k]
                for k in ("sample_start", "sample_end", "speech_end_sample", "endpoint_reason", "padding_samples")
            },
        }
        for i, row in enumerate(opening["diagnostic_finals"])
    ]
    rate = opening["diagnostic_finals"][0]["sample_rate"]
    buffered_metrics = [fixed_span_delivery(anchors, r["diagnostic_finals"], sample_rate=rate) for r in runs]
    positive = speech_intervals(opening)
    for anchor in anchors:
        anchor["required_intervals"] = [
            [max(a, anchor["sample_start"]), min(b, anchor["speech_end_sample"])]
            for a, b in positive
            if max(a, anchor["sample_start"]) < min(b, anchor["speech_end_sample"])
        ]
    metrics = [fixed_span_delivery(anchors, r["diagnostic_finals"], sample_rate=rate) for r in runs]
    before, current, after = metrics
    reasons = []
    runtime_checks = [runtime_errors(r, r.get("requested_runtime")) for r in runs]
    if any(runtime_checks):
        reasons.append("missing_or_mismatched_runtime_identity")
    quality = [caption_quality(r, r.get("reference_contract")) for r in runs]
    if any(q["status"] != "available" for q in quality):
        reasons.append("caption_reference_quality_unavailable")
    elif len({q["reference_contract_sha256"] for q in quality}) != 1:
        reasons.append("caption_reference_cohort_mismatch")
    else:
        for old in (quality[0], quality[2]):
            if quality[1]["wer"] > old["wer"] + max(0.01, old["wer"] * 0.05):
                reasons.append("production_caption_wer_regression")
            if old["term_opportunities"] and quality[1]["term_recall"] < old["term_recall"]:
                reasons.append("production_caption_term_recall_regression")
    if any(r.get("error") or r.get("completion_errors") for r in runs):
        reasons.append("failed_or_incomplete_run")
    if any(m["status"] != "complete" for m in metrics):
        reasons.append("missing_fixed_source_span")
    for control in (before, after):
        if current.get("p50_ms") is None or control.get("p50_ms") is None:
            reasons.append("missing_median")
            continue
        gain = control["p50_ms"] - current["p50_ms"]
        if not (gain >= 150 or gain >= control["p50_ms"] * 0.15):
            reasons.append("median_gain_below_gate")
        if current["p95_ms"] > control["p95_ms"] + max(100, control["p95_ms"] * 0.05):
            reasons.append("tail_regression")
    preview_loss = []
    memory_comparisons = []
    queue_comparisons = []
    responsiveness = []
    for control in (opening, closing):
        try:
            comparisons, failures = preview_comparisons(control, candidate)
            responsiveness.append(comparisons)
            reasons.extend(failures)
        except ValueError as exc:
            responsiveness.append({"status": "invalid_evidence", "error": str(exc)})
            reasons.append("missing_preview_responsiveness")

        # Cadence and STT backend can be deliberately changed by a declared arm.
        # Other loader/settings evidence must remain identical to both controls.
        def common_runtime(item):
            metadata = item.get("session_metadata", {})
            vad = {k: v for k, v in metadata.get("vad", {}).items() if k != "partial_interval"}
            stt = {k: v for k, v in metadata.get("stt_settings", {}).items() if k != "backend"}
            models = item.get("session_lifecycle", {}).get("models", {})
            return {
                "profile": metadata.get("profile"),
                "profile_artifacts": metadata.get("profile_artifacts"),
                "translation": metadata.get("translation"),
                "vad": vad,
                "stt_settings": stt,
                "translation_a": models.get("translation_a"),
                "marian": models.get("marian"),
            }

        if common_runtime(control) != common_runtime(candidate):
            reasons.append("unchanged_runtime_artifact_or_settings_mismatch")
        if control.get("session_metadata", {}).get("stt_backend") == candidate.get("session_metadata", {}).get(
            "stt_backend"
        ) and control.get("session_lifecycle", {}).get("models", {}).get("stt") != candidate.get(
            "session_lifecycle", {}
        ).get("models", {}).get("stt"):
            reasons.append("unchanged_stt_artifact_mismatch")
        reference = ranges([r for r in control["observed"]["partials"] if translated_preview(r)])
        actual = ranges([r for r in candidate["observed"]["partials"] if translated_preview(r)])
        total = sum(b - a for a, b in reference)
        loss = coverage_missing(reference, actual) / total if total else 0
        preview_loss.append(loss)
        if loss > 0.02:
            reasons.append("preview_source_coverage_loss")
        delivered = merged([(r["sample_start"], r["sample_end"]) for r in candidate["diagnostic_finals"]])
        if coverage_missing(speech_intervals(control, finalized_only=True), delivered) > 0:
            reasons.append("final_source_coverage_loss")
        for field in ("peak_rss_bytes", "peak_metal_bytes"):
            old = control.get("session_lifecycle", {}).get("memory", {}).get(field)
            new = candidate.get("session_lifecycle", {}).get("memory", {}).get(field)
            if field == "peak_metal_bytes" and control.get("session_metadata", {}).get("backend") == "cpu":
                continue
            if any(type(value) is not int or value < 0 for value in (old, new)):
                reasons.append("missing_memory_evidence")
                continue
            allowed = max(old * 0.1, 256 * 1024 * 1024)
            memory_comparisons.append({"field": field, "control": old, "candidate": new, "allowed_increase": allowed})
            if new > old + allowed:
                reasons.append("memory_regression")
        old_pressure = control["observed"]["session_summary"].get("final_queue_pressure", {})
        new_pressure = candidate["observed"]["session_summary"].get("final_queue_pressure", {})
        if any(
            not p
            or p.get("bookkeeping_truncated")
            or p.get("terminal_outstanding") != 0
            or p.get("put_failed")
            or p.get("unmatched_dequeues")
            for p in (old_pressure, new_pressure)
        ):
            reasons.append("missing_or_failed_queue_accounting")
        else:
            if new_pressure["max_pending"] > old_pressure["max_pending"] + 1:
                reasons.append("final_queue_high_water_regression")
            if new_pressure["max_wait_ms"] > old_pressure["max_wait_ms"] + max(100, old_pressure["max_wait_ms"] * 0.05):
                reasons.append("final_queue_wait_regression")
            trends = []
            for pressure in (old_pressure, new_pressure):
                count = min(16, pressure.get("dequeued", 0) // 2)
                first = pressure.get("first_window_wait_ms", [])[:count]
                last = pressure.get("last_window_wait_ms", [])[-count:] if count else []
                if count < 2 or len(first) != count or len(last) != count:
                    reasons.append("insufficient_queue_trend_samples")
                    break
                if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in first + last):
                    reasons.append("invalid_queue_trend_samples")
                    break
                trends.append(sum(last) / count - sum(first) / count)
            if len(trends) == 2:
                queue_comparisons.append(
                    {"field": "disjoint_queue_wait_growth_ms", "control": trends[0], "candidate": trends[1]}
                )
                if trends[1] > max(0, trends[0]) + max(100, old_pressure["max_wait_ms"] * 0.05):
                    reasons.append("final_queue_sustained_growth")
        for field in ("stt_queue_wait_ms", "translation_queue_wait_ms", "generation_lock_wait_ms_a"):
            if field == "generation_lock_wait_ms_a":
                samples, route_counts = [], []
                valid = True
                for item in (control, candidate):
                    values = []
                    counts = {"gemma": 0, "marian": 0, "unknown": 0}
                    for row in item["diagnostic_finals"]:
                        route, value = row.get("final_translation_route"), row.get(field)
                        if route not in ("gemma", "marian"):
                            counts["unknown"] += 1
                            reasons.append("missing_or_invalid_final_translation_route")
                            valid = False
                        elif route == "marian":
                            counts[route] += 1
                            if value is not None:
                                reasons.append("invalid_marian_generation_lock_evidence")
                                valid = False
                        else:
                            counts[route] += 1
                            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                                reasons.append("missing_stage_queue_evidence")
                                valid = False
                            else:
                                values.append(value)
                    samples.append(values)
                    route_counts.append(counts)
                old_values, new_values = samples
                old_tail, new_tail = stats(old_values)["p95"], stats(new_values)["p95"]
                comparable = valid and bool(old_values) and bool(new_values)
                queue_comparisons.append(
                    {
                        "field": field,
                        "control_p95_ms": old_tail,
                        "candidate_p95_ms": new_tail,
                        "control_sample_count": len(old_values),
                        "candidate_sample_count": len(new_values),
                        "control_route_counts": route_counts[0],
                        "candidate_route_counts": route_counts[1],
                        "status": (
                            "invalid_evidence"
                            if not valid
                            else "compared_gemma_calls"
                            if comparable
                            else "not_applicable_no_gemma_in_one_or_both_runs"
                        ),
                    }
                )
                if comparable and new_tail > old_tail + max(100, old_tail * 0.05):
                    reasons.append("stage_queue_tail_regression")
                continue
            old_values = [r.get(field) for r in control["diagnostic_finals"]]
            new_values = [r.get(field) for r in candidate["diagnostic_finals"]]
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in old_values + new_values):
                reasons.append("missing_stage_queue_evidence")
                continue
            old_tail, new_tail = stats(old_values)["p95"], stats(new_values)["p95"]
            queue_comparisons.append({"field": field, "control_p95_ms": old_tail, "candidate_p95_ms": new_tail})
            if new_tail > old_tail + max(100, old_tail * 0.05):
                reasons.append("stage_queue_tail_regression")
    return {
        "status": "latency_candidate" if not reasons else "rejected",
        "reasons": sorted(set(reasons)),
        "opening": before,
        "candidate": current,
        "closing": after,
        "preview_missing_fractions": preview_loss,
        "memory_comparisons": memory_comparisons,
        "queue_comparisons": queue_comparisons,
        "preview_responsiveness": responsiveness,
        "runtime_identity_errors": runtime_checks,
        "caption_reference_quality": {
            "opening": quality[0],
            "candidate": quality[1],
            "closing": quality[2],
            "wer_tolerance": "candidate <= both controls + max(0.01 absolute, 5% relative)",
            "term_gate": "No recall decrease when unique reference term opportunities > 0; otherwise N/A",
            "chrf_delta_vs_opening": quality[1]["chrf"] - quality[0]["chrf"]
            if quality[1]["chrf"] is not None and quality[0]["chrf"] is not None
            else None,
            "chrf_delta_vs_closing": quality[1]["chrf"] - quality[2]["chrf"]
            if quality[1]["chrf"] is not None and quality[2]["chrf"] is not None
            else None,
            "chrf_gate": "none; descriptive only, bilingual review pending",
        },
        "anchors_sha256": stable_hash(anchors),
        "buffered_span_diagnostic": buffered_metrics,
        "p95_claim_eligible": all(m["n"] >= 100 for m in metrics),
        "promotion": "Requires >=2/3 repeats plus independent quality, queue/memory and untouched confirmation gates",
    }


def report(args):
    provenance = json.loads((args.input / "provenance.json").read_text())
    runs = [json.loads(p.read_text()) for p in args.input.glob("*.json") if p.name != "provenance.json"]
    runs = [r for r in runs if "experiment" in r]
    indexed = defaultdict(list)
    for item in runs:
        indexed[tuple(item.get(k) for k in ("repeat", "experiment", "size", "clip_id"))].append(item)
    planned = list(schedule(provenance["spec"], provenance["repeats"]))
    expected = {(repeat, config["name"], size, clip["id"]): clip for repeat, config, clip, size in planned}
    inventory_errors = []
    for key in set(expected) | set(indexed):
        if key not in expected:
            inventory_errors.append(f"Unexpected run key: {key}")
        elif len(indexed[key]) != 1:
            inventory_errors.append(f"Expected exactly one run: {key}; found {len(indexed[key])}")
        else:
            item = indexed[key][0]
            repeat, name, size, clip_id = key
            session = f"{provenance['tag']}_{name}_{size}_r{repeat}_{clip_id}_{expected[key]['lang']}"
            if (
                item.get("session_id") != session
                or item.get("source_identity") != provenance["source_identity"]
                or item.get("clip", {}).get("sha256") != expected[key].get("sha256")
                or not provenance.get("runtime_contracts", {}).get(f"{name}|{size}|{clip_id}")
                or item.get("requested_runtime")
                != provenance.get("runtime_contracts", {}).get(f"{name}|{size}|{clip_id}")
                or not provenance.get("reference_contracts", {}).get(clip_id)
                or item.get("reference_contract") != provenance.get("reference_contracts", {}).get(clip_id)
            ):
                inventory_errors.append(f"Run identity mismatch: {key}")
    pairs = []
    for repeat, name, size, clip_id in expected:
        if name in {"baseline", "baseline_anchor"}:
            continue
        record = dict(experiment=name, repeat=repeat, size=size, clip_id=clip_id)
        try:
            keys = [(repeat, n, size, clip_id) for n in ("baseline", name, "baseline_anchor")]
            if any(len(indexed[key]) != 1 for key in keys):
                raise ValueError("Missing or duplicate candidate/control record")
            opening, candidate, closing = [indexed[key][0] for key in keys]
            record.update(session_id=candidate["session_id"], **score_pair(opening, candidate, closing))
            if inventory_errors:
                record["status"] = "invalid_inventory"
        except (ValueError, KeyError, IndexError) as exc:
            record.update(status="invalid_comparison", error=str(exc))
        pairs.append(record)
    write_json(
        args.output,
        {
            "schema_version": 1,
            "pairs": pairs,
            "run_count": len(runs),
            "expected_runs": len(expected),
            "inventory_complete": not inventory_errors,
            "inventory_errors": inventory_errors,
            "observed_final_counts": stats([r["observed"]["final_count"] for r in runs]),
            "defaults_changed": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--spec", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--tag", required=True)
    execute.add_argument("--repeats", type=int, default=3)
    execute.add_argument("--timeout", type=float, default=300)
    execute.add_argument("--ws-port", type=int, default=8865)
    execute.add_argument("--http-port", type=int, default=8866)
    summarize = sub.add_parser("report")
    summarize.add_argument("--input", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (run if args.command == "run" else report)(args)


if __name__ == "__main__":
    main()
