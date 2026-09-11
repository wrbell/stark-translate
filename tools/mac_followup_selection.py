"""Select independently qualified replay arms without promoting production defaults.

Pure JSON operations: no inference, artifact loading, or access to confirmation audio.
Selection remains scoped to one clip and model. A spec builder preserves that scope;
callers explicitly replace clips with separately frozen confirmation inputs afterward.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def select(report: dict) -> dict:
    groups = defaultdict(list)
    for pair in report.get("pairs", []):
        groups[tuple(pair.get(k) for k in ("experiment", "size", "clip_id"))].append(pair)
    valid_inventory = report.get("inventory_complete") is True and not report.get("inventory_errors")
    arms = []
    for (name, size, clip), pairs in sorted(groups.items(), key=lambda item: repr(item[0])):
        repeats = [p.get("repeat") for p in pairs]
        complete = len(repeats) == 3 and all(type(r) is int for r in repeats) and set(repeats) == {0, 1, 2}
        passed = [p for p in pairs if p.get("status") == "latency_candidate" and not p.get("reasons")]
        nonlatency_guards = complete and all(
            (p.get("status") == "latency_candidate" and p.get("reasons") == [])
            or (p.get("status") == "rejected" and p.get("reasons") == ["median_gain_below_gate"])
            for p in pairs
        )
        qualified = valid_inventory and nonlatency_guards and len(passed) >= 2
        gains = []
        for p in passed:
            current = p.get("candidate", {}).get("p50_ms")
            controls = [p.get(k, {}).get("p50_ms") for k in ("opening", "closing")]
            if type(current) in (int, float) and all(type(v) in (int, float) for v in controls):
                gains.append(min(v - current for v in controls))
        arms.append(
            {
                "experiment": name,
                "size": size,
                "clip_id": clip,
                "qualified_for_confirmation": qualified,
                "complete_three_repeats": complete,
                "all_repeats_nonlatency_guards_passed": nonlatency_guards,
                "passed_repeats": sorted(p["repeat"] for p in passed if type(p.get("repeat")) is int),
                "mean_conservative_gain_ms": sum(gains) / len(gains) if gains else None,
                "p95_claim_eligible": complete and all(p.get("p95_claim_eligible") is True for p in pairs),
                "rejection_reasons": sorted({r for p in pairs for r in p.get("reasons", [])}),
            }
        )
    return {
        "schema_version": 1,
        "inventory_complete": valid_inventory,
        "arms": arms,
        "qualified_count": sum(a["qualified_for_confirmation"] for a in arms),
        "defaults_changed": False,
        "limitations": "Engineering selection only. Historical references remain unavailable; independent public confirmation and bilingual review are required. Never pool repeat counts for p95 eligibility.",
    }


def build_spec(spec: dict, selection: dict, *, clip_id: str, size: str, names: list[str], combine=False) -> dict:
    """Build a scope-specific follow-up; combinations require independent successes."""
    if not names or len(set(names)) != len(names):
        raise ValueError("Explicit unique qualified arm names are required")
    qualified = {
        a["experiment"]
        for a in selection["arms"]
        if a["qualified_for_confirmation"] and a["clip_id"] == clip_id and a["size"] == size
    }
    if not selection["inventory_complete"] or set(names) - qualified:
        raise ValueError("Every arm must independently qualify on this clip and model")
    configurations = {c["name"]: c for c in spec["experiments"]}
    if "baseline" not in configurations or set(names) - configurations.keys():
        raise ValueError("Missing source configuration")
    clips = [c for c in spec["clips"] if c["id"] == clip_id]
    if len(clips) != 1 or size not in spec.get("sizes", ["e4b", "e2b"]):
        raise ValueError("Missing or ambiguous source scope")
    result = copy.deepcopy(spec)
    result.update(clips=copy.deepcopy(clips), sizes=[size])
    chosen = [copy.deepcopy(configurations[n]) for n in names]
    if combine:
        if len(chosen) < 2:
            raise ValueError("A combination needs at least two independent successes")
        merged = {"name": "combined_" + "_".join(names), "env": {}}
        for item in chosen:
            for key, value in item.items():
                if key == "name":
                    continue
                if key == "env":
                    for field, setting in value.items():
                        if field in merged["env"] and merged["env"][field] != setting:
                            raise ValueError("Conflicting independent settings")
                        merged["env"][field] = setting
                elif key in merged and merged[key] != value:
                    raise ValueError("Conflicting independent settings")
                else:
                    merged[key] = value
        chosen = [merged]
    result["experiments"] = [copy.deepcopy(configurations["baseline"]), *chosen]
    result["id"] = spec["id"] + "_qualified_followup"
    result["selection_scope"] = {
        "clip_id": clip_id,
        "size": size,
        "independent_arms": names,
        "combined": combine,
        "confirmation_completed": False,
        "defaults_changed": False,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    result = select(json.loads(raw))
    result["source_report"] = {
        "path": str(args.report.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
