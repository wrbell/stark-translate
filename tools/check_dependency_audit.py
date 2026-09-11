"""Fail closed on an incomplete installed-runtime pip-audit JSON report.

The locally built project can be absent from PyPI; every dependency must be
represented and auditable. This supplements pip-audit's vulnerability exit code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED = {"stark-translate", "pip", "setuptools", "faster-whisper", "ctranslate2", "onnxruntime", "piper-tts"}
FORBIDDEN = {"torch", "torchaudio", "torchvision", "mlx", "mlx-lm", "silero-vad", "bitsandbytes"}
MAC_REQUIRED = {
    "stark-translate",
    "pip",
    "setuptools",
    "torch",
    "torchaudio",
    "mlx",
    "mlx-lm",
    "mlx-whisper",
    "parakeet-mlx",
    "ctranslate2",
    "silero-vad",
}


def validate_lite_audit(report: dict) -> dict:
    """Retain the existing Lite validation API and result shape."""
    return validate_audit(report, runtime="lite")


def validate_audit(report: dict, *, runtime: str = "lite") -> dict:
    if runtime not in {"lite", "mac"}:
        raise ValueError(f"Unknown runtime: {runtime}")
    label = "Lite" if runtime == "lite" else "Mac"
    required = REQUIRED if runtime == "lite" else MAC_REQUIRED
    forbidden = FORBIDDEN if runtime == "lite" else set()
    dependencies = report.get("dependencies") if isinstance(report, dict) else None
    if not isinstance(dependencies, list) or not dependencies:
        raise ValueError(f"Audit must include the installed {label} dependency inventory")
    names, skipped = set(), []
    for item in dependencies:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("Invalid dependency audit entry")
        name = item["name"].lower().replace("_", "-")
        if name in names:
            raise ValueError(f"Duplicate audited dependency: {name}")
        names.add(name)
        if item.get("vulns"):
            raise ValueError(f"Known vulnerabilities remain in {name}")
        if item.get("skip_reason"):
            if name != "stark-translate" or "not found on PyPI" not in item["skip_reason"]:
                raise ValueError(f"Dependency was not audited: {name}")
            skipped.append(name)
        elif not isinstance(item.get("vulns"), list) or not item.get("version"):
            raise ValueError(f"Incomplete audit result for {name}")
    if required - names:
        raise ValueError(f"Installed {label} packages missing from audit: {sorted(required - names)}")
    if forbidden & names:
        raise ValueError(f"Non-{label} dependencies installed: {sorted(forbidden & names)}")
    return {"audited": len(names) - len(skipped), "skipped_local_project": skipped, "known_vulnerabilities": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["lite", "mac"], default="lite")
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        result = validate_audit(json.loads(args.report.read_text()), runtime=args.runtime)
    except (OSError, ValueError) as exc:
        label = "Lite" if args.runtime == "lite" else "Mac"
        parser.exit(1, f"Installed {label} audit failed: {exc}\n")
    print(json.dumps({**result, "runtime": args.runtime}))


if __name__ == "__main__":
    main()
