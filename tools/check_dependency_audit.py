"""Fail closed on an incomplete installed-Lite pip-audit JSON report.

The locally built project can be absent from PyPI; every dependency must be
represented and auditable. This supplements pip-audit's vulnerability exit code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED = {"stark-translate", "pip", "setuptools", "faster-whisper", "ctranslate2", "onnxruntime", "piper-tts"}
FORBIDDEN = {"torch", "torchaudio", "torchvision", "mlx", "mlx-lm", "silero-vad", "bitsandbytes"}


def validate_lite_audit(report: dict) -> dict:
    dependencies = report.get("dependencies") if isinstance(report, dict) else None
    if not isinstance(dependencies, list) or not dependencies:
        raise ValueError("Audit must include the installed Lite dependency inventory")
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
    if REQUIRED - names:
        raise ValueError(f"Installed Lite packages missing from audit: {sorted(REQUIRED - names)}")
    if FORBIDDEN & names:
        raise ValueError(f"Non-Lite dependencies installed: {sorted(FORBIDDEN & names)}")
    return {"audited": len(names) - len(skipped), "skipped_local_project": skipped, "known_vulnerabilities": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        result = validate_lite_audit(json.loads(args.report.read_text()))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Installed Lite audit failed: {exc}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
