#!/usr/bin/env python3
"""Render and validate docs/backlog.json.

Usage:
  python tools/render_backlog.py validate
  python tools/render_backlog.py render [--check]
  python tools/render_backlog.py check-links [ROOT ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BACKLOG_JSON = ROOT / "docs" / "backlog.json"
BACKLOG_MD = ROOT / "docs" / "backlog.md"

ALLOWED_STATUS = {
    "implemented",
    "validated",
    "experimental",
    "pending_input_or_hardware",
    "deferred",
}
ALLOWED_PRIORITY = {"P0", "P1", "P2", "P3"}
REQUIRED_ITEM_KEYS = {
    "id",
    "title",
    "status",
    "priority",
    "machine",
    "dependencies",
    "sources",
    "acceptance",
    "next_action",
}

CANONICAL_DOC_PATHS = [
    "README.md",
    "CLAUDE.md",
    "AGENTS.md",
    "CLAUDE-macbook.md",
    "CLAUDE-windows.md",
    "docs/roadmap.md",
    "docs/backlog.md",
    "docs/current_architecture.md",
    "docs/overnight_status.md",
    "docs/mac_implementation_status.md",
    "engines/AGENTS.md",
    "engines/CLAUDE.md",
    "training/AGENTS.md",
    "training/CLAUDE.md",
    "tools/AGENTS.md",
    "tools/CLAUDE.md",
    "displays/AGENTS.md",
    "displays/CLAUDE.md",
    "features/AGENTS.md",
    "features/CLAUDE.md",
]

MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def load_backlog(path: Path = BACKLOG_JSON) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def validate_backlog(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    items = data.get("items")
    if not isinstance(items, list) or not items:
        errors.append("items must be a non-empty list")
        return errors

    seen_ids: set[str] = set()
    all_ids = {item.get("id") for item in items if isinstance(item, dict)}

    for index, item in enumerate(items):
        prefix = f"items[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue

        missing = REQUIRED_ITEM_KEYS - item.keys()
        if missing:
            errors.append(f"{prefix} missing keys: {sorted(missing)}")

        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            errors.append(f"{prefix}.id must be a non-empty string")
        elif item_id in seen_ids:
            errors.append(f"duplicate id: {item_id}")
        else:
            seen_ids.add(item_id)

        status = item.get("status")
        if status not in ALLOWED_STATUS:
            errors.append(f"{prefix}.status invalid: {status!r}")

        priority = item.get("priority")
        if priority not in ALLOWED_PRIORITY:
            errors.append(f"{prefix}.priority invalid: {priority!r}")

        for field in ("dependencies", "sources"):
            value = item.get(field)
            if not isinstance(value, list):
                errors.append(f"{prefix}.{field} must be a list")

        deps = item.get("dependencies") or []
        unknown_deps = [dep for dep in deps if dep not in all_ids]
        if unknown_deps:
            errors.append(f"{prefix}.dependencies unknown ids: {unknown_deps}")

        for source in item.get("sources") or []:
            if isinstance(source, str) and source.startswith("docs/"):
                source_path = ROOT / source.split("#", 1)[0]
                if not source_path.exists():
                    errors.append(f"{prefix} source missing file: {source}")

    integration = data.get("integration")
    if not isinstance(integration, dict):
        errors.append("integration must be an object")
    else:
        for key in ("local_branch", "local_version", "main_release_tag"):
            if key not in integration:
                errors.append(f"integration missing {key}")

    return errors


def _status_heading(status: str) -> str:
    return status.replace("_", " ").title()


def render_backlog_md(data: dict[str, Any]) -> str:
    integration = data.get("integration", {})
    defaults = data.get("defaults", {})
    mac = defaults.get("mac_inference", {})

    lines = [
        "# Remaining backlog — Stark Road Bilingual Speech-to-Text",
        "",
        "> **Canonical machine-readable source:** [`backlog.json`](./backlog.json).",
        "> Regenerate this file with `python tools/render_backlog.py render`.",
        f"> **Last updated:** {data.get('last_updated', 'unknown')}",
        "",
        "## Integration status",
        "",
        f"- **Main release:** `{integration.get('main_release_tag', '?')}` — "
        f"{integration.get('main_head_note', 'see git log')}",
        f"- **Local candidate:** `{integration.get('local_version', '?')}` on "
        f"`{integration.get('local_branch', '?')}` (base `{integration.get('local_base', '?')}`)",
        f"- **Publication:** {integration.get('publication', 'unknown').replace('_', ' ')}",
        "",
        integration.get("distinction", ""),
        "",
        "## Current Mac defaults",
        "",
        f"- EN STT: `{mac.get('stt_en', '?')}` · ES STT: `{mac.get('stt_es', '?')}`",
        f"- Partials: `{mac.get('partial_translation', '?')}` · Finals: `{mac.get('final_translation', '?')}`",
        f"- Silence `{mac.get('silence_seconds', '?')}` s · partial cadence `{mac.get('partial_cadence_seconds', '?')}` s · MTP `{mac.get('mtp', '?')}`",
        "",
        "See [`current_architecture.md`](./current_architecture.md) and "
        "[`mac_implementation_status.md`](./mac_implementation_status.md) for contracts and evidence.",
        "",
    ]

    items = data.get("items", [])
    by_status: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_status.setdefault(item["status"], []).append(item)

    status_order = [
        "pending_input_or_hardware",
        "experimental",
        "deferred",
        "implemented",
        "validated",
    ]

    for status in status_order:
        bucket = by_status.get(status, [])
        if not bucket:
            continue
        lines.append(f"## {_status_heading(status)}")
        lines.append("")
        for item in sorted(bucket, key=lambda x: (x["priority"], x["id"])):
            deps = item.get("dependencies") or []
            dep_text = ", ".join(f"`{d}`" for d in deps) if deps else "none"
            sources = item.get("sources") or []
            source_bits = []
            for source in sources:
                if source.startswith("http"):
                    source_bits.append(f"[{source.split('/')[-1]}]({source})")
                else:
                    source_bits.append(f"`{source}`")
            lines.extend(
                [
                    f"### `{item['id']}` — {item['title']}",
                    "",
                    f"- **Priority:** {item['priority']} · **Machine:** {item.get('machine', '?')}",
                    f"- **Depends on:** {dep_text}",
                    f"- **Sources:** {', '.join(source_bits)}",
                    f"- **Acceptance:** {item['acceptance']}",
                    f"- **Next action:** {item['next_action']}",
                    "",
                ]
            )

    counts = Counter(item["status"] for item in items)
    lines.extend(
        [
            "## Summary counts",
            "",
            "| Status | Count |",
            "|--------|------:|",
        ]
    )
    for status in status_order:
        if counts.get(status):
            lines.append(f"| {_status_heading(status)} | {counts[status]} |")
    lines.append("")
    return "\n".join(lines)


def resolve_markdown_target(source_file: Path, target: str) -> Path | None:
    target = target.strip()
    if not target or target.startswith(("http://", "https://", "mailto:")):
        return None
    if target.startswith("#"):
        return source_file
    path_part, _, _anchor = target.partition("#")
    if path_part.startswith("/"):
        candidate = ROOT / path_part.lstrip("/")
    else:
        candidate = (source_file.parent / path_part).resolve()
    return candidate


def check_local_links(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for doc_path in paths:
        if not doc_path.exists():
            errors.append(f"missing canonical doc: {doc_path.relative_to(ROOT)}")
            continue
        text = doc_path.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK_RE.finditer(text):
            target = match.group(1)
            resolved = resolve_markdown_target(doc_path, target)
            if resolved is None:
                continue
            if not resolved.exists():
                errors.append(
                    f"{doc_path.relative_to(ROOT)}: broken link `{target}`"
                )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate", help="Validate backlog.json schema and references")

    render_parser = sub.add_parser("render", help="Render docs/backlog.md")
    render_parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if rendered output differs from committed backlog.md",
    )

    links_parser = sub.add_parser(
        "check-links", help="Check local markdown links in canonical docs"
    )
    links_parser.add_argument(
        "roots",
        nargs="*",
        help="Optional doc paths relative to repo root",
    )

    args = parser.parse_args(argv)
    data = load_backlog()

    if args.command == "validate":
        errors = validate_backlog(data)
        if errors:
            for err in errors:
                print(err, file=sys.stderr)
            return 1
        print(f"OK: {len(data['items'])} backlog items validated")
        return 0

    if args.command == "render":
        errors = validate_backlog(data)
        if errors:
            for err in errors:
                print(err, file=sys.stderr)
            return 1
        rendered = render_backlog_md(data)
        if args.check:
            if not BACKLOG_MD.exists():
                print("docs/backlog.md missing", file=sys.stderr)
                return 1
            if BACKLOG_MD.read_text(encoding="utf-8") != rendered:
                print("docs/backlog.md is out of date; run render", file=sys.stderr)
                return 1
            print("OK: docs/backlog.md matches backlog.json")
            return 0
        BACKLOG_MD.write_text(rendered, encoding="utf-8")
        print(f"Wrote {BACKLOG_MD.relative_to(ROOT)}")
        return 0

    if args.command == "check-links":
        rel_paths = args.roots or CANONICAL_DOC_PATHS
        paths = [ROOT / rel for rel in rel_paths]
        errors = check_local_links(paths)
        if errors:
            for err in errors:
                print(err, file=sys.stderr)
            return 1
        print(f"OK: local links in {len(paths)} docs")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
