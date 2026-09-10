#!/usr/bin/env python3
"""Render and validate docs/backlog.json.

Usage:
  python tools/render_backlog.py validate
  python tools/render_backlog.py render [--check]
  python tools/render_backlog.py check-links [ROOT ...]

Status vocabulary (implementation state) is separate from ``certification``
(whether the item's stated acceptance has actually been met):

  in_progress                active engineering right now (code may be uncommitted)
  implemented                code exists on the local branch; acceptance not yet certified
  validated                  acceptance met with recorded evidence
  experimental               opt-in path kept off by default
  pending_input_or_hardware  blocked on human input, references, or hardware access
  deferred                   intentionally postponed (often a pending user decision)
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
    "in_progress",
    "implemented",
    "validated",
    "experimental",
    "pending_input_or_hardware",
    "deferred",
}
ALLOWED_CERTIFICATION = {"met", "pending", "not_applicable"}
ALLOWED_PRIORITY = {"P0", "P1", "P2", "P3"}
REQUIRED_ITEM_KEYS = {
    "id",
    "title",
    "status",
    "certification",
    "priority",
    "machine",
    "dependencies",
    "sources",
    "acceptance",
    "next_action",
}
OPTIONAL_ITEM_KEYS = {"issue_acceptance", "evidence", "notes"}

STATUS_ORDER = [
    "in_progress",
    "pending_input_or_hardware",
    "experimental",
    "deferred",
    "implemented",
    "validated",
]

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
_PATHLIKE_SUFFIXES = (".md", ".py", ".json", ".sh", ".yml", ".yaml", ".js", ".html", ".txt")


def load_backlog(path: Path = BACKLOG_JSON) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _looks_like_repo_path(source: str) -> bool:
    if source.startswith(("http://", "https://")):
        return False
    if "/" in source:
        return True
    return source.endswith(_PATHLIKE_SUFFIXES)


def validate_backlog(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if data.get("schema_version") != 2:
        errors.append("schema_version must be 2")

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
        unknown = set(item.keys()) - REQUIRED_ITEM_KEYS - OPTIONAL_ITEM_KEYS
        if unknown:
            errors.append(f"{prefix} unknown keys: {sorted(unknown)}")

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

        certification = item.get("certification")
        if certification not in ALLOWED_CERTIFICATION:
            errors.append(f"{prefix}.certification invalid: {certification!r}")
        if status == "validated" and certification != "met":
            errors.append(f"{prefix}: status 'validated' requires certification 'met'")
        if status in {"in_progress", "pending_input_or_hardware", "experimental"} and certification == "met":
            errors.append(f"{prefix}: status {status!r} cannot have certification 'met'")

        priority = item.get("priority")
        if priority not in ALLOWED_PRIORITY:
            errors.append(f"{prefix}.priority invalid: {priority!r}")

        for field in ("dependencies", "sources"):
            value = item.get(field)
            if not isinstance(value, list):
                errors.append(f"{prefix}.{field} must be a list")
        for field in ("evidence",):
            if field in item and not isinstance(item[field], list):
                errors.append(f"{prefix}.{field} must be a list when present")

        deps = item.get("dependencies") or []
        unknown_deps = [dep for dep in deps if dep not in all_ids]
        if unknown_deps:
            errors.append(f"{prefix}.dependencies unknown ids: {unknown_deps}")
        if item_id in deps:
            errors.append(f"{prefix} depends on itself")

        for source in item.get("sources") or []:
            if not isinstance(source, str):
                errors.append(f"{prefix} source must be a string: {source!r}")
                continue
            if _looks_like_repo_path(source):
                source_path = ROOT / source.split("#", 1)[0]
                if not source_path.exists():
                    errors.append(f"{prefix} source missing file: {source}")

    integration = data.get("integration")
    if not isinstance(integration, dict):
        errors.append("integration must be an object")
    else:
        for key in ("local_branch", "local_version", "main_release_tag", "draft_pr"):
            if key not in integration:
                errors.append(f"integration missing {key}")

    definitions = data.get("status_definitions")
    if not isinstance(definitions, dict) or set(definitions) != ALLOWED_STATUS:
        errors.append("status_definitions must define exactly the allowed statuses")

    return errors


def _status_heading(status: str) -> str:
    return status.replace("_", " ").title()


def _format_sources(sources: list[str]) -> str:
    bits = []
    for source in sources:
        if source.startswith("http"):
            label = source.rstrip("/").split("/")[-1]
            if "/issues/" in source:
                label = f"#{label}"
            elif "/pull/" in source:
                label = f"PR #{label}"
            bits.append(f"[{label}]({source})")
        else:
            bits.append(f"`{source}`")
    return ", ".join(bits) if bits else "none"


def render_backlog_md(data: dict[str, Any]) -> str:
    integration = data.get("integration", {})
    defaults = data.get("defaults", {})
    mac = defaults.get("mac_inference", {})
    definitions = data.get("status_definitions", {})

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
        f"- **Draft PR:** {integration.get('draft_pr', 'none')}",
        f"- **Publication:** {integration.get('publication', 'unknown')}",
        "",
        integration.get("distinction", ""),
        "",
        "## Status vocabulary",
        "",
        "| Status | Meaning |",
        "|--------|---------|",
    ]
    for status in STATUS_ORDER:
        lines.append(f"| `{status}` | {definitions.get(status, '')} |")
    lines.extend(
        [
            "",
            "`certification` records whether the item's stated acceptance has been met "
            "(`met`, `pending`, `not_applicable`) independently of implementation status.",
            "",
            "## Current Mac defaults",
            "",
            f"- EN STT: `{mac.get('stt_en', '?')}` · ES STT: `{mac.get('stt_es', '?')}`",
            f"- Partials: `{mac.get('partial_translation', '?')}` · Finals: `{mac.get('final_translation', '?')}`",
            f"- Silence `{mac.get('silence_seconds', '?')}` s · partial cadence "
            f"`{mac.get('partial_cadence_seconds', '?')}` s · MTP `{mac.get('mtp', '?')}`",
            "",
            "See [`current_architecture.md`](./current_architecture.md) and "
            "[`mac_implementation_status.md`](./mac_implementation_status.md) for contracts and evidence.",
            "",
        ]
    )

    items = data.get("items", [])
    by_status: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_status.setdefault(item["status"], []).append(item)

    for status in STATUS_ORDER:
        bucket = by_status.get(status, [])
        if not bucket:
            continue
        lines.append(f"## {_status_heading(status)}")
        lines.append("")
        for item in sorted(bucket, key=lambda x: (x["priority"], x["id"])):
            deps = item.get("dependencies") or []
            dep_text = ", ".join(f"`{d}`" for d in deps) if deps else "none"
            lines.extend(
                [
                    f"### `{item['id']}` — {item['title']}",
                    "",
                    f"- **Priority:** {item['priority']} · **Machine:** {item.get('machine', '?')} · "
                    f"**Certification:** {item['certification'].replace('_', ' ')}",
                    f"- **Depends on:** {dep_text}",
                    f"- **Sources:** {_format_sources(item.get('sources') or [])}",
                ]
            )
            if item.get("issue_acceptance"):
                lines.append(f"- **Issue acceptance (verbatim intent):** {item['issue_acceptance']}")
            lines.append(f"- **Acceptance:** {item['acceptance']}")
            for evidence in item.get("evidence") or []:
                lines.append(f"- **Evidence:** {evidence}")
            if item.get("notes"):
                lines.append(f"- **Notes:** {item['notes']}")
            lines.append(f"- **Next action:** {item['next_action']}")
            lines.append("")

    counts = Counter(item["status"] for item in items)
    lines.extend(
        [
            "## Summary counts",
            "",
            "| Status | Count |",
            "|--------|------:|",
        ]
    )
    for status in STATUS_ORDER:
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
                errors.append(f"{doc_path.relative_to(ROOT)}: broken link `{target}`")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate", help="Validate backlog.json schema and references")

    render_parser = sub.add_parser("render", help="Render docs/backlog.md")
    render_parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if rendered output differs from committed backlog.md",
    )

    links_parser = sub.add_parser("check-links", help="Check local markdown links in canonical docs")
    links_parser.add_argument("roots", nargs="*", help="Optional doc paths relative to repo root")

    args = parser.parse_args(argv)

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

    data = load_backlog()
    errors = validate_backlog(data)
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1

    if args.command == "validate":
        print(f"OK: {len(data['items'])} backlog items validated")
        return 0

    if args.command == "render":
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

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
