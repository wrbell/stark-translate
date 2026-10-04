#!/usr/bin/env python3
"""Check a Dependabot config for the github-actions cooldown policy.

The github-actions entry must:

- use schedule interval ``weekly``
- set ``cooldown.default-days`` to 3 or more
- avoid semver cooldown keys (Dependabot rejects them for this ecosystem)
- define a ``groups`` block

The script uses the Python 3.11 standard library only.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ECOSYSTEM = re.compile(r"^(\s*)-\s+package-ecosystem:\s*[\"']?([^\"'\s#]+)[\"']?\s*$")
INTERVAL = re.compile(r"interval:\s*[\"']?weekly[\"']?")
DEFAULT_DAYS = re.compile(r"default-days:\s*(\d+)")
SEMVER_DAYS = re.compile(r"semver-(?:major|minor|patch)-days:")


def _blocks(text: str) -> list[tuple[str, str]]:
    """Return ``(ecosystem, block text)`` for each updates entry."""
    lines = text.splitlines(keepends=True)
    starts: list[tuple[int, str]] = []
    offset = 0
    for line in lines:
        stripped = line.lstrip()
        if not stripped.startswith("#"):
            match = ECOSYSTEM.match(line.rstrip("\n"))
            if match:
                starts.append((offset, match.group(2)))
        offset += len(line)
    blocks: list[tuple[str, str]] = []
    for index, (start, name) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(text)
        blocks.append((name, text[start:end]))
    return blocks


def check_text(text: str, name: str = "dependabot.yml") -> list[str]:
    """Return human-readable problems. An empty list means the file passes."""
    problems: list[str] = []
    if re.search(r"^version:\s*2\s*$", text, re.MULTILINE) is None:
        problems.append(f"{name}: version is not 2")
    actions = [block for eco, block in _blocks(text) if eco == "github-actions"]
    if not actions:
        problems.append(f"{name}: missing package-ecosystem github-actions")
        return problems
    block = actions[0]
    if "directory:" not in block and "directories:" not in block:
        problems.append(f"{name}: github-actions entry has no directory")
    if INTERVAL.search(block) is None:
        problems.append(f"{name}: github-actions schedule interval is not weekly")
    if "cooldown:" not in block:
        problems.append(f"{name}: github-actions entry has no cooldown")
    else:
        days = DEFAULT_DAYS.search(block)
        if days is None:
            problems.append(f"{name}: github-actions cooldown has no default-days")
        elif int(days.group(1)) < 3:
            problems.append(f"{name}: github-actions cooldown default-days is below 3")
    if SEMVER_DAYS.search(block):
        problems.append(f"{name}: github-actions cooldown cannot use semver day keys")
    if "groups:" not in block:
        problems.append(f"{name}: github-actions entry has no groups block")
    return problems


def default_files() -> list[Path]:
    """Return Dependabot files that exist in the working directory."""
    found: list[Path] = []
    for rel in (
        "enforcement/dependabot/dependabot.yml",
        ".github/dependabot.yml",
    ):
        path = Path(rel)
        if path.is_file():
            found.append(path)
    return found


def main(argv: list[str] | None = None) -> int:
    """Check Dependabot files and return a process status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Dependabot files (default: the repo copies that exist)",
    )
    args = parser.parse_args(argv)
    files = args.files or default_files()
    if not files:
        print("ok: no Dependabot file")
        return 0
    problems: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError as exc:
            problems.append(f"{path}: not valid UTF-8 ({exc})")
            continue
        problems.extend(check_text(text, path.as_posix()))
    if problems:
        for problem in problems:
            print(problem)
        print(f"{len(problems)} problem(s)", file=sys.stderr)
        return 1
    print(f"ok: {len(files)} Dependabot file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
