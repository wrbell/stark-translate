#!/usr/bin/env python3
"""Fail when a gitleaks:allow comment has no reason.

A line that contains ``#`` and then ``gitleaks:allow`` must explain the
exception after the marker. The reason needs at least eight letters or
digits. The negative fixture under ``tests/fixtures/security-supply-chain/allow-fail/``
is the sample that fails this check. The repository scan skips that
directory so the sample can stay in the tree.

``.agents/`` and ``third_party/`` are vendored. The scan skips them.

The script uses the Python 3.11 standard library only.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

MARKER = "gitleaks:allow"
SKIP_DIRS = frozenset(
    {".git", ".agents", "third_party", "__pycache__", ".pytest_cache"}
)
SKIP_PREFIX = "tests/fixtures/security-supply-chain/allow-fail/"


@dataclass(frozen=True)
class Problem:
    """One bare gitleaks allow comment."""

    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def problems_in_text(rel: str, text: str) -> list[Problem]:
    """Return problems for allow comments in one file."""
    problems: list[Problem] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        lower = line.lower()
        index = lower.find(MARKER)
        if index == -1:
            continue
        if "#" not in line[:index]:
            continue
        rest = line[index + len(MARKER) :].lstrip(" \t:").strip()
        if len(re.sub(r"[^A-Za-z0-9]", "", rest)) < 8:
            problems.append(
                Problem(rel, lineno, "gitleaks:allow comment needs a reason")
            )
    return problems


def scan_root(root: Path) -> list[Problem]:
    """Scan text files under ``root`` for bare allow comments."""
    problems: list[Problem] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            if rel.startswith(SKIP_PREFIX):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeError, OSError):
                continue
            problems.extend(problems_in_text(rel, text))
    problems.sort(key=lambda item: (item.path, item.line))
    return problems


def main(argv: list[str] | None = None) -> int:
    """Scan for bare gitleaks allow comments and return a process status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("."),
        help="directory to scan (default: the working directory)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        print(f"{root}: not a directory", file=sys.stderr)
        return 2
    problems = scan_root(root)
    if problems:
        for problem in problems:
            print(problem)
        print(f"{len(problems)} problem(s)", file=sys.stderr)
        return 1
    print("ok: gitleaks allow comments")
    return 0


if __name__ == "__main__":
    sys.exit(main())
