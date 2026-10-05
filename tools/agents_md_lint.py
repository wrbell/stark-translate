#!/usr/bin/env python3
"""Check AGENTS.md, CLAUDE.md, Cursor rules, and the Clarity section.

The check fails when any of these is true:

- ``AGENTS.md`` is missing
- ``AGENTS.md`` has more than 200 lines
- ``CLAUDE.md`` exists and no line is exactly ``@AGENTS.md``
- ``.cursor/rules`` contains a plain ``.md`` file
- a ``.mdc`` file is not glob-scoped (``globs`` set and ``alwaysApply: false``)
- the Clarity section is missing or its text was changed

With no arguments, the script checks the repository that contains it.
It checks a root ``AGENTS.md`` when that file exists.
It checks ``templates/`` when ``templates/AGENTS.md`` exists.
A repository with no agent instruction files passes.
``.agents/`` and ``third_party/`` are not instruction roots.

Usage::

    python3 tools/agents_md_lint.py [DIRECTORY]

DIRECTORY is one instruction root (the folder that should contain
``AGENTS.md``). Omit it to discover roots in this repository.
The script uses the Python 3.11 standard library only.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

MAX_AGENTS_LINES = 200
SKIP_DIR_NAMES = frozenset({".agents", "third_party"})
CLARITY_SECTION = """\
## Clarity

Write explanations to Willem in Simplified Technical English. Use the same style
for a pull request description, a commit message body, and prose in a README or
another doc. Aim for about 80 percent of the rules. Full compliance with
ASD-STE100 is not the goal.

The rules are in the standards repository at `standards/writing-ste/ste.md`.
The upstream skill is
[simplified-technical-english](https://github.com/0xpili/simplified-technical-english/tree/1e148d670cba46685ad2b4c3f2354a637a7fdbbe)
(MIT, commit `1e148d670cba46685ad2b4c3f2354a637a7fdbbe`). Link to that skill.
Do not copy the skill into this repository again.

Code, identifiers, math, command-line output, and quoted error text are exempt.

When structure, flow, or architecture is the point, use a mermaid diagram.
For a complex result, offer a self-contained HTML page.
That page is a throwaway file.
Do not commit it unless Willem asks.

Make a video only when Willem asks for a video.
Do not add an API key or a secret.

`scripts/ste_check.py` in the upstream skill is an optional check on docs.
Do not use it as a CI gate.
"""


@dataclass(frozen=True)
class Problem:
    """Hold one agents-md lint problem."""

    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def _is_vendor(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts)


def _read(path: Path) -> str | Problem:
    """Return file text, or a problem when the file is not UTF-8."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        return Problem(path.name, 1, f"not valid UTF-8 ({exc})")


def _frontmatter(text: str) -> str | None:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[1:index])
    return None


def _globs_set(frontmatter: str) -> bool:
    lines = frontmatter.splitlines()
    for index, raw in enumerate(lines):
        stripped = raw.strip()
        match = re.match(r"globs:\s*(.*)$", stripped)
        if match is None:
            continue
        value = match.group(1).strip()
        if value in {"", "''", '""', "[]", "{}"}:
            rest = lines[index + 1 :]
            return any(item.strip().startswith("- ") for item in rest)
        return True
    return False


def _mdc_problems(path: Path, rel: str) -> list[Problem]:
    loaded = _read(path)
    if isinstance(loaded, Problem):
        return [Problem(rel, loaded.line, loaded.message)]
    frontmatter = _frontmatter(loaded)
    if frontmatter is None:
        return [
            Problem(
                rel,
                1,
                "mdc rule must set globs and alwaysApply: false",
            )
        ]
    problems: list[Problem] = []
    saw_false = False
    for number, raw in enumerate(frontmatter.splitlines(), start=2):
        match = re.match(r"alwaysApply:\s*(\S+)", raw.strip())
        if match is None:
            continue
        value = match.group(1).lower().strip("'\"")
        if value == "true":
            problems.append(
                Problem(rel, number, "mdc rule must set alwaysApply: false")
            )
        elif value == "false":
            saw_false = True
    if not saw_false:
        problems.append(Problem(rel, 1, "mdc rule must set alwaysApply: false"))
    if not _globs_set(frontmatter):
        problems.append(Problem(rel, 1, "mdc rule must set globs"))
    return problems


def check_instruction_dir(root: Path) -> list[Problem]:
    """Check one folder laid out like a repository root."""
    problems: list[Problem] = []
    agents = root / "AGENTS.md"
    if not agents.is_file():
        problems.append(Problem("AGENTS.md", 1, "AGENTS.md missing"))
    else:
        loaded = _read(agents)
        if isinstance(loaded, Problem):
            problems.append(Problem("AGENTS.md", loaded.line, loaded.message))
        else:
            count = len(loaded.splitlines())
            if count > MAX_AGENTS_LINES:
                problems.append(
                    Problem(
                        "AGENTS.md",
                        1,
                        f"AGENTS.md has {count} lines (>{MAX_AGENTS_LINES})",
                    )
                )
            if CLARITY_SECTION not in loaded:
                problems.append(
                    Problem(
                        "AGENTS.md",
                        1,
                        "Clarity section is missing or was changed",
                    )
                )
    claude = root / "CLAUDE.md"
    if claude.is_file():
        loaded = _read(claude)
        if isinstance(loaded, Problem):
            problems.append(Problem("CLAUDE.md", loaded.line, loaded.message))
        elif "@AGENTS.md" not in loaded.splitlines():
            problems.append(Problem("CLAUDE.md", 1, "CLAUDE.md must import @AGENTS.md"))
    rules = root / ".cursor" / "rules"
    if rules.is_dir():
        for path in sorted(rules.rglob("*")):
            if not path.is_file() or _is_vendor(path.relative_to(root)):
                continue
            rel = path.relative_to(root).as_posix()
            if path.suffix == ".md":
                problems.append(Problem(rel, 1, ".cursor/rules: use .mdc"))
            elif path.suffix == ".mdc":
                problems.extend(_mdc_problems(path, rel))
    return problems


def discover_roots(repo: Path) -> list[Path]:
    """Return instruction roots to check under a repository."""
    roots: list[Path] = []
    agents = repo / "AGENTS.md"
    if agents.is_file():
        roots.append(repo)
    templates = repo / "templates"
    if (templates / "AGENTS.md").is_file():
        roots.append(templates)
    if roots:
        return roots
    if (repo / "CLAUDE.md").is_file() or (repo / ".cursor" / "rules").is_dir():
        return [repo]
    return []


def _prefix(repo: Path, root: Path) -> str:
    relative = root.resolve().relative_to(repo.resolve()).as_posix()
    if relative == ".":
        return ""
    return relative + "/"


def check_repo(repo: Path) -> list[Problem]:
    """Discover instruction roots and check each one."""
    problems: list[Problem] = []
    for root in discover_roots(repo):
        prefix = _prefix(repo, root)
        for problem in check_instruction_dir(root):
            problems.append(
                Problem(prefix + problem.path, problem.line, problem.message)
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    """Check instruction files and return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=None,
        help="instruction root (default: discover roots in this repository)",
    )
    args = parser.parse_args(argv)
    if args.directory is None:
        repo = Path(__file__).resolve().parent.parent
        problems = check_repo(repo)
        checked = len(discover_roots(repo))
    else:
        root = args.directory
        if not root.is_dir():
            print(f"{root}: not a directory", file=sys.stderr)
            return 2
        problems = check_instruction_dir(root)
        checked = 1
    if problems:
        for problem in problems:
            print(problem)
        print(f"{len(problems)} problem(s)", file=sys.stderr)
        return 1
    if checked == 0:
        print("ok: nothing to check")
    else:
        print(f"ok: {checked} instruction root(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
