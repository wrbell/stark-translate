#!/usr/bin/env python3
"""Check README minimums, decision records, and CITATION.cff.

The script checks the repository root. It uses the Python 3.11 standard
library only.

README checks:

- ``README.md`` exists at the root and is under 500 KiB.
- The first line after the title is one description of at most 80
  characters, so markdownlint MD013 (80) and this check agree.
- The last level-2 heading is ``License``.
- A file longer than 100 lines has a table of contents.
- Links to files inside the repo are relative.

Decision records live in ``docs/decisions/``. Each file name matches
``NNNN-title-with-dashes.md``. Front matter has ``status`` and ``date``.
The MADR 4.0.0 headings are present.

``CITATION.cff`` is checked when it exists at the root.
Every ``CITATION.cff`` under ``templates/`` is checked too.
A missing file is not an error.

The README checks run on the root ``README.md`` of every repository,
with or without a ``templates/`` folder. When
``templates/README.template.md`` exists, as in this repository, the
same rules run on it too.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

SKIP_DIR_NAMES = frozenset({".agents", "third_party"})
README_LIMIT_BYTES = 500 * 1024
SHORT_DESCRIPTION_LIMIT = 80
TOC_LINE_LIMIT = 100
ADR_NAME = re.compile(r"^[0-9]{4}-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\.md$")
STATUSES = frozenset({"proposed", "accepted", "rejected", "deprecated", "superseded"})
FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
HEADING = re.compile(r"^(#{1,6})[ \t]+(.*\S)\s*$")
LINK_TARGET = re.compile(r"\]\(([^)\s]+)\)")
BLOB = re.compile(r"https?://[^/]+/[^/]+/[^/]+/blob/[^/]+/(.+)$")
MADR_CONTEXT = "Context and Problem Statement"
MADR_OPTIONS = "Considered Options"
MADR_OUTCOME = "Decision Outcome"


@dataclass(frozen=True)
class Problem:
    """Hold one docs-structure problem."""

    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def _vendor(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts)


def _visible_lines(text: str) -> list[tuple[int, str]]:
    """Return non-fence lines as ``(lineno, text)``."""
    visible: list[tuple[int, str]] = []
    fence: str | None = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        match = FENCE.match(raw)
        if match:
            marker = match.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence is None:
            visible.append((lineno, raw))
    return visible


def _heading(line: str) -> tuple[int, str] | None:
    match = HEADING.match(line.strip())
    if match is None:
        return None
    text = re.sub(r"\s+#+\s*$", "", match.group(2).strip()).strip()
    return len(match.group(1)), text


def check_readme_file(path: Path, root: Path, *, prose: bool) -> list[Problem]:
    """Check one README. ``prose`` turns on the content rules."""
    rel = path.relative_to(root).as_posix()
    problems: list[Problem] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        return [Problem(rel, 1, f"not valid UTF-8 ({exc})")]
    size = path.stat().st_size
    if size >= README_LIMIT_BYTES:
        problems.append(Problem(rel, 1, "README is not under 500 KiB"))
    if not prose:
        return problems
    visible = _visible_lines(text)
    headings: list[tuple[int, int, str]] = []
    for lineno, line in visible:
        parsed = _heading(line)
        if parsed is not None:
            level, title = parsed
            headings.append((lineno, level, title))
    content = [(lineno, line) for lineno, line in visible if line.strip()]
    if not content:
        return problems + [Problem(rel, 1, "README has no title")]
    first = _heading(content[0][1])
    if first is None or first[0] != 1:
        problems.append(Problem(rel, content[0][0], "README must start with a title"))
    else:
        title_line = content[0][0]
        rest = [item for item in content if item[0] > title_line]
        if not rest:
            problems.append(Problem(rel, title_line, "README has no short description"))
        else:
            desc_line, desc = rest[0]
            if _heading(desc) is not None:
                problems.append(Problem(rel, desc_line, "short description is missing"))
            elif len(desc.strip()) > SHORT_DESCRIPTION_LIMIT:
                problems.append(
                    Problem(
                        rel,
                        desc_line,
                        "short description is over 80 characters",
                    )
                )
            else:
                # The description is one line. The next source line is blank
                # or the file ends.
                lines = text.splitlines()
                if desc_line < len(lines) and lines[desc_line].strip():
                    problems.append(
                        Problem(
                            rel,
                            desc_line + 1,
                            "short description continues past one line",
                        )
                    )
    h2 = [(lineno, title) for lineno, level, title in headings if level == 2]
    if not h2 or h2[-1][1] != "License":
        line = h2[-1][0] if h2 else 1
        problems.append(Problem(rel, line, "the last level-2 heading is not License"))
    elif headings and headings[-1][0] != h2[-1][0]:
        problems.append(
            Problem(rel, headings[-1][0], "a heading follows the License section")
        )
    if len(text.splitlines()) > TOC_LINE_LIMIT:
        titles = {title.casefold() for _lineno, _level, title in headings}
        if "table of contents" not in titles and "contents" not in titles:
            problems.append(
                Problem(rel, 1, "README longer than 100 lines has no table of contents")
            )
    problems.extend(_relative_links(text, rel, root))
    return problems


def _relative_links(text: str, rel: str, root: Path) -> list[Problem]:
    problems: list[Problem] = []
    for lineno, line in _visible_lines(text):
        for target in LINK_TARGET.findall(line):
            bare = target.split("#", 1)[0].split("?", 1)[0]
            if bare.startswith(("#", "mailto:")) or bare == "":
                continue
            if bare.startswith("/") and not bare.startswith("//"):
                problems.append(
                    Problem(rel, lineno, f"use a relative link for {target}")
                )
                continue
            blob = BLOB.match(bare)
            if blob is None:
                continue
            repo_path = blob.group(1)
            if (root / repo_path).is_file():
                problems.append(
                    Problem(
                        rel,
                        lineno,
                        f"use a relative link for the repo file {repo_path}",
                    )
                )
    return problems


def _front_matter(text: str) -> tuple[str, int] | None:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for index, line in enumerate(lines[1:], start=2):
        if line.strip() == "---":
            return "\n".join(lines[1 : index - 1]), index
    return None


def _field(block: str, name: str) -> str | None:
    match = re.search(rf"(?m)^{name}:\s*(\S.*?)\s*$", block)
    if match is None:
        return None
    return match.group(1).strip().strip("\"'")


def check_decision(path: Path, root: Path) -> list[Problem]:
    """Check one MADR decision record."""
    rel = path.relative_to(root).as_posix()
    problems: list[Problem] = []
    if not ADR_NAME.match(path.name):
        problems.append(
            Problem(rel, 1, "decision file name is not NNNN-title-with-dashes.md")
        )
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        return problems + [Problem(rel, 1, f"not valid UTF-8 ({exc})")]
    matter = _front_matter(text)
    if matter is None:
        problems.append(Problem(rel, 1, "decision record has no front matter"))
        status = None
    else:
        block, _end = matter
        status = _field(block, "status")
        iso_date = _field(block, "date")
        if status not in STATUSES:
            problems.append(
                Problem(
                    rel,
                    2,
                    "status must be proposed, accepted, rejected, "
                    "deprecated, or superseded",
                )
            )
        if iso_date is None:
            problems.append(Problem(rel, 2, "front matter has no date"))
        else:
            try:
                date.fromisoformat(iso_date)
            except ValueError:
                problems.append(Problem(rel, 2, "date is not YYYY-MM-DD"))
    headings = [
        (lineno, level, title)
        for lineno, line in _visible_lines(text)
        if (parsed := _heading(line)) is not None
        for level, title in (parsed,)
    ]
    h2 = [(lineno, title) for lineno, level, title in headings if level == 2]
    titles = [title for _lineno, title in h2]
    if (
        MADR_CONTEXT not in titles
        or MADR_OPTIONS not in titles
        or MADR_OUTCOME not in titles
    ):
        problems.append(
            Problem(
                rel,
                1,
                "MADR headings are missing. Require Context and Problem "
                "Statement, Considered Options, and Decision Outcome",
            )
        )
    else:
        context_at = titles.index(MADR_CONTEXT)
        options_at = titles.index(MADR_OPTIONS)
        outcome_at = titles.index(MADR_OUTCOME)
        if not (context_at < options_at < outcome_at):
            problems.append(Problem(rel, h2[0][0], "MADR headings are out of order"))
        elif outcome_at != options_at + 1:
            problems.append(
                Problem(
                    rel,
                    h2[outcome_at][0],
                    "Decision Outcome does not follow Considered Options",
                )
            )
        else:
            start = h2[outcome_at][0]
            end = h2[outcome_at + 1][0] if outcome_at + 1 < len(h2) else None
            body_lines = text.splitlines()[start : end - 1 if end else None]
            body = "\n".join(body_lines)
            if "Chosen option:" not in body or "because" not in body:
                problems.append(
                    Problem(
                        rel,
                        start,
                        'Decision Outcome must contain "Chosen option:" and "because"',
                    )
                )
    if status == "superseded" and "](" not in text:
        problems.append(
            Problem(rel, 1, "a superseded record must link to the new record")
        )
    return problems


def check_decisions(root: Path) -> list[Problem]:
    """Check every decision record under ``docs/decisions``."""
    folder = root / "docs" / "decisions"
    if not folder.is_dir():
        return []
    problems: list[Problem] = []
    numbers: dict[str, str] = {}
    files = sorted(
        path
        for path in folder.rglob("*.md")
        if path.is_file() and not _vendor(path.relative_to(root))
    )
    for path in files:
        problems.extend(check_decision(path, root))
        match = ADR_NAME.match(path.name)
        if match is None:
            continue
        number = path.name.split("-", 1)[0]
        previous = numbers.get(number)
        rel = path.relative_to(root).as_posix()
        if previous is not None:
            problems.append(
                Problem(rel, 1, f"decision number {number} is also used by {previous}")
            )
        else:
            numbers[number] = rel
    return problems


def check_citation_text(text: str, rel: str) -> list[Problem]:
    """Check the Citation File Format fields this repo requires."""
    problems: list[Problem] = []
    if re.search(r"(?m)^cff-version:\s*1\.2\.0\s*$", text) is None:
        problems.append(Problem(rel, 1, "cff-version must be 1.2.0"))
    if re.search(r"(?m)^message:\s*\S+", text) is None:
        problems.append(Problem(rel, 1, "CITATION.cff has no message"))
    if re.search(r"(?m)^title:\s*\S+", text) is None:
        problems.append(Problem(rel, 1, "CITATION.cff has no title"))
    if re.search(r"(?m)^authors:\s*$", text) is None:
        problems.append(Problem(rel, 1, "CITATION.cff has no authors"))
    elif re.search(r"(?m)^[ ]+(?:-\s*)?(?:family-names|name):\s*\S+", text) is None:
        problems.append(Problem(rel, 1, "an author needs family-names or name"))
    match = re.search(r"(?m)^preferred-citation:\s*$", text)
    if match is None:
        problems.append(Problem(rel, 1, "CITATION.cff has no preferred-citation"))
        return problems
    block: list[str] = []
    for line in text[match.end() :].splitlines():
        if line.strip() == "" or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            break
        block.append(line)
    body = "\n".join(block)
    for key in ("type", "title", "authors", "year"):
        if re.search(rf"(?m)^[ ]+{key}:\s*\S*", body) is None:
            problems.append(Problem(rel, 1, f"preferred-citation has no {key}"))
    if re.search(r"(?m)^[ ]+year:\s*\d{4}\s*$", body) is None:
        problems.append(Problem(rel, 1, "preferred-citation year is not YYYY"))
    return problems


def check_citations(root: Path) -> list[Problem]:
    """Check the root citation file and every copy under ``templates``."""
    problems: list[Problem] = []
    paths = [root / "CITATION.cff"]
    paths.extend(sorted((root / "templates").glob("**/CITATION.cff")))
    seen: set[Path] = set()
    for path in paths:
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        rel = path.relative_to(root)
        if _vendor(rel):
            continue
        label = rel.as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError as exc:
            problems.append(Problem(label, 1, f"not valid UTF-8 ({exc})"))
            continue
        problems.extend(check_citation_text(text, label))
    return problems


def check_tree(root: Path) -> list[Problem]:
    """Check a repository root."""
    problems: list[Problem] = []
    readme = root / "README.md"
    if not readme.is_file():
        problems.append(Problem("README.md", 1, "the repository root has no README.md"))
    else:
        problems.extend(check_readme_file(readme, root, prose=True))
    template = root / "templates" / "README.template.md"
    if template.is_file():
        problems.extend(check_readme_file(template, root, prose=True))
    problems.extend(check_decisions(root))
    problems.extend(check_citations(root))
    return problems


def main(argv: list[str] | None = None) -> int:
    """Check the repository and return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=None,
        help="repository root (default: the parent of this script)",
    )
    args = parser.parse_args(argv)
    root = (args.root or Path(__file__).resolve().parent.parent).resolve()
    if not root.is_dir():
        print(f"{root}: not a directory", file=sys.stderr)
        return 2
    problems = check_tree(root)
    if problems:
        for problem in problems:
            print(problem)
        print(f"{len(problems)} problem(s)", file=sys.stderr)
        return 1
    print(f"ok: {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
