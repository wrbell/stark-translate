#!/usr/bin/env python3
"""Fail when a workflow action is not pinned, or permissions are missing.

A third-party ``uses`` value must end in a 40-character commit SHA.
``uses`` counts only as a mapping key on a step or a job.
Text inside a ``run`` block is not a ``uses`` key.
A local reusable workflow (``./...``) and a ``docker://`` image are not
third-party actions.

Every workflow needs a top-level ``permissions`` key.
The value may be ``read-all``, an empty map, or a map of scopes.
Each scope is ``read`` or ``none``. An inline map is the same as a block.
``none`` and an empty map are read-only or tighter.
``write-all`` is never valid.
Job-level write scopes are valid.

The script uses the Python 3.11 standard library only.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_KEY = re.compile(
    r"^(?P<key>(?:\"(?:\\.|[^\"])*\"|'(?:[^']|'')*'|[^:\"'#\s][^:]*?))"
    r"\s*:(?P<rest>.*)$"
)
_BLOCK = re.compile(r"[|>][+-]?\d*|[|>]\d+[+-]?")
_READ_SCOPES = frozenset({"read", "none"})


@dataclass(frozen=True)
class Problem:
    """One workflow pin or permission failure."""

    path: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


class _YamlError(Exception):
    """A workflow file is not in the YAML subset this checker parses."""

    def __init__(self, line: int, message: str) -> None:
        self.line = line
        super().__init__(message)


class _Scalar:
    def __init__(self, value: str, line: int) -> None:
        self.value = value
        self.line = line


class _MapNode:
    def __init__(self, line: int) -> None:
        self.line = line
        self.items: list[tuple[str, int, _Node]] = []


class _SeqNode:
    def __init__(self, line: int) -> None:
        self.line = line
        self.items: list[_Node] = []


_Node = _Scalar | _MapNode | _SeqNode


def _code(line: str) -> str:
    """Return the line without an unquoted YAML comment."""
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index].rstrip()
    return line.rstrip()


def _unquote(text: str) -> str:
    value = text.strip()
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1].replace("''", "'")
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return _unescape(value[1:-1])
    return value


def _unescape(text: str) -> str:
    out: list[str] = []
    index = 0
    while index < len(text):
        if text[index] == "\\" and index + 1 < len(text):
            escaped = text[index + 1]
            out.append(
                {"n": "\n", "t": "\t", "\\": "\\", '"': '"'}.get(escaped, escaped)
            )
            index += 2
            continue
        out.append(text[index])
        index += 1
    return "".join(out)


def _is_block_header(text: str) -> bool:
    return _BLOCK.fullmatch(text.strip()) is not None


def _split_flow(text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_single = False
    in_double = False
    escape = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_single:
            buf.append(char)
            if char == "'" and index + 1 < len(text) and text[index + 1] == "'":
                buf.append(text[index + 1])
                index += 2
                continue
            if char == "'":
                in_single = False
            index += 1
            continue
        if in_double:
            buf.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_double = False
            index += 1
            continue
        if char == "'":
            in_single = True
            buf.append(char)
        elif char == '"':
            in_double = True
            buf.append(char)
        elif char in "{[":
            depth += 1
            buf.append(char)
        elif char in "}]":
            depth -= 1
            buf.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(char)
        index += 1
    parts.append("".join(buf))
    return parts


def _flow_node(text: str, line: int) -> _Node:
    value = text.strip()
    if value.startswith("{"):
        if not value.endswith("}"):
            raise _YamlError(line, "unclosed flow mapping")
        node = _MapNode(line)
        for part in _split_flow(value[1:-1]):
            piece = part.strip()
            if piece == "":
                continue
            match = _KEY.match(piece)
            if match is None:
                raise _YamlError(line, "flow mapping entry is not a key")
            key = _unquote(match.group("key"))
            rest = match.group("rest").strip()
            child = _Scalar("", line) if rest == "" else _flow_node(rest, line)
            node.items.append((key, line, child))
        return node
    if value.startswith("["):
        if not value.endswith("]"):
            raise _YamlError(line, "unclosed flow sequence")
        node = _SeqNode(line)
        for part in _split_flow(value[1:-1]):
            piece = part.strip()
            if piece == "":
                continue
            node.items.append(_flow_node(piece, line))
        return node
    return _Scalar(_unquote(value), line)


class _Parser:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.count = len(lines)

    def parse(self) -> _Node:
        index = self._next_content(0)
        if index >= self.count:
            return _MapNode(1)
        node, _index = self._parse_node(index)
        return node

    def _next_content(self, index: int) -> int:
        while index < self.count:
            code = _code(self.lines[index]).strip()
            if code not in {"", "---", "..."}:
                return index
            index += 1
        return index

    def _indent(self, line: str, line_no: int) -> int:
        spaces = len(line) - len(line.lstrip(" "))
        if line[spaces:].startswith("\t"):
            raise _YamlError(line_no, "workflow indentation uses a tab")
        return spaces

    def _parse_node(self, index: int) -> tuple[_Node, int]:
        start = index
        code = _code(self.lines[index]).strip()
        indent = self._indent(self.lines[index], index + 1)
        if code.startswith("- ") or code == "-":
            node, index = self._parse_seq(index, indent)
        elif _KEY.match(code):
            node, index = self._parse_map(index, indent)
        else:
            raise _YamlError(index + 1, "expected a mapping or a sequence")
        if index <= start:
            raise _YamlError(start + 1, "parser did not advance")
        return node, index

    def _parse_map(self, index: int, indent: int) -> tuple[_MapNode, int]:
        node = _MapNode(index + 1)
        while index < self.count:
            nxt = self._next_content(index)
            if nxt >= self.count or self._indent(self.lines[nxt], nxt + 1) != indent:
                return node, nxt
            code = _code(self.lines[nxt]).strip()
            match = _KEY.match(code)
            if match is None:
                return node, nxt
            key = _unquote(match.group("key"))
            value, index = self._parse_value(nxt, indent, match.group("rest"), nxt + 1)
            node.items.append((key, nxt + 1, value))
        return node, index

    def _parse_seq(self, index: int, indent: int) -> tuple[_SeqNode, int]:
        node = _SeqNode(index + 1)
        while index < self.count:
            nxt = self._next_content(index)
            if nxt >= self.count or self._indent(self.lines[nxt], nxt + 1) != indent:
                return node, nxt
            raw = _code(self.lines[nxt])
            stripped = raw.strip()
            if not (stripped.startswith("- ") or stripped == "-"):
                return node, nxt
            body = raw[indent + 1 :]
            content = body.lstrip(" ")
            content_indent = indent + 1 + (len(body) - len(content))
            value, index = self._parse_seq_item(nxt, indent, content_indent, content)
            node.items.append(value)
        return node, index

    def _parse_seq_item(
        self,
        index: int,
        dash_indent: int,
        content_indent: int,
        content: str,
    ) -> tuple[_Node, int]:
        if content == "":
            nxt = self._next_content(index + 1)
            if (
                nxt >= self.count
                or self._indent(self.lines[nxt], nxt + 1) <= dash_indent
            ):
                return _Scalar("", index + 1), index + 1
            return self._parse_node(nxt)
        if content[0] in "{[":
            return self._parse_flow(index, content, index + 1)
        if _is_block_header(content):
            return self._parse_block(index, dash_indent, index + 1)
        match = _KEY.match(content)
        if match is None:
            return _Scalar(_unquote(content), index + 1), index + 1
        key_line = index + 1
        value, index = self._parse_value(
            index, content_indent, match.group("rest"), key_line
        )
        node = _MapNode(key_line)
        node.items.append((_unquote(match.group("key")), key_line, value))
        rest, index = self._parse_map(index, content_indent)
        node.items.extend(rest.items)
        return node, index

    def _parse_value(
        self, index: int, key_indent: int, rest: str, key_line: int
    ) -> tuple[_Node, int]:
        text = rest.strip()
        if _is_block_header(text):
            return self._parse_block(index, key_indent, key_line)
        if text.startswith(("{", "[")):
            return self._parse_flow(index, text, key_line)
        if text == "":
            nxt = self._next_content(index + 1)
            if nxt < self.count and self._indent(self.lines[nxt], nxt + 1) > key_indent:
                return self._parse_node(nxt)
            return _Scalar("", key_line), index + 1
        nxt = self._next_content(index + 1)
        if nxt < self.count and self._indent(self.lines[nxt], nxt + 1) > key_indent:
            raise _YamlError(nxt + 1, "unexpected indented line after a plain value")
        return _Scalar(_unquote(text), key_line), index + 1

    def _parse_block(
        self, index: int, parent_indent: int, key_line: int
    ) -> tuple[_Scalar, int]:
        index += 1
        content_indent: int | None = None
        chunks: list[str] = []
        while index < self.count:
            line = self.lines[index]
            if line.strip() == "":
                chunks.append("")
                index += 1
                continue
            indent = self._indent(line, index + 1)
            if indent <= parent_indent:
                break
            if content_indent is None:
                content_indent = indent
            if indent < content_indent:
                break
            chunks.append(line[content_indent:])
            index += 1
        return _Scalar("\n".join(chunks), key_line), index

    def _parse_flow(self, index: int, first: str, key_line: int) -> tuple[_Node, int]:
        opener = first.lstrip()[0]
        text, nxt = self._read_flow(index, first.strip())
        if text[0] != opener:
            raise _YamlError(key_line, "expected a flow collection")
        return _flow_node(text, key_line), nxt

    def _read_flow(self, index: int, first: str) -> tuple[str, int]:
        buf = first
        line_no = index
        pos = 0
        depth = 0
        in_single = False
        in_double = False
        escape = False
        while True:
            if pos >= len(buf):
                if depth == 0 and buf:
                    break
                line_no += 1
                if line_no >= self.count:
                    raise _YamlError(index + 1, "unclosed flow collection")
                buf = f"{buf}\n{self.lines[line_no]}"
                continue
            char = buf[pos]
            if in_single:
                if char == "'" and pos + 1 < len(buf) and buf[pos + 1] == "'":
                    pos += 2
                    continue
                if char == "'":
                    in_single = False
                pos += 1
                continue
            if in_double:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_double = False
                pos += 1
                continue
            if char == "'":
                in_single = True
            elif char == '"':
                in_double = True
            elif char == "#":
                newline = buf.find("\n", pos)
                if newline == -1:
                    buf = buf[:pos].rstrip()
                    pos = len(buf)
                    continue
                buf = buf[:pos].rstrip() + buf[newline:]
                continue
            elif char in "{[":
                depth += 1
            elif char in "}]":
                depth -= 1
                if depth == 0:
                    end = pos + 1
                    leftover = buf[end:].strip()
                    if leftover and not leftover.startswith("#"):
                        raise _YamlError(
                            line_no + 1, "trailing text after a flow collection"
                        )
                    return buf[:end], line_no + 1
            pos += 1
        raise _YamlError(index + 1, "unclosed flow collection")


def _parse_document(text: str) -> _Node:
    return _Parser(text.splitlines()).parse()


def _root_map(node: _Node) -> _MapNode | None:
    if isinstance(node, _MapNode):
        return node
    return None


def _check_top_permissions(rel: str, value: _Node) -> list[Problem]:
    if isinstance(value, _Scalar):
        if value.value == "read-all":
            return []
        if value.value == "write-all":
            return [Problem(rel, value.line, "permissions use write-all")]
        return [
            Problem(
                rel,
                value.line,
                "top-level permissions must be read-all or read scopes",
            )
        ]
    if not isinstance(value, _MapNode):
        return [
            Problem(
                rel,
                1,
                "top-level permissions must be read-all or read scopes",
            )
        ]
    problems: list[Problem] = []
    for _key, line, item in value.items:
        if not isinstance(item, _Scalar):
            problems.append(
                Problem(rel, line, "top-level permissions entry is not a scope")
            )
            continue
        if item.value == "write-all":
            problems.append(Problem(rel, line, "permissions use write-all"))
        elif item.value not in _READ_SCOPES:
            problems.append(
                Problem(
                    rel,
                    line,
                    "top-level permissions grant a scope other than read",
                )
            )
    return problems


def _write_all_in(rel: str, value: _Node) -> list[Problem]:
    if isinstance(value, _Scalar):
        if value.value == "write-all":
            return [Problem(rel, value.line, "permissions use write-all")]
        return []
    if isinstance(value, _MapNode):
        problems: list[Problem] = []
        for _key, line, item in value.items:
            if isinstance(item, _Scalar) and item.value == "write-all":
                problems.append(Problem(rel, line, "permissions use write-all"))
            else:
                problems.extend(_write_all_in(rel, item))
        return problems
    if isinstance(value, _SeqNode):
        problems = []
        for item in value.items:
            problems.extend(_write_all_in(rel, item))
        return problems
    return []


def _nested_write_all(rel: str, node: _Node, *, at_root: bool) -> list[Problem]:
    problems: list[Problem] = []
    if isinstance(node, _MapNode):
        for key, _line, value in node.items:
            if key == "permissions" and not at_root:
                problems.extend(_write_all_in(rel, value))
            elif key != "permissions" or not at_root:
                problems.extend(_nested_write_all(rel, value, at_root=False))
    elif isinstance(node, _SeqNode):
        for item in node.items:
            problems.extend(_nested_write_all(rel, item, at_root=False))
    return problems


def _check_uses_value(rel: str, line: int, value: _Node) -> list[Problem]:
    if not isinstance(value, _Scalar):
        return [Problem(rel, line, "uses value is not pinned to a SHA")]
    text = value.value.strip()
    if text.startswith(("./", "docker://")):
        return []
    if "@" not in text:
        return [Problem(rel, line, f"uses value {text!r} is not pinned to a SHA")]
    ref = text.rsplit("@", 1)[1]
    if SHA.fullmatch(ref) is None:
        return [
            Problem(
                rel,
                line,
                f"uses value {text!r} is not pinned to a 40-character SHA",
            )
        ]
    return []


def _check_uses(rel: str, node: _Node) -> list[Problem]:
    problems: list[Problem] = []
    if isinstance(node, _MapNode):
        for key, line, value in node.items:
            if key == "uses":
                problems.extend(_check_uses_value(rel, line, value))
            else:
                problems.extend(_check_uses(rel, value))
    elif isinstance(node, _SeqNode):
        for item in node.items:
            problems.extend(_check_uses(rel, item))
    return problems


def check_text(rel: str, text: str) -> list[Problem]:
    """Return pin and permission problems for one workflow file."""
    try:
        doc = _parse_document(text)
    except _YamlError as exc:
        return [Problem(rel, exc.line, f"workflow YAML could not be parsed ({exc})")]
    root = _root_map(doc)
    if root is None:
        return [Problem(rel, 1, "workflow has no top-level permissions")]
    problems: list[Problem] = []
    permissions = [
        (line, value) for key, line, value in root.items if key == "permissions"
    ]
    if not permissions:
        problems.append(Problem(rel, 1, "workflow has no top-level permissions"))
    for _line, value in permissions:
        problems.extend(_check_top_permissions(rel, value))
    problems.extend(_nested_write_all(rel, root, at_root=True))
    problems.extend(_check_uses(rel, root))
    return problems


def workflow_files(path: Path) -> list[Path]:
    """Return workflow files in ``path`` (a file or one directory)."""
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix in {".yml", ".yaml"}
    )


def check_paths(paths: list[Path]) -> list[Problem]:
    """Check every workflow file under the given paths."""
    problems: list[Problem] = []
    for path in paths:
        for workflow in workflow_files(path):
            try:
                text = workflow.read_text(encoding="utf-8")
            except UnicodeError as exc:
                problems.append(
                    Problem(workflow.as_posix(), 1, f"not valid UTF-8 ({exc})")
                )
                continue
            problems.extend(check_text(workflow.as_posix(), text))
    return problems


def main(argv: list[str] | None = None) -> int:
    """Check workflow files and return a process status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="workflow files or directories (default: .github/workflows)",
    )
    args = parser.parse_args(argv)
    paths = args.paths or [Path(".github/workflows")]
    problems = check_paths(paths)
    if problems:
        for problem in problems:
            print(problem)
        print(f"{len(problems)} problem(s)", file=sys.stderr)
        return 1
    print(f"ok: {len(paths)} path(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
