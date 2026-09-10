"""Exercise the actual LAN display handler over HTTP without pipeline imports."""

from __future__ import annotations

import ast
import http.client
import re
from pathlib import Path
from urllib.parse import urljoin

import pytest

from tools.display_server import MAX_ASSET_BYTES, PUBLIC_ASSETS, start_display_server

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = b"private recording and configuration must never be served"


@pytest.fixture(scope="module")
def display_http(tmp_path_factory):
    root = tmp_path_factory.mktemp("audience-assets")
    for relative in PUBLIC_ASSETS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"public {relative}".encode())
    for relative in (
        ".env",
        "settings.py",
        "metrics/diagnostics_session.jsonl",
        "metrics/session.log",
        "stark_data/live_sessions/session/chunk_0001.wav",
        "models/private.gguf",
        "displays/private.html",
        "displays/README.md",
        "displays/operator/index.html",
        "displays/operator/app.js",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PRIVATE)
    server = start_display_server(0, root, host="127.0.0.1")
    try:
        yield root, server
    finally:
        server.shutdown()
        server.server_close()


def request(server, path, method="GET"):
    client = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=2)
    try:
        client.request(method, path)
        response = client.getresponse()
        return response.status, response.read(), dict(response.getheaders())
    finally:
        client.close()


@pytest.mark.parametrize("relative", sorted(PUBLIC_ASSETS))
def test_public_pages_and_required_assets_keep_existing_urls(display_http, relative):
    _, server = display_http
    status, body, headers = request(server, "/" + relative + "?port=9876")
    assert status == 200 and body == f"public {relative}".encode()
    expected_type = "text/html" if relative.endswith(".html") else "application/javascript"
    assert headers["Content-Type"] == expected_type + "; charset=utf-8"
    assert headers["X-Content-Type-Options"] == "nosniff"
    status, body, headers = request(server, "/" + relative, "HEAD")
    assert status == 200 and body == b""
    assert int(headers["Content-Length"]) == len(f"public {relative}".encode())


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/displays/",
        "/displays",
        "/displays/operator/",
        "/.env",
        "/settings.py",
        "/metrics/diagnostics_session.jsonl",
        "/metrics/session.log",
        "/stark_data/live_sessions/session/chunk_0001.wav",
        "/models/private.gguf",
        "/displays/private.html",
        "/displays/README.md",
        "/displays/operator/index.html",
        "/displays/operator/app.js",
        "/displays/../.env",
        "/displays/%2e%2e/.env",
        "/displays/%252e%252e/.env",
        "/displays/%2e%2e%2f.env",
        "/displays/..%5c.env",
        "/displays/..\\.env",
        "/displays/mobile_display.html/../../.env",
        "/displays/mobile_display.html%00",
        "/%64isplays/mobile_display.html",
        "http://example.org/displays/mobile_display.html",
    ],
)
def test_private_files_directories_and_traversal_are_unreachable(display_http, path):
    _, server = display_http
    status, body, headers = request(server, path)
    assert status == 404
    assert PRIVATE not in body
    assert "Location" not in headers  # No directory redirect or listing.
    assert request(server, path, "HEAD")[:2] == (404, b"")


@pytest.mark.parametrize("target", [".env", "displays/church_display.html"])
def test_allowlisted_file_cannot_be_a_symlink_even_inside_displays(display_http, target):
    root, server = display_http
    path = root / "displays/mobile_display.html"
    content = path.read_bytes()
    path.unlink()
    try:
        path.symlink_to(root / target)
        assert request(server, "/displays/mobile_display.html")[0] == 404
    finally:
        path.unlink(missing_ok=True)
        path.write_bytes(content)


def test_allowlisted_asset_rejects_a_symlinked_parent_directory(display_http):
    root, server = display_http
    directory = root / "displays/operator/widgets"
    saved = root / "widget-backup"
    directory.rename(saved)
    try:
        directory.symlink_to(saved, target_is_directory=True)
        assert request(server, "/displays/operator/widgets/qr.js")[0] == 404
    finally:
        directory.unlink(missing_ok=True)
        saved.rename(directory)


def test_allowlisted_directory_and_oversized_file_are_rejected(display_http):
    root, server = display_http
    path = root / "displays/mobile_display.html"
    content = path.read_bytes()
    path.unlink()
    try:
        path.mkdir()
        assert request(server, "/displays/mobile_display.html")[0] == 404
        path.rmdir()
        with path.open("wb") as stream:
            stream.truncate(MAX_ASSET_BYTES + 1)
        assert request(server, "/displays/mobile_display.html")[0] == 404
    finally:
        if path.is_dir():
            path.rmdir()
        path.write_bytes(content)


def test_all_shipped_display_script_dependencies_are_allowlisted():
    for relative in PUBLIC_ASSETS:
        path = ROOT / relative
        assert path.is_file(), relative
        if path.suffix == ".html":
            for source in re.findall(r'<script\b[^>]*\bsrc="([^"]+)"', path.read_text()):
                resolved = urljoin("/" + relative, source).lstrip("/")
                assert resolved in PUBLIC_ASSETS, (relative, source)


def test_pipeline_delegates_http_start_to_display_only_server(monkeypatch, tmp_path):
    # Execute the actual integration function, without importing any ML modules.
    source = ROOT / "dry_run_ab.py"
    tree = ast.parse(source.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "start_http_server"
    )
    calls = []
    sentinel = object()

    def start(port, directory):
        calls.append((port, directory))
        return sentinel

    monkeypatch.setattr("tools.display_server.start_display_server", start)
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    assert namespace["start_http_server"](8080, tmp_path) is sentinel
    assert calls == [(8080, tmp_path)]


def test_opened_file_identity_rejects_replacement_race(display_http, monkeypatch):
    import os

    root, server = display_http
    path = root / "displays/mobile_display.html"
    saved = root / "original-mobile.html"
    original_open = os.open

    def swapped_open(candidate, flags, *args, **kwargs):
        if Path(candidate) != path:
            return original_open(candidate, flags, *args, **kwargs)
        path.rename(saved)
        try:
            path.write_bytes(PRIVATE)
            descriptor = original_open(candidate, flags, *args, **kwargs)
        finally:
            path.unlink(missing_ok=True)
            saved.rename(path)
        # The lexical path is restored, but this descriptor points to a different
        # file. The server must compare identities before reading or sending it.
        return descriptor

    monkeypatch.setattr("tools.display_server.os.open", swapped_open)
    status, body, _ = request(server, "/displays/mobile_display.html")
    assert status == 404 and PRIVATE not in body
