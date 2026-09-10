"""Real loopback HTTP and redirect-policy regressions; no model/device access."""

from __future__ import annotations

import hashlib
import io
import json
import threading
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from operator_app import preflight, setup
from operator_app.http_requests import HTTPPolicyError, _HTTPRedirects, open_http, read_bounded, validate_http_url


@contextmanager
def local_http(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()


class Handler(BaseHTTPRequestHandler):
    routes = {}
    requests = []

    def log_message(self, *args):
        pass

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        self.requests.append((self.command, self.path, self.headers.get("Range")))
        status, headers, body = self.routes.get((self.command, self.path), (404, {}, b"missing"))
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass


@pytest.fixture
def handler():
    class LocalHandler(Handler):
        requests = []
        routes = {}

    return LocalHandler


@pytest.mark.parametrize(
    "url",
    [
        "file:///private/tmp/example",
        "data:text/plain,example",
        "ftp://example.invalid/x",
        "https:///missing",
        "http://localhost:70000",
        "http://localhost/#fragment",
        "http://local host",
        "http://localhost/\nhealth",
    ],
)
def test_disallowed_initial_url_fails_before_any_opener(monkeypatch, url):
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: pytest.fail("opener created for invalid URL"))
    with pytest.raises(HTTPPolicyError):
        open_http(url, timeout=1)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8090",
        "http://[::1]:8090/v1",
        "https://huggingface.co/model/resolve/revision/file?token=example",
    ],
)
def test_allowed_http_https_request_parser(url):
    assert validate_http_url(url) == url
    request = _HTTPRedirects().redirect_request(
        urllib.request.Request("http://localhost/start"), None, 302, "Found", {}, url
    )
    assert request.full_url == url


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
@pytest.mark.parametrize(
    "target", ["file:///private/tmp/not-read", "ftp://example.invalid/not-contacted", "data:text/plain,not-read"]
)
def test_real_http_redirect_rejects_non_http_before_following(handler, code, target):
    handler.routes[("GET", "/start")] = (code, {"Location": target}, b"")
    with local_http(handler) as base, pytest.raises(HTTPPolicyError):
        open_http(base + "/start", timeout=1)
    assert handler.requests == [("GET", "/start", None)]


def test_relative_http_redirect_and_health_json_whitespace_work(handler):
    handler.routes[("GET", "/base/health?probe=1")] = (302, {"Location": "/healthy"}, b"")
    handler.routes[("GET", "/healthy")] = (200, {}, b'{"status": "ok"}')
    with local_http(handler) as base:
        result = preflight.check_llamacpp_server(base + "/base?probe=1")
    assert result["status"] == "pass"
    assert [request[1] for request in handler.requests] == ["/base/health?probe=1", "/healthy"]


def test_health_invalid_scheme_is_structured_failure(tmp_path):
    fixture = tmp_path / "harmless"
    fixture.write_text("NEVER_REFLECT_LOCAL_FIXTURE")
    result = preflight.check_llamacpp_server(fixture.as_uri() + "#")
    assert result["status"] == "fail" and "HTTP" in result["detail"]
    assert "NEVER_REFLECT_LOCAL_FIXTURE" not in result["detail"]


@pytest.mark.parametrize("advertise_length", [True, False])
def test_actual_oversized_health_response_is_structured_failure(handler, advertise_length):
    body = b"x" * (16 * 1024 + 10)
    headers = {"Content-Length": str(len(body))} if advertise_length else {}
    handler.routes[("GET", "/health")] = (200, headers, body)
    with local_http(handler) as base:
        result = preflight.check_llamacpp_server(base)
    assert result["status"] == "fail" and "16384" in result["detail"]


def test_bounded_read_never_requests_entire_unadvertised_body():
    class Stream(io.BytesIO):
        headers = {}
        requested = []

        def read(self, size=-1):
            self.requested.append(size)
            return super().read(size)

    response = Stream(b"x" * 200)
    with pytest.raises(HTTPPolicyError):
        read_bounded(response, limit=32)
    assert response.requested == [33]
    assert response.tell() == 33


def test_health_read_timeout_remains_structured(monkeypatch):
    monkeypatch.setattr(preflight, "open_http", lambda *a, **kw: (_ for _ in ()).throw(TimeoutError("fixture timeout")))
    result = preflight.check_llamacpp_server("http://127.0.0.1:8090")
    assert result["status"] == "warn" and "Not reachable" in result["detail"]


def test_redirect_violation_is_visible_in_health_and_setup_check(handler):
    handler.routes[("GET", "/health")] = (302, {"Location": "file:///private/tmp/no-read"}, b"")
    handler.routes[("HEAD", "/asset")] = (302, {"Location": "ftp://example.invalid/no-contact"}, b"")
    with local_http(handler) as base:
        assert preflight.check_llamacpp_server(base)["status"] == "fail"
        status, detail = setup._head_url(base + "/asset", timeout_s=1)
    assert status == "fail" and "HTTPPolicyError" in detail


def test_model_check_head_405_range_get_retains_redirect_policy(handler):
    handler.routes[("HEAD", "/asset")] = (405, {}, b"")
    handler.routes[("GET", "/asset")] = (302, {"Location": "/actual"}, b"")
    handler.routes[("GET", "/actual")] = (206, {}, b"a")
    with local_http(handler) as base:
        assert setup._head_url(base + "/asset", timeout_s=1) == ("pass", "HTTP 206 (range)")
    assert handler.requests == [
        ("HEAD", "/asset", None),
        ("GET", "/asset", "bytes=0-0"),
        ("GET", "/actual", "bytes=0-0"),
    ]


def test_model_check_405_fallback_rejects_ftp_redirect(handler):
    handler.routes[("HEAD", "/asset")] = (405, {}, b"")
    handler.routes[("GET", "/asset")] = (302, {"Location": "ftp://example.invalid/no-contact"}, b"")
    with local_http(handler) as base:
        status, detail = setup._head_url(base + "/asset", timeout_s=1)
    assert status == "fail" and "HTTPPolicyError" in detail


def test_model_setup_local_http_redirect_checks_digest_and_installs(tmp_path, handler):
    body = b"fixture model bytes"
    handler.routes[("GET", "/asset")] = (302, {"Location": "/actual"}, b"")
    handler.routes[("GET", "/actual")] = (200, {"Content-Length": str(len(body))}, body)
    with local_http(handler) as base:
        manifest = {
            "version": "fixture",
            "models": {
                "fixture": {
                    "type": "direct",
                    "url": base + "/asset",
                    "filename": "fixture.bin",
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "size_bytes": len(body),
                    "required_for": ["cuda"],
                }
            },
        }
        (tmp_path / "models.lock.json").write_text(json.dumps(manifest))
        assert setup.bootstrap_models(tmp_path / "cache", project_root=tmp_path, backend="cuda") == 0
    assert (tmp_path / "cache/fixture.bin").read_bytes() == body
    assert (tmp_path / "cache/fixture.bin.installed").is_file()


def test_download_and_check_file_schemes_cannot_copy_fixture(tmp_path):
    source = tmp_path / "fixture"
    source.write_bytes(b"local fixture")
    target = tmp_path / "new-cache/model"
    with pytest.raises(HTTPPolicyError):
        setup._download_direct(source.as_uri(), target, None)
    assert not target.parent.exists()
    assert setup._head_url(source.as_uri(), timeout_s=1)[0] == "fail"


def test_packaging_guard_requires_shared_http_helper(tmp_path):
    import zipfile

    from tools.release_artifacts import RUNTIME_REQUIRED, verify_artifact

    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in RUNTIME_REQUIRED - {"operator_app/http_requests.py"}:
            archive.writestr(name, "fixture")
    with pytest.raises(ValueError, match=r"operator_app/http_requests\.py"):
        verify_artifact(wheel)


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
@pytest.mark.parametrize(
    "target", ["file:///private/tmp/not-read", "ftp://example.invalid/not-contacted", "data:text/plain,no-read"]
)
def test_redirect_parser_rejects_and_closes_response_before_following(code, target):
    from email.message import Message
    from types import SimpleNamespace

    handler = _HTTPRedirects()
    handler.parent = SimpleNamespace(open=lambda *a, **kw: pytest.fail("followed a forbidden redirect"))
    headers = Message()
    headers["Location"] = target
    response = io.BytesIO(b"redirect response")
    with pytest.raises(HTTPPolicyError):
        getattr(handler, f"http_error_{code}")(
            urllib.request.Request("http://localhost/start"), response, code, "Found", headers
        )
    assert response.closed


def test_malformed_redirect_is_structured_and_closes_before_following():
    from email.message import Message

    response = io.BytesIO(b"unread redirect")
    headers = Message()
    headers["Location"] = "http://["
    with pytest.raises(HTTPPolicyError):
        _HTTPRedirects().http_error_302(
            urllib.request.Request("http://localhost/start"), response, 302, "Found", headers
        )
    assert response.closed


def test_valid_redirect_does_not_drain_original_response_body():
    from email.message import Message
    from types import SimpleNamespace

    class NoRead(io.BytesIO):
        def read(self, *args):
            pytest.fail("unbounded redirect response read")

    response = NoRead(b"redirect bytes must not be consumed")
    headers = Message()
    headers["Location"] = "/next"
    request = urllib.request.Request("http://localhost/start")
    request.timeout = 2
    handler = _HTTPRedirects()
    handler.parent = SimpleNamespace(open=lambda new, timeout: (new.full_url, timeout))
    assert handler.http_error_302(request, response, 302, "Found", headers) == ("http://localhost/next", 2)
    assert response.closed


def test_ignored_range_200_restarts_partial_instead_of_duplicating_prefix(tmp_path, handler):
    body = b"abcdef"
    target = tmp_path / "model.bin"
    target.with_suffix(".bin.partial").write_bytes(body[:2])
    handler.routes[("GET", "/model")] = (200, {"Content-Length": str(len(body))}, body)
    with local_http(handler) as base:
        setup._download_direct(base + "/model", target, len(body))
    assert target.read_bytes() == body
    assert handler.requests == [("GET", "/model", "bytes=2-")]
    assert not target.with_suffix(".bin.partial").exists()


def test_verified_206_appends_only_the_requested_suffix(tmp_path, handler):
    target = tmp_path / "model.bin"
    target.with_suffix(".bin.partial").write_bytes(b"ab")
    handler.routes[("GET", "/model")] = (206, {"Content-Length": "4", "Content-Range": "bytes 2-5/6"}, b"cdef")
    with local_http(handler) as base:
        setup._download_direct(base + "/model", target, 6)
    assert target.read_bytes() == b"abcdef"
    assert handler.requests == [("GET", "/model", "bytes=2-")]


@pytest.mark.parametrize(
    "headers",
    [
        {"Content-Length": "4"},
        {"Content-Length": "4", "Content-Range": "bytes 0-3/6"},
        {"Content-Length": "4", "Content-Range": "bytes 2-5/7"},
        {"Content-Length": "3", "Content-Range": "bytes 2-5/6"},
        {"Content-Length": "4", "Content-Range": "bytes 2-5/*"},
    ],
)
def test_bad_206_range_does_not_append_or_promote(tmp_path, handler, headers):
    target = tmp_path / "model.bin"
    target.write_bytes(b"previous target")
    partial = target.with_suffix(".bin.partial")
    partial.write_bytes(b"ab")
    handler.routes[("GET", "/model")] = (206, headers, b"cdef")
    with local_http(handler) as base, pytest.raises(ValueError):
        setup._download_direct(base + "/model", target, 6)
    assert target.read_bytes() == b"previous target" and partial.read_bytes() == b"ab"


@pytest.mark.parametrize(
    "status,headers,body,initial",
    [
        (200, {"Content-Length": "6"}, b"abc", b""),
        (206, {"Content-Length": "4", "Content-Range": "bytes 2-5/6"}, b"cd", b"ab"),
        (206, {"Content-Length": "2", "Content-Range": "bytes 2-3/6"}, b"cd", b"ab"),
        (200, {}, b"abcdefg", b"ab"),
    ],
)
def test_truncated_or_oversized_download_never_promotes_partial(tmp_path, handler, status, headers, body, initial):
    target = tmp_path / "model.bin"
    partial = target.with_suffix(".bin.partial")
    if initial:
        partial.write_bytes(initial)
    handler.routes[("GET", "/model")] = (status, headers, body)
    with local_http(handler) as base, pytest.raises(ValueError):
        setup._download_direct(base + "/model", target, 6)
    assert not target.exists() and partial.exists()
