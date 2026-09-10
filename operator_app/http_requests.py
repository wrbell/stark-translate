"""HTTP(S)-only requests for local configuration and pinned model setup.

These guards restrict URL/redirect schemes; they are not host authentication or
an SSRF sandbox. Configured local HTTP servers remain supported.
"""

from __future__ import annotations

import io
import urllib.request
from urllib.parse import urljoin, urlsplit


class HTTPPolicyError(ValueError):
    """A configured URL or response violates the HTTP request policy."""


def validate_http_url(url: str) -> str:
    """Reject non-network schemes and malformed authorities before I/O."""
    try:
        parsed = urlsplit(url)
        valid = (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and not parsed.fragment
            and not any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in url)
            and (parsed.port is None or 1 <= parsed.port <= 65535)
        )
    except (TypeError, ValueError) as exc:
        raise HTTPPolicyError("URL must use HTTP or HTTPS with a valid host and port") from exc
    if not valid:
        raise HTTPPolicyError("URL must use HTTP or HTTPS with a valid host/port and no fragment or whitespace")
    return url


class _HTTPRedirects(urllib.request.HTTPRedirectHandler):
    def http_error_302(self, req, fp, code, msg, headers):
        location = headers.get("Location") or headers.get("URI")
        if location is None:
            return None
        try:
            validate_http_url(urljoin(req.full_url, location))
        except (ValueError, TypeError) as exc:
            fp.close()
            raise HTTPPolicyError("Redirect must use a valid HTTP or HTTPS URL") from exc
        # urllib drains redirect bodies without a size limit. Close the actual
        # response immediately; an empty substitute retains its method/header
        # and loop-limit behavior without consuming untrusted redirect bytes.
        fp.close()
        return super().http_error_302(req, io.BytesIO(), code, msg, headers)

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # The default handler can follow FTP. Apply the same policy to every hop
        # before constructing its Request, including absolute/relative redirects.
        validate_http_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_http(request: str | urllib.request.Request, *, timeout: float):
    """Open an HTTP(S) request with a per-request redirect policy and timeout."""
    validate_http_url(request.full_url if isinstance(request, urllib.request.Request) else request)
    return urllib.request.build_opener(_HTTPRedirects()).open(request, timeout=timeout)


def read_bounded(response, *, limit: int) -> bytes:
    """Consume at most limit + 1 bytes; reject oversized health responses."""
    if limit < 1:
        raise ValueError("Response limit must be positive")
    length = response.headers.get("Content-Length")
    if length is not None:
        try:
            declared = int(length)
        except ValueError:
            declared = None
        if declared is not None and declared > limit:
            raise HTTPPolicyError(f"HTTP response exceeds {limit} bytes")
    body = response.read(limit + 1)
    if len(body) > limit:
        raise HTTPPolicyError(f"HTTP response exceeds {limit} bytes")
    return body
