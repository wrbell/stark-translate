"""Local operator boundary and browser origin checks; this is not authentication."""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from functools import lru_cache
from urllib.parse import urlsplit

from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
REMOTE_WARNING = (
    "Remote operator access is enabled without authentication. Reachable clients can control sessions "
    "and read private transcripts/audio. Use a trusted restricted network or an authenticated tunnel/proxy; "
    "audience caption ports do not require remote operator access."
)


def _loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower().rstrip(".") == "localhost"


def _authority(value: str, scheme: str) -> tuple[str, str, int]:
    """Parse a Host or Origin authority without accepting userinfo or path tricks."""
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or any(char.isspace() for char in value)
    ):
        raise ValueError("Invalid request authority")
    host = parsed.hostname.lower().rstrip(".")
    port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
    if not 1 <= port <= 65535 or parsed.scheme != scheme:
        raise ValueError("Invalid request scheme or port")
    return parsed.scheme, host, port


@lru_cache(maxsize=16)
def _policy(bind_host: str, extra_hosts: str) -> tuple[bool, frozenset[str]]:
    local_only = _loopback(bind_host)
    allowed = set(LOCAL_HOSTS)
    allowed.update(host.strip().lower().rstrip(".") for host in extra_hosts.split(",") if host.strip())
    if any("*" in host or "/" in host or "@" in host for host in allowed):
        raise ValueError("STARK_OPERATOR_ALLOWED_HOSTS must list exact hostnames or IP addresses, without ports")
    if not local_only:
        if bind_host not in {"0.0.0.0", "::"}:
            allowed.add(bind_host.lower().rstrip("."))
        # Never resolve arbitrary request hosts: that would permit DNS rebinding.
        # Custom proxy/DNS aliases require STARK_OPERATOR_ALLOWED_HOSTS explicitly.
        import psutil

        for addresses in psutil.net_if_addrs().values():
            for address in addresses:
                if address.family in {socket.AF_INET, socket.AF_INET6}:
                    allowed.add(address.address.lower())
        logger.warning(REMOTE_WARNING)
    return local_only, frozenset(allowed)


def configure_operator_host(host: str) -> None:
    """Called by the supported launcher before importing the ASGI application."""
    os.environ["STARK_OPERATOR_BIND_HOST"] = host
    _policy(host, os.environ.get("STARK_OPERATOR_ALLOWED_HOSTS", ""))


class OperatorBoundaryMiddleware:
    """Guard HTTP and WebSockets, while retaining originless local CLI clients."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        reason = None
        try:
            local_only, allowed = _policy(
                os.environ.get("STARK_OPERATOR_BIND_HOST", "127.0.0.1"),
                os.environ.get("STARK_OPERATOR_ALLOWED_HOSTS", ""),
            )
            peer = (scope.get("client") or ("", 0))[0]
            if local_only and not _loopback(peer):
                reason = "Operator is restricted to local connections"
            headers = scope.get("headers", ())
            hosts = [value.decode("latin1") for key, value in headers if key.lower() == b"host"]
            origins = [value.decode("latin1") for key, value in headers if key.lower() == b"origin"]
            if len(hosts) != 1 or len(origins) > 1:
                raise ValueError("A single Host and at most one Origin are required")
            scheme = "https" if scope.get("scheme") in {"https", "wss"} else "http"
            authority = _authority(f"{scheme}://{hosts[0]}", scheme)
            if authority[1] not in allowed:
                reason = "Untrusted operator Host header"
            if origins and _authority(origins[0], scheme) != authority:
                reason = "Cross-origin operator requests are not allowed"
            method = scope.get("method", "GET")
            fetch_sites = [value for key, value in headers if key.lower() == b"sec-fetch-site"]
            if (
                not origins
                and (scope["type"] == "websocket" or method not in {"GET", "HEAD", "OPTIONS"})
                and any(site not in {b"same-origin", b"none"} for site in fetch_sites)
            ):
                reason = "Cross-origin operator requests are not allowed"
        except (ValueError, OSError) as exc:
            reason = str(exc)
        if reason:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008, "reason": reason})
            else:
                await JSONResponse({"detail": reason}, status_code=403)(scope, receive, send)
            return
        await self.app(scope, receive, send)
