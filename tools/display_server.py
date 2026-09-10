"""Serve only the public audience display bundle, never the application directory.

The audience HTTP port is intentionally reachable on the LAN. Control APIs,
recordings, model files and source/configuration are not part of this server.
"""

from __future__ import annotations

import functools
import http.server
import os
import stat
import threading
from pathlib import Path

# Explicit assets, not an extension or directory allowlist. In particular, the
# operator control page and arbitrary files added beneath displays stay private.
PUBLIC_ASSETS = frozenset(
    {
        "displays/ab_display.html",
        "displays/audience_display.html",
        "displays/church_display.html",
        "displays/mobile_display.html",
        "displays/obs_overlay.html",
        "displays/display_connection.js",
        "displays/caption_telemetry.js",
        "displays/operator/widgets/qr.js",
    }
)
MAX_ASSET_BYTES = 2 * 1024 * 1024


def _asset_bytes(root: Path, relative: str) -> bytes:
    """Open only a confined regular file, rejecting symlinks and replacement races."""
    path = root / relative
    # root is the trusted supplied application/package root, already resolved.
    # Any redirected component below it (including Windows junctions) is denied.
    if path.resolve(strict=True) != path:
        raise OSError("Redirected display asset")
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise OSError("Symlink display asset")
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_ASSET_BYTES:
        raise OSError("Invalid display asset")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or path.resolve(strict=True) != path
        ):
            raise OSError("Display asset changed while opening")
        data = stream.read(MAX_ASSET_BYTES + 1)
    if len(data) > MAX_ASSET_BYTES:
        raise OSError("Display asset exceeds size limit")
    return data


class DisplayAssetHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, directory: Path, **kwargs):
        self.asset_root = directory
        super().__init__(*args, **kwargs)

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def do_GET(self):
        self._serve(send_body=True)

    def do_HEAD(self):
        self._serve(send_body=False)

    def _serve(self, *, send_body: bool):
        try:
            # Match the raw path exactly. Do not normalize/decode traversal or
            # alternate separators into a public name. Port queries are allowed.
            path = self.path.partition("?")[0]
            relative = path[1:] if path.startswith("/") else ""
            if relative not in PUBLIC_ASSETS:
                raise OSError("Not a public display asset")
            data = _asset_bytes(self.asset_root, relative)
        except (OSError, ValueError, RuntimeError):
            self.send_error(404)
            return
        self.send_response(200)
        content_type = "text/html" if relative.endswith(".html") else "application/javascript"
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if send_body:
            self.wfile.write(data)

    def log_message(self, format, *args):
        # Do not echo arbitrary request paths/queries into pipeline logs.
        pass


# Audience display access on the LAN is intentional; the handler is restricted
# to the explicit public bundle above. Operator APIs use a separate server.
def start_display_server(port: int, directory, *, host: str = "0.0.0.0"):  # nosec B104
    root = Path(directory).resolve(strict=True)
    handler = functools.partial(DisplayAssetHandler, directory=root)
    server = http.server.HTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, name="display-http", daemon=True)
    try:
        thread.start()
    except BaseException:
        server.server_close()
        raise
    return server
