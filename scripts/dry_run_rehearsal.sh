#!/usr/bin/env bash
# API control rehearsal: starts real sessions using the operator's configured input.
# It does not establish caption quality, audience rendering or physical-device health.
# Run with exclusive use of an idle operator and the intended rehearsed input source.
# OPERATOR_URL defaults to http://localhost:9000.
# REHEARSAL_TIMEOUT_S defaults to 120 seconds per startup/restart/stop.
# REHEARSAL_POLL_S defaults to 1 second. Increase the deadline for cold model loads.
# Requires bash and Python 3; STARK_PYTHON may select the Python executable.
set -euo pipefail

exec "${STARK_PYTHON:-python3}" -u - <<'PY'
import json
import math
import os
import signal
import sys
import time
import urllib.error
import urllib.request


class RehearsalFailure(Exception):
    pass


def positive_setting(name, default):
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise RehearsalFailure(f"{name} must be a positive finite number")
    return value


URL = os.environ.get("OPERATOR_URL", "http://localhost:9000").rstrip("/")
owned_session = None


def request(path, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(2048).decode(errors="replace")
        raise RehearsalFailure(f"{req.get_method()} {path}: HTTP {exc.code}: {detail}") from exc
    except (OSError, ValueError) as exc:
        raise RehearsalFailure(f"{req.get_method()} {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(result, dict):
        raise RehearsalFailure(f"{path}: expected a JSON object")
    return result


def identity(snapshot, session, lang):
    if snapshot.get("session_id") != session:
        raise RehearsalFailure("Session ownership changed; refusing further controls")
    if snapshot.get("config", {}).get("lang") != lang:
        raise RehearsalFailure(f"Session {session}: expected language {lang}")


def describe(snapshot):
    readiness = snapshot.get("readiness") or {}
    return f"state={snapshot.get('state')}, phase={readiness.get('phase')}, reason={snapshot.get('error') or readiness.get('reason')}"


def wait_for(session, lang, snapshot, *, stopped=False):
    deadline = time.monotonic() + timeout_s
    while True:
        identity(snapshot, session, lang)
        state = snapshot.get("state")
        readiness = snapshot.get("readiness") or {}
        health = snapshot.get("health") or {}
        if stopped and state == "idle":
            if snapshot.get("outcome") != "completed":
                raise RehearsalFailure(f"Session {session}: stop outcome={snapshot.get('outcome')}; completion was not proved")
            return
        if state == "error" or snapshot.get("error") or snapshot.get("outcome") in {"failed", "interrupted"}:
            raise RehearsalFailure(f"Session {session}: {describe(snapshot)}")
        if not stopped:
            if state in {"idle", "paused", "stopping"} or readiness.get("phase") in {"input_error", "failed"}:
                raise RehearsalFailure(f"Session {session}: {describe(snapshot)}")
            if health.get("recording", {}).get("ok") is False:
                raise RehearsalFailure(f"Session {session}: recording has required failures")
            if (
                state == "running"
                and readiness.get("ready") is True
                and readiness.get("stale") is False
                and health.get("stale") is False
                and health.get("input_seen") is True
            ):
                print(f"  ✓ ready: {session} ({lang})")
                return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            target = "completed stop" if stopped else "fresh audio readiness"
            raise RehearsalFailure(f"Timed out after {timeout_s:g}s waiting for {target}: {describe(snapshot)}")
        time.sleep(min(poll_s, remaining))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            continue
        snapshot = request("/api/session/status", timeout=min(5, remaining))


def accept_started(snapshot, lang, previous=None):
    global owned_session
    session = snapshot.get("session_id")
    if not isinstance(session, str) or not session:
        raise RehearsalFailure("Start/restart returned no session identity; inspect operator status")
    # The successful mutation response is our ownership receipt, even if its
    # readiness subsequently fails. Never infer ownership from a status poll.
    owned_session = session
    if session == previous:
        raise RehearsalFailure("Language restart reused the previous session identity")
    wait_for(session, lang, snapshot)
    return session


def cleanup():
    if owned_session is None:
        return
    try:
        current = request("/api/session/status", timeout=5)
        if current.get("session_id") != owned_session:
            print("  ! Cleanup skipped: active session belongs to someone else", file=sys.stderr)
        elif current.get("state") != "idle":
            result = request("/api/session/stop", {})
            if result.get("session_id") != owned_session or result.get("state") != "idle":
                raise RehearsalFailure("Owned session did not stop; inspect the operator")
            print(f"  ! Cleaned up owned session {owned_session}", file=sys.stderr)
    except Exception as exc:
        print(f"  ! Cleanup failed: {exc}; inspect the operator before retrying", file=sys.stderr)


def interrupted(signum, _frame):
    raise RehearsalFailure(f"Interrupted by signal {signum}")


signal.signal(signal.SIGINT, interrupted)
signal.signal(signal.SIGTERM, interrupted)
exit_code = 1
try:
    timeout_s = positive_setting("REHEARSAL_TIMEOUT_S", "120")
    poll_s = positive_setting("REHEARSAL_POLL_S", "1")
    print("[1] Health probe and idle ownership check")
    if request("/healthz").get("status") != "ok":
        raise RehearsalFailure("/healthz did not return status=ok")
    if request("/api/session/status").get("state") != "idle":
        raise RehearsalFailure("Operator is not idle; no session was started or stopped")
    print("[2] Pre-flight (English, default configuration)")
    preflight = request("/api/preflight?lang=en")
    if preflight.get("ok") is not True or preflight.get("status_counts", {}).get("fail") != 0:
        raise RehearsalFailure("Pre-flight has failed or missing checks; resolve them before starting")
    print("[3] Start and wait for fresh audio readiness")
    session = accept_started(request("/api/session/start", {"lang": "en"}), "en")
    print("[4] Language-flip round trip, waiting after each restart")
    lang = "en"
    for target in ("es", "en"):
        identity(request("/api/session/status"), session, lang)
        session = accept_started(request("/api/control/lang_flip", {}), target, previous=session)
        lang = target
    print("[5] Stop and verify completed outcome")
    identity(request("/api/session/status"), session, lang)
    wait_for(session, lang, request("/api/session/stop", {}), stopped=True)
    owned_session = None
    print("[6] Verse highlights endpoint")
    if not isinstance(request("/api/features/verses").get("highlights"), list):
        raise RehearsalFailure("Verse highlights response did not contain a list")
    print("✓ rehearsal passed — control readiness and completion checks passed")
    print("Audio quality, audience rendering and physical devices still require a separate rehearsal.")
    exit_code = 0
except Exception as exc:
    print(f"✗ rehearsal failed — {exc}", file=sys.stderr)
finally:
    # Trap failures and interruption, but never stop a session adopted from a
    # poll or one that replaced ours. One operator/controller is required.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    cleanup()
sys.exit(exit_code)
PY
