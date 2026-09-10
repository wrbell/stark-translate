"""Cleanup only subprocesses owned by this runner, including detached children."""

from __future__ import annotations

import os
import signal

import psutil


def descendants(pid):
    try:
        return {p.pid: p.create_time() for p in psutil.Process(pid).children(recursive=True)}
    except (psutil.Error, OSError):
        return {}


def cleanup_children(pid, known=None, *, group=True):
    """Bounded cleanup with creation-time checks to avoid targeting reused PIDs."""
    children = dict(known or {})
    children.update(descendants(pid))
    processes = []
    for child, born in children.items():
        try:
            proc = psutil.Process(child)
            if proc.create_time() == born:
                proc.terminate()
                processes.append(proc)
        except (psutil.Error, OSError):
            pass
    _, alive = psutil.wait_procs(processes, timeout=0.5)
    for proc in alive:
        try:
            proc.kill()
        except (psutil.Error, OSError):
            pass
    psutil.wait_procs(alive, timeout=0.5)
    # Popen(start_new_session=True) gives this owner a distinct process group.
    if group and os.name != "nt":
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
