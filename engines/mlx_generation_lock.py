"""Serialize access to one MLX model while allowing distinct models to overlap."""

from __future__ import annotations

import threading
import weakref
from contextlib import ExitStack, contextmanager

_registry_lock = threading.RLock()
_locks: dict[int, threading.RLock] = {}


def _forget(key: int) -> None:
    with _registry_lock:
        _locks.pop(key, None)


def _lock_for(model):
    key = id(model)
    with _registry_lock:
        if key not in _locks:
            _locks[key] = threading.RLock()
            try:
                weakref.finalize(model, _forget, key)
            except TypeError:
                # Some test doubles/native holders cannot be weak-referenced.
                # Production nn.Module instances release their entries on GC.
                pass
        return _locks[key]


@contextmanager
def generation_guard(*models):
    """Lock target and draft in deterministic order to avoid A/B deadlocks."""
    unique = {id(model): model for model in models if model is not None}
    with ExitStack() as stack:
        for key in sorted(unique):
            stack.enter_context(_lock_for(unique[key]))
        yield
