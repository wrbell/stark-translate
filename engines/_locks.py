"""Shared module-level locks for engines that touch global runtime state.

Why this module exists: multiple call sites (the live pipeline's Silero VAD
path, the HF MarianMT partial translator) hit PyTorch from different threads.
PyTorch is not thread-safe on macOS — concurrent forward passes from different
threads cause Metal heap corruption (intermittent SIGSEGV). Historically two
independent locks lived in ``dry_run_ab.py`` and ``engines/mlx_engine.py``,
which did not actually serialize the two call paths against each other. This
module exposes a single ``_pytorch_lock`` shared by both.

Note: ``MarianCT2Engine`` (CTranslate2) does NOT acquire this lock — CT2 is
internally thread-safe per https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html
and benefits from concurrent calls. The lock is only required on the HF
PyTorch path and for Silero VAD.
"""

import threading

_pytorch_lock = threading.Lock()
