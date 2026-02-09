# macOS libomp Conflict — Diagnosis and Fix

## Problem

`python dry_run_ab.py` crashes with a segfault during model loading:

```
OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
OMP: Error #179: Function pthread_mutex_init failed:
OMP: System error #22: Invalid argument
zsh: segmentation fault  python dry_run_ab.py
```

## Root Cause

Three separate copies of `libomp.dylib` (OpenMP runtime) exist on the system:

| Source | Path | Install Name |
|--------|------|-------------|
| PyTorch | `stt_env/.../torch/lib/libomp.dylib` (755 KB) | `/opt/homebrew/opt/libomp/lib/libomp.dylib` |
| Anaconda | `/Users/willem/anaconda3/lib/libomp.dylib` (902 KB) | `@rpath/libomp.dylib` |
| Homebrew | `/opt/homebrew/opt/libomp/` | — |

The venv (`stt_env`) was created from Anaconda's Python 3.11, so Anaconda's `lib/` is on the dynamic linker's search path.

**The conflict chain:**

1. `import torch` loads PyTorch's `libomp.dylib` — OpenMP initializes normally
2. During model loading, **numba** gets imported as a transitive dependency (via `librosa` or audio processing libs)
3. numba's `omppool.cpython-311-darwin.so` links against `@rpath/libomp.dylib`, which resolves to **Anaconda's** copy
4. Two different libomp instances now share global process state
5. When PyTorch's first `model.generate()` call triggers OpenMP's thread pool initialization, `pthread_mutex_init` finds corrupted mutex attributes from the other libomp → EINVAL → segfault

**Why it was hard to find:** The crash is intermittent and only happens in the full `dry_run_ab.py` script, not in isolated test scripts. numba isn't imported directly — it's pulled in by a transitive dependency, and its `omppool.so` is loaded lazily when numba's threading layer initializes.

## Fix

Two lines at the top of `dry_run_ab.py`, before any other imports:

```python
import os
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"  # Prevent numba from loading its own libomp
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"        # Safety net for any remaining duplicates
```

`NUMBA_THREADING_LAYER=workqueue` tells numba to use its simple built-in thread pool instead of OpenMP. numba's `omppool.so` never loads, Anaconda's libomp never gets pulled in, no conflict.

## What Didn't Work

| Approach | Result |
|----------|--------|
| `KMP_DUPLICATE_LIB_OK=TRUE` alone | Suppresses the "duplicate library" abort but the two copies still corrupt each other's mutex state |
| `OMP_NUM_THREADS=1` | Doesn't prevent libomp initialization, just limits thread count |
| `VECLIB_MAXIMUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OMP_MAX_ACTIVE_LEVELS=1` | Same — doesn't prevent the duplicate load |
| Setting env vars in shell before `python` | Same — the conflict is at shared library load time during Python execution |
| `torch.set_num_threads(1)` / `torch.set_num_interop_threads(1)` | Reduced frequency but didn't eliminate the race |
| Reordering model loads (MarianMT before MLX) | Moved the crash point but didn't prevent it |

## Related Fix: CTranslate2 Removed

CTranslate2 was previously used for fast MarianMT int8 inference (~50ms partials). CT2 also bundles its own libomp, creating a three-way conflict. Removed CT2 entirely — PyTorch MarianMT handles all partial translations now (~80ms, negligible difference for the pipeline).

## Related Fix: MLX Thread Safety

MLX's Metal backend is not thread-safe. All MLX inference (Whisper STT + TranslateGemma translation) must be serialized on a single thread:

```python
_pipeline_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")
```

Running MLX operations on separate threads causes SIGSEGV from concurrent Metal GPU access.

## Diagnosis Method

1. `DYLD_PRINT_LIBRARIES=1 python -c "import torch"` — showed only one libomp at import time
2. Isolated test scripts reproducing the exact model loading sequence — always worked
3. Debug prints in `load_marian()` — pinpointed crash to `model.generate()` (first PyTorch GEMM after MLX)
4. `find stt_env -name "*.so" | xargs otool -L | grep omp` — found numba's `omppool.so` linking against `@rpath/libomp.dylib`
5. `python -c "import torch, mlx_lm, ...; print('numba' in sys.modules)"` — confirmed numba is loaded as a transitive dependency

## Environment

- macOS 15 (Darwin 25.2.0), Apple M3 Pro, 18 GB unified memory
- Python 3.11.11 (Anaconda-based venv)
- PyTorch 2.x, MLX, numba (transitive dep)
