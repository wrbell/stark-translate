# Automated Adapter Deployment System

> Implementation plan for versioned LoRA adapter deployment from the training desktop
> (WSL/CUDA) to inference endpoints (Mac + NVIDIA church machines). Includes health
> checks, hot-reload, and automatic rollback.

**Status:** Planning
**Date:** 2026-03-01

---

## Problem Statement

After each fine-tuning cycle, adapters are manually transferred via USB or ad-hoc scp.
No versioning, no health checks, no rollback capability. Current pain points:

- No way to know which adapter version is running on which endpoint
- No automated verification that a new adapter produces correct output
- Rolling back means manually finding and copying old files
- Zero-downtime reload not possible (must restart pipeline)

**Goal:** One-command deployment with versioning, health checks, hot-reload (Mac), and
automatic rollback on failure.

---

## 1. Adapter Versioning

### 1.1 Version ID Format

```
cycle{N}_{YYYYMMDD}_{sha256[:8]}
```

Examples:
- `cycle1_20260315_a4b2c8d1` — first fine-tuning cycle, March 15, 2026
- `cycle3_20260401_f1e2d3c4` — third cycle, April 1, 2026

The SHA-256 is computed over the adapter weights file (`adapter_model.safetensors`),
providing content-addressable deduplication and corruption detection.

### 1.2 Storage Layout

**Training machine (WSL):**
```
adapters/
  manifest.json                          # All versions, endpoints, deployment records
  whisper/
    cycle1_20260315_a4b2c8d1/
      adapter_config.json
      adapter_model.safetensors
      training_metrics.json              # Final loss, eval WER, epochs, data size
    cycle2_20260401_b5c6d7e8/
      ...
  gemma_4b/
    cycle1_20260315_f1e2d3c4/
      adapter_config.json
      adapter_model.safetensors
      training_metrics.json
    ...
  gemma_12b/
    cycle1_20260315_e4d3c2b1/
      ...
```

**Inference machines:**
```
adapters/
  whisper/
    active/                              # Currently loaded adapter
      adapter_config.json
      adapter_model.safetensors
    previous/                            # One-step rollback slot
      adapter_config.json
      adapter_model.safetensors
  gemma_4b/
    active/
    previous/
  gemma_12b/
    active/
    previous/
  status.json                            # Current versions, last reload, health
```

### 1.3 Backward Compatibility

Existing `fine_tuned_*/` directories become symlinks to `adapters/{model}/active/`:

```bash
ln -sf adapters/whisper/active fine_tuned_whisper_mi
ln -sf adapters/gemma_4b/active fine_tuned_gemma_mi_A
ln -sf adapters/gemma_12b/active fine_tuned_gemma_mi_B
```

This preserves all existing code paths that reference `fine_tuned_*/` directories.

### 1.4 Manifest Schema

`adapters/manifest.json`:
```json
{
  "versions": {
    "whisper": [
      {
        "version_id": "cycle1_20260315_a4b2c8d1",
        "created": "2026-03-15T14:30:00Z",
        "cycle": 1,
        "sha256": "a4b2c8d1...",
        "training_metrics": {
          "final_loss": 0.42,
          "eval_wer": 8.3,
          "epochs": 3,
          "data_hours": 22.5
        }
      }
    ],
    "gemma_4b": [...],
    "gemma_12b": [...]
  },
  "endpoints": {
    "mac-dev": {
      "host": "192.168.1.100",
      "user": "willem",
      "adapter_dir": "/Users/willem/Code/vibes/SRTranslate/adapters",
      "reload_method": "sigusr1"
    },
    "church-rtx2070-1": {
      "host": "192.168.1.101",
      "user": "stark",
      "adapter_dir": "/home/stark/stark-translate/adapters",
      "reload_method": "restart"
    }
  },
  "deployments": [
    {
      "timestamp": "2026-03-15T15:00:00Z",
      "cycle": 1,
      "models": ["whisper", "gemma_4b"],
      "endpoints": ["mac-dev"],
      "status": "success",
      "health_check_results": {...}
    }
  ]
}
```

---

## 2. Transfer via rsync over SSH

Adapter files are small (LoRA adapters: 60-120 MB each). On gigabit LAN:

| Model | Adapter Size | Transfer Time |
|-------|-------------|---------------|
| Whisper LoRA (r=32) | ~60 MB | <1s |
| Gemma 4B QLoRA (r=16) | ~80 MB | <1s |
| Gemma 12B QLoRA (r=16) | ~120 MB | ~1s |
| **All three** | **~260 MB** | **<3s** |

```bash
rsync -avz --progress \
    adapters/whisper/cycle1_20260315_a4b2c8d1/ \
    willem@192.168.1.100:/Users/willem/Code/vibes/SRTranslate/adapters/whisper/staging/
```

No cloud infrastructure needed. SSH keys must be pre-configured between training and
inference machines.

---

## 3. Deploy Script — `tools/deploy_adapters.py`

### 3.1 CLI Interface

```bash
# Deploy all adapters from latest cycle to all endpoints
python tools/deploy_adapters.py --cycle 1 --all-adapters --endpoints mac-dev church-rtx2070-1

# Deploy specific model to specific endpoint
python tools/deploy_adapters.py --cycle 1 --models whisper gemma_4b --endpoints mac-dev

# Rollback to previous version
python tools/deploy_adapters.py --rollback --endpoints mac-dev

# Dry-run (show what would happen)
python tools/deploy_adapters.py --cycle 1 --all-adapters --endpoints mac-dev --dry-run
```

### 3.2 Six-Phase Pipeline

```
Phase 1: VERSION
    │  SHA-256 hash adapter weights
    │  Copy to versioned directory: adapters/{model}/cycle{N}_{date}_{hash}/
    │  Update manifest.json with version metadata + training metrics
    ▼
Phase 2: CONVERT (NVIDIA endpoints only)
    │  Merge LoRA into base model (reuse convert_models_to_both.py logic)
    │  Requantize merged model to target format (GGUF Q4_K_M or bitsandbytes)
    │  Skip for Mac endpoints (MLX loads adapters directly)
    ▼
Phase 3: TRANSFER
    │  rsync adapter files to endpoint staging directory
    │  Verify transfer with SHA-256 check on remote
    ▼
Phase 4: HEALTH CHECK (pre-activation)
    │  SSH → run tools/health_check.py on staging adapter
    │  5 test sentences with expected substrings
    │  If any check fails → ABORT, leave current active adapter unchanged
    ▼
Phase 5: ACTIVATE
    │  Swap staging/ → active/ (atomic rename)
    │  Move old active/ → previous/ (rollback slot)
    │  Signal reload:
    │    Mac: SIGUSR1 → hot-reload (~2-3s, partials continue)
    │    NVIDIA: systemctl restart stark-translate (~10-20s outage)
    ▼
Phase 6: VERIFY (post-activation)
    │  Re-run health check against now-active adapter
    │  If fails → automatic rollback (swap active/ ↔ previous/)
    │  Update status.json and deploy_log.jsonl
    ▼
    DONE (or ROLLED BACK with alert)
```

### 3.3 Atomic Swap

The activate step uses `os.rename()` (atomic on POSIX) to swap directories:

```python
def _activate_adapter(model_dir: Path) -> None:
    staging = model_dir / "staging"
    active = model_dir / "active"
    previous = model_dir / "previous"

    # Move current active → previous (overwrite old previous)
    if previous.exists():
        shutil.rmtree(previous)
    if active.exists():
        active.rename(previous)

    # Move staging → active (atomic on same filesystem)
    staging.rename(active)
```

---

## 4. Health Check — `tools/health_check.py`

### 4.1 Test Sentences

Five hardcoded sentences covering critical translation scenarios:

| # | English Input | Expected Substring | Tests |
|---|--------------|-------------------|-------|
| 1 | "Good morning, welcome to the service." | "buen" or "servicio" | Simple greeting |
| 2 | "The atonement of Christ provides redemption for sinners." | "expiaci" or "redenci" | Theological terms |
| 3 | "The apostle James wrote about faith and works." | "Jacobo" or "Santiago" | Proper name disambiguation |
| 4 | "By grace you have been saved through faith, not of works." | "gracia" and "fe" | Complex theological concept |
| 5 | "Amen." | non-empty output | Short utterance handling |

### 4.2 Check Logic

```python
def run_health_check(adapter_dir: Path, engine_factory) -> HealthCheckResult:
    """Run 5-sentence smoke test. Returns pass/fail with details."""
    results = []
    for sentence in TEST_SENTENCES:
        t0 = time.perf_counter()
        output = engine.translate(sentence.input)
        elapsed = time.perf_counter() - t0

        passed = (
            len(output.text) > 0
            and any(sub in output.text.lower() for sub in sentence.expected_substrings)
            and elapsed < 5.0
        )
        results.append(SentenceResult(
            input=sentence.input,
            output=output.text,
            latency_s=elapsed,
            passed=passed,
        ))

    return HealthCheckResult(
        all_passed=all(r.passed for r in results),
        results=results,
        timestamp=datetime.utcnow(),
    )
```

### 4.3 Failure Handling

- **Pre-activation failure** (Phase 4): Abort deployment. Current active adapter untouched.
  Log failure to `deploy_log.jsonl`. Print warning with failing sentences.
- **Post-activation failure** (Phase 6): Automatic rollback — swap `active/ ↔ previous/`.
  Re-signal reload. Log rollback event.

---

## 5. Hot-Reload (Mac)

### 5.1 SIGUSR1 Handler in `dry_run_ab.py`

```python
import signal

def _setup_reload_handler(pipeline):
    """Register SIGUSR1 handler for hot adapter reload."""
    def _handle_sigusr1(signum, frame):
        logger.info("SIGUSR1 received — scheduling adapter reload")
        asyncio.get_event_loop().call_soon_threadsafe(
            asyncio.ensure_future,
            pipeline._reload_adapters(),
        )
    signal.signal(signal.SIGUSR1, _handle_sigusr1)
```

### 5.2 Reload Coroutine

```python
async def _reload_adapters(self):
    """Hot-reload TranslateGemma adapters without stopping the pipeline.

    During reload (~2-3s):
    - VAD continues (inline, <1ms)
    - STT continues (Whisper doesn't use adapters in inference)
    - MarianMT partials continue (separate model)
    - Only TranslateGemma finals pause briefly
    """
    logger.info("Reloading adapters...")
    adapter_dir = self._settings.adapter_dir

    # Submit reload to pipeline pool (serialized MLX thread — MLX is NOT thread-safe)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(self._pipeline_pool, self._do_reload, adapter_dir)

    logger.info("Adapter reload complete")
```

### 5.3 What Happens During Reload

| Component | During Reload | Impact |
|-----------|--------------|--------|
| Silero VAD | Continues (inline) | None |
| Whisper STT | Continues (no adapter) | None |
| MarianMT partials | Continues (separate model) | None |
| TranslateGemma finals | **Paused** (~2-3s) | Finals delayed by reload time |

Practical impact: 1-2 final translations may be delayed by ~2-3s. Partials still
display in real-time. For a church service, this is imperceptible.

### 5.4 NVIDIA Endpoints

NVIDIA endpoints require full model reload (merged weights, not LoRA adapters):

```bash
# On NVIDIA endpoint
systemctl restart stark-translate
```

Expected outage: ~10-20s (model load time). Schedule deployments between services.

---

## 6. Rollback

### 6.1 Two-Slot Swap

Each model has exactly two slots: `active/` and `previous/`. Rollback is a swap:

```python
def rollback(model_dir: Path) -> None:
    active = model_dir / "active"
    previous = model_dir / "previous"
    temp = model_dir / "_rollback_temp"

    active.rename(temp)
    previous.rename(active)
    temp.rename(previous)
```

### 6.2 Automatic Rollback

Triggered when Phase 6 (post-activation health check) fails:

```
[DEPLOY] Phase 6 health check FAILED for gemma_4b on mac-dev
[DEPLOY] Sentence 2 failed: expected "expiaci" in output, got "la expiación de Cristo..."
[DEPLOY] Rolling back gemma_4b: active ↔ previous
[DEPLOY] Sending SIGUSR1 to reload previous adapter
[DEPLOY] Rollback complete. Previous adapter re-activated.
```

### 6.3 Manual Rollback

```bash
python tools/deploy_adapters.py --rollback --endpoints mac-dev
python tools/deploy_adapters.py --rollback --models gemma_4b --endpoints mac-dev
```

### 6.4 Base Model Fallback

If both `active/` and `previous/` are corrupted (unlikely but possible):

- **Mac (MLX):** `mlx_lm.load(model_id)` without `adapter_path=` loads base weights
- **NVIDIA:** Load base HF model without PEFT — instant recovery to pre-fine-tuning quality

This is always available as a last resort. No adapter needed.

---

## 7. Monitoring

### 7.1 Deploy Log — `adapters/deploy_log.jsonl`

Append-only log of every deployment action:

```json
{
  "timestamp": "2026-03-15T15:00:00Z",
  "action": "deploy",
  "cycle": 1,
  "models": ["whisper", "gemma_4b"],
  "endpoint": "mac-dev",
  "phases_completed": ["version", "transfer", "health_check", "activate", "verify"],
  "health_check_results": {
    "pre_activation": {"all_passed": true, "latency_p50_ms": 450},
    "post_activation": {"all_passed": true, "latency_p50_ms": 460}
  },
  "status": "success",
  "duration_s": 12.3
}
```

### 7.2 Endpoint Status — `adapters/status.json`

Per-endpoint current state (written after each deploy/rollback):

```json
{
  "mac-dev": {
    "whisper": {
      "active_version": "cycle1_20260315_a4b2c8d1",
      "previous_version": null,
      "last_reload": "2026-03-15T15:00:00Z",
      "health_status": "healthy"
    },
    "gemma_4b": {
      "active_version": "cycle1_20260315_f1e2d3c4",
      "previous_version": null,
      "last_reload": "2026-03-15T15:00:05Z",
      "health_status": "healthy"
    }
  }
}
```

### 7.3 Session Metadata

`dry_run_ab.py` logs the current adapter versions in its diagnostics JSONL output:

```json
{
  "session_id": "20260315_153000",
  "adapters": {
    "whisper": "cycle1_20260315_a4b2c8d1",
    "gemma_4b": "cycle1_20260315_f1e2d3c4",
    "gemma_12b": null
  }
}
```

This connects inference quality metrics back to specific adapter versions.

---

## 8. Files to Modify/Create (When Implementing)

| File | Change | Lines (est.) |
|------|--------|-------------|
| `tools/deploy_adapters.py` | **New**: 6-phase deploy pipeline CLI | ~350 |
| `tools/health_check.py` | **New**: 5-sentence smoke test + result types | ~150 |
| `engines/mlx_engine.py` | Add `adapter_path` param to `MLXGemmaEngine.__init__()`, add `reload()` method | ~30 |
| `engines/factory.py` | Forward `adapter_path` to engine constructors | ~10 |
| `settings.py` | Add `adapter_dir`, per-model adapter path fields | ~15 |
| `dry_run_ab.py` | Add `--adapter-dir` flag, SIGUSR1 handler, `_reload_adapters()` | ~40 |
| `.gitignore` | Add `adapters/` | ~1 |
| `tests/test_deploy.py` | **New**: version ID format, manifest schema, rollback swap, health check | ~120 |

**Total estimated new/changed code: ~716 lines**

---

## 9. Implementation Phases

| Phase | Scope | Effort | Deliverable |
|-------|-------|--------|-------------|
| **MVP** | Versioning + rsync deploy + health checks + rollback | 1-2 days | `deploy_adapters.py` + `health_check.py` working for Mac |
| **Hot-reload** | SIGUSR1 handler, zero-downtime adapter swap | 1 day | Mac hot-reload without pipeline restart |
| **NVIDIA support** | LoRA merge + format convert + restart-based reload | 1-2 days | Full deploy to NVIDIA church endpoints |
| **Monitoring** | Deploy log, status files, session metadata integration | 0.5 day | `deploy_log.jsonl` + `status.json` |

**Total: 3.5-5.5 days**

---

## 10. Prerequisites

Before implementing this system:

1. **SSH key exchange** between training machine and all inference endpoints
2. **At least one fine-tuned adapter** from a completed training cycle
3. **`rsync` installed** on all machines (pre-installed on macOS, `apt install rsync` on Ubuntu)
4. **Static LAN IPs** or hostname resolution for inference endpoints
5. **`systemd` service** for NVIDIA endpoints (for `restart`-based reload)
