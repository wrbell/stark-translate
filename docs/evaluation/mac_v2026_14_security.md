# v2026.14 Mac security-check scope and B615 triage

The configured CI checks passed. The expanded model-download scan still reports
26 medium B615 findings; they are not 26 equivalent live Mac downloads, and they
have not all been resolved. This review changed no runtime code or model artifacts.

## What was checked

The [lint workflow](../../.github/workflows/lint.yml) runs:

```bash
bandit -r engines/ features/ tools/ settings.py -s B101,B603,B607,B615 --severity-level medium
```

That invocation reported zero medium/high findings. It explicitly skips B615
(unpinned Hugging Face downloads), as well as assertions and two subprocess
checks. It does not scan `dry_run_ab.py`, `workers.py`, `operator_app/`, `scripts/`
or `training/`. Passing it is not a whole-repository security assessment.

The expanded run retained B615 while keeping the same source roots and other
exclusions. Its preserved JSON, `.cache/mac-roadmap/bandit-expanded.json`, was
generated at `2026-09-10T00:59:24Z`: 26 reported medium findings, all B615 with
high confidence, zero high findings and no scan errors. The metrics also count
27 low-severity observations below the reporting threshold; the triage below
covers the 26 reported findings only.

The [dependency-audit workflow](../../.github/workflows/security.yml) was also
reproduced for Mac, Windows and NVIDIA requirement files. All three filtered
audits reported zero known vulnerabilities; local evidence is in
`.cache/security-audit/{mac,windows,nvidia}.json`. The command uses `--no-deps`
after excluding MLX, Piper, selected audio/training packages, pandas, torch-family
entries and index options. It does not audit the complete installed Mac extras
environment, all transitive packages, model contents or upstream model code.
The workflow's change filter watches `requirements*.txt`, so a `pyproject.toml`
dependency-only change does not itself trigger that audit on a pull request.

## All 26 B615 call sites

Line numbers identify the reviewed source; later edits may move them. Counts
are call sites, not distinct models or evidence of malicious artifacts.

| Group | Count | Source and reported lines | Reachability and remaining limitation |
|---|---:|---|---|
| Local Marian CT2 tokenizer | 1 | `engines/cuda_engine.py`: 1057 | The factory selects a local directory containing `model.bin`; CT2 opens that directory before the tokenizer loads beside it. The normal Mac CT2 path is a local artifact load, not an unpinned Hub fetch at this call. The export's original HF revision remains unknown when its manifest lacks one. |
| Marian HF fallback | 2 | `engines/marian_hf_engine.py`: 74, 75 | Both arguments pass through the shared resolver. A prepared Mac installation uses the pinned local snapshot. If no matching cache exists, the resolver returns the repo ID and these calls can download the moving default revision. Custom repository IDs have the same limitation. |
| Piper missing-voice fallback | 2 | `engines/mlx_engine.py`: 1087, 1088 | TTS is optional. Setup-installed EN/ES voices resolve locally first, but a missing/custom stock voice reaches two `hf_hub_download` calls without `revision`. Legacy Piper cache files are also accepted without establishing their revision. |
| Optional CUDA/HF engines | 8 | `engines/cuda_engine.py`: 430, 431, 654, 655, 682; `engines/hf_whisper_engine.py`: 102, 103, 114 | Tokenizer/target/draft calls use caller IDs directly. Repository IDs can fetch unpinned revisions. These are alternate CUDA/HF translation or HF Whisper/speculative paths, not the current Parakeet/MLX Whisper + MLX Gemma default. Explicit local paths avoid a Hub download but do not prove artifact provenance. |
| Legacy benchmark helpers | 4 | `tools/benchmark_latency.py`: 309, 310, 338; `tools/test_adaptive_model.py`: 404 | Manually invoked Marian benchmark/adaptive helpers load unpinned HF models or tokenizers, even when the CT2 weights themselves are local. These calls are outside the frozen `tools/mac_evaluation.py` screening route. |
| Offline quality estimation | 2 | `tools/translation_qe.py`: 93, 94 | Optional tier-2 back-translation loads `Helsinki-NLP/opus-mt-es-en` directly without revision. The live pipeline's lightweight quality checks do not invoke this offline helper. |
| Conversion and corpus tooling | 7 | `tools/convert_models_to_both.py`: 165, 354, 473, 490, 522, 577; `tools/build_v1_corpus.py`: 116 | Six snapshot/model/tokenizer calls belong to manual export or LoRA-merge preparation; the seventh streams the OPUS training corpus. These are real unpinned fetches when those tools run. Deferring training or conversion does not fix their reproducibility. |
| **Total** | **26** | | |

## What the Mac manifest does and does not protect

[models.lock.json](../../models.lock.json) pins full Hub commit IDs for the Mac
Whisper, Parakeet, E4B and Marian defaults, plus optional E2B, TranslateGemma and
Piper EN/ES entries. [Setup](../../operator_app/setup.py) passes that revision to
`snapshot_download`. The [shared resolver](../../engines/model_paths.py) checks
managed install markers against the expected repo/revision and searches the
pinned Hub snapshot directory. Existing user-supplied paths remain valid overrides.

The Mac Gemma loader (`engines/mlx_engine.py`: 709, 728), MLX Whisper loaders and
Parakeet (`engines/parakeet_mlx_engine.py`: 159) all use that resolver. These wrapper
APIs do not appear in the 26 B615 findings. Prepared, matching local caches avoid
their network fallback; their absence from Bandit is not independent proof of
pinning. In normal inference mode the resolver returns the original ID on a
cache miss, and the downstream library can fetch its default revision. Preflight
uses offline lookup, but direct CLI/engine callers are not universally forced
through that gate. The offline benchmark environment constrains the recorded
runs, not every possible future invocation.

Completeness checks inspect required files and weight shards; they are not a
cryptographic revalidation of every cached file. Explicit paths, legacy caches
and local CT2 exports therefore need separate provenance. New session completion
metadata records the actual CT2 `model.bin` SHA-256 and export-manifest identity
after inference. That binds the measured weights for comparison; it neither
retroactively establishes an HF revision nor certifies the model's safety.

Two reachable download paths outside this B615 list also remain relevant:

- `dry_run_ab.py:1032` uses `torch.hub.load("snakers4/silero-vad", ...)` without
  a repo commit/tag. This is the default VAD path and also supplies its ONNX
  variant. An installed `silero-vad` package version does not pin that Hub code
  or the cached model artifact. The pipeline file is outside the CI scan roots.
- `features/live_diarize.py:221` uses SpeechBrain `from_hparams` with the
  `speechbrain/spkrec-ecapa-voxceleb` source and no revision. It is reachable only
  for optional diarization; B615 does not report that wrapper call here.

## Remaining work

Prefer a shared revision-aware download or an explicit missing-model error for
managed runtime models, with deliberate overrides kept visible. Pin and record
the actual Silero and optional SpeechBrain sources; then update the alternate
backend, conversion, corpus and offline-QE tools. Broaden audit roots and cover
the resolved extras environment before describing the project as comprehensively
audited. These are follow-up items; none were changed during the frozen latency
screening run. PyPI account setup and publication remain pending by user choice.
