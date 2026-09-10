# v2026.14 Mac security-check scope and B615 triage

The configured CI checks passed. The final expanded model-download scan reports
27 medium B615 findings: the original 26 call sites described below, plus one
new download whose revision is explicitly guarded by a full-commit check.
The additional report is a static-analysis false positive for pinning; the
original 26 findings and their remaining limitations have not all been resolved.
This review changed no runtime code, suppressions or model artifacts.

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
exclusions. Its final JSON, `.cache/mac-roadmap/bandit-expanded-final.json`, was
generated at `2026-09-10T02:35:10Z`: 27 reported medium findings, all B615 with
high confidence, zero high findings and no scan errors. The metrics also count
28 low-severity observations below the reporting threshold. The original
`bandit-expanded.json` remains preserved (26 reports at `2026-09-10T00:59:24Z`).
The final configured CI result is `.cache/mac-roadmap/bandit-ci-final.json`.

The [dependency-audit workflow](../../.github/workflows/security.yml) was also
reproduced for Mac, Windows and NVIDIA requirement files. All three filtered
audits reported zero known vulnerabilities; local evidence is in
`.cache/security-audit/{mac,windows,nvidia}.json`. The command uses `--no-deps`
after excluding MLX, Piper, selected audio/training packages, pandas, torch-family
entries and index options. It does not audit the complete installed Mac extras
environment, all transitive packages, model contents or upstream model code.
The workflow's change filter watches `requirements*.txt`, so a `pyproject.toml`
dependency-only change does not itself trigger that audit on a pull request.

## Original 26 B615 call sites

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

The 27th report is `tools/marian_ct2_setup.py:69`, which calls
`snapshot_download(repo_id=repo_id, revision=selected_revision)`. Before any
remote call, lines 43–46 require `selected_revision` to be a string matching
`[0-9a-f]{40}` or raise `ValueError`. The revision comes from the pinned manifest
or an explicit full-commit override; it cannot be an omitted revision or moving
branch at this call. Local paths return earlier and do not reach the download.
Bandit does not establish that guard when evaluating the variable argument, so
this report does not add an unpinned-download path. No `nosec` suppression was
added. A full commit pins the requested source; it does not certify model safety.

## What the Mac manifest does and does not protect

[models.lock.json](../../models.lock.json) pins full Hub commit IDs for the Mac
Whisper, Parakeet, E4B and Marian defaults, plus optional E2B, TranslateGemma and
Piper EN/ES entries. [Setup](../../operator_app/setup.py) passes that revision to
`snapshot_download`. The [shared resolver](../../engines/model_paths.py) checks
managed install markers against the expected repo/revision and searches the
pinned Hub snapshot directory. Existing user-supplied paths remain valid overrides.

The Mac Gemma loader (`engines/mlx_engine.py`: 709, 728), MLX Whisper loaders and
Parakeet (`engines/parakeet_mlx_engine.py`: 159) all use that resolver. These wrapper
APIs do not appear in the 27 B615 findings. Prepared, matching local caches avoid
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

Two paths outside this B615 list also matter when interpreting the recorded scan:

- At scan time `dry_run_ab.py:1032` used unpinned Torch Hub for Silero. After the
  48 frozen screens, the loader changed to the installed `silero-vad==6.2.1`
  package's bundled JIT/ONNX weights. Preflight checks both files; startup metadata
  records their actual hashes. Both variants passed real CPU inference with
  empty Torch/HF caches and network/Hub calls blocked. The package's model code,
  utility code and both weights match the prior local Hub cache byte for byte
  (`.cache/mac-roadmap/vad-cache-proof.json`). This preserves local artifact
  identity without inventing an upstream Hub commit. Historical screen evidence
  remains bound to its original source; the pipeline file is still outside the
  CI scan roots.
- `features/live_diarize.py:221` uses SpeechBrain `from_hparams` with the
  `speechbrain/spkrec-ecapa-voxceleb` source and no revision. It is reachable only
  for optional diarization; B615 does not report that wrapper call here.

## Remaining work

Mac setup now also derives both Marian CT2 directions from pinned HF sources
when no complete working adapter exists. It records source revision, converter
interpreter/version and file hashes before atomically publishing the managed
artifact. Existing adapters remain unchanged and keep their original provenance.

Prefer a shared revision-aware download or an explicit missing-model error for
remaining runtime cache misses, with deliberate overrides kept visible. Pin the
optional SpeechBrain source; then update the alternate backend, legacy conversion,
corpus and offline-QE tools. Broaden audit roots and cover
the resolved extras environment before describing the project as comprehensively
audited. These are follow-up items. The Silero/managed-CT2 setup fixes were applied
after the frozen screening run. The separate [24-run synthetic routing exercise](mac_v2026_14_routing/README.md)
completed successfully and records the packaged VAD artifact in every run;
that runtime evidence does not broaden the security scan's scope.
PyPI account setup and publication remain pending by user choice.
