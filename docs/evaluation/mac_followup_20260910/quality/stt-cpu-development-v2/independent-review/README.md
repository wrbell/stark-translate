# CPU small/base retained-data review

The public note is written only after `cpu-quality-recovery-v2-state.json`
records `stage=completed`, with a complete immutable index and adjacent report.
`audit_completed.py` exits without selection while that completion is absent.
It reads small JSON/source files only and never starts models, subprocesses,
audio devices, decoders, native package imports, or weight/archive hashing.

After completion, run the reader with `stt_env/bin/python -B` (Python 3.11).
It reuses the frozen pure `summarize_run` and `prepare.validate_quality` APIs,
checks the complete 12-worker/600-row inventory, and independently recomputes
edit totals and call/engine/RTF distributions. Results are written exclusively
to `completed-review.json`. Model identities use retained inventory bindings;
the model files themselves are never reopened.

The public report structure is: outcome and language-specific base eligibility;
12 separate per-run WER/call-delay/RSS rows; changed public examples and glossary
opportunities; exact selected models and frozen source/environment provenance;
failed v1 decoding-import stage preserved separately; production configuration
and acceptance limits. Isolated beam-5/four-thread quality cannot establish
beam-1/three-thread live caption gains or a production default change. Fifty
calls per run do not meet a 100-observation p95 claim; repeats are never pooled.
