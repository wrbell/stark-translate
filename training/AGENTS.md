# training/AGENTS.md — Fine-Tuning & Data (Agent Guide)

> Paired with [`CLAUDE.md`](./CLAUDE.md) (imports this file for Claude Code) and with the
> reference guide [`docs/agents/training.md`](../docs/agents/training.md) (programs and their state, data pipeline with
> verified flags, Whisper LoRA / W17, Gemma 4 tuning, hymn corpus, export and transfer). All
> training runs on **WSL/A2000**; exported artifacts move to the inference machines. Repo-wide
> constraints: [`../AGENTS.md`](../AGENTS.md).

## Agent constraints

- **Training cutoff:** 2026-03-14 — never train on post-cutoff eval sermons or the four
  fresh-eval video ids listed in `docs/agents/training.md` § Data organization.
- **Copyright:** public-domain Bible pairs only — no ESV/NASB/NIV/NLT/NVI/LBLA/RVR1960/DHH.
- **Corpus version:** `bible_data/aligned/verse_pairs_train_v2.jsonl`; v1 has the Platense
  misalignment. `run_gemma4_e4b_domain_sft.sh` now defaults to v2; explicitly selected
  missing corpora fail. Both production recipes support CPU-only `--dry-run` / `--preflight`
  through `tools/training_preflight.py`, before trainer/GPU imports. Missing WSL artifacts
  remain a failed preflight, never a completed CUDA gate.
- **Holdout safety:** Review/export must not overwrite local holdout; correction imports keep
  evaluation and training separate (`tests/test_correction_import_safety.py`).
- **Gemma 4:** train E2B and E4B separately; freeze PLE and vision/audio towers;
  `enable_thinking=False` on every example; QLoRA via Unsloth. `train_gemma4.py`,
  `train_gemma4_cpo.py`, `export_gguf.py` ran for the v1/v1.1/v2-cpo experiments but keep
  pre-run `UNTESTED` headers — cite them as "review before each run".
- **W17:** use Whisper's `out_proj`; local model config and source adapter rank, scale,
  full tensor coverage and shapes are checked before GPU imports. The recipe explicitly
  enables `--allow-target-expansion` for fresh added modules/DoRA magnitudes while retaining
  every W16 LoRA tensor. `--require-replay` forbids falling back to hard-only training.
  Hard-subset JSON is aligned into a separate audiofolder before training.
  Preflight, mining and alignment resolve literal source stems through the same WAV
  index; nested/mixed-case WAV extensions work, and ambiguous duplicate paths fail.
- **Jacobo candidates:** [`candidates/README.md`](candidates/README.md) records original
  synthetic, **unapproved** preference triples, provenance and overlap checks. The CPO
  loader rejects these before GPU imports. The missing v2 holdout and bilingual review
  remain gates; candidate preparation does not close #136 or #137.
- **Ship rule:** stock Gemma 4 E4B stays the default until a Mac A/B note (#135) says
  otherwise; v2-cpo is at parity and fails the Jacobo canary (#136).
- **Quoting flags:** every CLI flag in `docs/agents/training.md` was read from the script's `argparse`
  block at `c00e697`; re-check with `--help` after pulling. Paths marked (WSL) do not exist in
  a Mac checkout; a missing WSL artifact is a failed preflight, never a completed gate.
- Do not paste WER/COMET/latency numbers into guides; link the evidence document that
  defines the measurement.

## Pipeline order (WSL runbook)

1. Phase 4 preprocess → 2. Gemma 4 E4B domain SFT → GGUF → 3. W17 Whisper DoRA + hard-mix →
CT2 → 4. optional Parakeet EN bench on CUDA → 5. Mac transfer + Phase 7 evaluate →
6. active learning cycle → 7. CUDA latency proposal.
Runbook: [`docs/wsl_pipeline_refresh.md`](../docs/wsl_pipeline_refresh.md).

## Artifacts

States and evidence links for W16/W17, the Gemma 4 runs, the E4B domain SFT recipe, the
TranslateGemma sweep and the Hindi/Chinese decision are in [`docs/agents/training.md`](../docs/agents/training.md#programs-and-their-state).

## Backlog

`wsl-phase4`, `wsl-e4b-domain-sft`, `wsl-w17-export`, `wsl-training-recipe-checks`,
`cuda-latency-proposal`, `issue-135-mac-ab`, `issue-136-jacobo-cpo`, `issue-138-hindi-zero-shot`,
`hymn-translation-boundary`, `hymn-capture-suppression` in
[`docs/backlog.json`](../docs/backlog.json).

<!-- standards:begin -->
## Collection standards

Every project under `/Users/willem/Code` follows the shared standards in
`/Users/willem/Code/standards/` (index: `standards/STANDARDS.md`; future
standards: `standards/ROADMAP.md`).

- **Presentations:** build every deck from
  `standards/powerpoint template/Willem-Default.potx` (theme "Helena": Neue Haas
  Grotesk Text Pro, 16:9, teal/orange/red accent palette). Spec:
  `standards/powerpoint template/STANDARD.md`. Generate with
  `standards/powerpoint template/house_style.py` (open
  `Willem-Default-Base.pptx`, never the `.potx`) and gate with
  `standards/powerpoint template/deck_checks.py` before calling a deck done.
- **Deck rules:** no speaker notes in submitted decks; editable shapes, not
  chart images; numbered, linked superscript citations with a final References
  slide; no bottom rules, citation strips, or page counters; footer text only
  when a course or client requires it (for example `ME460 HWx`), which overrides
  the default of no footer; export the deliverable PDF with native PowerPoint
  and use LibreOffice renders only for QA.
- **Everything else:** do not invent facts, dates, or numbers; mark unknowns TBD
  and point at the source. Keep copyrighted course material out of git. This
  block is managed by `standards/tools/apply_standards.py`; edit
  `standards/ai-files/BLOCK-root.md`, not this copy.
- **AI use (school work):** no AI-generated or AI-modified images in any school
  deliverable; AI-written deliverable text only with written adviser
  pre-clearance (`docs/ai-clearances/`); never cite an AI tool as a source;
  never edit graded report text (the repo's `protected-paths.txt`; example:
  `standards/enforcement/senior-design-repo/sd-protected-paths.txt`). Log AI use
  in `docs/ai-use-log.md` and disclose it per
  `standards/standards/ai-use-disclosure/ai-use-disclosure.md`.
- **AI files:** one `AGENTS.md` (≤ 200 lines, Clarity verbatim); `CLAUDE.md` is
  `@AGENTS.md`. Gates: `standards/tools/agents_md_lint.py`, `ai_file_lint.py`.
<!-- standards:end -->
