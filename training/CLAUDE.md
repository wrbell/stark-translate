# training/ — Fine-Tuning & Data Preparation (Windows/WSL)

> Paired with [`AGENTS.md`](./AGENTS.md). Everything here runs on the **WSL2 training
> desktop** (A2000 Ada 16 GB, 64 GB RAM); exported artifacts move to the inference machines.
> Environment: [`CLAUDE-windows.md`](../CLAUDE-windows.md) · ordered execution:
> [`docs/wsl_pipeline_refresh.md`](../docs/wsl_pipeline_refresh.md).
>
> **Status (2026-09-10):** no WSL job has run since the Gemma 4 v2-cpo iteration
> (2026-04-30). Phase 4 full preprocess, the E4B domain SFT recipe, W17 and the CUDA latency
> proposal are scripted and **pending hardware time** (`docs/backlog.json`: `wsl-phase4`,
> `wsl-e4b-domain-sft`, `wsl-w17-export`, `cuda-latency-proposal`). Flags and defaults below
> were read from the scripts' `argparse` definitions at commit `c5fb689`; sections marked
> **historical** describe completed runs whose numbers live only in the linked evidence.
> Native Windows / RTX 2070 **inference** is a separate Lite runtime, not a training concern
> ([`docs/lite_profiles.md`](../docs/lite_profiles.md)).

## Programs and their state

| Program | Scripts | State | Evidence |
|---------|---------|-------|----------|
| **Whisper LoRA (STT)** | `train_whisper.py` → `export_ct2.py` | **W16** deployed as the CUDA STT (`adapters/whisper_turbo_ct2/active`, auto-preferred by `FasterWhisperEngine` and by the standard CPU/CUDA STT default). **W17** (DoRA + hard-mix) scripted in `run_w17_curriculum.sh`, untrained. Mac EN default is Parakeet MLX, which loads no LoRA; Mac ES is mlx-whisper, also no LoRA — W16 comparisons on the Mac use the CPU faster-whisper path (#135). | [`docs/archive/v2026.7/STT_BENCHMARK.md`](../docs/archive/v2026.7/STT_BENCHMARK.md) |
| **Gemma 4 E2B/E4B QLoRA + CPO (translation)** | `train_gemma4.py`, `train_gemma4_cpo.py`, `export_gguf.py`, `tools/build_preference_triples.py`, `qe_filter.py`, `glossary_annotate.py` | Spike, v1, v1.1 and v2-cpo **ran** (2026-04-29/30); v2-cpo reached statistical parity with stock E4B and still fails the Jacobo canary (#136). Stock E4B remains the default on Mac and CUDA. The production recipe `run_gemma4_e4b_domain_sft.sh` has **not** been run. The three scripts still carry `UNTESTED` file headers written before the first run — read them as "review before each run", not as "never executed". | [`docs/gemma4_tuning/v1_results.md`](../docs/gemma4_tuning/v1_results.md), [`v3_directions.md`](../docs/gemma4_tuning/v3_directions.md) |
| **TranslateGemma QLoRA** | `train_gemma.py`, `run_ablation.sh`, `run_b_series.sh`, `run_hybrid_*.sh`, `run_scale*.sh` | **Historical** S1–S9 sweep; superseded (TranslateGemma is already a translator; its Platense corpus half was misaligned). | [`docs/archive/training/gemma_tuning_test_matrix.md`](../docs/archive/training/gemma_tuning_test_matrix.md) |
| **Marian full fine-tune** | `train_marian.py` | Fallback with a lower ceiling; unused in production. Marian is deployed **stock** as the CT2 int8 partial translator. | — |
| **Piper TTS** | `prepare_piper_dataset.py`, `train_piper.py`, `export_piper_onnx.py`, `evaluate_piper.py` | Scripted; production uses stock Piper voices. | — |

**Corpus rule:** `bible_data/aligned/verse_pairs_train.jsonl` (v1) joined Platense by row order
and is misaligned from Psalms onward. Train on `bible_data/aligned/verse_pairs_train_v2.jsonl`
(rebuilt by `tools/rebuild_verse_pairs.py`); postmortem
[`docs/platense_alignment_bug.md`](../docs/platense_alignment_bug.md).
`run_gemma4_e4b_domain_sft.sh` now defaults `STARK_GEMMA4_VERSE` to the **v2 path**.
An explicit `STARK_GEMMA4_TRAIN` selects a premixed corpus; otherwise both verse and sermon
component paths must exist. Missing selected inputs fail instead of silently falling back.

Both production recipes accept `--dry-run` (alias `--preflight`). They read the actual
trainer parser declarations, local config/adapter metadata and corpus files with the Python
standard library; they do not import a trainer or GPU library. Set `STARK_TRAINING_PYTHON`
to a CPU interpreter if desired. The default required holdout is
`bible_data/aligned/verse_pairs_test_v2.jsonl`; `STARK_TRAINING_HOLDOUT` selects a required
holdout at another mount, and existing local holdouts/canaries are checked as well.
Passing preflight does not establish CUDA execution, tokenization, model quality or approval.

## Data pipeline

### Phase 1–2 — Download and 10-step preprocess (`preprocess_audio.py`)

Raw YouTube church audio → clean chunks. `preprocess_audio.py --input DIR --output DIR
[--download --urls FILE] [--skip-demucs] [--diarize|--skip-diarize] [--resume]`. Steps, in
order: download (`yt-dlp`) → 16 kHz mono WAV → SNR/clipping gate → `inaSpeechSegmenter`
speech/music/noise → `demucs` (`htdemucs`, `--two-stems vocals`) → bandpass + `noisereduce`
→ `pyloudnorm` (−16 LUFS, −1 dBTP) → Silero VAD chunking → optional `pyannote` diarization →
final gate (SNR, duration, silence ratio). Don't over-clean: Whisper was trained on noisy
audio, and training noise should match service conditions.

**Phase 4 corpus run:** `training/run_phase4_preprocess.sh` (env `STARK_RAW_DIR`,
`STARK_CLEANED_DIR`) wraps `run_phase4_corpus.py --input --output --resume [--skip-demucs]
[--diarize] [--dry-run]` and writes `stark_data/cleaned/phase4_status.json`. Gate:
`ready_for_training: true`, `errors == 0`, `completed > 0`; `--dry-run` does not satisfy it.

### Phase 3 — Quality assessment (`assess_quality.py`)

Subcommands `sample`, `review`, `cross-check`, `evaluate` (`--input`, `--n`, `--seed`,
`--output`, `--model`, `--spot-check`). Establish a baseline on 50–100 stratified segments
before training. The original strategy table (WER band → filtering strategy) predates the
Deepgram oracle and is kept in the archive; today the label source is Deepgram, so the
decision is about filtering, not re-transcription.

### Phase 4 — Labels

- **Deepgram Nova-3 oracle (current):** `transcribe_with_deepgram.py --input DIR --output DIR
  [--boost-terms FILE] [--lang] [--max-concurrent N] [--resume] [--api-key KEY]`; key from
  `STARK_DEEPGRAM__API_KEY` (nested `STARK_` settings). Boosted terms = the 50 Tier 1 keyterms
  in `bible_data/glossary/tier1_boost.json`. Output `.deepgram.json` per sermon with word
  timestamps and confidence; large files use a 300 s request timeout.
- **Whisper pseudo-labels (historical fallback):** `transcribe_church.py --input --output
  [--model] [--backend transformers|faster-whisper] [--batch-size] [--resume] [--oracle
  whisper|deepgram]`. Only when Deepgram output is unavailable.
- **Dataset build:** `prepare_whisper_dataset.py --gt-source deepgram` (default `whisper`)
  with `--chunks-dir`, `--transcripts-dir`, `--output`, `--eval-ratio`, `--seed`,
  `--no-filter`, `--no-balance`, `--copy`.
- **Alignment at scale:** `align_deepgram_chunks.py --whisper-chunks --deepgram-dir
  --audio-dir --output [--whisper-model] [--min-chars] [--eval-sources ...]
  [--preprocess-cache]`. `--preprocess-cache` streams 1,000-row Arrow shards
  (`_shards_train/`, `_shards_test/`), resumes past completed shards, and sets a 12 GB
  `RLIMIT_AS` cap; `recover_shards.py` rebuilds a `DatasetDict` from completed shards one at
  a time.

### Phase 4b — Bible parallel corpus (`prepare_bible_corpus.py`)

`--db-dir`, `--source`, `--output`, `--multi-ref`. Public-domain pairs only: KJV/ASV/WEB/BBE/YLT
↔ RVR1909, Platense, Español Sencillo (CC BY-SA). **Never** ESV, NASB, NIV, NLT, NVI, LBLA,
RVR1960, DHH. Sources: `bible-nlp/biblenlp-corpus`, `Helsinki-NLP/bible_para`,
`scrollmapper/bible_databases`. Holdout: `tools/build_eval_sets.py` (`--verse-count`,
`--train-path`, `--test-path`, `--seed`, `--dry-run`) — the current v2 holdout is
`bible_data/aligned/verse_pairs_test_v2.jsonl` (500 verses).

### Glossary (`build_glossary.py`, `tools/glossary.py`)

Two tiers: Tier 1 boost (50 terms, Deepgram `keyterm`), Tier 2 master (229 terms,
normalization, QE, active learning). Build with `python training/build_glossary.py
--build-tiers` (`--boost-size`, `--master-size`, `--from-hymns`, `--merge-hymn-allowlist`,
`--augment`). Files: `bible_data/glossary/tier1_boost.json`, `tier2_master.json`.
`tools/glossary.py` exposes `load_tier()`, `validate_boost()`, `build_and_save_tiers()`.

### Data organization

- Fixed cutoff **2026-03-14**: train on earlier sermons, evaluate on later ones.
  `tools/sort_sermons.py --output-dir stt-data --catalog stark_data/playlist_catalog.json`
  lays out `stt-data/{gospel,ministry,conference,throwback}/{year}/` plus `manifest.json`.
- Never train on the fresh-eval sermons (`4Es8SrciqV0`, `vRT5RswIHu8`, `FOVTvZednUQ`,
  `yOzWGOTvTaA`).
- `tools/lock_data.py` records SHA-256 lockfiles for training inputs.

## Whisper LoRA (`train_whisper.py`)

Defaults from the parser: `--model openai/whisper-large-v3-turbo`, `--target-modules q_proj
v_proj`, `--lora-r 32`, `--lora-alpha 64`, `--batch-size 4`, `--grad-accum 4` (effective 16),
`--epochs 3`, `--lr 1e-4`, `--replay-ratio 0.3` (general-English replay, 0 disables),
`--accent-balance` on, bf16 + gradient checkpointing. Extras: `--init-from ADAPTER` (load
adapter weights, fresh optimizer — curriculum), `--use-dora`, `--eval-chunked`, `--resume`.
Extended module set per the parser help: `q_proj v_proj k_proj out_proj fc1 fc2`.

**W17 recipe** (`run_w17_curriculum.sh`, env `STARK_WHISPER_DATASET`, `STARK_W16_ADAPTER`,
`STARK_HARD_MINED`, `STARK_HARD_SUBSET`, `STARK_W17_DATASET`, `STARK_W17_OUT`, `STARK_W17_CT2`):
CPU preflight → train-only mining → WER-bounded subset JSON (0.15–0.80, `--include-tier1`) →
`align_deepgram_chunks.py` into a separate audiofolder → train with `--init-from` W16,
`--use-dora`, `--allow-target-expansion`, expanded modules, `--replay-ratio 0.3`,
`--require-replay`, 1 epoch → `export_ct2.py` sanity gate → `manage_adapters.py
register --model whisper_turbo_ct2`. Never train hard-only (W15 lesson,
[`docs/archive/v2026.5/w15_postmortem.md`](../docs/archive/v2026.5/w15_postmortem.md)).
The recipe uses `out_proj`. Set `STARK_WHISPER_MODEL_CONFIG` to the local base
`config.json` when it cannot be resolved from the Hugging Face cache. Preflight checks
Whisper geometry against every source adapter tensor header, including rank and alpha.
Strict init-from remains the trainer default; explicit expansion permits only newly
selected target modules and newly introduced DoRA magnitudes to initialize fresh. Every
source tensor must load. Existing recipe output paths are refused to preserve earlier work.
Missing replay now fails W17 instead of silently producing hard-only training.
Preflight, mining and alignment share a literal-stem WAV index: nested `.wav`, `.WAV`
and mixed-case extensions work, filename glob characters are ordinary characters,
and duplicate matching paths fail instead of choosing a flat file or wildcard sibling.
Direct `train_whisper.py` invocations also require the local config before GPU imports;
cache resolution honors `HF_HUB_CACHE`, `HF_HOME` and `XDG_CACHE_HOME`. When introducing
DoRA on a source LoRA adapter, the trainer recomputes new magnitudes **after** loading W16
A/B matrices, using PEFT's `DoraLinearLayer.update_layer` so their initial scale follows
the merged direction. Unsupported PEFT APIs fail before training; CUDA numerical behavior
still requires the WSL run. This follows the [upstream PEFT implementation](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/dora.py).

**Hard-example mining (W15 lineage):**

1. `mine_hard_examples.py --adapter --chunks-json --deepgram-dir --audio-dir --output
   [--model] [--batch-size] [--tier1-glossary bible_data/glossary/tier1_boost.json]
   [--resume]` — per-chunk WER vs Deepgram, Tier 1 term detection, JSONL.
2. `build_hard_subset.py --mined --chunks-json --output [--wer-min 0.15] [--wer-max 0.80]
   [--target-size 10000] [--max-per-source] [--include-tier1]`.
3. `filter_chunks_by_confidence.py --input --output [--metric logprob|confidence|combined]
   [--target-size] [--min-duration] [--max-duration]`.
4. `train_whisper.py --init-from <adapter>` → re-mine → repeat (2–4 cycles typical).

**Evaluation:** `eval_whisper_wer.py --adapter --eval-set [--model] [--max-samples]
[--output] [--sweep]`; engine-level bench `tools/benchmark_stt_engines.py --variant ...
--manifest tools/stt_bench_manifest.json` (41 clips). Gate for W17: match or beat W16 on
overall and Tier 1 WER with no p95 regression — reference numbers only in the v2026.7
benchmark document.

**Ablation history (W0–W15):** matrix in
[`docs/archive/training/whisper_tuning_test_matrix.md`](../docs/archive/training/whisper_tuning_test_matrix.md);
W12 trained the W7 config on the full Deepgram-aligned set (198K chunks / 328 sermons); the
Arrow cache lives under `/mnt/d/Data/stt-data/whisper_dataset_sttdata/.preprocessed_cache/`.

## Gemma 4 tuning (`train_gemma4.py`, `train_gemma4_cpo.py`, `export_gguf.py`)

Plan and results: [`docs/gemma4_tuning/`](../docs/gemma4_tuning/overview.md). Rules baked
into the trainers: E2B is a MatFormer slice of E4B, so **train each size separately**;
freeze Per-Layer Embeddings and the vision/audio towers; apply `enable_thinking=False` to every
example; QLoRA through Unsloth to fit 16 GB.

| Script | Parser defaults (c5fb689) |
|--------|---------------------------|
| `train_gemma4.py` | `--base unsloth/gemma-4-E4B-it` (`unsloth/gemma-4-E2B-it` for E2B); data via `--train-data` or `--verse-pairs` / `--sermon-pairs` / `--glossary-pairs` (`--max-pairs`); `--lora-r 8`, `--lora-alpha 8`, `--lr 2e-4`, `--epochs 2` (`--max-steps` overrides), `--per-device-batch-size 2`, `--grad-accum 8`, `--max-seq-length 1024`, `--packing` on, `--warmup-steps 5`, `--save-steps`, `--seed` |
| `train_gemma4_cpo.py` | `--triples` (required, `{prompt, chosen, rejected}` JSONL), `--init-adapter` (continue an SFT LoRA), `--beta 0.1`, `--epochs 1`, `--lora-r 8` (ignored with `--init-adapter`), `--max-prompt-length`, `--max-seq-length` |
| `export_gguf.py` | `--adapter`, `--base`, `--output`, `--qtype Q4_K_M`, `--outtype`, `--llama-cpp-dir`, `--sanity-test` (`--sanity-n 8` canaries through a temporary `llama-server` on `--sanity-port`, non-empty output + expected substrings), `--skip-merge`, `--skip-quantize`, `--keep-intermediate` |
| `tools/build_preference_triples.py` | `generate` (llama-server HTTP: `--server-url`, `--model`, `--candidates`, `--temperature`, `--max-tokens`) and `score` (CometKiwi-XL, `--margin`) |
| `qe_filter.py` | CometKiwi threshold filter for synthetic sermon pairs (`--threshold`, `--rejected-output`, `--scores-output`) |

**Production recipe:** `training/run_gemma4_e4b_domain_sft.sh` → `train_gemma4.py` (r=8,
α=8, 2 epochs, lr 2e-4, packing) → `export_gguf.py --qtype Q4_K_M --sanity-test` →
`models/gemma-4-e4b-it-q4km-domain.gguf`. The verse component defaults to v2
(see Corpus rule). Ship rule: stock Gemma 4 E4B stays the default until a Mac A/B note
(#135) says otherwise; next experiments are ranked in
[`v3_directions.md`](../docs/gemma4_tuning/v3_directions.md) (few-shot disambiguation in
the prompt, re-ranking, better preference pools).

Original Jacobo/Santiago preference candidates and their reproducible overlap manifest
are in [`candidates/README.md`](candidates/README.md). They are unapproved synthetic
teaching examples, not correction evidence or human translations. The CPO loader rejects
rows marked unapproved before importing Unsloth; historical triples without approval
metadata retain their existing format contract. The v2 holdout, semantic overlap review,
bilingual approval and a separate reviewed/scored export remain pending before CPO use.

## TranslateGemma QLoRA (historical, `train_gemma.py`)

Flags `--bible-data`, `--glossary-data`, `--sermon-data`, `--lora-r`, `--lora-alpha`,
`--epochs`, `--lr`, `--max-seq-length`, `--max-pairs`, `--max-steps`, `--neftune`,
`--replay-ratio`, `--lora-dropout`, `--glossary-oversample`, `--resume`. The S1–S9 sweep
(config → verse/sermon ratio → scale; S6 balanced 1:1 winner at COMET parity with the 12B
base) trained on the misaligned v1 corpus and is superseded by the Gemma 4 program.
Results and the hybrid 60/40 12B-vs-DeepL data design:
[`docs/archive/training/gemma_tuning_test_matrix.md`](../docs/archive/training/gemma_tuning_test_matrix.md),
[`docs/archive/training/benchmark_training.md`](../docs/archive/training/benchmark_training.md).
`generate_hybrid_synthetic.py` (`--ratio-deepl`, `--train-only`, `_provenance.json`
sidecars) and `benchmark_gemma4.py` (`--models tg4b tg12b e2b e4b`, `--skip-comet`) belong to
this era; the llama.cpp-vs-HF comparison that made Q4_K_M the CUDA default is in
[`docs/archive/v2026.5/BENCHMARK.md`](../docs/archive/v2026.5/BENCHMARK.md).

## Theological vocabulary

| English | Spanish options | Rule |
|---------|-----------------|------|
| Atonement | *expiación* vs *propiciación* | Removal of sin vs appeasing wrath |
| Covenant | *pacto* vs *alianza* | Protestant vs Catholic register — match audience |
| James | *Jacobo* (person) vs *Santiago* (epistle) | Context — the open canary (#136) |
| Breaking of bread | *partimiento del pan* | Fixed phrase |
| Righteousness / grace | *justicia* / *gracia* | Theological sense over everyday sense |

Mitigations: tiered glossary, Deepgram keyterms, canary set `theological_canaries.py`
(18 entries; `tools/health_check.py --n-canaries` defaults to 8, `export_gguf.py --sanity-n`
to 8), few-shot prompt examples proposed in `v3_directions.md`.

## Evaluation

- **Translation:** `evaluate_translation.py --adapter --base-model --test [--max-samples]
  [--compare-base] [--glossary-only] [--marian] [--deepl-key] [--output-file]` — SacreBLEU,
  chrF++, COMET; `evaluate_sermon.py --chunks --adapter --base-model [--ceiling-model]
  [--segment] [--deepl-key]` for sermon chunks. Holdout: v2 500-verse set; sermon eval 422
  chunks (`v1_results.md`). Human review of adequacy/fluency/theological precision remains
  the deciding gate and has not been run.
- **STT:** `eval_whisper_wer.py`, `tools/benchmark_stt_engines.py`, Parakeet bench
  `tools/benchmark_parakeet_en.py [--manifest] [--limit] [--skip-parakeet] [--device]`.
- **Go/no-go (from the plan):** WER > 10 % relative improvement minimum; canary ≥ 7/8 with
  8/8 target; stop when the worst metric improves < 2 % relative for two consecutive cycles.

## Adapter export and transfer

| Tool | Purpose |
|------|---------|
| `export_gguf.py` | Gemma: merge LoRA → bf16 HF → GGUF f16 → Q4_K_M via llama.cpp; feeds `LlamaCppEngine` (CUDA) — MLX uses the safetensors adapter directory directly (`--adapter-dir`). |
| `export_ct2.py` | Whisper: merge LoRA → HF → CTranslate2 (`--quantization int8_float16` default; `int8` for CPU/Lite-style deployments). Built-in sanity gate transcribes **5 canary clips** from `stark_data/whisper_dataset_deepgram/eval/` and aborts when WER exceeds `--sanity-wer-max 0.30`; `--no-sanity` skips it. Output loads in `FasterWhisperEngine` unchanged. |
| `tools/manage_adapters.py` | `register --adapter DIR --model NAME [--version V] [--eval-file JSON]` (version defaults to the directory name; SHA-256 of `adapter_model.safetensors` recorded), `activate --model --version [--base-model] [--max-latency]` (runs `health_check.py`), `rollback`, `list`, `export --model --target user@host:path` (rsync). Manifest `adapters/manifest.json` holds `versions`, `active`, `previous` per model. |
| `tools/deploy_adapters.py` | `--cycle N --models ... --endpoints local|mac-dev [--all-adapters] [--dry-run] [--rollback] [--skip-health]` — version → transfer → health check → activate → verify, local/rsync endpoints ([`docs/deploy.md`](../docs/deploy.md) for what is implemented vs designed). |

Transfer path: WSL `fine_tuned_*/` or `adapters/<model>/<version>/` → scp/rsync/USB → Mac
`adapters/`; then `python tools/health_check.py --backend mlx --adapter DIR` before
activation. Details and the Mac-side consumers: [`CLAUDE-windows.md`](../CLAUDE-windows.md)
§ Model Transfer to Mac.

## Adding a new language corpus

Hindi and Chinese remain **pending user decisions** (#138). `tools/offline_hindi.py`
(church audio → Parakeet English → Gemma Hindi, evaluation only; see
[`docs/evaluation/overnight_hindi/README.md`](../docs/evaluation/overnight_hindi/README.md))
has a completed offline R&D report, with no live integration or further work in the
EN↔ES latency program. No Hindi training data has been prepared. When a decision lands ([`docs/archive/research/multi_lingual.md`](../docs/archive/research/multi_lingual.md)):

1. Aligned verse pairs from `bible-nlp/biblenlp-corpus` (Hindi IRV `hin2017`, Chinese CUV-S `cmn-cu89s`).
2. `prepare_bible_corpus.py` → JSONL with `source_lang_code`, `target_lang_code`, `source_text`, `target_text`, `verse_id`.
3. Glossary of 100–150 terms via `build_glossary.py` as template (honorifics, denominational terms).
4. QLoRA at higher rank (r=32) and longer `--max-seq-length` for Hindi token fertility.
5. Evaluate with chrF++ first (morphology, no-space scripts), then COMET and term accuracy; `--tokenize zh` for Chinese SacreBLEU.
6. Same copyright rules as Spanish.

## Related work (unchanged)

eBible Corpus (2023); "From Priest to Doctor" (COLING 2025); BibleNLP community; domain-adapted
Whisper reports (aviation, industrial jargon). No published Whisper fine-tuning for church
speech — the gap this project addresses.
