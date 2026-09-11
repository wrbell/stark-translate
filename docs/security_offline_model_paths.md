# Operator-only offline model paths — 2026-09-11

The [retained source inventory](evaluation/mac_followup_20260910/live-hf-source-inventory.json)
records **105** `training_export_evaluation` call sites with
`residual_remote_risk: true`. These are the unpinned offline acquisition paths,
not 105 distinct models or Bandit findings. The same role also contains 24 sites
marked false and one unverified legacy Piper named loader (addressed separately
in the [optional-live follow-up](evaluation/mac_followup_20260910/live-hf-source-inventory.md#2026-09-11-follow-up)).
The JSON is historical evidence; the table below preserves its original site IDs.

These paths run only through explicit training, export, conversion, benchmark,
corpus preparation or evaluation commands on the WSL training box, or through
an operator's deliberate offline command on an inference machine. They are not
part of automatic church live-caption startup. “Offline” describes their role:
these commands can still access the network to acquire models or datasets.
Importing a helper does not authorize acquisition. Training stays on WSL;
standalone evaluation and scoring can also be operator-run on the Mac.

The calls include Transformers models/processors/tokenizers, Unsloth and PEFT
training/export sources, CT2 conversion, MLX draft and STT experiments, batch
Whisper/pyannote processing, COMET/BERTScore/LaBSE quality scoring, Torch Hub VAD,
and corpus or shell/heredoc acquisition. Explicit adapters, downstream checkpoint
loads, and already local-only calls are distinguished in the full JSON.

## CI policy and promotion to live

B615 is skipped in the repository's Bandit CI configuration for these paths;
this is a repository-wide skip, not a per-call exemption or proof of safety.
See the [Bandit invocation in the lint workflow](../.github/workflows/lint.yml)
and the [historical security assessment](evaluation/mac_v2026_14_security.md).
That invocation scans `engines/`, `features/`, `tools/`, `operator_app/` and
`settings.py`; training scripts and shell/heredoc sites are outside its roots.
This follow-up changes none of those settings. Model deserialization, remote
code, Torch Hub execution and transitive downloads remain separate concerns.

Before promoting an offline loader into a live path:

1. Identify every model, tokenizer, processor, adapter and nested dependency it
   can fetch, including fallback/draft models and wrappers' implicit defaults.
2. Record a verified full 40-character HF commit and required files in
   `models.lock.json`. Keep optional entries out of default setup with
   `required_for: []`. Do not borrow a revision from a converted or similarly
   named repository. Prepare local converted assets with source provenance.
3. Route Transformers loads through `resolve_hf_model_source` and forward its
   kwargs to every loader. For MLX wrappers, resolve a pinned complete snapshot
   through `resolve_model_for_loading` before passing a local path. Reject remote
   IDs without a manifest pin; never retry a moving branch after failure.
4. For revision-less loaders use `resolve_local_model_for_loading` and their
   local-only API: faster-whisper gets a directory plus `local_files_only=True`;
   NeMo gets one local `.nemo` checkpoint via `restore_from` (a snapshot entry
   lists that file in `required_files` and sets `weights_required: false`).
   Piper first resolves its local ONNX/config pair with `resolve_piper_voice`;
   named voices must also have a valid `pinned_hf_entry`.
5. Mock every downstream loader and downloader in tests. Cover pinned local and
   remote policies where supported, missing/malformed pins, partial caches,
   and fallback/draft failure. Check that rejection makes no loader/network call.
   Separately review native compatibility and the live quality gates before
   changing an operational default.

## The 105 residual offline sites

Original line numbers identify the retained inventory, not current source offsets.

| Original site | Enclosing symbol | Loader |
| --- | --- | --- |
| `engines/mlx_spec.py:80` | `load_gemma4_drafter` | `drafter_cls.from_pretrained` |
| `features/diarize.py:293` | `run_diarization` | `Pipeline.from_pretrained` |
| `features/diarize.py:362` | `transcribe_segments` | `mlx_whisper.transcribe` |
| `features/diarize.py:409` | `transcribe_segments` | `mlx_whisper.transcribe` |
| `features/diarize.py:575` | `main` | `Inference` |
| `scripts/cuda/convert_gemma4_assistant_gguf.sh:76` | `shell_or_embedded_python` | `huggingface-cli download "${repo}" --local-dir "${dest}"` |
| `scripts/cuda/convert_gemma4_assistant_gguf.sh:80` | `shell_or_embedded_python` | `snapshot_download` |
| `tools/batch_translate.py:161` | `segment_audio_vad` | `torch.hub.load` |
| `tools/benchmark_latency.py:208` | `_load_mlx_whisper` | `mlx_whisper.transcribe` |
| `tools/benchmark_latency.py:226` | `_load_mlx_gemma` | `load` |
| `tools/benchmark_latency.py:293` | `_load_vad` | `torch.hub.load` |
| `tools/benchmark_latency.py:309` | `_load_marian` | `MarianTokenizer.from_pretrained` |
| `tools/benchmark_latency.py:310` | `_load_marian` | `MarianMTModel.from_pretrained` |
| `tools/benchmark_latency.py:338` | `_load_ct2_marian` | `MarianTokenizer.from_pretrained` |
| `tools/benchmark_latency.py:422` | `bench_stt_suite._run_stt` | `mlx_whisper.transcribe` |
| `tools/benchmark_latency.py:427` | `bench_stt_suite._run_stt` | `mlx_whisper.transcribe` |
| `tools/benchmark_latency.py:961` | `bench_metal_cache` | `mlx_whisper.transcribe` |
| `tools/benchmark_latency.py:966` | `bench_metal_cache` | `mlx_whisper.transcribe` |
| `tools/benchmark_latency.py:1020` | `bench_e2e_suite` | `mlx_whisper.transcribe` |
| `tools/benchmark_latency.py:1028` | `bench_e2e_suite` | `mlx_whisper.transcribe` |
| `tools/benchmark_latency.py:1039` | `bench_e2e_suite` | `mlx_whisper.transcribe` |
| `tools/benchmark_latency.py:1048` | `bench_e2e_suite` | `mlx_whisper.transcribe` |
| `tools/benchmark_latency.py:1061` | `bench_e2e_suite` | `mlx_whisper.transcribe` |
| `tools/benchmark_latency.py:1068` | `bench_e2e_suite` | `mlx_whisper.transcribe` |
| `tools/benchmark_latency.py:1082` | `bench_e2e_suite` | `mlx_whisper.transcribe` |
| `tools/benchmark_latency.py:1093` | `bench_e2e_suite` | `mlx_whisper.transcribe` |
| `tools/benchmark_translate_engines.py:254` | `run_cometkiwi` | `download_model` |
| `tools/build_preference_triples.py:205` | `cmd_score` | `download_model` |
| `tools/build_v1_corpus.py:116` | `sample_opus` | `load_dataset` |
| `tools/convert_models_to_both.py:165` | `export_whisper_mlx` | `snapshot_download` |
| `tools/convert_models_to_both.py:228` | `export_whisper_ct2` | `ctranslate2.converters.TransformersConverter` |
| `tools/convert_models_to_both.py:271` | `validate_whisper_mlx` | `mlx_whisper.transcribe` |
| `tools/convert_models_to_both.py:354` | `export_gemma_mlx` | `snapshot_download` |
| `tools/convert_models_to_both.py:473` | `export_gemma_cuda` | `AutoModelForCausalLM.from_pretrained` |
| `tools/convert_models_to_both.py:480` | `export_gemma_cuda` | `PeftModel.from_pretrained` |
| `tools/convert_models_to_both.py:490` | `export_gemma_cuda` | `AutoTokenizer.from_pretrained` |
| `tools/convert_models_to_both.py:522` | `export_gemma_cuda` | `snapshot_download` |
| `tools/convert_models_to_both.py:577` | `export_marian_pytorch` | `snapshot_download` |
| `tools/health_check.py:102` | `_load_mlx` | `load` |
| `tools/kpi_report.py:228` | `_run_cometkiwi` | `download_model` |
| `tools/live_caption_monitor.py:1546` | `run_wav` | `mlx_whisper.transcribe` |
| `tools/mts_acceptance_probe.py:152` | `run_probe` | `mlx_load` |
| `tools/score_comet22.py:50` | `score_with_comet` | `download_model` |
| `tools/score_comet22.py:100` | `main` | `download_model` |
| `tools/stt_benchmark.py:148` | `bench_mlx_whisper` | `mlx_whisper.transcribe` |
| `tools/stt_benchmark.py:161` | `bench_mlx_whisper` | `mlx_whisper.transcribe` |
| `tools/stt_benchmark.py:166` | `bench_mlx_whisper` | `mlx_whisper.transcribe` |
| `tools/test_adaptive_model.py:404` | `load_ct2_marian` | `MarianTokenizer.from_pretrained` |
| `tools/test_adaptive_model.py:423` | `load_gemma_4b` | `load` |
| `tools/translation_qe.py:93` | `_load_backtranslation` | `MarianTokenizer.from_pretrained` |
| `tools/translation_qe.py:94` | `_load_backtranslation` | `MarianMTModel.from_pretrained` |
| `tools/translation_qe.py:115` | `tier2_score` | `bert_score` |
| `tools/translation_qe.py:139` | `_load_labse` | `SentenceTransformer` |
| `tools/validate_session.py:263` | `segment_audio_vad` | `torch.hub.load` |
| `training/align_deepgram_chunks.py:419` | `_build_preprocessed_cache` | `WhisperProcessor.from_pretrained` |
| `training/assess_quality.py:384` | `cross_check` | `WhisperModel` |
| `training/benchmark_gemma4.py:102` | `load_model_nf4` | `AutoTokenizer.from_pretrained` |
| `training/benchmark_gemma4.py:107` | `load_model_nf4` | `AutoModelForCausalLM.from_pretrained` |
| `training/benchmark_gemma4.py:460` | `run_benchmark` | `download_model` |
| `training/benchmark_quantize.py:60` | `` | `AutoModelForCausalLM.from_pretrained` |
| `training/benchmark_quantize.py:61` | `` | `AutoTokenizer.from_pretrained` |
| `training/benchmark_quantize.py:165` | `` | `AutoTokenizer.from_pretrained` |
| `training/benchmark_quantize.py:168` | `` | `AutoModelForCausalLM.from_pretrained` |
| `training/eval_whisper_wer.py:53` | `evaluate_adapter` | `WhisperProcessor.from_pretrained` |
| `training/eval_whisper_wer.py:58` | `evaluate_adapter` | `WhisperForConditionalGeneration.from_pretrained` |
| `training/eval_whisper_wer.py:76` | `evaluate_adapter` | `PeftModel.from_pretrained` |
| `training/evaluate_piper.py:303` | `transcribe_audio` | `mlx_whisper.transcribe` |
| `training/evaluate_sermon.py:389` | `evaluate_sermon` | `download_model` |
| `training/evaluate_sermon.py:458` | `evaluate_sermon` | `download_model` |
| `training/evaluate_translation.py:44` | `load_gemma_model` | `AutoTokenizer.from_pretrained` |
| `training/evaluate_translation.py:53` | `load_gemma_model` | `AutoModelForCausalLM.from_pretrained` |
| `training/evaluate_translation.py:61` | `load_gemma_model` | `PeftModel.from_pretrained` |
| `training/evaluate_translation.py:74` | `load_marian_model` | `MarianTokenizer.from_pretrained` |
| `training/evaluate_translation.py:75` | `load_marian_model` | `MarianMTModel.from_pretrained` |
| `training/evaluate_translation.py:205` | `evaluate_biblical_translation` | `download_model` |
| `training/evaluate_translation.py:450` | `evaluate_deepl_reference` | `download_model` |
| `training/export_ct2.py:116` | `merge_adapter_to_bf16` | `AutoModelForSpeechSeq2Seq.from_pretrained` |
| `training/export_ct2.py:124` | `merge_adapter_to_bf16` | `PeftModel.from_pretrained` |
| `training/export_ct2.py:143` | `merge_adapter_to_bf16` | `AutoProcessor.from_pretrained` |
| `training/export_ct2.py:144` | `merge_adapter_to_bf16` | `WhisperFeatureExtractor.from_pretrained` |
| `training/export_gguf.py:61` | `merge_adapter_to_bf16` | `AutoModelForCausalLM.from_pretrained` |
| `training/export_gguf.py:69` | `merge_adapter_to_bf16` | `PeftModel.from_pretrained` |
| `training/export_gguf.py:81` | `merge_adapter_to_bf16` | `AutoTokenizer.from_pretrained` |
| `training/mine_hard_examples.py:114` | `mine` | `WhisperProcessor.from_pretrained` |
| `training/mine_hard_examples.py:115` | `mine` | `WhisperForConditionalGeneration.from_pretrained` |
| `training/mine_hard_examples.py:123` | `mine` | `PeftModel.from_pretrained` |
| `training/prepare_bible_corpus.py:149` | `download_biblenlp` | `load_dataset` |
| `training/prepare_bible_corpus.py:166` | `download_helsinki` | `load_dataset` |
| `training/preprocess_audio.py:261` | `vad_chunk` | `torch.hub.load` |
| `training/preprocess_audio.py:328` | `diarize_speakers` | `Pipeline.from_pretrained` |
| `training/qe_filter.py:77` | `score_pairs` | `download_model` |
| `training/train_gemma.py:201` | `_load_replay_pairs` | `load_dataset` |
| `training/train_gemma.py:383` | `fine_tune_gemma` | `AutoTokenizer.from_pretrained` |
| `training/train_gemma.py:389` | `fine_tune_gemma` | `AutoModelForCausalLM.from_pretrained` |
| `training/train_gemma4.py:210` | `train` | `FastModel.from_pretrained` |
| `training/train_gemma4_cpo.py:95` | `train` | `FastModel.from_pretrained` |
| `training/train_gemma4_cpo.py:109` | `train` | `PeftModel.from_pretrained` |
| `training/train_marian.py:61` | `fine_tune_marian` | `MarianTokenizer.from_pretrained` |
| `training/train_marian.py:62` | `fine_tune_marian` | `MarianMTModel.from_pretrained` |
| `training/train_whisper.py:81` | `prepare_mixed_dataset` | `load_dataset` |
| `training/train_whisper.py:334` | `fine_tune_whisper` | `WhisperProcessor.from_pretrained` |
| `training/train_whisper.py:338` | `fine_tune_whisper` | `WhisperForConditionalGeneration.from_pretrained` |
| `training/transcribe_church.py:47` | `transcribe_with_transformers` | `pipeline` |
| `training/transcribe_church.py:168` | `transcribe_with_faster_whisper` | `WhisperModel` |
| `training/transcribe_sermons.sh:143` | `shell_or_embedded_python` | `model = WhisperModel` |
