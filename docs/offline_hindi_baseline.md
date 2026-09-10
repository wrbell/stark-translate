# Offline Hindi audio baseline

This development probe runs actual local church audio through Parakeet English
STT, saves those inputs once, and translates them with the shared Gemma 4 engine
using `target_lang="hi"`. It adds no live language mode. All artifacts are marked
`evaluation_only`; no review, training export, or model-default promotion occurs.

The frozen [manifest](../tools/offline_hindi_manifest.json) binds the December
45-second clip and February 150-second clip used in prior Mac testing by SHA-256.
Eight existing theological canaries provide additional text-only probes. The
local corpus has no Hindi reference translations: references and Hindi term
requirements remain null. Generation success, Devanagari presence, and retained
Latin words are availability diagnostics, not translation accuracy scores.

Run one command at a time on an otherwise idle Mac with the models already cached:

```bash
python tools/offline_hindi.py transcribe --data-root . --output metrics/hindi_audio_stt
python tools/offline_hindi.py translate --transcripts metrics/hindi_audio_stt --size e4b --runs 3 --output metrics/hindi_audio_e4b
python tools/offline_hindi.py translate --transcripts metrics/hindi_audio_stt --size e2b --runs 3 --output metrics/hindi_audio_e2b
python tools/offline_hindi.py report --transcripts metrics/hindi_audio_stt --translations metrics/hindi_audio_e4b metrics/hindi_audio_e2b --output metrics/hindi_audio_report.json
```

Each output directory must be new. Interrupted attempts retain partial rows and
a failed status; they cannot be used as completed report inputs. Original audio,
existing session diagnostics, and holdout files are read only. The two translation
commands reuse the same hashed STT rows. They load one Gemma model per process,
use the canonical Gemma 4 prompt/EOS handling with thinking off, and keep adapters,
drafts, prefix caching, and terminology prompt modifications off. No network model
downloads are allowed by these commands.

The STT stage uses installed Silero weights to make bounded offline utterances,
with 500 ms silence, 30 ms speech padding, and an 8-second maximum. These are
offline segment boundaries, not a live endpoint latency measurement. Rows retain
the source WAV hash, original format, exact sample bounds after 16 kHz resampling,
the hash of the actual float32 STT input, raw English transcript/confidence, and
Hindi generation. Parakeet does not support forced-language decoding or prompt
biasing. This baseline applies no STT corrections.

Model paths/revisions, settings, source hashes, package versions, VAD artifact,
peak process memory, generation budget status, and every actual output are saved.
The report separates translation durations for audio-derived utterances and text
probes. These durations exclude VAD/STT and are not speech-to-caption latency.
The baseline does not determine whether Hindi QLoRA is warranted.

A bilingual reviewer must inspect the source audio, English recognition, Hindi
meaning, theological terms, negation, omissions, and any Latin fragments. To add
real reference verse pairs, copy the manifest to a new file, add text probes with
their exact English source, `reference_hi`, and `reference_provenance` identifying
the edition/source and verse, then pass that new file through `--manifest` for
every stage. Do not label generated Hindi as a reference. A new manifest requires
a new transcription artifact even when its audio is unchanged, keeping the
evaluation identity explicit. Human Hindi quality approval and live Hindi support
remain pending.
