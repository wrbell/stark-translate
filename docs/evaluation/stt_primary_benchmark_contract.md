# Primary STT benchmark identity

`benchmark_stt_engines.py`, the STT arm in `benchmark_mlx_accel.py`, and
`stt_roundtrip_compare.py` validate the engine's actual public model ID after
loading and before benchmark warmup or measurements. A failed primary that
loads another model is ineligible; it must never be labeled as a Turbo result.
Explicit `--model-id` overrides are the requested identity for this check.

`fallback_on_low_conf=False` only disables per-transcription fallback in the
shared MLX engine. It does not disable startup fallback. These tools therefore
also call the model-free `tools.benchmark_identity.load_primary_model` guard.
The acceleration tool's STT arm disables quality retries as well. Load/identity
failures release the rejected engine and stop an active benchmark memory sampler.
The acceleration report records an error and `eligible: false`, without latency;
the other tools fail before emitting primary-model measurements.

This is a public model-identity check, not a replacement for file hashes,
revision provenance, language compatibility, or reference quality. No inference
default or confidence threshold changes. Existing reports are not rewritten.
