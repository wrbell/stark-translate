# Input-level pilot: rejected before inference

The first actual five-recording English production pilot and closing control
used the unmodified FLEURS-derived float32 samples at CLI gain 1. Both exited
normally but produced zero previews/finals: all seven final attempts in each
run were rejected by the existing RMS energy gate. The source ledger and
completion guard retained this failure; it is not a latency result or a model
failure. No performance cohort was selected from these runs.

The original English recordings have RMS levels approximately 0.0005–0.0026,
below the pipeline's 0.008 STT gate. Spanish levels are substantially different.
Changing the pipeline threshold or using one large gain across both languages
would confound the comparison. Original WAVs, hashes and completed isolated STT
quality runs remain unchanged.

New replay copies use the declared existing speech target RMS 0.08, separately
for every original recording, with gain capped to keep peak amplitude at 0.95.
This is linear float32 scaling without clipping, followed by exactly one second
of digital silence. Each recording's gain, original/output RMS and peak, source
hash, references and sample boundaries are retained in the new provenance.
The same fixed rule applies to both models and later untouched confirmation;
it is not a latency-tuned filter. CLI gain remains 1.

The earlier protocol is preserved under
[pre_normalization](../protocol/pre_normalization/standard-screen.json). The
[new provenance](../protocol/public-replay-provenance.json) and hash-bound
protocols use distinct normalized-v2 identities. This is edited public read
speech, not a natural service or human microphone accuracy certificate.
