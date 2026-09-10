# Mac E4B / E2B comparison

E4B remains the default. No model-selection UI has been added.

Human-reference audio coverage: {"approved_natural_utterances": {"en": 0, "es": 0}, "natural_audio_gate": false}

Natural Spanish / two-speaker / bilingual review gates remain pending until explicitly validated.

## Translation on identical input

chrF++ measures agreement with public-domain verse wording, not overall translation accuracy. References use unique book/chapter/verse alignment to KJV and RVR1909; ambiguous source text has no score. Existing model outputs are preserved when references are repaired. Canary checks require every listed term; older benchmarks checked only the first term.

| Model | Prompt | Direction | Items | References | p50 ms | p95 ms | Canaries | chrF++ |
|---|---|---|---:|---:|---:|---:|---:|---:|

### All theological canaries

| Input | Model / prompt | Result | Required terms | Output |
|---|---|---|---|---|

## E2B tradeoff relative to E4B

| Prompt | Direction | Median latency reduction | Canary pass difference | chrF++ difference |
|---|---|---:|---:|---:|

## STT on saved session audio

Unreviewed machine transcripts are not WER references. Unconfirmed session recordings cannot pass the natural-speech gate.

| Engine | Language | Audio items | Approved references | WER | p50 ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|

## Real-time server latency

Speech end is estimated from captured VAD-positive frames. Server-final latency ends at payload readiness; it is not browser display latency. Clips, inference code/config cohorts, source types and endpoints are kept separate. Cohort IDs bind recorded source hashes, package versions and effective settings. A matching lifecycle source hash observed at startup takes precedence over later pipeline-file snapshots; this does not prove imported engine bytecode, and older snapshot timing may be ambiguous.

| Experiment | Model | Clip / cohort | Language/source | Endpoint | n | p50 ms | p95 ms |
|---|---|---|---|---|---:|---:|---:|
| routing_conservative | e2b | routing_synthetic_en / d833d80754638195 | en/synthetic_piper | silence | 9 | 1119.1 | 1343.4 |
| routing_conservative | e2b | routing_synthetic_es / 1796f95d14b7f5d0 | es/synthetic_piper | silence | 9 | 1920.4 | 1978.2 |
| routing_conservative | e4b | routing_synthetic_en / d833d80754638195 | en/synthetic_piper | silence | 9 | 1381.9 | 3252.0 |
| routing_conservative | e4b | routing_synthetic_es / 1796f95d14b7f5d0 | es/synthetic_piper | silence | 9 | 2031.5 | 2915.2 |
| routing_legacy | e2b | routing_synthetic_en / 17ac8e31b8aae9cf | en/synthetic_piper | silence | 9 | 904.8 | 1143.1 |
| routing_legacy | e2b | routing_synthetic_es / 2acd67857d21a690 | es/synthetic_piper | silence | 9 | 1623.8 | 1956.6 |
| routing_legacy | e4b | routing_synthetic_en / 17ac8e31b8aae9cf | en/synthetic_piper | silence | 9 | 915.9 | 1945.6 |
| routing_legacy | e4b | routing_synthetic_es / 2acd67857d21a690 | es/synthetic_piper | silence | 9 | 1720.8 | 2972.3 |

## Partial delivery and visible browser timing

First-partial delay starts at the first captured speech frame; one earliest delay is counted per known utterance. Update gaps use chronological emissions, including speaking pauses. These partial timings end at server readiness, before browser rendering.

| Experiment | Model | Clip / cohort | Source | Metric | n | p50 ms | p95 ms |
|---|---|---|---|---|---:|---:|---:|
| routing_conservative | e2b | routing_synthetic_en / d833d80754638195 | en/synthetic_piper | captured_end_to_partial_ms | 18 | 144.4 | 714.3 |
| routing_conservative | e2b | routing_synthetic_en / d833d80754638195 | en/synthetic_piper | partial_update_gap_ms | 15 | 637.3 | 2709.2 |
| routing_conservative | e2b | routing_synthetic_en / d833d80754638195 | en/synthetic_piper | first_partial_ms | 9 | 920.4 | 1345.7 |
| routing_conservative | e2b | routing_synthetic_es / 1796f95d14b7f5d0 | es/synthetic_piper | captured_end_to_partial_ms | 6 | 580.0 | 588.9 |
| routing_conservative | e2b | routing_synthetic_es / 1796f95d14b7f5d0 | es/synthetic_piper | partial_update_gap_ms | 3 | 3375.4 | 3380.6 |
| routing_conservative | e2b | routing_synthetic_es / 1796f95d14b7f5d0 | es/synthetic_piper | first_partial_ms | 6 | 1492.0 | 1804.9 |
| routing_conservative | e4b | routing_synthetic_en / d833d80754638195 | en/synthetic_piper | captured_end_to_partial_ms | 14 | 155.6 | 424.0 |
| routing_conservative | e4b | routing_synthetic_en / d833d80754638195 | en/synthetic_piper | partial_update_gap_ms | 11 | 676.9 | 2830.7 |
| routing_conservative | e4b | routing_synthetic_en / d833d80754638195 | en/synthetic_piper | first_partial_ms | 8 | 940.2 | 1338.0 |
| routing_conservative | e4b | routing_synthetic_es / 1796f95d14b7f5d0 | es/synthetic_piper | captured_end_to_partial_ms | 6 | 583.5 | 617.9 |
| routing_conservative | e4b | routing_synthetic_es / 1796f95d14b7f5d0 | es/synthetic_piper | partial_update_gap_ms | 3 | 3367.9 | 3400.5 |
| routing_conservative | e4b | routing_synthetic_es / 1796f95d14b7f5d0 | es/synthetic_piper | first_partial_ms | 6 | 1498.2 | 1833.9 |
| routing_legacy | e2b | routing_synthetic_en / 17ac8e31b8aae9cf | en/synthetic_piper | captured_end_to_partial_ms | 19 | 148.5 | 698.4 |
| routing_legacy | e2b | routing_synthetic_en / 17ac8e31b8aae9cf | en/synthetic_piper | partial_update_gap_ms | 16 | 623.5 | 2721.0 |
| routing_legacy | e2b | routing_synthetic_en / 17ac8e31b8aae9cf | en/synthetic_piper | first_partial_ms | 9 | 1159.0 | 1345.8 |
| routing_legacy | e2b | routing_synthetic_es / 2acd67857d21a690 | es/synthetic_piper | captured_end_to_partial_ms | 6 | 577.2 | 589.1 |
| routing_legacy | e2b | routing_synthetic_es / 2acd67857d21a690 | es/synthetic_piper | partial_update_gap_ms | 3 | 3379.6 | 3381.5 |
| routing_legacy | e2b | routing_synthetic_es / 2acd67857d21a690 | es/synthetic_piper | first_partial_ms | 6 | 1489.2 | 1805.1 |
| routing_legacy | e4b | routing_synthetic_en / 17ac8e31b8aae9cf | en/synthetic_piper | captured_end_to_partial_ms | 16 | 154.6 | 317.3 |
| routing_legacy | e4b | routing_synthetic_en / 17ac8e31b8aae9cf | en/synthetic_piper | partial_update_gap_ms | 13 | 640.0 | 2850.1 |
| routing_legacy | e4b | routing_synthetic_en / 17ac8e31b8aae9cf | en/synthetic_piper | first_partial_ms | 8 | 857.6 | 1344.0 |
| routing_legacy | e4b | routing_synthetic_es / 2acd67857d21a690 | es/synthetic_piper | captured_end_to_partial_ms | 5 | 582.6 | 968.8 |
| routing_legacy | e4b | routing_synthetic_es / 2acd67857d21a690 | es/synthetic_piper | partial_update_gap_ms | 2 | 3041.3 | 3116.7 |
| routing_legacy | e4b | routing_synthetic_es / 2acd67857d21a690 | es/synthetic_piper | first_partial_ms | 5 | 1790.7 | 1798.6 |

### Visible final latency by browser session and endpoint

Only visible schema-2 final ACKs matched to a unique CSV chunk contribute. Each browser session, capture timing source and endpoint has its own distribution; clients and silence/forced-cut/EOF endpoints are never pooled. Duplicate chunk acknowledgments and unmatched/stale events are excluded. Speech-end-to-ack includes return-network time. Coverage alone does not make a run acceptance eligible; the latency gate remains pending.

| Experiment | Model | Clip / cohort | Session / client | Endpoint / timing source | Metric | n | p50 ms | p95 ms | ACKs / finals |
|---|---|---|---|---|---|---:|---:|---:|---|

### Visible final acknowledgment coverage

Coverage counts each finalized chunk once when any visible client acknowledged it. Missing acknowledgments are missing evidence, not proof of display failure; replay shutdown can race the final browser acknowledgment.

| Experiment | Model | Clip / cohort | Received final chunks | Finalized chunks |
|---|---|---|---:|---:|
| routing_conservative | e2b | routing_synthetic_en / d833d80754638195 | 0 | 9 |
| routing_conservative | e2b | routing_synthetic_es / 1796f95d14b7f5d0 | 0 | 9 |
| routing_conservative | e4b | routing_synthetic_en / d833d80754638195 | 0 | 9 |
| routing_conservative | e4b | routing_synthetic_es / 1796f95d14b7f5d0 | 0 | 9 |
| routing_legacy | e2b | routing_synthetic_en / 17ac8e31b8aae9cf | 0 | 9 |
| routing_legacy | e2b | routing_synthetic_es / 2acd67857d21a690 | 0 | 9 |
| routing_legacy | e4b | routing_synthetic_en / 17ac8e31b8aae9cf | 0 | 9 |
| routing_legacy | e4b | routing_synthetic_es / 2acd67857d21a690 | 0 | 9 |

## Changed translation examples

## Pending gates

- ≥50 approved natural-speech reference utterances per language.
- Bilingual review of meaning errors and terminology preferences (blind_review.jsonl).
- Visible-browser render acknowledgments for the sub-second caption-delivery gate.
- Physical second-output / hotplug and real two-speaker validation.

Recorded failed runs: 0. Excluded incompatible/duplicate runs: 0. Details are retained in comparison.json.
