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
| conservative_marian | e2b | screening_45s / 75d97c4e65cbef53 | en/church_replay | smart_cut | 12 | 3098.9 | 4191.5 |
| conservative_marian | e2b | screening_45s / 75d97c4e65cbef53 | en/church_replay | hard_cut | 3 | 1289.4 | 1790.1 |
| conservative_marian | e2b | screening_45s / 75d97c4e65cbef53 | en/church_replay | silence | 6 | 1706.0 | 2191.3 |
| conservative_marian | e4b | screening_45s / 75d97c4e65cbef53 | en/church_replay | smart_cut | 12 | 3474.6 | 4682.3 |
| conservative_marian | e4b | screening_45s / 75d97c4e65cbef53 | en/church_replay | hard_cut | 3 | 1958.3 | 2734.4 |
| conservative_marian | e4b | screening_45s / 75d97c4e65cbef53 | en/church_replay | silence | 6 | 2009.3 | 2210.4 |
| final_aware_partials | e2b | screening_45s / ada3e8e10d7f0d3e | en/church_replay | smart_cut | 12 | 3057.1 | 3987.8 |
| final_aware_partials | e2b | screening_45s / ada3e8e10d7f0d3e | en/church_replay | hard_cut | 3 | 1278.6 | 1278.6 |
| final_aware_partials | e2b | screening_45s / ada3e8e10d7f0d3e | en/church_replay | silence | 6 | 1502.8 | 1683.1 |
| final_aware_partials | e4b | screening_45s / ada3e8e10d7f0d3e | en/church_replay | smart_cut | 12 | 3629.3 | 4175.0 |
| final_aware_partials | e4b | screening_45s / ada3e8e10d7f0d3e | en/church_replay | hard_cut | 3 | 1860.5 | 3434.0 |
| final_aware_partials | e4b | screening_45s / ada3e8e10d7f0d3e | en/church_replay | silence | 6 | 2075.7 | 2934.3 |
| idle_warmup | e2b | screening_45s / 03f6856a73cde1f6 | en/church_replay | smart_cut | 12 | 3089.9 | 3802.3 |
| idle_warmup | e2b | screening_45s / 03f6856a73cde1f6 | en/church_replay | hard_cut | 3 | 1082.6 | 1131.8 |
| idle_warmup | e2b | screening_45s / 03f6856a73cde1f6 | en/church_replay | silence | 6 | 1353.8 | 1496.6 |
| idle_warmup | e4b | screening_45s / 03f6856a73cde1f6 | en/church_replay | smart_cut | 12 | 3730.2 | 5358.3 |
| idle_warmup | e4b | screening_45s / 03f6856a73cde1f6 | en/church_replay | hard_cut | 3 | 2414.6 | 2462.1 |
| idle_warmup | e4b | screening_45s / 03f6856a73cde1f6 | en/church_replay | silence | 6 | 1826.3 | 2756.7 |
| onnx_vad | e2b | screening_45s / d81ff9df7ef67153 | en/church_replay | smart_cut | 12 | 3073.4 | 3785.2 |
| onnx_vad | e2b | screening_45s / d81ff9df7ef67153 | en/church_replay | hard_cut | 3 | 1269.4 | 1276.5 |
| onnx_vad | e2b | screening_45s / d81ff9df7ef67153 | en/church_replay | silence | 6 | 1504.0 | 1677.7 |
| onnx_vad | e4b | screening_45s / d81ff9df7ef67153 | en/church_replay | smart_cut | 12 | 3427.5 | 4656.2 |
| onnx_vad | e4b | screening_45s / d81ff9df7ef67153 | en/church_replay | hard_cut | 3 | 1924.0 | 1969.6 |
| onnx_vad | e4b | screening_45s / d81ff9df7ef67153 | en/church_replay | silence | 6 | 1959.7 | 2182.3 |
| screening_baseline | e2b | screening_45s / bc7df10a30718535 | en/church_replay | smart_cut | 12 | 3090.6 | 3837.3 |
| screening_baseline | e2b | screening_45s / bc7df10a30718535 | en/church_replay | hard_cut | 3 | 1242.3 | 1285.6 |
| screening_baseline | e2b | screening_45s / bc7df10a30718535 | en/church_replay | silence | 6 | 1516.0 | 1702.1 |
| screening_baseline | e4b | screening_45s / bc7df10a30718535 | en/church_replay | smart_cut | 12 | 3606.9 | 4784.2 |
| screening_baseline | e4b | screening_45s / bc7df10a30718535 | en/church_replay | hard_cut | 3 | 2506.1 | 2935.8 |
| screening_baseline | e4b | screening_45s / bc7df10a30718535 | en/church_replay | silence | 6 | 2002.9 | 2823.4 |
| silence_035 | e2b | screening_45s / 8c42749813225951 | en/church_replay | silence | 9 | 1188.8 | 1643.7 |
| silence_035 | e2b | screening_45s / 8c42749813225951 | en/church_replay | smart_cut | 12 | 3043.1 | 4757.8 |
| silence_035 | e2b | screening_45s / 8c42749813225951 | en/church_replay | hard_cut | 3 | 1293.3 | 1302.6 |
| silence_035 | e4b | screening_45s / 8c42749813225951 | en/church_replay | silence | 9 | 1628.9 | 3024.5 |
| silence_035 | e4b | screening_45s / 8c42749813225951 | en/church_replay | smart_cut | 12 | 3373.9 | 5478.1 |
| silence_035 | e4b | screening_45s / 8c42749813225951 | en/church_replay | hard_cut | 3 | 2016.0 | 2557.3 |
| silence_04 | e2b | screening_45s / 8acb7971fe82e92c | en/church_replay | silence | 9 | 1257.6 | 1674.3 |
| silence_04 | e2b | screening_45s / 8acb7971fe82e92c | en/church_replay | smart_cut | 12 | 3016.6 | 4965.3 |
| silence_04 | e2b | screening_45s / 8acb7971fe82e92c | en/church_replay | hard_cut | 3 | 1289.1 | 1328.3 |
| silence_04 | e4b | screening_45s / 8acb7971fe82e92c | en/church_replay | silence | 9 | 1652.8 | 2385.0 |
| silence_04 | e4b | screening_45s / 8acb7971fe82e92c | en/church_replay | smart_cut | 12 | 3349.1 | 5683.9 |
| silence_04 | e4b | screening_45s / 8acb7971fe82e92c | en/church_replay | hard_cut | 3 | 2073.6 | 3043.3 |
| terminology | e2b | screening_45s / 27e7ad96f32fbd3f | en/church_replay | smart_cut | 12 | 3087.2 | 3926.0 |
| terminology | e2b | screening_45s / 27e7ad96f32fbd3f | en/church_replay | hard_cut | 3 | 1293.2 | 1297.3 |
| terminology | e2b | screening_45s / 27e7ad96f32fbd3f | en/church_replay | silence | 6 | 1496.8 | 1608.4 |
| terminology | e4b | screening_45s / 27e7ad96f32fbd3f | en/church_replay | smart_cut | 12 | 3428.1 | 5246.0 |
| terminology | e4b | screening_45s / 27e7ad96f32fbd3f | en/church_replay | hard_cut | 3 | 2005.1 | 2733.1 |
| terminology | e4b | screening_45s / 27e7ad96f32fbd3f | en/church_replay | silence | 6 | 1975.5 | 2964.0 |

## Partial delivery and visible browser timing

First-partial delay starts at the first captured speech frame; one earliest delay is counted per known utterance. Update gaps use chronological emissions, including speaking pauses. These partial timings end at server readiness, before browser rendering.

| Experiment | Model | Clip / cohort | Source | Metric | n | p50 ms | p95 ms |
|---|---|---|---|---|---:|---:|---:|
| conservative_marian | e2b | screening_45s / 75d97c4e65cbef53 | en/church_replay | captured_end_to_partial_ms | 201 | 206.1 | 545.7 |
| conservative_marian | e2b | screening_45s / 75d97c4e65cbef53 | en/church_replay | partial_update_gap_ms | 198 | 621.8 | 1221.0 |
| conservative_marian | e2b | screening_45s / 75d97c4e65cbef53 | en/church_replay | first_partial_ms | 21 | 974.0 | 4043.9 |
| conservative_marian | e4b | screening_45s / 75d97c4e65cbef53 | en/church_replay | captured_end_to_partial_ms | 196 | 215.5 | 710.2 |
| conservative_marian | e4b | screening_45s / 75d97c4e65cbef53 | en/church_replay | partial_update_gap_ms | 193 | 628.0 | 1277.5 |
| conservative_marian | e4b | screening_45s / 75d97c4e65cbef53 | en/church_replay | first_partial_ms | 21 | 868.8 | 4166.5 |
| final_aware_partials | e2b | screening_45s / ada3e8e10d7f0d3e | en/church_replay | captured_end_to_partial_ms | 190 | 202.8 | 477.0 |
| final_aware_partials | e2b | screening_45s / ada3e8e10d7f0d3e | en/church_replay | partial_update_gap_ms | 187 | 625.7 | 1252.1 |
| final_aware_partials | e2b | screening_45s / ada3e8e10d7f0d3e | en/church_replay | first_partial_ms | 21 | 1352.2 | 3678.9 |
| final_aware_partials | e4b | screening_45s / ada3e8e10d7f0d3e | en/church_replay | captured_end_to_partial_ms | 173 | 213.4 | 651.4 |
| final_aware_partials | e4b | screening_45s / ada3e8e10d7f0d3e | en/church_replay | partial_update_gap_ms | 170 | 628.1 | 1801.5 |
| final_aware_partials | e4b | screening_45s / ada3e8e10d7f0d3e | en/church_replay | first_partial_ms | 21 | 1986.4 | 3830.2 |
| idle_warmup | e2b | screening_45s / 03f6856a73cde1f6 | en/church_replay | captured_end_to_partial_ms | 206 | 198.8 | 322.2 |
| idle_warmup | e2b | screening_45s / 03f6856a73cde1f6 | en/church_replay | partial_update_gap_ms | 203 | 621.8 | 1066.8 |
| idle_warmup | e2b | screening_45s / 03f6856a73cde1f6 | en/church_replay | first_partial_ms | 21 | 784.4 | 3657.7 |
| idle_warmup | e4b | screening_45s / 03f6856a73cde1f6 | en/church_replay | captured_end_to_partial_ms | 187 | 226.9 | 845.1 |
| idle_warmup | e4b | screening_45s / 03f6856a73cde1f6 | en/church_replay | partial_update_gap_ms | 184 | 626.7 | 1378.3 |
| idle_warmup | e4b | screening_45s / 03f6856a73cde1f6 | en/church_replay | first_partial_ms | 21 | 1334.8 | 4768.0 |
| onnx_vad | e2b | screening_45s / d81ff9df7ef67153 | en/church_replay | captured_end_to_partial_ms | 207 | 194.9 | 462.6 |
| onnx_vad | e2b | screening_45s / d81ff9df7ef67153 | en/church_replay | partial_update_gap_ms | 204 | 620.6 | 1059.2 |
| onnx_vad | e2b | screening_45s / d81ff9df7ef67153 | en/church_replay | first_partial_ms | 21 | 783.5 | 3645.0 |
| onnx_vad | e4b | screening_45s / d81ff9df7ef67153 | en/church_replay | captured_end_to_partial_ms | 199 | 214.1 | 606.8 |
| onnx_vad | e4b | screening_45s / d81ff9df7ef67153 | en/church_replay | partial_update_gap_ms | 196 | 624.5 | 1245.5 |
| onnx_vad | e4b | screening_45s / d81ff9df7ef67153 | en/church_replay | first_partial_ms | 21 | 883.3 | 3913.4 |
| screening_baseline | e2b | screening_45s / bc7df10a30718535 | en/church_replay | captured_end_to_partial_ms | 207 | 192.2 | 434.2 |
| screening_baseline | e2b | screening_45s / bc7df10a30718535 | en/church_replay | partial_update_gap_ms | 204 | 621.9 | 1059.5 |
| screening_baseline | e2b | screening_45s / bc7df10a30718535 | en/church_replay | first_partial_ms | 21 | 782.4 | 3656.2 |
| screening_baseline | e4b | screening_45s / bc7df10a30718535 | en/church_replay | captured_end_to_partial_ms | 195 | 212.8 | 672.1 |
| screening_baseline | e4b | screening_45s / bc7df10a30718535 | en/church_replay | partial_update_gap_ms | 192 | 622.4 | 1253.2 |
| screening_baseline | e4b | screening_45s / bc7df10a30718535 | en/church_replay | first_partial_ms | 21 | 1385.5 | 3728.7 |
| silence_035 | e2b | screening_45s / 8c42749813225951 | en/church_replay | captured_end_to_partial_ms | 210 | 193.8 | 330.1 |
| silence_035 | e2b | screening_45s / 8c42749813225951 | en/church_replay | partial_update_gap_ms | 207 | 621.7 | 976.7 |
| silence_035 | e2b | screening_45s / 8c42749813225951 | en/church_replay | first_partial_ms | 24 | 773.6 | 4481.9 |
| silence_035 | e4b | screening_45s / 8c42749813225951 | en/church_replay | captured_end_to_partial_ms | 201 | 208.1 | 582.2 |
| silence_035 | e4b | screening_45s / 8c42749813225951 | en/church_replay | partial_update_gap_ms | 198 | 622.8 | 1169.5 |
| silence_035 | e4b | screening_45s / 8c42749813225951 | en/church_replay | first_partial_ms | 22 | 1293.2 | 4645.9 |
| silence_04 | e2b | screening_45s / 8acb7971fe82e92c | en/church_replay | captured_end_to_partial_ms | 209 | 193.5 | 335.2 |
| silence_04 | e2b | screening_45s / 8acb7971fe82e92c | en/church_replay | partial_update_gap_ms | 206 | 620.9 | 1010.9 |
| silence_04 | e2b | screening_45s / 8acb7971fe82e92c | en/church_replay | first_partial_ms | 24 | 779.3 | 4487.1 |
| silence_04 | e4b | screening_45s / 8acb7971fe82e92c | en/church_replay | captured_end_to_partial_ms | 205 | 206.4 | 476.9 |
| silence_04 | e4b | screening_45s / 8acb7971fe82e92c | en/church_replay | partial_update_gap_ms | 202 | 622.7 | 1096.8 |
| silence_04 | e4b | screening_45s / 8acb7971fe82e92c | en/church_replay | first_partial_ms | 24 | 877.1 | 4592.6 |
| terminology | e2b | screening_45s / 27e7ad96f32fbd3f | en/church_replay | captured_end_to_partial_ms | 206 | 200.3 | 455.8 |
| terminology | e2b | screening_45s / 27e7ad96f32fbd3f | en/church_replay | partial_update_gap_ms | 203 | 622.7 | 1069.5 |
| terminology | e2b | screening_45s / 27e7ad96f32fbd3f | en/church_replay | first_partial_ms | 21 | 787.2 | 3667.6 |
| terminology | e4b | screening_45s / 27e7ad96f32fbd3f | en/church_replay | captured_end_to_partial_ms | 192 | 211.9 | 660.5 |
| terminology | e4b | screening_45s / 27e7ad96f32fbd3f | en/church_replay | partial_update_gap_ms | 189 | 620.8 | 1374.7 |
| terminology | e4b | screening_45s / 27e7ad96f32fbd3f | en/church_replay | first_partial_ms | 21 | 1396.4 | 4466.6 |

### Visible final latency by browser session and endpoint

Only visible schema-2 final ACKs matched to a unique CSV chunk contribute. Each browser session, capture timing source and endpoint has its own distribution; clients and silence/forced-cut/EOF endpoints are never pooled. Duplicate chunk acknowledgments and unmatched/stale events are excluded. Speech-end-to-ack includes return-network time. Coverage alone does not make a run acceptance eligible; the latency gate remains pending.

| Experiment | Model | Clip / cohort | Session / client | Endpoint / timing source | Metric | n | p50 ms | p95 ms | ACKs / finals |
|---|---|---|---|---|---|---:|---:|---:|---|

### Visible final acknowledgment coverage

Coverage counts each finalized chunk once when any visible client acknowledged it. Missing acknowledgments are missing evidence, not proof of display failure; replay shutdown can race the final browser acknowledgment.

| Experiment | Model | Clip / cohort | Received final chunks | Finalized chunks |
|---|---|---|---:|---:|
| conservative_marian | e2b | screening_45s / 75d97c4e65cbef53 | 0 | 21 |
| conservative_marian | e4b | screening_45s / 75d97c4e65cbef53 | 0 | 21 |
| final_aware_partials | e2b | screening_45s / ada3e8e10d7f0d3e | 0 | 21 |
| final_aware_partials | e4b | screening_45s / ada3e8e10d7f0d3e | 0 | 21 |
| idle_warmup | e2b | screening_45s / 03f6856a73cde1f6 | 0 | 21 |
| idle_warmup | e4b | screening_45s / 03f6856a73cde1f6 | 0 | 21 |
| onnx_vad | e2b | screening_45s / d81ff9df7ef67153 | 0 | 21 |
| onnx_vad | e4b | screening_45s / d81ff9df7ef67153 | 0 | 21 |
| screening_baseline | e2b | screening_45s / bc7df10a30718535 | 0 | 21 |
| screening_baseline | e4b | screening_45s / bc7df10a30718535 | 0 | 21 |
| silence_035 | e2b | screening_45s / 8c42749813225951 | 0 | 24 |
| silence_035 | e4b | screening_45s / 8c42749813225951 | 0 | 24 |
| silence_04 | e2b | screening_45s / 8acb7971fe82e92c | 0 | 24 |
| silence_04 | e4b | screening_45s / 8acb7971fe82e92c | 0 | 24 |
| terminology | e2b | screening_45s / 27e7ad96f32fbd3f | 0 | 21 |
| terminology | e4b | screening_45s / 27e7ad96f32fbd3f | 0 | 21 |

## Changed translation examples

## Pending gates

- ≥50 approved natural-speech reference utterances per language.
- Bilingual review of meaning errors and terminology preferences (blind_review.jsonl).
- Visible-browser render acknowledgments for the sub-second caption-delivery gate.
- Physical second-output / hotplug and real two-speaker validation.

Recorded failed runs: 0. Excluded incompatible/duplicate runs: 0. Details are retained in comparison.json.
