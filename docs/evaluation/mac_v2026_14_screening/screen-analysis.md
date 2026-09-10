# Frozen 45-second screening analysis

All 48 runs completed. Values below are recomputed from raw CSV/JSONL, never averaged from persisted percentiles. Units are milliseconds; each cell is p50 / p95. No default promotion or visible-display latency claim follows from these results.

## Final payload readiness

|Config|Model|Finals|All endpoints|Silence|Smart cut|Hard cut|
|---|---|---:|---|---|---|---|
|screening_baseline|e4b|21|2751.7 / 4059.6 (n=21)|2002.9 / 2823.4 (n=6)|3606.9 / 4784.2 (n=12)|2506.1 / 2935.8 (n=3)|
|screening_baseline|e2b|21|1670.2 / 3763.3 (n=21)|1516.0 / 1702.1 (n=6)|3090.6 / 3837.3 (n=12)|1242.3 / 1285.6 (n=3)|
|idle_warmup|e4b|21|2756.7 / 5228.6 (n=21)|1826.3 / 2756.7 (n=6)|3730.2 / 5358.3 (n=12)|2414.6 / 2462.1 (n=3)|
|idle_warmup|e2b|21|1495.1 / 3779.4 (n=21)|1353.8 / 1496.6 (n=6)|3089.9 / 3802.3 (n=12)|1082.6 / 1131.8 (n=3)|
|final_aware_partials|e4b|21|2766.5 / 4114.5 (n=21)|2075.7 / 2934.3 (n=6)|3629.3 / 4175.0 (n=12)|1860.5 / 3434.0 (n=3)|
|final_aware_partials|e2b|21|1603.1 / 3810.1 (n=21)|1502.8 / 1683.1 (n=6)|3057.1 / 3987.8 (n=12)|1278.6 / 1278.6 (n=3)|
|silence_04|e4b|24|2176.8 / 5088.9 (n=24)|1652.8 / 2385.0 (n=9)|3349.1 / 5683.9 (n=12)|2073.6 / 3043.3 (n=3)|
|silence_04|e2b|24|1488.5 / 4757.2 (n=24)|1257.6 / 1674.3 (n=9)|3016.6 / 4965.3 (n=12)|1289.1 / 1328.3 (n=3)|
|silence_035|e4b|24|2606.9 / 5047.8 (n=24)|1628.9 / 3024.5 (n=9)|3373.9 / 5478.1 (n=12)|2016.0 / 2557.3 (n=3)|
|silence_035|e2b|24|1463.3 / 4739.9 (n=24)|1188.8 / 1643.7 (n=9)|3043.2 / 4757.8 (n=12)|1293.3 / 1302.6 (n=3)|
|conservative_marian|e4b|21|2210.4 / 4469.8 (n=21)|2009.3 / 2210.4 (n=6)|3474.6 / 4682.3 (n=12)|1958.3 / 2734.4 (n=3)|
|conservative_marian|e2b|21|1798.4 / 4135.6 (n=21)|1706.0 / 2191.3 (n=6)|3098.9 / 4191.5 (n=12)|1289.4 / 1790.1 (n=3)|
|terminology|e4b|21|2733.1 / 4704.3 (n=21)|1975.5 / 2964.0 (n=6)|3428.1 / 5246.0 (n=12)|2005.1 / 2733.1 (n=3)|
|terminology|e2b|21|1601.4 / 3801.1 (n=21)|1496.8 / 1608.4 (n=6)|3087.2 / 3926.0 (n=12)|1293.2 / 1297.3 (n=3)|
|onnx_vad|e4b|21|2155.5 / 4142.8 (n=21)|1959.7 / 2182.3 (n=6)|3427.5 / 4656.2 (n=12)|1924.0 / 1969.6 (n=3)|
|onnx_vad|e2b|21|1664.3 / 3771.0 (n=21)|1504.0 / 1677.7 (n=6)|3073.4 / 3785.2 (n=12)|1269.4 / 1276.5 (n=3)|

## Partial readiness

|Config|Model|Partial count|First partial|Update gap|Capture end to ready|Final utterances without partial|
|---|---|---:|---|---|---|---:|
|screening_baseline|e4b|195|1385.5 / 3728.7 (n=21)|622.4 / 1253.2 (n=192)|212.8 / 672.1 (n=195)|0|
|screening_baseline|e2b|207|782.4 / 3656.2 (n=21)|621.9 / 1059.5 (n=204)|192.2 / 434.2 (n=207)|0|
|idle_warmup|e4b|187|1334.8 / 4768.0 (n=21)|626.7 / 1378.3 (n=184)|226.9 / 845.1 (n=187)|0|
|idle_warmup|e2b|206|784.4 / 3657.7 (n=21)|621.8 / 1066.8 (n=203)|198.8 / 322.2 (n=206)|0|
|final_aware_partials|e4b|173|1986.4 / 3830.2 (n=21)|628.1 / 1801.5 (n=170)|213.4 / 651.4 (n=173)|0|
|final_aware_partials|e2b|190|1352.2 / 3678.9 (n=21)|625.7 / 1252.1 (n=187)|202.8 / 477.0 (n=190)|0|
|silence_04|e4b|205|877.1 / 4592.6 (n=24)|622.7 / 1096.8 (n=202)|206.4 / 476.9 (n=205)|0|
|silence_04|e2b|209|779.4 / 4487.1 (n=24)|620.9 / 1010.9 (n=206)|193.5 / 335.2 (n=209)|0|
|silence_035|e4b|201|1293.2 / 4645.9 (n=22)|622.8 / 1169.5 (n=198)|208.1 / 582.2 (n=201)|2|
|silence_035|e2b|210|773.6 / 4481.9 (n=24)|621.7 / 976.7 (n=207)|193.8 / 330.1 (n=210)|0|
|conservative_marian|e4b|196|868.8 / 4166.5 (n=21)|628.0 / 1277.5 (n=193)|215.5 / 710.2 (n=196)|0|
|conservative_marian|e2b|201|974.0 / 4043.9 (n=21)|621.8 / 1221.0 (n=198)|206.1 / 545.7 (n=201)|0|
|terminology|e4b|192|1396.4 / 4466.6 (n=21)|620.8 / 1374.7 (n=189)|211.8 / 660.5 (n=192)|0|
|terminology|e2b|206|787.2 / 3667.6 (n=21)|622.7 / 1069.5 (n=203)|200.3 / 455.8 (n=206)|0|
|onnx_vad|e4b|199|883.3 / 3913.4 (n=21)|624.5 / 1245.5 (n=196)|214.1 / 606.8 (n=199)|0|
|onnx_vad|e2b|207|783.5 / 3645.0 (n=21)|620.6 / 1059.2 (n=204)|194.9 / 462.6 (n=207)|0|

## Matched later finals for lower silence

Same model and repeat, same endpoint_reason and speech_end_sample, identical lowercased/whitespace-normalized English, audio sample interval intersection/union >=0.90. Matches must be unique; changed text/segmentation excluded.

Each lower-silence run adds 'Yeah.' → 'Sí.' through Marian and changes the first smart-cut transcript. Those two variant captions are excluded from the matched comparison. The raw silence median therefore has a different content and route mix.

|Config|Model|Endpoint|Matched baseline p50/p95|Variant p50/p95|Paired delta p50/p95|
|---|---|---|---|---|---|
|silence_04|e4b|all|2344.4 / 4028.9 (n=18)|2176.8 / 3990.6 (n=18)|-41.1 / 671.2 (n=18)|
|silence_04|e4b|silence|2002.9 / 2823.4 (n=6)|2141.4 / 2385.0 (n=6)|-53.1 / 671.2 (n=6)|
|silence_04|e4b|smart_cut|2754.2 / 4028.9 (n=9)|2751.2 / 3990.6 (n=9)|-44.0 / 273.2 (n=9)|
|silence_04|e4b|hard_cut|2506.1 / 2935.8 (n=3)|2073.6 / 3043.3 (n=3)|107.5 / 133.1 (n=3)|
|silence_04|e2b|all|1516.0 / 3763.3 (n=18)|1488.5 / 3688.3 (n=18)|-6.6 / 86.0 (n=18)|
|silence_04|e2b|silence|1516.0 / 1702.1 (n=6)|1468.2 / 1674.3 (n=6)|-72.3 / 32.7 (n=6)|
|silence_04|e2b|smart_cut|2419.1 / 3763.3 (n=9)|2433.3 / 3688.3 (n=9)|-5.6 / 17.7 (n=9)|
|silence_04|e2b|hard_cut|1242.3 / 1285.6 (n=3)|1289.1 / 1328.3 (n=3)|3.5 / 86.0 (n=3)|
|silence_035|e4b|all|2344.4 / 4028.9 (n=18)|2606.9 / 4127.7 (n=18)|-28.1 / 841.8 (n=18)|
|silence_035|e4b|silence|2002.9 / 2823.4 (n=6)|2161.3 / 3024.5 (n=6)|-35.8 / 841.8 (n=6)|
|silence_035|e4b|smart_cut|2754.2 / 4028.9 (n=9)|2750.9 / 4127.7 (n=9)|-24.0 / 98.8 (n=9)|
|silence_035|e4b|hard_cut|2506.1 / 2935.8 (n=3)|2016.0 / 2557.3 (n=3)|-490.1 / 616.8 (n=3)|
|silence_035|e2b|all|1516.0 / 3763.3 (n=18)|1463.3 / 3649.2 (n=18)|-11.8 / 93.5 (n=18)|
|silence_035|e2b|silence|1516.0 / 1702.1 (n=6)|1441.3 / 1643.7 (n=6)|-99.2 / 4.1 (n=6)|
|silence_035|e2b|smart_cut|2419.1 / 3763.3 (n=9)|2407.7 / 3649.2 (n=9)|-11.4 / 93.5 (n=9)|
|silence_035|e2b|hard_cut|1242.3 / 1285.6 (n=3)|1293.3 / 1302.6 (n=3)|49.7 / 52.6 (n=3)|

## Execution evidence

|Config|Model|Warmups requested/executed|Busy suppressed|Partial final-decode suppressed|Marian/Gemma finals|
|---|---|---|---:|---:|---|
|screening_baseline|e4b|21/12|0|0|0/21|
|screening_baseline|e2b|21/12|0|0|0/21|
|idle_warmup|e4b|135/6|129|0|0/21|
|idle_warmup|e2b|99/6|93|0|0/21|
|final_aware_partials|e4b|21/12|0|30|0/21|
|final_aware_partials|e2b|21/12|0|17|0/21|
|silence_04|e4b|30/12|0|0|3/21|
|silence_04|e2b|32/12|0|0|3/21|
|silence_035|e4b|48/15|0|0|3/21|
|silence_035|e2b|40/15|0|0|3/21|
|conservative_marian|e4b|21/12|0|0|0/21|
|conservative_marian|e2b|21/12|0|0|0/21|
|terminology|e4b|25/12|0|0|0/21|
|terminology|e2b|21/12|0|0|0/21|
|onnx_vad|e4b|21/12|0|0|0/21|
|onnx_vad|e2b|21/12|0|0|0/21|

## Interpretation limits

- **percentiles:** Raw observations pooled within model/config/endpoint only; p50 statistics.median, p95 nearest rank ceil(.95*n)-1. Small-sample p95 often equals the maximum.
- **timing:** Schema2 replay_realtime; estimated speech end to final payload readiness, not browser delivery. All runs use the same 45s English clip at 1x and 0.6s partial cadence.
- **first_partial:** Earliest speech_start_to_partial_ms per utterance_id per session; no-emission final utterances explicitly counted. Cut remainders can begin before the next partial is admitted.
- **partial_gap:** Consecutive server emission timestamps within a session; includes pauses and cut boundaries.
- **counters:** One latest session_summary per completed session; zeros preserved and repeated snapshots never summed.
- **matched_comparison:** Same model and repeat, same endpoint_reason and speech_end_sample, identical lowercased/whitespace-normalized English, audio sample interval intersection/union >=0.90. Matches must be unique; changed text/segmentation excluded.
- **provenance:** Pipeline SHA observed at startup; other recorded core source hashes and package versions identical. This does not prove imported bytecode, eliminate temporal/system variance, or make sequential configuration runs randomized trials.

Full per-endpoint component metrics, all 20 counters, per-run memory/model provenance, and raw-file hashes are in screen-analysis.json. The recommendation below uses these complete results.

## Recommendation

Do not run an additional combination from this screen; retain every experiment as opt-in and keep E4B/0.5s silence/0.6s cadence defaults.

- Idle-only warmup improves E2B all-final median 1670.2→1495.1ms, but E4B p95 worsens 4059.6→5228.6ms. The E2B tail is effectively unchanged.
- Final-aware partials execute real suppressions but delay first partials: E4B p50 1385.5→1986.4ms and E2B 782.4→1352.2ms. E4B update-gap p95 rises 1253.2→1801.5ms.
- Both lower-silence settings add a short Yeah/Sí Marian final and alter the first smart-cut transcript. On matched later silence finals, E4B median is 2002.9ms baseline versus 2141.45/2161.35ms, and E2B is 1516.0 versus 1468.2/1441.3ms. These are modest/inconsistent changes, not the larger raw-median gain.
- Conservative routing sends zero finals to Marian on this clip, matching baseline route decisions; its timing difference does not measure routing benefit. Its E4B median nevertheless shifts 2751.7→2210.4ms, illustrating variation without the intended branch executing.
- ONNX shows a promising E4B all-final median 2155.5ms versus 2751.7, but its p95 4142.8ms versus 4059.6 is slightly higher and E2B is essentially unchanged 1664.3 versus 1670.2. The unexercised control also has a large E4B median shift, so a combination is not justified by this small sequential screen.
- Terminology is a quality experiment: automatic text canaries improve in the separate quality report, while this replay does not establish a latency win or human meaning quality.
- No browser clients were observed in these runs: final ACK coverage 0/348. This is server readiness, not a caption-delivery acceptance result.
