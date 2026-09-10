# Mac E4B / E2B comparison

E4B remains the default. No model-selection UI has been added.

Human-reference audio coverage: {"approved_natural_utterances": {"en": 0, "es": 0}, "natural_audio_gate": false}

Natural Spanish / two-speaker / bilingual review gates remain pending until explicitly validated.

## Translation on identical input

chrF++ measures agreement with public-domain verse wording, not overall translation accuracy. References use unique book/chapter/verse alignment to KJV and RVR1909; ambiguous source text has no score. Existing model outputs are preserved when references are repaired. Canary checks require every listed term; older benchmarks checked only the first term.

| Model | Prompt | Direction | Items | References | p50 ms | p95 ms | Canaries | chrF++ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| e2b | church | en→es | 43 | 25 | 553.1 | 893.5 | 14/18 | 45.9 |
| e2b | church | es→en | 25 | 24 | 619.9 | 969.3 | 0/0 | 46.0 |
| e2b | none | en→es | 43 | 25 | 516.6 | 832.4 | 11/18 | 45.1 |
| e2b | none | es→en | 25 | 24 | 618.9 | 1067.3 | 0/0 | 46.0 |
| e4b | church | en→es | 43 | 25 | 955.1 | 1560.1 | 15/18 | 46.9 |
| e4b | church | es→en | 25 | 24 | 1083.1 | 1740.5 | 0/0 | 48.7 |
| e4b | none | en→es | 43 | 25 | 839.9 | 1442.6 | 13/18 | 44.4 |
| e4b | none | es→en | 25 | 24 | 989.1 | 1560.6 | 0/0 | 48.7 |

### All theological canaries

| Input | Model / prompt | Result | Required terms | Output |
|---|---|---|---|---|
| canary_00 | e2b / church | pass | expiación, cristo, dios | La expiación de Cristo nos reconcilia con Dios. |
| canary_01 | e2b / church | miss | santiago, fe, obras | Jacobo escribió sobre la fe y las obras. |
| canary_02 | e2b / church | pass | propiciación, pecados, sangre, cristo | La propiciación por nuestros pecados fue la sangre de Cristo. |
| canary_03 | e2b / church | pass | partimiento, pan | El partimiento del pan es un solemne recuerdo. |
| canary_04 | e2b / church | pass | pablo, corintios, resurrección | Pablo escribió a los corintios sobre la resurrección. |
| canary_05 | e2b / church | pass | justificación, fe | La justificación es por fe solamente, no por obras de la ley. |
| canary_06 | e2b / church | pass | espíritu, pecado | El Espíritu Santo convence al mundo del pecado y de la justicia. |
| canary_07 | e2b / church | pass | gracia, misericordia, cristo | La gracia y la misericordia se encuentran en la cruz de nuestro Señor Jesucristo. |
| canary_08 | e2b / church | miss | propiciatorio | Nos reunimos ante el asiento de la misericordia en oración. |
| canary_09 | e2b / church | pass | mesa, señor | La mesa del Señor está extendida para su pueblo. |
| canary_10 | e2b / church | pass | nombre | Estamos reunidos en Tu Nombre el primer día de la semana. |
| canary_11 | e2b / church | pass | poco, señor | Un poco más, y veremos al Señor. |
| canary_12 | e2b / church | pass | sangre, jesús | Nada sino la sangre de Jesús puede lavar mi pecado. |
| canary_13 | e2b / church | miss | fiador, pacto | Cristo es la garantía de un mejor pacto. |
| canary_14 | e2b / church | pass | memoria, mesa | Esto en memoria de Mí en Su mesa. |
| canary_15 | e2b / church | miss | propiciatorio, propiciación | En el asiento de misericordia recordamos su propiciación por nuestros pecados. |
| canary_16 | e2b / church | pass | lugar, evangelio | Pero hay lugar en la cena del evangelio para el alma cansada. |
| canary_17 | e2b / church | pass | peregrin, calvario | Venimos como peregrinos a Calvario y a la cruz. |
| canary_00 | e2b / none | pass | expiación, cristo, dios | La expiación de Cristo nos reconcilia con Dios. |
| canary_01 | e2b / none | miss | santiago, fe, obras | James escribió sobre la fe y las obras. |
| canary_02 | e2b / none | pass | propiciación, pecados, sangre, cristo | La propiciación por nuestros pecados fue la sangre de Cristo. |
| canary_03 | e2b / none | miss | partimiento, pan | Romper el pan es un solemne recuerdo. |
| canary_04 | e2b / none | pass | pablo, corintios, resurrección | Pablo escribió a los corintios sobre la resurrección. |
| canary_05 | e2b / none | pass | justificación, fe | La justificación es por fe solamente, no por obras de la ley. |
| canary_06 | e2b / none | pass | espíritu, pecado | El Espíritu Santo convence al mundo del pecado y de la justicia. |
| canary_07 | e2b / none | pass | gracia, misericordia, cristo | La gracia y la misericordia se encuentran en la cruz de nuestro Señor Jesucristo. |
| canary_08 | e2b / none | miss | propiciatorio | Nos reunimos ante el asiento de la misericordia en oración. |
| canary_09 | e2b / none | pass | mesa, señor | La mesa del Señor está extendida para su pueblo. |
| canary_10 | e2b / none | pass | nombre | Nos reunimos en Tu Nombre el primer día de la semana. |
| canary_11 | e2b / none | pass | poco, señor | Un poco más, y veremos al Señor. |
| canary_12 | e2b / none | pass | sangre, jesús | Nada sino la sangre de Jesús puede lavar mi pecado. |
| canary_13 | e2b / none | miss | fiador, pacto | Cristo es la garantía de un mejor pacto. |
| canary_14 | e2b / none | miss | memoria, mesa | Esto es en recuerdo de Mí en Su mesa. |
| canary_15 | e2b / none | miss | propiciatorio, propiciación | En el asiento de la misericordia recordamos su propiciación por nuestros pecados. |
| canary_16 | e2b / none | miss | lugar, evangelio | Sin embargo, hay espacio en el banquete del evangelio para el alma cansada. |
| canary_17 | e2b / none | pass | peregrin, calvario | Venimos como peregrinos a Calvario y a la cruz. |
| canary_00 | e4b / church | pass | expiación, cristo, dios | La expiación de Cristo nos reconcilia con Dios. |
| canary_01 | e4b / church | pass | santiago, fe, obras | Santiago escribió sobre fe y obras. |
| canary_02 | e4b / church | pass | propiciación, pecados, sangre, cristo | La propiciación por nuestros pecados fue la sangre de Cristo. |
| canary_03 | e4b / church | pass | partimiento, pan | El partimiento del pan es un solemne recuerdo. |
| canary_04 | e4b / church | pass | pablo, corintios, resurrección | Pablo escribió a los corintios sobre la resurrección. |
| canary_05 | e4b / church | pass | justificación, fe | La justificación es por la fe solamente, no por las obras de la ley. |
| canary_06 | e4b / church | pass | espíritu, pecado | El Espíritu Santo convence al mundo de pecado y de justicia. |
| canary_07 | e4b / church | pass | gracia, misericordia, cristo | La gracia y la misericordia se encuentran en la cruz de nuestro Señor Jesucristo. |
| canary_08 | e4b / church | miss | propiciatorio | Nos reunimos ante el trono de la misericordia en oración. |
| canary_09 | e4b / church | pass | mesa, señor | La mesa del Señor está extendida para Su pueblo. |
| canary_10 | e4b / church | pass | nombre | Nos reunimos en Tu Nombre el primer día de la semana. |
| canary_11 | e4b / church | pass | poco, señor | En poco tiempo, veremos al Señor. |
| canary_12 | e4b / church | pass | sangre, jesús | Nada sino la sangre de Jesús puede lavar mi pecado. |
| canary_13 | e4b / church | miss | fiador, pacto | Cristo es la garantía de un pacto mejor. |
| canary_14 | e4b / church | pass | memoria, mesa | Haced esto en memoria Mía en su mesa. |
| canary_15 | e4b / church | miss | propiciatorio, propiciación | En el lugar de misericordia recordamos Su propiciación por nuestros pecados. |
| canary_16 | e4b / church | pass | lugar, evangelio | Aun así hay lugar en el festín del evangelio para el alma cansada. |
| canary_17 | e4b / church | pass | peregrin, calvario | Venimos como peregrinos a Calvario y a la cruz. |
| canary_00 | e4b / none | pass | expiación, cristo, dios | La expiación de Cristo nos reconcilia con Dios. |
| canary_01 | e4b / none | miss | santiago, fe, obras | James escribió sobre fe y obras. |
| canary_02 | e4b / none | pass | propiciación, pecados, sangre, cristo | La propiciación por nuestros pecados fue la sangre de Cristo. |
| canary_03 | e4b / none | miss | partimiento, pan | El partir del pan es un solemne recuerdo. |
| canary_04 | e4b / none | pass | pablo, corintios, resurrección | Pablo escribió a los corintios sobre la resurrección. |
| canary_05 | e4b / none | pass | justificación, fe | La justificación es por la fe solamente, no por las obras de la ley. |
| canary_06 | e4b / none | pass | espíritu, pecado | El Espíritu Santo convence al mundo de pecado y de justicia. |
| canary_07 | e4b / none | pass | gracia, misericordia, cristo | La gracia y la misericordia se encuentran en la cruz de nuestro Señor Jesucristo. |
| canary_08 | e4b / none | miss | propiciatorio | Nos reunimos ante el trono de la misericordia en oración. |
| canary_09 | e4b / none | pass | mesa, señor | La mesa del Señor está extendida para Su pueblo. |
| canary_10 | e4b / none | pass | nombre | Nos reunimos en Tu Nombre el primer día de la semana. |
| canary_11 | e4b / none | pass | poco, señor | En poco tiempo, veremos al Señor. |
| canary_12 | e4b / none | pass | sangre, jesús | Nada sino la sangre de Jesús puede lavar mi pecado. |
| canary_13 | e4b / none | miss | fiador, pacto | Cristo es la garantía de un pacto mejor. |
| canary_14 | e4b / none | pass | memoria, mesa | Esto se hace en memoria de Mí en su mesa. |
| canary_15 | e4b / none | miss | propiciatorio, propiciación | En el asiento de la misericordia recordamos Su propiciación por nuestros pecados. |
| canary_16 | e4b / none | pass | lugar, evangelio | Sin embargo, hay lugar en el banquete del evangelio para el alma cansada. |
| canary_17 | e4b / none | pass | peregrin, calvario | Venimos como peregrinos a Calvario y a la cruz. |

## E2B tradeoff relative to E4B

| Prompt | Direction | Median latency reduction | Canary pass difference | chrF++ difference |
|---|---|---:|---:|---:|
| church | en→es | 42.1% | -1 | -1.0 |
| church | es→en | 42.8% | +0 | -2.6 |
| none | en→es | 38.5% | -2 | 0.7 |
| none | es→en | 37.4% | +0 | -2.6 |

## STT on saved session audio

Unreviewed machine transcripts are not WER references. Unconfirmed session recordings cannot pass the natural-speech gate.

| Engine | Language | Audio items | Approved references | WER | p50 ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|
| mlx | en | 50 | 0 | — | 568.5 | 639.2 |
| mlx | es | 11 | 0 | — | 557.8 | 4311.6 |
| parakeet-mlx | en | 50 | 0 | — | 132.2 | 179.8 |
| parakeet-mlx | es | 11 | 0 | — | 131.1 | 212.9 |

## Real-time server latency

Speech end is estimated from captured VAD-positive frames. Server-final latency ends at payload readiness; it is not browser display latency. Clips, inference code/config cohorts, source types and endpoints are kept separate. Cohort IDs bind recorded source hashes, package versions and effective settings. A matching lifecycle source hash observed at startup takes precedence over later pipeline-file snapshots; this does not prove imported engine bytecode, and older snapshot timing may be ambiguous.

| Experiment | Model | Clip / cohort | Language/source | Endpoint | n | p50 ms | p95 ms |
|---|---|---|---|---|---:|---:|---:|
| baseline | e2b | replay_0 / 3526aa21834d827a | en/church_replay | smart_cut | 16 | 3248.0 | 6948.4 |
| baseline | e2b | replay_0 / 3526aa21834d827a | en/church_replay | hard_cut | 2 | 1241.1 | 1254.7 |
| baseline | e2b | replay_0 / 3526aa21834d827a | en/church_replay | silence | 8 | 1612.6 | 2031.4 |
| baseline | e2b | replay_1 / 3526aa21834d827a | en/church_replay | silence | 20 | 1275.9 | 1549.9 |
| baseline | e2b | replay_1 / 3526aa21834d827a | en/church_replay | smart_cut | 10 | 3096.4 | 6127.7 |
| baseline | e2b | replay_2 / 48eb81606c27942b | es/synthetic_piper | silence | 1 | 2516.2 | 2516.2 |
| baseline | e2b | replay_0 / 9afe129c566152c8 | en/church_replay | smart_cut | 16 | 3251.8 | 6951.3 |
| baseline | e2b | replay_0 / 9afe129c566152c8 | en/church_replay | hard_cut | 2 | 1286.8 | 1341.8 |
| baseline | e2b | replay_0 / 9afe129c566152c8 | en/church_replay | silence | 8 | 1516.4 | 1897.0 |
| baseline | e2b | replay_1 / 53238c11744c545c | en/church_replay | silence | 20 | 2185.1 | 4564.7 |
| baseline | e2b | replay_1 / 53238c11744c545c | en/church_replay | smart_cut | 10 | 4396.6 | 6485.7 |
| baseline | e2b | replay_2 / 1e9abe1552c4f44a | es/synthetic_piper | silence | 1 | 2994.9 | 2994.9 |
| baseline | e2b | replay_0 / 64764cec49138bed | en/church_replay | smart_cut | 16 | 3472.9 | 6963.9 |
| baseline | e2b | replay_0 / 64764cec49138bed | en/church_replay | hard_cut | 2 | 1661.2 | 1979.5 |
| baseline | e2b | replay_0 / 64764cec49138bed | en/church_replay | silence | 8 | 1550.9 | 2599.7 |
| baseline | e2b | replay_1 / 0ff78eb5c4ab3218 | en/church_replay | silence | 20 | 1360.4 | 2014.2 |
| baseline | e2b | replay_1 / 0ff78eb5c4ab3218 | en/church_replay | smart_cut | 10 | 3186.6 | 6119.9 |
| baseline | e2b | replay_2 / cd7d5d2f3546ada5 | es/synthetic_piper | silence | 1 | 2702.9 | 2702.9 |
| baseline | e4b | replay_0 / 3526aa21834d827a | en/church_replay | smart_cut | 16 | 3812.8 | 6955.1 |
| baseline | e4b | replay_0 / 3526aa21834d827a | en/church_replay | hard_cut | 2 | 2856.3 | 3069.3 |
| baseline | e4b | replay_0 / 3526aa21834d827a | en/church_replay | silence | 8 | 2009.7 | 2700.4 |
| baseline | e4b | replay_1 / 3526aa21834d827a | en/church_replay | silence | 20 | 1969.2 | 2864.8 |
| baseline | e4b | replay_1 / 3526aa21834d827a | en/church_replay | smart_cut | 10 | 4622.5 | 7035.7 |
| baseline | e4b | replay_2 / 48eb81606c27942b | es/synthetic_piper | silence | 1 | 3594.8 | 3594.8 |
| baseline | e4b | replay_0 / 54e9183bfd940250 | en/church_replay | smart_cut | 16 | 4360.1 | 6954.1 |
| baseline | e4b | replay_0 / 54e9183bfd940250 | en/church_replay | hard_cut | 2 | 2543.9 | 2918.8 |
| baseline | e4b | replay_0 / 54e9183bfd940250 | en/church_replay | silence | 8 | 2333.4 | 3603.2 |
| baseline | e4b | replay_1 / dc50640c05c14f19 | en/church_replay | silence | 20 | 3469.5 | 16820.9 |
| baseline | e4b | replay_1 / dc50640c05c14f19 | en/church_replay | smart_cut | 10 | 6687.5 | 19766.1 |
| baseline | e4b | replay_2 / fab17c9d480b8567 | es/synthetic_piper | silence | 1 | 6303.6 | 6303.6 |
| baseline | e4b | replay_0 / 64764cec49138bed | en/church_replay | smart_cut | 16 | 5169.2 | 17250.6 |
| baseline | e4b | replay_0 / 64764cec49138bed | en/church_replay | hard_cut | 2 | 3464.2 | 4613.5 |
| baseline | e4b | replay_0 / 64764cec49138bed | en/church_replay | silence | 8 | 2803.7 | 16191.5 |
| baseline | e4b | replay_1 / 64764cec49138bed | en/church_replay | silence | 20 | 1894.0 | 2592.4 |
| baseline | e4b | replay_1 / 64764cec49138bed | en/church_replay | smart_cut | 10 | 4311.7 | 6414.9 |
| baseline | e4b | replay_2 / cd7d5d2f3546ada5 | es/synthetic_piper | silence | 1 | 3750.8 | 3750.8 |

## Partial delivery and visible browser timing

First-partial delay starts at the first captured speech frame; one earliest delay is counted per known utterance. Update gaps use chronological emissions, including speaking pauses. These partial timings end at server readiness, before browser rendering.

| Experiment | Model | Clip / cohort | Source | Metric | n | p50 ms | p95 ms |
|---|---|---|---|---|---:|---:|---:|
| baseline | e2b | replay_0 / 3526aa21834d827a | en/church_replay | captured_end_to_partial_ms | 221 | 226.2 | 648.4 |
| baseline | e2b | replay_0 / 3526aa21834d827a | en/church_replay | partial_update_gap_ms | 220 | 623.0 | 1220.2 |
| baseline | e2b | replay_0 / 3526aa21834d827a | en/church_replay | first_partial_ms | 26 | 1422.1 | 4997.4 |
| baseline | e2b | replay_1 / 3526aa21834d827a | en/church_replay | captured_end_to_partial_ms | 205 | 171.3 | 297.2 |
| baseline | e2b | replay_1 / 3526aa21834d827a | en/church_replay | partial_update_gap_ms | 204 | 619.9 | 1598.8 |
| baseline | e2b | replay_1 / 3526aa21834d827a | en/church_replay | first_partial_ms | 32 | 747.3 | 5215.9 |
| baseline | e2b | replay_2 / 48eb81606c27942b | es/synthetic_piper | captured_end_to_partial_ms | 10 | 649.8 | 1235.1 |
| baseline | e2b | replay_2 / 48eb81606c27942b | es/synthetic_piper | partial_update_gap_ms | 9 | 641.5 | 1039.6 |
| baseline | e2b | replay_2 / 48eb81606c27942b | es/synthetic_piper | first_partial_ms | 1 | 1187.4 | 1187.4 |
| baseline | e2b | replay_0 / 9afe129c566152c8 | en/church_replay | captured_end_to_partial_ms | 222 | 226.6 | 456.2 |
| baseline | e2b | replay_0 / 9afe129c566152c8 | en/church_replay | partial_update_gap_ms | 221 | 624.2 | 1189.0 |
| baseline | e2b | replay_0 / 9afe129c566152c8 | en/church_replay | first_partial_ms | 26 | 1423.3 | 4620.9 |
| baseline | e2b | replay_1 / 53238c11744c545c | en/church_replay | captured_end_to_partial_ms | 166 | 315.9 | 1403.0 |
| baseline | e2b | replay_1 / 53238c11744c545c | en/church_replay | partial_update_gap_ms | 165 | 649.4 | 2241.3 |
| baseline | e2b | replay_1 / 53238c11744c545c | en/church_replay | first_partial_ms | 31 | 1519.9 | 6058.2 |
| baseline | e2b | replay_2 / 1e9abe1552c4f44a | es/synthetic_piper | captured_end_to_partial_ms | 5 | 832.4 | 1195.1 |
| baseline | e2b | replay_2 / 1e9abe1552c4f44a | es/synthetic_piper | partial_update_gap_ms | 4 | 1267.9 | 1578.7 |
| baseline | e2b | replay_2 / 1e9abe1552c4f44a | es/synthetic_piper | first_partial_ms | 1 | 1336.6 | 1336.6 |
| baseline | e2b | replay_0 / 64764cec49138bed | en/church_replay | captured_end_to_partial_ms | 213 | 230.0 | 661.1 |
| baseline | e2b | replay_0 / 64764cec49138bed | en/church_replay | partial_update_gap_ms | 212 | 627.0 | 1268.8 |
| baseline | e2b | replay_0 / 64764cec49138bed | en/church_replay | first_partial_ms | 26 | 1455.7 | 4670.8 |
| baseline | e2b | replay_1 / 0ff78eb5c4ab3218 | en/church_replay | captured_end_to_partial_ms | 196 | 203.7 | 696.0 |
| baseline | e2b | replay_1 / 0ff78eb5c4ab3218 | en/church_replay | partial_update_gap_ms | 195 | 626.9 | 1641.4 |
| baseline | e2b | replay_1 / 0ff78eb5c4ab3218 | en/church_replay | first_partial_ms | 32 | 810.0 | 5544.1 |
| baseline | e2b | replay_2 / cd7d5d2f3546ada5 | es/synthetic_piper | captured_end_to_partial_ms | 10 | 653.0 | 1297.5 |
| baseline | e2b | replay_2 / cd7d5d2f3546ada5 | es/synthetic_piper | partial_update_gap_ms | 9 | 657.1 | 910.0 |
| baseline | e2b | replay_2 / cd7d5d2f3546ada5 | es/synthetic_piper | first_partial_ms | 1 | 1211.2 | 1211.2 |
| baseline | e4b | replay_0 / 3526aa21834d827a | en/church_replay | captured_end_to_partial_ms | 202 | 259.9 | 894.4 |
| baseline | e4b | replay_0 / 3526aa21834d827a | en/church_replay | partial_update_gap_ms | 201 | 628.8 | 1565.1 |
| baseline | e4b | replay_0 / 3526aa21834d827a | en/church_replay | first_partial_ms | 26 | 1576.0 | 5762.8 |
| baseline | e4b | replay_1 / 3526aa21834d827a | en/church_replay | captured_end_to_partial_ms | 178 | 228.4 | 1197.9 |
| baseline | e4b | replay_1 / 3526aa21834d827a | en/church_replay | partial_update_gap_ms | 177 | 631.3 | 2077.7 |
| baseline | e4b | replay_1 / 3526aa21834d827a | en/church_replay | first_partial_ms | 30 | 1026.0 | 5815.1 |
| baseline | e4b | replay_2 / 48eb81606c27942b | es/synthetic_piper | captured_end_to_partial_ms | 10 | 676.0 | 1230.0 |
| baseline | e4b | replay_2 / 48eb81606c27942b | es/synthetic_piper | partial_update_gap_ms | 9 | 676.3 | 1124.3 |
| baseline | e4b | replay_2 / 48eb81606c27942b | es/synthetic_piper | first_partial_ms | 1 | 1231.6 | 1231.6 |
| baseline | e4b | replay_0 / 54e9183bfd940250 | en/church_replay | captured_end_to_partial_ms | 189 | 276.7 | 1277.6 |
| baseline | e4b | replay_0 / 54e9183bfd940250 | en/church_replay | partial_update_gap_ms | 188 | 629.0 | 1752.6 |
| baseline | e4b | replay_0 / 54e9183bfd940250 | en/church_replay | first_partial_ms | 26 | 1732.5 | 5463.9 |
| baseline | e4b | replay_1 / dc50640c05c14f19 | en/church_replay | captured_end_to_partial_ms | 116 | 320.4 | 1853.7 |
| baseline | e4b | replay_1 / dc50640c05c14f19 | en/church_replay | partial_update_gap_ms | 115 | 644.4 | 3081.0 |
| baseline | e4b | replay_1 / dc50640c05c14f19 | en/church_replay | first_partial_ms | 23 | 1676.3 | 6519.2 |
| baseline | e4b | replay_2 / fab17c9d480b8567 | es/synthetic_piper | captured_end_to_partial_ms | 4 | 1061.5 | 1949.8 |
| baseline | e4b | replay_2 / fab17c9d480b8567 | es/synthetic_piper | partial_update_gap_ms | 3 | 1302.1 | 1595.9 |
| baseline | e4b | replay_2 / fab17c9d480b8567 | es/synthetic_piper | first_partial_ms | 1 | 2557.9 | 2557.9 |
| baseline | e4b | replay_0 / 64764cec49138bed | en/church_replay | captured_end_to_partial_ms | 155 | 350.5 | 1487.3 |
| baseline | e4b | replay_0 / 64764cec49138bed | en/church_replay | partial_update_gap_ms | 154 | 649.8 | 2192.2 |
| baseline | e4b | replay_0 / 64764cec49138bed | en/church_replay | first_partial_ms | 24 | 2467.5 | 5193.2 |
| baseline | e4b | replay_1 / 64764cec49138bed | en/church_replay | captured_end_to_partial_ms | 179 | 235.1 | 998.0 |
| baseline | e4b | replay_1 / 64764cec49138bed | en/church_replay | partial_update_gap_ms | 178 | 638.6 | 1973.4 |
| baseline | e4b | replay_1 / 64764cec49138bed | en/church_replay | first_partial_ms | 31 | 836.9 | 6099.5 |
| baseline | e4b | replay_2 / cd7d5d2f3546ada5 | es/synthetic_piper | captured_end_to_partial_ms | 6 | 990.7 | 1686.8 |
| baseline | e4b | replay_2 / cd7d5d2f3546ada5 | es/synthetic_piper | partial_update_gap_ms | 5 | 885.7 | 1663.5 |
| baseline | e4b | replay_2 / cd7d5d2f3546ada5 | es/synthetic_piper | first_partial_ms | 1 | 2177.7 | 2177.7 |

### Visible final latency by browser session and endpoint

Only visible schema-2 final ACKs matched to a unique CSV chunk contribute. Each browser session, capture timing source and endpoint has its own distribution; clients and silence/forced-cut/EOF endpoints are never pooled. Duplicate chunk acknowledgments and unmatched/stale events are excluded. Speech-end-to-ack includes return-network time. Coverage alone does not make a run acceptance eligible; the latency gate remains pending.

| Experiment | Model | Clip / cohort | Session / client | Endpoint / timing source | Metric | n | p50 ms | p95 ms | ACKs / finals |
|---|---|---|---|---|---|---:|---:|---:|---|

### Visible final acknowledgment coverage

Coverage counts each finalized chunk once when any visible client acknowledged it. Missing acknowledgments are missing evidence, not proof of display failure; replay shutdown can race the final browser acknowledgment.

| Experiment | Model | Clip / cohort | Received final chunks | Finalized chunks |
|---|---|---|---:|---:|
| baseline | e2b | replay_0 / 3526aa21834d827a | 0 | 26 |
| baseline | e2b | replay_1 / 3526aa21834d827a | 0 | 30 |
| baseline | e2b | replay_2 / 48eb81606c27942b | 0 | 1 |
| baseline | e2b | replay_0 / 9afe129c566152c8 | 0 | 26 |
| baseline | e2b | replay_1 / 53238c11744c545c | 0 | 30 |
| baseline | e2b | replay_2 / 1e9abe1552c4f44a | 0 | 1 |
| baseline | e2b | replay_0 / 64764cec49138bed | 0 | 26 |
| baseline | e2b | replay_1 / 0ff78eb5c4ab3218 | 0 | 30 |
| baseline | e2b | replay_2 / cd7d5d2f3546ada5 | 0 | 1 |
| baseline | e4b | replay_0 / 3526aa21834d827a | 0 | 26 |
| baseline | e4b | replay_1 / 3526aa21834d827a | 0 | 30 |
| baseline | e4b | replay_2 / 48eb81606c27942b | 0 | 1 |
| baseline | e4b | replay_0 / 54e9183bfd940250 | 0 | 26 |
| baseline | e4b | replay_1 / dc50640c05c14f19 | 0 | 30 |
| baseline | e4b | replay_2 / fab17c9d480b8567 | 0 | 1 |
| baseline | e4b | replay_0 / 64764cec49138bed | 0 | 26 |
| baseline | e4b | replay_1 / 64764cec49138bed | 0 | 30 |
| baseline | e4b | replay_2 / cd7d5d2f3546ada5 | 0 | 1 |

## Changed translation examples

**church_bilingual_063d72c8e57313c9_canary_01** — James wrote about faith and works.

- E4B: Santiago escribió sobre fe y obras.
- E2B: Jacobo escribió sobre la fe y las obras.

**church_bilingual_063d72c8e57313c9_canary_05** — Justification is by faith alone, not by works of the law.

- E4B: La justificación es por la fe solamente, no por las obras de la ley.
- E2B: La justificación es por fe solamente, no por obras de la ley.

**church_bilingual_063d72c8e57313c9_canary_06** — The Holy Spirit convicts the world of sin and righteousness.

- E4B: El Espíritu Santo convence al mundo de pecado y de justicia.
- E2B: El Espíritu Santo convence al mundo del pecado y de la justicia.

**church_bilingual_063d72c8e57313c9_canary_08** — We gather before the mercy seat in prayer.

- E4B: Nos reunimos ante el trono de la misericordia en oración.
- E2B: Nos reunimos ante el asiento de la misericordia en oración.

**church_bilingual_063d72c8e57313c9_canary_09** — The table of the Lord is spread for His people.

- E4B: La mesa del Señor está extendida para Su pueblo.
- E2B: La mesa del Señor está extendida para su pueblo.

**church_bilingual_063d72c8e57313c9_canary_10** — We are gathered to Thy Name on the first day of the week.

- E4B: Nos reunimos en Tu Nombre el primer día de la semana.
- E2B: Estamos reunidos en Tu Nombre el primer día de la semana.

**church_bilingual_063d72c8e57313c9_canary_11** — A little while, and we shall see the Lord.

- E4B: En poco tiempo, veremos al Señor.
- E2B: Un poco más, y veremos al Señor.

**church_bilingual_063d72c8e57313c9_canary_13** — Christ is the Surety of a better covenant.

- E4B: Cristo es la garantía de un pacto mejor.
- E2B: Cristo es la garantía de un mejor pacto.

**church_bilingual_063d72c8e57313c9_canary_14** — This do in remembrance of Me at His table.

- E4B: Haced esto en memoria Mía en su mesa.
- E2B: Esto en memoria de Mí en Su mesa.

**church_bilingual_063d72c8e57313c9_canary_15** — At the mercy seat we remember His propitiation for our sins.

- E4B: En el lugar de misericordia recordamos Su propiciación por nuestros pecados.
- E2B: En el asiento de misericordia recordamos su propiciación por nuestros pecados.

**church_bilingual_063d72c8e57313c9_canary_16** — Yet there is room at the gospel feast for the weary soul.

- E4B: Aun así hay lugar en el festín del evangelio para el alma cansada.
- E2B: Pero hay lugar en la cena del evangelio para el alma cansada.

**church_bilingual_063d72c8e57313c9_verse_00_en** — They look at themselves, then go on their way,

- E4B: Se miran a sí mismos, luego se van por su camino.
- E2B: Se miran a sí mismos, luego se van.

**church_bilingual_063d72c8e57313c9_verse_00_es** — El servidor se volvió a contar todo esto a su amo. Entonces, lleno de ira el dueño de casa, dijo a su servidor: “Sal en seguida a las calles y callejuelas de la ciudad; y tráeme aquí los pobres, y lisiados, y ciegos y cojos”.

- E4B: The servant told his master all of this again. Then, filled with anger, the master of the house said to his servant: "Go immediately out into the streets and alleys of the city; and bring me here the poor, the lame, the blind, and the crippled."
- E2B: The servant recounted all this to his master again. Then, filled with anger, the master of the house said to his servant: "Go immediately to the streets and alleys of the city; and bring me here the poor, the insane, the blind, and the lame."

**church_bilingual_063d72c8e57313c9_verse_01_en** — The trees of the Lord drink their fill the cedars he planted on Lebanon,

- E4B: Los árboles del Señor sacian su sed, los cedros que plantó en Líbano.
- E2B: Los árboles del Señor beben hasta saciarse los cedros que plantó en el Líbano,

**church_bilingual_063d72c8e57313c9_verse_01_es** — Llénanse de jugo los árboles de Jehová, los cedros del Líbano que él plantó.

- E4B: Let the trees of the Lord be full of juice, the cedars of Lebanon that he planted.
- E2B: Let the trees of the Lord be filled with juice, the cedars of Lebanon that he planted.

**church_bilingual_063d72c8e57313c9_verse_02_en** — From Kibroth-hattaavah the people journeyed unto Hazeroth; and they abode at Hazeroth.

- E4B: Desde Kibroth-hattaavah el pueblo viajó a Hazeroth; y allí permanecieron en Hazeroth.
- E2B: Desde Kibroth-hattaavah el pueblo viajó a Hazeroth; y se hospedaron en Hazeroth.

**church_bilingual_063d72c8e57313c9_verse_02_es** — No lo hago así con mi siervo Moisés,   el cual es fiel en toda mi casa.

- E4B: I do not do this with my servant Moses, who is faithful in my whole house.
- E2B: I don't do that to my servant Moses, who is faithful in my whole house.

**church_bilingual_063d72c8e57313c9_verse_03_en** — and from the Nethinim, whom David and the heads gave for the service of the Levites, two hundred and twenty Nethinim, all of them defined by name.

- E4B: y de los Nethinim, a quienes David y los jefes dieron para el servicio de los Levitas, doscientos veinte Nethinim, todos nombrados.
- E2B: y de los nehtimím, a quienes David y los jefes dieron por el servicio de los levitas, doscientos veinte nehtimím, todos definidos por nombre.

**church_bilingual_063d72c8e57313c9_verse_03_es** — Pesé y entregué en sus manos seiscientos cincuenta talentos de plata, utensilios de plata por cien talentos, cien talentos de oro,

- E4B: I weighed and delivered into your hands six hundred fifty talents of silver, silver utensils for one hundred talents, one hundred talents of gold,
- E2B: I weighed and delivered in his hands six hundred fifty talents of silver, silver utensils for one hundred talents, one hundred talents of gold,

**church_bilingual_063d72c8e57313c9_verse_04_en** — And the name of the daughter of Asher <FI>is<Fi> Sarah.

- E4B: Y el nombre de la hija de Aser es Sara.
- E2B: Y el nombre de la hija de Asher es Sara.

**church_bilingual_063d72c8e57313c9_verse_05_en** — If thou lend money to any of my people with thee that is poor, thou shalt not be to him as a creditor; neither shall ye lay upon him interest.

- E4B: Si prestas dinero a alguno de mi pueblo que esté pobre contigo, no serás para él como acreedor; ni le imponerás interés.
- E2B: Si prestas dinero a alguno de mi pueblo con el que está pobre, no serás acreedor de él; ni le pondrás interés.

**church_bilingual_063d72c8e57313c9_verse_05_es** — No blasfemarás contra Dios, ni maldecirás al príncipe de tu pueblo.

- E4B: You shall not blaspheme God, nor curse the prince of your people.
- E2B: You shall not swear blasphemies against God, nor curse the prince of your people.

**church_bilingual_063d72c8e57313c9_verse_06_en** — But I counted it necessary to send to you Epaphroditus, my brother and fellow-worker and fellow-soldier, and your messenger and minister to my need;

- E4B: Pero consideré necesario enviaros a Epafrodito, mi hermano, compañero de trabajo y compañero de armas, y vuestro mensajero y ministro para mi necesidad;
- E2B: Pero consideré necesario enviarles a Epafrodito, mi hermano y compañero de trabajo y compañero de soldado, y vuestro mensajero y ministro para mi necesidad;

**church_bilingual_063d72c8e57313c9_verse_06_es** — Él no es Dios de muertos, sino de vivos. Vosotros estáis, pues, en un gran error”.

- E4B: He is not a god of the dead, but of the living. You are, therefore, in a great error.
- E2B: He is not the God of the dead, but of the living. You are, therefore, in a great error.

**church_bilingual_063d72c8e57313c9_verse_07_en** — And this is the testimony of John, when the Jews sent out of Jerusalem priests and Levites, that they might question him, `Who art thou?'

- E4B: Y este es el testimonio de Juan, cuando los judíos enviaron de Jerusalén sacerdotes y levitas, para que le preguntaran: "¿Quién eres tú?"
- E2B: Y este es el testimonio de Juan, cuando los judíos enviaron sacerdotes y levitas de Jerusalén para que le cuestionaran, "¿Quién eres?"

**church_bilingual_063d72c8e57313c9_verse_07_es** — Y éste es el testimonio de Juan, cuando los Judíos enviaron de Jerusalem sacerdotes y Levitas, que le preguntasen: ¿Tú, quién eres?

- E4B: And this is the testimony of John, when the Jews sent priests and Levites from Jerusalem to ask him: Who are you?
- E2B: And this is the testimony of John, when the Jews sent priests and Levites from Jerusalem to ask him: "Who are you?"

**church_bilingual_063d72c8e57313c9_verse_08_en** — And thou shalt make holy garments for Aaron thy brother, for glory and for beauty.

- E4B: Y harás vestiduras santas para Aarón tu hermano, por gloria y por hermosura.
- E2B: Y harás vestiduras santas para Aarón tu hermano, para gloria y para belleza.

**church_bilingual_063d72c8e57313c9_verse_08_es** — Tomarán para ello oro, jacinto, púrpura escarlata y carmesí y tejido de lino fino.”

- E4B: They will take for it gold, sapphire, scarlet purple, and fine linen cloth.
- E2B: They will take gold, jasmine, purple scarlet and crimson and fine linen fabric for it.

**church_bilingual_063d72c8e57313c9_verse_09_en** — and they said “I will come quickly.”Blessed will he be who lays to heart the words of the prophecy contained in this book.’

- E4B: Y dijeron: "Vendré pronto". Bienaventurado el que atesore las palabras de la profecía contenidas en este libro.
- E2B: y dijeron “vendré pronto”.Bienaventurado será aquel que medite las palabras de la profecía contenida en este libro.

**church_bilingual_063d72c8e57313c9_verse_09_es** — ¿No ha dicho la Escritura que el Cristo ha de venir del linaje de David, y de Belén, la aldea de David?”

- E4B: Has the Scripture not said that the Christ is to come from the lineage of David, and from Bethlehem, the town of David?
- E2B: Hasn't the Scripture said that Christ must come from the lineage of David, and from Bethlehem, the village of David?

**church_bilingual_063d72c8e57313c9_verse_10_en** — and again it saith, `Rejoice ye nations, with His people;'

- E4B: y de nuevo dice, «Regocijaos, naciones, con Su pueblo;»
- E2B: y de nuevo dice, "Regocíate, naciones, con su pueblo".

**church_bilingual_063d72c8e57313c9_verse_10_es** — ¿Y sus hermanas no están todas entre nosotros? Entonces, ¿de dónde le viene todo esto?”

- E4B: And aren't all your sisters among us? So, where does all this come from?
- E2B: And aren't his sisters all among us? Then, where does all this come from?

**church_bilingual_063d72c8e57313c9_verse_11_en** — So two or three cities wandered unto one city, to drink water; but they were not satisfied: yet have ye not returned unto me, saith the Lord.

- E4B: Así dos o tres ciudades vagaron a una ciudad, para beber agua; pero no se saciaron: sin embargo, no habéis vuelto a mí, dice el Señor.
- E2B: Así que dos o tres ciudades vagaron hacia una ciudad para beber agua; pero no se saciaron: pero no habéis regresado a mí, dice el Señor.

**church_bilingual_063d72c8e57313c9_verse_11_es** — Al principio del reinado de Joakim, hijo de Josías, rey de Judá, habló Yahvé en estos términos:

- E4B: At the beginning of the reign of Joakim, son of Josiah, king of Judah, Yahweh spoke in these terms:
- E2B: At the beginning of Joakim's reign, son of Josiah, king of Judah, Yahweh spoke in these terms:

**church_bilingual_063d72c8e57313c9_verse_12_en** — And the God of peace shall bruise Satan under your feet shortly. The grace of our Lord Jesus Christ be with you. Amen.

- E4B: Y el Dios de paz aplastará a Satanás bajo vuestros pies pronto. La gracia de nuestro Señor Jesucristo sea con vosotros. Amén.
- E2B: Y el Dios de paz quebrantará a Satanás bajo vuestros pies pronto. La gracia de nuestro Señor Jesucristo esté con vosotros. Amén.

**church_bilingual_063d72c8e57313c9_verse_12_es** — Vosotros, al contrario, decís: “Cualquiera que diga a su padre o a su madre: “Es ofrenda (para el Templo) aquello con lo cual yo te podría haber socorrido,

- E4B: You, on the contrary, say: "Whoever says to his father or his mother: 'It is an offering (for the Temple) of what I could have helped you with,'
- E2B: You all, on the contrary, say: "Anyone who tells his father or mother: 'It is an offering (for the Temple) that with which I could have helped you'"

**church_bilingual_063d72c8e57313c9_verse_13_es** — Recibid mi enseñanza, y no plata; y ciencia antes que el oro escogido.

- E4B: Receive my teaching, and not silver; and knowledge before chosen gold.
- E2B: Receive my teaching, and not money; and knowledge before the gold chosen.

**church_bilingual_063d72c8e57313c9_verse_14_es** — “La Pascua, como sabéis, será dentro de dos días, y el Hijo del hombre va a ser entregado para que lo crucifiquen”.

- E4B: "Easter, as you know, will be in two days, and the Son of Man is going to be delivered to be crucified."
- E2B: Easter, as you know, will be in two days, and the Son of man is going to be handed over to be crucified.

**church_bilingual_063d72c8e57313c9_verse_15_en** — And by the desire of all the people, the feast went on for another seven days, and they kept the seven days with joy.

- E4B: Y por el deseo de todo el pueblo, la fiesta continuó por siete días más, y celebraron los siete días con gozo.
- E2B: Y por el deseo de todo el pueblo, la fiesta continuó por otros siete días, y guardaron los siete días con gozo.

**church_bilingual_063d72c8e57313c9_verse_15_es** — Y toda aquella multitud determinó que celebrasen otros siete días; y celebraron otros siete días con alegría.

- E4B: And all that multitude determined that they celebrate for seven more days; and they celebrated for seven more days with joy.
- E2B: And all that crowd decided to celebrate seven more days; and they celebrated seven more days with joy.

**church_bilingual_063d72c8e57313c9_verse_16_en** — `Remember that which Amalek hath done to thee in the way, in your going out from Egypt,

- E4B: Recuerda lo que Amalec te hizo en el camino, al salir de Egipto.
- E2B: Acuerda lo que Amalec ha hecho contigo en el camino, cuando salisteis de Egipto,

**church_bilingual_063d72c8e57313c9_verse_16_es** — El sacerdote recibirá el canasto de tu mano y lo pondrá delante del altar de Yahvé, tu Dios.

- E4B: The priest will receive the basket from your hand and place it before the altar of Yahweh, your God.
- E2B: The priest will receive the basket from your hand and place it in front of the altar of Yahweh, your God.

**church_bilingual_063d72c8e57313c9_verse_17_es** — Que edificáis á Sión con sangre, y á Jerusalem con injusticia;

- E4B: That you build Zion with blood, and Jerusalem with injustice;
- E2B: What are you building in Zion with blood, and in Jerusalem with injustice;

**church_bilingual_063d72c8e57313c9_verse_18_en** — He took them and sent them over the stream with all he had.

- E4B: Los tomó y los envió al otro lado del arroyo con todo lo que tenía.
- E2B: Los tomó y los envió sobre el arroyo con todo lo que tenía.

**church_bilingual_063d72c8e57313c9_verse_18_es** — Tomólos pues, y pasólos el arroyo, é hizo pasar lo que tenía.

- E4B: So he took them, and he passed them across the stream, and he made pass what he had.
- E2B: He took them, and passed them through the stream, he made pass what he had.

**church_bilingual_063d72c8e57313c9_verse_19_en** — And the Lord your God will give to you and to your seed a circumcision of the heart, so that, loving him with all your heart and all your soul, you may have life.

- E4B: Y el Señor tu Dios te dará a ti y a tu descendencia una circuncisión del corazón, para que, amándolo con todo tu corazón y con toda tu alma, tengas vida.
- E2B: Y el Señor tu Dios te dará a ti y a tu descendencia una circuncisión del corazón, para que, amándolo con todo tu corazón y toda tu alma, tengas vida.

**church_bilingual_063d72c8e57313c9_verse_19_es** — No está en el cielo, de suerte que puedas decir: ‘¿Quién subirá por nosotros al cielo para que nos la traiga y nos la enseñe, y nosotros la pongamos por obra?’

- E4B: It is not in heaven, luckily you can say: 'Who will go up for us to heaven to bring it to us and show it to us, so that we may put it into practice?'
- E2B: It is not in heaven, so you can say: 'Who will go up with us to heaven to bring it to us and show it to us, and we will put it into practice?'

**church_bilingual_063d72c8e57313c9_verse_20_es** — Entonces él se levantó y se fué á su casa.

- E4B: So he got up and went to his house.
- E2B: Then he got up and went to his house.

**church_bilingual_063d72c8e57313c9_verse_21_es** — Entonces él se levantó y se fué á su casa.

- E4B: So he got up and went to his house.
- E2B: Then he got up and went to his house.

**church_bilingual_063d72c8e57313c9_verse_22_en** — And he stayeth yet other seven days, and addeth to send forth the dove from the ark;

- E4B: Y se queda siete días más, y añade para enviar la paloma del arca;
- E2B: Y permanece otros siete días, y añade a enviar la paloma del arca;

**church_bilingual_063d72c8e57313c9_verse_22_es** — Y esperó aún otros siete días, y volvió á enviar la paloma fuera del arca.

- E4B: And he waited another seven days, and sent the dove out of the ark again.
- E2B: And he waited another seven days, and returned to send the dove out of the ark.

**church_bilingual_063d72c8e57313c9_verse_23_en** — And in the evening, at sundown, they took to him all who were diseased, and those who had evil spirits.

- E4B: Y por la tarde, al atardecer, le trajeron todos los enfermos y los que tenían espíritus malignos.
- E2B: Y en la tarde, al anochecer, le llevaron a él a todos los que estaban enfermos y a los que tenían espíritus malignos.

**church_bilingual_063d72c8e57313c9_verse_23_es** — Por tanto, así dice, Yahvé, el Señor: He aquí que haré venir sobre ti la espada, y exterminaré en ti hombres y bestias.

- E4B: Therefore, thus says Yahweh, the Lord: Behold, I will bring the sword upon you, and I will exterminate men and beasts in you.
- E2B: Therefore, so says Yahweh, the Lord: Behold, I will bring the sword upon you, and I will exterminate men and beasts in you.

**church_bilingual_063d72c8e57313c9_verse_24_en** — Thou hast turned my mourning to dancing for me, Thou hast loosed my sackcloth, And girdest me <FI>with<Fi> joy.

- E4B: Has cambiado mi luto por danza para mí, has quitado mi cilicio, y me vistes de gozo.
- E2B: Has convertido mi luto en danza por mí, has desatado mi saco y me has atado con gozo.

**church_bilingual_063d72c8e57313c9_verse_24_es** — Has tornado mi endecha en baile; desataste mi saco, y ceñísteme de alegría.

- E4B: You have tornado my right hand in dance; you have unleashed my sack, and girded me with joy.
- E2B: You have torn my edge in dance; you unleashed my sack, and tightened me with joy.

**none_bilingual_063d72c8e57313c9_canary_01** — James wrote about faith and works.

- E4B: James escribió sobre fe y obras.
- E2B: James escribió sobre la fe y las obras.

**none_bilingual_063d72c8e57313c9_canary_03** — The breaking of bread is a solemn remembrance.

- E4B: El partir del pan es un solemne recuerdo.
- E2B: Romper el pan es un solemne recuerdo.

**none_bilingual_063d72c8e57313c9_canary_05** — Justification is by faith alone, not by works of the law.

- E4B: La justificación es por la fe solamente, no por las obras de la ley.
- E2B: La justificación es por fe solamente, no por obras de la ley.

**none_bilingual_063d72c8e57313c9_canary_06** — The Holy Spirit convicts the world of sin and righteousness.

- E4B: El Espíritu Santo convence al mundo de pecado y de justicia.
- E2B: El Espíritu Santo convence al mundo del pecado y de la justicia.

**none_bilingual_063d72c8e57313c9_canary_08** — We gather before the mercy seat in prayer.

- E4B: Nos reunimos ante el trono de la misericordia en oración.
- E2B: Nos reunimos ante el asiento de la misericordia en oración.

**none_bilingual_063d72c8e57313c9_canary_09** — The table of the Lord is spread for His people.

- E4B: La mesa del Señor está extendida para Su pueblo.
- E2B: La mesa del Señor está extendida para su pueblo.

**none_bilingual_063d72c8e57313c9_canary_11** — A little while, and we shall see the Lord.

- E4B: En poco tiempo, veremos al Señor.
- E2B: Un poco más, y veremos al Señor.

**none_bilingual_063d72c8e57313c9_canary_13** — Christ is the Surety of a better covenant.

- E4B: Cristo es la garantía de un pacto mejor.
- E2B: Cristo es la garantía de un mejor pacto.

**none_bilingual_063d72c8e57313c9_canary_14** — This do in remembrance of Me at His table.

- E4B: Esto se hace en memoria de Mí en su mesa.
- E2B: Esto es en recuerdo de Mí en Su mesa.

**none_bilingual_063d72c8e57313c9_canary_15** — At the mercy seat we remember His propitiation for our sins.

- E4B: En el asiento de la misericordia recordamos Su propiciación por nuestros pecados.
- E2B: En el asiento de la misericordia recordamos su propiciación por nuestros pecados.

**none_bilingual_063d72c8e57313c9_canary_16** — Yet there is room at the gospel feast for the weary soul.

- E4B: Sin embargo, hay lugar en el banquete del evangelio para el alma cansada.
- E2B: Sin embargo, hay espacio en el banquete del evangelio para el alma cansada.

**none_bilingual_063d72c8e57313c9_verse_00_en** — They look at themselves, then go on their way,

- E4B: Se miran, luego se van de camino,
- E2B: Se miran a sí mismos, luego se van.

**none_bilingual_063d72c8e57313c9_verse_00_es** — El servidor se volvió a contar todo esto a su amo. Entonces, lleno de ira el dueño de casa, dijo a su servidor: “Sal en seguida a las calles y callejuelas de la ciudad; y tráeme aquí los pobres, y lisiados, y ciegos y cojos”.

- E4B: The servant told his master all of this again. Then, filled with anger, the master of the house said to his servant: "Go immediately out into the streets and alleys of the city; and bring me here the poor, the lame, the blind, and the crippled."
- E2B: The servant recounted all this to his master again. Then, filled with anger, the master of the house said to his servant: "Go immediately to the streets and alleys of the city; and bring me here the poor, the insane, the blind, and the lame."

**none_bilingual_063d72c8e57313c9_verse_01_en** — The trees of the Lord drink their fill the cedars he planted on Lebanon,

- E4B: Los árboles del Señor sacian su sed, los cedros que plantó en Líbano.
- E2B: Los árboles del Señor beben hasta saciarse los cedros que plantó en el Líbano.

**none_bilingual_063d72c8e57313c9_verse_01_es** — Llénanse de jugo los árboles de Jehová, los cedros del Líbano que él plantó.

- E4B: Let the trees of the Lord be full of juice, the cedars of Lebanon that he planted.
- E2B: Let the trees of the Lord be filled with juice, the cedars of Lebanon that he planted.

**none_bilingual_063d72c8e57313c9_verse_02_en** — From Kibroth-hattaavah the people journeyed unto Hazeroth; and they abode at Hazeroth.

- E4B: Desde Kibroth-hattaavah el pueblo viajó a Hazeroth; y allí residieron en Hazeroth.
- E2B: Desde Kibroth-hattaavah el pueblo viajó a Hazeroth; y se alojaron en Hazeroth.

**none_bilingual_063d72c8e57313c9_verse_02_es** — No lo hago así con mi siervo Moisés,   el cual es fiel en toda mi casa.

- E4B: I do not do this with my servant Moses, who is faithful in my whole house.
- E2B: I don't do that to my servant Moses, who is faithful in my whole house.

**none_bilingual_063d72c8e57313c9_verse_03_en** — and from the Nethinim, whom David and the heads gave for the service of the Levites, two hundred and twenty Nethinim, all of them defined by name.

- E4B: y de los Nethinim, a quienes David y los jefes dieron para el servicio de los levitas, doscientos veinte Nethinim, todos nombrados.
- E2B: y de los netinim, a quienes David y los jefes dieron por el servicio de los levitas, doscientos veinte netinim, todos definidos por nombre.

**none_bilingual_063d72c8e57313c9_verse_03_es** — Pesé y entregué en sus manos seiscientos cincuenta talentos de plata, utensilios de plata por cien talentos, cien talentos de oro,

- E4B: I weighed and delivered into your hands six hundred fifty talents of silver, silver utensils for one hundred talents, one hundred talents of gold,
- E2B: I weighed and delivered in his hands six hundred fifty talents of silver, silver utensils for one hundred talents, one hundred talents of gold,

**none_bilingual_063d72c8e57313c9_verse_04_en** — And the name of the daughter of Asher <FI>is<Fi> Sarah.

- E4B: Y el nombre de la hija de Aser es Sara.
- E2B: Y el nombre de la hija de Asher es Sarah.

**none_bilingual_063d72c8e57313c9_verse_05_en** — If thou lend money to any of my people with thee that is poor, thou shalt not be to him as a creditor; neither shall ye lay upon him interest.

- E4B: Si prestas dinero a alguno de mi pueblo que esté contigo y sea pobre, no serás para él como acreedor; ni le imponerás interés.
- E2B: Si prestas dinero a alguno de mis pueblos con el que esté pobre, no serás acreedor de él; ni le pondrás interés.

**none_bilingual_063d72c8e57313c9_verse_05_es** — No blasfemarás contra Dios, ni maldecirás al príncipe de tu pueblo.

- E4B: You shall not blaspheme God, nor curse the prince of your people.
- E2B: You shall not swear blasphemies against God, nor curse the prince of your people.

**none_bilingual_063d72c8e57313c9_verse_06_en** — But I counted it necessary to send to you Epaphroditus, my brother and fellow-worker and fellow-soldier, and your messenger and minister to my need;

- E4B: Pero consideré necesario enviarte Epafrodito, mi hermano y compañero de trabajo y compañero de armas, y tu mensajero y ministro para mi necesidad;
- E2B: Pero consideré necesario enviarte a Epafrodito, mi hermano, compañero de trabajo y compañero de soldado, y tu mensajero y ministro para mi necesidad;

**none_bilingual_063d72c8e57313c9_verse_06_es** — Él no es Dios de muertos, sino de vivos. Vosotros estáis, pues, en un gran error”.

- E4B: He is not a god of the dead, but of the living. You are, therefore, in a great error.
- E2B: He is not the God of the dead, but of the living. You are, therefore, in a great error.

**none_bilingual_063d72c8e57313c9_verse_07_en** — And this is the testimony of John, when the Jews sent out of Jerusalem priests and Levites, that they might question him, `Who art thou?'

- E4B: Y este es el testimonio de Juan, cuando los judíos enviaron de Jerusalén sacerdotes y levitas, para que le preguntaran: "¿Quién eres tú?"
- E2B: Y este es el testimonio de Juan, cuando los judíos enviaron sacerdotes y levitas de Jerusalén para interrogarlo, "¿Quién eres?"

**none_bilingual_063d72c8e57313c9_verse_07_es** — Y éste es el testimonio de Juan, cuando los Judíos enviaron de Jerusalem sacerdotes y Levitas, que le preguntasen: ¿Tú, quién eres?

- E4B: And this is the testimony of John, when the Jews sent priests and Levites from Jerusalem to ask him: Who are you?
- E2B: And this is the testimony of John, when the Jews sent priests and Levites from Jerusalem to ask him: "Who are you?"

**none_bilingual_063d72c8e57313c9_verse_08_en** — And thou shalt make holy garments for Aaron thy brother, for glory and for beauty.

- E4B: Y harás vestiduras santas para Aarón tu hermano, por gloria y por hermosura.
- E2B: Y harás vestiduras santas para Aarón tu hermano, para gloria y para belleza.

**none_bilingual_063d72c8e57313c9_verse_08_es** — Tomarán para ello oro, jacinto, púrpura escarlata y carmesí y tejido de lino fino.”

- E4B: They will take for it gold, sapphire, scarlet purple, and fine linen cloth.
- E2B: They will take gold, jasmine, purple scarlet and crimson and fine linen fabric for it.

**none_bilingual_063d72c8e57313c9_verse_09_en** — and they said “I will come quickly.”Blessed will he be who lays to heart the words of the prophecy contained in this book.’

- E4B: y dijeron: "Vendré pronto". Bendito sea aquel que atesore las palabras de la profecía contenidas en este libro.
- E2B: y dijeron "vendré pronto". Bendito será el que medite las palabras de la profecía contenida en este libro.

**none_bilingual_063d72c8e57313c9_verse_09_es** — ¿No ha dicho la Escritura que el Cristo ha de venir del linaje de David, y de Belén, la aldea de David?”

- E4B: Has the Scripture not said that the Christ is to come from the lineage of David, and from Bethlehem, the town of David?
- E2B: Hasn't the Scripture said that Christ must come from the lineage of David, and from Bethlehem, the village of David?

**none_bilingual_063d72c8e57313c9_verse_10_en** — and again it saith, `Rejoice ye nations, with His people;'

- E4B: y de nuevo dice: "Regocíjense las naciones con Su pueblo;"
- E2B: y de nuevo dice, "Regocíate, naciones, con su pueblo".

**none_bilingual_063d72c8e57313c9_verse_10_es** — ¿Y sus hermanas no están todas entre nosotros? Entonces, ¿de dónde le viene todo esto?”

- E4B: And aren't all your sisters among us? So, where does all this come from?
- E2B: And aren't his sisters all among us? Then, where does all this come from?

**none_bilingual_063d72c8e57313c9_verse_11_en** — So two or three cities wandered unto one city, to drink water; but they were not satisfied: yet have ye not returned unto me, saith the Lord.

- E4B: Así dos o tres ciudades vagaron hacia una ciudad, para beber agua; pero no se saciaron: sin embargo, no habéis vuelto a mí, dice el Señor.
- E2B: Así que dos o tres ciudades vagaron hacia una ciudad para beber agua; pero no se conformaron: pero no habéis vuelto a mí, dice el Señor.

**none_bilingual_063d72c8e57313c9_verse_11_es** — Al principio del reinado de Joakim, hijo de Josías, rey de Judá, habló Yahvé en estos términos:

- E4B: At the beginning of the reign of Joakim, son of Josiah, king of Judah, Yahweh spoke in these terms:
- E2B: At the beginning of Joakim's reign, son of Josiah, king of Judah, Yahweh spoke in these terms:

**none_bilingual_063d72c8e57313c9_verse_12_en** — And the God of peace shall bruise Satan under your feet shortly. The grace of our Lord Jesus Christ be with you. Amen.

- E4B: Y el Dios de paz aplastará a Satanás bajo tus pies pronto. La gracia de nuestro Señor Jesucristo esté con ustedes. Amén.
- E2B: Y el Dios de la paz quebrantará a Satanás bajo vuestros pies pronto. La gracia de nuestro Señor Jesucristo esté con vosotros. Amén.

**none_bilingual_063d72c8e57313c9_verse_12_es** — Vosotros, al contrario, decís: “Cualquiera que diga a su padre o a su madre: “Es ofrenda (para el Templo) aquello con lo cual yo te podría haber socorrido,

- E4B: You, on the contrary, say: "Whoever says to his father or his mother: 'It is an offering (for the Temple) of what I could have helped you with,'
- E2B: You all, on the contrary, say: "Anyone who tells his father or mother: 'It is an offering (for the Temple) that with which I could have helped you'"

**none_bilingual_063d72c8e57313c9_verse_13_en** — Take my teaching, and not silver; get knowledge in place of the best gold.

- E4B: Toma mi enseñanza, y no plata; adquiere conocimiento en lugar del mejor oro.
- E2B: Toma mi enseñanza, y no plata; consigue conocimiento en lugar del mejor oro.

**none_bilingual_063d72c8e57313c9_verse_13_es** — Recibid mi enseñanza, y no plata; y ciencia antes que el oro escogido.

- E4B: Receive my teaching, and not silver; and knowledge before chosen gold.
- E2B: Receive my teaching, and not money; and knowledge before the gold chosen.

**none_bilingual_063d72c8e57313c9_verse_14_es** — “La Pascua, como sabéis, será dentro de dos días, y el Hijo del hombre va a ser entregado para que lo crucifiquen”.

- E4B: "Easter, as you know, will be in two days, and the Son of Man is going to be delivered to be crucified."
- E2B: Easter, as you know, will be in two days, and the Son of man is going to be handed over to be crucified.

**none_bilingual_063d72c8e57313c9_verse_15_en** — And by the desire of all the people, the feast went on for another seven days, and they kept the seven days with joy.

- E4B: Y por el deseo de todo el pueblo, la fiesta continuó por otros siete días, y mantuvieron los siete días con alegría.
- E2B: Y por el deseo de toda la gente, la fiesta continuó por otros siete días, y mantuvieron los siete días con alegría.

**none_bilingual_063d72c8e57313c9_verse_15_es** — Y toda aquella multitud determinó que celebrasen otros siete días; y celebraron otros siete días con alegría.

- E4B: And all that multitude determined that they celebrate for seven more days; and they celebrated for seven more days with joy.
- E2B: And all that crowd decided to celebrate seven more days; and they celebrated seven more days with joy.

**none_bilingual_063d72c8e57313c9_verse_16_en** — `Remember that which Amalek hath done to thee in the way, in your going out from Egypt,

- E4B: Recuerda lo que Amalec te hizo en el camino, al salir de Egipto,
- E2B: Recuerda lo que Amalec te hizo en el camino, en tu salida de Egipto,

**none_bilingual_063d72c8e57313c9_verse_16_es** — El sacerdote recibirá el canasto de tu mano y lo pondrá delante del altar de Yahvé, tu Dios.

- E4B: The priest will receive the basket from your hand and place it before the altar of Yahweh, your God.
- E2B: The priest will receive the basket from your hand and place it in front of the altar of Yahweh, your God.

**none_bilingual_063d72c8e57313c9_verse_17_en** — Building up Zion with blood, And Jerusalem with iniquity.

- E4B: Construyendo Sion con sangre, Y Jerusalén con iniquidad.
- E2B: Construyendo Sion con sangre, y Jerusalén con iniquidad.

**none_bilingual_063d72c8e57313c9_verse_17_es** — Que edificáis á Sión con sangre, y á Jerusalem con injusticia;

- E4B: That you build Zion with blood, and Jerusalem with injustice;
- E2B: What are you building in Zion with blood, and in Jerusalem with injustice;

**none_bilingual_063d72c8e57313c9_verse_18_en** — He took them and sent them over the stream with all he had.

- E4B: Los tomó y los envió al otro lado del arroyo con todo lo que tenía.
- E2B: Los tomó y los envió por el arroyo con todo lo que tenía.

**none_bilingual_063d72c8e57313c9_verse_18_es** — Tomólos pues, y pasólos el arroyo, é hizo pasar lo que tenía.

- E4B: So he took them, and he passed them across the stream, and he made pass what he had.
- E2B: He took them, and passed them through the stream, he made pass what he had.

**none_bilingual_063d72c8e57313c9_verse_19_en** — And the Lord your God will give to you and to your seed a circumcision of the heart, so that, loving him with all your heart and all your soul, you may have life.

- E4B: Y el Señor tu Dios te dará a ti y a tu descendencia una circuncisión del corazón, para que, amándolo con todo tu corazón y con toda tu alma, tengas vida.
- E2B: Y el Señor tu Dios te dará y a tu descendencia una circuncisión del corazón, para que, amándolo con todo tu corazón y con toda tu alma, tengas vida.

**none_bilingual_063d72c8e57313c9_verse_19_es** — No está en el cielo, de suerte que puedas decir: ‘¿Quién subirá por nosotros al cielo para que nos la traiga y nos la enseñe, y nosotros la pongamos por obra?’

- E4B: It is not in heaven, luckily you can say: 'Who will go up for us to heaven to bring it to us and show it to us, so that we may put it into practice?'
- E2B: It is not in heaven, so you can say: 'Who will go up with us to heaven to bring it to us and show it to us, and we will put it into practice?'

**none_bilingual_063d72c8e57313c9_verse_20_es** — Entonces él se levantó y se fué á su casa.

- E4B: So he got up and went to his house.
- E2B: Then he got up and went to his house.

**none_bilingual_063d72c8e57313c9_verse_21_es** — Entonces él se levantó y se fué á su casa.

- E4B: So he got up and went to his house.
- E2B: Then he got up and went to his house.

**none_bilingual_063d72c8e57313c9_verse_22_en** — And he stayeth yet other seven days, and addeth to send forth the dove from the ark;

- E4B: Y se queda siete días más, y añade para enviar la paloma del arca;
- E2B: Y permanece otros siete días, y añade a enviar la paloma del arca;

**none_bilingual_063d72c8e57313c9_verse_22_es** — Y esperó aún otros siete días, y volvió á enviar la paloma fuera del arca.

- E4B: And he waited another seven days, and sent the dove out of the ark again.
- E2B: And he waited another seven days, and returned to send the dove out of the ark.

**none_bilingual_063d72c8e57313c9_verse_23_en** — And in the evening, at sundown, they took to him all who were diseased, and those who had evil spirits.

- E4B: Y por la tarde, al atardecer, le trajeron todos los enfermos y los que tenían espíritus malignos.
- E2B: Y por la noche, al atardecer, se llevaron a él a todos los que estaban enfermos y a los que tenían espíritus malignos.

**none_bilingual_063d72c8e57313c9_verse_23_es** — Por tanto, así dice, Yahvé, el Señor: He aquí que haré venir sobre ti la espada, y exterminaré en ti hombres y bestias.

- E4B: Therefore, thus says Yahweh, the Lord: Behold, I will bring the sword upon you, and I will exterminate men and beasts in you.
- E2B: Therefore, so says Yahweh, the Lord: Behold, I will bring the sword upon you, and I will exterminate men and beasts in you.

**none_bilingual_063d72c8e57313c9_verse_24_en** — Thou hast turned my mourning to dancing for me, Thou hast loosed my sackcloth, And girdest me <FI>with<Fi> joy.

- E4B: Has convertido mi luto en danza para mí, has quitado mi cilicio y me has ceñido con gozo.
- E2B: Has convertido mi luto en danza por mí, has desatado mi saco y me has atado con alegría.

**none_bilingual_063d72c8e57313c9_verse_24_es** — Has tornado mi endecha en baile; desataste mi saco, y ceñísteme de alegría.

- E4B: You have tornado my right hand in dance; you have unleashed my sack, and girded me with joy.
- E2B: You have torn my edge in dance; you unleashed my sack, and tightened me with joy.

## Pending gates

- ≥50 approved natural-speech reference utterances per language.
- Bilingual review of meaning errors and terminology preferences (blind_review.jsonl).
- Visible-browser render acknowledgments for the sub-second caption-delivery gate.
- Physical second-output / hotplug and real two-speaker validation.

Recorded failed runs: 0. Excluded incompatible/duplicate runs: 0. Details are retained in comparison.json.
