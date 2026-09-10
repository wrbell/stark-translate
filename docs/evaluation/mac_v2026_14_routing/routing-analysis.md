# Observed synthetic routing values

All 72 observed source captions match the corresponding declared script after punctuation/case/whitespace normalization. Each row below contains three actual repeats. Timings are p50 / nearest-rank p95 in milliseconds; with three observations p95 is the maximum.

Per-caption routes are derived from `tps_a` and reconciled with authoritative session-summary counts. Confidence is a rounded 1+mean-logprob proxy, not a calibrated probability.

|Policy|Model|Source language|Phrase|Confidence|Derived route|Speech end → final|STT|Translation|
|---|---|---|---:|---|---|---|---|---|
|conservative|e2b|en|1|1.0|marian: 3|1119.1 / 1343.4|576.4 / 710.6|43.8 / 44.0|
|conservative|e2b|en|2|1.0|marian: 3|853.5 / 872.9|325.7 / 343.6|35.1 / 35.3|
|conservative|e2b|en|3|1.0|gemma: 3|1144.1 / 1194.8|319.4 / 358.1|302.8 / 304.9|
|conservative|e2b|es|1|0.9|marian: 3|1920.4 / 1975.0|1386.9 / 1452.3|38.2 / 41.2|
|conservative|e2b|es|2|0.91|marian: 3|1418.1 / 1423.1|888.3 / 894.1|34.5 / 35.3|
|conservative|e2b|es|3|0.94|gemma: 3|1954.7 / 1978.2|1120.0 / 1124.9|308.6 / 319.7|
|conservative|e4b|en|1|1.0|marian: 3|1831.6 / 1866.7|499.3 / 1246.0|39.1 / 40.2|
|conservative|e4b|en|2|1.0|marian: 3|990.4 / 1597.4|450.9 / 1051.1|43.2 / 44.1|
|conservative|e4b|en|3|1.0|gemma: 3|1381.9 / 3252.0|393.5 / 534.3|424.5 / 561.7|
|conservative|e4b|es|1|0.9|marian: 3|1859.1 / 2067.6|1325.3 / 1546.2|28.1 / 41.3|
|conservative|e4b|es|2|0.91|marian: 3|1522.0 / 2031.5|995.6 / 1499.9|26.3 / 37.8|
|conservative|e4b|es|3|0.94|gemma: 3|2214.8 / 2915.2|1201.8 / 1701.9|435.4 / 440.7|
|legacy|e2b|en|1|1.0|marian: 3|1005.6 / 1143.1|474.0 / 617.7|36.1 / 36.2|
|legacy|e2b|en|2|1.0|marian: 3|874.0 / 918.0|339.8 / 381.5|37.5 / 39.0|
|legacy|e2b|en|3|1.0|marian: 3|865.1 / 904.8|332.8 / 376.4|35.6 / 39.4|
|legacy|e2b|es|1|0.9|marian: 3|1918.1 / 1956.6|1375.5 / 1426.9|41.6 / 44.2|
|legacy|e2b|es|2|0.91|marian: 3|1410.2 / 1420.3|880.0 / 886.3|34.8 / 35.4|
|legacy|e2b|es|3|0.94|marian: 3|1623.8 / 1637.7|1103.0 / 1126.0|26.3 / 26.4|
|legacy|e4b|en|1|1.0|marian: 3|1893.0 / 1945.6|551.9 / 1211.2|36.0 / 43.0|
|legacy|e4b|en|2|1.0|marian: 3|928.5 / 1098.2|392.9 / 562.9|42.7 / 43.5|
|legacy|e4b|en|3|1.0|marian: 3|881.1 / 903.4|360.9 / 388.7|24.6 / 25.0|
|legacy|e4b|es|1|0.9|marian: 3|2573.2 / 2972.3|1914.0 / 1988.4|32.4 / 32.7|
|legacy|e4b|es|2|0.91|marian: 3|1153.7 / 1578.4|626.5 / 725.0|22.8 / 26.4|
|legacy|e4b|es|3|0.94|marian: 3|1720.8 / 1789.3|1200.1 / 1269.8|24.2 / 25.0|

## Actual captions and translations

Each language/phrase has 12 observations across both models, policies and repeats. The observed strings below are stable across all 12. EN phrase 1 differs from its script only by a final question mark.

|Language|Phrase|Observed source|Observed translation|Declared allowlist member|
|---|---:|---|---|---|
|en|1|Would you please take your seats?|¿Podrían sentarse, por favor?|yes|
|en|2|Please turn to the next page.|Por favor, pase a la siguiente página.|yes|
|en|3|The window is open.|La ventana está abierta.|no|
|es|1|Por favor tomen asiento.|Please have a seat.|yes|
|es|2|Pueden sentarse.|You may sit down.|yes|
|es|3|La ventana está abierta.|The window is open.|no|

The JSON retains all 72 individual observations, absolute audio sample bounds, per-phrase component distributions, all per-session counters, resolved model identity, memory and packaged VAD provenance. Component timings are separate observations and are not presented as a complete additive breakdown of end-to-end time.

No model/default, natural-speech quality or browser-delivery gate is promoted by this operational exercise.
