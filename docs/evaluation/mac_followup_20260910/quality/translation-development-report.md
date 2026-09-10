# Public speech quality evaluation

Manifest: `9de09bead155cef2a62f58ef0be91c43e35ced2e91672fbcfa641f358c498385`.

These are isolated whole-recording STT or fixed-reference-text translation measurements. They exclude VAD, queueing, audio capture and browser delivery. No model promotion or human quality approval is implied.

| Engine | Direction | Partition / repeat | Complete + identity valid | n | Call p50 / p95 ms | WER | chrF | Canary |
|---|---|---|---|---:|---:|---:|---:|---:|
| e4b | en | development / 0 | True | 68 | 1041.553 / 1852.789 | — | 57.027 | 13/18 |
| e2b | en | development / 0 | True | 68 | 664.431 / 1097.901 | — | 56.799 | 11/18 |
| e4b | es | development / 0 | True | 50 | 1029.688 / 1596.315 | — | 61.343 | 0/0 |
| e2b | es | development / 0 | True | 50 | 653.806 / 997.546 | — | 60.252 | 0/0 |
| e2b | en | development / 1 | True | 68 | 664.746 / 1049.086 | — | 56.799 | 11/18 |
| e4b | en | development / 1 | True | 68 | 1045.407 / 1914.626 | — | 57.027 | 13/18 |
| e2b | es | development / 1 | True | 50 | 631.810 / 886.548 | — | 60.252 | 0/0 |
| e4b | es | development / 1 | True | 50 | 992.821 / 1507.105 | — | 61.343 | 0/0 |
| e4b | en | development / 2 | True | 68 | 1003.906 / 1783.788 | — | 57.027 | 13/18 |
| e2b | en | development / 2 | True | 68 | 627.666 / 1007.731 | — | 56.799 | 11/18 |
| e4b | es | development / 2 | True | 50 | 1023.565 / 1509.759 | — | 61.343 | 0/0 |
| e2b | es | development / 2 | True | 50 | 622.667 / 982.845 | — | 60.252 | 0/0 |
