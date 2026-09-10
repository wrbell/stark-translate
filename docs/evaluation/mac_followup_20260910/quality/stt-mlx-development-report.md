# Public speech quality evaluation

Manifest: `9de09bead155cef2a62f58ef0be91c43e35ced2e91672fbcfa641f358c498385`.

These are isolated whole-recording STT or fixed-reference-text translation measurements. They exclude VAD, queueing, audio capture and browser delivery. No model promotion or human quality approval is implied.

| Engine | Direction | Partition / repeat | Complete + identity valid | n | Call p50 / p95 ms | WER | chrF | Canary |
|---|---|---|---|---:|---:|---:|---:|---:|
| parakeet-mlx | en | development / 0 | True | 50 | 224.412 / 319.954 | 0.050 | — | 0/0 |
| whisper-mlx | en | development / 0 | True | 50 | 699.043 / 776.356 | 0.044 | — | 0/0 |
| parakeet-mlx | es | development / 0 | True | 50 | 226.287 / 332.316 | 0.039 | — | 0/0 |
| whisper-mlx | es | development / 0 | True | 50 | 718.797 / 855.611 | 0.030 | — | 0/0 |
| whisper-mlx | en | development / 1 | True | 50 | 702.984 / 777.721 | 0.044 | — | 0/0 |
| parakeet-mlx | en | development / 1 | True | 50 | 220.326 / 315.756 | 0.050 | — | 0/0 |
| whisper-mlx | es | development / 1 | True | 50 | 720.893 / 885.538 | 0.030 | — | 0/0 |
| parakeet-mlx | es | development / 1 | True | 50 | 223.887 / 348.381 | 0.039 | — | 0/0 |
| parakeet-mlx | en | development / 2 | True | 50 | 220.300 / 323.532 | 0.050 | — | 0/0 |
| whisper-mlx | en | development / 2 | True | 50 | 739.380 / 818.443 | 0.044 | — | 0/0 |
| parakeet-mlx | es | development / 2 | True | 50 | 270.079 / 392.249 | 0.039 | — | 0/0 |
| whisper-mlx | es | development / 2 | True | 50 | 875.934 / 1051.408 | 0.030 | — | 0/0 |
