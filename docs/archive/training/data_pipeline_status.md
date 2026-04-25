# Data Pipeline Status (2026-03-27)

## Audio Inventory

| Source | WAVs | Location | Type |
|--------|------|----------|------|
| Midwest (original) | 35 | `stark_data/raw/midwest/` (D: via symlink) | Gospel (31), Conference (2), Gospel+Baptism (1), Ministry (1) |
| YouTube (downloaded) | 332 | `stt-data/` → `D:\Data\stt-data\sorted\` | Gospel (150), Ministry (103), Conference (6), Throwback (73) |
| Fresh eval | 1 | `stark_data/raw/eval_fresh/` | Ministry (3/22/26) |
| **Total** | **368** | | ~362 hours of audio |

## Processing Pipeline Status

### Midwest Sermons (35) — COMPLETE

| Step | Status | Output |
|------|--------|--------|
| Audio | 35/35 | `stark_data/raw/midwest/*.wav` |
| Deepgram STT | 35/35 | `stark_data/deepgram_transcripts/*.deepgram.json` |
| Faster-whisper chunks | 35/35 | `ablation/sermon_whisper_chunks_expanded.json` (24,595 chunks) |
| Alignment | 17,207 done | `stark_data/whisper_dataset_deepgram/.preprocessed_cache/` (16,956 train + 251 eval) |
| DeepL translation | 21,828 pairs | `bible_data/synthetic/deepl_sermon_pairs_full.jsonl` |

### YouTube Sermons (332) — IN PROGRESS

| Step | Status | Blocked by | ETA | Cost |
|------|--------|------------|-----|------|
| Audio | 332/332 | — | Done | Free |
| Deepgram STT | **0/332 (running)** | — | ~6 hrs (tonight) | ~$93 |
| Faster-whisper chunks | **0/332** | GPU (W9 training) | After W9 (~Sat AM) + 12 hrs | Free |
| Alignment | **blocked** | Deepgram + faster-whisper | After both complete + 5 hrs | Free |
| DeepL translation | **blocked** | Alignment (need chunks first) | After alignment + ~25 hrs API | ~$109 |

### Fresh Eval Sermon (1) — COMPLETE

| Step | Status | Output |
|------|--------|--------|
| Audio | 1/1 | `stark_data/raw/eval_fresh/Teaching_Message_(3_22_26)_FOVTvZednUQ.wav` |
| Deepgram STT | 1/1 | `stark_data/eval_fresh_transcripts/*.deepgram.json` |
| DeepL translation | 1/1 | `stark_data/eval_fresh_transcripts/*.eval_reference.json` (699 segments) |
| Faster-whisper chunks | 0/1 | Not yet processed |

## Processing Dependencies

```
Audio download ──────────────────────────────────────── DONE (368 WAVs)
       │
       ├── Deepgram STT (API, no GPU) ──────────────── Midwest DONE, YouTube RUNNING
       │         │
       │         └── DeepL translation (API, no GPU) ── Midwest DONE, YouTube BLOCKED
       │
       └── Faster-whisper chunks (GPU) ─────────────── Midwest DONE, YouTube BLOCKED (W9 on GPU)
                  │
                  └── Alignment (Deepgram + whisper) ── Midwest DONE, YouTube BLOCKED
                           │
                           └── Whisper training dataset ─ Midwest DONE (17K chunks)
                                                          YouTube target: ~100-200K chunks
```

## What's Running Right Now

| Job | PID | Resource | Progress | ETA |
|-----|-----|----------|----------|-----|
| W9 Whisper training (r=64) | GPU | A2000 16GB | ~130/3555 (3%) | Sat Mar 28 AM |
| Deepgram (332 YouTube sermons) | API | No GPU | 0/332 starting | Tonight ~9 PM |

## Cost Summary

| Service | Midwest (done) | YouTube (pending) | Total |
|---------|---------------|-------------------|-------|
| Deepgram STT | $9 (35 sermons) | ~$93 (332 sermons) | ~$102 |
| DeepL translation | ~$3 (21K pairs) | ~$109 (est. 200K+ chunks) | ~$112 |
| **Total API cost** | **$12** | **~$202** | **~$214** |

## Next Steps (in order)

1. **Deepgram finishes** (tonight) — 332 `.deepgram.json` files ready
2. **W9 finishes** (Sat AM) — GPU freed
3. **Faster-whisper on 332 sermons** (Sat, ~12 hrs GPU) — produces chunk boundaries
4. **Alignment** (Sat night, ~5 hrs) — merges Deepgram text with whisper chunk boundaries
5. **Train W12** (Sun, W7 config on ~100K chunks, 1 epoch, ~45 hrs) — the big data scaling run
6. **DeepL on new chunks** ($109, parallel with training) — for TranslateGemma if needed later

## Training Data Cutoff

**Cutoff date: 2026-03-14**

- **Train**: all sermons dated before 3/14/26
- **Eval**: sermons dated 3/14/26 and after
- Currently 1 eval sermon (3/15/26, 251 chunks). More post-cutoff sermons from YouTube downloads will expand eval set.
- The 3/22/26 fresh ministry sermon is reserved for live testing, not training.

## Whisper Training Results So Far

| Run | Config | WER(norm) | Status |
|-----|--------|-----------|--------|
| Base | — | 20.78% | — |
| **W7** | 3 epochs, r=32, 17K chunks | **5.63%** | **Best** |
| W8 | 5 epochs, r=32, 17K chunks | 7.96% | Overfitting |
| W9 | 3 epochs, r=64, 17K chunks | ? | Training |
| W12 | 1 epoch, r=32, ~100K chunks | ? | Blocked on data pipeline |
