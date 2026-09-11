# P3 — end-of-utterance classifier (Smart Turn v3) offline feasibility: **no-go**

**Question.** Could a semantic end-of-turn classifier shorten the fixed 0.5 s silence countdown (≈ 0.49 s of every
silence final) by declaring “complete” earlier on true utterance ends while staying quiet on internal pauses?
Shorter uniform triggers (0.4/0.35 s) were already screened and rejected on quality/preview guards; a classifier
would have to separate the two cases.

**Model.** `pipecat-ai/smart-turn-v3` (Pipecat / Daily, BSD-2, US origin), file `smart-turn-v3.2-cpu.onnx`, revision
`f766f81d3cfdf7737ac64aad813d91bbfd56bf93`, SHA256 `2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f`
([`pin.json`](pin.json)); inference reproduced from upstream `inference.py` (SHA256 `39a6422b…`): last ≤ 8 s of 16 kHz
audio left-padded to 128 000 samples, `WhisperFeatureExtractor(chunk_length=8)` with `do_normalize=True`, ONNX input
`(1, 80, 800)`, sigmoid probability, shipped threshold 0.5. `onnxruntime 1.24.2` / `transformers 5.12.1` in `venv`.
No threshold was tuned and nothing in the pipeline changed.

**Dataset (exact reconstruction, [`eou_offline.py`](eou_offline.py)).** From the six traced series-3 control replays
(`lb0912_{A,B}_ctl_r{0,1,2}`; 16 kHz clips, capture coordinates ÷ 3):
- positives = every silence-ended final's audio from its utterance start to speech end + 200 / 300 / 480 ms
  (267 finals; 480 ms is where the pipeline actually finalizes);
- negatives = every internal VAD-negative run of ≥ 6 frames (32 ms frames, i.e. ≥ 192 ms) that resumed speech inside the
  same utterance and started ≥ 0.7 s into it (the minimum-final guard), cut at run start + 200 / 300 ms (132 pauses;
  bins 6–8 / 9–11 / ≥ 12 frames).
- extra positives ([`eou_endurance.py`](eou_endurance.py)): the 401 retained silence-final chunks of the 2026-09-10
  installed-service endurance replay (`20260910_043120_839144_en`), trimmed from their 448 ms tail to +200 / 300 / 480 ms.

## Result (probability > 0.5 = “complete”; [`eou_offline.json`](eou_offline.json), [`eou_endurance.json`](eou_endurance.json))

| cohort | true ends judged complete @ +200 / +300 / +480 ms | internal pauses judged complete @ +200 / +300 ms | pauses by length 6–8 / 9–11 / ≥ 12 frames (complete @ +200) | rank AUC @ +200 / +300 |
|---|---|---|---|---|
| clip A (63 ends, 39 pauses) | 23.8 % / 14.3 % / 19.0 % | 30.8 % / 23.1 % | 40.0 % / 0.0 % / 66.7 % | 0.36 / 0.39 |
| clip B (204 ends, 93 pauses) | 67.6 % / 69.1 % / 67.6 % | 54.8 % / 54.8 % | 58.3 % / 37.5 % / 63.6 % | 0.56 / 0.59 |
| pooled (267 ends, 132 pauses) | 57.3 % / 56.2 % / 56.2 % | 47.7 % / 45.5 % | 52.9 % / 23.1 % / 64.3 % | 0.54 / 0.57 |
| endurance service (401 ends, no pauses cut) | 71.8 % / 73.8 % / 73.8 % | — | — | — |

CPU cost per call: 9.0 ms p50 / 18.3 ms p95 (pooled), 15.7 / 19.2 ms on the endurance chunks.

Implied trade if a “complete” verdict shortened the countdown (pooled, per 360 s run of ≈ 70 finals, 44.5 silence-ended):
a 300 ms countdown would finalize ≈ 25 finals 180 ms earlier and add ≈ 5.5 spurious finals (+7.8 % finals);
a 200 ms countdown ≈ 25.5 finals 280 ms earlier and ≈ 8.5 spurious finals (+12 %).

**Verdict: no-go.** At the shipped threshold the classifier calls internal pauses “complete” almost as often as true
ends (46–48 % vs 56–57 % pooled; AUC ≈ 0.55, chance-like), and on clip A it misses three quarters of the true ends
while still firing on a third of the pauses. The endurance cohort's higher hit rate (72–74 %) has no matching pause
set and does not change the separation. No declared arm follows from this study; the 0.5 s countdown remains a policy
floor. This study does not evaluate the classifier for conversational turn-taking, its intended use.
