# KPI Definitions & Tuning Guide

Five key performance indicators for monitoring live session quality. Each metric has defined targets, measurement methodology, and tuning levers.

## Overview

| Rank | KPI | Target | How Measured | Current Baseline |
|------|-----|--------|-------------|------------------|
| 1 | E2E Latency P95 | <3.0s | `true_e2e_ms` in CSV → `kpi_report.py` | ~1.2s (4B pipeline, excl. VAD) |
| 2 | WER (proxy) | <10% cross-system | `validate_session.py` cross-system | ~19.6% cross-system |
| 3 | Translation QE | >0.85 mean | `qe_a` in CSV; optional CometKiwi | heuristic QE ~0.85–1.0 |
| 4 | Word Stability | >95% | `word_stability_pct` in CSV | marian_similarity ~0.5 avg |
| 5 | Processing Rate | >99.5% | `chunks_completed / chunks_attempted` | ~95% (empty STT drops) |

---

## 1. End-to-End Latency

**Definition:** Wall-clock time from first audio frame of an utterance to translated text broadcast. Includes VAD accumulation, STT, and translation.

**Formula:**
```
true_e2e_ms = (translation_complete_time - utterance_first_audio_frame_time) × 1000
```

When `true_e2e_ms` is unavailable (historical data), falls back to `e2e_latency_ms` (queue-entry → translation-complete, excludes VAD accumulation).

**Measurement:**
- Column: `true_e2e_ms` (new), `e2e_latency_ms` (legacy)
- Tool: `python tools/kpi_report.py metrics/ab_metrics_*.csv`
- Percentiles: P50, P95, P99 reported; P95 is the primary target

**Thresholds:**

| Grade | P95 Value |
|-------|-----------|
| Excellent | <2.0s |
| Target | <3.0s |
| Needs Work | ≥3.0s |

**Current baseline:** ~1.2s P50 e2e (4B pipeline, excludes VAD). True E2E will be higher by the utterance duration (~2–8s of VAD buffering).

**Tuning levers (ordered by impact):**
1. Reduce `CHUNK_DURATION` / `MAX_UTTERANCE` — shorter chunks = lower VAD accumulation
2. Use 4B-only mode (skip 12B) — saves ~600ms translation time
3. Enable prompt caching — saves ~100ms on repeat phrases
4. Lower `SILENCE_TRIGGER` — faster silence detection = sooner finalization
5. Use MarianMT-only for simple phrases (`should_use_marian_only`) — saves ~200-400ms

**Queue stalls:** Outliers >10s usually indicate pipeline queue backup (previous chunk still translating). These are reported separately as "stall count" rather than inflating percentiles.

---

## 2. Word Error Rate (WER)

**Definition:** Fraction of words incorrectly transcribed by STT, measured against a reference. No ground-truth church test set exists yet, so two proxy measurements are used.

**Proxy A — Cross-system WER:** Compare local Whisper output against YouTube's auto-captions on the same livestream. Neither is ground truth; 5–15% disagreement is normal.

**Proxy B — Confidence distribution:** STT confidence (`stt_confidence` column) as a proxy for accuracy. Low-confidence chunks correlate with higher WER.

**Measurement:**
- Cross-system: `python tools/validate_session.py metrics/ab_metrics_*.csv`
- Confidence proxy: `python tools/kpi_report.py metrics/ab_metrics_*.csv` (reports mean, P10, P25)
- Empty STT rate and hallucination rate from diagnostics

**Thresholds (confidence proxy):**

| Grade | Mean Confidence |
|-------|-----------------|
| Excellent | >0.80 |
| Target | >0.65 |
| Needs Work | ≤0.65 |

**Current baseline:** Cross-system WER ~19.6% (March 1 session). Mean confidence ~0.60–0.70 depending on audio quality.

**Tuning levers:**
1. Fine-tune Whisper with church audio (LoRA) — expected 10–30% relative WER reduction
2. Improve `WHISPER_PROMPT` with domain-specific vocabulary
3. Increase `BEAM_SIZE` from 1 to 5 — better accuracy, +200ms latency
4. Enable `WORD_TIMESTAMPS` for per-word confidence filtering
5. Lower `VAD_THRESHOLD` for better speech detection in reverberant rooms
6. Add soundboard input (direct line, no room reverb)

---

## 3. Translation Quality (QE)

**Definition:** Reference-free quality score for English→Spanish translation output.

**Tier 1 (always-on):** Heuristic QE (`qe_a`/`qe_b` in CSV) — length ratio + untranslated content detection. Score 0.0–1.0.

**Tier 2 (optional, post-session):** CometKiwi (`Unbabel/wmt22-cometkiwi-da`) — neural reference-free QE. Source + translation → 0.0–1.0 score.

**Measurement:**
- Heuristic: `qe_a` column in CSV (always available)
- CometKiwi: `python tools/kpi_report.py metrics/ab_metrics_*.csv --comet`

**Thresholds (heuristic QE):**

| Grade | Mean QE |
|-------|---------|
| Excellent | >0.95 |
| Target | >0.85 |
| Needs Work | ≤0.85 |

**Thresholds (CometKiwi):**

| Grade | Mean Score |
|-------|-----------|
| Excellent | >0.85 |
| Target | >0.82 |
| Needs Work | ≤0.82 |

**Current baseline:** Heuristic QE averages ~0.85–1.0 on well-formed sentences. CometKiwi not yet measured.

**Tuning levers:**
1. Fine-tune TranslateGemma with biblical parallel text (QLoRA) — expected +3–8 BLEU
2. Expand theological glossary for soft constraint training
3. Switch from MarianMT to Gemma for all translations (not just finals)
4. Enable 12B model for higher-quality translations at +600ms cost
5. Post-process: LanguageTool Spanish grammar check

---

## 4. Word Stability

**Definition:** Percentage of words in the partial (MarianMT) translation that are preserved in the final (TranslateGemma) translation. Measures visual consistency — high stability means less jarring text replacement on screen.

**Formula:**
```
word_stability_pct = LCS(partial_words, final_words) / len(partial_words)
```

Where LCS is the longest common subsequence (preserves word order).

**Measurement:**
- Column: `word_stability_pct` (new), `marian_similarity` (legacy Jaccard)
- Tool: `python tools/kpi_report.py metrics/ab_metrics_*.csv`

**Thresholds:**

| Grade | Mean Stability |
|-------|---------------|
| Excellent | >0.98 |
| Target | >0.95 |
| Needs Work | ≤0.95 |

**Current baseline:** `marian_similarity` (Jaccard) averages ~0.50. LCS-based stability will likely be higher since it accounts for word order.

**Tuning levers:**
1. Fine-tune MarianMT on biblical text to align its output closer to TranslateGemma
2. Use TranslateGemma 4B for partials too (at latency cost)
3. Constrained decoding: seed TranslateGemma with MarianMT output tokens
4. Display-side: fade transition instead of hard replace (UX mitigation)

---

## 5. Processing Rate

**Definition:** Fraction of audio chunks that complete the full pipeline (STT + translation + broadcast) without being dropped.

**Formula:**
```
processing_rate = chunks_completed / chunks_attempted
```

**Drop categories:**
- `empty_stt` — Whisper returns empty text (breath, noise, silence)
- `hallucination` — suppressed by `_should_suppress()` filters
- `dedup` — consecutive duplicate text suppressed
- `low_energy` — pre-STT RMS gate filters out sub-threshold audio
- `garbage` — `_is_garbage_text()` filter

**Measurement:**
- Counters in session summary output
- Session-level diagnostics JSONL
- Tool: `python tools/kpi_report.py metrics/ab_metrics_*.csv`

**Thresholds:**

| Grade | Processing Rate |
|-------|----------------|
| Excellent | >99.5% |
| Target | >98% |
| Needs Work | ≤98% |

**Current baseline:** ~95% (several empty STT drops per session, plus hallucination suppression).

**Tuning levers:**
1. Fine-tune Whisper to reduce empty STT on short utterances
2. Adjust `VAD_THRESHOLD` to filter noise earlier (before STT)
3. Increase `_MIN_UTTERANCE_DUR` to skip more sub-second fragments
4. Improve mic placement / gain for cleaner input signal
5. Lower `MUSIC_THRESHOLD` if hymn detection is too aggressive

---

## Appendix: Running the KPI Report

### Basic usage (historical CSV data)

```bash
# Single session
python tools/kpi_report.py metrics/ab_metrics_20260301_180046_en.csv

# Most recent session
python tools/kpi_report.py --latest

# Multiple sessions
python tools/kpi_report.py metrics/ab_metrics_20260301_*.csv

# Save report to file
python tools/kpi_report.py metrics/ab_metrics_20260301_180046_en.csv --output report.md
```

### With CometKiwi (requires `pip install unbabel-comet`)

```bash
python tools/kpi_report.py metrics/ab_metrics_20260301_180046_en.csv --comet
```

### Cross-system WER (requires YouTube transcript)

```bash
python tools/validate_session.py metrics/ab_metrics_20260301_180046_en.csv
```
