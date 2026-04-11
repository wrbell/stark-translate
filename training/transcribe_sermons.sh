#!/usr/bin/env bash
# transcribe_sermons.sh — Batch Whisper Transcription of Gospel Sermons
#
# Transcribes all untranscribed WAV files in stark_data/raw/midwest/ using
# faster-whisper large-v3 (fp16, ~5GB VRAM, ~8x realtime).
#
# Outputs chunks in the same format as ablation/sermon_whisper_chunks.json:
#   {"en", "start", "end", "confidence", "avg_logprob", "source"}
#
# Merges with existing chunks → ablation/sermon_whisper_chunks_expanded.json
#
# Usage:
#   bash training/transcribe_sermons.sh
#   nohup bash training/transcribe_sermons.sh &

set -euo pipefail

# --- Setup ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

VENV="/home/wbell/stt_train_env/bin/activate"
source "$VENV"

INPUT_DIR="stark_data/raw/midwest"
TRANSCRIPT_DIR="ablation/sermon_transcripts"
EXISTING="ablation/sermon_whisper_chunks.json"
OUTPUT="ablation/sermon_whisper_chunks_expanded.json"
LOG="ablation/transcribe_sermons_log.txt"

mkdir -p "$TRANSCRIPT_DIR"

echo "=== Batch Sermon Transcription — $(date) ===" | tee "$LOG"
echo "Input: $INPUT_DIR" | tee -a "$LOG"
echo "Output: $OUTPUT" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>&1 | tee -a "$LOG"
echo "" | tee -a "$LOG"

# --- Find WAV files, skip already-transcribed conference sermons ---
# The 2 conference sermons are already in sermon_whisper_chunks.json
SKIP_PATTERN="Conference"

TOTAL_FILES=0
SKIP_FILES=0
declare -a WAV_FILES=()

for wav in "$INPUT_DIR"/*.wav; do
    [[ -f "$wav" ]] || continue
    TOTAL_FILES=$(( TOTAL_FILES + 1 ))
    basename_wav="$(basename "$wav")"

    # Skip conference files already transcribed
    if [[ "$basename_wav" == *"$SKIP_PATTERN"* ]]; then
        echo "SKIP (already transcribed): $basename_wav" | tee -a "$LOG"
        SKIP_FILES=$(( SKIP_FILES + 1 ))
        continue
    fi

    # Skip if transcript already exists (resume support)
    stem="${basename_wav%.wav}"
    if [[ -f "$TRANSCRIPT_DIR/${stem}.json" ]]; then
        echo "SKIP (transcript exists): $basename_wav" | tee -a "$LOG"
        SKIP_FILES=$(( SKIP_FILES + 1 ))
        continue
    fi

    WAV_FILES+=("$wav")
done

echo "" | tee -a "$LOG"
echo "Found $TOTAL_FILES WAV files, skipping $SKIP_FILES, transcribing ${#WAV_FILES[@]}" | tee -a "$LOG"
echo "" | tee -a "$LOG"

if [[ ${#WAV_FILES[@]} -eq 0 ]]; then
    echo "No files to transcribe. Proceeding to merge step." | tee -a "$LOG"
fi

# --- VRAM logging helper ---
log_vram() {
    local label="$1"
    echo "[VRAM] $label — $(date '+%H:%M:%S')" | tee -a "$LOG"
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total \
        --format=csv,noheader,nounits 2>/dev/null | \
        awk -F', ' '{printf "[VRAM]   Used: %s MiB / %s MiB total (%.0f%% free)\n", $1, $3, $2/$3*100}' | \
        tee -a "$LOG"
}

log_vram "Before whisper load"

# --- Transcribe each file with faster-whisper ---
TRANSCRIBE_START=$(date +%s)
PROCESSED=0

for wav in "${WAV_FILES[@]}"; do
    basename_wav="$(basename "$wav")"
    stem="${basename_wav%.wav}"
    out_json="$TRANSCRIPT_DIR/${stem}.json"

    echo "--- [$((PROCESSED + 1))/${#WAV_FILES[@]}] Transcribing: $basename_wav — $(date)" | tee -a "$LOG"

    FILE_START=$(date +%s)

    # Run faster-whisper via Python inline
    # Outputs per-segment JSON with confidence scores
    python -c "
import json
import subprocess
import sys
import time

import torch

try:
    from faster_whisper import WhisperModel
except ImportError:
    print('ERROR: faster-whisper not installed. Run: pip install faster-whisper', file=sys.stderr)
    sys.exit(1)

def log_vram_py(label):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f'[VRAM-py] {label}: {alloc:.1f}GB alloc / {reserved:.1f}GB reserved / peak {peak:.1f}GB')
    # Also log nvidia-smi for total GPU picture
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total',
             '--format=csv,noheader,nounits'], text=True).strip()
        parts = out.split(', ')
        print(f'[VRAM-smi] {label}: {parts[0]} MiB used / {parts[2]} MiB total')
    except Exception:
        pass

wav_path = '$wav'
out_path = '$out_json'
source = '${stem}'

log_vram_py('Before whisper load')

print(f'Loading faster-whisper large-v3 on CUDA...')
model = WhisperModel('large-v3', device='cuda', compute_type='float16')

log_vram_py('After whisper load (large-v3 fp16)')

print(f'Transcribing {wav_path}...')
start = time.time()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
segments_iter, info = model.transcribe(
    wav_path,
    language='en',
    word_timestamps=True,
    vad_filter=True,
)

chunks = []
for seg in segments_iter:
    text = seg.text.strip()
    if not text:
        continue
    # Compute word-level mean confidence
    word_conf = 0.0
    if seg.words:
        word_conf = sum(w.probability for w in seg.words) / len(seg.words)
    chunks.append({
        'en': text,
        'start': round(seg.start, 2),
        'end': round(seg.end, 2),
        'confidence': round(word_conf, 4),
        'avg_logprob': round(seg.avg_logprob, 4),
        'source': source,
    })

elapsed = time.time() - start
print(f'Transcribed {len(chunks)} chunks in {elapsed:.0f}s (duration={info.duration:.0f}s, {info.duration/elapsed:.1f}x realtime)')

log_vram_py('After transcription (peak during inference)')

with open(out_path, 'w') as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)
print(f'Saved to {out_path}')
" 2>&1 | tee -a "$LOG"

    FILE_END=$(date +%s)
    FILE_ELAPSED=$(( FILE_END - FILE_START ))
    PROCESSED=$(( PROCESSED + 1 ))
    echo "[$PROCESSED/${#WAV_FILES[@]}] Done in ${FILE_ELAPSED}s" | tee -a "$LOG"
    log_vram "After file $PROCESSED/${#WAV_FILES[@]} (whisper large-v3)"
    echo "" | tee -a "$LOG"
done

TRANSCRIBE_END=$(date +%s)
TRANSCRIBE_ELAPSED=$(( TRANSCRIBE_END - TRANSCRIBE_START ))
echo "Transcription complete: $PROCESSED files in ${TRANSCRIBE_ELAPSED}s ($(( TRANSCRIBE_ELAPSED / 60 )) min)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# --- Merge all chunks into expanded file ---
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "Merging all chunks → $OUTPUT — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

python -c "
import json, glob, os

existing_path = '$EXISTING'
transcript_dir = '$TRANSCRIPT_DIR'
output_path = '$OUTPUT'

# Load existing chunks
all_chunks = []
if os.path.exists(existing_path):
    with open(existing_path) as f:
        existing = json.load(f)
    all_chunks.extend(existing)
    print(f'Loaded {len(existing)} existing chunks from {existing_path}')
else:
    print(f'WARNING: {existing_path} not found, starting fresh')

# Load new transcripts
new_count = 0
for transcript_file in sorted(glob.glob(os.path.join(transcript_dir, '*.json'))):
    with open(transcript_file) as f:
        chunks = json.load(f)
    all_chunks.extend(chunks)
    new_count += len(chunks)
    source = os.path.basename(transcript_file).replace('.json', '')
    print(f'  {source}: {len(chunks)} chunks')

print(f'Total new chunks: {new_count}')
print(f'Total combined: {len(all_chunks)}')

# Deduplicate by (source, start, en text)
seen = set()
unique = []
for c in all_chunks:
    key = (c.get('source', ''), c.get('start', 0), c.get('en', ''))
    if key not in seen:
        seen.add(key)
        unique.append(c)

dupes = len(all_chunks) - len(unique)
if dupes:
    print(f'Removed {dupes} duplicates')
print(f'Final unique chunks: {len(unique)}')

# Filter by minimum length
MIN_CHARS = 20
filtered = [c for c in unique if len(c.get('en', '')) >= MIN_CHARS]
removed = len(unique) - len(filtered)
if removed:
    print(f'Removed {removed} chunks < {MIN_CHARS} chars')
print(f'Final filtered chunks: {len(filtered)}')

with open(output_path, 'w') as f:
    json.dump(filtered, f, indent=2, ensure_ascii=False)
print(f'Saved to {output_path}')

# Per-source breakdown
sources = {}
for c in filtered:
    s = c.get('source', 'unknown')
    sources[s] = sources.get(s, 0) + 1
print()
print('Per-source breakdown:')
for s in sorted(sources):
    print(f'  {s}: {sources[s]}')
" 2>&1 | tee -a "$LOG"

PIPELINE_END=$(date +%s)
TOTAL_ELAPSED=$(( PIPELINE_END - TRANSCRIBE_START ))
echo "" | tee -a "$LOG"
echo "=== Transcription pipeline complete — ${TOTAL_ELAPSED}s ($(( TOTAL_ELAPSED / 60 )) min) — $(date) ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Outputs:" | tee -a "$LOG"
echo "  Per-sermon transcripts: $TRANSCRIPT_DIR/*.json" | tee -a "$LOG"
echo "  Merged chunks: $OUTPUT" | tee -a "$LOG"
echo "  Log: $LOG" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Next steps:" | tee -a "$LOG"
echo "  1. Regenerate hybrid data from expanded pool:" | tee -a "$LOG"
echo "     python training/generate_hybrid_synthetic.py \\" | tee -a "$LOG"
echo "       --deepl-key \"\$DEEPL_KEY\" \\" | tee -a "$LOG"
echo "       --whisper-chunks $OUTPUT \\" | tee -a "$LOG"
echo "       --max-chunks 5000 --ratio-deepl 0.40 \\" | tee -a "$LOG"
echo "       --output bible_data/synthetic/hybrid_sermon_pairs_expanded.jsonl" | tee -a "$LOG"
echo "  2. Re-run winning S4/S5/S6 config with expanded data" | tee -a "$LOG"
echo "=== Done ===" | tee -a "$LOG"
