#!/usr/bin/env bash
# download_from_catalog.sh — Download playlist videos into sorted stt-data/ dirs
#
# Downloads all videos from stt-data/manifest.json into the sorted
# stt-data/{type}/{year}/ directory structure. Groups URLs by target dir
# for efficient batching via download_sermons.py.
#
# Usage:
#   bash training/download_from_catalog.sh
#   bash training/download_from_catalog.sh --dry-run
#   nohup bash training/download_from_catalog.sh &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

VENV="/home/wbell/stt_train_env/bin/activate"
source "$VENV"

MANIFEST="stt-data/manifest.json"
LOG="stt-data/download_log.txt"
DRY_RUN="${1:-}"

mkdir -p stt-data

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: Manifest not found at $MANIFEST" >&2
    echo "  Run: python tools/sort_sermons.py --output-dir stt-data --catalog ... first" >&2
    exit 1
fi

echo "=== Downloading from manifest — $(date) ===" | tee "$LOG"

# Use Python to group URLs by target_dir, then download each batch
python -c "
import json, subprocess, sys, os, time
from pathlib import Path
from collections import defaultdict

manifest = json.load(open('$MANIFEST'))
dry_run = '$DRY_RUN' == '--dry-run'

# Group catalog entries by target_dir
batches = defaultdict(list)
for entry in manifest:
    if entry.get('origin') != 'catalog':
        continue
    if entry.get('wav_path'):
        continue  # already downloaded
    target = entry.get('target_dir', 'stt-data/unknown')
    url = entry.get('url', '')
    vid = entry.get('video_id', '')
    if url and vid:
        # Check if already downloaded
        target_path = Path(target)
        if target_path.exists() and list(target_path.glob(f'*{vid}*.wav')):
            continue
        batches[target].append((url, vid))

total = sum(len(urls) for urls in batches.values())
print(f'Videos to download: {total} across {len(batches)} directories')
for target, urls in sorted(batches.items()):
    print(f'  {target}: {len(urls)} videos')
print()

if total == 0:
    print('Nothing to download.')
    sys.exit(0)

downloaded = 0
failed = 0
start_time = time.time()

for target_dir, url_list in sorted(batches.items()):
    os.makedirs(target_dir, exist_ok=True)

    # Write URLs to a temp file for batch download
    url_file = Path(target_dir) / '_download_urls.txt'
    url_file.write_text('\n'.join(url for url, vid in url_list))

    print(f'--- Batch: {target_dir} ({len(url_list)} videos) — {time.strftime(\"%H:%M:%S\")}')

    if dry_run:
        for url, vid in url_list:
            print(f'  [DRY RUN] {vid}: {url}')
        continue

    # Download entire batch at once
    cmd = [
        sys.executable, 'download_sermons.py',
        '-s', str(url_file),
        '--output-dir', target_dir,
        '--no-video-json',
        '--skip-existing',
    ]

    result = subprocess.run(cmd, capture_output=False)

    # Count what was downloaded
    new_wavs = list(Path(target_dir).glob('*.wav'))
    batch_downloaded = len(new_wavs)
    downloaded += batch_downloaded
    if result.returncode != 0:
        failed += len(url_list) - batch_downloaded
        print(f'  WARNING: Some downloads may have failed (exit code {result.returncode})')

    print(f'  Batch done: {batch_downloaded} WAVs in {target_dir}')

    # Clean up temp file
    url_file.unlink(missing_ok=True)

elapsed = time.time() - start_time
print()
print(f'=== Download complete — {time.strftime(\"%H:%M:%S\")} ===')
print(f'  Downloaded: {downloaded}')
print(f'  Failed: {failed}')
print(f'  Elapsed: {elapsed/60:.0f} min')
print()
print('Next steps:')
print('  1. Re-run sort_sermons.py to update manifest with new files')
print('  2. Transcribe new files with Deepgram + faster-whisper')
" 2>&1 | tee -a "$LOG"
