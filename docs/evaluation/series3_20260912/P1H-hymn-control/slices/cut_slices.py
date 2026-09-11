"""Lossless PCM16 slices of the parent recording for the #193 annotation packet (no resampling, no normalization)."""
import hashlib, json, wave, datetime
from pathlib import Path
ROOT = Path('/Users/willem/Code/vibes/SRTranslate'); OUT = ROOT / '.cache/series3-20260912/P1H/slices'; OUT.mkdir(parents=True, exist_ok=True)
PARENT = ROOT / 'stark_data/raw/Gospel_Message_(12_14_25)_5D2rOMvkwrk.wav'
SLICES = {'hymn-search-400-470': (6400000, 7520000), 'hymn-search-700-780': (11200000, 12480000)}
def sha(p):
    with open(p, 'rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
parent_sha = sha(PARENT)
with wave.open(str(PARENT), 'rb') as w:
    params = w.getparams(); assert (params.nchannels, params.sampwidth, params.framerate, params.comptype) == (1, 2, 16000, 'NONE'), params
    total = params.nframes
    for name, (a, b) in SLICES.items():
        assert 0 <= a < b <= total; w.setpos(a); frames = w.readframes(b - a)
        target = OUT / f'{name}.wav'
        with wave.open(str(target), 'wb') as o:
            o.setnchannels(1); o.setsampwidth(2); o.setframerate(16000); o.writeframes(frames)
        receipt = {'slice': name, 'parent_path': str(PARENT), 'parent_sha256': parent_sha, 'parent_frames': total,
                   'source_native_sample_offset': a, 'source_native_sample_end': b, 'frames': b - a, 'duration_s': (b - a) / 16000,
                   'sample_rate': 16000, 'channels': 1, 'sample_width_bytes': 2, 'path': str(target), 'sha256': sha(target), 'size_bytes': target.stat().st_size,
                   'method': 'Lossless PCM frame slice with wave.setpos/readframes; no resampling, normalization, synthetic padding or removed intervals.',
                   'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'labels': 'none; acoustic annotation pending (issue #193)'}
        (OUT / f'{name}.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(name, receipt['duration_s'], 's', receipt['sha256'][:16], receipt['size_bytes'])
print('parent', parent_sha[:16], total)
