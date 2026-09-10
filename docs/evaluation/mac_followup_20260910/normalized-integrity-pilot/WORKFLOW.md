# Larger-cohort reporting and archival procedure

This procedure is prepared for the Standard 96-run, CPU Lite 24-run and Spanish
Parakeet 18-run screens. Their results are not implied by this pilot. Execute only
after the corresponding runner has released its model window and its frozen output
inventory is complete. Use a fresh output directory for each cohort; retain failed
attempts without replacing them with favorable reruns.

1. Generate the paired report and selection **outside** the raw cohort directory:

   ```sh
   python -m tools.mac_followup_latency report --input RUNS --output REPORT.json
   python -m tools.mac_followup_selection --report REPORT.json --output SELECTION.json
   ```

   Record the exact reporter source revision alongside the runner revision. The
   report checks the declared inventory, session/source/model identities, exact
   reference contracts and pairs each candidate with opening and closing controls.
   Selection requires exactly three unique repetitions; at least two must meet the
   median gain gate against both controls. All three must pass other guards: the
   only permitted unsuccessful third result is `median_gain_below_gate`.

2. Archive unmodified evidence using the supplied standard-library script:

   ```sh
   python docs/evaluation/mac_followup_20260910/normalized-integrity-pilot/archive_cohort.py.txt \
     --input RUNS --output ARCHIVE --runner-log RUNNER.log \
     --metrics FROZEN_RUNTIME/metrics --report REPORT.json --selection SELECTION.json
   ```

   The archive includes every raw run JSON, provenance, available production CSVs
   and session logs, the runner log, report, selection and per-run summary. The
   SHA-256 index binds exact bytes; raw provenance binds source/dependency settings,
   requested artifacts, audio hashes, normalization and references. Copies are
   checked and existing payloads are never overwritten. Audio remains separately
   retained with its original/derived hashes and public attribution; do not call the
   archive a self-contained audio bundle. Do not copy private blinded-review keys.

3. Write one Markdown report per cohort. Present results separately by **language,
   actual model/profile, candidate and repetition**. CPU Lite's `e2b` scheduling
   label does not mean Gemma ran: identify actual Marian finals and CPU Whisper
   from runtime contracts. Show all arms, including no qualified arms, failures and
   rejected combinations. Keep earlier unnormalized input and older source cohorts
   separate; never pool incompatible metric definitions or input hashes.

4. For each pair, expose the report's fixed-source opening/candidate/closing median,
   individual guard failures, queue and memory comparisons, first-preview/update-gap
   evidence, production transcript WER, glossary opportunities/recall and descriptive
   reference chrF deltas. Present `by_opening_endpoint_reason` from each fixed-span
   result separately; preserve all raw endpoint samples from `run-summary.json`.
   Do not merge silence endings, forced cuts and EOF into a single latency claim.
   Legacy processing metrics remain raw historical fields, not caption delivery.

5. Preserve `p95_claim_eligible`: at least 100 eligible observations are required in
   **each run**. Three small repetitions do not become a qualifying tail cohort by
   summing their counts. Missing preview/quality evidence is unavailable, not zero.
   Schema-2 server speech-end-to-final and fixed-source delivery are distinct from
   browser render acknowledgments; this headless replay does not certify physical
   display visibility. Human meaning review remains pending regardless of chrF.

6. Follow up only on independently qualified settings in their selected clip/model
   scope. Retain a machine-readable no-qualified result when applicable. Evaluate
   combinations afresh rather than adding individual gains. Use untouched public
   confirmation and the separate historical replays with adequate timeouts; missing
   historical reference quality stays unavailable and requires independent public
   quality evidence. Do not relax the existing reference guard or imply production
   promotion. Record confirmation scope, any endpoint sample shortage and external
   human/device gates explicitly.

## Compressed delivery for larger cohorts

For the 96/24/18-run cohorts, add `--compress` to the archive command above.
This stores full raw run JSON, traces and optional production logs/CSVs in
`raw-evidence.tar.gz`; do not commit their expanded `raw/` directory. Original
payload bytes are preserved, including negative results. Readable `provenance.json`,
`run-summary.json`, `report.json`, `selection.json` and `runner.log` remain alongside
it. Keep the existing small pilot archive unchanged.

Tar members have sorted names, fixed mode 0644, zero ownership and modification
timestamps, and empty user/group names. Gzip uses an empty filename and timestamp
zero, with level 6. Thus the same input bytes produce the same compressed artifact
within the recorded Python/zlib runtime. The index records the compressed SHA-256
and size **and each original member's SHA-256 and size**. The script rereads every
compressed member and validates its exact bytes before writing the final index.
No audio files or private review keys are selected.

To reproduce, run the same command into a new directory with `--compress`, using
the original completed cohort and identical report/selection files. To inspect raw
records without extracting into the repository:

```sh
python - <<'PY'
import hashlib, json, pathlib, tarfile
root = pathlib.Path('ARCHIVE')
index = json.loads((root / 'evidence-index.json').read_text())
entry = next(f for f in index['files'] if f['path'] == 'raw-evidence.tar.gz')
archive = root / entry['path']
assert hashlib.sha256(archive.read_bytes()).hexdigest() == entry['sha256']
with tarfile.open(archive, 'r:gz') as tar:
    assert [m.name for m in tar.getmembers()] == [m['path'] for m in entry['members']]
    for expected in entry['members']:
        raw = tar.extractfile(expected['path']).read()
        assert len(raw) == expected['size_bytes']
        assert hashlib.sha256(raw).hexdigest() == expected['sha256']
print('Compressed artifact and all original members verified')
PY
```

Compression is archival work after a timed cohort releases its window. Do not
compress hundreds of megabytes concurrently with inference. The current pilot's
original evidence index binds its retained raw payloads and remains unchanged.
