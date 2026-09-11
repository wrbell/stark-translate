# Correction handoff and remaining Mac acceptance

The current attended review is a preserved **unapproved replay**, so it cannot
supply the training correction required by #137. The next useful Mac work is to
review an eligible original recorded Sunday session, obtain actual transcript
and bilingual approval, then run the export → scratch merge → CPU trainer
preflight commands below. No new microphone capture is required to review an
existing recording. None of these missing approvals was manufactured here.

The [machine handoff receipt](correction-handoff.json) records inspected source
hashes, actual negative checks and remaining gates. The attended session
`20260910_092824_526799_en` has one final, revision 2, both approval flags false,
`session_kind=replay`, and unchanged predicted/corrected text. It demonstrates
saving a review draft, not a human correction. Its
[original evidence](../attended_mic_20260910/README.md) remains separate.

## Actual CPU-only rejection check

The original diagnostics, lifecycle, metadata, sidecar and retained WAV were
copied to `.cache/mac-en-es-closeout/correction-handoff-20260910/session_snapshot`.
The [exact audit source](correction-handoff-audit.py.txt) only operates on that
copy. `ReviewStore.export` rejects both `train` and `eval` exports with
“Approve at least one transcript or translation before exporting”. No bundle,
export registry, training corpus or approval was created. All five original
file hashes match after the check.

The legacy `prepare_finetune_data.py export-translation` and `export-whisper`
commands return exit 0 when they find zero eligible rows; neither creates an
output here. A zero exit from those commands alone is therefore not a corpus,
merge or retraining receipt. The live ReviewStore export API provides the
explicit-approval contract used below. `tools/export_review.py` is an Excel
workbook utility, not this portable training-bundle exporter. The legacy
`apply-corrections` subcommand rewrites diagnostics; it is not used in this
sidecar-preserving handoff. The legacy `summary` command can import
`dry_run_ab.py`, so it was not run during the coordinator's model timing slot.

Two concrete importer gaps were repaired in `tools/merge_corrections.py`:

- Explicit denied/unknown approval fields, unapproved status or exclusion now
  reject before corpus writes. Whisper requires only transcript approval when
  that field is present; translation additionally checks bilingual approval.
  Unmarked legacy/schema-2 training projections retain their format contract.
  `training_eligible=false`/unknown and `usage=evaluation_only` also refuse a
  supplied `session_kind=live` override.
- Missing/empty correction inputs reject instead of producing a successful
  zero-input write. Registered holdout files, including the v2 holdout and
  resolved aliases, cannot be training inputs or output targets.

The focused safety suite passes **39 tests**. It also retains valid idempotent
reimport, stale-revision and audio-identity checks. These are fixture tests,
separate from the actual unapproved-session rejection above. No source session
was edited or approved to make a test pass.

## Exact scratch workflow after real approval

First select a completed session whose original recording is eligible for
training and whose provenance is explicitly `live`, with both texts actually
approved in Review. Do not relabel this attended replay, public evaluation clips,
synthetic speech or an existing evaluation export. Confirm original recording
provenance/date against the 2026-03-14 training boundary and existing holdouts;
a replay run's date is not evidence of the original recording date. The original
Sunday source and human review receipt remain necessary for #137.

Set `REVIEW_SESSION` to that actual approved session ID and choose a new dated
scratch directory. The following source-only command reads the current review,
refuses to proceed without a bilingual approved live record, copies the session
and its existing approval/provenance records, and exports from the copy. It never
sets an approval flag or changes the originals. All selected audio must retain
its normal session-relative diagnostics path for a portable Whisper export.

```bash
export REVIEW_SESSION='ACTUAL_APPROVED_ELIGIBLE_SESSION_ID'
export CORRECTION_HANDOFF_DIR="$PWD/.cache/mac-en-es-closeout/corrections-20260910-approved-1"
/Users/willem/anaconda3/bin/python3.11 -S - <<'PY'
import hashlib, json, os, shutil
from pathlib import Path
from tools.review_data import ReviewStore, validate_export_download
from tools.session_lifecycle import require_completed
root = Path.cwd()
session = os.environ['REVIEW_SESSION']
destination = Path(os.environ['CORRECTION_HANDOFF_DIR'])
store = ReviewStore(root)
require_completed(root, session)
records = store.records(session)
eligible = [r for r in records if not r['excluded'] and r['transcript_approved'] and r['translation_approved']]
assert eligible, 'No actual bilingual-approved correction is available'
assert all(r['session_kind'] == 'live' for r in eligible), 'Replay/unknown provenance cannot be promoted'
paths = [store.diagnostics_path(session), store.sidecar_path(session),
         root / f'metrics/session_lifecycle_{session}.json', root / f'metrics/session_metadata_{session}.json']
for name in ['session_provenance.jsonl', 'export_registry.jsonl']:
    path = store.corrections / name
    if path.exists(): paths.append(path)
for record in records:
    if record['excluded'] or not record['transcript_approved']:
        continue
    try:
        audio = store.audio_path(session, record['chunk_id'], record)
    except FileNotFoundError:
        continue
    assert not Path(record['audio_path']).is_absolute(), 'Retain a portable session-relative audio path'
    paths.append(audio)
paths = list(dict.fromkeys(paths))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before = {str(p.relative_to(root)): sha(p) for p in paths}
destination.mkdir(parents=True, exist_ok=False)
snapshot = destination / 'snapshot'
for path in paths:
    copied = snapshot / path.relative_to(root)
    copied.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, copied)
result = ReviewStore(snapshot).export(session, split='train')
validate_export_download(Path(result['archive']), result['bundle_id'])
assert result['translation_pairs'] > 0
assert before == {str(p.relative_to(root)): sha(p) for p in paths}, 'Original changed during handoff'
result['original_source_hashes'] = before
(destination / 'export-receipt.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
PY
```

The command is conditional and **has not been executed**: no qualifying approved
session is currently identified. It uses the real existing review flags and the
exporter's independent completion/split/privacy validation. If the session has
an evaluation assignment anywhere in its records, the exporter refuses it.

After a real export, merge its translation projection against the real versioned
training corpus into a new output. The explicit checks below prevent a missing
base corpus from being mistaken for an intentional empty corpus. They do not
create, relabel or substitute a holdout.

```bash
set -euo pipefail
REVIEW_BUNDLE_ID="$(/Users/willem/anaconda3/bin/python3.11 -S -c \
  'import json,os,pathlib; print(json.loads((pathlib.Path(os.environ["CORRECTION_HANDOFF_DIR"])/"export-receipt.json").read_text())["bundle_id"])')"
export REVIEW_BUNDLE_ID
export REVIEW_BUNDLE_DIR="$CORRECTION_HANDOFF_DIR/snapshot/stark_data/corrections/exports/$REVIEW_BUNDLE_ID"
export CORRECTION_BASE_TRAIN="$PWD/bible_data/aligned/verse_pairs_train_v2.jsonl"
export CORRECTION_V2_HOLDOUT="$PWD/bible_data/aligned/verse_pairs_test_v2.jsonl"
test -s "$CORRECTION_BASE_TRAIN" && test -s "$CORRECTION_V2_HOLDOUT"
test -s "$REVIEW_BUNDLE_DIR/translation/train.jsonl"
/Users/willem/anaconda3/bin/python3.11 -S tools/merge_corrections.py translation \
  --corrections "$REVIEW_BUNDLE_DIR/translation/train.jsonl" \
  --train-jsonl "$CORRECTION_BASE_TRAIN" \
  --output "$CORRECTION_HANDOFF_DIR/translation_train_20260910.jsonl" \
  > "$CORRECTION_HANDOFF_DIR/merge-receipt.json"
```

Require a real sample ID/revision from the export in the resulting dated corpus,
`added` or `updated` greater than zero on the first import, and unchanged hashes
for the original base and holdout. A second import into that scratch output should
be idempotent. Do not call a merger run a retrain.

The actual Gemma trainer parser supports a one-step smoke configuration. Its
CPU-only dataset/configuration dry run reads the same merged input and checks all
available holdouts before any trainer import:

```bash
/Users/willem/anaconda3/bin/python3.11 -S tools/training_preflight.py gemma \
  --holdout "$CORRECTION_V2_HOLDOUT" \
  --report "$CORRECTION_HANDOFF_DIR/gemma_smoke_preflight.json" -- \
  --base unsloth/gemma-4-E4B-it \
  --train-data "$CORRECTION_HANDOFF_DIR/translation_train_20260910.jsonl" \
  --output "$CORRECTION_HANDOFF_DIR/gemma_smoke_adapter" \
  --lora-r 8 --lora-alpha 8 --max-steps 1 --warmup-steps 0 \
  --per-device-batch-size 1 --grad-accum 1 --no-packing
```

Retain full argv, timestamps, exit code, input/holdout hashes and the preflight
report. This is an actual parser/data dry run for the smoke trainer configuration;
model loading, tokenization, optimizer steps, adapter generation and CUDA behavior
remain untested. The future WSL command uses the exact trainer arguments after
`--` with `training/train_gemma4.py`; no GPU command runs on this Mac. A first
retrain dry run is permitted by #137, but a synthetic fixture or zero-input check
cannot replace its approved real-correction prerequisite.

An optional Whisper import uses the existing verified parser:
`tools/merge_corrections.py whisper --corrections "$REVIEW_BUNDLE_DIR"
--train-dir <existing-language-train-dir> --output-dir <new-scratch-train-dir>
--language en` (or `es` in a separate Spanish corpus). The complete W17 recipe
also requires its real base config, W16 adapter, Deepgram/chunk inputs and replay
corpus; its [runbook](../../wsl_pipeline_refresh.md) remains the execution source.

## Remaining runnable Mac work and external gates

| Gate | Current evidence | Exact next action / receipt |
|---|---|---|
| Review queue and draft persistence | Actual attended replay revision 2, unapproved; its review priority is 0 | Use **all captions** if flagged-only hides the item. The existing note is a persistence check, not a corrected training pair. |
| Recorded Sunday correction | No qualifying approved session identified | Human reviews an eligible original session, saves an actual correction and separate approvals; retain original provenance and revisioned sidecar. No new mic is required. |
| Portable export and dated scratch merge | Negative real-session check and positive fixtures exist | Run the conditional scratch workflow after approval; retain bundle manifest/ZIP hash, sample ID/revision, merge receipt and unchanged original hashes. |
| CPU smoke-trainer preflight | Implemented, fixtures pass | Provide the real v2 train/holdout files and run the exact one-step-config preflight above against the actual merged correction. Missing mounted files remain a failure. |
| CUDA smoke/actual retraining | Not run | WSL/A2000 window plus actual data/config/adapter inputs; retain loader, tokenizer, optimizer and resulting adapter receipts. |
| Attended EN/ES mic speech | Input/readiness passed, zero speech frames/finals | During an attended session speak EN and ES; retain captions, browser ACKs and review. This is #131, separate from correction import. |
| New candidate/quality comparisons | Coordinator owns currently running frozen-source jobs | Use their actual EN/ES results and later candidate replay; no parallel model/build/install jobs or promotion from this handoff. |

The canonical backlog and GitHub issues were not changed. Approval, data origin,
CUDA execution and attended speech gates keep their actual pending state.
