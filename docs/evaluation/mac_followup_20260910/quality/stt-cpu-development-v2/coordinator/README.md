# CPU quality recovery and adaptive Lite v2

These are new scratch coordinators. The original Standard96, Spanish18 and
Lite-cadence24 cohorts are read-only completed inputs; none is scheduled again.
The original CPU-quality v1 cohort and its 600 missing-SoundFile error rows,
failed serial-queue receipt and failed waiting-adaptive receipt are preserved.
No device access is allowed. The execution flags authorize only a model window
already reserved by the root coordinator; preparation alone never runs models.

## CPU predecessor

`run_cpu_quality_recovery_v2.py` uses the unchanged `eddb0ad` runtime and fresh
full CPU evaluation interpreter:

```text
.cache/mac-en-es-closeout/quality/cpu-eval-environment-v2/venv/bin/python
```

Its only execution commands are 12 serial CPU STT workers (small/base × EN/ES ×
three repeats, 50 development recordings each) and the quality report. Source
and model paths match the original protocol. `base` is quality assessment only;
it does not silently replace the live Lite model. This runner checks all three
completed latency cohorts and reproduces their existing selections before and
after quality execution. It never calls their runner main functions.

New quality paths:

```text
.cache/mac-en-es-closeout/quality/stt-cpu-development-v2/index.json
.cache/mac-en-es-closeout/quality/stt-cpu-development-v2-report.json
.cache/mac-en-es-closeout/cpu-quality-recovery-v2-state.json
```

The terminal state must be `completed`, preserve defaults and original failures,
and pass the real preparation helper's quality/source/model/reference checks.
It records the coordinator's PID, process creation time and exact argv in
`owner`. A failed worker, report, input binding, cleanup, or final validation
stops the recovery. Outputs/logs use fresh paths and exclusive creation.

Exact reviewed launch argv (root coordinator only, once):

```bash
/Users/willem/Code/vibes/SRTranslate/stt_env/bin/python /Users/willem/Code/vibes/SRTranslate/.cache/mac-en-es-closeout/recovery-preparation/run_cpu_quality_recovery_v2.py --output /Users/willem/Code/vibes/SRTranslate/.cache/mac-en-es-closeout/cpu-quality-recovery-v2 --execute-reserved-serial-window
```

The root has launched this predecessor after review. Its state file is the
source of progress; do not rerun that command or edit the bound predecessor,
original helper scripts, frozen runtime, completed cohorts, or failed receipts.

## Adaptive waiter

`execute_lite_recovery_v2.py` requires a filled copy of
`inputs.recovery.example.json`. Keep the template unchanged. The template has
all fixed v2 paths; only the actual predecessor `pid` and `created_at` are null.
Copy the three ownership fields directly from the new CPU state's `owner`:

```python
import json
from pathlib import Path

root = Path('/Users/willem/Code/vibes/SRTranslate')
cache = root / '.cache/mac-en-es-closeout'
prep = cache / 'recovery-preparation'
inputs = json.loads((prep / 'inputs.recovery.example.json').read_text())
state = json.loads((cache / 'cpu-quality-recovery-v2-state.json').read_text())
for field in ('pid', 'created_at', 'argv'):
    inputs['recovery_gate'][field] = state['owner'][field]
with (prep / 'inputs.recovery.v2.json').open('x') as stream:
    stream.write(json.dumps(inputs, indent=2) + '\n')
```

The root may then launch the waiter once with absolute paths and a fresh output:

```bash
/Users/willem/Code/vibes/SRTranslate/stt_env/bin/python /Users/willem/Code/vibes/SRTranslate/.cache/mac-en-es-closeout/recovery-preparation/execute_lite_recovery_v2.py --inputs /Users/willem/Code/vibes/SRTranslate/.cache/mac-en-es-closeout/recovery-preparation/inputs.recovery.v2.json --output /Users/willem/Code/vibes/SRTranslate/.cache/mac-en-es-closeout/lite-adaptive-followups-v2 --execute-reserved-serial-window
```

The waiter verifies PID creation time and exact argv, observes descendant
identity, waits until the coordinator exits, and requires the matching successful
terminal state with 12 completed quality runs / 600 completed rows. Inventory
failure, PID reuse, or a surviving observed child never grants a model window.
The completed state becomes a fixed hash binding before preparation starts.

The unchanged original `prepare.py` freezes/validates the actual completed
inputs. The new coordinator then permits exactly:

1. EN deadline100/deadline250 with unchanged 0.6-second cadence, three repeats and
   opening/closing controls: 12 runs.
2. The same independently in ES: 12 runs.
3. Reproducible report and selection for each language, then a new preparation
   pass containing two formal `no_combination` plans and **zero model commands**.

The retained cadence selection is 0/4 qualified, so an interaction model run is
explicitly rejected. A successful independent deadline screen cannot invent a
qualified cadence component. No confirmation, default change, base-model live
implementation, or p95 certification is implied.

## Interpreter and safety boundaries

Adaptive replays keep the original fully validated Lite interpreter:
`.cache/overnight-20260910/lite-runtime/bin/python`. This preserves dependency
scope with the completed cadence cohort. The new CPU quality interpreter is
separate; its installation receipt confirms original package versions plus
added evaluation dependencies, CPU only, and silent EN/ES file decoding. It is
not silently substituted into cadence comparisons.

Both coordinators use frozen-source checks and the reviewed process cleanup
semantics, with birth-time tracking and cleanup even after ordinary nonzero
exit, timeout or interruption. The adaptive driver's seven validation/cleanup
function bodies exactly match the original reviewed driver. Declared command
module, interpreter, cwd, environment, ordering, count, deadline and output
bindings are verified. Generated input paths, bundles, specs and terminal plans
are bound before/after their use. Ambient `STARK_*`, `PYTHONPATH` and `PYTHONHOME`
are removed; actual prepared replay argv must retain explicit `--no-tts`, file
input and fixed gain. No microphone, native input/output test, speaker playback,
network model download, install, or prior-cohort rerun is scheduled.

`check_recovery.py` executes only tiny stdlib fixtures with actual pure
preparation/configuration code and fake process inventories. It verifies exact
CPU commands, 24-run replay command construction, no-combination behavior,
predecessor identity/terminal requirements, and failure guards. It asserts that
NumPy, Torch, MLX, sounddevice, faster-whisper and psutil were not imported.
`validation.json` records 25 passed fixtures. Ruff passed for the new scripts.
These are source/fixture checks, not completed experiment evidence.
