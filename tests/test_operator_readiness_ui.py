"""Readiness drives the actual header, live panel and recovery controls."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_operator(script):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to execute the operator UI")
    # A collaborating worktree may test its controller against the already
    # integrated page/harness without copying or editing those owned fixtures.
    fixtures = Path(os.environ.get("STARK_OPERATOR_TEST_FIXTURES", str(ROOT)))
    bootstrap = f"""
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const appSource = fs.readFileSync({json.dumps(str(ROOT / "displays/operator/app.js"))}, 'utf8');
const fixtures = {json.dumps(str(fixtures))};
const originalRead = fs.readFileSync.bind(fs);
fs.readFileSync = (name, ...args) => path.resolve(String(name)) === path.join(fixtures, 'displays/operator/app.js')
  ? appSource : originalRead(name, ...args);
const {{createHarness, realHealth, settle}} = require(path.join(fixtures, 'tests/frontend/operator_harness.js'));
"""
    result = subprocess.run(
        [
            node,
            "-e",
            bootstrap + "(async () => {\n" + script + "\n})().catch(e => {console.error(e); process.exit(1);});",
        ],
        cwd=fixtures,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_main_status_distinguishes_initial_health_wait_input_error_staleness_and_recovery():
    run_operator(r"""
const h = createHarness();
await h.start();
const base = {session_id: 's1', config: {lang: 'en'}, state: 'starting'};
await h.setStatus({...base, readiness: {ready: false, phase: 'unknown', stale: true, age_s: null, updated_at: null, reason: 'Pipeline health is unavailable'}});
assert.strictEqual(h.text('state-pill'), 'Starting…');
assert(h.text('pipeline-readiness').startsWith('Waiting for the caption process'));
assert(!h.text('pipeline-readiness').includes('just now'));
assert.strictEqual(h.el('stop-btn').disabled, false);

await h.setStatus({...base, state: 'running', readiness: {ready: false, phase: 'listening', stale: false, updated_at: 1, reason: 'No audio frames have arrived'}});
assert.strictEqual(h.text('state-pill'), 'Waiting for input');
assert.strictEqual(h.text('live-title'), 'Waiting for input');
assert(h.text('state-detail').includes('No audio frames have arrived'));
assert(h.text('live-subtitle').includes('Check the selected sound input'));
assert(!h.text('live-subtitle').includes('Captions are being sent'));
assert.strictEqual(h.el('stop-btn').disabled, false);

await h.setStatus({...base, state: 'running', readiness: {ready: false, phase: 'input_error', stale: false, updated_at: 2, reason: 'Permission denied'}});
assert.strictEqual(h.text('state-pill'), 'Needs attention');
assert.strictEqual(h.text('live-title'), 'Needs attention');
assert(h.el('state-pill').className.includes('bad'));
assert(h.text('live-subtitle').includes('Permission denied'));
assert(h.text('live-subtitle').includes('microphone permission'));
assert.strictEqual(h.el('stop-btn').disabled, false);

// A previously reporting process is unhealthy even when its latest status
// loses the timestamp; it must not fall back to the initial waiting state.
await h.setStatus({...base, state: 'running', readiness: {ready: true, phase: 'ready', stale: true, age_s: null, updated_at: null, reason: 'Status file unreadable'}});
assert.strictEqual(h.text('state-pill'), 'Needs attention');
assert.strictEqual(h.text('live-title'), 'Needs attention');
assert(h.text('live-subtitle').includes('Status file unreadable'));
assert(h.text('live-subtitle').includes('Use Stop'));
assert(!h.text('pipeline-readiness').includes('just now'));

await h.setStatus({...base, state: 'running', readiness: {ready: true, phase: 'ready', stale: false, updated_at: 3}, last_event: 'Models loaded; waiting for audio readiness'});
assert.strictEqual(h.text('state-pill'), 'Live');
assert.strictEqual(h.text('live-title'), 'Live');
assert.strictEqual(h.el('live-event').hidden, true);
assert.strictEqual(h.text('pipeline-readiness'), 'Captions are ready.');

// A new session resets the earlier-health observation.
await h.setStatus({...base, session_id: 's2', state: 'running', readiness: {ready: false, stale: true, updated_at: null, age_s: null}});
assert.strictEqual(h.text('state-pill'), 'Waiting for status');
assert.strictEqual(h.text('live-title'), 'Waiting for status');

// Old servers without readiness retain the established contract.
await h.setStatus({...base, session_id: 'legacy', state: 'running'});
assert.strictEqual(h.text('state-pill'), 'Live');
assert.strictEqual(h.text('live-title'), 'Live');
assert.strictEqual(h.el('pipeline-readiness').hidden, true);
""")


def test_acknowledged_pause_resume_and_disabled_chunk_saving_have_no_stale_claims():
    run_operator(r"""
const h = createHarness();
await h.start();
const base = {session_id: 's1', config: {lang: 'en'}};
await h.setStatus({...base, state: 'paused', readiness: {ready: false, phase: 'paused', stale: false, updated_at: 1}, last_event: 'Pause requested; waiting for pipeline acknowledgment'});
assert.strictEqual(h.text('state-pill'), 'Paused');
assert.strictEqual(h.text('live-title'), 'Paused');
assert.strictEqual(h.text('pipeline-readiness'), 'Captions are paused.');
assert.strictEqual(h.el('live-event').hidden, true);
assert.strictEqual(h.el('stop-btn').disabled, false);

await h.setStatus({...base, state: 'running', readiness: {ready: true, phase: 'ready', stale: false, updated_at: 2}, last_event: 'Resume requested; waiting for pipeline acknowledgment', health: realHealth({recording: {audio_enabled: false, ok: true, required_failures: 0}})});
assert.strictEqual(h.text('live-title'), 'Live');
assert.strictEqual(h.el('live-event').hidden, true);
const chips = h.el('health-list').children.map(li => li.textContent);
assert(chips.includes('Original chunk audio is not being saved'));
assert(!chips.includes('Not recording audio'));
assert.strictEqual(h.el('stop-btn').disabled, false);

// Current meaningful events remain visible.
await h.setStatus({...base, state: 'running', readiness: {ready: true, phase: 'ready', stale: false, updated_at: 3}, last_event: 'Input device recovered'});
assert.strictEqual(h.text('live-event'), 'Input device recovered');
assert.strictEqual(h.el('live-event').hidden, false);
""")
