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


def test_selected_microphone_follows_identity_across_device_index_changes_and_missing_inventory():
    run_operator(r"""
const h = createHarness({capabilities: {audio_tests: true}, devices: {inputs: [
  {index: 0, name: 'MacBook Pro Microphone', host_api: 'Core Audio', channels: 1}
], outputs: []}});
await h.start();
h.el('mic-device').value = '0';
h.state.devices.inputs = [
  {index: 0, name: 'WR17.1 Microphone', host_api: 'Core Audio', channels: 1},
  {index: 1, name: 'MacBook Pro Microphone', host_api: 'Core Audio', channels: 1}
];
await h.app.refreshDevices(false);
assert.strictEqual(h.app.readForm().mic_device, 1);
assert.strictEqual(h.app.readForm().mic_device_name, 'MacBook Pro Microphone');
assert.strictEqual(h.app.readForm().mic_host_api, 'Core Audio');
await h.app.refreshPreflight();
const preflight = new URLSearchParams(h.lastFetch('/api/preflight', 'GET').url.split('?')[1]);
assert.strictEqual(preflight.get('input_device_name'), 'MacBook Pro Microphone');
assert.strictEqual(preflight.get('input_host_api'), 'Core Audio');

// A disappeared mic remains explicitly selected, so both start and test-input
// reject it rather than adopting WR17.1 or the computer default.
h.state.devices.inputs = h.state.devices.inputs.slice(0, 1);
await h.app.refreshDevices(false);
assert.strictEqual(h.app.readForm().mic_device, undefined);
assert.strictEqual(h.app.readForm().mic_device_name, 'MacBook Pro Microphone');
h.route('POST', '/api/audio/test-input', () => h.response({ok: false}, 422));
h.el('mic-test-btn').click();
await settle();
const body = JSON.parse(h.lastFetch('/api/audio/test-input').init.body);
assert.strictEqual(body.device, null);
assert.strictEqual(body.device_name, 'MacBook Pro Microphone');
assert.strictEqual(body.device_host_api, 'Core Audio');
""")


def test_status_preserves_full_unavailable_identity_and_bounds_ambiguous_placeholders():
    run_operator(r"""
const mic = (index, api) => ({index, name: 'Shared Mic', host_api: api, channels: 1});
const h = createHarness({devices: {inputs: [mic(0, 'Core Audio'), mic(1, 'Core Audio')], outputs: []}});
await h.start();
const status = api => ({state: 'running', session_id: 's1', config: {
  lang: 'en', mic_device: 8, mic_device_name: 'Shared Mic', mic_host_api: api
}});
for (let i = 0; i < 10; i++) await h.setStatus(status('Core Audio'));
assert.strictEqual(h.el('mic-device').options.length, 4); // default + 2 real + 1 placeholder
assert.strictEqual(h.app.readForm().mic_device, undefined);
assert.strictEqual(h.app.readForm().mic_host_api, 'Core Audio');
await h.setStatus(status('Other API'));
assert.strictEqual(h.el('mic-device').options.length, 4);
assert.strictEqual(h.app.readForm().mic_host_api, 'Other API');
// A service restart can restore the identity before device enumeration finishes.
h.state.devices.inputs = [mic(7, 'Other API')];
await h.app.refreshDevices(false);
assert.strictEqual(h.app.readForm().mic_device, 7);
assert.strictEqual(h.app.readForm().mic_host_api, 'Other API');
""")
