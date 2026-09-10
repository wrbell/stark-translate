"""Drive the real operator page (index.html + app.js + widgets) under Node.

Every test loads ``displays/operator/index.html`` into the small DOM in
``tests/frontend/fake_dom.js`` and runs the shipped JavaScript against routed
fetch/WebSocket fakes. The fixtures ``realCapabilities`` and ``realHealth``
mirror what ``operator_app`` emits today; older/legacy shapes are exercised
alongside them. Nothing here talks to a server, opens a device or loads a model.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRELUDE = (
    "const {createHarness, settle, deferred, response, preflightPayload, realCapabilities, realHealth} = "
    "require(process.cwd() + '/tests/frontend/operator_harness.js');\n"
    "const assert = require('assert');\n"
)


def run_node(script: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to execute the operator UI")
    wrapped = (
        PRELUDE + "(async () => {\n" + script + "\n})().catch(error => { console.error(error); process.exit(1); });"
    )
    result = subprocess.run([node, "-e", wrapped], cwd=ROOT, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        pytest.fail(f"node exited {result.returncode}\n{result.stdout}\n{result.stderr}")


def test_tabs_keep_jargon_in_advanced_and_stay_keyboard_accessible():
    run_node(r"""
const h = createHarness();
const tabs = h.document.querySelectorAll('[role="tab"]');
assert.deepStrictEqual(tabs.map(t => t.dataset.tab), ['prepare', 'live', 'sessions', 'help', 'advanced']);
for (const tab of tabs) {
  const panel = h.document.getElementById(tab.getAttribute('aria-controls'));
  assert(panel, `panel for ${tab.id}`);
  assert.strictEqual(panel.getAttribute('role'), 'tabpanel');
  assert.strictEqual(panel.getAttribute('aria-labelledby'), tab.id);
}
// Implementation vocabulary is confined to the Advanced tab. The Prepare panel is scanned before the
// server's own check details render: those raw details stay visible on purpose for the setup owner.
const jargon = [/\bvad\b/i, /backend/i, /\bmlx\b/i, /\bcuda\b/i, /llama/i, /diariz/i, /a\/b/i, /\bhf\b/i, /vram/i,
  /\bp50\b/i, /websocket/i, /\bengine\b/i, /preflight/i, /subprocess/i, /sigstop/i, /gemma/i, /whisper/i, /parakeet/i];
const scan = name => {
  const text = h.panelText(name);
  for (const pattern of jargon) assert(!pattern.test(text), `${name} panel exposes ${pattern}`);
};
scan('prepare');
await h.start();
assert.strictEqual(h.app.activeTab, 'prepare');
assert.strictEqual(h.el('panel-live').hidden, true);
await h.el('tab-live').click();
assert.strictEqual(h.el('panel-live').hidden, false);
assert.strictEqual(h.el('panel-prepare').hidden, true);
assert.strictEqual(h.el('tab-live').getAttribute('aria-selected'), 'true');
assert.strictEqual(h.el('tab-prepare').getAttribute('aria-selected'), 'false');
assert.strictEqual(h.el('tab-prepare').tabIndex, -1);
assert.strictEqual(h.storage.get('stark-operator-tab'), 'live');
const tablist = h.document.querySelector('[role="tablist"]');
await tablist.dispatchEvent(new h.context.Event('keydown', {key: 'ArrowRight'}));
assert.strictEqual(h.app.activeTab, 'sessions');
assert.strictEqual(h.document.activeElement, h.el('tab-sessions'));
await tablist.dispatchEvent(new h.context.Event('keydown', {key: 'End'}));
assert.strictEqual(h.app.activeTab, 'advanced');
await tablist.dispatchEvent(new h.context.Event('keydown', {key: 'ArrowRight'}));
assert.strictEqual(h.app.activeTab, 'prepare');
for (const name of ['live', 'sessions', 'help']) scan(name);
const prepareRows = h.el('checks').children;
assert(prepareRows.length > 0);
for (const row of prepareRows) row.querySelector('.detail').remove();
scan('prepare'); // the page's own Prepare wording stays plain once server details are set aside
const advanced = h.panelText('advanced');
for (const word of ['Backend', 'Engine', 'VAD', 'A/B', 'diarization', 'VRAM']) assert(advanced.includes(word), word);
// Technical controls outside the Prepare form still belong to the configuration form.
const names = Array.from(h.el('config-form').elements).map(e => e.name);
for (const name of ['lang', 'mic_device', 'tts', 'tts_output_mode', 'output_device', 'tts_device_en', 'tts_device_es',
  'backend', 'engine', 'vad_threshold', 'run_ab', 'diarize', 'profile']) assert(names.includes(name), name);
// Screen-reader affordances.
assert.strictEqual(h.el('state-pill').getAttribute('role'), 'status');
assert.strictEqual(h.el('readiness-summary').getAttribute('aria-live'), 'polite');
assert.strictEqual(h.el('attention').getAttribute('role'), 'alert');
assert.strictEqual(h.el('review-status').getAttribute('aria-live'), 'polite');
// Review markup keeps every id review.js binds to.
for (const id of ['session', 'pending', 'flagged', 'refresh', 'previous-page', 'next-page', 'restore', 'status', 'items',
  'editor', 'context', 'audio', 'audio-status', 'language', 'source-label', 'source', 'transcript-approved', 'target-label',
  'target', 'translation-approved', 'original', 'note', 'excluded', 'save', 'next', 'split', 'export', 'download']) h.el(`review-${id}`);
// The chosen tab survives a reload.
const again = createHarness({storage: [...h.storage]});
assert.strictEqual(again.app.activeTab, 'prepare');
""")


def test_preflight_failure_clears_the_ready_flag_immediately():
    run_node(r"""
const h = createHarness();
await h.start();
assert.strictEqual(h.app.preflightOk, true);
assert.strictEqual(h.el('start-btn').disabled, false);
assert.strictEqual(h.text('state-pill'), 'Ready');
assert.strictEqual(h.text('readiness-summary'), 'Ready to start — 1 warning to be aware of.');
h.route('GET', '/api/preflight', () => { throw new Error('connection refused'); });
await h.app.refreshPreflight(); await settle();
assert.strictEqual(h.app.preflightOk, false);
assert.strictEqual(h.el('start-btn').disabled, true);
assert(h.text('readiness-summary').startsWith("Couldn't check readiness"), h.text('readiness-summary'));
assert.strictEqual(h.text('state-pill'), 'Not ready');
// A failing server check is explained in plain words while the raw detail stays visible.
h.route('GET', '/api/preflight', () => response(preflightPayload({checks: [
  {name: 'Microphone', status: 'fail', detail: 'No input devices found'},
  {name: 'Models', status: 'pass', detail: 'mlx-parakeet-v3, mlx-gemma4-e4b'}]})));
await h.app.refreshPreflight(); await settle();
assert.strictEqual(h.text('readiness-summary'), '1 problem to fix before starting.');
const rows = h.el('checks').children;
assert.strictEqual(rows[0].querySelector('.name').textContent, 'Microphone');
assert(rows[0].querySelector('.advice').textContent.includes('Plug in the USB microphone'));
assert.strictEqual(rows[0].querySelector('.detail').textContent, 'No input devices found');
assert.strictEqual(rows[1].querySelector('.name').textContent, 'Language models');
assert.strictEqual(h.el('start-btn').disabled, true);
assert(h.text('preflight-detail').includes('No input devices found'));
// Lite check names get plain titles too.
h.route('GET', '/api/preflight', () => response(preflightPayload({backend: 'cpu', checks: [
  {name: 'Lite hardware', status: 'fail', detail: '2 physical cores, 7.5 GiB RAM; lite-cpu floor 4 cores / 8 GiB.'},
  {name: 'Managed llama-server', status: 'pass', detail: 'llama-server (b10883)'},
  {name: 'Diarization', status: 'fail', detail: 'Diarization is outside the lite memory profile'}]})));
await h.app.refreshPreflight(); await settle();
assert.deepStrictEqual(h.el('checks').children.map(li => li.querySelector('.name').textContent),
  ['Computer memory and cores', 'Translation server (managed)', 'Speaker labels']);
assert(h.text('preflight-meta').includes('processor only'));
// A slower, older check cannot resurrect the ready flag after a newer failure.
const slow = deferred();
h.route('GET', '/api/preflight', () => slow.promise);
const first = h.app.refreshPreflight();
h.route('GET', '/api/preflight', () => { throw new Error('offline'); });
await h.app.refreshPreflight();
slow.resolve(response(preflightPayload()));
await first; await settle();
assert.strictEqual(h.app.preflightOk, false);
assert.strictEqual(h.el('start-btn').disabled, true);
// Readiness is not re-polled while captions run; stopping triggers a fresh check.
h.unroute('GET', '/api/preflight');
await h.setStatus({state: 'running', config: {lang: 'en'}});
const before = h.fetchLog.filter(f => f.url.startsWith('/api/preflight')).length;
h.runIntervals(); await settle();
assert.strictEqual(h.fetchLog.filter(f => f.url.startsWith('/api/preflight')).length, before);
await h.setStatus({state: 'idle', outcome: 'completed', config: {lang: 'en'}});
assert.strictEqual(h.fetchLog.filter(f => f.url.startsWith('/api/preflight')).length, before + 1);
assert.strictEqual(h.app.preflightOk, true);
""")


def test_status_failure_shows_disconnected_instead_of_green_running():
    run_node(r"""
const h = createHarness({status: {state: 'running', session_id: 's1', started_at: '2026-09-09T10:00:00', config: {lang: 'en'}}});
await h.start();
assert.strictEqual(h.text('state-pill'), 'Live');
assert(h.el('state-pill').className.includes('ok'));
assert.strictEqual(h.app.activeTab, 'live'); // reloading during a live session lands on Live
h.route('GET', '/api/session/status', () => { throw new Error('socket hang up'); });
await h.app.refreshStatus(); await settle();
assert.strictEqual(h.text('state-pill'), 'Not connected');
assert(!h.el('state-pill').className.includes('ok'));
assert(h.el('state-pill').className.includes('stale'));
assert(h.text('state-detail').includes('Last known: Live'), h.text('state-detail'));
assert(h.text('connection-age').startsWith('Not connected'), h.text('connection-age'));
assert(h.el('connection-age').className.includes('stale'));
assert.strictEqual(h.text('live-title'), 'Not connected');
assert.strictEqual(h.el('pause-btn').disabled, true);
assert.strictEqual(h.el('flip-btn').disabled, true);
assert.strictEqual(h.el('stop-btn').disabled, false); // recovery stays possible
assert.strictEqual(h.el('start-btn').disabled, true);
// A server error is also not "running".
h.route('GET', '/api/session/status', () => response({detail: 'boom'}, 500));
await h.app.refreshStatus(); await settle();
assert.strictEqual(h.text('state-pill'), 'Not connected');
// Recovery restores the live view.
h.unroute('GET', '/api/session/status');
await h.app.refreshStatus(); await settle();
assert.strictEqual(h.text('state-pill'), 'Live');
assert.strictEqual(h.el('pause-btn').disabled, false);
assert(h.text('connection-age').startsWith('Connected'), h.text('connection-age'));
// Silence from the poller also turns stale, from the age ticker alone.
h.clock.value += 7000;
h.app.renderConnectionAge();
assert.strictEqual(h.text('state-pill'), 'Not connected');
assert(h.text('connection-age').includes('No status update'), h.text('connection-age'));
assert.strictEqual(h.el('pause-btn').disabled, true);
// A stale poll resolving after a newer control response cannot flip the state back.
const slow = deferred();
h.route('GET', '/api/session/status', () => slow.promise);
const poll = h.app.refreshStatus();
h.route('POST', '/api/session/stop', () => response({state: 'idle', outcome: 'completed'}));
await h.el('stop-btn').click();
assert.strictEqual(h.app.state, 'idle');
assert.strictEqual(h.text('state-pill'), 'Ready');
assert(h.text('state-detail').includes('finished normally'));
assert(h.text('connection-age').startsWith('Connected'));
slow.resolve(response({state: 'running', config: {lang: 'en'}}));
await poll; await settle();
assert.strictEqual(h.app.state, 'idle');
assert.strictEqual(h.text('state-pill'), 'Ready');
""")


def test_error_state_keeps_stop_available_and_error_persistent():
    run_node(r"""
const h = createHarness();
await h.start();
const crash = 'subprocess exited rc=1 unexpectedly\nModuleNotFoundError: mlx';
await h.setStatus({state: 'error', error: crash, last_event: 'subprocess crashed rc=1', config: {lang: 'en'}});
assert.strictEqual(h.text('state-pill'), 'Needs attention');
assert.strictEqual(h.el('attention').hidden, false);
assert(h.text('attention-text').includes('ModuleNotFoundError'));
assert.strictEqual(h.el('attention-stop').hidden, false);
assert.strictEqual(h.el('attention-stop').disabled, false);
assert.strictEqual(h.el('stop-btn').disabled, false);
assert.strictEqual(h.el('start-btn').disabled, false); // readiness is still green
assert.strictEqual(h.text('live-title'), 'Needs attention');
assert.strictEqual(h.text('live-event'), 'subprocess crashed rc=1');
// The banner persists across polls until the state changes.
await h.setStatus({state: 'error', error: crash, config: {lang: 'en'}});
assert.strictEqual(h.el('attention').hidden, false);
await h.el('attention-dismiss').click();
assert.strictEqual(h.el('attention').hidden, true);
await h.setStatus({state: 'error', error: crash, config: {lang: 'en'}});
assert.strictEqual(h.el('attention').hidden, true); // the same error stays dismissed…
assert(h.text('live-subtitle').includes('subprocess exited')); // …but the Live tab keeps showing it
await h.setStatus({state: 'error', error: 'pipeline did not stop within 10s', config: {lang: 'en'}});
assert.strictEqual(h.el('attention').hidden, false); // a different error re-surfaces
// "Stop and reset" posts the real stop endpoint; success clears the banner and reports the outcome.
h.route('POST', '/api/session/stop', () => response({state: 'idle', outcome: 'failed',
  last_event: 'Session stopped with errors; review is available, export is blocked'}));
await h.el('attention-stop').click(); await settle();
assert.strictEqual(h.lastFetch('/api/session/stop').url, '/api/session/stop');
assert.strictEqual(h.el('attention').hidden, true);
assert.strictEqual(h.text('state-pill'), 'Stopped');
assert(h.text('state-detail').includes('failed'));
assert.strictEqual(h.el('stop-btn').disabled, true);
// A failed stop is reported as a failure and the state is re-polled, never assumed.
await h.setStatus({state: 'error', error: 'stuck', config: {lang: 'en'}});
h.route('POST', '/api/session/stop', () => response({detail: 'cannot stop'}, 500));
await h.el('stop-btn').click(); await settle();
assert.strictEqual(h.text('attention-title'), "Couldn't stop captions");
assert(h.text('attention-text').includes('cannot stop'));
assert.strictEqual(h.app.state, 'error');
assert.strictEqual(h.el('stop-btn').disabled, false);
await h.el('attention-dismiss').click();
await h.setStatus({state: 'error', error: 'stuck', config: {lang: 'en'}});
assert.strictEqual(h.text('attention-title'), 'Captions need attention'); // state error returns after dismissal
""")


def test_delayed_responses_never_overwrite_newer_choices():
    run_node(r"""
const h = createHarness();
await h.start();
// 1. A slow device listing must not clobber a microphone chosen meanwhile.
const slowDevices = deferred();
h.route('GET', '/api/devices', () => slowDevices.promise);
const listing = h.app.refreshDevices(false);
h.unroute('GET', '/api/devices');
h.state.devices = {inputs: [{index: 1, name: 'USB Mic', channels: 1}, {index: 3, name: 'Yeti', channels: 2}], outputs: [], change_seq: 1};
await h.app.refreshDevices(false);
h.el('mic-device').value = '3';
assert.strictEqual(h.el('mic-device').value, '3');
slowDevices.resolve(response({inputs: [{index: 1, name: 'USB Mic', channels: 1}], outputs: [], change_seq: 0}));
await listing; await settle();
assert.strictEqual(h.el('mic-device').value, '3');
assert(h.el('mic-device').options.some(o => o.value === '3' && o.textContent.startsWith('Yeti')));
// Duplicate device names are disambiguated with their index; unique ones are not.
h.state.devices = {inputs: [{index: 1, name: 'USB Audio', channels: 1}, {index: 2, name: 'USB Audio', channels: 1}, {index: 3, name: 'Yeti', channels: 2}], outputs: []};
await h.app.refreshDevices(false);
assert.deepStrictEqual(h.el('mic-device').options.map(o => o.textContent), ['Automatic (computer default)', 'USB Audio (1ch) #1', 'USB Audio (1ch) #2', 'Yeti (2ch)']);
// 2. A slow capabilities probe cannot undo a newer one.
const slowCaps = deferred();
h.route('GET', '/api/capabilities', () => slowCaps.promise);
const probe = h.app.probeCapabilities();
h.route('GET', '/api/capabilities', () => response(realCapabilities()));
await h.app.probeCapabilities();
assert.strictEqual(h.el('profile-field').hidden, false);
slowCaps.resolve(response({detail: 'Not Found'}, 404));
await probe; await settle();
assert.strictEqual(h.el('profile-field').hidden, false);
// 3. A stale status poll cannot overwrite the language confirmed by a newer control response.
await h.setStatus({state: 'running', config: {lang: 'es'}});
assert.strictEqual(h.text('flip-btn'), 'Switch to English speaker');
const slowStatus = deferred();
h.route('GET', '/api/session/status', () => slowStatus.promise);
const poll = h.app.refreshStatus();
h.route('POST', '/api/control/lang_flip', () => response({state: 'starting', config: {lang: 'en'}}));
await h.el('flip-btn').click();
assert.strictEqual(h.el('lang-select').value, 'en');
assert.strictEqual(h.text('flip-btn'), 'Switch to Spanish speaker');
slowStatus.resolve(response({state: 'running', config: {lang: 'es'}}));
await poll; await settle();
assert.strictEqual(h.el('lang-select').value, 'en');
assert.strictEqual(h.app.state, 'starting');
// 4. A slow storage read cannot repaint over a newer one.
h.unroute('GET', '/api/session/status');
const slowStorage = deferred();
h.route('GET', '/api/storage', () => slowStorage.promise);
const reading = h.app.refreshStorage();
h.route('GET', '/api/storage', () => response({free_bytes: 5 * 1024 ** 3, used_bytes: 1024 ** 3, sessions: 2}));
await h.app.refreshStorage();
assert(h.text('storage-summary').includes('2 sessions stored'));
slowStorage.resolve(response({free_bytes: 1, used_bytes: 1, sessions: 99}));
await reading; await settle();
assert(h.text('storage-summary').includes('2 sessions stored'));
""")


def test_start_failures_are_explained_and_never_shown_as_progress():
    run_node(r"""
const h = createHarness();
await h.start();
await h.el('tab-live').click();
h.route('POST', '/api/session/start', () => response({detail: {code: 'preflight_failed',
  message: 'The selected configuration is not ready. Resolve the failed checks before starting.',
  checks: [{name: 'Microphone', status: 'fail', detail: 'No input devices found'}]}}, 422));
await h.el('start-btn').click(); await settle();
assert.strictEqual(h.app.preflightOk, false);
assert.strictEqual(h.el('start-btn').disabled, true);
assert.strictEqual(h.text('attention-title'), "Can't start yet");
assert(h.text('attention-text').includes('not ready'));
assert.strictEqual(h.app.activeTab, 'prepare');
assert.strictEqual(h.text('readiness-summary'), '1 problem to fix before starting.');
assert.strictEqual(h.text('start-status'), 'Not started.');
assert.strictEqual(h.text('state-pill'), 'Not ready');
assert.strictEqual(h.app.state, 'idle');
// Busy server: explained with the kind of work, not retried silently.
await h.app.refreshPreflight(); await settle();
assert.strictEqual(h.el('start-btn').disabled, false);
h.route('POST', '/api/session/start', () => response({detail: {code: 'work_busy', message: 'Finish summary before starting another job',
  work: {kind: 'summary', id: 'x1'}}}, 409));
await h.el('start-btn').click(); await settle();
assert.strictEqual(h.text('attention-title'), 'The operator service is busy');
assert(h.text('attention-advice').includes('busy with summary'));
assert.strictEqual(h.app.state, 'idle');
assert.strictEqual(h.el('start-btn').disabled, false);
// Current and older servers answer 409 with a plain string.
h.route('POST', '/api/session/start', () => response({detail: 'A session is already running'}, 409));
await h.el('start-btn').click(); await settle();
assert(h.text('attention-text').includes('already running'));
// A string 422 is the server refusing contradictory settings; a validation array is a plain failure.
h.route('POST', '/api/session/start', () => response({detail: 'Profile lite-cpu requires backend cpu, not mlx'}, 422));
await h.el('start-btn').click(); await settle();
assert.strictEqual(h.text('attention-title'), "The selected settings can't be used together");
assert.strictEqual(h.app.activeTab, 'prepare');
h.route('POST', '/api/session/start', () => response({detail: {code: 'profile_unavailable', message: 'Install the selected runtime profile'}}, 422));
await h.el('start-btn').click(); await settle();
assert.strictEqual(h.text('attention-title'), 'This profile is not installed');
h.route('POST', '/api/session/start', () => response({detail: [{loc: ['body', 'lang'], msg: 'string does not match regex', type: 'value_error'}]}, 422));
await h.el('start-btn').click(); await settle();
assert.strictEqual(h.text('attention-title'), "Couldn't start captions");
assert(h.text('attention-text').includes('string does not match regex'));
// A real start switches to Live and reports the server's state, not an assumed one.
h.route('POST', '/api/session/start', () => response({state: 'starting', session_id: 's2', config: {lang: 'en'},
  last_event: 'subprocess launching', readiness: {phase: 'idle', ready: false, reason: 'idle', stale: false}}));
await h.el('start-btn').click(); await settle();
assert.strictEqual(h.app.activeTab, 'live');
assert.strictEqual(h.text('state-pill'), 'Starting…');
assert.strictEqual(h.text('live-title'), 'Starting…');
assert.strictEqual(h.el('stop-btn').disabled, false);
assert.strictEqual(h.el('pause-btn').disabled, true);
assert.strictEqual(h.el('attention').hidden, true);
assert.strictEqual(h.el('summary-btn').disabled, true);
const body = JSON.parse(h.lastFetch('/api/session/start').init.body);
assert.deepStrictEqual(Object.keys(body).sort(), ['backend', 'diarize', 'engine', 'lang', 'log_level', 'run_ab', 'tts', 'tts_output_mode', 'vad_threshold']);
""")


def test_new_status_fields_render_and_old_servers_still_work():
    run_node(r"""
const h = createHarness();
await h.start();
await h.setStatus({state: 'starting', session_id: 's1', started_at: '2026-09-09T10:00:00', config: {lang: 'en'},
  readiness: {phase: 'loading', ready: false, reason: 'weights still loading', updated_at: 1, age_s: 2, stale: false},
  health: realHealth({phase: 'loading', input_seen: false, input_age_s: null, caption_age_s: null, clients: 0,
    queues: {audio: 0, capture_handoff: 0, finals: 2, stream_tokens: 0}}),
  effective_profile: {name: 'lite-cpu'}, work: {kind: 'live session', id: 's1'}, outcome: null});
assert(h.text('pipeline-readiness').includes('loading language models'), h.text('pipeline-readiness'));
assert(h.text('pipeline-readiness').includes('weights still loading'));
const health = h.el('health-list').children.map(li => li.textContent);
assert(health.includes('No sound heard yet'), health);
assert(health.includes('No captions yet'), health);
assert(health.includes('Displays connected: 0'));
assert(health.includes('Backlog: 2'));
assert(health.includes('Recording audio'));
assert(h.text('live-language').includes('profile: Lite — no graphics card'), h.text('live-language'));
assert(h.text('live-elapsed').startsWith('Running for'));
assert.strictEqual(h.el('start-btn').disabled, true);
// Legacy field shapes from older servers still render without inventing anything.
await h.setStatus({state: 'running', session_id: 's1', started_at: '2026-09-09T10:00:00', config: {lang: 'en'},
  readiness: {phase: 'listening', ready: true, reason: null, age_s: 9, stale: true},
  health: {input_age_s: 1.5, clients: 2, errors: 3, recording: true, persistence: {ok: false, reason: 'disk full'}},
  effective_profile: 'lite-cuda-8gb'});
assert(h.text('pipeline-readiness').startsWith('The caption process has not reported'), h.text('pipeline-readiness'));
const bad = h.el('health-list').children.filter(li => li.className === 'bad').map(li => li.textContent);
assert(bad.some(t => t.startsWith('Saving problems: disk full')), bad);
assert(bad.includes('Errors: 3'));
assert(h.el('health-list').children.map(li => li.textContent).includes('Recording audio'));
assert(h.text('live-language').includes('profile: Lite — 8 GB NVIDIA graphics card'));
// Busy work while idle blocks Start with an explanation.
await h.setStatus({state: 'idle', outcome: 'completed', work: {kind: 'summary', id: 'b1'}});
assert.strictEqual(h.el('start-btn').disabled, true);
assert(h.text('start-hint').includes('busy with summary'));
assert.strictEqual(h.text('state-pill'), 'Busy');
await h.setStatus({state: 'idle', outcome: 'completed', work: null});
assert.strictEqual(h.el('start-btn').disabled, false);
assert.strictEqual(h.text('state-pill'), 'Ready');
assert(h.text('state-detail').includes('finished normally'));
await h.setStatus({state: 'idle', outcome: 'interrupted', last_event: 'Session stopped; recording is incomplete'});
assert.strictEqual(h.text('state-pill'), 'Stopped');
assert(h.text('live-subtitle').includes('interrupted'));
assert.strictEqual(h.el('live-event').hidden, true); // server events are shown only while active or in error
// Servers without the new fields keep working and nothing is invented.
await h.setStatus({state: 'running', session_id: 's1', config: {lang: 'en'}});
assert.strictEqual(h.el('health-list').hidden, true);
assert.strictEqual(h.el('pipeline-readiness').hidden, true);
assert.strictEqual(h.text('live-language'), 'English speaker → Spanish captions');
assert.strictEqual(h.text('live-elapsed'), '');
// Capabilities embedded in the status snapshot are honoured too, including the legacy "full" id.
await h.setStatus({state: 'idle', capabilities: {profiles: [{id: 'full'}, {id: 'lite-cuda-8gb', available: false}, {id: 'turbo'}]}});
assert.strictEqual(h.el('profile-field').hidden, false);
assert.deepStrictEqual(h.el('profile-select').options.map(o => o.value), ['', 'standard', 'lite-cuda-8gb', 'turbo']);
assert.strictEqual(h.el('profile-select').options[2].disabled, true);
""")


def test_profiles_follow_server_default_then_explicit_choice():
    run_node(r"""
const lastPreflightProfile = h => {
  const url = h.fetchLog.filter(f => f.url.startsWith('/api/preflight')).pop().url;
  return new URLSearchParams(url.split('?')[1] || '').get('profile');
};
// A Lite launcher (STARK_PROFILE) advertises its default; with no explicit choice the page follows it.
const lite = createHarness({capabilities: realCapabilities({default_profile: 'lite-cpu'})});
await lite.start();
assert.strictEqual(lite.el('profile-field').hidden, false);
assert.strictEqual(lite.el('profile-select').value, '');
assert.strictEqual(lite.el('profile-select').options[0].textContent, 'Server default: Lite — no graphics card');
assert.deepStrictEqual(lite.el('profile-select').options.map(o => o.value), ['', 'standard', 'lite-cpu', 'lite-cpu-quality', 'lite-cuda-8gb']);
assert.strictEqual(lite.app.effectiveProfile, 'lite-cpu');
assert.strictEqual(lite.app.readForm().profile, 'lite-cpu');
assert(lite.text('profile-hint').includes("Following the operator service's default"), lite.text('profile-hint'));
assert.strictEqual(lastPreflightProfile(lite), 'lite-cpu'); // readiness is re-checked for the advertised default
// A legacy persisted "full" means standard; an explicit persisted choice wins while the server offers it.
const legacy = createHarness({storage: [['stark-operator-profile', 'full']], capabilities: realCapabilities({default_profile: 'lite-cpu'})});
await legacy.start();
assert.strictEqual(legacy.el('profile-select').value, 'standard');
assert.strictEqual(legacy.app.readForm().profile, 'standard');
// A persisted choice the server no longer offers falls back to the default.
const gone = createHarness({storage: [['stark-operator-profile', 'lite-cuda-8gb']],
  capabilities: realCapabilities({profiles: ['standard', 'lite-cpu'], default_profile: 'lite-cpu'})});
await gone.start();
assert.strictEqual(gone.el('profile-select').value, '');
assert.strictEqual(gone.app.effectiveProfile, 'lite-cpu');
// Choosing a profile persists it and re-checks readiness for exactly that profile.
const h = createHarness({capabilities: realCapabilities()});
await h.start();
assert.strictEqual(h.app.effectiveProfile, 'standard');
assert.strictEqual(lastPreflightProfile(h), 'standard');
h.el('profile-select').value = 'lite-cpu-quality';
await h.el('profile-select').fire('change'); await settle();
assert.strictEqual(h.storage.get('stark-operator-profile'), 'lite-cpu-quality');
assert.strictEqual(h.app.readForm().profile, 'lite-cpu-quality');
assert.strictEqual(lastPreflightProfile(h), 'lite-cpu-quality');
assert(h.text('profile-hint').includes('Adds a slower processor-only model'), h.text('profile-hint'));
const reloaded = createHarness({storage: [...h.storage], capabilities: realCapabilities()});
await reloaded.start();
assert.strictEqual(reloaded.app.readForm().profile, 'lite-cpu-quality');
// The preflight reply names the profile and backend it checked, in plain words.
h.state.preflight = preflightPayload({backend: 'cpu', profile: {name: 'lite-cpu-quality', backend: 'cpu'},
  effective_profile: {profile: 'lite-cpu-quality', backend: 'cpu', lang: 'en'}});
await h.app.refreshPreflight(); await settle();
assert(h.text('preflight-meta').includes('checked for Lite quality — no graphics card, slower final captions on processor only'), h.text('preflight-meta'));
// A running session shows its confirmed profile; the choice returns afterwards.
await h.setStatus({state: 'running', session_id: 's1', config: {lang: 'en', profile: 'lite-cuda-8gb'}, effective_profile: {name: 'lite-cuda-8gb'}});
assert.strictEqual(h.el('profile-select').value, 'lite-cuda-8gb');
assert.strictEqual(h.el('profile-select').disabled, true);
assert(h.text('profile-hint').includes('running session'));
assert(h.text('live-language').includes('profile: Lite — 8 GB NVIDIA graphics card'));
await h.setStatus({state: 'idle', outcome: 'completed'});
assert.strictEqual(h.el('profile-select').value, 'lite-cpu-quality');
assert.strictEqual(h.el('profile-select').disabled, false);
// Returning to the server default persists "follow the default", not a copy of it.
h.el('profile-select').value = '';
await h.el('profile-select').fire('change'); await settle();
assert.strictEqual(h.storage.get('stark-operator-profile'), '');
assert.strictEqual(h.app.readForm().profile, 'standard');
// Old servers: no profile control, nothing sent.
const old = createHarness();
await old.start();
assert.strictEqual(old.el('profile-field').hidden, true);
assert.strictEqual(old.app.readForm().profile, undefined);
assert.strictEqual(lastPreflightProfile(old), null);
""")


def test_lite_profile_locks_contradictory_technical_settings():
    run_node(r"""
const h = createHarness({capabilities: realCapabilities()});
await h.start();
const c = name => h.el('config-form').elements.namedItem(name);
c('backend').value = 'mlx';
await c('backend').fire('change');
c('run_ab').checked = true;
c('diarize').checked = true;
assert.strictEqual(h.app.readForm().backend, 'mlx');
h.el('profile-select').value = 'lite-cpu';
await h.el('profile-select').fire('change'); await settle();
assert.strictEqual(c('backend').value, 'auto');
for (const name of ['backend', 'engine', 'run_ab', 'diarize']) assert.strictEqual(c(name).disabled, true, name);
assert.strictEqual(c('run_ab').checked, false);
assert.strictEqual(c('diarize').checked, false);
assert.strictEqual(h.el('lite-note').hidden, false);
const body = h.app.readForm();
assert.deepStrictEqual([body.profile, body.backend, body.run_ab, body.diarize], ['lite-cpu', 'auto', false, false]);
// Back to standard: the technical settings are editable again.
h.el('profile-select').value = 'standard';
await h.el('profile-select').fire('change'); await settle();
for (const name of ['backend', 'engine', 'run_ab', 'diarize']) assert.strictEqual(c(name).disabled, false, name);
assert.strictEqual(h.el('lite-note').hidden, true);
// A Lite server default locks them too, without any explicit choice.
const lite = createHarness({capabilities: realCapabilities({default_profile: 'lite-cuda-8gb'})});
await lite.start();
const lc = name => lite.el('config-form').elements.namedItem(name);
assert.strictEqual(lc('backend').disabled, true);
assert.strictEqual(lite.el('lite-note').hidden, false);
// A running session shows the confirmed values; afterwards the profile rule applies again.
await lite.setStatus({state: 'running', config: {lang: 'en', profile: 'lite-cuda-8gb', backend: 'cuda', engine: 'llamacpp'}});
assert.strictEqual(lc('backend').value, 'cuda');
assert.strictEqual(lc('backend').disabled, true);
await lite.setStatus({state: 'idle', outcome: 'completed'});
assert.strictEqual(lc('backend').value, 'auto');
assert.strictEqual(lc('backend').disabled, true);
// The server's own conflict answers are explained.
lite.route('POST', '/api/session/start', () => response({detail: 'Profile lite-cuda-8gb requires backend cuda, not mlx'}, 422));
await lite.el('start-btn').click(); await settle();
assert.strictEqual(lite.text('attention-title'), "The selected settings can't be used together");
""")


def test_audio_tests_use_real_endpoints_and_work_lease():
    run_node(r"""
const h = createHarness({capabilities: realCapabilities()});
await h.start();
assert.strictEqual(h.el('mic-test-btn').disabled, false);
assert.strictEqual(h.el('output-test-btn').hidden, false);
assert.strictEqual(h.el('device-note').hidden, false); // audio_devices_validated: false
assert(h.text('mic-test-status').includes('Nothing is recorded or saved'));
h.el('mic-device').value = '1';
const probe = deferred();
h.route('POST', '/api/audio/test-input', () => probe.promise);
const clicking = h.el('mic-test-btn').click(); await settle();
assert.deepStrictEqual(JSON.parse(h.lastFetch('/api/audio/test-input').init.body), {device: 1, duration_s: 2});
assert(h.text('mic-test-status').includes('Listening for 2 seconds on USB Mic (1ch)'), h.text('mic-test-status'));
assert.strictEqual(h.el('start-btn').disabled, true); // Start waits for the pending device test
assert(h.text('start-hint').includes('Wait for the audio test'));
assert.strictEqual(h.el('mic-test-btn').disabled, true);
probe.resolve(response({ok: true, device: 1, duration_s: 2, rms: 0.1, peak: 0.42, samples: 96000, recorded: false}));
await clicking; await settle();
assert(h.text('mic-test-status').includes('Measured peak 42% (average 10%)'), h.text('mic-test-status'));
assert(h.text('mic-test-status').includes('nothing was recorded or saved'));
assert.strictEqual(h.el('mic-level-bar').style.width, '42%');
assert.strictEqual(h.el('start-btn').disabled, false);
// Silence, refusals and busy leases are reported as exactly that.
h.route('POST', '/api/audio/test-input', () => response({ok: true, device: 1, duration_s: 2, rms: 0, peak: 0, samples: 96000, recorded: false}));
await h.el('mic-test-btn').click(); await settle();
assert(h.text('mic-test-status').includes('No sound was detected'));
h.route('POST', '/api/audio/test-input', () => response({detail: {code: 'audio_unavailable',
  message: 'Audio device could not open. Check its connection and microphone permission.'}}, 422));
await h.el('mic-test-btn').click(); await settle();
assert(h.text('mic-test-status').includes('could not use USB Mic (1ch): Audio device could not open'), h.text('mic-test-status'));
h.route('POST', '/api/audio/test-input', () => response({detail: {code: 'work_busy', message: 'Finish summary before starting another job'}}, 409));
await h.el('mic-test-btn').click(); await settle();
assert(h.text('mic-test-status').includes('Try again when it finishes'), h.text('mic-test-status'));
h.route('POST', '/api/audio/test-input', () => response({ok: false}));
await h.el('mic-test-btn').click(); await settle();
assert(h.text('mic-test-status').includes('did not report a result'));
// The speaker test never claims anyone heard anything.
h.route('POST', '/api/audio/test-output', () => response({ok: true, device: null, duration_s: 0.4}));
await h.el('output-test-btn').click(); await settle();
assert.deepStrictEqual(JSON.parse(h.lastFetch('/api/audio/test-output').init.body), {device: null, duration_s: 0.4});
assert(h.text('mic-test-status').includes('confirm you heard it'), h.text('mic-test-status'));
assert(!/heard successfully|speakers work/i.test(h.text('mic-test-status')));
// Idle-only: another job or a live session blocks the tests.
await h.setStatus({state: 'idle', work: {kind: 'summary', id: 'abc'}});
assert.strictEqual(h.el('mic-test-btn').disabled, true);
await h.setStatus({state: 'running', config: {lang: 'en'}, work: {kind: 'live session', id: 's1'}});
assert.strictEqual(h.el('mic-test-btn').disabled, true);
assert.strictEqual(h.el('output-test-btn').disabled, true);
await h.setStatus({state: 'idle', outcome: 'completed'});
assert.strictEqual(h.el('mic-test-btn').disabled, false);
// A server that advertises the flag but lacks the route.
h.route('POST', '/api/audio/test-input', () => response({detail: 'Not Found'}, 404));
await h.el('mic-test-btn').click(); await settle();
assert(h.text('mic-test-status').includes('not available'));
// No capabilities at all: the controls stay off and honest.
const old = createHarness();
await old.start();
assert.strictEqual(old.el('mic-test-btn').disabled, true);
assert.strictEqual(old.el('output-test-btn').hidden, true);
assert.strictEqual(old.el('device-note').hidden, true);
assert(old.text('mic-test-status').includes('not available'));
""")


def test_pause_and_resume_wait_for_acknowledgment():
    run_node(r"""
const h = createHarness({capabilities: realCapabilities()});
await h.start();
await h.setStatus({state: 'running', session_id: 's1', config: {lang: 'en'}, health: realHealth()});
h.route('POST', '/api/control/pause', () => response({state: 'running', session_id: 's1', config: {lang: 'en'},
  last_event: 'Pause requested; waiting for pipeline acknowledgment', health: realHealth()}));
await h.el('pause-btn').click(); await settle();
assert.strictEqual(h.text('state-pill'), 'Live'); // not paused until the caption process confirms
assert(h.text('live-subtitle').startsWith('Pausing… waiting for the caption process to confirm'), h.text('live-subtitle'));
assert.strictEqual(h.text('live-event'), 'Pause requested; waiting for pipeline acknowledgment');
assert.strictEqual(h.el('pause-btn').disabled, true);
assert.strictEqual(h.el('resume-btn').disabled, true);
assert.strictEqual(h.app.pendingControl.kind, 'pause');
// Polls that still say running keep waiting; a long wait says so.
await h.setStatus({state: 'running', session_id: 's1', config: {lang: 'en'}, health: realHealth()});
assert(h.text('live-subtitle').startsWith('Pausing…'));
h.clock.value += 16000;
await h.setStatus({state: 'running', session_id: 's1', config: {lang: 'en'}, health: realHealth()});
assert(h.text('live-subtitle').includes('No confirmation yet'), h.text('live-subtitle'));
// Acknowledged through health.
await h.setStatus({state: 'paused', session_id: 's1', config: {lang: 'en'}, health: realHealth({phase: 'paused'})});
assert.strictEqual(h.text('state-pill'), 'Paused');
assert.strictEqual(h.app.pendingControl, null);
assert.strictEqual(h.el('resume-btn').disabled, false);
h.route('POST', '/api/control/resume', () => response({state: 'paused', session_id: 's1', config: {lang: 'en'},
  last_event: 'Resume requested; waiting for pipeline acknowledgment'}));
await h.el('resume-btn').click(); await settle();
assert(h.text('live-subtitle').startsWith('Resuming…'));
assert.strictEqual(h.el('resume-btn').disabled, true);
await h.setStatus({state: 'running', session_id: 's1', config: {lang: 'en'}});
assert.strictEqual(h.text('live-subtitle'), 'Captions are being sent to the audience display.');
assert.strictEqual(h.el('pause-btn').disabled, false);
// An immediate state change (older servers) needs no waiting.
h.route('POST', '/api/control/pause', () => response({state: 'paused', session_id: 's1', config: {lang: 'en'}}));
await h.el('pause-btn').click(); await settle();
assert.strictEqual(h.text('state-pill'), 'Paused');
assert.strictEqual(h.app.pendingControl, null);
// A refused control is shown and the state re-polled.
h.route('POST', '/api/control/resume', () => response({detail: 'cannot resume from state=paused'}, 409));
await h.el('resume-btn').click(); await settle();
assert.strictEqual(h.text('attention-title'), "Couldn't resume captions");
""")


def test_summary_renders_bilingual_text_and_keeps_failures_retryable():
    run_node(r"""
const h = createHarness({capabilities: realCapabilities()});
await h.start();
await h.setStatus({state: 'idle', outcome: 'completed', session_id: 's1', csv_path: 'metrics/ab_metrics_s1.csv'});
assert.strictEqual(h.el('summary-btn').disabled, false);
let task = {task_id: 't1', csv_path: 'metrics/ab_metrics_s1.csv', output_path: 'metrics/summary_s1.json', state: 'pending',
  started_at: 1, finished_at: null, return_code: null, error: null, result: null};
h.route('POST', '/api/features/summary', () => response(task));
h.route('GET', '/api/features/summary/t1', () => response(task));
await h.el('summary-btn').click(); await settle();
assert(h.text('summary-status').includes('Working on the summary (pending)'), h.text('summary-status'));
assert.strictEqual(h.el('summary-btn').disabled, true);
assert.strictEqual(h.el('summary-cancel').hidden, false);
assert.strictEqual(h.el('summary-result').hidden, true);
const poll = h.intervals.at(-1);
assert.strictEqual(poll.ms, 2000);
task = {...task, state: 'running'};
await poll.fn(); await settle();
assert(h.text('summary-status').includes('(running)'));
await h.setStatus({state: 'idle', outcome: 'completed', work: {kind: 'summary', id: 't1'}});
assert.strictEqual(h.el('summary-btn').disabled, true);
task = {...task, state: 'done', return_code: 0, finished_at: 2, result: {
  english: "God's grace is enough.", spanish: 'La gracia de Dios es suficiente.', speakers: null,
  format: 'short-session excerpt', translation_method: 'recorded prediction',
  notice: 'Too little text for a sermon summary; original excerpts and translations are shown. Translations are unreviewed predictions.',
  metadata: {input_files: ['metrics/ab_metrics_s1.csv'], model: null, timestamp: 'x', total_segments: 1, total_words: 4,
    diarized: false, source_languages: ['en'], content_mode: 'excerpt', transcript_truncated: false, human_reviewed: false}}};
await poll.fn(); await settle();
assert.strictEqual(h.text('summary-status'), 'Finished.');
assert.strictEqual(h.el('summary-result').hidden, false);
assert.strictEqual(h.text('summary-english'), "God's grace is enough.");
assert.strictEqual(h.text('summary-spanish'), 'La gracia de Dios es suficiente.');
assert.strictEqual(h.el('summary-notice').hidden, false);
assert(h.text('summary-notice').startsWith('Too little text'));
assert(h.text('summary-meta').includes('Excerpt of the recorded text, not a summary'), h.text('summary-meta'));
assert(h.text('summary-meta').includes('not reviewed by a person'));
const sessions = h.panelText('sessions');
assert(!sessions.includes('content_mode') && !sessions.includes('{'), 'raw JSON must stay out of the Sessions tab');
assert(h.text('summary-raw').includes('"content_mode": "excerpt"'));
assert.strictEqual(h.el('summary-cancel').hidden, true);
await h.setStatus({state: 'idle', outcome: 'completed'});
assert.strictEqual(h.el('summary-btn').disabled, false);
// A model summary shows its format; a failure stays visible and retryable.
task = {...task, task_id: 't2', state: 'done', result: {english: 'Summary EN', spanish: 'Resumen ES',
  format: 'up to 3 sentences (undiarized)', translation_method: 'gemma (direct)',
  metadata: {content_mode: 'model_summary', human_reviewed: false, total_words: 900}}};
h.route('POST', '/api/features/summary', () => response(task));
h.route('GET', '/api/features/summary/t2', () => response(task));
await h.el('summary-btn').click(); await settle();
await h.intervals.at(-1).fn(); await settle();
assert(h.text('summary-meta').includes('Model summary (up to 3 sentences (undiarized))'), h.text('summary-meta'));
assert.strictEqual(h.el('summary-notice').hidden, true);
task = {...task, task_id: 't3', state: 'error', return_code: 1, error: 'No transcript content found', result: null};
h.route('POST', '/api/features/summary', () => response(task));
h.route('GET', '/api/features/summary/t3', () => response(task));
await h.el('summary-btn').click(); await settle();
await h.intervals.at(-1).fn(); await settle();
assert(h.text('summary-status').includes('Summary failed: No transcript content found (exit code 1)'), h.text('summary-status'));
assert.strictEqual(h.el('summary-result').hidden, true);
assert.strictEqual(h.el('summary-btn').disabled, false);
assert.strictEqual(h.text('summary-btn'), 'Try again');
// The service being busy is explained; nothing pretends to run.
h.route('POST', '/api/features/summary', () => response({detail: {code: 'work_busy',
  message: 'Finish audio test before starting another job', work: {kind: 'audio test', id: 'x'}}}, 409));
await h.el('summary-btn').click(); await settle();
assert(h.text('summary-status').includes('busy with audio test'), h.text('summary-status'));
assert.strictEqual(h.el('summary-btn').disabled, false);
// Cancel posts the cancel endpoint and shows the server's answer.
task = {...task, task_id: 't4', state: 'running', error: null, return_code: null};
h.route('POST', '/api/features/summary', () => response(task));
h.route('POST', '/api/features/summary/t4/cancel', () => response({...task, state: 'error', error: 'cancelled by operator'}));
await h.el('summary-btn').click(); await settle();
assert.strictEqual(h.el('summary-cancel').hidden, false);
await h.el('summary-cancel').click(); await settle();
assert.strictEqual(h.lastFetch('/api/features/summary/t4/cancel').url, '/api/features/summary/t4/cancel');
assert(h.text('summary-status').includes('Summary failed: cancelled by operator'));
assert.strictEqual(h.el('summary-cancel').hidden, true);
// While another job holds the work lease the button waits.
await h.setStatus({state: 'idle', outcome: 'completed', work: {kind: 'audio test', id: 'z'}});
assert.strictEqual(h.el('summary-btn').disabled, true);
""")


def test_caption_preview_works_without_diarization_and_never_acknowledges():
    run_node(r"""
const h = createHarness();
await h.start();
assert.strictEqual(h.sockets().filter(s => s.url.endsWith(':8765')).length, 0);
assert(h.text('caption-status').includes('connects while captions are running'));
await h.setStatus({state: 'starting', session_id: 's1', config: {lang: 'en', diarize: false}});
const sock = h.sockets().find(s => s.url === 'ws://localhost:8765');
assert(sock, 'caption socket opened without diarization');
assert(h.text('caption-status').includes('Waiting for the caption service'), h.text('caption-status'));
sock.open();
sock.message({type: 'lang_config', session_id: 'abc', source_label: 'English', target_label: 'Español'});
sock.message({type: 'translation', stage: 'partial', chunk_id: 'u1', english: 'for God so', spanish_a: 'porque de tal', session_id: 'abc', event_id: 'e1'});
let items = h.el('caption-view').children;
assert.strictEqual(items.length, 1);
assert.strictEqual(items[0].className, 'partial');
assert.strictEqual(items[0].querySelector('.tgt').textContent, 'porque de tal');
sock.message({type: 'translation', stage: 'complete', chunk_id: 7, english: 'For God so loved the world',
  spanish_a: 'Porque de tal manera amó Dios al mundo', speaker: 'Speaker A', session_id: 'abc', event_id: 'e2'});
items = h.el('caption-view').children;
assert.strictEqual(items.length, 1);
assert.strictEqual(items[0].className, 'final');
assert.strictEqual(items[0].querySelector('.spk').textContent, 'Speaker A:');
assert.strictEqual(items[0].querySelector('.tgt').textContent, 'Porque de tal manera amó Dios al mundo');
assert(h.text('caption-status').includes('English → Español'), h.text('caption-status'));
sock.message({type: 'music_hold', active: true, session_id: 'abc'});
assert(h.text('caption-status').includes('music or silence'));
// Only the newest lines are shown.
for (let i = 10; i < 20; i++) sock.message({type: 'translation', stage: 'complete', chunk_id: i, english: `line ${i}`, spanish_a: `línea ${i}`, session_id: 'abc'});
assert.strictEqual(h.el('caption-view').children.length, 6);
assert.strictEqual(h.el('caption-view').children[5].querySelector('.src').textContent, 'line 19');
// A new session id resets the preview; the preview never acknowledges renders.
sock.message({type: 'lang_config', session_id: 'def', source_label: 'English', target_label: 'Español'});
assert.strictEqual(h.el('caption-view').children[0].className, 'empty');
assert.deepStrictEqual(sock.sent, []);
// Losing the socket while live is reported, not hidden, and a reconnect is scheduled.
await h.setStatus({state: 'running', session_id: 's1', config: {lang: 'en', diarize: false}});
assert.strictEqual(h.sockets().filter(s => s.url === 'ws://localhost:8765').length, 1);
sock.close();
assert(h.text('caption-status').includes('Not receiving captions'), h.text('caption-status'));
assert(h.timeouts.some(t => t.ms === 3000));
// Stopping closes the preview and clears it.
await h.setStatus({state: 'idle', outcome: 'completed'});
assert.strictEqual(h.app.captionClient, null);
assert(h.text('caption-status').includes('connects while captions are running'));
// The metrics frame's diarization block only feeds the Advanced speaker metric now.
h.app.renderMetrics({resources: {}, latency: {}, audio: {change_seq: 0, diarization: {current_speaker: 'Speaker B', transitions: 3, recent: [1, 2]}}});
assert.strictEqual(h.text('metric-speaker'), 'Speaker B');
assert.strictEqual(h.el('caption-view').children[0].className, 'empty');
""")


def test_health_captions_and_level_feed_live_without_websocket():
    run_node(r"""
const h = createHarness({capabilities: realCapabilities()});
await h.start();
const health = realHealth({
  captions: [{chunk_id: 1, stage: 'complete', english: 'Hello everyone', spanish_a: 'Hola a todos', source_lang: 'en', target_lang: 'es', speaker: null}],
  input_level: 0.31, errors: [{stage: 'stt', code: 'RuntimeError', at: 1}], error_count: 1,
  storage: {free_bytes: 500 * 1024 ** 2, low_space: true}, recording: {audio_enabled: true, required_failures: 1, ok: false},
  persistence: {ok: false, failed: 1, completed: 2, pending: 0}});
await h.setStatus({state: 'running', session_id: 's1', started_at: '2026-09-10T10:00:00', config: {lang: 'en', diarize: false},
  health, readiness: {phase: 'ready', ready: true, reason: 'ready', updated_at: 1, age_s: 0.2, stale: false},
  effective_profile: {name: 'standard'}, work: {kind: 'live session', id: 's1'}});
// Captions come from the status feed while the caption socket is not open.
const rows = h.el('caption-view').children;
assert.strictEqual(rows.length, 1);
assert.strictEqual(rows[0].className, 'final');
assert.strictEqual(rows[0].querySelector('.tgt').textContent, 'Hola a todos');
assert(h.text('caption-status').includes('status feed'), h.text('caption-status'));
// Sound level and health chips use the real fields.
assert.strictEqual(h.el('live-level-bar').style.width, '31%');
assert.strictEqual(h.text('live-level-text'), 'Sound level 31%');
const chips = h.el('health-list').children.map(li => [li.className, li.textContent]);
assert(chips.some(([cls, t]) => cls === 'bad' && t === 'Errors: 1'), chips);
assert(chips.some(([cls, t]) => cls === 'bad' && t.startsWith('Recording problems')), chips);
assert(chips.some(([cls, t]) => cls === 'bad' && t.startsWith('Saving problems')), chips);
assert(chips.some(([cls, t]) => cls === 'bad' && t.startsWith('Low disk space')), chips);
assert(chips.some(([, t]) => t === 'Displays connected: 1'));
assert.strictEqual(h.text('pipeline-readiness'), 'Captions are ready.');
// Once the socket delivers captions they take over; the status-feed rows fill in until then.
const sock = h.sockets().find(s => s.url === 'ws://localhost:8765');
sock.open();
assert.strictEqual(h.el('caption-view').children[0].querySelector('.tgt').textContent, 'Hola a todos');
assert(h.text('caption-status').includes('until new ones arrive'), h.text('caption-status'));
sock.message({type: 'translation', stage: 'partial', chunk_id: 'u2', english: 'and', spanish_a: 'y'});
assert.strictEqual(h.el('caption-view').children[0].className, 'partial');
// A microphone problem phase is explained in plain words.
await h.setStatus({state: 'running', session_id: 's1', config: {lang: 'en'}, health: realHealth({phase: 'input_error', captions: []}),
  readiness: {phase: 'input_error', ready: false, reason: 'input error', stale: false}});
assert(h.text('pipeline-readiness').includes('microphone problem'), h.text('pipeline-readiness'));
// Stale health is reported, not hidden.
await h.setStatus({state: 'running', session_id: 's1', config: {lang: 'en'}, health: {session_id: 's1', phase: 'unknown', stale: true, age_s: null},
  readiness: {phase: 'unknown', ready: false, reason: 'Pipeline health is unavailable', updated_at: null, age_s: null, stale: true}});
assert(h.text('pipeline-readiness').includes('has not reported'), h.text('pipeline-readiness'));
assert.strictEqual(h.el('live-level-bar').parentNode.hidden, true);
// After stop, nothing lingers.
await h.setStatus({state: 'idle', outcome: 'completed'});
assert.strictEqual(h.el('health-list').hidden, true);
assert.strictEqual(h.el('caption-view').children[0].className, 'empty');
""")


def test_audience_links_share_only_reachable_addresses():
    run_node(r"""
const h = createHarness({capabilities: realCapabilities()});
await h.start();
assert.strictEqual(h.text('audience-url'), 'http://localhost:8080/displays/mobile_display.html?port=8765');
assert(h.text('audience-note').includes('only works on this computer'));
assert.strictEqual(h.el('audience-copy').disabled, true);
assert.strictEqual(h.el('audience-qr').hidden, true);
await h.el('audience-open').click();
assert.deepStrictEqual(h.opened, ['http://localhost:8080/displays/audience_display.html?port=8765']);
assert.strictEqual(h.el('other-displays').children.length, 2);
assert(h.el('other-displays').children[0].querySelector('a').href.endsWith('church_display.html?port=8765'));
// Reached through the LAN address, the server builds shareable links and the page draws the QR code.
const lan = createHarness({location: {hostname: '192.168.1.20', host: '192.168.1.20:9000'},
  capabilities: realCapabilities({host: '192.168.1.20', http: 8090, ws: 8770})});
await lan.start();
assert.strictEqual(lan.text('audience-url'), 'http://192.168.1.20:8090/displays/mobile_display.html?port=8770');
assert(lan.text('audience-note').includes('same Wi-Fi'));
assert.strictEqual(lan.el('audience-qr').hidden, false);
assert(lan.el('audience-qr').getContext('2d').calls.some(c => c[0] === 'fillRect'));
assert.strictEqual(lan.el('audience-copy').disabled, false);
await lan.el('audience-copy').click();
assert.deepStrictEqual(lan.clipboard, ['http://192.168.1.20:8090/displays/mobile_display.html?port=8770']);
assert.strictEqual(lan.text('caption-endpoint'), 'ws://192.168.1.20:8770');
// Without capabilities the links derive from the page address.
const old = createHarness({location: {hostname: '192.168.1.20', host: '192.168.1.20:9000'}});
await old.start();
assert.strictEqual(old.text('audience-url'), 'http://192.168.1.20:8080/displays/mobile_display.html');
assert.strictEqual(old.el('audience-qr').hidden, false);
assert.strictEqual(old.el('other-displays').hidden, true);
const links = old.context.StarkOperator.audienceLinks({hostname: '127.0.0.1'}, {});
assert.strictEqual(links.shareable, false);
""")


def test_storage_and_support_bundle_degrade_gracefully():
    run_node(r"""
const sessions = [{session: '20260909_101500_000000_en', pending: 3, status: 'completed', exportable: true},
  {session: '20260909_120000_000000_es', pending: 1, active: true}];
const h = createHarness({reviewSessions: {sessions}, capabilities: realCapabilities()});
await h.start();
h.state.storage = {free_bytes: 120 * 1024 ** 3, used_bytes: 3.5 * 1024 ** 3, total_bytes: 200 * 1024 ** 3, low_space: false,
  sessions: [{session_id: '20260909_101500_000000_en', status: 'completed', cleanup_bytes: 12345678, originals_preserved: true},
    {session_id: '20260909_120000_000000_es', status: 'running', cleanup_bytes: 1024, originals_preserved: true}],
  cleanup_scope: 'Completed-session operational logs only; original audio, diagnostics, corrections and exports are preserved'};
await h.el('tab-sessions').click(); await settle();
assert(h.text('storage-summary').includes('Free: 120.0 GB'), h.text('storage-summary'));
const boxes = h.el('storage-sessions').querySelectorAll('input[type="checkbox"]');
assert.strictEqual(boxes.length, 2);
assert.strictEqual(boxes[1].disabled, true); // a running session cannot be cleaned
assert(h.el('storage-sessions').children[0].textContent.includes('logs 11.8 MB'));
assert(h.text('storage-summary').includes('Completed-session operational logs only'), h.text('storage-summary'));
h.route('POST', '/api/storage/cleanup/preview', () => response({preview_id: 'p1', files: ['metrics/session_20260909_101500_000000_en.log'], bytes: 12345678, originals_preserved: true}));
await h.el('storage-cleanup-preview').click(); await settle();
assert.deepStrictEqual(JSON.parse(h.lastFetch('/api/storage/cleanup/preview').init.body), {session_ids: ['20260909_101500_000000_en']});
assert(h.text('storage-status').includes('1 log file (11.8 MB) can be removed'), h.text('storage-status'));
assert.strictEqual(h.el('storage-cleanup').disabled, false);
h.route('POST', '/api/storage/cleanup', () => response({removed_bytes: 12345678, originals_preserved: true}));
await h.el('storage-cleanup').click(); await settle();
assert.deepStrictEqual(JSON.parse(h.lastFetch('/api/storage/cleanup').init.body), {preview_id: 'p1'});
assert(h.text('storage-status').includes('Removed the listed logs (11.8 MB)'), h.text('storage-status'));
assert.strictEqual(h.el('storage-cleanup').disabled, true);
// A refusal from the server is shown as a failure.
h.route('POST', '/api/storage/cleanup/preview', () => response({detail: 'Finish this session before exporting; review drafts are saved'}, 409));
await h.el('storage-cleanup-preview').click(); await settle();
assert(h.text('storage-status').includes("Couldn't preview cleanup: Finish this session"));
// Low space is called out.
h.state.storage = {...h.state.storage, low_space: true, free_bytes: 512 * 1024 ** 2};
await h.app.refreshStorage(); await settle();
assert(h.text('storage-summary').includes('Low disk space'));

// Support bundle: a session id is required; the most recent session is the default.
await h.el('tab-help').click(); await settle();
assert.strictEqual(h.el('support-session').options[0].textContent, 'Most recent session (20260909_101500_000000_en)');
h.route('POST', '/api/support/preview', () => response({preview_id: 'sp1', files: ['metadata.json'], bytes: 0,
  privacy: {text_included: false, audio_included: false, message: 'Metadata excludes transcript, raw log messages, environment and local model paths.'}}));
await h.el('support-preview').click(); await settle();
assert.deepStrictEqual(JSON.parse(h.lastFetch('/api/support/preview').init.body), {session_id: '20260909_101500_000000_en', include_text: false, include_audio: false});
assert(h.text('support-status').includes('Metadata plus 0 B of optional attachments for session 20260909_101500_000000_en. Metadata excludes transcript'), h.text('support-status'));
assert.strictEqual(h.el('support-export').disabled, false);
h.route('POST', '/api/support/export', () => response({bundle_id: 'b1', download_url: '/api/support/download/b1'}));
await h.el('support-export').click(); await settle();
assert.deepStrictEqual(JSON.parse(h.lastFetch('/api/support/export').init.body), {preview_id: 'sp1'});
assert.strictEqual(h.el('support-download').href, '/api/support/download/b1');
// Including text is stated in the preview; changing the options invalidates the old preview.
h.el('support-include-text').checked = true;
await h.el('support-include-text').fire('change');
assert.strictEqual(h.el('support-export').disabled, true);
assert.strictEqual(h.el('support-download').hidden, true);
h.route('POST', '/api/support/preview', () => response({preview_id: 'sp2', files: ['metadata.json', 'metrics/diagnostics_x.jsonl'], bytes: 2048,
  privacy: {text_included: true, audio_included: false, message: 'Attachments contain private session content when selected.'}}));
await h.el('support-preview').click(); await settle();
assert(h.text('support-status').includes('Includes caption text.'), h.text('support-status'));
// A stale export answer cannot re-enable an outdated link.
const slow = deferred();
h.route('POST', '/api/support/export', () => slow.promise);
const exporting = h.el('support-export').click();
h.el('support-include-audio').checked = true;
await h.el('support-include-audio').fire('change');
slow.resolve(response({download_url: '/obsolete'}));
await exporting; await settle();
assert.strictEqual(h.el('support-download').hidden, true);
// Nothing to describe yet on a fresh install.
const fresh = createHarness({capabilities: realCapabilities()});
await fresh.start();
await fresh.el('support-preview').click(); await settle();
assert(fresh.text('support-status').includes('No session to describe yet'));
// Flags off, or an older server, degrade honestly.
const off = createHarness({capabilities: realCapabilities({storage: false, support: false})});
await off.start();
assert(off.text('storage-summary').includes('not available'));
assert.strictEqual(off.el('support-preview').disabled, true);
const old = createHarness();
await old.start();
assert(old.text('storage-summary').includes('not available'));
old.state.reviewSessions = {sessions};
await old.app.refreshSupportSessions();
old.route('POST', '/api/support/preview', () => response({detail: 'Not Found'}, 404));
await old.el('support-preview').click(); await settle();
assert(old.text('support-status').includes('not available'));
""")


def test_qr_and_caption_widgets():
    run_node(r"""
const h = createHarness();
const {StarkQR, StarkCaptions} = h.context;
// Values crossing the vm boundary have another realm's prototypes; compare by structure.
const same = (actual, expected) => assert.deepStrictEqual(JSON.parse(JSON.stringify(actual)), expected);
// Format information for ECC level L, mask 0 is the published 15-bit string.
same(StarkQR.formatBits(0, 1), [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0]);
const url = 'http://192.168.1.20:8080/displays/mobile_display.html';
assert.strictEqual(url.length, 53);
const grid = StarkQR.generate(url);
assert.strictEqual(grid.length, 29); // version 3
assert(grid.every(row => row.length === 29 && row.every(v => v === 0 || v === 1)));
same(grid[0].slice(0, 8), [1, 1, 1, 1, 1, 1, 1, 0]);
same(grid[3].slice(0, 8), [1, 0, 1, 1, 1, 0, 1, 0]);
same(grid[6].slice(8, 13), [1, 0, 1, 0, 1]); // timing pattern
assert.strictEqual(grid[grid.length - 8][8], 1); // dark module
same(grid[grid.length - 1].slice(0, 8), [1, 1, 1, 1, 1, 1, 1, 0]);
assert.strictEqual(StarkQR.versionFor(17), 1);
assert.strictEqual(StarkQR.versionFor(18), 2);
assert.strictEqual(StarkQR.versionFor(230), 9);
assert.strictEqual(StarkQR.generate('x'.repeat(231)), null);
assert.strictEqual(StarkQR.generate('http://a.b/ñ').length, 21);
const canvas = h.document.createElement('canvas');
assert.strictEqual(StarkQR.draw(canvas, 'x'.repeat(231), 100), false);
assert.strictEqual(StarkQR.draw(canvas, 'http://a.b/c', 100), true);
assert.strictEqual(canvas.width, canvas.height);
assert(canvas.getContext('2d').calls.filter(c => c[0] === 'fillRect').length > 100);

const m = StarkCaptions.createModel({limit: 3});
assert.strictEqual(m.apply({type: 'lang_config', session_id: 'a', source_label: 'English', target_label: 'Español'}), true);
same(m.labels(), {source: 'English', target: 'Español'});
assert.strictEqual(m.apply({type: 'translation', stage: 'partial', chunk_id: 'u1', english: 'hel', spanish_a: 'ho', session_id: 'a'}), true);
assert.strictEqual(m.apply({type: 'translation', stage: 'partial', chunk_id: 'u1', english: 'hello', spanish_a: 'hola', session_id: 'a'}), true);
same(m.sentences().map(s => [s.id, s.source, s.partial]), [['p-u1', 'hello', true]]);
assert.strictEqual(m.apply({type: 'translation_start', chunk_id: 1, english: 'hello there', session_id: 'a'}), true);
same(m.sentences().map(s => [s.id, s.target, s.streaming]), [['stream-1', 'hola', true]]);
assert.strictEqual(m.apply({type: 'translation_stream', chunk_id: 1, partial_spanish_a: 'hola ahí', session_id: 'a'}), true);
assert.strictEqual(m.sentences()[0].target, 'hola ahí');
assert.strictEqual(m.apply({type: 'translation', stage: 'complete', chunk_id: 1, english: 'hello there', spanish_a: 'hola, ahí', session_id: 'a'}), true);
same(m.sentences().map(s => [s.id, s.target, s.partial, s.streaming]), [[1, 'hola, ahí', false, false]]);
assert.strictEqual(m.apply({type: 'speaker_update', chunk_id: 1, speaker: 'Speaker A', session_id: 'a'}), true);
assert.strictEqual(m.sentences()[0].speaker, 'Speaker A');
assert.strictEqual(m.apply({type: 'translation', stage: 'complete', chunk_id: 2, english: 'x', spanish_a: 'y', session_id: 'b'}), false);
assert.strictEqual(m.apply({type: 'rolling_stats', session_id: 'b'}), false);
assert.strictEqual(m.apply('garbage'), false);
for (const id of [2, 3, 4]) m.apply({type: 'translation', stage: 'complete', chunk_id: id, english: 'x', spanish_a: 'y', session_id: 'a'});
same(m.sentences().map(s => s.id), [2, 3, 4]);
assert.strictEqual(m.apply({type: 'music_hold', active: true, session_id: 'a'}), true);
assert.strictEqual(m.musicHold(), true);
assert.strictEqual(m.apply({type: 'lang_config', session_id: 'b'}), true);
same(m.sentences(), []);
assert.strictEqual(m.musicHold(), false);
const legacy = StarkCaptions.createModel();
assert.strictEqual(legacy.apply({type: 'translation', stage: 'complete', chunk_id: 9, english: 'x', spanish_a: 'y'}), true);
// Pure helpers used by the page.
const {formatAge, formatBytes, describeState, readinessSummary, pickProfiles, defaultProfileId, normalizeProfileId,
  isLiteProfile, humanPhase, describeLevel} = h.context.StarkOperator;
assert.strictEqual(formatAge(1), 'just now');
assert.strictEqual(formatAge(45), '45 s ago');
assert.strictEqual(formatAge(600), '10 min ago');
assert.strictEqual(formatBytes(1536), '2 KB');
assert.strictEqual(describeState({state: 'running', connection: 'stale'}).label, 'Not connected');
assert.strictEqual(describeState({state: 'idle', preflightOk: true}).tone, 'ok');
assert.strictEqual(readinessSummary(null).tone, 'pending');
assert.strictEqual(readinessSummary({checks: [{status: 'fail'}, {status: 'fail'}]}).text, '2 problems to fix before starting.');
same(pickProfiles({profiles: ['turbo', 'lite-cpu', 'full']}).map(p => [p.id, p.lite]), [['turbo', true], ['lite-cpu', true], ['standard', false]]);
assert.strictEqual(pickProfiles({profiles: ['lite-cpu']})[0].label, 'Lite — no graphics card');
assert.strictEqual(normalizeProfileId('full'), 'standard');
assert.strictEqual(isLiteProfile('standard'), false);
assert.strictEqual(defaultProfileId({default_profile: 'lite-cpu'}, pickProfiles({profiles: ['standard', 'lite-cpu']})), 'lite-cpu');
assert.strictEqual(defaultProfileId({default_profile: 'nope'}, pickProfiles({profiles: ['standard']})), 'standard');
assert.strictEqual(humanPhase('warming_up_gpu'), 'warming up gpu');
assert.strictEqual(humanPhase('input_error'), 'microphone problem — no sound can be captured');
assert.strictEqual(describeLevel(0.42, 0.1).percent, 42);
assert(describeLevel(0, 0).text.includes('No sound was detected'));
assert(describeLevel(0.95).text.includes('Too loud'));
assert(describeLevel(undefined).text.includes('no level was reported'));
""")


def test_repeated_polls_do_not_churn_live_regions():
    run_node(r"""
const h = createHarness({capabilities: realCapabilities()});
await h.start();
await h.setStatus({state: 'running', session_id: 's1', config: {lang: 'en'}, capabilities: realCapabilities(), health: realHealth()});
const sock = h.sockets().find(s => s.url === 'ws://localhost:8765');
sock.open();
sock.message({type: 'translation', stage: 'complete', chunk_id: 1, english: 'hello', spanish_a: 'hola'});
const pillNode = h.el('state-pill').childNodes[0];
const captionRow = h.el('caption-view').children[0];
const profileOption = h.el('profile-select').options[1];
const readiness = h.el('readiness-summary').childNodes[0];
for (let i = 0; i < 3; i++) { await h.app.refreshStatus(); await h.app.refreshPreflight(); }
await settle();
assert.strictEqual(h.el('state-pill').childNodes[0] === pillNode, true); // aria-live text was not rewritten
assert.strictEqual(h.el('caption-view').children[0] === captionRow, true); // caption list was not rebuilt
assert.strictEqual(h.el('profile-select').options[1] === profileOption, true); // embedded capabilities did not rebuild the dropdown
assert.strictEqual(h.el('readiness-summary').childNodes[0] === readiness, true);
// Real changes still render.
sock.message({type: 'translation', stage: 'complete', chunk_id: 2, english: 'again', spanish_a: 'otra vez'});
assert.strictEqual(h.el('caption-view').children[0] === captionRow, false);
await h.setStatus({state: 'paused', session_id: 's1', config: {lang: 'en'}});
assert.strictEqual(h.text('state-pill'), 'Paused');
assert.strictEqual(h.el('state-pill').childNodes[0] === pillNode, false);
// The summary button stays disabled while its task is still being polled.
await h.setStatus({state: 'idle', outcome: 'completed'});
h.route('POST', '/api/features/summary', () => response({task_id: 't1', state: 'running'}));
h.route('GET', '/api/features/summary/t1', () => response({task_id: 't1', state: 'running'}));
await h.el('summary-btn').click(); await settle();
assert.strictEqual(h.el('summary-btn').disabled, true);
await h.setStatus({state: 'idle', outcome: 'completed'});
assert.strictEqual(h.el('summary-btn').disabled, true);
h.route('GET', '/api/features/summary/t1', () => response({task_id: 't1', state: 'done', return_code: 0, result: {english: 'ok', spanish: 'vale', format: 'up to 3 sentences (undiarized)', metadata: {content_mode: 'model_summary', human_reviewed: false}}}));
await h.intervals.at(-1).fn(); await settle();
assert.strictEqual(h.el('summary-btn').disabled, false);
assert.strictEqual(h.text('summary-status'), 'Finished.');
""")
