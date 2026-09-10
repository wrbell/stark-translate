"""Drive the real operator page (index.html + app.js + widgets) under Node.

Every test loads ``displays/operator/index.html`` into the small DOM in
``tests/frontend/fake_dom.js`` and runs the shipped JavaScript against routed
fetch/WebSocket fakes. Nothing here talks to a server or loads a model.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRELUDE = (
    "const {createHarness, settle, deferred, response, preflightPayload} = "
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
h.route('POST', '/api/session/stop', () => response({state: 'idle', outcome: 'failed'}));
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
h.route('GET', '/api/capabilities', () => response({profiles: ['full', 'lite-cpu']}));
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
h.route('POST', '/api/session/start', () => response({detail: {code: 'preflight_failed', message: 'Microphone check failed',
  checks: [{name: 'Microphone', status: 'fail', detail: 'No input devices found'}]}}, 422));
await h.el('start-btn').click(); await settle();
assert.strictEqual(h.app.preflightOk, false);
assert.strictEqual(h.el('start-btn').disabled, true);
assert.strictEqual(h.text('attention-title'), "Can't start yet");
assert(h.text('attention-text').includes('Microphone check failed'));
assert.strictEqual(h.app.activeTab, 'prepare');
assert.strictEqual(h.text('readiness-summary'), '1 problem to fix before starting.');
assert.strictEqual(h.text('start-status'), 'Not started.');
assert.strictEqual(h.text('state-pill'), 'Not ready');
assert.strictEqual(h.app.state, 'idle');
// Busy server: explained with the kind of work, not retried silently.
await h.app.refreshPreflight(); await settle();
assert.strictEqual(h.el('start-btn').disabled, false);
h.route('POST', '/api/session/start', () => response({detail: {code: 'work_busy', message: 'Export in progress',
  work: {kind: 'review_export', id: 'x1'}}}, 409));
await h.el('start-btn').click(); await settle();
assert.strictEqual(h.text('attention-title'), 'The operator service is busy');
assert(h.text('attention-advice').includes('review export'));
assert.strictEqual(h.app.state, 'idle');
assert.strictEqual(h.el('start-btn').disabled, false);
// Older servers answer 409 with a plain string.
h.route('POST', '/api/session/start', () => response({detail: 'session 20260909_x already running'}, 409));
await h.el('start-btn').click(); await settle();
assert(h.text('attention-text').includes('already running'));
// Validation errors from the framework are flattened into one line.
h.route('POST', '/api/session/start', () => response({detail: [{loc: ['body', 'lang'], msg: 'string does not match regex', type: 'value_error'}]}, 422));
await h.el('start-btn').click(); await settle();
assert.strictEqual(h.text('attention-title'), "Couldn't start captions");
assert(h.text('attention-text').includes('string does not match regex'));
// A real start switches to Live and reports the server's state, not an assumed one.
h.route('POST', '/api/session/start', () => response({state: 'starting', session_id: 's2', config: {lang: 'en'}}));
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
  readiness: {phase: 'loading_models', ready: false, reason: 'weights still loading', updated_at: 'x', age_s: 2, stale: false},
  health: {schema_version: 1, session_id: 's1', queues: {stt: 0, translate: 2}, input_age_s: 1.5, caption_age_s: null,
    clients: 0, errors: 0, persistence: {ok: true}, recording: true},
  effective_profile: 'lite-cpu', work: null, outcome: null});
assert(h.text('pipeline-readiness').includes('loading language models'), h.text('pipeline-readiness'));
assert(h.text('pipeline-readiness').includes('weights still loading'));
const health = h.el('health-list').children.map(li => li.textContent);
assert(health.some(t => t.startsWith('Sound last heard')), health);
assert(health.includes('Displays connected: 0'));
assert(health.includes('Backlog: 2'));
assert(health.includes('Recording audio'));
assert(h.text('live-language').includes('profile: lite-cpu'));
assert(h.text('live-elapsed').startsWith('Running for'));
await h.setStatus({state: 'running', session_id: 's1', started_at: '2026-09-09T10:00:00', config: {lang: 'en'},
  readiness: {phase: 'listening', ready: true, reason: null, age_s: 9, stale: true},
  health: {persistence: {ok: false, reason: 'disk full'}, errors: [{msg: 'x'}]}});
assert(h.text('pipeline-readiness').startsWith('Stale'), h.text('pipeline-readiness'));
const bad = h.el('health-list').children.filter(li => li.className === 'bad').map(li => li.textContent);
assert(bad.some(t => t.startsWith('Saving problems: disk full')), bad);
assert(bad.includes('Errors: 1'));
// Busy work while idle blocks Start with an explanation.
await h.setStatus({state: 'idle', outcome: 'completed', work: {kind: 'support_export', id: 'b1'}});
assert.strictEqual(h.el('start-btn').disabled, true);
assert(h.text('start-hint').includes('support export'));
assert.strictEqual(h.text('state-pill'), 'Busy');
await h.setStatus({state: 'idle', outcome: 'completed', work: null});
assert.strictEqual(h.el('start-btn').disabled, false);
assert.strictEqual(h.text('state-pill'), 'Ready');
assert(h.text('state-detail').includes('finished normally'));
await h.setStatus({state: 'idle', outcome: 'interrupted'});
assert.strictEqual(h.text('state-pill'), 'Stopped');
assert(h.text('live-subtitle').includes('interrupted'));
// Servers without the new fields keep working and nothing is invented.
await h.setStatus({state: 'running', session_id: 's1', config: {lang: 'en'}});
assert.strictEqual(h.el('health-list').hidden, true);
assert.strictEqual(h.el('pipeline-readiness').hidden, true);
assert.strictEqual(h.text('live-language'), 'English speaker → Spanish captions');
assert.strictEqual(h.text('live-elapsed'), '');
// Capabilities embedded in the status snapshot are honoured too.
await h.setStatus({state: 'idle', capabilities: {profiles: [{id: 'full'}, {id: 'lite-cuda-8gb', available: false}, {id: 'turbo'}]}});
assert.strictEqual(h.el('profile-field').hidden, false);
assert.deepStrictEqual(h.el('profile-select').options.map(o => o.value), ['', 'full', 'lite-cuda-8gb']);
assert.strictEqual(h.el('profile-select').options[2].disabled, true);
""")


def test_profiles_and_audio_tests_are_capability_gated():
    run_node(r"""
const h = createHarness();
await h.start();
assert.strictEqual(h.el('profile-field').hidden, true);
assert.strictEqual(h.el('mic-test-btn').disabled, true);
assert.strictEqual(h.el('output-test-btn').hidden, true);
assert(h.text('mic-test-status').includes('not available'));
assert.strictEqual(h.app.readForm().profile, undefined);
h.route('GET', '/api/capabilities', () => response({profiles: ['full', 'lite-cpu', 'lite-cuda-8gb', 'turbo'],
  audio: {input_test: true, output_test: {url: '/api/audio/speaker-check'}}}));
await h.app.probeCapabilities(); await settle();
assert.strictEqual(h.el('profile-field').hidden, false);
assert.deepStrictEqual(h.el('profile-select').options.map(o => o.value), ['', 'full', 'lite-cpu', 'lite-cuda-8gb']);
h.el('profile-select').value = 'lite-cpu';
await h.el('profile-select').fire('change');
assert.strictEqual(h.app.readForm().profile, 'lite-cpu');
assert.strictEqual(h.storage.get('stark-operator-profile'), 'lite-cpu');
const reloaded = createHarness({storage: [...h.storage], capabilities: {profiles: ['full', 'lite-cpu']}});
await reloaded.start();
assert.strictEqual(reloaded.app.readForm().profile, 'lite-cpu');
// Microphone test: a real request with an honest result.
assert.strictEqual(h.el('mic-test-btn').disabled, false);
h.el('mic-device').value = '1';
h.route('POST', '/api/audio/input-test', () => response({state: 'done', level_peak: 0.42, level_rms: 0.1}));
await h.el('mic-test-btn').click(); await settle();
assert.deepStrictEqual(JSON.parse(h.lastFetch('/api/audio/input-test').init.body), {device: 1, seconds: 3});
assert(h.text('mic-test-status').includes('Sound detected (42%'), h.text('mic-test-status'));
assert.strictEqual(h.el('mic-level-bar').style.width, '42%');
h.route('POST', '/api/audio/input-test', () => response({state: 'failed', message: 'device busy'}));
await h.el('mic-test-btn').click(); await settle();
assert(h.text('mic-test-status').includes('failed: device busy'));
h.route('POST', '/api/audio/input-test', () => response({state: 'done'}));
await h.el('mic-test-btn').click(); await settle();
assert(h.text('mic-test-status').includes('did not report a level'));
h.route('POST', '/api/audio/input-test', () => response({state: 'done', level_peak: 0}));
await h.el('mic-test-btn').click(); await settle();
assert(h.text('mic-test-status').includes('No sound was detected'));
h.route('POST', '/api/audio/input-test', () => response({detail: 'Not Found'}, 404));
await h.el('mic-test-btn').click(); await settle();
assert(h.text('mic-test-status').includes('not available'));
// Speaker test uses the advertised URL and reports what the server said.
assert.strictEqual(h.el('output-test-btn').hidden, false);
h.route('POST', '/api/audio/speaker-check', () => response({state: 'done', message: 'Played a chime on Speakers.'}));
await h.el('output-test-btn').click(); await settle();
assert(h.text('mic-test-status').includes('Played a chime'));
// Tests are unavailable while captions run.
await h.setStatus({state: 'running', config: {lang: 'en'}});
assert.strictEqual(h.el('mic-test-btn').disabled, true);
assert.strictEqual(h.el('output-test-btn').disabled, true);
// Losing the capability hides the controls and drops the profile from the request.
await h.setStatus({state: 'idle'});
h.route('GET', '/api/capabilities', () => response({detail: 'Not Found'}, 404));
await h.app.probeCapabilities(); await settle();
assert.strictEqual(h.el('profile-field').hidden, true);
assert.strictEqual(h.app.readForm().profile, undefined);
assert.strictEqual(h.el('output-test-btn').hidden, true);
assert.strictEqual(h.el('mic-test-btn').disabled, true);
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


def test_audience_links_share_only_reachable_addresses():
    run_node(r"""
const h = createHarness();
await h.start();
assert.strictEqual(h.text('audience-url'), 'http://localhost:8080/displays/mobile_display.html');
assert(h.text('audience-note').includes('only works on this computer'));
assert.strictEqual(h.el('audience-copy').disabled, true);
assert.strictEqual(h.el('audience-qr').hidden, true);
await h.el('audience-open').click();
assert.deepStrictEqual(h.opened, ['http://localhost:8080/displays/audience_display.html']);
// Opened from the LAN address, the link is shareable and gets a QR code drawn from it.
const lan = createHarness({location: {hostname: '192.168.1.20', host: '192.168.1.20:9000'}});
await lan.start();
assert.strictEqual(lan.text('audience-url'), 'http://192.168.1.20:8080/displays/mobile_display.html');
assert(lan.text('audience-note').includes('same Wi-Fi'));
assert.strictEqual(lan.el('audience-qr').hidden, false);
assert(lan.el('audience-qr').getContext('2d').calls.some(c => c[0] === 'fillRect'));
assert.strictEqual(lan.el('audience-copy').disabled, false);
await lan.el('audience-copy').click();
assert.deepStrictEqual(lan.clipboard, ['http://192.168.1.20:8080/displays/mobile_display.html']);
// Server-advertised addresses win over the page address.
lan.route('GET', '/api/capabilities', () => response({audience: {lan_host: '10.0.0.5', http_port: 8090, ws_port: 8770}}));
await lan.app.probeCapabilities(); await settle();
assert.strictEqual(lan.text('audience-url'), 'http://10.0.0.5:8090/displays/mobile_display.html?port=8770');
assert.strictEqual(lan.text('caption-endpoint'), 'ws://192.168.1.20:8770');
await lan.el('audience-open').click();
assert.strictEqual(lan.opened.at(-1), 'http://10.0.0.5:8090/displays/audience_display.html');
const links = lan.context.StarkOperator.audienceLinks({hostname: '127.0.0.1'}, {});
assert.strictEqual(links.shareable, false);
""")


def test_storage_and_support_bundle_degrade_gracefully():
    run_node(r"""
const sessions = [{session: '20260909_101500_000000_en', pending: 3, status: 'completed', exportable: true},
  {session: '20260909_120000_000000_es', pending: 1, active: true}];
const h = createHarness({reviewSessions: {sessions}});
await h.start();
assert(h.text('storage-summary').includes('not available'));
assert.strictEqual(h.el('storage-cleanup-preview').disabled, true);
h.state.storage = {free_bytes: 120 * 1024 ** 3, used_bytes: 3.5 * 1024 ** 3, sessions: [
  {session_id: '20260909_101500_000000_en', bytes: 2 * 1024 ** 3, status: 'completed'},
  {session_id: '20260909_120000_000000_es', bytes: 1.5 * 1024 ** 3, status: 'running'}]};
await h.el('tab-sessions').click(); await settle();
assert(h.text('storage-summary').includes('Free: 120.0 GB'), h.text('storage-summary'));
assert.strictEqual(h.el('storage-sessions').children.length, 2);
assert.strictEqual(h.el('storage-cleanup-preview').disabled, false);
assert.strictEqual(h.el('storage-cleanup').disabled, true);
h.el('storage-sessions').querySelectorAll('input[type="checkbox"]')[0].checked = true;
h.route('POST', '/api/storage/cleanup/preview', () => response({preview_id: 'p1', files: ['metrics/session_a.log', 'metrics/operator.log.1'], bytes: 12345678}));
await h.el('storage-cleanup-preview').click(); await settle();
assert.deepStrictEqual(JSON.parse(h.lastFetch('/api/storage/cleanup/preview').init.body), {session_ids: ['20260909_101500_000000_en']});
assert(h.text('storage-status').includes('2 files (11.8 MB) can be removed'), h.text('storage-status'));
assert.strictEqual(h.el('storage-cleanup').disabled, false);
h.route('POST', '/api/storage/cleanup', () => response({removed: 2, bytes: 12345678}));
await h.el('storage-cleanup').click(); await settle();
assert.deepStrictEqual(JSON.parse(h.lastFetch('/api/storage/cleanup').init.body), {preview_id: 'p1'});
assert(h.text('storage-status').includes('Removed 2 files'));
assert.strictEqual(h.el('storage-cleanup').disabled, true);
h.route('POST', '/api/storage/cleanup/preview', () => response({detail: 'Not Found'}, 404));
await h.el('storage-cleanup-preview').click(); await settle();
assert(h.text('storage-status').includes('not available'));
assert.strictEqual(h.el('storage-cleanup').disabled, true);

// Support bundle: sessions listed, preview gated on the server, export produces a link.
await h.el('tab-help').click(); await settle();
assert.deepStrictEqual(h.el('support-session').options.map(o => o.value), ['', '20260909_101500_000000_en', '20260909_120000_000000_es']);
h.route('POST', '/api/support/preview', () => response({detail: 'Not Found'}, 404));
await h.el('support-preview').click(); await settle();
assert(h.text('support-status').includes('not available'));
assert.strictEqual(h.el('support-export').disabled, true);
h.el('support-session').value = '20260909_101500_000000_en';
h.route('POST', '/api/support/preview', () => response({preview_id: 'sp1', files: [{path: 'operator.log', bytes: 2048}, 'session.log'],
  bytes: 4096, privacy: 'No caption text or audio included.'}));
await h.el('support-preview').click(); await settle();
assert.deepStrictEqual(JSON.parse(h.lastFetch('/api/support/preview').init.body),
  {session_id: '20260909_101500_000000_en', include_text: false, include_audio: false});
assert(h.text('support-status').includes('2 files, 4 KB. No caption text or audio included.'), h.text('support-status'));
assert.strictEqual(h.el('support-files').children.length, 2);
assert.strictEqual(h.el('support-export').disabled, false);
h.route('POST', '/api/support/export', () => response({download_url: '/api/support/bundles/sp1.zip'}));
await h.el('support-export').click(); await settle();
assert.deepStrictEqual(JSON.parse(h.lastFetch('/api/support/export').init.body), {preview_id: 'sp1'});
assert.strictEqual(h.el('support-download').hidden, false);
assert.strictEqual(h.el('support-download').href, '/api/support/bundles/sp1.zip');
// Changing what to include invalidates the preview and the download.
h.el('support-include-text').checked = true;
await h.el('support-include-text').fire('change');
assert.strictEqual(h.el('support-export').disabled, true);
assert.strictEqual(h.el('support-download').hidden, true);
// An export answered after the options changed cannot re-enable an outdated link.
h.route('POST', '/api/support/preview', () => response({preview_id: 'sp2', files: [], bytes: 0}));
await h.el('support-preview').click(); await settle();
assert.deepStrictEqual(JSON.parse(h.lastFetch('/api/support/preview').init.body).include_text, true);
const slow = deferred();
h.route('POST', '/api/support/export', () => slow.promise);
const exporting = h.el('support-export').click();
h.el('support-include-audio').checked = true;
await h.el('support-include-audio').fire('change');
slow.resolve(response({download_url: '/obsolete'}));
await exporting; await settle();
assert.strictEqual(h.el('support-download').hidden, true);
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
const {formatAge, formatBytes, describeState, readinessSummary, pickProfiles, humanPhase} = h.context.StarkOperator;
assert.strictEqual(formatAge(1), 'just now');
assert.strictEqual(formatAge(45), '45 s ago');
assert.strictEqual(formatAge(600), '10 min ago');
assert.strictEqual(formatBytes(1536), '2 KB');
assert.strictEqual(describeState({state: 'running', connection: 'stale'}).label, 'Not connected');
assert.strictEqual(describeState({state: 'idle', preflightOk: true}).tone, 'ok');
assert.strictEqual(readinessSummary(null).tone, 'pending');
assert.strictEqual(readinessSummary({checks: [{status: 'fail'}, {status: 'fail'}]}).text, '2 problems to fix before starting.');
same(pickProfiles({profiles: ['turbo', 'lite-cpu']}).map(p => p.id), ['lite-cpu']);
assert.strictEqual(humanPhase('warming_up_gpu'), 'warming up gpu');
""")
