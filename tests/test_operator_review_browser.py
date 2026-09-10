"""Execute the actual review UI across delayed saves, reads, and conflicts."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_review_preserves_drafts_and_revisions_across_request_races():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to execute the review UI")
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const source = fs.readFileSync('displays/operator/review.js', 'utf8');
const settle = () => new Promise(resolve => setImmediate(resolve));
function deferred() { let resolve; const promise = new Promise(r => resolve = r); return {promise, resolve}; }
function harness(initialStorage = []) {
  const elements = new Map(), storage = new Map(initialStorage), intervals = [], puts = [];
  let record = {session: 'live_en', chunk_id: 2, revision: 0, source_lang: 'en',
    source_text: 'Original', corrected_source_text: 'Original', corrected_translation_text: 'Traducción'};
  let nextGet, nextPut, nextSessions;
  function el(id) {
    if (!elements.has(id)) elements.set(id, {value: '', checked: false, hidden: false, listeners: {},
      addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); },
      replaceChildren(...options) { this.options = options; this.value = options[0]?.value || ''; },
      paused: true, pauseCalls: 0,
      pause() { this.paused = true; this.pauseCalls++; }, removeAttribute() {},
      fire(name) { return Promise.all((this.listeners[name] || []).map(fn => fn({}))); }});
    return elements.get(id);
  }
  const response = (data, ok = true) => ({ok, status: ok ? 200 : 409, json: async () => data});
  const ctx = {console, URLSearchParams, document: {getElementById: id => el(id.slice(7))},
    window: {addEventListener() {}}, setInterval: fn => intervals.push(fn),
    Option: function(text, value) { this.text = text; this.value = value; },
    localStorage: {getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value),
      removeItem: key => storage.delete(key)},
    fetch: async (url, options) => {
      if (options.method === 'PUT') {
        puts.push(JSON.parse(options.body));
        if (nextPut) { const pending = nextPut; nextPut = null; return pending.promise; }
        record = {...record, ...puts.at(-1), revision: record.revision + 1};
        return response(record);
      }
      if (url.endsWith('/sessions')) {
        if (nextSessions) { const pending = nextSessions; nextSessions = null; return pending.promise; }
        return response({sessions: [{session: record.session, pending: 1}]});
      }
      if (nextGet) { const pending = nextGet; nextGet = null; return pending.promise; }
      return response({segments: [record], total: 1});
    }};
  vm.createContext(ctx); vm.runInContext(source, ctx);
  return {el, puts, storage, response, get record() { return record; },
    set record(value) { record = value; },
    get pendingRead() { return nextGet = deferred(); }, get pendingWrite() { return nextPut = deferred(); },
    get pendingSessions() { return nextSessions = deferred(); },
    tick: () => intervals[0](), edit: async text => { el('source').value = text; await el('source').fire('input'); }};
}
(async () => {
  // Conflict refresh must load the same chunk's latest revision and keep the draft recoverable.
  let h = harness(); await settle();
  await h.edit('My draft');
  const failed = h.pendingWrite, save = h.el('save').fire('click');
  failed.resolve(h.response({detail: 'Edited elsewhere'}, false)); await save;
  h.record = {...h.record, revision: 1, corrected_source_text: 'Other reviewer'};
  await h.el('refresh').fire('click');
  assert.strictEqual(h.el('source').value, 'Other reviewer');
  assert.strictEqual(h.el('restore').hidden, false);
  assert.strictEqual([...h.storage.keys()].filter(key => key !== 'stark-review-selection').length, 1);
  await h.el('restore').fire('click');
  assert.strictEqual(h.el('source').value, 'My draft');
  await h.el('save').fire('click');
  assert.strictEqual(h.puts.at(-1).expected_revision, 1);
  assert.strictEqual([...h.storage.keys()].filter(key => key !== 'stark-review-selection').length, 0);

  // A pre-save GET that finishes after the PUT cannot restore revision zero.
  h = harness(); await settle();
  const delayed = h.pendingRead, old = {...h.record}; h.tick(); await settle();
  await h.edit('Saved text'); await h.el('save').fire('click');
  delayed.resolve(h.response({segments: [old], total: 1})); await settle();
  assert.strictEqual(h.el('source').value, 'Saved text');
  await h.edit('Next edit'); await h.el('save').fire('click');
  assert.strictEqual(h.puts.at(-1).expected_revision, 1);

  // Edits made during a save remain a local draft attached to the new revision.
  h = harness(); await settle(); await h.edit('First edit');
  const saving = h.pendingWrite, promise = h.el('save').fire('click');
  await h.edit('Later edit');
  saving.resolve(h.response({...h.record, revision: 1, corrected_source_text: 'First edit'})); await promise;
  assert.strictEqual(h.el('source').value, 'Later edit');
  assert.strictEqual(JSON.parse(h.storage.get('stark-review-live_en-2')).expected_revision, 1);

  // A conflict GET also cannot discard typing that happened while it was in flight.
  h = harness(); await settle(); await h.edit('Draft');
  const conflict = h.pendingWrite, attempt = h.el('save').fire('click');
  conflict.resolve(h.response({detail: 'Conflict'}, false)); await attempt;
  const reload = h.pendingRead, refresh = h.el('refresh').fire('click');
  await h.edit('Newer draft');
  reload.resolve(h.response({segments: [{...h.record, revision: 2}], total: 1})); await refresh;
  assert.strictEqual(h.el('source').value, 'Newer draft');

  // Delayed session-list polling must not change selection while editing.
  h = harness(); await settle(); const sessionPoll = h.pendingSessions; h.tick();
  await h.edit('Keep selection');
  sessionPoll.resolve(h.response({sessions: [{session: 'other_en', pending: 1}]})); await settle();
  assert.strictEqual(h.el('session').value, 'live_en');
  assert.strictEqual(h.el('source').value, 'Keep selection');

  // Polling preserves the current audio element even when another reviewer saves.
  h = harness(); await settle();
  h.el('audio').paused = false;
  const paused = h.el('audio').pauseCalls;
  h.tick(); await settle();
  assert.strictEqual(h.el('audio').pauseCalls, paused);
  h.record = {...h.record, revision: 1, corrected_source_text: 'External revision'};
  h.tick(); await settle();
  assert.strictEqual(h.el('audio').pauseCalls, paused);
  assert.strictEqual(h.el('source').value, 'Original');
  await h.el('refresh').fire('click');
  assert.strictEqual(h.el('source').value, 'External revision');
  assert.strictEqual(h.el('audio').pauseCalls, paused + 1);

  // Selection and filters survive reload alongside any local draft.
  h = harness([['stark-review-selection', JSON.stringify({session: 'live_en', chunk: 2,
    pending: false, flagged: true, offset: 0})],
    ['stark-review-live_en-2', JSON.stringify({expected_revision: 0, source_lang: 'en',
      corrected_source_text: 'Recovered draft'})]]);
  await settle();
  assert.strictEqual(h.el('pending').checked, false);
  assert.strictEqual(h.el('flagged').checked, true);
  assert.strictEqual(h.el('items').value, '2');
  assert.strictEqual(h.el('source').value, 'Recovered draft');
})().catch(error => { console.error(error); process.exit(1); });
"""
    subprocess.run([node, "-e", script], cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True)
