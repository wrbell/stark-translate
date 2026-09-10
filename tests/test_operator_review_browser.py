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


def test_review_queue_context_and_downloads_follow_current_server_state():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to execute the review UI")
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const source = fs.readFileSync('displays/operator/review.js', 'utf8');
const settle = () => new Promise(resolve => setImmediate(resolve));
function deferred() { let resolve; const promise = new Promise(r => resolve = r); return {promise, resolve}; }
function row(chunk, extra = {}) { return {session: 'first_en', chunk_id: chunk, revision: 0,
  source_lang: 'en', source_text: `Source ${chunk}`, corrected_source_text: `Source ${chunk}`,
  corrected_translation_text: `Translation ${chunk}`, review_priority: 1, audio_available: true,
  transcript_approved: false, translation_approved: false, context: {previous: '', next: ''}, ...extra}; }
function harness(initial, pending = false) {
  const elements = new Map(), storage = new Map(), intervals = [], puts = [], gets = [];
  let rows = initial, pendingExport = null, pendingGet = null, beforePut = null;
  function el(id) {
    if (!elements.has(id)) elements.set(id, {value: '', checked: false, hidden: false, listeners: {},
      addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); },
      replaceChildren(...options) { this.options = options; this.value = options[0]?.value || ''; },
      paused: true, pauseCalls: 0, sourceWrites: 0,
      pause() { this.paused = true; this.pauseCalls++; },
      removeAttribute(name) { delete this[name]; },
      fire(name) { return Promise.all((this.listeners[name] || []).map(fn => fn({}))); }});
    return elements.get(id);
  }
  el('pending').checked = pending; el('split').value = 'train';
  const response = (data, ok = true) => ({ok, status: ok ? 200 : 409,
    json: async () => JSON.parse(JSON.stringify(data))});
  const document = {getElementById: id => el(id.slice(7)), activeElement: null};
  const ctx = {console, URLSearchParams, document, window: {addEventListener() {}},
    setInterval: fn => intervals.push(fn), Option: function(text, value) { this.text = text; this.value = value; },
    localStorage: {getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value),
      removeItem: key => storage.delete(key)},
    fetch: async (url, options) => {
      if (options.method === 'POST') {
        if (pendingExport) { const next = pendingExport; pendingExport = null; return next.promise; }
        const session = decodeURIComponent(url.split('/')[3]), split = JSON.parse(options.body).split;
        return response({download_url: `/bundle/${session}/${split}`, stt_samples: {en: 1, es: 0}, translation_pairs: 1});
      }
      if (options.method === 'PUT') {
        const update = JSON.parse(options.body), chunk = Number(url.split('/').at(-1));
        const session = decodeURIComponent(url.split('/')[3]);
        puts.push({session, chunk, ...update});
        const index = rows.findIndex(r => r.session === session && r.chunk_id === chunk);
        if (update.expected_revision !== rows[index].revision) return response({detail: 'Revision conflict'}, false);
        rows[index] = {...rows[index], ...update, revision: rows[index].revision + 1};
        if (beforePut) { beforePut(); beforePut = null; }
        return response(rows[index]);
      }
      if (url.endsWith('/sessions')) return response({sessions: [...new Set(rows.map(r => r.session))]
        .map(session => ({session, pending: 1, exportable: true}))});
      gets.push(url);
      if (pendingGet) { const next = pendingGet; pendingGet = null; return next.promise; }
      const query = new URLSearchParams(url.split('?')[1]), session = decodeURIComponent(url.split('/')[3]);
      let selected = rows.filter(r => r.session === session);
      if (query.get('pending_only') === 'true') selected = selected.filter(r => !r.excluded &&
        !(r.transcript_approved && r.translation_approved));
      if (query.get('flagged_only') === 'true') selected = selected.filter(r => r.review_priority >= 1);
      selected.sort((a, b) => b.review_priority - a.review_priority || a.chunk_id - b.chunk_id);
      const offset = Number(query.get('offset')), limit = Number(query.get('limit'));
      return response({segments: selected.slice(offset, offset + limit), total: selected.length, offset});
    }};
  vm.createContext(ctx); vm.runInContext(source, ctx);
  return {el, document, puts, gets, storage, response, get rows() { return rows; },
    set rows(value) { rows = value; }, set beforePut(fn) { beforePut = fn; },
    get pendingExport() { return pendingExport = deferred(); }, get pendingGet() { return pendingGet = deferred(); },
    tick: () => intervals[0](), select: async chunk => { el('items').value = String(chunk); await el('items').fire('change'); },
    edit: async (id, value) => { el(id).value = value; await el(id).fire('input'); }};
}
(async () => {
  // Save-and-next crosses the 200-row boundary and loads a newer server revision.
  let h = harness(Array.from({length: 201}, (_, i) => row(i + 1))); await settle();
  await h.select(200); await h.edit('note', 'Draft only');
  h.beforePut = () => { h.rows[200] = {...h.rows[200], revision: 7, corrected_source_text: 'Latest 201'}; };
  await h.el('next').fire('click');
  assert.strictEqual(h.el('items').value, '201');
  assert.strictEqual(h.el('source').value, 'Latest 201');
  assert(h.gets.some(url => url.endsWith('offset=200')));
  await h.edit('note', 'Review latest'); await h.el('save').fire('click');
  assert.strictEqual(h.puts.at(-1).expected_revision, 7);

  // Filtered-out anchor/earlier rows cannot shift the next page past unseen work.
  h = harness(Array.from({length: 205}, (_, i) => row(i + 1)), true); await settle();
  await h.select(200);
  for (const id of ['transcript-approved', 'translation-approved']) {
    h.el(id).checked = true; await h.el(id).fire('input');
  }
  h.beforePut = () => { for (let i = 0; i < 3; i++) h.rows[i] = {...h.rows[i], transcript_approved: true, translation_approved: true};
    h.rows[200] = {...h.rows[200], revision: 4, corrected_source_text: 'Other reviewer 201'}; };
  await h.el('next').fire('click');
  assert.strictEqual(h.el('items').value, '201');
  assert.strictEqual(h.el('source').value, 'Other reviewer 201');
  assert.strictEqual(h.el('transcript-approved').checked, false);
  assert.strictEqual(h.el('translation-approved').checked, false);

  // Priority order, not numeric chunk order, determines the next segment.
  h = harness([row(1, {review_priority: 2}), row(2, {review_priority: 3}), row(3)]); await settle();
  assert.strictEqual(h.el('items').value, '2');
  await h.el('next').fire('click'); assert.strictEqual(h.el('items').value, '1');

  // Typing while next-page data is in flight cancels navigation and preserves its draft.
  const delayed = h.pendingGet, navigation = h.el('next').fire('click'); await settle();
  await h.edit('source', 'Keep typing');
  delayed.resolve(h.response({segments: [row(3)], total: 1, offset: 0})); await navigation;
  assert.strictEqual(h.el('source').value, 'Keep typing');
  assert.strictEqual(h.el('items').value, '1');
  assert.strictEqual(JSON.parse(h.storage.get('stark-review-first_en-1')).corrected_source_text, 'Keep typing');

  // Live context updates around a draft without changing focus, selection, approval or playback.
  h = harness([row(1)]); await settle();
  await h.edit('source', 'Unfinished correction');
  h.document.activeElement = h.el('source'); h.el('source').selectionStart = 5;
  h.el('audio').paused = false;
  const pauses = h.el('audio').pauseCalls, audioSource = h.el('audio').src;
  h.rows[0] = {...h.rows[0], context: {previous: '', next: 'New finalized context'}};
  h.rows.push(row(2)); h.tick(); await settle();
  assert(h.el('context').textContent.includes('New finalized context'));
  assert.strictEqual(h.el('source').value, 'Unfinished correction');
  assert.strictEqual(h.document.activeElement, h.el('source'));
  assert.strictEqual(h.el('source').selectionStart, 5);
  assert.strictEqual(h.el('audio').pauseCalls, pauses);
  assert.strictEqual(h.el('audio').src, audioSource);
  assert.strictEqual(h.el('transcript-approved').checked, false);
  await h.el('save').fire('click'); assert.strictEqual(h.puts.at(-1).expected_revision, 0);

  // Metadata-only explicit refresh keeps playback; newly retained audio becomes available.
  h.rows[0] = {...h.rows[0], context: {previous: 'Before', next: 'After explicit refresh'}};
  await h.el('refresh').fire('click');
  assert(h.el('context').textContent.includes('After explicit refresh'));
  assert.strictEqual(h.el('audio').pauseCalls, pauses);
  h = harness([row(1, {audio_available: false})]); await settle();
  assert.strictEqual(h.el('audio').hidden, true);
  h.rows[0] = {...h.rows[0], audio_available: true}; h.tick(); await settle();
  assert.strictEqual(h.el('audio').hidden, false);
  assert(h.el('audio').src.endsWith('/segments/1/audio'));

  // A download identifies its session/split and is cleared by split, edits, saves, or session changes.
  h = harness([row(1), row(1, {session: 'second_es', source_lang: 'es'})]); await settle();
  await h.el('export').fire('click');
  assert.strictEqual(h.el('download').hidden, false);
  assert.strictEqual(h.el('download').textContent, 'Download first_en train bundle');
  h.el('split').value = 'eval'; await h.el('split').fire('change');
  assert.strictEqual(h.el('download').hidden, true); assert.strictEqual(h.el('download').href, undefined);
  await h.el('export').fire('click'); assert.strictEqual(h.el('download').href, '/bundle/first_en/eval');
  await h.edit('note', 'New draft'); assert.strictEqual(h.el('download').hidden, true);
  await h.el('save').fire('click'); assert.strictEqual(h.el('download').hidden, true);
  await h.el('export').fire('click');
  h.el('session').value = 'second_es'; await h.el('session').fire('change');
  assert.strictEqual(h.el('download').hidden, true); assert.strictEqual(h.el('download').href, undefined);

  // A response from an obsolete export cannot re-enable a download for another purpose/session.
  const oldExport = h.pendingExport, exporting = h.el('export').fire('click'); await settle();
  h.el('split').value = 'train'; await h.el('split').fire('change');
  oldExport.resolve(h.response({download_url: '/obsolete', stt_samples: {en: 0, es: 1}, translation_pairs: 1}));
  await exporting; assert.strictEqual(h.el('download').hidden, true);
  assert.strictEqual(h.el('download').href, undefined);
})().catch(error => { console.error(error); process.exit(1); });
"""
    subprocess.run([node, "-e", script], cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True)
