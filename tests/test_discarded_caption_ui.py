"""Exercise discard behavior through each shipped page's JavaScript handler; no models."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DISPLAYS = ["audience_display", "ab_display", "church_display", "mobile_display", "obs_overlay"]


def run_node(script):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for production JavaScript handler tests")
    result = subprocess.run(
        [node, "-e", "const assert = require('assert');\n" + script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("display", DISPLAYS)
def test_discard_removes_only_matching_partial_and_blocks_late_preview(display):
    run_node(
        "const name = "
        + repr(display)
        + ";\n"
        + r"""
const {createDisplay} = require('./tests/frontend/caption_display_harness.js');
const h = createDisplay(name);
const config = session_id => h.send({type: 'lang_config', session_id, source_label: 'English', target_label: 'Spanish'});
const caption = (uid, text, extras = {}) => h.send({type: 'translation', session_id: 's1', stage: 'partial',
  chunk_id: uid, utterance_id: uid, english: text, spanish_a: text, spanish_b: text, ...extras});
const discard = (uid, extras = {}) => h.send({type: 'utterance_discarded', session_id: 's1', utterance_id: uid, reason: 'short_silence', ...extras});
config('s1');
caption(100, 'KEPT_FINAL', {stage: 'complete', chunk_id: 7});
caption(70, 'DISCARDED_PREVIEW', {chunk_id: 7}); // utterance and final chunk IDs collide
assert(h.text().includes('KEPT_FINAL'), h.text());
assert(h.text().includes('DISCARDED_PREVIEW'), h.text());
for (const uid of [null, undefined, 0, -1, '70', 1.5, 2**54]) discard(uid);
discard(70, {session_id: 'old'});
discard(70, {session_id: undefined});
assert(h.text().includes('DISCARDED_PREVIEW'), h.text());
let sent = h.socket.sent.length;
discard(70);
assert.strictEqual(h.socket.sent.length, sent, 'removal cannot acknowledge a caption render');
assert(!h.text().includes('DISCARDED_PREVIEW'), h.text());
assert(h.text().includes('KEPT_FINAL'), h.text());
caption(70, 'LATE_DISCARDED_PREVIEW', {chunk_id: 888});
assert(!h.text().includes('LATE_DISCARDED_PREVIEW'), h.text());
assert.strictEqual(h.socket.sent.length, sent, 'ignored preview cannot borrow a later render');
caption(90, 'NEWER_PREVIEW');
discard(70);
assert(h.text().includes('NEWER_PREVIEW'), h.text());
discard(undefined, {chunk_id: 90});
assert(h.text().includes('NEWER_PREVIEW'), h.text());
discard(90);
assert(!h.text().includes('NEWER_PREVIEW'), h.text());
caption(42, 'LEGACY_PREVIEW', {utterance_id: undefined});
discard(42);
assert(!h.text().includes('LEGACY_PREVIEW'), h.text());
caption(70, 'FINAL_AFTER_DISCARD', {stage: 'complete', chunk_id: 70});
discard(70);
assert(h.text().includes('FINAL_AFTER_DISCARD'), h.text());
config('s2');
caption(70, 'NEW_SESSION_PREVIEW', {session_id: 's2'});
discard(70);
assert(h.text().includes('NEW_SESSION_PREVIEW'), h.text());
discard(70, {session_id: 's2'});
assert(!h.text().includes('NEW_SESSION_PREVIEW'), h.text());
"""
    )


@pytest.mark.parametrize("display", ["audience_display", "ab_display", "obs_overlay"])
def test_discard_never_removes_or_tombstones_final_stream(display):
    run_node(
        "const name = "
        + repr(display)
        + ";\n"
        + r"""
const {createDisplay} = require('./tests/frontend/caption_display_harness.js');
const h = createDisplay(name);
h.send({type: 'lang_config', session_id: 's1', source_label: 'English', target_label: 'Spanish'});
h.send({type: 'translation_start', session_id: 's1', chunk_id: 9, utterance_id: 90, english: 'STREAM_SOURCE'});
h.send({type: 'translation_stream', session_id: 's1', chunk_id: 9, utterance_id: 90, partial_spanish_a: 'STREAM_TARGET'});
assert(h.text().includes('STREAM_TARGET'), h.text());
h.send({type: 'utterance_discarded', session_id: 's1', utterance_id: 90});
assert(h.text().includes('STREAM_TARGET'), h.text());
h.send({type: 'translation_stream', session_id: 's1', chunk_id: 9, utterance_id: 90, partial_spanish_a: 'STREAM_CONTINUES'});
assert(h.text().includes('STREAM_CONTINUES'), h.text());
"""
    )


def test_operator_discards_socket_and_health_fallback_without_resurrection():
    run_node(r"""
const {createHarness, realCapabilities, realHealth} = require('./tests/frontend/operator_harness.js');
(async () => {
const h = createHarness({capabilities: realCapabilities()});
await h.start();
const final = {session_id: 's1', chunk_id: 7, utterance_id: 100, stage: 'complete', english: 'KEPT_FINAL', spanish_a: 'KEPT_FINAL'};
const partial = {session_id: 's1', chunk_id: 7, utterance_id: 70, stage: 'partial', english: 'DISCARDED_PREVIEW', spanish_a: 'DISCARDED_PREVIEW'};
const status = rows => ({state: 'running', session_id: 's1', config: {lang: 'en'}, health: realHealth({session_id: 's1', captions: rows})});
await h.setStatus(status([partial]));
const socket = h.sockets().find(s => s.url === 'ws://localhost:8765');
socket.open();
socket.message({type: 'lang_config', session_id: 's1'});
const view = () => h.text('caption-view');
assert(view().includes('DISCARDED_PREVIEW'), view());
// The socket model is empty: this must also hide the status-feed-only preview.
socket.message({type: 'utterance_discarded', session_id: 's1', utterance_id: 70});
assert(!view().includes('DISCARDED_PREVIEW'), view());
await h.setStatus(status([final, partial]));
assert(!view().includes('DISCARDED_PREVIEW'), view());
assert(view().includes('KEPT_FINAL'), view());
socket.message({type: 'translation', ...partial});
assert(!view().includes('DISCARDED_PREVIEW'), view());
socket.close();
assert(!view().includes('DISCARDED_PREVIEW'), view());
assert(view().includes('KEPT_FINAL'), view());
socket.open();
socket.message({type: 'translation', ...final});
const later = {...partial, chunk_id: 8, utterance_id: 80, english: 'NEW_PREVIEW', spanish_a: 'NEW_PREVIEW'};
socket.message({type: 'translation', ...later});
socket.message({type: 'utterance_discarded', session_id: 'old', utterance_id: 80});
socket.message({type: 'utterance_discarded', session_id: 's1', utterance_id: '80'});
assert(view().includes('NEW_PREVIEW'), view());
socket.message({type: 'utterance_discarded', session_id: 's1', utterance_id: 80});
assert(!view().includes('NEW_PREVIEW'), view());
assert(view().includes('KEPT_FINAL'), view());
socket.message({type: 'translation_start', session_id: 's1', chunk_id: 80, utterance_id: 80, english: 'STREAM_SOURCE'});
socket.message({type: 'translation_stream', session_id: 's1', chunk_id: 80, partial_spanish_a: 'STREAM_TARGET'});
socket.message({type: 'utterance_discarded', session_id: 's1', utterance_id: 80});
assert(view().includes('STREAM_TARGET'), view());
socket.message({type: 'lang_config', session_id: 's2'});
socket.message({type: 'translation', ...partial, session_id: 's2'});
assert(view().includes('DISCARDED_PREVIEW'), view());
socket.message({type: 'utterance_discarded', session_id: 's1', utterance_id: 70});
assert(view().includes('DISCARDED_PREVIEW'), view());
socket.message({type: 'utterance_discarded', session_id: 's2', utterance_id: 70});
assert(!view().includes('DISCARDED_PREVIEW'), view());
assert.deepStrictEqual(socket.sent, []);
})().catch(e => { console.error(e); process.exit(1); });
""")


@pytest.mark.parametrize("display", DISPLAYS)
def test_discard_requires_session_handshake_and_preserves_reconnect_tombstone(display):
    run_node(
        "const name = "
        + repr(display)
        + ";\n"
        + r"""
const {createDisplay} = require('./tests/frontend/caption_display_harness.js');
const h = createDisplay(name);
const partial = {type: 'translation', session_id: 's1', stage: 'partial', chunk_id: 1,
  english: 'PRE_HANDSHAKE', spanish_a: 'PRE_HANDSHAKE', spanish_b: 'PRE_HANDSHAKE'};
h.send(partial);
h.send({type: 'utterance_discarded', session_id: 's1', utterance_id: 1});
assert(h.text().includes('PRE_HANDSHAKE'), h.text());
const config = {type: 'lang_config', session_id: 's1', source_label: 'English', target_label: 'Spanish'};
h.send(config);
h.send(partial);
h.send({type: 'utterance_discarded', session_id: 's1', utterance_id: 1});
assert(!h.text().includes('PRE_HANDSHAKE'), h.text());
h.send(config); // reconnect to the same session is not a fresh utterance namespace
h.send(partial);
assert(!h.text().includes('PRE_HANDSHAKE'), h.text());
"""
    )


def test_ab_discard_also_removes_cached_marian_preview():
    run_node(r"""
const {createDisplay} = require('./tests/frontend/caption_display_harness.js');
const h = createDisplay('ab_display');
h.send({type: 'lang_config', session_id: 's1', source_label: 'English', target_label: 'Spanish'});
h.send({type: 'translation', session_id: 's1', stage: 'partial', chunk_id: 7, utterance_id: 70,
  english: 'SOURCE_PREVIEW', spanish_a: 'DISCARDED_MARIAN'});
h.send({type: 'utterance_discarded', session_id: 's1', utterance_id: 70});
h.send({type: 'translation', session_id: 's1', stage: 'complete', chunk_id: 7, utterance_id: 70,
  english: 'FINAL_SOURCE', spanish_a: 'FINAL_TARGET'});
assert(!h.text().includes('DISCARDED_MARIAN'), h.text());
assert(h.text().includes('FINAL_TARGET'), h.text());
""")
