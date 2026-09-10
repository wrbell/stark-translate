"""Published finals supersede only their own capture previews, without models."""

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest

from tests.test_discarded_caption_ui import DISPLAYS, run_node
from tools.caption_delivery import CaptionDelivery, FinalCaptionHistory
from tools.pipeline_health import PipelineHealth


def caption(uid, stage="partial", session="s", chunk=None):
    return {
        "type": "translation",
        "session_id": session,
        "stage": stage,
        "utterance_id": uid,
        "chunk_id": uid if chunk is None else chunk,
    }


def test_delivery_keeps_prepublication_preview_and_only_removes_finalized_pending_identity():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        sent = []

        async def send(raw):
            message = json.loads(raw)
            if not sent:
                entered.set()
                await release.wait()
            sent.append(message)

        client = Mock(send=send, close=AsyncMock())
        delivery = CaptionDelivery()
        early = caption(7)
        delivery.publish(client, early)
        await entered.wait()  # Already admitted while the final computes.
        delivery.publish(client, caption(7))
        newer = caption(8)
        delivery.publish(client, newer)
        final = caption(7, "complete", chunk=8)  # Chunk number collides with newer capture.
        delivery.publish(client, final)
        delivery.publish(client, caption(7))  # Worker finishes after final publication.
        delivery.publish(client, caption(None, chunk=7))  # Legacy partial uses capture ID in chunk_id.
        other_session = caption(7, session="other")
        delivery.publish(client, other_session)
        release.set()
        await delivery.close()
        assert sent == [early, newer, final, other_session]

    asyncio.run(run())


@pytest.mark.parametrize("uid", [None, False, True, 0, -1, "7", 7.5, 2**53])
def test_unidentified_or_malformed_final_does_not_close_its_numeric_chunk(uid):
    async def run():
        sent = []

        async def send(raw):
            sent.append(json.loads(raw))

        client = Mock(send=send, close=AsyncMock())
        delivery = CaptionDelivery()
        final, partial = caption(uid, "complete", chunk=7), caption(7)
        delivery.publish(client, final)
        delivery.publish(client, partial)
        await delivery.close()
        assert sent == [final, partial]

    asyncio.run(run())


def test_health_filters_late_partial_and_keeps_newer_preview_and_foreign_session_out(tmp_path):
    health = PipelineHealth(tmp_path, "s")
    health.caption(caption(7))
    health.caption(caption(None, chunk=7))
    health.caption(caption(8))
    health.caption(caption(7, "complete", chunk=8))
    health.caption(caption(7))
    health.caption(caption(None, chunk=7))
    health.caption(caption(8, "complete", session="stale"))
    health.caption(caption(8))
    rows = health.snapshot()["captions"]
    assert [(r["stage"], r["utterance_id"]) for r in rows] == [("partial", 8), ("complete", 7), ("partial", 8)]
    assert all(r["session_id"] == "s" for r in rows)


def test_final_identity_retention_is_bounded_and_expired_results_cannot_resurrect():
    history = FinalCaptionHistory()
    for uid in range(2, 1000, 2):
        history.observe(uid)
    assert len(history.ids) == 128
    assert history.blocks(2) and history.blocks(998)
    assert not history.blocks(997) and not history.blocks(1000)


@pytest.mark.parametrize("display", DISPLAYS)
def test_shipped_handler_final_does_not_reopen_or_erase_newer_utterance(display):
    run_node(
        "const name = "
        + repr(display)
        + ";\n"
        + r"""
const {createDisplay} = require('./tests/frontend/caption_display_harness.js');
const h = createDisplay(name);
const config = session_id => h.send({type:'lang_config', session_id, source_label:'English', target_label:'Spanish'});
const partial = (uid, text, extra={}) => h.send({type:'translation',session_id:'s',stage:'partial',
  chunk_id:uid,utterance_id:uid,english:text,spanish_a:text,...extra});
config('s');
partial(7,'EARLY_PREVIEW');
assert(h.text().includes('EARLY_PREVIEW'),h.text());
h.send({type:'translation_start',session_id:'s',chunk_id:8,utterance_id:7,english:'FINAL_IN_PROGRESS'});
partial(7,'PREVIEW_DURING_FINAL');
assert(h.text().includes('PREVIEW_DURING_FINAL'),h.text());
partial(8,'NEWER_PREVIEW');
if (name === 'ab_display') {
  h.send({type:'translation_start',session_id:'s',chunk_id:8,utterance_id:7,english:'FINAL_IN_PROGRESS'});
  for (const panelId of ['a-en','c-en','c-es']) {
    const ids=Array.from(h.document.getElementById(panelId).children).map(el=>el.id);
    const stream=ids.indexOf(panelId+'-chunk-stream-8'), preview=ids.indexOf(panelId+'-chunk-p-8');
    assert(stream>=0 && preview>stream, 'older stream must precede newer partial: '+ids);
  }
}
h.send({type:'translation',session_id:'s',stage:'complete',chunk_id:8,utterance_id:7,
  english:'AUTHORITATIVE_FINAL',spanish_a:'AUTHORITATIVE_FINAL'});
assert(h.text().includes('AUTHORITATIVE_FINAL'),h.text());
assert(h.text().includes('NEWER_PREVIEW'),h.text());
if (name === 'ab_display') {
  for (const panelId of ['a-en','b-en','b-es','c-en','c-es']) {
    const ids=Array.from(h.document.getElementById(panelId).children).map(el=>el.id);
    const final=ids.indexOf(panelId+'-chunk-8'), preview=ids.indexOf(panelId+'-chunk-p-8');
    assert(final>=0 && preview>final, 'older final must precede newer partial: '+ids);
  }
}
const sent=h.socket.sent.length;
partial(7,'STALE_AFTER_FINAL');
assert(!h.text().includes('STALE_AFTER_FINAL'),h.text());
assert.strictEqual(h.socket.sent.length,sent,'ignored late partial must not create render evidence');
partial(8,'NEWER_UPDATE');
assert(h.text().includes('NEWER_UPDATE'),h.text());
config('s'); // reconnect retains final identities
partial(7,'STALE_RECONNECT');
assert(!h.text().includes('STALE_RECONNECT'),h.text());
config('new-session');
partial(7,'FRESH_NAMESPACE',{session_id:'new-session'});
assert(h.text().includes('FRESH_NAMESPACE'),h.text());
h.send({type:'translation',session_id:'s',stage:'complete',chunk_id:7,utterance_id:7,
  english:'STALE_SESSION',spanish_a:'STALE_SESSION'});
partial(7,'FRESH_UPDATE',{session_id:'new-session'});
assert(h.text().includes('FRESH_UPDATE'),h.text());
"""
    )


def test_shared_and_operator_guards_have_bounded_exact_final_history():
    run_node(r"""
const fs=require('fs'),vm=require('vm'); const ctx={};ctx.window=ctx;vm.createContext(ctx);
vm.runInContext(fs.readFileSync('displays/display_connection.js','utf8'),ctx);
vm.runInContext(fs.readFileSync('displays/operator/widgets/captions.js','utf8'),ctx);
const model=ctx.StarkCaptions.createModel();
const guard=ctx.StarkDisplayConnection.sessionGuard(()=>{});
const consumers=[message=>guard(message),message=>model.apply(message)];
for (const consume of consumers) {
  consume({type:'lang_config',session_id:'s'});
  for(const uid of [null,undefined,0,-1,'7',7.5,2**53]) {
    consume({type:'translation',stage:'complete',session_id:'s',chunk_id:7,utterance_id:uid});
    assert(consume({type:'translation',stage:'partial',session_id:'s',chunk_id:7,utterance_id:7}));
  }
  for(let uid=2;uid<1000;uid+=2)consume({type:'translation',stage:'complete',session_id:'s',chunk_id:uid,utterance_id:uid});
  assert.strictEqual(consume({type:'translation',stage:'partial',session_id:'s',chunk_id:2,utterance_id:2}),false);
  assert(consume({type:'translation',stage:'partial',session_id:'s',chunk_id:997,utterance_id:997}));
  assert.strictEqual(consume({type:'translation',stage:'partial',session_id:'stale',chunk_id:999,utterance_id:999}),false);
}
""")


def test_operator_socket_and_health_fallback_reject_postfinal_partial():
    run_node(r"""
const {createHarness,realCapabilities,realHealth}=require('./tests/frontend/operator_harness.js');
(async()=>{
const h=createHarness({capabilities:realCapabilities()}); await h.start();
const final={session_id:'s',stage:'complete',chunk_id:8,utterance_id:7,english:'FINAL',spanish_a:'FINAL'};
const partial={session_id:'s',stage:'partial',chunk_id:7,utterance_id:7,english:'LATE',spanish_a:'LATE'};
const newer={...partial,chunk_id:8,utterance_id:8,english:'NEWER',spanish_a:'NEWER'};
const status=rows=>({state:'running',session_id:'s',config:{lang:'en'},health:realHealth({session_id:'s',captions:rows})});
await h.setStatus(status([]));const socket=h.sockets().find(s=>s.url==='ws://localhost:8765');socket.open();
socket.message({type:'lang_config',session_id:'s'});
socket.message({type:'translation',...newer});socket.message({type:'translation',...final});
assert(h.text('caption-view').includes('NEWER'),h.text('caption-view'));
socket.message({type:'translation',...partial});assert(!h.text('caption-view').includes('LATE'));
socket.close();await h.setStatus(status([final,partial,newer]));
assert(!h.text('caption-view').includes('LATE'),h.text('caption-view'));
assert(h.text('caption-view').includes('NEWER'),h.text('caption-view'));
assert.deepStrictEqual(socket.sent,[]);
})().catch(e=>{console.error(e);process.exit(1);});
""")
