"""Run the browser encoder against independent Nayuki v1.8.0 reference grids.

The checked-in hashes come from byte-mode ECC-L/mask-0 qrcodegen output, not
this widget. ZXing-C++ independently decoded those exact widget matrices; see
fixtures/operator_qr_oracle.json and docs/evaluation/operator_qr_verification_20260910.json.
No QR package or network access is required by these regression tests.
"""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/operator_qr_oracle.json"


def _node(script, payload=None):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to execute the actual browser QR encoder")
    result = subprocess.run(
        [node, "-e", script],
        cwd=ROOT,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    return json.loads(result.stdout)


def test_all_supported_versions_match_independent_encoder():
    fixture = json.loads(FIXTURE.read_text())
    cases = fixture["cases"]
    matrices = _node(
        """
const fs = require('fs');
require('./displays/operator/widgets/qr.js');
const texts = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(texts.map(text => StarkQR.generate(text))));
""",
        [case["text"] for case in cases],
    )
    assert {case["version"] for case in cases} == set(range(1, 10))
    for case, matrix in zip(cases, matrices, strict=True):
        assert len(matrix) == case["size"], case["name"]
        assert all(len(row) == case["size"] for row in matrix), case["name"]
        flattened = bytes(value for row in matrix for value in row)
        assert set(flattened) <= {0, 1}, case["name"]
        assert hashlib.sha256(flattened).hexdigest() == case["matrix_sha256"], case["name"]


def test_canvas_renders_oracle_matrix_with_quiet_zone_and_byte_limit():
    result = _node(
        """
const assert = require('assert');
require('./displays/operator/widgets/qr.js');
function canvas() {
  const rectangles = [], ctx = {fillStyle: '', fillRect(...rect) { rectangles.push([this.fillStyle, ...rect]); }};
  return {rectangles, getContext(kind) { assert.strictEqual(kind, '2d'); return ctx; }};
}
const input = 'x'.repeat(230), grid = StarkQR.generate(input), c = canvas();
assert.strictEqual(StarkQR.draw(c, input, 200), true);
assert.strictEqual(c.width, 183); assert.strictEqual(c.height, 183);
assert.deepStrictEqual(c.rectangles[0], ['#fff', 0, 0, 183, 183]);
const expected = [];
for (let r=0;r<grid.length;r++) for (let col=0;col<grid.length;col++) {
  if (grid[r][col]) expected.push(['#000', (col+4)*3, (r+4)*3, 3, 3]);
}
assert.deepStrictEqual(c.rectangles.slice(1), expected);
assert.strictEqual(StarkQR.generate('x'.repeat(231)), null);
assert.strictEqual(StarkQR.generate('é'.repeat(115)).length, 53);
assert.strictEqual(StarkQR.generate('é'.repeat(116)), null);
const rejected = canvas();
assert.strictEqual(StarkQR.draw(rejected, 'x'.repeat(231), 200), false);
assert.strictEqual(rejected.rectangles.length, 0);
process.stdout.write(JSON.stringify({ok:true}));
"""
    )
    assert result == {"ok": True}


def test_audience_overlay_uses_verified_widget_and_retains_oversized_link():
    result = _node(
        """
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const html = fs.readFileSync('displays/audience_display.html', 'utf8');
const scripts = [...html.matchAll(/<script[^>]*src="([^"]+)"/g)].map(match => match[1]);
assert(scripts.includes('operator/widgets/qr.js'));
assert(!html.includes('function generateQR('));
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {hidden:false, textContent:'', listeners:{},
    classList:{add() {},remove() {}}, getContext() { return {fillRect() {}}; },
    addEventListener(event, fn) { this.listeners[event] = fn; }});
  return elements.get(id);
}
let url = 'http://192.168.1.20:8080/displays/mobile_display.html?port=8765';
const ctx = {location:{}, document:{getElementById:element, querySelector:element},
  StarkDisplayConnection:{mobileUrl:() => url}};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync('displays/operator/widgets/qr.js','utf8'),ctx);
const start = html.indexOf('  (function() {',html.indexOf('// ── QR Code overlay'));
const end = html.indexOf('  })();',start) + '  })();'.length;
assert(start > 0 && end > start);
vm.runInContext(html.slice(start,end),ctx);
const click = () => element('.header').listeners.click({});
click();
assert.strictEqual(element('qrUrl').textContent,url);
assert.strictEqual(element('qrCanvas').hidden,false);
assert(element('qrCanvas').width > 0);
url = 'https://example.org/' + 'x'.repeat(231);
click();
assert.strictEqual(element('qrUrl').textContent,url);
assert.strictEqual(element('qrCanvas').hidden,true);
url = 'http://a.b/'; click();
assert.strictEqual(element('qrCanvas').hidden,false);
process.stdout.write(JSON.stringify({ok:true}));
"""
    )
    assert result == {"ok": True}
