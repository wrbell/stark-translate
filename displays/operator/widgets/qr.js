// Offline QR encoder for the operator page (byte mode, ECC-L, versions 1-9).
// Ported from the audience display's inline encoder so the operator can show
// the phone link without a network dependency. Pure functions, no framework.
//
//   StarkQR.generate("http://…") -> square grid of 0/1 modules (or null)
//   StarkQR.draw(canvas, "http://…", 200) -> paints the grid; returns false
//     when the text does not fit (callers must then hide the canvas).

(function (global) {
  "use strict";

  // Byte-mode capacities for ECC level L. Version 10 needs four RS blocks, so
  // the encoder stops at version 9 rather than emit an undecodable code.
  const CAPACITY = [0, 17, 32, 53, 78, 106, 134, 154, 192, 230];
  const DATA_BITS = {1: 152, 2: 272, 3: 440, 4: 640, 5: 864, 6: 1088, 7: 1248, 8: 1552, 9: 1856};
  const EC_INFO = {
    1: {dc: 19, ec: 7, b: 1}, 2: {dc: 34, ec: 10, b: 1}, 3: {dc: 55, ec: 15, b: 1},
    4: {dc: 80, ec: 20, b: 1}, 5: {dc: 108, ec: 26, b: 1}, 6: {dc: 136, ec: 18, b: 2},
    7: {dc: 156, ec: 20, b: 2}, 8: {dc: 194, ec: 24, b: 2}, 9: {dc: 232, ec: 30, b: 2},
  };
  const ALIGNMENT = {2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34], 7: [6, 22, 38], 8: [6, 24, 42], 9: [6, 26, 46]};

  function utf8Bytes(text) {
    const out = [];
    for (const ch of String(text)) {
      const cp = ch.codePointAt(0);
      if (cp < 0x80) out.push(cp);
      else if (cp < 0x800) out.push(0xc0 | (cp >> 6), 0x80 | (cp & 63));
      else if (cp < 0x10000) out.push(0xe0 | (cp >> 12), 0x80 | ((cp >> 6) & 63), 0x80 | (cp & 63));
      else out.push(0xf0 | (cp >> 18), 0x80 | ((cp >> 12) & 63), 0x80 | ((cp >> 6) & 63), 0x80 | (cp & 63));
    }
    return out;
  }

  function versionFor(length) {
    for (let v = 1; v < CAPACITY.length; v++) if (CAPACITY[v] >= length) return v;
    return 0;
  }

  function rsEncode(data, numEC) {
    const gfExp = new Uint8Array(512);
    const gfLog = new Uint8Array(256);
    let x = 1;
    for (let i = 0; i < 255; i++) {
      gfExp[i] = x; gfLog[x] = i;
      x = (x << 1) ^ (x >= 128 ? 0x11d : 0);
    }
    for (let i = 255; i < 512; i++) gfExp[i] = gfExp[i - 255];
    const gfMul = (a, b) => (a === 0 || b === 0 ? 0 : gfExp[gfLog[a] + gfLog[b]]);
    let gen = [1];
    for (let i = 0; i < numEC; i++) {
      const next = new Array(gen.length + 1).fill(0);
      for (let j = 0; j < gen.length; j++) {
        next[j] ^= gen[j];
        next[j + 1] ^= gfMul(gen[j], gfExp[i]);
      }
      gen = next;
    }
    const msg = new Array(data.length + numEC).fill(0);
    for (let i = 0; i < data.length; i++) msg[i] = data[i];
    for (let i = 0; i < data.length; i++) {
      const coef = msg[i];
      if (coef !== 0) for (let j = 1; j < gen.length; j++) msg[i + j] ^= gfMul(gen[j], coef);
    }
    return msg.slice(data.length);
  }

  function addErrorCorrection(codewords, info) {
    const {dc, ec, b} = info;
    const blockSize = Math.floor(dc / b);
    const blocks = [];
    let offset = 0;
    for (let i = 0; i < b; i++) {
      const size = i < b - (dc % b) ? blockSize : blockSize + 1;
      blocks.push(codewords.slice(offset, offset + size));
      offset += size;
    }
    const ecBlocks = blocks.map(block => rsEncode(block, ec));
    const result = [];
    const maxData = Math.max(...blocks.map(block => block.length));
    for (let i = 0; i < maxData; i++) for (const block of blocks) if (i < block.length) result.push(block[i]);
    for (let i = 0; i < ec; i++) for (const block of ecBlocks) if (i < block.length) result.push(block[i]);
    return result;
  }

  function formatBits(mask, eccLevel) {
    const value = (eccLevel << 3) | mask;
    let rem = value;
    for (let i = 0; i < 10; i++) rem = (rem << 1) ^ ((rem >> 9) ? 0x537 : 0);
    const bits = ((value << 10) | rem) ^ 0x5412;
    const out = [];
    for (let i = 14; i >= 0; i--) out.push((bits >> i) & 1);
    return out;
  }

  function placeFinder(grid, reserved, row, col) {
    const size = grid.length;
    for (let r = -1; r <= 7; r++) {
      for (let c = -1; c <= 7; c++) {
        const rr = row + r, cc = col + c;
        if (rr < 0 || rr >= size || cc < 0 || cc >= size) continue;
        const outer = r === 0 || r === 6 || c === 0 || c === 6;
        const inner = r >= 2 && r <= 4 && c >= 2 && c <= 4;
        grid[rr][cc] = (outer || inner) && r >= 0 && r <= 6 && c >= 0 && c <= 6 ? 1 : 0;
        reserved[rr][cc] = true;
      }
    }
  }

  function placeAlignment(grid, reserved, centerR, centerC) {
    for (let r = -2; r <= 2; r++) {
      for (let c = -2; c <= 2; c++) {
        grid[centerR + r][centerC + c] = Math.abs(r) === 2 || Math.abs(c) === 2 || (r === 0 && c === 0) ? 1 : 0;
        reserved[centerR + r][centerC + c] = true;
      }
    }
  }

  function placeFormat(grid, bits) {
    const size = grid.length;
    const first = [[8, 0], [8, 1], [8, 2], [8, 3], [8, 4], [8, 5], [8, 7], [8, 8], [7, 8], [5, 8], [4, 8], [3, 8], [2, 8], [1, 8], [0, 8]];
    const second = [];
    for (let i = 1; i <= 7; i++) second.push([size - i, 8]);
    for (let i = 8; i >= 1; i--) second.push([8, size - i]);
    for (let i = 0; i < 15; i++) {
      grid[first[i][0]][first[i][1]] = bits[i];
      grid[second[i][0]][second[i][1]] = bits[i];
    }
  }

  function generate(text) {
    const data = utf8Bytes(text);
    const ver = versionFor(data.length);
    if (!ver) return null;
    const size = ver * 4 + 17;
    const totalBits = DATA_BITS[ver];
    const bits = [];
    const push = (value, len) => { for (let i = len - 1; i >= 0; i--) bits.push((value >> i) & 1); };
    push(0b0100, 4);
    push(data.length, 8);
    for (const byte of data) push(byte, 8);
    push(0, Math.min(4, totalBits - bits.length));
    while (bits.length % 8 !== 0) bits.push(0);
    for (let pad = 0xec; bits.length < totalBits; pad ^= 0xfd) push(pad, 8);
    const codewords = [];
    for (let i = 0; i < bits.length; i += 8) {
      let byte = 0;
      for (let j = 0; j < 8; j++) byte = (byte << 1) | bits[i + j];
      codewords.push(byte);
    }
    const all = addErrorCorrection(codewords, EC_INFO[ver]);

    const grid = Array.from({length: size}, () => Array(size).fill(null));
    const reserved = Array.from({length: size}, () => Array(size).fill(false));
    placeFinder(grid, reserved, 0, 0);
    placeFinder(grid, reserved, 0, size - 7);
    placeFinder(grid, reserved, size - 7, 0);
    for (let i = 8; i < size - 8; i++) {
      grid[6][i] = i % 2 === 0 ? 1 : 0; grid[i][6] = i % 2 === 0 ? 1 : 0;
      reserved[6][i] = true; reserved[i][6] = true;
    }
    for (const r of ALIGNMENT[ver] || []) {
      for (const c of ALIGNMENT[ver] || []) {
        if (reserved[r][c]) continue;
        placeAlignment(grid, reserved, r, c);
      }
    }
    for (let i = 0; i < 8; i++) {
      reserved[8][i] = true; reserved[i][8] = true;
      reserved[8][size - 1 - i] = true; reserved[size - 1 - i][8] = true;
    }
    reserved[8][8] = true;
    grid[size - 8][8] = 1; reserved[size - 8][8] = true;

    const dataBits = [];
    for (const byte of all) for (let i = 7; i >= 0; i--) dataBits.push((byte >> i) & 1);
    let index = 0, upward = true;
    for (let col = size - 1; col >= 1; col -= 2) {
      if (col === 6) col = 5;
      for (let step = 0; step < size; step++) {
        const row = upward ? size - 1 - step : step;
        for (const c of [col, col - 1]) {
          if (reserved[row][c]) continue;
          grid[row][c] = index < dataBits.length ? dataBits[index++] : 0;
        }
      }
      upward = !upward;
    }
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        if (!reserved[r][c] && grid[r][c] !== null && (r + c) % 2 === 0) grid[r][c] ^= 1;
      }
    }
    placeFormat(grid, formatBits(0, 1));
    for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) if (grid[r][c] === null) grid[r][c] = 0;
    return grid;
  }

  function draw(canvas, text, size) {
    const modules = generate(text);
    if (!modules) return false;
    const n = modules.length;
    const quiet = 4;
    const cell = Math.max(1, Math.floor((size || 200) / (n + quiet * 2)));
    canvas.width = cell * (n + quiet * 2);
    canvas.height = canvas.width;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#000";
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        if (modules[r][c]) ctx.fillRect((c + quiet) * cell, (r + quiet) * cell, cell, cell);
      }
    }
    return true;
  }

  global.StarkQR = {generate, draw, formatBits, versionFor, capacity: CAPACITY.slice()};
})(typeof window !== "undefined" ? window : globalThis);
