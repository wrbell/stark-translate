// Tiny canvas sparkline. Pure functions, no framework.
// Usage:
//   drawSparkline(canvas, [12, 14, 11, ...], { color: "#2563aa", min: 0, max: 100 });

(function (global) {
  "use strict";

  function drawSparkline(canvas, values, opts) {
    opts = opts || {};
    const color = opts.color || "#2563aa";
    const fill = opts.fill || null; // optional fill-under color
    const ctx = canvas.getContext("2d");
    const dpr = global.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || canvas.width;
    const cssH = canvas.clientHeight || canvas.height;
    if (canvas.width !== cssW * dpr || canvas.height !== cssH * dpr) {
      canvas.width = cssW * dpr;
      canvas.height = cssH * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    if (!values || values.length === 0) {
      ctx.strokeStyle = "#ddd";
      ctx.beginPath();
      ctx.moveTo(0, cssH / 2);
      ctx.lineTo(cssW, cssH / 2);
      ctx.stroke();
      return;
    }

    let min = opts.min;
    let max = opts.max;
    if (min === undefined) min = Math.min.apply(null, values);
    if (max === undefined) max = Math.max.apply(null, values);
    if (max - min < 1e-9) {
      max = min + 1; // avoid divide-by-zero on flat series
    }

    const stepX = values.length > 1 ? cssW / (values.length - 1) : 0;
    const yFor = (v) => cssH - ((v - min) / (max - min)) * (cssH - 2) - 1;

    if (fill) {
      ctx.beginPath();
      ctx.moveTo(0, cssH);
      values.forEach((v, i) => ctx.lineTo(i * stepX, yFor(v)));
      ctx.lineTo((values.length - 1) * stepX, cssH);
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
    }

    ctx.beginPath();
    values.forEach((v, i) => {
      const x = i * stepX;
      const y = yFor(v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  global.drawSparkline = drawSparkline;
})(window);
