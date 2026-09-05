// Models section of the playground: ready-to-use neural networks, separate from the operators.
// Each model is an ONNX graph (vendored under models/ or fetched from the Hugging Face hub) run with
// onnxruntime-web; the task-specific decoding (boxes, faces, depth colour map, upscaled image) is here.
(function () {
  "use strict";

  function el(tag, cls, attrs) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    for (const key in attrs || {}) node.setAttribute(key, attrs[key]);
    return node;
  }

  const sessions = {};   // url -> Promise<InferenceSession>
  const labelsCache = {};
  let listRows = [];

  // ------------------------------------------------------------------ sidebar list

  function buildList(index, side, select, ROOT) {
    const nav = el("nav", "pg-side-list pg-models-list", { id: "pg-models-list", "aria-label": "Models" });
    nav.hidden = document.body.dataset.mode !== "models";
    listRows = [];
    index.tasks.forEach(function (task) {
      const models = index.models.filter(function (m) { return m.task === task; });
      if (!models.length) return;
      const section = el("section", "pg-pkg");
      const h = el("h2");
      h.textContent = task;
      const n = el("span", "pg-pkg-count");
      n.textContent = String(models.length);
      h.appendChild(n);
      section.appendChild(h);
      const ul = el("ul", "pg-list");
      models.forEach(function (m) {
        const li = el("li", "pg-row pg-row-live");
        const a = el("a", "pg-row-name pg-row-model", { href: ROOT + "models/" + m.slug + "/", title: m.summary });
        const title = el("span", "pg-model-title");
        title.textContent = m.title || m.name;
        const sub = el("span", "pg-model-sub");
        const subText = el("span");
        subText.textContent = m.subtitle || m.task.toLowerCase();
        const size = el("span", "pg-model-size");
        size.textContent = m.size_mb >= 1 ? Math.round(m.size_mb) + " MB" : (m.size_mb * 1000).toFixed(0) + " kB";
        sub.appendChild(subText);
        sub.appendChild(size);
        a.appendChild(title);
        a.appendChild(sub);
        a.addEventListener("click", function (ev) { ev.preventDefault(); select(m, true); });
        li.appendChild(a);
        ul.appendChild(li);
        listRows.push({ model: m, a: a, li: li, text: (m.name + " " + m.id + " " + m.task + " " + m.summary).toLowerCase() });
      });
      section.appendChild(ul);
      nav.appendChild(section);
    });
    side.appendChild(nav);
    const search = document.getElementById("pg-search");
    if (search) {
      search.addEventListener("input", function () {
        const q = search.value.trim().toLowerCase();
        listRows.forEach(function (r) { r.li.hidden = !!q && r.text.indexOf(q) === -1; });
      });
    }
  }

  // ---- runs used today, from the API; announced to whoever shows it
  let quota = null, quotaError = null;
  const quotaListeners = [];
  // seconds of server time, shown as a short duration
  function fmtSecs(s) { s = Number(s) || 0; return s >= 60 ? Math.floor(s / 60) + " min" + (Math.round(s % 60) ? " " + Math.round(s % 60) + " s" : "") : s >= 10 ? Math.round(s) + " s" : s.toFixed(1) + " s"; }
  function quotaText() {
    const A = window.KorniaAuth;
    if (!A || !A.user) return "";
    if (quotaError) return quotaError;
    if (!quota) return "";
    return fmtSecs(quota.used) + " of " + fmtSecs(quota.limit) + " of server time used today" + (quota.remaining === 0 ? " · resets at 00:00 UTC" : "");
  }
  function setQuota(q, err) { quota = q; quotaError = err || null; quotaListeners.forEach(function (fn) { fn(quotaText()); }); }
  async function refreshQuota() {
    const A = window.KorniaAuth;
    if (!A || !A.user) { setQuota(null); return; }
    if (!A.apiBase) { setQuota(null, "the server side is not deployed yet"); return; }
    try {
      const r = await fetch(A.apiBase + "/v1/me", { headers: { Authorization: "Bearer " + await A.token() } });
      const j = await r.json().catch(function () { return {}; });
      if (r.ok) setQuotaAndAnnounce(j.quota);
      else setQuota(null, (j.detail || ("HTTP " + r.status)).indexOf("verify your email") !== -1 ? "verify your email address to run models" : (j.detail || ("HTTP " + r.status)));
    } catch (e) { setQuota(null, "the server did not answer"); }
  }
  function onQuota(fn) { quotaListeners.push(fn); fn(quotaText()); }
  function setQuotaAndAnnounce(q, err) { setQuota(q, err); window.dispatchEvent(new CustomEvent("kornia-quota", { detail: quota })); }
  if (window.KorniaAuth) window.KorniaAuth.onChange(function () { document.body.classList.toggle("pg-signed-in", !!window.KorniaAuth.user); refreshQuota(); });
  else document.addEventListener("DOMContentLoaded", function () { if (window.KorniaAuth) window.KorniaAuth.onChange(function () { document.body.classList.toggle("pg-signed-in", !!window.KorniaAuth.user); refreshQuota(); }); });

  function highlight(model) {
    listRows.forEach(function (r) {
      const on = r.model.id === model.id;
      r.a.classList.toggle("pg-row-current", on);
      if (on) r.a.setAttribute("aria-current", "page"); else r.a.removeAttribute("aria-current");
    });
  }

  // ------------------------------------------------------------------ loading with progress

  function fetchWithProgress(url, onProgress) {
    return fetch(url).then(function (response) {
      if (!response.ok) throw new Error("HTTP " + response.status + " fetching the model");
      const total = Number(response.headers.get("Content-Length")) || 0;
      if (!response.body || !response.body.getReader) return response.arrayBuffer().then(function (b) { return new Uint8Array(b); });
      const reader = response.body.getReader();
      const chunks = [];
      let received = 0;
      function pump() {
        return reader.read().then(function (r) {
          if (r.done) return;
          chunks.push(r.value);
          received += r.value.length;
          onProgress(received, total);
          return pump();
        });
      }
      return pump().then(function () {
        const out = new Uint8Array(received);
        let offset = 0;
        chunks.forEach(function (c) { out.set(c, offset); offset += c.length; });
        return out;
      });
    });
  }

  function absUrl(url, ROOT) { return /^https?:/.test(url) ? url : ROOT + url; }

  function getSessionByUrl(url, onProgress) {
    if (!sessions[url]) {
      sessions[url] = fetchWithProgress(url, onProgress || function () {}).then(function (bytes) {
        return ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      }).catch(function (err) { delete sessions[url]; throw err; });
    }
    return sessions[url];
  }

  // the main graph reports progress; the auxiliary graphs (descriptor, warp) are small and load alongside
  function getSession(model, ROOT, onProgress) {
    const extras = Object.keys(model.graphs || {}).map(function (k) { return getSessionByUrl(absUrl(model.graphs[k], ROOT)); });
    return Promise.all([getSessionByUrl(absUrl(model.url, ROOT), onProgress)].concat(extras)).then(function (all) { return all[0]; });
  }

  function getLabels(name, ROOT) {
    if (!labelsCache[name]) labelsCache[name] = fetch(ROOT + "models/" + name + "_labels.json").then(function (r) { return r.json(); });
    return labelsCache[name];
  }

  // ------------------------------------------------------------------ tensors

  function canvasToTensor(canvas, input) {
    const size = input.size;
    const data = canvas.getContext("2d").getImageData(0, 0, size, size).data;
    const plane = size * size;
    if (input.gray) {
      const g = new Float32Array(plane);
      const scale = input.scale || 1;
      for (let i = 0; i < plane; i++) g[i] = ((0.299 * data[4 * i] + 0.587 * data[4 * i + 1] + 0.114 * data[4 * i + 2]) / 255) * scale;
      return new ort.Tensor("float32", g, [1, 1, size, size]);
    }
    const out = new Float32Array(3 * plane);
    const mean = input.mean || [0, 0, 0];
    const std = input.std || [1, 1, 1];
    const scale = input.scale || 1;
    for (let i = 0; i < plane; i++) {
      for (let c = 0; c < 3; c++) {
        out[c * plane + i] = ((data[4 * i + c] / 255) * scale - mean[c]) / std[c];
      }
    }
    return new ort.Tensor("float32", out, [1, 3, size, size]);
  }

  function drawImageTensor(tensor, canvas) {
    const dims = tensor.dims;
    const c = dims[dims.length - 3], h = dims[dims.length - 2], w = dims[dims.length - 1];
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(w, h);
    const plane = h * w;
    const src = tensor.data;
    for (let i = 0; i < plane; i++) {
      const r = src[i], g = c === 1 ? r : src[plane + i], b = c === 1 ? r : src[2 * plane + i];
      img.data[4 * i] = Math.max(0, Math.min(255, Math.round(r * 255)));
      img.data[4 * i + 1] = Math.max(0, Math.min(255, Math.round(g * 255)));
      img.data[4 * i + 2] = Math.max(0, Math.min(255, Math.round(b * 255)));
      img.data[4 * i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  // A compact approximation of the Turbo colour map (near = bright, far = dark blue).
  const TURBO = [[48, 18, 59], [70, 107, 227], [40, 187, 236], [31, 233, 168], [135, 253, 78], [225, 220, 55], [253, 149, 39], [223, 64, 17], [122, 4, 3]];
  function turbo(t) {
    const x = Math.max(0, Math.min(1, t)) * (TURBO.length - 1);
    const i = Math.min(TURBO.length - 2, Math.floor(x));
    const f = x - i;
    const a = TURBO[i], b = TURBO[i + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }

  function drawDepth(tensor, canvas) {
    const dims = tensor.dims;
    const h = dims[dims.length - 2], w = dims[dims.length - 1];
    const src = tensor.data;
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < h * w; i++) { const v = src[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
    const scale = mx > mn ? 1 / (mx - mn) : 1;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(w, h);
    for (let i = 0; i < h * w; i++) {
      const rgb = turbo((src[i] - mn) * scale);
      img.data[4 * i] = rgb[0]; img.data[4 * i + 1] = rgb[1]; img.data[4 * i + 2] = rgb[2]; img.data[4 * i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  // ------------------------------------------------------------------ detections

  function drawBox(ctx, x, y, w, h, label, color) {
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);
    if (label) {
      ctx.font = "12px Inter, system-ui, sans-serif";
      const tw = ctx.measureText(label).width + 8;
      ctx.fillStyle = color;
      ctx.fillRect(x, Math.max(0, y - 16), tw, 16);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x + 4, Math.max(12, y - 4));
    }
  }

  const PALETTE = ["#2563eb", "#16a34a", "#dc2626", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#65a30d"];

  function drawDetections(results, model, canvasIn, canvasOut, labels, threshold) {
    const out = results[Object.keys(results)[0]];
    const rows = out.dims[out.dims.length - 2], cols = out.dims[out.dims.length - 1];
    canvasOut.width = canvasIn.width;
    canvasOut.height = canvasIn.height;
    const ctx = canvasOut.getContext("2d");
    ctx.drawImage(canvasIn, 0, 0);
    // boxes are in the model's own coordinate space (RT-DETR resizes to 480 inside the graph)
    const k = canvasOut.width / (model.output.coord_size || canvasIn.width);
    let kept = 0;
    for (let i = 0; i < rows; i++) {
      const cls = out.data[i * cols], score = out.data[i * cols + 1];
      if (score < threshold) continue;
      const x = out.data[i * cols + 2] * k, y = out.data[i * cols + 3] * k, w = out.data[i * cols + 4] * k, h = out.data[i * cols + 5] * k;
      const name = labels && labels[Math.round(cls)] ? labels[Math.round(cls)] : String(Math.round(cls));
      drawBox(ctx, x, y, w, h, name + " " + Math.round(score * 100) + "%", PALETTE[Math.round(cls) % PALETTE.length]);
      kept++;
    }
    return kept + " object" + (kept === 1 ? "" : "s") + " above " + Math.round(threshold * 100) + "%";
  }

  // ------------------------------------------------------------------ YuNet faces: priors, decode, NMS (port of kornia.contrib.FaceDetector)

  const priorsCache = {};
  function priors(cfg, size) {
    const key = size + ":" + cfg.steps.join(",");
    if (priorsCache[key]) return priorsCache[key];
    const fm2 = [Math.floor(Math.floor((size + 1) / 2) / 2), Math.floor(Math.floor((size + 1) / 2) / 2)];
    const maps = [];
    let cur = [Math.floor(fm2[0] / 2), Math.floor(fm2[1] / 2)];
    for (let k = 0; k < 4; k++) { maps.push(cur); cur = [Math.floor(cur[0] / 2), Math.floor(cur[1] / 2)]; }
    const out = [];
    maps.forEach(function (f, k) {
      for (let i = 0; i < f[0]; i++) for (let j = 0; j < f[1]; j++) cfg.min_sizes[k].forEach(function (ms) {
        out.push([(j + 0.5) * cfg.steps[k] / size, (i + 0.5) * cfg.steps[k] / size, ms / size, ms / size]);
      });
    });
    priorsCache[key] = out;
    return out;
  }

  function nms(boxes, scores, thr) {
    const order = scores.map(function (s, i) { return i; }).sort(function (a, b) { return scores[b] - scores[a]; });
    const keep = [];
    const removed = new Set();
    for (let oi = 0; oi < order.length; oi++) {
      const i = order[oi];
      if (removed.has(i)) continue;
      keep.push(i);
      const a = boxes[i], areaA = (a[2] - a[0]) * (a[3] - a[1]);
      for (let oj = oi + 1; oj < order.length; oj++) {
        const j = order[oj];
        if (removed.has(j)) continue;
        const b = boxes[j];
        const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
        const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
        const inter = w * h;
        const iou = inter / (areaA + (b[2] - b[0]) * (b[3] - b[1]) - inter);
        if (iou > thr) removed.add(j);
      }
    }
    return keep;
  }

  // loc/conf/iou -> {boxes:[x1,y1,x2,y2], scores, keypoints:[[x,y]x5]} in pixels of a size x size input
  function decodeFaces(results, cfg, size, threshold) {
    const loc = results.loc, conf = results.conf, iou = results.iou;
    const pri = priors(cfg, size);
    const v = cfg.variance;
    const n = loc.dims[1];
    const boxes = [], scores = [], kps = [];
    for (let i = 0; i < n; i++) {
      const p = pri[i];
      const l = loc.data.subarray(i * 14, i * 14 + 14);
      const cx = p[0] + l[0] * v[0] * p[2], cy = p[1] + l[1] * v[0] * p[3];
      const w = p[2] * Math.exp(l[2] * v[1]), h = p[3] * Math.exp(l[3] * v[1]);
      const score = Math.sqrt(conf.data[i * 2 + 1] * Math.max(0, Math.min(1, iou.data[i])));
      if (score <= threshold) continue;
      boxes.push([(cx - w / 2) * size, (cy - h / 2) * size, (cx + w / 2) * size, (cy + h / 2) * size]);
      scores.push(score);
      const pts = [];
      for (let k = 0; k < 5; k++) pts.push([(p[0] + l[4 + 2 * k] * v[0] * p[2]) * size, (p[1] + l[5 + 2 * k] * v[0] * p[3]) * size]);
      kps.push(pts);
    }
    const keep = nms(boxes, scores, cfg.nms);
    return { boxes: keep.map(function (i) { return boxes[i]; }), scores: keep.map(function (i) { return scores[i]; }), keypoints: keep.map(function (i) { return kps[i]; }) };
  }

  function paintFaces(ctx, faces) {
    faces.boxes.forEach(function (b, i) {
      drawBox(ctx, b[0], b[1], b[2] - b[0], b[3] - b[1], Math.round(faces.scores[i] * 100) + "%", "#16a34a");
      ctx.fillStyle = "#facc15";
      faces.keypoints[i].forEach(function (pt) { ctx.beginPath(); ctx.arc(pt[0], pt[1], 2.5, 0, Math.PI * 2); ctx.fill(); });
    });
  }

  function drawFaces(results, model, canvasIn, canvasOut, threshold) {
    const faces = decodeFaces(results, model.output, canvasIn.width, threshold);   // priors follow the actual input size
    canvasOut.width = canvasIn.width;
    canvasOut.height = canvasIn.height;
    const ctx = canvasOut.getContext("2d");
    ctx.drawImage(canvasIn, 0, 0);
    paintFaces(ctx, faces);
    return faces.boxes.length + " face" + (faces.boxes.length === 1 ? "" : "s") + " above " + Math.round(threshold * 100) + "%";
  }

  // ------------------------------------------------------------------ local features

  // positions that are the maximum of their k x k window and above thr, as [x, y, value], strongest first
  function heatmapPeaks(data, w, h, k, thr, topK) {
    const r = Math.floor(k / 2);
    const out = [];
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const v = data[y * w + x];
        if (v <= thr) continue;
        let isMax = true;
        for (let dy = -r; dy <= r && isMax; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= h) continue;
          for (let dx = -r; dx <= r; dx++) {
            const xx = x + dx;
            if (xx < 0 || xx >= w || (dx === 0 && dy === 0)) continue;
            if (data[yy * w + xx] > v) { isMax = false; break; }
          }
        }
        if (isMax) out.push([x, y, v]);
      }
    }
    out.sort(function (a, b) { return b[2] - a[2]; });
    return topK ? out.slice(0, topK) : out;
  }

  // grid_sample(bilinear, align_corners=false, zeros) at XFeat's 2*x/(W-1)-1 coordinates; fm is (C, h, w)
  function sampleBilinear(fm, C, h, w, x, y, W, H) {
    const ix = ((2 * x / (W - 1)) * w) / 2 + (w - 1) / 2 - w / 2 + 0.5 * 0; // = x/(W-1)*w - 0.5
    const iy = ((2 * y / (H - 1)) * h) / 2 + (h - 1) / 2 - h / 2 + 0.5 * 0;
    const px = x / (W - 1) * w - 0.5, py = y / (H - 1) * h - 0.5;
    const x0 = Math.floor(px), y0 = Math.floor(py);
    const out = new Float32Array(C);
    for (let dy = 0; dy <= 1; dy++) for (let dx = 0; dx <= 1; dx++) {
      const xi = x0 + dx, yi = y0 + dy;
      if (xi < 0 || xi >= w || yi < 0 || yi >= h) continue;
      const wgt = (1 - Math.abs(px - xi)) * (1 - Math.abs(py - yi));
      for (let c = 0; c < C; c++) out[c] += fm[c * h * w + yi * w + xi] * wgt;
    }
    void ix; void iy;
    return out;
  }

  function normalizeRows(descs) {
    descs.forEach(function (d) {
      let n = 0;
      for (let i = 0; i < d.length; i++) n += d[i] * d[i];
      n = Math.sqrt(n) || 1;
      for (let i = 0; i < d.length; i++) d[i] /= n;
    });
    return descs;
  }

  // XFeat: heatmap peaks scored by the reliability map, descriptors sampled from the 1/8-resolution map
  function xfeatFeatures(results, cfg, size) {
    const heat = results.keypoint_heatmap, rel = results.reliability, desc = results.descriptors;
    const H = heat.dims[2], W = heat.dims[3], h = rel.dims[2], w = rel.dims[3], C = desc.dims[1];
    const peaks = heatmapPeaks(heat.data, W, H, cfg.nms, cfg.threshold, 0);
    const scored = peaks.map(function (p) { return [p[0], p[1], p[2] * sampleBilinear(rel.data, 1, h, w, p[0], p[1], W, H)[0]]; })
      .filter(function (p) { return p[2] > 0; }).sort(function (a, b) { return b[2] - a[2]; }).slice(0, cfg.top_k);
    const descs = normalizeRows(scored.map(function (p) { return sampleBilinear(desc.data, C, h, w, p[0], p[1], W, H); }));
    return { points: scored.map(function (p) { return [p[0] * size / W, p[1] * size / H]; }), descriptors: descs };
  }

  // mutual nearest neighbours on unit descriptors; returns [i, j, cosine]
  function matchMNN(a, b, minCos) {
    if (!a.length || !b.length) return [];
    const bestA = new Array(a.length), bestB = new Int32Array(b.length).fill(-1), bestBv = new Float32Array(b.length).fill(-2);
    for (let i = 0; i < a.length; i++) {
      let bj = -1, bv = -2;
      for (let j = 0; j < b.length; j++) {
        let s = 0;
        const x = a[i], y = b[j];
        for (let k = 0; k < x.length; k++) s += x[k] * y[k];
        if (s > bv) { bv = s; bj = j; }
        if (s > bestBv[j]) { bestBv[j] = s; bestB[j] = i; }
      }
      bestA[i] = [bj, bv];
    }
    const out = [];
    for (let i = 0; i < a.length; i++) {
      const j = bestA[i][0];
      if (j >= 0 && bestB[j] === i && bestA[i][1] >= minCos) out.push([i, j, bestA[i][1]]);
    }
    return out;
  }

  // a warped copy of the input, through the homepage's affine graph (random parameters, held until re-warp).
  // Deliberately gentle: a few degrees, a few percent of translation and scale, no shear, so the pair stays
  // a realistic "next frame" and most keypoints have a counterpart to match.
  function randomAffine(size) {
    const u = function (lo, hi) { return lo + Math.random() * (hi - lo); };
    return { angle: u(-8, 8), tx: u(-0.05, 0.05) * size, ty: u(-0.05, 0.05) * size, scale: u(0.95, 1.05), shear: 0 };
  }
  function warpCanvas(sessionWarp, canvasIn, canvasB, size, params) {
    const feeds = { image: canvasToTensor(canvasIn, { size: size, scale: 1 }) };
    ["angle", "tx", "ty", "scale", "shear"].forEach(function (n) { feeds[n] = new ort.Tensor("float32", new Float32Array([params[n]]), [1]); });
    return sessionWarp.run(feeds).then(function (r) { drawImageTensor(r[Object.keys(r)[0]], canvasB); });
  }

  // grayscale patches around keypoints: a `support` px window resized to `patch` px, as (N, 1, patch, patch)
  function extractPatches(canvas, points, support, patch) {
    const scratch = document.createElement("canvas");
    scratch.width = patch; scratch.height = patch;
    const sctx = scratch.getContext("2d");
    const out = new Float32Array(points.length * patch * patch);
    points.forEach(function (p, n) {
      sctx.fillStyle = "#000";
      sctx.fillRect(0, 0, patch, patch);
      sctx.drawImage(canvas, p[0] - support / 2, p[1] - support / 2, support, support, 0, 0, patch, patch);
      const d = sctx.getImageData(0, 0, patch, patch).data;
      for (let i = 0; i < patch * patch; i++) out[n * patch * patch + i] = (0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2]) / 255;
    });
    return new ort.Tensor("float32", out, [points.length, 1, patch, patch]);
  }

  function drawMatches(canvasA, canvasB, canvasOut, fa, fb, matches) {
    const size = canvasA.width;
    canvasOut.width = size * 2 + 8;
    canvasOut.height = size;
    const ctx = canvasOut.getContext("2d");
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, canvasOut.width, canvasOut.height);
    ctx.drawImage(canvasA, 0, 0);
    ctx.drawImage(canvasB, size + 8, 0);
    const off = size + 8;
    ctx.lineWidth = 1;
    matches.forEach(function (m, i) {
      const p = fa.points[m[0]], q = fb.points[m[1]];
      ctx.strokeStyle = "hsla(" + ((i * 47) % 360) + ", 90%, 60%, 0.75)";
      ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0] + off, q[1]); ctx.stroke();
    });
    ctx.fillStyle = "#facc15";
    fa.points.forEach(function (p) { ctx.fillRect(p[0] - 1.5, p[1] - 1.5, 3, 3); });
    fb.points.forEach(function (p) { ctx.fillRect(p[0] + off - 1.5, p[1] - 1.5, 3, 3); });
  }

  // ------------------------------------------------------------------ classification

  function drawClassification(results, canvasIn, canvasOut, labels, top) {
    const logits = results[Object.keys(results)[0]].data;
    let mx = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];
    let sum = 0;
    const probs = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) { probs[i] = Math.exp(logits[i] - mx); sum += probs[i]; }
    const idx = Array.from(probs.keys()).sort(function (a, b) { return probs[b] - probs[a]; }).slice(0, top);
    canvasOut.width = canvasIn.width;
    canvasOut.height = canvasIn.height;
    const ctx = canvasOut.getContext("2d");
    ctx.drawImage(canvasIn, 0, 0);
    ctx.fillStyle = "rgba(15, 23, 42, 0.72)";
    ctx.fillRect(0, canvasOut.height - (top * 22 + 12), canvasOut.width, top * 22 + 12);
    ctx.font = "12px Inter, system-ui, sans-serif";
    idx.forEach(function (i, row) {
      const y = canvasOut.height - (top - row) * 22 - 4;
      const p = probs[i] / sum;
      ctx.fillStyle = "rgba(37, 99, 235, 0.85)";
      ctx.fillRect(8, y - 12, (canvasOut.width - 16) * p, 14);
      ctx.fillStyle = "#fff";
      ctx.fillText((labels ? labels[i] : String(i)) + "  " + (100 * p).toFixed(1) + "%", 12, y);
    });
    return labels ? labels[idx[0]] + " " + (100 * probs[idx[0]] / sum).toFixed(1) + "%" : "class " + idx[0];
  }

  // Draw a model's raw outputs for the image in canvasIn onto canvasOut; returns the status text.
  function drawResults(model, results, canvasIn, canvasOut, opts) {
    opts = opts || {};
    const kind = model.output.kind;
    if (kind === "detections") return drawDetections(results, model, canvasIn, canvasOut, opts.labels, opts.threshold || model.output.threshold || 0.5);
    if (kind === "faces_yunet") return drawFaces(results, model, canvasIn, canvasOut, opts.threshold || model.output.threshold || 0.5);
    if (kind === "depth") { drawDepth(results[Object.keys(results)[0]], canvasOut); return "relative depth, near is bright"; }
    if (kind === "classification") return drawClassification(results, canvasIn, canvasOut, opts.labels, model.output.top || 5);
    const t = results[Object.keys(results)[0]];
    drawImageTensor(t, canvasOut);
    return "upscaled to " + t.dims[3] + "×" + t.dims[2];
  }

  // ------------------------------------------------------------------ the server: one call, one result

  function canvasBlob(canvas) { return new Promise(function (res) { canvas.toBlob(res, "image/png"); }); }

  // Send one or two canvases to the API for `model`; resolves with the JSON result. Rejects with a readable error.
  async function runOnServer(model, canvasA, canvasB) {
    const A = window.KorniaAuth;
    if (!A || !A.configured) throw new Error("Accounts are not configured on this copy of the site.");
    if (!A.user) throw new Error("sign in first");
    if (!A.apiBase) throw new Error("the server side is not deployed yet");
    const form = new FormData();
    form.append("image", await canvasBlob(canvasA), "a.png");
    if (canvasB) form.append("image1", await canvasBlob(canvasB), "b.png");
    const r = await fetch(A.apiBase + "/v1/run/" + model.server.endpoint, { method: "POST", body: form, headers: { Authorization: "Bearer " + await A.token() } });
    const j = await r.json().catch(function () { return {}; });
    if (!r.ok) {
      const d = j.detail || ("HTTP " + r.status);
      throw new Error(d.indexOf("verify your email") !== -1 ? "Your account's email is not verified yet; open the link in the verification mail, then try again." : d);
    }
    return j;
  }

  // Draw a server result for the frame(s) it was computed on; returns a one-line summary and stores it on res.summary.
  function drawServerResult(model, res, canvasIn, canvasB, canvasOut, opts) {
    opts = opts || {};
    let summary = "";
    if (res.kind === "image") {
      const img = new Image();
      img.onload = function () {
        canvasOut.width = img.width; canvasOut.height = img.height;
        const ctx = canvasOut.getContext("2d");
        ctx.drawImage(img, 0, 0);
        if (res.colormap === "turbo") {
          const d = ctx.getImageData(0, 0, img.width, img.height);
          if (opts.onDepth) { const raw = new Float32Array(img.width * img.height); for (let i = 0; i < raw.length; i++) raw[i] = d.data[i * 4] / 255; opts.onDepth(raw, img.width, img.height); }
          for (let i = 0; i < d.data.length; i += 4) { const c = turbo(d.data[i] / 255); d.data[i] = c[0]; d.data[i + 1] = c[1]; d.data[i + 2] = c[2]; }
          ctx.putImageData(d, 0, 0);
        }
      };
      img.src = res.png;
      summary = res.label || "image";
    } else if (res.kind === "detections") {
      const thr = opts.threshold || model.output.threshold || 0.4;
      canvasOut.width = canvasIn.width; canvasOut.height = canvasIn.height;
      const ctx = canvasOut.getContext("2d");
      ctx.drawImage(canvasIn, 0, 0);
      const k = canvasOut.width / res.size[0];
      let kept = 0;
      res.boxes.forEach(function (r) {
        if (r[1] < thr) return;
        const name = res.labels && res.labels[r[0]] ? res.labels[r[0]] : String(r[0]);
        drawBox(ctx, r[2] * k, r[3] * k, r[4] * k, r[5] * k, name + " " + Math.round(r[1] * 100) + "%", PALETTE[r[0] % PALETTE.length]);
        kept++;
      });
      summary = kept + " object" + (kept === 1 ? "" : "s") + " above " + Math.round(thr * 100) + "%";
    } else if (res.kind === "faces") {
      const thr = opts.threshold || model.output.threshold || 0.5;
      canvasOut.width = canvasIn.width; canvasOut.height = canvasIn.height;
      const ctx = canvasOut.getContext("2d");
      ctx.drawImage(canvasIn, 0, 0);
      const k = canvasOut.width / res.size[0];
      const kept = res.faces.filter(function (f) { return f.score >= thr; });
      paintFaces(ctx, { boxes: kept.map(function (f) { return f.box.map(function (v) { return v * k; }); }), scores: kept.map(function (f) { return f.score; }), keypoints: kept.map(function (f) { return f.keypoints.map(function (p) { return [p[0] * k, p[1] * k]; }); }) });
      summary = kept.length + " face" + (kept.length === 1 ? "" : "s") + " above " + Math.round(thr * 100) + "%";
    } else if (res.kind === "matches") {
      const kA = canvasIn.width / res.size0[0], kB = (canvasB || canvasIn).width / res.size1[0];
      const fa = { points: res.keypoints0.map(function (p) { return [p[0] * kA, p[1] * kA]; }) }, fb = { points: res.keypoints1.map(function (p) { return [p[0] * kB, p[1] * kB]; }) };
      drawMatches(canvasIn, canvasB || canvasIn, canvasOut, fa, fb, res.keypoints0.map(function (_, i) { return [i, i, res.confidence[i]]; }));
      summary = res.matches + " matches" + (res.keypoints_all ? " from " + res.keypoints_all[0] + " + " + res.keypoints_all[1] + " keypoints" : "");
    } else if (res.kind === "classification") {
      canvasOut.width = canvasIn.width; canvasOut.height = canvasIn.height;
      const ctx = canvasOut.getContext("2d");
      ctx.drawImage(canvasIn, 0, 0);
      const top = res.top.length;
      ctx.fillStyle = "rgba(15, 23, 42, 0.72)";
      ctx.fillRect(0, canvasOut.height - (top * 22 + 12), canvasOut.width, top * 22 + 12);
      ctx.font = "12px Inter, system-ui, sans-serif";
      res.top.forEach(function (t, row) {
        const y = canvasOut.height - (top - row) * 22 - 4;
        ctx.fillStyle = "rgba(37, 99, 235, 0.85)";
        ctx.fillRect(8, y - 12, (canvasOut.width - 16) * t.prob, 14);
        ctx.fillStyle = "#fff";
        ctx.fillText(t.label + "  " + (100 * t.prob).toFixed(1) + "%", 12, y);
      });
      summary = res.top[0].label + " " + (100 * res.top[0].prob).toFixed(1) + "%";
    }
    res.summary = summary;
    return summary;
  }

  // ------------------------------------------------------------------ shared: "sign in to …" prompt, and the upload button

  let signInDialog = null;
  function askSignIn(what, onSigned) {
    const A = window.KorniaAuth;
    if (!signInDialog) {
      signInDialog = el("dialog", "pg-dialog pg-dialog-signin");
      signInDialog.innerHTML = '<button type="button" class="pg-dialog-close" aria-label="Close"><i class="fas fa-xmark" aria-hidden="true"></i></button><h2></h2><p class="pg-dialog-text"></p>';
      signInDialog.querySelector(".pg-dialog-close").addEventListener("click", function () { signInDialog.close(); });
      signInDialog.addEventListener("click", function (e) { if (e.target === signInDialog) signInDialog.close(); });   // the backdrop
      const row = el("div", "pg-dialog-actions");
      const cancel = el("button", "pg-btn pg-btn-ghost", { type: "button" }); cancel.textContent = "Not now";
      const gh = el("button", "pg-btn pg-btn-ghost", { type: "button" }); gh.innerHTML = '<i class="fab fa-github" aria-hidden="true"></i> GitHub';
      const gg = el("button", "pg-btn pg-btn-ghost", { type: "button" }); gg.innerHTML = '<i class="fab fa-google" aria-hidden="true"></i> Google';
      const mail = el("a", "pg-btn", { href: "#" }); mail.innerHTML = '<i class="fas fa-envelope" aria-hidden="true"></i> Email or new account';
      row.appendChild(cancel); row.appendChild(gh); row.appendChild(gg); row.appendChild(mail);
      signInDialog.appendChild(row);
      document.body.appendChild(signInDialog);
      cancel.addEventListener("click", function () { signInDialog.close(); });
      function via(p) {
        signInDialog.close();
        const cb = signInDialog._onSigned;
        if (!window.KorniaAuth || !window.KorniaAuth.configured) return;
        window.KorniaAuth.signIn(p).then(function () { if (cb) cb(); }).catch(function () {});
      }
      gh.addEventListener("click", function () { via("github"); });
      gg.addEventListener("click", function () { via("google"); });
      mail.addEventListener("click", function (e) { e.preventDefault(); location.href = (window.KorniaAuth ? window.KorniaAuth.hubUrl : "../../dashboard/") + "?next=" + encodeURIComponent(location.pathname + location.search); });
    }
    signInDialog.querySelector("h2").textContent = "Sign in to " + what;
    signInDialog.querySelector(".pg-dialog-text").textContent = A && A.configured ? "Accounts are free; a daily quota of runs comes with one." : "Accounts are not configured on this copy of the site.";
    signInDialog._onSigned = onSigned || null;
    signInDialog.showModal();
  }

  // ---- targeted ONNX export: a modal that asks the server for a graph built the way the visitor wants.
  // The server sends the operator's real kornia signature; every parameter can take any legal value (a Python
  // literal), baked into the graph. "Keep the page's sliders as graph inputs" is the other mode: the page's own
  // parameter set, with live values left as inputs.
  let exportDlg = null;
  const signatureCache = {};
  // the report after an export: what was built, its contract, and how it was checked; the file can be saved again
  function exportReport(host, rep, blob, extra) {
    host.innerHTML = "";
    const card = el("div", "pg-export-report");
    const head = el("div", "pg-export-report-head");
    const title = el("strong"); title.textContent = rep.file;
    const again = el("button", "pg-btn pg-btn-small", { type: "button" });
    again.innerHTML = '<i class="fas fa-file-arrow-down" aria-hidden="true"></i> Save again';
    again.addEventListener("click", function () { const url = URL.createObjectURL(blob); const a = el("a", "", { href: url, download: rep.file }); document.body.appendChild(a); a.click(); a.remove(); setTimeout(function () { URL.revokeObjectURL(url); }, 4000); });
    head.appendChild(title); head.appendChild(again); card.appendChild(head);
    const size = rep.bytes >= 1048576 ? (rep.bytes / 1048576).toFixed(1) + " MB" : (rep.bytes / 1024).toFixed(0) + " KB";
    const shapeText = function (sh) { return sh.map(function (d) { return typeof d === "number" ? d : d; }).join("×"); };
    const rows = [
      ["Graph", size + " · opset " + rep.opset + (rep.dynamic ? " · dynamic batch, height and width" : " · fixed shape") + (rep.ops ? " · " + rep.ops.length + " operator types" : "")],
      ["Inputs", rep.inputs.map(function (i) { return i[0] + " " + shapeText(i[1]); }).join(", ") + (extra && extra.inputNote ? " · " + extra.inputNote : "")],
      ["Outputs", rep.outputs ? rep.outputs.map(function (o) { return o[0] + " " + shapeText(o[1]); }).join(", ") : "out " + shapeText(rep.output_shape)],
    ];
    if (rep.params && Object.keys(rep.params).length) rows.push(["Baked in", Object.keys(rep.params).map(function (k) { return k + "=" + JSON.stringify(rep.params[k]); }).join(", ")]);
    if (rep.note) rows.push(["Contains", rep.note]);
    rows.push(["Checked", rep.max_error != null ? "matches PyTorch to " + rep.max_error.toExponential(1) + (extra && extra.relative ? " (relative)" : "") + " on a random input" : "random augmentation: outputs differ by design, shapes verified"]);
    rows.push(["Built with", "kornia " + rep.kornia + " · torch " + rep.torch + " · " + (rep.ms >= 1000 ? (rep.ms / 1000).toFixed(1) + " s" : rep.ms + " ms") + " of server time"]);
    const table = el("table", "pg-export-report-table");
    rows.forEach(function (r) { const tr = el("tr"); const th = el("th"); th.textContent = r[0]; const td = el("td"); td.textContent = r[1]; tr.appendChild(th); tr.appendChild(td); table.appendChild(tr); });
    card.appendChild(table);
    const run = el("pre", "pg-export-report-code");
    const inName = rep.inputs[0][0], inShape = rep.inputs[0][1];
    run.textContent = 'import onnxruntime as ort, numpy as np\nsess = ort.InferenceSession("' + rep.file + '")\nx = np.random.rand(' + inShape.map(function (d) { return typeof d === "number" ? d : (d === "B" || d === "N" ? 1 : 256); }).join(", ") + ').astype(np.float32)\nout = sess.run(None, {"' + inName + '": x})';
    card.appendChild(run);
    const foot = el("p", "hub-item-meta");
    foot.textContent = "Runs with onnxruntime, onnxruntime-web, TensorRT and OpenVINO through their ONNX importers, and the ort crate in Rust. The download is in your browser's download list; nothing is kept on the server.";
    card.appendChild(foot);
    host.appendChild(card);
  }
  function exportDialog(op, registry, params, ROOT) {
    const volume = op.input_kind === "volume";
    const A = window.KorniaAuth;
    if (!A || !A.user) { askSignIn("export " + op.name + " on kornia's server", function () { exportDialog(op, registry, params, ROOT); }); return; }
    if (!A.apiBase) { alert("the server side is not deployed yet"); return; }
    const live = op.params.filter(function (p) { return p.kind === "live"; });
    const opsets = volume ? [20] : [13, 17, 18, 20];
    const defOpset = op.opset || registry.onnx_opset || 18;
    if (exportDlg) exportDlg.remove();
    exportDlg = el("dialog", "pg-dialog pg-dialog-signin pg-export");
    exportDlg.innerHTML = '<button type="button" class="pg-dialog-close" aria-label="Close"><i class="fas fa-xmark" aria-hidden="true"></i></button>'
      + "<h2>Export " + op.name + " to ONNX</h2>"
      + '<p class="pg-dialog-text">Built on kornia\'s server from <code>' + op.id + '</code> with the values below, dry-run, then checked against PyTorch in onnxruntime before you get the file. Server time counts toward your day.</p>'
      + '<div class="pg-export-grid">'
      + '<label><span>Opset</span><select name="opset">' + opsets.map(function (o) { return '<option value="' + o + '"' + (o === defOpset ? " selected" : "") + ">" + o + (o === 13 ? " · widest runtime support" : o === 20 ? " · newest" : "") + "</option>"; }).join("") + "</select></label>"
      + (volume ? '<label><span>Input</span><span class="pg-export-fixed">1×1×' + op.volume.size + "×" + op.volume.size + "×" + op.volume.size + " (the graph size)</span></label>"
        : '<label><span>Input shape</span><span class="pg-export-shape"><select name="shape"><option value="fixed">fixed</option><option value="dynamic">dynamic (any batch, H, W)</option></select><input name="batch" type="number" min="1" max="16" value="1" aria-label="batch"> × 3 × <input name="height" type="number" min="16" max="2048" value="' + (registry.size || 256) + '" aria-label="height"> × <input name="width" type="number" min="16" max="2048" value="' + (registry.size || 256) + '" aria-label="width"></span></label>')
      + '<label><span>Parameters</span><span class="pg-export-choice"><label><input type="radio" name="mode" value="full" checked> every parameter of the call, any value, baked in</label>' + (live.length ? '<label><input type="radio" name="mode" value="live"> the page\'s sliders (' + live.map(function (p) { return p.name; }).join(", ") + ') stay graph inputs</label>' : "") + "</span></label>"
      + '<div class="pg-export-sig" data-sig><p class="hub-item-meta">loading the signature…</p></div>'
      + '<label><span>Precision</span><span class="pg-export-choice"><label><input type="radio" name="fp" value="fp32" checked> float32</label><label><input type="radio" name="fp" value="fp16"> float16 weights, float32 in and out</label></span></label>'
      + "</div>"
      + '<p class="pg-status pg-export-status"></p>'
      + '<div class="pg-dialog-actions"><button type="button" class="pg-btn pg-btn-ghost" data-act="cancel">Not now</button><button type="button" class="pg-btn" data-act="export"><i class="fas fa-cloud-arrow-down" aria-hidden="true"></i> Export on server</button></div>';
    document.body.appendChild(exportDlg);
    const status = exportDlg.querySelector(".pg-export-status");
    const sigBox = exportDlg.querySelector("[data-sig]");
    const close = function () { exportDlg.close(); };
    exportDlg.querySelector(".pg-dialog-close").addEventListener("click", close);
    exportDlg.querySelector('[data-act="cancel"]').addEventListener("click", close);
    exportDlg.addEventListener("click", function (e) { if (e.target === exportDlg) close(); });
    const shapeSel = exportDlg.querySelector('select[name="shape"]');
    if (shapeSel) shapeSel.addEventListener("change", function () { exportDlg.querySelectorAll('input[name="batch"], input[name="height"], input[name="width"]').forEach(function (i) { i.disabled = shapeSel.value === "dynamic"; }); });

    // the page's current values, as the literals the call takes (the page's tuple-taking parameters show as pairs)
    function pageLiteral(name) {
      const p = op.params.find(function (q) { return q.name === name; });
      if (!p) return null;
      const v = params[name];
      if (p.literals && p.literals[v]) return p.literals[v];
      const s = p.type === "str" ? String(v) : String(Number(v));
      const paired = /^(kernel_size|sigma|sigma_space|size|translate)$/.test(name) && /\(\{[^}]+\}, \{[^}]+\}\)/.test(op.snippet || "");
      return paired ? "(" + s + ", " + s + ")" : (p.type === "str" ? s : s);
    }
    function renderSignature(sig) {
      sigBox.innerHTML = "";
      if (!sig.params.length) { sigBox.innerHTML = '<p class="hub-item-meta">This operator takes no parameters besides the image.</p>'; return; }
      const table = el("table", "pg-export-table");
      table.innerHTML = "<thead><tr><th>parameter</th><th>type</th><th>value</th></tr></thead>";
      const body = el("tbody");
      sig.params.forEach(function (p) {
        const tr = el("tr");
        const fromPage = pageLiteral(p.name);
        const value = fromPage != null ? fromPage : (p.default != null ? p.default : "");
        tr.innerHTML = "<td><code>" + p.name + (p.required ? " *" : "") + "</code></td><td><span class=\"pg-export-type\">" + (p.type || "") + "</span></td><td></td>";
        const input = el("input", "pg-export-value", { type: "text", name: "sig:" + p.name, value: value, placeholder: p.required ? "required" : "default " + p.default, spellcheck: "false" });
        if (fromPage != null) input.title = "from the page";
        tr.lastElementChild.appendChild(input);
        body.appendChild(tr);
      });
      table.appendChild(body);
      sigBox.appendChild(table);
      const hint = el("p", "hub-item-meta");
      hint.textContent = "Python literals: 5, 2.5, (3, 3), True, 'reflect', None. Leave a field at its default to keep it. * required." + (sig.kind === "class" ? " These are the constructor's arguments; the module is applied to the image." : "");
      sigBox.appendChild(hint);
    }
    (signatureCache[op.slug] || (signatureCache[op.slug] = fetch(A.apiBase + "/v1/signature/" + op.slug).then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })))
      .then(renderSignature).catch(function (e) { sigBox.innerHTML = '<p class="pg-status pg-error">could not load the signature: ' + (e.message || e) + "</p>"; });
    const modeInputs = exportDlg.querySelectorAll('input[name="mode"]');
    modeInputs.forEach(function (r) { r.addEventListener("change", function () { sigBox.hidden = exportDlg.querySelector('input[name="mode"]:checked').value !== "full"; }); });

    function request() {
      const f = function (n) { return exportDlg.querySelector('[name="' + n + '"]'); };
      const mode = exportDlg.querySelector('input[name="mode"]:checked').value;
      const body = { opset: Number(f("opset").value), fp16: exportDlg.querySelector('input[name="fp"]:checked').value === "fp16", mode: mode };
      if (!volume) {
        body.dynamic = shapeSel && shapeSel.value === "dynamic";
        body.batch = Number(f("batch").value); body.height = Number(f("height").value); body.width = Number(f("width").value);
      }
      if (mode === "full") {
        body.params = {};
        exportDlg.querySelectorAll('input[name^="sig:"]').forEach(function (i) { const v = i.value.trim(); if (v !== "" && v !== i.placeholder.replace(/^default /, "")) body.params[i.name.slice(4)] = v; else if (i.placeholder === "required" && v === "") body.params[i.name.slice(4)] = ""; });
      } else {
        body.bake = false;
        body.params = {};
        op.params.forEach(function (p) { body.params[p.name] = params[p.name]; });
      }
      return body;
    }
    async function run() {
      const btn = exportDlg.querySelector('[data-act="export"]');
      btn.disabled = true; status.classList.remove("pg-error"); status.innerHTML = ""; status.textContent = "exporting and checking against PyTorch…";
      try {
        const r = await fetch(A.apiBase + "/v1/export/" + op.slug, { method: "POST", headers: { Authorization: "Bearer " + await A.token(), "Content-Type": "application/json" }, body: JSON.stringify(request()) });
        if (!r.ok) { const j = await r.json().catch(function () { return {}; }); throw new Error(j.detail || ("HTTP " + r.status)); }
        const rep = JSON.parse(r.headers.get("X-Export-Report") || "{}");
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        const a = el("a", "", { href: url, download: rep.file || (op.slug + ".onnx") });
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
        setQuota({ used: rep.quota.used, limit: rep.quota.limit, remaining: Math.max(0, rep.quota.limit - rep.quota.used) }); window.dispatchEvent(new CustomEvent("kornia-quota", { detail: quota }));
        exportReport(status, rep, blob, { inputNote: volume ? "float32 volume in [0, 1]" : "float32 RGB in [0, 1], NCHW" });
      } catch (e) { status.classList.add("pg-error"); status.textContent = e.message || String(e); }
      btn.disabled = false;
    }
    exportDlg.querySelector('[data-act="export"]').addEventListener("click", run);
    exportDlg.showModal();
  }

  // the same, for a pretrained model: opset, one of the sizes the server offers, batch, dynamic, fp16
  let exportInfo = null;
  function modelExportDialog(model, ROOT) {
    const A = window.KorniaAuth;
    if (!A || !A.user) { askSignIn("export " + model.name + " on kornia's server", function () { modelExportDialog(model, ROOT); }); return; }
    if (!A.apiBase) { alert("the server side is not deployed yet"); return; }
    const names = model.exports || [];
    (exportInfo || (exportInfo = fetch(A.apiBase + "/v1/models").then(function (r) { return r.json(); }).then(function (j) { return j.exports || {}; }))).then(function (info) {
      if (exportDlg) exportDlg.remove();
      exportDlg = el("dialog", "pg-dialog pg-dialog-signin pg-export");
      const first = info[names[0]] || { sizes: [model.input.size], dynamic: false, batch: false, note: "" };
      exportDlg.innerHTML = '<button type="button" class="pg-dialog-close" aria-label="Close"><i class="fas fa-xmark" aria-hidden="true"></i></button>'
        + "<h2>Export " + model.name + " to ONNX</h2>"
        + '<p class="pg-dialog-text">Built on kornia\'s server from the pretrained weights, then checked against PyTorch in onnxruntime before you get the file. One export counts as one run.</p>'
        + '<div class="pg-export-grid">'
        + (names.length > 1 ? '<label><span>Part</span><select name="which">' + names.map(function (n) { return '<option value="' + n + '">' + n + (info[n] ? " · " + info[n].note : "") + "</option>"; }).join("") + "</select></label>" : "")
        + '<label><span>Opset</span><select name="opset"><option value="13">13 · widest runtime support</option><option value="17" selected>17</option><option value="18">18</option><option value="20">20 · newest</option></select></label>'
        + '<label><span>Input size</span><span class="pg-export-shape"><select name="size">' + first.sizes.map(function (z) { return '<option value="' + z + '">' + z + "×" + z + "</option>"; }).join("") + "</select>"
        + (first.batch || first.dynamic ? ' batch <input name="batch" type="number" min="1" max="16" value="1" aria-label="batch">' : "")
        + (first.dynamic ? ' <label><input type="checkbox" name="dynamic"> dynamic batch, height and width</label>' : "") + "</span></label>"
        + '<label><span>Precision</span><span class="pg-export-choice"><label><input type="radio" name="fp" value="fp32" checked> float32</label><label data-fp16' + (first.fp16 === false ? " hidden" : "") + '><input type="radio" name="fp" value="fp16"> float16 weights, float32 in and out</label></span></label>'
        + '<label><span>Contains</span><span class="pg-export-fixed" data-note>' + (first.note || "") + "</span></label>"
        + "</div>"
        + '<p class="pg-status pg-export-status"></p>'
        + '<div class="pg-dialog-actions"><button type="button" class="pg-btn pg-btn-ghost" data-act="cancel">Not now</button><button type="button" class="pg-btn" data-act="export"><i class="fas fa-cloud-arrow-down" aria-hidden="true"></i> Export on server</button></div>';
      document.body.appendChild(exportDlg);
      const status = exportDlg.querySelector(".pg-export-status");
      const close = function () { exportDlg.close(); };
      exportDlg.querySelector(".pg-dialog-close").addEventListener("click", close);
      exportDlg.querySelector('[data-act="cancel"]').addEventListener("click", close);
      exportDlg.addEventListener("click", function (e) { if (e.target === exportDlg) close(); });
      const which = exportDlg.querySelector('select[name="which"]');
      if (which) which.addEventListener("change", function () {
        const inf = info[which.value] || first;
        const sz = exportDlg.querySelector('select[name="size"]'); sz.innerHTML = inf.sizes.map(function (z) { return '<option value="' + z + '">' + z + "×" + z + "</option>"; }).join("");
        exportDlg.querySelector("[data-note]").textContent = inf.note || "";
        const dyn = exportDlg.querySelector('input[name="dynamic"]'); if (dyn) dyn.parentElement.hidden = !inf.dynamic;
        const fp = exportDlg.querySelector("[data-fp16]"); fp.hidden = inf.fp16 === false; if (fp.hidden) exportDlg.querySelector('input[name="fp"][value="fp32"]').checked = true;
      });
      exportDlg.querySelector('[data-act="export"]').addEventListener("click", async function () {
        const btn = this; btn.disabled = true; status.classList.remove("pg-error"); status.innerHTML = ""; status.textContent = "exporting and checking against PyTorch… (large models take a while)";
        const f = function (n) { return exportDlg.querySelector('[name="' + n + '"]'); };
        const body = { opset: Number(f("opset").value), size: Number(f("size").value), batch: f("batch") ? Number(f("batch").value) : 1, dynamic: !!(f("dynamic") && f("dynamic").checked), fp16: exportDlg.querySelector('input[name="fp"]:checked').value === "fp16" };
        const name = which ? which.value : names[0];
        try {
          const r = await fetch(A.apiBase + "/v1/export_model/" + name, { method: "POST", headers: { Authorization: "Bearer " + await A.token(), "Content-Type": "application/json" }, body: JSON.stringify(body) });
          if (!r.ok) { const j = await r.json().catch(function () { return {}; }); throw new Error(j.detail || ("HTTP " + r.status)); }
          const rep = JSON.parse(r.headers.get("X-Export-Report") || "{}");
          const blob = await r.blob(); const url = URL.createObjectURL(blob);
          const a = el("a", "", { href: url, download: rep.file || (name + ".onnx") }); document.body.appendChild(a); a.click(); a.remove();
          setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
          setQuota({ used: rep.quota.used, limit: rep.quota.limit, remaining: Math.max(0, rep.quota.limit - rep.quota.used) }); window.dispatchEvent(new CustomEvent("kornia-quota", { detail: quota }));
          const inf = info[name] || first;
          exportReport(status, rep, blob, { relative: true, inputNote: (model.input.gray ? "float32 grayscale" : "float32 RGB") + (model.input.scale === 255 ? " in [0, 255]" : model.input.mean ? ", ImageNet-normalised" : " in [0, 1]") + ", NCHW" });
        } catch (e) { status.classList.add("pg-error"); status.textContent = e.message || String(e); }
        btn.disabled = false;
      });
      exportDlg.showModal();
    });
  }

  // where a run happens: a small two-way switch that sits above the Run button
  function targetSwitch(opts) {
    const box = el("div", "pg-target", { role: "group", "aria-label": "Where to run" });
    let value = opts.value || (opts.browser ? "browser" : "server");
    const buttons = {};
    [["browser", "in browser"], ["server", "on server"]].forEach(function (pair) {
      const b = el("button", "", { type: "button", "aria-pressed": value === pair[0] ? "true" : "false" });
      b.textContent = pair[1];
      b.disabled = !opts[pair[0]];
      b.addEventListener("click", function () { if (b.disabled) return; value = pair[0]; paint(); if (opts.onChange) opts.onChange(value); });
      buttons[pair[0]] = b; box.appendChild(b);
    });
    function paint() { Object.keys(buttons).forEach(function (k) { buttons[k].setAttribute("aria-pressed", k === value ? "true" : "false"); }); }
    return { el: box, get: function () { return value; }, set: function (v) { value = v; paint(); }, allow: function (browser, server) { buttons.browser.disabled = !browser; buttons.server.disabled = !server; if (!opts[value] && false) {} if ((value === "browser" && !browser) || (value === "server" && !server)) { value = browser ? "browser" : "server"; paint(); } } };
  }

  // "upload your own image" thumbnail: greyed until signed in (a click then asks to sign in); signed in it opens
  // the file picker and calls onImage(img, name) with a loaded HTMLImageElement
  function uploadButton(onImage) {
    const btn = el("button", "pg-thumb-upload", { type: "button", title: "Upload your own image", "aria-label": "Upload your own image" });
    btn.innerHTML = '<i class="fas fa-upload" aria-hidden="true"></i>';
    const file = el("input", "", { type: "file", accept: "image/*" });
    file.hidden = true;
    btn.appendChild(file);
    function paint() { btn.classList.toggle("is-locked", !(window.KorniaAuth && window.KorniaAuth.user)); }
    paint();
    if (window.KorniaAuth) window.KorniaAuth.onChange(paint); else document.addEventListener("DOMContentLoaded", function () { if (window.KorniaAuth) window.KorniaAuth.onChange(paint); });
    btn.addEventListener("click", function (e) {
      if (e.target === file) return;
      if (!(window.KorniaAuth && window.KorniaAuth.user)) { askSignIn("upload your own image", function () { file.click(); }); return; }
      file.click();
    });
    file.addEventListener("change", function () {
      const f = file.files && file.files[0];
      if (!f) return;
      const url = URL.createObjectURL(f);
      const img = new Image();
      img.onload = function () { URL.revokeObjectURL(url); onImage(img, f.name); };
      img.onerror = function () { URL.revokeObjectURL(url); };
      img.src = url;
      file.value = "";
    });
    return btn;
  }

  // ------------------------------------------------------------------ the widget

  function render(model, registry, index, container, ROOT) {
    const root = el("article", "pg-widget pg-model");
    const head = el("div", "pg-model-head pg-head");
    const h1 = el("h1");
    h1.textContent = model.name;
    const task = el("span", "pg-tag pg-tag-task");
    task.textContent = model.task;
    head.appendChild(h1);
    head.appendChild(task);
    root.appendChild(head);
    const summary = el("p", "pg-summary");
    summary.textContent = model.summary;
    root.appendChild(summary);

    // stage
    const isFeatures = model.output.kind.indexOf("features_") === 0;
    const hasBrowser = !!model.url;                   // a graph the browser can download and run
    const canServer = !!model.server;                 // an endpoint on kornia's server
    let onServer = !hasBrowser;   // where the last result came from, for the threshold slider
    const stage = el("div", "pg-model-stage" + (isFeatures ? " pg-model-stage-wide" : ""));
    const figIn = el("figure");
    const canvasIn = el("canvas", "", { width: model.input.size, height: model.input.size });
    const canvasB = document.createElement("canvas");   // the warped copy, for feature models
    canvasB.width = model.input.size; canvasB.height = model.input.size;
    let affine = randomAffine(model.input.size);
    const capIn = el("figcaption");
    capIn.textContent = "input · " + model.input.size + "×" + model.input.size;
    figIn.appendChild(canvasIn);
    figIn.appendChild(capIn);
    const figOut = el("figure");
    // feature models draw the frame and its warped copy side by side, so their output box is 2:1 from the start
    const canvasOut = el("canvas", "", { width: isFeatures ? 2 * model.input.size : model.input.size, height: model.input.size });
    const capOut = el("figcaption");
    capOut.textContent = isFeatures ? "frame | warped copy, with matches" : "output";
    figOut.appendChild(canvasOut);
    figOut.appendChild(capOut);
    stage.appendChild(figIn);
    stage.appendChild(figOut);
    root.appendChild(stage);

    // samples: images + clips
    const thumbs = el("div", "pg-thumbs-pick pg-model-thumbs", { role: "group", "aria-label": "Sample inputs" });
    const samples = (model.sample ? [model.sample] : []).concat(registry.images.filter(function (im) { return im.selectable !== false; }));
    let currentImage = samples[0];
    function placeholder(text) {
      const ctx = canvasOut.getContext("2d");
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(0, 0, canvasOut.width, canvasOut.height);
      ctx.fillStyle = "#94a3b8";
      ctx.font = "14px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(text, canvasOut.width / 2, canvasOut.height / 2);
    }
    // nothing runs by itself: a sample change only redraws the input and clears the output
    function drawStill(im) {
      if (!im || !im.file) return;
      const img = new Image();
      img.onload = function () {
        canvasIn.getContext("2d").drawImage(img, 0, 0, model.input.size, model.input.size);
        lastResults = null;
        placeholder("press Run on this frame");
      };
      img.src = ROOT + im.file;
    }
    samples.forEach(function (im, i) {
      const b = el("button", i === 0 ? "pg-selected" : "", { type: "button", title: im.label });
      const img = el("img", "", { src: ROOT + im.thumb, alt: im.label, width: 56, height: 56 });
      b.appendChild(img);
      b.addEventListener("click", function () {
        thumbs.querySelectorAll("button").forEach(function (t) { t.classList.remove("pg-selected"); });
        b.classList.add("pg-selected");
        currentImage = im;
        drawStill(im);
      });
      thumbs.appendChild(b);
    });
    const upBtn = uploadButton(function (img, name) {
      const N = model.input.size;   // the model's fixed input: cover-fit the photo into that square
      thumbs.querySelectorAll("button").forEach(function (t) { t.classList.remove("pg-selected"); });
      upBtn.classList.add("pg-selected");
      const ctx = canvasIn.getContext("2d");
      const s = Math.max(N / img.width, N / img.height);
      const w = img.width * s, h = img.height * s;
      ctx.fillStyle = "#000"; ctx.fillRect(0, 0, N, N);
      ctx.drawImage(img, (N - w) / 2, (N - h) / 2, w, h);
      currentImage = { id: "upload", label: name, file: null };
      lastResults = null; lastServer = null;
      placeholder("press Run on this frame");
      capIn.textContent = "input · " + N + "×" + N + " · " + name;
    });
    thumbs.appendChild(upBtn);
    const barRow = el("div", "pg-bar");   // one row under the stage: samples left, actions right
    barRow.appendChild(thumbs);
    root.appendChild(barRow);
    // depth becomes a point cloud you can orbit: a panel under the stage, built on the first result
    let cloud = null, cloudPanel = null;
    function showCloud(rel, w, h) {
      if (model.output.kind !== "depth") return;
      if (!cloudPanel) {
        cloudPanel = el("figure", "pg-3d");
        const c = el("canvas", "pg-3d-canvas");
        const cap = el("figcaption");
        cap.textContent = "the depth as a point cloud, coloured by the frame · drag to orbit, wheel to zoom · relative depth, so the scale is indicative";
        cloudPanel.appendChild(c); cloudPanel.appendChild(cap);
        stage.insertAdjacentElement("afterend", cloudPanel);
        cloud = import(ROOT + "../viewer3d.js").then(function (mod) { return mod.createViewer(c).then(function (v) { return { mod: mod, viewer: v }; }); });
      }
      cloud.then(function (o) {
        const depth = o.mod.inverseToDepth(rel, 0.5, 6);
        const rgba = canvasIn.getContext("2d").getImageData(0, 0, canvasIn.width, canvasIn.height).data;
        const pc = o.mod.cloudFromDepth(depth, w, h, rgba, canvasIn.width, canvasIn.height, 60, w > 300 ? 2 : 1);
        o.viewer.setCloud(pc.positions, pc.colors, { view: { yaw: 0.4, pitch: 0.3 } });
      }).catch(function (e) { console.warn("point cloud unavailable", e); });
    }
    const panel = el("div", "pg-cell-controls pg-controls-cell pg-model-panel");   // thresholds, quota, progress

    // actions: Run loads the model on demand (after asking), Download mirrors the operator pages
    const sizeText = model.size_mb >= 1 ? Math.round(model.size_mb) + " MB" : (model.size_mb * 1000).toFixed(0) + " kB";
    const modelUrl = hasBrowser ? (/^https?:/.test(model.url) ? model.url : ROOT + model.url) : "";
    // one Run button; the small switch above it picks this browser (downloads the graph, no account) or kornia's server
    const actions = el("div", "pg-actions pg-model-actions");
    const target = targetSwitch({ browser: hasBrowser, server: canServer, value: hasBrowser ? "browser" : "server" });
    const runGroup = el("div", "pg-run-group");
    const runBtn = el("button", "pg-btn", { type: "button" });
    runBtn.innerHTML = '<i class="fas fa-play" aria-hidden="true"></i> Run';
    runGroup.appendChild(target.el); runGroup.appendChild(runBtn);
    actions.appendChild(runGroup);
    let threshold = isFeatures ? model.output.min_cossim : (model.output.threshold || 0.5);
    if (isFeatures) {
      const rew = el("button", "pg-btn pg-btn-ghost", { type: "button" });
      rew.innerHTML = '<i class="fas fa-dice" aria-hidden="true"></i> Re-warp';
      rew.addEventListener("click", function () { affine = randomAffine(model.input.size); if (ready) run(); });
      actions.appendChild(rew);
    }
    if (model.output.kind === "detections" || model.output.kind === "faces_yunet" || isFeatures) {
      const param = el("div", "pg-param");
      const name = el("label");
      name.textContent = isFeatures ? "cosine ≥" : "score ≥";
      const range = el("input", "", { type: "range", min: isFeatures ? "0.3" : "0.1", max: isFeatures ? "0.98" : "0.9", step: isFeatures ? "0.02" : "0.05", value: String(threshold) });
      const val = el("output");
      val.textContent = threshold.toFixed(2);
      range.addEventListener("input", function () { threshold = Number(range.value); val.textContent = threshold.toFixed(2); if (onServer && lastServer) { drawServerResult(model, lastServer, canvasIn, canvasB, canvasOut, { threshold: threshold }); status.textContent = lastServer.summary; } else if (lastResults) draw(lastResults); });
      param.appendChild(name); param.appendChild(range); param.appendChild(val);
      panel.appendChild(param);
    }
    const status = el("span", "pg-status");
    function idleStatus() { return ""; }
    status.textContent = idleStatus();
    actions.appendChild(status);
    barRow.appendChild(actions);
    root.appendChild(panel);
    const progress = el("div", "pg-progress");
    const bar = el("div");
    progress.appendChild(bar);
    progress.hidden = true;
    panel.appendChild(progress);

    // the question asked before the first download
    const dialog = el("dialog", "pg-dialog");
    const dh = el("h2");
    dh.textContent = "Load " + model.name + "?";
    const dp1 = el("p");
    dp1.textContent = "This downloads " + sizeText + " from " + model.hosted + " into your browser tab. The image you run it on never leaves your browser.";
    const dp2 = el("p", "pg-dialog-warn");
    dp2.innerHTML = '<i class="fas fa-triangle-exclamation" aria-hidden="true"></i> Models run on the CPU with WebAssembly here, so one frame can take a few seconds. For real-time use, download the ONNX and run it with onnxruntime.';
    const drow = el("div", "pg-dialog-actions");
    const dCancel = el("button", "pg-btn pg-btn-ghost", { type: "button", value: "cancel" });
    dCancel.textContent = "Not now";
    const dOk = el("button", "pg-btn", { type: "button", value: "ok" });
    dOk.innerHTML = '<i class="fas fa-download" aria-hidden="true"></i> Load ' + sizeText + " and run";
    drow.appendChild(dCancel); drow.appendChild(dOk);
    dialog.appendChild(dh); dialog.appendChild(dp1); dialog.appendChild(dp2); dialog.appendChild(drow);
    root.appendChild(dialog);
    // ---- the server path
    let serverBusy = false, lastServer = null;
    async function runServer() {
      const A = window.KorniaAuth;
      if (!A || !A.user) { askSignIn("run " + model.name + " on kornia's server", runServer); return; }   // dismissable
      if (serverBusy) return;
      serverBusy = true;
      onServer = true;
      runBtn.disabled = true;
      status.classList.remove("pg-error");
      const pair = model.server.inputs === 2;
      status.textContent = pair ? "warping the copy, sending both frames…" : "sending the frame to the server…";
      const t0 = performance.now();
      try {
        if (pair) {
          const sw = await getSessionByUrl(absUrl(model.graphs.warp, ROOT));
          await warpCanvas(sw, canvasIn, canvasB, model.input.size, affine);
        }
        const res = await runOnServer(model, canvasIn, pair ? canvasB : null);
        lastServer = res;
        setQuota({ used: res.quota.used, limit: res.quota.limit, remaining: Math.max(0, res.quota.limit - res.quota.used) }); window.dispatchEvent(new CustomEvent("kornia-quota", { detail: quota }));
        drawServerResult(model, res, canvasIn, canvasB, canvasOut, { threshold: threshold, onDepth: showCloud });
        status.textContent = res.summary + " · " + res.ms + " ms on the server";
      } catch (e) {
        status.classList.add("pg-error");
        status.textContent = e.message || String(e);
      }
      serverBusy = false;
      runBtn.disabled = false;
    }

    function askToLoad() {
      return new Promise(function (resolve) {
        if (typeof dialog.showModal !== "function") { resolve(window.confirm(dh.textContent + " " + dp1.textContent + " It may take a few seconds per frame.")); return; }
        function done(ok) { dialog.close(); dCancel.removeEventListener("click", no); dOk.removeEventListener("click", yes); resolve(ok); }
        function no() { done(false); }
        function yes() { done(true); }
        dCancel.addEventListener("click", no);
        dOk.addEventListener("click", yes);
        dialog.addEventListener("cancel", function onCancel(e) { e.preventDefault(); dialog.removeEventListener("cancel", onCancel); done(false); });
        dialog.showModal();
      });
    }

    const code = el("section", "pg-code", { id: "code" });
    const tabs = el("div", "pg-tabs", { role: "tablist" });
    const panels = [];
    // the snippets from the build, plus an ONNX tab: the file to download and the few lines that run it
    const entries = model.snippets.map(function (sn) { return { label: /onnx/i.test(sn.label) ? "ONNX" : sn.label, code: sn.code, onnx: /onnx/i.test(sn.label) }; });
    if ((hasBrowser || (model.exports && model.exports.length)) && !entries.some(function (e) { return e.onnx; })) entries.push({ label: "ONNX", code: hasBrowser ? onnxRunnerSnippet() : serverExportSnippet(), onnx: true });
    entries.forEach(function (sn, i) {
      const t = el("button", "pg-tab" + (i === 0 ? " pg-tab-active" : ""), { type: "button", role: "tab", "aria-selected": i === 0 ? "true" : "false" });
      t.textContent = sn.label;
      const pane = el("div", "pg-pane", { role: "tabpanel" });
      if (sn.onnx && (hasBrowser || (model.exports && model.exports.length))) {
        const head = el("div", "pg-onnx-head");
        if (hasBrowser) {
          const dl = el("a", "pg-btn pg-btn-small", { href: modelUrl, target: "_blank", rel: "noopener", download: "" });
          dl.innerHTML = '<i class="fas fa-file-arrow-down" aria-hidden="true"></i> Download ONNX (' + sizeText + ")";
          head.appendChild(dl);
        }
        if (model.exports && model.exports.length) {
          const cu = el("button", "pg-btn pg-btn-ghost pg-btn-small", { type: "button", title: "Export this model with your own opset, input size and precision, on kornia's server" });
          cu.innerHTML = '<i class="fas fa-sliders" aria-hidden="true"></i> Customise…';
          cu.addEventListener("click", function () { modelExportDialog(model, ROOT); });
          head.appendChild(cu);
        }
        const note = el("span", "pg-onnx-note");
        note.textContent = (hasBrowser ? model.input.size + "×" + model.input.size + " input · from " + model.hosted + " · " : "") + "runs with onnxruntime, onnxruntime-web and the ort crate";
        head.appendChild(note); pane.appendChild(head);
      }
      const pre = el("pre", "pg-snippet");
      const c = el("code", "language-python");
      c.textContent = sn.code;
      pre.appendChild(c);
      pane.appendChild(pre);
      pane.hidden = i !== 0;
      if (window.hljs) window.hljs.highlightElement(c);
      t.addEventListener("click", function () {
        tabs.querySelectorAll(".pg-tab").forEach(function (x) { x.classList.remove("pg-tab-active"); x.setAttribute("aria-selected", "false"); });
        t.classList.add("pg-tab-active"); t.setAttribute("aria-selected", "true");
        panels.forEach(function (p) { p.hidden = p !== pane; });
      });
      tabs.appendChild(t);
      panels.push(pane);
    });
    function serverExportSnippet() {
      const S = model.input.size;
      return ["# This model is too large for the browser; Customise… above exports it on kornia's server with your opset and size.", "# pip install onnxruntime numpy pillow", "import numpy as np", "import onnxruntime as ort", "from PIL import Image", "",
        'sess = ort.InferenceSession("kornia-' + model.exports[0] + '-op17-1x3x' + S + 'x' + S + '.onnx")   # the exported file',
        'img = Image.open("photo.jpg").convert("RGB").resize((' + S + ", " + S + "))",
        "x = np.asarray(img, np.float32).transpose(2, 0, 1)[None] / 255",
        'outs = sess.run(None, {"image": x})', "print([o.shape for o in outs])"].join("\n");
    }
    function onnxRunnerSnippet() {
      const S = model.input.size, gray = !!model.input.gray, file = modelUrl.split("/").pop();
      const lines = ["# pip install onnxruntime numpy pillow", "import numpy as np", "import onnxruntime as ort", "from PIL import Image", "",
        'sess = ort.InferenceSession("' + file + '")   # the file from the Download button above',
        'img = Image.open("photo.jpg").convert("' + (gray ? "L" : "RGB") + '").resize((' + S + ", " + S + "))",
        "x = np.asarray(img, np.float32)" + (model.input.scale === 255 ? "                 # 0–255, as this model expects" : " / 255                       # 0–1")];
      if (model.input.mean) lines.push("x = (x - np.array(" + JSON.stringify(model.input.mean) + ", np.float32)) / np.array(" + JSON.stringify(model.input.std) + ", np.float32)   # ImageNet normalisation");
      lines.push(gray ? "x = x[None, None]                                 # (1, 1, H, W)" : "x = x.transpose(2, 0, 1)[None]                    # (1, 3, H, W)");
      lines.push('outs = sess.run(None, {"' + model.input.name + '": x.astype(np.float32)})');
      lines.push("print([o.shape for o in outs])" + (model.output.kind === "classification" ? "   # class scores; argmax gives the ImageNet label" : model.output.kind === "faces_yunet" ? "   # the raw heads; this page decodes them into faces" : model.output.kind === "detections" ? "   # (1, N, 6): class_id, score, x, y, w, h" : ""));
      return lines.join("\n");
    }
    code.appendChild(tabs);
    panels.forEach(function (p) { code.appendChild(p); });
    root.appendChild(code);

    // model card
    const card = el("div", "pg-model-card", { id: "details" });
    const table = el("table");
    [["Task", model.task], ["Input", model.input.size + "×" + model.input.size + " RGB" + (model.input.mean ? ", ImageNet-normalised" : model.input.scale === 255 ? ", 0–255" : ", 0–1")],
     ["Size", model.params + " · " + model.size_mb + " MB ONNX"], ["Hosted", model.hosted], ["Licence", model.license]].forEach(function (row) {
      const tr = el("tr"); const th = el("th"); th.textContent = row[0]; const td = el("td"); td.textContent = row[1]; tr.appendChild(th); tr.appendChild(td); table.appendChild(tr);
    });
    const tr = el("tr"); const th = el("th"); th.textContent = "Paper"; const td = el("td");
    const pa = el("a", "", { href: model.paper.url, target: "_blank", rel: "noopener" }); pa.textContent = model.paper.title; td.appendChild(pa);
    td.appendChild(document.createTextNode(" · "));
    const sa = el("a", "", { href: model.source, target: "_blank", rel: "noopener" }); sa.textContent = "source"; td.appendChild(sa);
    tr.appendChild(th); tr.appendChild(td); table.appendChild(tr);
    if (model.note) { const tr2 = el("tr"); const th2 = el("th"); th2.textContent = "Note"; const td2 = el("td"); td2.textContent = model.note; tr2.appendChild(th2); tr2.appendChild(td2); table.appendChild(tr2); }
    card.appendChild(table);
    const links = el("div", "pg-details-links");
    links.innerHTML = '<a id="pg-link-docs" href="' + (model.doc_url || "https://kornia.readthedocs.io") + '" target="_blank" rel="noopener"><i class="fas fa-book" aria-hidden="true"></i> API reference</a>'
      + ''
      + '<a id="pg-link-issue" href="https://github.com/kornia/kornia.github.io/issues/new?title=' + encodeURIComponent("playground model: " + model.id) + '" target="_blank" rel="noopener"><i class="fas fa-bug" aria-hidden="true"></i> Report an issue</a>';
    card.appendChild(links);
    root.appendChild(card);
    container.appendChild(root);

    // ---- running
    let ready = false, running = false, lastResults = null, labels = null;
    function draw(results) {
      if (isFeatures) {
        const m = matchMNN(results.a.descriptors, results.b.descriptors, threshold);
        drawMatches(canvasIn, canvasB, canvasOut, results.a, results.b, m);
        status.textContent = results.a.points.length + " + " + results.b.points.length + " keypoints, " + m.length + " mutual matches at cosine ≥ " + threshold.toFixed(2) + lastTime;
        return;
      }
      if (model.output.kind === "classification") { status.textContent = drawClassification(results, canvasIn, canvasOut, labels, model.output.top || 5) + lastTime; return; }
      if (model.output.kind === "detections") status.textContent = drawDetections(results, model, canvasIn, canvasOut, labels, threshold) + lastTime;
      else if (model.output.kind === "faces_yunet") status.textContent = drawFaces(results, model, canvasIn, canvasOut, threshold) + lastTime;
      else if (model.output.kind === "depth") { const t = results[Object.keys(results)[0]]; drawDepth(t, canvasOut); showCloud(t.data, t.dims[t.dims.length - 1], t.dims[t.dims.length - 2]); status.textContent = "relative depth, near is bright" + lastTime; }
      else { const t = results[Object.keys(results)[0]]; drawImageTensor(t, canvasOut); capOut.textContent = "output · " + t.dims[3] + "×" + t.dims[2]; status.textContent = "upscaled ×" + (model.output.scale || 1) + lastTime; }
    }
    let lastTime = "";
    function run() {
      if (!ready || running) return Promise.resolve();
      running = true;
      runBtn.disabled = true;
      status.classList.remove("pg-error");
      status.textContent = "running…";
      const t0 = performance.now();
      const mainSession = sessions[modelUrl];
      let job;
      if (isFeatures) {
        // frame and warped copy through the same pipeline, then matched
        const size = model.input.size;
        const grayIn = function (canvas) { return canvasToTensor(canvas, Object.assign({}, model.input, { gray: true })); };
        const features = function (canvas) {
          return mainSession.then(function (s) {
            const feeds = {};
            feeds[model.input.name] = model.input.gray ? grayIn(canvas) : canvasToTensor(canvas, model.input);
            return s.run(feeds);
          }).then(function (r) {
            if (model.output.kind === "features_xfeat") return xfeatFeatures(r, model.output, size);
            // keynet + hardnet: peaks of the response map, patches around them, then the descriptor graph
            const resp = r[Object.keys(r)[0]];
            // keep the whole support window inside the frame: drawImage would otherwise squash a clipped source rect
            const m = model.output.support / 2;
            const peaks = heatmapPeaks(resp.data, resp.dims[3], resp.dims[2], model.output.nms, 0, model.output.top_k * 2)
              .filter(function (p) { return p[0] >= m && p[1] >= m && p[0] < size - m && p[1] < size - m; }).slice(0, model.output.top_k);
            const pts = peaks.map(function (p) { return [p[0], p[1]]; });
            if (!pts.length) return { points: [], descriptors: [] };
            return sessions[absUrl(model.graphs.descriptor, ROOT)].then(function (sd) {
              return sd.run({ patches: extractPatches(canvas, pts, model.output.support, model.output.patch) });
            }).then(function (rd) {
              const d = rd[Object.keys(rd)[0]];
              const n = d.dims[0], dim = d.dims[1];
              const descs = [];
              for (let i = 0; i < n; i++) descs.push(new Float32Array(d.data.subarray(i * dim, (i + 1) * dim)));
              return { points: pts, descriptors: normalizeRows(descs) };
            });
          });
        };
        job = sessions[absUrl(model.graphs.warp, ROOT)].then(function (sw) { return warpCanvas(sw, canvasIn, canvasB, size, affine); })
          .then(function () { return features(canvasIn); })
          .then(function (a) { return features(canvasB).then(function (b) { return { a: a, b: b }; }); });
      } else {
        const feeds = {};
        feeds[model.input.name] = canvasToTensor(canvasIn, model.input);
        job = mainSession.then(function (s) { return s.run(feeds); });
      }
      return job.then(function (results) {
        lastResults = results;
        lastTime = " · " + (performance.now() - t0).toFixed(0) + " ms in your browser";
        draw(results);
      }).catch(function (err) {
        status.classList.add("pg-error");
        status.textContent = "run failed: " + (err.message || err);
      }).then(function () { running = false; runBtn.disabled = false; });
    }
    function load() {
      runBtn.disabled = true;
      progress.hidden = false;
      status.classList.remove("pg-error");
      status.textContent = "downloading…";
      const t0 = performance.now();
      const labelsReady = model.output.labels ? getLabels(model.output.labels, ROOT).then(function (l) { labels = l; }) : Promise.resolve();
      return Promise.all([getSession(model, ROOT, function (got, total) {
        bar.style.width = total ? (100 * got / total).toFixed(1) + "%" : "50%";
        status.textContent = "downloading… " + (got / 1e6).toFixed(1) + (total ? " / " + (total / 1e6).toFixed(1) : "") + " MB";
      }), labelsReady]).then(function () {
        ready = true;
        progress.hidden = true;
        runBtn.disabled = false;
        status.textContent = "loaded in " + ((performance.now() - t0) / 1000).toFixed(1) + " s";
      }).catch(function (err) {
        progress.hidden = true;
        runBtn.disabled = false;
        status.classList.add("pg-error");
        status.textContent = "could not load the model: " + (err.message || err);
        throw err;
      });
    }
    runBtn.addEventListener("click", function () {
      if (target.get() === "server") { runServer(); return; }
      onServer = false;
      if (typeof ort === "undefined") { status.classList.add("pg-error"); status.textContent = "onnxruntime-web has not finished loading; try again in a moment."; return; }
      if (ready) { run(); return; }
      askToLoad().then(function (ok) {
        if (!ok) { status.textContent = "Not loaded. Press Run whenever you want to try it."; return; }
        return load().then(run);
      }).catch(function () {});
    });
    drawStill(currentImage);
  }

  window.PGModels = { buildList: buildList, highlight: highlight, render: render, decodeFaces: decodeFaces, paintFaces: paintFaces, drawBox: drawBox, drawMatches: drawMatches, drawResults: drawResults, getLabels: getLabels, askSignIn: askSignIn, uploadButton: uploadButton, targetSwitch: targetSwitch, exportDialog: exportDialog, modelExportDialog: modelExportDialog,
    runOnServer: runOnServer, drawServerResult: drawServerResult, onQuota: onQuota, refreshQuota: refreshQuota, setQuota: setQuota, xfeatFeatures: xfeatFeatures, matchMNN: matchMNN };
})();
