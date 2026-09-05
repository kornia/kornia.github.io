// kornia.org/playground
//
// One shell for every page: a sidebar listing every image operator (search + filter) and a detail
// pane showing the selected operator's widget. index.html and ops/<slug>/index.html share this
// script; the latter preselects its operator through body[data-op]. Clicking a sidebar entry swaps
// the widget in place and pushes the operator page URL, so every state has a real static URL.
// ONNX operators run through onnxruntime-web (loaded before this file); frame-mode operators swap
// pre-rendered images. Vanilla JS, no build step.
(function () {
  "use strict";

  // Absolute URL of playground/, resolved once from body[data-base] (the relative path from this
  // page). Every asset URL and pushed history entry is built from it, so switching operators with
  // pushState (which changes the document URL) never makes relative paths pile up.
  const ROOT = new URL(document.body.dataset.base || "./", window.location.href).href;
  const RUN_DELAY_MS = 30;
  const RUST_VERSION = "0.1.14";
  let preferredLang = "python"; // code tab opened by default; the sidebar's Rust filter switches it

  // ------------------------------------------------------------------ helpers

  function el(tag, cls, attrs) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    for (const key in attrs || {}) node.setAttribute(key, attrs[key]);
    return node;
  }

  function fmt(value, param) {
    if (param.type === "int") return String(Math.round(value));
    if (param.type === "str") return String(value);
    const s = Number(value).toFixed(4);
    return s.replace(/\.?0+$/, "") || "0";
  }

  // Rust wants a float literal for f32 parameters
  function fmtRust(value, param) {
    const s = fmt(value, param);
    return param.type === "float" && s.indexOf(".") === -1 ? s + ".0" : s;
  }

  function nearest(values, target) {
    let best = 0;
    for (let i = 1; i < values.length; i++) {
      if (Math.abs(values[i] - target) < Math.abs(values[best] - target)) best = i;
    }
    return best;
  }

  function fill(template, values) {
    return template.replace(/\{(\w+)\}/g, function (m, key) { return key in values ? values[key] : m; });
  }

  function loadImage(src) {
    return new Promise(function (resolve, reject) {
      const img = new Image();
      img.onload = function () { resolve(img); };
      img.onerror = function () { reject(new Error("could not load " + src)); };
      img.src = src;
    });
  }

  function canvasToTensor(canvas, size) {
    const data = canvas.getContext("2d").getImageData(0, 0, size, size).data;
    const plane = size * size;
    const out = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      out[i] = data[4 * i] / 255;
      out[plane + i] = data[4 * i + 1] / 255;
      out[2 * plane + i] = data[4 * i + 2] / 255;
    }
    return new ort.Tensor("float32", out, [1, 3, size, size]);
  }

  // (1, C, H, W) float32 -> canvas. C in {1, 3, 4}; "normalize" stretches min..max to 0..1.
  function tensorToCanvas(tensor, canvas, display) {
    const dims = tensor.dims;
    const channels = dims[1], h = dims[2], w = dims[3];
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(w, h);
    const plane = h * w;
    const src = tensor.data;
    let lo = 0, scale = 1;
    if (display === "normalize") {
      let mn = Infinity, mx = -Infinity;
      const n = Math.min(channels, 3) * plane;
      for (let i = 0; i < n; i++) { const v = src[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
      lo = mn; scale = mx > mn ? 1 / (mx - mn) : 1;
    }
    for (let i = 0; i < plane; i++) {
      const r = (src[i] - lo) * scale;
      const g = channels === 1 ? r : (src[plane + i] - lo) * scale;
      const b = channels === 1 ? r : (src[2 * plane + i] - lo) * scale;
      img.data[4 * i] = Math.max(0, Math.min(255, Math.round(r * 255)));
      img.data[4 * i + 1] = Math.max(0, Math.min(255, Math.round(g * 255)));
      img.data[4 * i + 2] = Math.max(0, Math.min(255, Math.round(b * 255)));
      img.data[4 * i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  // ------------------------------------------------------------------ code snippets

  function pythonSnippet(op, state, registry) {
    const values = { img: "img" };
    op.params.forEach(function (p) {
      values[p.name] = fmt(state.params[p.name], p);
      // predefined kernels: {kernel_literal} expands to the tensor literal of the chosen kernel
      if (p.literals) values[p.name + "_literal"] = p.literals[state.params[p.name]];
      // values derived from a choice for the snippet, e.g. {strength_one_minus} -> 1 - strength
      if (p.derived && p.derived[String(state.params[p.name])]) {
        const d = p.derived[String(state.params[p.name])];
        for (const k in d) values[p.name + "_" + k] = d[k];
      }
    });
    const image = registry.images.find(function (im) { return im.id === state.image; });
    const lines = ["import torch", "import kornia", "from kornia.io import load_image", ""];
    if (image) {
      lines.push("# the " + registry.size + " px sample shown in the playground (\"" + image.label + "\"):");
      lines.push("# https://kornia.org/playground/" + image.file);
      lines.push("img = load_image(\"" + image.id + ".png\")[None]  # (1, 3, " + registry.size + ", " + registry.size + ") float in [0, 1]");
    } else {
      lines.push("img = load_image(\"my_photo.png\")[None]  # your own image: (1, 3, H, W) float in [0, 1]");
    }
    lines.push("");
    if (op.guidance) {
      lines.push("# the fixed guidance image (\"" + op.guidance.label + "\"): https://kornia.org/playground/" + op.guidance.file);
      lines.push("guide = load_image(\"" + op.guidance.id + ".png\")[None]");
      lines.push("");
    }
    if (state.kind === "video") {
      const v = (registry.videos || []).find(function (x) { return x.id === state.image; });
      if (v) lines.push("# in the playground the same call runs on every frame of " + v.mp4.split("/").pop() + " (" + v.license + ")");
    }
    if (op.stochastic) lines.push("torch.manual_seed(0)  # the browser draws its own random parameters");
    lines.push("out = " + fill(op.snippet, values));
    if (op.module_snippet) {
      lines.push("");
      lines.push("# the same operator as an nn.Module, e.g. inside nn.Sequential:");
      lines.push("out = " + fill(op.module_snippet, values));
    }
    return lines.join("\n");
  }

  function rustSnippet(op, state, registry) {
    const r = op.rust;
    if (!r) return null;
    const values = {};
    op.params.forEach(function (p) { values[p.name] = fmtRust(state.params[p.name], p); });
    const image = registry.images.find(function (im) { return im.id === state.image; });
    const uses = ["kornia_image::{Image, ImageSize, allocator::CpuAllocator}", "kornia_io::functional::read_image_any_rgb8"].concat(r.use);
    const lines = uses.map(function (u) { return "use " + u + ";"; });
    lines.push("");
    if (image) {
      lines.push("// kornia-rs " + RUST_VERSION + ". The same " + registry.size + " px sample (\"" + image.label + "\"):");
      lines.push("// https://kornia.org/playground/" + image.file);
    } else {
      lines.push("// kornia-rs " + RUST_VERSION + ", on your own image:");
    }
    lines.push("let img = read_image_any_rgb8(\"" + (image ? image.id : "my_photo") + ".png\")?;          // Image<u8, 3>");
    lines.push("let img = img.cast_and_scale::<f32>(1.0 / 255.0)?;   // Image<f32, 3> in [0, 1]");
    const size = r.out_size ? fill(r.out_size, values) : "img.size()";
    lines.push("let mut out = Image::<f32, " + r.channels + ", _>::from_size_val(" + size + ", 0.0, CpuAllocator)?;");
    r.call.forEach(function (line) { lines.push(fill(line, values)); });
    if (r.note) lines.push("// " + r.note);
    return lines.join("\n");
  }

  const sessions = {};
  function getSession(path) {
    if (!sessions[path]) {
      sessions[path] = ort.InferenceSession.create(path, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    }
    return sessions[path];
  }

  // ------------------------------------------------------------------ the widget

  // ------------------------------------------------------------------ volume operators: two ray-marched views

  function buildVolumeWidget(op, registry, container) {
    const vol = (registry.volumes || [])[0];
    const N = op.volume.size;
    const state = { params: {} };
    op.params.forEach(function (p) { state.params[p.name] = p.default; });
    const root = el("article", "pg-op pg-op-volume", { id: op.slug });
    const frames = op.mode === "frames";   // the output was rendered offline per choice; nothing runs here
    const head = el("div", "pg-op-head pg-head");
    const title = el("h2"); title.textContent = op.name;
    const code = el("code"); code.textContent = op.id;
    const tag = el("span", "pg-tag"); tag.textContent = frames ? "3D volume · pre-rendered" : "3D volume";
    head.appendChild(title); head.appendChild(code); head.appendChild(tag);
    root.appendChild(head);
    const summary = el("p", "pg-summary");
    summary.textContent = op.summary;
    root.appendChild(summary);
    const grid = el("div", "pg-grid");
    root.appendChild(grid);

    // ---- stage: input volume | output volume, one shared orbit
    const stage = el("div", "pg-vol-stage pg-cell-images", { id: "demo" });
    const figIn = el("figure"), figOut = el("figure");
    const canvasIn = el("canvas"), canvasOut = el("canvas");
    const capIn = el("figcaption"), capOut = el("figcaption");
    capIn.textContent = "input · " + N + "×" + N + "×" + N + " · drag to orbit, wheel to zoom";
    capOut.textContent = "output" + (op.output && op.output.display === "normalize" ? " (normalised for display)" : "");
    figIn.appendChild(canvasIn); figIn.appendChild(capIn); figOut.appendChild(canvasOut); figOut.appendChild(capOut);
    stage.appendChild(figIn); stage.appendChild(figOut);
    grid.appendChild(stage);
    let viewers = null;
    function getViewers() {
      if (!viewers) {
        viewers = import(ROOT + "../viewer3d.js").then(function (mod) {
          return Promise.all([mod.createVolumeViewer(canvasIn), mod.createVolumeViewer(canvasOut)]).then(function (vs) {
            vs[0].onOrbit(function () { vs[1].setView(vs[0].view()); });
            vs[1].onOrbit(function () { vs[0].setView(vs[1].view()); });
            return { a: vs[0], b: vs[1] };
          });
        });
      }
      return viewers;
    }

    // ---- bar: the sample on the left, status on the right
    const bar = el("div", "pg-bar pg-cell-bar");
    const thumbs = el("div", "pg-thumbs");
    const pick = el("div", "pg-thumbs-pick pg-thumbs-volumes");
    const sampleBtn = el("button", "pg-thumb-volume pg-selected", { type: "button", title: vol ? vol.label : "volume" });
    sampleBtn.innerHTML = '<i class="fas fa-cube" aria-hidden="true"></i>&nbsp; ' + (vol ? vol.label : "volume");
    pick.appendChild(sampleBtn); thumbs.appendChild(pick); bar.appendChild(thumbs);
    const actions = el("div", "pg-actions");
    if (op.stochastic && !frames) {
      const reroll = el("button", "pg-btn", { type: "button" });
      reroll.innerHTML = '<i class="fas fa-dice" aria-hidden="true"></i> Re-roll';
      reroll.addEventListener("click", schedule);
      actions.appendChild(reroll);
    }
    const status = el("span", "pg-status");
    status.textContent = "loading the volume…";
    actions.appendChild(status);
    bar.appendChild(actions);
    grid.appendChild(bar);

    // ---- parameters: the same live / select controls as the image operators
    const controls = el("div", "pg-controls");
    op.params.forEach(function (p) {
      const row = el("div", "pg-param");
      const label = el("label", "", { for: op.slug + "-" + p.name }); label.textContent = p.label || p.name;
      const out = el("output"); out.textContent = fmt(p.default, p);
      let input;
      if (p.kind === "select") {
        input = el("select", "", { id: op.slug + "-" + p.name });
        p.choices.forEach(function (c, i) { const o = el("option", "", { value: c }); o.textContent = p.labels ? p.labels[i] : fmt(c, p); if (c === p.default) o.selected = true; input.appendChild(o); });
        input.addEventListener("change", function () { state.params[p.name] = p.type === "str" ? input.value : Number(input.value); out.textContent = fmt(state.params[p.name], p); schedule(); });
      } else {
        input = el("input", "", { type: "range", id: op.slug + "-" + p.name, min: p.min, max: p.max, step: p.step, value: p.default });
        input.addEventListener("input", function () { state.params[p.name] = Number(input.value); out.textContent = fmt(state.params[p.name], p); schedule(); });
      }
      row.appendChild(label); row.appendChild(input); row.appendChild(out); controls.appendChild(row);
    });
    if (!op.params.length) { const none = el("p", "pg-note"); none.textContent = "This operator has no parameters."; controls.appendChild(none); }
    const controlsCell = el("div", "pg-cell-controls pg-controls-cell", { id: "parameters" });
    controlsCell.appendChild(controls);
    grid.appendChild(controlsCell);

    // ---- code: Python, and ONNX with the download when there is a graph
    const opset = op.opset || registry.onnx_opset;
    function graphKey() {
      if (!op.select_order || !op.select_order.length) return "default";
      return op.select_order.map(function (name) { const p = op.params.find(function (q) { return q.name === name; }); return fmt(state.params[name], p); }).join("|");
    }
    function graphFile() { const key = graphKey(); return op.slug + (key === "default" ? "" : "-" + key.replace(/\|/g, "_")) + ".onnx"; }
    function pyValues() {
      const values = { img: "vol" };
      op.params.forEach(function (p) {
        values[p.name] = fmt(state.params[p.name], p);
        if (p.literals) values[p.name + "_literal"] = p.literals[state.params[p.name]];
        if (p.derived && p.derived[String(state.params[p.name])]) { const d = p.derived[String(state.params[p.name])]; for (const k in d) values[p.name + "_" + k] = d[k]; }
      });
      return values;
    }
    function pythonCode() {
      const lines = ["import numpy as np", "import torch", "import kornia", "",
        "# the " + N + "³ phantom shown in the playground, raw float32 (D, H, W): https://kornia.org/playground/" + (vol ? vol.file : "volumes/phantom.bin"),
        'vol = torch.from_numpy(np.fromfile("phantom.bin", np.float32).reshape(1, 1, ' + N + ", " + N + ", " + N + "))", ""];
      if (op.stochastic) lines.push("torch.manual_seed(0)  # the browser draws its own random parameters");
      lines.push("out = " + fill(op.snippet, pyValues()));
      if (op.module_snippet) { lines.push(""); lines.push("# the same operator as an nn.Module:"); lines.push("out = " + fill(op.module_snippet, pyValues())); }
      return lines.join("\n");
    }
    function onnxCode() {
      const feeds = ['"' + op.inputs[0] + '": vol'];
      let k = 1;
      op.params.filter(function (p) { return p.kind === "live"; }).forEach(function (p) { feeds.push('"' + op.inputs[k++] + '": np.array([' + fmt(state.params[p.name], p) + '], np.float32)'); });
      return ["# pip install onnxruntime numpy", "import numpy as np", "import onnxruntime as ort", "",
        "# the file from the Download button above: opset " + opset + ", a fixed 1×1×" + N + "×" + N + "×" + N + " volume",
        'sess = ort.InferenceSession("' + graphFile() + '")',
        'vol = np.fromfile("phantom.bin", np.float32).reshape(1, 1, ' + N + ", " + N + ", " + N + ")   # any (1, 1, D, H, W) float volume in [0, 1] of this size",
        "out = sess.run(None, {" + feeds.join(", ") + "})[0]"].join("\n");
    }
    const codeBox = el("div", "pg-code", { id: "code" });
    const tabs = el("div", "pg-tabs", { role: "tablist" });
    const panes = {}, pres = {};
    const download = el("a", "pg-btn pg-btn-small", { download: "" });
    download.innerHTML = '<i class="fas fa-file-arrow-down" aria-hidden="true"></i> Download ONNX';
    const customise = el("button", "pg-btn pg-btn-ghost pg-btn-small", { type: "button", title: "Export this operator with your own opset and parameters, on kornia's server" });
    customise.innerHTML = '<i class="fas fa-sliders" aria-hidden="true"></i> Customise…';
    customise.addEventListener("click", function () { if (window.PGModels && window.PGModels.exportDialog) window.PGModels.exportDialog(op, registry, state.params, ROOT); });
    const onnxNote = el("span", "pg-onnx-note");
    (frames ? ["python"] : ["python", "onnx"]).forEach(function (lang, i) {
      const tab = el("button", "pg-tab" + (i === 0 ? " pg-tab-active" : ""), { type: "button", role: "tab", "data-lang": lang, "aria-selected": i === 0 ? "true" : "false" });
      tab.textContent = lang === "python" ? "Python" : "ONNX";
      tab.addEventListener("click", function () {
        tabs.querySelectorAll(".pg-tab").forEach(function (t) { t.classList.remove("pg-tab-active"); t.setAttribute("aria-selected", "false"); });
        tab.classList.add("pg-tab-active"); tab.setAttribute("aria-selected", "true");
        for (const key in panes) panes[key].hidden = key !== lang;
      });
      tabs.appendChild(tab);
      const pane = el("div", "pg-pane", { role: "tabpanel" }); pane.hidden = i !== 0;
      if (lang === "onnx") { const headRow = el("div", "pg-onnx-head"); headRow.appendChild(download); headRow.appendChild(customise); headRow.appendChild(onnxNote); pane.appendChild(headRow); }
      const pre = el("pre"); const c = el("code", "language-python"); pre.appendChild(c); pane.appendChild(pre);
      panes[lang] = pane; pres[lang] = c;
    });
    const copy = el("button", "pg-copy", { type: "button" }); copy.textContent = "copy";
    copy.addEventListener("click", function () {
      const visible = panes.python.hidden ? pres.onnx : pres.python;
      navigator.clipboard.writeText(visible.textContent).then(function () { copy.textContent = "copied"; setTimeout(function () { copy.textContent = "copy"; }, 1200); });
    });
    tabs.appendChild(copy);
    codeBox.appendChild(tabs);
    codeBox.appendChild(panes.python); if (panes.onnx) codeBox.appendChild(panes.onnx);
    grid.appendChild(codeBox);

    // ---- details + links
    const details = el("div", "pg-details pg-model-card pg-cell-details", { id: "details" });
    const table = el("table"); details.appendChild(table);
    const links = el("div", "pg-details-links");
    links.innerHTML = '<a id="pg-link-docs" href="' + op.doc_url + '" target="_blank" rel="noopener"><i class="fas fa-book" aria-hidden="true"></i> API reference</a>'
      + ''
      + '<a id="pg-link-issue" href="https://github.com/kornia/kornia.github.io/issues/new?title=' + encodeURIComponent("playground: " + op.id) + '" target="_blank" rel="noopener"><i class="fas fa-bug" aria-hidden="true"></i> Report an issue</a>';
    details.appendChild(links);
    grid.appendChild(details);
    container.appendChild(root);

    function setCode(c, text) { c.textContent = text; if (typeof hljs !== "undefined") { c.removeAttribute("data-highlighted"); hljs.highlightElement(c); } }
    function renderCode() {
      setCode(pres.python, pythonCode());
      if (frames) {
        table.innerHTML = "";
        [["Rendering", "volumes rendered offline with kornia " + registry.kornia + "; this operator cannot export to ONNX yet" + (op.frames.note ? ": " + op.frames.note : "")],
         ["Input", "a 1×1×" + N + "×" + N + "×" + N + " volume, float in [0, 1]"], ["Choices", op.frames.values.length + " along " + (op.frames.param || "the defaults")]].forEach(function (r) {
          const tr = el("tr"); const th = el("th"); th.textContent = r[0]; const td = el("td"); td.textContent = r[1]; tr.appendChild(th); tr.appendChild(td); table.appendChild(tr);
        });
        return;
      }
      setCode(pres.onnx, onnxCode());
      const key = graphKey();
      download.href = ROOT + op.graphs[key];
      download.setAttribute("download", graphFile());
      onnxNote.textContent = (op.graph_kb ? op.graph_kb[key] + " KB · " : "") + "opset " + opset + " · 1×1×" + N + "×" + N + "×" + N + " only · runs with onnxruntime, onnxruntime-web and the ort crate";
      table.innerHTML = "";
      [["ONNX graph", (op.graph_kb ? op.graph_kb[key] + " KB, " : "") + "opset " + opset], ["Input", "a 1×1×" + N + "×" + N + "×" + N + " volume, float in [0, 1]"],
       ["Output", op.output ? op.output.depth + "×" + op.output.height + "×" + op.output.width : ""], ["Exported with", "kornia " + registry.kornia + " / torch " + registry.torch.split("+")[0]],
       op.sampling_folded ? ["Randomness", "the random parameters were drawn once at export, so the graph is deterministic"] : null].filter(Boolean).forEach(function (r) {
        const tr = el("tr"); const th = el("th"); th.textContent = r[0]; const td = el("td"); td.textContent = r[1]; tr.appendChild(th); tr.appendChild(td); table.appendChild(tr);
      });
    }

    // ---- running
    let inputTensor = null, timer = null, running = false, pending = false;
    function schedule() { renderCode(); clearTimeout(timer); timer = setTimeout(run, RUN_DELAY_MS); }
    function showFrame() {
      const idx = op.frames.param === null ? 0 : nearest(op.frames.values, state.params[op.frames.param]);
      const shape = op.frames.shapes[idx];
      fetch(ROOT + op.frames.dir + "/phantom/" + String(idx).padStart(2, "0") + ".bin").then(function (r) { return r.arrayBuffer(); }).then(function (buf) {
        return getViewers().then(function (v) { v.b.setVolume(new Float32Array(buf), shape[0], shape[1], shape[2]); status.textContent = "volume " + (idx + 1) + " of " + op.frames.values.length; });
      }).catch(function (e) { status.classList.add("pg-error"); status.textContent = "could not load the frame: " + (e.message || e); });
    }
    function run() {
      if (frames) { showFrame(); return; }
      if (typeof ort === "undefined") { status.classList.add("pg-error"); status.textContent = "onnxruntime-web did not load; the download and code still work."; return; }
      if (running) { pending = true; return; }
      if (!inputTensor) return;
      running = true;
      status.classList.remove("pg-error");
      const feeds = {}; feeds[op.inputs[0]] = inputTensor;
      let next = 1;
      op.params.filter(function (p) { return p.kind === "live"; }).forEach(function (p) { feeds[op.inputs[next++]] = new ort.Tensor("float32", new Float32Array([state.params[p.name]]), [1]); });
      const t0 = performance.now();
      getSession(ROOT + op.graphs[graphKey()]).then(function (session) { return session.run(feeds); }).then(function (results) {
        const out = results[Object.keys(results)[0]];
        const dims = out.dims, d = dims[dims.length - 3], h = dims[dims.length - 2], w = dims[dims.length - 1];
        return getViewers().then(function (v) { v.b.setVolume(out.data, d, h, w, { normalize: op.output && op.output.display === "normalize" }); status.textContent = (performance.now() - t0).toFixed(0) + " ms on your machine"; });
      }).catch(function (err) { status.classList.add("pg-error"); status.textContent = "could not run the graph: " + (err.message || err); })
        .then(function () { running = false; if (pending) { pending = false; run(); } });
    }
    renderCode();
    fetch(ROOT + (vol ? vol.file : "volumes/phantom.bin")).then(function (r) { return r.arrayBuffer(); }).then(function (buf) {
      const data = new Float32Array(buf);
      if (!frames && typeof ort !== "undefined") inputTensor = new ort.Tensor("float32", data, [1, 1, N, N, N]);
      return getViewers().then(function (v) { v.a.setVolume(data, N, N, N); status.textContent = frames ? "" : "loading graph…"; run(); });
    }).catch(function (e) { status.classList.add("pg-error"); status.textContent = "could not load the volume: " + (e.message || e); });
  }

  function buildWidget(op, registry, container) {
    if (op.input_kind === "volume") return buildVolumeWidget(op, registry, container);
    const size = registry.size;
    const state = { image: registry.images[0].id, kind: "image", params: {} };
    op.params.forEach(function (p) { state.params[p.name] = p.default; });

    const root = el("article", "pg-op", { id: op.slug });
    const head = el("div", "pg-op-head pg-head");
    const title = el("h2");
    title.textContent = op.name;
    const code = el("code");
    code.textContent = op.id;
    head.appendChild(title);
    head.appendChild(code);
    if (op.mode === "frames") {
      const tag = el("span", "pg-tag");
      tag.textContent = "pre-rendered frames";
      head.appendChild(tag);
    }
    root.appendChild(head);

    const summary = el("p", "pg-summary");
    summary.textContent = op.summary + (op.pre ? " (" + op.pre + ".)" : "");
    root.appendChild(summary);

    const grid = el("div", "pg-grid");
    root.appendChild(grid);
    const bar = el("div", "pg-bar pg-cell-bar");   // one row under the stage: samples left, actions right

    // ---- images column
    const left = el("div", "", { id: "demo" });
    const images = el("div", "pg-images");
    const figIn = el("figure");
    const canvasIn = el("canvas", "", { width: size, height: size });
    const capIn = el("figcaption");
    capIn.textContent = "input";
    figIn.appendChild(canvasIn);
    figIn.appendChild(capIn);
    const figOut = el("figure");
    const canvasOut = el("canvas", "", { width: size, height: size });
    const imgOut = el("img", "", { alt: op.name + " output", hidden: "" });
    const capOut = el("figcaption");
    capOut.textContent = "output" + (op.output && op.output.display === "normalize" ? " (normalised for display)" : "");
    figOut.appendChild(canvasOut);
    figOut.appendChild(imgOut);
    figOut.appendChild(capOut);

    images.appendChild(figIn);
    images.appendChild(figOut);
    left.appendChild(images);

    // picker row: the selectable samples on the left (under the input), the fixed guidance image,
    // when the operator has one, under the output at the same size
    const thumbs = el("div", "pg-thumbs");
    const pick = el("div", "pg-thumbs-pick");
    thumbs.appendChild(pick);
    function markSelected(btn) {
      thumbs.querySelectorAll("button").forEach(function (b) { b.classList.remove("pg-selected"); });
      btn.classList.add("pg-selected");
    }
    registry.images.filter(function (im) { return im.selectable !== false; }).forEach(function (im) {
      const btn = el("button", im.id === state.image ? "pg-selected" : "", { type: "button", title: im.label, "aria-label": im.label });
      const t = el("img", "", { src: ROOT +im.thumb, alt: "" });
      btn.appendChild(t);
      pick.appendChild(btn);
      btn.addEventListener("click", function () {
        stopVideo();
        state.image = im.id;
        state.kind = "image";
        capIn.textContent = "input";
        markSelected(btn);
        loadInput().then(schedule);
      });
    });
    // your own image: signed-in visitors only (the button asks otherwise); live graphs only, since
    // frame-mode operators are pre-rendered for the samples
    let uploaded = null, uploadBtn = null;
    // frame-mode operators have no graph to run here: the pre-rendered frames preview the samples, and
    // "Run on server" computes the exact result for any parameters, on your own image too
    const onServer = op.mode === "frames";
    if ((op.mode === "onnx" || op.mode === "frames") && window.PGModels && window.PGModels.uploadButton) {
      uploadBtn = window.PGModels.uploadButton(function (img, name) {
        stopVideo();
        uploaded = img;
        state.image = "upload";
        state.kind = "image";
        capIn.textContent = "input · " + name;
        thumbs.querySelectorAll("button").forEach(function (b) { b.classList.remove("pg-selected"); });
        if (op.mode === "frames") {
          loadInput().then(runServerOp);   // no graph can run here on a new image: the server does, at once
          return;
        }
        loadInput().then(schedule);
      });
      pick.appendChild(uploadBtn);
    }
    // short clips: the graph runs on every frame (frame-mode operators have no live graph to run)
    (registry.videos || []).forEach(function (v) {
      if (op.mode !== "onnx") return;
      const btn = el("button", "pg-thumb-video", { type: "button", title: v.label + " (" + v.seconds + " s, " + v.license + ")", "aria-label": "video: " + v.label });
      const t = el("img", "", { src: ROOT +v.thumb, alt: "" });
      const play = el("span", "pg-thumb-play", { "aria-hidden": "true" });
      play.innerHTML = '<i class="fas fa-play"></i>';
      btn.appendChild(t);
      btn.appendChild(play);
      pick.appendChild(btn);
      btn.addEventListener("click", function () {
        state.image = v.id;
        state.kind = "video";
        markSelected(btn);
        capIn.textContent = "input: " + v.label + " (" + v.license + ")";
        startVideo(v);
      });
    });
    if (op.guidance) {
      const guide = el("div", "pg-guide", { title: "guidance image: " + op.guidance.label + " (fixed)" });
      const small = el("img", "pg-guide-thumb", { src: ROOT + op.guidance.thumb, alt: "guidance image: " + op.guidance.label });
      const large = el("img", "pg-guide-large", { src: ROOT + op.guidance.file, alt: "" });
      const label = el("span");
      label.textContent = "guidance (fixed)";
      guide.appendChild(small);
      guide.appendChild(label);
      guide.appendChild(large);
      thumbs.appendChild(guide);
    }
    left.className = "pg-cell-images";
    grid.appendChild(left);
    bar.appendChild(thumbs);
    grid.appendChild(bar);

    // ---- controls column
    const controls = el("div", "pg-controls");
    const framesParam = op.mode === "frames" ? op.frames.param : null;
    op.params.forEach(function (p) {
      const row = el("div", "pg-param");
      const label = el("label", "", { for: op.slug + "-" + p.name });
      label.textContent = p.label || p.name;
      const out = el("output");
      out.textContent = fmt(p.default, p);
      let input;
      if (op.mode === "frames" && p.name !== framesParam) {
        row.classList.add("pg-fixed");
        input = el("span");
        input.textContent = "fixed";
      } else if (p.kind === "select" && op.mode !== "frames") {
        input = el("select", "", { id: op.slug + "-" + p.name });
        p.choices.forEach(function (c, i) {
          const option = el("option", "", { value: c });
          option.textContent = p.labels ? p.labels[i] : fmt(c, p);
          if (c === p.default) option.selected = true;
          input.appendChild(option);
        });
        input.addEventListener("change", function () {
          state.params[p.name] = p.type === "str" ? input.value : Number(input.value);
          out.textContent = fmt(state.params[p.name], p);
          schedule();
        });
      } else if (op.mode === "frames") {
        const values = op.frames.values;
        input = el("input", "", { type: "range", id: op.slug + "-" + p.name, min: 0, max: values.length - 1, step: 1 });
        input.value = p.kind === "select" ? Math.max(0, values.indexOf(p.default)) : nearest(values, p.default);
        state.params[p.name] = values[input.value];
        out.textContent = fmt(state.params[p.name], p);
        input.addEventListener("input", function () {
          state.params[p.name] = values[Number(input.value)];
          out.textContent = fmt(state.params[p.name], p);
          schedule();
        });
      } else {
        input = el("input", "", { type: "range", id: op.slug + "-" + p.name, min: p.min, max: p.max, step: p.step, value: p.default });
        input.addEventListener("input", function () {
          state.params[p.name] = Number(input.value);
          out.textContent = fmt(state.params[p.name], p);
          schedule();
        });
      }
      row.appendChild(label);
      row.appendChild(input);
      row.appendChild(out);
      controls.appendChild(row);
    });
    if (!op.params.length) {
      const none = el("p", "pg-note");
      none.textContent = "This operator has no parameters.";
      controls.appendChild(none);
    }

    // frame-mode operators: on kornia's server every parameter is live and your own image is accepted
    const frameParams = state.params;
    const liveParams = {};
    let liveControls = null, runBtn = null;
    let serverBusy = false;
    function markDirty() {
      renderCode();
      status.classList.remove("pg-error");
      runFrames();   // the nearest pre-rendered frame previews the change; Run computes it exactly
    }
    if (op.mode === "frames") {
      op.params.forEach(function (p) { liveParams[p.name] = p.default; });
      liveControls = el("div", "pg-controls");
      op.params.forEach(function (p) {
        const row = el("div", "pg-param");
        const label = el("label", "", { for: op.slug + "-live-" + p.name });
        label.textContent = p.label || p.name;
        const out = el("output");
        out.textContent = fmt(p.default, p);
        let input;
        if (p.kind === "select") {
          input = el("select", "", { id: op.slug + "-live-" + p.name });
          p.choices.forEach(function (ch, i) {
            const option = el("option", "", { value: ch });
            option.textContent = p.labels ? p.labels[i] : fmt(ch, p);
            if (ch === p.default) option.selected = true;
            input.appendChild(option);
          });
          input.addEventListener("change", function () {
            liveParams[p.name] = p.type === "str" ? input.value : Number(input.value);
            out.textContent = fmt(liveParams[p.name], p);
            markDirty();
          });
        } else {
          input = el("input", "", { type: "range", id: op.slug + "-live-" + p.name, min: p.min, max: p.max, step: p.step, value: p.default });
          input.addEventListener("input", function () {
            liveParams[p.name] = Number(input.value);
            out.textContent = fmt(liveParams[p.name], p);
            markDirty();
          });
        }
        row.appendChild(label);
        row.appendChild(input);
        row.appendChild(out);
        liveControls.appendChild(row);
      });
      if (!op.params.length) {
        const none = el("p", "pg-note");
        none.textContent = "This operator has no parameters.";
        liveControls.appendChild(none);
      }
      state.params = liveParams;
    }
    function clearOutput() {
      imgOut.hidden = true;
      canvasOut.hidden = false;
      canvasOut.getContext("2d").clearRect(0, 0, canvasOut.width, canvasOut.height);
    }
    function inputBlob() {
      if (state.image === "upload" && uploaded) {
        const MAX = 512, sc = Math.min(1, MAX / Math.max(uploaded.width, uploaded.height));   // the server would resize anyway
        const c = el("canvas", "", { width: Math.max(1, Math.round(uploaded.width * sc)), height: Math.max(1, Math.round(uploaded.height * sc)) });
        c.getContext("2d").drawImage(uploaded, 0, 0, c.width, c.height);
        return new Promise(function (resolve) { c.toBlob(resolve, "image/png"); });
      }
      const im = registry.images.find(function (i) { return i.id === state.image; });
      return fetch(ROOT + im.file).then(function (r) { return r.blob(); });
    }
    async function runServerOp() {
      const A = window.KorniaAuth;
      if (!A || !A.user) { window.PGModels.askSignIn("run " + op.name + " on kornia's server", runServerOp); return; }   // the prompt can be dismissed
      if (!A.apiBase) { status.classList.add("pg-error"); status.textContent = "the server side is not deployed yet"; return; }
      if (serverBusy) return;
      serverBusy = true;
      runBtn.disabled = true;
      status.classList.remove("pg-error");
      status.textContent = "running on the server…";
      const t0 = performance.now();
      try {
        const form = new FormData();
        form.append("image", await inputBlob(), "input.png");
        form.append("params", JSON.stringify(liveParams));
        const r = await fetch(A.apiBase + "/v1/run_op/" + op.slug, { method: "POST", body: form, headers: { Authorization: "Bearer " + await A.token() } });
        const j = await r.json().catch(function () { return {}; });
        if (!r.ok) {
          const d = j.detail || ("HTTP " + r.status);
          throw new Error(d.indexOf("verify your email") !== -1 ? "Your account's email is not verified yet; open the link in the verification mail, then try again." : d);
        }
        canvasOut.hidden = true;
        imgOut.hidden = false;
        imgOut.classList.add("pg-contain");
        imgOut.src = j.png;
        if (j.quota && window.PGModels.setQuota) window.PGModels.setQuota({ used: j.quota.used, limit: j.quota.limit, remaining: Math.max(0, j.quota.limit - j.quota.used) });
        status.textContent = j.ms + " ms on the server";
      } catch (e) {
        status.classList.add("pg-error");
        status.textContent = e.message || String(e);
      }
      serverBusy = false;
      runBtn.disabled = false;
    }

    const actions = el("div", "pg-actions");
    if (op.mode === "frames") {
      runBtn = el("button", "pg-btn", { type: "button" });
      runBtn.innerHTML = '<i class="fas fa-play" aria-hidden="true"></i> Run';
      runBtn.addEventListener("click", runServerOp);
      const group = el("div", "pg-run-group");
      if (window.PGModels && window.PGModels.targetSwitch) group.appendChild(window.PGModels.targetSwitch({ browser: false, server: true, value: "server" }).el);   // no graph runs here: the server is the only target
      group.appendChild(runBtn);
      actions.appendChild(group);
    }
    if (op.stochastic && op.mode === "onnx") {
      const reroll = el("button", "pg-btn", { type: "button" });
      reroll.innerHTML = '<i class="fas fa-dice" aria-hidden="true"></i> Re-roll';
      reroll.addEventListener("click", schedule);
      actions.appendChild(reroll);
    }
    let download = null, onnxNote = null, customise = null;
    if (op.mode === "onnx") {   // lives at the top of the ONNX tab, above the snippet that runs it
      download = el("a", "pg-btn pg-btn-small", { download: "" });
      download.innerHTML = '<i class="fas fa-file-arrow-down" aria-hidden="true"></i> Download ONNX';
      customise = el("button", "pg-btn pg-btn-ghost pg-btn-small", { type: "button", title: "Export this operator with your own opset, shape and parameters, on kornia's server" });
      customise.innerHTML = '<i class="fas fa-sliders" aria-hidden="true"></i> Customise…';
      customise.addEventListener("click", function () { if (window.PGModels && window.PGModels.exportDialog) window.PGModels.exportDialog(op, registry, state.params, ROOT); });
      onnxNote = el("span", "pg-onnx-note");
    }
    const status = el("span", "pg-status");
    status.textContent = op.mode === "frames" ? "" : "loading runtime…";
    actions.appendChild(status);
    const note = el("p", "pg-note");
    const controlsCell = el("div", "pg-cell-controls pg-controls-cell", { id: "parameters" });
    controlsCell.appendChild(op.mode === "frames" ? liveControls : controls);
    controlsCell.appendChild(note);
    grid.appendChild(controlsCell);
    bar.appendChild(actions);
    // the facts about the graph, as a table under the code (the same block the model pages have)
    const details = el("div", "pg-details pg-model-card pg-cell-details", { id: "details" });
    const detailsTable = el("table");
    details.appendChild(detailsTable);
    const links = el("div", "pg-details-links");
    links.innerHTML = '<a id="pg-link-docs" href="' + (op.doc_url || "https://kornia.readthedocs.io") + '" target="_blank" rel="noopener"><i class="fas fa-book" aria-hidden="true"></i> API reference</a>'
      + ''
      + '<a id="pg-link-issue" href="https://github.com/kornia/kornia.github.io/issues/new?title=' + encodeURIComponent("playground: " + op.id) + '" target="_blank" rel="noopener"><i class="fas fa-bug" aria-hidden="true"></i> Report an issue</a>';
    details.appendChild(links);
    function setDetails(rows) {
      detailsTable.innerHTML = "";
      rows.filter(Boolean).forEach(function (r) {
        const tr = el("tr"); const th = el("th"); th.textContent = r[0]; const td = el("td"); td.textContent = r[1];
        tr.appendChild(th); tr.appendChild(td); detailsTable.appendChild(tr);
      });
    }

    // ---- code tabs: Python / Rust
    const codeBox = el("div", "pg-code", { id: "code" });
    const tabs = el("div", "pg-tabs", { role: "tablist" });
    const panes = {};
    const pres = {};
    const langs = op.rust ? ["python", "rust"] : ["python"]; // no Rust tab when kornia-rs has no counterpart
    if (op.mode === "onnx") langs.push("onnx");                // how to run the downloaded graph
    const active = langs.indexOf(preferredLang) === -1 ? "python" : preferredLang;
    langs.forEach(function (lang) {
      const isActive = lang === active;
      const tab = el("button", "pg-tab" + (isActive ? " pg-tab-active" : ""), { type: "button", role: "tab", "data-lang": lang, "aria-selected": isActive ? "true" : "false" });
      tab.textContent = lang === "python" ? "Python" : lang === "rust" ? "Rust" : "ONNX";
      tab.addEventListener("click", function () {
        tabs.querySelectorAll(".pg-tab").forEach(function (t) { t.classList.remove("pg-tab-active"); t.setAttribute("aria-selected", "false"); });
        tab.classList.add("pg-tab-active");
        tab.setAttribute("aria-selected", "true");
        for (const key in panes) panes[key].hidden = key !== lang;
      });
      tabs.appendChild(tab);
      const pane = el("div", "pg-pane", { role: "tabpanel" });
      pane.hidden = lang !== active;
      if (lang === "onnx" && download) {   // the file itself, then the few lines that run it
        const head = el("div", "pg-onnx-head");
        head.appendChild(download);
        head.appendChild(customise);
        head.appendChild(onnxNote);
        pane.appendChild(head);
      }
      const pre = el("pre");
      const code = el("code", "language-" + (lang === "onnx" ? "python" : lang));
      pre.appendChild(code);
      pane.appendChild(pre);
      panes[lang] = pane;
      pres[lang] = code;
    });
    const copy = el("button", "pg-copy", { type: "button" });
    copy.textContent = "copy";
    copy.addEventListener("click", function () {
      const visible = langs.map(function (l) { return pres[l]; }).find(function (c) { return !c.parentElement.parentElement.hidden; }) || pres.python;
      navigator.clipboard.writeText(visible.textContent).then(function () {
        copy.textContent = "copied";
        setTimeout(function () { copy.textContent = "copy"; }, 1200);
      });
    });
    tabs.appendChild(copy);
    codeBox.appendChild(tabs);
    langs.forEach(function (lang) { codeBox.appendChild(panes[lang]); });
    grid.appendChild(codeBox);
    grid.appendChild(details);
    container.appendChild(root);

    // ---- behaviour
    let inputTensor = null;
    let guideTensor = null;
    let timer = null;
    let running = false;
    let pending = false;
    let video = null;      // the <video> element, created on first use
    let videoLoop = false; // true while frames are being pumped through the graph

    function stopVideo() {
      videoLoop = false;
      restartPass();   // the kept frames go with the clip
      if (video) video.pause();
    }

    function startVideo(v) {
      if (!video) {
        video = el("video", "", { muted: "", loop: "", playsinline: "", preload: "auto" });
        video.muted = true;
        video.loop = true;
        video.hidden = true;
        root.appendChild(video);
      }
      if (video.dataset.id !== v.id) {
        video.innerHTML = "";
        [["webm", "video/webm"], ["mp4", "video/mp4"]].forEach(function (pair) {
          if (v[pair[0]]) video.appendChild(el("source", "", { src: ROOT +v[pair[0]], type: pair[1] }));
        });
        video.dataset.id = v.id;
        video.load();
      }
      videoLoop = true;
      restartPass();
      const p = video.play();
      if (p && p.catch) p.catch(function () { /* autoplay refused: the poster frame still shows */ });
      pump();
    }

    // one frame per completed graph run: the clip plays as fast as the graph allows

    // one computed pass through the clip, kept in memory as output frames (capped), then replayed in step with the
    // looping clip so nothing computes for hours. Dropped when the clip stops, a parameter changes, or the
    // page section changes. Keyed by playback time, at most MAX_KEPT frames.
    const MAX_KEPT = 240;
    let pass = { t: -1, seen: 0, done: false, frames: [] };
    function restartPass() { pass = { t: -1, seen: 0, done: false, frames: [] }; }
    function keptNear(t) {
      const f = pass.frames; if (!f.length) return null;
      let lo = 0, hi = f.length - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (f[mid].t < t) lo = mid + 1; else hi = mid; }
      if (lo > 0 && Math.abs(f[lo - 1].t - t) < Math.abs(f[lo].t - t)) lo--;
      return f[lo];
    }
    function pump() {
      if (!videoLoop || !root.isConnected) { videoLoop = false; restartPass(); return; }
      if (video.readyState >= 2 && typeof ort !== "undefined") {
        const t = video.currentTime;
        if (pass.t >= 0) pass.seen += Math.max(0, t - pass.t);
        if (!pass.done && pass.t >= 0 && t < pass.t - 0.5 && pass.seen >= (video.duration || 4) * 0.9 && pass.frames.length > 1) pass.done = true;
        pass.t = t;
        canvasIn.getContext("2d").drawImage(video, 0, 0, size, size);
        if (pass.done) {   // replay the computed pass in step with the clip
          const kept = keptNear(t);
          if (kept) { canvasOut.getContext("2d").putImageData(kept.img, 0, 0); status.textContent = "computed once (" + pass.frames.length + " frames) · replaying with the clip"; }
          window.requestAnimationFrame(pump);
          return;
        }
        inputTensor = canvasToTensor(canvasIn, size);
        execute().then(function () {
          if (!pass.done && canvasOut.width && canvasOut.height) {
            // keep at most MAX_KEPT frames across the clip: thin out when the pass produces more
            if (pass.frames.length < MAX_KEPT || (pass.frames.length && t - pass.frames[pass.frames.length - 1].t >= (video.duration || 4) / MAX_KEPT)) {
              if (pass.frames.length >= MAX_KEPT) pass.frames.splice(0, 1);
              pass.frames.push({ t: t, img: canvasOut.getContext("2d").getImageData(0, 0, canvasOut.width, canvasOut.height) });
            }
          }
          window.requestAnimationFrame(pump);
        });
      } else {
        window.requestAnimationFrame(pump);
      }
    }

    document.addEventListener("visibilitychange", function () {
      if (!video || !videoLoop) return;
      if (document.hidden) video.pause(); else video.play().catch(function () {});
    });

    function graphKey() {
      if (!op.select_order || !op.select_order.length) return "default";
      return op.select_order.map(function (name) {
        const p = op.params.find(function (q) { return q.name === name; });
        return fmt(state.params[name], p);
      }).join("|");
    }

    function renderNote() {
      note.textContent = "";
      if (op.mode === "frames" && onServer) {
        setDetails([["Preview", "pre-rendered frames of the samples along " + (op.frames.param || "the defaults") + (op.frames.note ? "; " + op.frames.note : "")],
                    ["Run on server", "kornia " + registry.kornia + " with every parameter live; your own image is resized to 512 px on the longer side"]]);
      } else if (op.mode === "frames") {
        note.textContent = op.frames.note ? op.frames.note + "." : "";
        setDetails([["Rendering", "frames rendered offline with kornia " + registry.kornia + "; this operator cannot export with live parameters yet"],
                    op.frames.param ? ["Frames", op.frames.values.length + " along " + op.frames.param] : null]);
      } else {
        const key = graphKey();
        const kb = op.graph_kb ? op.graph_kb[key] : "?";
        setDetails([["ONNX graph", kb + " KB, opset " + registry.onnx_opset],
                    ["Input", op.dynamic ? "any batch and image size" : "fixed 1×3×" + size + "×" + size],
                    ["Exported with", "kornia " + registry.kornia + " / torch " + registry.torch.split("+")[0]],
                    op.sampling_folded ? ["Randomness", "the random parameters were drawn once at export, so the graph is deterministic"] : null]);
      }
      note.hidden = !note.textContent;
      if (op.mode !== "onnx") return;
      const key = graphKey();
      if (download) {
        download.href = ROOT +op.graphs[key];
        download.setAttribute("download", op.slug + (key === "default" ? "" : "-" + key.replace(/\|/g, "_")) + ".onnx");
        onnxNote.textContent = (op.graph_kb ? op.graph_kb[key] + " KB · " : "") + "opset " + registry.onnx_opset + " · " + (op.dynamic ? "any image size" : size + "×" + size + " only") + " · runs with onnxruntime, onnxruntime-web and the ort crate";
      }
    }

    function setCode(code, text) {
      code.textContent = text;
      if (typeof hljs !== "undefined") {
        code.removeAttribute("data-highlighted"); // re-highlight after every slider move
        hljs.highlightElement(code);
      }
    }

    function onnxSnippet() {
      const key = graphKey();
      const file = op.slug + (key === "default" ? "" : "-" + key.replace(/\|/g, "_")) + ".onnx";
      const feeds = ['"' + op.inputs[0] + '": img'];
      let k = 1;
      if (op.guidance) feeds.push('"' + op.inputs[k++] + '": guide');
      op.params.filter(function (p) { return p.kind === "live"; }).forEach(function (p) { feeds.push('"' + op.inputs[k++] + '": np.array([' + fmt(state.params[p.name], p) + '], np.float32)'); });
      return [
        "# pip install onnxruntime numpy",
        "import numpy as np",
        "import onnxruntime as ort",
        "",
        "# the file from the Download button above: opset " + registry.onnx_opset + ", " + (op.dynamic ? "any batch and image size" : "a fixed 1×3×" + size + "×" + size + " input"),
        'sess = ort.InferenceSession("' + file + '")',
        "img = np.random.rand(1, 3, " + (op.dynamic ? "480, 640" : size + ", " + size) + ").astype(np.float32)  # (1, 3, H, W) float in [0, 1]",
        (op.guidance ? "guide = np.random.rand(1, 3, " + size + ", " + size + ").astype(np.float32)\n" : "") + "out = sess.run(None, {" + feeds.join(", ") + "})[0]",
        "",
        "# the same file runs in the browser with onnxruntime-web (as this page does) and in Rust with the ort crate",
      ].join("\n");
    }
    function renderCode() {
      setCode(pres.python, pythonSnippet(op, state, registry));
      if (pres.rust) setCode(pres.rust, rustSnippet(op, state, registry));
      if (pres.onnx) setCode(pres.onnx, onnxSnippet());
    }

    function loadInput() {
      const haveOrt = typeof ort !== "undefined";
      const im = registry.images.find(function (i) { return i.id === state.image; });
      if (!im && state.image === "upload" && uploaded) {
        // the visitor's own image: cover-fitted for the fixed-size graphs, letterboxed when the server keeps its shape
        const ctx = canvasIn.getContext("2d");
        const cover = op.mode === "onnx";
        const s = (cover ? Math.max : Math.min)(size / uploaded.width, size / uploaded.height), w = uploaded.width * s, h = uploaded.height * s;
        ctx.clearRect(0, 0, size, size);
        if (cover) { ctx.fillStyle = "#000"; ctx.fillRect(0, 0, size, size); }
        ctx.drawImage(uploaded, (size - w) / 2, (size - h) / 2, w, h);
        inputTensor = op.mode === "onnx" && haveOrt ? canvasToTensor(canvasIn, size) : null;
        return Promise.resolve();
      }
      if (!im) return Promise.resolve(); // a clip is selected: frames come from the video loop
      const steps = [loadImage(ROOT +im.file).then(function (img) {
        canvasIn.getContext("2d").drawImage(img, 0, 0, size, size);
        inputTensor = op.mode === "onnx" && haveOrt ? canvasToTensor(canvasIn, size) : null;
      })];
      if (op.guidance && op.mode === "onnx" && haveOrt && !guideTensor) {
        steps.push(loadImage(ROOT +op.guidance.file).then(function (img) {
          const scratch = el("canvas", "", { width: size, height: size });
          scratch.getContext("2d").drawImage(img, 0, 0, size, size);
          guideTensor = canvasToTensor(scratch, size);
        }));
      }
      return Promise.all(steps);
    }

    function schedule() {
      renderCode();
      renderNote();
      restartPass();
      clearTimeout(timer);
      timer = setTimeout(run, RUN_DELAY_MS);
    }

    function runFrames() {
      // nothing runs by itself on the server; the pre-rendered frames still preview the samples there
      if (!registry.images.some(function (i) { return i.id === state.image; })) { if (onServer) { clearOutput(); status.textContent = "press Run"; } return; }
      const idx = op.frames.param === null ? 0 : nearest(op.frames.values, state.params[op.frames.param]);
      canvasOut.hidden = true;
      imgOut.hidden = false;
      imgOut.classList.remove("pg-contain");
      imgOut.src = ROOT +op.frames.dir + "/" + state.image + "/" + String(idx).padStart(2, "0") + ".webp";
      status.textContent = onServer ? "preview from the pre-rendered frames · Run on server computes it exactly" : "frame " + (idx + 1) + " of " + op.frames.values.length;
    }

    function run() {
      if (op.mode === "frames") { runFrames(); return; }
      if (typeof ort === "undefined") {
        status.classList.add("pg-error");
        status.textContent = "onnxruntime-web did not load (blocked CDN?); the live demo is unavailable, the download and code still work.";
        return;
      }
      if (running) { pending = true; return; }
      if (!inputTensor) return;
      running = true;
      execute().then(function () {
        running = false;
        if (pending) { pending = false; run(); }
      });
    }

    // one graph run on the current input tensor; resolves when the output is drawn
    function execute() {
      status.classList.remove("pg-error");
      const feeds = {};
      feeds[op.inputs[0]] = inputTensor;
      let next = 1;
      if (op.guidance) feeds[op.inputs[next++]] = guideTensor;
      op.params.filter(function (p) { return p.kind === "live"; }).forEach(function (p) {
        feeds[op.inputs[next++]] = new ort.Tensor("float32", new Float32Array([state.params[p.name]]), [1]);
      });
      const t0 = performance.now();
      return getSession(ROOT +op.graphs[graphKey()])
        .then(function (session) { return session.run(feeds); })
        .then(function (results) {
          const out = results[Object.keys(results)[0]];
          tensorToCanvas(out, canvasOut, op.output.display);
          const ms = performance.now() - t0;
          status.textContent = ms.toFixed(0) + " ms on your machine" + (state.kind === "video" && ms > 0 ? " (" + (1000 / ms).toFixed(0) + " fps)" : "");
        })
        .catch(function (err) {
          status.classList.add("pg-error");
          status.textContent = "could not run the graph: " + (err.message || err);
          videoLoop = false;
        });
    }

    renderCode();
    renderNote();
    loadInput().then(function () {
      if (op.mode === "frames") { runFrames(); return; }
      status.textContent = "loading graph…";
      run();
    }).catch(function (err) {
      status.classList.add("pg-error");
      status.textContent = err.message;
    });
  }

  function buildUnsupported(op, container) {
    const root = el("article", "pg-op pg-op-unsupported", { id: op.slug });
    const head = el("div", "pg-op-head");
    const title = el("h2");
    title.textContent = op.name;
    const code = el("code");
    code.textContent = op.id;
    head.appendChild(title);
    head.appendChild(code);
    root.appendChild(head);
    const p = el("p", "pg-summary");
    p.textContent = "No demo yet: " + (op.reason || "no demo yet") + ". ";
    const docLink = el("a", "", { href: op.doc_url, target: "_blank", rel: "noopener" });
    docLink.textContent = "API reference";
    p.appendChild(docLink);
    root.appendChild(p);
    container.appendChild(root);
  }

  // ------------------------------------------------------------------ sidebar + routing

  function buildSidebar(registry, select) {
    const container = document.getElementById("pg-catalog");
    const rows = [];
    registry.packages.forEach(function (pkg) {
      const rank = { live: 0, frames: 0, unsupported: 1 };
      const ops = registry.ops.filter(function (o) { return o.package === pkg.id; }).sort(function (a, b) {
        return (rank[a.status] - rank[b.status]) || a.name.localeCompare(b.name);
      });
      if (!ops.length) return;
      const section = el("section", "pg-pkg");
      const h = el("h2");
      h.textContent = pkg.title;
      const n = el("span", "pg-pkg-count");
      n.textContent = ops.filter(function (o) { return o.status !== "unsupported"; }).length + "/" + ops.length;
      h.appendChild(n);
      section.appendChild(h);
      const list = el("ul", "pg-list");
      ops.forEach(function (op) {
        const li = el("li", "pg-row pg-row-" + op.status);
        const a = el("a", "pg-row-name", { href: ROOT +"ops/" + op.slug + "/", "data-op": op.id, title: op.reason || op.summary || op.name });
        a.textContent = op.name;
        if (op.mode === "frames") {
          const tag = el("span", "pg-tag pg-tag-small");
          tag.textContent = "frames";
          a.appendChild(tag);
        }
        a.addEventListener("click", function (ev) {
          ev.preventDefault();
          select(op, true);
        });
        li.appendChild(a);
        list.appendChild(li);
        rows.push({ li: li, section: section, op: op, a: a, text: (op.name + " " + op.id + " " + (op.summary || "") + " " + (op.reason || "")).toLowerCase() });
      });
      section.appendChild(list);
      container.appendChild(section);
    });

    const search = document.getElementById("pg-search");
    const radios = document.querySelectorAll("#pg-filters input");
    function apply() {
      const q = search.value.trim().toLowerCase();
      let want = "all";
      radios.forEach(function (r) { if (r.checked) want = r.value; });
      // the Rust filter also makes Rust the default code tab, "All" goes back to Python
      const lang = want === "rust" ? "rust" : "python";
      if (lang !== preferredLang) {
        preferredLang = lang;
        const tab = document.querySelector('.pg-tab[data-lang="' + lang + '"]');
        if (tab) tab.click();
      }
      const visibleSections = new Set();
      rows.forEach(function (r) {
        // "rust": only operators kornia-rs has (the ones with a Rust tab)
        // "rust": only operators kornia-rs has; "3d": the volume operators; "all": everything
        const passes = want === "all" || (want === "rust" ? !!r.op.rust : want === "3d" ? r.op.input_kind === "volume" || /3d$/i.test(r.op.name) : r.op.status !== "unsupported");
        const ok = (!q || r.text.indexOf(q) !== -1) && passes;
        r.li.hidden = !ok;
        if (ok) visibleSections.add(r.section);
      });
      container.querySelectorAll(".pg-pkg").forEach(function (s) { s.hidden = !visibleSections.has(s); });
    }
    search.addEventListener("input", apply);
    radios.forEach(function (r) { r.addEventListener("change", apply); });
    apply();
    return {
      highlight: function (op) {
        rows.forEach(function (r) {
          const on = r.op.id === op.id;
          r.a.classList.toggle("pg-row-current", on);
          if (on) r.a.setAttribute("aria-current", "page"); else r.a.removeAttribute("aria-current");
        });
        // scroll the list only, never the page (scrollIntoView would move the whole page on narrow screens)
        const current = container.querySelector(".pg-row-current");
        if (current) {
          const top = current.getBoundingClientRect().top - container.getBoundingClientRect().top + container.scrollTop;
          if (top < container.scrollTop || top > container.scrollTop + container.clientHeight - 40) {
            container.scrollTop = Math.max(0, top - container.clientHeight / 2);
          }
        }
      },
    };
  }

  // Site header hamburger (the homepage wires the same markup in its inline script).
  function initNavToggle() {
    document.querySelectorAll(".nav-links a[data-section]").forEach(function (a) {
      const on = location.pathname.indexOf("/" + a.dataset.section + "/") !== -1;
      a.classList.toggle("active", on);
      if (on) a.setAttribute("aria-current", "page");
    });
    const toggle = document.querySelector(".nav-toggle");
    const links = document.getElementById("primary-nav-links");
    if (!toggle || !links) return;
    toggle.addEventListener("click", function () {
      const open = links.classList.toggle("is-open");
      toggle.classList.toggle("is-active", open);
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
    links.addEventListener("click", function (e) {
      const a = e.target.closest("a");
      if (!a) return;
      if (a.getAttribute("aria-haspopup")) {
        // a dropdown trigger: Docs has no page, so it only toggles; on a phone the first tap opens, the second follows
        const dd = a.closest(".nav-dropdown");
        const mobile = window.matchMedia("(max-width: 1024px)").matches;
        if (a.getAttribute("href") === "#" || (mobile && !dd.classList.contains("is-open"))) {
          e.preventDefault();
          const open = dd.classList.toggle("is-open");
          a.setAttribute("aria-expanded", open ? "true" : "false");
        }
        return;
      }
      links.classList.remove("is-open");
      toggle.classList.remove("is-active");
      toggle.setAttribute("aria-expanded", "false");
    });
    document.addEventListener("click", function (e) {
      if (e.target.closest(".nav-dropdown")) return;
      links.querySelectorAll(".nav-dropdown.is-open").forEach(function (dd) { dd.classList.remove("is-open"); });
    });
  }

  function init(registry) {
    initNavToggle();
    if (typeof ort !== "undefined") {
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@" + registry.onnxruntime_web + "/dist/";
      ort.env.wasm.numThreads = 1; // GitHub Pages does not send the COOP/COEP headers threads need
    }
    const detail = document.getElementById("pg-ops");
    const meta = document.getElementById("pg-meta");
    if (meta) meta.remove();

    if (document.body.dataset.view === "pipelines") {
      // the sidebar lists pipelines here (pipeline.js draws it); the other two tabs open their own pages
      document.body.dataset.mode = "pipes";
      document.querySelectorAll("#pg-mode button").forEach(function (b) {
        if (b.dataset.mode === "ops") b.addEventListener("click", function () { window.location.href = ROOT; });
        if (b.dataset.mode === "models") b.addEventListener("click", function () {
          fetch(ROOT + "models/index.json").then(function (r) { return r.json(); }).then(function (idx) {
            if (idx.models.length) window.location.href = ROOT + "models/" + idx.models[0].slug + "/";
          }).catch(function () {});
        });
      });
      if (window.PGPipelines) window.PGPipelines.init(registry, ROOT);
      return;
    }

    let sidebar = null;
    let modelsIndex = null;
    const modeButtons = document.querySelectorAll("#pg-mode button");
    function setMode(mode) {
      modeButtons.forEach(function (b) { b.setAttribute("aria-selected", b.dataset.mode === mode ? "true" : "false"); });
      document.body.dataset.mode = mode;
      const filters = document.getElementById("pg-filters");
      if (filters) filters.hidden = mode === "models";
      document.getElementById("pg-catalog").hidden = mode === "models";
      const list = document.getElementById("pg-models-list");
      if (list) list.hidden = mode !== "models";
      const search = document.getElementById("pg-search");
      if (search) search.placeholder = mode === "models" ? "Search models" : "Search operators";
    }
    modeButtons.forEach(function (b) { b.addEventListener("click", function () {
      if (b.dataset.mode === "pipes") { window.location.href = b.dataset.href || (ROOT + "pipelines/"); return; }   // the editor lives on its own page
      setMode(b.dataset.mode);
    }); });

    function selectModel(model, push) {
      setMode("models");
      detail.innerHTML = "";
      window.PGModels.render(model, registry, modelsIndex, detail, ROOT);
      document.title = model.name + " - Kornia Playground";
      const docs = document.getElementById("pg-link-docs");
      if (docs) docs.href = model.doc_url;
      const issue = document.getElementById("pg-link-issue");
      if (issue) issue.href = "https://github.com/kornia/kornia.github.io/issues/new?title=" + encodeURIComponent("playground model: " + model.id);
      if (sidebar) sidebar.highlight({ id: null });
      window.PGModels.highlight(model);
      if (push && window.history && window.history.pushState) {
        window.history.pushState({ model: model.id }, "", ROOT + "models/" + model.slug + "/");
      }
      if (push && window.innerWidth < 900) detail.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function select(op, push) {
      setMode("ops");
      detail.innerHTML = "";
      if (op.status === "unsupported") buildUnsupported(op, detail);
      else buildWidget(op, registry, detail);
      document.title = op.name + " - Kornia Playground";
      const docs = document.getElementById("pg-link-docs");
      if (docs) docs.href = op.doc_url;
      const issue = document.getElementById("pg-link-issue");
      if (issue) issue.href = "https://github.com/kornia/kornia.github.io/issues/new?title=" + encodeURIComponent("playground: " + op.id);
      if (sidebar) sidebar.highlight(op);
      if (push && window.history && window.history.pushState) {
        window.history.pushState({ op: op.id }, "", ROOT +"ops/" + op.slug + "/");
      }
      if (push && window.innerWidth < 900) detail.scrollIntoView({ behavior: "smooth", block: "start" });
      if (typeof ort === "undefined") {
        detail.querySelectorAll(".pg-status").forEach(function (s) {
          s.classList.add("pg-error");
          s.textContent = "onnxruntime-web did not load; live demos are unavailable.";
        });
      }
    }
    sidebar = buildSidebar(registry, select);

    const wanted = document.body.dataset.op || new URLSearchParams(window.location.search).get("op");
    const wantedModel = document.body.dataset.model;
    const initial = registry.ops.find(function (o) { return o.id === wanted || o.slug === wanted; }) ||
      registry.ops.find(function (o) { return o.id === "kornia.filters.gaussian_blur2d"; }) ||
      registry.ops.find(function (o) { return o.status === "live"; });
    if (initial && !wantedModel) select(initial, false);

    fetch(ROOT + "models/index.json", { cache: "no-cache" }).then(function (r) { return r.ok ? r.json() : null; }).then(function (idx) {
      if (!idx || !window.PGModels) return;
      modelsIndex = idx;
      window.PGModels.buildList(idx, document.getElementById("pg-side"), selectModel, ROOT);
      if (wantedModel) {
        const m = idx.models.find(function (x) { return x.id === wantedModel || x.slug === wantedModel; });
        if (m) selectModel(m, false); else if (initial) select(initial, false);
      }
    }).catch(function () { if (wantedModel && initial) select(initial, false); });

    window.addEventListener("popstate", function (ev) {
      const id = ev.state && ev.state.op;
      const mid = ev.state && ev.state.model;
      const op = id ? registry.ops.find(function (o) { return o.id === id; }) : null;
      const model = mid && modelsIndex ? modelsIndex.models.find(function (m) { return m.id === mid; }) : null;
      if (model) selectModel(model, false); else if (op) select(op, false); else if (initial) select(initial, false);
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    fetch(ROOT +"registry.json")
      .then(function (r) {
        if (!r.ok) throw new Error("registry.json: HTTP " + r.status);
        return r.json();
      })
      .then(init)
      .catch(function (err) {
        const p = el("p", "pg-status pg-error");
        p.textContent = "The playground registry could not be loaded (" + err.message + ").";
        document.getElementById("pg-ops").appendChild(p);
      });
  });
})();
