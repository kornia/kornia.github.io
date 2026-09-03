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
    const lines = [
      "import torch",
      "import kornia",
      "from kornia.io import load_image",
      "",
      "# the " + registry.size + " px sample shown in the playground (\"" + image.label + "\"):",
      "# https://kornia.org/playground/" + image.file,
      "img = load_image(\"" + image.id + ".png\")[None]  # (1, 3, " + registry.size + ", " + registry.size + ") float in [0, 1]",
      "",
    ];
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
    lines.push("// kornia-rs " + RUST_VERSION + ". The same " + registry.size + " px sample (\"" + image.label + "\"):");
    lines.push("// https://kornia.org/playground/" + image.file);
    lines.push("let img = read_image_any_rgb8(\"" + image.id + ".png\")?;          // Image<u8, 3>");
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

  function buildWidget(op, registry, container) {
    const size = registry.size;
    const state = { image: registry.images[0].id, kind: "image", params: {} };
    op.params.forEach(function (p) { state.params[p.name] = p.default; });

    const root = el("article", "pg-op", { id: op.slug });
    const head = el("div", "pg-op-head");
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
    thumbs.classList.add("pg-cell-thumbs");
    grid.appendChild(thumbs);  // second row, left: sample picker

    // ---- controls column
    const controls = el("div", "pg-controls", { id: "parameters" });
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

    const actions = el("div", "pg-actions");
    if (op.stochastic && op.mode === "onnx") {
      const reroll = el("button", "pg-btn", { type: "button" });
      reroll.innerHTML = '<i class="fas fa-dice" aria-hidden="true"></i> Re-roll';
      reroll.addEventListener("click", schedule);
      actions.appendChild(reroll);
    }
    let download = null;
    if (op.mode === "onnx") {
      download = el("a", "pg-btn pg-btn-ghost", { download: "" });
      download.innerHTML = '<i class="fas fa-download" aria-hidden="true"></i> Download ONNX';
      actions.appendChild(download);
    }
    const status = el("span", "pg-status");
    status.textContent = op.mode === "frames" ? "" : "loading runtime…";
    actions.appendChild(status);
    const note = el("p", "pg-note");
    controls.appendChild(note);
    controls.classList.add("pg-cell-controls");
    grid.appendChild(controls);
    actions.classList.add("pg-cell-actions");
    grid.appendChild(actions);  // second row, right: download / re-roll / status, level with the sample picker

    // ---- code tabs: Python / Rust
    const codeBox = el("div", "pg-code", { id: "code" });
    const tabs = el("div", "pg-tabs", { role: "tablist" });
    const panes = {};
    const pres = {};
    const langs = op.rust ? ["python", "rust"] : ["python"]; // no Rust tab when kornia-rs has no counterpart
    const active = langs.indexOf(preferredLang) === -1 ? "python" : preferredLang;
    langs.forEach(function (lang) {
      const isActive = lang === active;
      const tab = el("button", "pg-tab" + (isActive ? " pg-tab-active" : ""), { type: "button", role: "tab", "data-lang": lang, "aria-selected": isActive ? "true" : "false" });
      tab.textContent = lang === "python" ? "Python" : "Rust";
      tab.addEventListener("click", function () {
        tabs.querySelectorAll(".pg-tab").forEach(function (t) { t.classList.remove("pg-tab-active"); t.setAttribute("aria-selected", "false"); });
        tab.classList.add("pg-tab-active");
        tab.setAttribute("aria-selected", "true");
        for (const key in panes) panes[key].hidden = key !== lang;
      });
      tabs.appendChild(tab);
      const pane = el("div", "pg-pane", { role: "tabpanel" });
      pane.hidden = lang !== active;
      const pre = el("pre");
      const code = el("code", "language-" + lang);
      pre.appendChild(code);
      pane.appendChild(pre);
      panes[lang] = pane;
      pres[lang] = code;
    });
    const copy = el("button", "pg-copy", { type: "button" });
    copy.textContent = "copy";
    copy.addEventListener("click", function () {
      const visible = panes.rust && panes.python.hidden ? pres.rust : pres.python;
      navigator.clipboard.writeText(visible.textContent).then(function () {
        copy.textContent = "copied";
        setTimeout(function () { copy.textContent = "copy"; }, 1200);
      });
    });
    tabs.appendChild(copy);
    codeBox.appendChild(tabs);
    langs.forEach(function (lang) { codeBox.appendChild(panes[lang]); });
    grid.appendChild(codeBox);
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
      const p = video.play();
      if (p && p.catch) p.catch(function () { /* autoplay refused: the poster frame still shows */ });
      pump();
    }

    // one frame per completed graph run: the clip plays as fast as the graph allows
    function pump() {
      if (!videoLoop || !root.isConnected) { videoLoop = false; return; }
      if (video.readyState >= 2 && typeof ort !== "undefined") {
        canvasIn.getContext("2d").drawImage(video, 0, 0, size, size);
        inputTensor = canvasToTensor(canvasIn, size);
        execute().then(function () { window.requestAnimationFrame(pump); });
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
      if (op.mode === "frames") {
        const fixed = op.frames.note ? op.frames.note + ". " : "";
        note.textContent = fixed + "Frames rendered offline with kornia " + registry.kornia + "; this operator cannot export with live parameters yet.";
        return;
      }
      const key = graphKey();
      const kb = op.graph_kb ? op.graph_kb[key] : "?";
      const shape = op.dynamic ? "any batch and image size" : "fixed 1×3×" + size + "×" + size + " input";
      note.textContent = "ONNX graph: " + kb + " KB, opset " + registry.onnx_opset + ", " + shape + ", exported from kornia " +
        registry.kornia + " / torch " + registry.torch.split("+")[0] + "." +
        (op.sampling_folded ? " The random parameters were drawn once at export, so this graph is deterministic." : "");
      if (download) {
        download.href = ROOT +op.graphs[key];
        download.setAttribute("download", op.slug + (key === "default" ? "" : "-" + key.replace(/\|/g, "_")) + ".onnx");
      }
    }

    function setCode(code, text) {
      code.textContent = text;
      if (typeof hljs !== "undefined") {
        code.removeAttribute("data-highlighted"); // re-highlight after every slider move
        hljs.highlightElement(code);
      }
    }

    function renderCode() {
      setCode(pres.python, pythonSnippet(op, state, registry));
      if (pres.rust) setCode(pres.rust, rustSnippet(op, state, registry));
    }

    function loadInput() {
      const im = registry.images.find(function (i) { return i.id === state.image; });
      if (!im) return Promise.resolve(); // a clip is selected: frames come from the video loop
      const haveOrt = typeof ort !== "undefined";
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
      clearTimeout(timer);
      timer = setTimeout(run, RUN_DELAY_MS);
    }

    function runFrames() {
      const idx = op.frames.param === null ? 0 : nearest(op.frames.values, state.params[op.frames.param]);
      canvasOut.hidden = true;
      imgOut.hidden = false;
      imgOut.src = ROOT +op.frames.dir + "/" + state.image + "/" + String(idx).padStart(2, "0") + ".webp";
      status.textContent = "frame " + (idx + 1) + " of " + op.frames.values.length;
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
        const passes = want === "all" || (want === "rust" ? !!r.op.rust : r.op.status !== "unsupported");
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
    const toggle = document.querySelector(".nav-toggle");
    const links = document.getElementById("primary-nav-links");
    if (!toggle || !links) return;
    toggle.addEventListener("click", function () {
      const open = links.classList.toggle("is-open");
      toggle.classList.toggle("is-active", open);
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
    links.addEventListener("click", function (e) {
      if (e.target.closest("a")) {
        links.classList.remove("is-open");
        toggle.classList.remove("is-active");
        toggle.setAttribute("aria-expanded", "false");
      }
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
    const live = registry.ops.filter(function (o) { return o.status !== "unsupported"; }).length;
    meta.textContent = live + " of " + registry.ops.length + " operators have a demo · kornia " + registry.kornia + " · " + registry.generated_at;

    if (document.body.dataset.view === "pipelines") {
      // the sidebar only navigates here: an operator opens its page, the Models tab opens the first model
      const modelsTab = document.querySelector('#pg-mode [data-mode="models"]');
      if (modelsTab) modelsTab.addEventListener("click", function () {
        fetch(ROOT + "models/index.json").then(function (r) { return r.json(); }).then(function (idx) {
          if (idx.models.length) window.location.href = ROOT + "models/" + idx.models[0].slug + "/";
        }).catch(function () {});
      });
      buildSidebar(registry, function (op) { window.location.href = ROOT + "ops/" + op.slug + "/"; });
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
      if (meta) meta.hidden = mode === "models";
      const mmeta = document.getElementById("pg-models-meta");
      if (mmeta) mmeta.hidden = mode !== "models";
    }
    modeButtons.forEach(function (b) { b.addEventListener("click", function () { setMode(b.dataset.mode); }); });

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
      const mmeta = el("p", "pg-meta", { id: "pg-models-meta" });
      mmeta.textContent = idx.models.length + " models run in your browser with onnxruntime-web · the large ones download from the Hugging Face hub on demand";
      mmeta.hidden = document.body.dataset.mode !== "models";
      meta.parentNode.insertBefore(mmeta, meta.nextSibling);
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
