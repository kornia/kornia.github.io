// Homepage demo: three things from the stack running on short clips, in the browser, with no button to press.
//   Faces   YuNet face detection (playground/models/yunet.onnx, 350 kB) on the astronaut clip
//   Augment an affine augmentation whose parameters are drawn once per pass of the clip and held fixed
//           (playground/home/affine_256.onnx: the warp with its parameters as graph inputs)
//   Edges   Sobel from the operator catalog
// Everything is small, so all three graphs are fetched as soon as the runtime is ready.
(function () {
  "use strict";

  const ROOT = "playground/";
  const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.min.js";
  const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/";
  const TABS = [
    { key: "faces", label: "Faces", clip: "astronaut", caption: "YuNet face detection" },
    { key: "augment", label: "Augment", clip: "boats", caption: "RandomAffine, one draw per pass" },
    { key: "edges", label: "Edges", clip: "traffic", caption: "Sobel" },
  ];
  // RandomAffine's default ranges: degrees ±15, translate 10 %, scale 0.9–1.1, shear ±10°
  function drawAffine(size) {
    const u = function (a, b) { return a + Math.random() * (b - a); };
    return { angle: u(-15, 15), tx: u(-0.1, 0.1) * size, ty: u(-0.1, 0.1) * size, scale: u(0.9, 1.1), shear: u(-10, 10) };
  }

  const mount = document.getElementById("hero-demo");
  if (!mount) return;

  function el(tag, cls, attrs) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    for (const key in attrs || {}) node.setAttribute(key, attrs[key]);
    return node;
  }

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      if (typeof ort !== "undefined") { resolve(); return; }
      const s = el("script", "", { src: src });
      s.onload = resolve;
      s.onerror = function () { reject(new Error("onnxruntime-web did not load")); };
      document.head.appendChild(s);
    });
  }

  function canvasToTensor(canvas, size, scale) {
    const data = canvas.getContext("2d").getImageData(0, 0, size, size).data;
    const plane = size * size;
    const out = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      out[i] = data[4 * i] * scale;
      out[plane + i] = data[4 * i + 1] * scale;
      out[2 * plane + i] = data[4 * i + 2] * scale;
    }
    return new ort.Tensor("float32", out, [1, 3, size, size]);
  }

  function tensorToCanvas(tensor, canvas, normalize) {
    const dims = tensor.dims;
    const channels = dims[1], h = dims[2], w = dims[3];
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(w, h);
    const plane = h * w;
    const src = tensor.data;
    let lo = 0, scale = 1;
    if (normalize) {
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

  function init(registry, models) {
    const size = registry.size;
    const sobel = registry.ops.find(function (o) { return o.id === "kornia.filters.sobel" && o.mode === "onnx"; });
    const yunet = models && models.models.find(function (m) { return m.output.kind === "faces_yunet"; });
    if (!sobel || !yunet || !window.PGModels) { mount.hidden = true; return; }
    const graphs = {
      faces: ROOT + yunet.url,
      augment: ROOT + "home/affine_256.onnx",
      edges: ROOT + sobel.graphs[Object.keys(sobel.graphs)[0]],
    };

    // ---- markup
    const tabs = el("div", "hd-tabs", { role: "tablist" });
    const stage = el("div", "hd-stage");
    const figIn = el("figure", "hd-fig");
    const canvasIn = el("canvas", "", { width: size, height: size });
    const capIn = el("figcaption");
    capIn.textContent = "input";
    figIn.appendChild(canvasIn);
    figIn.appendChild(capIn);
    const arrow = el("div", "hd-arrow", { "aria-hidden": "true" });
    arrow.innerHTML = '<i class="fas fa-arrow-right"></i>';
    const figOut = el("figure", "hd-fig");
    const canvasOut = el("canvas", "", { width: size, height: size });
    const capOut = el("figcaption");
    capOut.textContent = "output";
    figOut.appendChild(canvasOut);
    figOut.appendChild(capOut);
    stage.appendChild(figIn);
    stage.appendChild(arrow);
    stage.appendChild(figOut);
    const controls = el("div", "hd-controls");
    const foot = el("div", "hd-foot");
    const status = el("span", "hd-status");
    status.textContent = "loading the runtime…";
    const link = el("a", "hd-link", { href: ROOT });
    foot.appendChild(status);
    foot.appendChild(link);
    mount.appendChild(tabs);
    mount.appendChild(stage);
    mount.appendChild(controls);
    mount.appendChild(foot);

    // ---- state
    let tab = TABS[0];
    let video = null;
    let ready = false;
    let running = false;
    let affine = drawAffine(size);
    let lastTime = 0;
    const sessions = {};

    function session(key) {
      if (!sessions[key]) sessions[key] = ort.InferenceSession.create(graphs[key], { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      return sessions[key];
    }

    function startClip(id) {
      if (video) { video.pause(); video.remove(); video = null; }
      const clip = (registry.videos || []).find(function (v) { return v.id === id; });
      if (!clip) return;
      video = el("video", "", { muted: "", loop: "", playsinline: "", preload: "auto" });
      video.muted = true;
      video.loop = true;
      video.hidden = true;
      [["webm", "video/webm"], ["mp4", "video/mp4"]].forEach(function (pair) {
        if (clip[pair[0]]) video.appendChild(el("source", "", { src: ROOT + clip[pair[0]], type: pair[1] }));
      });
      mount.appendChild(video);
      capIn.textContent = "input: " + clip.label;
      lastTime = 0;
      video.play().catch(function () { status.textContent = "press play in your browser to start the clip"; });
    }

    function renderControls() {
      controls.innerHTML = "";
      const cap = el("span", "hd-caption");
      cap.textContent = tab.caption;
      controls.appendChild(cap);
      if (tab.key === "augment") {
        const re = el("button", "hd-reroll", { type: "button" });
        re.innerHTML = '<i class="fas fa-dice" aria-hidden="true"></i> Re-roll';
        re.addEventListener("click", function () { affine = drawAffine(size); });
        controls.appendChild(re);
      }
      const targets = {
        faces: { text: "Open YuNet in the playground →", href: ROOT + "models/" + yunet.slug + "/" },
        augment: { text: "Open RandomAffine in the playground →", href: ROOT + "ops/augmentation-RandomAffine/" },
        edges: { text: "Open sobel in the playground →", href: ROOT + "ops/" + sobel.slug + "/" },
      };
      link.textContent = targets[tab.key].text;
      link.href = targets[tab.key].href;
    }

    function selectTab(next) {
      tab = next;
      tabs.querySelectorAll("button").forEach(function (b) {
        const on = b.dataset.key === tab.key;
        b.classList.toggle("hd-tab-active", on);
        b.setAttribute("aria-selected", on ? "true" : "false");
      });
      renderControls();
      startClip(tab.clip);
    }

    TABS.forEach(function (t, i) {
      const b = el("button", "hd-tab" + (i === 0 ? " hd-tab-active" : ""), { type: "button", role: "tab", "data-key": t.key, "aria-selected": i === 0 ? "true" : "false" });
      b.textContent = t.label;
      b.addEventListener("click", function () { selectTab(t); });
      tabs.appendChild(b);
    });

    // one frame through the current graph; the loop below calls it as fast as the graph allows
    function tick() {
      if (!ready || running || !video || video.readyState < 2) return Promise.resolve();
      running = true;
      // a new pass of the clip draws new augmentation parameters, held for the whole pass
      if (video.currentTime < lastTime - 0.5) affine = drawAffine(size);
      lastTime = video.currentTime;
      canvasIn.getContext("2d").drawImage(video, 0, 0, size, size);
      const key = tab.key;
      const feeds = {};
      if (key === "faces") feeds.image = canvasToTensor(canvasIn, size, 1);          // YuNet wants 0–255
      else feeds[key === "edges" ? sobel.inputs[0] : "image"] = canvasToTensor(canvasIn, size, 1 / 255);
      if (key === "augment") {
        ["angle", "tx", "ty", "scale", "shear"].forEach(function (name) { feeds[name] = new ort.Tensor("float32", new Float32Array([affine[name]]), [1]); });
      }
      const t0 = performance.now();
      return session(key).then(function (s) { return s.run(feeds); }).then(function (results) {
        if (key === "faces") {
          const faces = window.PGModels.decodeFaces(results, yunet.output, size, yunet.output.threshold || 0.5);
          canvasOut.width = size; canvasOut.height = size;
          const ctx = canvasOut.getContext("2d");
          ctx.drawImage(canvasIn, 0, 0);
          window.PGModels.paintFaces(ctx, faces);
        } else {
          tensorToCanvas(results[Object.keys(results)[0]], canvasOut, key === "edges");
        }
        status.textContent = "runs in your browser · " + (performance.now() - t0).toFixed(0) + " ms per frame";
      }).catch(function (err) {
        status.textContent = "live demo unavailable: " + (err.message || err);
        ready = false;
      }).then(function () { running = false; });
    }

    function loop() {
      if (!ready) return;
      tick().then(function () { window.requestAnimationFrame(loop); });
    }

    selectTab(TABS[0]);
    loadScript(ORT_URL).then(function () {
      ort.env.wasm.wasmPaths = ORT_WASM;
      ort.env.wasm.numThreads = 1;
      return Promise.all(Object.keys(graphs).map(session));   // all three are small: fetch them up front
    }).then(function () {
      ready = true;
      loop();
    }).catch(function (err) {
      status.textContent = err.message;
    });
  }

  function start() {
    Promise.all([
      fetch(ROOT + "registry.json").then(function (r) { return r.json(); }),
      fetch(ROOT + "models/index.json").then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; }),
    ]).then(function (pair) { init(pair[0], pair[1]); }).catch(function () { mount.hidden = true; });
  }

  if (document.readyState === "complete") start(); else window.addEventListener("load", start);
})();
