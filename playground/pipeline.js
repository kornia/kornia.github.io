// kornia.org/playground/pipelines: build a chain of operators, run it in the browser, export it.
//
// A pipeline is a list of steps, each one an operator from registry.json with chosen parameters.
// The exported ONNX graph of every step is fetched, decoded with protobuf.js (vendor/onnx.proto),
// renamed with a per-step prefix and spliced into one ModelProto: step i's image input is wired to
// step i-1's output, the slider parameters stay graph inputs (or are baked as initializers on
// request), and the result runs here through onnxruntime-web and downloads as a single .onnx.
// Pipelines are kept in localStorage; a share link carries one in the URL.
(function () {
  "use strict";

  const STORAGE_KEY = "kornia-playground-pipelines";
  const CONTAINERS = {
    sequential: {
      title: "torch.nn.Sequential",
      code: "torch.nn.Sequential",
      blurb: "Plain PyTorch container. Any operator with a module form.",
      accepts: function (op) { return true; },
      extra: "",
    },
    image_sequential: {
      title: "kornia.augmentation.ImageSequential",
      code: "kornia.augmentation.ImageSequential",
      blurb: "Kornia container for image operators and augmentations; supports random ordering and probabilities.",
      accepts: function (op) { return true; },
      extra: "",
    },
    augmentation_sequential: {
      title: "kornia.augmentation.AugmentationSequential",
      code: "kornia.augmentation.AugmentationSequential",
      blurb: "Augmentations applied consistently to images, masks, boxes and keypoints (data_keys).",
      accepts: function (op) { return op.package === "kornia.augmentation" || op.package === "kornia.geometry.transform" || op.package === "kornia.enhance" || op.package === "kornia.filters" || op.package === "kornia.color"; },
      extra: '    data_keys=["input"],\n',
    },
  };

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

  function fill(template, values) {
    return template.replace(/\{(\w+)\}/g, function (m, key) { return key in values ? values[key] : m; });
  }

  function valuesOf(op, params) {
    const values = { img: "img" };
    op.params.forEach(function (p) {
      const v = params[p.name];
      values[p.name] = fmt(v, p);
      if (p.literals) values[p.name + "_literal"] = p.literals[v];
      if (p.derived && p.derived[String(v)]) {
        const d = p.derived[String(v)];
        for (const k in d) values[p.name + "_" + k] = d[k];
      }
    });
    return values;
  }

  // "kornia.filters.GaussianBlur2d((5, 5), (1.5, 1.5))({img})" -> the constructor call
  function moduleExpr(op, params) {
    const template = op.module_snippet || op.snippet;
    const call = fill(template, valuesOf(op, params));
    return call.replace(/\(img\)\s*$/, "").replace(/\(img,\s*[^)]*\)\s*$/, "");
  }

  function graphKey(op, params) {
    if (!op.select_order || !op.select_order.length) return "default";
    return op.select_order.map(function (name) {
      const p = op.params.find(function (q) { return q.name === name; });
      return fmt(params[name], p);
    }).join("|");
  }

  function defaults(op) {
    const params = {};
    op.params.forEach(function (p) { params[p.name] = p.default; });
    return params;
  }

  function uid() { return Math.random().toString(36).slice(2, 10); }

  function loadAll() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); } catch (e) { return []; }
  }

  function saveAll(list) {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(list)); } catch (e) { /* private mode */ }
  }

  function b64url(str) { return btoa(unescape(encodeURIComponent(str))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, ""); }
  function unb64url(str) { return decodeURIComponent(escape(atob(str.replace(/-/g, "+").replace(/_/g, "/")))); }

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

  // ------------------------------------------------------------------ ONNX composition

  let ModelProto = null;   // protobuf.js type, loaded from vendor/onnx.proto
  const modelCache = {};   // path -> Uint8Array

  function loadSchema(root) {
    if (ModelProto) return Promise.resolve(ModelProto);
    return fetch(root + "vendor/onnx.proto").then(function (r) {
      if (!r.ok) throw new Error("onnx.proto: HTTP " + r.status);
      return r.text();
    }).then(function (text) {
      ModelProto = protobuf.parse(text, { keepCase: true }).root.lookupType("onnx.ModelProto");
      return ModelProto;
    });
  }

  function fetchModel(url) {
    if (!modelCache[url]) {
      modelCache[url] = fetch(url).then(function (r) {
        if (!r.ok) throw new Error(url.split("/").pop() + ": HTTP " + r.status);
        return r.arrayBuffer();
      }).then(function (buf) { return new Uint8Array(buf); });
    }
    return modelCache[url];
  }

  // Rename every tensor name in a graph (and nested subgraphs) through `map`.
  function renameGraph(graph, map) {
    (graph.node || []).forEach(function (node, i) {
      node.name = map(node.name || "node_" + i);   // unnamed nodes get unique names; duplicates are rejected
      node.input = (node.input || []).map(function (n) { return n ? map(n) : n; });
      node.output = (node.output || []).map(function (n) { return n ? map(n) : n; });
      (node.attribute || []).forEach(function (attr) {
        if (attr.g) renameGraph(attr.g, map);
        (attr.graphs || []).forEach(function (g) { renameGraph(g, map); });
      });
    });
    ["input", "output", "value_info"].forEach(function (field) {
      (graph[field] || []).forEach(function (vi) { vi.name = map(vi.name); });
    });
    (graph.initializer || []).forEach(function (t) { t.name = map(t.name); });
    (graph.sparse_initializer || []).forEach(function (t) { if (t.values) t.values.name = map(t.values.name); });
  }

  // Bring a graph exported at opset <= 17 up to opset 18: the reduce ops moved `axes` from an attribute to an
  // input, and Split needs `num_outputs` when it has no `split` input. Everything else the graphs here use is
  // unchanged between those versions. Initializers are appended to `inits`.
  const REDUCE_OPS = { ReduceMean: 1, ReduceMax: 1, ReduceMin: 1, ReduceProd: 1, ReduceL1: 1, ReduceL2: 1, ReduceLogSum: 1, ReduceLogSumExp: 1, ReduceSumSquare: 1 };
  function liftTo18(graph, prefix, inits) {
    let n = 0;
    (graph.node || []).forEach(function (node) {
      if (REDUCE_OPS[node.op_type]) {
        const i = (node.attribute || []).findIndex(function (a) { return a.name === "axes"; });
        if (i >= 0) {
          const axes = node.attribute[i].ints.map(Number);
          const name = prefix + "lift/axes_" + (n++);
          inits.push({ dims: [axes.length], data_type: 7, int64_data: axes, name: name });
          node.attribute.splice(i, 1);
          while (node.input.length < 1) node.input.push("");
          node.input = [node.input[0], name];
        }
      } else if (node.op_type === "Split" && !(node.input.length > 1 && node.input[1])) {
        if (!(node.attribute || []).some(function (a) { return a.name === "num_outputs"; })) {
          node.attribute = (node.attribute || []).concat([{ name: "num_outputs", type: 2, i: node.output.length }]);
        }
      }
      (node.attribute || []).forEach(function (attr) {
        if (attr.g) liftTo18(attr.g, prefix, inits);
        (attr.graphs || []).forEach(function (g) { liftTo18(g, prefix, inits); });
      });
    });
  }

  // steps: [{op, params, bytes} | {model, bytes}] -> encoded ModelProto. `bake` turns the live parameters
  // into constants. A model step must be last; its input normalisation becomes part of the graph, so the
  // download is a complete standalone model. `preview` adds the model's input image as an extra output.
  function compose(steps, bake, preview) {
    const models = steps.map(function (s) { return ModelProto.decode(s.bytes); });
    const nodes = [], inputs = [], initializers = [], valueInfo = [], extraOutputs = [];
    let prevOut = null;
    const paramInputs = []; // {name, value}
    models.forEach(function (m, i) {
      const prefix = "s" + i + "/";
      const g = m.graph;
      const imageIn = g.input[0].name;
      const map = function (n) { return prefix + n; };
      renameGraph(g, map);
      const onnxOpset = (m.opset_import || []).filter(function (o) { return !o.domain || o.domain === "ai.onnx"; }).map(function (o) { return Number(o.version); })[0] || 0;
      if (onnxOpset && onnxOpset < 18) liftTo18(g, prefix, initializers);
      const step = steps[i];
      if (step.model && prevOut !== null) {
        // (x * scale - mean) / std before the network, as constants; the image before it is kept for the preview
        const inp = step.model.input;
        let cur = prevOut;
        if (preview) extraOutputs.push({ name: cur, type: { tensor_type: { elem_type: 1 } } });
        const c = function (name, values, dims) { initializers.push({ dims: dims, data_type: 1, float_data: values, name: prefix + name }); return prefix + name; };
        if (inp.scale && inp.scale !== 1) { nodes.push({ op_type: "Mul", name: prefix + "pre_scale", input: [cur, c("scale_const", [inp.scale], [1])], output: [prefix + "pre_scaled"] }); cur = prefix + "pre_scaled"; }
        if (inp.mean) { nodes.push({ op_type: "Sub", name: prefix + "pre_mean", input: [cur, c("mean_const", inp.mean, [1, 3, 1, 1])], output: [prefix + "pre_centered"] }); cur = prefix + "pre_centered"; }
        if (inp.std) { nodes.push({ op_type: "Div", name: prefix + "pre_std", input: [cur, c("std_const", inp.std, [1, 3, 1, 1])], output: [prefix + "pre_normalized"] }); cur = prefix + "pre_normalized"; }
        prevOut = cur;
      }
      if (prevOut !== null) {
        // wire this step's image input to the previous output
        const from = prefix + imageIn;
        const rewire = function (graph) {
          graph.node.forEach(function (node) {
            node.input = node.input.map(function (n) { return n === from ? prevOut : n; });
            (node.attribute || []).forEach(function (attr) {
              if (attr.g) rewire(attr.g);
              (attr.graphs || []).forEach(rewire);
            });
          });
        };
        rewire(g);
      } else {
        inputs.push(g.input[0]);
      }
      // live parameters: keep as inputs or bake (a model has none)
      const live = step.op ? step.op.params.filter(function (p) { return p.kind === "live"; }) : [];
      const liveInputs = step.op ? g.input.slice(1) : [];
      liveInputs.forEach(function (vi, k) {
        const p = live[k];
        const value = p ? Number(step.params[p.name]) : 0;
        if (bake) {
          initializers.push({ dims: [1], data_type: 1, float_data: [value], name: vi.name });
        } else {
          inputs.push(vi);
          paramInputs.push({ name: vi.name, value: value, step: i, param: p ? p.name : vi.name });
        }
      });
      nodes.push.apply(nodes, g.node);
      initializers.push.apply(initializers, g.initializer || []);
      valueInfo.push.apply(valueInfo, g.value_info || []);
      prevOut = g.output[0].name;
    });
    const last = models[models.length - 1];
    // one opset per domain: the highest among the steps (the operator graphs are opset 18, the models 14 to 17)
    const opsets = {};
    models.forEach(function (m) { (m.opset_import || []).forEach(function (o) { const d = (!o.domain || o.domain === "ai.onnx") ? "" : o.domain; opsets[d] = Math.max(opsets[d] || 0, Number(o.version)); }); });
    opsets[""] = Math.max(opsets[""] || 0, 18);
    const model = ModelProto.create({
      ir_version: models.reduce(function (v, m) { return Math.max(v, Number(m.ir_version)); }, 8),
      producer_name: "kornia.org/playground",
      producer_version: "1",
      opset_import: Object.keys(opsets).map(function (d) { return { domain: d, version: opsets[d] }; }),
      graph: { name: "kornia_pipeline", node: nodes, input: inputs, output: last.graph.output.concat(extraOutputs), initializer: initializers, value_info: valueInfo },
    });
    return { bytes: ModelProto.encode(model).finish(), paramInputs: paramInputs, imageInput: inputs[0].name, outputName: last.graph.output[0].name,
             outputNames: last.graph.output.map(function (o) { return o.name; }), previewOutput: extraOutputs.length ? extraOutputs[0].name : null };
  }

  // ------------------------------------------------------------------ the app

  let registry = null;
  let modelsIndex = null;   // models/index.json, loaded at init
  let ROOT = "";
  let mount = null;

  // models a pipeline can end with: single image in, decoded by models.js; the two-pass feature models are out
  function pipelineModels() {
    if (!modelsIndex) return [];
    return modelsIndex.models.filter(function (m) { return m.output.kind.indexOf("features_") !== 0; });
  }
  function modelById(id) { return pipelineModels().find(function (m) { return m.id === id; }); }
  function absUrl(url) { return /^https?:/.test(url) ? url : ROOT + url; }
  function sizeText(mb) { return mb >= 1 ? Math.round(mb) + " MB" : (mb * 1000).toFixed(0) + " kB"; }

  function eligible(op) {
    return op.status === "live" && op.mode === "onnx" && !op.guidance && (op.module_snippet || op.package === "kornia.augmentation");
  }

  function opById(id) { return registry.ops.find(function (o) { return o.id === id; }); }

  function navigate(params) {
    const url = new URL(window.location.href);
    url.search = "";
    for (const k in params) url.searchParams.set(k, params[k]);
    window.history.pushState({}, "", url.toString());
    render();
  }

  function render() {
    const q = new URLSearchParams(window.location.search);
    mount.innerHTML = "";
    if (q.get("share")) {
      try {
        const pipe = JSON.parse(unb64url(q.get("share")));   // {name, container, steps, model}
        pipe.id = uid();
        pipe.name = (pipe.name || "Shared pipeline");
        const list = loadAll();
        list.push(pipe);
        saveAll(list);
        navigate({ p: pipe.id });
        return;
      } catch (e) { /* fall through to the list */ }
    }
    if (q.get("new") !== null && q.get("new") !== undefined && q.has("new")) return renderNew();
    if (q.get("p")) {
      const pipe = loadAll().find(function (x) { return x.id === q.get("p"); });
      if (pipe) return renderEditor(pipe);
    }
    renderList();
  }

  // ---- list
  function renderList() {
    const head = el("div", "pg-pipe-head");
    const h = el("h2");
    h.textContent = "Pipelines";
    head.appendChild(h);
    mount.appendChild(head);
    const lead = el("p", "pg-summary");
    lead.textContent = "Chain kornia operators, run the chain on the sample images here, then download it as one ONNX graph or copy the PyTorch code. Pipelines are saved in this browser.";
    mount.appendChild(lead);

    const list = loadAll();
    const grid = el("div", "pg-pipe-grid");
    const create = el("div", "pg-pipe-card pg-pipe-create");
    const createBtn = el("button", "pg-pipe-create-btn", { type: "button", "aria-expanded": "false" });
    createBtn.innerHTML = '<i class="fas fa-plus" aria-hidden="true"></i> Create a pipeline';
    const form = el("div", "pg-pipe-create-form");
    form.hidden = true;
    const nameRow = el("div", "pg-pipe-namerow");
    const label = el("label", "", { for: "pg-pipe-name" });
    label.textContent = "Name";
    const input = el("input", "pg-input", { id: "pg-pipe-name", type: "text", value: "My pipeline", maxlength: "60" });
    nameRow.appendChild(label);
    nameRow.appendChild(input);
    form.appendChild(nameRow);
    const hint = el("p", "pg-note");
    hint.textContent = "Pick the container the code will use:";
    form.appendChild(hint);
    const choices = el("div", "pg-pipe-choices");
    Object.keys(CONTAINERS).forEach(function (key) {
      const c = CONTAINERS[key];
      const card = el("button", "pg-pipe-choice", { type: "button" });
      const t = el("code");
      t.textContent = c.title;
      const b = el("span", "pg-note");
      b.textContent = c.blurb;
      card.appendChild(t);
      card.appendChild(b);
      card.addEventListener("click", function () {
        const pipe = { id: uid(), name: input.value.trim() || "My pipeline", container: key, steps: [], created: Date.now() };
        const all = loadAll();
        all.push(pipe);
        saveAll(all);
        navigate({ p: pipe.id });
      });
      choices.appendChild(card);
    });
    form.appendChild(choices);
    createBtn.addEventListener("click", function () {
      form.hidden = !form.hidden;
      createBtn.setAttribute("aria-expanded", form.hidden ? "false" : "true");
      create.classList.toggle("pg-pipe-create-open", !form.hidden);
      if (!form.hidden) input.focus();
    });
    create.appendChild(createBtn);
    create.appendChild(form);
    grid.appendChild(create);
    list.forEach(function (pipe) {
      const card = el("div", "pg-pipe-card");
      const name = el("a", "pg-pipe-name", { href: "?p=" + pipe.id });
      name.textContent = pipe.name;
      name.addEventListener("click", function (ev) { ev.preventDefault(); navigate({ p: pipe.id }); });
      const meta = el("p", "pg-note");
      meta.textContent = CONTAINERS[pipe.container].title + " · " + pipe.steps.length + " step" + (pipe.steps.length === 1 ? "" : "s");
      const steps = el("p", "pg-pipe-steps");
      steps.textContent = pipe.steps.map(function (s) { const op = opById(s.op); return op ? op.name : s.op; }).join(" → ") || "empty";
      const actions = el("div", "pg-pipe-actions");
      const dup = el("button", "pg-link", { type: "button" });
      dup.textContent = "duplicate";
      dup.addEventListener("click", function () {
        const copy = JSON.parse(JSON.stringify(pipe));
        copy.id = uid();
        copy.name = pipe.name + " (copy)";
        const all = loadAll();
        all.push(copy);
        saveAll(all);
        render();
      });
      const del = el("button", "pg-link pg-link-danger", { type: "button" });
      del.textContent = "delete";
      del.addEventListener("click", function () {
        saveAll(loadAll().filter(function (x) { return x.id !== pipe.id; }));
        render();
      });
      actions.appendChild(dup);
      actions.appendChild(del);
      card.appendChild(name);
      card.appendChild(meta);
      card.appendChild(steps);
      card.appendChild(actions);
      grid.appendChild(card);
    });
    mount.appendChild(grid);
  }

  // ---- new
  function renderNew() {
    const head = el("div", "pg-pipe-head");
    const h = el("h2");
    h.textContent = "New pipeline";
    head.appendChild(h);
    mount.appendChild(head);
    const lead = el("p", "pg-summary");
    lead.textContent = "Pick the container the code will use, then add steps.";
    mount.appendChild(lead);
    const nameRow = el("div", "pg-pipe-namerow");
    const label = el("label", "", { for: "pg-pipe-name" });
    label.textContent = "Name";
    const input = el("input", "pg-input", { id: "pg-pipe-name", type: "text", value: "My pipeline", maxlength: "60" });
    nameRow.appendChild(label);
    nameRow.appendChild(input);
    mount.appendChild(nameRow);
    const grid = el("div", "pg-pipe-grid");
    Object.keys(CONTAINERS).forEach(function (key) {
      const c = CONTAINERS[key];
      const card = el("button", "pg-pipe-card pg-pipe-choice", { type: "button" });
      const t = el("code");
      t.textContent = c.title;
      const b = el("p", "pg-note");
      b.textContent = c.blurb;
      card.appendChild(t);
      card.appendChild(b);
      card.addEventListener("click", function () {
        const pipe = { id: uid(), name: input.value.trim() || "My pipeline", container: key, steps: [], created: Date.now() };
        const all = loadAll();
        all.push(pipe);
        saveAll(all);
        navigate({ p: pipe.id });
      });
      grid.appendChild(card);
    });
    mount.appendChild(grid);
    const back = el("p", "pg-backlink");
    const a = el("a", "", { href: "?" });
    a.innerHTML = '<i class="fas fa-arrow-left" aria-hidden="true"></i> All pipelines';
    a.addEventListener("click", function (ev) { ev.preventDefault(); navigate({}); });
    back.appendChild(a);
    mount.appendChild(back);
  }

  // ---- editor
  function renderEditor(pipe) {
    const container = CONTAINERS[pipe.container];
    const candidates = registry.ops.filter(function (op) { return eligible(op) && container.accepts(op); });

    function persist() {
      const all = loadAll();
      const i = all.findIndex(function (x) { return x.id === pipe.id; });
      if (i >= 0) all[i] = pipe; else all.push(pipe);
      saveAll(all);
    }

    const head = el("div", "pg-pipe-head");
    const nameInput = el("input", "pg-input pg-pipe-title", { type: "text", value: pipe.name, maxlength: "60", "aria-label": "Pipeline name" });
    nameInput.addEventListener("change", function () { pipe.name = nameInput.value.trim() || "My pipeline"; persist(); refreshCode(); });
    const cont = el("code", "pg-pipe-container");
    cont.textContent = container.title;
    head.appendChild(nameInput);
    head.appendChild(cont);
    mount.appendChild(head);

    const layout = el("div", "pg-pipe-layout");
    mount.appendChild(layout);

    // steps block (appended to the layout after the preview below)
    const stepsCol = el("div", "pg-pipe-steps-col");
    const stepsList = el("ol", "pg-pipe-list");
    stepsCol.appendChild(stepsList);
    const addRow = el("div", "pg-pipe-add");
    const addSel = el("select", "pg-select", { "aria-label": "Operator to add" });
    const ph = el("option", "", { value: "" });
    ph.textContent = "Add a step…";
    addSel.appendChild(ph);
    const packages = registry.packages.slice();
    if (pipe.container === "augmentation_sequential") {
      // the augmentations are what this container is for: list them first
      packages.sort(function (a, b) { return (a.id === "kornia.augmentation" ? 0 : 1) - (b.id === "kornia.augmentation" ? 0 : 1); });
    }
    packages.forEach(function (pkg) {
      const ops = candidates.filter(function (o) { return o.package === pkg.id; });
      if (!ops.length) return;
      const group = el("optgroup", "", { label: pkg.title });
      ops.forEach(function (op) {
        const o = el("option", "", { value: op.id });
        o.textContent = op.name;
        group.appendChild(o);
      });
      addSel.appendChild(group);
    });
    // older saved pipelines carried the model as a step; it now lives in pipe.model
    pipe.steps = pipe.steps.filter(function (st) { if (st.model) { pipe.model = st.model; return false; } return true; });
    if (pipe.model && !modelById(pipe.model)) pipe.model = null;
    addSel.addEventListener("change", function () {
      if (!addSel.value) return;
      const op = opById(addSel.value);
      pipe.steps.push({ op: op.id, params: defaults(op) });
      addSel.value = "";
      persist();
      renderSteps();
      schedule();
    });
    addRow.appendChild(addSel);
    stepsCol.appendChild(addRow);

    // the model, optional and separate from the preprocessing steps: it always runs last
    const modelBox = el("div", "pg-pipe-model");
    const modelHead = el("div", "pg-pipe-model-head");
    const modelLabel = el("span", "pg-pipe-model-label");
    modelLabel.innerHTML = '<i class="fas fa-brain" aria-hidden="true"></i> Model <span class="pg-note">optional, runs after the steps above</span>';
    const modelSel = el("select", "pg-select", { "aria-label": "Model to run after the steps" });
    const none = el("option", "", { value: "" });
    none.textContent = "No model: the pipeline outputs an image";
    modelSel.appendChild(none);
    pipelineModels().forEach(function (m) {
      const o = el("option", "", { value: m.id });
      o.textContent = (m.title || m.name) + " · " + (m.subtitle || m.task.toLowerCase()) + " · " + sizeText(m.size_mb);
      if (pipe.model === m.id) o.selected = true;
      modelSel.appendChild(o);
    });
    modelSel.addEventListener("change", function () {
      pipe.model = modelSel.value || null;
      persist();
      renderModel();
      refreshWarnings();
      refreshCode();
      refreshRunLabel();
      schedule();
    });
    modelHead.appendChild(modelLabel);
    modelHead.appendChild(modelSel);
    modelBox.appendChild(modelHead);
    const modelRow = el("div", "pg-pipe-model-row");
    modelBox.appendChild(modelRow);
    function renderModel() {
      modelRow.innerHTML = "";
      const m = pipe.model ? modelById(pipe.model) : null;
      modelRow.hidden = !m;
      if (!m) return;
      const info = el("p", "pg-pipe-model-info");
      const a = el("a", "", { href: ROOT + "models/" + m.slug + "/", target: "_blank", rel: "noopener" });
      a.textContent = m.name;
      info.appendChild(a);
      info.appendChild(document.createTextNode(" — " + m.summary + " "));
      const tag = el("span", "pg-tag pg-tag-task");
      tag.textContent = sizeText(m.size_mb) + (/^https?:/.test(m.url) ? " from " + m.hosted : "");
      info.appendChild(tag);
      modelRow.appendChild(info);
    }
    renderModel();
    stepsCol.appendChild(modelBox);
    const warnings = el("ul", "pg-pipe-warnings");
    stepsCol.appendChild(warnings);

    // preview: input and output side by side, full width, above the steps
    const previewCol = el("div", "pg-pipe-preview");
    const images = el("div", "pg-images pg-pipe-images");
    const figIn = el("figure");
    const canvasIn = el("canvas", "", { width: registry.size, height: registry.size });
    const capIn = el("figcaption");
    capIn.textContent = "input";
    figIn.appendChild(canvasIn);
    figIn.appendChild(capIn);
    const figOut = el("figure");
    const canvasOut = el("canvas", "", { width: registry.size, height: registry.size });
    const capOut = el("figcaption");
    capOut.textContent = "pipeline output";
    figOut.appendChild(canvasOut);
    figOut.appendChild(capOut);
    images.appendChild(figIn);
    images.appendChild(figOut);
    const thumbs = el("div", "pg-thumbs pg-pipe-thumbs");
    const pick = el("div", "pg-thumbs-pick pg-pipe-pick");
    thumbs.appendChild(pick);
    const previewRow = el("div", "pg-pipe-preview-row");
    previewRow.appendChild(thumbs);
    previewRow.appendChild(images);
    previewCol.appendChild(previewRow);
    let image = registry.images[0].id;
    let sourceKind = "image";
    function markSelected(btn) {
      pick.querySelectorAll("button").forEach(function (b) { b.classList.remove("pg-selected"); });
      btn.classList.add("pg-selected");
    }
    registry.images.filter(function (im) { return im.selectable !== false; }).forEach(function (im) {
      const btn = el("button", im.id === image ? "pg-selected" : "", { type: "button", title: im.label, "aria-label": im.label });
      const t = el("img", "", { src: ROOT + im.thumb, alt: "" });
      btn.appendChild(t);
      pick.appendChild(btn);
      btn.addEventListener("click", function () {
        stopVideo();
        image = im.id;
        sourceKind = "image";
        capIn.textContent = "input";
        markSelected(btn);
        loadInput().then(schedule);
      });
    });
    (registry.videos || []).forEach(function (v) {
      const btn = el("button", "pg-thumb-video", { type: "button", title: v.label + " (" + v.seconds + " s, " + v.license + ")", "aria-label": "video: " + v.label });
      const t = el("img", "", { src: ROOT + v.thumb, alt: "" });
      const play = el("span", "pg-thumb-play", { "aria-hidden": "true" });
      play.innerHTML = '<i class="fas fa-play"></i>';
      btn.appendChild(t);
      btn.appendChild(play);
      pick.appendChild(btn);
      btn.addEventListener("click", function () {
        image = v.id;
        sourceKind = "video";
        capIn.textContent = "input: " + v.label + " (" + v.license + ")";
        markSelected(btn);
        startVideo(v);
      });
    });
    layout.appendChild(previewCol);
    layout.appendChild(stepsCol);
    const actions = el("div", "pg-actions pg-pipe-actions-row");
    const run = el("button", "pg-btn", { type: "button" });
    run.addEventListener("click", schedule);
    function refreshRunLabel() {
      const random = pipe.steps.some(function (st) { const o = opById(st.op); return o && o.stochastic; });
      run.innerHTML = random ? '<i class="fas fa-dice" aria-hidden="true"></i> Re-roll' : '<i class="fas fa-play" aria-hidden="true"></i> Run';
    }
    refreshRunLabel();
    const dlInputs = el("button", "pg-btn pg-btn-ghost", { type: "button", title: "The slider parameters stay graph inputs" });
    dlInputs.innerHTML = '<i class="fas fa-download" aria-hidden="true"></i> ONNX, parameters as inputs';
    dlInputs.addEventListener("click", function () { download(false); });
    const dlBaked = el("button", "pg-btn pg-btn-ghost", { type: "button", title: "The current parameter values are baked into the graph" });
    dlBaked.innerHTML = '<i class="fas fa-download" aria-hidden="true"></i> ONNX, parameters baked';
    dlBaked.addEventListener("click", function () { download(true); });
    const share = el("button", "pg-btn pg-btn-ghost", { type: "button" });
    share.innerHTML = '<i class="fas fa-link" aria-hidden="true"></i> Copy share link';
    share.addEventListener("click", function () {
      const url = new URL(window.location.href);
      url.search = "?share=" + b64url(JSON.stringify({ name: pipe.name, container: pipe.container, steps: pipe.steps, model: pipe.model || null }));
      navigator.clipboard.writeText(url.toString()).then(function () { share.innerHTML = '<i class="fas fa-check" aria-hidden="true"></i> Link copied'; setTimeout(function () { share.innerHTML = '<i class="fas fa-link" aria-hidden="true"></i> Copy share link'; }, 1500); });
    });
    const status = el("span", "pg-status");
    actions.appendChild(run);
    actions.appendChild(dlInputs);
    actions.appendChild(dlBaked);
    actions.appendChild(share);
    const actionsBlock = el("div", "pg-pipe-actions-block");
    actionsBlock.appendChild(actions);
    actionsBlock.appendChild(status);
    layout.appendChild(actionsBlock);

    // code
    const codeBox = el("div", "pg-code pg-pipe-code", { id: "code" });
    const tabs = el("div", "pg-tabs");
    const tab = el("span", "pg-tab pg-tab-active");
    tab.textContent = "Python";
    const copy = el("button", "pg-copy", { type: "button" });
    copy.textContent = "copy";
    const pre = el("pre");
    const code = el("code", "language-python");
    pre.appendChild(code);
    copy.addEventListener("click", function () {
      navigator.clipboard.writeText(code.textContent).then(function () { copy.textContent = "copied"; setTimeout(function () { copy.textContent = "copy"; }, 1200); });
    });
    tabs.appendChild(tab);
    tabs.appendChild(copy);
    codeBox.appendChild(tabs);
    codeBox.appendChild(pre);
    mount.appendChild(codeBox);

    const back = el("p", "pg-backlink");
    const a = el("a", "", { href: "?" });
    a.innerHTML = '<i class="fas fa-arrow-left" aria-hidden="true"></i> All pipelines';
    a.addEventListener("click", function (ev) { ev.preventDefault(); navigate({}); });
    back.appendChild(a);
    mount.appendChild(back);

    // ---- steps UI
    function renderSteps() {
      stepsList.innerHTML = "";
      if (!pipe.steps.length) {
        const li = el("li", "pg-note");
        li.textContent = pipe.model ? "No preprocessing yet: the model runs on the raw frame. Add steps below." : "No steps yet. Add one below.";
        stepsList.appendChild(li);
      }
      pipe.steps.forEach(function (step, idx) {
        const op = opById(step.op);
        const li = el("li", "pg-pipe-step");
        const num = el("span", "pg-pipe-step-num");
        num.textContent = String(idx + 1);
        const title = el("a", "pg-pipe-step-name", { href: ROOT + "ops/" + (op ? op.slug : "") + "/", target: "_blank", rel: "noopener", title: "open the operator page" });
        title.textContent = op ? op.name : step.op;
        const paramsBox = el("div", "pg-pipe-step-params");
        const tools = el("span", "pg-pipe-step-tools");
        const up = el("button", "pg-link", { type: "button", title: "move up", disabled: idx === 0 ? "" : null });
        up.innerHTML = '<i class="fas fa-arrow-up" aria-hidden="true"></i>';
        if (idx === 0) up.disabled = true;
        up.addEventListener("click", function () { pipe.steps.splice(idx - 1, 2, pipe.steps[idx], pipe.steps[idx - 1]); persist(); renderSteps(); schedule(); });
        const down = el("button", "pg-link", { type: "button", title: "move down" });
        down.innerHTML = '<i class="fas fa-arrow-down" aria-hidden="true"></i>';
        if (idx === pipe.steps.length - 1) down.disabled = true;
        down.addEventListener("click", function () { pipe.steps.splice(idx, 2, pipe.steps[idx + 1], pipe.steps[idx]); persist(); renderSteps(); schedule(); });
        const rm = el("button", "pg-link pg-link-danger", { type: "button", title: "remove" });
        rm.innerHTML = '<i class="fas fa-times" aria-hidden="true"></i>';
        rm.addEventListener("click", function () { pipe.steps.splice(idx, 1); persist(); renderSteps(); schedule(); });
        tools.appendChild(up);
        tools.appendChild(down);
        tools.appendChild(rm);
        li.appendChild(num);
        li.appendChild(title);
        li.appendChild(paramsBox);
        li.appendChild(tools);
        if (!op) { stepsList.appendChild(li); return; }
        if (!op.params.length) {
          const none = el("span", "pg-note");
          none.textContent = "no parameters";
          paramsBox.appendChild(none);
        }
        op.params.forEach(function (p) {
          const row = el("div", "pg-param pg-pipe-param");
          const label = el("label");
          label.textContent = p.label || p.name;
          const out = el("output");
          out.textContent = fmt(step.params[p.name], p);
          let input;
          if (p.kind === "select") {
            input = el("select", "");
            p.choices.forEach(function (c) {
              const option = el("option", "", { value: c });
              option.textContent = fmt(c, p);
              if (String(c) === String(step.params[p.name])) option.selected = true;
              input.appendChild(option);
            });
            input.addEventListener("change", function () {
              step.params[p.name] = p.type === "str" ? input.value : Number(input.value);
              out.textContent = fmt(step.params[p.name], p);
              persist();
              schedule();
            });
          } else {
            input = el("input", "", { type: "range", min: p.min, max: p.max, step: p.step, value: step.params[p.name] });
            input.addEventListener("input", function () {
              step.params[p.name] = Number(input.value);
              out.textContent = fmt(step.params[p.name], p);
              persist();
              schedule();
            });
          }
          row.appendChild(label);
          row.appendChild(input);
          row.appendChild(out);
          paramsBox.appendChild(row);
        });
        stepsList.appendChild(li);
      });
      refreshWarnings();
      refreshCode();
      refreshRunLabel();
    }

    function refreshWarnings() {
      warnings.innerHTML = "";
      const msgs = [];
      let size = registry.size, channels = 3;
      pipe.steps.forEach(function (step, idx) {
        const op = opById(step.op);
        if (!op) { msgs.push("Step " + (idx + 1) + ": operator " + step.op + " is not in this build."); return; }
        if (channels !== 3) msgs.push("Step " + (idx + 1) + " (" + op.name + ") receives a " + channels + "-channel image; kornia operators here expect RGB.");
        if (!op.dynamic && size !== registry.size) msgs.push("Step " + (idx + 1) + " (" + op.name + ") is a fixed " + registry.size + " px graph but receives " + size + " px.");
        if (op.output) {
          channels = op.output.channels;
          // the registry records the output at the default choice; follow the selected size or factor instead
          if (op.params.some(function (p) { return p.name === "size"; })) size = Number(step.params.size) || op.output.height;
          else if (op.params.some(function (p) { return p.name === "factor"; })) size = Math.round(size * Number(step.params.factor));
          else size = op.output.height === registry.size ? size : op.output.height;
        }
        if (!op.dynamic) msgs.push("Step " + (idx + 1) + " (" + op.name + ") exports at a fixed 1×3×" + registry.size + "×" + registry.size + " input, so the pipeline will too.");
      });
      const model = pipe.model ? modelById(pipe.model) : null;
      if (model) {
        if (model.input.fixed && size !== model.input.size) msgs.push((model.title || model.name) + " needs a " + model.input.size + "×" + model.input.size + " input but receives " + size + " px; add resize (" + model.input.size + ") as the last step.");
        if (channels !== 3) msgs.push((model.title || model.name) + " expects an RGB image; the steps above produce " + channels + " channel(s).");
        if (/^https?:/.test(model.url)) msgs.push((model.title || model.name) + " downloads " + sizeText(model.size_mb) + " from " + model.hosted + " the first time the pipeline runs; the exported ONNX will be about that size.");
      }
      const seen = {};
      msgs.forEach(function (m) {
        if (seen[m]) return;
        seen[m] = true;
        const li = el("li");
        li.textContent = m;
        warnings.appendChild(li);
      });
    }

    function pythonCode() {
      const lines = ["import torch", "import kornia", "import kornia.augmentation", "from kornia.io import load_image", ""];
      lines.push('img = load_image("' + image + '.png")[None]  # (1, 3, ' + registry.size + ", " + registry.size + ") float in [0, 1]");
      lines.push("");
      const model = pipe.model ? modelById(pipe.model) : null;
      if (model && model.pipeline) model.pipeline.imports.forEach(function (imp) { lines.splice(4, 0, imp); });
      lines.push("pipeline = " + container.code + "(");
      pipe.steps.forEach(function (step) {
        const op = opById(step.op);
        if (op) lines.push("    " + moduleExpr(op, step.params) + ",");
      });
      if (container.extra) lines.push(container.extra.replace(/\n$/, ""));
      lines.push(")");
      lines.push("out = pipeline(img)");
      if (model) {
        lines.push("");
        if (model.pipeline) {
          lines.push("model = " + model.pipeline.build);
          if (model.pipeline.note) lines.push("# " + model.pipeline.note);
          lines.push("result = " + model.pipeline.call);
        } else {
          lines.push("# then " + model.name + ": see its playground page for the call");
        }
      }
      return lines.join("\n");
    }

    function refreshCode() {
      code.textContent = pythonCode();
      if (typeof hljs !== "undefined") {
        code.removeAttribute("data-highlighted");
        hljs.highlightElement(code);
      }
    }

    // ---- running and exporting
    let inputTensor = null;
    let timer = null;
    let running = false;
    let pending = false;
    let composedKey = null;
    let composed = null; // {bytes, paramInputs, imageInput, session}
    let video = null;
    let videoLoop = false;

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
        previewCol.appendChild(video);
      }
      if (video.dataset.id !== v.id) {
        video.innerHTML = "";
        [["webm", "video/webm"], ["mp4", "video/mp4"]].forEach(function (pair) {
          if (v[pair[0]]) video.appendChild(el("source", "", { src: ROOT + v[pair[0]], type: pair[1] }));
        });
        video.dataset.id = v.id;
        video.load();
      }
      videoLoop = true;
      const pr = video.play();
      if (pr && pr.catch) pr.catch(function () {});
      pump();
    }

    function pump() {
      if (!videoLoop || !previewCol.isConnected) { videoLoop = false; return; }
      if (video.readyState >= 2 && typeof ort !== "undefined" && pipe.steps.length) {
        canvasIn.getContext("2d").drawImage(video, 0, 0, registry.size, registry.size);
        inputTensor = canvasToTensor(canvasIn, registry.size);
        execute().then(function () { window.requestAnimationFrame(pump); });
      } else {
        window.requestAnimationFrame(pump);
      }
    }

    document.addEventListener("visibilitychange", function () {
      if (!video || !videoLoop) return;
      if (document.hidden) video.pause(); else video.play().catch(function () {});
    });

    function loadInput() {
      const im = registry.images.find(function (i) { return i.id === image; });
      if (!im) return Promise.resolve();
      return loadImage(ROOT + im.file).then(function (img) {
        canvasIn.getContext("2d").drawImage(img, 0, 0, registry.size, registry.size);
        inputTensor = typeof ort !== "undefined" ? canvasToTensor(canvasIn, registry.size) : null;
      });
    }

    function stepModels() {
      const jobs = pipe.steps.map(function (step) {
        const op = opById(step.op);
        if (!op) throw new Error("operator " + step.op + " is not in this build");
        return fetchModel(ROOT + op.graphs[graphKey(op, step.params)]).then(function (bytes) { return { op: op, params: step.params, bytes: bytes }; });
      });
      const m = pipe.model ? modelById(pipe.model) : null;
      if (m) jobs.push(fetchModel(absUrl(m.url)).then(function (bytes) { return { model: m, bytes: bytes }; }));
      return Promise.all(jobs);
    }

    function build(bake, preview) {
      return loadSchema(ROOT).then(stepModels).then(function (steps) {
        if (!steps.length) throw new Error("add at least one step");
        if (steps.length === 1 && steps[0].model) throw new Error("add at least one preprocessing step, or use the model's own page");
        return compose(steps, bake, preview);
      });
    }

    function schedule() {
      refreshCode();
      refreshWarnings();
      clearTimeout(timer);
      timer = setTimeout(runPipeline, 60);
    }

    function runPipeline() {
      if (typeof ort === "undefined" || typeof protobuf === "undefined") {
        status.classList.add("pg-error");
        status.textContent = "onnxruntime-web or protobuf.js did not load; the preview is unavailable.";
        return;
      }
      if (!pipe.steps.length) { status.textContent = pipe.model ? "add at least one preprocessing step" : ""; canvasOut.getContext("2d").clearRect(0, 0, canvasOut.width, canvasOut.height); return; }
      if (running) { pending = true; return; }
      if (!inputTensor) return;
      running = true;
      execute().then(function () {
        running = false;
        if (pending) { pending = false; runPipeline(); }
      });
    }

    // one run of the composed graph on the current input; resolves when drawn
    function execute() {
      status.classList.remove("pg-error");
      // the graph only changes when the operators or a discrete choice change; live values are inputs
      const key = JSON.stringify([pipe.model].concat(pipe.steps.map(function (s) { const op = opById(s.op); return op ? [op.id, graphKey(op, s.params)] : [s.op]; })));
      const t0 = performance.now();
      const modelStep = pipe.model ? modelById(pipe.model) : null;
      if (!(composed && composedKey === key)) status.textContent = modelStep && /^https?:/.test(modelStep.url) && !modelCache[absUrl(modelStep.url)] ? "downloading " + sizeText(modelStep.size_mb) + " model, then composing…" : "composing…";
      const ready = (composed && composedKey === key)
        ? Promise.resolve(composed)
        : build(false, true).then(function (c) {
          return ort.InferenceSession.create(c.bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" }).then(function (session) {
            c.session = session;
            composed = c;
            composedKey = key;
            return c;
          });
        });
      return ready.then(function (c) {
        const feeds = {};
        feeds[c.imageInput] = inputTensor;
        c.paramInputs.forEach(function (pi) {
          feeds[pi.name] = new ort.Tensor("float32", new Float32Array([Number(pipe.steps[pi.step].params[pi.param])]), [1]);
        });
        return c.session.run(feeds).then(function (results) {
          const ms = performance.now() - t0;
          let detail = "";
          if (modelStep && window.PGModels) {
            // the image the model saw (before its normalisation) is the backdrop for boxes, faces, bars
            const pre = document.createElement("canvas");
            tensorToCanvas(results[c.previewOutput], pre, "clamp");
            const modelResults = {};
            c.outputNames.forEach(function (n) { modelResults[n.replace(/^s\d+\//, "")] = results[n]; });
            const labelsReady = modelStep.output.labels ? window.PGModels.getLabels(modelStep.output.labels, ROOT) : Promise.resolve(null);
            return labelsReady.then(function (labels) {
              detail = window.PGModels.drawResults(modelStep, modelResults, pre, canvasOut, { labels: labels });
              status.textContent = pipe.steps.length + " step" + (pipe.steps.length === 1 ? "" : "s") + " + " + (modelStep.title || modelStep.name) + ": " + detail + " · " + (c.bytes.length / 1048576).toFixed(1) + " MB graph, " + ms.toFixed(0) + " ms on your machine";
            });
          }
          const out = results[c.outputName] || results[Object.keys(results)[0]];
          const last = opById(pipe.steps[pipe.steps.length - 1].op);
          tensorToCanvas(out, canvasOut, last && last.output ? last.output.display : "clamp");
          status.textContent = pipe.steps.length + " step" + (pipe.steps.length === 1 ? "" : "s") + ", " + (c.bytes.length / 1024).toFixed(0) + " KB graph, " + ms.toFixed(0) + " ms on your machine" + (sourceKind === "video" && ms > 0 ? " (" + (1000 / ms).toFixed(0) + " fps)" : "");
        });
      }).catch(function (err) {
        status.classList.add("pg-error");
        status.textContent = "could not run the pipeline: " + (err.message || err);
        videoLoop = false;
      });
    }

    function download(bake) {
      status.classList.remove("pg-error");
      status.textContent = "composing…";
      build(bake).then(function (c) {
        const blob = new Blob([c.bytes], { type: "application/octet-stream" });
        const url = URL.createObjectURL(blob);
        const a = el("a", "", { href: url, download: pipe.name.replace(/[^\w.-]+/g, "_") + (bake ? "-baked" : "") + ".onnx" });
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
        status.textContent = (c.bytes.length > 2 * 1048576 ? (c.bytes.length / 1048576).toFixed(1) + " MB" : (c.bytes.length / 1024).toFixed(0) + " KB") + " ONNX downloaded" + (bake ? " (parameters baked)" : " (inputs: " + [c.imageInput].concat(c.paramInputs.map(function (p) { return p.name; })).join(", ") + ")");
      }).catch(function (err) {
        status.classList.add("pg-error");
        status.textContent = "could not export: " + (err.message || err);
      });
    }

    renderSteps();
    loadInput().then(schedule).catch(function (err) { status.classList.add("pg-error"); status.textContent = err.message; });
  }

  window.PGPipelines = {
    init: function (reg, root) {
      registry = reg;
      ROOT = root;
      mount = document.getElementById("pg-ops");
      window.addEventListener("popstate", render);
      fetch(ROOT + "models/index.json", { cache: "no-cache" }).then(function (r) { return r.ok ? r.json() : null; })
        .catch(function () { return null; })
        .then(function (idx) { modelsIndex = idx; render(); });
    },
  };
})();
