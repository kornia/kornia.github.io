// kornia.org/playground/pipelines: build a graph of operators, run it in the browser, export it.
//
// A pipeline is a directed acyclic graph. Nodes are inputs (an image, a mask, boxes, keypoints), kornia
// operators from registry.json, models from models/index.json, and outputs. Ports are typed; an edge joins
// an output port to an input port of a compatible type, and a node can have several of each. The exported
// ONNX graph of every operator and model is fetched, decoded with protobuf.js (vendor/onnx.proto), renamed
// with a per-node prefix and spliced into one ModelProto following the graph: every input node becomes a
// graph input, every output node a graph output, and the slider parameters stay graph inputs (or are baked
// as initializers on request). The result runs here through onnxruntime-web and downloads as one .onnx.
// Pipelines are kept in localStorage (and in the account, from the dashboard); a share link carries one in
// the URL. Version 1 pipelines (a straight list of steps and one model) are migrated on load.
(function () {
  "use strict";

  const STORAGE_KEY = "kornia-playground-pipelines";
  const VERSION = 2;

  // port types: what flows along an edge
  const TYPES = {
    image: { label: "image", color: "#2563eb", dims: function (S) { return [1, 3, S, S]; } },
    gray: { label: "gray", color: "#64748b", dims: function (S) { return [1, 1, S, S]; } },
    mask: { label: "mask", color: "#0ea5e9", dims: function (S) { return [1, 1, S, S]; } },
    boxes: { label: "boxes", color: "#f59e0b", dims: function () { return [1, "K", 4]; } },
    keypoints: { label: "keypoints", color: "#10b981", dims: function () { return [1, "P", 2]; } },
    depth: { label: "depth", color: "#8b5cf6", dims: function (S) { return [1, 1, S, S]; } },
    result: { label: "result", color: "#ef4444", dims: function () { return []; } },
  };
  const INPUT_KINDS = ["image", "mask", "boxes", "keypoints"];
  // what an input port of a given type takes
  const ACCEPTS = { image: ["image"], gray: ["gray", "mask", "depth"], mask: ["mask", "gray"], boxes: ["boxes"], keypoints: ["keypoints"], depth: ["depth", "gray"], result: [] };

  // ------------------------------------------------------------------ helpers

  function el(tag, cls, attrs) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (attrs) for (const k in attrs) node.setAttribute(k, attrs[k]);
    return node;
  }
  function fmt(value, param) {
    if (param && param.type === "str") return '"' + value + '"';
    if (param && param.type === "int") return String(Math.round(Number(value)));
    const n = Number(value);
    return Number.isInteger(n) ? n.toFixed(1) : String(Number(n.toFixed(4)));
  }
  function fill(template, values) {
    return template.replace(/\{(\w+)\}/g, function (m, key) { return key in values ? values[key] : m; });
  }
  function valuesOf(op, params, img) {
    const values = { img: img || "img" };
    op.params.forEach(function (p) {
      values[p.name] = fmt(params[p.name], p);
      if (p.literals) values[p.name + "_literal"] = p.literals[params[p.name]];
      if (p.derived && p.derived[String(params[p.name])]) {
        const d = p.derived[String(params[p.name])];
        for (const k in d) values[p.name + "_" + k] = d[k];
      }
    });
    return values;
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
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]").map(migrate); } catch (e) { return []; }
  }
  function saveAll(list) {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(list)); } catch (e) { /* private mode */ }
  }
  // ---- the account: a signed-in, verified visitor's changes are saved there as they happen; nothing to press
  const syncTimers = {};
  function canSync() { const A = window.KorniaAuth; return !!(A && A.pipelines && A.pipelines.canSync()); }
  function syncSoon(pipe) {
    if (!canSync()) return;
    clearTimeout(syncTimers[pipe.id]);
    syncTimers[pipe.id] = setTimeout(function () {
      window.KorniaAuth.pipelines.save(pipe).then(function () {
        const all = loadAll(); const i = all.findIndex(function (x) { return x.id === pipe.id; });
        if (i >= 0 && !all[i].synced) { all[i].synced = true; saveAll(all); pipe.synced = true; renderSidebar(); }
      }).catch(function (e) { console.warn("pipeline sync failed", e); });
    }, 800);
  }
  function unsync(id) { if (canSync()) window.KorniaAuth.pipelines.remove(id).catch(function () {}); }
  // on sign-in: the account's pipelines join this browser's list (the newer copy wins when both have one)
  let merged = false;
  function mergeFromAccount() {
    if (!canSync() || merged) return Promise.resolve();
    merged = true;
    return window.KorniaAuth.pipelines.list().then(function (remote) {
      const all = loadAll(); const byId = {}; all.forEach(function (p) { byId[p.id] = p; });
      remote.forEach(function (r) {
        const l = byId[r.id];
        if (!l) { byId[r.id] = migrate(r); return; }
        if ((r.updated || 0) > (l.updated || 0)) { byId[r.id] = Object.assign(migrate(r), { synced: true }); } else { l.synced = true; }
      });
      saveAll(Object.values(byId));
      renderSidebar();
    }).catch(function (e) { console.warn("could not read the account's pipelines", e); });
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
      modelCache[url] = window.KorniaCache ? window.KorniaCache.fetch(url) : fetch(url).then(function (r) {
        if (!r.ok) throw new Error(url.split("/").pop() + ": HTTP " + r.status);
        return r.arrayBuffer();
      }).then(function (buf) { return new Uint8Array(buf); });
      modelCache[url].catch(function () { delete modelCache[url]; });
    }
    return modelCache[url];
  }
  // Rename every tensor name in a graph (and nested subgraphs) through `map`.
  function renameGraph(graph, map) {
    (graph.node || []).forEach(function (node, i) {
      node.name = map(node.name || "node_" + i);
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
  // Bring a graph exported at opset <= 17 up to opset 18 (reduce ops' axes became an input; Split needs num_outputs).
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
  function valueInfo(name, dims) {
    return { name: name, type: { tensor_type: { elem_type: 1, shape: { dim: dims.map(function (d) { return typeof d === "number" ? { dim_value: d } : { dim_param: d }; }) } } } };
  }
  // Rewire every use of tensor `from` inside a graph (and subgraphs) to `to`.
  function rewire(graph, from, to) {
    (graph.node || []).forEach(function (node) {
      node.input = node.input.map(function (n) { return n === from ? to : n; });
      (node.attribute || []).forEach(function (attr) {
        if (attr.g) rewire(attr.g, from, to);
        (attr.graphs || []).forEach(function (g) { rewire(g, from, to); });
      });
    });
  }

  // ------------------------------------------------------------------ the graph model

  let registry = null;
  let modelsIndex = null;
  let ROOT = "";
  let mount = null;

  function opById(id) { return registry.ops.find(function (o) { return o.id === id; }); }
  function models() { return modelsIndex ? (modelsIndex.models || modelsIndex) : []; }
  function modelById(id) { return models().find(function (m) { return m.id === id || m.slug === id; }); }
  // operators a graph can hold: live graphs of image operators
  function eligible(op) { return op.mode === "onnx" && op.input_kind !== "volume"; }
  // models a graph can hold: the ones whose ONNX runs in the browser (server-only models stay on their own page)
  function graphModels() { return models().filter(function (m) { return m.url && m.output && m.output.kind !== "matches"; }); }
  function modelOutputType(m) {
    const k = m.output && m.output.kind;
    return k === "image" ? "image" : k === "depth" ? "depth" : "result";
  }

  // the typed ports of a node
  function portsOf(node) {
    if (node.kind === "input") return { inputs: [], outputs: [{ name: node.type, type: node.type }] };
    if (node.kind === "output") return { inputs: [{ name: "in", type: "any", accepts: Object.keys(TYPES) }], outputs: [] };
    if (node.kind === "model") {
      const m = modelById(node.model);
      if (!m) return { inputs: [], outputs: [] };
      const t = modelOutputType(m);
      return { inputs: [{ name: "image", type: "image", accepts: ["image"] }], outputs: [{ name: t, type: t }] };
    }
    const op = opById(node.op);
    if (!op) return { inputs: [], outputs: [] };
    const ports = op.ports || { inputs: [{ name: "image", type: "image", accepts: ["image"] }], outputs: [{ name: "image", type: "image" }] };
    return {
      inputs: ports.inputs.map(function (p) { return { name: p.name, type: p.type, optional: !!p.optional, accepts: p.accepts || ACCEPTS[p.type] || [p.type] }; }),
      outputs: ports.outputs.map(function (p) { return { name: p.name, type: p.type }; }),
    };
  }
  function nodeTitle(node) {
    if (node.kind === "input") return node.type;
    if (node.kind === "output") return node.label || "output";
    if (node.kind === "model") { const m = modelById(node.model); return m ? m.name : node.model; }
    const op = opById(node.op); return op ? op.name : node.op;
  }
  function edgeInto(pipe, nodeId, port) { return pipe.edges.find(function (e) { return e.to === nodeId && e.in === port; }); }
  function edgesFrom(pipe, nodeId) { return pipe.edges.filter(function (e) { return e.from === nodeId; }); }
  function nodeById(pipe, id) { return pipe.nodes.find(function (n) { return n.id === id; }); }
  // the type carried by an edge: the source port's type
  function edgeType(pipe, e) {
    const src = nodeById(pipe, e.from); if (!src) return null;
    const port = portsOf(src).outputs.find(function (p) { return p.name === e.out; });
    return port ? port.type : null;
  }
  function compatible(fromType, port) { return port.accepts.indexOf(fromType) !== -1; }
  function reaches(pipe, from, to) {   // is `to` downstream of `from`?
    const seen = {}; const stack = [from];
    while (stack.length) {
      const id = stack.pop();
      if (id === to) return true;
      if (seen[id]) continue; seen[id] = true;
      edgesFrom(pipe, id).forEach(function (e) { stack.push(e.to); });
    }
    return false;
  }
  // nodes in an order that respects every edge; throws on a cycle
  function topo(pipe) {
    const indeg = {}; pipe.nodes.forEach(function (n) { indeg[n.id] = 0; });
    pipe.edges.forEach(function (e) { if (indeg[e.to] !== undefined) indeg[e.to]++; });
    const ready = pipe.nodes.filter(function (n) { return indeg[n.id] === 0; }).map(function (n) { return n.id; });
    const order = [];
    while (ready.length) {
      const id = ready.shift(); order.push(id);
      edgesFrom(pipe, id).forEach(function (e) { if (--indeg[e.to] === 0) ready.push(e.to); });
    }
    if (order.length !== pipe.nodes.length) throw new Error("the graph has a cycle");
    return order.map(function (id) { return nodeById(pipe, id); });
  }
  // what stops the graph from running
  function problems(pipe) {
    const out = [];
    try { topo(pipe); } catch (e) { out.push(e.message); return out; }
    pipe.nodes.forEach(function (n) {
      portsOf(n).inputs.forEach(function (p) {
        if (!p.optional && !edgeInto(pipe, n.id, p.name)) out.push(nodeTitle(n) + ": " + (p.name === "in" ? "nothing is connected" : "input “" + p.name + "” is not connected"));
      });
      if (n.kind === "op" && !opById(n.op)) out.push(n.op + " is not in this build");
      if (n.kind === "model" && !modelById(n.model)) out.push(n.model + " is not in this build");
    });
    if (!pipe.nodes.some(function (n) { return n.kind === "output" && edgeInto(pipe, n.id, "in"); })) out.push("add an output node and connect it");
    return out;
  }
  // columns by depth, rows by order: for migrated and template graphs
  function autoLayout(pipe) {
    const depth = {};
    topo(pipe).forEach(function (n) {
      let d = 0;
      pipe.edges.filter(function (e) { return e.to === n.id; }).forEach(function (e) { d = Math.max(d, (depth[e.from] || 0) + 1); });
      depth[n.id] = d;
    });
    const rows = {};
    pipe.nodes.forEach(function (n) {
      const d = depth[n.id] || 0; rows[d] = (rows[d] || 0);
      n.x = 24 + d * 250; n.y = 24 + rows[d] * 150;
      rows[d]++;
    });
  }
  // version 1: {steps:[{op, params}], model} -> a chain
  function migrate(pipe) {
    if (!pipe || pipe.version === VERSION) return pipe;
    const nodes = [], edges = [];
    const input = { id: "in_" + uid(), kind: "input", type: "image" };
    nodes.push(input);
    let prev = { id: input.id, port: "image" };
    (pipe.steps || []).forEach(function (st) {
      const n = { id: "n_" + uid(), kind: "op", op: st.op, params: st.params || {} };
      nodes.push(n); edges.push({ from: prev.id, out: prev.port, to: n.id, in: "image" });
      prev = { id: n.id, port: "image" };
    });
    if (pipe.model) {
      const m = modelById(pipe.model);
      const n = { id: "m_" + uid(), kind: "model", model: m ? m.id : pipe.model };
      nodes.push(n); edges.push({ from: prev.id, out: prev.port, to: n.id, in: "image" });
      prev = { id: n.id, port: m ? modelOutputType(m) : "result" };
    }
    const out = { id: "out_" + uid(), kind: "output", label: "output" };
    nodes.push(out); edges.push({ from: prev.id, out: prev.port, to: out.id, in: "in" });
    const v2 = { id: pipe.id || uid(), name: pipe.name || "My pipeline", note: pipe.note || "", version: VERSION, created: pipe.created || Date.now(), nodes: nodes, edges: edges };
    try { autoLayout(v2); } catch (e) { /* leave positions */ }
    return v2;
  }

  // ------------------------------------------------------------------ composer

  // The op and model graphs are fetched by node, renamed with the node's id as prefix, and wired by the edges.
  // graph input and output names: by type, numbered when repeated; outputs by their label
  function ioNames(pipe) {
    const names = {}, used = {};
    const slug = function (t) { return String(t).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "output"; };
    const inputNames = {};
    pipe.nodes.forEach(function (n) {   // inputs first: image, image_2, mask, ...
      if (n.kind !== "input" || !edgesFrom(pipe, n.id).length) return;
      used[n.type] = (used[n.type] || 0) + 1;
      names[n.id] = used[n.type] > 1 ? n.type + "_" + used[n.type] : n.type;
      inputNames[names[n.id]] = true;
    });
    const outUsed = {};
    pipe.nodes.forEach(function (n) {   // outputs by label; a label that is also an input name gets _out
      if (n.kind !== "output" || !edgeInto(pipe, n.id, "in")) return;
      let base = slug(n.label || "output");
      if (inputNames[base]) base += "_out";
      outUsed[base] = (outUsed[base] || 0) + 1;
      names[n.id] = outUsed[base] > 1 ? base + "_" + outUsed[base] : base;
    });
    return names;
  }
  function compose(pipe, bake, preview, S) {
    const order = topo(pipe);
    const names = ioNames(pipe);
    const jobs = order.map(function (n) {
      if (n.kind === "op") {
        const op = opById(n.op);
        if (!op) throw new Error(n.op + " is not in this build");
        const multi = !!(op.graphs_multi && ["mask", "boxes", "keypoints"].some(function (m) { return edgeInto(pipe, n.id, m); }));
        const key = graphKey(op, n.params);
        const path = multi ? op.graphs_multi[key] : op.graphs[key];
        if (!path) throw new Error(op.name + ": no graph for these parameters");
        return fetchModel(ROOT + path).then(function (bytes) { return { node: n, op: op, multi: multi, bytes: bytes }; });
      }
      if (n.kind === "model") {
        const m = modelById(n.model);
        if (!m) throw new Error(n.model + " is not in this build");
        const url = /^https?:/.test(m.url) ? m.url : ROOT + m.url;
        return fetchModel(url).then(function (bytes) { return { node: n, model: m, bytes: bytes }; });
      }
      return Promise.resolve({ node: n });
    });
    return loadSchema(ROOT).then(function () { return Promise.all(jobs); }).then(function (parts) {
      const nodes = [], inputs = [], initializers = [], valueInfos = [], outputs = [];
      const paramInputs = [], graphInputs = [], graphOutputs = [], taps = {}, modelTaps = {};
      const tensorOf = {};   // nodeId -> {port: tensorName}
      const opsets = { "": 18 };
      let irVersion = 8;
      function sourceTensor(nodeId, port) {
        const e = edgeInto(pipe, nodeId, port);
        return e && tensorOf[e.from] ? tensorOf[e.from][e.out] : null;
      }
      parts.forEach(function (part) {
        const n = part.node;
        const prefix = n.id + "/";
        if (n.kind === "input") {
          if (!edgesFrom(pipe, n.id).length) return;   // unused inputs stay out of the graph
          const name = names[n.id];
          inputs.push(valueInfo(name, TYPES[n.type].dims(S)));
          graphInputs.push({ node: n.id, type: n.type, name: name });
          tensorOf[n.id] = {}; tensorOf[n.id][n.type] = name;
          return;
        }
        if (n.kind === "output") {
          const src = sourceTensor(n.id, "in");
          if (!src) return;
          const e = edgeInto(pipe, n.id, "in");
          const name = names[n.id];
          nodes.push({ op_type: "Identity", name: prefix + "id", input: [src], output: [name] });
          outputs.push({ name: name, type: { tensor_type: { elem_type: 1 } } });
          graphOutputs.push({ node: n.id, name: name, type: edgeType(pipe, e), from: e.from, fromPort: e.out });
          return;
        }
        const m = ModelProto.decode(part.bytes);
        const g = m.graph;
        irVersion = Math.max(irVersion, Number(m.ir_version));
        (m.opset_import || []).forEach(function (o) { const d = (!o.domain || o.domain === "ai.onnx") ? "" : o.domain; opsets[d] = Math.max(opsets[d] || 0, Number(o.version)); });
        const onnxOpset = (m.opset_import || []).filter(function (o) { return !o.domain || o.domain === "ai.onnx"; }).map(function (o) { return Number(o.version); })[0];
        renameGraph(g, function (x) { return prefix + x; });
        if (onnxOpset && onnxOpset < 18) liftTo18(g, prefix, initializers);
        const gInputNames = g.input.map(function (vi) { return vi.name; });
        tensorOf[n.id] = {};
        if (n.kind === "op") {
          const op = part.op;
          const inNames = part.multi ? ["image", "mask", "boxes", "keypoints"].map(function (x) { return prefix + x; }) : gInputNames;
          const img = sourceTensor(n.id, "image");
          if (!img) throw new Error(op.name + ": the image input is not connected");
          rewire(g, inNames[0], img);
          let k = 1;
          if (op.guidance && !part.multi) {
            const gp = (op.ports && op.ports.inputs[1] && op.ports.inputs[1].name) || "guide";
            const guide = sourceTensor(n.id, gp);
            if (!guide) throw new Error(op.name + ": the " + gp + " input is not connected");
            rewire(g, inNames[k++], guide);
          }
          if (part.multi) {
            // mask, boxes and keypoints: an edge, or a constant placeholder for the ports left free
            [["mask", [1, 1, S, S], null], ["boxes", [1, 1, 4], [0, 0, 1, 1]], ["keypoints", [1, 1, 2], [0, 0]]].forEach(function (spec, i) {
              const src = sourceTensor(n.id, spec[0]);
              const name = inNames[1 + i];
              if (src) { rewire(g, name, src); return; }
              const size = spec[1].reduce(function (a, b) { return a * b; }, 1);
              initializers.push({ dims: spec[1], data_type: 1, float_data: spec[2] || Array.from(new Float32Array(size)), name: name });
            });
            g.input = [];
            ["image", "mask", "boxes", "keypoints"].forEach(function (port, i) { tensorOf[n.id][port] = g.output[i].name; });
          } else {
            // the slider parameters: graph inputs, or baked
            const live = op.params.filter(function (p) { return p.kind === "live"; });
            gInputNames.slice(k).forEach(function (name, j) {
              const p = live[j];
              const value = p ? Number(n.params[p.name]) : 0;
              if (bake) initializers.push({ dims: [1], data_type: 1, float_data: [value], name: name });
              else { inputs.push(g.input.find(function (vi) { return vi.name === name; })); paramInputs.push({ name: name, node: n.id, param: p ? p.name : name }); }
            });
            g.input = [];
            tensorOf[n.id].image = g.output[0].name;
          }
          taps[n.id] = Object.assign({}, tensorOf[n.id]);
        } else {
          // a model: (x * scale - mean) / std before the network, as constants
          const model = part.model;
          const inp = model.input || {};
          let cur = sourceTensor(n.id, "image");
          if (!cur) throw new Error(model.name + ": the image input is not connected");
          const c = function (name, values, dims) { initializers.push({ dims: dims, data_type: 1, float_data: values, name: prefix + name }); return prefix + name; };
          const pre = cur;
          if (inp.scale && inp.scale !== 1) { nodes.push({ op_type: "Mul", name: prefix + "pre_scale", input: [cur, c("scale_const", [inp.scale], [1])], output: [prefix + "pre_scaled"] }); cur = prefix + "pre_scaled"; }
          if (inp.mean) { nodes.push({ op_type: "Sub", name: prefix + "pre_mean", input: [cur, c("mean_const", inp.mean, [1, 3, 1, 1])], output: [prefix + "pre_centred"] }); cur = prefix + "pre_centred"; }
          if (inp.std) { nodes.push({ op_type: "Div", name: prefix + "pre_std", input: [cur, c("std_const", inp.std, [1, 3, 1, 1])], output: [prefix + "pre_normed"] }); cur = prefix + "pre_normed"; }
          rewire(g, gInputNames[0], cur);
          g.input = [];
          const t = modelOutputType(model);
          tensorOf[n.id][t] = g.output[0].name;
          const outsByShort = {};
          g.output.forEach(function (o) { outsByShort[o.name.slice(prefix.length)] = o.name; });
          modelTaps[n.id] = { pre: pre, outputs: outsByShort, model: model };
          if (preview) g.output.forEach(function (o) { outputs.push({ name: o.name, type: { tensor_type: { elem_type: 1 } } }); });
        }
        nodes.push.apply(nodes, g.node);
        initializers.push.apply(initializers, g.initializer || []);
        valueInfos.push.apply(valueInfos, g.value_info || []);
      });
      if (preview) {
        // every operator's outputs and every model's input image, so the results panel can show any of them
        const seen = {}; outputs.forEach(function (o) { seen[o.name] = true; });
        Object.keys(taps).forEach(function (id) { Object.keys(taps[id]).forEach(function (port) { const t = taps[id][port]; if (!seen[t]) { seen[t] = true; outputs.push({ name: t, type: { tensor_type: { elem_type: 1 } } }); } }); });
        Object.keys(modelTaps).forEach(function (id) { const t = modelTaps[id].pre; if (!seen[t] && !graphInputs.some(function (gi) { return gi.name === t; })) { seen[t] = true; outputs.push({ name: t, type: { tensor_type: { elem_type: 1 } } }); } });
      }
      if (!outputs.length) throw new Error("connect an output");
      const model = ModelProto.create({
        ir_version: irVersion,
        producer_name: "kornia.org/playground",
        producer_version: "2",
        opset_import: Object.keys(opsets).map(function (d) { return { domain: d, version: opsets[d] }; }),
        graph: { name: "kornia_pipeline", node: nodes, input: inputs, output: outputs, initializer: initializers, value_info: valueInfos },
      });
      return { bytes: ModelProto.encode(model).finish(), inputs: graphInputs, paramInputs: paramInputs, outputs: graphOutputs, taps: taps, modelTaps: modelTaps };
    });
  }

  // ------------------------------------------------------------------ the JSON document: save to a file, load from one
  const FORMAT = "kornia-pipeline";
  function toDocument(pipe) {
    const nodes = pipe.nodes.map(function (n) {
      const o = { id: n.id, kind: n.kind };
      if (n.kind === "input") o.type = n.type;
      if (n.kind === "output") o.label = n.label || "output";
      if (n.kind === "op") { o.op = n.op; o.params = n.params || {}; }
      if (n.kind === "model") o.model = n.model;
      if (n.x !== undefined) { o.x = n.x; o.y = n.y; }
      return o;
    });
    return { format: FORMAT, version: VERSION, name: pipe.name, note: pipe.note || "", kornia: registry.kornia, generated: new Date().toISOString().slice(0, 10),
             nodes: nodes, edges: pipe.edges.map(function (e) { return { from: e.from, out: e.out, to: e.to, in: e.in }; }) };
  }
  function documentText(pipe) { return JSON.stringify(toDocument(pipe), null, 2); }
  // a parsed file -> a pipeline for this browser, or throws with a reason
  function fromDocument(docu) {
    if (!docu || typeof docu !== "object") throw new Error("not a JSON object");
    if (docu.steps && !docu.nodes) return migrate({ id: uid(), name: docu.name, note: docu.note, container: docu.container, steps: docu.steps, model: docu.model || null, created: Date.now() });
    if (!Array.isArray(docu.nodes) || !Array.isArray(docu.edges)) throw new Error("expected nodes and edges");
    const ids = {};
    docu.nodes.forEach(function (n, i) {
      if (!n || typeof n.id !== "string" || !n.id) throw new Error("node " + i + " has no id");
      if (ids[n.id]) throw new Error("node id " + n.id + " appears twice");
      ids[n.id] = true;
      if (n.kind === "input") { if (INPUT_KINDS.indexOf(n.type) === -1) throw new Error("input " + n.id + ": unknown type " + n.type); }
      else if (n.kind === "op") { if (!opById(n.op)) throw new Error("operator " + n.op + " is not in this build"); if (n.params && typeof n.params !== "object") throw new Error("op " + n.id + ": params must be an object"); }
      else if (n.kind === "model") { if (!modelById(n.model)) throw new Error("model " + n.model + " is not in this build"); }
      else if (n.kind !== "output") throw new Error("node " + n.id + ": unknown kind " + n.kind);
    });
    docu.edges.forEach(function (e, i) {
      if (!e || !ids[e.from] || !ids[e.to]) throw new Error("edge " + i + " names a node that does not exist");
      if (typeof e.out !== "string" || typeof e.in !== "string") throw new Error("edge " + i + " needs out and in port names");
    });
    const pipe = { id: uid(), name: String(docu.name || "Imported pipeline").slice(0, 60), note: String(docu.note || "").slice(0, 160), version: VERSION, created: Date.now(), updated: Date.now(),
                   nodes: docu.nodes.map(function (n) { const o = { id: n.id, kind: n.kind }; if (n.kind === "input") o.type = n.type; if (n.kind === "output") o.label = n.label || "output"; if (n.kind === "op") { const op = opById(n.op); o.op = n.op; o.params = Object.assign(defaults(op), n.params || {}); } if (n.kind === "model") o.model = modelById(n.model).id; if (typeof n.x === "number") { o.x = n.x; o.y = n.y; } return o; }),
                   edges: docu.edges.map(function (e) { return { from: e.from, out: e.out, to: e.to, in: e.in }; }) };
    topo(pipe);   // throws on a cycle
    if (!pipe.nodes.some(function (n) { return n.x !== undefined; })) autoLayout(pipe);
    return pipe;
  }
  function downloadJson(pipe) {
    const blob = new Blob([documentText(pipe)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = el("a", "", { href: url, download: pipe.name.replace(/[^\w.-]+/g, "_") + ".kornia-pipeline.json" });
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
  }
  // an "Import JSON" control: a button and a hidden file input; the file becomes a saved pipeline and opens
  function importControl(cls, onError) {
    const wrap = el("span", "pg-import");
    const btn = el("button", cls || "pg-btn pg-btn-ghost pg-btn-small", { type: "button", title: "Load a pipeline from a .json file saved here or written by hand" });
    btn.innerHTML = '<i class="fas fa-file-import" aria-hidden="true"></i> Import JSON';
    const file = el("input", "", { type: "file", accept: ".json,application/json" });
    file.hidden = true;
    btn.addEventListener("click", function () { file.value = ""; file.click(); });
    file.addEventListener("change", function () {
      const f = file.files && file.files[0]; if (!f) return;
      f.text().then(function (text) {
        const pipe = fromDocument(JSON.parse(text));
        const all = loadAll(); all.push(pipe); saveAll(all); syncSoon(pipe);
        navigate({ p: pipe.id });
      }).catch(function (e) { if (onError) onError("could not import " + f.name + ": " + (e.message || e)); else window.alert("could not import " + f.name + ": " + (e.message || e)); });
    });
    wrap.appendChild(btn); wrap.appendChild(file);
    return wrap;
  }

  // ------------------------------------------------------------------ templates

  function empty() { return { id: uid(), name: "My pipeline", version: VERSION, created: Date.now(), nodes: [], edges: [] }; }
  function addNode(p, n) { n.id = (n.kind === "input" ? "in_" : n.kind === "output" ? "out_" : n.kind === "model" ? "m_" : "n_") + uid(); p.nodes.push(n); return n; }
  function chain(opIds, modelSlug) {
    const p = empty();
    const i = addNode(p, { kind: "input", type: "image" });
    let prev = { id: i.id, port: "image" };
    opIds.forEach(function (id) {
      const op = opById(id);
      const n = addNode(p, { kind: "op", op: id, params: defaults(op) });
      p.edges.push({ from: prev.id, out: prev.port, to: n.id, in: "image" }); prev = { id: n.id, port: "image" };
    });
    if (modelSlug) {
      const m = modelById(modelSlug);
      const n = addNode(p, { kind: "model", model: m.id });
      p.edges.push({ from: prev.id, out: prev.port, to: n.id, in: "image" }); prev = { id: n.id, port: modelOutputType(m) };
    }
    const o = addNode(p, { kind: "output", label: "output" });
    p.edges.push({ from: prev.id, out: prev.port, to: o.id, in: "in" });
    return p;
  }
  const STARTERS = [
    { name: "Denoise, then edges", blurb: "Median blur to remove noise, then a Sobel edge map.", needs: ["kornia.filters.median_blur", "kornia.filters.sobel"],
      build: function () { return chain(["kornia.filters.median_blur", "kornia.filters.sobel"]); } },
    { name: "Augment image, mask, boxes and keypoints", blurb: "One random affine draw moves all four together, the way training data is augmented.", needs: ["kornia.augmentation.RandomAffine"], needsMulti: true,
      build: function () {
        const p = empty();
        const aug = addNode(p, { kind: "op", op: "kornia.augmentation.RandomAffine", params: defaults(opById("kornia.augmentation.RandomAffine")) });
        INPUT_KINDS.forEach(function (t) {
          const i = addNode(p, { kind: "input", type: t }); p.edges.push({ from: i.id, out: t, to: aug.id, in: t });
          const o = addNode(p, { kind: "output", label: t }); p.edges.push({ from: aug.id, out: t, to: o.id, in: "in" });
        });
        return p;
      } },
    { name: "Two branches", blurb: "The same image blurred on one branch and Laplacian-filtered on the other; two outputs.", needs: ["kornia.filters.gaussian_blur2d", "kornia.filters.laplacian"],
      build: function () {
        const p = empty();
        const i = addNode(p, { kind: "input", type: "image" });
        const a = addNode(p, { kind: "op", op: "kornia.filters.gaussian_blur2d", params: defaults(opById("kornia.filters.gaussian_blur2d")) });
        const b = addNode(p, { kind: "op", op: "kornia.filters.laplacian", params: defaults(opById("kornia.filters.laplacian")) });
        const oa = addNode(p, { kind: "output", label: "blurred" }), ob = addNode(p, { kind: "output", label: "edges" });
        p.edges.push({ from: i.id, out: "image", to: a.id, in: "image" }, { from: i.id, out: "image", to: b.id, in: "image" },
                     { from: a.id, out: "image", to: oa.id, in: "in" }, { from: b.id, out: "image", to: ob.id, in: "in" });
        return p;
      } },
    { name: "Blur, then find faces", blurb: "A Gaussian blur in front of YuNet: does the detector still find the faces?", needs: ["kornia.filters.gaussian_blur2d"], model: "face-detection-yunet",
      build: function () { return chain(["kornia.filters.gaussian_blur2d"], "face-detection-yunet"); } },
  ];
  function availableStarters() {
    return STARTERS.filter(function (st) {
      const ops = (st.needs || []).map(opById);
      if (ops.some(function (o) { return !o || !eligible(o); })) return false;
      if (st.needsMulti && !ops.every(function (o) { return o.graphs_multi; })) return false;
      if (st.model && !graphModels().some(function (m) { return m.slug === st.model; })) return false;
      return true;
    });
  }
  function starterPipe(st) { const p = st.build(); p.name = st.name; autoLayout(p); return p; }
  function summary(pipe) {
    try { return topo(pipe).filter(function (n) { return n.kind === "op" || n.kind === "model"; }).map(nodeTitle).join(" → ") || "empty"; } catch (e) { return "cycle"; }
  }
  function counts(pipe) { return String(pipe.nodes.filter(function (x) { return x.kind === "op" || x.kind === "model"; }).length); }

  // ------------------------------------------------------------------ routing and the sidebar

  let draft = null;
  let sideRows = [];
  function navigate(params) {
    const url = new URL(window.location.href);
    url.search = "";
    for (const k in params) url.searchParams.set(k, params[k]);
    window.history.pushState({}, "", url.toString());
    render();
  }
  function createFromStarter(st) {
    draft = starterPipe(st); draft.template = String(STARTERS.indexOf(st));
    navigate({ t: draft.template });
  }
  function pipelineLinks() {
    const links = el("div", "pg-details pg-model-card pg-pipe-links");
    const row = el("div", "pg-details-links");
    row.innerHTML = '<a href="https://kornia.readthedocs.io/en/latest/augmentation.container.html" target="_blank" rel="noopener"><i class="fas fa-book" aria-hidden="true"></i> Containers in kornia</a>'
      + '<a href="https://github.com/kornia/kornia.github.io/issues/new?title=playground%20pipelines" target="_blank" rel="noopener"><i class="fas fa-bug" aria-hidden="true"></i> Report an issue</a>';
    links.appendChild(row);
    return links;
  }
  function renderSidebar() {
    const side = document.getElementById("pg-pipes-list");
    if (!side) return;
    side.innerHTML = "";
    const q = new URLSearchParams(window.location.search);
    const current = q.get("p");
    const rows = [];
    function row(list, cls, title, sub, right, onClick, isCurrent, text) {
      const li = el("li", "pg-row pg-row-live");
      const a = el("a", "pg-row-name pg-row-model" + (cls ? " " + cls : ""), { href: "#" });
      const t = el("span", "pg-model-title"); t.textContent = title; a.appendChild(t);
      if (sub) { const sb = el("span", "pg-model-sub"); sb.textContent = sub; a.appendChild(sb); }
      if (right) { const r = el("span", "pg-model-size"); r.textContent = right; a.appendChild(r); }
      if (isCurrent) { a.classList.add("pg-row-current"); a.setAttribute("aria-current", "page"); }
      a.addEventListener("click", function (ev) { ev.preventDefault(); onClick(); });
      li.appendChild(a); list.appendChild(li);
      rows.push({ li: li, text: (text || title + " " + (sub || "")).toLowerCase() });
      return li;
    }
    const mine = el("section", "pg-pkg");
    const h = el("h2"); h.textContent = "Your pipelines";
    const all = loadAll();
    const n = el("span", "pg-pkg-count"); n.textContent = String(all.length); h.appendChild(n);
    mine.appendChild(h);
    const list = el("ul", "pg-list");
    row(list, "pg-row-new", "+ New pipeline", null, null, function () { newDraft(); }, false, "new pipeline");
    const impLi = el("li", "pg-row pg-row-live pg-row-import"); impLi.appendChild(importControl("pg-link")); list.appendChild(impLi);
    all.slice().sort(function (a, b) { return (b.created || 0) - (a.created || 0); }).forEach(function (pipe) {
      const li = row(list, "", pipe.name, pipe.note || summary(pipe), pipe.synced ? "☁" : "", function () { navigate({ p: pipe.id }); }, pipe.id === current);
      const tools = el("span", "pg-row-tools");
      const dup = el("button", "pg-row-tool", { type: "button", title: "Duplicate", "aria-label": "Duplicate " + pipe.name });
      dup.innerHTML = '<i class="fas fa-copy" aria-hidden="true"></i>';
      dup.addEventListener("click", function (ev) {
        ev.preventDefault(); ev.stopPropagation();
        const copy = JSON.parse(JSON.stringify(pipe));
        copy.id = uid(); copy.name = pipe.name + " (copy)"; copy.created = Date.now(); copy.updated = Date.now(); delete copy.synced;
        const list2 = loadAll(); list2.push(copy); saveAll(list2); syncSoon(copy);
        navigate({ p: copy.id });
      });
      const del = el("button", "pg-row-tool pg-row-tool-danger", { type: "button", title: "Delete", "aria-label": "Delete " + pipe.name });
      del.innerHTML = '<i class="fas fa-trash" aria-hidden="true"></i>';
      del.addEventListener("click", function (ev) {
        ev.preventDefault(); ev.stopPropagation();
        if (!window.confirm('Delete "' + pipe.name + '"? This cannot be undone.')) return;
        saveAll(loadAll().filter(function (x) { return x.id !== pipe.id; }));
        if (pipe.synced) unsync(pipe.id);
        if (pipe.id === current) navigate({}); else renderSidebar();
      });
      tools.appendChild(dup); tools.appendChild(del);
      li.appendChild(tools);
    });
    mine.appendChild(list);
    side.appendChild(mine);
    const starters = availableStarters();
    if (starters.length) {
      const sec = el("section", "pg-pkg");
      const h2 = el("h2"); h2.textContent = "Templates";
      const n2 = el("span", "pg-pkg-count"); n2.textContent = String(starters.length); h2.appendChild(n2);
      sec.appendChild(h2);
      const l2 = el("ul", "pg-list");
      starters.forEach(function (st) { row(l2, "", st.name, st.blurb, null, function () { createFromStarter(st); }, q.get("t") === String(STARTERS.indexOf(st))); });
      sec.appendChild(l2);
      side.appendChild(sec);
    }
    const search = document.getElementById("pg-search");
    if (search && !search.dataset.pipesBound) {
      search.dataset.pipesBound = "1";
      search.addEventListener("input", function () {
        const qq = search.value.trim().toLowerCase();
        sideRows.forEach(function (r) { r.li.hidden = !!qq && r.text.indexOf(qq) === -1; });
      });
    }
    sideRows = rows;
    if (search && search.value) search.dispatchEvent(new Event("input"));
  }
  function newDraft() {
    draft = empty();
    const i = addNode(draft, { kind: "input", type: "image" }); i.x = 24; i.y = 60;
    const o = addNode(draft, { kind: "output", label: "output" }); o.x = 560; o.y = 60;
    navigate({ draft: "1" });
  }

  function render() {
    const q = new URLSearchParams(window.location.search);
    mount.innerHTML = "";
    renderSidebar();
    if (q.has("t")) {
      const st = STARTERS[Number(q.get("t"))];
      if (st && availableStarters().indexOf(st) !== -1) {
        if (!draft || draft.template !== q.get("t")) { draft = starterPipe(st); draft.template = q.get("t"); }
        return renderEditor(draft, true);
      }
    }
    if (q.has("draft")) { if (draft) return renderEditor(draft, true); return renderList(); }
    if (q.get("share")) {
      try {
        const pipe = migrate(JSON.parse(unb64url(q.get("share"))));
        pipe.id = uid(); pipe.name = pipe.name || "Shared pipeline"; pipe.created = Date.now();
        const list = loadAll(); list.push(pipe); saveAll(list);
        navigate({ p: pipe.id });
        return;
      } catch (e) { /* fall through */ }
    }
    if (q.get("p")) {
      const pipe = loadAll().find(function (x) { return x.id === q.get("p"); });
      if (pipe) return renderEditor(pipe, false);
    }
    renderList();
  }

  // ------------------------------------------------------------------ list

  function renderList() {
    const head = el("div", "pg-pipe-head");
    const h = el("h2"); h.textContent = "Pipelines"; head.appendChild(h);
    mount.appendChild(head);
    const lead = el("p", "pg-summary");
    lead.textContent = "Wire kornia operators and models into a graph: several inputs (an image, its mask, boxes, keypoints), branches that split and join, several outputs. It runs on the samples here, then downloads as one ONNX graph or as PyTorch code.";
    mount.appendChild(lead);
    const grid = el("div", "pg-pipe-grid");
    const create = el("button", "pg-pipe-card pg-pipe-create pg-pipe-create-btn", { type: "button" });
    create.innerHTML = '<i class="fas fa-plus" aria-hidden="true"></i> Create a pipeline';
    create.addEventListener("click", newDraft);
    grid.appendChild(create);
    const imp = el("div", "pg-pipe-card pg-pipe-import");
    imp.appendChild(importControl("pg-btn pg-btn-ghost"));
    const impNote = el("p", "pg-note"); impNote.textContent = "A .json file saved from the editor, or written by hand: nodes with typed ports and the edges between them.";
    imp.appendChild(impNote);
    grid.appendChild(imp);
    mount.appendChild(grid);
    const starters = availableStarters();
    if (starters.length) {
      const sec = el("section", "pg-pipe-templates", { id: "templates" });
      const h3 = el("h3"); h3.textContent = "Start from a template"; sec.appendChild(h3);
      const row = el("div", "pg-pipe-template-grid");
      starters.forEach(function (st) {
        const card = el("button", "pg-pipe-template", { type: "button" });
        const t = el("strong"); t.textContent = st.name;
        const chainEl = el("span", "pg-pipe-steps"); chainEl.textContent = summary(st.build());
        const b = el("span", "pg-note"); b.textContent = st.blurb;
        card.appendChild(t); card.appendChild(chainEl); card.appendChild(b);
        card.addEventListener("click", function () { createFromStarter(st); });
        row.appendChild(card);
      });
      sec.appendChild(row);
      mount.appendChild(sec);
    }
    mount.appendChild(pipelineLinks());
  }

  // ------------------------------------------------------------------ editor

  function renderEditor(pipe, isDraft) {
    let unsaved = !!isDraft;
    pipe.nodes.forEach(function (n, i) { if (n.x === undefined) { n.x = 24 + (i % 4) * 250; n.y = 24 + Math.floor(i / 4) * 150; } });
    const S = registry.size;

    const snap = function () { return JSON.stringify({ nodes: pipe.nodes, edges: pipe.edges }); };
    let lastSnap = snap();
    const undoStack = [], redoStack = [];
    function record() {
      const now = snap();
      if (now === lastSnap) return;
      undoStack.push(lastSnap); if (undoStack.length > 100) undoStack.shift();
      redoStack.length = 0;
      lastSnap = now;
      refreshHistoryButtons();
    }
    function restore(json) {
      const st = JSON.parse(json);
      pipe.nodes = st.nodes; pipe.edges = st.edges;
      lastSnap = json;
      saveAll(loadAll().map(function (x) { return x.id === pipe.id ? pipe : x; }));
      rebuild(); refreshHistoryButtons();
    }
    function undo() { if (!undoStack.length) return; redoStack.push(lastSnap); restore(undoStack.pop()); }
    function redo() { if (!redoStack.length) return; undoStack.push(lastSnap); restore(redoStack.pop()); }
    function refreshHistoryButtons() { if (typeof undoBtn !== "undefined") { undoBtn.disabled = !undoStack.length; redoBtn.disabled = !redoStack.length; } }
    function onKey(ev) {
      if (!mount.contains(pane)) { document.removeEventListener("keydown", onKey); return; }
      const t = ev.target; if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
      if (!(ev.ctrlKey || ev.metaKey)) return;
      if (ev.key === "z" || ev.key === "Z") { ev.preventDefault(); if (ev.shiftKey) redo(); else undo(); }
      else if (ev.key === "y") { ev.preventDefault(); redo(); }
    }
    document.addEventListener("keydown", onKey);
    function persist() {
      record();
      const all = loadAll();
      const i = all.findIndex(function (x) { return x.id === pipe.id; });
      pipe.updated = Date.now();
      if (i >= 0) all[i] = pipe; else all.push(pipe);
      delete pipe.template;
      saveAll(all);
      syncSoon(pipe);
      if (unsaved) {
        unsaved = false; draft = null;
        const url = new URL(window.location.href); url.search = "?p=" + pipe.id;
        window.history.replaceState({}, "", url.toString());
        if (draftNote) draftNote.remove();
        del.hidden = false;
      }
      renderSidebar();
    }

    // ---- head
    const head = el("div", "pg-pipe-head");
    const nameInput = el("input", "pg-input pg-pipe-title", { type: "text", value: pipe.name, maxlength: "60", "aria-label": "Pipeline name" });
    nameInput.addEventListener("change", function () { pipe.name = nameInput.value.trim() || "My pipeline"; persist(); refreshCode(); });
    head.appendChild(nameInput);
    const headActions = el("div", "pg-pipe-head-actions");
    const dup = el("button", "pg-link", { type: "button", title: "Duplicate this pipeline" });
    dup.innerHTML = '<i class="fas fa-copy" aria-hidden="true"></i> Duplicate';
    dup.addEventListener("click", function () {
      const copy = JSON.parse(JSON.stringify(pipe));
      copy.id = uid(); copy.name = pipe.name + " (copy)"; copy.created = Date.now(); copy.updated = Date.now(); delete copy.synced;
      const all = loadAll(); all.push(copy); saveAll(all); syncSoon(copy);
      navigate({ p: copy.id });
    });
    const del = el("button", "pg-link pg-link-danger", { type: "button", title: "Delete this pipeline" });
    del.innerHTML = '<i class="fas fa-trash" aria-hidden="true"></i> Delete';
    del.hidden = unsaved;
    del.addEventListener("click", function () {
      if (!window.confirm('Delete "' + pipe.name + '"? This cannot be undone.')) return;
      saveAll(loadAll().filter(function (x) { return x.id !== pipe.id; }));
      if (pipe.synced) unsync(pipe.id);
      navigate({});
    });
    const share = el("button", "pg-link", { type: "button", title: "Copy a link that carries this pipeline" });
    share.innerHTML = '<i class="fas fa-link" aria-hidden="true"></i> Share link';
    share.addEventListener("click", function () {
      const url = new URL(window.location.href); url.search = "?share=" + b64url(JSON.stringify({ name: pipe.name, version: VERSION, nodes: pipe.nodes, edges: pipe.edges }));
      navigator.clipboard.writeText(url.toString()).then(function () { share.innerHTML = '<i class="fas fa-check" aria-hidden="true"></i> Copied'; setTimeout(function () { share.innerHTML = '<i class="fas fa-link" aria-hidden="true"></i> Share link'; }, 1500); });
    });
    const exp = el("button", "pg-link", { type: "button", title: "Save this pipeline as a .json file" });
    exp.innerHTML = '<i class="fas fa-file-arrow-down" aria-hidden="true"></i> Export JSON';
    exp.addEventListener("click", function () { downloadJson(pipe); });
    headActions.appendChild(exp); headActions.appendChild(share); headActions.appendChild(dup); headActions.appendChild(del);
    head.appendChild(headActions);
    mount.appendChild(head);
    const noteInput = el("input", "pg-input pg-pipe-note", { type: "text", value: pipe.note || "", maxlength: "160", placeholder: "What is this pipeline for? A comment shown in your lists.", "aria-label": "Comment" });
    noteInput.addEventListener("change", function () { pipe.note = noteInput.value.trim(); persist(); });
    mount.appendChild(noteInput);
    let draftNote = null;
    if (unsaved) {
      draftNote = el("p", "pg-note pg-pipe-draft");
      draftNote.innerHTML = '<i class="fas fa-circle-info" aria-hidden="true"></i> Not saved yet: it joins your pipelines the first time you change something in it.';
      mount.appendChild(draftNote);
    }

    const layout = el("div", "pg-pipe-layout");
    mount.appendChild(layout);

    // ---- toolbar: add nodes
    const bar = el("div", "pg-graph-bar");
    function addSelect(label, groups, onPick) {
      const sel = el("select", "pg-select", { "aria-label": label });
      const ph = el("option", "", { value: "" }); ph.textContent = label; sel.appendChild(ph);
      groups.forEach(function (gr) {
        const og = el("optgroup", "", { label: gr.label });
        gr.items.forEach(function (it) { const o = el("option", "", { value: it.value }); o.textContent = it.text; og.appendChild(o); });
        sel.appendChild(og);
      });
      sel.addEventListener("change", function () { if (sel.value) { onPick(sel.value); sel.value = ""; } });
      bar.appendChild(sel);
      return sel;
    }
    addSelect("+ Input", [{ label: "Inputs", items: INPUT_KINDS.map(function (t) { return { value: t, text: t + (t === "image" ? " (the sample, your image or a clip)" : " (the sample's annotations)") }; }) }], function (t) {
      const n = addNode(pipe, { kind: "input", type: t }); place(n); persist(); rebuild();
    });
    // a searchable picker: a button opens a panel with a search box and the grouped list; typing filters it
    function picker(label, groups, onPick) {
      const wrap = el("div", "pg-picker");
      const btn = el("button", "pg-btn pg-btn-ghost pg-btn-small", { type: "button", "aria-haspopup": "listbox", "aria-expanded": "false" });
      btn.innerHTML = '<i class="fas fa-plus" aria-hidden="true"></i> ' + label;
      const panel = el("div", "pg-picker-panel"); panel.hidden = true;
      const search = el("input", "pg-input pg-input-small", { type: "search", placeholder: "search " + label.toLowerCase() + "s…", "aria-label": "Search " + label.toLowerCase() + "s" });
      const list = el("div", "pg-picker-list", { role: "listbox" });
      panel.appendChild(search); panel.appendChild(list);
      const rows = [];
      groups.forEach(function (gr) {
        const h = el("div", "pg-picker-group"); h.textContent = gr.label; list.appendChild(h);
        gr.items.forEach(function (it) {
          const r = el("button", "pg-picker-item", { type: "button", role: "option" });
          r.innerHTML = "<strong>" + it.text + "</strong>" + (it.sub ? '<span class="pg-note">' + it.sub + "</span>" : "");
          r.addEventListener("click", function () { close(); onPick(it.value); });
          list.appendChild(r); rows.push({ el: r, head: h, text: (it.text + " " + (it.sub || "") + " " + gr.label).toLowerCase() });
        });
      });
      function filter() {
        const q = search.value.trim().toLowerCase();
        const visibleHeads = new Set();
        rows.forEach(function (r) { const on = !q || r.text.indexOf(q) !== -1; r.el.hidden = !on; if (on) visibleHeads.add(r.head); });
        list.querySelectorAll(".pg-picker-group").forEach(function (h) { h.hidden = !visibleHeads.has(h); });
      }
      search.addEventListener("input", filter);
      search.addEventListener("keydown", function (ev) {
        if (ev.key === "Escape") { close(); btn.focus(); }
        if (ev.key === "Enter") { const first = rows.find(function (r) { return !r.el.hidden; }); if (first) first.el.click(); }
      });
      function open() { panel.hidden = false; btn.setAttribute("aria-expanded", "true"); search.value = ""; filter(); search.focus(); document.addEventListener("pointerdown", outside); }
      function close() { panel.hidden = true; btn.setAttribute("aria-expanded", "false"); document.removeEventListener("pointerdown", outside); }
      function outside(ev) { if (!wrap.contains(ev.target)) close(); }
      btn.addEventListener("click", function () { if (panel.hidden) open(); else close(); });
      wrap.appendChild(btn); wrap.appendChild(panel);
      bar.appendChild(wrap);
      return wrap;
    }
    picker("Operator", registry.packages.map(function (pkg) {
      return { label: pkg.title, items: registry.ops.filter(function (o) { return eligible(o) && o.package === pkg.id; }).map(function (o) { return { value: o.id, text: o.name, sub: o.graphs_multi ? "mask, boxes, keypoints" : o.guidance ? "two images" : (o.summary || "").split(/[.;]/)[0] }; }) };
    }).filter(function (g) { return g.items.length; }), function (id) {
      const n = addNode(pipe, { kind: "op", op: id, params: defaults(opById(id)) }); place(n); persist(); rebuild();
    });
    if (graphModels().length) picker("Model", [{ label: "Models, run in the browser", items: graphModels().map(function (m) { return { value: m.id, text: m.name, sub: (m.task || "") + " · " + modelOutputType(m) }; }) }], function (id) {
      const n = addNode(pipe, { kind: "model", model: id }); place(n); persist(); rebuild();
    });
    const addOut = el("button", "pg-btn pg-btn-ghost pg-btn-small", { type: "button" });
    addOut.innerHTML = '<i class="fas fa-plus" aria-hidden="true"></i> Output';
    addOut.addEventListener("click", function () { const n = addNode(pipe, { kind: "output", label: "output " + (pipe.nodes.filter(function (x) { return x.kind === "output"; }).length + 1) }); place(n); persist(); rebuild(); });
    bar.appendChild(addOut);
    const tidy = el("button", "pg-btn pg-btn-ghost pg-btn-small", { type: "button", title: "Arrange the nodes by depth" });
    tidy.innerHTML = '<i class="fas fa-wand-magic-sparkles" aria-hidden="true"></i> Tidy';
    tidy.addEventListener("click", function () { try { autoLayout(pipe); persist(); rebuild(); } catch (e) { /* cycle */ } });
    bar.appendChild(tidy);
    const spacer = el("span", "pg-graph-hint");
    bar.appendChild(spacer);
    const undoBtn = el("button", "pg-btn pg-btn-ghost pg-btn-small", { type: "button", title: "Undo (Ctrl+Z)" });
    undoBtn.innerHTML = '<i class="fas fa-rotate-left" aria-hidden="true"></i>';
    undoBtn.addEventListener("click", function () { undo(); });
    const redoBtn = el("button", "pg-btn pg-btn-ghost pg-btn-small", { type: "button", title: "Redo (Ctrl+Shift+Z)" });
    redoBtn.innerHTML = '<i class="fas fa-rotate-right" aria-hidden="true"></i>';
    redoBtn.addEventListener("click", function () { redo(); });
    bar.appendChild(undoBtn); bar.appendChild(redoBtn);
    const run = el("button", "pg-btn pg-btn-small pg-graph-run", { type: "button" });
    run.innerHTML = '<i class="fas fa-play" aria-hidden="true"></i> Run';
    run.addEventListener("click", function () { explicit = true; schedule(false); });
    bar.appendChild(run);
    layout.appendChild(bar);
    function place(n) {   // a free spot in the visible part of the pane: inputs left, operators and models in the middle, outputs right
      const left = pane.scrollLeft + 24, width = Math.max(600, pane.clientWidth);
      n.x = n.kind === "input" ? left : n.kind === "output" ? left + Math.max(520, width - 280) : left + Math.max(260, Math.round((width - 220) / 2));
      let y = pane.scrollTop + 24;
      const taken = function (yy) { return pipe.nodes.some(function (o) { return o !== n && Math.abs((o.x || 0) - n.x) < 200 && Math.abs((o.y || 0) - yy) < 130; }); };
      while (taken(y)) y += 150;
      n.y = y;
    }

    // ---- the graph pane
    const pane = el("div", "pg-graph", { tabindex: "0", "aria-label": "Pipeline graph" });
    const wires = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    wires.setAttribute("class", "pg-wires");
    pane.appendChild(wires);
    const nodesLayer = el("div", "pg-nodes");
    pane.appendChild(nodesLayer);
    layout.appendChild(pane);
    const problemsBox = el("ul", "pg-pipe-warnings");
    layout.appendChild(problemsBox);

    function portEl(nodeId, dir, name) { return nodesLayer.querySelector('.pg-port[data-node="' + nodeId + '"][data-dir="' + dir + '"][data-port="' + name + '"]'); }
    function portCenter(p) {
      const dot = p.querySelector(".pg-port-dot"), r = dot.getBoundingClientRect(), pr = pane.getBoundingClientRect();
      return { x: r.left + r.width / 2 - pr.left + pane.scrollLeft, y: r.top + r.height / 2 - pr.top + pane.scrollTop };
    }
    let tempWire = null;
    function drawWires() {
      while (wires.firstChild) wires.removeChild(wires.firstChild);
      let w = pane.clientWidth, h = pane.clientHeight;
      pipe.nodes.forEach(function (n) { w = Math.max(w, (n.x || 0) + 260); h = Math.max(h, (n.y || 0) + 200); });
      nodesLayer.style.width = w + "px"; nodesLayer.style.height = h + "px";
      wires.setAttribute("width", w); wires.setAttribute("height", h);
      pipe.edges.forEach(function (e) {
        const a = portEl(e.from, "out", e.out), b = portEl(e.to, "in", e.in);
        if (!a || !b) return;
        const p1 = portCenter(a), p2 = portCenter(b);
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        const dx = Math.max(40, Math.abs(p2.x - p1.x) / 2);
        path.setAttribute("d", "M" + p1.x + "," + p1.y + " C" + (p1.x + dx) + "," + p1.y + " " + (p2.x - dx) + "," + p2.y + " " + p2.x + "," + p2.y);
        path.setAttribute("class", "pg-wire");
        path.setAttribute("stroke", (TYPES[edgeType(pipe, e)] || TYPES.image).color);
        path.addEventListener("click", function () { pipe.edges = pipe.edges.filter(function (x) { return x !== e; }); persist(); rebuild(); });
        const title = document.createElementNS("http://www.w3.org/2000/svg", "title"); title.textContent = "remove this wire"; path.appendChild(title);
        wires.appendChild(path);
      });
      if (tempWire) wires.appendChild(tempWire);
    }

    function paramControls(n, op) {
      const box = el("div", "pg-node-params");
      op.params.forEach(function (p) {
        const row = el("label", "pg-node-param");
        const name = el("span"); name.textContent = p.label || p.name; row.appendChild(name);
        if (p.kind === "select") {
          const input = el("select", "pg-select pg-select-small");
          p.choices.forEach(function (c, i) { const o = el("option", "", { value: c }); o.textContent = p.labels ? p.labels[i] : fmt(c, p); if (String(c) === String(n.params[p.name])) o.selected = true; input.appendChild(o); });
          input.addEventListener("change", function () { n.params[p.name] = p.type === "str" ? input.value : Number(input.value); persist(); schedule(true); });
          row.appendChild(input);
        } else {
          const input = el("input", "", { type: "range", min: p.min, max: p.max, step: p.step, value: n.params[p.name] });
          const out = el("output"); out.textContent = fmt(n.params[p.name], p);
          input.addEventListener("input", function () { n.params[p.name] = Number(input.value); out.textContent = fmt(n.params[p.name], p); persist(); schedule(false); });
          row.appendChild(input); row.appendChild(out);
        }
        row.addEventListener("pointerdown", function (ev) { ev.stopPropagation(); });
        box.appendChild(row);
      });
      return box;
    }
    function buildNode(n) {
      const ports = portsOf(n);
      const card = el("div", "pg-node pg-node-" + n.kind, { "data-id": n.id });
      card.style.left = (n.x || 0) + "px"; card.style.top = (n.y || 0) + "px";
      const headEl = el("div", "pg-node-head");
      const t = el("span", "pg-node-title"); t.textContent = nodeTitle(n);
      headEl.appendChild(t);
      if (n.kind === "op") { const c = el("code", "pg-node-id"); c.textContent = ((opById(n.op) || {}).package || "").replace("kornia.", ""); headEl.appendChild(c); }
      if (n.kind === "model") { const c = el("span", "pg-tag pg-tag-task pg-tag-small"); c.textContent = (modelById(n.model) || {}).task || "model"; headEl.appendChild(c); }
      if (n.kind === "input") { const c = el("span", "pg-tag pg-tag-small"); c.textContent = "input"; headEl.appendChild(c); }
      const rm = el("button", "pg-node-remove", { type: "button", title: "Remove", "aria-label": "Remove " + nodeTitle(n) });
      rm.innerHTML = '<i class="fas fa-xmark" aria-hidden="true"></i>';
      rm.addEventListener("pointerdown", function (ev) { ev.stopPropagation(); });
      rm.addEventListener("click", function () { pipe.nodes = pipe.nodes.filter(function (x) { return x !== n; }); pipe.edges = pipe.edges.filter(function (e) { return e.from !== n.id && e.to !== n.id; }); persist(); rebuild(); });
      headEl.appendChild(rm);
      card.appendChild(headEl);
      const body = el("div", "pg-node-body");
      const inCol = el("div", "pg-ports pg-ports-in"), outCol = el("div", "pg-ports pg-ports-out");
      ports.inputs.forEach(function (p) {
        const pe = el("div", "pg-port" + (p.optional ? " pg-port-optional" : ""), { "data-node": n.id, "data-dir": "in", "data-port": p.name, title: p.name + ": " + (p.accepts || [p.type]).join(" or ") + (p.optional ? " (optional)" : "") });
        const dot = el("span", "pg-port-dot"); dot.style.borderColor = (TYPES[p.type] || TYPES.image).color;
        const e = edgeInto(pipe, n.id, p.name);
        if (e) dot.style.background = (TYPES[edgeType(pipe, e)] || TYPES.image).color;
        pe.appendChild(dot); const l = el("span", "pg-port-name"); l.textContent = p.name === "in" ? "" : p.name; pe.appendChild(l);
        inCol.appendChild(pe);
      });
      ports.outputs.forEach(function (p) {
        const pe = el("div", "pg-port", { "data-node": n.id, "data-dir": "out", "data-port": p.name, title: p.name + ": " + p.type });
        const l = el("span", "pg-port-name"); l.textContent = p.name; pe.appendChild(l);
        const dot = el("span", "pg-port-dot"); dot.style.borderColor = TYPES[p.type].color; dot.style.background = TYPES[p.type].color; pe.appendChild(dot);
        outCol.appendChild(pe);
      });
      body.appendChild(inCol);
      const mid = el("div", "pg-node-mid");
      if (n.kind === "op") { const op = opById(n.op); if (op) mid.appendChild(paramControls(n, op)); }
      if (n.kind === "output") {
        const label = el("input", "pg-input pg-input-small", { type: "text", value: n.label || "output", maxlength: "24", "aria-label": "Output name" });
        label.addEventListener("pointerdown", function (ev) { ev.stopPropagation(); });
        label.addEventListener("change", function () { n.label = label.value.trim() || "output"; persist(); rebuild(); });
        mid.appendChild(label);
      }
      if (n.kind === "input" && n.type !== "image") { const note = el("span", "pg-note"); note.textContent = "the sample's " + n.type; mid.appendChild(note); }
      if (n.kind === "model") { const m = modelById(n.model); if (m) { const a = el("a", "pg-note", { href: ROOT + "models/" + m.slug + "/", target: "_blank", rel: "noopener" }); a.textContent = m.subtitle || m.task; mid.appendChild(a); } }
      body.appendChild(mid);
      body.appendChild(outCol);
      card.appendChild(body);
      // drag to move
      headEl.addEventListener("pointerdown", function (ev) {
        if (ev.button !== 0) return;
        ev.preventDefault();
        const start = { x: ev.clientX, y: ev.clientY, nx: n.x || 0, ny: n.y || 0 };
        function move(e2) { n.x = Math.max(0, start.nx + e2.clientX - start.x); n.y = Math.max(0, start.ny + e2.clientY - start.y); card.style.left = n.x + "px"; card.style.top = n.y + "px"; drawWires(); }
        function up() { document.removeEventListener("pointermove", move); document.removeEventListener("pointerup", up); persist(); }
        document.addEventListener("pointermove", move); document.addEventListener("pointerup", up);
      });
      return card;
    }
    // connect: drag from an output port to an input port; dragging an input port detaches its wire
    nodesLayer.addEventListener("pointerdown", function (ev) {
      const port = ev.target.closest(".pg-port"); if (!port) return;
      ev.preventDefault(); ev.stopPropagation();
      if (port.dataset.dir === "in") {
        const e = edgeInto(pipe, port.dataset.node, port.dataset.port);
        if (e) { pipe.edges = pipe.edges.filter(function (x) { return x !== e; }); persist(); rebuild(); }
        return;
      }
      const from = { node: port.dataset.node, port: port.dataset.port };
      const src = nodeById(pipe, from.node);
      const fromType = portsOf(src).outputs.find(function (p) { return p.name === from.port; }).type;
      const p1 = portCenter(port);
      tempWire = document.createElementNS("http://www.w3.org/2000/svg", "path");
      tempWire.setAttribute("class", "pg-wire pg-wire-temp");
      tempWire.setAttribute("stroke", TYPES[fromType].color);
      wires.appendChild(tempWire);
      nodesLayer.querySelectorAll('.pg-port[data-dir="in"]').forEach(function (pe) {
        const tn = nodeById(pipe, pe.dataset.node);
        const pp = portsOf(tn).inputs.find(function (p) { return p.name === pe.dataset.port; });
        const ok = pp && compatible(fromType, pp) && tn.id !== from.node && !reaches(pipe, tn.id, from.node);
        pe.classList.toggle("pg-port-ok", !!ok);
      });
      function move(e2) {
        const pr = pane.getBoundingClientRect();
        const x = e2.clientX - pr.left + pane.scrollLeft, y = e2.clientY - pr.top + pane.scrollTop;
        const dx = Math.max(40, Math.abs(x - p1.x) / 2);
        tempWire.setAttribute("d", "M" + p1.x + "," + p1.y + " C" + (p1.x + dx) + "," + p1.y + " " + (x - dx) + "," + y + " " + x + "," + y);
      }
      function up(e2) {
        document.removeEventListener("pointermove", move); document.removeEventListener("pointerup", up);
        if (tempWire) { tempWire.remove(); tempWire = null; }
        nodesLayer.querySelectorAll(".pg-port-ok").forEach(function (pe) { pe.classList.remove("pg-port-ok"); });
        const target = document.elementFromPoint(e2.clientX, e2.clientY);
        const tp = target && target.closest ? target.closest('.pg-port[data-dir="in"]') : null;
        if (!tp) return;
        const to = { node: tp.dataset.node, port: tp.dataset.port };
        const tn = nodeById(pipe, to.node);
        const pp = portsOf(tn).inputs.find(function (p) { return p.name === to.port; });
        if (!pp || !compatible(fromType, pp)) { flash(fromType + " does not fit " + (to.port === "in" ? "that output" : to.port) + (pp ? " (" + pp.accepts.join(" or ") + ")" : "")); return; }
        if (tn.id === from.node || reaches(pipe, tn.id, from.node)) { flash("that would make a cycle"); return; }
        pipe.edges = pipe.edges.filter(function (e) { return !(e.to === to.node && e.in === to.port); });
        pipe.edges.push({ from: from.node, out: from.port, to: to.node, in: to.port });
        persist(); rebuild();
      }
      document.addEventListener("pointermove", move); document.addEventListener("pointerup", up);
    });
    function flash(msg) { status.classList.add("pg-error"); status.textContent = msg; setTimeout(function () { status.classList.remove("pg-error"); }, 2500); }

    function rebuild() {
      nodesLayer.innerHTML = "";
      pipe.nodes.forEach(function (n) { nodesLayer.appendChild(buildNode(n)); });
      drawWires();
      refreshProblems();
      refreshInputs();
      refreshOutputs();
      refreshCode();
      loadInputs().then(function () { schedule(true); }).catch(function (err) { status.classList.add("pg-error"); status.textContent = err.message; });
    }
    window.addEventListener("resize", drawWires);
    pane.addEventListener("scroll", drawWires);
    function refreshProblems() {
      problemsBox.innerHTML = "";
      problems(pipe).forEach(function (m) { const li = el("li"); li.textContent = m; problemsBox.appendChild(li); });
      problemsBox.hidden = !problemsBox.children.length;
    }

    // ---- inputs (left) and outputs (right): one panel per input node, one figure per output node
    const io = el("div", "pg-io");
    const inBlock = el("section", "pg-io-block pg-io-inputs"); const inH = el("h3"); inH.textContent = "Inputs"; inBlock.appendChild(inH);
    const inList = el("div", "pg-io-list"); inBlock.appendChild(inList);
    const outBlock = el("section", "pg-io-block pg-io-outputs"); const outH = el("h3"); outH.textContent = "Outputs"; outBlock.appendChild(outH);
    const outList = el("div", "pg-io-list"); outBlock.appendChild(outList);
    io.appendChild(inBlock); io.appendChild(outBlock);
    layout.appendChild(io);
    const inPanels = {};   // input node id -> {node, panel, canvas, cap, tensor, annotations, source: {kind, id}, uploaded}
    const outFigs = {};    // output node id -> {fig, canvas, cap}
    const firstSample = registry.images.filter(function (im) { return im.selectable !== false; })[0].id;
    function firstImageInput() {
      const id = Object.keys(inPanels).find(function (k) { return inPanels[k].node.type === "image"; });
      return id ? inPanels[id] : null;
    }
    function thumbButton(src, title, cls) {
      const btn = el("button", cls || "", { type: "button", title: title, "aria-label": title });
      btn.appendChild(el("img", "", { src: src, alt: "" }));
      return btn;
    }
    function markPicked(P) { P.side.querySelectorAll("button").forEach(function (b) { b.classList.toggle("pg-selected", b.dataset.pick === P.source.kind + ":" + P.source.id); }); }
    // a small rendering of one sample's mask, boxes or keypoints, alone: the picker thumbnail for annotation inputs
    function annotationThumb(type, a, px) {
      const c = el("canvas", "", { width: px, height: px });
      const ctx = c.getContext("2d");
      ctx.fillStyle = "#f1f5f9"; ctx.fillRect(0, 0, px, px);
      if (!a) return c;
      const k = px / S;
      if (type === "mask" && a.mask) {
        const m = a.mask, tmp = el("canvas", "", { width: m.width, height: m.height }), layer = tmp.getContext("2d").createImageData(m.width, m.height);
        for (let i = 0; i < m.data.length; i++) if (m.data[i] > 0.5) { layer.data[4 * i] = 37; layer.data[4 * i + 1] = 99; layer.data[4 * i + 2] = 235; layer.data[4 * i + 3] = 200; }
        tmp.getContext("2d").putImageData(layer, 0, 0); ctx.drawImage(tmp, 0, 0, px, px);
      }
      if (type === "boxes") { ctx.strokeStyle = "#f59e0b"; ctx.lineWidth = 1.5; a.boxes.forEach(function (b) { ctx.strokeRect(b[0] * k, b[1] * k, (b[2] - b[0]) * k, (b[3] - b[1]) * k); }); }
      if (type === "keypoints") { ctx.fillStyle = "#10b981"; a.keypoints.forEach(function (q) { ctx.beginPath(); ctx.arc(q[0] * k, q[1] * k, 1.6, 0, Math.PI * 2); ctx.fill(); }); }
      return c;
    }
    function buildInputPanel(n) {
      const panel = el("div", "pg-io-item pg-io-item-" + n.type);
      const fig = el("figure"); const canvas = el("canvas", "", { width: S, height: S }); const cap = el("figcaption");
      fig.appendChild(canvas); fig.appendChild(cap);
      const side = el("div", "pg-io-side");
      const P = { node: n, panel: panel, canvas: canvas, cap: cap, side: side, tensor: null, annotations: null, source: { kind: "sample", id: firstSample }, uploaded: null };
      registry.images.filter(function (im) { return im.selectable !== false; }).forEach(function (im) {
        const btn = n.type === "image" ? thumbButton(ROOT + im.thumb, im.label) : el("button", "", { type: "button", title: im.label + ": its " + n.type, "aria-label": im.label + ": its " + n.type });
        btn.dataset.pick = "sample:" + im.id;
        if (n.type !== "image" && window.PGOverlays) window.PGOverlays.loadAnnotations(ROOT, im).then(function (a) { btn.appendChild(annotationThumb(n.type, a, 44)); });
        btn.addEventListener("click", function () { if (videoPanel === P) stopVideo(); P.source = { kind: "sample", id: im.id }; markPicked(P); loadInputPanel(P).then(function () { schedule(false); }); });
        side.appendChild(btn);
      });
      if (n.type === "image") {
        if (window.PGModels && window.PGModels.uploadButton) {
          const up = window.PGModels.uploadButton(function (img, name) {
            if (videoPanel === P) stopVideo(); P.uploaded = img; P.source = { kind: "upload", id: name }; markPicked(P);
            loadInputPanel(P).then(function () { schedule(false); });
          });
          side.appendChild(up);
        }
        (registry.videos || []).forEach(function (v) {
          const btn = thumbButton(ROOT + v.thumb, v.label + " (" + v.seconds + " s, " + v.license + ")", "pg-thumb-video"); btn.dataset.pick = "video:" + v.id;
          const play = el("span", "pg-thumb-play", { "aria-hidden": "true" }); play.innerHTML = '<i class="fas fa-play"></i>'; btn.appendChild(play);
          btn.addEventListener("click", function () { stopVideo(); P.source = { kind: "video", id: v.id }; markPicked(P); cap.textContent = "image · " + v.label; startVideo(v, P); });
          side.appendChild(btn);
        });
      }
      panel.appendChild(fig); panel.appendChild(side);
      markPicked(P);
      return P;
    }
    function refreshInputs() {
      Object.keys(inPanels).forEach(function (id) { if (!nodeById(pipe, id)) { if (videoPanel === inPanels[id]) stopVideo(); inPanels[id].panel.remove(); delete inPanels[id]; } });
      pipe.nodes.filter(function (n) { return n.kind === "input"; }).forEach(function (n) {
        if (!inPanels[n.id]) { inPanels[n.id] = buildInputPanel(n); inList.appendChild(inPanels[n.id].panel); }
      });
      inBlock.classList.toggle("pg-io-empty", !Object.keys(inPanels).length);
    }
    // draw the panel's source and prepare its tensor or annotations
    function loadInputPanel(P) {
      const haveOrt = typeof ort !== "undefined";
      const n = P.node;
      if (P.source.kind === "video") return Promise.resolve();   // frames come from the clip
      if (P.source.kind === "upload" && P.uploaded) {
        const ctx = P.canvas.getContext("2d");
        const sc = Math.max(S / P.uploaded.width, S / P.uploaded.height), w = P.uploaded.width * sc, h = P.uploaded.height * sc;
        ctx.fillStyle = "#000"; ctx.fillRect(0, 0, S, S); ctx.drawImage(P.uploaded, (S - w) / 2, (S - h) / 2, w, h);
        P.tensor = haveOrt ? canvasToTensor(P.canvas, S) : null; P.annotations = null;
        P.cap.textContent = "image · " + P.source.id;
        return Promise.resolve();
      }
      const im = registry.images.find(function (i) { return i.id === P.source.id; }) || registry.images[0];
      return Promise.all([loadImage(ROOT + im.file), window.PGOverlays ? window.PGOverlays.loadAnnotations(ROOT, im) : Promise.resolve(null)]).then(function (r) {
        const img = r[0]; P.annotations = r[1];
        if (n.type === "image") { P.canvas.getContext("2d").drawImage(img, 0, 0, S, S); P.tensor = haveOrt ? canvasToTensor(P.canvas, S) : null; P.cap.textContent = "image · " + im.label; return; }
        // an annotation input: the annotation alone, on a plain ground
        const ctx = P.canvas.getContext("2d"); ctx.fillStyle = "#f1f5f9"; ctx.fillRect(0, 0, S, S);
        const on = {}; on[n.type] = true;
        if (P.annotations) window.PGOverlays.drawOverlays(P.canvas, { mask: P.annotations.mask, boxes: P.annotations.boxes, keypoints: P.annotations.keypoints }, on);
        P.cap.textContent = n.type + " · " + im.label + (P.annotations ? (n.type === "boxes" ? " · " + P.annotations.boxes.length : n.type === "keypoints" ? " · " + P.annotations.keypoints.length : "") : " · none");
      });
    }
    function loadInputs() { return Promise.all(Object.keys(inPanels).map(function (id) { return loadInputPanel(inPanels[id]); })); }
    function refreshOutputs() {
      Object.keys(outFigs).forEach(function (id) { if (!nodeById(pipe, id)) { outFigs[id].fig.remove(); delete outFigs[id]; } });
      pipe.nodes.filter(function (n) { return n.kind === "output"; }).forEach(function (n) {
        if (!outFigs[n.id]) {
          const fig = el("figure"); const c = el("canvas", "", { width: S, height: S }); const cap = el("figcaption");
          fig.appendChild(c); fig.appendChild(cap); outList.appendChild(fig);
          outFigs[n.id] = { fig: fig, canvas: c, cap: cap };
        }
        const e = edgeInto(pipe, n.id, "in");
        outFigs[n.id].cap.textContent = (n.label || "output") + (e ? " · " + edgeType(pipe, e) : " · not connected");
      });
      outBlock.classList.toggle("pg-io-empty", !Object.keys(outFigs).length);
    }
    // the status line, under the two blocks
    const actions = el("div", "pg-actions pg-pipe-actions-row pg-io-actions");
    const status = el("span", "pg-status");
    actions.appendChild(status);
    layout.appendChild(actions);

    // ---- code: Python and ONNX
    const codeBox = el("div", "pg-code pg-pipe-code", { id: "code" });
    const tabs = el("div", "pg-tabs", { role: "tablist" });
    const tabPy = el("button", "pg-tab pg-tab-active", { type: "button", role: "tab", "aria-selected": "true" }); tabPy.textContent = "Python";
    const tabOnnx = el("button", "pg-tab", { type: "button", role: "tab", "aria-selected": "false" }); tabOnnx.textContent = "ONNX";
    const tabJson = el("button", "pg-tab", { type: "button", role: "tab", "aria-selected": "false" }); tabJson.textContent = "JSON";
    const copy = el("button", "pg-copy", { type: "button" }); copy.textContent = "copy";
    tabs.appendChild(tabPy); tabs.appendChild(tabOnnx); tabs.appendChild(tabJson); tabs.appendChild(copy);
    codeBox.appendChild(tabs);
    const panePy = el("div", "pg-pane", { role: "tabpanel" }), paneOnnx = el("div", "pg-pane", { role: "tabpanel" }), paneJson = el("div", "pg-pane", { role: "tabpanel" });
    paneOnnx.hidden = true; paneJson.hidden = true;
    const jsonHead = el("div", "pg-onnx-head");
    const dlJson = el("button", "pg-btn pg-btn-small", { type: "button", title: "Save this pipeline as a .json file; Import JSON loads it back, here or in another browser" });
    dlJson.innerHTML = '<i class="fas fa-file-arrow-down" aria-hidden="true"></i> Download JSON';
    dlJson.addEventListener("click", function () { downloadJson(pipe); });
    jsonHead.appendChild(dlJson);
    const jsonNote = el("span", "pg-onnx-note"); jsonNote.textContent = "the pipeline itself: nodes with typed ports, edges, parameters; load it with Import JSON";
    jsonHead.appendChild(jsonNote);
    paneJson.appendChild(jsonHead);
    const preJson = el("pre"), codeJson = el("code", "language-json"); preJson.appendChild(codeJson); paneJson.appendChild(preJson);
    const prePy = el("pre"), codePy = el("code", "language-python"); prePy.appendChild(codePy); panePy.appendChild(prePy);
    const onnxHead = el("div", "pg-onnx-head");
    const dlInputs = el("button", "pg-btn pg-btn-small", { type: "button", title: "The slider parameters stay graph inputs" });
    dlInputs.innerHTML = '<i class="fas fa-file-arrow-down" aria-hidden="true"></i> Download ONNX';
    dlInputs.addEventListener("click", function () { download(false); });
    const dlBaked = el("button", "pg-btn pg-btn-ghost pg-btn-small", { type: "button", title: "The current parameter values are baked into the graph" });
    dlBaked.innerHTML = '<i class="fas fa-file-arrow-down" aria-hidden="true"></i> Download baked';
    dlBaked.addEventListener("click", function () { download(true); });
    onnxHead.appendChild(dlInputs); onnxHead.appendChild(dlBaked);
    paneOnnx.appendChild(onnxHead);
    const preOnnx = el("pre"), codeOnnx = el("code", "language-python"); preOnnx.appendChild(codeOnnx); paneOnnx.appendChild(preOnnx);
    codeBox.appendChild(panePy); codeBox.appendChild(paneOnnx); codeBox.appendChild(paneJson);
    const tabMap = { py: [tabPy, panePy, null], onnx: [tabOnnx, paneOnnx, null], json: [tabJson, paneJson, null] };
    function selectTab(which) {
      Object.keys(tabMap).forEach(function (k) { tabMap[k][0].classList.toggle("pg-tab-active", k === which); tabMap[k][0].setAttribute("aria-selected", k === which ? "true" : "false"); tabMap[k][1].hidden = k !== which; });
    }
    tabPy.addEventListener("click", function () { selectTab("py"); });
    tabOnnx.addEventListener("click", function () { selectTab("onnx"); });
    tabJson.addEventListener("click", function () { selectTab("json"); });
    copy.addEventListener("click", function () {
      const visible = !panePy.hidden ? codePy : !paneOnnx.hidden ? codeOnnx : codeJson;
      navigator.clipboard.writeText(visible.textContent).then(function () { copy.textContent = "copied"; setTimeout(function () { copy.textContent = "copy"; }, 1200); });
    });
    layout.appendChild(codeBox);
    layout.appendChild(pipelineLinks());

    function setCode(code, text) {
      code.textContent = text;
      if (typeof hljs !== "undefined") { code.removeAttribute("data-highlighted"); hljs.highlightElement(code); }
    }
    function varName(n) { return n.id.replace(/[^a-zA-Z0-9_]/g, "_"); }
    function pythonCode() {
      let order;
      try { order = topo(pipe); } catch (e) { return "# " + e.message; }
      const imports = ["import torch", "import kornia", "from kornia.io import load_image"];
      const init = [], fwd = [], args = [], rets = [];
      const varOf = {};   // "node/port" -> python expression
      const used = {};    // argument names: image, image_2, mask, ...
      order.forEach(function (n) {
        if (n.kind === "input") {
          if (!edgesFrom(pipe, n.id).length) return;
          used[n.type] = (used[n.type] || 0) + 1;
          const a = used[n.type] > 1 ? n.type + "_" + used[n.type] : n.type;
          args.push(a); varOf[n.id + "/" + n.type] = a; return;
        }
        const src = function (port) { const e = edgeInto(pipe, n.id, port); return e ? varOf[e.from + "/" + e.out] : null; };
        if (n.kind === "output") { const s = src("in"); if (s) rets.push('"' + (n.label || "output") + '": ' + s); return; }
        const v = varName(n);
        if (n.kind === "model") {
          const m = modelById(n.model); if (!m) return;
          const s = src("image") || "image";
          if (m.pipeline) {
            (m.pipeline.imports || []).forEach(function (imp) { if (imports.indexOf(imp) === -1) imports.push(imp); });
            init.push("self." + v + " = " + m.pipeline.build);
            fwd.push(v + " = " + m.pipeline.call.replace(/\bmodel\b/, "self." + v).replace(/\bout\b/g, s));
          } else {
            fwd.push(v + " = self." + v + "(" + s + ")   # " + m.name + ": see its page for the call");
          }
          varOf[n.id + "/" + modelOutputType(m)] = v; return;
        }
        const op = opById(n.op); if (!op) return;
        const s = src("image") || "image";
        const multi = op.graphs_multi && ["mask", "boxes", "keypoints"].some(function (m) { return edgeInto(pipe, n.id, m); });
        const values = valuesOf(op, n.params, s);
        if (op.guidance) values.guide = src((op.ports && op.ports.inputs[1] && op.ports.inputs[1].name) || "guide") || "image2";
        if (multi) {
          const ctor = fill(op.snippet.replace(/\(\{img\}\)\s*$/, ""), values);
          const keys = ["input"], ins = [s], outs = [v + "_image"];
          varOf[n.id + "/image"] = v + "_image";
          ["mask", "boxes", "keypoints"].forEach(function (mkey) { const ss = src(mkey); if (ss) { keys.push(mkey === "boxes" ? "bbox_xyxy" : mkey); ins.push(ss); outs.push(v + "_" + mkey); varOf[n.id + "/" + mkey] = v + "_" + mkey; } });
          init.push("self." + v + " = kornia.augmentation.AugmentationSequential(" + ctor + ", data_keys=" + JSON.stringify(keys).replace(/"/g, "'") + ")");
          fwd.push(outs.join(", ") + " = self." + v + "(" + ins.join(", ") + ")");
        } else if (op.module_snippet) {
          const mod = fill(op.module_snippet.replace(/\(\{img\}(?:,\s*\{guide\})?\)\s*$/, ""), values);
          init.push("self." + v + " = " + mod);
          fwd.push(v + " = self." + v + "(" + s + (op.guidance ? ", " + values.guide : "") + ")");
          varOf[n.id + "/image"] = v;
        } else {
          fwd.push(v + " = " + fill(op.snippet, values));
          varOf[n.id + "/image"] = v;
        }
      });
      const lines = imports.concat(["", "", "class Pipeline(torch.nn.Module):", "    def __init__(self):", "        super().__init__()"]);
      init.forEach(function (l) { lines.push("        " + l); });
      if (!init.length) lines.push("        pass");
      lines.push("", "    def forward(self, " + (args.length ? args.join(", ") : "image") + "):");
      fwd.forEach(function (l) { lines.push("        " + l); });
      lines.push("        return {" + rets.join(", ") + "}");
      lines.push("", "", "pipeline = Pipeline().eval()");
      const base = firstImageInput();
      const im = base && base.source.kind === "sample" ? registry.images.find(function (i) { return i.id === base.source.id; }) : null;
      lines.push("image = load_image(\"" + (im ? im.id : "my_photo") + ".png\")[None]   # (1, 3, H, W) float in [0, 1]" + (im ? "   # https://kornia.org/playground/" + im.file : ""));
      args.filter(function (a) { return /^image_\d+$/.test(a); }).forEach(function (a) { lines.push(a + " = load_image(\"second.png\")[None]   # another image of the same size"); });
      if (args.indexOf("mask") !== -1) lines.push("mask = torch.zeros(1, 1, image.shape[-2], image.shape[-1])   # (1, 1, H, W) float");
      if (args.indexOf("boxes") !== -1) lines.push("boxes = torch.tensor([[[40.0, 50.0, 120.0, 140.0]]])   # (1, K, 4) xyxy in pixels");
      if (args.indexOf("keypoints") !== -1) lines.push("keypoints = torch.tensor([[[60.0, 70.0], [200.0, 60.0]]])   # (1, P, 2) xy in pixels");
      lines.push("with torch.no_grad():", "    outputs = pipeline(" + (args.length ? args.join(", ") : "image") + ")");
      return lines.join("\n");
    }
    // the contract of the composed graph: every input and output with its shape, range and decoding
    function onnxSnippet() {
      const file = pipe.name.replace(/[^\w.-]+/g, "_");
      let order, names;
      try { order = topo(pipe); names = ioNames(pipe); } catch (e) { return "# " + e.message; }
      const ins = order.filter(function (n) { return n.kind === "input" && names[n.id]; });
      const outs = order.filter(function (n) { return n.kind === "output" && names[n.id]; }).map(function (n) { const e = edgeInto(pipe, n.id, "in"); const src = nodeById(pipe, e.from); return { name: names[n.id], type: edgeType(pipe, e), src: src }; });
      const live = [];
      order.forEach(function (n) { if (n.kind === "op") { const op = opById(n.op); if (op && !(op.graphs_multi && ["mask", "boxes", "keypoints"].some(function (m) { return edgeInto(pipe, n.id, m); }))) op.params.filter(function (q) { return q.kind === "live"; }).forEach(function (q) { live.push(nodeTitle(n) + "." + q.name); }); } });
      const dyn = order.every(function (n) { return n.kind !== "op" || (opById(n.op) || {}).dynamic; }) && !order.some(function (n) { return n.kind === "model"; });
      const HW = dyn ? "H, W (any size)" : S + ", " + S;
      const shape = { image: "(1, 3, " + HW + ")  float32 RGB in [0, 1], NCHW", gray: "(1, 1, " + HW + ")  float32 in [0, 1]", mask: "(1, 1, " + HW + ")  float32, 1 inside, 0 outside", boxes: "(1, K, 4)  float32 xyxy in pixels of the image", keypoints: "(1, P, 2)  float32 xy in pixels of the image", depth: "(1, 1, " + HW + ")  float32 relative depth", result: "model tensors, see below" };
      const L = [
        "# " + file + ".onnx: the pipeline as one graph. Download above (parameters as inputs) or Download baked (parameters fixed).",
        "# Runs with onnxruntime, onnxruntime-web, the ort crate in Rust, and the ONNX importers of TensorRT and OpenVINO.",
        "#",
        "# INPUTS",
      ];
      ins.forEach(function (n) { L.push("#   " + names[n.id].padEnd(26) + shape[n.type]); });
      live.forEach(function (name) { L.push("#   " + name.padEnd(26) + "(1,)  float32, a slider value (absent from the baked file)"); });
      L.push("#", "# OUTPUTS");
      outs.forEach(function (o) {
        let d = shape[o.type] || o.type;
        if (o.type === "result" && o.src && o.src.kind === "model") { const m = modelById(o.src.model); d = (m ? m.name : "model") + ": " + (m && m.output ? m.output.kind : "") + " (raw tensors; decode as on its page)"; }
        L.push("#   " + o.name.padEnd(26) + d);
      });
      L.push("", "# pip install onnxruntime numpy", "import numpy as np", "import onnxruntime as ort", "", 'sess = ort.InferenceSession("' + file + '.onnx")', "feeds = {}");
      ins.forEach(function (n) {
        const v = names[n.id];
        if (n.type === "image") L.push(v + " = np.random.rand(1, 3, " + (dyn ? "480, 640" : S + ", " + S) + ").astype(np.float32)   # your frame: uint8 HWC -> img[None].transpose(0, 3, 1, 2) / 255", 'feeds["' + v + '"] = ' + v);
        else if (n.type === "mask") L.push('feeds["' + v + '"] = np.zeros((1, 1, ' + S + ', ' + S + '), np.float32)   # your mask, 1 inside');
        else if (n.type === "boxes") L.push('feeds["' + v + '"] = np.array([[[40, 50, 120, 140]]], np.float32)   # (1, K, 4) xyxy pixels');
        else if (n.type === "keypoints") L.push('feeds["' + v + '"] = np.array([[[60, 70], [200, 60]]], np.float32)   # (1, P, 2) xy pixels');
      });
      if (live.length) L.push("for i in sess.get_inputs()[" + ins.length + ":]:", "    feeds[i.name] = np.array([0.5], np.float32)   # each slider, by name");
      L.push('outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))');
      const types = {}; outs.forEach(function (o) { types[o.type] = true; });
      L.push("", "# DECODING each output type, into what a robot or a dataset wants");
      if (types.image) L.push("def to_rgb8(t):   # (1, 3, H, W) float -> (H, W, 3) uint8, e.g. sensor_msgs/Image rgb8", "    return (np.clip(t[0], 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)");
      if (types.gray || types.mask) L.push("def to_mono8(t):   # (1, 1, H, W) float -> (H, W) uint8, e.g. sensor_msgs/Image mono8", "    return (np.clip(t[0, 0], 0, 1) * 255).astype(np.uint8)");
      if (types.depth) L.push("def to_depth(t):   # (1, 1, H, W) relative depth -> (H, W) float32, e.g. sensor_msgs/Image 32FC1 (scale is arbitrary)", "    return t[0, 0].astype(np.float32)");
      if (types.boxes) L.push("def to_boxes(t):   # (1, K, 4) -> list of (x1, y1, x2, y2) in pixels, e.g. vision_msgs/Detection2D per box", "    return [tuple(map(float, b)) for b in t[0]]");
      if (types.keypoints) L.push("def to_points(t):   # (1, P, 2) -> list of (x, y) in pixels", "    return [tuple(map(float, k)) for k in t[0]]");
      outs.forEach(function (o) {
        if (o.type === "result" && o.src && o.src.kind === "model") {
          const m = modelById(o.src.model); const k = m && m.output ? m.output.kind : "";
          if (k === "detections") L.push("def to_detections(t, size=" + (m.output.coord_size || m.input.size) + ", threshold=" + (m.output.threshold || 0.4) + "):   # " + m.name + ": rows of class, score, x, y, w, h in the model's " + (m.output.coord_size || m.input.size) + " px frame", "    rows = t[0] if t.ndim == 3 else t", "    keep = rows[rows[:, 1] >= threshold]", "    return [(int(c), float(s), float(x), float(y), float(w), float(h)) for c, s, x, y, w, h in keep[:, :6]]   # scale x, y, w, h by your_width / size, your_height / size");
          else if (k === "classification") L.push("def to_classes(t, top=5):   # " + m.name + ": logits over the ImageNet classes", "    p = np.exp(t[0] - t[0].max()); p /= p.sum()", "    return [(int(i), float(p[i])) for i in np.argsort(-p)[:top]]   # labels: https://kornia.org/playground/models/imagenet_labels.json");
          else L.push("# " + o.name + ": " + (m ? m.name : "model") + " publishes " + k + " tensors; the decoder is in the page's JavaScript (playground/models.js, draw" + k.replace(/_.*/, "").replace(/^\w/, function (c) { return c.toUpperCase(); }) + ") and on the model's page");
        }
      });
      L.push("", "# IN ROS 2: no ROS package is needed. Feed the file to a generic ONNX inference node (NVIDIA Isaac ROS DNN inference on Jetson,",
             "# the OpenVINO node on Intel, or an onnxruntime wrapper), set the input contract above, and publish the decoded outputs as",
             "# sensor_msgs/Image (rgb8, mono8, 32FC1), vision_msgs/Detection2DArray for boxes, geometry_msgs/PoseArray for keypoints.");
      return L.join("\n");
    }
    function refreshCode() { setCode(codePy, pythonCode()); setCode(codeOnnx, onnxSnippet()); setCode(codeJson, documentText(pipe)); }

    // ---- running
    let videoPanel = null;   // the image input panel a clip plays into
    let timer = null, running = false, pending = false;
    let composed = null, composedKey = null;
    let explicit = false;   // Run was pressed: models are included
    let video = null, videoLoop = false;
    function stopVideo() { videoLoop = false; restartPass(); if (video) video.pause(); videoPanel = null; }
    function startVideo(v, P) {
      videoPanel = P;
      if (!video) {
        video = el("video", "", { muted: "", loop: "", playsinline: "", preload: "auto" });
        video.muted = true; video.loop = true; video.hidden = true; io.appendChild(video);
      }
      if (video.dataset.id !== v.id) {
        video.innerHTML = "";
        [["webm", "video/webm"], ["mp4", "video/mp4"]].forEach(function (pair) { if (v[pair[0]]) video.appendChild(el("source", "", { src: ROOT + v[pair[0]], type: pair[1] })); });
        video.dataset.id = v.id; video.load();
      }
      videoLoop = true; restartPass();
      const pr = video.play(); if (pr && pr.catch) pr.catch(function () {});
      pump();
    }
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
      if (!videoLoop || !io.isConnected || !videoPanel) { videoLoop = false; restartPass(); return; }
      if (video.readyState >= 2 && typeof ort !== "undefined") {
        const t = video.currentTime;
        if (pass.t >= 0) pass.seen += Math.max(0, t - pass.t);
        if (!pass.done && pass.t >= 0 && t < pass.t - 0.5 && pass.seen >= (video.duration || 4) * 0.9 && pass.frames.length > 1) pass.done = true;
        pass.t = t;
        videoPanel.canvas.getContext("2d").drawImage(video, 0, 0, S, S);
        if (pass.done) {
          const kept = keptNear(t);
          if (kept) { kept.imgs.forEach(function (im) { const f = outFigs[im.id]; if (f) f.canvas.getContext("2d").putImageData(im.data, 0, 0); }); status.textContent = "computed once (" + pass.frames.length + " frames) · replaying with the clip"; }
          window.requestAnimationFrame(pump); return;
        }
        videoPanel.tensor = canvasToTensor(videoPanel.canvas, S);
        if (hasModel() && !explicit) { window.requestAnimationFrame(pump); return; }
        execute().then(function () {
          if (!pass.done && (pass.frames.length < MAX_KEPT || (pass.frames.length && t - pass.frames[pass.frames.length - 1].t >= (video.duration || 4) / MAX_KEPT))) {
            if (pass.frames.length >= MAX_KEPT) pass.frames.splice(0, 1);
            pass.frames.push({ t: t, imgs: Object.keys(outFigs).map(function (id) { const c = outFigs[id].canvas; return { id: id, data: c.getContext("2d").getImageData(0, 0, c.width, c.height) }; }) });
          }
          window.requestAnimationFrame(pump);
        });
      } else window.requestAnimationFrame(pump);
    }
    document.addEventListener("visibilitychange", function () { if (!video || !videoLoop) return; if (document.hidden) video.pause(); else video.play().catch(function () {}); });
    function hasModel() { return pipe.nodes.some(function (n) { return n.kind === "model"; }); }
    function schedule(structural) {
      refreshCode(); refreshProblems();
      restartPass();
      if (structural) composed = null;
      clearTimeout(timer);
      timer = setTimeout(runGraph, 60);
    }
    function runGraph() {
      if (typeof ort === "undefined" || typeof protobuf === "undefined") { status.classList.add("pg-error"); status.textContent = "onnxruntime-web or protobuf.js did not load; the preview is unavailable."; return; }
      const issues = problems(pipe);
      if (issues.length) { status.textContent = "fix the graph to run it"; return; }
      if (hasModel() && !explicit) { status.textContent = "press Run to include the model" + (pipe.nodes.filter(function (n) { return n.kind === "model"; }).length > 1 ? "s" : ""); return; }
      if (running) { pending = true; return; }
      if (Object.keys(inPanels).some(function (k) { return inPanels[k].node.type === "image" && !inPanels[k].tensor; })) return;
      running = true;
      execute().then(function () { running = false; if (pending) { pending = false; runGraph(); } });
    }
    function feedsFor(c) {
      const feeds = {};
      c.inputs.forEach(function (gi) {
        const P = inPanels[gi.node];
        const annotations = P ? P.annotations : null;
        if (gi.type === "image") feeds[gi.name] = P && P.tensor ? P.tensor : new ort.Tensor("float32", new Float32Array(3 * S * S), [1, 3, S, S]);
        else if (gi.type === "mask") feeds[gi.name] = new ort.Tensor("float32", annotations && annotations.mask ? annotations.mask.data : new Float32Array(S * S), [1, 1, S, S]);
        else if (gi.type === "boxes") { const b = annotations ? annotations.boxes : [[S * 0.2, S * 0.2, S * 0.6, S * 0.7]]; const d = new Float32Array(b.length * 4); b.forEach(function (r, i) { d[4 * i] = r[0]; d[4 * i + 1] = r[1]; d[4 * i + 2] = r[2]; d[4 * i + 3] = r[3]; }); feeds[gi.name] = new ort.Tensor("float32", d, [1, b.length, 4]); }
        else if (gi.type === "keypoints") { const k = annotations ? annotations.keypoints : [[S * 0.5, S * 0.5]]; const d = new Float32Array(k.length * 2); k.forEach(function (r, i) { d[2 * i] = r[0]; d[2 * i + 1] = r[1]; }); feeds[gi.name] = new ort.Tensor("float32", d, [1, k.length, 2]); }
      });
      c.paramInputs.forEach(function (pi) { const n = nodeById(pipe, pi.node); feeds[pi.name] = new ort.Tensor("float32", new Float32Array([Number(n.params[pi.param])]), [1]); });
      return feeds;
    }
    // draw one output figure from the results
    function drawOutput(out, results, c) {
      const f = outFigs[out.node]; if (!f) return;
      const t = results[out.name]; if (!t) return;
      const type = out.type;
      if (type === "image" || type === "gray") { const src = nodeById(pipe, out.from); const o = src && src.kind === "op" ? opById(src.op) : null; tensorToCanvas(t, f.canvas, o && o.output ? o.output.display : "clamp"); return; }
      if (type === "depth") { tensorToCanvas(t, f.canvas, "normalize"); return; }
      if (type === "mask") {
        f.canvas.width = S; f.canvas.height = S;
        const ctx = f.canvas.getContext("2d"); ctx.fillStyle = "#f1f5f9"; ctx.fillRect(0, 0, S, S);
        window.PGOverlays.drawOverlays(f.canvas, { mask: t }, { mask: true });
        return;
      }
      if (type === "boxes" || type === "keypoints") {
        // the annotation alone, on the same plain ground as the input panels; the coordinates are in the graph's frame
        f.canvas.width = S; f.canvas.height = S;
        const ctx = f.canvas.getContext("2d"); ctx.fillStyle = "#f1f5f9"; ctx.fillRect(0, 0, S, S);
        const items = {}; items[type] = window.PGOverlays.rows2d(t, type === "boxes" ? 4 : 2);
        const on = {}; on[type] = true;
        window.PGOverlays.drawOverlays(f.canvas, items, on);
        f.cap.textContent = (nodeById(pipe, out.node).label || "output") + " · " + type + " · " + t.dims[1];
        return;
      }
      if (type === "result" && window.PGModels) {
        const mt = c.modelTaps[out.from]; if (!mt) return;
        const pre = el("canvas");
        const preT = results[mt.pre]; if (preT) tensorToCanvas(preT, pre, "clamp"); else { pre.width = S; pre.height = S; const base = firstImageInput(); if (base) pre.getContext("2d").drawImage(base.canvas, 0, 0); }
        const modelResults = {}; Object.keys(mt.outputs).forEach(function (short) { modelResults[short] = results[mt.outputs[short]]; });
        const labelsReady = mt.model.output.labels ? window.PGModels.getLabels(mt.model.output.labels, ROOT) : Promise.resolve(null);
        return labelsReady.then(function (labels) { const detail = window.PGModels.drawResults(mt.model, modelResults, pre, f.canvas, { labels: labels }); if (detail) f.cap.textContent = (nodeById(pipe, out.node).label || "output") + " · " + detail; });
      }
    }
    function execute() {
      status.classList.remove("pg-error");
      const key = JSON.stringify([pipe.nodes.map(function (n) { return [n.id, n.op, n.model, n.kind === "op" && opById(n.op) ? graphKey(opById(n.op), n.params) : null]; }), pipe.edges]);
      const t0 = performance.now();
      const ready = (composed && composedKey === key) ? Promise.resolve(composed)
        : compose(pipe, false, true, S).then(function (c) {
          status.textContent = "loading the graph…";
          return ort.InferenceSession.create(c.bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" }).then(function (session) { c.session = session; composed = c; composedKey = key; return c; });
        });
      return ready.then(function (c) {
        return c.session.run(feedsFor(c)).then(function (results) {
          const ms = performance.now() - t0;
          return Promise.all(c.outputs.map(function (out) { return drawOutput(out, results, c); })).then(function () {
            explicit = false;
            const nOps = pipe.nodes.filter(function (n) { return n.kind === "op" || n.kind === "model"; }).length;
            status.textContent = nOps + " node" + (nOps === 1 ? "" : "s") + ", " + c.outputs.length + " output" + (c.outputs.length === 1 ? "" : "s") + " · " + (c.bytes.length / 1024).toFixed(0) + " KB graph · " + ms.toFixed(0) + " ms on your machine";
          });
        });
      }).catch(function (err) {
        console.error("pipeline run failed", err);
        status.classList.add("pg-error");
        status.textContent = "could not run the pipeline: " + (err.message || err);
        videoLoop = false;
      });
    }
    function download(bake) {
      status.classList.remove("pg-error");
      status.textContent = "composing…";
      compose(pipe, bake, false, S).then(function (c) {
        const blob = new Blob([c.bytes], { type: "application/octet-stream" });
        const url = URL.createObjectURL(blob);
        const a = el("a", "", { href: url, download: pipe.name.replace(/[^\w.-]+/g, "_") + (bake ? "-baked" : "") + ".onnx" });
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
        status.textContent = (c.bytes.length > 2 * 1048576 ? (c.bytes.length / 1048576).toFixed(1) + " MB" : (c.bytes.length / 1024).toFixed(0) + " KB") + " ONNX · inputs: " + c.inputs.map(function (i) { return i.type; }).join(", ") + (c.paramInputs.length ? " + " + c.paramInputs.length + " parameter" + (c.paramInputs.length === 1 ? "" : "s") : "") + " · outputs: " + c.outputs.length;
      }).catch(function (err) { status.classList.add("pg-error"); status.textContent = "could not export: " + (err.message || err); });
    }

    rebuild();
    refreshHistoryButtons();
  }

  window.PGPipelines = {
    init: function (reg, root) {
      registry = reg; ROOT = root;
      mount = document.getElementById("pg-ops");
      window.addEventListener("popstate", render);
      fetch(ROOT + "models/index.json", { cache: "no-cache" }).then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; })
        .then(function (idx) { modelsIndex = idx; render(); if (window.KorniaAuth) window.KorniaAuth.onChange(function () { if (canSync()) mergeFromAccount(); }); });
    },
  };
})();
