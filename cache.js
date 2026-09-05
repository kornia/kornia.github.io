// kornia.org: a persistent download cache for the ONNX graphs, models and volumes the experiments fetch.
//
// Downloads go through KorniaCache.fetch(url): the Cache API keeps the response on this device across visits, so a
// graph or a model is fetched once and read from disk after that. A small index in localStorage records what is
// kept (size, when, a label) for the dashboard, which lists the entries and can remove them. Nothing here needs an
// account: the cache is the browser's. Turning it off (KorniaCache.setEnabled(false)) makes every fetch go to the
// network again and keeps nothing.
(function () {
  "use strict";
  const CACHE = "kornia-downloads-v1";
  const INDEX = "kornia-downloads-index";
  const OFF = "kornia-downloads-off";
  const supported = typeof window !== "undefined" && "caches" in window && typeof window.caches.open === "function";

  function readIndex() { try { return JSON.parse(localStorage.getItem(INDEX) || "{}"); } catch (e) { return {}; } }
  function writeIndex(ix) { try { localStorage.setItem(INDEX, JSON.stringify(ix)); } catch (e) { /* private mode */ } }
  function enabled() { try { return supported && localStorage.getItem(OFF) !== "1"; } catch (e) { return supported; } }
  function labelFor(url) {
    const name = url.split("?")[0].split("/").pop();
    if (/huggingface\.co/.test(url)) { const m = url.match(/huggingface\.co\/([^/]+\/[^/]+)/); return (m ? m[1] + " · " : "") + name; }
    const g = url.match(/graphs\/([^/]+)\/([^/]+)$/);
    if (g) return g[1] + " · " + g[2];
    return name;
  }
  function kind(url) {
    if (/\/graphs\//.test(url)) return "operator graph";
    if (/huggingface\.co|\/models\//.test(url)) return "model";
    if (/\.bin(\?|$)/.test(url)) return "volume";
    return "file";
  }

  // read a response with progress; resolves to a Uint8Array
  function readBody(response, onProgress) {
    const total = Number(response.headers.get("Content-Length")) || 0;
    if (!response.body || !response.body.getReader || !onProgress) return response.arrayBuffer().then(function (b) { return new Uint8Array(b); });
    const reader = response.body.getReader();
    const chunks = []; let received = 0;
    function pump() {
      return reader.read().then(function (r) {
        if (r.done) return;
        chunks.push(r.value); received += r.value.length; onProgress(received, total);
        return pump();
      });
    }
    return pump().then(function () {
      const out = new Uint8Array(received); let off = 0;
      chunks.forEach(function (c) { out.set(c, off); off += c.length; });
      return out;
    });
  }

  function fetchBytes(url, onProgress) {
    const abs = new URL(url, window.location.href).href;
    if (!enabled()) return fetch(abs).then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status + " fetching " + labelFor(abs)); return readBody(r, onProgress); });
    return caches.open(CACHE).then(function (cache) {
      return cache.match(abs).then(function (hit) {
        if (hit) { if (onProgress) onProgress(1, 1); return readBody(hit, null); }
        return fetch(abs).then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status + " fetching " + labelFor(abs));
          return readBody(r, onProgress).then(function (bytes) {
            // stored as a fresh response: the original body is consumed, and the stored copy needs no headers
            const stored = new Response(bytes, { headers: { "Content-Type": "application/octet-stream", "Content-Length": String(bytes.length) } });
            return cache.put(abs, stored).then(function () {
              const ix = readIndex(); ix[abs] = { bytes: bytes.length, at: Date.now(), label: labelFor(abs), kind: kind(abs) }; writeIndex(ix);
            }).catch(function () { /* quota: keep going without storing */ }).then(function () { return bytes; });
          });
        });
      });
    }).catch(function () { return fetch(abs).then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return readBody(r, onProgress); }); });
  }

  function list() {
    const ix = readIndex();
    return Object.keys(ix).map(function (url) { return Object.assign({ url: url }, ix[url]); }).sort(function (a, b) { return b.at - a.at; });
  }
  function remove(url) {
    const ix = readIndex(); delete ix[url]; writeIndex(ix);
    if (!supported) return Promise.resolve();
    return caches.open(CACHE).then(function (c) { return c.delete(url); }).catch(function () {});
  }
  function clear() {
    writeIndex({});
    if (!supported) return Promise.resolve();
    return caches.delete(CACHE).catch(function () {});
  }
  function usage() {
    const total = list().reduce(function (n, e) { return n + (e.bytes || 0); }, 0);
    const est = navigator.storage && navigator.storage.estimate ? navigator.storage.estimate().catch(function () { return null; }) : Promise.resolve(null);
    return est.then(function (e) { return { bytes: total, entries: list().length, quota: e && e.quota ? e.quota : null, used: e && e.usage ? e.usage : null }; });
  }
  function setEnabled(on) { try { if (on) localStorage.removeItem(OFF); else localStorage.setItem(OFF, "1"); } catch (e) { /* private mode */ } }

  window.KorniaCache = { supported: supported, enabled: enabled, setEnabled: setEnabled, fetch: fetchBytes, list: list, remove: remove, clear: clear, usage: usage, label: labelFor };
})();
