// kornia dashboard: accounts on Firebase Authentication (GitHub, Google, email) and per-user data in Firestore.
// The page is static; every call goes from the browser to Firebase with the user's own token, and
// firestore.rules decides what that token may touch.
import { initializeApp } from "https://www.gstatic.com/firebasejs/12.18.0/firebase-app.js";
import {
  getAuth, onAuthStateChanged, signInWithPopup, GithubAuthProvider, GoogleAuthProvider,
  createUserWithEmailAndPassword, signInWithEmailAndPassword, sendEmailVerification, sendPasswordResetEmail,
  signOut, updateProfile,
} from "https://www.gstatic.com/firebasejs/12.18.0/firebase-auth.js";
import {
  getFirestore, collection, doc, getDocs, setDoc, deleteDoc, serverTimestamp, query, orderBy,
} from "https://www.gstatic.com/firebasejs/12.18.0/firebase-firestore.js";

const cfg = window.KORNIA_FIREBASE || {};
const $ = (id) => document.getElementById(id);
const PIPELINES_KEY = "kornia-playground-pipelines";
const USERS = cfg.usersRoot || "users";   // users/ in production, dev_users/ on the preview and localhost
const b64url = (s) => btoa(unescape(encodeURIComponent(s))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");

function show(id) {
  ["hub-setup", "hub-signin", "hub-account"].forEach((k) => { $(k).hidden = k !== id; });
  document.body.classList.toggle("db-signed-in", id === "hub-account");
}
function note(text, error) {
  const n = $("hub-note");
  n.textContent = text || "";
  n.classList.toggle("pg-error", !!error);
}
const friendly = (e) => ({
  "auth/email-already-in-use": "That email already has an account. Sign in, or reset the password.",
  "auth/invalid-email": "That does not look like an email address.",
  "auth/weak-password": "Use at least 8 characters.",
  "auth/invalid-credential": "Wrong email or password.",
  "auth/user-not-found": "No account with that email.",
  "auth/wrong-password": "Wrong email or password.",
  "auth/popup-closed-by-user": "The sign-in window was closed before finishing.",
  "auth/account-exists-with-different-credential": "This email is already registered with another sign-in method. Use that one.",
  "auth/too-many-requests": "Too many attempts; wait a minute.",
  "auth/unauthorized-domain": "This domain is not authorised in the Firebase console yet.",
}[e && e.code] || (e && e.message) || String(e));

if (!cfg.configured) {
  show("hub-setup");
} else {
  const app = initializeApp(cfg);
  const auth = getAuth(app);
  const db = getFirestore(app);
  auth.useDeviceLanguage();

  // ---- sign in
  $("hub-github").addEventListener("click", () => signInWithPopup(auth, new GithubAuthProvider()).catch((e) => note(friendly(e), true)));
  $("hub-google").addEventListener("click", () => signInWithPopup(auth, new GoogleAuthProvider()).catch((e) => note(friendly(e), true)));
  $("hub-email-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const email = $("hub-email").value.trim(), pw = $("hub-password").value;
    const mode = ev.submitter && ev.submitter.dataset.mode;
    note("");
    if (mode === "register") {
      createUserWithEmailAndPassword(auth, email, pw).then((cred) => sendEmailVerification(cred.user))
        .then(() => note("Account created. Check your inbox for the verification link, then reload."))
        .catch((e) => note(friendly(e), true));
    } else if (mode === "reset") {
      if (!email) { note("Type your email first.", true); return; }
      sendPasswordResetEmail(auth, email).then(() => note("Password reset mail sent.")).catch((e) => note(friendly(e), true));
    } else {
      signInWithEmailAndPassword(auth, email, pw).catch((e) => note(friendly(e), true));
    }
  });
  $("hub-signout").addEventListener("click", () => signOut(auth));
  $("hub-resend").addEventListener("click", () => sendEmailVerification(auth.currentUser).then(() => note("Verification mail sent again.")).catch((e) => note(friendly(e), true)));

  // ---- sections: the sidebar switches them; the hash remembers the one open
  const sections = [...document.querySelectorAll(".db-section")];
  const tabs = [...document.querySelectorAll(".db-nav [data-section]")];
  let verified = false, current = "overview";
  function showSection(name) {
    if (!sections.some((s) => s.dataset.section === name)) name = "overview";
    current = name;
    sections.forEach((s) => { s.hidden = s.dataset.section !== name; });
    if (name === "downloads" && typeof renderCache === "function") renderCache();   // the list may have grown since the page loaded
    tabs.forEach((t) => t.setAttribute("aria-selected", t.dataset.section === name ? "true" : "false"));
    // an unverified account sees the notice where the data would be
    const active = sections.find((s) => s.dataset.section === name);
    const gated = active && active.hasAttribute("data-needs-verified") && !verified;
    active.querySelectorAll(":scope > :not(.db-head)").forEach((n) => { n.hidden = gated; });
  }
  tabs.forEach((t) => t.addEventListener("click", (e) => { e.preventDefault(); history.replaceState(null, "", "#" + t.dataset.section); showSection(t.dataset.section); }));
  showSection((location.hash || "#overview").slice(1));
  window.addEventListener("hashchange", () => showSection(location.hash.slice(1)));

  // ---- account
  let user = null;
  onAuthStateChanged(auth, (u) => {
    user = u;
    if (!u) { show("hub-signin"); return; }
    show("hub-account");
    $("hub-name").textContent = u.displayName || u.email;
    const provider = u.providerData[0] ? u.providerData[0].providerId.replace(".com", "") : "email";
    $("hub-meta").textContent = u.email + (u.emailVerified ? "" : " · not verified");
    $("hub-avatar").src = u.photoURL || "../assets/kornia-logo-mini.svg";
    $("db-acc-name").textContent = u.displayName || "(no name)";
    $("db-acc-email").textContent = u.email + (u.emailVerified ? " · verified" : " · not verified");
    $("db-acc-provider").textContent = provider === "password" ? "email and password" : provider;
    $("db-acc-since").textContent = u.metadata && u.metadata.creationTime ? new Date(u.metadata.creationTime).toLocaleDateString() : "";
    $("hub-unverified").hidden = u.emailVerified;
    verified = u.emailVerified || devApi;   // the rules require a verified email before any data access; pricing and account never need it
    showSection(current);
    if (u.emailVerified) loadPipelines();
  });

  // ---- saved pipelines: browser storage <-> users/{uid}/pipelines
  function localPipelines() { try { return JSON.parse(localStorage.getItem(PIPELINES_KEY) || "[]"); } catch (e) { return []; } }
  async function loadPipelines() {
    const list = $("hub-pipelines");
    list.innerHTML = "<li class='pg-note'>loading…</li>";
    try {
      const snap = await getDocs(query(collection(db, USERS, user.uid, "pipelines"), orderBy("updated", "desc")));
      list.innerHTML = "";
      if (snap.empty) list.innerHTML = "<li class='pg-note'>Nothing saved yet. Build one in the experiment while signed in and it appears here.</li>";
      snap.forEach((d) => {
        const p = d.data();
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = "../playground/pipelines/?share=" + b64url(JSON.stringify(p.version === 2 ? { name: p.name, version: 2, nodes: p.nodes, edges: p.edges } : { name: p.name, container: p.container, steps: p.steps, model: p.model || null }));
        a.textContent = p.name;
        const meta = noteField(p.note || "", async (text) => {
          await window.KorniaAuth.pipelines.note(d.id, text);
          const local = localPipelines(); const lp = local.find((x) => x.id === d.id); if (lp) { lp.note = text; localStorage.setItem(PIPELINES_KEY, JSON.stringify(local)); }
        });
        const rm = document.createElement("button");
        rm.type = "button"; rm.className = "pg-btn pg-btn-ghost pg-btn-small hub-btn-danger"; rm.innerHTML = '<i class="fas fa-trash" aria-hidden="true"></i> Delete';
        rm.addEventListener("click", () => {
          if (!window.confirm('Delete "' + p.name + '"? This cannot be undone.')) return;
          deleteDoc(doc(db, USERS, user.uid, "pipelines", d.id)).then(() => {
            localStorage.setItem(PIPELINES_KEY, JSON.stringify(localPipelines().filter((x) => x.id !== d.id)));
            loadPipelines();
          });
        });
        li.className = "hub-pipe-row";
        const line = document.createElement("div"); line.className = "hub-pipe-line";
        const tools = document.createElement("span"); tools.className = "hub-pipe-tools"; tools.appendChild(rm);
        line.appendChild(a); line.appendChild(tools);
        li.appendChild(line); li.appendChild(meta);
        list.appendChild(li);
      });
      $("db-pipes").textContent = String(snap.size);
      // the account's copies count as saved in this browser too, so nothing shows twice
      const ids = new Set(); snap.forEach((d) => ids.add(d.id));
      const local = localPipelines(); let changed = false;
      local.forEach((p) => { const on = ids.has(p.id); if (!!p.synced !== on) { p.synced = on; changed = true; } });
      if (changed) localStorage.setItem(PIPELINES_KEY, JSON.stringify(local));
      renderLocal();
    } catch (e) { list.innerHTML = ""; note("could not read your pipelines: " + friendly(e), true); }
  }
  // the comment under a pipeline's name: click to edit, saved on change
  function noteField(text, onSave) {
    const input = document.createElement("input");
    input.type = "text"; input.className = "hub-pipe-note"; input.maxLength = 160; input.value = text; input.placeholder = "add a comment"; input.setAttribute("aria-label", "Comment");
    input.addEventListener("change", async () => { input.disabled = true; try { await onSave(input.value.trim()); } catch (e) { note("could not save the comment: " + friendly(e), true); } input.disabled = false; });
    return input;
  }
  function markLocal(id, synced) {
    const local = localPipelines(); const p = local.find((x) => x.id === id);
    if (p) { p.synced = synced; localStorage.setItem(PIPELINES_KEY, JSON.stringify(local)); }
  }
  function renderLocal() {
    const panel = $("hub-local-panel"), list = $("hub-local");
    if (!panel) return;
    const pending = localPipelines().filter((p) => !p.synced);
    panel.hidden = !pending.length;
    list.innerHTML = "";
    pending.forEach((p) => {
      const li = document.createElement("li"); li.className = "hub-pipe-row";
      const a = document.createElement("a"); a.href = "../playground/pipelines/?p=" + p.id; a.textContent = p.name;
      const meta = noteField(p.note || "", async (text) => { const local = localPipelines(); const lp = local.find((x) => x.id === p.id); if (lp) { lp.note = text; localStorage.setItem(PIPELINES_KEY, JSON.stringify(local)); } p.note = text; });
      const tools = document.createElement("span"); tools.className = "hub-pipe-tools";
      const save = document.createElement("button"); save.type = "button"; save.className = "pg-btn pg-btn-small"; save.innerHTML = '<i class="fas fa-cloud-arrow-up" aria-hidden="true"></i> Save';
      save.addEventListener("click", async () => {
        save.disabled = true;
        try { await window.KorniaAuth.pipelines.save(p); markLocal(p.id, true); await loadPipelines(); }
        catch (e) { save.disabled = false; note("could not save: " + friendly(e), true); }
      });
      const rm = document.createElement("button"); rm.type = "button"; rm.className = "pg-btn pg-btn-ghost pg-btn-small hub-btn-danger"; rm.innerHTML = '<i class="fas fa-trash" aria-hidden="true"></i> Delete';
      rm.addEventListener("click", () => {
        if (!window.confirm('Delete "' + p.name + '" from this browser?')) return;
        localStorage.setItem(PIPELINES_KEY, JSON.stringify(localPipelines().filter((x) => x.id !== p.id))); renderLocal();
      });
      tools.appendChild(save); tools.appendChild(rm);
      const line = document.createElement("div"); line.className = "hub-pipe-line";
      line.appendChild(a); line.appendChild(tools);
      li.appendChild(line); li.appendChild(meta);
      list.appendChild(li);
    });
  }
  // ---- quota, from hub-api; and the ?next= return trip from a playground page
  // ?api=http://127.0.0.1:8765 points the page at a local copy of the service; that copy accepts unverified accounts
  const params = new URLSearchParams(location.search);
  const apiBase = (params.get("api") || cfg.apiBase || "").replace(/\/$/, "");
  const devApi = /^https?:\/\/(localhost|127\.0\.0\.1)(:|\/|$)/.test(apiBase);
  const next = params.get("next");
  const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; };

  // ---- downloads kept by this browser (cache.js): listed, removable, switchable
  const fmtBytes = (b) => b >= 1073741824 ? (b / 1073741824).toFixed(1) + " GB" : b >= 1048576 ? (b / 1048576).toFixed(1) + " MB" : b >= 1024 ? Math.round(b / 1024) + " KB" : b + " B";
  function renderCache() {
    const C = window.KorniaCache;
    [["db-cache", "db-cache-summary", true], ["db-cache-out", "db-cache-summary-out", false]].forEach(([tableId, sumId, full]) => {
      const table = $(tableId), sum = $(sumId);
      if (!table || !sum) return;
      const body = table.querySelector("tbody"); body.innerHTML = "";
      if (!C || !C.supported) { sum.textContent = "this browser does not keep downloads (no Cache API)"; return; }
      const entries = C.list();
      C.usage().then((u) => { sum.textContent = (entries.length ? entries.length + " file" + (entries.length === 1 ? "" : "s") + ", " + fmtBytes(u.bytes) : "nothing kept yet") + (u.quota ? " · the browser allows about " + fmtBytes(u.quota) + " for this site" : "") + (C.enabled() ? "" : " · keeping is off"); });
      if (!entries.length) { const tr = el("tr"); const td = el("td", "hub-item-meta", "Run an experiment and its graph or model appears here."); td.colSpan = full ? 5 : 3; tr.appendChild(td); body.appendChild(tr); return; }
      entries.forEach((e) => {
        const tr = el("tr");
        const name = el("td"); name.appendChild(el("code", "", e.label)); name.title = e.url; tr.appendChild(name);
        if (full) tr.appendChild(el("td", "hub-item-meta", e.kind));
        tr.appendChild(el("td", "", fmtBytes(e.bytes)));
        if (full) tr.appendChild(el("td", "hub-item-meta", new Date(e.at).toLocaleDateString()));
        const act = el("td"); const rm = el("button", "pg-link pg-link-danger", "remove"); rm.type = "button";
        rm.addEventListener("click", () => C.remove(e.url).then(renderCache)); act.appendChild(rm); tr.appendChild(act);
        body.appendChild(tr);
      });
    });
    const toggle = $("db-cache-enabled");
    if (toggle && C) toggle.checked = C.enabled();
  }
  ["db-cache-clear", "db-cache-clear-out"].forEach((id) => { const b = $(id); if (b) b.addEventListener("click", () => { if (!window.KorniaCache) return; if (!window.confirm("Remove every kept download? They are fetched again when needed.")) return; window.KorniaCache.clear().then(renderCache); }); });
  if ($("db-cache-enabled")) $("db-cache-enabled").addEventListener("change", (e) => { if (window.KorniaCache) { window.KorniaCache.setEnabled(e.target.checked); if (!e.target.checked) window.KorniaCache.clear().then(renderCache); else renderCache(); } });
  renderCache();

  const NAMES = { rtdetr: "RT-DETR", yunet: "YuNet", xfeat: "XFeat", keynet_hardnet: "KeyNet + HardNet", loftr: "LoFTR", dexined: "DexiNed", depth_anything: "Depth Anything", tinyvit: "TinyViT", small_sr: "ESPCN" };
  const untilReset = () => { const now = new Date(), next = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + 1)); const s = Math.max(0, Math.round((next - now) / 1000)), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60); return (h ? h + " h " : "") + m + " min"; };
  const fmtSecs = (s) => { s = Number(s) || 0; return s >= 60 ? Math.floor(s / 60) + " min" + (Math.round(s % 60) ? " " + Math.round(s % 60) + " s" : "") : s >= 10 ? Math.round(s) + " s" : s.toFixed(1) + " s"; };
  const niceName = (what) => NAMES[what] || what.replace(/^[a-z]+-/, "").replace(/_/g, "_");
  function renderUsage(j) {
    const quota = j.quota, usage = j.usage || { days: [], total: 0, today_by: {}, recent: [] };
    $("hub-tier").textContent = (j.tier === "sponsor" ? "sponsor" : "free") + (cfg.env && cfg.env !== "prod" ? " · " + cfg.env : "");
    $("hub-tier").hidden = false;
    const pct = Math.min(100, 100 * quota.used / Math.max(1, quota.limit));
    $("db-today").textContent = fmtSecs(quota.used); $("db-today-sub").textContent = pct.toFixed(0) + "% of " + fmtSecs(quota.limit) + " · " + (quota.runs || 0) + " request" + (quota.runs === 1 ? "" : "s");
    $("db-month").textContent = fmtSecs(usage.total_seconds || 0);
    $("db-plan").textContent = j.tier === "sponsor" ? "Sponsor" : "Free"; $("db-plan-sub").textContent = fmtSecs(quota.limit) + " of server time a day";
    $("db-free-runs").textContent = fmtSecs(quota.limit);
    const ex = $("db-exports"); ex.innerHTML = "";
    const exports = usage.recent.filter((r) => /^export:/.test(r.what));
    if (!exports.length) ex.appendChild(el("li", "hub-item-meta", "No exports yet today."));
    exports.forEach((r) => { const li = el("li"); const when = new Date(r.at); li.appendChild(el("code", "", niceName(r.what.replace(/^export:/, "")))); li.appendChild(el("span", "hub-item-meta", (isNaN(when) ? r.at : when.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })) + " · export")); ex.appendChild(li); });
    $("hub-today-bar").style.width = Math.min(100, 100 * quota.used / Math.max(1, quota.limit)).toFixed(1) + "%";
    $("hub-today-bar").classList.toggle("is-full", quota.remaining === 0);
    $("hub-quota").textContent = fmtSecs(quota.used) + " / " + fmtSecs(quota.limit) + " · " + pct.toFixed(0) + "% used · " + fmtSecs(quota.remaining) + " left · resets in " + untilReset() + " (00:00 UTC)";
    // 30 days of bars, oldest left, today right
    const chart = $("hub-chart"); chart.innerHTML = "";
    const byDate = {}; usage.days.forEach((d) => { byDate[d.date] = d.seconds != null ? d.seconds : d.runs; });
    const today = new Date(); let max = 1;
    const cells = [];
    for (let i = 29; i >= 0; i--) {
      const d = new Date(today.getTime() - i * 86400000); const key = d.toISOString().slice(0, 10);
      const n = byDate[key] || 0; max = Math.max(max, n); cells.push({ key, n });
    }
    cells.forEach((c) => {
      const b = el("div", "hub-chart-bar" + (c.n ? "" : " is-empty")); b.title = c.key + ": " + fmtSecs(c.n) + " of server time";
      const fill = el("div"); fill.style.height = Math.max(c.n ? 6 : 2, Math.round(100 * c.n / max)) + "%"; b.appendChild(fill); chart.appendChild(b);
    });
    $("hub-chart-caption").textContent = fmtSecs(usage.total_seconds || 0) + " of server time across " + usage.total + " request" + (usage.total === 1 ? "" : "s") + " in the last 30 days";
    const by = $("hub-by"); by.innerHTML = "";
    Object.entries(usage.today_by).sort((a, b) => b[1] - a[1]).forEach(([what, n]) => { by.appendChild(el("span", "hub-chip", niceName(what.replace(/^export:/, "")) + (/^export:/.test(what) ? " export" : "") + " · " + fmtSecs(n))); });
    const list = $("hub-recent"); list.innerHTML = "";
    if (!usage.recent.length) list.appendChild(el("li", "hub-item-meta", "No server runs yet today. Every run of a model or a frame-mode operator on the server shows up here."));
    usage.recent.forEach((r) => {
      const li = el("li"); const when = new Date(r.at);
      li.appendChild(el("code", "", niceName(r.what.replace(/^export:/, "")) + (/^export:/.test(r.what) ? " · export" : "")));
      li.appendChild(el("span", "hub-item-meta", (r.s != null ? fmtSecs(r.s) + " · " : "") + (isNaN(when) ? r.at : when.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }))));
      list.appendChild(li);
    });
  }
  async function refreshAccount(u) {
    const q = $("hub-quota");
    if (!apiBase) { q.textContent = "The server side is not deployed yet."; return; }
    try {
      const r = await fetch(apiBase + "/v1/me", { headers: { Authorization: "Bearer " + await u.getIdToken() } });
      const j = await r.json();
      if (r.ok) renderUsage(j); else q.textContent = j.detail || "usage unavailable";
    } catch (e) { q.textContent = "the server did not answer"; }
  }
  onAuthStateChanged(auth, (u) => {
    if (!u) return;
    if (u.emailVerified || devApi) refreshAccount(u);
    if (next && /^\/[^/]/.test(next)) { setTimeout(() => { location.href = next; }, 600); }
  });
  $("hub-delete").addEventListener("click", async () => {
    if (!user) return;
    const typed = prompt('This deletes your pipelines, your run history and the account itself. Type "delete" to confirm.');
    if (typed !== "delete") return;
    try {
      if (apiBase) {
        const r = await fetch(apiBase + "/v1/me", { method: "DELETE", headers: { Authorization: "Bearer " + await user.getIdToken() } });
        const j = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(j.detail || ("HTTP " + r.status));
      }
      localStorage.removeItem(PIPELINES_KEY);
      await signOut(auth);
      note("Your account and its data are deleted.");
    } catch (e) { note("could not delete the account: " + friendly(e), true); }
  });

  $("hub-rename").addEventListener("click", () => {
    const name = prompt("Display name", user.displayName || "");
    if (name === null) return;
    updateProfile(user, { displayName: name.trim() }).then(() => { $("hub-name").textContent = name.trim() || user.email; $("db-acc-name").textContent = name.trim() || "(no name)"; }).catch((e) => note(friendly(e), true));
  });
}
