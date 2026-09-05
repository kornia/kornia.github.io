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
      if (snap.empty) list.innerHTML = "<li class='pg-note'>Nothing saved yet. Build one in the playground, then press Sync below.</li>";
      snap.forEach((d) => {
        const p = d.data();
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = "../playground/pipelines/?share=" + b64url(JSON.stringify({ name: p.name, container: p.container, steps: p.steps, model: p.model || null }));
        a.textContent = p.name;
        const meta = document.createElement("span");
        meta.className = "hub-item-meta";
        meta.textContent = (p.steps || []).length + " step" + ((p.steps || []).length === 1 ? "" : "s") + (p.model ? " + model" : "") + " · " + (p.container || "");
        const rm = document.createElement("button");
        rm.type = "button"; rm.className = "pg-link pg-link-danger"; rm.title = "delete from the cloud"; rm.innerHTML = '<i class="fas fa-times" aria-hidden="true"></i>';
        rm.addEventListener("click", () => deleteDoc(doc(db, USERS, user.uid, "pipelines", d.id)).then(loadPipelines));
        li.appendChild(a); li.appendChild(meta); li.appendChild(rm);
        list.appendChild(li);
      });
      $("hub-local-count").textContent = String(localPipelines().length);
      $("db-pipes").textContent = String(snap.size);
    } catch (e) { list.innerHTML = ""; note("could not read your pipelines: " + friendly(e), true); }
  }
  $("hub-sync").addEventListener("click", async () => {
    const local = localPipelines();
    if (!local.length) { note("No pipelines in this browser to upload."); return; }
    try {
      await Promise.all(local.map((p) => setDoc(doc(db, USERS, user.uid, "pipelines", p.id), {
        name: p.name, container: p.container, steps: p.steps, model: p.model || null, created: p.created || Date.now(), updated: serverTimestamp(),
      }, { merge: true })));
      note(local.length + " pipeline" + (local.length === 1 ? "" : "s") + " saved to your account.");
      loadPipelines();
    } catch (e) { note("sync failed: " + friendly(e), true); }
  });
  $("hub-pull").addEventListener("click", async () => {
    try {
      const snap = await getDocs(collection(db, USERS, user.uid, "pipelines"));
      const local = localPipelines();
      const byId = {}; local.forEach((p) => { byId[p.id] = p; });
      snap.forEach((d) => { const p = d.data(); byId[d.id] = { id: d.id, name: p.name, container: p.container, steps: p.steps, model: p.model || null, created: p.created || Date.now() }; });
      localStorage.setItem(PIPELINES_KEY, JSON.stringify(Object.values(byId)));
      note(snap.size + " pipeline" + (snap.size === 1 ? "" : "s") + " now available in this browser's playground.");
      $("hub-local-count").textContent = String(Object.keys(byId).length);
    } catch (e) { note("could not download: " + friendly(e), true); }
  });
  // ---- quota, from hub-api; and the ?next= return trip from a playground page
  // ?api=http://127.0.0.1:8765 points the page at a local copy of the service; that copy accepts unverified accounts
  const params = new URLSearchParams(location.search);
  const apiBase = (params.get("api") || cfg.apiBase || "").replace(/\/$/, "");
  const devApi = /^https?:\/\/(localhost|127\.0\.0\.1)(:|\/|$)/.test(apiBase);
  const next = params.get("next");
  const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; };
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
