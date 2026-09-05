// Site-wide account state: the header's Sign in / avatar control on every page, and a small API other
// scripts use (window.KorniaAuth) to get the signed-in user's ID token for the hub API.
// Needs dashboard/firebase-config.js loaded first; does nothing when the project is not configured.
import { initializeApp, getApps } from "https://www.gstatic.com/firebasejs/12.18.0/firebase-app.js";
import { getAuth, onAuthStateChanged, signInWithPopup, GithubAuthProvider, GoogleAuthProvider, signOut } from "https://www.gstatic.com/firebasejs/12.18.0/firebase-auth.js";

const cfg = window.KORNIA_FIREBASE || {};
const control = document.getElementById("nav-auth");
const root = new URL("./", import.meta.url).href;   // this module lives at the site root; a module has no currentScript

let user = null;
const listeners = [];
// The signed-in state is remembered locally so the header can paint the avatar on the very first frame,
// before Firebase has confirmed the session; the confirmation then corrects it if needed.
const CACHE = "kornia-auth-state";
function readCache() { try { return JSON.parse(localStorage.getItem(CACHE) || "null"); } catch (e) { return null; } }
function writeCache(u) { try { if (u) localStorage.setItem(CACHE, JSON.stringify({ displayName: u.displayName, email: u.email, photoURL: u.photoURL })); else localStorage.removeItem(CACHE); } catch (e) { /* storage unavailable */ } }
window.KorniaAuth = {
  configured: !!cfg.configured,
  apiBase: (cfg.apiBase || "").replace(/\/$/, ""),
  get user() { return user; },
  onChange(fn) { listeners.push(fn); if (ready) fn(user); },
  token() { return user ? user.getIdToken() : Promise.resolve(null); },
  signIn(provider) { return signInWithPopup(auth, provider === "github" ? new GithubAuthProvider() : new GoogleAuthProvider()); },
  signOut() { return signOut(auth); },
  hubUrl: root + "dashboard/",
  // the account's pipelines (Firestore, users/<uid>/pipelines, or dev_users/ on the dev site); needs a verified email
  pipelines: {
    canSync() { return !!(user && user.emailVerified); },
    async save(p) {
      const f = await firestore();
      const data = { name: p.name, note: p.note || "", version: p.version || 1, container: p.container || null, steps: p.steps || null, model: p.model || null, nodes: p.nodes || null, edges: p.edges || null,
                     created: p.created || Date.now(), updated_ms: p.updated || Date.now(), updated: f.serverTimestamp() };
      await f.setDoc(f.doc(f.db, cfg.usersRoot || "users", user.uid, "pipelines", p.id), data);
    },
    async list() {
      const f = await firestore();
      const snap = await f.getDocs(f.collection(f.db, cfg.usersRoot || "users", user.uid, "pipelines"));
      const out = [];
      snap.forEach((d) => { const p = d.data(); out.push(p.version === 2 ? { id: d.id, name: p.name, note: p.note || "", version: 2, nodes: p.nodes, edges: p.edges, created: p.created || Date.now(), updated: p.updated_ms || 0, synced: true }
                                                        : { id: d.id, name: p.name, note: p.note || "", container: p.container, steps: p.steps, model: p.model || null, created: p.created || Date.now(), updated: p.updated_ms || 0, synced: true }); });
      return out;
    },
    async remove(id) { const f = await firestore(); await f.deleteDoc(f.doc(f.db, cfg.usersRoot || "users", user.uid, "pipelines", id)); },
    async note(id, text) { const f = await firestore(); await f.setDoc(f.doc(f.db, cfg.usersRoot || "users", user.uid, "pipelines", id), { note: text, updated_ms: Date.now(), updated: f.serverTimestamp() }, { merge: true }); },
  },
};
let firestoreMod = null;
async function firestore() {
  if (!firestoreMod) {
    const m = await import("https://www.gstatic.com/firebasejs/12.18.0/firebase-firestore.js");
    firestoreMod = { db: m.getFirestore(getApps()[0]), doc: m.doc, setDoc: m.setDoc, getDocs: m.getDocs, collection: m.collection, deleteDoc: m.deleteDoc, serverTimestamp: m.serverTimestamp };
  }
  return firestoreMod;
}
let ready = false, auth = null;

function el(tag, cls, attrs) { const n = document.createElement(tag); if (cls) n.className = cls; for (const k in attrs || {}) n.setAttribute(k, attrs[k]); return n; }

function render(provisional) {
  if (!control) return;
  control.innerHTML = "";
  const u = user || provisional;
  if (!u) {
    const wrap = el("div", "nav-dropdown nav-auth-dd");
    const btn = el("a", "btn nav-signin-btn", { href: root + "dashboard/", role: "button", "aria-haspopup": "menu", "aria-expanded": "false" });
    btn.innerHTML = '<i class="fas fa-user" aria-hidden="true"></i> Sign in';
    const menu = el("div", "nav-dropdown-content nav-auth-menu");
    [["github", '<i class="fab fa-github" aria-hidden="true"></i>', "Continue with GitHub"], ["google", '<i class="fab fa-google" aria-hidden="true"></i>', "Continue with Google"]].forEach(([p, icon, label]) => {
      const a = el("a", "", { href: "#" });
      a.innerHTML = '<span class="nav-dd-icon">' + icon + '</span><span><span class="nav-dd-name">' + label + "</span></span>";
      a.addEventListener("click", (e) => { e.preventDefault(); window.KorniaAuth.signIn(p).catch((err) => alert(err.message || err)); });
      menu.appendChild(a);
    });
    const mail = el("a", "nav-dd-all", { href: root + "dashboard/?next=" + encodeURIComponent(location.pathname + location.search) });
    mail.textContent = "Email, or create an account →";
    menu.appendChild(mail);
    wrap.appendChild(btn); wrap.appendChild(menu);
    control.appendChild(wrap);
  } else {
    const wrap = el("div", "nav-dropdown nav-auth-dd");
    const btn = el("a", "nav-avatar", { href: root + "dashboard/", role: "button", "aria-haspopup": "menu", "aria-expanded": "false", title: u.displayName || u.email });
    const img = el("img", "", { alt: "", width: 28, height: 28, referrerpolicy: "no-referrer", src: u.photoURL || root + "assets/kornia-logo-mini.svg" });
    btn.appendChild(img);
    const menu = el("div", "nav-dropdown-content nav-auth-menu");
    const who = el("div", "nav-auth-who");
    who.innerHTML = "<strong>" + (u.displayName || "Signed in") + "</strong><span>" + (u.email || "") + "</span>";
    menu.appendChild(who);
    // today's server time, from the quota the playground scripts keep; fetched here on pages without them
    const use = el("div", "nav-auth-usage");
    use.innerHTML = '<div class="nav-auth-usage-row"><div class="nav-auth-usage-bar"><div></div></div><span class="nav-auth-usage-mark">…</span></div><span class="nav-auth-usage-reset"></span>';
    menu.appendChild(use);
    const fmtSecs = (s) => { s = Number(s) || 0; return s >= 60 ? Math.floor(s / 60) + " min" + (Math.round(s % 60) ? " " + Math.round(s % 60) + " s" : "") : s >= 10 ? Math.round(s) + " s" : s.toFixed(1) + " s"; };
    function untilReset() {   // the budget resets at midnight UTC
      const now = new Date(), next = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + 1));
      const s = Math.max(0, Math.round((next - now) / 1000)), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
      return "resets in " + (h ? h + " h " : "") + m + " min";
    }
    let resetTimer = null;
    function paintUsage(q) {
      if (!q) return;
      const bar = use.querySelector(".nav-auth-usage-bar > div");
      bar.style.width = Math.min(100, 100 * q.used / Math.max(1, q.limit)).toFixed(1) + "%";
      bar.classList.toggle("is-full", q.remaining === 0);
      use.querySelector(".nav-auth-usage-mark").textContent = fmtSecs(q.used) + " / " + fmtSecs(q.limit);
      use.querySelector(".nav-auth-usage-reset").textContent = untilReset();
      clearInterval(resetTimer); resetTimer = setInterval(() => { use.querySelector(".nav-auth-usage-reset").textContent = untilReset(); }, 30000);
    }
    window.addEventListener("kornia-quota", (e) => paintUsage(e.detail));
    // the provisional paint works from the cached profile, which cannot fetch; the confirmed user can
    if (!window.KorniaAuth.apiBase) use.hidden = true;
    else if (user && typeof user.getIdToken === "function") {
      user.getIdToken().then((tok) => fetch(window.KorniaAuth.apiBase + "/v1/me", { headers: { Authorization: "Bearer " + tok } })).then((r) => r.ok ? r.json() : null).then((j) => { if (j) paintUsage(j.quota); else use.querySelector("span").textContent = "server usage unavailable"; }).catch(() => { use.querySelector("span").textContent = "server usage unavailable"; });
    }
    const acc = el("a", "", { href: root + "dashboard/" });
    acc.innerHTML = '<span class="nav-dd-icon"><i class="fas fa-id-badge" aria-hidden="true"></i></span><span><span class="nav-dd-name">Dashboard</span><span class="nav-dd-desc">Runs, exports, pipelines</span></span>';
    const out = el("a", "", { href: "#" });
    out.innerHTML = '<span class="nav-dd-icon"><i class="fas fa-right-from-bracket" aria-hidden="true"></i></span><span><span class="nav-dd-name">Sign out</span></span>';
    out.addEventListener("click", (e) => { e.preventDefault(); window.KorniaAuth.signOut(); });
    menu.appendChild(acc); menu.appendChild(out);
    wrap.appendChild(btn); wrap.appendChild(menu);
    control.appendChild(wrap);
  }
  // click toggles the menu (site.js binds only the dropdowns present at load)
  const dd = control.querySelector(".nav-dropdown"), trig = control.querySelector("[aria-haspopup]");
  trig.addEventListener("click", (e) => {
    const mobile = window.matchMedia("(max-width: 1024px)").matches;
    if (!mobile) return;                       // desktop: hover opens, click follows the link (the account page)
    if (!dd.classList.contains("is-open")) { e.preventDefault(); dd.classList.add("is-open"); }
  });
}

if (cfg.configured && control) {
  const cached = readCache();
  if (cached) render(cached);                    // no flash of "Sign in" for someone who is signed in
  const app = getApps().length ? getApps()[0] : initializeApp(cfg);
  auth = getAuth(app);
  onAuthStateChanged(auth, (u) => {
    user = u; ready = true;
    writeCache(u);
    render();
    listeners.forEach((fn) => fn(user));
  });
} else if (control) {
  render();
}
