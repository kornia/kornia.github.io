// Light/dark theme. Runs synchronously in <head> so the first paint already has the right colours.
// Choice is stored in localStorage; with no choice the system preference applies (see the CSS media query).
(function () {
  var KEY = "kornia-theme";
  function stored() { try { return localStorage.getItem(KEY); } catch (e) { return null; } }
  function systemDark() {
    try { return window.matchMedia("(prefers-color-scheme: dark)").matches; } catch (e) { return false; }
  }
  // always leave an explicit attribute on <html>: the stored choice, else the system preference right now
  function apply(theme) {
    if (theme !== "dark" && theme !== "light") theme = "dark";   // the blueprint is the site's face; a stored choice of light still wins
    document.documentElement.setAttribute("data-theme", theme);
  }
  function effective() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t) return t;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  apply(stored());
  function paintButtons() {
    var dark = effective() === "dark";
    document.querySelectorAll(".nav-theme").forEach(function (b) {
      b.innerHTML = dark ? '<i class="fas fa-sun" aria-hidden="true"></i>' : '<i class="fas fa-moon" aria-hidden="true"></i>';
      b.setAttribute("aria-label", dark ? "Switch to the light theme" : "Switch to the dark theme");
      b.title = b.getAttribute("aria-label");
    });
  }
  document.addEventListener("DOMContentLoaded", function () {
    paintButtons();
    document.querySelectorAll(".nav-theme").forEach(function (b) {
      b.addEventListener("click", function () {
        var next = effective() === "dark" ? "light" : "dark";
        apply(next);
        try { localStorage.setItem(KEY, next); } catch (e) { /* storage unavailable */ }
        paintButtons();
      });
    });
    // follow the system while no choice is stored
    if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () { if (!stored()) apply(null); paintButtons(); });
  });
})();

// On phones the experiment and robot sidebars (.pg-side) collapse into a bar at the top of the content: the section
// tabs, and one dropdown listing every entry of the sidebar's lists, grouped as the lists are. Picking an entry
// clicks its link, so the page's own navigation applies. Rebuilt whenever the sidebar's contents change.
document.addEventListener("DOMContentLoaded", function () {
  var side = document.querySelector(".pg-side");
  if (!side) return;
  var bar = document.createElement("div");
  bar.className = "pg-side-mobile";
  side.parentNode.insertBefore(bar, side);
  var timer = null;
  function build() {
    bar.innerHTML = "";
    var mode = side.querySelector("#pg-mode");
    if (mode) {   // the section tabs: forwarded to the real ones
      var tabs = document.createElement("div");
      tabs.className = "pg-mode pg-mode-mobile";
      Array.prototype.forEach.call(mode.querySelectorAll("button"), function (b) {
        var c = b.cloneNode(true);
        c.addEventListener("click", function () { b.click(); });
        tabs.appendChild(c);
      });
      bar.appendChild(tabs);
    }
    // only the entries a visitor would see in the sidebar: nothing inside a hidden list or section
    function shown(el) {
      for (var n = el; n && n !== side; n = n.parentElement) { if (n.hidden || n.getAttribute("aria-hidden") === "true" || n.style.display === "none") return false; }
      return true;
    }
    var links = Array.prototype.filter.call(side.querySelectorAll("a[href]"), shown);
    if (!links.length) return;
    var sel = document.createElement("select");
    sel.className = "pg-select pg-side-select";
    sel.setAttribute("aria-label", side.getAttribute("aria-label") || "Sections");
    var groups = {}, order = [];
    links.forEach(function (a, i) {
      var sec = a.closest("section");
      var h = sec ? sec.querySelector("h2") : null;
      var label = h ? (h.childNodes[0] && h.childNodes[0].textContent || h.textContent).trim() : "";
      if (!groups[label]) { groups[label] = []; order.push(label); }
      var title = a.querySelector(".pg-model-title") || a;
      groups[label].push({ i: i, text: (title.childNodes[0] && title.childNodes[0].textContent || title.textContent).trim(), current: a.classList.contains("pg-row-current") || a.getAttribute("aria-current") === "page" });
    });
    order.forEach(function (label) {
      var parent = sel;
      if (label) { parent = document.createElement("optgroup"); parent.label = label; sel.appendChild(parent); }
      groups[label].forEach(function (it) {
        var o = document.createElement("option");
        o.value = String(it.i); o.textContent = it.text; if (it.current) o.selected = true;
        parent.appendChild(o);
      });
    });
    if (!links.some(function (a) { return a.classList.contains("pg-row-current") || a.getAttribute("aria-current") === "page"; })) {
      var ph = document.createElement("option"); ph.value = ""; ph.textContent = "Choose…"; ph.selected = true; sel.insertBefore(ph, sel.firstChild);
    }
    sel.addEventListener("change", function () { var a = links[Number(sel.value)]; if (a) a.click(); });
    bar.appendChild(sel);
  }
  new MutationObserver(function () { clearTimeout(timer); timer = setTimeout(build, 120); }).observe(side, { childList: true, subtree: true, attributes: true, attributeFilter: ["class", "aria-current", "aria-selected", "hidden", "style"] });
  build();
});
