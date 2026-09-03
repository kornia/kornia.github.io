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
    if (theme !== "dark" && theme !== "light") theme = systemDark() ? "dark" : "light";
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
