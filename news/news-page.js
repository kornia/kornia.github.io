// The news page: news.json rendered as a timeline grouped by month, with project filters and search.
(function () {
  "use strict";
  const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
  const list = document.getElementById("news-timeline");
  const tagsBox = document.getElementById("news-tags");
  const search = document.getElementById("news-search");
  const empty = document.getElementById("news-empty");

  function parseDate(text) {
    const d = new Date(text);
    return isNaN(d) ? null : d;
  }
  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  fetch("../news.json", { cache: "no-cache" }).then(function (r) { return r.json(); }).then(function (news) {
    const items = news.items.map(function (it) { return Object.assign({}, it, { when: parseDate(it.date) }); })
      .sort(function (a, b) { return (b.when || 0) - (a.when || 0); });
    const tags = news.tags.filter(function (t) { return t !== "all"; });
    let active = "all";

    // filter chips
    const allChip = el("button", "news-tag is-active", "All");
    allChip.type = "button"; allChip.dataset.tag = "all";
    tagsBox.appendChild(allChip);
    tags.forEach(function (t) {
      const b = el("button", "news-tag", t);
      b.type = "button"; b.dataset.tag = t;
      tagsBox.appendChild(b);
    });
    tagsBox.addEventListener("click", function (e) {
      const b = e.target.closest(".news-tag");
      if (!b) return;
      active = b.dataset.tag;
      tagsBox.querySelectorAll(".news-tag").forEach(function (x) { x.classList.toggle("is-active", x === b); });
      render();
    });
    search.addEventListener("input", render);

    function render() {
      const q = search.value.trim().toLowerCase();
      list.innerHTML = "";
      let shown = 0, lastMonth = null;
      items.forEach(function (it) {
        const inTag = active === "all" || (it.tags || []).indexOf(active) !== -1;
        const text = (it.title + " " + it.content + " " + (it.tags || []).join(" ")).toLowerCase();
        if (!inTag || (q && text.indexOf(q) === -1)) return;
        const month = it.when ? MONTHS[it.when.getMonth()] + " " + it.when.getFullYear() : "";
        if (month !== lastMonth) {
          const h = el("li", "timeline-month");
          h.appendChild(el("h2", "", month));
          list.appendChild(h);
          lastMonth = month;
        }
        const li = el("li", "timeline-item");
        const dot = el("span", "timeline-dot");
        dot.setAttribute("aria-hidden", "true");
        li.appendChild(dot);
        const card = el("article", "news-card");
        const meta = el("div", "news-meta");
        meta.appendChild(el("time", "", it.when ? it.when.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" }) : it.date));
        (it.tags || []).forEach(function (t) { meta.appendChild(el("span", "news-chip", t)); });
        card.appendChild(meta);
        const h3 = el("h3");
        if (it.link) {
          const a = el("a", "", it.title);
          a.href = it.link; a.target = "_blank"; a.rel = "noopener";
          h3.appendChild(a);
        } else h3.textContent = it.title;
        card.appendChild(h3);
        const body = el("div", "news-content");
        body.innerHTML = it.content;   // news.json is maintained in this repository
        card.appendChild(body);
        if (it.link) {
          const more = el("a", "news-more", "Read more →");
          more.href = it.link; more.target = "_blank"; more.rel = "noopener";
          card.appendChild(more);
        }
        li.appendChild(card);
        list.appendChild(li);
        shown++;
      });
      empty.hidden = shown > 0;
    }
    render();
  }).catch(function () {
    list.innerHTML = "";
    list.appendChild(el("li", "timeline-loading", "The news could not be loaded."));
  });
})();
