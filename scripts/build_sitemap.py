#!/usr/bin/env python3
"""Write sitemap.xml from every public page in the checkout.

    python scripts/build_sitemap.py

Every index.html and top-level .html becomes one URL under https://www.kornia.org/, except pages marked
noindex (the dashboard, the old hub redirect, the 404 page). lastmod is the file's last commit date, so a
regenerated page that did not change keeps its date. Run after a playground build or a page edit.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = "https://www.kornia.org"

PRIORITY = [
    (re.compile(r"^$"), "1.0", "weekly"),
    (re.compile(r"^playground/$"), "0.9", "weekly"),
    (re.compile(r"^playground/models/"), "0.8", "monthly"),
    (re.compile(r"^playground/ops/"), "0.7", "monthly"),
    (re.compile(r"^(playground/pipelines|robot)/$"), "0.8", "monthly"),
    (re.compile(r"^news/$"), "0.7", "weekly"),
    (re.compile(r".*"), "0.6", "monthly"),
]


def last_commit(path: Path) -> str:
    out = subprocess.run(["git", "log", "-1", "--format=%cs", "--", str(path)], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return out or subprocess.run(["git", "log", "-1", "--format=%cs"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main() -> int:
    pages = sorted(set(ROOT.glob("*.html")) | set(ROOT.glob("**/index.html")))
    urls = []
    for page in pages:
        rel = page.relative_to(ROOT).as_posix()
        if any(part.startswith(".") or part in ("node_modules", "vendor", "scripts") for part in page.relative_to(ROOT).parts):
            continue
        head = page.read_text(errors="ignore")[:6000]
        if re.search(r'name="robots"[^>]*noindex', head):
            continue
        url = "" if rel == "index.html" else (rel[: -len("index.html")] if rel.endswith("/index.html") else rel)
        priority, freq = next((p, f) for rx, p, f in PRIORITY if rx.match(url))
        urls.append((url, last_commit(page), freq, priority))
    body = "\n".join(
        f"  <url>\n    <loc>{HOST}/{u}</loc>\n    <lastmod>{d}</lastmod>\n    <changefreq>{f}</changefreq>\n    <priority>{p}</priority>\n  </url>"
        for u, d, f, p in urls
    )
    (ROOT / "sitemap.xml").write_text('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + body + "\n</urlset>\n")
    print(f"sitemap.xml: {len(urls)} URLs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
