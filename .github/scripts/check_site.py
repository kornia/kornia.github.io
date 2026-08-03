#!/usr/bin/env python3
"""Structural checks for the kornia.org static site.

Deliberately dependency-free and offline, so it is fast and cannot fail
because a third-party site rate-limited us. External URLs are checked
separately by link-check.yml, which is scheduled and non-blocking.

The checks here are the ones that map to bugs this repo has actually
shipped:

  * a footer link to news.html, a file that did not exist (404 in production)
  * hardcoded stats that no code path could ever update
  * news content living inside index.html, so news.js broke if it moved

Exit code 1 if any error is found. Warnings do not fail the build.
"""

from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]

# Attributes that can carry a URL we are able to resolve locally.
URL_ATTRS = {"href", "src", "poster"}

errors: list[str] = []
warnings: list[str] = []


def error(msg: str) -> None:
    errors.append(msg)


def warn(msg: str) -> None:
    warnings.append(msg)


class PageParser(HTMLParser):
    """Collects ids, link targets and a few a11y-relevant facts."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.links: list[tuple[str, str]] = []  # (attr, value)
        self.imgs_without_alt: list[str] = []
        self.blank_targets_without_rel: list[str] = []
        self.headings: list[tuple[int, str]] = []
        self._heading: tuple[int, list[str]] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: (v or "") for k, v in attrs}

        if "id" in a and a["id"]:
            self.ids.append(a["id"])

        for attr in URL_ATTRS:
            if attr in a and a[attr]:
                self.links.append((attr, a[attr]))

        if tag == "img" and "alt" not in a:
            self.imgs_without_alt.append(a.get("src", "<no src>"))

        if tag == "a" and a.get("target") == "_blank":
            rel = a.get("rel", "")
            if "noopener" not in rel and "noreferrer" not in rel:
                self.blank_targets_without_rel.append(a.get("href", "<no href>"))

        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading = (int(tag[1]), [])

    def handle_data(self, data: str) -> None:
        if self._heading is not None:
            self._heading[1].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self._heading is not None:
            level, parts = self._heading
            self.headings.append((level, "".join(parts).strip()))
            self._heading = None


def check_page(path: Path) -> None:
    rel = path.relative_to(ROOT)
    parser = PageParser()
    parser.feed(path.read_text(encoding="utf-8"))

    # --- duplicate ids -----------------------------------------------------
    seen: set[str] = set()
    for i in parser.ids:
        if i in seen:
            error(f"{rel}: duplicate id={i!r} (breaks anchors and aria-controls)")
        seen.add(i)

    # --- link targets ------------------------------------------------------
    for attr, value in parser.links:
        value = value.strip()
        if not value:
            error(f"{rel}: empty {attr}=''")
            continue

        parsed = urlparse(value)

        # Absolute, protocol-relative, or non-navigational scheme: not ours.
        if parsed.scheme or value.startswith("//"):
            continue
        if value.startswith(("mailto:", "tel:", "data:", "javascript:")):
            continue

        # Pure fragment: must resolve to an id on this page.
        if value.startswith("#"):
            frag = unquote(value[1:])
            if frag and frag not in parser.ids:
                error(f"{rel}: {attr}='{value}' points at an id that does not exist")
            continue

        # Local path, optionally with a fragment.
        target_path = unquote(parsed.path)
        if not target_path:
            continue

        candidate = (ROOT / target_path.lstrip("/")).resolve() if target_path.startswith("/") \
            else (path.parent / target_path).resolve()

        if candidate.is_dir():
            if not (candidate / "index.html").exists():
                warn(f"{rel}: {attr}='{value}' is a directory with no index.html")
            continue

        if not candidate.exists():
            error(f"{rel}: {attr}='{value}' does not exist on disk")

    # --- accessibility -----------------------------------------------------
    for src in parser.imgs_without_alt:
        error(f"{rel}: <img src='{src}'> has no alt attribute")

    for href in parser.blank_targets_without_rel:
        warn(f"{rel}: target=_blank without rel=noopener -> {href}")

    # --- heading order -----------------------------------------------------
    h1s = [t for lvl, t in parser.headings if lvl == 1]
    if len(h1s) == 0:
        error(f"{rel}: no <h1>")
    elif len(h1s) > 1:
        warn(f"{rel}: {len(h1s)} <h1> elements: {h1s}")

    prev = 0
    for level, text in parser.headings:
        if prev and level > prev + 1:
            warn(f"{rel}: heading jumps h{prev} -> h{level} ({text[:40]!r})")
        prev = level

    # --- head essentials ---------------------------------------------------
    html = path.read_text(encoding="utf-8")
    if path.name == "index.html":
        required = {
            "meta description": r'<meta\s+name="description"',
            "canonical": r'<link\s+rel="canonical"',
            "og:title": r'property="og:title"',
            "og:image": r'property="og:image"',
            "twitter:card": r'name="twitter:card"',
            "favicon": r'<link\s+rel="icon"',
        }
        for label, pattern in required.items():
            if not re.search(pattern, html, re.I):
                error(f"{rel}: missing {label}")

    # --- JSON-LD validity --------------------------------------------------
    for block in re.findall(
        r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>', html, re.S
    ):
        try:
            json.loads(block)
        except json.JSONDecodeError as exc:
            error(f"{rel}: invalid JSON-LD ({exc})")

    # --- subresource integrity on CDN assets -------------------------------
    for tag in re.findall(r"<(?:script|link)\s[^>]*>", html, re.I):
        if "cdn" not in tag and "cdnjs" not in tag:
            continue
        if "fonts.googleapis" in tag or "fonts.gstatic" in tag or "preconnect" in tag:
            continue  # Google Fonts serves varying CSS; SRI is not applicable
        if "integrity=" not in tag:
            warn(f"{rel}: CDN asset without integrity= -> {tag[:90]}")


def check_json(name: str, validate) -> None:
    path = ROOT / name
    if not path.exists():
        error(f"{name}: missing")
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        error(f"{name}: invalid JSON ({exc})")
        return
    validate(data)


def validate_news(data) -> None:
    if not isinstance(data, dict):
        error("news.json: top level must be an object")
        return
    tags, items = data.get("tags"), data.get("items")
    if not isinstance(tags, list) or "all" not in tags:
        error("news.json: 'tags' must be a list containing 'all'")
    if not isinstance(items, list) or not items:
        error("news.json: 'items' must be a non-empty list")
        return

    known = set(tags or [])
    for n, item in enumerate(items):
        where = f"news.json item {n}"
        for field in ("date", "title", "content", "tags"):
            if field not in item:
                error(f"{where}: missing '{field}'")
        for t in item.get("tags", []):
            if t not in known:
                error(f"{where}: tag {t!r} is not declared in the top-level 'tags' list")


def validate_stats(data) -> None:
    if not isinstance(data, dict):
        error("stats.json: top level must be an object")
        return
    if "generated_at" not in data:
        error("stats.json: missing 'generated_at'")
    repos = data.get("repos")
    if not isinstance(repos, dict) or not repos:
        error("stats.json: 'repos' must be a non-empty object")
        return
    for name, entry in repos.items():
        stars = (entry or {}).get("stars")
        if not isinstance(stars, int) or stars < 0:
            error(f"stats.json: repos[{name!r}].stars must be a non-negative integer")


def check_stat_contract() -> None:
    """Every [data-stat] element must be resolvable from stats.json.

    This is the check that would have caught the original bug, where the
    numbers a visitor actually reads carried no data-repo attribute and so
    were never updated by any code path.
    """
    index = ROOT / "index.html"
    stats_path = ROOT / "stats.json"
    if not index.exists() or not stats_path.exists():
        return
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return  # already reported

    html = index.read_text(encoding="utf-8")
    repos = set(stats.get("repos", {}))
    packages = set(stats.get("packages", {}))

    found = 0
    for tag in re.findall(r"<[^>]*\bdata-stat=[^>]*>", html):
        found += 1
        kind = re.search(r'data-stat="([^"]+)"', tag)
        kind = kind.group(1) if kind else ""
        if kind == "stars":
            m = re.search(r'data-repo="([^"]+)"', tag)
            if not m:
                error(f"index.html: data-stat='stars' without data-repo -> {tag[:80]}")
            elif m.group(1) not in repos:
                error(f"index.html: data-repo='{m.group(1)}' is not present in stats.json")
        elif kind == "downloads":
            m = re.search(r'data-pkg="([^"]+)"', tag)
            if not m:
                error(f"index.html: data-stat='downloads' without data-pkg -> {tag[:80]}")
            elif m.group(1) not in packages:
                error(f"index.html: data-pkg='{m.group(1)}' is not present in stats.json")
        else:
            error(f"index.html: unknown data-stat={kind!r}")

    if found == 0:
        warn("index.html: no [data-stat] elements; every number on the page is hardcoded")


def main() -> int:
    pages = sorted(ROOT.glob("*.html"))
    if not pages:
        error("no .html files found at the repository root")

    for page in pages:
        check_page(page)

    check_json("news.json", validate_news)
    check_json("stats.json", validate_stats)
    check_stat_contract()

    for w in warnings:
        print(f"warning: {w}")
    for e in errors:
        print(f"error: {e}")

    print(
        f"\nchecked {len(pages)} page(s): "
        f"{len(errors)} error(s), {len(warnings)} warning(s)"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
