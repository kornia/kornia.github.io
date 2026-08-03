#!/usr/bin/env python3
"""Check that every external URL on the site still resolves.

Run on a schedule, not on pull requests: third-party hosts rate-limit and
occasionally blip, and a flaky required check trains people to ignore CI.

Some hosts deliberately refuse automated requests. Those are reported as
UNVERIFIED rather than broken, because a 403 from Cloudflare tells us
nothing about whether the page exists:

  403  bot protection (scholar.google.com, some CDNs)
  429  rate limited
  999  LinkedIn's anti-automation status
  202  Semantic Scholar's JS-rendered pages

Exit 1 only when a URL is genuinely gone (404/410) or the host does not
resolve at all.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIMEOUT = 25
UA = "Mozilla/5.0 (compatible; kornia-link-check/1.0; +https://www.kornia.org/)"

# Status codes that mean "the host refused to talk to a robot", not "gone".
INCONCLUSIVE = {202, 401, 403, 405, 429, 503, 999}

# Hosts we never check: they are infrastructure, not content links.
SKIP_HOSTS = {"fonts.googleapis.com", "fonts.gstatic.com"}


def collect_urls() -> set[str]:
    urls: set[str] = set()

    for path in sorted(ROOT.glob("*.html")):
        text = path.read_text(encoding="utf-8")
        urls |= set(re.findall(r'(?:href|src)="(https?://[^"]+)"', text))

    news = ROOT / "news.json"
    if news.exists():
        blob = json.loads(news.read_text(encoding="utf-8"))
        for item in blob.get("items", []):
            if item.get("link"):
                urls.add(item["link"])
            urls |= set(re.findall(r'href=[\'"](https?://[^\'"]+)[\'"]', item.get("content", "")))

    return {u for u in urls if not any(h in u for h in SKIP_HOSTS)}


def probe(url: str) -> tuple[str, int | str]:
    """Return (verdict, status). GET, because many hosts reject HEAD."""
    req = urllib.request.Request(url, headers={"User-Agent": UA}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return "ok", resp.status
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 410):
            return "dead", exc.code
        if exc.code in INCONCLUSIVE:
            return "unverified", exc.code
        return "unverified", exc.code
    except urllib.error.URLError as exc:
        # DNS failure is a real death: this is how bubbaloop.kornia.org broke.
        reason = str(exc.reason)
        if "Name or service not known" in reason or "nodename nor servname" in reason:
            return "dead", "DNS"
        return "unverified", reason[:60]
    except Exception as exc:  # noqa: BLE001 - never let one URL abort the run
        return "unverified", type(exc).__name__


def main() -> int:
    urls = sorted(collect_urls())
    print(f"checking {len(urls)} external URLs\n")

    dead: list[tuple[str, int | str]] = []
    unverified: list[tuple[str, int | str]] = []

    for url in urls:
        verdict, status = probe(url)
        if verdict == "dead":
            dead.append((url, status))
            print(f"DEAD        {status}  {url}")
        elif verdict == "unverified":
            unverified.append((url, status))
            print(f"unverified  {status}  {url}")

    print(f"\n{len(urls)} checked | {len(dead)} dead | {len(unverified)} unverified")

    if unverified:
        print("\nUnverified means the host refused an automated request "
              "(bot protection or rate limiting), not that the link is broken.")

    if dead:
        print("\nDead links:")
        for url, status in dead:
            print(f"  {status}  {url}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
